"""
AutoResearch — Automated Hyperparameter Search

Inspired by Karpathy's autoresearch: runs short training experiments,
evaluates them, keeps improvements, and iterates. A simple evolutionary
search over hyperparameters with simulated evaluation.
"""

import asyncio
import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from bashgym.gym.trainer import TrainerConfig
from bashgym.gym.training_goal import (
    GoalProgress,
    OutcomeAggregator,
    TrainingGoal,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Search Space Definition
# =============================================================================

SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "learning_rate": {
        "type": "float",
        "min": 1e-6,
        "max": 1e-3,
        "log_scale": True,
    },
    "lora_r": {
        "type": "int",
        "min": 4,
        "max": 128,
        "choices": [4, 8, 16, 32, 64, 128],
    },
    "lora_alpha": {
        "type": "int",
        "min": 8,
        "max": 256,
    },
    "lora_dropout": {
        "type": "float",
        "min": 0.0,
        "max": 0.3,
    },
    "warmup_ratio": {
        "type": "float",
        "min": 0.0,
        "max": 0.3,
    },
    "gradient_accumulation_steps": {
        "type": "int",
        "min": 1,
        "max": 64,
        "choices": [1, 2, 4, 8, 16, 32, 64],
    },
    "batch_size": {
        "type": "int",
        "min": 1,
        "max": 32,
        "choices": [1, 2, 4, 8, 16, 32],
    },
    "max_seq_length": {
        "type": "int",
        "min": 512,
        "max": 8192,
        "choices": [512, 1024, 2048, 4096, 8192],
    },
    "load_in_4bit": {
        "type": "bool",
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AutoResearchConfig:
    """Configuration for autoresearch hyperparameter search."""

    # Which TrainerConfig fields to mutate
    search_params: List[str] = field(default_factory=lambda: [
        "learning_rate", "lora_r", "lora_alpha", "warmup_ratio",
    ])

    # Budget
    max_experiments: int = 50
    train_minutes: float = 5.0   # Minutes per experiment (for real training)
    train_steps: int = 100       # Fixed steps instead of time

    # Data
    dataset_subset_ratio: float = 0.1  # Use 10% of training data for fast iteration

    # Eval
    eval_metric: str = "val_loss"  # What to optimize (lower is better for loss)

    # Mutation
    mutation_rate: float = 0.3   # Probability of mutating each param
    mutation_scale: float = 0.2  # Scale of mutations (20% change)


@dataclass
class ExperimentResult:
    """Result of a single autoresearch experiment."""

    experiment_id: int
    config_snapshot: Dict[str, Any]
    metric_value: float
    improved: bool
    duration_seconds: float
    timestamp: str
    goal_progress: Optional[Dict[str, Any]] = None


class AutoResearchStatus:
    """Possible states for the autoresearch loop."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# =============================================================================
# Simulation Helpers
# =============================================================================

def _simulate_loss(config: TrainerConfig, experiment_number: int, total_experiments: int) -> float:
    """Simulate a realistic validation loss for a given config.

    The simulation models these dynamics:
    - Learning rate sweet spot around 1e-5 to 5e-5
    - Extreme learning rates (>1e-3 or <1e-7) produce high loss
    - Higher LoRA rank generally helps but with diminishing returns
    - LoRA alpha / rank ratio matters (good range: 1.5-3.0)
    - Some warmup helps, too much hurts
    - Gaussian noise so not every experiment improves
    - Over 50 experiments the best should trend from ~2.5 to ~1.5-1.8
    """
    loss = 0.0

    # --- Learning rate contribution ---
    lr = config.learning_rate
    # Optimal zone: 1e-5 to 5e-5.  log10 of that is -5 to -4.3
    log_lr = math.log10(max(lr, 1e-10))
    optimal_log_lr = -4.7  # ~2e-5
    lr_penalty = (log_lr - optimal_log_lr) ** 2
    # Scale so being at the sweet spot contributes ~0, being 2 orders off adds ~1.5
    loss += 0.4 * lr_penalty

    # Extreme LR penalty
    if lr > 5e-4:
        loss += 2.0 * (math.log10(lr) - math.log10(5e-4))
    if lr < 1e-7:
        loss += 1.5 * (math.log10(1e-7) - math.log10(max(lr, 1e-10)))

    # --- LoRA rank contribution ---
    lora_r = config.lora_r
    # Diminishing returns: rank 16 is decent, 32 is good, 64+ marginal
    rank_benefit = math.log2(max(lora_r, 1)) / math.log2(128)  # 0 to 1
    loss += 0.8 * (1.0 - rank_benefit)

    # --- LoRA alpha / rank ratio ---
    alpha_ratio = config.lora_alpha / max(config.lora_r, 1)
    # Sweet spot around 2.0
    ratio_penalty = (alpha_ratio - 2.0) ** 2 * 0.05
    loss += min(ratio_penalty, 0.5)

    # --- Warmup ratio contribution ---
    warmup = config.warmup_ratio
    # Small warmup (~0.03-0.1) helps; too much (>0.2) hurts
    warmup_penalty = 0.0
    if warmup < 0.01:
        warmup_penalty = 0.15
    elif warmup > 0.2:
        warmup_penalty = 0.3 * (warmup - 0.2)
    loss += warmup_penalty

    # --- Load in 4bit contribution ---
    if not config.load_in_4bit:
        # Full precision can sometimes be slightly better
        loss -= 0.05

    # --- Base loss ---
    # With default config (optimal LR, lora_r=16, ratio=2.0, warmup=0.1):
    #   lr_penalty=0, rank_cost=~0.26, ratio_penalty=0, warmup=0 → ~0.26 from params
    # We want defaults to land around 2.3-2.5 (before noise), so base ≈ 2.1
    loss += 2.1

    # --- Slight improvement over time (search learns) ---
    progress = experiment_number / max(total_experiments, 1)
    loss -= 0.15 * progress  # Small natural improvement from exploration

    # --- Gaussian noise ---
    noise = random.gauss(0, 0.12)
    loss += noise

    # Clamp to realistic range
    return max(0.3, min(5.0, loss))


# =============================================================================
# AutoResearcher
# =============================================================================

class AutoResearcher:
    """Automated hyperparameter search via iterative experimentation.

    Uses a simple evolutionary strategy: start from the user's config,
    mutate parameters, keep improvements, iterate.

    Optionally accepts a TrainingGoal for multi-criteria optimization
    with constraint enforcement and stall detection. When no goal is
    provided, behavior is identical to the original single-metric approach.
    """

    def __init__(
        self,
        config: AutoResearchConfig,
        base_trainer_config: TrainerConfig,
        goal: Optional[TrainingGoal] = None,
    ):
        self.config = config
        self.best_config = copy.deepcopy(base_trainer_config)
        self.best_metric: float = float("inf")  # Lower is better for loss
        self.experiments: List[ExperimentResult] = []
        self.status: str = AutoResearchStatus.IDLE
        self._running = False
        self._paused = False
        self._error: Optional[str] = None
        self._started_at: Optional[str] = None
        self._completed_at: Optional[str] = None

        # Goal-based optimization (optional)
        self.goal = goal
        self.aggregator: Optional[OutcomeAggregator] = None
        if goal is not None:
            self.aggregator = OutcomeAggregator(goal)

    # -----------------------------------------------------------------
    # Mutation
    # -----------------------------------------------------------------

    def mutate_config(self, config: TrainerConfig) -> TrainerConfig:
        """Create a mutated version of the config.

        For each searchable param, probabilistically apply a mutation.
        """
        mutated = copy.deepcopy(config)

        for param in self.config.search_params:
            if param not in SEARCH_SPACE:
                logger.warning(f"Param '{param}' not in SEARCH_SPACE, skipping")
                continue

            # Decide whether to mutate this param
            if random.random() > self.config.mutation_rate:
                continue

            spec = SEARCH_SPACE[param]
            current_value = getattr(mutated, param, None)
            if current_value is None:
                continue

            new_value = self._mutate_value(current_value, spec)
            setattr(mutated, param, new_value)

        return mutated

    def _mutate_value(self, current: Any, spec: Dict[str, Any]) -> Any:
        """Mutate a single value according to its search space spec."""
        param_type = spec["type"]

        if param_type == "bool":
            return not current

        if param_type == "float":
            if spec.get("log_scale"):
                # Mutate in log space
                log_val = math.log(max(current, 1e-12))
                scale = self.config.mutation_scale * 2.0  # Wider in log space
                log_val += random.gauss(0, scale)
                new_val = math.exp(log_val)
            else:
                # Linear mutation
                delta = current * self.config.mutation_scale
                new_val = current + random.gauss(0, delta) if delta > 0 else random.uniform(spec["min"], spec["max"])
            return max(spec["min"], min(spec["max"], new_val))

        if param_type == "int":
            if "choices" in spec:
                # Pick from discrete choices, biased toward neighbors
                choices = spec["choices"]
                if current in choices:
                    idx = choices.index(current)
                    # Move up or down by 1 with some probability, or random
                    if random.random() < 0.7:
                        direction = random.choice([-1, 1])
                        new_idx = max(0, min(len(choices) - 1, idx + direction))
                    else:
                        new_idx = random.randint(0, len(choices) - 1)
                    return choices[new_idx]
                else:
                    return random.choice(choices)
            else:
                # Continuous int range
                delta = max(1, int(current * self.config.mutation_scale))
                new_val = current + random.randint(-delta, delta)
                return max(spec["min"], min(spec["max"], new_val))

        return current  # Fallback

    # -----------------------------------------------------------------
    # Experiment execution
    # -----------------------------------------------------------------

    def run_experiment(
        self,
        config: TrainerConfig,
        dataset_path: Path,
        experiment_number: int,
    ) -> float:
        """Run a short training experiment and return the eval metric.

        Currently simulated: sleeps 2-3 seconds and returns a realistic
        loss value. Will be replaced with actual short training runs once
        the simulation is validated.
        """
        # Simulate compute time
        sleep_time = random.uniform(2.0, 3.0)
        time.sleep(sleep_time)

        return _simulate_loss(
            config,
            experiment_number,
            self.config.max_experiments,
        )

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    async def run_loop(
        self,
        dataset_path: Path,
        callback: Optional[Callable[..., Coroutine]] = None,
    ):
        """Main autoresearch loop.

        Args:
            dataset_path: Path to training data (used when real training is enabled).
            callback: Async callable invoked after each experiment with
                (result, best_config, best_metric).
        """
        self._running = True
        self._paused = False
        self.status = AutoResearchStatus.RUNNING
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._error = None
        self._completed_at = None

        # Import event bus lazily to avoid circular imports
        try:
            from bashgym.events import event_bus
            from bashgym.events.types import GoalProgressed
            _has_events = True
        except ImportError:
            _has_events = False

        logger.info(
            f"[AutoResearch] Starting loop: {self.config.max_experiments} experiments, "
            f"params={self.config.search_params}"
            + (f", goal={len(self.goal.criteria)} criteria" if self.goal else "")
        )

        try:
            for i in range(self.config.max_experiments):
                if not self._running:
                    logger.info("[AutoResearch] Stopped by user")
                    self.status = AutoResearchStatus.STOPPED
                    break

                # Handle pause
                while self._paused:
                    await asyncio.sleep(0.5)
                    if not self._running:
                        break

                if not self._running:
                    self.status = AutoResearchStatus.STOPPED
                    break

                # Mutate
                candidate = self.mutate_config(self.best_config)

                # Evaluate (run in executor to avoid blocking the event loop)
                loop = asyncio.get_running_loop()
                start = time.time()
                metric = await loop.run_in_executor(
                    None,
                    self.run_experiment,
                    candidate,
                    dataset_path,
                    i + 1,
                )
                duration = time.time() - start

                # Keep or revert
                improved = metric < self.best_metric
                if improved:
                    prev_best = self.best_metric
                    self.best_config = candidate
                    self.best_metric = metric
                    logger.info(
                        f"[AutoResearch] Experiment {i+1}: IMPROVED to {metric:.4f} "
                        f"(was {prev_best:.4f})"
                    )

                # Goal-based progress tracking
                goal_progress_data: Optional[Dict[str, Any]] = None
                if self.aggregator is not None:
                    # Build metrics dict for the aggregator
                    experiment_metrics = {
                        self.config.eval_metric: metric,
                        "experiment_number": i + 1,
                        "duration_seconds": round(duration, 2),
                    }
                    progress = self.aggregator.record(experiment_metrics)
                    goal_progress_data = {
                        "criteria_scores": progress.criteria_scores,
                        "weighted_score": progress.weighted_score,
                        "constraints_status": progress.constraints_status,
                        "recommendation": progress.recommendation,
                        "reasoning": progress.reasoning,
                    }

                    # Emit goal progress event
                    if _has_events:
                        event_bus.emit(GoalProgressed(
                            experiment_id=str(i + 1),
                            weighted_score=progress.weighted_score,
                            recommendation=progress.recommendation,
                            criteria_scores=progress.criteria_scores,
                            constraints_status=progress.constraints_status,
                            reasoning=progress.reasoning,
                        ))

                    # Act on recommendation
                    if progress.recommendation == "complete":
                        logger.info(
                            f"[AutoResearch] Goal recommends COMPLETE at experiment {i+1}: "
                            f"{progress.reasoning}"
                        )
                        self.status = AutoResearchStatus.COMPLETED
                        # Build result before breaking
                        result = ExperimentResult(
                            experiment_id=i + 1,
                            config_snapshot={
                                param: getattr(candidate, param)
                                for param in self.config.search_params
                                if hasattr(candidate, param)
                            },
                            metric_value=round(metric, 6),
                            improved=improved,
                            duration_seconds=round(duration, 2),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            goal_progress=goal_progress_data,
                        )
                        self.experiments.append(result)
                        if callback:
                            try:
                                await callback(result, self.best_config, self.best_metric)
                            except Exception as cb_err:
                                logger.warning(f"[AutoResearch] Callback error: {cb_err}")
                        break

                    elif progress.recommendation == "adjust":
                        # Increase mutation rate to explore more
                        old_rate = self.config.mutation_rate
                        self.config.mutation_rate = min(
                            1.0, self.config.mutation_rate * 1.5
                        )
                        logger.info(
                            f"[AutoResearch] Goal recommends ADJUST: mutation_rate "
                            f"{old_rate:.2f} -> {self.config.mutation_rate:.2f}"
                        )

                result = ExperimentResult(
                    experiment_id=i + 1,
                    config_snapshot={
                        param: getattr(candidate, param)
                        for param in self.config.search_params
                        if hasattr(candidate, param)
                    },
                    metric_value=round(metric, 6),
                    improved=improved,
                    duration_seconds=round(duration, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    goal_progress=goal_progress_data,
                )
                self.experiments.append(result)

                if callback:
                    try:
                        await callback(result, self.best_config, self.best_metric)
                    except Exception as cb_err:
                        logger.warning(f"[AutoResearch] Callback error: {cb_err}")

            else:
                # Completed all experiments
                self.status = AutoResearchStatus.COMPLETED

        except Exception as exc:
            logger.error(f"[AutoResearch] Loop failed: {exc}", exc_info=True)
            self.status = AutoResearchStatus.FAILED
            self._error = str(exc)

        finally:
            self._running = False
            self._completed_at = datetime.now(timezone.utc).isoformat()
            logger.info(
                f"[AutoResearch] Finished. Status={self.status}, "
                f"experiments={len(self.experiments)}, best_metric={self.best_metric:.4f}"
            )

        return self.best_config, self.experiments

    # -----------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------

    def stop(self):
        """Stop the autoresearch loop."""
        self._running = False

    def pause(self):
        """Pause the autoresearch loop."""
        self._paused = True
        self.status = AutoResearchStatus.PAUSED

    def resume(self):
        """Resume the autoresearch loop."""
        self._paused = False
        if self.status == AutoResearchStatus.PAUSED:
            self.status = AutoResearchStatus.RUNNING

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a serializable status dict."""
        status = {
            "status": self.status,
            "total_experiments": self.config.max_experiments,
            "completed_experiments": len(self.experiments),
            "best_metric": round(self.best_metric, 6) if self.best_metric < float("inf") else None,
            "best_config": {
                param: getattr(self.best_config, param)
                for param in self.config.search_params
                if hasattr(self.best_config, param)
            },
            "search_params": self.config.search_params,
            "mutation_rate": self.config.mutation_rate,
            "mutation_scale": self.config.mutation_scale,
            "eval_metric": self.config.eval_metric,
            "experiments": [asdict(e) for e in self.experiments],
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "error": self._error,
        }

        # Include goal status if present
        if self.aggregator is not None and self.aggregator.history:
            latest = self.aggregator.history[-1][1]
            status["goal_progress"] = {
                "weighted_score": latest.weighted_score,
                "criteria_scores": latest.criteria_scores,
                "constraints_status": latest.constraints_status,
                "recommendation": latest.recommendation,
            }

        return status
