"""
AutoResearch — Automated Evolutionary Search

Inspired by Karpathy's autoresearch: runs short experiments,
evaluates them, keeps improvements, and iterates. A simple evolutionary
search with a pluggable SearchSpace abstraction.

The core pattern (mutate -> evaluate -> keep if better) is generic.
Strategy-specific behavior is encapsulated in SearchSpace subclasses:
- HyperparamSearchSpace: mutates TrainerConfig fields (learning_rate, lora_r, etc.)
- SchemaSearchSpace: (future) mutates Data Designer pipeline configs
"""

import asyncio
import copy
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bashgym.gym.trainer import TrainerConfig

logger = logging.getLogger(__name__)


# =============================================================================
# SearchSpace ABC
# =============================================================================


class SearchSpace(ABC):
    """Abstract search space for evolutionary optimization.

    Defines what gets mutated and how candidates are evaluated.
    AutoResearcher uses this to support different optimization targets:
    - HyperparamSearchSpace: mutates TrainerConfig fields (learning_rate, lora_r, etc.)
    - SchemaSearchSpace: mutates Data Designer pipeline configs (temperatures, columns, judges)
    """

    @abstractmethod
    def mutate(self, config: Any) -> Any:
        """Create a mutated version of the config."""
        ...

    @abstractmethod
    def evaluate(self, config: Any, experiment_number: int, total_experiments: int) -> float:
        """Evaluate a config and return a metric (lower is better)."""
        ...

    @abstractmethod
    def get_config_snapshot(self, config: Any) -> dict[str, Any]:
        """Extract a serializable snapshot of the searchable params."""
        ...


# =============================================================================
# Search Space Definition
# =============================================================================

SEARCH_SPACE: dict[str, dict[str, Any]] = {
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
    search_params: list[str] = field(
        default_factory=lambda: [
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "warmup_ratio",
        ]
    )

    # Budget
    max_experiments: int = 50
    train_minutes: float = 5.0  # Minutes per experiment (for real training)
    train_steps: int = 100  # Fixed steps instead of time

    # Data
    dataset_subset_ratio: float = 0.1  # Use 10% of training data for fast iteration

    # Eval
    eval_metric: str = "val_loss"  # What to optimize (lower is better for loss)

    # Mode: "simulate" for fast heuristic, "real" for actual short training runs
    mode: str = "simulate"

    # Mutation
    mutation_rate: float = 0.3  # Probability of mutating each param
    mutation_scale: float = 0.2  # Scale of mutations (20% change)


@dataclass
class ExperimentResult:
    """Result of a single autoresearch experiment."""

    experiment_id: int
    config_snapshot: dict[str, Any]
    metric_value: float
    improved: bool
    duration_seconds: float
    timestamp: str


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
# Mutation Helpers
# =============================================================================


def _mutate_value(current: Any, spec: dict[str, Any], mutation_scale: float) -> Any:
    """Mutate a single value according to its search space spec.

    This is a module-level function used by HyperparamSearchSpace.
    """
    param_type = spec["type"]

    if param_type == "bool":
        return not current

    if param_type == "float":
        if spec.get("log_scale"):
            # Mutate in log space
            log_val = math.log(max(current, 1e-12))
            scale = mutation_scale * 2.0  # Wider in log space
            log_val += random.gauss(0, scale)
            new_val = math.exp(log_val)
        else:
            # Linear mutation
            delta = current * mutation_scale
            new_val = (
                current + random.gauss(0, delta)
                if delta > 0
                else random.uniform(spec["min"], spec["max"])
            )
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
            delta = max(1, int(current * mutation_scale))
            new_val = current + random.randint(-delta, delta)
            return max(spec["min"], min(spec["max"], new_val))

    return current  # Fallback


# =============================================================================
# HyperparamSearchSpace
# =============================================================================


class HyperparamSearchSpace(SearchSpace):
    """Search space for training hyperparameters (the original AutoResearch behavior)."""

    def __init__(
        self,
        search_params: list[str],
        mutation_rate: float = 0.3,
        mutation_scale: float = 0.2,
        mode: str = "simulate",
        train_steps: int = 100,
        dataset_path: Path | None = None,
        val_dataset_path: Path | None = None,
    ):
        self.search_params = search_params
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.mode = mode
        self.train_steps = train_steps
        self._dataset_path = dataset_path
        self._val_dataset_path = val_dataset_path

    def mutate(self, config: TrainerConfig) -> TrainerConfig:
        """Create a mutated version of the TrainerConfig."""
        mutated = copy.deepcopy(config)

        for param in self.search_params:
            if param not in SEARCH_SPACE:
                logger.warning(f"Param '{param}' not in SEARCH_SPACE, skipping")
                continue

            # Decide whether to mutate this param
            if random.random() > self.mutation_rate:
                continue

            spec = SEARCH_SPACE[param]
            current_value = getattr(mutated, param, None)
            if current_value is None:
                continue

            new_value = _mutate_value(current_value, spec, self.mutation_scale)
            setattr(mutated, param, new_value)

        return mutated

    def evaluate(
        self, config: TrainerConfig, experiment_number: int, total_experiments: int
    ) -> float:
        """Evaluate a config by running a real or simulated training experiment."""
        if self.mode == "real":
            return self._run_real_experiment(config, experiment_number, total_experiments)

        # Simulate mode: sleep briefly and return heuristic loss
        time.sleep(random.uniform(2.0, 3.0))
        return _simulate_loss(config, experiment_number, total_experiments)

    def get_config_snapshot(self, config: TrainerConfig) -> dict[str, Any]:
        """Extract a serializable snapshot of the searchable params."""
        return {
            param: getattr(config, param) for param in self.search_params if hasattr(config, param)
        }

    def _run_real_experiment(
        self,
        config: TrainerConfig,
        experiment_number: int,
        total_experiments: int,
    ) -> float:
        """Run a real short training experiment and return eval/final loss."""
        from bashgym.gym.trainer import Trainer

        # Configure for a short experiment
        exp_config = copy.deepcopy(config)
        exp_config.max_steps = self.train_steps
        exp_config.eval_strategy = "steps"
        exp_config.eval_steps = max(10, self.train_steps // 3)
        exp_config.auto_export_gguf = False  # No export during search
        exp_config.save_steps = 999999  # Don't save checkpoints
        exp_config.logging_steps = 5

        import tempfile

        with tempfile.TemporaryDirectory(prefix=f"autoresearch_exp{experiment_number}_") as tmpdir:
            exp_config.output_dir = tmpdir

            trainer = Trainer(exp_config)

            dataset_path = self._dataset_path
            val_path = self._val_dataset_path

            if not dataset_path or not dataset_path.exists():
                logger.error(
                    f"[AutoResearch] Real mode: dataset_path={dataset_path} not found, "
                    "falling back to simulation"
                )
                return _simulate_loss(config, experiment_number, total_experiments)

            try:
                run = trainer.train_sft(
                    dataset_path=dataset_path,
                    val_dataset_path=val_path,
                )

                # Prefer eval_loss, fall back to final_loss
                eval_loss = run.metrics.get("eval_loss")
                final_loss = run.metrics.get("final_loss", 5.0)

                metric = eval_loss if eval_loss is not None else final_loss
                logger.info(
                    f"[AutoResearch] Experiment {experiment_number}: "
                    f"eval_loss={eval_loss}, final_loss={final_loss}, metric={metric}"
                )
                return float(metric)

            except Exception as e:
                logger.error(f"[AutoResearch] Real experiment {experiment_number} failed: {e}")
                return 5.0  # Worst-case metric so this config is rejected


# =============================================================================
# AutoResearcher
# =============================================================================


class AutoResearcher:
    """Automated evolutionary search via iterative experimentation.

    Uses a simple evolutionary strategy: start from a config,
    mutate parameters, keep improvements, iterate.

    The search behavior is defined by a pluggable SearchSpace:
    - HyperparamSearchSpace (default): mutates TrainerConfig hyperparameters
    - Custom SearchSpace subclasses can optimize other targets
    """

    def __init__(
        self,
        config: AutoResearchConfig,
        base_trainer_config: TrainerConfig,
        dataset_path: Path | None = None,
        val_dataset_path: Path | None = None,
        search_space: SearchSpace | None = None,
    ):
        self.config = config
        self.best_config = copy.deepcopy(base_trainer_config)
        self.best_metric: float = float("inf")  # Lower is better for loss
        self.experiments: list[ExperimentResult] = []
        self.status: str = AutoResearchStatus.IDLE
        self._running = False
        self._paused = False
        self._error: str | None = None
        self._started_at: str | None = None
        self._completed_at: str | None = None
        self._dataset_path = dataset_path
        self._val_dataset_path = val_dataset_path

        # Use provided search space or create default HyperparamSearchSpace
        self.search_space = search_space or HyperparamSearchSpace(
            search_params=config.search_params,
            mutation_rate=config.mutation_rate,
            mutation_scale=config.mutation_scale,
            mode=config.mode,
            train_steps=config.train_steps,
            dataset_path=dataset_path,
            val_dataset_path=val_dataset_path,
        )

    # -----------------------------------------------------------------
    # Mutation
    # -----------------------------------------------------------------

    def mutate_config(self, config: TrainerConfig) -> TrainerConfig:
        """Create a mutated version of the config. Delegates to search_space."""
        return self.search_space.mutate(config)

    def _mutate_value(self, current: Any, spec: dict[str, Any]) -> Any:
        """Mutate a single value. Delegates to module-level _mutate_value.

        Kept for backward compatibility.
        """
        return _mutate_value(current, spec, self.config.mutation_scale)

    # -----------------------------------------------------------------
    # Experiment execution
    # -----------------------------------------------------------------

    def run_experiment(
        self,
        config: TrainerConfig,
        dataset_path: Path,
        experiment_number: int,
    ) -> float:
        """Run experiment. Delegates to search_space.evaluate()."""
        return self.search_space.evaluate(config, experiment_number, self.config.max_experiments)

    def _run_real_experiment(
        self,
        config: TrainerConfig,
        experiment_number: int,
    ) -> float:
        """Run a real short training experiment. Delegates to search_space.

        Kept for backward compatibility.
        """
        if isinstance(self.search_space, HyperparamSearchSpace):
            return self.search_space._run_real_experiment(
                config, experiment_number, self.config.max_experiments
            )
        # Fallback for non-hyperparam search spaces
        return self.search_space.evaluate(config, experiment_number, self.config.max_experiments)

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    async def run_loop(
        self,
        dataset_path: Path,
        callback: Callable[..., Coroutine] | None = None,
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

        logger.info(
            f"[AutoResearch] Starting loop: {self.config.max_experiments} experiments, "
            f"params={self.config.search_params}"
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

                result = ExperimentResult(
                    experiment_id=i + 1,
                    config_snapshot=self.search_space.get_config_snapshot(candidate),
                    metric_value=round(metric, 6),
                    improved=improved,
                    duration_seconds=round(duration, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
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

    def get_status(self) -> dict[str, Any]:
        """Return a serializable status dict."""
        return {
            "status": self.status,
            "total_experiments": self.config.max_experiments,
            "completed_experiments": len(self.experiments),
            "best_metric": round(self.best_metric, 6) if self.best_metric < float("inf") else None,
            "best_config": self.search_space.get_config_snapshot(self.best_config),
            "search_params": self.config.search_params,
            "mutation_rate": self.config.mutation_rate,
            "mutation_scale": self.config.mutation_scale,
            "eval_metric": self.config.eval_metric,
            "mode": self.config.mode,
            "experiments": [asdict(e) for e in self.experiments],
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "error": self._error,
        }
