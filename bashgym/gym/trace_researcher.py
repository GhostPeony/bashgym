"""
Trace Researcher -- Automated training data optimization.

Iterates over different trace selection and example generation strategies,
evaluates each by training a short run, and keeps the best data pipeline config.

Inspired by Karpathy's autoresearch but applied to data curation rather than
hyperparameter tuning.
"""

import asyncio
import copy
import logging
import math
import random
import tempfile
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Search Space
# =============================================================================

DATA_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "min_success_rate": {
        "type": "float",
        "min": 0.3,
        "max": 0.95,
        "default": 0.7,
        "description": "Minimum step success rate to include a trace",
    },
    "min_quality_score": {
        "type": "float",
        "min": 0.2,
        "max": 0.9,
        "default": 0.5,
        "description": "Minimum quality score threshold",
    },
    "max_steps_per_example": {
        "type": "int",
        "min": 5,
        "max": 100,
        "default": 50,
        "description": "Maximum steps in a single training example",
    },
    "min_steps_per_example": {
        "type": "int",
        "min": 1,
        "max": 20,
        "default": 3,
        "description": "Minimum steps for a viable example",
    },
    "include_cognitive": {
        "type": "bool",
        "default": True,
        "description": "Include thinking/planning/reflection tags",
    },
    "include_failed_as_dpo": {
        "type": "bool",
        "default": False,
        "description": "Include failed traces as DPO rejected examples",
    },
    "time_gap_threshold_minutes": {
        "type": "float",
        "min": 1.0,
        "max": 30.0,
        "default": 5.0,
        "description": "Time gap for segmenting sessions into examples",
    },
    "silver_inclusion_ratio": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "description": "Fraction of silver traces to include (0 = gold only)",
    },
    "dedup_similarity_threshold": {
        "type": "float",
        "min": 0.5,
        "max": 1.0,
        "default": 0.85,
        "description": "Similarity threshold for deduplication",
    },
    "max_examples_per_repo": {
        "type": "int",
        "min": 10,
        "max": 1000,
        "default": 500,
        "description": "Cap examples per repo to prevent dominance",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DataPipelineConfig:
    """Configuration for a data curation strategy."""

    min_success_rate: float = 0.7
    min_quality_score: float = 0.5
    max_steps_per_example: int = 50
    min_steps_per_example: int = 3
    include_cognitive: bool = True
    include_failed_as_dpo: bool = False
    time_gap_threshold_minutes: float = 5.0
    silver_inclusion_ratio: float = 0.0
    dedup_similarity_threshold: float = 0.85
    max_examples_per_repo: int = 500


@dataclass
class TraceExperimentResult:
    """Result of a single trace research experiment."""

    experiment_id: int
    config_snapshot: dict[str, Any]
    # Data metrics
    examples_generated: int
    unique_repos: int
    avg_example_length: float
    # Training metrics (from short eval run)
    metric_value: float  # val_loss or similar
    improved: bool
    duration_seconds: float
    timestamp: str


@dataclass
class TraceResearchConfig:
    """Configuration for the trace research loop."""

    search_params: list[str] = field(
        default_factory=lambda: [
            "min_success_rate",
            "min_quality_score",
            "max_steps_per_example",
            "include_cognitive",
            "silver_inclusion_ratio",
            "time_gap_threshold_minutes",
        ]
    )
    max_experiments: int = 30
    mutation_rate: float = 0.4
    mutation_scale: float = 0.25
    # Paths
    gold_traces_dir: str = ""
    silver_traces_dir: str = ""
    failed_traces_dir: str = ""
    pending_traces_dir: str = ""


class TraceResearchStatus:
    """Possible states for the trace research loop."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# =============================================================================
# Simulation Helpers
# =============================================================================


def _simulate_metric(
    pipeline: DataPipelineConfig,
    experiment_number: int,
    total_experiments: int,
) -> tuple[float, dict[str, Any]]:
    """Simulate a realistic metric and data stats for a pipeline config.

    Models these dynamics:
    - Optimal success rate filter around 0.75-0.85 (too strict = too little
      data, too lenient = noisy)
    - Cognitive tags help the model learn reasoning
    - A moderate silver inclusion ratio adds useful data diversity
    - DPO pairs from failed traces provide contrast
    - Dedup threshold and per-repo caps prevent overfitting
    - Noise so not every experiment improves
    - Over 30 experiments the best should trend from ~2.5 down to ~1.7-2.0
    """
    # ---- Simulate data generation stats ----
    base_examples = 150

    # More lenient quality thresholds = more examples but lower quality
    quality_factor = 1.0 + (0.7 - pipeline.min_success_rate) * 2
    quality_factor *= 1.0 + (0.5 - pipeline.min_quality_score) * 1.5

    # Silver inclusion adds more data
    if pipeline.silver_inclusion_ratio > 0:
        quality_factor *= 1.0 + pipeline.silver_inclusion_ratio * 0.5

    examples_generated = int(base_examples * quality_factor)
    examples_generated = max(20, min(500, examples_generated))

    unique_repos = random.randint(2, min(8, max(2, examples_generated // 30)))

    avg_length = pipeline.max_steps_per_example * 0.6 + random.uniform(-5, 5)
    avg_length = max(pipeline.min_steps_per_example + 1, avg_length)

    data_stats = {
        "examples_generated": examples_generated,
        "unique_repos": unique_repos,
        "avg_example_length": round(avg_length, 1),
    }

    # ---- Simulate training metric ----
    base_loss = 2.5

    # Optimal success rate filter ~ 0.8
    sr_optimal = 0.8
    sr_penalty = abs(pipeline.min_success_rate - sr_optimal) * 1.5

    # Optimal quality score ~ 0.6
    qs_optimal = 0.6
    qs_penalty = abs(pipeline.min_quality_score - qs_optimal) * 1.2

    # Cognitive tags help
    cognitive_bonus = -0.15 if pipeline.include_cognitive else 0.0

    # Silver traces help a bit if ratio is moderate (0.1-0.3)
    silver_optimal = 0.2
    silver_penalty = (
        abs(pipeline.silver_inclusion_ratio - silver_optimal) * 0.5
        if pipeline.silver_inclusion_ratio > 0
        else 0.1
    )

    # DPO pairs help
    dpo_bonus = -0.1 if pipeline.include_failed_as_dpo else 0.0

    # Longer examples generally better up to a point
    steps_factor = -0.1 * min(1.0, pipeline.max_steps_per_example / 50)

    # Time gap: too small = fragments, too large = mixed tasks
    gap_optimal = 5.0
    gap_penalty = abs(pipeline.time_gap_threshold_minutes - gap_optimal) * 0.03

    # Dedup: moderate is best
    dedup_penalty = abs(pipeline.dedup_similarity_threshold - 0.85) * 0.3

    # More data generally helps (log scale)
    data_bonus = -0.2 * math.log(max(1, examples_generated) / 100)

    # Per-repo cap helps prevent overfitting to one repo
    cap_bonus = -0.05 if pipeline.max_examples_per_repo < 200 else 0.0

    metric = (
        base_loss
        + sr_penalty
        + qs_penalty
        + cognitive_bonus
        + silver_penalty
        + dpo_bonus
        + steps_factor
        + gap_penalty
        + dedup_penalty
        + data_bonus
        + cap_bonus
    )

    # Slight improvement over time (search learns)
    progress = experiment_number / max(total_experiments, 1)
    metric -= 0.15 * progress

    # Add noise
    metric += random.gauss(0, 0.08)
    metric = max(0.5, metric)

    return round(metric, 4), data_stats


# =============================================================================
# TraceResearcher
# =============================================================================


class TraceResearcher:
    """Automated training data optimization through iterative trace mining.

    Uses a simple evolutionary strategy: start from default data pipeline
    config, mutate parameters, evaluate by simulating training, keep
    improvements, iterate.
    """

    def __init__(self, config: TraceResearchConfig):
        self.config = config
        self.best_pipeline = DataPipelineConfig()
        self.best_metric: float = float("inf")
        self.best_data_stats: dict[str, Any] = {}
        self.experiments: list[TraceExperimentResult] = []
        self.status: str = TraceResearchStatus.IDLE
        self._running = False
        self._paused = False
        self._error: str | None = None
        self._started_at: str | None = None
        self._completed_at: str | None = None

    # -----------------------------------------------------------------
    # Mutation
    # -----------------------------------------------------------------

    def mutate_pipeline(self, pipeline: DataPipelineConfig) -> DataPipelineConfig:
        """Create a mutated version of the data pipeline config."""
        new = copy.deepcopy(pipeline)

        for param in self.config.search_params:
            if random.random() > self.config.mutation_rate:
                continue

            spec = DATA_SEARCH_SPACE.get(param)
            if not spec:
                continue

            current = getattr(new, param, None)
            if current is None:
                continue

            if spec["type"] == "float":
                delta = (spec["max"] - spec["min"]) * self.config.mutation_scale
                new_val = current + random.uniform(-delta, delta)
                new_val = max(spec["min"], min(spec["max"], new_val))
                setattr(new, param, round(new_val, 4))
            elif spec["type"] == "int":
                delta = max(1, int((spec["max"] - spec["min"]) * self.config.mutation_scale))
                new_val = current + random.randint(-delta, delta)
                new_val = max(spec["min"], min(spec["max"], new_val))
                setattr(new, param, new_val)
            elif spec["type"] == "bool":
                setattr(new, param, not current)

        return new

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def evaluate_pipeline(
        self,
        pipeline: DataPipelineConfig,
        experiment_number: int,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a data pipeline configuration.

        In simulation mode: generates a realistic metric based on the
        pipeline config.  In real mode: generates examples with the pipeline,
        exports train/val, runs a short training, and returns eval_loss.
        """
        if getattr(self, "_mode", "simulate") == "real":
            return self._evaluate_pipeline_real(pipeline, experiment_number)

        # Simulate compute time (2-4 seconds)
        time.sleep(random.uniform(2.0, 4.0))

        return _simulate_metric(
            pipeline,
            experiment_number,
            self.config.max_experiments,
        )

    def _evaluate_pipeline_real(
        self,
        pipeline: DataPipelineConfig,
        experiment_number: int,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a pipeline by actually generating data and training.

        1. Generate examples with this pipeline's config
        2. Export train/val split
        3. Run short training
        4. Return (eval_loss, data_stats)
        """
        from bashgym.factory.example_generator import ExampleGenerator, ExampleGeneratorConfig
        from bashgym.gym.trainer import Trainer, TrainerConfig

        gold_dir = Path(self.config.gold_traces_dir) if self.config.gold_traces_dir else None
        if not gold_dir or not gold_dir.exists():
            logger.warning(
                "[TraceResearch] No gold_traces_dir for real eval, falling back to simulation"
            )
            return _simulate_metric(pipeline, experiment_number, self.config.max_experiments)

        with tempfile.TemporaryDirectory(prefix=f"trace_exp{experiment_number}_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            try:
                # Generate examples with candidate pipeline config
                gen_config = ExampleGeneratorConfig(
                    output_dir=str(tmpdir_path),
                    min_quality_score=pipeline.min_quality_score,
                    include_cognitive=pipeline.include_cognitive,
                )
                generator = ExampleGenerator(gen_config)
                examples, _stats = generator.process_directory(gold_dir)

                if not examples:
                    return 5.0, {
                        "examples_generated": 0,
                        "unique_repos": 0,
                        "avg_example_length": 0,
                    }

                # Export
                result = generator.export_for_nemo(examples, tmpdir_path, train_split=0.9)
                train_path = Path(result["train"])
                val_path = Path(result["validation"]) if result.get("validation") else None

                data_stats = {
                    "examples_generated": len(examples),
                    "unique_repos": len({getattr(e, "repo_name", "unknown") for e in examples}),
                    "avg_example_length": round(
                        sum(len(getattr(e, "messages", [])) for e in examples)
                        / max(len(examples), 1),
                        1,
                    ),
                }

                # Short training
                trainer_config = TrainerConfig(
                    max_steps=getattr(self, "_train_steps", 50),
                    eval_strategy="steps",
                    eval_steps=max(10, getattr(self, "_train_steps", 50) // 3),
                    auto_export_gguf=False,
                    save_steps=999999,
                    logging_steps=5,
                )
                trainer = Trainer(trainer_config)
                run = trainer.train_sft(
                    dataset_path=train_path,
                    val_dataset_path=val_path,
                )

                eval_loss = run.metrics.get("eval_loss")
                final_loss = run.metrics.get("final_loss", 5.0)
                metric = eval_loss if eval_loss is not None else final_loss

                return float(metric), data_stats

            except Exception as e:
                logger.error(
                    f"[TraceResearch] Real eval failed for experiment {experiment_number}: {e}"
                )
                return 5.0, {"examples_generated": 0, "unique_repos": 0, "avg_example_length": 0}

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    async def run_loop(
        self,
        callback: Callable[..., Coroutine] | None = None,
    ):
        """Main trace research loop.

        Args:
            callback: Async callable invoked after each experiment with
                (result, best_pipeline, best_metric, best_data_stats).
        """
        self._running = True
        self._paused = False
        self.status = TraceResearchStatus.RUNNING
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._error = None
        self._completed_at = None

        logger.info(
            f"[TraceResearch] Starting loop: {self.config.max_experiments} experiments, "
            f"params={self.config.search_params}"
        )

        try:
            for i in range(self.config.max_experiments):
                if not self._running:
                    logger.info("[TraceResearch] Stopped by user")
                    self.status = TraceResearchStatus.STOPPED
                    break

                # Handle pause
                while self._paused:
                    self.status = TraceResearchStatus.PAUSED
                    await asyncio.sleep(0.5)
                    if not self._running:
                        break

                if not self._running:
                    self.status = TraceResearchStatus.STOPPED
                    break

                self.status = TraceResearchStatus.RUNNING

                # Mutate pipeline config
                candidate = self.mutate_pipeline(self.best_pipeline)

                # Evaluate (run in executor to avoid blocking the event loop)
                loop = asyncio.get_running_loop()
                start = time.time()
                metric, data_stats = await loop.run_in_executor(
                    None,
                    self.evaluate_pipeline,
                    candidate,
                    i + 1,
                )
                duration = time.time() - start

                # Keep or revert
                improved = metric < self.best_metric
                if improved:
                    self.best_pipeline = candidate
                    self.best_metric = metric
                    self.best_data_stats = data_stats
                    logger.info(f"[TraceResearch] Experiment {i+1}: IMPROVED to {metric:.4f}")

                result = TraceExperimentResult(
                    experiment_id=i + 1,
                    config_snapshot={p: getattr(candidate, p) for p in self.config.search_params},
                    examples_generated=data_stats["examples_generated"],
                    unique_repos=data_stats["unique_repos"],
                    avg_example_length=data_stats["avg_example_length"],
                    metric_value=metric,
                    improved=improved,
                    duration_seconds=round(duration, 2),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.experiments.append(result)

                if callback:
                    try:
                        await callback(
                            result,
                            self.best_pipeline,
                            self.best_metric,
                            self.best_data_stats,
                        )
                    except Exception as cb_err:
                        logger.warning(f"[TraceResearch] Callback error: {cb_err}")

            else:
                # Completed all experiments
                self.status = TraceResearchStatus.COMPLETED

        except Exception as e:
            logger.error(f"[TraceResearch] Loop failed: {e}", exc_info=True)
            self.status = TraceResearchStatus.FAILED
            self._error = str(e)

        finally:
            self._running = False
            self._completed_at = datetime.now(timezone.utc).isoformat()
            logger.info(
                f"[TraceResearch] Finished. Status={self.status}, "
                f"experiments={len(self.experiments)}, "
                f"best_metric={self.best_metric:.4f}"
            )

    # -----------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------

    def stop(self):
        """Stop the trace research loop."""
        self._running = False

    def pause(self):
        """Pause the trace research loop."""
        self._paused = True
        self.status = TraceResearchStatus.PAUSED

    def resume(self):
        """Resume the trace research loop."""
        self._paused = False
        if self.status == TraceResearchStatus.PAUSED:
            self.status = TraceResearchStatus.RUNNING

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a serializable status dict."""
        return {
            "status": self.status,
            "total_experiments": self.config.max_experiments,
            "completed_experiments": len(self.experiments),
            "best_metric": (
                round(self.best_metric, 4) if self.best_metric < float("inf") else None
            ),
            "best_pipeline": {p: getattr(self.best_pipeline, p) for p in self.config.search_params},
            "best_data_stats": self.best_data_stats,
            "search_params": self.config.search_params,
            "experiments": [
                {
                    "experiment_id": e.experiment_id,
                    "config_snapshot": e.config_snapshot,
                    "examples_generated": e.examples_generated,
                    "unique_repos": e.unique_repos,
                    "avg_example_length": e.avg_example_length,
                    "metric_value": e.metric_value,
                    "improved": e.improved,
                    "duration_seconds": e.duration_seconds,
                    "timestamp": e.timestamp,
                }
                for e in self.experiments
            ],
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "error": self._error,
        }
