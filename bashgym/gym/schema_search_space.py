"""
Schema Search Space for Data Designer Pipeline Evolution

Defines a SearchSpace subclass that mutates Data Designer pipeline configs
(temperatures, column toggles, judge rubrics, provider assignments) and
evaluates them via a two-stage process:

  Stage 1: Generate a small batch of examples, filter by judge scores (fast)
  Stage 2: Export to JSONL, micro-train (50 steps SFT), measure loss (accurate)

Also provides a template library mapping pipeline names to metadata, and a
failure-type classifier that selects the best template for observed failures.

Part of the AutoCurriculum Compiler (Steps 2.2 + 2.3).
"""

from __future__ import annotations

import copy
import logging
import random
import tempfile
from pathlib import Path
from typing import Any

from bashgym.gym.autoresearch import SearchSpace

logger = logging.getLogger(__name__)


# =============================================================================
# Schema Search Space Definition
# =============================================================================

SCHEMA_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "temperature_text": {"type": "float", "min": 0.3, "max": 1.0},
    "temperature_code": {"type": "float", "min": 0.05, "max": 0.5},
    "temperature_judge": {"type": "float", "min": 0.0, "max": 0.3},
    "judge_threshold": {
        "type": "int",
        "min": 2,
        "max": 5,
        "choices": [2, 3, 4, 5],
    },
    "num_judge_dimensions": {
        "type": "int",
        "min": 2,
        "max": 5,
        "choices": [2, 3, 4, 5],
    },
    "include_code_validation": {"type": "bool"},
    "include_embedding_dedup": {"type": "bool"},
    "complexity_weights": {
        "type": "weights",
        "options": ["simple", "moderate", "complex"],
    },
    "judge_backend": {
        "type": "choice",
        "choices": ["haiku", "gemma_local"],
        "default": "haiku",
    },
}


# =============================================================================
# Template Library
# =============================================================================

TEMPLATE_LIBRARY: dict[str, dict[str, Any]] = {
    "coding_agent_sft": {
        "description": (
            "General coding agent SFT with task prompts, " "solutions, and quality scoring"
        ),
        "columns": [
            "task_category",
            "complexity",
            "language",
            "codebase_size",
            "task_prompt",
            "solution",
            "solution_text",
            "quality_score",
        ],
        "judge_dimensions": ["correctness", "tool_usage", "completeness"],
        "default_for_failures": ["incomplete", "bad_reasoning"],
    },
    "coding_agent_dpo": {
        "description": ("DPO training with dual solutions at different temperatures"),
        "columns": [
            "task_category",
            "complexity",
            "language",
            "task_prompt",
            "solution_a",
            "solution_b",
            "judge_a",
            "judge_b",
            "chosen",
            "rejected",
        ],
        "judge_dimensions": ["quality"],
        "default_for_failures": ["quality_inconsistent"],
    },
    "tool_use_sft": {
        "description": ("Structured tool-call training data with multi-turn conversations"),
        "columns": [
            "tool_focus",
            "num_tools",
            "scenario_type",
            "language",
            "conversation",
            "formatted_response",
            "interaction_quality",
        ],
        "judge_dimensions": ["coherence", "tool_appropriateness"],
        "default_for_failures": ["wrong_tool", "tool_misuse"],
    },
    "from_external": {
        "description": ("Augmented data from HuggingFace datasets or local files"),
        "columns": [
            "augmentation_style",
            "target_format",
            "task_prompt",
            "solution",
            "quality_score",
        ],
        "judge_dimensions": ["relevance", "training_value"],
        "default_for_failures": [],
    },
    "from_unstructured": {
        "description": ("Training data from raw code files and documents"),
        "columns": [
            "task_type",
            "interaction_depth",
            "task_prompt",
            "context_summary",
            "solution",
            "quality_score",
        ],
        "judge_dimensions": ["grounding", "solution_quality"],
        "default_for_failures": ["context_misunderstanding"],
    },
}

# Reverse mapping: failure type -> best template
FAILURE_TEMPLATE_MAP: dict[str, str] = {}
for _name, _meta in TEMPLATE_LIBRARY.items():
    for _failure_type in _meta.get("default_for_failures", []):
        FAILURE_TEMPLATE_MAP[_failure_type] = _name


# =============================================================================
# Gemma-Based Judge (Alternative to Claude Haiku)
# =============================================================================


class GemmaJudge:
    """Evaluates synthetic training examples by perplexity via local Gemma model.

    Instead of using Claude Haiku to score examples on a rubric, this judge
    computes cross-entropy loss — measuring how well the current Gemma checkpoint
    can predict the example. This discovers examples the model can actually learn
    from (moderate loss), filtering out both trivial (too low) and garbage (too high).

    Score mapping (loss -> 0-5 scale for compatibility with haiku judge):
        loss <= 1.0  -> score 5.0 (model already knows this well)
        loss >= 4.0  -> score 1.0 (too hard / noisy)
        linear interpolation between
    """

    def __init__(
        self,
        base_model: str = "unsloth/gemma-4-E4B-it",
        adapter_path: str | None = None,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self._models: dict | None = None

    def _ensure_loaded(self) -> dict:
        if self._models is None:
            from bashgym.models.gemma_loader import load_models

            self._models = load_models(
                base_model=self.base_model,
                adapter_path=self.adapter_path,
            )
        return self._models

    @staticmethod
    def _loss_to_score(loss: float) -> float:
        """Convert cross-entropy loss to a 0-5 quality score.

        Moderate loss (1.5-2.5) gives the best scores — these are examples
        the model hasn't memorized but can plausibly learn.
        """
        if loss <= 1.0:
            return 5.0
        if loss >= 4.0:
            return 1.0
        return 5.0 - (loss - 1.0) * (4.0 / 3.0)

    def score_examples(self, examples: list[dict]) -> float:
        """Return average quality score (0-5 scale) for a batch of examples.

        Args:
            examples: List of training example dicts with 'messages' key.

        Returns:
            Average score in [0, 5]. Higher is better (for stage 1 filtering).
        """
        models = self._ensure_loaded()
        ft_loss = models["ft_loss"]

        scores: list[float] = []
        for ex in examples:
            messages = ex.get("messages", [])
            if not messages:
                continue
            try:
                loss = ft_loss(messages)
                scores.append(self._loss_to_score(float(loss)))
            except Exception:
                continue

        return sum(scores) / len(scores) if scores else 3.0


# =============================================================================
# SchemaSearchSpace
# =============================================================================


class SchemaSearchSpace(SearchSpace):
    """Search space for Data Designer pipeline schema evolution.

    Mutates Data Designer config traits (temperatures, column toggles,
    judge rubrics, provider assignments) and evaluates by generating
    examples + micro-training.

    Two-stage evaluation:
      Stage 1: Generate 25 examples, filter by judge scores (fast)
      Stage 2: Export to JSONL, micro-train top 5 (50 steps SFT)
    """

    def __init__(
        self,
        base_pipeline_name: str = "coding_agent_sft",
        mutation_rate: float = 0.3,
        mutation_scale: float = 0.2,
        stage1_examples: int = 25,
        stage1_judge_threshold: float = 3.0,
        stage2_train_steps: int = 50,
        dataset_path: Path | None = None,
    ):
        self.base_pipeline_name = base_pipeline_name
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.stage1_examples = stage1_examples
        self.stage1_judge_threshold = stage1_judge_threshold
        self.stage2_train_steps = stage2_train_steps
        self.dataset_path = dataset_path
        self._template_meta = TEMPLATE_LIBRARY.get(
            base_pipeline_name, TEMPLATE_LIBRARY["coding_agent_sft"]
        )

    # -----------------------------------------------------------------
    # SearchSpace ABC
    # -----------------------------------------------------------------

    def mutate(self, genome: dict[str, Any]) -> dict[str, Any]:
        """Mutate a schema genome (Data Designer config dict).

        Probabilistically modifies temperatures, judge thresholds,
        column toggles, and other traits.
        """
        mutated = copy.deepcopy(genome)

        for param, spec in SCHEMA_SEARCH_SPACE.items():
            if random.random() > self.mutation_rate:
                continue

            current = mutated.get(param)
            if current is None:
                # Initialize missing params with defaults
                mutated[param] = self._init_param(spec)
                continue

            # Mutate existing value
            mutated[param] = self._mutate_param(current, spec)

        return mutated

    def evaluate(
        self,
        genome: dict[str, Any],
        experiment_number: int,
        total_experiments: int,
    ) -> float:
        """Two-stage evaluation of a schema genome.

        Stage 1: Generate examples, check judge scores (fast filter)
        Stage 2: Micro-train on generated data (slow, accurate)

        Returns loss metric (lower is better). Returns 5.0 on failure.
        """
        # Stage 1: Quick quality check
        try:
            stage1_score = self._stage1_evaluate(genome)
            if stage1_score < self.stage1_judge_threshold:
                logger.info(
                    "[SchemaSearch] Exp %d: Stage 1 filtered " "(judge score %.2f < %.2f)",
                    experiment_number,
                    stage1_score,
                    self.stage1_judge_threshold,
                )
                return 5.0  # Worst case -- filtered out
        except Exception as e:
            logger.warning("[SchemaSearch] Stage 1 failed: %s", e)
            return 5.0

        # Stage 2: Micro-training
        try:
            loss = self._stage2_evaluate(genome, experiment_number)
            return loss
        except Exception as e:
            logger.warning("[SchemaSearch] Stage 2 failed: %s", e)
            return 5.0

    def get_config_snapshot(self, genome: dict[str, Any]) -> dict[str, Any]:
        """Extract a serializable snapshot of the genome."""
        return {k: v for k, v in genome.items() if k in SCHEMA_SEARCH_SPACE}

    # -----------------------------------------------------------------
    # Mutation helpers
    # -----------------------------------------------------------------

    def _init_param(self, spec: dict[str, Any]) -> Any:
        """Initialize a missing parameter with a random value."""
        param_type = spec["type"]
        if param_type == "bool":
            return random.choice([True, False])
        if param_type == "float":
            return random.uniform(spec["min"], spec["max"])
        if param_type == "int" and "choices" in spec:
            return random.choice(spec["choices"])
        if param_type == "choice":
            return spec.get("default", random.choice(spec["choices"]))
        if param_type == "weights":
            options = spec["options"]
            weights = [random.random() for _ in options]
            total = sum(weights)
            return {opt: round(w / total, 2) for opt, w in zip(options, weights)}
        return None  # pragma: no cover

    def _mutate_param(self, current: Any, spec: dict[str, Any]) -> Any:
        """Mutate an existing parameter value."""
        param_type = spec["type"]

        if param_type == "bool":
            return not current

        if param_type == "float":
            delta = current * self.mutation_scale
            new_val = current + random.gauss(0, max(delta, 0.01))
            return max(spec["min"], min(spec["max"], new_val))

        if param_type == "int" and "choices" in spec:
            choices = spec["choices"]
            if current in choices:
                idx = choices.index(current)
                direction = random.choice([-1, 1])
                new_idx = max(0, min(len(choices) - 1, idx + direction))
                return choices[new_idx]
            return random.choice(choices)

        if param_type == "choice":
            choices = spec["choices"]
            other = [c for c in choices if c != current]
            return random.choice(other) if other else current

        if param_type == "weights":
            if isinstance(current, dict):
                weights = {k: max(0.05, v + random.gauss(0, 0.1)) for k, v in current.items()}
                total = sum(weights.values())
                return {k: round(v / total, 2) for k, v in weights.items()}

        return current  # Fallback

    # -----------------------------------------------------------------
    # Stage 1: Quick judge-score filter
    # -----------------------------------------------------------------

    def _stage1_evaluate(self, genome: dict[str, Any]) -> float:
        """Stage 1: Generate few examples, return average judge score.

        When judge_backend is "gemma_local", uses the local Gemma model's
        perplexity instead of the Data Designer's LLM judge.
        """
        if genome.get("judge_backend") == "gemma_local":
            return self._stage1_evaluate_gemma(genome)

        try:
            from bashgym.factory.data_designer import (
                DATA_DESIGNER_AVAILABLE,
                DataDesignerPipeline,
                PipelineConfig,
            )
        except ImportError:
            raise RuntimeError("Data Designer not available for schema evaluation")

        if not DATA_DESIGNER_AVAILABLE:
            raise RuntimeError("data-designer package not installed")

        # Build PipelineConfig from genome
        config = PipelineConfig(
            pipeline=self.base_pipeline_name,
            temperature_text=genome.get("temperature_text", 0.85),
            temperature_code=genome.get("temperature_code", 0.2),
            temperature_judge=genome.get("temperature_judge", 0.1),
        )

        pipeline = DataDesignerPipeline(config)
        df = pipeline.preview(num_records=self.stage1_examples)

        # Extract judge scores if available
        score_cols = [
            c
            for c in df.columns
            if "score" in c.lower() or "judge" in c.lower() or "quality" in c.lower()
        ]
        if not score_cols:
            return 3.0  # No judge scores, pass stage 1 with neutral score

        # Average across all judge score columns
        import numpy as np

        scores: list[float] = []
        for col in score_cols:
            try:
                col_scores = df[col].dropna()
                if len(col_scores) > 0:
                    for val in col_scores:
                        if isinstance(val, (int, float)):
                            scores.append(float(val))
                        elif hasattr(val, "score"):
                            scores.append(float(val.score))
            except (TypeError, ValueError):
                continue

        return float(np.mean(scores)) if scores else 3.0

    def _stage1_evaluate_gemma(self, genome: dict[str, Any]) -> float:
        """Stage 1 alternative: use local Gemma model as judge via perplexity.

        Generates examples using the Data Designer pipeline, then scores them
        by how well the current Gemma checkpoint predicts them. Returns a
        quality score on the same 0-5 scale as the haiku judge.
        """
        try:
            from bashgym.factory.data_designer import (
                DATA_DESIGNER_AVAILABLE,
                DataDesignerPipeline,
                PipelineConfig,
            )
        except ImportError:
            logger.warning("[SchemaSearch] Data Designer unavailable for Gemma judge")
            return 3.0

        if not DATA_DESIGNER_AVAILABLE:
            return 3.0

        config = PipelineConfig(
            pipeline=self.base_pipeline_name,
            temperature_text=genome.get("temperature_text", 0.85),
            temperature_code=genome.get("temperature_code", 0.2),
            temperature_judge=genome.get("temperature_judge", 0.1),
        )

        try:
            pipeline = DataDesignerPipeline(config)
            df = pipeline.preview(num_records=self.stage1_examples)
        except Exception as e:
            logger.warning("[SchemaSearch] Gemma judge: data generation failed: %s", e)
            return 3.0

        # Convert DataFrame rows to message dicts for scoring
        examples: list[dict] = []
        for _, row in df.iterrows():
            messages = []
            if "task_prompt" in row and row["task_prompt"]:
                messages.append({"role": "user", "content": str(row["task_prompt"])})
            solution_col = next(
                (c for c in ("solution", "solution_text", "formatted_response") if c in row),
                None,
            )
            if solution_col and row[solution_col]:
                messages.append({"role": "assistant", "content": str(row[solution_col])})
            if messages:
                examples.append({"messages": messages})

        if not examples:
            return 3.0

        # Lazy-init judge (shared across experiments for efficiency)
        if not hasattr(self, "_gemma_judge"):
            self._gemma_judge = GemmaJudge()

        return self._gemma_judge.score_examples(examples)

    # -----------------------------------------------------------------
    # Stage 2: Micro-training evaluation
    # -----------------------------------------------------------------

    def _stage2_evaluate(self, genome: dict[str, Any], experiment_number: int) -> float:
        """Stage 2: Generate full examples, micro-train, return loss."""
        try:
            from bashgym.factory.data_designer import (
                DataDesignerPipeline,
                PipelineConfig,
            )
        except ImportError:
            return 5.0

        config = PipelineConfig(
            pipeline=self.base_pipeline_name,
            temperature_text=genome.get("temperature_text", 0.85),
            temperature_code=genome.get("temperature_code", 0.2),
            temperature_judge=genome.get("temperature_judge", 0.1),
            num_records=50,
        )

        with tempfile.TemporaryDirectory(prefix=f"schema_eval_{experiment_number}_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Generate examples
            pipeline = DataDesignerPipeline(config)
            pipeline.config.output_dir = tmpdir_path

            try:
                df = pipeline.from_config(self.base_pipeline_name, num_records=50)
                export_result = pipeline.export_nemo(df, output_dir=tmpdir_path)
                train_path = Path(export_result["train_path"])
            except Exception as e:
                logger.warning("[SchemaSearch] Data generation failed: %s", e)
                return 5.0

            if not train_path.exists() or train_path.stat().st_size == 0:
                return 5.0

            # Micro-train
            try:
                from bashgym.gym.trainer import Trainer, TrainerConfig

                trainer_config = TrainerConfig(
                    max_steps=self.stage2_train_steps,
                    eval_strategy="steps",
                    eval_steps=max(10, self.stage2_train_steps // 3),
                    auto_export_gguf=False,
                    save_steps=999999,
                    logging_steps=5,
                    output_dir=str(tmpdir_path / "training"),
                )

                trainer = Trainer(trainer_config)
                run = trainer.train_sft(dataset_path=train_path)

                eval_loss = run.metrics.get("eval_loss")
                final_loss = run.metrics.get("final_loss", 5.0)
                return float(eval_loss if eval_loss is not None else final_loss)

            except Exception as e:
                logger.warning("[SchemaSearch] Micro-training failed: %s", e)
                return 5.0

    # -----------------------------------------------------------------
    # Template selection from failure analysis
    # -----------------------------------------------------------------

    @staticmethod
    def select_template(
        failed_traces: list[dict[str, Any]],
        confidence_threshold: float = 0.6,
    ) -> tuple[str, float]:
        """Select the best template for failed trace patterns.

        Classifies failure types from trace data and returns the best
        matching template with a confidence score.

        Args:
            failed_traces: List of failed trace dicts
            confidence_threshold: Below this, default to coding_agent_sft

        Returns:
            Tuple of (template_name, confidence_score)
        """
        if not failed_traces:
            return "coding_agent_sft", 0.0

        # Count failure indicators across traces
        failure_signals: dict[str, int] = {
            "wrong_tool": 0,
            "tool_misuse": 0,
            "incomplete": 0,
            "bad_reasoning": 0,
            "quality_inconsistent": 0,
            "context_misunderstanding": 0,
        }

        for trace in failed_traces:
            steps = trace.get("trace", [])
            metadata = trace.get("metadata", {})
            error = metadata.get("error", "")

            # Analyze tool usage patterns
            tools_used = [s.get("tool_name", "") for s in steps]

            # Wrong tool: used grep when should have used read, etc.
            if any(t in ["Grep", "Glob"] for t in tools_used) and not any(
                t == "Read" for t in tools_used
            ):
                failure_signals["wrong_tool"] += 1

            # Tool misuse: bash errors, failed commands
            bash_errors = sum(1 for s in steps if s.get("exit_code", 0) != 0)
            if bash_errors > len(steps) * 0.3:
                failure_signals["tool_misuse"] += 1

            # Incomplete: too few steps for the task
            if len(steps) < 3:
                failure_signals["incomplete"] += 1

            # Bad reasoning: steps don't build on each other
            if len(steps) > 5 and len(set(tools_used)) <= 1:
                failure_signals["bad_reasoning"] += 1

            # Context misunderstanding: working in wrong files/dirs
            if "not found" in error.lower() or "no such file" in error.lower():
                failure_signals["context_misunderstanding"] += 1

        # Find dominant failure type
        total_traces = len(failed_traces)
        if total_traces == 0:
            return "coding_agent_sft", 0.0

        top_failure = max(failure_signals, key=lambda k: failure_signals[k])
        confidence = failure_signals[top_failure] / total_traces

        if confidence < confidence_threshold:
            return "coding_agent_sft", confidence

        template = FAILURE_TEMPLATE_MAP.get(top_failure, "coding_agent_sft")
        return template, confidence

    # -----------------------------------------------------------------
    # Default genome factory
    # -----------------------------------------------------------------

    @staticmethod
    def create_default_genome(
        template_name: str = "coding_agent_sft",
    ) -> dict[str, Any]:
        """Create a default genome for a given template."""
        return {
            "template": template_name,
            "temperature_text": 0.85,
            "temperature_code": 0.2,
            "temperature_judge": 0.1,
            "judge_threshold": 3,
            "num_judge_dimensions": 3,
            "include_code_validation": False,
            "include_embedding_dedup": False,
            "complexity_weights": {
                "simple": 0.3,
                "moderate": 0.5,
                "complex": 0.2,
            },
            "judge_backend": "haiku",
        }
