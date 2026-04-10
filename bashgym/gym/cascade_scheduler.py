"""
Cascade RL Scheduler — Sequential domain-by-domain RL training.

Inspired by Nemotron Cascade 2: trains each coding domain independently
with tailored reward functions, then distills domain experts into a
unified student via MOPD.

Domains are trained sequentially — each stage's final checkpoint becomes
the base model for the next stage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =========================================================================
# Domain Taxonomy
# =========================================================================


@dataclass
class CascadeDomain:
    """A training domain for cascade RL."""

    name: str
    description: str
    reward_mode: str  # "syntax" | "execution" | "verification"
    tool_filter: list[str]  # Tool names that indicate this domain
    min_steps: int = 1  # Minimum steps for a segment to belong to this domain

    def matches(self, example: dict[str, Any]) -> bool:
        """Check if a training example belongs to this domain.

        Checks both formats:
        - messages[].tool_calls[].function.name (OpenAI/NeMo format)
        - trace[].tool_name (BashGym native trace format)
        """
        tools_used: set[str] = set()
        step_count = 0

        # Check messages format (OpenAI/NeMo training examples)
        messages = example.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tools_used.add(fn.get("name", ""))
                step_count += 1

        # Check trace format (BashGym native traces)
        trace = example.get("trace", [])
        for step in trace:
            tool = step.get("tool_name") or step.get("tool") or ""
            if tool:
                tools_used.add(tool)
            step_count += 1

        # Check tool presence
        if self.tool_filter:
            if not tools_used.intersection(set(self.tool_filter)):
                return False

        # Check minimum steps
        if step_count < self.min_steps:
            return False

        return True


DOMAIN_TAXONOMY: dict[str, CascadeDomain] = {
    "file_operations": CascadeDomain(
        name="file_operations",
        description="Read, write, and edit files — core file manipulation",
        reward_mode="syntax",
        tool_filter=["Read", "Write", "Edit"],
        min_steps=1,
    ),
    "bash_commands": CascadeDomain(
        name="bash_commands",
        description="Shell execution, scripting, and system commands",
        reward_mode="execution",
        tool_filter=["Bash"],
        min_steps=1,
    ),
    "search_and_navigate": CascadeDomain(
        name="search_and_navigate",
        description="Grep, glob, codebase exploration and navigation",
        reward_mode="execution",
        tool_filter=["Grep", "Glob"],
        min_steps=1,
    ),
    "multi_step_reasoning": CascadeDomain(
        name="multi_step_reasoning",
        description="Planning and multi-tool reasoning chains",
        reward_mode="verification",
        tool_filter=[],  # Any tools — but must have many steps
        min_steps=5,
    ),
}


# =========================================================================
# Repo-Based Domains
# =========================================================================


@dataclass
class RepoCascadeDomain(CascadeDomain):
    """Domain that matches training examples by repo name rather than tool usage.

    Used for repo-by-repo cascade RL: each repository gets its own training
    stage, allowing the model to specialize per-project before unification.
    """

    repo_names: list[str] = field(default_factory=list)

    def matches(self, example: dict[str, Any]) -> bool:
        """Match by repo name in metadata, with min_steps check."""
        # Check both top-level and metadata.primary_repo (format varies)
        primary_repo = example.get("primary_repo") or example.get("metadata", {}).get("primary_repo", {})
        name = primary_repo.get("name", "") if isinstance(primary_repo, dict) else ""

        if self.repo_names and name not in self.repo_names:
            return False

        # Count steps from either format
        step_count = len(example.get("trace", []))
        if not step_count:
            step_count = sum(
                1 for m in example.get("messages", []) if m.get("role") == "assistant"
            )

        return step_count >= self.min_steps


def build_repo_domains(
    gold_traces_dir: Path,
    min_examples: int = 10,
) -> dict[str, RepoCascadeDomain]:
    """Scan gold traces and create a CascadeDomain per repo with enough examples.

    Args:
        gold_traces_dir: Directory containing gold trace JSON files.
        min_examples: Minimum traces per repo to create a domain.

    Returns:
        Dict mapping domain name to RepoCascadeDomain.
    """
    from collections import Counter

    repo_counts: Counter[str] = Counter()

    for trace_file in gold_traces_dir.glob("*.json"):
        try:
            data = json.loads(trace_file.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict):
                continue
            # Check both top-level and metadata.primary_repo (format varies)
            primary_repo = data.get("primary_repo") or data.get("metadata", {}).get("primary_repo", {})
            name = primary_repo.get("name", "") if isinstance(primary_repo, dict) else ""
            if name:
                repo_counts[name] += 1
        except (json.JSONDecodeError, OSError):
            continue

    domains: dict[str, RepoCascadeDomain] = {}
    for repo_name, count in repo_counts.most_common():
        if count < min_examples:
            continue
        domain_key = f"repo_{repo_name}"
        domains[domain_key] = RepoCascadeDomain(
            name=domain_key,
            description=f"Repo-specific training for {repo_name} ({count} traces)",
            reward_mode="verification",
            tool_filter=[],
            min_steps=2,
            repo_names=[repo_name],
        )
        logger.info("[Cascade] Repo domain '%s': %d traces", domain_key, count)

    return domains


# =========================================================================
# Stage and Config Dataclasses
# =========================================================================


@dataclass
class CascadeStage:
    """A single stage in the cascade RL pipeline."""

    domain: CascadeDomain
    stage_number: int
    base_model: str
    output_path: Path
    run_id: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    checkpoint_path: Path | None = None
    error_message: str | None = None
    examples_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.name,
            "stage_number": self.stage_number,
            "base_model": self.base_model,
            "output_path": str(self.output_path),
            "run_id": self.run_id,
            "status": self.status,
            "metrics": self.metrics,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "error_message": self.error_message,
            "examples_count": self.examples_count,
        }


@dataclass
class CascadeConfig:
    """Configuration for a cascade RL training run."""

    domains: list[str] = field(default_factory=lambda: list(DOMAIN_TAXONOMY.keys()))
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    dataset_path: Path = field(default_factory=lambda: Path("data/gold_traces"))
    output_dir: Path = field(default_factory=lambda: Path.home() / ".bashgym" / "cascade")

    # Per-stage GRPO settings
    grpo_num_generations: int = 4
    grpo_temperature: float = 0.7
    train_steps_per_stage: int = 200
    learning_rate: float = 2e-5

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = True

    # Cascade control
    early_stopping_patience: int = 3
    min_reward_improvement: float = 0.05
    skip_empty_domains: bool = True  # Skip if domain has 0 examples
    min_domain_examples: int = 10  # Minimum examples to train on

    # Execution
    use_remote_ssh: bool = False
    mode: str = "real"  # "real" or "simulate"

    # Repo-based domains
    repo_domains_enabled: bool = False  # Build domains from repo names in gold traces
    repo_domains_dir: str = ""  # Path to gold traces for repo scanning
    weakest_first: bool = False  # Sort domains by val loss (weakest trains first)

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)


@dataclass
class CascadeResult:
    """Result of a complete cascade RL run."""

    stages: list[CascadeStage]
    best_checkpoints: dict[str, Path]  # domain -> checkpoint path
    total_duration_seconds: float
    status: str  # "completed" | "failed" | "stopped"
    unified_model_path: Path | None = None  # After MOPD distillation


# =========================================================================
# MOPD — Multi-domain On-Policy Distillation
# =========================================================================


@dataclass
class MOPDConfig:
    """Configuration for Multi-domain On-Policy Distillation.

    MOPD takes the best checkpoint from each domain-specific cascade RL stage
    and distills them into a single unified student model. Each domain checkpoint
    acts as a "teacher" for its domain's data.
    """

    domain_checkpoints: dict[str, str] = field(
        default_factory=dict
    )  # domain_name -> checkpoint path
    domain_datasets: dict[str, str] = field(
        default_factory=dict
    )  # domain_name -> filtered dataset path
    student_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Base model for student
    distillation_alpha: float = 0.5  # Balance between soft (teacher) and hard (ground truth) labels
    temperature: float = 2.0  # Softmax temperature for distillation
    train_steps: int = 500
    learning_rate: float = 2e-5
    lora_r: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = True
    output_path: Path = field(default_factory=lambda: Path.home() / ".bashgym" / "cascade" / "mopd")
    use_remote_ssh: bool = False

    def __post_init__(self):
        self.output_path = Path(self.output_path)


# =========================================================================
# Cascade Scheduler
# =========================================================================


class CascadeScheduler:
    """Orchestrates sequential domain-by-domain RL training.

    Cascade training flow:
    1. Filter dataset by domain
    2. Run GRPO for each domain with domain-specific reward
    3. Chain checkpoints: stage N output -> stage N+1 base model
    4. Track per-domain metrics and checkpoints
    5. Optional: MOPD distillation to merge domain experts
    """

    def __init__(self, config: CascadeConfig):
        self.config = config
        self.stages: list[CascadeStage] = []
        self.status: str = "idle"  # idle | running | completed | failed | stopped
        self._running = False
        self._started_at: str | None = None
        self._completed_at: str | None = None
        self._error: str | None = None
        self._current_stage: int = 0

        # Build domain taxonomy: merge tool-based and repo-based domains
        available_domains: dict[str, CascadeDomain] = dict(DOMAIN_TAXONOMY)

        if config.repo_domains_enabled:
            repo_dir = Path(config.repo_domains_dir) if config.repo_domains_dir else config.dataset_path
            if repo_dir.exists():
                repo_domains = build_repo_domains(
                    repo_dir, min_examples=config.min_domain_examples
                )
                available_domains.update(repo_domains)
                # If repo domains are enabled and found, use them as the domain list
                if repo_domains:
                    config.domains = list(repo_domains.keys())
                    logger.info(
                        "[Cascade] Using %d repo-based domains: %s",
                        len(repo_domains),
                        list(repo_domains.keys()),
                    )

        # Build stages from domain list
        for i, domain_name in enumerate(config.domains):
            domain = available_domains.get(domain_name)
            if domain is None:
                raise ValueError(
                    f"Unknown domain: {domain_name}. "
                    f"Available: {list(available_domains.keys())}"
                )

            stage = CascadeStage(
                domain=domain,
                stage_number=i + 1,
                base_model=config.base_model,  # Will be updated during execution
                output_path=config.output_dir / f"stage_{i + 1}_{domain_name}",
                run_id=f"cascade-{domain_name}-{int(time.time())}",
            )
            self.stages.append(stage)

    def sort_domains_by_loss(
        self,
        val_examples: list[dict[str, Any]],
        loss_fn: Callable[[list[dict]], float] | None = None,
    ) -> None:
        """Re-order stages so the highest-loss (weakest) domain trains first.

        Args:
            val_examples: Validation examples for loss scoring.
            loss_fn: Function that takes a list of message dicts and returns loss.
                     If None, stages keep their current order.
        """
        if not loss_fn or not val_examples:
            return

        domain_losses: dict[str, float] = {}
        for stage in self.stages:
            # Filter val examples for this domain
            domain_examples = [ex for ex in val_examples if stage.domain.matches(ex)]
            if not domain_examples:
                domain_losses[stage.domain.name] = 0.0
                continue

            # Average loss across domain examples
            losses = []
            for ex in domain_examples[:20]:  # Cap to avoid slowness
                msgs = ex.get("messages", [])
                if msgs:
                    try:
                        losses.append(loss_fn(msgs))
                    except Exception:
                        continue
            domain_losses[stage.domain.name] = (
                sum(losses) / len(losses) if losses else 0.0
            )

        # Sort stages: highest loss (weakest) first
        self.stages.sort(
            key=lambda s: domain_losses.get(s.domain.name, 0.0),
            reverse=True,
        )

        # Renumber stages
        for i, stage in enumerate(self.stages):
            stage.stage_number = i + 1

        logger.info(
            "[Cascade] Weakest-first order: %s",
            [(s.domain.name, round(domain_losses.get(s.domain.name, 0), 4)) for s in self.stages],
        )

    async def run_cascade(
        self,
        callback: Callable | None = None,
    ) -> CascadeResult:
        """Execute the cascade RL pipeline — all stages sequentially."""
        from datetime import datetime, timezone

        self.status = "running"
        self._running = True
        self._started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        best_checkpoints: dict[str, Path] = {}
        prev_checkpoint: str | None = None

        try:
            for stage in self.stages:
                if not self._running:
                    self.status = "stopped"
                    break

                self._current_stage = stage.stage_number

                # Update base model to previous stage's output
                if prev_checkpoint:
                    stage.base_model = prev_checkpoint

                # Filter dataset for this domain
                filtered_path = self._filter_dataset(stage.domain)
                stage.examples_count = self._count_jsonl(filtered_path)

                # Skip if too few examples
                if stage.examples_count < self.config.min_domain_examples:
                    logger.info(
                        f"[Cascade] Skipping {stage.domain.name}: "
                        f"only {stage.examples_count} examples "
                        f"(min: {self.config.min_domain_examples})"
                    )
                    stage.status = "skipped"
                    if callback:
                        await callback("stage-skipped", stage)
                    continue

                # Run the stage
                stage.status = "running"
                stage.started_at = datetime.now(timezone.utc).isoformat()

                if callback:
                    await callback("stage-started", stage)

                try:
                    if self.config.mode == "simulate":
                        metrics = await self._simulate_stage(stage)
                    else:
                        metrics = await self._run_stage(stage, filtered_path)

                    stage.metrics = metrics
                    stage.status = "completed"
                    stage.completed_at = datetime.now(timezone.utc).isoformat()

                    # Track checkpoint
                    checkpoint = stage.output_path / "final"
                    if checkpoint.exists():
                        stage.checkpoint_path = checkpoint
                        best_checkpoints[stage.domain.name] = checkpoint
                        prev_checkpoint = str(checkpoint)
                    elif self.config.mode == "simulate":
                        # In simulate mode, use base model as "checkpoint"
                        stage.checkpoint_path = stage.output_path
                        best_checkpoints[stage.domain.name] = stage.output_path
                        prev_checkpoint = stage.base_model

                    logger.info(
                        f"[Cascade] Stage {stage.stage_number} ({stage.domain.name}) "
                        f"completed: reward={metrics.get('mean_reward', 'N/A')}"
                    )

                except Exception as e:
                    stage.status = "failed"
                    stage.error_message = str(e)
                    stage.completed_at = datetime.now(timezone.utc).isoformat()
                    logger.error(f"[Cascade] Stage {stage.stage_number} failed: {e}")

                    if callback:
                        await callback("stage-failed", stage)

                    # Continue to next stage (don't abort entire cascade)
                    continue

                if callback:
                    await callback("stage-completed", stage)

            if self._running:
                self.status = "completed"

        except Exception as e:
            self.status = "failed"
            self._error = str(e)
            logger.error(f"[Cascade] Failed: {e}")

        self._running = False
        self._completed_at = datetime.now(timezone.utc).isoformat()
        duration = time.time() - start_time

        return CascadeResult(
            stages=self.stages,
            best_checkpoints=best_checkpoints,
            total_duration_seconds=duration,
            status=self.status,
        )

    async def _run_stage(self, stage: CascadeStage, dataset_path: Path) -> dict[str, Any]:
        """Run a single GRPO stage for a domain."""
        from bashgym.gym.trainer import GRPOTrainer, TrainerConfig, TrainingStrategy

        config = TrainerConfig(
            base_model=stage.base_model,
            strategy=TrainingStrategy.GRPO,
            grpo_num_generations=self.config.grpo_num_generations,
            grpo_temperature=self.config.grpo_temperature,
            grpo_reward_mode=stage.domain.reward_mode,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.train_steps_per_stage,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            load_in_4bit=self.config.load_in_4bit,
            output_dir=str(stage.output_path),
            use_remote_ssh=self.config.use_remote_ssh,
        )

        trainer = GRPOTrainer(config)

        run = await asyncio.to_thread(
            trainer.train_grpo,
            dataset_path=dataset_path,
            verifier_fn=lambda p, r: 0.0,  # Reward is in the generated script
            run_id=stage.run_id,
            training_metadata={
                "task_domain": stage.domain.name,
                "cascade_stage": stage.stage_number,
                "cascade_run_id": stage.run_id,
            },
        )

        return run.metrics

    async def _simulate_stage(self, stage: CascadeStage) -> dict[str, Any]:
        """Simulate a cascade stage for testing (no actual training)."""
        import random

        await asyncio.sleep(random.uniform(1.0, 3.0))

        base_reward = {"syntax": 0.7, "execution": 0.5, "verification": 0.3}.get(
            stage.domain.reward_mode, 0.5
        )
        improvement = random.uniform(0.0, 0.2) * (1 + stage.stage_number * 0.1)

        stage.output_path.mkdir(parents=True, exist_ok=True)

        return {
            "mean_reward": round(base_reward + improvement, 4),
            "final_loss": round(random.uniform(1.5, 2.5), 4),
            "steps": self.config.train_steps_per_stage,
            "domain": stage.domain.name,
            "reward_mode": stage.domain.reward_mode,
            "simulated": True,
        }

    def _filter_dataset(self, domain: CascadeDomain) -> Path:
        """Filter the training dataset for a specific domain.

        Reads both .json (one trace per file) and .jsonl (one trace per line).
        """
        filtered: list[dict[str, Any]] = []
        dataset_path = self.config.dataset_path

        if dataset_path.is_dir():
            # Read .json files (BashGym trace format — one JSON object per file)
            for json_file in dataset_path.glob("*.json"):
                try:
                    example = json.loads(json_file.read_text(encoding="utf-8", errors="replace"))
                    if domain.matches(example):
                        filtered.append(example)
                except (json.JSONDecodeError, OSError):
                    continue

            # Read .jsonl files (NeMo training format — one JSON per line)
            for jsonl_file in dataset_path.glob("*.jsonl"):
                try:
                    with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                example = json.loads(line)
                                if domain.matches(example):
                                    filtered.append(example)
                            except json.JSONDecodeError:
                                continue
                except OSError:
                    continue
        elif dataset_path.is_file():
            with open(dataset_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                        if domain.matches(example):
                            filtered.append(example)
                    except json.JSONDecodeError:
                        continue

        # Convert raw traces → GRPO format using the dataset converter
        from bashgym.datasets.converters import trace_to_grpo_example
        from bashgym.datasets.validator import validate_dataset

        # Detect if the base model is multimodal (Gemma 4 family) — needs list-of-parts content
        base = (self.config.base_model or "").lower()
        multimodal = "gemma-4" in base or "gemma4" in base

        grpo_examples = []
        skipped = 0
        for trace in filtered:
            ex = trace_to_grpo_example(trace, multimodal_format=multimodal)
            if ex is not None:
                grpo_examples.append(ex)
            else:
                skipped += 1
        logger.info(
            f"[Cascade] trace→GRPO ({domain.name}): {len(grpo_examples)} converted, "
            f"{skipped} skipped (multimodal_format={multimodal})"
        )

        # Write filtered dataset
        output_dir = self.config.output_dir / "filtered"
        output_dir.mkdir(parents=True, exist_ok=True)
        filtered_path = output_dir / f"{domain.name}.jsonl"

        with open(filtered_path, "w") as f:
            for example in grpo_examples:
                f.write(json.dumps(example) + "\n")

        # Validate the dataset before training touches it
        result = validate_dataset(filtered_path, format="grpo", quiet=True)
        if not result.is_valid:
            logger.error(
                f"[Cascade] {domain.name} dataset failed validation: "
                f"{result.error_count} errors, {result.valid_examples}/{result.total_examples} valid"
            )
            for issue in result.issues[:5]:
                logger.error(f"  {issue.severity} line {issue.line}: {issue.field} — {issue.message}")
        else:
            logger.info(
                f"[Cascade] Filtered {len(filtered)} traces → "
                f"{len(grpo_examples)} GRPO examples for {domain.name} (validated ✓)"
            )

        return filtered_path

    @staticmethod
    def _count_jsonl(path: Path) -> int:
        """Count lines in a JSONL file."""
        if not path.exists():
            return 0
        with open(path) as f:
            return sum(1 for line in f if line.strip())

    def stop(self):
        """Stop the cascade after the current stage completes."""
        self._running = False

    def get_status(self) -> dict[str, Any]:
        """Get current cascade status."""
        return {
            "status": self.status,
            "current_stage": self._current_stage,
            "total_stages": len(self.stages),
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "error": self._error,
            "stages": [s.to_dict() for s in self.stages],
            "domains": [s.domain.name for s in self.stages],
        }

    def create_mopd_config(
        self,
        cascade_result: CascadeResult,
        student_model: str | None = None,
        **kwargs: Any,
    ) -> MOPDConfig:
        """Create MOPD config from cascade results.

        Automatically maps domain checkpoints and filtered datasets
        from the completed cascade stages.
        """
        domain_checkpoints = {}
        domain_datasets = {}

        for stage in cascade_result.stages:
            if stage.status != "completed" or stage.checkpoint_path is None:
                continue
            domain_checkpoints[stage.domain.name] = str(stage.checkpoint_path)
            # Filtered dataset was written during cascade
            filtered_path = self.config.output_dir / "filtered" / f"{stage.domain.name}.jsonl"
            if filtered_path.exists():
                domain_datasets[stage.domain.name] = str(filtered_path)

        return MOPDConfig(
            domain_checkpoints=domain_checkpoints,
            domain_datasets=domain_datasets,
            student_model=student_model or self.config.base_model,
            output_path=self.config.output_dir / "mopd",
            use_remote_ssh=self.config.use_remote_ssh,
            **kwargs,
        )


# =========================================================================
# MOPD Distillation — Standalone Function
# =========================================================================


async def distill_cascade(
    config: MOPDConfig,
    callback: Callable | None = None,
) -> dict[str, Any]:
    """Distill domain-specific cascade checkpoints into a unified student.

    MOPD (Multi-domain On-Policy Distillation) flow:
    1. For each domain checkpoint, load its filtered dataset
    2. Combine all domain datasets into a unified distillation dataset
    3. Run offline distillation: student learns from all domain teachers

    Uses the existing DISTILLATION training strategy (offline mode).
    Each domain checkpoint acts as teacher for its domain's data subset.

    Args:
        config: MOPD configuration with domain checkpoints and settings
        callback: Optional progress callback

    Returns:
        Dict with distillation results (run metrics, output path)
    """
    from bashgym.gym.trainer import Trainer, TrainerConfig, TrainingStrategy

    logger.info(
        f"[MOPD] Starting distillation: {len(config.domain_checkpoints)} domains, "
        f"student={config.student_model}"
    )

    if not config.domain_checkpoints:
        raise ValueError("No domain checkpoints provided for MOPD")

    # Step 1: Combine domain datasets into unified distillation dataset
    config.output_path.mkdir(parents=True, exist_ok=True)
    combined_path = config.output_path / "combined_distillation.jsonl"

    total_examples = 0
    domain_counts: dict[str, int] = {}

    with open(combined_path, "w") as out_f:
        for domain_name, dataset_path in config.domain_datasets.items():
            ds_path = Path(dataset_path)
            if not ds_path.exists():
                logger.warning(f"[MOPD] Dataset not found for {domain_name}: {ds_path}")
                continue

            count = 0
            with open(ds_path) as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                        # Tag with domain for provenance
                        example["_mopd_domain"] = domain_name
                        example["_mopd_teacher"] = str(
                            config.domain_checkpoints.get(domain_name, "")
                        )
                        out_f.write(json.dumps(example) + "\n")
                        count += 1
                    except json.JSONDecodeError:
                        continue

            domain_counts[domain_name] = count
            total_examples += count
            logger.info(f"[MOPD] Added {count} examples from {domain_name}")

    if total_examples == 0:
        raise ValueError("No training examples found across all domains")

    logger.info(
        f"[MOPD] Combined dataset: {total_examples} examples " f"from {len(domain_counts)} domains"
    )

    if callback:
        await callback(
            "mopd-dataset-ready",
            {
                "total_examples": total_examples,
                "domain_counts": domain_counts,
            },
        )

    # Step 2: Run distillation using existing DISTILLATION strategy
    # Use the first available domain checkpoint as the "teacher model" for the script
    # (In offline MOPD, the teacher outputs are already in the dataset)
    first_teacher = next(iter(config.domain_checkpoints.values()), config.student_model)

    trainer_config = TrainerConfig(
        base_model=config.student_model,
        strategy=TrainingStrategy.DISTILLATION,
        teacher_model=first_teacher,
        teacher_temperature=config.temperature,
        distillation_alpha=config.distillation_alpha,
        on_policy_distillation=False,  # Offline MOPD
        learning_rate=config.learning_rate,
        max_steps=config.train_steps,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        load_in_4bit=config.load_in_4bit,
        output_dir=str(config.output_path / "training"),
        use_remote_ssh=config.use_remote_ssh,
        auto_export_gguf=True,
    )

    trainer = Trainer(trainer_config)

    if callback:
        await callback(
            "mopd-training-started",
            {
                "student_model": config.student_model,
                "teacher_checkpoints": list(config.domain_checkpoints.keys()),
                "train_steps": config.train_steps,
            },
        )

    # Run distillation (synchronous, in thread to not block event loop)
    try:
        run = await asyncio.to_thread(
            trainer.train_distillation,
            dataset_path=combined_path,
            run_id=f"mopd-{int(time.time())}",
            training_metadata={
                "cascade_mopd": True,
                "domain_checkpoints": {k: str(v) for k, v in config.domain_checkpoints.items()},
                "domain_counts": domain_counts,
                "total_examples": total_examples,
            },
        )

        result = {
            "status": "completed",
            "run_id": run.run_id,
            "output_path": str(run.output_path),
            "metrics": run.metrics,
            "domain_counts": domain_counts,
            "total_examples": total_examples,
        }

        if callback:
            await callback("mopd-completed", result)

        logger.info(f"[MOPD] Distillation complete: {run.run_id}")
        return result

    except Exception as e:
        error_result = {
            "status": "failed",
            "error": str(e),
            "domain_counts": domain_counts,
            "total_examples": total_examples,
        }

        if callback:
            await callback("mopd-failed", error_result)

        logger.error(f"[MOPD] Distillation failed: {e}")
        return error_result
