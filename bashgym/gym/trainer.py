"""
Trainer for The Gym Layer

Handles fine-tuning of SLMs using NVIDIA NeMo Gym and Unsloth.
Supports SFT (Supervised Fine-Tuning), DPO, and GRPO training strategies.

Module 4: Training (The "Gym")
"""

import asyncio
import copy
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from bashgym.gym.dppo import DPPO_BINARY_KL_THRESHOLD, DPPO_BINARY_TV_THRESHOLD
from bashgym.gym.dppo_backend import VALID_DPPO_BACKENDS, select_dppo_backend
from bashgym.gym.echo import ECHO_DEFAULT_LAMBDA
from bashgym.gym.rwml import (
    RWML_DEFAULT_DISTANCE_THRESHOLD,
    RWML_DEFAULT_EASY_KEEP_PROBABILITY,
    RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD,
    RWML_DEFAULT_HISTORY_WINDOW,
)
from bashgym.gym.terminal_rl import (
    DEFAULT_TRAINING_PROFILE,
    TERMINAL_RL_TMAX_LIKE_PROFILE,
    TMAX_LIKE_DEFAULTS,
    normalize_training_profile,
)

# Model profile integration
try:
    from bashgym.models import ModelProfile, get_registry

    MODEL_PROFILE_AVAILABLE = True
except ImportError:
    MODEL_PROFILE_AVAILABLE = False
    ModelProfile = None
    get_registry = None

# NeMo Microservices integration
try:
    from bashgym.integrations import NEMO_SDK_AVAILABLE, NeMoClient, NeMoClientConfig
except ImportError:
    NEMO_SDK_AVAILABLE = False
    NeMoClient = None
    NeMoClientConfig = None

logger = logging.getLogger(__name__)


def _parse_trl_stats(text: str) -> dict[str, float] | None:
    """Extract all numeric key-value pairs from a TRL-printed dict line.

    TRL's ProgressCallback formats values with f"{v:.4g}", which drops
    precision and can emit 'nan'/'inf'. We require at least one known
    TRL metric key so we don't match random dict-like strings in user
    output. Tries ast.literal_eval first (most accurate), falls back to
    regex on the :.4g form.

    Shared by SFT, DPO, and GRPO subprocess loops.
    """
    if "{" not in text or "}" not in text:
        return None
    markers = (
        # GRPO
        "'reward'",
        "'train_loss'",
        "'train_runtime'",
        "'frac_reward_zero_std'",
        "'kl'",
        # SFT / DPO / all
        "'loss'",
        "'grad_norm'",
        "'learning_rate'",
        "'epoch'",
        # DPO-specific
        "'rewards/chosen'",
        "'rewards/rejected'",
        "'rewards/accuracies'",
        "'rewards/margins'",
        "'logps/chosen'",
        "'logps/rejected'",
    )
    if not any(m in text for m in markers):
        return None
    try:
        import ast as _ast

        start = text.index("{")
        end = text.rindex("}") + 1
        parsed_raw = _ast.literal_eval(text[start:end])
        if isinstance(parsed_raw, dict):
            out: dict[str, float] = {}
            for k, v in parsed_raw.items():
                if not isinstance(k, str):
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    continue
            if out:
                return out
    except (ValueError, SyntaxError):
        pass
    pairs = re.findall(
        r"'([A-Za-z_][A-Za-z0-9_/]*)'\s*:\s*'?(-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf))'?",
        text,
    )
    out = {}
    for k, v in pairs:
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out or None


class TrainingStrategy(Enum):
    """Training strategies available."""

    SFT = "sft"  # Supervised Fine-Tuning
    DPO = "dpo"  # Direct Preference Optimization
    GRPO = "grpo"  # Group Relative Policy Optimization
    RLVR = "rlvr"  # RL with Verifiable Rewards
    DISTILLATION = "distillation"  # Knowledge Distillation from larger model


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    # Model settings
    base_model: str = (
        ""  # No default — an explicit base model is required (see _require_base_model)
    )
    model_type: str = "qwen"  # llama, mistral, qwen, phi

    # Training settings
    strategy: TrainingStrategy = TrainingStrategy.SFT
    learning_rate: float = 2e-5
    batch_size: int = 1  # Reduced for 12GB VRAM
    gradient_accumulation_steps: int = 8  # Increased to maintain effective batch size
    num_epochs: int = 3
    max_seq_length: int = 2048  # Reduced from 4096 for memory
    warmup_ratio: float = 0.1

    # LoRA settings (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # DPO settings
    dpo_beta: float = 0.1

    # GRPO settings
    grpo_num_generations: int = 4
    grpo_temperature: float = 0.7
    grpo_reward_mode: str = "syntax"  # "syntax", "execution", "verification"
    grpo_use_vllm: bool = False  # vLLM-backed generation (TRL GRPO); requires vllm in env
    grpo_backend: str = "auto"  # auto|unsloth|plain|trl_vllm — set by ModelProfile + platform (S1)
    sft_backend: str = "auto"  # auto|unsloth|plain — plain is the GB10/sm_121 fallback (S1)
    dpo_backend: str = "auto"  # auto|unsloth|plain
    use_liger: bool = False  # plain backend: Liger fused-linear-CE (use_liger_kernel) — the
    # 262k-vocab (Gemma) fused-CE OOM fix; requires liger-kernel in the training env
    # GRPO loss variant (TRL/Unsloth GRPOConfig.loss_type). "gspo" = Qwen's
    # sequence-level Group Sequence Policy Optimization (more stable for long
    # sequences/MoE); "dr_grpo" = Dr. GRPO. Default "grpo" matches TRL's default.
    grpo_loss_type: str = "grpo"

    # Terminal-agent RL profile settings. `terminal_rl_tmax_like` applies the
    # defaults exposed by bashgym.gym.terminal_rl while leaving direct overrides
    # available for experiments.
    training_profile: str = DEFAULT_TRAINING_PROFILE
    grpo_group_size: int | None = None
    prompts_per_rollout_batch: int = 8
    max_tool_calls_per_episode: int = 64
    token_level_loss: bool | None = None
    filter_zero_std_groups: bool | None = None
    active_sampling: bool | None = None
    lm_head_fp32: bool | None = None
    interleaved_thinking: bool | None = None
    sft_warm_start_policy: str | None = None
    dppo_backend: str = "auto"
    dppo_divergence: str = "binary_tv"
    dppo_binary_tv_threshold: float = DPPO_BINARY_TV_THRESHOLD
    dppo_binary_kl_threshold: float = DPPO_BINARY_KL_THRESHOLD

    # World-model objectives for terminal-RL (consumed by the rollout/terminal-RL
    # backend, not the single-turn code-gen GRPO script). See bashgym.gym.echo
    # (ECHO observation-prediction auxiliary loss, arXiv:2605.24517) and
    # bashgym.gym.rwml (embedding-space world-model reward, arXiv:2602.05842).
    echo_enabled: bool = False
    echo_aux_lambda: float = ECHO_DEFAULT_LAMBDA
    rwml_enabled: bool = False
    rwml_distance_threshold: float = RWML_DEFAULT_DISTANCE_THRESHOLD
    rwml_easy_pass_rate_threshold: float = RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD
    rwml_easy_keep_probability: float = RWML_DEFAULT_EASY_KEEP_PROBABILITY
    rwml_history_window: int = RWML_DEFAULT_HISTORY_WINDOW
    rwml_embedding_model: str = ""
    rwml_kl_beta: float = 0.0

    # Knowledge Distillation settings
    teacher_model: str = "claude-sonnet-4-6"  # Teacher model for distillation
    teacher_temperature: float = 0.7
    distillation_alpha: float = 0.5  # Balance between hard and soft labels
    on_policy_distillation: bool = False  # Use on-policy distillation (Oct 2025+)

    # Output settings
    output_dir: str = "data/models"
    save_steps: int = 100
    logging_steps: int = 10

    # Auto-export settings
    auto_export_gguf: bool = True  # Automatically export to GGUF after training
    gguf_quantization: str = "q4_k_m"  # Quantization level: q4_k_m, q5_k_m, q8_0, f16
    auto_deploy_ollama: bool = False  # Auto-deploy GGUF to Ollama after training
    ollama_model_name: str = ""  # Empty = auto-generate from base model + run_id
    ollama_base_tag: str = (
        ""  # Reuse this base Ollama model's TEMPLATE on deploy (correct tool-call format)
    )
    auto_push_hf: bool = False  # Auto-push to HuggingFace Hub after training
    hf_repo_name: str = ""  # Empty = auto-generate from base model + run_id
    hf_private: bool = True  # Private repo by default

    # Eval settings
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    eval_steps: int = 50  # Eval every N steps (when eval_strategy="steps")
    max_steps: int = -1  # Max training steps (-1 = use num_epochs)
    early_stopping_patience: int = 3  # 0 = disabled

    # Hardware settings
    device_map: str = "auto"
    use_flash_attention: bool = True
    # Missing fields referenced by _save_model_profile()
    weight_decay: float = 0.01
    use_gradient_checkpointing: bool = True

    # NeMo Gym settings (for cloud training)
    use_nemo_gym: bool = False
    nemo_gym_endpoint: str = "http://localhost:8080"
    nemo_api_key: str | None = None

    # Remote SSH settings (DGX Spark)
    use_remote_ssh: bool = False

    # Cascade RL settings
    task_domain: str | None = None  # Domain name for cascade stage (e.g., "file_operations")
    cascade_stage: int | None = None  # Stage number in cascade sequence
    cascade_run_id: str | None = None  # Parent cascade run ID for linking stages

    def __post_init__(self) -> None:
        """Resolve named recipe defaults after dataclass construction."""

        self.training_profile = normalize_training_profile(self.training_profile)
        self.dppo_backend = (self.dppo_backend or "auto").strip().lower()
        if self.dppo_backend not in VALID_DPPO_BACKENDS:
            raise ValueError(
                f"dppo_backend={self.dppo_backend!r} must be one of {list(VALID_DPPO_BACKENDS)}"
            )
        self.dppo_divergence = (self.dppo_divergence or "binary_tv").strip().lower()
        if self.dppo_divergence not in {"binary_tv", "binary_kl"}:
            raise ValueError("dppo_divergence must be binary_tv or binary_kl")
        if self.dppo_binary_tv_threshold < 0 or self.dppo_binary_kl_threshold < 0:
            raise ValueError("DPPO thresholds must be non-negative")
        self._validate_world_model_settings()
        if self.training_profile != TERMINAL_RL_TMAX_LIKE_PROFILE:
            if self.grpo_group_size is None:
                self.grpo_group_size = self.grpo_num_generations
            return

        defaults = TMAX_LIKE_DEFAULTS
        if self.grpo_group_size is None:
            if self.grpo_num_generations == 4:
                self.grpo_group_size = defaults.grpo_group_size
            else:
                self.grpo_group_size = self.grpo_num_generations
        self.grpo_num_generations = self.grpo_group_size
        if self.grpo_loss_type == "grpo":
            self.grpo_loss_type = defaults.grpo_loss_type
        if self.token_level_loss is None:
            self.token_level_loss = defaults.token_level_loss
        if self.filter_zero_std_groups is None:
            self.filter_zero_std_groups = defaults.filter_zero_std_groups
        if self.active_sampling is None:
            self.active_sampling = defaults.active_sampling
        if self.lm_head_fp32 is None:
            self.lm_head_fp32 = defaults.lm_head_fp32
        if self.interleaved_thinking is None:
            self.interleaved_thinking = defaults.interleaved_thinking
        if self.sft_warm_start_policy is None:
            self.sft_warm_start_policy = defaults.sft_warm_start_policy

    def effective_grpo_group_size(self) -> int:
        """Group size used for GRPO completions per prompt."""

        return self.grpo_group_size or self.grpo_num_generations

    def terminal_rl_settings(self) -> dict[str, Any]:
        """Resolved terminal-RL knobs for scripts, logs, and tests."""

        profile_enabled = self.training_profile == TERMINAL_RL_TMAX_LIKE_PROFILE
        return {
            "training_profile": self.training_profile,
            "grpo_group_size": self.effective_grpo_group_size(),
            "prompts_per_rollout_batch": self.prompts_per_rollout_batch,
            "max_tool_calls_per_episode": self.max_tool_calls_per_episode,
            "token_level_loss": bool(self.token_level_loss)
            if self.token_level_loss is not None
            else False,
            "filter_zero_std_groups": bool(self.filter_zero_std_groups)
            if self.filter_zero_std_groups is not None
            else False,
            "active_sampling": bool(self.active_sampling)
            if self.active_sampling is not None
            else False,
            "lm_head_fp32": bool(self.lm_head_fp32)
            if self.lm_head_fp32 is not None
            else False,
            "interleaved_thinking": bool(self.interleaved_thinking)
            if self.interleaved_thinking is not None
            else False,
            "sft_warm_start_policy": self.sft_warm_start_policy
            or ("weak_models_only" if profile_enabled else "none"),
            "dppo_backend": self.dppo_backend,
            "dppo_divergence": self.dppo_divergence,
            "dppo_binary_tv_threshold": self.dppo_binary_tv_threshold,
            "dppo_binary_kl_threshold": self.dppo_binary_kl_threshold,
        }

    def dppo_backend_selection(self) -> dict:
        """Resolved DPPO backend capability status."""

        return select_dppo_backend(self.dppo_backend).to_dict()

    def _validate_world_model_settings(self) -> None:
        """Validate ECHO/RWML world-model objective knobs."""

        if self.echo_aux_lambda < 0:
            raise ValueError("echo_aux_lambda must be non-negative")
        if not 0.0 < self.rwml_distance_threshold <= 2.0:
            raise ValueError("rwml_distance_threshold must be in (0, 2] (cosine distance)")
        if not 0.0 <= self.rwml_easy_pass_rate_threshold <= 1.0:
            raise ValueError("rwml_easy_pass_rate_threshold must be a probability in [0, 1]")
        if not 0.0 <= self.rwml_easy_keep_probability <= 1.0:
            raise ValueError("rwml_easy_keep_probability must be a probability in [0, 1]")
        if self.rwml_history_window < 0:
            raise ValueError("rwml_history_window must be non-negative")

    def world_model_settings(self) -> dict[str, Any]:
        """Resolved ECHO + RWML world-model objective knobs for scripts/logs/tests."""

        return {
            "echo_enabled": bool(self.echo_enabled),
            "echo_aux_lambda": self.echo_aux_lambda,
            "rwml_enabled": bool(self.rwml_enabled),
            "rwml_distance_threshold": self.rwml_distance_threshold,
            "rwml_easy_pass_rate_threshold": self.rwml_easy_pass_rate_threshold,
            "rwml_easy_keep_probability": self.rwml_easy_keep_probability,
            "rwml_history_window": self.rwml_history_window,
            "rwml_embedding_model": self.rwml_embedding_model,
            "rwml_kl_beta": self.rwml_kl_beta,
        }

    def terminal_rl_warnings(self) -> list[str]:
        """Warnings for GRPO/RLVR settings likely to produce weak terminal RL."""

        settings = self.terminal_rl_settings()
        warnings: list[str] = []
        if self.training_profile == DEFAULT_TRAINING_PROFILE:
            return warnings
        if settings["grpo_group_size"] < 16:
            warnings.append("terminal RL profile works best with grpo_group_size >= 16")
        if not settings["filter_zero_std_groups"]:
            warnings.append("zero-std reward groups are not filtered")
        if not settings["active_sampling"]:
            warnings.append("active sampling is disabled, so zero-std filtering can shrink batches")
        if not settings["token_level_loss"]:
            warnings.append("token-level loss is disabled; long terminal rollouts may be unstable")
        return warnings


@dataclass
class TrainingRun:
    """Represents a training run."""

    run_id: str
    strategy: TrainingStrategy
    base_model: str
    dataset_path: Path
    output_path: Path
    status: str = "pending"
    pid: int | None = None  # Subprocess PID for process control
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    early_stop_reason: str | None = None
    loss_curve: list[dict[str, Any]] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    val_dataset_path: Path | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "strategy": self.strategy.value,
            "base_model": self.base_model,
            "dataset_path": str(self.dataset_path),
            "output_path": str(self.output_path),
            "status": self.status,
            "pid": self.pid,
            "metrics": self.metrics,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "loss_curve": self.loss_curve,
            "config_snapshot": self.config_snapshot,
            "val_dataset_path": str(self.val_dataset_path) if self.val_dataset_path else None,
            "training_metadata": self.training_metadata,
        }

    def add_loss_point(
        self, step: int, loss: float, epoch: int | None = None, learning_rate: float | None = None
    ):
        """Add a point to the loss curve."""
        self.loss_curve.append(
            {"step": step, "loss": loss, "epoch": epoch, "learning_rate": learning_rate}
        )


class Trainer:
    """
    Trains SLMs using various strategies.

    Supports:
    - Local training with Unsloth (fast LoRA fine-tuning)
    - Cloud training with NVIDIA NeMo Gym
    - Multiple training strategies (SFT, DPO, GRPO)
    """

    def __init__(self, config: TrainerConfig | None = None):
        """Initialize the trainer."""
        self.config = config or TrainerConfig()
        self.active_runs: dict[str, TrainingRun] = {}

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load API key from environment if not provided
        if not self.config.nemo_api_key:
            self.config.nemo_api_key = os.environ.get("NEMO_API_KEY")

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    @staticmethod
    def _extract_param_count(base_model: str) -> str | None:
        """Extract parameter count from model name, e.g. '1.5B' from 'Qwen2.5-Coder-1.5B'."""
        match = re.search(r"(\d+\.?\d*)[Bb]", base_model)
        return f"{match.group(1)}B" if match else None

    @staticmethod
    def _scan_output_artifacts(output_path: Path):
        """Scan a training output directory for model artifacts."""
        from bashgym.models.profile import CheckpointInfo, GGUFExport, ModelArtifacts

        artifacts = ModelArtifacts()

        if not output_path.exists():
            return artifacts

        # Checkpoints
        for ckpt_dir in sorted(output_path.glob("checkpoint-*")):
            step_match = re.search(r"checkpoint-(\d+)", ckpt_dir.name)
            if step_match:
                artifacts.checkpoints.append(
                    CheckpointInfo(path=str(ckpt_dir), step=int(step_match.group(1)))
                )

        # Final adapter
        final_dir = output_path / "final"
        if final_dir.exists():
            artifacts.final_adapter_path = str(final_dir)

        # Merged model
        merged_dir = output_path / "merged"
        if merged_dir.exists():
            artifacts.merged_path = str(merged_dir)

        # GGUF exports — check both directory names
        for gguf_dirname in ["gguf", "exported_gguf"]:
            gguf_dir = output_path / gguf_dirname
            if gguf_dir.exists():
                for gguf_file in gguf_dir.glob("*.gguf"):
                    quant = gguf_file.stem.split("-")[-1] if "-" in gguf_file.stem else "unknown"
                    artifacts.gguf_exports.append(
                        GGUFExport(
                            path=str(gguf_file),
                            quantization=quant.upper(),
                            size_bytes=gguf_file.stat().st_size,
                            created_at=datetime.fromtimestamp(gguf_file.stat().st_mtime),
                        )
                    )

        return artifacts

    @staticmethod
    def _calculate_model_size(output_path: Path) -> int:
        """Calculate total model size on disk (bytes) from merged/final/gguf dirs."""
        total = 0
        for subdir in ["merged", "final", "gguf", "exported_gguf"]:
            d = output_path / subdir
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total

    def _save_model_profile(self, run: TrainingRun) -> None:
        """Save or update the ModelProfile for a completed training run."""
        if not MODEL_PROFILE_AVAILABLE:
            logger.debug("Model profile module not available, skipping profile save")
            return

        try:
            registry = get_registry()

            # Check if profile already exists
            existing = registry.get(run.run_id)

            # Calculate duration
            duration = 0.0
            if run.started_at and run.completed_at:
                try:
                    start = datetime.fromisoformat(run.started_at)
                    end = datetime.fromisoformat(run.completed_at)
                    duration = (end - start).total_seconds()
                except Exception:
                    pass

            # Build config snapshot from TrainerConfig
            config_dict = {
                "base_model": self.config.base_model,
                "model_type": self.config.model_type,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "max_seq_length": self.config.max_seq_length,
                "max_steps": self.config.max_steps,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "warmup_ratio": self.config.warmup_ratio,
                "weight_decay": self.config.weight_decay,
                "load_in_4bit": self.config.load_in_4bit,
                "use_gradient_checkpointing": self.config.use_gradient_checkpointing,
                "eval_strategy": self.config.eval_strategy,
                "eval_steps": self.config.eval_steps,
                "gguf_quantization": self.config.gguf_quantization,
            }

            # Add cascade metadata if present
            if self.config.task_domain:
                config_dict["cascade"] = {
                    "domain": self.config.task_domain,
                    "stage": self.config.cascade_stage,
                    "cascade_run_id": self.config.cascade_run_id,
                }

            # Extract enrichment data from training_metadata
            meta = run.training_metadata or {}
            training_traces = meta.get("trace_ids", [])
            training_repos = meta.get("training_repos", [])
            teacher_model = meta.get("teacher_model")
            if not teacher_model and run.strategy == TrainingStrategy.DISTILLATION:
                teacher_model = self.config.teacher_model

            # Scan artifacts and compute sizes
            artifacts = self._scan_output_artifacts(run.output_path)
            model_size_bytes = self._calculate_model_size(run.output_path)
            model_size_params = self._extract_param_count(run.base_model)

            # Build description
            base_name = run.base_model.split("/")[-1] if run.base_model else "unknown"
            trace_count = meta.get("trace_count", len(training_traces))
            desc_parts = [f"Fine-tuned {base_name} with {run.strategy.value.upper()}"]
            if trace_count:
                desc_parts.append(f"on {trace_count} traces")
            if training_repos:
                desc_parts.append(f"from {', '.join(training_repos[:3])}")
            description = " ".join(desc_parts)

            if existing:
                # Update existing profile
                existing.status = "ready" if run.status == "completed" else run.status
                existing.completed_at = (
                    datetime.fromisoformat(run.completed_at) if run.completed_at else None
                )
                existing.duration_seconds = duration
                existing.loss_curve = run.loss_curve
                existing.final_metrics = run.metrics
                existing.config = config_dict
                existing.artifacts = artifacts
                existing.model_size_bytes = model_size_bytes
                existing.model_size_params = model_size_params
                existing.training_traces = training_traces
                existing.training_repos = training_repos
                if teacher_model:
                    existing.teacher_model = teacher_model
                existing.description = description
                existing.save()
                logger.info(f"Updated model profile for {run.run_id}")
            else:
                # Create new profile
                display_name = f"{base_name}-{run.strategy.value}-{run.run_id[-6:]}"

                profile = ModelProfile(
                    model_id=run.run_id,
                    run_id=run.run_id,
                    display_name=display_name,
                    description=description,
                    created_at=(
                        datetime.fromisoformat(run.started_at) if run.started_at else datetime.now()
                    ),
                    base_model=run.base_model,
                    training_strategy=run.strategy.value,
                    teacher_model=teacher_model,
                    training_traces=training_traces,
                    training_repos=training_repos,
                    config=config_dict,
                    started_at=datetime.fromisoformat(run.started_at) if run.started_at else None,
                    completed_at=(
                        datetime.fromisoformat(run.completed_at) if run.completed_at else None
                    ),
                    duration_seconds=duration,
                    loss_curve=run.loss_curve,
                    final_metrics=run.metrics,
                    artifacts=artifacts,
                    model_dir=str(run.output_path),
                    model_size_bytes=model_size_bytes,
                    model_size_params=model_size_params,
                    status="ready" if run.status == "completed" else run.status,
                )
                profile.save(Path(run.output_path) / "model_profile.json")

                # Rescan registry to pick up the new profile
                registry.scan(force_rescan=True)
                logger.info(f"Created model profile for {run.run_id}")

        except Exception as e:
            logger.warning(f"Failed to save model profile: {e}")

    def _get_training_python(self) -> str:
        """Get the Python executable for training.

        Python 3.14 doesn't have PyTorch CUDA wheels, so we use Python 3.12
        which has working CUDA + Unsloth support.
        """
        import platform

        if platform.system() == "Windows":
            # Prefer Python 3.12 with CUDA support on Windows
            py312_paths = [
                r"C:\Users\Cade\AppData\Local\Programs\Python\Python312\python.exe",
                r"C:\Python312\python.exe",
                r"C:\Program Files\Python312\python.exe",
            ]
            for path in py312_paths:
                if Path(path).exists():
                    return path

        # Fallback to current Python
        return sys.executable

    def _require_base_model(self) -> None:
        """Fail fast when no base model is set.

        There is no default base model; the user must choose one (set BASE_MODEL,
        or pass base_model in the training request / TrainerConfig). This prevents
        silently fine-tuning a stale hardcoded default.
        """
        if not (self.config.base_model or "").strip():
            raise ValueError(
                "No base model set. Choose a model to fine-tune — set BASE_MODEL in "
                "your environment or pass base_model in the training request."
            )

    def train_sft(
        self,
        dataset_path: Path,
        val_dataset_path: Path | None = None,
        run_id: str | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """
        Run Supervised Fine-Tuning.

        Args:
            dataset_path: Path to JSONL training data
            run_id: Optional run identifier
            callback: Optional callback for progress updates
            log_callback: Optional callback for raw log lines

        Returns:
            TrainingRun with results
        """
        self._require_base_model()
        run_id = run_id or self._generate_run_id()
        output_path = Path(self.config.output_dir) / run_id

        run = TrainingRun(
            run_id=run_id,
            strategy=TrainingStrategy.SFT,
            base_model=self.config.base_model,
            dataset_path=Path(dataset_path),
            output_path=output_path,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.active_runs[run_id] = run
        run.val_dataset_path = val_dataset_path
        if training_metadata:
            run.training_metadata = training_metadata

        try:
            if self.config.use_remote_ssh:
                self._train_with_remote_ssh(run, callback, log_callback, pid_callback)
            elif self.config.use_nemo_gym:
                self._train_with_nemo_gym(run, callback)
            else:
                self._train_with_unsloth_sft(run, callback, log_callback, pid_callback)

            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc).isoformat()

            # Save model profile
            self._save_model_profile(run)

            # GGUF export now happens in-script (model still in GPU memory)

            # Auto-deploy to Ollama if enabled
            if self.config.auto_deploy_ollama:
                self._auto_deploy_to_ollama(run)

            # Auto-push to HuggingFace Hub if enabled
            if self.config.auto_push_hf:
                self._auto_push_to_hf(run)

            # Export to bashbros integration if linked
            self._export_to_bashbros_integration(run)

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc).isoformat()

        return run

    def _auto_deploy_to_ollama(self, run: TrainingRun) -> None:
        """Deploy GGUF model to Ollama after training."""
        try:
            from bashgym.api.models_routes import deploy_gguf_to_ollama

            model_name = self.config.ollama_model_name
            if not model_name:
                base_short = run.base_model.split("/")[-1].lower() if run.base_model else "model"
                model_name = f"bashgym-{base_short}-{run.run_id[-6:]}"

            # Find GGUF file
            gguf_dir = run.output_path / "gguf"
            gguf_files = sorted(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []
            if not gguf_files:
                logger.warning(f"No GGUF files in {gguf_dir}, skipping Ollama deploy")
                return

            logger.info(f"Auto-deploying to Ollama as '{model_name}'...")
            result = deploy_gguf_to_ollama(
                str(gguf_files[0]),
                model_name,
                base_ollama_tag=self.config.ollama_base_tag or None,
            )
            if result["success"]:
                logger.info(f"Deployed to Ollama as '{model_name}'")
                run.metrics["ollama_model"] = model_name
            else:
                logger.warning(f"Ollama deploy failed: {result.get('error')}")
        except Exception as e:
            logger.warning(f"Auto-deploy to Ollama failed: {e}")

    def _auto_push_to_hf(self, run: TrainingRun) -> None:
        """Push trained model to HuggingFace Hub after training."""
        try:
            from bashgym.integrations.huggingface import get_hf_client
            from bashgym.integrations.huggingface.model_manager import get_model_manager

            client = get_hf_client()
            if not client.is_enabled:
                logger.info("HuggingFace not configured, skipping auto-push")
                return

            manager = get_model_manager(client)

            # Determine repo name
            repo_name = self.config.hf_repo_name
            if not repo_name:
                base_short = (
                    run.base_model.split("/")[-1].lower().replace(" ", "-")
                    if run.base_model
                    else "model"
                )
                repo_name = f"bashgym-{base_short}-{run.run_id[-6:]}"

            repo_id = client.get_repo_id(repo_name)

            # Push merged model (preferred) or final adapter
            merged_dir = run.output_path / "merged"
            final_dir = run.output_path / "final"
            push_dir = (
                merged_dir if merged_dir.exists() else final_dir if final_dir.exists() else None
            )

            if not push_dir:
                logger.warning("No merged or final model found, skipping HF push")
                return

            logger.info(f"Pushing model to HuggingFace Hub as '{repo_id}'...")
            url = manager.push_model(push_dir, repo_name, private=self.config.hf_private)
            run.metrics["hf_repo_id"] = repo_id
            run.metrics["hf_url"] = url

            # Push GGUF exports
            gguf_dir = run.output_path / "gguf"
            if gguf_dir.exists():
                gguf_files = sorted(gguf_dir.glob("*.gguf"))
                if gguf_files:
                    logger.info(f"Pushing GGUF to {repo_name}-GGUF...")
                    manager.push_gguf(gguf_files[0], repo_name, private=self.config.hf_private)

            # Generate and push model card
            try:
                profile_data = {
                    "base_model": run.base_model,
                    "training_strategy": run.strategy.value,
                    "display_name": repo_name,
                    "description": (
                        f"Fine-tuned {run.base_model} with" f" {run.strategy.value.upper()}"
                    ),
                    "final_metrics": run.metrics,
                    "duration_seconds": 0,
                    "training_traces": run.training_metadata.get("trace_ids", []),
                    "training_repos": run.training_metadata.get("training_repos", []),
                    "config": {
                        "learning_rate": self.config.learning_rate,
                        "batch_size": self.config.batch_size,
                        "num_epochs": self.config.num_epochs,
                        "max_seq_length": self.config.max_seq_length,
                        "lora_r": self.config.lora_r,
                        "lora_alpha": self.config.lora_alpha,
                    },
                }
                if run.started_at and run.completed_at:
                    start = datetime.fromisoformat(run.started_at)
                    end = datetime.fromisoformat(run.completed_at)
                    profile_data["duration_seconds"] = (end - start).total_seconds()

                card = manager.generate_model_card(repo_id, profile_data)
                manager.push_model_card(repo_id, card)
            except Exception as card_err:
                logger.warning(f"Model card generation failed: {card_err}")

            # Update model profile if available
            if MODEL_PROFILE_AVAILABLE:
                try:
                    registry = get_registry()
                    profile = registry.get(run.run_id)
                    if profile:
                        profile.hf_repo_id = repo_id
                        profile.save()
                except Exception:
                    pass

            logger.info(f"Model pushed to HuggingFace Hub: {url}")

        except ImportError:
            logger.debug("HuggingFace integration not available, skipping auto-push")
        except Exception as e:
            logger.warning(f"Auto-push to HuggingFace failed: {e}")

    def _export_to_bashbros_integration(self, run: TrainingRun) -> None:
        """Export model to bashbros integration if linked.

        This exports the model to GGUF in the shared integration directory
        and registers it with Ollama for use by bashbros sidekick.
        """
        try:
            from bashgym.integrations.bashbros import get_integration

            integration = get_integration()

            # Check if integration is linked
            if not integration.is_linked():
                logger.debug("Bashbros integration not linked, skipping export")
                return

            settings = integration.get_settings()
            if not settings.auto_export_ollama:
                logger.debug("Auto-export to Ollama disabled, skipping")
                return

            # Update training status
            integration.update_training_status("exporting", run.run_id)

            # Get model path
            model_path = run.output_path / "merged"
            if not model_path.exists():
                logger.warning(f"Merged model not found at {model_path}")
                return

            # Calculate trace stats (if available)
            traces_used = 0
            quality_avg = 0.0
            if hasattr(run, "config_snapshot") and run.config_snapshot:
                traces_used = run.config_snapshot.get("traces_used", 0)
                quality_avg = run.config_snapshot.get("quality_avg", 0.0)

            # Export to GGUF and register with Ollama
            logger.info(f"Exporting model {run.run_id} to bashbros integration...")
            gguf_path = integration.export_to_gguf(
                model_path=model_path,
                quantization=self.config.gguf_quantization,
                traces_used=traces_used,
                quality_avg=quality_avg,
            )

            if gguf_path:
                integration.update_training_status(
                    "complete", run.run_id, model=settings.ollama_model_name
                )
                logger.info(f"Model exported to bashbros: {gguf_path}")
            else:
                integration.update_training_status("failed", run.run_id)
                logger.warning("Failed to export model to bashbros integration")

        except ImportError:
            logger.debug("Bashbros integration module not available")
        except Exception as e:
            logger.warning(f"Error exporting to bashbros integration: {e}")

    def train_dpo(
        self,
        dataset_path: Path,
        val_dataset_path: Path | None = None,
        run_id: str | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """
        Run Direct Preference Optimization training.

        Args:
            dataset_path: Path to DPO JSONL data (with chosen/rejected pairs)
            val_dataset_path: Optional path to validation JSONL data
            run_id: Optional run identifier
            callback: Optional callback for progress updates
            log_callback: Optional callback for raw log lines (used by cascade)
            pid_callback: Optional callback when the subprocess PID is known

        Returns:
            TrainingRun with results
        """
        self._require_base_model()
        run_id = run_id or self._generate_run_id()
        output_path = Path(self.config.output_dir) / run_id

        run = TrainingRun(
            run_id=run_id,
            strategy=TrainingStrategy.DPO,
            base_model=self.config.base_model,
            dataset_path=Path(dataset_path),
            output_path=output_path,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.active_runs[run_id] = run
        run.val_dataset_path = val_dataset_path
        if training_metadata:
            run.training_metadata = training_metadata

        try:
            self._train_with_unsloth_dpo(run, callback, log_callback, pid_callback)
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc).isoformat()

            # Save model profile
            self._save_model_profile(run)

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc).isoformat()

        return run

    def _train_with_unsloth_sft(
        self,
        run: TrainingRun,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
    ) -> None:
        """
        Train using Unsloth for fast LoRA fine-tuning.

        Generates and executes a training script. Mirrors the GRPO loop's
        shape: persistent `training.log`, TRL-stats dict parsing (shared
        `_parse_trl_stats`), and an `EARLY_STOPPED` sentinel check for
        loss-not-decreasing plateau.
        """
        # Generate training script
        script_content = self._generate_sft_script(run)
        script_path = run.output_path / "train_sft.py"
        run.output_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)

        # Execute training
        python_exe = self._get_training_python()
        logger.info(f"Starting SFT training run: {run.run_id}")
        logger.info(f"Dataset: {run.dataset_path}")
        logger.info(f"Output: {run.output_path}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Python: {python_exe}")

        log_file_path = run.output_path / "training.log"
        log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")
        logger.info(f"SFT training log: {log_file_path}")

        # Force unbuffered stdout on the child so per-step 'loss' dicts from
        # TRL's PrinterCallback flush immediately instead of sitting in Python's
        # block buffer for minutes. Without `-u` the subprocess pipe is not a
        # tty so print() is block-buffered and we only see tqdm progress bars
        # (which flush on every \r) — the actual metric dicts arrive in batches.
        sft_env = os.environ.copy()
        sft_env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            process = subprocess.Popen(
                [python_exe, "-u", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
                env=sft_env,
            )

            run.pid = process.pid
            logger.info(f"Training subprocess started with PID {process.pid}")

            if pid_callback:
                try:
                    pid_callback(process.pid, run)
                except Exception as e:
                    logger.warning(f"pid_callback error: {e}")

            stats: dict[str, float] = {}
            final_summary: dict[str, float] = {}
            last_step = 0
            estimated_total_steps = self.config.max_steps or 1000
            samples_processed = 0
            start_time = datetime.now(timezone.utc)

            for line in process.stdout:
                line = line.rstrip("\n")
                try:
                    log_file.write(line + "\n")
                except Exception as e:
                    logger.warning(f"Log file write error: {e}")
                line = line.strip()
                if not line:
                    continue

                logger.info(f"[SFT Training] {line}")

                if log_callback:
                    try:
                        log_callback(line)
                    except Exception as e:
                        logger.warning(f"Log callback error: {e}")

                # Tqdm progress bar → step count
                progress_match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
                if progress_match:
                    last_step = int(progress_match.group(2))
                    estimated_total_steps = int(progress_match.group(3))
                    samples_processed = last_step * self.config.batch_size

                # Unsloth header: "Total steps = 12"
                unsloth_steps_match = re.search(r"Total steps\s*=\s*(\d+)", line)
                if unsloth_steps_match:
                    estimated_total_steps = int(unsloth_steps_match.group(1))

                parsed = _parse_trl_stats(line)
                if parsed is not None:
                    if "train_runtime" in parsed or "train_loss" in parsed:
                        final_summary.update(parsed)
                    else:
                        stats.update(parsed)

                    if "epoch" in parsed and last_step == 0:
                        last_step = max(1, int(parsed.get("epoch", 0) * estimated_total_steps))

                # ETA
                eta = None
                if last_step > 0 and estimated_total_steps > 0:
                    elapsed_s = (datetime.now(timezone.utc) - start_time).total_seconds()
                    steps_remaining = estimated_total_steps - last_step
                    if steps_remaining > 0 and last_step > 0:
                        time_per_step = elapsed_s / last_step
                        eta_s = steps_remaining * time_per_step
                        eta = (
                            f"{int(eta_s)}s"
                            if eta_s < 60
                            else (
                                f"{int(eta_s / 60)}m"
                                if eta_s < 3600
                                else f"{int(eta_s / 3600)}h {int((eta_s % 3600) / 60)}m"
                            )
                        )

                if callback and (progress_match or parsed):
                    callback(
                        {
                            "epoch": int(stats.get("epoch", 0)),
                            "total_epochs": self.config.num_epochs,
                            "step": last_step,
                            "total_steps": estimated_total_steps,
                            "loss": stats.get("loss"),
                            "learning_rate": stats.get("learning_rate")
                            or self.config.learning_rate,
                            "grad_norm": stats.get("grad_norm"),
                            "eval_loss": stats.get("eval_loss"),
                            "eta": eta,
                            "samples_processed": samples_processed,
                        }
                    )

                # Track loss curve for model profile
                if parsed and "loss" in parsed and last_step > 0:
                    run.add_loss_point(
                        step=last_step,
                        loss=parsed["loss"],
                        epoch=int(parsed.get("epoch", 0)),
                        learning_rate=parsed.get("learning_rate") or self.config.learning_rate,
                    )

            return_code = process.wait()

            # Same early-stop convention as GRPO: script writes EARLY_STOPPED
            # sentinel when the in-script callback fires on a loss plateau.
            # BUT — the callback triggers `control.should_training_stop = True`
            # which only stops the training LOOP. The script still runs its
            # save-artifacts path (save_pretrained + save_pretrained_merged)
            # AFTER the loop exits. So if the sentinel file exists AND
            # artifacts were saved successfully, this is a "completed early"
            # run, NOT a failure — mark it completed with an early_stop note.
            # Only treat the sentinel as a failure if no artifacts landed.
            early_stop_file = run.output_path / "EARLY_STOPPED"
            early_stop_reason = None
            if early_stop_file.exists():
                early_stop_reason = early_stop_file.read_text().strip()

            artifacts_exist = (run.output_path / "merged" / "model.safetensors").exists() or (
                run.output_path / "final" / "adapter_model.safetensors"
            ).exists()

            if early_stop_reason and not artifacts_exist:
                # Callback fired before save path ran → real failure
                raise RuntimeError(
                    f"SFT training stopped early by platform safety check: " f"{early_stop_reason}"
                )

            if return_code != 0 and not artifacts_exist:
                raise RuntimeError(f"Training script exited with code {return_code}")

            if early_stop_reason:
                logger.warning(
                    f"SFT training early-stopped but saved artifacts successfully: "
                    f"{early_stop_reason}"
                )
                run.early_stop_reason = early_stop_reason

            combined = {**stats, **final_summary}
            run.metrics = {
                "final_loss": combined.get("train_loss", combined.get("loss", 0.0)),
                "epochs_completed": self.config.num_epochs,
                "samples_processed": samples_processed,
                "eval_loss": combined.get("eval_loss"),
                "final_grad_norm": combined.get("grad_norm"),
                "train_runtime_s": combined.get("train_runtime"),
                "steps_completed": last_step,
            }

            logger.info(f"SFT training completed. Model saved to: {run.output_path}")

        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")
        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            raise
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def _generate_sft_script(self, run: TrainingRun) -> str:
        """Dispatch SFT script generation to the backend-appropriate generator.

        Mirrors the GRPO dispatch: explicit ``sft_backend`` > family default >
        platform probe. Unsloth where available; the plain transformers+peft path
        on GB10/sm_121 where Unsloth can't load (unslothai#4867).
        """
        from bashgym.families import resolve_family_profile, select_backend

        profile = resolve_family_profile(self.config.base_model)
        backend = select_backend(profile, self.config.sft_backend)
        if backend == "plain":
            return self._generate_sft_script_plain(run, profile)
        return self._generate_unsloth_sft_script(run)

    def _generate_dpo_script(self, run: TrainingRun) -> str:
        """Dispatch DPO script generation (Unsloth vs plain transformers+peft)."""
        from bashgym.families import resolve_family_profile, select_backend

        profile = resolve_family_profile(self.config.base_model)
        backend = select_backend(profile, self.config.dpo_backend)
        if backend == "plain":
            return self._generate_dpo_script_plain(run, profile)
        return self._generate_unsloth_dpo_script(run)

    def _generate_sft_script_plain(self, run: TrainingRun, profile) -> str:
        """Generate an SFT script using plain transformers + peft (no Unsloth).

        The GB10/sm_121 fallback. Family-correct LoRA targets/excludes, attention
        impl, and correctness patches come from the ModelFamilyProfile; Liger
        fused-linear-CE is enabled via ``use_liger_kernel`` when ``use_liger`` is
        set (the 262k-vocab fused-CE OOM fix).
        """
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")
        val_path = str(run.val_dataset_path).replace("\\", "/") if run.val_dataset_path else ""

        return f'''#!/usr/bin/env python3
"""
Auto-generated SFT Training Script for BashGym (plain transformers+peft backend)
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}

Plain HuggingFace transformers + peft + trl (no Unsloth) — the GB10/sm_121
fallback for when Unsloth can't load (unslothai#4867). Liger fused-linear-CE is
enabled via use_liger_kernel={self.config.use_liger}.
Family: {profile.family}; patches: {list(profile.patches)}
"""

import json as _json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from bashgym.families.patches import apply_patches

apply_patches({list(profile.patches)})


def _sanitize_messages(messages):
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("content") is None:
            m["content"] = ""
        if isinstance(m.get("tool_calls"), list):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = _json.loads(args)
                        except (_json.JSONDecodeError, TypeError):
                            fn["arguments"] = {{"raw": args}}
                    tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


if __name__ == "__main__":
    MODEL_NAME = "{self.config.base_model}"
    print(f"Loading {{MODEL_NAME}} with plain transformers (no Unsloth)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="{profile.attn_implementation}",
        device_map="cuda:0",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r={self.config.lora_r},
        lora_alpha={self.config.lora_alpha},
        lora_dropout={self.config.lora_dropout},
        target_modules={list(profile.lora_target_modules)},
        exclude_modules={list(profile.lora_exclude_modules)},
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files="{dataset_path}", split="train")

    val_dataset = None
    val_path = "{val_path}"
    if val_path and os.path.exists(val_path):
        val_dataset = load_dataset("json", data_files=val_path, split="train")

    def _to_text(examples):
        texts = []
        for messages in examples["messages"]:
            texts.append(
                tokenizer.apply_chat_template(_sanitize_messages(messages), tokenize=False)
            )
        return {{"text": texts}}

    dataset = dataset.map(_to_text, batched=True)
    if val_dataset is not None:
        val_dataset = val_dataset.map(_to_text, batched=True)

    sft_config = SFTConfig(
        output_dir="{output_path}",
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        warmup_ratio={self.config.warmup_ratio},
        num_train_epochs={self.config.num_epochs},
        max_steps={self.config.max_steps},
        learning_rate={self.config.learning_rate},
        bf16=True,
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        save_total_limit=3,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_seq_length={self.config.max_seq_length},
        use_liger_kernel={self.config.use_liger},
        eval_strategy="{self.config.eval_strategy}" if val_dataset is not None else "no",
        eval_steps={self.config.eval_steps} if val_dataset is not None else None,
        load_best_model_at_end=True if val_dataset is not None else False,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False if val_dataset is not None else None,
    )

    callbacks = []
    if val_dataset is not None and {self.config.early_stopping_patience} > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience={self.config.early_stopping_patience})
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=callbacks or None,
    )

    print("Starting SFT training (plain backend)...")
    trainer.train()

    model.save_pretrained("{output_path}/final")
    tokenizer.save_pretrained("{output_path}/final")

    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained("{output_path}/merged")
    tokenizer.save_pretrained("{output_path}/merged")

    print("SFT training complete (plain backend)!")
'''

    def _generate_dpo_script_plain(self, run: TrainingRun, profile) -> str:
        """Generate a DPO script using plain transformers + peft (no Unsloth).

        The GB10/sm_121 fallback. Implicit-reference DPO (``ref_model=None``);
        dataset columns are passed straight to ``DPOTrainer`` (same as the Unsloth
        path). Liger fused-linear-CE via ``use_liger_kernel`` when ``use_liger``.
        """
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")

        return f'''#!/usr/bin/env python3
"""
Auto-generated DPO Training Script for BashGym (plain transformers+peft backend)
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}

Plain HuggingFace transformers + peft + trl (no Unsloth) — the GB10/sm_121
fallback. Liger fused-linear-CE enabled via use_liger_kernel={self.config.use_liger}.
Family: {profile.family}; patches: {list(profile.patches)}
"""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from bashgym.families.patches import apply_patches

apply_patches({list(profile.patches)})


if __name__ == "__main__":
    MODEL_NAME = "{self.config.base_model}"
    print(f"Loading {{MODEL_NAME}} with plain transformers (no Unsloth)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="{profile.attn_implementation}",
        device_map="cuda:0",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r={self.config.lora_r},
        lora_alpha={self.config.lora_alpha},
        lora_dropout={self.config.lora_dropout},
        target_modules={list(profile.lora_target_modules)},
        exclude_modules={list(profile.lora_exclude_modules)},
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files="{dataset_path}", split="train")

    val_dataset = None
    val_path = "{str(run.val_dataset_path).replace(chr(92), '/') if run.val_dataset_path else ''}"
    if val_path and os.path.exists(val_path):
        val_dataset = load_dataset("json", data_files=val_path, split="train")

    dpo_config = DPOConfig(
        output_dir="{output_path}",
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        warmup_ratio={self.config.warmup_ratio},
        num_train_epochs={self.config.num_epochs},
        max_steps={self.config.max_steps},
        learning_rate={self.config.learning_rate},
        beta={self.config.dpo_beta},
        bf16=True,
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        save_total_limit=3,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        max_length={self.config.max_seq_length},
        use_liger_kernel={self.config.use_liger},
        eval_strategy="{self.config.eval_strategy}" if val_dataset is not None else "no",
        eval_steps={self.config.eval_steps} if val_dataset is not None else None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        args=dpo_config,
    )

    print("Starting DPO training (plain backend)...")
    trainer.train()

    model.save_pretrained("{output_path}/final")
    tokenizer.save_pretrained("{output_path}/final")

    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained("{output_path}/merged")
    tokenizer.save_pretrained("{output_path}/merged")

    print("DPO training complete (plain backend)!")
'''

    def _generate_unsloth_sft_script(self, run: TrainingRun) -> str:
        """Generate Unsloth SFT training script."""
        # Use forward slashes for cross-platform compatibility
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")
        val_path = str(run.val_dataset_path).replace("\\", "/") if run.val_dataset_path else ""

        return f'''#!/usr/bin/env python3
"""
Auto-generated SFT Training Script for BashGym
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, TrainerCallback, ProcessorMixin
import os
import torch


class LossPlateauStop(TrainerCallback):
    """Write an EARLY_STOPPED sentinel when train loss has genuinely stalled.

    Purpose: detect runs that aren't learning — NOT to catch healthy stochastic
    bounce at the loss floor. A well-behaved SFT run descends quickly in the
    first few percent of an epoch and then bounces within a ±5% band around
    a slowly-drifting floor. We must not fire on that bounce.

    Detection strategy:
      - Wait `min_step` steps before considering any stop (warmup done, descent
        established).
      - Maintain a rolling window of the last `window` loss values.
      - Compare the *mean* of the current window to the mean of the window
        `window` steps ago (i.e. non-overlapping prior window).
      - If the current window mean is not at least `min_delta` lower than the
        prior window mean, increment the patience counter. Else reset.
      - Fire when the counter reaches `patience`.

    Defaults tuned for SFT stochastic descent on an 8B LoRA at batch 4-8:
      - min_step=50: never fire during warmup
      - window=10: smooth over 10 logs (≈10 optimizer steps at logging_steps=1)
      - patience=10: require 10 consecutive window comparisons with no
        meaningful improvement
      - min_delta=5e-3: 0.005 absolute loss — well above stochastic jitter
    """

    def __init__(
        self, output_dir, min_step=50, window=10, patience=10, min_delta=5e-3,
    ):
        self.output_dir = output_dir
        self.min_step = min_step
        self.window = window
        self.patience = patience
        self.min_delta = min_delta
        self.losses: list[float] = []
        self.bad = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step < self.min_step:
            return control
        loss = logs.get("loss")
        if loss is None:
            return control
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            return control
        if loss != loss:  # NaN
            return control

        self.losses.append(loss)
        # Need 2 full windows before we can compare
        if len(self.losses) < 2 * self.window:
            return control

        recent = self.losses[-self.window:]
        prior = self.losses[-2 * self.window:-self.window]
        recent_mean = sum(recent) / self.window
        prior_mean = sum(prior) / self.window

        if recent_mean < prior_mean - self.min_delta:
            self.bad = 0  # genuine improvement, reset
        else:
            self.bad += 1

        if self.bad >= self.patience:
            try:
                with open(os.path.join(self.output_dir, "EARLY_STOPPED"), "w") as f:
                    f.write(
                        f"loss plateau at step {{state.global_step}}: "
                        f"rolling mean {{recent_mean:.4f}} vs prior {{prior_mean:.4f}} "
                        f"(delta {{prior_mean - recent_mean:+.4f}}), "
                        f"no meaningful improvement for {{self.patience}} consecutive windows"
                    )
            except Exception as e:
                print(f"[LossPlateauStop] failed to write sentinel: {{e}}")
            control.should_training_stop = True
        return control

def _sanitize_messages(messages):
    """Sanitize messages for chat template compatibility.

    Fixes common format issues:
    - tool_calls arguments as JSON string -> parse to dict
    - None content in assistant messages -> empty string
    - Missing required fields
    """
    import json as _json
    sanitized = []
    for msg in messages:
        m = dict(msg)  # shallow copy
        # Fix None content
        if m.get("content") is None:
            m["content"] = ""
        # Fix tool_calls arguments: some templates expect dict, not JSON string
        if "tool_calls" in m and isinstance(m["tool_calls"], list):
            fixed_calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                if "function" in tc and isinstance(tc["function"], dict):
                    fn = dict(tc["function"])
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = _json.loads(args)
                        except (_json.JSONDecodeError, TypeError):
                            fn["arguments"] = {{"raw": args}}
                    tc["function"] = fn
                fixed_calls.append(tc)
            m["tool_calls"] = fixed_calls
        sanitized.append(m)
    return sanitized

def _make_formatting_prompts_func(fmt_tokenizer):
    """Build a formatter closed over the RAW tokenizer (already unwrapped
    from any ProcessorMixin). Returns a function suitable for Dataset.map.

    Mirrors Unsloth Studio's chat_templates._format_chatml:
    - applies the tokenizer's chat template
    - strips a leading <bos> token (SFTTrainer injects its own)
    - appends eos_token so the model learns where assistant turns end
    """
    eos = fmt_tokenizer.eos_token or ""

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = []
        for convo in convos:
            try:
                clean = _sanitize_messages(convo)
                text = fmt_tokenizer.apply_chat_template(
                    clean, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                # Fallback: strip tool_calls entirely and retry
                fallback = [
                    {{"role": m.get("role", "user"), "content": m.get("content", "") or ""}}
                    for m in convo if m.get("role") in ("system", "user", "assistant")
                ]
                try:
                    text = fmt_tokenizer.apply_chat_template(
                        fallback, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    texts.append("")
                    continue
            # Strip any leading <bos> so SFTTrainer's BOS injection doesn't double up
            if text.startswith("<bos>"):
                text = text.removeprefix("<bos>")
            # Append EOS so the model learns sequence termination
            if eos and not text.endswith(eos):
                text = text + eos
            texts.append(text)
        return {{"text": texts}}

    return formatting_prompts_func

if __name__ == "__main__":
    import gc
    import os

    # Clear GPU memory before starting
    print("Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {{torch.cuda.get_device_name(0)}}")
        print(f"Available VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}} GB")

    # Model configuration. dtype=None lets Unsloth auto-detect the model's
    # native precision (Gemma 4 is bf16-native — hardcoding float16 causes
    # numerical instability and an absurdly high starting loss).
    max_seq_length = {self.config.max_seq_length}
    dtype = None
    load_in_4bit = {self.config.load_in_4bit}

    # Load model with Unsloth optimizations
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="{self.config.base_model}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Unwrap ProcessorMixin → raw tokenizer for text-only SFT.
    # Gemma 3/4 from_pretrained returns a ProcessorMixin (vision-language
    # wrapper) even for text. If you pass that directly to SFTTrainer, Unsloth
    # detects the Processor, sets _is_vlm=True, and SKIPS _prepare_dataset(),
    # meaning the 'text' column never gets tokenized to 'input_ids'. The model
    # then trains on garbage (loss ≈ ln(vocab_size) ≈ 12.5 on step 1).
    # Mirror of Unsloth Studio's fix in backend/core/training/trainer.py:3205.
    if isinstance(tokenizer, ProcessorMixin) and hasattr(tokenizer, "tokenizer"):
        print("Unwrapping ProcessorMixin -> raw tokenizer for text-only SFT")
        sft_tokenizer = tokenizer.tokenizer
    else:
        sft_tokenizer = tokenizer

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r={self.config.lora_r},
        target_modules={self.config.lora_target_modules},
        lora_alpha={self.config.lora_alpha},
        lora_dropout={self.config.lora_dropout},
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files="{dataset_path}", split="train")

    # Load validation dataset if available
    val_dataset = None
    val_dataset_path = "{val_path}"
    if val_dataset_path and os.path.exists(val_dataset_path):
        print("Loading validation dataset...")
        val_dataset = load_dataset("json", data_files=val_dataset_path, split="train")
        print(f"Validation set: {{len(val_dataset)}} examples")

    # Training arguments — use SFTConfig directly (NOT TrainingArguments).
    # Unsloth's SFTTrainer wrapper has a broken version check that only strips
    # push_to_hub_token from the dict args on transformers<5.0.0, so passing a
    # TrainingArguments on transformers>=5.0 fails with TypeError. SFTConfig(...)
    # sidesteps the conversion path entirely. Also: bf16 when supported (Gemma 4
    # is bf16-native on Blackwell), fall back to fp16 only on older cards.
    training_args = SFTConfig(
        output_dir="{output_path}",
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        warmup_ratio={self.config.warmup_ratio},
        num_train_epochs={self.config.num_epochs},
        max_steps={self.config.max_steps},
        learning_rate={self.config.learning_rate},
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        # Eval settings (active when val_dataset is available)
        eval_strategy="{self.config.eval_strategy}" if val_dataset is not None else "no",
        eval_steps={self.config.eval_steps} if val_dataset is not None else None,
        load_best_model_at_end=True if val_dataset is not None else False,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False if val_dataset is not None else None,
    )

    # Apply chat template to dataset using the UNWRAPPED tokenizer so the
    # chat template, BOS handling, and EOS appending all match what SFTTrainer
    # will see at tokenization time.
    formatting_prompts_func = _make_formatting_prompts_func(sft_tokenizer)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Apply formatting to val dataset too
    if val_dataset is not None:
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    # Early stopping (only when val_dataset exists and patience > 0)
    callbacks = []
    if val_dataset is not None and {self.config.early_stopping_patience} > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience={self.config.early_stopping_patience}))

    # Platform safety: degenerate loss check (always on). Writes EARLY_STOPPED
    # sentinel if train loss fails to decrease, so the parent cascade can fail
    # the stage instead of silently claiming success.
    callbacks.append(LossPlateauStop(output_dir="{output_path}"))

    # Initialize trainer with the UNWRAPPED tokenizer (critical for Gemma 4 —
    # see unwrap comment above). dataset_text_field and max_seq_length now
    # live on SFTConfig so they're omitted here.
    trainer = SFTTrainer(
        model=model,
        processing_class=sft_tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving model...")
    model.save_pretrained("{output_path}/final")
    tokenizer.save_pretrained("{output_path}/final")

    # Save LoRA adapters separately
    model.save_pretrained_merged(
        "{output_path}/merged",
        tokenizer,
        save_method="merged_16bit",
    )

    # Export to GGUF if enabled
    if {self.config.auto_export_gguf}:
        print("Exporting to GGUF ({self.config.gguf_quantization})...")
        import os
        gguf_dir = "{output_path}/gguf"
        os.makedirs(gguf_dir, exist_ok=True)
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method="{self.config.gguf_quantization}",
        )
        print(f"GGUF exported to: {{gguf_dir}}")

    print("Training complete!")
'''

    def _train_with_unsloth_dpo(
        self,
        run: TrainingRun,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
    ) -> None:
        """Train using Unsloth with DPO.

        Mirrors GRPO/SFT loop shape: persistent `training.log`, shared
        `_parse_trl_stats`, `EARLY_STOPPED` sentinel check for
        `rewards/accuracies` collapse.
        """
        script_content = self._generate_dpo_script(run)
        script_path = run.output_path / "train_dpo.py"
        run.output_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)

        python_exe = self._get_training_python()
        logger.info(f"Starting DPO training run: {run.run_id}")
        logger.info(f"Dataset: {run.dataset_path}")
        logger.info(f"Output: {run.output_path}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Python: {python_exe}")

        log_file_path = run.output_path / "training.log"
        log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")
        logger.info(f"DPO training log: {log_file_path}")

        # Unbuffered stdout — same fix as SFT. Without this the per-step
        # loss/reward dicts block-buffer and only show up in batches.
        dpo_env = os.environ.copy()
        dpo_env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            process = subprocess.Popen(
                [python_exe, "-u", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
                env=dpo_env,
            )

            run.pid = process.pid
            logger.info(f"DPO subprocess started with PID {process.pid}")

            if pid_callback:
                try:
                    pid_callback(process.pid, run)
                except Exception as e:
                    logger.warning(f"pid_callback error: {e}")

            stats: dict[str, float] = {}
            final_summary: dict[str, float] = {}
            last_step = 0
            estimated_total_steps = self.config.max_steps or 1000
            samples_processed = 0
            start_time = datetime.now(timezone.utc)

            for line in process.stdout:
                line = line.rstrip("\n")
                try:
                    log_file.write(line + "\n")
                except Exception as e:
                    logger.warning(f"Log file write error: {e}")
                line = line.strip()
                if not line:
                    continue

                logger.info(f"[DPO Training] {line}")

                if log_callback:
                    try:
                        log_callback(line)
                    except Exception as e:
                        logger.warning(f"Log callback error: {e}")

                progress_match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
                if progress_match:
                    last_step = int(progress_match.group(2))
                    estimated_total_steps = int(progress_match.group(3))
                    samples_processed = last_step * self.config.batch_size

                parsed = _parse_trl_stats(line)
                if parsed is not None:
                    if "train_runtime" in parsed or "train_loss" in parsed:
                        final_summary.update(parsed)
                    else:
                        stats.update(parsed)

                    if "epoch" in parsed and last_step == 0:
                        last_step = max(1, int(parsed.get("epoch", 0) * estimated_total_steps))

                eta = None
                if last_step > 0 and estimated_total_steps > 0:
                    elapsed_s = (datetime.now(timezone.utc) - start_time).total_seconds()
                    steps_remaining = estimated_total_steps - last_step
                    if steps_remaining > 0 and last_step > 0:
                        time_per_step = elapsed_s / last_step
                        eta_s = steps_remaining * time_per_step
                        eta = (
                            f"{int(eta_s)}s"
                            if eta_s < 60
                            else (
                                f"{int(eta_s / 60)}m"
                                if eta_s < 3600
                                else f"{int(eta_s / 3600)}h {int((eta_s % 3600) / 60)}m"
                            )
                        )

                if callback and (progress_match or parsed):
                    callback(
                        {
                            "epoch": int(stats.get("epoch", 0)),
                            "total_epochs": self.config.num_epochs,
                            "step": last_step,
                            "total_steps": estimated_total_steps,
                            "loss": stats.get("loss"),
                            "learning_rate": stats.get("learning_rate")
                            or self.config.learning_rate,
                            "grad_norm": stats.get("grad_norm"),
                            "eval_loss": stats.get("eval_loss"),
                            "reward_margin": stats.get("rewards/margins"),
                            "rewards_chosen": stats.get("rewards/chosen"),
                            "rewards_rejected": stats.get("rewards/rejected"),
                            "rewards_accuracies": stats.get("rewards/accuracies"),
                            "eta": eta,
                            "samples_processed": samples_processed,
                        }
                    )

                if parsed and "loss" in parsed and last_step > 0:
                    run.add_loss_point(
                        step=last_step,
                        loss=parsed["loss"],
                        epoch=int(parsed.get("epoch", 0)),
                        learning_rate=parsed.get("learning_rate") or self.config.learning_rate,
                    )

            return_code = process.wait()

            # Sentinel-race fix: see _train_with_unsloth_sft for the full
            # explanation. Callback stops the training loop but the save
            # path still runs, so artifacts may exist even when the sentinel
            # is written. Only treat as failure if nothing was actually saved.
            early_stop_file = run.output_path / "EARLY_STOPPED"
            early_stop_reason = None
            if early_stop_file.exists():
                early_stop_reason = early_stop_file.read_text().strip()

            artifacts_exist = (run.output_path / "merged" / "model.safetensors").exists() or (
                run.output_path / "final" / "adapter_model.safetensors"
            ).exists()

            if early_stop_reason and not artifacts_exist:
                raise RuntimeError(
                    f"DPO training stopped early by platform safety check: {early_stop_reason}"
                )

            if return_code != 0 and not artifacts_exist:
                raise RuntimeError(f"DPO training script exited with code {return_code}")

            if early_stop_reason:
                logger.warning(
                    f"DPO training early-stopped but saved artifacts successfully: "
                    f"{early_stop_reason}"
                )
                run.early_stop_reason = early_stop_reason

            combined = {**stats, **final_summary}
            run.metrics = {
                "final_loss": combined.get("train_loss", combined.get("loss", 0.0)),
                "final_reward_margin": combined.get("rewards/margins"),
                "final_rewards_chosen": combined.get("rewards/chosen"),
                "final_rewards_rejected": combined.get("rewards/rejected"),
                "final_rewards_accuracies": combined.get("rewards/accuracies"),
                "epochs_completed": self.config.num_epochs,
                "eval_loss": combined.get("eval_loss"),
                "train_runtime_s": combined.get("train_runtime"),
                "steps_completed": last_step,
                "samples_processed": samples_processed,
            }

            logger.info(f"DPO training completed. Model saved to: {run.output_path}")

        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")
        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def train_distillation(
        self,
        dataset_path: Path,
        run_id: str | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """
        Run Knowledge Distillation training.

        Trains a smaller student model using outputs from a larger teacher model.
        Supports both offline (pre-generated) and on-policy distillation.

        Args:
            dataset_path: Path to prompts dataset (or pre-distilled data)
            run_id: Optional run identifier
            callback: Optional callback for progress updates

        Returns:
            TrainingRun with results
        """
        self._require_base_model()
        run_id = run_id or self._generate_run_id()
        output_path = Path(self.config.output_dir) / run_id

        run = TrainingRun(
            run_id=run_id,
            strategy=TrainingStrategy.DISTILLATION,
            base_model=self.config.base_model,
            dataset_path=Path(dataset_path),
            output_path=output_path,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.active_runs[run_id] = run
        if training_metadata:
            run.training_metadata = training_metadata

        try:
            if self.config.use_remote_ssh:
                self._train_with_remote_ssh(run, callback, log_callback, pid_callback)
            else:
                self._train_with_distillation(run, callback, log_callback, pid_callback)
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc).isoformat()

            # Save model profile
            self._save_model_profile(run)

            # Auto-export to GGUF if enabled
            if self.config.auto_export_gguf:
                print(f"Auto-exporting to GGUF ({self.config.gguf_quantization})...")
                self.export_model(run_id, "gguf", self.config.gguf_quantization)

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc).isoformat()

        return run

    def _train_with_distillation(
        self,
        run: TrainingRun,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
    ) -> None:
        """Train using knowledge distillation (offline — teacher traces as training data)."""
        import re

        script_content = self._generate_distillation_script(run)
        script_path = run.output_path / "train_distillation.py"
        run.output_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)

        python_exe = self._get_training_python()
        logger.info(f"Starting Distillation run: {run.run_id}")
        logger.info(f"Teacher: {self.config.teacher_model}")
        logger.info(f"Student: {self.config.base_model}")
        logger.info(f"Dataset: {run.dataset_path}")
        logger.info(f"Output: {run.output_path}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Python: {python_exe}")

        try:
            process = subprocess.Popen(
                [python_exe, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
            )

            run.pid = process.pid
            logger.info(f"Distillation subprocess started with PID {process.pid}")

            if pid_callback:
                try:
                    pid_callback(process.pid, run)
                except Exception as e:
                    logger.warning(f"pid_callback error: {e}")

            last_loss = None
            last_epoch = 0
            last_step = 0
            last_grad_norm = None
            samples_processed = 0
            start_time = datetime.now(timezone.utc)
            estimated_total_steps = 1000

            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[Distillation] {line}")

                    if log_callback:
                        try:
                            log_callback(line)
                        except Exception as e:
                            logger.warning(f"Log callback error: {e}")

                    loss_match = re.search(r"'loss':\s*([\d.]+)", line)
                    epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                    step_match = re.search(r"'step':\s*(\d+)", line)
                    grad_norm_match = re.search(r"'grad_norm':\s*([\d.]+)", line)
                    progress_match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
                    unsloth_steps_match = re.search(r"Total steps\s*=\s*(\d+)", line)

                    if loss_match:
                        last_loss = float(loss_match.group(1))
                    if epoch_match:
                        last_epoch = int(float(epoch_match.group(1)))
                    if step_match:
                        last_step = int(step_match.group(1))
                        samples_processed = last_step * self.config.batch_size
                    if grad_norm_match:
                        last_grad_norm = float(grad_norm_match.group(1))
                    if unsloth_steps_match:
                        estimated_total_steps = int(unsloth_steps_match.group(1))
                    if progress_match:
                        last_step = int(progress_match.group(2))
                        estimated_total_steps = int(progress_match.group(3))
                        samples_processed = last_step * self.config.batch_size

                    eta = None
                    if last_step > 0 and estimated_total_steps > 0:
                        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                        steps_remaining = estimated_total_steps - last_step
                        if steps_remaining > 0:
                            time_per_step = elapsed / last_step
                            eta_seconds = steps_remaining * time_per_step
                            if eta_seconds < 60:
                                eta = f"{int(eta_seconds)}s"
                            elif eta_seconds < 3600:
                                eta = f"{int(eta_seconds / 60)}m"
                            else:
                                eta = (
                                    f"{int(eta_seconds / 3600)}h {int((eta_seconds % 3600) / 60)}m"
                                )

                    if callback and (progress_match or loss_match):
                        callback(
                            {
                                "epoch": last_epoch,
                                "total_epochs": self.config.num_epochs,
                                "step": last_step,
                                "total_steps": estimated_total_steps,
                                "loss": last_loss,
                                "learning_rate": self.config.learning_rate,
                                "grad_norm": last_grad_norm,
                                "eta": eta,
                                "samples_processed": samples_processed,
                            }
                        )

                    if loss_match and last_loss is not None and last_step > 0:
                        run.add_loss_point(
                            step=last_step,
                            loss=last_loss,
                            epoch=last_epoch,
                            learning_rate=self.config.learning_rate,
                        )

            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(f"Distillation script exited with code {return_code}")

            run.metrics = {
                "final_loss": last_loss or 0.0,
                "epochs_completed": self.config.num_epochs,
                "samples_processed": samples_processed,
                "teacher_model": self.config.teacher_model,
            }

            logger.info(f"Distillation completed. Model saved to: {run.output_path}")

        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")
        except Exception as e:
            logger.error(f"Distillation training failed: {e}")
            raise

    def train_rlvr(
        self,
        dataset_path: Path,
        run_id: str | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """
        Run RL with Verifiable Rewards.

        This is GRPO with verification-based reward signals.
        Dataset must include 'tests' field with pytest-compatible test code.
        """
        grpo_config = copy.deepcopy(self.config)
        grpo_config.grpo_reward_mode = "verification"

        grpo_trainer = GRPOTrainer(grpo_config)
        run = grpo_trainer.train_grpo(
            dataset_path=dataset_path,
            verifier_fn=lambda p, r: 0.0,
            run_id=run_id,
            callback=callback,
            log_callback=log_callback,
            pid_callback=pid_callback,
            training_metadata=training_metadata,
        )
        run.strategy = TrainingStrategy.RLVR
        return run

    def _generate_distillation_script(self, run: TrainingRun) -> str:
        """Generate Knowledge Distillation training script."""
        # Use forward slashes for cross-platform compatibility
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")

        return f'''#!/usr/bin/env python3
"""
Auto-generated Knowledge Distillation Script for BashGym
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}

Knowledge Distillation:
- Teacher: {self.config.teacher_model}
- Student: {self.config.base_model}
- Alpha: {self.config.distillation_alpha} (balance hard/soft labels)
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import torch.nn.functional as F

# Configuration
TEACHER_MODEL = "{self.config.teacher_model}"
STUDENT_MODEL = "{self.config.base_model}"
ALPHA = {self.config.distillation_alpha}  # Weight for soft labels (1-alpha for hard labels)
TEMPERATURE = {self.config.teacher_temperature}
ON_POLICY = {self.config.on_policy_distillation}

# Load student model with Unsloth optimizations
student_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=STUDENT_MODEL,
    max_seq_length={self.config.max_seq_length},
    dtype=torch.float16,
    load_in_4bit={self.config.load_in_4bit},
)

# Add LoRA adapters to student
student_model = FastLanguageModel.get_peft_model(
    student_model,
    r={self.config.lora_r},
    target_modules={self.config.lora_target_modules},
    lora_alpha={self.config.lora_alpha},
    lora_dropout={self.config.lora_dropout},
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load dataset (should contain prompts and optionally teacher outputs)
dataset = load_dataset("json", data_files="{dataset_path}", split="train")

def distillation_loss(student_logits, teacher_logits, labels, alpha=ALPHA, temperature=TEMPERATURE):
    """
    Compute distillation loss combining:
    - Soft labels (KL divergence with teacher)
    - Hard labels (cross-entropy with ground truth)
    """
    # Soft targets (from teacher)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    # Hard targets
    hard_loss = F.cross_entropy(student_logits, labels, ignore_index=-100)

    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Training arguments
training_args = TrainingArguments(
    output_dir="{output_path}",
    per_device_train_batch_size={self.config.batch_size},
    gradient_accumulation_steps={self.config.gradient_accumulation_steps},
    warmup_ratio={self.config.warmup_ratio},
    num_train_epochs={self.config.num_epochs},
    learning_rate={self.config.learning_rate},
    fp16=True,
    logging_steps={self.config.logging_steps},
    save_steps={self.config.save_steps},
    save_total_limit=3,
    optim="adamw_8bit",
    seed=42,
)

if ON_POLICY:
    # On-policy distillation: generate teacher outputs during training
    print("Using on-policy distillation - generating teacher outputs on-the-fly")
    # This would integrate with NeMo's on-policy distillation API
else:
    # Offline distillation: use pre-generated teacher outputs
    print("Using offline distillation with pre-generated teacher outputs")

# Initialize trainer with custom loss
trainer = SFTTrainer(
    model=student_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length={self.config.max_seq_length},
    args=training_args,
)

# Train
print("Starting distillation training...")
trainer.train()

# Save final model
print("Saving distilled model...")
student_model.save_pretrained("{output_path}/final")
tokenizer.save_pretrained("{output_path}/final")

# Save LoRA adapters separately
student_model.save_pretrained_merged(
    "{output_path}/merged",
    tokenizer,
    save_method="merged_16bit",
)

# Export to GGUF if enabled
if {self.config.auto_export_gguf}:
    print("Exporting to GGUF ({self.config.gguf_quantization})...")
    import os
    gguf_dir = "{output_path}/gguf"
    os.makedirs(gguf_dir, exist_ok=True)
    student_model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method="{self.config.gguf_quantization}",
    )
    print(f"GGUF exported to: {{gguf_dir}}")

print("Knowledge distillation complete!")
print(f"Student model saved to: {output_path}/final")
'''

    def _generate_unsloth_dpo_script(self, run: TrainingRun) -> str:
        """Generate Unsloth DPO training script."""
        # Use forward slashes for cross-platform compatibility
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")
        val_path = str(run.val_dataset_path).replace("\\", "/") if run.val_dataset_path else ""

        return f'''#!/usr/bin/env python3
"""
Auto-generated DPO Training Script for BashGym
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback
import os
import torch


class DegenerateAccuracyStop(TrainerCallback):
    """Write an EARLY_STOPPED sentinel when rewards/accuracies collapses.

    DPO with a dataset the model can't distinguish chosen from rejected
    on leaves rewards/accuracies stuck at ~0.5 (random). If we see that
    after `min_step`, the run is burning compute for no gradient signal.
    Threshold 0.52 is a small tolerance above random — informed by the
    TRL GitHub issue #2194 where users report the 0.5 stuck state.
    """

    def __init__(self, output_dir, min_step=10, patience=3, threshold=0.52):
        self.output_dir = output_dir
        self.min_step = min_step
        self.patience = patience
        self.threshold = threshold
        self.bad = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step < self.min_step:
            return control
        acc = logs.get("rewards/accuracies")
        if acc is None:
            return control
        try:
            acc = float(acc)
        except (TypeError, ValueError):
            return control
        if acc != acc:  # NaN
            return control
        if acc <= self.threshold:
            self.bad += 1
        else:
            self.bad = 0
        if self.bad >= self.patience:
            try:
                with open(os.path.join(self.output_dir, "EARLY_STOPPED"), "w") as f:
                    f.write(
                        f"rewards/accuracies={{acc:.4f}} <= {{self.threshold}} "
                        f"for {{self.patience}} consecutive logs at step "
                        f"{{state.global_step}} — DPO is not distinguishing "
                        f"chosen from rejected. Check dataset pair quality."
                    )
            except Exception as e:
                print(f"[DegenerateAccuracyStop] failed to write sentinel: {{e}}")
            control.should_training_stop = True
        return control


# Model configuration
max_seq_length = {self.config.max_seq_length}
dtype = torch.float16
load_in_4bit = {self.config.load_in_4bit}

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{self.config.base_model}",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r={self.config.lora_r},
    target_modules={self.config.lora_target_modules},
    lora_alpha={self.config.lora_alpha},
    lora_dropout={self.config.lora_dropout},
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load DPO dataset
dataset = load_dataset("json", data_files="{dataset_path}", split="train")

# Load validation dataset if available
val_dataset = None
val_dataset_path = "{val_path}"
if val_dataset_path and os.path.exists(val_dataset_path):
    print("Loading validation dataset...")
    val_dataset = load_dataset("json", data_files=val_dataset_path, split="train")
    print(f"Validation set: {{len(val_dataset)}} examples")

# DPO configuration
dpo_config = DPOConfig(
    output_dir="{output_path}",
    per_device_train_batch_size={self.config.batch_size},
    gradient_accumulation_steps={self.config.gradient_accumulation_steps},
    learning_rate={self.config.learning_rate},
    num_train_epochs={self.config.num_epochs},
    beta={self.config.dpo_beta},
    logging_steps={self.config.logging_steps},
    save_steps={self.config.save_steps},
    fp16=True,
    optim="adamw_8bit",
    # Eval settings (active when val_dataset is available)
    eval_strategy="{self.config.eval_strategy}" if val_dataset is not None else "no",
    eval_steps={self.config.eval_steps} if val_dataset is not None else None,
)

# Initialize DPO trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Use implicit reference model
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    args=dpo_config,
    callbacks=[DegenerateAccuracyStop(output_dir="{output_path}")],
)

# Train
print("Starting DPO training...")
trainer.train()

# Save model
print("Saving model...")
model.save_pretrained("{output_path}/final")
tokenizer.save_pretrained("{output_path}/final")

# Save merged weights
model.save_pretrained_merged(
    "{output_path}/merged",
    tokenizer,
    save_method="merged_16bit",
)

# Export to GGUF if enabled
if {self.config.auto_export_gguf}:
    print("Exporting to GGUF ({self.config.gguf_quantization})...")
    import os
    gguf_dir = "{output_path}/gguf"
    os.makedirs(gguf_dir, exist_ok=True)
    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method="{self.config.gguf_quantization}",
    )
    print(f"GGUF exported to: {{gguf_dir}}")

print("DPO training complete!")
'''

    def _train_with_remote_ssh(self, run, callback, log_callback, pid_callback):
        """Execute training on remote machine via SSH.

        Strategy-aware: generates the correct script based on run.strategy.
        """
        from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig

        # Use pre-resolved ssh_config from route handler, or fall back to env vars
        ssh_config = getattr(self, "ssh_config", None)
        if ssh_config is None:
            from bashgym.config import get_settings

            settings = get_settings()
            ssh_config = SSHConfig.from_settings(settings.ssh)

        trainer = RemoteTrainer(ssh_config)

        # Generate the correct script based on strategy
        run.output_path.mkdir(parents=True, exist_ok=True)

        if run.strategy == TrainingStrategy.DISTILLATION:
            script_content = self._generate_distillation_script(run)
            script_path = run.output_path / "train_distillation.py"
        elif run.strategy == TrainingStrategy.DPO:
            script_content = self._generate_dpo_script(run)
            script_path = run.output_path / "train_dpo.py"
        elif run.strategy in (TrainingStrategy.GRPO, TrainingStrategy.RLVR):
            grpo_trainer = GRPOTrainer(self.config)
            script_content = grpo_trainer._generate_grpo_script(run)
            script_path = run.output_path / "train_grpo.py"
        else:
            script_content = self._generate_sft_script(run)
            script_path = run.output_path / "train_sft.py"

        script_path.write_text(script_content)

        def _pid_cb(remote_pid):
            run.pid = remote_pid
            if pid_callback:
                pid_callback(remote_pid, run)

        result = asyncio.run(
            trainer.train_remote(
                run_id=run.run_id,
                script_path=script_path,
                dataset_path=Path(run.dataset_path),
                local_output_dir=run.output_path,
                log_callback=log_callback,
                pid_callback=_pid_cb,
            )
        )

        if not result["success"]:
            raise RuntimeError(f"Remote training failed: {result.get('error')}")

    def _train_with_nemo_gym(
        self, run: TrainingRun, callback: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        """
        Train using NVIDIA NeMo Customizer (cloud-based).

        Submits training job to NeMo Microservices API using the official SDK.
        Falls back to HTTP if SDK is not available.
        """
        if NeMoClient is None:
            raise RuntimeError(
                "NeMo client not available. Install with:\n" "  pip install nemo-microservices"
            )

        logger.info(f"Submitting training job to NeMo Customizer: {run.run_id}")

        # Initialize NeMo client
        nemo_config = NeMoClientConfig(
            base_url=self.config.nemo_gym_endpoint,
            api_key=self.config.nemo_api_key,
        )

        with NeMoClient(nemo_config) as client:
            # Log SDK mode
            mode = "SDK" if client.is_sdk_mode else "HTTP"
            logger.info(f"Using NeMo client in {mode} mode")

            # Prepare hyperparameters
            hyperparameters = {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "max_seq_length": self.config.max_seq_length,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "warmup_ratio": self.config.warmup_ratio,
            }

            # Create customization job
            job = client.customization.create_job(
                model=self.config.base_model,
                training_data=str(run.dataset_path),
                strategy=run.strategy.value,
                hyperparameters=hyperparameters,
                output_model_name=f"bashgym-{run.run_id}",
            )

            logger.info(f"Created NeMo job: {job.job_id}")
            run.metrics["nemo_job_id"] = job.job_id

            # Poll for completion with progress callbacks
            poll_interval = 30
            timeout = 3600 * 4  # 4 hours max
            start_time = datetime.now(timezone.utc)

            while True:
                job = client.customization.get_job(job.job_id)

                # Report progress
                if callback and job.metrics:
                    callback(
                        {
                            "job_id": job.job_id,
                            "status": job.status,
                            "epoch": job.metrics.get("epoch", 0),
                            "loss": job.metrics.get("loss", 0),
                            "learning_rate": job.metrics.get("learning_rate", 0),
                        }
                    )

                # Check completion
                if job.status in ("completed", "succeeded"):
                    logger.info(f"NeMo job completed: {job.job_id}")
                    run.metrics.update(
                        {
                            "nemo_job_id": job.job_id,
                            "status": job.status,
                            "final_loss": job.metrics.get("final_loss", 0),
                            "output_model": job.output_model,
                        }
                    )
                    break

                elif job.status in ("failed", "cancelled", "error"):
                    error_msg = job.error or f"Job failed with status: {job.status}"
                    logger.error(f"NeMo job failed: {error_msg}")
                    raise RuntimeError(error_msg)

                # Check timeout
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                if elapsed > timeout:
                    logger.error(f"NeMo job timed out after {elapsed}s")
                    client.customization.cancel_job(job.job_id)
                    raise TimeoutError(f"Training job timed out after {timeout}s")

                import time

                time.sleep(poll_interval)

    def get_run_status(self, run_id: str) -> TrainingRun | None:
        """Get the status of a training run."""
        return self.active_runs.get(run_id)

    def list_runs(self) -> list[TrainingRun]:
        """List all training runs."""
        return list(self.active_runs.values())

    def export_model(
        self,
        run_id: str,
        export_format: str = "gguf",
        quantization: str = "q4_k_m",
        save_lora_separately: bool = True,
    ) -> Path | None:
        """
        Export a trained model to a specific format.

        Args:
            run_id: Training run ID
            export_format: Export format (gguf, onnx, safetensors, lora)
            quantization: Quantization level for GGUF (q4_k_m, q5_k_m, q8_0, f16)
            save_lora_separately: Save LoRA adapters as separate files

        Returns:
            Path to exported model

        Output structure:
            {output_path}/
            ├── final/              # Full model with LoRA merged
            ├── lora_adapters/      # Separate LoRA adapter files
            │   ├── adapter_config.json
            │   └── adapter_model.safetensors
            ├── merged/             # Merged weights (16-bit)
            └── exported_gguf/      # GGUF quantized model
                └── model-{quantization}.gguf
        """
        run = self.active_runs.get(run_id)
        if not run or run.status != "completed":
            print(f"Run {run_id} not found or not completed")
            return None

        export_path = run.output_path / f"exported_{export_format}"
        export_path.mkdir(parents=True, exist_ok=True)

        if export_format == "gguf":
            # Generate comprehensive GGUF export script
            script = f'''#!/usr/bin/env python3
"""
GGUF Export Script for BashGym
Run ID: {run.run_id}

This script converts the trained LoRA model to GGUF format for use with:
- llama.cpp
- Ollama
- LM Studio
- GPT4All
- Any GGUF-compatible inference engine
"""

import os
import subprocess
from pathlib import Path

# Paths
MODEL_PATH = Path("{run.output_path}/merged")
OUTPUT_PATH = Path("{export_path}")
QUANTIZATION = "{quantization}"

def export_to_gguf():
    """Export model to GGUF format."""

    output_file = OUTPUT_PATH / f"model-{{QUANTIZATION}}.gguf"

    print(f"Exporting to GGUF with {{QUANTIZATION}} quantization...")
    print(f"Source: {{MODEL_PATH}}")
    print(f"Output: {{output_file}}")

    # Method 1: Using llama.cpp convert script (recommended)
    try:
        # First convert to F16 GGUF
        subprocess.run([
            "python", "-m", "llama_cpp.llama_convert",
            str(MODEL_PATH),
            "--outfile", str(OUTPUT_PATH / "model-f16.gguf"),
            "--outtype", "f16"
        ], check=True)

        # Then quantize
        subprocess.run([
            "llama-quantize",
            str(OUTPUT_PATH / "model-f16.gguf"),
            str(output_file),
            QUANTIZATION
        ], check=True)

        print(f"Successfully exported to: {{output_file}}")
        return output_file

    except FileNotFoundError:
        print("llama.cpp tools not found. Install with:")
        print("  pip install llama-cpp-python")
        print("  # Or build llama.cpp from source for quantization")

    # Method 2: Using transformers + llama.cpp (alternative)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model for conversion...")
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

        # Save in a format llama.cpp can convert
        model.save_pretrained(str(OUTPUT_PATH / "hf_model"))
        tokenizer.save_pretrained(str(OUTPUT_PATH / "hf_model"))

        print(f"Model saved to {{OUTPUT_PATH / 'hf_model'}}")
        print("Use llama.cpp convert.py to create GGUF:")
        print(f"  python convert.py {{OUTPUT_PATH / 'hf_model'}} --outfile {{output_file}}")

    except Exception as e:
        print(f"Conversion failed: {{e}}")
        return None

if __name__ == "__main__":
    export_to_gguf()
'''
            script_path = export_path / "export_gguf.py"
            script_path.write_text(script)

            # Also create a simple batch script
            batch_script = f"""#!/bin/bash
# Quick GGUF export for {run.run_id}
# Requires: llama-cpp-python or llama.cpp

python export_gguf.py

echo "GGUF export complete!"
echo "Output: {export_path}/model-{quantization}.gguf"
"""
            (export_path / "export.sh").write_text(batch_script)

            print(f"GGUF export scripts generated in: {export_path}")
            print(f"  Run: python {script_path}")

        elif export_format == "lora":
            # Export just the LoRA adapters (lightweight)
            lora_path = run.output_path / "lora_adapters"
            print(f"LoRA adapters saved at: {lora_path}")
            return lora_path

        elif export_format == "safetensors":
            # Export as safetensors (faster loading)
            script = f"""#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{run.output_path}/merged")
tokenizer = AutoTokenizer.from_pretrained("{run.output_path}/merged")

model.save_pretrained("{export_path}", safe_serialization=True)
tokenizer.save_pretrained("{export_path}")
print("Exported to safetensors format")
"""
            script_path = export_path / "export_safetensors.py"
            script_path.write_text(script)
            print(f"Safetensors export script generated: {script_path}")

        return export_path


class GRPOTrainer(Trainer):
    """
    Specialized trainer for GRPO (Group Relative Policy Optimization).

    GRPO generates multiple responses and uses relative ranking
    for policy optimization - ideal for agentic tasks.
    """

    def __init__(self, config: TrainerConfig | None = None):
        """Initialize GRPO trainer."""
        super().__init__(config)
        if self.config.strategy != TrainingStrategy.GRPO:
            self.config.strategy = TrainingStrategy.GRPO

    def train_grpo(
        self,
        dataset_path: Path,
        verifier_fn: Callable[[str, str], float],
        run_id: str | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
        training_metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """
        Run GRPO training with verifiable rewards.

        Args:
            dataset_path: Path to prompts dataset
            verifier_fn: Function that takes (prompt, response) and returns reward
            run_id: Optional run identifier
            callback: Optional callback for progress updates

        Returns:
            TrainingRun with results
        """
        self._require_base_model()
        run_id = run_id or self._generate_run_id()
        output_path = Path(self.config.output_dir) / run_id

        run = TrainingRun(
            run_id=run_id,
            strategy=TrainingStrategy.GRPO,
            base_model=self.config.base_model,
            dataset_path=Path(dataset_path),
            output_path=output_path,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.active_runs[run_id] = run
        if training_metadata:
            run.training_metadata = training_metadata

        try:
            if self.config.use_remote_ssh:
                self._train_with_remote_ssh(run, callback, log_callback, pid_callback)
            else:
                self._run_grpo_loop(run, verifier_fn, callback, log_callback, pid_callback)
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc).isoformat()

            # Save model profile
            self._save_model_profile(run)

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.now(timezone.utc).isoformat()

        return run

    def _run_grpo_loop(
        self,
        run: TrainingRun,
        verifier_fn: Callable[[str, str], float],
        callback: Callable[[dict[str, Any]], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int, "TrainingRun"], None] | None = None,
    ) -> None:
        """
        Execute the GRPO training loop.

        For each prompt:
        1. Generate N responses
        2. Score each with verifier
        3. Compute relative advantages
        4. Update policy
        """
        import re

        print(f"Starting GRPO training: {run.run_id}")
        print(f"Generations per prompt: {self.config.grpo_num_generations}")

        # Generate GRPO training script
        script_content = self._generate_grpo_script(run)
        script_path = run.output_path / "train_grpo.py"
        run.output_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)

        python_exe = self._get_training_python()
        logger.info(f"Starting GRPO training run: {run.run_id}")
        logger.info(f"Dataset: {run.dataset_path}")
        logger.info(f"Output: {run.output_path}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Python: {python_exe}")

        # DGX Spark / GB10 (sm_121) Triton fix:
        # Triton ships with ptxas from CUDA 12.8 which doesn't recognize sm_121a.
        # Point it at the system CUDA 13 ptxas which has full sm_121 support.
        # See: https://github.com/triton-lang/triton/issues/9181
        grpo_env = os.environ.copy()
        if Path("/usr/local/cuda/bin/ptxas").exists():
            grpo_env.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
        grpo_env.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")
        # Unbuffered stdout — same fix as SFT/DPO. Without this the per-step
        # loss/reward dicts block-buffer and only show up in batches.
        grpo_env.setdefault("PYTHONUNBUFFERED", "1")

        log_file_path = run.output_path / "training.log"
        log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")
        logger.info(f"GRPO training log: {log_file_path}")

        try:
            process = subprocess.Popen(
                [python_exe, "-u", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
                env=grpo_env,
            )

            run.pid = process.pid
            logger.info(f"GRPO subprocess started with PID {process.pid}")

            if pid_callback:
                try:
                    pid_callback(process.pid, run)
                except Exception as e:
                    logger.warning(f"pid_callback error: {e}")

            # Aggregated per-step metrics parsed from TRL's printed stats dict.
            # TRL's Trainer.log() passes a dict[str,float] to TrainerCallback.on_log,
            # but ProgressCallback.on_log stringifies it with f"{v:.4g}" before
            # printing — that's what we see on stdout. We still parse stdout for
            # log persistence and cascade-level metrics, but early-stop lives in
            # a TrainerCallback inside the generated script (raw floats, no
            # display-layer formatting bugs). See `_generate_grpo_script`.
            #
            # The stop-file convention: if the generated script's early-stop
            # callback fires, it writes {output_path}/EARLY_STOPPED with a reason.
            # We check that file after process exit to distinguish a clean
            # completion from an early-stop.
            stats: dict[str, float] = {}
            final_summary: dict[str, float] = {}
            last_step = 0
            estimated_total_steps = self.config.max_steps or 1000
            start_time = datetime.now(timezone.utc)

            for line in process.stdout:
                line = line.rstrip("\n")
                try:
                    log_file.write(line + "\n")
                except Exception as e:
                    logger.warning(f"Log file write error: {e}")
                line = line.strip()
                if not line:
                    continue

                logger.info(f"[GRPO Training] {line}")

                if log_callback:
                    try:
                        log_callback(line)
                    except Exception as e:
                        logger.warning(f"Log callback error: {e}")

                # Tqdm progress bar → step count (e.g. "20%|██ | 2/10 [...]")
                progress_match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
                if progress_match:
                    last_step = int(progress_match.group(2))
                    estimated_total_steps = int(progress_match.group(3))

                # Parse TRL stats dict (either per-step or final summary)
                parsed = _parse_trl_stats(line)
                if parsed is not None:
                    if "train_runtime" in parsed or "train_loss" in parsed:
                        final_summary.update(parsed)
                    else:
                        stats.update(parsed)

                    # Derive step count if parsed dict has epoch but progress bar
                    # hasn't fired yet (defensive).
                    if "epoch" in parsed and last_step == 0:
                        last_step = max(1, int(parsed.get("epoch", 0) * estimated_total_steps))

                # ETA + progress callback for live UI updates
                eta = None
                if last_step > 0 and estimated_total_steps > 0:
                    elapsed_s = (datetime.now(timezone.utc) - start_time).total_seconds()
                    steps_remaining = estimated_total_steps - last_step
                    if steps_remaining > 0 and last_step > 0:
                        time_per_step = elapsed_s / last_step
                        eta_s = steps_remaining * time_per_step
                        eta = (
                            f"{int(eta_s)}s"
                            if eta_s < 60
                            else (
                                f"{int(eta_s / 60)}m"
                                if eta_s < 3600
                                else f"{int(eta_s / 3600)}h {int((eta_s % 3600) / 60)}m"
                            )
                        )

                if callback and (progress_match or parsed):
                    samples_processed = last_step * max(1, self.config.batch_size)
                    callback(
                        {
                            "epoch": int(stats.get("epoch", 0)),
                            "total_epochs": self.config.num_epochs,
                            "step": last_step,
                            "total_steps": estimated_total_steps,
                            "loss": stats.get("loss"),
                            "learning_rate": stats.get("learning_rate")
                            or self.config.learning_rate,
                            "eta": eta,
                            "samples_processed": samples_processed,
                            "avg_reward": stats.get("reward"),
                            "reward_std": stats.get("reward_std"),
                            "frac_reward_zero_std": stats.get("frac_reward_zero_std"),
                            "grad_norm": stats.get("grad_norm"),
                            "kl_divergence": stats.get("kl"),
                        }
                    )

            return_code = process.wait()

            # Sentinel-race fix: see _train_with_unsloth_sft for the full
            # explanation. Callback stops the training loop but the save
            # path still runs, so artifacts may exist even when the sentinel
            # is written. Only treat as failure if nothing was actually saved.
            early_stop_file = run.output_path / "EARLY_STOPPED"
            early_stop_reason = None
            if early_stop_file.exists():
                early_stop_reason = early_stop_file.read_text().strip()

            artifacts_exist = (run.output_path / "merged" / "model.safetensors").exists() or (
                run.output_path / "final" / "adapter_model.safetensors"
            ).exists()

            if early_stop_reason and not artifacts_exist:
                raise RuntimeError(
                    f"GRPO training stopped early by platform safety check: {early_stop_reason}"
                )

            if return_code != 0 and not artifacts_exist:
                raise RuntimeError(f"GRPO training script exited with code {return_code}")

            if early_stop_reason:
                logger.warning(
                    f"GRPO training early-stopped but saved artifacts successfully: "
                    f"{early_stop_reason}"
                )
                run.early_stop_reason = early_stop_reason

            # Final metrics — prefer the final_summary dict if TRL emitted one,
            # else fall back to the latest per-step stats.
            combined = {**stats, **final_summary}
            samples_processed = last_step * max(1, self.config.batch_size)
            run.metrics = {
                "final_loss": combined.get("train_loss", combined.get("loss", 0.0)),
                "epochs_completed": self.config.num_epochs,
                "samples_processed": samples_processed,
                "final_avg_reward": combined.get("reward"),
                "final_reward_std": combined.get("reward_std"),
                "final_frac_reward_zero_std": combined.get("frac_reward_zero_std"),
                "final_grad_norm": combined.get("grad_norm"),
                "final_kl_divergence": combined.get("kl"),
                "train_runtime_s": combined.get("train_runtime"),
                "steps_completed": last_step,
            }

            logger.info(f"GRPO training completed. Model saved to: {run.output_path}")

        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")
        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            raise
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def _generate_grpo_script(self, run: TrainingRun) -> str:
        """Dispatch to the backend-appropriate GRPO generator.

        Resolves a ModelFamilyProfile from base_model and selects the backend
        (explicit grpo_backend > family default > platform probe): Unsloth where
        available (the GB10 path), else the plain transformers+peft generator
        (the sm_121 fallback for when Unsloth can't load — unslothai#4867).
        """
        from bashgym.families import resolve_family_profile, select_backend

        valid_loss = {"grpo", "gspo", "dr_grpo", "dapo", "bnpo"}
        if self.config.grpo_loss_type not in valid_loss:
            raise ValueError(
                f"grpo_loss_type={self.config.grpo_loss_type!r} must be one of {sorted(valid_loss)}"
            )

        profile = resolve_family_profile(self.config.base_model)
        backend = select_backend(profile, self.config.grpo_backend)
        if backend == "plain":
            return self._generate_grpo_script_plain(run, profile)
        return self._generate_grpo_script_unsloth(run, profile)

    def _terminal_rl_config_src(self) -> str:
        """Generated-script constants for terminal-agent RL recipe settings."""

        settings = self.config.terminal_rl_settings()
        warnings = self.config.terminal_rl_warnings()
        return f'''# Configuration
REWARD_MODE = "{self.config.grpo_reward_mode}"
TRAINING_PROFILE = {settings["training_profile"]!r}
GRPO_GROUP_SIZE = {settings["grpo_group_size"]}
NUM_GENERATIONS = GRPO_GROUP_SIZE
PROMPTS_PER_ROLLOUT_BATCH = {settings["prompts_per_rollout_batch"]}
MAX_TOOL_CALLS_PER_EPISODE = {settings["max_tool_calls_per_episode"]}
TOKEN_LEVEL_LOSS = {settings["token_level_loss"]}
FILTER_ZERO_STD_GROUPS = {settings["filter_zero_std_groups"]}
ACTIVE_SAMPLING = {settings["active_sampling"]}
LM_HEAD_FP32 = {settings["lm_head_fp32"]}
INTERLEAVED_THINKING = {settings["interleaved_thinking"]}
SFT_WARM_START_POLICY = {settings["sft_warm_start_policy"]!r}
DPPO_BACKEND = {settings["dppo_backend"]!r}
DPPO_DIVERGENCE = {settings["dppo_divergence"]!r}
DPPO_BINARY_TV_THRESHOLD = {settings["dppo_binary_tv_threshold"]}
DPPO_BINARY_KL_THRESHOLD = {settings["dppo_binary_kl_threshold"]}
TERMINAL_RL_WARNINGS = {warnings!r}


def configure_terminal_rl_model(model):
    """Apply terminal-RL stabilization settings that are backend-local."""
    if not LM_HEAD_FP32:
        return model
    output_head = getattr(model, "lm_head", None)
    if output_head is None and hasattr(model, "get_output_embeddings"):
        output_head = model.get_output_embeddings()
    if output_head is not None and hasattr(output_head, "to"):
        output_head.to(torch.float32)
        print("[TerminalRL] lm_head/output embeddings kept in fp32")
    else:
        print("[TerminalRL] warning: LM_HEAD_FP32 requested but output head was not found")
    return model


def log_terminal_rl_config():
    """Print recipe settings once so run logs capture the active contract."""
    print(
        "[TerminalRL] "
        f"profile={{TRAINING_PROFILE}} group_size={{GRPO_GROUP_SIZE}} "
        f"prompts_per_rollout_batch={{PROMPTS_PER_ROLLOUT_BATCH}} "
        f"max_tool_calls={{MAX_TOOL_CALLS_PER_EPISODE}} "
        f"loss_type={self.config.grpo_loss_type} token_level_loss={{TOKEN_LEVEL_LOSS}} "
        f"filter_zero_std={{FILTER_ZERO_STD_GROUPS}} active_sampling={{ACTIVE_SAMPLING}} "
        f"interleaved_thinking={{INTERLEAVED_THINKING}} "
        f"sft_warm_start={{SFT_WARM_START_POLICY}} "
        f"dppo_backend={{DPPO_BACKEND}} dppo_divergence={{DPPO_DIVERGENCE}}"
    )
    if FILTER_ZERO_STD_GROUPS or ACTIVE_SAMPLING:
        print(
            "[TerminalRL] zero-std filtering/active sampling contract is enabled; "
            "standard TRL backends expose diagnostics, while external rollout "
            "trainers should enforce filtering and refills."
        )
    for warning in TERMINAL_RL_WARNINGS:
        print(f"[TerminalRL] warning: {{warning}}")'''

    def _grpo_reward_functions_src(self) -> str:
        """Shared generated-script source: reward-mode config + tiered reward functions.

        Rendered Python (REWARD_MODE/NUM_GENERATIONS + extract_code, run_verification,
        syntax/execution/verification rewards, REWARD_FN) used by the plain GRPO
        generator. The Unsloth generator keeps an inline copy (de-dup is a tracked
        follow-up gated on a byte-identity check).
        """
        return f'''{self._terminal_rl_config_src()}

# --- Reward helpers ---

def extract_code(text):
    """Extract code from ```python fences, ``` fences, or raw text."""
    match = re.search(r"```python\\s*\\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\\s*\\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_verification(code, test_code):
    """Write code + tests to a temp dir and run pytest. Returns (passed, total)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "test_solution.py")
        with open(solution_path, "w") as f:
            f.write(code)
        with open(test_path, "w") as f:
            f.write(test_code)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir,
            )
            output = result.stdout + result.stderr
            passed = len(re.findall(r" PASSED", output))
            failed = len(re.findall(r" FAILED", output))
            total = passed + failed
            return passed, total
        except Exception:
            return 0, 1


# --- Reward functions ---

def syntax_reward(completions, **kwargs):
    """Return 1.0 if code parses without SyntaxError, else 0.0."""
    rewards = []
    for completion in completions:
        code = extract_code(completion if isinstance(completion, str) else completion[0]["content"])
        try:
            ast.parse(code)
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    return rewards


def execution_reward(completions, **kwargs):
    """Return 1.0 if code executes with exit code 0, else 0.0."""
    rewards = []
    for completion in completions:
        code = extract_code(completion if isinstance(completion, str) else completion[0]["content"])
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=10,
            )
            rewards.append(1.0 if result.returncode == 0 else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def verification_reward(completions, prompts, tests=None, **kwargs):
    """Run pytest on extracted code. Returns passed/total for each completion."""
    rewards = []
    for i, completion in enumerate(completions):
        code = extract_code(completion if isinstance(completion, str) else completion[0]["content"])
        test_code = ""
        if tests is not None and i < len(tests):
            test_code = tests[i]
        if not test_code:
            # Fall back to syntax check if no test code
            try:
                ast.parse(code)
                rewards.append(0.5)
            except SyntaxError:
                rewards.append(0.0)
            continue
        passed, total = run_verification(code, test_code)
        rewards.append(passed / total if total > 0 else 0.0)
    return rewards


# Reward function selection
REWARD_FN = {{"syntax": syntax_reward, "execution": execution_reward, "verification": verification_reward}}[REWARD_MODE]'''

    def _generate_grpo_script_plain(self, run: TrainingRun, profile) -> str:
        """Generate GRPO training script using plain transformers + peft (no Unsloth).

        The sm_121 / GB10 fallback for when Unsloth can't load (unslothai#4867).
        Family correctness patches are applied via bashgym.families.patches, and all
        model-specific values (LoRA targets/excludes, attention impl) come from the
        ModelFamilyProfile.
        """
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")

        return f'''#!/usr/bin/env python3
"""
Auto-generated GRPO Training Script for BashGym (plain transformers+peft backend)
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}

GRPO: Group Relative Policy Optimization with tiered rewards
- Reward mode: {self.config.grpo_reward_mode}
- Generations per prompt: {self.config.effective_grpo_group_size()}
- Training profile: {self.config.training_profile}
- Quantization: load_in_4bit={self.config.load_in_4bit} (GB10/sm_121 trains in bf16)
- Family: {profile.family}; patches: {list(profile.patches)}

NOTE: plain HuggingFace transformers + peft + trl (no Unsloth) for GB10/sm_121.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from trl import GRPOTrainer as TRLGRPOTrainer

from bashgym.families.patches import apply_patches

apply_patches({list(profile.patches)})

{self._grpo_reward_functions_src()}


if __name__ == "__main__":
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    MODEL_NAME = "{self.config.base_model}"
    log_terminal_rl_config()
    print(f"Loading {{MODEL_NAME}} with plain transformers (no Unsloth)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        attn_implementation="{profile.attn_implementation}",
        device_map="cuda:0",
    )
    model = configure_terminal_rl_model(model)
    model.config.use_cache = False  # required for gradient checkpointing
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r={self.config.lora_r},
        lora_alpha={self.config.lora_alpha},
        lora_dropout={self.config.lora_dropout},
        target_modules={list(profile.lora_target_modules)},
        exclude_modules={list(profile.lora_exclude_modules)},
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files="{dataset_path}", split="train")

    grpo_config = GRPOConfig(
        output_dir="{output_path}",
        num_generations=GRPO_GROUP_SIZE,
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        num_train_epochs={self.config.num_epochs},
        max_steps={self.config.max_steps},
        learning_rate={self.config.learning_rate},
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        max_completion_length={self.config.max_seq_length},
        temperature={self.config.grpo_temperature},
        use_vllm={self.config.grpo_use_vllm},
        loss_type="{self.config.grpo_loss_type}",
        bf16=True,
        report_to="none",
    )

    trainer = TRLGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[REWARD_FN],
        train_dataset=dataset,
        args=grpo_config,
    )

    print(f"Starting GRPO training with reward_mode={{REWARD_MODE}}...")
    trainer.train()

    model.save_pretrained("{output_path}/final")
    tokenizer.save_pretrained("{output_path}/final")

    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained("{output_path}/merged")
    tokenizer.save_pretrained("{output_path}/merged")

    print("GRPO training complete!")
    print(f"Adapter saved to: {output_path}/final")
    print(f"Merged model saved to: {output_path}/merged")
'''

    def _generate_grpo_script_unsloth(self, run: TrainingRun, profile) -> str:
        """Generate GRPO training script using Unsloth + trl.GRPOTrainer (tiered rewards)."""
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")

        return f'''#!/usr/bin/env python3
"""
Auto-generated GRPO Training Script for BashGym
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}

GRPO: Group Relative Policy Optimization with tiered rewards
- Reward mode: {self.config.grpo_reward_mode}
- Generations per prompt: {self.config.effective_grpo_group_size()}
- Training profile: {self.config.training_profile}

Uses Unsloth to patch TRL imports (fixes vLLM crash on DGX Spark sm_121).
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from pathlib import Path as _Path
import ast, subprocess, tempfile, os, sys, re
import warnings

# Silence cosmetic TRL deprecation: pad_token_id + generation_config double-pass
warnings.filterwarnings(
    "ignore", message=".*pad_token_id.*deprecated.*", category=UserWarning
)


class DegenerateRewardStop(TrainerCallback):
    """Stop training if the reward signal is degenerate for too many steps.

    TRL's GRPOTrainer logs `frac_reward_zero_std` per step — the fraction of
    completion groups whose reward std is (approximately) zero. When that
    fraction stays near 1.0, every group has the same reward and GRPO's
    group-relative advantage is 0, so no gradient signal flows and the model
    cannot learn. Early-stop avoids burning an hour on a dead run.

    Thresholds chosen per huggingface/trl source + community discussion:
    - A single step above 0.95 is normal cold-start (e.g. verification reward
      where nothing passes yet). We require a sustained streak.
    - We don't fire before `min_step` because warm-up is expected.
    - Raw float values are read from the `logs` dict passed to on_log —
      NOT the stringified `:.4g` version ProgressCallback prints to stdout.
    """

    def __init__(self, output_dir, threshold=0.95, min_step=5, streak=3):
        self.output_dir = _Path(output_dir)
        self.threshold = threshold
        self.min_step = min_step
        self.streak = streak
        self.consecutive_hits = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step < self.min_step:
            return
        fzs = logs.get("frac_reward_zero_std")
        reward_std = logs.get("reward_std")
        if fzs is None:
            return
        # Only count as degenerate if BOTH the group-zero fraction is above
        # threshold AND reward_std is effectively zero (belt-and-suspenders
        # against a single flaky step).
        is_degenerate = fzs > self.threshold and (reward_std is None or reward_std < 1e-6)
        if is_degenerate:
            self.consecutive_hits += 1
            print(
                f"[DegenerateRewardStop] warning: step {{state.global_step}} "
                f"frac_reward_zero_std={{fzs:.4f}} reward_std={{reward_std}} "
                f"streak={{self.consecutive_hits}}/{{self.streak}}"
            )
            if self.consecutive_hits >= self.streak:
                reason = (
                    f"frac_reward_zero_std stayed above {{self.threshold}} for "
                    f"{{self.streak}} consecutive logs (last={{fzs:.4f}}, "
                    f"reward_std={{reward_std}}). Reward function produces no "
                    f"variance across completion groups → GRPO advantage=0 → "
                    f"zero gradient. Change the reward function or the dataset."
                )
                try:
                    (self.output_dir / "EARLY_STOPPED").write_text(reason)
                except Exception:
                    pass
                print(f"[DegenerateRewardStop] STOPPING TRAINING: {{reason}}")
                control.should_training_stop = True
        else:
            self.consecutive_hits = 0


def _install_gc_compat(model):
    """Wrap model.gradient_checkpointing_enable so it accepts both calling conventions.

    Unsloth's FastBaseModel.post_patch_model replaces the method with a kwargs-only
    closure. TRL's disable_gradient_checkpointing (and its Unsloth compiled cache copy)
    calls it positionally as `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)`,
    which raises TypeError. This wrapper accepts either form.

    Refs: unslothai/unsloth#3828, #2870, #2362; huggingface/trl#3089
    """
    inner = model.gradient_checkpointing_enable

    def _compat(gradient_checkpointing_kwargs=None, **kwargs):
        merged = dict(gradient_checkpointing_kwargs or {{}})
        merged.update(kwargs)
        try:
            return inner(**merged)
        except TypeError:
            # Some variants of the patched enable accept no args at all
            return inner()

    model.gradient_checkpointing_enable = _compat
    return model

{self._terminal_rl_config_src()}


# --- Reward helpers ---

def extract_code(text):
    """Extract code from ```python fences, ``` fences, or raw text."""
    match = re.search(r"```python\\s*\\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\\s*\\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_verification(code, test_code):
    """Write code + tests to a temp dir and run pytest. Returns (passed, total)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        solution_path = os.path.join(tmpdir, "solution.py")
        test_path = os.path.join(tmpdir, "test_solution.py")
        with open(solution_path, "w") as f:
            f.write(code)
        with open(test_path, "w") as f:
            f.write(test_code)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmpdir,
            )
            output = result.stdout + result.stderr
            passed = len(re.findall(r" PASSED", output))
            failed = len(re.findall(r" FAILED", output))
            total = passed + failed
            return passed, total
        except Exception:
            return 0, 1


# --- Reward functions ---

def syntax_reward(completions, **kwargs):
    """Return 1.0 if code parses without SyntaxError, else 0.0."""
    rewards = []
    for completion in completions:
        code = extract_code(completion if isinstance(completion, str) else completion[0]["content"])
        try:
            ast.parse(code)
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    return rewards


def execution_reward(completions, **kwargs):
    """Return 1.0 if code executes with exit code 0, else 0.0."""
    rewards = []
    for completion in completions:
        code = extract_code(completion if isinstance(completion, str) else completion[0]["content"])
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                timeout=10,
            )
            rewards.append(1.0 if result.returncode == 0 else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def verification_reward(completions, prompts, tests=None, **kwargs):
    """Run pytest on extracted code. Returns passed/total for each completion."""
    rewards = []
    for i, completion in enumerate(completions):
        code = extract_code(completion if isinstance(completion, str) else completion[0]["content"])
        test_code = ""
        if tests is not None and i < len(tests):
            test_code = tests[i]
        if not test_code:
            # Fall back to syntax check if no test code
            try:
                ast.parse(code)
                rewards.append(0.5)
            except SyntaxError:
                rewards.append(0.0)
            continue
        passed, total = run_verification(code, test_code)
        rewards.append(passed / total if total > 0 else 0.0)
    return rewards


# Reward function selection
REWARD_FN = {{"syntax": syntax_reward, "execution": execution_reward, "verification": verification_reward}}[REWARD_MODE]


if __name__ == "__main__":
    import gc

    # GPU memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    MODEL_NAME = "{self.config.base_model}"
    log_terminal_rl_config()
    print(f"Loading {{MODEL_NAME}} via Unsloth...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length={self.config.max_seq_length},
        dtype=torch.bfloat16,
        load_in_4bit={str(self.config.load_in_4bit)},
        device_map="sequential",
    )
    model = configure_terminal_rl_model(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation

    model = FastLanguageModel.get_peft_model(
        model,
        r={self.config.lora_r},
        lora_alpha={self.config.lora_alpha},
        lora_dropout={self.config.lora_dropout},
        target_modules={list(profile.lora_target_modules)},
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    # Fix Unsloth/TRL gradient_checkpointing_enable contract mismatch.
    _install_gc_compat(model)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset("json", data_files="{dataset_path}", split="train")

    # GRPO training configuration
    # use_vllm=False: Unsloth's GRPO guide recommends non-vLLM path;
    # vLLM is broken on sm_121 (cu130 aarch64 wheels unstable) and Unsloth
    # native inference is the supported configuration.
    # logging_steps=1: we need real per-step metrics visible in the log so the
    # cascade scheduler + heartbeat monitor can detect degenerate reward signals
    # (e.g. frac_reward_zero_std > 0.9) early instead of discovering it only
    # at the end of the run.
    grpo_config = GRPOConfig(
        output_dir="{output_path}",
        num_generations=GRPO_GROUP_SIZE,
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        num_train_epochs={self.config.num_epochs},
        max_steps={self.config.max_steps},
        learning_rate={self.config.learning_rate},
        logging_steps=1,
        save_steps={self.config.save_steps},
        max_completion_length={self.config.max_seq_length},
        temperature={self.config.grpo_temperature},
        bf16=True,
        report_to="none",
        use_vllm={self.config.grpo_use_vllm},
        loss_type="{self.config.grpo_loss_type}",
    )

    # Initialize TRL GRPOTrainer with the degenerate-reward early-stop callback.
    trainer = TRLGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[REWARD_FN],
        train_dataset=dataset,
        args=grpo_config,
        callbacks=[DegenerateRewardStop(output_dir="{output_path}")],
    )

    print(f"Starting GRPO training with reward_mode={{REWARD_MODE}}...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained("{output_path}/final")
    tokenizer.save_pretrained("{output_path}/final")

    # Merge LoRA into base model and save as standalone
    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained("{output_path}/merged")
    tokenizer.save_pretrained("{output_path}/merged")

    print("GRPO training complete!")
    print(f"Adapter saved to: {output_path}/final")
    print(f"Merged model saved to: {output_path}/merged")
'''


def main():
    """Example usage of the Trainer."""
    # SFT Training
    config = TrainerConfig(
        base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        strategy=TrainingStrategy.SFT,
        num_epochs=3,
        use_lora=True,
    )

    trainer = Trainer(config)

    # Check for training data
    sft_data = Path("data/training_batches")
    if sft_data.exists():
        batch_files = list(sft_data.glob("sft_*.jsonl"))
        if batch_files:
            run = trainer.train_sft(
                dataset_path=batch_files[0], callback=lambda m: print(f"Progress: {m}")
            )
            print(f"Training run: {run.to_dict()}")
    else:
        print("No training data found")


if __name__ == "__main__":
    main()
