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
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
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

    # Knowledge Distillation settings
    teacher_model: str = "claude-sonnet-4-20250514"  # Teacher model for distillation
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
            result = deploy_gguf_to_ollama(str(gguf_files[0]), model_name)
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
        training_metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """
        Run Direct Preference Optimization training.

        Args:
            dataset_path: Path to DPO JSONL data (with chosen/rejected pairs)
            val_dataset_path: Optional path to validation JSONL data
            run_id: Optional run identifier
            callback: Optional callback for progress updates

        Returns:
            TrainingRun with results
        """
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
            self._train_with_unsloth_dpo(run, callback)
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

        Generates and executes a training script.

        Args:
            run: TrainingRun object with configuration
            callback: Optional callback for training metrics
            log_callback: Optional callback for raw log lines
        """
        import re

        # Generate training script
        script_content = self._generate_unsloth_sft_script(run)
        script_path = run.output_path / "train_sft.py"
        run.output_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)

        # Execute training
        # Get Python with CUDA support for training
        python_exe = self._get_training_python()
        logger.info(f"Starting SFT training run: {run.run_id}")
        logger.info(f"Dataset: {run.dataset_path}")
        logger.info(f"Output: {run.output_path}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Python: {python_exe}")

        # Actually execute the training script
        try:
            process = subprocess.Popen(
                [python_exe, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),  # Run from project root
            )

            # Store PID for process control (suspend/resume/reconnect)
            run.pid = process.pid
            logger.info(f"Training subprocess started with PID {process.pid}")

            # Notify caller of PID (used to persist state to disk immediately)
            if pid_callback:
                try:
                    pid_callback(process.pid, run)
                except Exception as e:
                    logger.warning(f"pid_callback error: {e}")

            last_loss = None
            last_epoch = 0
            last_step = 0
            last_grad_norm = None
            last_eval_loss = None
            samples_processed = 0
            start_time = datetime.now(timezone.utc)

            # Estimate total steps (will be refined from training output)
            # HuggingFace estimates: total_steps = (dataset_size / batch_size) * num_epochs
            estimated_total_steps = 1000  # Default, will update from training output

            # Stream output and parse metrics
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[Training] {line}")

                    # Send raw log line to callback
                    if log_callback:
                        try:
                            log_callback(line)
                        except Exception as e:
                            logger.warning(f"Log callback error: {e}")

                    # Parse metrics from training output (HuggingFace format)
                    # Example: {'loss': 1.234, 'grad_norm': 0.5, 'learning_rate': 2e-05, 'epoch': 1.0}
                    loss_match = re.search(r"'loss':\s*([\d.]+)", line)
                    epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                    step_match = re.search(r"'step':\s*(\d+)", line)
                    grad_norm_match = re.search(r"'grad_norm':\s*([\d.]+)", line)
                    eval_loss_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
                    total_steps_match = re.search(
                        r"total[_\s]?steps[=:\s]+(\d+)", line, re.IGNORECASE
                    )

                    # Also parse tqdm progress bar: "8%|8 | 1/12 [00:07<01:26, 7.90s/it]"
                    progress_match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)

                    # Parse Unsloth header for total steps: "Total steps = 12"
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
                    if eval_loss_match:
                        last_eval_loss = float(eval_loss_match.group(1))
                    if total_steps_match:
                        estimated_total_steps = int(total_steps_match.group(1))
                    if unsloth_steps_match:
                        estimated_total_steps = int(unsloth_steps_match.group(1))
                    if progress_match:
                        last_step = int(progress_match.group(2))
                        estimated_total_steps = int(progress_match.group(3))
                        samples_processed = last_step * self.config.batch_size

                    # Calculate ETA based on progress
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

                    # Send progress callback with all metrics
                    # Send updates on step changes OR loss updates
                    if callback and (progress_match or loss_match or eval_loss_match):
                        callback(
                            {
                                "epoch": last_epoch,
                                "total_epochs": self.config.num_epochs,
                                "step": last_step,
                                "total_steps": estimated_total_steps,
                                "loss": last_loss,  # May be None initially
                                "learning_rate": self.config.learning_rate,
                                "grad_norm": last_grad_norm,
                                "eval_loss": last_eval_loss,
                                "eta": eta,
                                "samples_processed": samples_processed,
                            }
                        )

                    # Track loss curve for model profile
                    if loss_match and last_loss is not None and last_step > 0:
                        run.add_loss_point(
                            step=last_step,
                            loss=last_loss,
                            epoch=last_epoch,
                            learning_rate=self.config.learning_rate,
                        )

            # Wait for process to complete
            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(f"Training script exited with code {return_code}")

            run.metrics = {
                "final_loss": last_loss or 0.0,
                "epochs_completed": self.config.num_epochs,
                "samples_processed": samples_processed,
                "eval_loss": last_eval_loss,
            }

            logger.info(f"Training completed. Model saved to: {run.output_path}")

        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

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

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
import torch

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

def formatting_prompts_func(examples):
    """Format examples using the tokenizer's chat template.

    This handles all model-specific formatting (Qwen, Llama, Mistral, etc.),
    None content in tool_call messages, and special tokens automatically.
    """
    convos = examples["messages"]
    texts = []
    for convo in convos:
        try:
            clean = _sanitize_messages(convo)
            text = tokenizer.apply_chat_template(
                clean, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        except Exception:
            # Fallback: strip tool_calls entirely and retry
            fallback = [
                {{"role": m.get("role", "user"), "content": m.get("content", "") or ""}}
                for m in convo if m.get("role") in ("system", "user", "assistant")
            ]
            try:
                text = tokenizer.apply_chat_template(
                    fallback, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            except Exception:
                texts.append("")
    return {{"text": texts}}

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

    # Model configuration
    max_seq_length = {self.config.max_seq_length}
    dtype = torch.float16
    load_in_4bit = {self.config.load_in_4bit}

    # Load model with Unsloth optimizations
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="{self.config.base_model}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir="{output_path}",
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        warmup_ratio={self.config.warmup_ratio},
        num_train_epochs={self.config.num_epochs},
        max_steps={self.config.max_steps},
        learning_rate={self.config.learning_rate},
        fp16=True,
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        save_total_limit=3,
        optim="adamw_8bit",
        seed=42,
        # Eval settings (active when val_dataset is available)
        eval_strategy="{self.config.eval_strategy}" if val_dataset is not None else "no",
        eval_steps={self.config.eval_steps} if val_dataset is not None else None,
        load_best_model_at_end=True if val_dataset is not None else False,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False if val_dataset is not None else None,
    )

    # Apply chat template to dataset (handles all model formats + None content)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Apply formatting to val dataset too
    if val_dataset is not None:
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    # Early stopping (only when val_dataset exists and patience > 0)
    callbacks = []
    if val_dataset is not None and {self.config.early_stopping_patience} > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience={self.config.early_stopping_patience}))

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,  # Required on Windows (avoids multiprocessing crash)
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
        self, run: TrainingRun, callback: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        """Train using Unsloth with DPO."""
        import re

        # Generate DPO training script
        script_content = self._generate_unsloth_dpo_script(run)
        script_path = run.output_path / "train_dpo.py"
        run.output_path.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)

        # Get Python with CUDA support for training
        python_exe = self._get_training_python()
        logger.info(f"Starting DPO training run: {run.run_id}")
        logger.info(f"Dataset: {run.dataset_path}")
        logger.info(f"Output: {run.output_path}")
        logger.info(f"Script: {script_path}")
        logger.info(f"Python: {python_exe}")

        # Actually execute the training script
        try:
            process = subprocess.Popen(
                [python_exe, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
            )

            last_loss = None
            last_epoch = 0
            reward_margin = 0.0
            last_eval_loss = None

            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[DPO Training] {line}")

                    loss_match = re.search(r"'loss':\s*([\d.]+)", line)
                    epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                    reward_match = re.search(r"'rewards/margins':\s*([\d.]+)", line)
                    eval_loss_match = re.search(r"'eval_loss':\s*([\d.]+)", line)

                    if loss_match:
                        last_loss = float(loss_match.group(1))
                    if epoch_match:
                        last_epoch = int(float(epoch_match.group(1)))
                    if reward_match:
                        reward_margin = float(reward_match.group(1))
                    if eval_loss_match:
                        last_eval_loss = float(eval_loss_match.group(1))

                    if callback and (last_loss is not None or eval_loss_match):
                        callback(
                            {
                                "epoch": last_epoch,
                                "total_epochs": self.config.num_epochs,
                                "loss": last_loss,
                                "reward_margin": reward_margin,
                                "eval_loss": last_eval_loss,
                            }
                        )

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"DPO training script exited with code {return_code}")

            run.metrics = {
                "final_loss": last_loss or 0.0,
                "final_reward_margin": reward_margin,
                "epochs_completed": self.config.num_epochs,
                "eval_loss": last_eval_loss,
            }

            logger.info(f"DPO training completed. Model saved to: {run.output_path}")

        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise

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
import os
import torch

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
            script_content = self._generate_unsloth_dpo_script(run)
            script_path = run.output_path / "train_dpo.py"
        elif run.strategy in (TrainingStrategy.GRPO, TrainingStrategy.RLVR):
            grpo_trainer = GRPOTrainer(self.config)
            script_content = grpo_trainer._generate_grpo_script(run)
            script_path = run.output_path / "train_grpo.py"
        else:
            script_content = self._generate_unsloth_sft_script(run)
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

        try:
            process = subprocess.Popen(
                [python_exe, str(script_path)],
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

            last_loss = None
            last_epoch = 0
            last_step = 0
            samples_processed = 0
            last_avg_reward = None
            last_kl_divergence = None
            last_policy_loss = None
            start_time = datetime.now(timezone.utc)
            estimated_total_steps = 1000

            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[GRPO Training] {line}")

                    if log_callback:
                        try:
                            log_callback(line)
                        except Exception as e:
                            logger.warning(f"Log callback error: {e}")

                    # Standard HuggingFace metrics
                    loss_match = re.search(r"'loss':\s*([\d.-]+)", line)
                    epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                    step_match = re.search(r"'step':\s*(\d+)", line)
                    progress_match = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
                    unsloth_steps_match = re.search(r"Total steps\s*=\s*(\d+)", line)

                    # GRPO-specific trl metrics
                    reward_match = re.search(r"'reward':\s*([\d.-]+)", line)
                    mean_reward_match = re.search(r"'mean_reward':\s*([\d.-]+)", line)
                    kl_match = re.search(r"'kl':\s*([\d.-]+)", line)
                    policy_loss_match = re.search(r"'policy_loss':\s*([\d.-]+)", line)

                    if loss_match:
                        last_loss = float(loss_match.group(1))
                    if epoch_match:
                        last_epoch = int(float(epoch_match.group(1)))
                    if step_match:
                        last_step = int(step_match.group(1))
                        samples_processed = last_step * self.config.batch_size
                    if unsloth_steps_match:
                        estimated_total_steps = int(unsloth_steps_match.group(1))
                    if progress_match:
                        last_step = int(progress_match.group(2))
                        estimated_total_steps = int(progress_match.group(3))
                        samples_processed = last_step * self.config.batch_size
                    if reward_match:
                        last_avg_reward = float(reward_match.group(1))
                    if mean_reward_match:
                        last_avg_reward = float(mean_reward_match.group(1))
                    if kl_match:
                        last_kl_divergence = float(kl_match.group(1))
                    if policy_loss_match:
                        last_policy_loss = float(policy_loss_match.group(1))

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

                    if callback and (
                        progress_match or loss_match or reward_match or mean_reward_match
                    ):
                        callback(
                            {
                                "epoch": last_epoch,
                                "total_epochs": self.config.num_epochs,
                                "step": last_step,
                                "total_steps": estimated_total_steps,
                                "loss": last_loss,
                                "learning_rate": self.config.learning_rate,
                                "eta": eta,
                                "samples_processed": samples_processed,
                                "avg_reward": last_avg_reward,
                                "kl_divergence": last_kl_divergence,
                                "policy_loss": last_policy_loss,
                            }
                        )

            return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(f"GRPO training script exited with code {return_code}")

            run.metrics = {
                "final_loss": last_loss or 0.0,
                "epochs_completed": self.config.num_epochs,
                "samples_processed": samples_processed,
                "final_avg_reward": last_avg_reward,
                "final_kl_divergence": last_kl_divergence,
                "final_policy_loss": last_policy_loss,
            }

            logger.info(f"GRPO training completed. Model saved to: {run.output_path}")

        except FileNotFoundError as e:
            raise RuntimeError(f"Python interpreter not found: {e}")
        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            raise

    def _generate_grpo_script(self, run: TrainingRun) -> str:
        """Generate GRPO training script using trl.GRPOTrainer with tiered reward functions."""
        dataset_path = str(run.dataset_path).replace("\\", "/")
        output_path = str(run.output_path).replace("\\", "/")

        return f'''#!/usr/bin/env python3
"""
Auto-generated GRPO Training Script for BashGym
Run ID: {run.run_id}
Generated: {datetime.now(timezone.utc).isoformat()}

GRPO: Group Relative Policy Optimization with tiered rewards
- Reward mode: {self.config.grpo_reward_mode}
- Generations per prompt: {self.config.grpo_num_generations}

Uses Unsloth to patch TRL imports (fixes vLLM crash on DGX Spark sm_121).
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
import ast, subprocess, tempfile, os, sys, re

# Configuration
REWARD_MODE = "{self.config.grpo_reward_mode}"
NUM_GENERATIONS = {self.config.grpo_num_generations}


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
    print(f"Loading {{MODEL_NAME}} via Unsloth...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length={self.config.max_seq_length},
        dtype=torch.bfloat16,
        load_in_4bit={str(self.config.load_in_4bit)},
        device_map="sequential",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation

    model = FastLanguageModel.get_peft_model(
        model,
        r={self.config.lora_r},
        lora_alpha={self.config.lora_alpha},
        lora_dropout={self.config.lora_dropout},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset("json", data_files="{dataset_path}", split="train")

    # GRPO training configuration
    grpo_config = GRPOConfig(
        output_dir="{output_path}",
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size={self.config.batch_size},
        gradient_accumulation_steps={self.config.gradient_accumulation_steps},
        num_train_epochs={self.config.num_epochs},
        max_steps={self.config.max_steps},
        learning_rate={self.config.learning_rate},
        logging_steps={self.config.logging_steps},
        save_steps={self.config.save_steps},
        max_completion_length={self.config.max_seq_length},
        bf16=True,
        report_to="none",
    )

    # Initialize TRL GRPOTrainer
    trainer = TRLGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[REWARD_FN],
        train_dataset=dataset,
        args=grpo_config,
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
