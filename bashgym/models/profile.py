"""
Model Profile - Rich metadata for trained models

Captures full lifecycle data:
- Identity: name, tags, description
- Lineage: base model, training strategy, source traces
- Training: config, loss curve, metrics
- Artifacts: checkpoints, exports
- Evaluations: benchmarks, custom evals
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class LossPoint:
    """A single point on the loss curve."""
    step: int
    loss: float
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[int] = None


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    path: str
    step: int
    epoch: Optional[int] = None
    loss: Optional[float] = None


@dataclass
class GGUFExport:
    """Information about a GGUF export."""
    path: str
    quantization: str  # e.g., "Q4_K_M", "Q8_0"
    size_bytes: int
    created_at: datetime


@dataclass
class ModelArtifacts:
    """All artifacts associated with a model."""
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    final_adapter_path: Optional[str] = None
    merged_path: Optional[str] = None
    gguf_exports: List[GGUFExport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoints": [
                {"path": c.path, "step": c.step, "epoch": c.epoch, "loss": c.loss}
                for c in self.checkpoints
            ],
            "final_adapter_path": self.final_adapter_path,
            "merged_path": self.merged_path,
            "gguf_exports": [
                {
                    "path": g.path,
                    "quantization": g.quantization,
                    "size_bytes": g.size_bytes,
                    "created_at": g.created_at.isoformat()
                }
                for g in self.gguf_exports
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelArtifacts":
        return cls(
            checkpoints=[
                CheckpointInfo(
                    path=c["path"],
                    step=c["step"],
                    epoch=c.get("epoch"),
                    loss=c.get("loss")
                )
                for c in data.get("checkpoints", [])
            ],
            final_adapter_path=data.get("final_adapter_path"),
            merged_path=data.get("merged_path"),
            gguf_exports=[
                GGUFExport(
                    path=g["path"],
                    quantization=g["quantization"],
                    size_bytes=g["size_bytes"],
                    created_at=datetime.fromisoformat(g["created_at"])
                )
                for g in data.get("gguf_exports", [])
            ]
        )


@dataclass
class BenchmarkResult:
    """Result from a standard benchmark."""
    benchmark_name: str
    score: float  # Primary score (e.g., pass@1)
    passed: int
    total: int
    metrics: Dict[str, float] = field(default_factory=dict)  # Additional metrics
    evaluated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "passed": self.passed,
            "total": self.total,
            "metrics": self.metrics,
            "evaluated_at": self.evaluated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        return cls(
            benchmark_name=data["benchmark_name"],
            score=data["score"],
            passed=data["passed"],
            total=data["total"],
            metrics=data.get("metrics", {}),
            evaluated_at=datetime.fromisoformat(data["evaluated_at"])
        )


@dataclass
class CustomEvalResult:
    """Result from a custom evaluation set."""
    eval_set_id: str
    eval_type: str  # "replay" or "variation"
    passed: int
    total: int
    pass_rate: float
    failures: List[Dict[str, Any]] = field(default_factory=list)  # Failed case details
    evaluated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_set_id": self.eval_set_id,
            "eval_type": self.eval_type,
            "passed": self.passed,
            "total": self.total,
            "pass_rate": self.pass_rate,
            "failures": self.failures,
            "evaluated_at": self.evaluated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomEvalResult":
        return cls(
            eval_set_id=data["eval_set_id"],
            eval_type=data["eval_type"],
            passed=data["passed"],
            total=data["total"],
            pass_rate=data["pass_rate"],
            failures=data.get("failures", []),
            evaluated_at=datetime.fromisoformat(data["evaluated_at"])
        )


@dataclass
class EvaluationRecord:
    """A timestamped evaluation snapshot for tracking trends."""
    evaluated_at: datetime
    benchmarks: Dict[str, float]  # benchmark_name -> score
    custom_eval_score: Optional[float] = None
    overall_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluated_at": self.evaluated_at.isoformat(),
            "benchmarks": self.benchmarks,
            "custom_eval_score": self.custom_eval_score,
            "overall_score": self.overall_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRecord":
        return cls(
            evaluated_at=datetime.fromisoformat(data["evaluated_at"]),
            benchmarks=data.get("benchmarks", {}),
            custom_eval_score=data.get("custom_eval_score"),
            overall_score=data.get("overall_score")
        )


@dataclass
class ModelProfile:
    """
    Complete profile for a trained model.

    Captures the full lifecycle: identity, lineage, training, artifacts, and evaluations.
    """

    # Identity
    model_id: str                                    # Unique identifier
    run_id: str                                      # Training run ID
    display_name: str                                # User-editable name
    description: str = ""                            # User-editable description
    tags: List[str] = field(default_factory=list)   # e.g., ["production", "v3"]
    starred: bool = False                            # Pinned to top
    created_at: datetime = field(default_factory=datetime.now)

    # Lineage
    base_model: str = ""                             # e.g., "Qwen2.5-Coder-1.5B"
    training_strategy: str = "sft"                   # sft, dpo, grpo, distillation
    teacher_model: Optional[str] = None              # If distillation
    training_traces: List[str] = field(default_factory=list)  # Gold trace IDs used
    parent_model: Optional[str] = None               # If fine-tuned from trained model
    training_repos: List[str] = field(default_factory=list)   # Repos traces came from

    # Training
    config: Dict[str, Any] = field(default_factory=dict)  # Full TrainerConfig
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    loss_curve: List[Dict[str, Any]] = field(default_factory=list)  # [{step, loss}, ...]
    final_metrics: Dict[str, float] = field(default_factory=dict)   # {final_loss, epochs}

    # Artifacts
    artifacts: ModelArtifacts = field(default_factory=ModelArtifacts)
    model_dir: str = ""  # Path to model directory

    # Evaluations
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    custom_evals: Dict[str, CustomEvalResult] = field(default_factory=dict)
    evaluation_history: List[EvaluationRecord] = field(default_factory=list)

    # Operational
    model_size_bytes: int = 0
    model_size_params: Optional[str] = None          # e.g., "1.5B"
    inference_latency_ms: Optional[float] = None
    status: str = "pending"                          # pending, training, ready, needs_eval, archived
    deployed_to: Optional[str] = None                # e.g., "ollama:bashgym-v3"

    # Computed properties
    @property
    def custom_eval_pass_rate(self) -> Optional[float]:
        """Aggregate pass rate across all custom evals."""
        if not self.custom_evals:
            return None
        total_passed = sum(e.passed for e in self.custom_evals.values())
        total_total = sum(e.total for e in self.custom_evals.values())
        return (total_passed / total_total * 100) if total_total > 0 else 0.0

    @property
    def benchmark_avg_score(self) -> Optional[float]:
        """Average score across all benchmarks."""
        if not self.benchmarks:
            return None
        return sum(b.score for b in self.benchmarks.values()) / len(self.benchmarks)

    @property
    def model_size_display(self) -> str:
        """Human-readable model size."""
        if self.model_size_bytes == 0:
            return "Unknown"
        gb = self.model_size_bytes / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f}GB"
        mb = self.model_size_bytes / (1024 ** 2)
        return f"{mb:.0f}MB"

    @property
    def training_duration_display(self) -> str:
        """Human-readable training duration."""
        if self.duration_seconds == 0:
            return "Unknown"
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        return {
            # Identity
            "model_id": self.model_id,
            "run_id": self.run_id,
            "display_name": self.display_name,
            "description": self.description,
            "tags": self.tags,
            "starred": self.starred,
            "created_at": self.created_at.isoformat(),

            # Lineage
            "base_model": self.base_model,
            "training_strategy": self.training_strategy,
            "teacher_model": self.teacher_model,
            "training_traces": self.training_traces,
            "parent_model": self.parent_model,
            "training_repos": self.training_repos,

            # Training
            "config": self.config,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "loss_curve": self.loss_curve,
            "final_metrics": self.final_metrics,

            # Artifacts
            "artifacts": self.artifacts.to_dict(),
            "model_dir": self.model_dir,

            # Evaluations
            "benchmarks": {k: v.to_dict() for k, v in self.benchmarks.items()},
            "custom_evals": {k: v.to_dict() for k, v in self.custom_evals.items()},
            "evaluation_history": [e.to_dict() for e in self.evaluation_history],

            # Operational
            "model_size_bytes": self.model_size_bytes,
            "model_size_params": self.model_size_params,
            "inference_latency_ms": self.inference_latency_ms,
            "status": self.status,
            "deployed_to": self.deployed_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelProfile":
        """Deserialize profile from dictionary."""
        return cls(
            # Identity
            model_id=data["model_id"],
            run_id=data["run_id"],
            display_name=data["display_name"],
            description=data.get("description", ""),
            tags=data.get("tags", []),
            starred=data.get("starred", False),
            created_at=datetime.fromisoformat(data["created_at"]),

            # Lineage
            base_model=data.get("base_model", ""),
            training_strategy=data.get("training_strategy", "sft"),
            teacher_model=data.get("teacher_model"),
            training_traces=data.get("training_traces", []),
            parent_model=data.get("parent_model"),
            training_repos=data.get("training_repos", []),

            # Training
            config=data.get("config", {}),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_seconds=data.get("duration_seconds", 0.0),
            loss_curve=data.get("loss_curve", []),
            final_metrics=data.get("final_metrics", {}),

            # Artifacts
            artifacts=ModelArtifacts.from_dict(data.get("artifacts", {})),
            model_dir=data.get("model_dir", ""),

            # Evaluations
            benchmarks={
                k: BenchmarkResult.from_dict(v)
                for k, v in data.get("benchmarks", {}).items()
            },
            custom_evals={
                k: CustomEvalResult.from_dict(v)
                for k, v in data.get("custom_evals", {}).items()
            },
            evaluation_history=[
                EvaluationRecord.from_dict(e)
                for e in data.get("evaluation_history", [])
            ],

            # Operational
            model_size_bytes=data.get("model_size_bytes", 0),
            model_size_params=data.get("model_size_params"),
            inference_latency_ms=data.get("inference_latency_ms"),
            status=data.get("status", "pending"),
            deployed_to=data.get("deployed_to"),
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """Save profile to JSON file."""
        if path is None:
            path = Path(self.model_dir) / "model_profile.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path) -> "ModelProfile":
        """Load profile from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def add_benchmark_result(self, result: BenchmarkResult):
        """Add a benchmark result and update history."""
        self.benchmarks[result.benchmark_name] = result
        self._update_evaluation_history()

    def add_custom_eval_result(self, result: CustomEvalResult):
        """Add a custom eval result and update history."""
        self.custom_evals[result.eval_set_id] = result
        self._update_evaluation_history()

    def _update_evaluation_history(self):
        """Add current evaluation state to history."""
        record = EvaluationRecord(
            evaluated_at=datetime.now(),
            benchmarks={k: v.score for k, v in self.benchmarks.items()},
            custom_eval_score=self.custom_eval_pass_rate,
            overall_score=self.benchmark_avg_score
        )
        self.evaluation_history.append(record)

    def update_status(self):
        """Update status based on current state."""
        if not self.completed_at:
            self.status = "training"
        elif not self.benchmarks and not self.custom_evals:
            self.status = "needs_eval"
        else:
            # Check for regression (simplified - compare to previous in history)
            if len(self.evaluation_history) >= 2:
                prev = self.evaluation_history[-2]
                curr = self.evaluation_history[-1]
                if curr.overall_score and prev.overall_score:
                    if curr.overall_score < prev.overall_score * 0.95:  # 5% drop
                        self.status = "regression_detected"
                        return
            self.status = "ready"
