"""Training backend abstraction — run a fine-tune on a chosen compute target.

Separates *what* to train (``TrainingSpec``) from *where* (``TrainingBackend``):
a local GPU, an SSH-reachable host (the existing ``remote_trainer``), or a
managed fine-tuning API (Together, Fireworks, OpenAI). Backends share one
contract — ``submit`` -> ``poll`` -> artifacts — with a normalized
``TrainingStatus`` so callers don't special-case the platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def terminal(self) -> bool:
        return self in (
            TrainingStatus.SUCCEEDED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        )


@dataclass
class TrainingSpec:
    """Platform-agnostic description of a fine-tune to run."""

    base_model: str
    dataset_path: Path
    strategy: str = "sft"  # sft | dpo
    n_epochs: int = 1
    learning_rate: float = 1e-5
    suffix: str = ""  # name suffix for the resulting model
    extra: dict[str, Any] = field(default_factory=dict)  # platform-specific job knobs


@dataclass
class TrainingJob:
    job_id: str
    backend: str
    status: TrainingStatus = TrainingStatus.PENDING
    output_model: str | None = None  # the resulting fine-tuned model id/path
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "backend": self.backend,
            "status": self.status.value,
            "output_model": self.output_model,
            "error": self.error,
        }


class TrainingBackend(ABC):
    """Submit a fine-tune to a compute target and track it to completion."""

    @property
    @abstractmethod
    def backend_type(self) -> str: ...

    @abstractmethod
    async def submit(self, spec: TrainingSpec) -> TrainingJob: ...

    @abstractmethod
    async def poll(self, job: TrainingJob) -> TrainingJob: ...

    async def cancel(self, job: TrainingJob) -> TrainingJob:
        """Best-effort cancel; backends override where the platform supports it."""
        return job
