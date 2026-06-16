"""Training backends — submit a fine-tune to local, SSH, or managed-API compute."""

from .base import TrainingBackend, TrainingJob, TrainingSpec, TrainingStatus
from .managed import DIALECTS, FineTuneDialect, ManagedFineTuneBackend

__all__ = [
    "TrainingBackend",
    "TrainingJob",
    "TrainingSpec",
    "TrainingStatus",
    "ManagedFineTuneBackend",
    "FineTuneDialect",
    "DIALECTS",
]
