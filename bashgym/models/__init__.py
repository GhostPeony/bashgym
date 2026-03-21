"""
Model Registry and Profile Management

Provides model lifecycle management:
- ModelProfile: Rich metadata for trained models
- ModelRegistry: Index and query trained models
- Custom evaluation generation and tracking
"""

from .evaluator import (
    CustomEvalGenerator,
    CustomEvalRunner,
    CustomEvalSet,
    EvalCase,
    EvalCaseResult,
    EvalVerification,
    get_eval_generator,
)
from .profile import (
    BenchmarkResult,
    CustomEvalResult,
    EvaluationRecord,
    ModelArtifacts,
    ModelProfile,
)
from .registry import ModelRegistry, get_registry

__all__ = [
    "ModelProfile",
    "ModelArtifacts",
    "EvaluationRecord",
    "BenchmarkResult",
    "CustomEvalResult",
    "ModelRegistry",
    "get_registry",
    "CustomEvalGenerator",
    "CustomEvalRunner",
    "CustomEvalSet",
    "EvalCase",
    "EvalVerification",
    "EvalCaseResult",
    "get_eval_generator",
]
