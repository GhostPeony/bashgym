"""Verification-layer APIs with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "bashgym.judge.verifier": (
        "Verifier",
        "VerificationConfig",
        "VerificationResult",
        "VerificationStatus",
    ),
    "bashgym.judge.evaluator": (
        "EvaluatorClient",
        "EvaluatorConfig",
        "EvaluationResult",
        "JudgeScore",
    ),
    "bashgym.judge.guardrails": (
        "NemoGuard",
        "GuardrailsConfig",
        "GuardrailResult",
        "CheckResult",
    ),
    "bashgym.judge.benchmarks": (
        "BenchmarkRunner",
        "BenchmarkConfig",
        "BenchmarkResult",
        "BenchmarkType",
    ),
}
_EXPORTS = {name: module_name for module_name, names in _MODULE_EXPORTS.items() for name in names}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
