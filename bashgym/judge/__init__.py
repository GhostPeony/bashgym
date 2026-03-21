"""Judge - Verification Layer

Includes:
- Verifier: Test execution and verification
- EvaluatorClient: NeMo Evaluator integration
- NemoGuard: Guardrails and safety checks
- BenchmarkRunner: Standard code benchmarks
"""

from bashgym.judge.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkType,
)
from bashgym.judge.evaluator import EvaluationResult, EvaluatorClient, EvaluatorConfig, JudgeScore
from bashgym.judge.guardrails import CheckResult, GuardrailResult, GuardrailsConfig, NemoGuard
from bashgym.judge.verifier import (
    VerificationConfig,
    VerificationResult,
    VerificationStatus,
    Verifier,
)

__all__ = [
    # Verifier
    "Verifier",
    "VerificationConfig",
    "VerificationResult",
    "VerificationStatus",
    # Evaluator
    "EvaluatorClient",
    "EvaluatorConfig",
    "EvaluationResult",
    "JudgeScore",
    # Guardrails
    "NemoGuard",
    "GuardrailsConfig",
    "GuardrailResult",
    "CheckResult",
    # Benchmarks
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkType",
]
