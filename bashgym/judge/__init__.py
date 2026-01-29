"""Judge - Verification Layer

Includes:
- Verifier: Test execution and verification
- EvaluatorClient: NeMo Evaluator integration
- NemoGuard: Guardrails and safety checks
- BenchmarkRunner: Standard code benchmarks
"""

from bashgym.judge.verifier import Verifier, VerificationConfig, VerificationResult, VerificationStatus
from bashgym.judge.evaluator import EvaluatorClient, EvaluatorConfig, EvaluationResult, JudgeScore
from bashgym.judge.guardrails import NemoGuard, GuardrailsConfig, GuardrailResult, CheckResult
from bashgym.judge.benchmarks import BenchmarkRunner, BenchmarkConfig, BenchmarkResult, BenchmarkType

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
