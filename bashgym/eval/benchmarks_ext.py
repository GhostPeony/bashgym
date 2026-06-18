"""External benchmark orchestration (S6).

Runs third-party eval harnesses against a served model and normalizes their
results into one ``BenchmarkReport``: lm-eval (MMLU/GSM8K/IFEval/HellaSwag),
Terminal-Bench, BFCL, and SWE-bench. Each benchmark is a ``BenchmarkSpec``
(argv to invoke the harness + a parser for its result JSON). The subprocess run
is injected via ``run_command``, so this module stays hermetic — it builds
commands and parses/normalizes results. Scores record onto the model profile
through the registry's ``add_benchmark_result``.

The agent-harness command builders target the documented CLIs; pass an explicit
``extra_args`` list to adapt to a specific harness version.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .forgetting import DEFAULT_FORGETTING_TASKS, lm_eval_command, parse_lm_eval_results


@dataclass
class BenchmarkResult:
    name: str
    score: float  # normalized primary metric in [0, 1]
    passed: int = 0
    total: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "total": self.total,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult]

    @property
    def scores(self) -> dict[str, float]:
        return {r.name: r.score for r in self.results if r.error is None}

    @property
    def failures(self) -> list[str]:
        return [r.name for r in self.results if r.error is not None]

    def to_dict(self) -> dict:
        return {
            "scores": self.scores,
            "failures": self.failures,
            "results": [r.to_dict() for r in self.results],
        }


# ── Result parsers (harness JSON -> normalized BenchmarkResult) ──────────────


def parse_lm_eval(name: str, data: dict) -> BenchmarkResult:
    """lm-eval / NeMo Evaluator: average the per-task headline scores."""
    scores = parse_lm_eval_results(data)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    return BenchmarkResult(name=name, score=avg, total=len(scores), metrics=scores)


def parse_resolved_rate(name: str, data: dict) -> BenchmarkResult:
    """Terminal-Bench / SWE-bench style: resolved instances over total."""
    resolved = data.get("resolved", data.get("n_resolved", data.get("num_resolved", 0)))
    total = data.get("total", data.get("n_instances", data.get("num_instances", 0)))
    if isinstance(resolved, list):
        resolved = len(resolved)
    score = resolved / total if total else 0.0
    return BenchmarkResult(name=name, score=score, passed=int(resolved), total=int(total))


def parse_accuracy(name: str, data: dict) -> BenchmarkResult:
    """BFCL style: an overall accuracy, reported as a fraction or a percentage."""
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    acc = (
        data.get("overall_accuracy")
        or data.get("accuracy")
        or summary.get("overall_accuracy")
        or summary.get("overall")
        or 0.0
    )
    acc = float(acc)
    if acc > 1.0:  # percentage -> fraction
        acc /= 100.0
    return BenchmarkResult(name=name, score=acc, metrics={"accuracy": acc})


# ── Harness command builders (argv) ─────────────────────────────────────────


def terminal_bench_command(
    model: str,
    *,
    agent: str = "terminus",
    dataset: str = "terminal-bench-core",
    n_attempts: int = 1,
    output_path: str = "tb_results",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Terminal-Bench ``tb run``. The served endpoint is set via the litellm env
    (``OPENAI_API_BASE`` / key) the harness reads."""
    argv = [
        "tb",
        "run",
        "--dataset",
        dataset,
        "--agent",
        agent,
        "--model",
        model,
        "--n-attempts",
        str(n_attempts),
        "--output-path",
        output_path,
    ]
    return argv + (extra_args or [])


def bfcl_command(
    model: str,
    *,
    test_category: str = "all",
    backend: str = "openai",
    extra_args: list[str] | None = None,
) -> list[str]:
    """Berkeley Function-Calling Leaderboard generation (``bfcl generate``)."""
    argv = [
        "bfcl",
        "generate",
        "--model",
        model,
        "--test-category",
        test_category,
        "--backend",
        backend,
    ]
    return argv + (extra_args or [])


def swebench_command(
    model: str,
    *,
    subset: str = "lite",
    output_path: str = "swebench_results",
    extra_args: list[str] | None = None,
) -> list[str]:
    """SWE-bench via mini-swe-agent batch mode."""
    argv = ["mini-swe-agent", "--model", model, "--subset", subset, "--output", output_path]
    return argv + (extra_args or [])


# ── Orchestration ───────────────────────────────────────────────────────────


@dataclass
class BenchmarkSpec:
    name: str
    argv: list[str]
    parser: Callable[[str, dict], BenchmarkResult]


def forgetting_suite_spec(
    base_url: str, *, model_name: str = "candidate", tasks=DEFAULT_FORGETTING_TASKS, **kwargs: Any
) -> BenchmarkSpec:
    """The lm-eval forgetting suite as a BenchmarkSpec."""
    return BenchmarkSpec(
        name="forgetting",
        argv=lm_eval_command(base_url, model_name=model_name, tasks=tasks, **kwargs),
        parser=parse_lm_eval,
    )


def run_benchmarks(
    specs: list[BenchmarkSpec], run_command: Callable[[list[str]], dict]
) -> BenchmarkReport:
    """Run each benchmark and normalize results.

    ``run_command(argv) -> result_dict`` is the subprocess seam: it invokes the
    harness and returns its parsed JSON output. A harness that raises is recorded
    as a failed BenchmarkResult rather than aborting the sweep.
    """
    results: list[BenchmarkResult] = []
    for spec in specs:
        try:
            data = run_command(spec.argv)
            results.append(spec.parser(spec.name, data))
        except Exception as exc:  # noqa: BLE001 - one failed harness must not abort the suite
            results.append(BenchmarkResult(name=spec.name, score=0.0, error=str(exc)))
    return BenchmarkReport(results=results)


def record_benchmarks(registry: Any, model_id: str, report: BenchmarkReport) -> None:
    """Write each successful benchmark onto a model profile via the registry."""
    for r in report.results:
        if r.error is None:
            registry.add_benchmark_result(model_id, r.name, r.score, r.passed, r.total, r.metrics)
