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

import re
from collections.abc import Callable, Mapping, Sequence
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


# ── External result ingestion (manual harness JSON -> normalized report) ─────


_SCORE_KEYS = (
    "score",
    "overall",
    "overall_score",
    "overall_acc",
    "overall accuracy",
    "overall acc",
    "pass_rate",
    "success_rate",
    "resolved_rate",
    "resolution_rate",
    "accuracy",
    "overall_accuracy",
    "mean",
    "mean_reward",
    "pass@1",
    "pass_at_1",
    "percent_resolved",
)
_PASSED_KEYS = (
    "passed",
    "resolved",
    "resolved_instances",
    "instances_resolved",
    "n_resolved",
    "num_resolved",
    "successes",
    "success_count",
    "correct",
)
_TOTAL_KEYS = (
    "total",
    "total_instances",
    "n_instances",
    "num_instances",
    "n_tasks",
    "total_tasks",
    "count",
    "num_total",
)
_CONTAINER_KEYS = ("benchmarks", "benchmark_results", "results", "scores")
_DETAIL_KEYS = ("metrics", "summary", "aggregate", "overall", "result")
_BFCL_V4_WEIGHTS = {
    "agentic": 0.4,
    "multi_turn": 0.3,
    "live": 0.1,
    "non_live": 0.1,
    "hallucination": 0.1,
}
_BFCL_CONTAINER_KEYS = (
    "categories",
    "category_scores",
    "breakdown",
    "breakdowns",
    "data_overall",
    "data_live",
    "data_non_live",
    "data_multi_turn",
    "summary",
    "scores",
    "metrics",
)
_SWE_RESOLVED_KEYS = (*_PASSED_KEYS, "resolved_submitted")
_SWE_SUBMITTED_KEYS = (
    "submitted",
    "submitted_instances",
    "instances_submitted",
    "n_submitted",
    "num_submitted",
)
_SWE_COMPLETED_KEYS = (
    "completed",
    "completed_instances",
    "instances_completed",
    "n_completed",
    "num_completed",
)
_SWE_TOTAL_KEYS = (*_TOTAL_KEYS, "instances")
_SWE_ERROR_KEYS = ("errors", "error_count", "num_errors", "failed", "failures")
_SWE_PENDING_KEYS = ("pending", "pending_evaluations", "num_pending")
_SWE_INSTANCE_RESULT_KEYS = ("instance_results", "instances", "instance_results_jsonl", "results")
_SWE_RESOLVED_STATUS = {"resolved", "passed", "success", "succeeded", "fixed", "true", "1"}
_SWE_UNRESOLVED_STATUS = {
    "unresolved",
    "failed",
    "failure",
    "error",
    "errored",
    "timeout",
    "timed_out",
    "false",
    "0",
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        raw = value.strip().replace("%", "")
        if not raw:
            return None
        try:
            parsed = float(raw)
        except ValueError:
            return None
        return parsed / 100.0 if value.strip().endswith("%") else parsed
    return None


def _fraction(value: Any) -> float | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return max(0.0, min(1.0, numeric))


def _count_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    if _is_sequence(value):
        return len(value)
    return None


def _normalize_name(raw: str | None, *, fallback: str = "external_benchmark") -> str:
    name = (raw or fallback).strip()
    if not name:
        name = fallback
    name = name.replace("@", "_").replace("/", "_").replace("-", "_").replace(".", "_")
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return name or fallback


def _metric_name(raw: str | None, *, fallback: str = "metric") -> str:
    return _normalize_name(raw, fallback=fallback)


def _value_by_normalized_key(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    targets = {_normalize_name(key) for key in keys}
    for key, value in data.items():
        if _normalize_name(str(key)) in targets:
            return value
    return None


def _name_from_record(record: Mapping[str, Any], fallback: str) -> str:
    for key in ("benchmark_name", "benchmark", "name", "dataset", "task"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_name(value, fallback=fallback)
    return _normalize_name(fallback)


def _numeric_metrics(data: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        if _is_mapping(value) or _is_sequence(value):
            continue
        numeric = _coerce_float(value)
        if numeric is not None:
            metrics[str(key)] = numeric
    return metrics


def _flatten_numeric_metrics(
    data: Mapping[str, Any],
    *,
    prefix: str = "",
    max_depth: int = 4,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if max_depth < 0:
        return metrics
    for key, value in data.items():
        metric_key = _metric_name(str(key))
        path = f"{prefix}.{metric_key}" if prefix else metric_key
        if _is_mapping(value):
            metrics.update(
                _flatten_numeric_metrics(dict(value), prefix=path, max_depth=max_depth - 1)
            )
            continue
        if _is_sequence(value):
            continue
        numeric = _coerce_float(value)
        if numeric is not None:
            metrics[path] = numeric
    return metrics


def _find_score(data: Mapping[str, Any]) -> float | None:
    value = _value_by_normalized_key(data, _SCORE_KEYS)
    if value is not None:
        score = _fraction(value)
        if score is not None:
            return score
    for detail_key in _DETAIL_KEYS:
        detail = _value_by_normalized_key(data, (detail_key,))
        if _is_mapping(detail):
            score = _find_score(detail)
            if score is not None:
                return score
    return None


def _find_count(data: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    value = _value_by_normalized_key(data, keys)
    if value is not None:
        count = _count_value(value)
        if count is not None:
            return count
    for detail_key in _DETAIL_KEYS:
        detail = _value_by_normalized_key(data, (detail_key,))
        if _is_mapping(detail):
            count = _find_count(detail, keys)
            if count is not None:
                return count
    return None


def _is_bfcl_name(name: str | None) -> bool:
    return "bfcl" in _normalize_name(name)


def _is_swebench_name(name: str | None) -> bool:
    normalized = _normalize_name(name)
    return "swebench" in normalized.replace("_", "")


def _is_first_class_external_name(name: str | None) -> bool:
    return _is_bfcl_name(name) or _is_swebench_name(name)


def _candidate_mappings(data: Mapping[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(data)]
    for key in keys:
        value = _value_by_normalized_key(data, (key,))
        if _is_mapping(value):
            rows.append(dict(value))
        elif _is_sequence(value):
            rows.extend(dict(row) for row in value if _is_mapping(row))
    return rows


def _canonical_bfcl_category(raw: str) -> str | None:
    normalized = _normalize_name(raw)
    compact = normalized.replace("_", "")
    if "nonlive" in compact:
        return "non_live"
    if "multiturn" in compact:
        return "multi_turn"
    if "hallucination" in normalized or "irrelevance" in normalized:
        return "hallucination"
    if "agentic" in normalized:
        return "agentic"
    if normalized == "live" or normalized.startswith("live_"):
        return "live"
    return None


def _score_from_metric_like(value: Any) -> float | None:
    if _is_mapping(value):
        return _find_score(value)
    return _fraction(value)


def _bfcl_category_scores(data: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    rows = _candidate_mappings(data, _BFCL_CONTAINER_KEYS)
    for row in rows:
        for key, value in row.items():
            category = _canonical_bfcl_category(str(key))
            if category is None:
                continue
            score = _score_from_metric_like(value)
            if score is not None:
                scores[category] = score
    return scores


def parse_bfcl_results(name: str, data: dict) -> BenchmarkResult:
    """BFCL score JSON/CSV exports with V4 category drilldown."""
    normalized_name = _normalize_name(name)
    rows = _candidate_mappings(data, _BFCL_CONTAINER_KEYS)
    score = next((candidate for row in rows if (candidate := _find_score(row)) is not None), None)
    categories: dict[str, float] = {}
    for row in rows:
        categories.update(_bfcl_category_scores(row))

    if score is None and all(category in categories for category in _BFCL_V4_WEIGHTS):
        score = sum(categories[category] * weight for category, weight in _BFCL_V4_WEIGHTS.items())
    elif score is None and categories:
        score = sum(categories.values()) / len(categories)

    passed = _find_count(data, _PASSED_KEYS)
    total = _find_count(data, _TOTAL_KEYS)
    if score is None and passed is not None and total:
        score = passed / total
    if score is None:
        return BenchmarkResult(
            name=normalized_name,
            score=0.0,
            passed=passed or 0,
            total=total or 0,
            error="no BFCL overall accuracy, V4 category scores, or correct/total count found",
        )

    metrics: dict[str, float] = {}
    for row in rows:
        metrics.update(_flatten_numeric_metrics(row))
    for category, category_score in categories.items():
        metrics[f"category.{category}"] = category_score
    if all(category in categories for category in _BFCL_V4_WEIGHTS):
        metrics["bfcl_v4_weighted_score"] = sum(
            categories[category] * weight for category, weight in _BFCL_V4_WEIGHTS.items()
        )
    metrics.setdefault("accuracy", score)

    return BenchmarkResult(
        name=normalized_name,
        score=score,
        passed=passed or (int(round(score * total)) if total else 0),
        total=total or 0,
        metrics=metrics,
    )


def _instance_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in _SWE_INSTANCE_RESULT_KEYS:
        value = _value_by_normalized_key(data, (key,))
        if _is_sequence(value):
            rows = [dict(row) for row in value if _is_mapping(row)]
            if rows:
                return rows
        if _is_mapping(value):
            rows = [dict(row) for row in value.values() if _is_mapping(row)]
            if rows and all(_looks_like_swe_instance(row) for row in rows):
                return rows
    return []


def _looks_like_swe_instance(row: Mapping[str, Any]) -> bool:
    return any(
        _value_by_normalized_key(row, (key,)) is not None
        for key in (
            "instance_id",
            "task_id",
            "repo",
            "resolved",
            "passed",
            "success",
            "status",
        )
    )


def _truthy_status(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = _normalize_name(value)
        if normalized in _SWE_RESOLVED_STATUS:
            return True
        if normalized in _SWE_UNRESOLVED_STATUS:
            return False
    return None


def _swe_instance_resolved(row: Mapping[str, Any]) -> bool | None:
    for key in ("resolved", "passed", "success", "patch_successfully_applied"):
        value = _value_by_normalized_key(row, (key,))
        resolved = _truthy_status(value)
        if resolved is not None:
            return resolved
    for key in ("status", "result", "outcome"):
        value = _value_by_normalized_key(row, (key,))
        resolved = _truthy_status(value)
        if resolved is not None:
            return resolved
    return None


def _swe_instance_completed(row: Mapping[str, Any]) -> bool | None:
    for key in ("completed", "evaluated", "ran"):
        value = _value_by_normalized_key(row, (key,))
        completed = _truthy_status(value)
        if completed is not None:
            return completed
    status = _value_by_normalized_key(row, ("status",))
    if isinstance(status, str):
        normalized = _normalize_name(status)
        if normalized in {"pending", "queued"}:
            return False
        if normalized in _SWE_RESOLVED_STATUS | _SWE_UNRESOLVED_STATUS:
            return True
    return None


def _swe_instance_repo(row: Mapping[str, Any]) -> str | None:
    repo = _value_by_normalized_key(row, ("repo", "repository"))
    if isinstance(repo, str) and repo.strip():
        return _metric_name(repo.replace("/", "_"), fallback="unknown_repo")
    instance_id = _value_by_normalized_key(row, ("instance_id", "task_id", "id"))
    if isinstance(instance_id, str) and "__" in instance_id:
        owner, rest = instance_id.split("__", 1)
        repo_name = rest.split("-", 1)[0]
        return _metric_name(f"{owner}_{repo_name}", fallback="unknown_repo")
    return None


def parse_swebench_results(name: str, data: dict) -> BenchmarkResult:
    """SWE-bench results.json, sb-cli report JSON, or instance_results rows."""
    normalized_name = _normalize_name(name)
    rows = _instance_rows(data)

    resolved = _find_count(data, _SWE_RESOLVED_KEYS)
    submitted = _find_count(data, _SWE_SUBMITTED_KEYS)
    completed = _find_count(data, _SWE_COMPLETED_KEYS)
    total = _find_count(data, _SWE_TOTAL_KEYS)
    errors = _find_count(data, _SWE_ERROR_KEYS)
    pending = _find_count(data, _SWE_PENDING_KEYS)

    repo_counts: dict[str, dict[str, int]] = {}
    if rows:
        resolved_values = [value for row in rows if (value := _swe_instance_resolved(row)) is not None]
        completed_values = [
            value for row in rows if (value := _swe_instance_completed(row)) is not None
        ]
        if resolved is None:
            resolved = sum(1 for value in resolved_values if value)
        if completed is None and completed_values:
            completed = sum(1 for value in completed_values if value)
        if total is None:
            total = len(rows)
        if submitted is None:
            submitted = len(rows)
        for row in rows:
            repo = _swe_instance_repo(row)
            instance_resolved = _swe_instance_resolved(row)
            if repo is None or instance_resolved is None:
                continue
            counts = repo_counts.setdefault(repo, {"resolved": 0, "total": 0})
            counts["total"] += 1
            counts["resolved"] += int(instance_resolved)

    denominator = submitted or total or 0
    score = _find_score(data)
    if score is None and resolved is not None and denominator:
        score = resolved / denominator
    if score is None:
        return BenchmarkResult(
            name=normalized_name,
            score=0.0,
            passed=resolved or 0,
            total=denominator,
            error="no SWE-bench resolution rate or resolved/submitted count found",
        )

    metrics = _flatten_numeric_metrics(data)
    metrics["resolution_rate"] = score
    if resolved is not None:
        metrics["resolved_instances"] = float(resolved)
    if submitted is not None:
        metrics["submitted_instances"] = float(submitted)
    if completed is not None:
        metrics["completed_instances"] = float(completed)
    if total is not None:
        metrics["total_instances"] = float(total)
    if errors is not None:
        metrics["error_count"] = float(errors)
    if pending is not None:
        metrics["pending_evaluations"] = float(pending)
    for repo, counts in repo_counts.items():
        if counts["total"]:
            metrics[f"repo.{repo}.resolution_rate"] = counts["resolved"] / counts["total"]
            metrics[f"repo.{repo}.resolved_instances"] = float(counts["resolved"])
            metrics[f"repo.{repo}.total_instances"] = float(counts["total"])

    return BenchmarkResult(
        name=normalized_name,
        score=score,
        passed=resolved or (int(round(score * denominator)) if denominator else 0),
        total=denominator,
        metrics=metrics,
    )


def _find_trial_score(record: Any) -> float | None:
    if _is_mapping(record):
        for key in ("reward", "score", "passed", "success"):
            if key in record:
                score = _fraction(record[key])
                if score is not None:
                    return score
        for detail_key in _DETAIL_KEYS:
            detail = record.get(detail_key)
            score = _find_trial_score(detail)
            if score is not None:
                return score
    return _fraction(record)


def _aggregate_trial_results(name: str, rows: Sequence[Any]) -> BenchmarkResult:
    scores = [score for row in rows if (score := _find_trial_score(row)) is not None]
    if not scores:
        return BenchmarkResult(
            name=name,
            score=0.0,
            total=len(rows),
            error="no numeric score/reward found in trial results",
        )
    score = sum(scores) / len(scores)
    passed = sum(1 for value in scores if value >= 1.0)
    return BenchmarkResult(
        name=name,
        score=score,
        passed=passed,
        total=len(scores),
        metrics={"mean_reward": score, "trials": float(len(scores))},
    )


def _parse_external_result(name: str, payload: Any) -> BenchmarkResult:
    normalized_name = _normalize_name(name)
    if _is_bfcl_name(normalized_name):
        data = {"data_overall": list(payload)} if _is_sequence(payload) else payload
        if _is_mapping(data):
            return parse_bfcl_results(normalized_name, dict(data))
    if _is_swebench_name(normalized_name):
        data = {"instance_results": list(payload)} if _is_sequence(payload) else payload
        if _is_mapping(data):
            return parse_swebench_results(normalized_name, dict(data))
    if _is_sequence(payload):
        return _aggregate_trial_results(normalized_name, payload)
    if not _is_mapping(payload):
        score = _fraction(payload)
        if score is None:
            return BenchmarkResult(
                name=normalized_name,
                score=0.0,
                error="external benchmark result must be an object, list, or numeric score",
            )
        return BenchmarkResult(name=normalized_name, score=score, metrics={"score": score})

    data = dict(payload)
    for trial_key in ("trials", "episodes", "attempts"):
        rows = data.get(trial_key)
        if _is_sequence(rows):
            aggregate = _aggregate_trial_results(normalized_name, rows)
            if aggregate.error is None:
                aggregate.metrics.update(_numeric_metrics(data))
                return aggregate

    score = _find_score(data)
    passed = _find_count(data, _PASSED_KEYS)
    total = _find_count(data, _TOTAL_KEYS)
    if score is None and passed is not None and total:
        score = passed / total
    if score is None:
        return BenchmarkResult(
            name=normalized_name,
            score=0.0,
            passed=passed or 0,
            total=total or 0,
            error="no score, accuracy, pass rate, reward, or resolved/total count found",
        )

    metrics = _numeric_metrics(data)
    for detail_key in ("metrics", "summary"):
        detail = data.get(detail_key)
        if _is_mapping(detail):
            for key, value in _numeric_metrics(detail).items():
                metrics[f"{detail_key}.{key}"] = value
    return BenchmarkResult(
        name=normalized_name,
        score=score,
        passed=passed or (int(round(score * total)) if total else 0),
        total=total or 0,
        metrics=metrics,
    )


def _collect_external_results(payload: Any, *, fallback_name: str) -> list[BenchmarkResult]:
    if _is_sequence(payload):
        rows = list(payload)
        if all(_is_mapping(row) for row in rows):
            named = [
                _parse_external_result(_name_from_record(row, fallback_name), row)
                for row in rows
                if _name_from_record(row, fallback_name) != fallback_name
            ]
            if named:
                return named
        return [_aggregate_trial_results(_normalize_name(fallback_name), rows)]

    if not _is_mapping(payload):
        return [_parse_external_result(fallback_name, payload)]

    data = dict(payload)
    record_name = _name_from_record(data, fallback_name)
    if record_name != _normalize_name(fallback_name) or any(key in data for key in _SCORE_KEYS):
        return [_parse_external_result(record_name, data)]

    for key in _CONTAINER_KEYS:
        value = data.get(key)
        if value is None:
            continue
        if _is_sequence(value):
            return _collect_external_results(value, fallback_name=fallback_name)
        if _is_mapping(value):
            return [
                _parse_external_result(str(name), result)
                for name, result in value.items()
                if _is_mapping(result) or _is_sequence(result) or _coerce_float(result) is not None
            ]

    if any(key in data for key in ("trials", "episodes", "attempts", *_PASSED_KEYS, *_TOTAL_KEYS)):
        return [_parse_external_result(fallback_name, data)]

    return [
        _parse_external_result(str(name), result)
        for name, result in data.items()
        if _is_mapping(result) or _is_sequence(result) or _coerce_float(result) is not None
    ]


def normalize_external_benchmark_results(
    payload: Any,
    *,
    benchmark_name: str | None = None,
    source: str | None = None,
) -> BenchmarkReport:
    """Normalize pasted external harness JSON into registry-ready benchmark scores.

    Supports aggregate summaries (``score``, ``accuracy``, ``resolved``/``total``),
    named result maps/lists, and Harbor-style trial lists with numeric rewards.
    """
    fallback_name = benchmark_name or source or "external_benchmark"
    if benchmark_name or _is_first_class_external_name(fallback_name):
        results = [_parse_external_result(fallback_name, payload)]
    else:
        results = _collect_external_results(payload, fallback_name=fallback_name)
    if not results:
        raise ValueError("no external benchmark results found")
    return BenchmarkReport(results=results)


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


def harbor_terminal_bench_command(
    model: str,
    *,
    dataset: str = "terminal-bench@2.0",
    agent: str = "terminus",
    n_concurrent: int = 4,
    environment: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Harbor-native Terminal-Bench run.

    Harbor is the current official harness for Terminal-Bench-style container
    evaluations and can also produce rollout artifacts for RL workflows.
    """
    argv = [
        "harbor",
        "run",
        "--dataset",
        dataset,
        "--agent",
        agent,
        "--model",
        model,
        "--n-concurrent",
        str(n_concurrent),
    ]
    if environment:
        argv.extend(["--env", environment])
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
