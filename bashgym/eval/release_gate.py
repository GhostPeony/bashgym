"""Combine trace, forgetting, and environment gates into one release verdict."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

RELEASE_GATE_SCHEMA_VERSION = "bashgym.release_gate.v1"

ENVIRONMENT_GATE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("holdout_gate", "environment holdout"),
    ("holdout_comparison", "environment holdout comparison"),
    ("spurious_reward_control", "environment spurious reward control"),
)
EXTERNAL_BENCHMARK_SECTION = "external_benchmarks"
WORLD_MODEL_QUALITY_SECTION = "world_model_quality"
LEARNED_REWARD_SECTION = "learned_reward_evidence"

WORLD_MODEL_QUALITY_ALIASES: dict[str, tuple[str, ...]] = {
    "echo_loss": ("echo_loss", "echoLoss", "environment_prediction_loss"),
    "echo_loss_delta": ("echo_loss_delta", "echoLossDelta"),
    "rwml_pass_rate": ("rwml_pass_rate", "rwmlPassRate", "world_model_pass_rate"),
    "embedding_distance_mean": (
        "embedding_distance_mean",
        "embeddingDistanceMean",
        "rwml_embedding_distance_mean",
    ),
    "embedding_distance_p95": (
        "embedding_distance_p95",
        "embeddingDistanceP95",
        "rwml_embedding_distance_p95",
    ),
    "exit_code_accuracy": ("exit_code_accuracy", "exitCodeAccuracy"),
    "test_result_accuracy": ("test_result_accuracy", "testResultAccuracy"),
}


def _as_reason_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item)
        if text:
            out.append(text)
    return out


def _unwrap_gate_payload(payload: Any) -> dict[str, Any] | None:
    """Accept raw gate results or full API responses containing ``result``."""

    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return payload


def _environment_gate_reasons(
    environment_evidence: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    reasons: list[str] = []
    provided_sections: list[str] = []
    blocking_sections: list[str] = []

    for section, label in ENVIRONMENT_GATE_SECTIONS:
        payload = _unwrap_gate_payload(environment_evidence.get(section))
        if payload is None:
            continue

        provided_sections.append(section)
        gate = payload.get("gate") if isinstance(payload, dict) else None
        if not isinstance(gate, dict) or not isinstance(gate.get("ship"), bool):
            reasons.append(f"{label}: missing gate ship verdict")
            blocking_sections.append(section)
            continue

        if gate["ship"]:
            continue

        blocking_sections.append(section)
        gate_reasons = _as_reason_list(gate.get("reasons"))
        if gate_reasons:
            reasons.extend(f"{label}: {reason}" for reason in gate_reasons)
        else:
            reasons.append(f"{label}: gate did not ship")

    return reasons, provided_sections, blocking_sections


def _unwrap_external_benchmark_payload(payload: Any) -> dict[str, Any] | None:
    """Accept raw BenchmarkReport dicts or full ingest responses containing report."""

    if not isinstance(payload, dict):
        return None
    report = payload.get("report")
    if isinstance(report, dict):
        return report
    if "scores" in payload or "results" in payload or "failures" in payload:
        return payload
    return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _metric_last(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("last", "value", "mean"):
            number = _float(value.get(key))
            if number is not None:
                return number
        return None
    if isinstance(value, list) and value:
        return _metric_last(value[-1])
    return _float(value)


def _metric_delta(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("delta", "change"):
            number = _float(value.get(key))
            if number is not None:
                return number
        first = _metric_last(value.get("first"))
        last = _metric_last(value.get("last"))
        if first is not None and last is not None:
            return last - first
    if isinstance(value, list) and len(value) >= 2:
        first = _metric_last(value[0])
        last = _metric_last(value[-1])
        if first is not None and last is not None:
            return last - first
    return None


def _unwrap_world_model_quality_payload(payload: Any) -> dict[str, Any] | None:
    """Accept raw world-model quality dicts or run-analysis/release wrappers."""

    if not isinstance(payload, dict):
        return None
    for key in ("result", "report", "world_model_quality"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def _world_model_quality_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_world_model_quality_payload(evidence.get(WORLD_MODEL_QUALITY_SECTION))
    if payload is None:
        return {
            "present": False,
            "diagnostic_only": True,
            "signal": "missing",
            "metrics": {},
            "findings": [],
            "coverage": None,
        }

    metric_sources: list[dict[str, Any]] = []
    for key in ("metrics", "training_metrics"):
        source = payload.get(key)
        if isinstance(source, dict):
            metric_sources.append(source)
    metric_sources.append(payload)

    metrics: dict[str, float] = {}
    for canonical, aliases in WORLD_MODEL_QUALITY_ALIASES.items():
        for source in metric_sources:
            for alias in aliases:
                if alias not in source:
                    continue
                number = (
                    _metric_delta(source[alias])
                    if canonical.endswith("_delta")
                    else _metric_last(source[alias])
                )
                if number is not None:
                    metrics[canonical] = number
                    break
            if canonical in metrics:
                break

    if "echo_loss_delta" not in metrics:
        for source in metric_sources:
            echo_value = source.get("echo_loss") or source.get("echoLoss")
            delta = _metric_delta(echo_value)
            if delta is not None:
                metrics["echo_loss_delta"] = delta
                break

    coverage = payload.get("coverage")
    if coverage is None and isinstance(payload.get("replay_summary"), dict):
        replay_summary = payload["replay_summary"]
        coverage = replay_summary.get("world_model") or {
            "world_model_records": replay_summary.get("world_model_records")
        }

    findings: list[str] = []
    if not metrics:
        findings.append("world model quality evidence has no recognized ECHO/RWML metrics")

    echo_delta = metrics.get("echo_loss_delta")
    if echo_delta is not None and echo_delta > 0:
        findings.append("ECHO loss increased across the observed window")

    rwml_pass_rate = metrics.get("rwml_pass_rate")
    if rwml_pass_rate is not None and rwml_pass_rate < 0.5:
        findings.append("RWML pass rate is below the suggested smoke threshold")

    distance_mean = metrics.get("embedding_distance_mean")
    if distance_mean is not None and distance_mean > 0.2:
        findings.append("mean RWML embedding distance is above the starter threshold")

    signal = "present"
    if not metrics:
        signal = "missing_quality_metrics"
    elif findings:
        signal = "needs_attention"
    elif echo_delta is not None and echo_delta < 0:
        signal = "improving"

    return {
        "present": True,
        "diagnostic_only": True,
        "signal": signal,
        "metrics": metrics,
        "findings": findings,
        "coverage": coverage if isinstance(coverage, dict) else None,
    }


def _unwrap_learned_reward_payload(payload: Any) -> dict[str, Any] | None:
    """Accept raw reward_eval.json or fixture/report wrappers containing it."""

    if not isinstance(payload, dict):
        return None
    for key in ("reward_eval", "result", "report", "learned_reward_evidence"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def _learned_reward_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_learned_reward_payload(evidence.get(LEARNED_REWARD_SECTION))
    if payload is None:
        return {
            "present": False,
            "diagnostic_only": True,
            "signal": "missing",
            "metrics": {},
            "findings": [],
        }

    raw_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metric_names = (
        "heldout_pair_accuracy",
        "calibration_error",
        "reward_margin",
        "length_bias",
        "reward_variance",
        "eval_only_leakage_count",
        "eval_only_leakage_rate",
        "pair_count",
        "prediction_records",
        "evaluated_records",
    )
    metrics: dict[str, float] = {}
    for name in metric_names:
        number = _float(raw_metrics.get(name))
        if number is None:
            number = _float(payload.get(name))
        if number is not None:
            metrics[name] = number

    normalized_findings: list[str] = []
    if isinstance(payload.get("findings"), list):
        for item in payload["findings"]:
            if isinstance(item, dict):
                code = item.get("code")
                message = item.get("message")
                if message:
                    normalized_findings.append(
                        f"learned reward {code or 'finding'}: {message}"
                    )
                continue
            text = str(item)
            if text:
                normalized_findings.append(f"learned reward: {text}")

    if not metrics:
        normalized_findings.append("learned reward evidence has no recognized reward metrics")
    if bool(payload.get("ok")) is False:
        normalized_findings.append("learned reward evidence reports ok=false")
    leakage_count = metrics.get("eval_only_leakage_count")
    if leakage_count is not None and leakage_count > 0:
        normalized_findings.append("learned reward evidence reports eval-only leakage")
    calibration = metrics.get("calibration_error")
    if calibration is not None and calibration > 0.2:
        normalized_findings.append("learned reward calibration error is above the starter threshold")
    pair_accuracy = metrics.get("heldout_pair_accuracy")
    if pair_accuracy is not None and pair_accuracy < 0.6:
        normalized_findings.append("learned reward heldout pair accuracy is below the starter threshold")
    reward_variance = metrics.get("reward_variance")
    if reward_variance is not None and reward_variance == 0.0:
        normalized_findings.append("learned reward predictions have zero variance")

    signal = "present"
    if not metrics:
        signal = "missing_quality_metrics"
    elif normalized_findings:
        signal = "needs_attention"
    elif pair_accuracy is not None and pair_accuracy >= 0.7:
        signal = "healthy"

    return {
        "present": True,
        "diagnostic_only": True,
        "signal": signal,
        "metrics": metrics,
        "findings": normalized_findings,
    }


def _external_benchmark_reasons(
    evidence: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    reasons: list[str] = []
    provided_sections: list[str] = []
    blocking_sections: list[str] = []

    payload = evidence.get(EXTERNAL_BENCHMARK_SECTION)
    report = _unwrap_external_benchmark_payload(payload)
    required = bool(evidence.get("external_benchmarks_required"))
    if report is None:
        if required:
            reasons.append("external benchmark evidence required but missing")
            blocking_sections.append(EXTERNAL_BENCHMARK_SECTION)
        return reasons, provided_sections, blocking_sections

    provided_sections.append(EXTERNAL_BENCHMARK_SECTION)
    failures = _as_reason_list(report.get("failures"))
    if failures:
        blocking_sections.append(EXTERNAL_BENCHMARK_SECTION)
        reasons.extend(f"external benchmark {name}: failed or missing score" for name in failures)

    scores = report.get("scores") if isinstance(report.get("scores"), dict) else {}
    thresholds = {}
    if isinstance(payload, dict) and isinstance(payload.get("min_scores"), dict):
        thresholds.update(payload["min_scores"])
    if isinstance(evidence.get("external_benchmark_min_scores"), dict):
        thresholds.update(evidence["external_benchmark_min_scores"])

    for name, raw_threshold in thresholds.items():
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            reasons.append(f"external benchmark {name}: invalid minimum score")
            if EXTERNAL_BENCHMARK_SECTION not in blocking_sections:
                blocking_sections.append(EXTERNAL_BENCHMARK_SECTION)
            continue
        try:
            score = float(scores[name])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"external benchmark {name}: score required but missing")
            if EXTERNAL_BENCHMARK_SECTION not in blocking_sections:
                blocking_sections.append(EXTERNAL_BENCHMARK_SECTION)
            continue
        if score < threshold:
            reasons.append(
                f"external benchmark {name}: score {score:.3f} < required {threshold:.3f}"
            )
            if EXTERNAL_BENCHMARK_SECTION not in blocking_sections:
                blocking_sections.append(EXTERNAL_BENCHMARK_SECTION)

    return reasons, provided_sections, blocking_sections


def combine_release_gate_evidence(
    heldout_report: dict[str, Any],
    environment_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fold optional environment gate evidence into a held-out trace report.

    ``heldout_report`` is a ``HeldoutReport.to_dict()``-style payload. Environment
    evidence may contain pass@k metrics plus any precomputed gate results from the
    environment endpoints. Pass@k is carried as supporting evidence; only the gate
    sections block release.
    """

    report = deepcopy(heldout_report)
    trace_ship = bool(report.get("ship"))
    trace_reasons = _as_reason_list(report.get("reasons"))

    evidence = deepcopy(environment_evidence or {})
    evidence_required = bool(evidence.get("required"))
    environment_reasons, provided_sections, blocking_sections = _environment_gate_reasons(evidence)
    if evidence_required and not provided_sections:
        environment_reasons.append("environment gate evidence required but missing")
    (
        external_reasons,
        external_sections,
        blocking_external_sections,
    ) = _external_benchmark_reasons(evidence)
    world_model_quality = _world_model_quality_summary(evidence)
    world_model_sections = [WORLD_MODEL_QUALITY_SECTION] if world_model_quality["present"] else []
    learned_reward = _learned_reward_summary(evidence)
    learned_reward_sections = [LEARNED_REWARD_SECTION] if learned_reward["present"] else []

    environment_ship = not environment_reasons
    external_benchmark_ship = not external_reasons
    combined_reasons = [*trace_reasons, *environment_reasons, *external_reasons]
    ship = trace_ship and environment_ship and external_benchmark_ship

    report["ship"] = ship
    report["reasons"] = combined_reasons
    report["release_gate"] = {
        "schema_version": RELEASE_GATE_SCHEMA_VERSION,
        "ship": ship,
        "trace_ship": trace_ship,
        "environment_ship": environment_ship,
        "external_benchmark_ship": external_benchmark_ship,
        "trace_reasons": trace_reasons,
        "environment_reasons": environment_reasons,
        "external_benchmark_reasons": external_reasons,
        "world_model_quality": world_model_quality,
        "world_model_quality_present": world_model_quality["present"],
        "world_model_quality_diagnostic_only": world_model_quality["diagnostic_only"],
        "world_model_quality_signal": world_model_quality["signal"],
        "world_model_quality_findings": world_model_quality["findings"],
        "learned_reward_evidence": learned_reward,
        "learned_reward_evidence_present": learned_reward["present"],
        "learned_reward_evidence_diagnostic_only": learned_reward["diagnostic_only"],
        "learned_reward_evidence_signal": learned_reward["signal"],
        "learned_reward_evidence_findings": learned_reward["findings"],
        "environment_required": evidence_required,
        "environment_sections": provided_sections,
        "blocking_environment_sections": blocking_sections,
        "external_benchmark_sections": external_sections,
        "blocking_external_benchmark_sections": blocking_external_sections,
        "world_model_quality_sections": world_model_sections,
        "learned_reward_evidence_sections": learned_reward_sections,
    }
    if environment_evidence is not None:
        report["environment_evidence"] = evidence
    return report
