"""Agent-readable training run analysis.

This module turns persisted run metrics, optional DPPO replay, and optional
release evidence into a compact diagnostic payload. It is intentionally
conservative: loss/reward curves can suggest next steps, but held-out and
environment gates decide whether a model is release-ready.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bashgym.eval.dppo_replay import read_dppo_replay_jsonl, summarize_dppo_replay_records
from bashgym.gym.run_metrics import read_run_metrics

RUN_ANALYSIS_SCHEMA_VERSION = "bashgym.run_analysis.v1"


def read_jsonl_metrics(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL metrics file, skipping blank or malformed lines."""

    metrics: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                metrics.append(item)
    return metrics


def _number(point: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = point.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _series(metrics: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for point in metrics:
        value = _number(point, *keys)
        if value is not None:
            values.append(value)
    return values


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "first": None, "last": None, "min": None, "max": None, "delta": None}
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "delta": values[-1] - values[0],
    }


def summarize_training_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute coarse summaries from SFT/DPO/GRPO-style metric JSONL."""

    steps = _series(metrics, "step", "global_step")
    loss = _summary(_series(metrics, "loss", "train_loss", "eval_loss"))
    reward = _summary(_series(metrics, "reward", "train_reward_mean", "valid_reward_mean"))
    reward_std = _summary(_series(metrics, "reward_std", "rewardStd"))
    frac_reward_zero_std = _summary(
        _series(metrics, "frac_reward_zero_std", "fracRewardZeroStd")
    )
    pass_at_1 = _summary(
        _series(metrics, "pass@1", "pass_at_1", "heldout_pass@1", "heldout_pass_at_1")
    )
    pass_at_k = _summary(
        _series(metrics, "pass@k", "pass_at_k", "heldout_pass@k", "heldout_pass_at_k")
    )
    timeout_rate = _summary(_series(metrics, "timeout_rate", "timeoutRate"))
    tamper_rate = _summary(_series(metrics, "tamper_rate", "tamperRate"))
    echo_loss = _summary(_series(metrics, "echo_loss", "echoLoss"))
    rwml_pass_rate = _summary(_series(metrics, "rwml_pass_rate", "rwmlPassRate"))
    embedding_distance_mean = _summary(
        _series(metrics, "embedding_distance_mean", "rwml_embedding_distance_mean")
    )
    embedding_distance_p95 = _summary(
        _series(metrics, "embedding_distance_p95", "rwml_embedding_distance_p95")
    )
    exit_code_accuracy = _summary(_series(metrics, "exit_code_accuracy", "exitCodeAccuracy"))
    test_result_accuracy = _summary(
        _series(metrics, "test_result_accuracy", "testResultAccuracy")
    )
    grad_norm = _summary(_series(metrics, "grad_norm", "gradNorm"))
    learning_rate = _summary(_series(metrics, "learning_rate", "learningRate"))

    return {
        "points": len(metrics),
        "first_step": int(steps[0]) if steps else None,
        "last_step": int(steps[-1]) if steps else None,
        "loss": loss,
        "reward": reward,
        "reward_std": reward_std,
        "frac_reward_zero_std": frac_reward_zero_std,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "timeout_rate": timeout_rate,
        "tamper_rate": tamper_rate,
        "echo_loss": echo_loss,
        "rwml_pass_rate": rwml_pass_rate,
        "embedding_distance_mean": embedding_distance_mean,
        "embedding_distance_p95": embedding_distance_p95,
        "exit_code_accuracy": exit_code_accuracy,
        "test_result_accuracy": test_result_accuracy,
        "grad_norm": grad_norm,
        "learning_rate": learning_rate,
    }


def _finding(
    severity: str,
    code: str,
    message: str,
    *,
    evidence: dict[str, Any] | None = None,
    next_step: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "evidence": evidence or {},
        "next": next_step,
    }


def _load_release_evidence(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("release evidence must be a JSON object")
    return payload


def _release_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "present": False,
            "ship": None,
            "reasons": [],
            "release_gate": None,
            "world_model_quality": None,
        }
    release_gate = payload.get("release_gate")
    gate = release_gate if isinstance(release_gate, dict) else {}
    ship = payload.get("ship", gate.get("ship"))
    reasons = payload.get("reasons", gate.get("trace_reasons", []))
    if not isinstance(reasons, list):
        reasons = []
    quality = gate.get("world_model_quality")
    if not isinstance(quality, dict):
        quality = payload.get("world_model_quality")
    if not isinstance(quality, dict):
        quality = None
    return {
        "present": True,
        "ship": bool(ship) if isinstance(ship, bool) else None,
        "reasons": [str(reason) for reason in reasons],
        "release_gate": release_gate if isinstance(release_gate, dict) else None,
        "world_model_quality": quality,
    }


def _verdict_level(findings: list[dict[str, Any]]) -> str:
    severities = {finding["severity"] for finding in findings}
    if "blocker" in severities:
        return "blocked"
    if "warning" in severities:
        return "needs_attention"
    if "missing" in severities:
        return "insufficient_evidence"
    return "ready_for_eval"


def build_training_analysis(
    *,
    metrics: list[dict[str, Any]],
    run_id: str | None = None,
    metrics_path: Path | str | None = None,
    replay_path: Path | str | None = None,
    release_evidence_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build an agent-readable analysis payload."""

    metric_summary = summarize_training_metrics(metrics)
    replay_summary = None
    if replay_path is not None:
        replay_records = read_dppo_replay_jsonl(str(replay_path))
        replay_summary = summarize_dppo_replay_records(replay_records)

    release_payload = _load_release_evidence(release_evidence_path)
    release_summary = _release_summary(release_payload)
    findings: list[dict[str, Any]] = []

    if metric_summary["points"] == 0:
        findings.append(
            _finding(
                "missing",
                "no_metrics",
                "No training metrics were found.",
                next_step="Start a run or pass --metrics PATH to a metrics.jsonl artifact.",
            )
        )
    else:
        loss = metric_summary["loss"]
        if loss["count"] >= 2 and loss["delta"] is not None and loss["delta"] > 0:
            findings.append(
                _finding(
                    "warning",
                    "loss_increased",
                    "Training loss increased across the observed metric window.",
                    evidence={"first": loss["first"], "last": loss["last"], "delta": loss["delta"]},
                    next_step="Check learning rate, truncation, data formatting, and whether eval loss behaves differently.",
                )
            )

        reward_std_last = metric_summary["reward_std"]["last"]
        if reward_std_last is not None and reward_std_last <= 0.0:
            findings.append(
                _finding(
                    "warning",
                    "zero_reward_variance",
                    "Latest GRPO reward_std is zero, so policy-gradient learning has little contrast.",
                    evidence={"reward_std": reward_std_last},
                    next_step="Use active sampling, rebalance task difficulty, or return to SFT/curriculum until attempts vary.",
                )
            )

        zero_std_last = metric_summary["frac_reward_zero_std"]["last"]
        if zero_std_last is not None and zero_std_last >= 0.5:
            findings.append(
                _finding(
                    "warning",
                    "many_zero_std_groups",
                    "At least half of recent GRPO groups have zero reward standard deviation.",
                    evidence={"frac_reward_zero_std": zero_std_last},
                    next_step="Filter zero-std groups for policy updates and preserve them only for world-model/curriculum data.",
                )
            )

        tamper_last = metric_summary["tamper_rate"]["last"]
        if tamper_last is not None and tamper_last > 0:
            findings.append(
                _finding(
                    "blocker",
                    "tamper_detected",
                    "Tamper rate is non-zero.",
                    evidence={"tamper_rate": tamper_last},
                    next_step="Fix verifier/protected-file guardrails before treating rewards or pass@k as trustworthy.",
                )
            )

        timeout_last = metric_summary["timeout_rate"]["last"]
        if timeout_last is not None and timeout_last > 0.25:
            findings.append(
                _finding(
                    "warning",
                    "high_timeout_rate",
                    "Timeout rate is above the default environment-gate tolerance.",
                    evidence={"timeout_rate": timeout_last},
                    next_step="Reduce max tool calls, improve prompt budgets, or split long tasks before more RL.",
                )
            )

    pass1 = metric_summary["pass_at_1"]["last"]
    passk = metric_summary["pass_at_k"]["last"]
    if pass1 is None and passk is None and not release_summary["present"]:
        findings.append(
            _finding(
                "missing",
                "missing_heldout_evidence",
                "No held-out pass@k or release evidence was provided.",
                next_step="Run held-out trace and executable environment gates before deciding whether the model improved.",
            )
        )

    if release_summary["present"] and release_summary["ship"] is False:
        findings.append(
            _finding(
                "blocker",
                "release_gate_blocked",
                "Release evidence says this model should not ship.",
                evidence={"reasons": release_summary["reasons"]},
                next_step="Address each release-gate reason, then rerun the held-out/environment evidence.",
            )
        )

    release_world_model_quality = release_summary.get("world_model_quality")
    if isinstance(release_world_model_quality, dict):
        quality_findings = release_world_model_quality.get("findings")
        if (
            release_world_model_quality.get("present") is True
            and release_world_model_quality.get("signal") == "needs_attention"
            and isinstance(quality_findings, list)
        ):
            findings.append(
                _finding(
                    "warning",
                    "world_model_quality_needs_attention",
                    "Release evidence includes ECHO/RWML quality warnings.",
                    evidence={"findings": [str(item) for item in quality_findings]},
                    next_step="Use prediction-error outliers for curriculum mining and compare quality trends against held-out pass@k.",
                )
            )

    world_model_quality_metric_count = sum(
        metric_summary[key]["count"]
        for key in (
            "echo_loss",
            "rwml_pass_rate",
            "embedding_distance_mean",
            "embedding_distance_p95",
            "exit_code_accuracy",
            "test_result_accuracy",
        )
    )
    release_has_world_model_quality = bool(
        isinstance(release_world_model_quality, dict)
        and release_world_model_quality.get("present") is True
    )

    if replay_summary is not None:
        world_model = replay_summary.get("world_model") or {}
        if replay_summary.get("records", 0) == 0:
            findings.append(
                _finding(
                    "missing",
                    "empty_replay",
                    "DPPO replay contains no records.",
                    next_step="Export served-model rollouts before trying train-logprob replay or DPPO smoke launch.",
                )
            )
        elif replay_summary.get("world_model_records", 0) == 0:
            findings.append(
                _finding(
                    "missing",
                    "missing_world_model_replay",
                    "Replay has no world_model payloads.",
                    next_step="Export replay with include_world_model_replay=true before ECHO/RWML training.",
                )
            )
        elif (
            world_model.get("records", 0) > 0
            and world_model_quality_metric_count == 0
            and not release_has_world_model_quality
        ):
            findings.append(
                _finding(
                    "missing",
                    "world_model_quality_missing",
                    "Replay has world-model coverage but no ECHO/RWML quality metrics were found.",
                    evidence={
                        "world_model_records": replay_summary.get("world_model_records", 0),
                        "rwml_transitions": world_model.get("rwml_transitions", 0),
                    },
                    next_step="Run an installed-backend smoke that logs ECHO loss or RWML pass rate before dashboarding release quality.",
                )
            )

    if not findings:
        findings.append(
            _finding(
                "info",
                "no_immediate_issues",
                "No obvious metric or evidence blockers were detected.",
                next_step="Compare against a base model on held-out pass@k and environment release gates.",
            )
        )

    docs = [
        {"topic": "metrics", "path": "docs/training/metrics-runbook.md"},
        {"topic": "strategy", "path": "docs/training/strategy-guide.md"},
    ]
    if replay_summary is not None:
        docs.append({"topic": "world-models", "path": "docs/training/world-models.md"})

    level = _verdict_level(findings)
    return {
        "schema_version": RUN_ANALYSIS_SCHEMA_VERSION,
        "ok": True,
        "run_id": run_id,
        "inputs": {
            "metrics_path": str(Path(metrics_path).resolve()) if metrics_path is not None else None,
            "replay_path": str(Path(replay_path).resolve()) if replay_path is not None else None,
            "release_evidence_path": (
                str(Path(release_evidence_path).resolve())
                if release_evidence_path is not None
                else None
            ),
        },
        "training_metrics": metric_summary,
        "replay_summary": replay_summary,
        "release_evidence": release_summary,
        "verdict": {
            "level": level,
            "summary": {
                "blocked": level == "blocked",
                "has_heldout_signal": pass1 is not None or passk is not None or release_summary["present"],
                "has_world_model_coverage": bool(
                    replay_summary and replay_summary.get("world_model_records", 0) > 0
                ),
                "has_world_model_quality": world_model_quality_metric_count > 0
                or release_has_world_model_quality,
            },
        },
        "findings": findings,
        "docs": docs,
        "next": [
            {
                "reason": finding["message"],
                "action": finding["next"],
            }
            for finding in findings[:5]
        ],
    }


def analyze_run_artifacts(
    *,
    run_id: str | None = None,
    models_dir: Path | None = None,
    metrics_path: Path | str | None = None,
    replay_path: Path | str | None = None,
    release_evidence_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load requested artifacts and return ``build_training_analysis`` output."""

    if metrics_path is not None:
        metrics = read_jsonl_metrics(metrics_path)
        resolved_metrics_path = metrics_path
    elif run_id is not None:
        metrics = read_run_metrics(run_id, models_dir=models_dir) or []
        base = models_dir or Path("data/models")
        resolved_metrics_path = base / run_id / "metrics.jsonl"
    else:
        raise ValueError("Either run_id or metrics_path is required")

    return build_training_analysis(
        metrics=metrics,
        run_id=run_id,
        metrics_path=resolved_metrics_path,
        replay_path=replay_path,
        release_evidence_path=release_evidence_path,
    )
