"""Paired-bootstrap comparison gates for executable environment holdouts."""

from __future__ import annotations

from typing import Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.environment_holdout import (
    environment_group_key,
    environment_holdout_contamination,
    make_environment_holdout_split,
)
from bashgym.eval.environment_passk import (
    EnvironmentAttempt,
    EnvironmentPassKReport,
    evaluate_environment_attempts,
)
from bashgym.eval.stats import paired_bootstrap

ENVIRONMENT_HOLDOUT_COMPARISON_SCHEMA_VERSION = "bashgym.environment_holdout_comparison.v1"


def _attempt_environment_id(attempt: EnvironmentAttempt | dict[str, Any]) -> str:
    if isinstance(attempt, EnvironmentAttempt):
        return attempt.environment_id
    return str(attempt.get("environment_id") or "")


def _holdout_attempts(
    attempts: list[EnvironmentAttempt | dict[str, Any]],
    holdout_ids: set[str],
) -> list[EnvironmentAttempt]:
    return [
        (
            attempt
            if isinstance(attempt, EnvironmentAttempt)
            else EnvironmentAttempt.from_dict(attempt)
        )
        for attempt in attempts
        if _attempt_environment_id(attempt) in holdout_ids
    ]


def _pass_metric(report: EnvironmentPassKReport, env_id: str, metric: str) -> float:
    row = report.to_dict()["per_environment"].get(env_id) or {}
    return float(row.get(metric, 0.0))


def _attempt_summary_value(report: EnvironmentPassKReport, key: str) -> float:
    value = report.to_dict()["attempt_summary"].get(key)
    return float(value or 0.0)


def _tamper_rate(report: EnvironmentPassKReport) -> float:
    data = report.to_dict()
    status_dist = data["attempt_summary"]["verifier_status_distribution"]
    tamper_count = int(status_dist.get("tampered", 0))
    return tamper_count / report.n_attempts if report.n_attempts else 0.0


def evaluate_environment_holdout_comparison_gate(
    environments: list[EnvironmentSpec | dict[str, Any]],
    base_attempts: list[EnvironmentAttempt | dict[str, Any]],
    candidate_attempts: list[EnvironmentAttempt | dict[str, Any]],
    *,
    split_by: str = "task_family",
    cluster_by: str = "task_family",
    holdout_fraction: float = 0.2,
    seed: int = 0,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    compare_k: int = 1,
    min_delta: float = 0.0,
    min_candidate_pass_at_1: float = 0.0,
    require_ci_excludes_zero: bool = True,
    max_candidate_timeout_rate: float = 0.25,
    max_candidate_tamper_rate: float = 0.0,
    require_no_contamination: bool = True,
    n_resamples: int = 1000,
) -> dict[str, Any]:
    """Compare base vs candidate holdout performance with clustered bootstrap."""

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if compare_k <= 0:
        raise ValueError("compare_k must be positive")
    if min_delta < -1 or min_delta > 1:
        raise ValueError("min_delta must be between -1 and 1")
    if min_candidate_pass_at_1 < 0 or min_candidate_pass_at_1 > 1:
        raise ValueError("min_candidate_pass_at_1 must be between 0 and 1")
    if max_candidate_timeout_rate < 0 or max_candidate_timeout_rate > 1:
        raise ValueError("max_candidate_timeout_rate must be between 0 and 1")
    if max_candidate_tamper_rate < 0 or max_candidate_tamper_rate > 1:
        raise ValueError("max_candidate_tamper_rate must be between 0 and 1")

    ks = sorted({int(k) for k in k_values})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers")
    if compare_k not in ks:
        ks = sorted({*ks, compare_k})

    split = make_environment_holdout_split(
        environments,
        by=split_by,
        fraction=holdout_fraction,
        seed=seed,
    )
    holdout_ids = {env.id for env in split.holdout}
    base_report = evaluate_environment_attempts(
        split.holdout,
        _holdout_attempts(base_attempts, holdout_ids),
        k_values=ks,
    )
    candidate_report = evaluate_environment_attempts(
        split.holdout,
        _holdout_attempts(candidate_attempts, holdout_ids),
        k_values=ks,
    )

    metric = f"pass@{compare_k}"
    deltas: list[float] = []
    clusters: list[str] = []
    per_environment: dict[str, dict[str, Any]] = {}
    for env in split.holdout:
        base_score = _pass_metric(base_report, env.id, metric)
        candidate_score = _pass_metric(candidate_report, env.id, metric)
        delta = candidate_score - base_score
        cluster = environment_group_key(env, cluster_by)
        deltas.append(delta)
        clusters.append(cluster)
        per_environment[env.id] = {
            "cluster": cluster,
            "base": base_score,
            "candidate": candidate_score,
            "delta": delta,
        }

    bootstrap = paired_bootstrap(deltas, clusters, n_resamples=n_resamples, seed=seed)
    contamination = environment_holdout_contamination(split)
    candidate_pass_at_1 = float(candidate_report.pass_at_k.get("pass@1", 0.0))
    timeout_rate = _attempt_summary_value(candidate_report, "timeout_rate")
    tamper_rate = _tamper_rate(candidate_report)

    reasons: list[str] = []
    if bootstrap.mean < min_delta:
        reasons.append(f"{metric} delta {bootstrap.mean:.3f} < required {min_delta:.3f}")
    if require_ci_excludes_zero and not bootstrap.better:
        reasons.append(
            f"{metric} CI [{bootstrap.ci_low:.3f}, {bootstrap.ci_high:.3f}] does not clear zero"
        )
    if candidate_pass_at_1 < min_candidate_pass_at_1:
        reasons.append(
            f"candidate pass@1 {candidate_pass_at_1:.3f} < required {min_candidate_pass_at_1:.3f}"
        )
    if timeout_rate > max_candidate_timeout_rate:
        reasons.append(
            f"candidate timeout rate {timeout_rate:.3f} > allowed {max_candidate_timeout_rate:.3f}"
        )
    if tamper_rate > max_candidate_tamper_rate:
        reasons.append(
            f"candidate tamper rate {tamper_rate:.3f} > allowed {max_candidate_tamper_rate:.3f}"
        )
    if require_no_contamination and contamination:
        reasons.append(f"holdout contamination detected: {len(contamination)} hashes")

    return {
        "schema_version": ENVIRONMENT_HOLDOUT_COMPARISON_SCHEMA_VERSION,
        "split": split.manifest(),
        "cluster_by": cluster_by,
        "compare_metric": metric,
        "contamination": contamination,
        "base_report": base_report.to_dict(),
        "candidate_report": candidate_report.to_dict(),
        "per_environment": per_environment,
        "bootstrap": {
            "mean": bootstrap.mean,
            "ci_low": bootstrap.ci_low,
            "ci_high": bootstrap.ci_high,
            "significant": bootstrap.significant,
            "better": bootstrap.better,
            "n": bootstrap.n,
            "n_clusters": bootstrap.n_clusters,
        },
        "gate": {
            "ship": not reasons,
            "reasons": reasons,
            "thresholds": {
                "min_delta": min_delta,
                "min_candidate_pass_at_1": min_candidate_pass_at_1,
                "require_ci_excludes_zero": require_ci_excludes_zero,
                "max_candidate_timeout_rate": max_candidate_timeout_rate,
                "max_candidate_tamper_rate": max_candidate_tamper_rate,
                "require_no_contamination": require_no_contamination,
            },
            "observed": {
                "delta": bootstrap.mean,
                "ci_low": bootstrap.ci_low,
                "ci_high": bootstrap.ci_high,
                "candidate_pass_at_1": candidate_pass_at_1,
                "candidate_timeout_rate": timeout_rate,
                "candidate_tamper_rate": tamper_rate,
            },
        },
    }
