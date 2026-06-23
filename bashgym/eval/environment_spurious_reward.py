"""Spurious-reward negative controls for executable environment evals."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.environment_holdout import (
    environment_holdout_contamination,
    make_environment_holdout_split,
)
from bashgym.eval.environment_passk import (
    EnvironmentAttempt,
    EnvironmentPassKReport,
    evaluate_environment_attempts,
)

ENVIRONMENT_SPURIOUS_REWARD_SCHEMA_VERSION = "bashgym.environment_spurious_reward_control.v1"


def _attempt_environment_id(attempt: EnvironmentAttempt | dict[str, Any]) -> str:
    if isinstance(attempt, EnvironmentAttempt):
        return attempt.environment_id
    return str(attempt.get("environment_id") or "")


def _normalize_attempts(
    attempts: list[EnvironmentAttempt | dict[str, Any]],
) -> list[EnvironmentAttempt]:
    return [
        attempt if isinstance(attempt, EnvironmentAttempt) else EnvironmentAttempt.from_dict(attempt)
        for attempt in attempts
    ]


def _holdout_attempts(
    attempts: list[EnvironmentAttempt | dict[str, Any]],
    holdout_ids: set[str],
) -> list[EnvironmentAttempt]:
    return [
        attempt
        for attempt in _normalize_attempts(attempts)
        if _attempt_environment_id(attempt) in holdout_ids
    ]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p05": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _summarize_pass_at_k(rows: list[dict[str, float]], k_values: list[int]) -> dict[str, dict[str, float]]:
    return {
        f"pass@{k}": _summary([float(row.get(f"pass@{k}", 0.0)) for row in rows])
        for k in k_values
    }


def _spurious_attempts(
    attempts: list[EnvironmentAttempt],
    *,
    rng: random.Random,
    pass_probability: float,
    trial_index: int,
) -> list[EnvironmentAttempt]:
    out: list[EnvironmentAttempt] = []
    for attempt in attempts:
        passed = rng.random() < pass_probability
        metadata = dict(attempt.metadata)
        metadata.update(
            {
                "control": "spurious_random_reward",
                "control_trial": trial_index,
                "control_pass_probability": pass_probability,
            }
        )
        out.append(
            replace(
                attempt,
                passed=passed,
                reward=1.0 if passed else 0.0,
                verifier_status="spurious_passed" if passed else "spurious_failed",
                metadata=metadata,
            )
        )
    return out


def _report_pass_at_k(report: EnvironmentPassKReport) -> dict[str, float]:
    return {name: float(value) for name, value in report.pass_at_k.items()}


def evaluate_environment_spurious_reward_control(
    environments: list[EnvironmentSpec | dict[str, Any]],
    attempts: list[EnvironmentAttempt | dict[str, Any]],
    *,
    control_attempts: list[EnvironmentAttempt | dict[str, Any]] | None = None,
    split_by: str = "task_family",
    holdout_fraction: float = 0.2,
    seed: int = 0,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    n_trials: int = 200,
    random_pass_probability: float = 0.05,
    min_observed_pass_at_1: float = 0.0,
    max_control_pass_at_1: float = 0.25,
    min_lift_over_control: float = 0.0,
    require_no_contamination: bool = True,
) -> dict[str, Any]:
    """Audit whether a random/spurious reward control would also pass the gate.

    If ``control_attempts`` are provided, they are treated as real attempts from a
    spurious-reward policy. Otherwise, the audit creates deterministic random
    pass/fail labels over the same holdout attempts and compares the observed
    pass@1 against the 95th-percentile random-control pass@1.
    """

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if random_pass_probability < 0 or random_pass_probability > 1:
        raise ValueError("random_pass_probability must be between 0 and 1")
    if min_observed_pass_at_1 < 0 or min_observed_pass_at_1 > 1:
        raise ValueError("min_observed_pass_at_1 must be between 0 and 1")
    if max_control_pass_at_1 < 0 or max_control_pass_at_1 > 1:
        raise ValueError("max_control_pass_at_1 must be between 0 and 1")
    if min_lift_over_control < -1 or min_lift_over_control > 1:
        raise ValueError("min_lift_over_control must be between -1 and 1")

    ks = sorted({int(k) for k in k_values})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers")

    split = make_environment_holdout_split(
        environments,
        by=split_by,
        fraction=holdout_fraction,
        seed=seed,
    )
    holdout_ids = {env.id for env in split.holdout}
    observed_attempts = _holdout_attempts(attempts, holdout_ids)
    observed_report = evaluate_environment_attempts(split.holdout, observed_attempts, k_values=ks)
    contamination = environment_holdout_contamination(split)

    if control_attempts is not None:
        control_report = evaluate_environment_attempts(
            split.holdout,
            _holdout_attempts(control_attempts, holdout_ids),
            k_values=ks,
        )
        pass_rows = [_report_pass_at_k(control_report)]
        control = {
            "mode": "provided_attempts",
            "n_trials": 0,
            "random_pass_probability": None,
            "report": control_report.to_dict(),
            "pass_at_k_summary": _summarize_pass_at_k(pass_rows, ks),
        }
        control_stat = "mean"
    else:
        rng = random.Random(seed)
        pass_rows = []
        for trial_index in range(n_trials):
            report = evaluate_environment_attempts(
                split.holdout,
                _spurious_attempts(
                    observed_attempts,
                    rng=rng,
                    pass_probability=random_pass_probability,
                    trial_index=trial_index,
                ),
                k_values=ks,
            )
            pass_rows.append(_report_pass_at_k(report))
        control = {
            "mode": "simulated_random_labels",
            "n_trials": n_trials,
            "random_pass_probability": random_pass_probability,
            "report": None,
            "pass_at_k_summary": _summarize_pass_at_k(pass_rows, ks),
        }
        control_stat = "p95"

    observed_pass_at_1 = float(observed_report.pass_at_k.get("pass@1", 0.0))
    control_pass_at_1 = float(control["pass_at_k_summary"]["pass@1"][control_stat])
    lift = observed_pass_at_1 - control_pass_at_1

    reasons: list[str] = []
    if observed_pass_at_1 < min_observed_pass_at_1:
        reasons.append(
            f"observed pass@1 {observed_pass_at_1:.3f} < required {min_observed_pass_at_1:.3f}"
        )
    if control_pass_at_1 > max_control_pass_at_1:
        reasons.append(
            f"spurious control pass@1 {control_pass_at_1:.3f} > allowed {max_control_pass_at_1:.3f}"
        )
    if lift < min_lift_over_control:
        reasons.append(f"observed-control lift {lift:.3f} < required {min_lift_over_control:.3f}")
    if require_no_contamination and contamination:
        reasons.append(f"holdout contamination detected: {len(contamination)} hashes")

    return {
        "schema_version": ENVIRONMENT_SPURIOUS_REWARD_SCHEMA_VERSION,
        "split": split.manifest(),
        "contamination": contamination,
        "observed_report": observed_report.to_dict(),
        "control": control,
        "gate": {
            "ship": not reasons,
            "reasons": reasons,
            "thresholds": {
                "min_observed_pass_at_1": min_observed_pass_at_1,
                "max_control_pass_at_1": max_control_pass_at_1,
                "min_lift_over_control": min_lift_over_control,
                "require_no_contamination": require_no_contamination,
            },
            "observed": {
                "observed_pass_at_1": observed_pass_at_1,
                "control_pass_at_1": control_pass_at_1,
                "control_stat": control_stat,
                "lift_over_control": lift,
            },
        },
    }
