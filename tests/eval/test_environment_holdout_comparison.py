"""Tests for paired-bootstrap environment holdout comparison gates."""

from __future__ import annotations

import pytest

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.environment_holdout_comparison import (
    ENVIRONMENT_HOLDOUT_COMPARISON_SCHEMA_VERSION,
    evaluate_environment_holdout_comparison_gate,
)
from bashgym.eval.environment_passk import EnvironmentAttempt


def _env(env_id: str, family: str) -> EnvironmentSpec:
    return EnvironmentSpec(
        id=env_id,
        instruction=f"Task for {family}",
        metadata={"task_family": family, "generator_seed": family},
        files={"README.md": f"{family}\n"},
    )


def _attempts(
    environments: list[EnvironmentSpec],
    *,
    passed_by_env: dict[str, bool],
) -> list[EnvironmentAttempt]:
    return [
        EnvironmentAttempt(
            env.id,
            attempt_index=0,
            passed=passed_by_env.get(env.id, False),
            verifier_status="passed" if passed_by_env.get(env.id, False) else "failed",
        )
        for env in environments
    ]


def test_environment_holdout_comparison_ships_on_positive_clustered_delta():
    environments = [
        _env("env_a", "a"),
        _env("env_b", "b"),
        _env("env_c", "c"),
        _env("env_d", "d"),
    ]
    base_attempts = _attempts(environments, passed_by_env={})
    candidate_attempts = _attempts(
        environments,
        passed_by_env={env.id: True for env in environments},
    )

    result = evaluate_environment_holdout_comparison_gate(
        environments,
        base_attempts,
        candidate_attempts,
        split_by="task_family",
        cluster_by="task_family",
        holdout_fraction=0.5,
        seed=3,
        k_values=[1],
        compare_k=1,
        min_delta=0.5,
        require_ci_excludes_zero=True,
        n_resamples=50,
    )

    assert result["schema_version"] == ENVIRONMENT_HOLDOUT_COMPARISON_SCHEMA_VERSION
    assert result["compare_metric"] == "pass@1"
    assert result["base_report"]["pass_at_k"]["pass@1"] == pytest.approx(0.0)
    assert result["candidate_report"]["pass_at_k"]["pass@1"] == pytest.approx(1.0)
    assert result["bootstrap"]["mean"] == pytest.approx(1.0)
    assert result["bootstrap"]["better"] is True
    assert result["gate"]["ship"] is True
    assert result["gate"]["reasons"] == []


def test_environment_holdout_comparison_blocks_when_ci_does_not_clear_zero():
    environments = [
        _env("env_a", "a"),
        _env("env_b", "b"),
        _env("env_c", "c"),
        _env("env_d", "d"),
    ]
    base_attempts = _attempts(environments, passed_by_env={})
    candidate_attempts = _attempts(environments, passed_by_env={})

    result = evaluate_environment_holdout_comparison_gate(
        environments,
        base_attempts,
        candidate_attempts,
        split_by="task_family",
        cluster_by="task_family",
        holdout_fraction=0.5,
        seed=3,
        k_values=[1],
        require_ci_excludes_zero=True,
        n_resamples=50,
    )

    assert result["bootstrap"]["mean"] == pytest.approx(0.0)
    assert result["bootstrap"]["better"] is False
    assert result["gate"]["ship"] is False
    assert any("does not clear zero" in reason for reason in result["gate"]["reasons"])


def test_environment_holdout_comparison_blocks_candidate_tamper():
    environments = [_env("env_a", "a"), _env("env_b", "b")]
    base_attempts = _attempts(environments, passed_by_env={})
    candidate_attempts = _attempts(environments, passed_by_env={env.id: True for env in environments})
    for attempt in candidate_attempts:
        attempt.verifier_status = "tampered"

    result = evaluate_environment_holdout_comparison_gate(
        environments,
        base_attempts,
        candidate_attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=0,
        k_values=[1],
        max_candidate_tamper_rate=0.0,
        n_resamples=20,
    )

    assert result["gate"]["ship"] is False
    assert any("candidate tamper rate" in reason for reason in result["gate"]["reasons"])


def test_environment_holdout_comparison_rejects_bad_cluster_key():
    environments = [_env("env_a", "a"), _env("env_b", "b")]

    with pytest.raises(ValueError, match="split_by must be one of"):
        evaluate_environment_holdout_comparison_gate(
            environments,
            _attempts(environments, passed_by_env={}),
            _attempts(environments, passed_by_env={}),
            cluster_by="bad",
            k_values=[1],
        )
