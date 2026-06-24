"""Tests for spurious-reward negative controls over environment holdouts."""

from __future__ import annotations

import pytest

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.environment_passk import EnvironmentAttempt
from bashgym.eval.environment_spurious_reward import (
    ENVIRONMENT_SPURIOUS_REWARD_SCHEMA_VERSION,
    evaluate_environment_spurious_reward_control,
)


def _env(env_id: str, family: str) -> EnvironmentSpec:
    return EnvironmentSpec(
        id=env_id,
        instruction=f"Task for {family}",
        metadata={"task_family": family},
        files={"README.md": f"{family}\n"},
    )


def _attempts(environments: list[EnvironmentSpec], *, passed: bool) -> list[EnvironmentAttempt]:
    return [
        EnvironmentAttempt(
            env.id,
            attempt_index=0,
            passed=passed,
            reward=1.0 if passed else 0.0,
            verifier_status="passed" if passed else "failed",
        )
        for env in environments
    ]


def test_spurious_reward_control_simulation_is_deterministic_and_can_pass():
    environments = [_env("env_a", "a"), _env("env_b", "b"), _env("env_c", "c"), _env("env_d", "d")]
    attempts = _attempts(environments, passed=True)

    first = evaluate_environment_spurious_reward_control(
        environments,
        attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=3,
        k_values=[1],
        n_trials=12,
        random_pass_probability=0.0,
        max_control_pass_at_1=0.0,
        min_lift_over_control=0.5,
    )
    second = evaluate_environment_spurious_reward_control(
        environments,
        attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=3,
        k_values=[1],
        n_trials=12,
        random_pass_probability=0.0,
        max_control_pass_at_1=0.0,
        min_lift_over_control=0.5,
    )

    assert first == second
    assert first["schema_version"] == ENVIRONMENT_SPURIOUS_REWARD_SCHEMA_VERSION
    assert first["control"]["mode"] == "simulated_random_labels"
    assert first["control"]["pass_at_k_summary"]["pass@1"]["p95"] == pytest.approx(0.0)
    assert first["gate"]["observed"]["observed_pass_at_1"] == pytest.approx(1.0)
    assert first["gate"]["ship"] is True


def test_spurious_reward_control_blocks_when_random_control_also_passes():
    environments = [_env("env_a", "a"), _env("env_b", "b"), _env("env_c", "c"), _env("env_d", "d")]
    attempts = _attempts(environments, passed=True)

    result = evaluate_environment_spurious_reward_control(
        environments,
        attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=3,
        k_values=[1],
        n_trials=4,
        random_pass_probability=1.0,
        max_control_pass_at_1=0.5,
    )

    assert result["control"]["pass_at_k_summary"]["pass@1"]["p95"] == pytest.approx(1.0)
    assert result["gate"]["ship"] is False
    assert any("spurious control pass@1" in reason for reason in result["gate"]["reasons"])


def test_spurious_reward_control_accepts_provided_control_attempts():
    environments = [_env("env_a", "a"), _env("env_b", "b"), _env("env_c", "c"), _env("env_d", "d")]
    attempts = _attempts(environments, passed=True)
    control_attempts = _attempts(environments, passed=True)

    result = evaluate_environment_spurious_reward_control(
        environments,
        attempts,
        control_attempts=control_attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=3,
        k_values=[1],
        max_control_pass_at_1=0.5,
    )

    assert result["control"]["mode"] == "provided_attempts"
    assert result["control"]["report"]["pass_at_k"]["pass@1"] == pytest.approx(1.0)
    assert result["gate"]["observed"]["control_stat"] == "mean"
    assert result["gate"]["ship"] is False


def test_spurious_reward_control_does_not_mutate_attempts():
    environments = [_env("env_a", "a"), _env("env_b", "b")]
    attempts = _attempts(environments, passed=False)
    before = [attempt.to_dict() for attempt in attempts]

    evaluate_environment_spurious_reward_control(
        environments,
        attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=0,
        k_values=[1],
        n_trials=3,
        random_pass_probability=1.0,
    )

    assert [attempt.to_dict() for attempt in attempts] == before


def test_spurious_reward_control_rejects_bad_probability():
    with pytest.raises(ValueError, match="random_pass_probability"):
        evaluate_environment_spurious_reward_control(
            [_env("env_a", "a")],
            _attempts([_env("env_a", "a")], passed=True),
            k_values=[1],
            random_pass_probability=1.5,
        )
