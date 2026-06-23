"""Tests for executable-environment pass@k reporting."""

from __future__ import annotations

import pytest

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.environment_passk import (
    EnvironmentAttempt,
    evaluate_environment_attempts,
    evaluate_environment_pass_at_k,
)


def _attempts() -> list[EnvironmentAttempt]:
    return [
        EnvironmentAttempt(
            "env_a",
            0,
            True,
            verifier_status="passed",
            tool_calls=3,
            action_tokens=20,
            observation_tokens=80,
        ),
        EnvironmentAttempt(
            "env_a",
            1,
            False,
            verifier_status="failed",
            tool_calls=4,
            action_tokens=30,
            observation_tokens=90,
        ),
        EnvironmentAttempt("env_a", 2, True, verifier_status="passed", timeout=False),
        EnvironmentAttempt("env_a", 3, False, verifier_status="timeout", timeout=True),
        EnvironmentAttempt("env_b", 0, False, verifier_status="failed"),
        EnvironmentAttempt("env_b", 1, False, verifier_status="failed"),
        EnvironmentAttempt("env_b", 2, False, verifier_status="failed"),
        EnvironmentAttempt("env_b", 3, False, verifier_status="failed"),
    ]


def test_evaluate_environment_attempts_reports_passk_and_telemetry():
    report = evaluate_environment_attempts(
        ["env_a", "env_b"],
        _attempts(),
        k_values=[1, 2],
    )
    data = report.to_dict()

    assert data["n_environments"] == 2
    assert data["n_attempts"] == 8
    assert data["pass_at_k"]["pass@1"] == pytest.approx(0.25)
    assert data["pass_at_k"]["pass@2"] == pytest.approx((5 / 6) / 2)
    assert data["per_environment"]["env_a"]["passes"] == 2
    assert data["per_environment"]["env_b"]["pass@2"] == 0.0
    assert data["attempt_summary"]["timeout_rate"] == pytest.approx(0.125)
    assert data["attempt_summary"]["mean_action_tokens"] == pytest.approx(25.0)
    assert data["attempt_summary"]["mean_observation_tokens"] == pytest.approx(85.0)
    assert data["attempt_summary"]["verifier_status_distribution"]["failed"] == 5


def test_evaluate_environment_attempts_rejects_unknown_environment():
    with pytest.raises(ValueError, match="unknown environments"):
        evaluate_environment_attempts(
            ["env_a"],
            [EnvironmentAttempt("env_b", 0, True)],
            k_values=[1],
        )


def test_evaluate_environment_attempts_requires_enough_samples():
    with pytest.raises(ValueError, match="needs at least 2"):
        evaluate_environment_attempts(
            ["env_a"],
            [EnvironmentAttempt("env_a", 0, True)],
            k_values=[2],
        )


def test_evaluate_environment_pass_at_k_collects_with_injected_runner():
    envs = [
        EnvironmentSpec(id="env_a", instruction="Task A"),
        EnvironmentSpec(id="env_b", instruction="Task B"),
    ]

    def run_episode(env, attempt):
        return {
            "environment_id": env.id,
            "attempt_index": attempt,
            "passed": env.id == "env_a" and attempt == 0,
            "verifier_status": "passed" if env.id == "env_a" and attempt == 0 else "failed",
        }

    report = evaluate_environment_pass_at_k(envs, run_episode, n_samples=2, k_values=[1])

    assert report.pass_at_k["pass@1"] == pytest.approx(0.25)
    assert report.to_dict()["per_environment"]["env_a"]["success_rate"] == 0.5
