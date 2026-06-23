"""Tests for contamination-aware environment holdout gates."""

from __future__ import annotations

import pytest

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval.environment_holdout import (
    environment_group_key,
    environment_hash,
    environment_holdout_contamination,
    evaluate_environment_holdout_gate,
    make_environment_holdout_split,
)
from bashgym.eval.environment_passk import EnvironmentAttempt


def _env(env_id: str, family: str, domain: str = "cli") -> EnvironmentSpec:
    return EnvironmentSpec(
        id=env_id,
        instruction=f"Task for {family}",
        domain=domain,
        metadata={"task_family": family, "generator_seed": family},
        files={"README.md": f"{family}\n"},
    )


def _attempts(environments: list[EnvironmentSpec], *, passed: bool = True) -> list[EnvironmentAttempt]:
    return [
        EnvironmentAttempt(env.id, attempt_index=0, passed=passed, verifier_status="passed" if passed else "failed")
        for env in environments
    ]


def test_environment_holdout_split_is_grouped_and_deterministic():
    environments = [_env("env_a", "a"), _env("env_b", "b"), _env("env_c", "c"), _env("env_d", "d")]

    first = make_environment_holdout_split(environments, by="task_family", fraction=0.5, seed=7)
    second = make_environment_holdout_split(environments, by="task_family", fraction=0.5, seed=7)

    assert first.manifest() == second.manifest()
    assert {environment_group_key(env, "task_family") for env in first.train}.isdisjoint(
        {environment_group_key(env, "task_family") for env in first.holdout}
    )
    assert len(first.holdout) == 2
    assert len(first.train) == 2


def test_environment_hash_ignores_ids_to_catch_copied_tasks():
    assert environment_hash(_env("env_original", "same")) == environment_hash(_env("env_copy", "same"))


def test_environment_holdout_contamination_detects_copied_task_across_groups():
    original = _env("env_original", "train_family")
    copied = EnvironmentSpec(
        id="env_copy",
        instruction=original.instruction,
        domain=original.domain,
        metadata={"task_family": "holdout_family", "generator_seed": "holdout_family"},
        files=dict(original.files),
    )
    split = make_environment_holdout_split([original, copied], by="task_family", fraction=0.5, seed=0)

    assert environment_holdout_contamination(split) == [environment_hash(original)]


def test_environment_holdout_gate_reports_ship_verdict_on_holdout_only():
    environments = [_env("env_a", "a"), _env("env_b", "b"), _env("env_c", "c"), _env("env_d", "d")]
    split = make_environment_holdout_split(environments, by="task_family", fraction=0.5, seed=3)
    attempts = _attempts(environments, passed=False)
    for attempt in attempts:
        if attempt.environment_id in {env.id for env in split.holdout}:
            attempt.passed = True
            attempt.verifier_status = "passed"

    result = evaluate_environment_holdout_gate(
        environments,
        attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=3,
        k_values=[1],
        min_pass_at_1=0.5,
    )

    assert result["split"]["holdout_ids"] == split.manifest()["holdout_ids"]
    assert result["report"]["n_environments"] == 2
    assert result["report"]["pass_at_k"]["pass@1"] == pytest.approx(1.0)
    assert result["gate"]["ship"] is True
    assert result["gate"]["reasons"] == []


def test_environment_holdout_gate_blocks_low_pass_rate_and_tamper():
    environments = [_env("env_a", "a"), _env("env_b", "b")]
    split = make_environment_holdout_split(environments, by="task_family", fraction=0.5, seed=0)
    attempts = _attempts(environments, passed=True)
    for attempt in attempts:
        if attempt.environment_id in {env.id for env in split.holdout}:
            attempt.passed = False
            attempt.verifier_status = "tampered"
            attempt.metadata["tamper_detected"] = True

    result = evaluate_environment_holdout_gate(
        environments,
        attempts,
        split_by="task_family",
        holdout_fraction=0.5,
        seed=0,
        k_values=[1],
        min_pass_at_1=0.5,
        max_tamper_rate=0.0,
    )

    assert result["gate"]["ship"] is False
    assert any("pass@1" in reason for reason in result["gate"]["reasons"])
    assert any("tamper rate" in reason for reason in result["gate"]["reasons"])


def test_environment_holdout_gate_rejects_bad_split_key():
    with pytest.raises(ValueError, match="split_by must be one of"):
        make_environment_holdout_split([_env("env_a", "a")], by="bad")
