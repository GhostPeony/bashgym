from __future__ import annotations

import pytest

from bashgym.campaigns.reward_integrity import (
    AutoResearchRewardIntegrityEvidence,
    RewardConstraint,
    build_reward_integrity_evidence,
    build_reward_spec,
)
from bashgym.environments.contracts import (
    EnvironmentSpec,
    RewardComponentSpec,
    RewardConstraintSpec,
    VerifierSpec,
)


def _environment() -> EnvironmentSpec:
    return EnvironmentSpec(
        id="env_reward_integrity",
        instruction="Return a correct answer in the required format.",
        source="unit",
        verifier=VerifierSpec(
            command="python verify.py",
            reward_type="components",
            reward_components=[
                RewardComponentSpec(
                    name="correctness",
                    weight=1.0,
                    hard_constraint=RewardConstraintSpec(minimum=0.8),
                ),
                RewardComponentSpec(name="format", weight=0.2),
            ],
        ),
    )


def test_reward_spec_reuses_verifier_components_without_exposing_executable_details() -> None:
    spec = build_reward_spec(_environment())

    assert spec.environment_id == "env_reward_integrity"
    assert [item.name for item in spec.components] == ["correctness", "format"]
    assert spec.components[0].hard_constraint.model_dump(mode="json") == {
        "schema_version": "bashgym.reward_constraint.v1",
        "minimum": 0.8,
        "maximum": None,
    }
    payload = spec.model_dump(mode="json")
    assert payload["reward_spec_digest"]
    assert "command" not in str(payload)
    assert "path" not in str(payload)


def test_reward_integrity_projects_component_distributions_constraints_and_canaries() -> None:
    evidence = build_reward_integrity_evidence(
        _environment(),
        reward_rows=(
            {"correctness": 1.0, "format": 0.5},
            {"correctness": 0.5, "format": 1.0},
        ),
        canary_summary={
            "total": 4,
            "guarded": 3,
            "failed": 1,
            "guard_rate": 0.75,
            "categories": {
                "env_manifest_tamper": 1,
                "private_fixture_tamper": 1,
                "tests_tamper": 1,
                "verifier_tamper": 1,
            },
        },
    )

    components = {item.name: item for item in evidence.components}
    assert components["correctness"].mean == 0.75
    assert components["correctness"].standard_deviation == 0.25
    assert components["correctness"].hard_constraint_violations == 1
    assert components["correctness"].hard_constraint_violation_rate == 0.5
    assert components["format"].hard_constraint_violation_rate == 0.0
    assert evidence.hard_constraint_violation_rate == 0.5
    assert evidence.canaries.total == 4
    assert evidence.canaries.failure_rate == 0.25
    assert evidence.canaries.categories == (
        "env_manifest_tamper",
        "private_fixture_tamper",
        "tests_tamper",
        "verifier_tamper",
    )
    assert evidence.method_evidence() == {
        "reward_spec_verified": True,
        "reward_canary_cases": 4,
        "reward_canary_failure_rate": 0.25,
        "hard_constraint_violation_rate": 0.5,
    }


def test_reward_integrity_rejects_mismatched_components_and_inconsistent_canaries() -> None:
    with pytest.raises(ValueError, match="reward component mismatch"):
        build_reward_integrity_evidence(
            _environment(),
            reward_rows=({"correctness": 1.0},),
            canary_summary={"total": 1, "guarded": 1, "failed": 0, "guard_rate": 1.0},
        )

    with pytest.raises(ValueError, match="canary summary"):
        build_reward_integrity_evidence(
            _environment(),
            reward_rows=({"correctness": 1.0, "format": 1.0},),
            canary_summary={"total": 4, "guarded": 4, "failed": 1, "guard_rate": 1.0},
        )


def test_reward_constraint_requires_a_finite_ordered_bound() -> None:
    with pytest.raises(ValueError, match="at least one bound"):
        RewardConstraintSpec()
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        RewardConstraintSpec(minimum=1.0, maximum=0.0)
    with pytest.raises(ValueError, match="finite"):
        RewardConstraintSpec(minimum=float("inf"))
    with pytest.raises(ValueError, match="at least one bound"):
        RewardConstraint()
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        RewardConstraint(minimum=1.0, maximum=0.0)


def test_reward_integrity_contract_rejects_tampered_aggregate_counts() -> None:
    evidence = build_reward_integrity_evidence(
        _environment(),
        reward_rows=({"correctness": 1.0, "format": 1.0},),
        canary_summary={
            "total": 1,
            "guarded": 1,
            "failed": 0,
            "guard_rate": 1.0,
        },
    )
    payload = evidence.model_dump(mode="json")
    payload["hard_constraint_violations"] = 1

    with pytest.raises(ValueError, match="hard-constraint totals"):
        AutoResearchRewardIntegrityEvidence.model_validate(payload)
