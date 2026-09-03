"""Built-in executor adapter policy: registered kinds, stages, repair eligibility."""

import pytest

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.executor_adapters import (
    DevelopmentEvaluationExecutorAdapter,
    FakeExecutorAdapter,
    SshRemoteExecutorAdapter,
    build_default_registry,
)
from bashgym.campaigns.executors import fake_digest
from bashgym.campaigns.runtime import ActionSpec

REMOTE_STAGES = frozenset(
    {
        StageKind.DATA_BUILD,
        StageKind.CONTRACT_EVALUATION,
        StageKind.SMOKE_TRAINING,
        StageKind.FULL_TRAINING,
        StageKind.DEVELOPMENT_EVALUATION,
    }
)


def test_default_registry_holds_exactly_the_three_built_in_kinds() -> None:
    assert build_default_registry().kinds() == (
        "development_evaluation",
        "fake",
        "ssh_remote",
    )


def test_fake_adapter_accepts_every_stage_and_allows_repair() -> None:
    adapter = FakeExecutorAdapter()

    assert adapter.kind == "fake"
    assert adapter.allowed_stages == frozenset(StageKind)
    assert len(adapter.allowed_stages) == 9
    assert adapter.repair_allowed() is True
    assert adapter.reuses_completed_results is False


def test_ssh_remote_adapter_is_restricted_to_compute_stages_and_forbids_repair() -> None:
    adapter = SshRemoteExecutorAdapter()

    assert adapter.kind == "ssh_remote"
    assert adapter.allowed_stages == REMOTE_STAGES
    assert StageKind.PROMOTION not in adapter.allowed_stages
    assert adapter.repair_allowed() is False
    assert adapter.reuses_completed_results is True


def test_development_evaluation_adapter_is_restricted_to_its_stage_and_allows_repair() -> None:
    adapter = DevelopmentEvaluationExecutorAdapter()

    assert adapter.kind == "development_evaluation"
    assert adapter.allowed_stages == frozenset({StageKind.DEVELOPMENT_EVALUATION})
    assert adapter.repair_allowed() is True
    assert adapter.reuses_completed_results is False


def test_registry_reports_the_stages_each_adapter_declares() -> None:
    registry = build_default_registry()

    for adapter in (
        FakeExecutorAdapter(),
        SshRemoteExecutorAdapter(),
        DevelopmentEvaluationExecutorAdapter(),
    ):
        assert registry.allowed_stages(adapter.kind) == adapter.allowed_stages
        assert registry.get(adapter.kind).repair_allowed() == adapter.repair_allowed()


@pytest.mark.parametrize(
    ("executor_kind", "stage"),
    [
        ("ssh_remote", StageKind.PROMOTION),
        ("development_evaluation", StageKind.FULL_TRAINING),
    ],
)
def test_action_spec_rejects_a_stage_outside_the_registered_set(executor_kind, stage) -> None:
    with pytest.raises(ValueError, match="restricted to its registered stages"):
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=stage,
            input_contract={},
            candidate_digest=fake_digest("candidate"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind=executor_kind,
        )


@pytest.mark.parametrize(
    ("executor_kind", "stage"),
    [
        ("fake", StageKind.PROMOTION),
        ("ssh_remote", StageKind.FULL_TRAINING),
        ("development_evaluation", StageKind.DEVELOPMENT_EVALUATION),
    ],
)
def test_action_spec_accepts_a_stage_inside_the_registered_set(executor_kind, stage) -> None:
    spec = ActionSpec(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        stage_index=0,
        stage=stage,
        input_contract={},
        candidate_digest=fake_digest("candidate"),
        manifest_revision=1,
        budget_unit="gpu_hours",
        budget_reservation=0.25,
        executor_kind=executor_kind,
    )

    assert spec.executor_kind == executor_kind
