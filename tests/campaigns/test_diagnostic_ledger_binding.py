"""Checkpoint identity against the real migrations and scheduler (no remote launch)."""

from datetime import timedelta

import pytest

from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignSpec,
    AutoResearchProposalControl,
    AutoResearchRepository,
    AutoResearchStopRules,
    ExperimentRole,
)
from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.persistence import CampaignPersistenceError
from bashgym.campaigns.runtime import ActionSpec
from tests.campaigns.test_worker import START, active_repository, seed_validated_study


def test_parent_binding_resolves_real_scheduled_evaluation_record(tmp_path):
    active_repository(tmp_path / "campaigns.sqlite3")
    repository = AutoResearchRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    repository.create_autoresearch_spec(
        AutoResearchCampaignSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            primary_metric="accuracy",
            metric_direction="maximize",
            stop_rules=AutoResearchStopRules(
                max_attempts=4,
                budget_unit="gpu_hours",
                max_total_cost=8,
                minimum_improvement=0.01,
            ),
        )
    )
    plan = seed_validated_study(repository, "parent", stage=StageKind.DEVELOPMENT_EVALUATION)
    seed_validated_study(repository, "probe", sequence=2, stage=StageKind.CONTRACT_EVALUATION)
    repository.register_autoresearch_proposal(
        AutoResearchProposalControl(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            proposal_id="proposal-probe",
            role=ExperimentRole.DIAGNOSTIC,
            parent_proposal_id="proposal-parent",
            created_at=START,
        )
    )
    leader = repository.acquire_lease(
        "campaign-worker-leader", "worker-test", ttl=timedelta(seconds=60), now=START
    )
    attempt = repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="parent",
            stage_index=0,
            stage=StageKind.DEVELOPMENT_EVALUATION,
            input_contract=plan.items[0].input_contract,
            candidate_digest="d" * 64,
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.1,
            executor_kind="ssh_remote",
            executor_config={
                "stage": "development_evaluation",
                "evaluated_model_digest": "a" * 64,
                "evaluation_binding": {
                    "evaluation_suite_id": "suite-a",
                    "evaluation_code_digest": "b" * 64,
                    "dataset_version_id": "data-a",
                    "dataset_content_digest": "c" * 64,
                },
            },
        ),
        leader,
        expected_campaign_version=4,
        now=START,
    )
    binding = {"parent_model_digest": "a" * 64}
    with pytest.raises(CampaignPersistenceError, match="parent_checkpoint_mismatch"):
        repository.validate_diagnostic_parent_binding(
            "workspace-a", "campaign-1", "proposal-probe", binding
        )
    # Fixture models the completed executor record; no adapter or compute is launched.
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_actions SET status = 'completed' WHERE action_id = ?",
            (attempt.action_id,),
        )
        connection.execute(
            "UPDATE campaign_attempts SET status = 'completed' WHERE attempt_id = ?",
            (attempt.attempt_id,),
        )
    repository.validate_diagnostic_parent_binding(
        "workspace-a", "campaign-1", "proposal-probe", binding
    )
    with pytest.raises(CampaignPersistenceError, match="parent_checkpoint_mismatch"):
        repository.validate_diagnostic_parent_binding(
            "workspace-a", "campaign-1", "proposal-probe", {"parent_model_digest": "f" * 64}
        )
