"""Durable proposal, controller selection, and bounded evidence tests."""

import json
import sqlite3

import pytest

from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import (
    AutonomyProfile,
    CampaignTrigger,
    Capability,
    CredentialKind,
    ProposalStatus,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyProposalSubmission,
    StudyStatus,
)
from bashgym.campaigns.persistence import (
    CampaignRepository,
    IdempotencyConflictError,
    InvalidProposalTransitionError,
)
from bashgym.campaigns.proposals import validate_proposal_submission
from bashgym.campaigns.service import CampaignControllerService, CampaignService
from tests.campaigns.test_persistence import campaign, create, manifest


def principal(repository, profile=AutonomyProfile.CODEX_TRUSTED):
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id=("hermes-agent" if profile == AutonomyProfile.HERMES_BOUNDED else "codex-agent"),
        autonomy_profile=profile,
        workspace_ids=("workspace-a",),
    )
    return auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)


def activate(repository: CampaignRepository) -> None:
    create(repository)
    version = 1
    for trigger, key in (
        (CampaignTrigger.VALIDATE, "validate"),
        (CampaignTrigger.VALIDATION_PASSED, "ready"),
        (CampaignTrigger.START, "start"),
    ):
        result = repository.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="campaign-controller",
            credential_kind=CredentialKind.CONTROLLER,
            correlation_id=key,
            idempotency_key=key,
        )
        version = result.campaign.version


def proposal(
    proposal_id: str,
    *,
    priority: int = 50,
    estimated_cost: float = 1.0,
    recipe_schema: bool = True,
) -> StudyProposalSubmission:
    recipe = {
        "data_scope_id": "memexai-approved-training",
        **({"schema_version": "recipe.v1"} if recipe_schema else {}),
    }
    return StudyProposalSubmission(
        proposal_id=proposal_id,
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        hypothesis=f"Hypothesis for {proposal_id}",
        study_family="embedding-retrieval",
        primary_variable="learning_rate",
        controlled_variables=("batch_size",),
        expected_outcome="Improve development MRR.",
        falsification_criterion="Reject if development MRR regresses.",
        estimated_cost=estimated_cost,
        priority=priority,
        dataset_recipe=recipe,
        training_recipe={"schema_version": "recipe.v1"},
        evaluation_recipe={"schema_version": "recipe.v1"},
        stage_plan=StagePlan(
            items=(
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Development evidence is required.",
                ),
            )
        ),
        rationale="A bounded change with a falsifiable development gate.",
    )


@pytest.fixture
def repository(tmp_path):
    value = CampaignRepository(tmp_path / "campaigns.sqlite3")
    value.initialize()
    activate(value)
    return value


def submit(service, value, actor, version, key):
    return service.submit_proposal(
        value,
        expected_version=version,
        principal=actor,
        correlation_id=f"correlation-{key}",
        idempotency_key=key,
    )


def test_hermes_priority_is_normalized_and_replay_is_exact(repository):
    service = CampaignService(repository)
    actor = principal(repository, AutonomyProfile.HERMES_BOUNDED)
    first = submit(service, proposal("proposal-1", priority=99), actor, 4, "submit-1")
    replay = submit(service, proposal("proposal-1", priority=99), actor, 4, "submit-1")

    assert first.record.proposal.priority == 50
    assert first.record.proposal.planner_actor_id == "hermes-agent"
    assert replay.replayed is True
    assert replay.record == first.record
    with pytest.raises(IdempotencyConflictError):
        submit(service, proposal("proposal-2"), actor, 4, "submit-1")


def test_invalid_proposal_is_rejected_without_creating_study(repository):
    service = CampaignService(repository)
    result = submit(
        service,
        proposal("proposal-invalid", recipe_schema=False),
        principal(repository),
        4,
        "invalid",
    )

    assert result.record.proposal.status == ProposalStatus.REJECTED
    assert result.record.validation.reason_codes == ("proposal_recipe_schema_missing",)
    assert result.event.event_type == "campaign:proposal-rejected"
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_studies").fetchone()[0] == 0


def _live_training_proposal(proposal_id: str) -> StudyProposalSubmission:
    return proposal(proposal_id).model_copy(
        update={
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
            },
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.SMOKE_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Prove the pinned remote recipe in a bounded smoke.",
                    ),
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Run the pinned recipe inside the approved budget.",
                    ),
                )
            ),
        }
    )


def test_live_training_requires_declared_compute_capabilities(repository):
    service = CampaignService(repository)
    result = submit(
        service,
        _live_training_proposal("proposal-live-missing-caps"),
        principal(repository),
        4,
        "live-missing-caps",
    )

    assert result.record.validation.reason_codes == (
        "proposal_compute_smoke_capability_missing",
        "proposal_compute_training_capability_missing",
    )


def test_external_handoff_uses_generic_opt_in_and_keeps_legacy_capability_read_only(
    repository,
):
    actor = principal(repository)
    generic = proposal("proposal-generic-handoff").model_copy(
        update={"required_capabilities": frozenset({Capability.HANDOFF_EXTERNAL_PREPARE})}
    )

    denied = validate_proposal_submission(
        generic,
        manifest(),
        actor,
        existing_prerequisite_ids=frozenset(),
    )
    assert denied.reason_codes == ("proposal_external_handoff_not_approved",)

    allowed = validate_proposal_submission(
        generic,
        manifest().model_copy(update={"allow_external_handoff": True}),
        actor,
        existing_prerequisite_ids=frozenset(),
    )
    assert allowed.valid is True

    legacy_actor = actor.model_copy(
        update={"capabilities": actor.capabilities | {Capability.HANDOFF_MEMEXAI_PREPARE}}
    )
    legacy = generic.model_copy(
        update={
            "proposal_id": "proposal-legacy-handoff",
            "required_capabilities": frozenset({Capability.HANDOFF_MEMEXAI_PREPARE}),
        }
    )
    legacy_result = validate_proposal_submission(
        legacy,
        manifest().model_copy(update={"allow_memexai_handoff": True}),
        legacy_actor,
        existing_prerequisite_ids=frozenset(),
    )
    assert legacy_result.reason_codes == ("proposal_legacy_handoff_read_only",)


def test_live_training_rejects_actor_supplied_execution_material(repository):
    service = CampaignService(repository)
    value = _live_training_proposal("proposal-live-command").model_copy(
        update={
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {
                    "executor_kind": "registered_training",
                    "script_path": "/tmp/actor.py",
                },
            },
            "required_capabilities": frozenset(
                {Capability.COMPUTE_SMOKE, Capability.COMPUTE_TRAIN_WITHIN_BUDGET}
            ),
        }
    )
    result = submit(service, value, principal(repository), 4, "live-command")

    assert result.record.validation.reason_codes == (
        "proposal_executable_material_forbidden",
        "proposal_runtime_keys_not_allowed",
    )


def test_live_training_accepts_mode_only_with_declared_capabilities(repository):
    service = CampaignService(repository)
    value = _live_training_proposal("proposal-live-valid").model_copy(
        update={
            "required_capabilities": frozenset(
                {Capability.COMPUTE_SMOKE, Capability.COMPUTE_TRAIN_WITHIN_BUDGET}
            )
        }
    )
    result = submit(service, value, principal(repository), 4, "live-valid")

    assert result.record.validation.valid is True
    assert result.record.proposal.status == ProposalStatus.SUBMITTED


def test_registered_compute_evaluation_requires_declared_capability(repository):
    service = CampaignService(repository)
    value = proposal("proposal-live-evaluation").model_copy(
        update={
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Evaluate the immutable base on approved private compute.",
                    ),
                )
            ),
        }
    )
    missing = submit(service, value, principal(repository), 4, "live-eval-missing")
    assert missing.record.validation.reason_codes == (
        "proposal_development_evaluation_capability_missing",
    )

    accepted = submit(
        service,
        value.model_copy(
            update={
                "proposal_id": "proposal-live-evaluation-valid",
                "required_capabilities": frozenset({Capability.EVAL_DEVELOPMENT}),
            }
        ),
        principal(repository),
        missing.campaign.version,
        "live-eval-valid",
    )
    assert accepted.record.validation.valid is True


def test_withdraw_requires_submitted_status_and_expected_version(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    submitted = submit(service, proposal("proposal-1"), actor, 4, "submit")
    withdrawn = service.withdraw_proposal(
        "workspace-a",
        "campaign-1",
        "proposal-1",
        expected_version=submitted.campaign.version,
        principal=actor,
        correlation_id="withdraw",
        idempotency_key="withdraw",
    )
    replay = service.withdraw_proposal(
        "workspace-a",
        "campaign-1",
        "proposal-1",
        expected_version=submitted.campaign.version,
        principal=actor,
        correlation_id="withdraw",
        idempotency_key="withdraw",
    )

    assert withdrawn.record.proposal.status == ProposalStatus.WITHDRAWN
    assert replay.replayed is True
    with pytest.raises(InvalidProposalTransitionError):
        service.withdraw_proposal(
            "workspace-a",
            "campaign-1",
            "proposal-1",
            expected_version=withdrawn.campaign.version,
            principal=actor,
            correlation_id="withdraw-again",
            idempotency_key="withdraw-again",
        )


def test_controller_selection_is_deterministic_and_creates_exactly_one_study(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    first = submit(service, proposal("proposal-low", priority=20), actor, 4, "low")
    second = submit(
        service,
        proposal("proposal-costly", priority=90, estimated_cost=3),
        actor,
        first.campaign.version,
        "costly",
    )
    third = submit(
        service,
        proposal("proposal-cheap", priority=90, estimated_cost=1),
        actor,
        second.campaign.version,
        "cheap",
    )
    controller = CampaignControllerService(repository, controller_id="campaign-controller")
    selected = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=third.campaign.version,
        correlation_id="select",
        idempotency_key="select",
    )
    replay = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=third.campaign.version,
        correlation_id="select",
        idempotency_key="select",
    )

    assert selected is not None
    assert selected.record.proposal.proposal_id == "proposal-cheap"
    assert selected.record.proposal.status == ProposalStatus.ACCEPTED
    assert selected.study.status.value == "validated"
    assert replay == selected.__class__(
        selected.campaign,
        selected.event,
        selected.record,
        selected.study,
        replayed=True,
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_studies").fetchone()[0] == 1


def test_controller_skips_higher_priority_proposal_with_unready_prerequisite(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    initial = submit(service, proposal("proposal-prior"), actor, 4, "prior")
    controller = CampaignControllerService(repository, controller_id="campaign-controller")
    prior = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=initial.campaign.version,
        correlation_id="select-prior",
        idempotency_key="select-prior",
    )
    assert prior is not None
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = NULL
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ("workspace-a", "campaign-1"),
        )
    blocked_submission = proposal("proposal-blocked", priority=100).model_copy(
        update={"prerequisite_study_ids": (prior.study.study_id,)}
    )
    blocked = submit(
        service,
        blocked_submission,
        actor,
        prior.campaign.version,
        "blocked",
    )
    ready = submit(
        service,
        proposal("proposal-ready", priority=50),
        actor,
        blocked.campaign.version,
        "ready-independent",
    )
    selected = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=ready.campaign.version,
        correlation_id="select-ready",
        idempotency_key="select-ready",
    )

    assert selected is not None
    assert selected.record.proposal.proposal_id == "proposal-ready"

    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_studies SET status = ?
            WHERE workspace_id = ? AND study_id = ?
            """,
            (StudyStatus.COMPLETED.value, "workspace-a", prior.study.study_id),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = NULL
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ("workspace-a", "campaign-1"),
        )
    unblocked = controller.select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=selected.campaign.version,
        correlation_id="select-unblocked",
        idempotency_key="select-unblocked",
    )
    assert unblocked is not None
    assert unblocked.record.proposal.proposal_id == "proposal-blocked"


def test_advance_request_never_accepts_proposal(repository):
    service = CampaignService(repository)
    actor = principal(repository)
    submitted = submit(service, proposal("proposal-1"), actor, 4, "submit")
    advanced = service.request_advance(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        principal=actor,
        correlation_id="advance",
        idempotency_key="advance",
    )

    assert advanced.event.event_type == "campaign:advance-requested"
    assert advanced.campaign.active_study_id is None
    assert service.proposals("workspace-a", "campaign-1", actor)[0].proposal.status == (
        ProposalStatus.SUBMITTED
    )
    with sqlite3.connect(repository.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_studies").fetchone()[0] == 0


def test_evidence_snapshot_is_bounded_and_excludes_rows_and_uris(repository, tmp_path):
    service = CampaignService(repository)
    actor = principal(repository)
    rejected = submit(
        service,
        proposal("proposal-invalid", recipe_schema=False),
        actor,
        4,
        "reject",
    )
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_evaluations(
                workspace_id, campaign_id, evaluation_id, evaluation_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "protected-eval",
                json.dumps({"protected_row": "NEVER_SURFACE_THIS"}),
                campaign().created_at.isoformat(),
            ),
        )
        nemo_reference = {
            "artifact_id": "artifact-nemo",
            "artifact_sha256": "b" * 64,
            "bundle_digest": "c" * 64,
            "environment_id": "star-count-v1",
            "environment_digest": "d" * 64,
            "rollout_batch_digest": "e" * 64,
            "token_evidence_digest": "f" * 64,
            "refit_receipt_digest": "1" * 64,
            "rollout_count": 2,
            "mean_total_reward": 0.75,
            "training_step": 4,
            "policy_revision": 4,
        }
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id, uri,
                sha256, size_bytes, schema_name, sealed, valid, metadata_json, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, 10, ?, 1, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "artifact-nemo",
                str(tmp_path / "private" / "nemo_gym_campaign_evidence.json"),
                "b" * 64,
                "nemo_gym_campaign_evidence.v1",
                json.dumps(
                    {
                        "nemo_gym": nemo_reference,
                        "raw_rollout": "NEVER_SURFACE_THIS",
                    }
                ),
                campaign().created_at.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id, uri,
                sha256, size_bytes, schema_name, sealed, valid, metadata_json, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, 10, ?, 1, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "artifact-safe",
                str(tmp_path / "private" / "model.bin"),
                "a" * 64,
                "model.v1",
                json.dumps({"private_excerpt": "NEVER_SURFACE_THIS"}),
                campaign().created_at.isoformat(),
            ),
        )

    snapshot = service.evidence("workspace-a", "campaign-1", actor)
    serialized = snapshot.model_dump_json()
    assert snapshot.campaign_version == rejected.campaign.version
    assert snapshot.proposal_counts[ProposalStatus.REJECTED] == 1
    assert snapshot.artifact_references[0].artifact_id == "artifact-safe"
    assert snapshot.nemo_gym_evidence_references[0].artifact_id == "artifact-nemo"
    assert snapshot.nemo_gym_evidence_references[0].rollout_count == 2
    assert "NEVER_SURFACE_THIS" not in serialized
    assert "model.bin" not in serialized
    assert "uri" not in serialized.casefold()
    assert service.evidence("workspace-a", "campaign-1", actor).snapshot_digest == (
        snapshot.snapshot_digest
    )
