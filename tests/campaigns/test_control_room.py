"""Atomic, bounded control-room snapshot repository tests."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from bashgym._compat import UTC
from bashgym.campaigns import control_room as control_room_module
from bashgym.campaigns import transitions as campaign_transitions
from bashgym.campaigns.autoresearch import AutoResearchRepository
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    CampaignControlRoomStateV1,
    Capability,
    ControllerObservationV1,
    CredentialKind,
    ReadinessSummaryV1,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyProposal,
)
from bashgym.campaigns.control_room import (
    CandidateProvenance,
    ChampionProvenance,
    build_control_room_snapshot,
)
from bashgym.campaigns.persistence import (
    CampaignPersistenceError,
    CampaignRepository,
    RecordNotFoundError,
)
from bashgym.campaigns.remote import RemoteRunIdentity
from tests.campaigns.test_persistence import campaign, create, revision


class TracedCampaignRepository(CampaignRepository):
    """Observe only the connection used by the snapshot read."""

    trace_snapshot = False
    snapshot_connection_count = 0
    snapshot_statements: list[str]

    @contextmanager
    def _connection(self, *, immediate: bool = False):
        with super()._connection(immediate=immediate) as connection:
            if self.trace_snapshot:
                self.snapshot_connection_count += 1
                self.snapshot_statements = []
                connection.set_trace_callback(self.snapshot_statements.append)
            yield connection


@pytest.fixture
def repository(tmp_path) -> TracedCampaignRepository:
    value = TracedCampaignRepository(tmp_path / "campaigns.sqlite3")
    value.initialize()
    create(value)
    other = campaign(workspace_id="workspace-b", campaign_id="campaign-other")
    value.create_campaign(
        other,
        revision(other),
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="correlation-other",
        idempotency_key="create-other",
    )
    return value


def ready_snapshot(durable):
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    return build_control_room_snapshot(
        durable,
        ControllerObservationV1(
            controller_observation_version=1,
            state="online",
            observed_at=now,
            heartbeat_age_seconds=0,
            lease_expires_at=now + timedelta(seconds=10),
            controller_instance_id="resident-worker",
            safe_guidance=None,
        ),
        ReadinessSummaryV1(
            materializable=True,
            launch_ready=True,
            checked_at=now,
            activation_receipt_digest=None,
            doctor_receipt_digest=None,
            blocking_codes=(),
        ),
        principal=ActorPrincipal(
            actor_id="reader",
            autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
            credential_id="reader-credential",
            credential_kind=CredentialKind.ACCESS,
            workspace_ids=("workspace-a",),
            capabilities=frozenset({Capability.CAMPAIGN_READ}),
            expires_at=now + timedelta(hours=1),
        ),
        snapshot_at=now,
    )


def test_snapshot_reads_campaign_cursor_and_summaries_in_one_transaction(repository):
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            UPDATE campaign_events SET payload_json = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ('{"raw_event_payload":"must-not-leak"}', "workspace-a", "campaign-1"),
        )
        connection.execute(
            """
            INSERT INTO campaign_proposals(
                workspace_id, campaign_id, proposal_id, status, priority,
                estimated_cost, creation_sequence, proposal_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "proposal-private",
                "submitted",
                50,
                1.0,
                1,
                '{"private_candidate_mapping":"must-not-leak"}',
                "2026-07-16T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id, uri,
                sha256, size_bytes, schema_name, sealed, valid, metadata_json, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "artifact-private",
                "file:///private/operator/path/restricted.json",
                "a" * 64,
                10,
                "restricted-evaluation.v1",
                1,
                1,
                '{"secret":"must-not-leak"}',
                "2026-07-16T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_protected_epochs(
                workspace_id, protected_epoch_id, target_contract_key,
                protected_set_hash, candidate_lock_digest, lease_state,
                access_count, result_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "protected-private",
                "memexai-embedding-v1",
                "b" * 64,
                "c" * 64,
                "consumed",
                1,
                '{"raw_restricted_evaluation":"must-not-leak"}',
                "2026-07-16T00:00:00+00:00",
                "2026-07-16T00:00:00+00:00",
            ),
        )

    repository.trace_snapshot = True
    snapshot = repository.read_control_room_snapshot("workspace-a", "campaign-1")
    repository.trace_snapshot = False

    assert isinstance(snapshot, CampaignControlRoomStateV1)
    assert snapshot.campaign.workspace_id == "workspace-a"
    assert snapshot.campaign.campaign_id == "campaign-1"
    assert snapshot.campaign.version == 1
    assert snapshot.latest_event_cursor == 1
    assert snapshot.proposals.total == 1
    assert snapshot.proposals.by_status[0].status == "submitted"
    assert snapshot.proposals.by_status[0].count == 1
    assert snapshot.artifacts.total == 1
    assert snapshot.artifacts.sealed == 1
    assert snapshot.artifacts.valid == 1
    assert repository.snapshot_connection_count == 1
    assert repository.snapshot_statements[0] == "BEGIN"

    rendered = snapshot.model_dump_json()
    assert "campaign-other" not in rendered
    assert "private/operator/path" not in rendered
    assert "must-not-leak" not in rendered
    assert "raw_event_payload" not in rendered
    assert "protected-private" not in rendered


def test_complete_projection_read_is_atomic_bounded_and_truthful(repository):
    repository.trace_snapshot = True

    projection = repository.read_control_room_projection("workspace-a", "campaign-1")

    repository.trace_snapshot = False
    assert projection.campaign.workspace_id == "workspace-a"
    assert projection.campaign.campaign_id == "campaign-1"
    assert projection.campaign.version == 1
    assert projection.manifest.revision == 1
    assert projection.latest_event_cursor == 1
    assert projection.collection_counts == {
        "events": 1,
        "proposals": 0,
        "studies": 0,
        "attempts": 0,
        "artifacts": 0,
        "comparisons": 0,
        "human_work": 0,
    }
    assert projection.active_studies == ()
    assert projection.active_actions == ()
    assert projection.active_attempts == ()
    assert projection.human_work == ()
    assert projection.agents == ()
    assert repository.snapshot_connection_count == 1
    assert repository.snapshot_statements[0] == "BEGIN"
    normalized_sql = " ".join(repository.snapshot_statements).upper()
    assert normalized_sql.count("LIMIT 2") >= 3


def test_public_projection_exposes_every_safe_field_and_no_future_state(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    principal = ActorPrincipal(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        credential_id="credential-1",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset(Capability),
        authorization_revision=3,
        expires_at=now + timedelta(hours=1),
    )
    controller = ControllerObservationV1(
        controller_observation_version=7,
        state="offline",
        observed_at=now,
        heartbeat_age_seconds=None,
        lease_expires_at=None,
        controller_instance_id=None,
        safe_guidance="Reconcile the resident controller.",
    )
    readiness = ReadinessSummaryV1(
        materializable=False,
        launch_ready=False,
        checked_at=now,
        activation_receipt_digest=None,
        doctor_receipt_digest=None,
        blocking_codes=("campaign_not_ready",),
    )

    snapshot = build_control_room_snapshot(
        durable,
        controller,
        readiness,
        principal=principal,
        snapshot_at=now,
    )

    assert snapshot.workspace_id == "workspace-a"
    assert snapshot.campaign_id == "campaign-1"
    assert snapshot.aggregate_version == 1
    assert snapshot.manifest_revision == 1
    assert snapshot.authorization_revision == 3
    assert snapshot.latest_event_cursor == 1
    assert snapshot.campaign.champion_ref is None
    assert snapshot.bindings.model.binding_id == "memexai-embedding-v1"
    assert snapshot.bindings.model.display_label == "text-retrieval"
    assert snapshot.bindings.model.immutable_digest is not None
    assert snapshot.bindings.data.binding_id == "memexai-approved-training"
    assert snapshot.bindings.data.immutable_digest is None
    assert snapshot.bindings.source.binding_id == "bashgym-source-v1"
    assert snapshot.bindings.compute.binding_id == "ssh-gpu-lab"
    assert tuple(phase.phase_id for phase in snapshot.journey) == (
        "setup",
        "baseline",
        "experiments",
        "human_review",
        "decision",
    )
    assert snapshot.active_work is None
    assert snapshot.champion is None
    assert snapshot.candidate is None
    assert snapshot.metrics == ()
    assert tuple(resource.unit for resource in snapshot.budget.resources) == (
        "gpu_hours",
        "study_count",
    )
    assert snapshot.human_work.blocking_count == 0
    assert snapshot.human_work.newest == ()
    assert snapshot.agents == ()
    assert snapshot.collections.events.count == 1
    assert snapshot.collections.human_work.count == 0
    assert snapshot.decision_surface.blocker.code == "campaign_not_ready"
    assert snapshot.decision_surface.blocker.secondary_codes == ("controller_offline",)
    assert snapshot.decision_surface.attention_owner == "bashgym"
    assert snapshot.decision_surface.promotion_eligible is False
    assert tuple(action.action for action in snapshot.decision_surface.next_actions) == ("cancel",)
    assert snapshot.decision_surface.recovery_actions == ("inspect", "reconcile_controller")

    rendered = snapshot.model_dump_json()
    for forbidden in (
        "base_model_ref",
        "Qwen/Qwen3",
        "repository_path",
        "remote_pid",
        "private_candidate_mapping",
        "raw_restricted_evaluation",
    ):
        assert forbidden not in rendered


def test_snapshot_promotion_eligibility_uses_the_shared_mutation_gate(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "active",
                "best_development_candidate_ref": "candidate-digest",
            }
        ),
        latest_gate={
            "decision_id": "comparison-1",
            "verdict": "passed",
            "candidate_digest": "candidate-digest",
        },
        protected_gate_passed=True,
    )
    principal = ActorPrincipal(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        credential_id="credential-1",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset(Capability),
        authorization_revision=1,
        expires_at=now + timedelta(hours=1),
    )
    controller = ControllerObservationV1(
        controller_observation_version=1,
        state="offline",
        observed_at=now,
        heartbeat_age_seconds=None,
        lease_expires_at=None,
        controller_instance_id=None,
        safe_guidance="Start the resident controller.",
    )
    readiness = ReadinessSummaryV1(
        materializable=True,
        launch_ready=True,
        checked_at=now,
        activation_receipt_digest=None,
        doctor_receipt_digest=None,
        blocking_codes=(),
    )

    mutation_gate = campaign_transitions.evaluate_promotion_gate(
        active_action_id=None,
        comparison_verdict="passed",
        candidate_digest="candidate-digest",
        protected_required=True,
        protected_passed=True,
        human_work_complete=True,
    )
    snapshot = build_control_room_snapshot(
        durable,
        controller,
        readiness,
        principal=principal,
        snapshot_at=now,
    )

    assert mutation_gate.eligible is True
    assert snapshot.decision_surface.promotion_eligible is mutation_gate.eligible
    assert "promote" in {action.action for action in snapshot.decision_surface.next_actions}
    decision = next(phase for phase in snapshot.journey if phase.phase_id == "decision")
    assert decision.state == "ready"
    assert decision.primary_blocker is None
    assert decision.next_action_ids == ("promote",)


def test_snapshot_decision_phase_exposes_shared_promotion_blockers(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "active",
                "best_development_candidate_ref": "candidate-digest",
            }
        ),
        latest_gate={
            "decision_id": "comparison-1",
            "verdict": "failed",
            "candidate_digest": "candidate-digest",
        },
        protected_gate_passed=False,
    )
    principal = ActorPrincipal(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        credential_id="credential-1",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset(Capability),
        authorization_revision=1,
        expires_at=now + timedelta(hours=1),
    )
    controller = ControllerObservationV1(
        controller_observation_version=1,
        state="online",
        observed_at=now,
        heartbeat_age_seconds=0,
        lease_expires_at=now + timedelta(seconds=10),
        controller_instance_id="resident-worker",
        safe_guidance=None,
    )
    readiness = ReadinessSummaryV1(
        materializable=True,
        launch_ready=True,
        checked_at=now,
        activation_receipt_digest=None,
        doctor_receipt_digest=None,
        blocking_codes=(),
    )

    snapshot = build_control_room_snapshot(
        durable,
        controller,
        readiness,
        principal=principal,
        snapshot_at=now,
    )

    decision = next(phase for phase in snapshot.journey if phase.phase_id == "decision")
    assert snapshot.decision_surface.promotion_eligible is False
    assert "promote" not in {action.action for action in snapshot.decision_surface.next_actions}
    assert decision.state == "blocked"
    assert decision.primary_blocker is not None
    assert decision.primary_blocker.code == "campaign_development_gate_not_passed"
    assert "campaign_protected_gate_not_passed" in decision.primary_blocker.secondary_codes
    assert snapshot.decision_surface.attention_owner == "bashgym"
    assert snapshot.decision_surface.blocker is not None
    assert snapshot.decision_surface.blocker.code == "campaign_development_gate_not_passed"
    assert "campaign_protected_gate_not_passed" in (
        snapshot.decision_surface.blocker.secondary_codes
    )

    lifecycle_blocked = replace(
        durable,
        campaign=durable.campaign.model_copy(update={"status": "paused"}),
        latest_gate={
            "decision_id": "comparison-2",
            "verdict": "passed",
            "candidate_digest": "candidate-digest",
        },
        protected_gate_passed=True,
    )
    lifecycle_snapshot = build_control_room_snapshot(
        lifecycle_blocked,
        controller,
        readiness,
        principal=principal,
        snapshot_at=now,
    )
    lifecycle_decision = next(
        phase for phase in lifecycle_snapshot.journey if phase.phase_id == "decision"
    )
    assert lifecycle_decision.state == "blocked"
    assert lifecycle_decision.primary_blocker is not None
    assert lifecycle_decision.primary_blocker.code == "campaign_promotion_transition_unavailable"


def test_override_champion_reports_only_candidate_keyed_actual_evidence(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "completed",
                "best_development_candidate_ref": "candidate-digest",
                "champion_ref": "champion:target:1:candidate-digest",
            }
        ),
        latest_gate={
            "decision_id": "comparison-1",
            "verdict": "failed",
            "candidate_digest": "candidate-digest",
        },
        candidate_provenance=CandidateProvenance(
            candidate_digest="candidate-digest",
            source_attempt_ids=("attempt-owned",),
            source_artifact_ids=("artifact-owned",),
            latest_comparable_evaluation_id="evaluation-for-candidate",
        ),
        champion_provenance=ChampionProvenance(
            candidate_digest="candidate-digest",
            decision_id="comparison-1",
            comparison_verdict="failed",
            override=True,
        ),
    )
    principal = ActorPrincipal(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        credential_id="credential-1",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset(Capability),
        authorization_revision=1,
        expires_at=now + timedelta(hours=1),
    )
    controller = ControllerObservationV1(
        controller_observation_version=1,
        state="online",
        observed_at=now,
        heartbeat_age_seconds=0,
        lease_expires_at=now + timedelta(seconds=10),
        controller_instance_id="resident-worker",
        safe_guidance=None,
    )
    readiness = ReadinessSummaryV1(
        materializable=True,
        launch_ready=True,
        checked_at=now,
        activation_receipt_digest=None,
        doctor_receipt_digest=None,
        blocking_codes=(),
    )

    snapshot = build_control_room_snapshot(
        durable,
        controller,
        readiness,
        principal=principal,
        snapshot_at=now,
    )

    assert snapshot.champion is not None
    assert snapshot.champion.comparison_verdict == "failed"
    assert snapshot.champion.latest_comparable_evaluation_id == "evaluation-for-candidate"
    assert snapshot.champion.source_attempt_ids == ("attempt-owned",)
    assert snapshot.champion.source_artifact_ids == ("artifact-owned",)


def test_durable_champion_claim_is_correlated_to_its_candidate_and_decision(repository):
    candidate_digest = "c" * 64
    target_model = campaign().target_model.model_copy(
        update={"target_contract_key": "registry:provider:model-v1"}
    )
    target_key = target_model.target_contract_key
    champion_ref = f"champion:{target_key}:1:{candidate_digest}"
    now_text = "2026-07-16T12:00:00+00:00"
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO campaign_gate_decisions(
                workspace_id, campaign_id, decision_id, decision_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "comparison-override",
                json.dumps({"candidate_digest": candidate_digest, "verdict": "failed"}),
                now_text,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_champions(
                workspace_id, target_contract_key, revision, champion_json, created_at
            ) VALUES (?, ?, 1, ?, ?)
            """,
            (
                "workspace-a",
                target_key,
                json.dumps(
                    {
                        "schema_version": "campaign_champion.v1",
                        "campaign_id": "campaign-1",
                        "candidate_digest": candidate_digest,
                        "development_decision_id": "comparison-override",
                        "protected_gate_passed": False,
                        "override": True,
                        "override_reason": "Approved exception",
                        "actor_id": "operator",
                        "created_at": now_text,
                    }
                ),
                now_text,
            ),
        )
        connection.execute(
            """
            UPDATE campaigns SET status = 'completed', target_model_json = ?,
                champion_ref = ?, best_development_candidate_ref = ?, updated_at = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (
                target_model.model_dump_json(),
                champion_ref,
                candidate_digest,
                now_text,
                "workspace-a",
                "campaign-1",
            ),
        )

    durable = repository.read_control_room_projection("workspace-a", "campaign-1")

    assert durable.champion_provenance == ChampionProvenance(
        candidate_digest=candidate_digest,
        decision_id="comparison-override",
        comparison_verdict="failed",
        override=True,
    )
    assert "campaign_projection_champion_ref_malformed" not in (durable.projection_invariant_codes)
    snapshot = ready_snapshot(durable)
    assert snapshot.champion is not None
    assert snapshot.champion.comparison_verdict == "failed"


@pytest.mark.parametrize(
    ("decision_json", "expected_invariant"),
    (
        (None, "campaign_projection_champion_decision_missing"),
        (
            json.dumps({"candidate_digest": "d" * 64, "verdict": "passed"}),
            "campaign_projection_champion_decision_mismatch",
        ),
        ("not-json", "campaign_projection_champion_decision_mismatch"),
    ),
)
def test_broken_champion_decision_fails_closed_without_trusted_provenance(
    repository, decision_json, expected_invariant
):
    candidate_digest = "c" * 64
    target_key = campaign().target_model.target_contract_key
    champion_ref = f"champion:{target_key}:1:{candidate_digest}"
    now_text = "2026-07-16T12:00:00+00:00"
    with sqlite3.connect(repository.db_path) as connection:
        if decision_json is not None:
            connection.execute(
                """
                INSERT INTO campaign_gate_decisions(
                    workspace_id, campaign_id, decision_id,
                    decision_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "workspace-a",
                    "campaign-1",
                    "comparison-missing",
                    decision_json,
                    now_text,
                ),
            )
        connection.execute(
            """
            INSERT INTO campaign_champions(
                workspace_id, target_contract_key, revision, champion_json, created_at
            ) VALUES (?, ?, 1, ?, ?)
            """,
            (
                "workspace-a",
                target_key,
                json.dumps(
                    {
                        "schema_version": "campaign_champion.v1",
                        "campaign_id": "campaign-1",
                        "candidate_digest": candidate_digest,
                        "development_decision_id": "comparison-missing",
                        "protected_gate_passed": False,
                        "override": True,
                        "override_reason": "Approved exception",
                        "actor_id": "operator",
                        "created_at": now_text,
                    }
                ),
                now_text,
            ),
        )
        connection.execute(
            """
            UPDATE campaigns SET status = 'completed', champion_ref = ?,
                best_development_candidate_ref = ?, updated_at = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (
                champion_ref,
                candidate_digest,
                now_text,
                "workspace-a",
                "campaign-1",
            ),
        )

    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    snapshot = ready_snapshot(durable)

    assert durable.champion_provenance is None
    assert expected_invariant in durable.projection_invariant_codes
    assert snapshot.champion is not None
    assert snapshot.champion.comparison_verdict is None
    assert snapshot.champion.source_attempt_ids == ()
    assert snapshot.champion.source_artifact_ids == ()
    assert snapshot.decision_surface.blocker is not None
    assert expected_invariant in snapshot.decision_surface.blocker.secondary_codes


def test_valid_running_stage_plan_projects_active_work_without_invariant_failure(repository):
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    proposal = StudyProposal(
        proposal_id="proposal-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        hypothesis="Changing batch size improves held-out retrieval.",
        study_family="embedding-retrieval",
        primary_variable="training.batch_size",
        controlled_variables=("dataset.revision", "evaluation.suite"),
        expected_outcome="Held-out retrieval improves.",
        falsification_criterion="Reject if held-out retrieval regresses.",
        estimated_cost=1.0,
        dataset_recipe={"schema_version": "recipe.v1"},
        training_recipe={"schema_version": "recipe.v1"},
        evaluation_recipe={"schema_version": "recipe.v1"},
        stage_plan=StagePlan(
            items=(
                StagePlanItem(
                    stage=StageKind.SMOKE_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="Bounded smoke check",
                ),
            )
        ),
        planner_actor_id="planner",
        rationale="Test one controlled training change.",
        creation_sequence=1,
        created_at=now,
    )
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO campaign_proposals(
                workspace_id, campaign_id, proposal_id, status, priority,
                estimated_cost, creation_sequence, proposal_json, created_at
            ) VALUES (?, ?, ?, 'accepted', 50, 1.0, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                proposal.proposal_id,
                proposal.model_dump_json(),
                now.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_studies(
                workspace_id, campaign_id, study_id, proposal_id, status,
                current_stage_index, stage_plan_json, candidate_digest,
                version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'smoke_training', 0, ?, ?, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "study-1",
                proposal.proposal_id,
                proposal.stage_plan.model_dump_json(),
                "a" * 64,
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.execute(
            """
            UPDATE campaigns SET status = 'active', active_study_id = ?, active_action_id = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ("study-1", "action-1", "workspace-a", "campaign-1"),
        )
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    durable = replace(
        durable,
        active_actions=(
            {
                "action_id": "action-1",
                "study_id": "study-1",
                "stage_index": 0,
                "stage_kind": "smoke_training",
                "status": "running",
                "candidate_digest": "a" * 64,
            },
        ),
        active_attempts=(
            {
                "attempt_id": "attempt-1",
                "action_id": "action-1",
                "study_id": "study-1",
                "status": "running",
                "executor_json": '{"kind":"fake"}',
                "stage_kind": "smoke_training",
                "candidate_digest": "a" * 64,
            },
        ),
    )
    principal = ActorPrincipal(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        credential_id="credential-1",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset(Capability),
        authorization_revision=1,
        expires_at=now + timedelta(hours=1),
    )
    snapshot = build_control_room_snapshot(
        durable,
        ControllerObservationV1(
            controller_observation_version=1,
            state="online",
            observed_at=now,
            heartbeat_age_seconds=0,
            lease_expires_at=now + timedelta(seconds=10),
            controller_instance_id="resident-worker",
            safe_guidance=None,
        ),
        ReadinessSummaryV1(
            materializable=True,
            launch_ready=True,
            checked_at=now,
            activation_receipt_digest=None,
            doctor_receipt_digest=None,
            blocking_codes=(),
        ),
        principal=principal,
        snapshot_at=now,
    )

    assert snapshot.decision_surface.blocker is None
    assert snapshot.active_work.study_id == "study-1"
    assert snapshot.active_work.attempt_id == "attempt-1"
    assert snapshot.active_work.executor_type == "fake"
    assert snapshot.active_work.hypothesis_summary == proposal.hypothesis
    assert snapshot.active_work.primary_variable_summary == proposal.primary_variable
    assert snapshot.active_work.controlled_variable_summary == proposal.controlled_variables
    assert snapshot.active_work.progress_fraction is None
    assert snapshot.active_work.eta_seconds is None


def test_candidate_projection_never_copies_unproven_outcome_references(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    candidate_digest = "c" * 64
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "active",
                "best_development_candidate_ref": candidate_digest,
            }
        ),
        latest_gate={
            "decision_id": "comparison-selected",
            "verdict": "failed",
            "candidate_digest": candidate_digest,
        },
        candidate_provenance=CandidateProvenance(
            candidate_digest="d" * 64,
            source_attempt_ids=("attempt-from-unrelated-result",),
            source_artifact_ids=("private_candidate_mapping",),
            latest_comparable_evaluation_id="evaluation-for-unrelated-candidate",
        ),
    )
    principal = ActorPrincipal(
        actor_id="reader",
        autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
        credential_id="reader-credential",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset({Capability.CAMPAIGN_READ}),
        expires_at=now + timedelta(hours=1),
    )

    snapshot = build_control_room_snapshot(
        durable,
        ControllerObservationV1(
            controller_observation_version=1,
            state="online",
            observed_at=now,
            heartbeat_age_seconds=0,
            lease_expires_at=now + timedelta(seconds=10),
            controller_instance_id="resident-worker",
            safe_guidance=None,
        ),
        ReadinessSummaryV1(
            materializable=True,
            launch_ready=True,
            checked_at=now,
            activation_receipt_digest=None,
            doctor_receipt_digest=None,
            blocking_codes=(),
        ),
        principal=principal,
        snapshot_at=now,
    )

    assert snapshot.candidate is not None
    assert snapshot.candidate.candidate_ref == candidate_digest
    assert snapshot.candidate.source_attempt_ids == ()
    assert snapshot.candidate.source_artifact_ids == ()
    assert snapshot.candidate.latest_comparable_evaluation_id is None
    rendered = snapshot.model_dump_json()
    assert "attempt-from-unrelated-result" not in rendered
    assert "private_candidate_mapping" not in rendered
    assert "evaluation-for-unrelated-candidate" not in rendered


def test_candidate_gate_mismatch_never_displays_the_other_candidate_verdict(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    candidate_digest = "c" * 64
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "active",
                "best_development_candidate_ref": candidate_digest,
            }
        ),
        latest_gate={
            "decision_id": "comparison-other",
            "verdict": "passed",
            "candidate_digest": "d" * 64,
        },
        projection_invariant_codes=("campaign_projection_candidate_identity_mismatch",),
    )
    snapshot = build_control_room_snapshot(
        durable,
        ControllerObservationV1(
            controller_observation_version=1,
            state="online",
            observed_at=now,
            heartbeat_age_seconds=0,
            lease_expires_at=now + timedelta(seconds=10),
            controller_instance_id="resident-worker",
            safe_guidance=None,
        ),
        ReadinessSummaryV1(
            materializable=True,
            launch_ready=True,
            checked_at=now,
            activation_receipt_digest=None,
            doctor_receipt_digest=None,
            blocking_codes=(),
        ),
        principal=ActorPrincipal(
            actor_id="reader",
            autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
            credential_id="reader-credential",
            credential_kind=CredentialKind.ACCESS,
            workspace_ids=("workspace-a",),
            capabilities=frozenset({Capability.CAMPAIGN_READ}),
            expires_at=now + timedelta(hours=1),
        ),
        snapshot_at=now,
    )

    assert snapshot.candidate is not None
    assert snapshot.candidate.candidate_ref == candidate_digest
    assert snapshot.candidate.comparison_verdict is None
    assert snapshot.candidate.gate_state == "blocked"
    assert snapshot.decision_surface.promotion_eligible is False


def test_durable_read_correlates_candidate_provenance_to_owned_rows(repository):
    AutoResearchRepository(repository.db_path).initialize()
    selected_digest = "c" * 64
    unrelated_digest = "d" * 64
    stage_plan = json.dumps(
        {
            "schema_version": "campaign_stage_plan.v1",
            "items": [
                {
                    "schema_version": "campaign_stage_plan_item.v1",
                    "stage": "development_evaluation",
                    "disposition": "required",
                    "reason": "Candidate comparison",
                    "input_contract": {},
                    "output_contract": {},
                }
            ],
        }
    )
    with sqlite3.connect(repository.db_path) as connection:
        for suffix, digest, sequence in (
            ("selected", selected_digest, 1),
            ("unrelated", unrelated_digest, 2),
        ):
            connection.execute(
                """
                INSERT INTO campaign_proposals(
                    workspace_id, campaign_id, proposal_id, status, priority,
                    estimated_cost, creation_sequence, proposal_json, created_at
                ) VALUES (?, ?, ?, 'accepted', 50, 0, ?, '{}', ?)
                """,
                (
                    "workspace-a",
                    "campaign-1",
                    f"proposal-{suffix}",
                    sequence,
                    f"2026-07-16T00:00:0{sequence}+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_studies(
                    workspace_id, campaign_id, study_id, proposal_id, status,
                    current_stage_index, stage_plan_json, candidate_digest,
                    version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'completed', 0, ?, ?, 1, ?, ?)
                """,
                (
                    "workspace-a",
                    "campaign-1",
                    f"study-{suffix}",
                    f"proposal-{suffix}",
                    stage_plan,
                    digest,
                    f"2026-07-16T00:00:0{sequence}+00:00",
                    f"2026-07-16T00:00:0{sequence}+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_actions(
                    workspace_id, campaign_id, study_id, action_id, stage_index,
                    stage_kind, input_digest, status, candidate_digest,
                    manifest_revision, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, 'development_evaluation', ?, 'completed', ?, 1, 1, ?, ?)
                """,
                (
                    "workspace-a",
                    "campaign-1",
                    f"study-{suffix}",
                    f"action-{suffix}",
                    "a" * 64,
                    digest,
                    f"2026-07-16T00:00:0{sequence}+00:00",
                    f"2026-07-16T00:00:0{sequence}+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_attempts(
                    workspace_id, action_id, attempt_id, attempt_number,
                    claim_generation, status, executor_json, created_at, updated_at
                ) VALUES (?, ?, ?, 1, 1, 'completed', '{}', ?, ?)
                """,
                (
                    "workspace-a",
                    f"action-{suffix}",
                    f"attempt-{suffix}",
                    f"2026-07-16T00:00:0{sequence}+00:00",
                    f"2026-07-16T00:00:0{sequence}+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_artifacts(
                    workspace_id, campaign_id, artifact_id, producer_action_id,
                    uri, sha256, size_bytes, schema_name, sealed, valid,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 'evaluation.v1', 1, 1, '{}', ?)
                """,
                (
                    "workspace-a",
                    "campaign-1",
                    f"artifact-{suffix}",
                    f"action-{suffix}",
                    f"file:///private/{suffix}",
                    ("e" if suffix == "selected" else "f") * 64,
                    f"2026-07-16T00:00:0{sequence}+00:00",
                ),
            )
        selected_result = {
            "study_id": "study-selected",
            "attempt_ids": [
                "attempt-selected",
                "attempt-unrelated",
                "attempt-unknown",
            ],
            "evidence_references": [
                "artifact-selected",
                "artifact-unrelated",
                "private_candidate_mapping",
            ],
        }
        connection.execute(
            """
            INSERT INTO autoresearch_results(
                workspace_id, campaign_id, result_id, proposal_id, result_json,
                result_digest, decision_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "result-selected",
                "proposal-selected",
                json.dumps(selected_result),
                "1" * 64,
                json.dumps({"eligible_for_best": True, "decision": "keep"}),
                "2026-07-16T00:00:03+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO autoresearch_results(
                workspace_id, campaign_id, result_id, proposal_id, result_json,
                result_digest, decision_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "result-unrelated-later",
                "proposal-unrelated",
                json.dumps(
                    {
                        "study_id": "study-unrelated",
                        "attempt_ids": ["attempt-unrelated"],
                        "evidence_references": ["artifact-unrelated"],
                    }
                ),
                "2" * 64,
                json.dumps({"eligible_for_best": False, "decision": "discard"}),
                "2026-07-16T00:00:04+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_gate_decisions(
                workspace_id, campaign_id, decision_id, decision_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "comparison-selected",
                json.dumps({"candidate_digest": selected_digest, "verdict": "failed"}),
                "2026-07-16T00:00:05+00:00",
            ),
        )
        for evaluation_id, digest, created_at in (
            ("evaluation-selected", selected_digest, "2026-07-16T00:00:05+00:00"),
            ("evaluation-unrelated-later", unrelated_digest, "2026-07-16T00:00:06+00:00"),
        ):
            connection.execute(
                """
                INSERT INTO campaign_evaluations(
                    workspace_id, campaign_id, evaluation_id, evaluation_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "workspace-a",
                    "campaign-1",
                    evaluation_id,
                    json.dumps({"candidate_digest": digest}),
                    created_at,
                ),
            )
        connection.execute(
            """
            UPDATE campaigns SET status = 'active', best_development_candidate_ref = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (selected_digest, "workspace-a", "campaign-1"),
        )

    projection = repository.read_control_room_projection("workspace-a", "campaign-1")

    assert projection.candidate_provenance is not None
    assert projection.candidate_provenance.candidate_digest == selected_digest
    assert projection.candidate_provenance.source_attempt_ids == ("attempt-selected",)
    assert projection.candidate_provenance.source_artifact_ids == ("artifact-selected",)
    assert projection.candidate_provenance.latest_comparable_evaluation_id == "evaluation-selected"


def test_oversized_legacy_projection_is_bounded_and_fails_closed(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    candidate_digest = "c" * 64
    oversized_manifest = durable.manifest.manifest.model_copy(
        update={"budget_limits": {f"unit_{index:03d}": 1.0 for index in range(65)}}
    )
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "active",
                "best_development_candidate_ref": candidate_digest,
            }
        ),
        manifest=durable.manifest.model_copy(update={"manifest": oversized_manifest}),
        latest_gate={
            "decision_id": "comparison-1",
            "verdict": "passed",
            "candidate_digest": candidate_digest,
        },
        candidate_provenance=CandidateProvenance(
            candidate_digest=candidate_digest,
            source_attempt_ids=tuple(f"attempt-{index:03d}" for index in range(101)),
            source_artifact_ids=tuple(f"artifact-{index:03d}" for index in range(101)),
            latest_comparable_evaluation_id="evaluation-1",
        ),
        projection_invariant_codes=(
            "campaign_projection_budget_resources_exceeded",
            "campaign_projection_candidate_references_exceeded",
        ),
    )
    principal = ActorPrincipal(
        actor_id="reader",
        autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
        credential_id="reader-credential",
        credential_kind=CredentialKind.ACCESS,
        workspace_ids=("workspace-a",),
        capabilities=frozenset({Capability.CAMPAIGN_READ}),
        expires_at=now + timedelta(hours=1),
    )

    snapshot = build_control_room_snapshot(
        durable,
        ControllerObservationV1(
            controller_observation_version=1,
            state="online",
            observed_at=now,
            heartbeat_age_seconds=0,
            lease_expires_at=now + timedelta(seconds=10),
            controller_instance_id="resident-worker",
            safe_guidance=None,
        ),
        ReadinessSummaryV1(
            materializable=True,
            launch_ready=True,
            checked_at=now,
            activation_receipt_digest=None,
            doctor_receipt_digest=None,
            blocking_codes=(),
        ),
        principal=principal,
        snapshot_at=now,
    )

    assert len(snapshot.budget.resources) == 64
    assert snapshot.candidate is not None
    assert len(snapshot.candidate.source_attempt_ids) == 100
    assert len(snapshot.candidate.source_artifact_ids) == 100
    assert snapshot.decision_surface.blocker.code == "campaign_projection_invariant_failed"
    assert snapshot.decision_surface.blocker.secondary_codes[:2] == (
        "campaign_projection_budget_resources_exceeded",
        "campaign_projection_candidate_references_exceeded",
    )
    assert snapshot.decision_surface.attention_owner == "bashgym"
    assert snapshot.decision_surface.promotion_eligible is False
    assert snapshot.decision_surface.next_actions == ()
    decision = next(phase for phase in snapshot.journey if phase.phase_id == "decision")
    assert decision.state == "blocked"
    assert decision.attention_owner == "bashgym"
    assert decision.primary_blocker == snapshot.decision_surface.blocker
    assert set(snapshot.decision_surface.recovery_actions) <= {
        "inspect",
        "reconcile_controller",
        "reconcile_attempt",
    }


@pytest.mark.parametrize(
    "invariant_code",
    (
        "campaign_projection_budget_resources_exceeded",
        "campaign_projection_candidate_artifacts_malformed",
        "campaign_projection_candidate_attempts_malformed",
        "campaign_projection_candidate_identity_mismatch",
        "campaign_projection_candidate_outcome_ambiguous",
        "campaign_projection_candidate_owned_rows_exceeded",
        "campaign_projection_candidate_references_exceeded",
        "campaign_projection_gate_decision_malformed",
        "campaign_projection_champion_claim_mismatch",
        "campaign_projection_champion_claim_missing",
        "campaign_projection_champion_decision_mismatch",
        "campaign_projection_champion_decision_missing",
        "campaign_projection_champion_ref_malformed",
    ),
)
def test_each_projection_invariant_blocks_a_passed_promotion_gate(repository, invariant_code):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    candidate_digest = "c" * 64
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "active",
                "best_development_candidate_ref": candidate_digest,
            }
        ),
        latest_gate={
            "decision_id": "comparison-1",
            "verdict": "passed",
            "candidate_digest": candidate_digest,
        },
        protected_gate_passed=True,
        projection_invariant_codes=(invariant_code,),
    )
    snapshot = build_control_room_snapshot(
        durable,
        ControllerObservationV1(
            controller_observation_version=1,
            state="online",
            observed_at=now,
            heartbeat_age_seconds=0,
            lease_expires_at=now + timedelta(seconds=10),
            controller_instance_id="resident-worker",
            safe_guidance=None,
        ),
        ReadinessSummaryV1(
            materializable=True,
            launch_ready=True,
            checked_at=now,
            activation_receipt_digest=None,
            doctor_receipt_digest=None,
            blocking_codes=(),
        ),
        principal=ActorPrincipal(
            actor_id="reader",
            autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
            credential_id="reader-credential",
            credential_kind=CredentialKind.ACCESS,
            workspace_ids=("workspace-a",),
            capabilities=frozenset({Capability.CAMPAIGN_READ}),
            expires_at=now + timedelta(hours=1),
        ),
        snapshot_at=now,
    )

    decision = next(phase for phase in snapshot.journey if phase.phase_id == "decision")
    assert snapshot.decision_surface.promotion_eligible is False
    assert snapshot.decision_surface.next_actions == ()
    assert snapshot.decision_surface.attention_owner == "bashgym"
    assert snapshot.decision_surface.blocker is not None
    assert snapshot.decision_surface.blocker.code == "campaign_projection_invariant_failed"
    assert invariant_code in snapshot.decision_surface.blocker.secondary_codes
    assert decision.state == "blocked"
    assert decision.attention_owner == "bashgym"
    assert decision.primary_blocker == snapshot.decision_surface.blocker


def test_new_campaign_rejects_more_budget_resources_than_public_contract(repository):
    value = campaign(workspace_id="workspace-a", campaign_id="campaign-oversized")
    oversized_manifest = revision(value).manifest.model_copy(
        update={"budget_limits": {f"unit_{index:03d}": 1.0 for index in range(65)}}
    )

    with pytest.raises(
        CampaignPersistenceError,
        match="campaign_budget_resource_limit_exceeded",
    ):
        repository.create_campaign(
            value,
            revision(value).model_copy(update={"manifest": oversized_manifest}),
            actor_id="codex-agent",
            credential_kind=CredentialKind.ACCESS,
            correlation_id="oversized-budget",
            idempotency_key="oversized-budget",
        )


def test_snapshot_workspace_scope_fails_closed(repository):
    with pytest.raises(RecordNotFoundError):
        repository.read_control_room_projection("workspace-b", "campaign-1")


def test_projection_keeps_one_sqlite_snapshot_when_writer_commits_mid_read(repository, monkeypatch):
    original_table_exists = control_room_module._table_exists
    writer_committed = False

    def commit_writer_after_initial_projection_reads(connection, table):
        nonlocal writer_committed
        if not writer_committed:
            with sqlite3.connect(repository.db_path) as writer:
                writer.execute(
                    """
                    UPDATE campaigns SET version = 2, updated_at = ?
                    WHERE workspace_id = ? AND campaign_id = ?
                    """,
                    ("2026-07-16T12:00:00+00:00", "workspace-a", "campaign-1"),
                )
                writer.execute(
                    """
                    INSERT INTO campaign_events(
                        event_id, workspace_id, campaign_id, sequence,
                        aggregate_version, event_type, payload_json, actor_id,
                        credential_kind, correlation_id, idempotency_key, created_at
                    ) VALUES (?, ?, ?, 2, 2, ?, '{}', ?, ?, ?, ?, ?)
                    """,
                    (
                        "event-concurrent-writer",
                        "workspace-a",
                        "campaign-1",
                        "campaign:concurrent-test",
                        "writer",
                        "controller",
                        "concurrent-writer",
                        "concurrent-writer",
                        "2026-07-16T12:00:00+00:00",
                    ),
                )
            writer_committed = True
        return original_table_exists(connection, table)

    monkeypatch.setattr(
        control_room_module,
        "_table_exists",
        commit_writer_after_initial_projection_reads,
    )

    projection = repository.read_control_room_projection("workspace-a", "campaign-1")

    assert writer_committed is True
    assert projection.campaign.version == 1
    assert projection.latest_event_cursor == 1
    assert projection.collection_counts["events"] == 1
    assert repository.get_campaign("workspace-a", "campaign-1").version == 2


def test_unknown_remote_identity_is_opaque_and_requires_reconciliation(repository):
    now_text = "2026-07-16T12:00:00+00:00"
    stage_plan = json.dumps(
        {
            "schema_version": "campaign_stage_plan.v1",
            "items": [
                {
                    "schema_version": "campaign_stage_plan_item.v1",
                    "stage": "smoke_training",
                    "disposition": "required",
                    "reason": "Bounded smoke check",
                    "input_contract": {},
                    "output_contract": {},
                }
            ],
        }
    )
    identity = RemoteRunIdentity(
        compute_profile_id="ssh-gpu-lab",
        run_id="remote-run-1",
        remote_run_directory="/private/remote/run-1",
        remote_pid=4242,
        process_group_id=4242,
        process_start_ticks=9001,
        boot_id="private-boot-id",
        command_hash="a" * 64,
        launch_manifest_sha256="d" * 64,
        launched_at=datetime(2026, 7, 16, 12, 0, tzinfo=UTC),
    )
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO campaign_proposals(
                workspace_id, campaign_id, proposal_id, status, priority,
                estimated_cost, creation_sequence, proposal_json, created_at
            ) VALUES (?, ?, ?, 'accepted', 50, 0, 1, '{}', ?)
            """,
            ("workspace-a", "campaign-1", "proposal-remote", now_text),
        )
        connection.execute(
            """
            INSERT INTO campaign_studies(
                workspace_id, campaign_id, study_id, proposal_id, status,
                current_stage_index, stage_plan_json, candidate_digest,
                version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'smoke_training', 0, ?, ?, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "study-remote",
                "proposal-remote",
                stage_plan,
                "b" * 64,
                now_text,
                now_text,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_actions(
                workspace_id, campaign_id, study_id, action_id, stage_index,
                stage_kind, input_digest, status, candidate_digest,
                manifest_revision, version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, 'smoke_training', ?, 'running', ?, 1, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "study-remote",
                "action-remote",
                "c" * 64,
                "b" * 64,
                now_text,
                now_text,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_attempts(
                workspace_id, action_id, attempt_id, attempt_number,
                claim_generation, status, executor_json, created_at, updated_at
            ) VALUES (?, ?, ?, 1, 1, 'unknown', ?, ?, ?)
            """,
            (
                "workspace-a",
                "action-remote",
                "attempt-remote",
                json.dumps({"kind": "ssh_remote"}),
                now_text,
                now_text,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_remote_runs(
                workspace_id, attempt_id, claim_generation, identity_json, state,
                metric_cursor_json, log_cursor_json, created_at, updated_at
            ) VALUES (?, ?, 1, ?, 'unknown', '{}', '{}', ?, ?)
            """,
            (
                "workspace-a",
                "attempt-remote",
                identity.model_dump_json(),
                now_text,
                now_text,
            ),
        )
        connection.execute(
            """
            UPDATE campaigns SET status = 'active', active_study_id = ?,
                active_action_id = ?, updated_at = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (
                "study-remote",
                "action-remote",
                now_text,
                "workspace-a",
                "campaign-1",
            ),
        )

    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    snapshot = build_control_room_snapshot(
        durable,
        ControllerObservationV1(
            controller_observation_version=1,
            state="online",
            observed_at=now,
            heartbeat_age_seconds=0,
            lease_expires_at=now + timedelta(seconds=10),
            controller_instance_id="resident-worker",
            safe_guidance=None,
        ),
        ReadinessSummaryV1(
            materializable=True,
            launch_ready=True,
            checked_at=now,
            activation_receipt_digest=None,
            doctor_receipt_digest=None,
            blocking_codes=(),
        ),
        principal=ActorPrincipal(
            actor_id="reader",
            autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
            credential_id="reader-credential",
            credential_kind=CredentialKind.ACCESS,
            workspace_ids=("workspace-a",),
            capabilities=frozenset({Capability.CAMPAIGN_READ}),
            expires_at=now + timedelta(hours=1),
        ),
        snapshot_at=now,
    )

    assert snapshot.active_work is not None
    assert snapshot.active_work.executor_type == "ssh_remote"
    assert snapshot.active_work.process_identity is not None
    assert snapshot.active_work.process_identity.run_id == "remote-run-1"
    assert snapshot.active_work.process_identity.compute_profile_id == "ssh-gpu-lab"
    assert snapshot.active_work.process_identity.state == "unknown"
    assert snapshot.decision_surface.blocker is not None
    assert snapshot.decision_surface.blocker.code == "campaign_process_identity_unknown"
    assert "reconcile_attempt" in snapshot.decision_surface.recovery_actions
    rendered = snapshot.model_dump_json()
    assert "private-boot-id" not in rendered
    assert "/private/remote/run-1" not in rendered
    assert "4242" not in rendered
