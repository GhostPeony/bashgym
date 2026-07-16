"""Atomic, bounded control-room snapshot repository tests."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from bashgym.campaigns import transitions as campaign_transitions
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    CampaignControlRoomStateV1,
    Capability,
    ControllerObservationV1,
    CredentialKind,
    ReadinessSummaryV1,
)
from bashgym.campaigns.control_room import build_control_room_snapshot
from bashgym.campaigns.persistence import CampaignRepository, RecordNotFoundError
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
    assert "promote" in {
        action.action for action in snapshot.decision_surface.next_actions
    }


def test_valid_running_stage_plan_projects_active_work_without_invariant_failure(repository):
    durable = repository.read_control_room_projection("workspace-a", "campaign-1")
    now = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    durable = replace(
        durable,
        campaign=durable.campaign.model_copy(
            update={
                "status": "active",
                "active_study_id": "study-1",
                "active_action_id": "action-1",
            }
        ),
        active_studies=(
            {
                "study_id": "study-1",
                "proposal_id": "proposal-1",
                "status": "smoke_training",
                "current_stage_index": 0,
                "stage_plan_json": json.dumps(
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
                ),
            },
        ),
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
                "executor_json": '{"executor_kind":"fake"}',
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
    assert snapshot.active_work.progress_fraction is None
    assert snapshot.active_work.eta_seconds is None


def test_snapshot_workspace_scope_fails_closed(repository):
    with pytest.raises(RecordNotFoundError):
        repository.read_control_room_snapshot("workspace-b", "campaign-1")
