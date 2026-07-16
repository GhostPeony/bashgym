"""Atomic, bounded control-room snapshot repository tests."""

import sqlite3
from contextlib import contextmanager

import pytest

from bashgym.campaigns.contracts import CampaignControlRoomStateV1, CredentialKind
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


def test_snapshot_workspace_scope_fails_closed(repository):
    with pytest.raises(RecordNotFoundError):
        repository.read_control_room_snapshot("workspace-b", "campaign-1")
