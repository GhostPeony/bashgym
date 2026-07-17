"""Campaign migrations, workspace ownership, CAS, event, and replay tests."""

import hashlib
import sqlite3

import pytest

from bashgym.campaigns.contracts import (
    Campaign,
    CampaignKind,
    CampaignManifest,
    CampaignStatus,
    CampaignTrigger,
    CredentialKind,
    ManifestRevision,
    TargetModelContract,
)
from bashgym.campaigns.persistence import (
    MIGRATIONS,
    CampaignRepository,
    IdempotencyConflictError,
    RecordNotFoundError,
    RevisionConflictError,
)
from bashgym.campaigns.transitions import InvalidCampaignTransitionError


def campaign(workspace_id: str = "workspace-a", campaign_id: str = "campaign-1") -> Campaign:
    return Campaign(
        campaign_id=campaign_id,
        workspace_id=workspace_id,
        title="MemexAI embedding campaign",
        kind=CampaignKind.EMBEDDING_RETRIEVAL,
        objective="Improve development retrieval without regressing the frozen champion.",
        target_model=TargetModelContract(
            target_contract_key="memexai-embedding-v1",
            base_model_ref="Qwen/Qwen3-Embedding-0.6B",
            task="text-retrieval",
            representation_contract={"pooling": "last-token", "normalize": True},
        ),
        owner_actor_id="codex-agent",
    )


def manifest() -> CampaignManifest:
    return CampaignManifest(
        approved_data_scopes=("memexai-approved-training",),
        compute_profile_id="ssh-gpu-lab",
        budget_limits={"gpu_hours": 12.0, "study_count": 5.0},
        evaluation_plan={
            "development_query_set": "dev-18-v1",
            "source_repository_binding_id": "bashgym-source-v1",
        },
        promotion_gates={"mrr_at_10_delta_min": 0.0},
        protected_artifact_refs=("frozen-test-36-v1",),
    )


def revision(value: Campaign) -> ManifestRevision:
    return ManifestRevision(
        workspace_id=value.workspace_id,
        campaign_id=value.campaign_id,
        revision=1,
        manifest=manifest(),
        actor_id="codex-agent",
        correlation_id="correlation-create",
    )


@pytest.fixture
def repository(tmp_path):
    value = CampaignRepository(tmp_path / "state" / "campaigns.sqlite3")
    value.initialize()
    return value


def create(repository: CampaignRepository, value: Campaign | None = None):
    value = value or campaign()
    return repository.create_campaign(
        value,
        revision(value),
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="correlation-create",
        idempotency_key=f"create-{value.campaign_id}",
    )


def transition(repository, trigger, version, *, key, payload=None):
    return repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        trigger,
        expected_version=version,
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id=f"correlation-{key}",
        idempotency_key=key,
        payload=payload,
    )


def test_initialize_applies_checksum_migration_and_all_owned_tables(repository):
    assert repository.schema_versions() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert repository.journal_mode() == "wal"
    assert repository.foreign_keys_enabled() is True

    with sqlite3.connect(repository.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'campaign_%'"
            )
        }
    assert {
        "campaign_manifest_revisions",
        "campaign_proposals",
        "campaign_studies",
        "campaign_actions",
        "campaign_attempts",
        "campaign_artifacts",
        "campaign_evaluations",
        "campaign_gate_decisions",
        "campaign_champions",
        "campaign_budget_ledger",
        "campaign_events",
        "campaign_actor_credentials",
        "campaign_access_tokens",
        "campaign_protected_epochs",
        "campaign_scheduler_leases",
        "campaign_mutations",
        "campaign_source_approvals",
        "campaign_action_control_requests",
        "campaign_exports",
        "campaign_code_lineages",
        "campaign_human_work",
        "campaign_human_receipts",
        "campaign_human_promotions",
        "campaign_human_mutations",
    } <= tables


def test_v9_database_migrates_existing_auth_and_controller_rows_to_revision_one(tmp_path):
    path = tmp_path / "campaigns-v9.sqlite3"
    applied_at = "2026-07-16T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE campaign_schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        for version, name, statements in MIGRATIONS[:-1]:
            for statement in statements:
                connection.execute(statement)
            checksum = hashlib.sha256("\n".join(statements).encode("utf-8")).hexdigest()
            connection.execute(
                """
                INSERT INTO campaign_schema_migrations(version, name, checksum, applied_at)
                VALUES (?, ?, ?, ?)
                """,
                (version, name, checksum, applied_at),
            )
        connection.execute(
            """
            INSERT INTO campaign_actor_credentials(
                credential_id, actor_id, autonomy_profile, credential_kind,
                workspace_ids_json, token_salt, token_hash, issued_at,
                expires_at, token_not_before, revoked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                "cred-legacy",
                "codex-agent",
                "codex_trusted",
                "refresh",
                '["workspace-a"]',
                "salt",
                "hash",
                applied_at,
                "2026-08-16T00:00:00+00:00",
                applied_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_scheduler_leases(
                lease_key, owner_id, generation, expires_at, heartbeat_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "scheduler:legacy",
                "worker-a",
                1,
                "2026-07-16T00:00:15+00:00",
                applied_at,
            ),
        )

    migrated = CampaignRepository(path)
    migrated.initialize()

    credential = migrated.get_actor_credential("cred-legacy")
    lease = migrated.get_lease("scheduler:legacy")
    assert migrated.schema_versions()[-1] == 15
    assert credential is not None and credential.authorization_revision == 1
    assert lease is not None and lease.controller_observation_version == 1


def test_create_is_atomic_manifest_is_immutable_and_replay_is_exact(repository):
    value = campaign()
    first = create(repository, value)
    replay = create(repository, value)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.campaign == first.campaign
    assert replay.event == first.event
    saved_manifest = repository.get_manifest_revision("workspace-a", "campaign-1", 1)
    assert saved_manifest.manifest == manifest()
    assert saved_manifest.manifest_hash == revision(value).manifest_hash


def test_idempotency_key_with_different_body_conflicts(repository):
    create(repository)
    first = transition(
        repository,
        CampaignTrigger.VALIDATE,
        1,
        key="validate-1",
        payload={"source": "controller"},
    )
    replay = transition(
        repository,
        CampaignTrigger.VALIDATE,
        1,
        key="validate-1",
        payload={"source": "controller"},
    )

    assert first.campaign.status == CampaignStatus.VALIDATING
    assert replay.replayed is True
    with pytest.raises(IdempotencyConflictError):
        transition(
            repository,
            CampaignTrigger.VALIDATE,
            1,
            key="validate-1",
            payload={"source": "different"},
        )


def test_transition_cas_event_cursor_and_illegal_transition_are_atomic(repository):
    create(repository)
    validating = transition(repository, CampaignTrigger.VALIDATE, 1, key="validate-1")
    ready = transition(repository, CampaignTrigger.VALIDATION_PASSED, 2, key="ready-1")
    active = transition(repository, CampaignTrigger.START, 3, key="start-1")

    assert [validating.campaign.version, ready.campaign.version, active.campaign.version] == [
        2,
        3,
        4,
    ]
    assert active.campaign.status == CampaignStatus.ACTIVE
    events = repository.list_events("workspace-a", "campaign-1")
    assert [cursor for cursor, _event in events] == sorted(cursor for cursor, _event in events)
    assert [event.event_type for _cursor, event in events] == [
        "campaign:created",
        "campaign:validation-started",
        "campaign:ready",
        "campaign:started",
    ]

    with pytest.raises(RevisionConflictError):
        transition(repository, CampaignTrigger.PAUSE, 3, key="stale-pause")
    with pytest.raises(InvalidCampaignTransitionError):
        transition(repository, CampaignTrigger.RESUME, 4, key="illegal-resume")
    assert repository.get_campaign("workspace-a", "campaign-1").version == 4
    assert len(repository.list_events("workspace-a", "campaign-1")) == 4


def test_cross_workspace_reads_are_indistinguishable_from_missing(repository):
    create(repository)

    with pytest.raises(RecordNotFoundError) as wrong_workspace:
        repository.get_campaign("workspace-b", "campaign-1")
    with pytest.raises(RecordNotFoundError) as missing:
        repository.get_campaign("workspace-a", "missing")

    assert str(wrong_workspace.value) == str(missing.value) == "campaign not found"
    assert repository.list_campaigns("workspace-b") == []
    with pytest.raises(RecordNotFoundError, match="campaign not found"):
        repository.list_events("workspace-b", "campaign-1")


def test_reopen_preserves_campaign_and_idempotent_response(tmp_path):
    path = tmp_path / "campaigns.sqlite3"
    before = CampaignRepository(path)
    before.initialize()
    create(before)

    after = CampaignRepository(path)
    after.initialize()
    replay = create(after)

    assert replay.replayed is True
    assert after.get_campaign("workspace-a", "campaign-1").status == CampaignStatus.DRAFT
