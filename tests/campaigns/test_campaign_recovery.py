"""Durable, bounded campaign recovery authority tests."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.campaign_recovery import (
    CampaignRecoveryConflictError,
    CampaignRecoveryRepository,
    RecoveryAction,
    RecoveryRequest,
)
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
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.worker import CampaignWorker, SimulatedWorkerCrashError
from tests.campaigns.test_worker import schedule, seed_validated_study

NOW = datetime(2026, 7, 16, 20, 30, tzinfo=UTC)
INSTALLATION = "ins_" + "1" * 32
LEASE_KEY = "controller:/private/host/must-not-leak"
PRIVATE_OWNER = "operator@private-host-must-not-leak"
PRIVATE_CHECKPOINT = "C:/Users/private/checkpoints/must-not-leak"
PRIVATE_ARTIFACT = "ssh://private-host/home/operator/must-not-leak"
RECOVERY_SEAL_KEY_VERSION = "recovery-worker-test-v1"


def _recovery_sealer() -> ArtifactSealer:
    return ArtifactSealer(b"r" * 32, key_version=RECOVERY_SEAL_KEY_VERSION)


def _campaign(workspace_id: str = "workspace-a", campaign_id: str = "campaign-1") -> Campaign:
    return Campaign(
        campaign_id=campaign_id,
        workspace_id=workspace_id,
        title="Portable recovery campaign",
        kind=CampaignKind.EMBEDDING_RETRIEVAL,
        objective="Recover the exact registered campaign without replacing its bindings.",
        target_model=TargetModelContract(
            target_contract_key="operator-model-binding",
            base_model_ref="operator-selected/model@0123456789abcdef0123456789abcdef01234567",
            task="text-retrieval",
        ),
        owner_actor_id="human-operator",
    )


def _manifest() -> CampaignManifest:
    return CampaignManifest(
        approved_data_scopes=("operator-data-binding",),
        compute_profile_id="registered-private-compute",
        budget_limits={"gpu_hours": 2.0},
        evaluation_plan={
            "dataset_binding_id": "operator-data-binding",
            "evaluation_suite_id": "operator-evaluator-binding",
        },
        promotion_gates={"requires_human": True},
    )


def _create_campaign(repository: CampaignRuntimeRepository, value: Campaign) -> None:
    repository.create_campaign(
        value,
        ManifestRevision(
            workspace_id=value.workspace_id,
            campaign_id=value.campaign_id,
            revision=1,
            manifest=_manifest(),
            actor_id="human-operator",
            correlation_id=f"create-{value.campaign_id}",
        ),
        actor_id="human-operator",
        credential_kind=CredentialKind.ACCESS,
        correlation_id=f"create-{value.campaign_id}",
        idempotency_key=f"create-{value.campaign_id}",
    )


def _insert_artifact(repository: CampaignRuntimeRepository, source_id: str, schema: str) -> None:
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id, uri,
                sha256, size_bytes, schema_name, sealed, valid, metadata_json, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, 1, ?, 1, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                source_id,
                f"file:///private/{source_id}/must-not-leak",
                "a" * 64,
                schema,
                json.dumps({"secret": "must-not-leak", "path": PRIVATE_CHECKPOINT}),
                NOW.isoformat(),
            ),
        )


@pytest.fixture
def repositories(tmp_path):
    campaigns = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    campaigns.initialize()
    _create_campaign(campaigns, _campaign())
    _create_campaign(campaigns, _campaign("workspace-b", "campaign-other"))
    _insert_artifact(campaigns, PRIVATE_CHECKPOINT, "checkpoint.v1")
    _insert_artifact(campaigns, PRIVATE_ARTIFACT, "adapter.v1")
    recovery = CampaignRecoveryRepository(campaigns.db_path, sealer=_recovery_sealer())
    recovery.initialize()
    recovery.register_installation(
        installation_id=INSTALLATION,
        controller_owner_id=PRIVATE_OWNER,
        controller_lease_key=LEASE_KEY,
    )
    for kind, logical_id, label in (
        ("model", "operator-model-binding", "Operator selected model"),
        ("data", "operator-data-binding", None),
        ("evaluator", "operator-evaluator-binding", None),
        ("compute", "registered-private-compute", None),
    ):
        recovery.register_binding(
            installation_id=INSTALLATION,
            kind=kind,
            logical_id=logical_id,
            availability="reachable",
            display_label=label,
            integration_label=None,
        )
    recovery.bind_campaign(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        installation_id=INSTALLATION,
        lineage_mode="clone",
        parent_campaign_id=None,
        checkpoint_source_id=PRIVATE_CHECKPOINT,
        artifact_source_id=PRIVATE_ARTIFACT,
        schema_compatible=True,
        revision_compatible=True,
    )
    return campaigns, recovery


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from _all_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_strings(nested)


def _request(
    snapshot: dict, *, action: RecoveryAction = RecoveryAction.RESUME, key="idem_" + "9" * 32
):
    eligibility = snapshot["eligibility"]
    return RecoveryRequest(
        action=action,
        idempotency_key=key,
        workspace_id=snapshot["workspace_id"],
        campaign_id=snapshot["campaign_id"],
        eligibility_receipt_id=eligibility["receipt_id"],
        doctor_evidence_id=snapshot["doctor"]["evidence_id"],
        expected_campaign_revision=snapshot["lineage"]["campaign_revision"],
        expected_event_cursor=snapshot["lineage"]["event_cursor"],
        expected_aggregate_version=snapshot["lineage"]["aggregate_version"],
        expected_controller_lease_id=snapshot["controller"]["lease_id"],
        checkpoint_id=snapshot["lineage"]["checkpoint_id"],
        artifact_id=snapshot["lineage"]["artifact_id"],
        human_confirmed=True,
    )


def test_projection_is_exact_bounded_and_contains_no_private_authority(repositories):
    _campaigns, recovery = repositories

    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)

    assert snapshot["schema_version"] == "campaign_recovery.v1"
    assert snapshot["installation"] == {
        "installation_id": INSTALLATION,
        "lineage_mode": "clone",
    }
    assert re.fullmatch(r"mdl_[0-9a-f]{32}", snapshot["bindings"]["model_id"])
    assert re.fullmatch(r"ckpt_[0-9a-f]{32}", snapshot["lineage"]["checkpoint_id"])
    assert re.fullmatch(r"art_[0-9a-f]{32}", snapshot["lineage"]["artifact_id"])
    assert snapshot["eligibility"]["allowed_actions"] == ["resume", "repair"]
    assert snapshot["execution_consumer"] == {
        "supported": True,
        "ready": False,
        "reason_code": "controller_unowned",
    }
    assert len(snapshot["receipts"]) <= 32
    assert len(snapshot["consumed_idempotency_keys"]) <= 64
    encoded = json.dumps(snapshot, sort_keys=True)
    for canary in (
        "must-not-leak",
        "private-host",
        "operator@",
        "C:/Users",
        "ssh://",
        "file://",
        "secret",
        "uri",
        "path",
        "credential",
    ):
        assert canary.casefold() not in encoded.casefold()
    assert recovery.verify_receipt(snapshot["eligibility"]["receipt_id"])


def test_projection_is_workspace_isolated_and_missing_clone_bindings_fail_closed(repositories):
    _campaigns, recovery = repositories
    with pytest.raises(CampaignRecoveryConflictError, match="not registered"):
        recovery.project("workspace-b", "campaign-other", now=NOW)

    clone = "ins_" + "2" * 32
    recovery.register_installation(
        installation_id=clone,
        controller_owner_id="clone-private-owner",
        controller_lease_key="clone-private-lease",
    )
    for kind, logical_id in (
        ("model", "operator-model-binding"),
        ("data", "operator-data-binding"),
        ("evaluator", "operator-evaluator-binding"),
    ):
        recovery.register_binding(
            installation_id=clone,
            kind=kind,
            logical_id=logical_id,
            availability="reachable",
        )
    recovery.bind_campaign(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        installation_id=clone,
        lineage_mode="fork",
        parent_campaign_id="parent-campaign",
        checkpoint_source_id=PRIVATE_CHECKPOINT,
        artifact_source_id=PRIVATE_ARTIFACT,
        schema_compatible=True,
        revision_compatible=True,
    )

    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=2))

    assert snapshot["installation"]["lineage_mode"] == "fork"
    assert snapshot["bindings"]["compute_id"] is None
    assert snapshot["compute"]["availability"] == "unknown"
    assert snapshot["eligibility"]["decision"] == "blocked"
    assert snapshot["eligibility"]["allowed_actions"] == []
    assert "clone-private" not in json.dumps(snapshot)


def test_portable_parent_campaign_identity_must_be_public_safe(repositories):
    _campaigns, recovery = repositories

    with pytest.raises(ValueError, match="parent campaign"):
        recovery.bind_campaign(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            installation_id=INSTALLATION,
            lineage_mode="fork",
            parent_campaign_id="ssh://private-host/must-not-leak",
            checkpoint_source_id=PRIVATE_CHECKPOINT,
            artifact_source_id=PRIVATE_ARTIFACT,
            schema_compatible=True,
            revision_compatible=True,
        )


@pytest.mark.parametrize(
    "label",
    [
        "Cade MacBook Pro",
        "sk-proj-AbCdEf123456",
        "github_pat_private_token",
        "Operator laptop",
    ],
)
def test_model_display_label_rejects_personal_devices_and_credentials(repositories, label):
    _campaigns, recovery = repositories

    with pytest.raises(ValueError, match="public-safe"):
        recovery.register_binding(
            installation_id=INSTALLATION,
            kind="model",
            logical_id="operator-model-binding",
            availability="reachable",
            display_label=label,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("eligibility_receipt_id", "rcpt_" + "0" * 32),
        ("doctor_evidence_id", "evd_" + "0" * 32),
        ("expected_campaign_revision", 99),
        ("expected_event_cursor", 99),
        ("expected_aggregate_version", 99),
        ("checkpoint_id", "ckpt_" + "0" * 32),
        ("artifact_id", "art_" + "0" * 32),
    ],
)
def test_recovery_rejects_stale_or_wrong_sealed_authority(repositories, field, replacement):
    _campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot).model_copy(update={field: replacement})

    with pytest.raises(CampaignRecoveryConflictError):
        recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))


def test_resume_is_idempotent_restart_safe_and_never_resets_lineage(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    original_lineage = dict(snapshot["lineage"])

    first, replayed = recovery.request(
        request, actor_id="human-operator", now=NOW + timedelta(seconds=1)
    )
    reopened = CampaignRecoveryRepository(campaigns.db_path, sealer=_recovery_sealer())
    reopened.initialize()
    second, replayed_after_restart = reopened.request(
        request, actor_id="human-operator", now=NOW + timedelta(seconds=2)
    )

    assert replayed is False
    assert replayed_after_restart is True
    assert first == second
    assert first["outcome"] == "accepted"
    assert first["lineage"] == original_lineage
    assert first["approval_receipt"]["kind"] == "recovery"
    assert first["approval_receipt"]["receipt_id"] != first["receipt"]["receipt_id"]
    assert reopened.verify_receipt(first["approval_receipt"]["receipt_id"])
    assert reopened.verify_receipt(first["receipt"]["receipt_id"])
    after = reopened.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=3))
    assert after["lineage"] == original_lineage
    assert request.idempotency_key in after["consumed_idempotency_keys"]
    conflict = request.model_copy(update={"action": RecoveryAction.REPAIR})
    with pytest.raises(CampaignRecoveryConflictError, match="idempotency"):
        reopened.request(conflict, actor_id="human-operator", now=NOW + timedelta(seconds=4))


def test_repair_request_is_sealed_and_idempotent(repositories):
    _campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(
        snapshot,
        action=RecoveryAction.REPAIR,
        key="idem_" + "6" * 32,
    )

    first, replayed = recovery.request(
        request, actor_id="human-operator", now=NOW + timedelta(seconds=1)
    )
    second, replayed_again = recovery.request(
        request, actor_id="human-operator", now=NOW + timedelta(seconds=2)
    )

    assert replayed is False
    assert replayed_again is True
    assert first == second
    assert first["action"] == "repair"
    assert recovery.verify_receipt(first["receipt"]["receipt_id"])


def test_active_foreign_lease_is_denied_and_only_expired_lease_can_be_taken_over(repositories):
    campaigns, recovery = repositories
    campaigns.acquire_lease(LEASE_KEY, "foreign-private-owner", ttl=timedelta(minutes=5), now=NOW)
    blocked = recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=1))
    assert blocked["controller"]["state"] == "foreign_lease"
    assert blocked["eligibility"]["decision"] == "blocked"

    forced = _request(blocked, action=RecoveryAction.TAKEOVER)
    with pytest.raises(CampaignRecoveryConflictError, match="eligible"):
        recovery.request(forced, actor_id="human-operator", now=NOW + timedelta(seconds=2))

    expired_at = NOW + timedelta(minutes=6)
    eligible = recovery.project("workspace-a", "campaign-1", now=expired_at)
    assert eligible["controller"]["state"] == "expired"
    assert eligible["eligibility"]["allowed_actions"] == ["takeover"]
    takeover = _request(
        eligible,
        action=RecoveryAction.TAKEOVER,
        key="idem_" + "8" * 32,
    )
    outcome, replayed = recovery.request(
        takeover, actor_id="human-operator", now=expired_at + timedelta(seconds=1)
    )
    repeated, replayed_again = recovery.request(
        takeover, actor_id="human-operator", now=expired_at + timedelta(seconds=2)
    )
    assert replayed is False
    assert replayed_again is True
    assert repeated == outcome
    assert outcome["action"] == "takeover"
    assert outcome["lineage"] == eligible["lineage"]
    current = campaigns.get_lease(LEASE_KEY)
    assert current is not None
    assert current.owner_id == PRIVATE_OWNER
    assert current.generation == 2


def test_missing_artifacts_and_incompatible_schema_or_revision_are_explicit(repositories):
    campaigns, recovery = repositories
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            "DELETE FROM campaign_artifacts WHERE workspace_id=? AND artifact_id=?",
            ("workspace-a", PRIVATE_CHECKPOINT),
        )
    recovery.bind_campaign(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        installation_id=INSTALLATION,
        lineage_mode="clone",
        parent_campaign_id=None,
        checkpoint_source_id=PRIVATE_CHECKPOINT,
        artifact_source_id=PRIVATE_ARTIFACT,
        schema_compatible=False,
        revision_compatible=False,
    )

    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)

    assert snapshot["artifacts"] == {"checkpoint": "missing", "artifact": "available"}
    assert snapshot["compatibility"] == {
        "schema": "incompatible",
        "revision": "incompatible",
    }
    assert snapshot["eligibility"]["decision"] == "blocked"
    assert snapshot["eligibility"]["allowed_actions"] == []


def test_expired_receipt_and_changed_controller_observation_are_rejected(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project(
        "workspace-a", "campaign-1", now=NOW, eligibility_ttl=timedelta(seconds=5)
    )
    request = _request(snapshot)
    with pytest.raises(CampaignRecoveryConflictError, match="expired"):
        recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=6))

    fresh = recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=7))
    campaigns.acquire_lease(
        LEASE_KEY, PRIVATE_OWNER, ttl=timedelta(minutes=1), now=NOW + timedelta(seconds=8)
    )
    with pytest.raises(CampaignRecoveryConflictError, match="controller"):
        recovery.request(
            _request(fresh, key="idem_" + "7" * 32),
            actor_id="human-operator",
            now=NOW + timedelta(seconds=9),
        )


def test_projection_bounds_receipts_and_consumed_keys(repositories):
    _campaigns, recovery = repositories
    for index in range(70):
        now = NOW + timedelta(seconds=index * 2)
        snapshot = recovery.project("workspace-a", "campaign-1", now=now)
        request = _request(snapshot, key=f"idem_{index:032x}")
        recovery.request(request, actor_id="human-operator", now=now + timedelta(seconds=1))

    bounded = recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=141))
    assert len(bounded["receipts"]) == 32
    assert len(bounded["consumed_idempotency_keys"]) == 64
    assert all(len(value) <= 160 for value in _all_strings(bounded))


def test_projection_fails_closed_when_a_current_receipt_seal_is_tampered(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            "UPDATE campaign_recovery_receipts SET receipt_digest=? WHERE receipt_id=?",
            ("sha256_" + "0" * 64, snapshot["eligibility"]["receipt_id"]),
        )

    with pytest.raises(CampaignRecoveryConflictError, match="seal"):
        recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=1))


def test_receipts_use_external_versioned_sealing_and_legacy_rows_fail_closed(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    assert snapshot["receipts"]

    with sqlite3.connect(campaigns.db_path) as connection:
        versions = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT seal_key_version FROM campaign_recovery_receipts"
            ).fetchall()
        }
        colocated_key = connection.execute(
            "SELECT value FROM campaign_recovery_meta WHERE key='seal_key'"
        ).fetchone()
    assert versions == {RECOVERY_SEAL_KEY_VERSION}
    assert colocated_key is None

    wrong_authority = CampaignRecoveryRepository(
        campaigns.db_path,
        sealer=ArtifactSealer(b"w" * 32, key_version="wrong-recovery-key-v1"),
    )
    wrong_authority.initialize()
    with pytest.raises(CampaignRecoveryConflictError, match="key version"):
        wrong_authority.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=1))

    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            "UPDATE campaign_recovery_receipts SET seal_key_version='legacy-colocated-key'"
        )
    with pytest.raises(CampaignRecoveryConflictError, match="key version"):
        recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=2))


def test_legacy_receipt_schema_migrates_without_trusting_colocated_key(tmp_path):
    campaigns = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    campaigns.initialize()
    _create_campaign(campaigns, _campaign())
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE campaign_recovery_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO campaign_recovery_meta(key, value) VALUES ('seal_key', '00');
            CREATE TABLE campaign_recovery_receipts (
                receipt_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                state_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                emitted_at TEXT NOT NULL,
                expires_at TEXT
            );
            INSERT INTO campaign_recovery_receipts VALUES (
                'rcpt_00000000000000000000000000000000',
                'workspace-a', 'campaign-1', 'doctor',
                '0000000000000000000000000000000000000000000000000000000000000000',
                '{}', 'sha256_0000000000000000000000000000000000000000000000000000000000000000',
                '2026-07-16T20:30:00Z', NULL
            );
            """
        )
    recovery = CampaignRecoveryRepository(campaigns.db_path, sealer=_recovery_sealer())
    recovery.initialize()

    with sqlite3.connect(campaigns.db_path) as connection:
        migrated = connection.execute(
            "SELECT seal_key_version FROM campaign_recovery_receipts"
        ).fetchone()
        colocated_key = connection.execute(
            "SELECT value FROM campaign_recovery_meta WHERE key='seal_key'"
        ).fetchone()
    assert migrated == (None,)
    assert colocated_key is None
    with pytest.raises(CampaignRecoveryConflictError, match="key version"):
        recovery.verify_receipt("rcpt_00000000000000000000000000000000")


def test_stale_eligibility_cannot_be_rebound_by_tampering_receipt_metadata(repositories):
    campaigns, recovery = repositories
    eligible = recovery.project("workspace-a", "campaign-1", now=NOW)
    eligible_id = eligible["eligibility"]["receipt_id"]
    recovery.register_binding(
        installation_id=INSTALLATION,
        kind="compute",
        logical_id="registered-private-compute",
        availability="inaccessible",
    )
    blocked = recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=1))
    assert blocked["eligibility"]["decision"] == "blocked"

    with sqlite3.connect(campaigns.db_path) as connection:
        current_state = connection.execute(
            """
            SELECT state_digest FROM campaign_recovery_receipts
            WHERE workspace_id=? AND campaign_id=? AND kind='eligibility'
            ORDER BY emitted_at DESC, receipt_id DESC LIMIT 1
            """,
            ("workspace-a", "campaign-1"),
        ).fetchone()[0]
        connection.execute(
            """
            UPDATE campaign_recovery_receipts
            SET state_digest=?, emitted_at='2099-01-01T00:00:00Z'
            WHERE receipt_id=?
            """,
            (current_state, eligible_id),
        )

    with pytest.raises(CampaignRecoveryConflictError, match="seal|binding"):
        recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=2))


def test_restart_replay_rebuilds_from_sealed_receipts_not_mutable_response_json(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    original, _replayed = recovery.request(
        request, actor_id="human-operator", now=NOW + timedelta(seconds=1)
    )
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            "UPDATE campaign_recovery_mutations SET response_json=? WHERE workspace_id=?",
            ('{"secret":"response-must-not-leak"}', "workspace-a"),
        )

    reopened = CampaignRecoveryRepository(campaigns.db_path, sealer=_recovery_sealer())
    reopened.initialize()
    replay, replayed = reopened.request(
        request, actor_id="human-operator", now=NOW + timedelta(seconds=2)
    )

    assert replayed is True
    assert replay == original
    assert "must-not-leak" not in json.dumps(replay)


def test_restart_replay_rejects_a_mutable_request_hash_rebound_to_new_authority(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))
    forged = request.model_copy(
        update={"expected_event_cursor": request.expected_event_cursor + 999}
    )
    forged_hash = hashlib.sha256(
        json.dumps(
            forged.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            """
            UPDATE campaign_recovery_mutations SET request_hash=?
            WHERE workspace_id=? AND actor_id=? AND idempotency_key=?
            """,
            (forged_hash, "workspace-a", "human-operator", request.idempotency_key),
        )

    with pytest.raises(CampaignRecoveryConflictError, match="replay.*binding"):
        recovery.request(forged, actor_id="human-operator", now=NOW + timedelta(seconds=2))


def test_projection_verifies_every_bounded_historical_receipt_before_returning_it(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    outcome, _replayed = recovery.request(
        _request(snapshot), actor_id="human-operator", now=NOW + timedelta(seconds=1)
    )
    recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=2))
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            "UPDATE campaign_recovery_receipts SET payload_json=? WHERE receipt_id=?",
            (
                '{"workspace_id":"historical-must-not-leak"}',
                outcome["receipt"]["receipt_id"],
            ),
        )

    with pytest.raises(CampaignRecoveryConflictError, match="historical.*seal"):
        recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=3))


def _resident_worker(campaigns, recovery, tmp_path, *, worker_id="resident-worker"):
    worker = CampaignWorker(
        campaigns,
        tmp_path / "artifacts",
        _recovery_sealer(),
        data_directory=tmp_path / "data-root",
        worker_id=worker_id,
    )
    recovery.register_installation(
        installation_id=INSTALLATION,
        controller_owner_id=worker_id,
        controller_lease_key=worker.leader_key,
    )
    return worker


def test_projection_proves_recovery_consumer_readiness_from_live_controller_lease(
    repositories, tmp_path
):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    campaigns.acquire_lease(
        worker.leader_key,
        worker.worker_id,
        ttl=timedelta(seconds=30),
        now=NOW,
    )

    snapshot = recovery.project(
        "workspace-a", "campaign-1", now=NOW + timedelta(seconds=1)
    )

    assert snapshot["execution_consumer"] == {
        "supported": True,
        "ready": True,
        "reason_code": "ready",
    }


def _pause_campaign(campaigns):
    for trigger, version, key in (
        (CampaignTrigger.VALIDATE, 1, "recovery-validate"),
        (CampaignTrigger.VALIDATION_PASSED, 2, "recovery-ready"),
        (CampaignTrigger.START, 3, "recovery-start"),
        (CampaignTrigger.PAUSE, 4, "recovery-pause"),
    ):
        campaigns.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="human-operator",
            credential_kind=CredentialKind.ACCESS,
            correlation_id=key,
            idempotency_key=key,
        )


def test_accepted_resume_is_not_execution_and_resident_worker_completes_it(
    repositories, tmp_path
):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    _pause_campaign(campaigns)
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)

    accepted, replayed = recovery.request(
        request, actor_id="human-operator", now=NOW + timedelta(seconds=1)
    )

    assert replayed is False
    assert accepted["outcome"] == "accepted"
    assert campaigns.get_campaign("workspace-a", "campaign-1").status == CampaignStatus.PAUSED
    assert recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    ) == {
        "schema_version": "campaign_recovery_execution.v1",
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "action": "resume",
        "status": "accepted",
        "outcome_code": None,
        "attempt_id": None,
    }

    assert worker.run_once(now=NOW + timedelta(seconds=2)) == "recovery_resumed"
    assert campaigns.get_campaign("workspace-a", "campaign-1").status == CampaignStatus.ACTIVE
    completed = recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )
    assert completed["status"] == "completed"
    assert completed["outcome_code"] == "campaign_resumed"
    assert "resident-worker" not in json.dumps(completed)


def test_mutable_execution_lifecycle_tampering_fails_closed(repositories):
    campaigns, recovery = repositories
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            """UPDATE campaign_recovery_requests
               SET status='executing', claim_owner_id='FORGED-WORKER-CANARY',
                   claim_generation=99
               WHERE workspace_id=? AND campaign_id=? AND idempotency_key=?""",
            ("workspace-a", "campaign-1", request.idempotency_key),
        )

    with pytest.raises(CampaignRecoveryConflictError, match="execution.*seal"):
        recovery.execution_status("workspace-a", "campaign-1", request.idempotency_key)
    with pytest.raises(CampaignRecoveryConflictError, match="execution.*seal"):
        recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=2))


def test_recovery_claim_is_single_owner_restart_safe_and_idempotent(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    _pause_campaign(campaigns)
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))
    leader = campaigns.acquire_lease(
        worker.leader_key, worker.worker_id, ttl=timedelta(seconds=5), now=NOW + timedelta(seconds=2)
    )

    claim = recovery.claim_next(
        leader=leader,
        worker_id=worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=2),
    )
    assert claim is not None
    assert claim.status == "executing"
    assert recovery.claim_next(
        leader=leader,
        worker_id="not-the-lease-owner",
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=3),
    ) is None

    reopened = CampaignRecoveryRepository(campaigns.db_path, sealer=_recovery_sealer())
    reopened.initialize()
    assert reopened.claim_next(
        leader=leader,
        worker_id=worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=4),
    ) is None
    successor_leader = campaigns.acquire_lease(
        worker.leader_key,
        worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=8),
    )
    retried = reopened.claim_next(
        leader=successor_leader,
        worker_id=worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=8),
    )
    assert retried is not None
    assert retried.request_id == claim.request_id
    assert retried.claim_generation == claim.claim_generation + 1


def test_expired_unbound_repair_rechecks_authority_before_retry(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(
        snapshot, action=RecoveryAction.REPAIR, key="idem_" + "3" * 32
    )
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))
    leader = campaigns.acquire_lease(
        worker.leader_key,
        worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=2),
    )
    claim = recovery.claim_next(
        leader=leader,
        worker_id=worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=2),
    )
    assert claim is not None and claim.attempt_id is None
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            "UPDATE campaigns SET version=version+1 WHERE workspace_id=? AND campaign_id=?",
            ("workspace-a", "campaign-1"),
        )
    successor = campaigns.acquire_lease(
        worker.leader_key,
        worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=8),
    )

    blocked = recovery.claim_next(
        leader=successor,
        worker_id=worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=8),
    )
    assert blocked is not None and blocked.status == "blocked"
    assert recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )["outcome_code"] == "authority_changed"


def test_stale_recovery_authority_blocks_without_execution(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    _pause_campaign(campaigns)
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))
    campaigns.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.RESUME,
        expected_version=5,
        actor_id="human-operator",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="external-resume",
        idempotency_key="external-resume",
    )

    assert worker.run_once(now=NOW + timedelta(seconds=2)) == "recovery_blocked"
    outcome = recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )
    assert outcome["status"] == "blocked"
    assert outcome["outcome_code"] == "authority_changed"


def test_changed_lineage_after_acceptance_fails_closed(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    _pause_campaign(campaigns)
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            """
            UPDATE campaign_recovery_targets SET checkpoint_source_id=?
            WHERE workspace_id=? AND campaign_id=?
            """,
            (PRIVATE_ARTIFACT, "workspace-a", "campaign-1"),
        )

    assert worker.run_once(now=NOW + timedelta(seconds=2)) == "recovery_blocked"
    assert recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )["outcome_code"] == "authority_changed"


def test_generic_repair_without_exact_sealed_attempt_needs_operator(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(
        snapshot, action=RecoveryAction.REPAIR, key="idem_" + "5" * 32
    )
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))

    assert worker.run_once(now=NOW + timedelta(seconds=2)) == "recovery_blocked"
    outcome = recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )
    assert outcome == {
        "schema_version": "campaign_recovery_execution.v1",
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "action": "repair",
        "status": "blocked",
        "outcome_code": "needs_operator",
        "attempt_id": None,
    }
    assert len(json.dumps(outcome)) < 512


def test_resume_reopens_after_crash_and_replays_transition_idempotently(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    _pause_campaign(campaigns)
    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW)
    request = _request(snapshot)
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=1))
    leader = campaigns.acquire_lease(
        worker.leader_key,
        worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=2),
    )
    claim = recovery.claim_next(
        leader=leader,
        worker_id=worker.worker_id,
        ttl=timedelta(seconds=5),
        now=NOW + timedelta(seconds=2),
    )
    assert claim is not None
    campaigns.transition_campaign(
        claim.workspace_id,
        claim.campaign_id,
        CampaignTrigger.RESUME,
        expected_version=claim.expected_aggregate_version,
        actor_id="campaign-recovery-worker",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id=f"recovery-{claim.request_id}",
        idempotency_key=f"recovery-resume-{claim.request_id}",
        payload={"recovery_request_id": claim.request_id},
    )
    assert recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )["status"] == "executing"

    reopened = CampaignRecoveryRepository(campaigns.db_path, sealer=_recovery_sealer())
    reopened.initialize()
    successor = _resident_worker(
        campaigns, reopened, tmp_path, worker_id=worker.worker_id
    )
    assert successor.run_once(now=NOW + timedelta(seconds=8)) == "recovery_resumed"
    assert campaigns.get_campaign("workspace-a", "campaign-1").version == 6
    assert successor.recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )["status"] == "completed"


def test_repair_reconciles_only_the_exact_existing_sealed_attempt(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    for trigger, version, key in (
        (CampaignTrigger.VALIDATE, 1, "repair-validate"),
        (CampaignTrigger.VALIDATION_PASSED, 2, "repair-ready"),
        (CampaignTrigger.START, 3, "repair-start"),
    ):
        campaigns.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="human-operator",
            credential_kind=CredentialKind.ACCESS,
            correlation_id=key,
            idempotency_key=key,
        )
    plan = seed_validated_study(campaigns)
    schedule(campaigns, worker, plan)
    with pytest.raises(SimulatedWorkerCrashError):
        worker.run_once(now=NOW, crash_after_seal=True)
    unfinished = campaigns.list_unfinished_attempts()
    assert len(unfinished) == 1

    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=1))
    request = _request(
        snapshot, action=RecoveryAction.REPAIR, key="idem_" + "4" * 32
    )
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=2))

    assert worker.run_once(now=NOW + timedelta(seconds=3)) == "recovery_repaired"
    outcome = recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )
    assert outcome["status"] == "completed"
    assert outcome["outcome_code"] == "attempt_reconciled"
    assert outcome["attempt_id"] == unfinished[0].attempt_id
    assert campaigns.list_unfinished_attempts() == []


def test_repair_with_multiple_sealed_local_attempts_needs_operator(repositories, tmp_path):
    campaigns, recovery = repositories
    worker = _resident_worker(campaigns, recovery, tmp_path)
    for trigger, version, key in (
        (CampaignTrigger.VALIDATE, 1, "ambiguous-repair-validate"),
        (CampaignTrigger.VALIDATION_PASSED, 2, "ambiguous-repair-ready"),
        (CampaignTrigger.START, 3, "ambiguous-repair-start"),
    ):
        campaigns.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="human-operator",
            credential_kind=CredentialKind.ACCESS,
            correlation_id=key,
            idempotency_key=key,
        )
    plan = seed_validated_study(campaigns)
    schedule(campaigns, worker, plan)
    with pytest.raises(SimulatedWorkerCrashError):
        worker.run_once(now=NOW, crash_after_seal=True)
    first = campaigns.list_unfinished_attempts()[0]
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            """
            INSERT INTO campaign_attempts(
                workspace_id, action_id, attempt_id, attempt_number, claim_generation,
                status, lease_owner, lease_expires_at, heartbeat_at, executor_json,
                result_json, created_at, updated_at
            )
            SELECT workspace_id, action_id, 'attempt-ambiguous-local', 2, claim_generation,
                   status, lease_owner, lease_expires_at, heartbeat_at, executor_json,
                   result_json, created_at, updated_at
            FROM campaign_attempts WHERE workspace_id=? AND attempt_id=?
            """,
            (first.workspace_id, first.attempt_id),
        )
    second = campaigns.get_attempt("workspace-a", "attempt-ambiguous-local")
    worker.sealed_path(second).mkdir(parents=True)

    snapshot = recovery.project("workspace-a", "campaign-1", now=NOW + timedelta(seconds=1))
    request = _request(
        snapshot, action=RecoveryAction.REPAIR, key="idem_" + "6" * 32
    )
    recovery.request(request, actor_id="human-operator", now=NOW + timedelta(seconds=2))

    assert worker.run_once(now=NOW + timedelta(seconds=3)) == "recovery_blocked"
    outcome = recovery.execution_status(
        "workspace-a", "campaign-1", request.idempotency_key
    )
    assert outcome["status"] == "blocked"
    assert outcome["outcome_code"] == "needs_operator"
    assert outcome["attempt_id"] is None
    assert len(campaigns.list_unfinished_attempts()) == 2
