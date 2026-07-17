"""Durable, blinded, human-only campaign oversight tests."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import AutonomyProfile, CredentialKind, canonical_hash
from bashgym.campaigns.evaluation import DevelopmentGateContract, compare_development_evaluations
from bashgym.campaigns.human_oversight import (
    HumanOversightConflictError,
    HumanOversightIntegrityError,
    HumanOversightRepository,
)
from bashgym.campaigns.persistence import (
    CampaignPersistenceError,
    IdempotencyConflictError,
    PromotionGateFailedError,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from tests.campaigns.test_evaluation import _artifact, _rows
from tests.campaigns.test_persistence import campaign, create, manifest

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
TEST_SEAL_KEY = b"human-oversight-test-seal-key-material-v1"


def _principal(repository, *, actor_id="desktop-user", profile=AutonomyProfile.DESKTOP_USER):
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id=actor_id,
        autonomy_profile=profile,
        workspace_ids=("workspace-a",),
    )
    access = auth.exchange_refresh(refresh.raw_token)
    return auth.authenticate_access(access.raw_token)


def _seed(oversight, *, work_id="hw_0123456789abcdef", campaign_revision=1, replacement_for=None):
    return oversight.enqueue(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id=work_id,
        campaign_revision=campaign_revision,
        blocking=True,
        rubric={
            "rubric_id": "rub_01234567",
            "version": 1,
            "instructions": "Choose the stronger response.",
            "choices": [
                {"choice_id": "left", "label": "Left"},
                {"choice_id": "right", "label": "Right"},
                {"choice_id": "tie", "label": "No material difference"},
            ],
        },
        sample={
            "prompt": "Explain the bounded behavior.",
            "left": {"label": "A", "display": "Public candidate A"},
            "right": {"label": "B", "display": "Public candidate B"},
        },
        protected_context={
            "candidate_mapping": "SECRET-MAPPING-CANARY",
            "raw_path": "C:/private/restricted.jsonl",
            "api_key": "ghp_secret_canary",
        },
        replacement_for_work_id=replacement_for,
        now=NOW,
    )


def _prepare_ordinary_promotion_gate(repository):
    unprotected_manifest = manifest().model_copy(
        update={"protected_artifact_refs": ()}
    )
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaigns
            SET status = 'active', version = 2,
                best_development_candidate_ref = 'candidate-reviewed'
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
            """
        )
        connection.execute(
            """
            UPDATE campaign_manifest_revisions SET manifest_json = ?
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
              AND revision = 1
            """,
            (unprotected_manifest.model_dump_json(),),
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
                "gate-human-review-repro",
                json.dumps(
                    {
                        "verdict": "passed",
                        "candidate_digest": "candidate-reviewed",
                    }
                ),
                NOW.isoformat(),
            ),
        )


@pytest.fixture
def repository(tmp_path):
    value = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    value.initialize()
    create(value)
    return value


@pytest.fixture
def oversight(repository):
    return HumanOversightRepository(
        repository,
        sealer=ArtifactSealer(TEST_SEAL_KEY, key_version="human-test-v1"),
    )


def test_migration_and_bounded_public_queue_are_fail_closed(repository, oversight):
    _seed(oversight)
    queue = oversight.read_queue(
        "workspace-a", "campaign-1", _principal(repository), now=NOW, limit=50
    )

    assert queue["schema_version"] == "human_work_queue.v1"
    assert queue["campaign_revision"] == 1
    assert queue["reviewer"] == {"authenticated": True, "review_capability": True}
    assert len(queue["items"]) == 1
    assert queue["items"][0]["state"] == "pending"
    assert queue["items"][0]["claimed_by_current_reviewer"] is False
    assert queue["promotion"]["state"] == "blocked_by_review"
    public_json = json.dumps(queue)
    assert "SECRET-MAPPING-CANARY" not in public_json
    assert "restricted.jsonl" not in public_json
    assert "ghp_secret_canary" not in public_json

    with sqlite3.connect(repository.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "campaign_human_work",
        "campaign_human_receipts",
        "campaign_human_promotions",
        "campaign_human_mutations",
    } <= tables


def test_internal_enqueue_is_idempotent_and_emits_one_payload_free_public_hint(repository, oversight):
    first = _seed(oversight)
    replay = _seed(oversight)
    assert replay == first
    events = [
        event
        for _cursor, event in repository.list_events("workspace-a", "campaign-1")
        if event.event_type == "campaign:human-work-enqueued"
    ]
    assert len(events) == 1
    assert events[0].payload == {
        "work_id": "hw_0123456789abcdef",
        "campaign_revision": 1,
        "item_version": 1,
        "state": "pending",
    }


def test_agents_cannot_read_or_mutate_human_work(repository, oversight):
    seeded = _seed(oversight)
    agent = _principal(repository, actor_id="codex-agent", profile=AutonomyProfile.CODEX_TRUSTED)

    with pytest.raises(PermissionError, match="human_reviewer_required"):
        oversight.read_queue("workspace-a", "campaign-1", agent, now=NOW)
    with pytest.raises(PermissionError, match="human_reviewer_required"):
        oversight.claim(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            work_id="hw_0123456789abcdef",
            expected_campaign_revision=1,
            expected_version=1,
            expected_state="pending",
            principal=agent,
            correlation_id="corr-agent-denied",
            idempotency_key=seeded["items"][0]["claim_idempotency_key"],
            now=NOW,
        )


def test_claim_is_atomic_replayable_and_rejects_wrong_reviewer(repository, oversight):
    seeded = _seed(oversight)
    reviewer = _principal(repository)
    key = seeded["items"][0]["claim_idempotency_key"]
    request = dict(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id="hw_0123456789abcdef",
        expected_campaign_revision=1,
        expected_version=1,
        expected_state="pending",
        principal=reviewer,
        correlation_id="corr-claim",
        idempotency_key=key,
        now=NOW,
    )

    first = oversight.claim(**request)
    replay = oversight.claim(**request)
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.queue == first.queue
    assert first.queue["items"][0]["state"] == "claimed"
    assert first.queue["items"][0]["version"] == 2
    assert first.queue["items"][0]["claimed_by_current_reviewer"] is True

    other = _principal(repository, actor_id="other-desktop")
    other_queue = oversight.read_queue("workspace-a", "campaign-1", other, now=NOW)
    assert other_queue["items"][0]["claimed_by_current_reviewer"] is False
    with pytest.raises(HumanOversightConflictError):
        oversight.claim(**{**request, "principal": other, "expected_version": 2})
    with pytest.raises(IdempotencyConflictError):
        oversight.claim(**{**request, "expected_version": 999})


def test_mutation_replay_rejects_tampered_stored_response(repository, oversight):
    seeded = _seed(oversight)
    reviewer = _principal(repository)
    request = dict(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=1, expected_state="pending", principal=reviewer,
        correlation_id="corr-claim",
        idempotency_key=seeded["items"][0]["claim_idempotency_key"], now=NOW,
    )
    oversight.claim(**request)
    with sqlite3.connect(repository.db_path) as connection:
        response = json.loads(
            connection.execute(
                "SELECT response_json FROM campaign_human_mutations"
            ).fetchone()[0]
        )
        response["private_path"] = "C:/restricted/REPLAY-SECRET-CANARY"
        connection.execute(
            "UPDATE campaign_human_mutations SET response_json = ?",
            (json.dumps(response),),
        )
    with pytest.raises(HumanOversightIntegrityError):
        oversight.claim(**request)


def test_mutation_replay_rejects_forged_response_with_recomputed_public_digest(
    repository, oversight
):
    seeded = _seed(oversight)
    reviewer = _principal(repository)
    request = dict(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=1, expected_state="pending", principal=reviewer,
        correlation_id="corr-claim-forged-replay",
        idempotency_key=seeded["items"][0]["claim_idempotency_key"], now=NOW,
    )
    oversight.claim(**request)
    with sqlite3.connect(repository.db_path) as connection:
        response = json.loads(
            connection.execute(
                "SELECT response_json FROM campaign_human_mutations"
            ).fetchone()[0]
        )
        response["event_id"] = "evt-human-forged-replay"
        connection.execute(
            """
            UPDATE campaign_human_mutations
            SET response_json = ?, response_digest = ?
            """,
            (
                json.dumps(response, sort_keys=True, separators=(",", ":")),
                f"sha256:{canonical_hash(response)}",
            ),
        )
    with pytest.raises(HumanOversightIntegrityError):
        oversight.claim(**request)


def test_concurrent_claim_race_has_exactly_one_winner(repository, oversight):
    seeded = _seed(oversight)
    first = _principal(repository, actor_id="reviewer-one")
    second = _principal(repository, actor_id="reviewer-two")
    key = seeded["items"][0]["claim_idempotency_key"]

    def attempt(principal):
        try:
            result = oversight.claim(
                workspace_id="workspace-a", campaign_id="campaign-1",
                work_id="hw_0123456789abcdef", expected_campaign_revision=1,
                expected_version=1, expected_state="pending", principal=principal,
                correlation_id=f"corr-{principal.actor_id}", idempotency_key=key, now=NOW,
            )
            return result.queue["items"][0]["claimed_by_current_reviewer"]
        except HumanOversightConflictError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (first, second)))
    assert sorted(outcomes) == [False, True]
    assert len(
        [
            event
            for _cursor, event in repository.list_events("workspace-a", "campaign-1")
            if event.event_type == "campaign:human-work-claimed"
        ]
    ) == 1


def test_expired_lease_reclaims_with_new_fence_and_stale_submit_fails(repository, oversight):
    seeded = _seed(oversight)
    reviewer = _principal(repository)
    claimed = oversight.claim(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id="hw_0123456789abcdef",
        expected_campaign_revision=1,
        expected_version=1,
        expected_state="pending",
        principal=reviewer,
        correlation_id="corr-claim",
        idempotency_key=seeded["items"][0]["claim_idempotency_key"],
        now=NOW,
    )
    old_submit_key = claimed.queue["items"][0]["submit_idempotency_key"]
    later = NOW + timedelta(minutes=16)
    expired = oversight.read_queue("workspace-a", "campaign-1", reviewer, now=later)
    assert expired["items"][0]["state"] == "expired"
    reclaimed = oversight.claim(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id="hw_0123456789abcdef",
        expected_campaign_revision=1,
        expected_version=2,
        expected_state="expired",
        principal=reviewer,
        correlation_id="corr-reclaim",
        idempotency_key=claimed.queue["items"][0]["claim_idempotency_key"],
        now=later,
    )
    assert reclaimed.queue["items"][0]["version"] == 3
    assert reclaimed.queue["items"][0]["submit_idempotency_key"] != old_submit_key
    wrong_reviewer = _principal(repository, actor_id="other-desktop")
    with pytest.raises(HumanOversightConflictError):
        oversight.submit(
            workspace_id="workspace-a", campaign_id="campaign-1",
            work_id="hw_0123456789abcdef", expected_campaign_revision=1,
            expected_version=3, expected_rubric_version=1, decision="prefer_left",
            rationale="Not my lease.", principal=wrong_reviewer,
            correlation_id="corr-wrong-reviewer",
            idempotency_key=reclaimed.queue["items"][0]["submit_idempotency_key"], now=later,
        )
    with pytest.raises(HumanOversightConflictError):
        oversight.submit(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            work_id="hw_0123456789abcdef",
            expected_campaign_revision=1,
            expected_version=2,
            expected_rubric_version=1,
            decision="prefer_left",
            rationale="Old lease response.",
            principal=reviewer,
            correlation_id="corr-stale-submit",
            idempotency_key=old_submit_key,
            now=later,
        )


def test_submit_seals_stable_receipt_and_promotion_is_separate(repository, oversight):
    seeded = _seed(oversight)
    reviewer = _principal(repository)
    claimed = oversight.claim(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=1, expected_state="pending", principal=reviewer,
        correlation_id="corr-claim", idempotency_key=seeded["items"][0]["claim_idempotency_key"], now=NOW,
    )
    submit_key = claimed.queue["items"][0]["submit_idempotency_key"]
    submit_request = dict(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=2, expected_rubric_version=1,
        decision="prefer_left", rationale="A is materially stronger.",
        principal=reviewer, correlation_id="corr-submit", idempotency_key=submit_key,
        now=NOW + timedelta(minutes=1),
    )
    submitted = oversight.submit(**submit_request)
    replay = oversight.submit(**submit_request)
    receipt = submitted.queue["items"][0]["receipt"]
    assert replay.replayed is True
    assert replay.queue == submitted.queue
    assert receipt["receipt_digest"].startswith("sha256:")
    assert submitted.queue["promotion"]["state"] == "awaiting_human_decision"
    assert submitted.queue["promotion"]["eligible_receipt_id"] == receipt["receipt_id"]
    assert submitted.queue["promotion"]["state"] != "promoted"
    with pytest.raises(IdempotencyConflictError):
        oversight.submit(**{**submit_request, "decision": "prefer_right"})

    promotion_key = submitted.queue["promotion"]["idempotency_key"]
    held = oversight.decide_promotion(
        workspace_id="workspace-a", campaign_id="campaign-1",
        receipt_id=receipt["receipt_id"], work_id="hw_0123456789abcdef",
        expected_campaign_revision=1, expected_item_version=3,
        expected_rubric_version=1, expected_promotion_version=2,
        expected_promotion_state="awaiting_human_decision", decision="hold",
        principal=reviewer, correlation_id="corr-hold", idempotency_key=promotion_key,
        now=NOW + timedelta(minutes=2),
    )
    assert held.queue["promotion"]["state"] == "blocked_by_review"


def test_incomplete_blocking_work_keeps_promotion_fail_closed(repository, oversight):
    first_seed = _seed(oversight)
    _seed(oversight, work_id="hw_fedcba9876543210")
    reviewer = _principal(repository)
    claimed = oversight.claim(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=1, expected_state="pending", principal=reviewer,
        correlation_id="corr-first-claim",
        idempotency_key=next(
            item["claim_idempotency_key"]
            for item in first_seed["items"]
            if item["work_id"] == "hw_0123456789abcdef"
        ),
        now=NOW,
    )
    submitted = oversight.submit(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=2, expected_rubric_version=1, decision="prefer_left",
        rationale="Only the first blocker is complete.", principal=reviewer,
        correlation_id="corr-first-submit",
        idempotency_key=next(
            item["submit_idempotency_key"]
            for item in claimed.queue["items"]
            if item["work_id"] == "hw_0123456789abcdef"
        ),
        now=NOW + timedelta(minutes=1),
    )
    assert submitted.queue["promotion"] == {
        **submitted.queue["promotion"],
        "state": "blocked_by_review",
        "eligible_receipt_id": None,
    }


def test_receipt_tampering_fails_closed(repository, oversight):
    seeded = _seed(oversight)
    reviewer = _principal(repository)
    claimed = oversight.claim(
        workspace_id="workspace-a", campaign_id="campaign-1", work_id="hw_0123456789abcdef",
        expected_campaign_revision=1, expected_version=1, expected_state="pending",
        principal=reviewer, correlation_id="corr-claim",
        idempotency_key=seeded["items"][0]["claim_idempotency_key"], now=NOW,
    )
    oversight.submit(
        workspace_id="workspace-a", campaign_id="campaign-1", work_id="hw_0123456789abcdef",
        expected_campaign_revision=1, expected_version=2, expected_rubric_version=1,
        decision="prefer_left", rationale="Valid decision.", principal=reviewer,
        correlation_id="corr-submit", idempotency_key=claimed.queue["items"][0]["submit_idempotency_key"],
        now=NOW + timedelta(minutes=1),
    )
    with sqlite3.connect(repository.db_path) as connection:
        row = connection.execute(
            "SELECT sealed_payload_json FROM campaign_human_receipts WHERE work_id = ?",
            ("hw_0123456789abcdef",),
        ).fetchone()
        forged_payload = json.loads(row[0])
        forged_payload["decision"] = "prefer_right"
        connection.execute(
            """
            UPDATE campaign_human_receipts
            SET decision = 'prefer_right', sealed_payload_json = ?, receipt_digest = ?
            WHERE work_id = ?
            """,
            (
                json.dumps(forged_payload, sort_keys=True, separators=(",", ":")),
                f"sha256:{canonical_hash(forged_payload)}",
                "hw_0123456789abcdef",
            ),
        )
    with pytest.raises(HumanOversightIntegrityError):
        oversight.read_queue("workspace-a", "campaign-1", reviewer, now=NOW + timedelta(minutes=2))


def test_public_projection_rejects_stored_shape_tampering_before_serialization(repository, oversight):
    _seed(oversight)
    reviewer = _principal(repository)
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE campaign_human_work SET public_sample_json = ? WHERE work_id = ?",
            (
                json.dumps(
                    {
                        "prompt": "Prompt",
                        "left": {"label": "A", "display": "Left"},
                        "right": {"label": "B", "display": "Right"},
                        "private_path": "C:/restricted/SHOULD-NOT-SERIALIZE",
                    }
                ),
                "hw_0123456789abcdef",
            ),
        )
    with pytest.raises(HumanOversightIntegrityError):
        oversight.read_queue("workspace-a", "campaign-1", reviewer, now=NOW)


def test_revision_invalidation_requires_explicit_lineage_replacement_and_survives_restart(repository, oversight):
    _seed(oversight)
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaigns SET manifest_revision = 2 WHERE workspace_id = ? AND campaign_id = ?",
            ("workspace-a", "campaign-1"),
        )
    reviewer = _principal(repository)
    invalidated = oversight.read_queue("workspace-a", "campaign-1", reviewer, now=NOW)
    assert invalidated["campaign_revision"] == 2
    assert invalidated["items"][0]["campaign_revision"] == 1
    with pytest.raises(HumanOversightConflictError):
        oversight.claim(
            workspace_id="workspace-a", campaign_id="campaign-1", work_id="hw_0123456789abcdef",
            expected_campaign_revision=1, expected_version=1, expected_state="pending",
            principal=reviewer, correlation_id="corr-old-revision",
            idempotency_key=invalidated["items"][0]["claim_idempotency_key"], now=NOW,
        )

    replaced = _seed(
        oversight,
        work_id="hw_fedcba9876543210",
        campaign_revision=2,
        replacement_for="hw_0123456789abcdef",
    )
    states = {item["work_id"]: item["state"] for item in replaced["items"]}
    assert states == {
        "hw_fedcba9876543210": "pending",
        "hw_0123456789abcdef": "replaced",
    }
    restarted = CampaignRuntimeRepository(repository.db_path)
    restarted.initialize()
    recovered = HumanOversightRepository(
        restarted,
        sealer=ArtifactSealer(TEST_SEAL_KEY, key_version="human-test-v1"),
    ).read_queue(
        "workspace-a", "campaign-1", reviewer, now=NOW
    )
    assert recovered == replaced


def test_queue_limit_is_bounded(repository, oversight):
    for index in range(51):
        oversight.enqueue(
            workspace_id="workspace-a", campaign_id="campaign-1",
            work_id=f"hw_{index:016d}", campaign_revision=1, blocking=False,
            rubric={"rubric_id": "rub_01234567", "version": 1, "instructions": "Choose.", "choices": [{"choice_id": "tie", "label": "Tie"}]},
            sample={"prompt": "Prompt", "left": {"label": "A", "display": "Left"}, "right": {"label": "B", "display": "Right"}},
            protected_context={"row": index}, now=NOW,
        )
    reviewer = _principal(repository)
    assert len(oversight.read_queue("workspace-a", "campaign-1", reviewer, now=NOW, limit=50)["items"]) == 50
    with pytest.raises(ValueError, match="limit"):
        oversight.read_queue("workspace-a", "campaign-1", reviewer, now=NOW, limit=51)
    with pytest.raises(ValueError, match="protected context"):
        oversight.enqueue(
            workspace_id="workspace-a", campaign_id="campaign-1",
            work_id="hw_oversizedcontext1", campaign_revision=1, blocking=False,
            rubric={"rubric_id": "rub_01234567", "version": 1, "instructions": "Choose.", "choices": [{"choice_id": "tie", "label": "Tie"}]},
            sample={"prompt": "Prompt", "left": {"label": "A", "display": "Left"}, "right": {"label": "B", "display": "Right"}},
            protected_context={"blob": "x" * 70_000}, now=NOW,
        )


def test_workspace_isolation_allows_same_work_identity_without_cross_scope_leakage(repository, oversight):
    _seed(oversight)
    create(repository, campaign("workspace-b", "campaign-b"))
    oversight.enqueue(
        workspace_id="workspace-b", campaign_id="campaign-b",
        work_id="hw_0123456789abcdef", campaign_revision=1, blocking=True,
        rubric={"rubric_id": "rub_01234567", "version": 1, "instructions": "Choose.", "choices": [{"choice_id": "tie", "label": "Tie"}]},
        sample={"prompt": "Other workspace", "left": {"label": "A", "display": "Other left"}, "right": {"label": "B", "display": "Other right"}},
        protected_context={"canary": "WORKSPACE-B-SECRET"}, now=NOW,
    )
    workspace_a_reviewer = _principal(repository)
    with pytest.raises(PermissionError, match="workspace"):
        oversight.read_queue("workspace-b", "campaign-b", workspace_a_reviewer, now=NOW)

    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="workspace-b-reviewer",
        autonomy_profile=AutonomyProfile.DESKTOP_USER,
        workspace_ids=("workspace-b",),
    )
    workspace_b_reviewer = auth.authenticate_access(
        auth.exchange_refresh(refresh.raw_token).raw_token
    )
    queue_b = oversight.read_queue(
        "workspace-b", "campaign-b", workspace_b_reviewer, now=NOW
    )
    assert queue_b["items"][0]["sample"]["prompt"] == "Other workspace"
    assert "WORKSPACE-B-SECRET" not in json.dumps(queue_b)


def test_pending_blocking_human_work_prevents_campaign_promotion(repository, oversight):
    _seed(oversight)
    _prepare_ordinary_promotion_gate(repository)

    with pytest.raises(PromotionGateFailedError, match="campaign_gate_failed"):
        repository.promote_candidate(
            "workspace-a",
            "campaign-1",
            expected_version=2,
            actor_id="desktop-user",
            credential_kind=CredentialKind.ACCESS,
            correlation_id="corr-promotion-must-wait-for-human",
            idempotency_key="promotion-must-wait-for-human",
        )


def test_pending_human_work_blocks_new_comparison_until_sealed_submission(
    repository, oversight
):
    queue = _seed(oversight)
    comparison = compare_development_evaluations(
        _artifact(digest="a" * 64, rows=_rows(count=18, videos=3, rank=1)),
        _artifact(digest="b" * 64, rows=_rows(count=18, videos=3, rank=2)),
        DevelopmentGateContract(bootstrap_samples=100),
    )
    with pytest.raises(CampaignPersistenceError, match="campaign_human_work_incomplete"):
        repository.store_development_comparison(
            "workspace-a", "campaign-1", comparison, now=NOW
        )

    reviewer = _principal(repository)
    claimed = oversight.claim(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=1, expected_state="pending", principal=reviewer,
        correlation_id="corr-claim-comparison",
        idempotency_key=queue["items"][0]["claim_idempotency_key"], now=NOW,
    )
    oversight.submit(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id="hw_0123456789abcdef", expected_campaign_revision=1,
        expected_version=2, expected_rubric_version=1,
        decision="no_material_difference", rationale="Equivalent under the rubric.",
        principal=reviewer, correlation_id="corr-submit-comparison",
        idempotency_key=claimed.queue["items"][0]["submit_idempotency_key"],
        now=NOW + timedelta(minutes=1),
    )
    assert repository.store_development_comparison(
        "workspace-a", "campaign-1", comparison, now=NOW + timedelta(minutes=2)
    ).startswith("gate-")


def test_codex_override_cannot_bypass_pending_human_authority(repository, oversight):
    _seed(oversight)
    _prepare_ordinary_promotion_gate(repository)
    codex = _principal(
        repository,
        actor_id="codex-agent",
        profile=AutonomyProfile.CODEX_TRUSTED,
    )

    with pytest.raises(PermissionError):
        CampaignService(repository).promote(
            "workspace-a",
            "campaign-1",
            expected_version=2,
            principal=codex,
            correlation_id="corr-codex-human-override",
            idempotency_key="codex-human-override",
            override_reason="Agent cannot override a pending human decision.",
        )


def test_bounded_queue_keeps_the_eligible_promotion_receipt_in_projection(
    repository, oversight
):
    for index in range(51):
        oversight.enqueue(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            work_id=f"hw_{index:016d}",
            campaign_revision=1,
            blocking=index == 50,
            rubric={
                "rubric_id": "rub_01234567",
                "version": 1,
                "instructions": "Choose.",
                "choices": [{"choice_id": "tie", "label": "Tie"}],
            },
            sample={
                "prompt": "Prompt",
                "left": {"label": "A", "display": "Left"},
                "right": {"label": "B", "display": "Right"},
            },
            protected_context={"row": index},
            now=NOW,
        )
    reviewer = _principal(repository)
    target = "hw_0000000000000050"
    with repository._connection() as connection:
        claim_key = connection.execute(
            """
            SELECT claim_idempotency_key FROM campaign_human_work
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
              AND work_id = ?
            """,
            (target,),
        ).fetchone()["claim_idempotency_key"]
    oversight.claim(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id=target,
        expected_campaign_revision=1,
        expected_version=1,
        expected_state="pending",
        principal=reviewer,
        correlation_id="corr-claim-hidden-eligible",
        idempotency_key=claim_key,
        now=NOW,
    )
    with repository._connection() as connection:
        submit_key = connection.execute(
            """
            SELECT submit_idempotency_key FROM campaign_human_work
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
              AND work_id = ?
            """,
            (target,),
        ).fetchone()["submit_idempotency_key"]
    submitted = oversight.submit(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id=target,
        expected_campaign_revision=1,
        expected_version=2,
        expected_rubric_version=1,
        decision="no_material_difference",
        rationale="The blinded samples are equivalent.",
        principal=reviewer,
        correlation_id="corr-submit-hidden-eligible",
        idempotency_key=submit_key,
        now=NOW + timedelta(minutes=1),
    ).queue
    eligible = submitted["promotion"]["eligible_receipt_id"]
    assert eligible is not None
    assert eligible in {
        item["receipt"]["receipt_id"]
        for item in submitted["items"]
        if item["receipt"] is not None
    }
