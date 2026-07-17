"""Persisted campaign operator actions behind REST, CLI, MCP, and Hermes."""

import json

import pytest

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import (
    ActionStatus,
    AttemptStatus,
    AutonomyProfile,
    ProtectedEvaluationResult,
    StudyStatus,
)
from bashgym.campaigns.human_oversight import HumanOversightRepository
from bashgym.campaigns.persistence import (
    PromotionGateFailedError,
    ProtectedLeaseDeniedError,
)
from bashgym.campaigns.runtime import ActionIdentityMismatchError, CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from tests.campaigns.test_persistence import campaign, manifest
from tests.campaigns.test_proposals import activate, principal, proposal
from tests.campaigns.test_remote_persistence import _claimed_attempt, _identity


def active_repository(tmp_path) -> CampaignRuntimeRepository:
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    activate(repository)
    return repository


def test_manifest_budget_and_source_actions_are_versioned_and_replay_exact(tmp_path):
    repository = active_repository(tmp_path)
    service = CampaignService(repository)
    actor = principal(repository)

    revised_manifest = manifest().model_copy(update={"max_proposal_rounds": 8})
    revised = service.revise_manifest(
        "workspace-a",
        "campaign-1",
        revised_manifest,
        reason="Allow the bounded three-study dry campaign.",
        expected_version=4,
        principal=actor,
        correlation_id="revise",
        idempotency_key="revise",
    )
    replay = service.revise_manifest(
        "workspace-a",
        "campaign-1",
        revised_manifest,
        reason="Allow the bounded three-study dry campaign.",
        expected_version=4,
        principal=actor,
        correlation_id="revise",
        idempotency_key="revise",
    )
    assert revised.details["revision"] == 2
    assert replay.replayed is True
    assert repository.get_manifest_revision("workspace-a", "campaign-1", 2).manifest == revised_manifest

    budget = service.amend_budget(
        "workspace-a",
        "campaign-1",
        "gpu_hours",
        2.0,
        reason="Approved capacity extension.",
        expected_version=revised.campaign.version,
        principal=actor,
        correlation_id="budget",
        idempotency_key="budget",
    )
    assert budget.entry.limit_delta == 2.0
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours")[
        "limit_delta"
    ] == 2.0

    evidence = {
        "provenance": "Hugging Face dataset card and pinned artifact",
        "license": "apache-2.0",
        "privacy_review": "No personal data detected by the approved review.",
        "contamination_review": "No protected evaluation overlap.",
        "artifact_sha256": "a" * 64,
        "notes": None,
    }
    approved = service.approve_source(
        "workspace-a",
        "campaign-1",
        "hf-community-evals-v1",
        evidence,
        expected_version=budget.campaign.version,
        principal=actor,
        correlation_id="source",
        idempotency_key="source",
    )
    serialized = json.dumps(approved.details)
    assert approved.details["source_id"] == "hf-community-evals-v1"
    assert "Hugging Face" not in serialized
    with repository._connection() as connection:
        stored = connection.execute(
            "SELECT evidence_json FROM campaign_source_approvals"
        ).fetchone()
    assert json.loads(stored["evidence_json"]) == evidence


def test_abandon_retry_and_exact_identity_force_stop_requests(tmp_path):
    repository = active_repository(tmp_path / "abandon")
    service = CampaignService(repository)
    actor = principal(repository)
    submitted = service.submit_proposal(
        proposal("proposal-abandon"),
        expected_version=4,
        principal=actor,
        correlation_id="proposal",
        idempotency_key="proposal",
    )
    selection = repository.select_next_proposal_as_controller(
        "workspace-a",
        "campaign-1",
        expected_version=submitted.campaign.version,
        controller_id="controller",
        correlation_id="select",
        idempotency_key="select",
    )
    abandoned = service.abandon_study(
        "workspace-a",
        "campaign-1",
        selection.study.study_id,
        reason="Dry-campaign branch is no longer useful.",
        expected_version=selection.campaign.version,
        principal=actor,
        correlation_id="abandon",
        idempotency_key="abandon",
    )
    assert abandoned.details["study_id"] == selection.study.study_id
    with repository._connection() as connection:
        status = connection.execute(
            "SELECT status FROM campaign_studies WHERE study_id = ?",
            (selection.study.study_id,),
        ).fetchone()["status"]
    assert status == StudyStatus.ABANDONED.value

    remote_repository, attempt = _claimed_attempt(tmp_path / "force")
    remote_service = CampaignService(remote_repository)
    remote_actor = principal(remote_repository)
    identity = _identity(attempt.attempt_id)
    remote_repository.register_remote_identity(attempt, identity)
    campaign_version = remote_repository.get_campaign("workspace-a", "campaign-1").version
    with pytest.raises(ActionIdentityMismatchError):
        remote_service.request_force_stop(
            "workspace-a",
            "campaign-1",
            attempt.action_id,
            identity.model_copy(update={"remote_pid": identity.remote_pid + 1}),
            reason="Identity mismatch must fail closed.",
            expected_version=campaign_version,
            principal=remote_actor,
            correlation_id="force-bad",
            idempotency_key="force-bad",
        )
    requested = remote_service.request_force_stop(
        "workspace-a",
        "campaign-1",
        attempt.action_id,
        identity,
        reason="Operator-confirmed exact identity stop.",
        expected_version=campaign_version,
        principal=remote_actor,
        correlation_id="force",
        idempotency_key="force",
    )
    assert requested.details["state"] == "pending"
    assert (
        remote_repository.pending_force_stop_request(
            "workspace-a", attempt.action_id, identity
        )
        == requested.details["request_id"]
    )


def test_failed_action_retry_preserves_hashes_and_reserves_a_new_attempt(tmp_path):
    repository, attempt = _claimed_attempt(tmp_path)
    service = CampaignService(repository)
    actor = principal(repository)
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_attempts SET status = ? WHERE attempt_id = ?",
            (AttemptStatus.FAILED.value, attempt.attempt_id),
        )
        connection.execute(
            "UPDATE campaign_actions SET status = ? WHERE action_id = ?",
            (ActionStatus.FAILED.value, attempt.action_id),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_action_id = NULL, active_study_id = NULL,
                version = version + 1 WHERE workspace_id = ? AND campaign_id = ?
            """,
            (attempt.workspace_id, attempt.campaign_id),
        )
        reservation = connection.execute(
            "SELECT unit, reserved_delta FROM campaign_budget_ledger WHERE action_id = ?",
            (attempt.action_id,),
        ).fetchone()
        connection.execute(
            """
            INSERT INTO campaign_budget_ledger(
                workspace_id, campaign_id, entry_id, unit, entry_kind,
                reserved_delta, actual_delta, limit_delta, action_id,
                evidence_json, actor_id, created_at
            ) VALUES (?, ?, ?, ?, 'release', ?, 0, 0, ?, '{}', 'controller', ?)
            """,
            (
                attempt.workspace_id,
                attempt.campaign_id,
                f"release-{attempt.action_id}",
                reservation["unit"],
                -float(reservation["reserved_delta"]),
                attempt.action_id,
                campaign().created_at.isoformat(),
            ),
        )
    version = repository.get_campaign("workspace-a", "campaign-1").version
    retried = service.retry_action(
        "workspace-a",
        "campaign-1",
        attempt.action_id,
        expected_version=version,
        principal=actor,
        correlation_id="retry",
        idempotency_key="retry",
    )
    replay = service.retry_action(
        "workspace-a",
        "campaign-1",
        attempt.action_id,
        expected_version=version,
        principal=actor,
        correlation_id="retry",
        idempotency_key="retry",
    )
    assert retried.details["attempt_number"] == 2
    assert retried.details["input_digest"] == attempt.input_digest
    assert retried.details["candidate_digest"] == attempt.candidate_digest
    assert replay.replayed is True
    assert len(repository.list_attempts("workspace-a", "campaign-1")) == 2


def test_protected_epoch_is_one_use_and_promotion_fails_closed_then_commits(tmp_path):
    repository = active_repository(tmp_path)
    service = CampaignService(repository)
    actor = principal(repository)
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_gate_decisions(
                workspace_id, campaign_id, decision_id, decision_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "gate-pass",
                json.dumps({"verdict": "passed", "candidate_digest": "c" * 64}),
                campaign().created_at.isoformat(),
            ),
        )
    leased = service.acquire_protected_lease(
        "workspace-a",
        "campaign-1",
        expected_version=4,
        principal=actor,
        correlation_id="protected",
        idempotency_key="protected",
    )
    replay = service.acquire_protected_lease(
        "workspace-a",
        "campaign-1",
        expected_version=4,
        principal=actor,
        correlation_id="protected",
        idempotency_key="protected",
    )
    assert replay.replayed is True
    with pytest.raises(ProtectedLeaseDeniedError):
        service.acquire_protected_lease(
            "workspace-a",
            "campaign-1",
            expected_version=leased.campaign.version,
            principal=actor,
            correlation_id="protected-again",
            idempotency_key="protected-again",
        )
    with pytest.raises(PromotionGateFailedError):
        service.promote(
            "workspace-a",
            "campaign-1",
            expected_version=leased.campaign.version,
            principal=actor,
            correlation_id="promote-too-early",
            idempotency_key="promote-too-early",
        )
    protected_result = service.record_protected_evaluation(
        "workspace-a",
        "campaign-1",
        ProtectedEvaluationResult(
            protected_epoch_id=leased.details["protected_epoch_id"],
            candidate_digest="c" * 64,
            passed=True,
            metrics={"recall_at_10": 0.84},
            artifact_sha256="d" * 64,
        ),
        expected_version=leased.campaign.version,
        principal=actor,
        correlation_id="protected-result",
        idempotency_key="protected-result",
    )
    assert protected_result.details["passed"] is True

    oversight = HumanOversightRepository(
        repository,
        sealer=ArtifactSealer(
            b"operator-action-human-seal-key-v1",
            key_version="operator-human-v1",
        ),
    )
    queue = oversight.enqueue(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id="hw_promotiongate0001",
        campaign_revision=1,
        blocking=True,
        rubric={
            "rubric_id": "rub_gate0001",
            "version": 1,
            "instructions": "Choose the stronger response.",
            "choices": [
                {"choice_id": "left", "label": "Left"},
                {"choice_id": "right", "label": "Right"},
                {"choice_id": "tie", "label": "No material difference"},
            ],
        },
        sample={
            "prompt": "Review the bounded candidate evidence.",
            "left": {"label": "A", "display": "Public candidate A"},
            "right": {"label": "B", "display": "Public candidate B"},
        },
        protected_context={"candidate_mapping": "server-owned"},
    )
    with pytest.raises(PromotionGateFailedError):
        service.promote(
            "workspace-a",
            "campaign-1",
            expected_version=protected_result.campaign.version,
            principal=actor,
            correlation_id="promote-before-human-decision",
            idempotency_key="promote-before-human-decision",
        )

    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="desktop-reviewer",
        autonomy_profile=AutonomyProfile.DESKTOP_USER,
        workspace_ids=("workspace-a",),
    )
    reviewer = auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)
    claimed = oversight.claim(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id="hw_promotiongate0001",
        expected_campaign_revision=1,
        expected_version=1,
        expected_state="pending",
        principal=reviewer,
        correlation_id="claim-human-gate",
        idempotency_key=queue["items"][0]["claim_idempotency_key"],
    )
    submitted = oversight.submit(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id="hw_promotiongate0001",
        expected_campaign_revision=1,
        expected_version=2,
        expected_rubric_version=1,
        decision="prefer_left",
        rationale="Candidate A meets the rubric.",
        principal=reviewer,
        correlation_id="submit-human-gate",
        idempotency_key=claimed.queue["items"][0]["submit_idempotency_key"],
    )
    receipt = submitted.queue["items"][0]["receipt"]
    approved = oversight.decide_promotion(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        receipt_id=receipt["receipt_id"],
        work_id="hw_promotiongate0001",
        expected_campaign_revision=1,
        expected_item_version=3,
        expected_rubric_version=1,
        expected_promotion_version=2,
        expected_promotion_state="awaiting_human_decision",
        decision="promote",
        principal=reviewer,
        correlation_id="approve-human-gate",
        idempotency_key=submitted.queue["promotion"]["idempotency_key"],
    )
    assert approved.queue["promotion"]["state"] == "promoted"
    promoted = service.promote(
        "workspace-a",
        "campaign-1",
        expected_version=protected_result.campaign.version,
        principal=actor,
        correlation_id="promote",
        idempotency_key="promote",
    )
    assert promoted.campaign.status.value == "completed"
    assert promoted.details["protected_gate_passed"] is True
    assert promoted.campaign.champion_ref.startswith("champion:memexai-embedding-v1:")


def test_export_is_server_managed_replayable_and_path_free(tmp_path):
    repository = active_repository(tmp_path / "state")
    service = CampaignService(repository, export_root=tmp_path / "exports")
    actor = principal(repository)
    exported = service.export(
        "workspace-a",
        "campaign-1",
        ("markdown", "json", "csv", "png", "docx", "pdf"),
        expected_version=4,
        principal=actor,
        correlation_id="export",
        idempotency_key="export",
    )
    replay = service.export(
        "workspace-a",
        "campaign-1",
        ("markdown", "json", "csv", "png", "docx", "pdf"),
        expected_version=4,
        principal=actor,
        correlation_id="export",
        idempotency_key="export",
    )
    assert replay.replayed is True
    assert exported.details["export_id"].startswith("export-")
    assert "path" not in json.dumps(exported.details).casefold()
    output = tmp_path / "exports" / "workspace-a" / "campaign-1" / exported.details["export_id"]
    assert (output / "campaign_report.docx").is_file()
    assert (output / "campaign_report.pdf").is_file()
