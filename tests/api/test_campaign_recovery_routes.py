"""Authenticated recovery authority REST boundary tests."""

from __future__ import annotations

import json
from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api.campaign_recovery_routes import campaign_recovery_router
from bashgym.api.routes import create_app
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
from bashgym.campaigns.contracts import AutonomyProfile
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from tests.campaigns.test_campaign_recovery import (
    INSTALLATION,
    LEASE_KEY,
    PRIVATE_ARTIFACT,
    PRIVATE_CHECKPOINT,
    PRIVATE_OWNER,
    _campaign,
    _create_campaign,
    _insert_artifact,
)


def _client(tmp_path):
    campaigns = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    campaigns.initialize()
    _create_campaign(campaigns, _campaign())
    _insert_artifact(campaigns, PRIVATE_CHECKPOINT, "checkpoint.v1")
    _insert_artifact(campaigns, PRIVATE_ARTIFACT, "adapter.v1")
    recovery = CampaignRecoveryRepository(
        campaigns.db_path,
        sealer=ArtifactSealer(b"a" * 32, key_version="recovery-api-test-v1"),
    )
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
    auth = CampaignAuthService(campaigns)
    tokens = {}
    for name, profile in (
        ("desktop", AutonomyProfile.DESKTOP_USER),
        ("codex", AutonomyProfile.CODEX_TRUSTED),
    ):
        refresh = auth.issue_refresh_credential(
            actor_id=f"{name}-principal",
            autonomy_profile=profile,
            workspace_ids=("workspace-a",),
            ttl=timedelta(hours=1),
        )
        tokens[name] = auth.exchange_refresh(refresh.raw_token).raw_token
    app = FastAPI()
    app.state.campaign_repository = campaigns
    app.state.campaign_auth_service = auth
    app.state.campaign_recovery_repository = recovery
    app.include_router(campaign_recovery_router)
    return TestClient(app), tokens, recovery


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


def _payload(snapshot, *, key="idem_" + "9" * 32, action="resume"):
    return {
        "action": action,
        "idempotencyKey": key,
        "workspaceId": snapshot["workspace_id"],
        "campaignId": snapshot["campaign_id"],
        "eligibilityReceiptId": snapshot["eligibility"]["receipt_id"],
        "doctorEvidenceId": snapshot["doctor"]["evidence_id"],
        "expectedCampaignRevision": snapshot["lineage"]["campaign_revision"],
        "expectedEventCursor": snapshot["lineage"]["event_cursor"],
        "expectedAggregateVersion": snapshot["lineage"]["aggregate_version"],
        "expectedControllerLeaseId": snapshot["controller"]["lease_id"],
        "checkpointId": snapshot["lineage"]["checkpoint_id"],
        "artifactId": snapshot["lineage"]["artifact_id"],
        "humanConfirmed": True,
    }


def test_create_app_registers_campaign_recovery_routes():
    def registered_paths(routes):
        for route in routes:
            path = getattr(route, "path", None)
            if path is not None:
                yield path
            original_router = getattr(route, "original_router", None)
            if original_router is not None:
                yield from registered_paths(original_router.routes)

    paths = set(registered_paths(create_app().routes))

    assert {
        "/api/campaigns/{campaign_id}/recovery",
        "/api/campaigns/{campaign_id}/recovery/{action}",
    } <= paths


def test_projection_requires_authentication_and_exact_workspace_scope(tmp_path):
    http, tokens, _recovery = _client(tmp_path)
    assert (
        http.get(
            "/api/campaigns/campaign-1/recovery", params={"workspaceId": "workspace-a"}
        ).status_code
        == 401
    )

    response = http.get(
        "/api/campaigns/campaign-1/recovery",
        params={"workspaceId": "workspace-a"},
        headers=_headers(tokens["desktop"]),
    )
    assert response.status_code == 200
    assert response.json()["schema_version"] == "campaign_recovery.v1"
    wrong = http.get(
        "/api/campaigns/campaign-1/recovery",
        params={"workspaceId": "workspace-b"},
        headers=_headers(tokens["desktop"]),
    )
    assert wrong.status_code == 403


def test_human_confirmed_mutation_replays_but_agent_principal_is_denied(tmp_path):
    http, tokens, _recovery = _client(tmp_path)
    snapshot = http.get(
        "/api/campaigns/campaign-1/recovery",
        params={"workspaceId": "workspace-a"},
        headers=_headers(tokens["desktop"]),
    ).json()
    payload = _payload(snapshot)

    denied = http.post(
        "/api/campaigns/campaign-1/recovery/resume",
        headers=_headers(tokens["codex"]),
        json=payload,
    )
    assert denied.status_code == 403
    first = http.post(
        "/api/campaigns/campaign-1/recovery/resume",
        headers=_headers(tokens["desktop"]),
        json=payload,
    )
    second = http.post(
        "/api/campaigns/campaign-1/recovery/resume",
        headers=_headers(tokens["desktop"]),
        json=payload,
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.headers["X-BashGym-Replayed"] == "false"
    assert second.headers["X-BashGym-Replayed"] == "true"

    reconciled = http.get(
        "/api/campaigns/campaign-1/recovery",
        params={"workspaceId": "workspace-a"},
        headers=_headers(tokens["desktop"]),
    )
    assert reconciled.status_code == 200
    assert reconciled.json()["latest_execution"] == {
        "schema_version": "campaign_recovery_execution.v1",
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "action": "resume",
        "status": "accepted",
        "outcome_code": None,
        "attempt_id": None,
    }
    assert len(json.dumps(reconciled.json()["latest_execution"])) < 512


def test_server_revalidates_current_doctor_state_and_never_trusts_client_freshness(tmp_path):
    http, tokens, recovery = _client(tmp_path)
    snapshot = http.get(
        "/api/campaigns/campaign-1/recovery",
        params={"workspaceId": "workspace-a"},
        headers=_headers(tokens["desktop"]),
    ).json()
    recovery.register_binding(
        installation_id=INSTALLATION,
        kind="compute",
        logical_id="registered-private-compute",
        availability="inaccessible",
    )
    payload = _payload(snapshot)
    payload["freshness"] = "live"
    invalid = http.post(
        "/api/campaigns/campaign-1/recovery/resume",
        headers=_headers(tokens["desktop"]),
        json=payload,
    )
    assert invalid.status_code == 422

    payload.pop("freshness")
    stale = http.post(
        "/api/campaigns/campaign-1/recovery/resume",
        headers=_headers(tokens["desktop"]),
        json=payload,
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "campaign_recovery_conflict"


def test_validation_and_conflict_errors_are_bounded_and_do_not_echo_canaries(tmp_path):
    http, tokens, _recovery = _client(tmp_path)
    response = http.post(
        "/api/campaigns/campaign-1/recovery/resume",
        headers=_headers(tokens["desktop"]),
        json={"workspaceId": "workspace-a", "password": "super-secret-must-not-leak"},
    )
    assert response.status_code == 422
    encoded = json.dumps(response.json())
    assert "super-secret" not in encoded
    assert len(encoded) < 500

    unsafe_query = http.get(
        "/api/campaigns/campaign-1/recovery",
        params={"workspaceId": "../query-secret-must-not-leak"},
        headers=_headers(tokens["desktop"]),
    )
    unsafe_action = http.post(
        "/api/campaigns/campaign-1/recovery/action-secret-must-not-leak",
        headers=_headers(tokens["desktop"]),
        json={},
    )
    assert unsafe_query.status_code == unsafe_action.status_code == 422
    assert "must-not-leak" not in json.dumps(unsafe_query.json())
    assert "must-not-leak" not in json.dumps(unsafe_action.json())

    root_canary = "root-list-secret-must-not-leak"
    root_list = http.post(
        "/api/campaigns/campaign-1/recovery/resume",
        headers=_headers(tokens["desktop"]),
        json=[root_canary],
    )
    assert root_list.status_code == 422
    assert root_canary not in json.dumps(root_list.json())
