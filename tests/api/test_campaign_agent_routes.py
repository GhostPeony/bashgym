"""REST tests for the secret-free campaign-agent broker boundary."""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api.campaign_agent_routes import (
    campaign_agent_credential_router,
    campaign_agent_router,
)
from bashgym.api.campaign_routes import campaign_auth_router
from bashgym.api.routes import create_app
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.campaign_agent_contracts import CampaignAgentFamily
from bashgym.campaigns.campaign_agent_sessions import (
    CampaignAgentSessionOriginVerifier,
    EncryptedCampaignAgentCredentialBroker,
)
from bashgym.campaigns.campaign_agents import BrokeredCampaignAgentCredential
from bashgym.campaigns.contracts import (
    AutonomyProfile,
    Campaign,
    CampaignKind,
    CampaignManifest,
    CredentialKind,
    ManifestRevision,
    TargetModelContract,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository


def _seed(repository: CampaignRuntimeRepository, campaign_id="campaign-1"):
    campaign = Campaign(
        workspace_id="workspace-a",
        campaign_id=campaign_id,
        title="Operator-selected model campaign",
        kind=CampaignKind.GENERAL,
        objective="Improve an explicitly bound trainable model.",
        target_model=TargetModelContract(
            target_contract_key="operator-model-binding-v1",
            base_model_ref="registry://trainable/model@immutable-revision",
            task="operator-selected-task",
        ),
        owner_actor_id="desktop-user",
    )
    manifest = CampaignManifest(
        approved_data_scopes=("approved-data",),
        compute_profile_id="registered-private-compute",
        budget_limits={"gpu_hours": 1.0},
        evaluation_plan={"suite": "registered-eval"},
        promotion_gates={"primary_metric": 0.0},
    )
    repository.create_campaign(
        campaign,
        ManifestRevision(
            workspace_id="workspace-a",
            campaign_id=campaign_id,
            revision=1,
            manifest=manifest,
            actor_id="desktop-user",
            correlation_id=f"create-{campaign_id}",
        ),
        actor_id="desktop-user",
        credential_kind=CredentialKind.ACCESS,
        correlation_id=f"create-{campaign_id}",
        idempotency_key=f"create-{campaign_id}",
    )


def _client(tmp_path, *, broker=True, verified=True):
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    _seed(repository)
    _seed(repository, "campaign-2")
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="desktop-user",
        autonomy_profile=AutonomyProfile.DESKTOP_USER,
        workspace_ids=("workspace-a",),
        ttl=timedelta(hours=1),
    )
    delivered = []
    app = FastAPI()
    app.state.campaign_repository = repository
    app.state.campaign_auth_service = auth
    app.state.campaign_human_seal_key = b"campaign-agent-route-seal-key-v1"
    app.state.campaign_human_seal_key_version = "campaign-agent-route-seal-v1"
    if verified:
        app.state.campaign_agent_origin_verifier = (
            lambda scope, family, origin, session, principal: (
                scope.workspace_id == "workspace-a"
                and scope.campaign_id == "campaign-1"
                and origin == "codex-origin"
                and session == "session-1"
                and principal == "codex-agent"
            )
        )
    if broker:
        app.state.campaign_agent_credential_broker = delivered.append
    app.include_router(campaign_auth_router)
    app.include_router(campaign_agent_router)
    app.include_router(campaign_agent_credential_router)
    http = TestClient(app)
    exchange = http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {refresh.raw_token}"},
    )
    assert exchange.status_code == 200
    return http, exchange.json()["raw_token"], delivered


def _desktop_bootstrap_client(
    tmp_path,
    monkeypatch,
    *,
    managed: bool,
    repository: CampaignRuntimeRepository | None = None,
    auth: CampaignAuthService | None = None,
    with_authority: bool = True,
):
    new_repository = repository is None
    repository = repository or CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    if new_repository:
        _seed(repository)
    auth = auth or CampaignAuthService(repository)
    bootstrap = "bgcb.managed-desktop-launch." + "desktop-bootstrap-secret-material-123456789"
    monkeypatch.setenv("BASHGYM_DESKTOP_BOOTSTRAP_SECRET", bootstrap)
    auth.install_desktop_bootstrap(bootstrap)
    app = FastAPI()
    app.state.campaign_repository = repository
    app.state.campaign_auth_service = auth
    app.state.campaign_worker_managed = managed
    app.state.campaign_desktop_bootstrap_checked = True
    if with_authority:
        app.state.campaign_human_seal_key = b"campaign-agent-route-seal-key-v1"
        app.state.campaign_human_seal_key_version = "campaign-agent-route-seal-v1"
    app.include_router(campaign_auth_router)
    app.include_router(campaign_agent_router)
    app.include_router(campaign_agent_credential_router)
    return TestClient(app), repository, auth, bootstrap


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _session_public_key():
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return private, base64.urlsafe_b64encode(public).decode().rstrip("=")


def _session_payload(public_key):
    return {
        "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
        "agentFamily": "codex",
        "agentOrigin": "codex-origin",
        "agentPrincipalId": "codex-agent",
        "sessionId": "session-1",
        "ephemeralPublicKey": public_key,
        "ttlSeconds": 300,
        "idempotencyKey": "register-session-1",
    }


def _grant_payload(key="grant-1"):
    return {
        "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
        "agentFamily": "codex",
        "agentOrigin": "codex-origin",
        "agentPrincipalId": "codex-agent",
        "sessionId": "session-1",
        "requestedCapabilities": ["campaign_observe", "training_launch"],
        "grantedCapabilities": ["campaign_observe"],
        "idempotencyKey": key,
    }


def _attach_payload(receipt, key="attach-1", base=None):
    return {
        "action": "attach",
        "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
        "agentFamily": "codex",
        "agentOrigin": "codex-origin",
        "agentPrincipalId": "codex-agent",
        "sessionId": "session-1",
        "requestedCapabilities": ["campaign_observe", "training_launch"],
        "grantedCapabilities": ["campaign_observe"],
        "confirmationReceipt": receipt,
        "baseAttachmentVersion": base,
        "idempotencyKey": key,
    }


def _agent_heartbeat(http, token):
    return http.post(
        "/api/campaign-agent/heartbeat",
        headers=_bearer(token),
        json={
            "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
            "agentFamily": "codex",
            "agentOrigin": "codex-origin",
            "agentPrincipalId": "codex-agent",
            "sessionId": "session-1",
        },
    )


def test_create_app_registers_campaign_agent_routes():
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
        "/api/campaigns/{campaign_id}/agent-grant",
        "/api/campaigns/{campaign_id}/agent-attachment",
        "/api/campaigns/{campaign_id}/agent-attachment/{attachment_id}/revoke",
        "/api/campaigns/{campaign_id}/agent-attachment/audit",
        "/api/campaigns/{campaign_id}/agent-attachment/receipts",
        "/api/campaign-agent/heartbeat",
        "/api/campaign-agent/actions/observe",
        "/api/campaign-agent/actions/artifacts",
    } <= paths


def test_grant_attach_get_and_heartbeat_never_return_bearer(tmp_path):
    http, access, delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    )
    assert grant.status_code == 200
    assert set(grant.json()) == {
        "schemaVersion",
        "issuer",
        "receiptId",
        "receiptDigest",
        "humanPrincipal",
        "scope",
        "agentFamily",
        "agentOrigin",
        "agentPrincipalId",
        "sessionId",
        "requestedCapabilities",
        "grantedCapabilities",
        "capabilityDigest",
        "grantRevision",
        "issuedAt",
        "expiresAt",
    }

    attach = http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant.json()),
    )
    assert attach.status_code == 200
    assert attach.headers["X-BashGym-Replayed"] == "false"
    assert len(delivered) == 1
    secret = delivered[0].raw_token
    assert secret not in attach.text
    assert "raw_token" not in attach.text

    public = http.get(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert public.status_code == 200
    assert public.json()["scope"] == attach.json()["scope"]
    assert public.json()["attachment"] == attach.json()["attachment"]
    assert public.json()["audit_events"] == attach.json()["audit_events"]
    assert secret not in public.text

    heartbeat = http.post(
        "/api/campaign-agent/heartbeat",
        headers=_bearer(secret),
        json={
            "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
            "agentFamily": "codex",
            "agentOrigin": "codex-origin",
            "agentPrincipalId": "codex-agent",
            "sessionId": "session-1",
            "resumeCursor": "public-cursor-1",
            "resumeSequence": 1,
            "expectedResumeCursor": None,
        },
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["attachment"]["provenance"]["resume_cursor"] == "public-cursor-1"
    assert secret not in heartbeat.text


def test_observe_action_derives_scope_and_principal_from_bearer(tmp_path):
    http, access, delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    ).json()
    attached = http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant),
    )
    assert attached.status_code == 200
    secret = delivered[0].raw_token
    assert _agent_heartbeat(http, secret).status_code == 200

    observed = http.get(
        "/api/campaign-agent/actions/observe",
        headers=_bearer(secret),
    )

    assert observed.status_code == 200
    assert observed.json() == {
        "schemaVersion": "campaign_agent_observation.v1",
        "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
        "campaign": {
            "status": "draft",
            "version": 1,
            "manifestRevision": 1,
            "activeStudyId": None,
            "activeActionId": None,
            "latestEventCursor": 1,
        },
        "agent": {
            "attachmentId": attached.json()["attachment"]["attachment_id"],
            "attachmentVersion": 1,
            "agentFamily": "codex",
            "agentPrincipalId": "codex-agent",
            "authorizedCapability": "campaign_observe",
        },
    }
    assert secret not in observed.text


def test_observe_action_ignores_spoofed_scope_and_capability_inputs(tmp_path):
    http, access, delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    ).json()
    http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant),
    )
    assert _agent_heartbeat(http, delivered[0].raw_token).status_code == 200

    response = http.get(
        "/api/campaign-agent/actions/observe",
        headers={
            **_bearer(delivered[0].raw_token),
            "X-Agent-Principal": "spoofed-root",
            "X-Agent-Capability": "promotion_override",
        },
        params={
            "workspace_id": "sibling-workspace",
            "campaign_id": "campaign-2",
            "agent_family": "hermes",
        },
    )

    assert response.status_code == 200
    assert response.json()["scope"] == {
        "workspaceId": "workspace-a",
        "campaignId": "campaign-1",
    }
    assert response.json()["agent"]["agentPrincipalId"] == "codex-agent"
    assert response.json()["agent"]["authorizedCapability"] == "campaign_observe"
    assert "spoofed" not in response.text
    assert "promotion_override" not in response.text


def test_observe_action_requires_the_fixed_observe_capability(tmp_path):
    http, access, delivered = _client(tmp_path)
    grant_payload = _grant_payload()
    grant_payload["requestedCapabilities"] = ["training_launch"]
    grant_payload["grantedCapabilities"] = ["training_launch"]
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=grant_payload,
    ).json()
    attach_payload = _attach_payload(grant)
    attach_payload["requestedCapabilities"] = ["training_launch"]
    attach_payload["grantedCapabilities"] = ["training_launch"]
    assert (
        http.post(
            "/api/campaigns/campaign-1/agent-attachment",
            headers=_bearer(access),
            json=attach_payload,
        ).status_code
        == 200
    )
    secret = delivered[0].raw_token
    assert _agent_heartbeat(http, secret).status_code == 200

    response = http.get(
        "/api/campaign-agent/actions/observe",
        headers=_bearer(secret),
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "campaign_agent_authorization_denied"
    assert secret not in response.text


def test_observe_action_fails_closed_when_host_liveness_is_lost(tmp_path):
    http, access, delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    ).json()
    http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant),
    )
    secret = delivered[0].raw_token
    assert _agent_heartbeat(http, secret).status_code == 200
    http.app.state.campaign_agent_origin_verifier = lambda *_args: False

    response = http.get(
        "/api/campaign-agent/actions/observe",
        headers=_bearer(secret),
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "campaign_agent_authorization_denied"
    assert secret not in response.text


def test_observe_action_fails_closed_after_revocation_or_expiry(tmp_path, monkeypatch):
    http, access, delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    ).json()
    attached = http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant),
    ).json()
    secret = delivered[0].raw_token
    assert _agent_heartbeat(http, secret).status_code == 200

    monkeypatch.setattr(
        "bashgym.campaigns.campaign_agents.utc_now",
        lambda: datetime.now(UTC) + timedelta(days=1),
    )
    expired = http.get(
        "/api/campaign-agent/actions/observe",
        headers=_bearer(secret),
    )
    assert expired.status_code == 401
    assert secret not in expired.text

    monkeypatch.undo()
    revoked = http.post(
        f"/api/campaigns/campaign-1/agent-attachment/{attached['attachment']['attachment_id']}/revoke",
        headers=_bearer(access),
        json={
            "action": "revoke",
            "attachmentId": attached["attachment"]["attachment_id"],
            "attachmentVersion": 1,
            "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
            "actorId": "desktop-user",
            "idempotencyKey": "revoke-before-observe",
        },
    )
    assert revoked.status_code == 200
    denied = http.get(
        "/api/campaign-agent/actions/observe",
        headers=_bearer(secret),
    )
    assert denied.status_code == 401
    assert secret not in denied.text


def test_observe_action_requires_a_current_agent_heartbeat(tmp_path):
    http, access, delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    ).json()
    http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant),
    )
    secret = delivered[0].raw_token

    response = http.get(
        "/api/campaign-agent/actions/observe",
        headers=_bearer(secret),
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "campaign_agent_authorization_denied"
    assert secret not in response.text


def test_artifact_read_action_returns_only_the_public_bounded_projection(tmp_path):
    http, access, delivered = _client(tmp_path)
    grant_payload = _grant_payload()
    grant_payload["requestedCapabilities"] = ["artifact_read"]
    grant_payload["grantedCapabilities"] = ["artifact_read"]
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=grant_payload,
    ).json()
    attach_payload = _attach_payload(grant)
    attach_payload["requestedCapabilities"] = ["artifact_read"]
    attach_payload["grantedCapabilities"] = ["artifact_read"]
    assert (
        http.post(
            "/api/campaigns/campaign-1/agent-attachment",
            headers=_bearer(access),
            json=attach_payload,
        ).status_code
        == 200
    )
    secret = delivered[0].raw_token
    assert _agent_heartbeat(http, secret).status_code == 200
    private_uri = "file://private/campaign/output.jsonl"
    private_metadata = "PRIVATE-ARTIFACT-METADATA-CANARY"
    with sqlite3.connect(tmp_path / "campaigns.sqlite3") as connection:
        connection.execute(
            """INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id,
                uri, sha256, size_bytes, schema_name, sealed, valid,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "workspace-a",
                "campaign-1",
                "artifact-public-1",
                None,
                private_uri,
                "a" * 64,
                42,
                "campaign_training_log.v1",
                1,
                1,
                json.dumps({"private": private_metadata}),
                "2026-07-17T00:00:00Z",
            ),
        )

    response = http.get(
        "/api/campaign-agent/actions/artifacts",
        headers=_bearer(secret),
        params={"limit": 1},
    )

    assert response.status_code == 200
    assert response.json() == {
        "schemaVersion": "campaign_agent_artifact_page.v1",
        "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
        "items": [
            {
                "artifactId": "artifact-public-1",
                "producerActionId": None,
                "sha256": "a" * 64,
                "sizeBytes": 42,
                "schemaName": "campaign_training_log.v1",
                "sealed": True,
                "valid": True,
                "createdAt": "2026-07-17T00:00:00Z",
            }
        ],
        "nextCursor": None,
        "hasMore": False,
    }
    assert secret not in response.text
    assert private_uri not in response.text
    assert private_metadata not in response.text


def test_mutating_action_proxy_routes_remain_absent_until_fixed_contracts_exist(tmp_path):
    """TODO: add each route only with its own server-owned action contract.

    Training launch still needs a server-selected runnable action plus a bounded
    compute request; pause-self needs durable ownership/CAS semantics; artifact
    propose needs an allowlisted schema and authority-owned sealing. A generic
    route forwarder must never fill those contract gaps.
    """

    http, _access, _delivered = _client(tmp_path)
    for path in (
        "/api/campaign-agent/actions/training-launch",
        "/api/campaign-agent/actions/training-pause-self",
        "/api/campaign-agent/actions/artifact-propose",
        "/api/campaign-agent/actions/forward",
    ):
        response = http.post(path, json={"target": "/api/campaigns/campaign-1/start"})
        assert response.status_code == 404


def test_action_adapters_accept_campaign_agent_bearers_only_in_authorization_header(
    tmp_path,
):
    http, _access, _delivered = _client(tmp_path)
    canary = "bgag.query-body.SECRET-BEARER-CANARY-12345678901234567890"

    query_attempt = http.get(
        "/api/campaign-agent/actions/observe",
        params={"access_token": canary},
    )
    body_attempt = http.request(
        "GET",
        "/api/campaign-agent/actions/artifacts",
        json={"token": canary},
    )

    for response in (query_attempt, body_attempt):
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "campaign_auth_required"
        assert canary not in response.text


def test_attach_fails_closed_when_desktop_broker_is_unavailable(tmp_path):
    http, access, _delivered = _client(tmp_path, broker=False)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    )
    response = http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant.json()),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "campaign_agent_broker_unavailable"
    public = http.get(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert public.json()["attachment"] is None


def test_route_scope_actor_spoof_and_sibling_access_fail_closed(tmp_path):
    http, access, _delivered = _client(tmp_path)
    mismatch = http.post(
        "/api/campaigns/campaign-2/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    )
    assert mismatch.status_code == 403

    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    )
    attach = http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant.json()),
    ).json()
    revoke = http.post(
        f"/api/campaigns/campaign-1/agent-attachment/{attach['attachment']['attachment_id']}/revoke",
        headers=_bearer(access),
        json={
            "action": "revoke",
            "attachmentId": attach["attachment"]["attachment_id"],
            "attachmentVersion": 1,
            "scope": {"workspaceId": "workspace-a", "campaignId": "campaign-1"},
            "actorId": "spoofed-human",
            "idempotencyKey": "revoke-1",
        },
    )
    assert revoke.status_code == 403

    sibling = http.get(
        "/api/campaigns/campaign-2/agent-attachment",
        headers=_bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert sibling.status_code == 200
    assert sibling.json()["attachment"] is None
    assert "codex-origin" not in json.dumps(sibling.json())


def test_transport_rejects_unknown_fields_and_noncanonical_identifiers(tmp_path):
    http, access, _delivered = _client(tmp_path)
    unknown = _grant_payload()
    unknown["secret"] = "API-SECRET-CANARY"
    response = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=unknown,
    )
    assert response.status_code == 422
    assert "API-SECRET-CANARY" not in response.text

    root_list = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=["ROOT-LIST-SECRET-CANARY"],
    )
    assert root_list.status_code == 422
    assert "ROOT-LIST-SECRET-CANARY" not in root_list.text

    semantic = _grant_payload("semantic-invalid")
    semantic["requestedCapabilities"] = ["campaign_observe", "campaign_observe"]
    response = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=semantic,
    )
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "campaign_agent_request_invalid",
        "message": "Campaign-agent request validation failed.",
    }

    malformed = _grant_payload()
    malformed["scope"]["workspaceId"] = "-workspace-a"
    response = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=malformed,
    )
    assert response.status_code == 422


def test_audit_and_receipt_pages_expose_explicit_resume_metadata(tmp_path):
    http, access, _delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    ).json()
    http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant),
    )

    audit = http.get(
        "/api/campaigns/campaign-1/agent-attachment/audit",
        headers=_bearer(access),
        params={"workspace_id": "workspace-a", "after_sequence": 0, "limit": 1},
    )
    receipts = http.get(
        "/api/campaigns/campaign-1/agent-attachment/receipts",
        headers=_bearer(access),
        params={"workspace_id": "workspace-a", "after_version": 0, "limit": 1},
    )

    assert audit.status_code == 200
    assert set(audit.json()) == {"items", "next_cursor", "has_more"}
    assert audit.json()["next_cursor"] == audit.json()["items"][-1]["sequence"]
    assert receipts.status_code == 200
    assert receipts.json()["next_cursor"] == 1


def test_tampered_attachment_projection_fails_closed_without_echoing_data(tmp_path):
    http, access, _delivered = _client(tmp_path)
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    ).json()
    attached = http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant),
    )
    assert attached.status_code == 200

    with sqlite3.connect(tmp_path / "campaigns.sqlite3") as connection:
        connection.execute(
            """UPDATE campaign_agent_attachments
               SET agent_origin = ?
               WHERE workspace_id = ? AND campaign_id = ?""",
            ("TAMPERED-ORIGIN-CANARY", "workspace-a", "campaign-1"),
        )

    response = http.get(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        params={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "campaign_agent_integrity_failed",
        "message": "Campaign-agent authority state could not be verified.",
    }
    assert "TAMPERED-ORIGIN-CANARY" not in response.text


def test_desktop_session_routes_do_not_enable_host_bindings_implicitly(tmp_path):
    http, access, _delivered = _client(tmp_path, broker=False, verified=False)
    _private, public_key = _session_public_key()

    registered = http.post(
        "/api/campaign-agent/sessions",
        headers=_bearer(access),
        json=_session_payload(public_key),
    )

    assert registered.status_code == 201
    assert set(registered.json()) == {
        "schemaVersion",
        "registrationId",
        "scope",
        "agentFamily",
        "agentOrigin",
        "agentPrincipalId",
        "sessionId",
        "publicKeyDigest",
        "registeredAt",
        "expiresAt",
        "status",
    }
    assert not hasattr(http.app.state, "campaign_agent_origin_verifier")
    assert not hasattr(http.app.state, "campaign_agent_credential_broker")
    denied = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload(),
    )
    assert denied.status_code == 403


def test_encrypted_delivery_claim_and_revoke_routes_are_secret_safe(tmp_path):
    http, access, _delivered = _client(tmp_path, broker=False, verified=False)
    _private, public_key = _session_public_key()
    registered = http.post(
        "/api/campaign-agent/sessions",
        headers=_bearer(access),
        json=_session_payload(public_key),
    ).json()
    registration_id = registered["registrationId"]
    raw_token = "bgag.credential-route.SECRET-ROUTE-CANARY"
    now = datetime.now(UTC).replace(microsecond=0)
    broker = EncryptedCampaignAgentCredentialBroker(
        http.app.state.campaign_agent_session_repository,
        clock=lambda: now,
    )
    broker(
        BrokeredCampaignAgentCredential(
            attachment_id="attachment-route",
            credential_id="credential-route",
            raw_token=raw_token,
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            agent_family=CampaignAgentFamily.CODEX,
            agent_origin="codex-origin",
            session_id="session-1",
            agent_principal_id="codex-agent",
            granted_capabilities=(),
            authorization_revision=1,
            issued_at=now,
            expires_at=now + timedelta(minutes=5),
        )
    )

    claimed = http.post(
        f"/api/campaign-agent/sessions/{registration_id}/deliveries/claim",
        headers=_bearer(access),
        json={},
    )
    assert claimed.status_code == 200
    assert raw_token not in claimed.text
    assert claimed.json()["algorithm"] == "X25519-HKDF-SHA256+CHACHA20-POLY1305"
    assert (
        http.post(
            f"/api/campaign-agent/sessions/{registration_id}/deliveries/claim",
            headers=_bearer(access),
            json={},
        ).status_code
        == 409
    )

    revoked = http.post(
        f"/api/campaign-agent/sessions/{registration_id}/revoke",
        headers=_bearer(access),
        json={},
    )
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"


def test_desktop_bootstrap_exchange_does_not_activate_bindings_on_unmanaged_server(
    tmp_path, monkeypatch
):
    http, _repository, _auth, bootstrap = _desktop_bootstrap_client(
        tmp_path, monkeypatch, managed=False
    )

    exchange = http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )
    assert exchange.status_code == 200
    access = exchange.json()["raw_token"]
    _private, public_key = _session_public_key()
    registered = http.post(
        "/api/campaign-agent/sessions",
        headers=_bearer(access),
        json=_session_payload(public_key),
    )

    assert registered.status_code == 201
    assert not hasattr(http.app.state, "campaign_agent_origin_verifier")
    assert not hasattr(http.app.state, "campaign_agent_credential_broker")
    denied = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload("unmanaged-grant"),
    )
    assert denied.status_code == 403


def test_managed_server_requires_current_bootstrap_exchange_before_activation(
    tmp_path, monkeypatch
):
    http, _repository, auth, bootstrap = _desktop_bootstrap_client(
        tmp_path, monkeypatch, managed=True
    )
    refresh = auth.issue_refresh_credential(
        actor_id="desktop-user",
        autonomy_profile=AutonomyProfile.DESKTOP_USER,
        workspace_ids=("workspace-a",),
        ttl=timedelta(hours=1),
    )
    access = auth.exchange_refresh(refresh.raw_token).raw_token
    _private, public_key = _session_public_key()
    assert (
        http.post(
            "/api/campaign-agent/sessions",
            headers=_bearer(access),
            json=_session_payload(public_key),
        ).status_code
        == 201
    )
    assert not hasattr(http.app.state, "campaign_agent_origin_verifier")

    monkeypatch.setenv(
        "BASHGYM_DESKTOP_BOOTSTRAP_SECRET",
        "bgcb.current-desktop-launch." + "current-bootstrap-secret-material-123456789",
    )
    old_exchange = http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )
    assert old_exchange.status_code == 200
    assert not hasattr(http.app.state, "campaign_agent_origin_verifier")
    assert (
        http.post(
            "/api/campaigns/campaign-1/agent-grant",
            headers=_bearer(old_exchange.json()["raw_token"]),
            json=_grant_payload("stale-bootstrap-grant"),
        ).status_code
        == 403
    )


def test_authenticated_managed_desktop_exchange_activates_cached_service_bindings(
    tmp_path, monkeypatch
):
    http, _repository, auth, bootstrap = _desktop_bootstrap_client(
        tmp_path, monkeypatch, managed=True
    )
    refresh = auth.issue_refresh_credential(
        actor_id="desktop-user",
        autonomy_profile=AutonomyProfile.DESKTOP_USER,
        workspace_ids=("workspace-a",),
        ttl=timedelta(hours=1),
    )
    ordinary_access = auth.exchange_refresh(refresh.raw_token).raw_token
    cached = http.get(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(ordinary_access),
        params={"workspace_id": "workspace-a"},
    )
    assert cached.status_code == 200
    cached_service = http.app.state.campaign_agent_service

    exchange = http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )

    assert exchange.status_code == 200
    assert http.app.state.campaign_agent_service is cached_service
    assert isinstance(
        http.app.state.campaign_agent_origin_verifier,
        CampaignAgentSessionOriginVerifier,
    )
    assert isinstance(
        http.app.state.campaign_agent_credential_broker,
        EncryptedCampaignAgentCredentialBroker,
    )
    assert cached_service.origin_verifier is http.app.state.campaign_agent_origin_verifier
    assert cached_service.credential_broker is http.app.state.campaign_agent_credential_broker

    access = exchange.json()["raw_token"]
    _private, public_key = _session_public_key()
    registered = http.post(
        "/api/campaign-agent/sessions",
        headers=_bearer(access),
        json=_session_payload(public_key),
    )
    assert registered.status_code == 201
    grant = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload("managed-grant"),
    )
    assert grant.status_code == 200
    attached = http.post(
        "/api/campaigns/campaign-1/agent-attachment",
        headers=_bearer(access),
        json=_attach_payload(grant.json(), key="managed-attach"),
    )
    assert attached.status_code == 200
    claimed = http.post(
        f"/api/campaign-agent/sessions/{registered.json()['registrationId']}/deliveries/claim",
        headers=_bearer(access),
        json={},
    )
    assert claimed.status_code == 200
    assert "raw_token" not in claimed.text
    assert bootstrap not in claimed.text


def test_managed_desktop_bindings_lazy_initialize_after_restart_and_honor_revocation(
    tmp_path, monkeypatch
):
    first, repository, auth, bootstrap = _desktop_bootstrap_client(
        tmp_path, monkeypatch, managed=True
    )
    first_exchange = first.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )
    assert first_exchange.status_code == 200
    _private, public_key = _session_public_key()
    registered = first.post(
        "/api/campaign-agent/sessions",
        headers=_bearer(first_exchange.json()["raw_token"]),
        json=_session_payload(public_key),
    )
    assert registered.status_code == 201

    restarted, _repository, _auth, _bootstrap = _desktop_bootstrap_client(
        tmp_path,
        monkeypatch,
        managed=True,
        repository=repository,
        auth=auth,
    )
    assert not hasattr(restarted.app.state, "campaign_agent_session_repository")
    restart_exchange = restarted.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )
    assert restart_exchange.status_code == 200
    access = restart_exchange.json()["raw_token"]
    grant = restarted.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload("restart-grant"),
    )
    assert grant.status_code == 200

    revoked = restarted.post(
        f"/api/campaign-agent/sessions/{registered.json()['registrationId']}/revoke",
        headers=_bearer(access),
        json={},
    )
    assert revoked.status_code == 200
    denied = restarted.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload("revoked-grant"),
    )
    assert denied.status_code == 403


def test_managed_desktop_session_tamper_fails_closed_without_echo(tmp_path, monkeypatch):
    http, repository, _auth, bootstrap = _desktop_bootstrap_client(
        tmp_path, monkeypatch, managed=True
    )
    exchange = http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )
    access = exchange.json()["raw_token"]
    _private, public_key = _session_public_key()
    assert (
        http.post(
            "/api/campaign-agent/sessions",
            headers=_bearer(access),
            json=_session_payload(public_key),
        ).status_code
        == 201
    )
    canary = "TAMPERED-DESKTOP-SESSION-PRIVATE-CANARY"
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE campaign_agent_host_sessions SET agent_origin=?",
            (canary,),
        )

    denied = http.post(
        "/api/campaigns/campaign-1/agent-grant",
        headers=_bearer(access),
        json=_grant_payload("tampered-session-grant"),
    )

    assert denied.status_code == 403
    assert canary not in denied.text
    assert bootstrap not in denied.text


def test_managed_desktop_activation_failure_is_safe_and_never_returns_access_secret(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("BASHGYM_CAMPAIGN_SEAL_KEY", raising=False)
    monkeypatch.setattr("bashgym.api.campaign_routes.get_secret", lambda _key: None)
    http, _repository, _auth, bootstrap = _desktop_bootstrap_client(
        tmp_path,
        monkeypatch,
        managed=True,
        with_authority=False,
    )

    response = http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "campaign_agent_authority_unavailable",
        "message": "The campaign-agent desktop authority is unavailable.",
    }
    assert bootstrap not in response.text
    assert "raw_token" not in response.text
