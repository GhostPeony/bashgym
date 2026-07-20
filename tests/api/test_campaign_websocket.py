from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect

from bashgym._compat import UTC
from bashgym.api.websocket import (
    CampaignHintV1,
    ConnectionManager,
    build_campaign_hint,
    handle_websocket,
    manager,
)
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import AutonomyProfile, CampaignTrigger, CredentialKind
from bashgym.campaigns.persistence import CampaignHintSource, CampaignRepository
from tests.api.test_campaign_routes import bearer, campaign_client, exchange
from tests.campaigns.test_persistence import campaign, revision


def test_campaign_hint_is_exact_low_entropy_projection_at_send_time() -> None:
    source = CampaignHintSource(
        cursor=41,
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        aggregate_version=7,
        event_type="campaign:created",
        correlation_id="correlation-safe",
    )
    hint = build_campaign_hint(
        source, emitted_at=datetime(2026, 7, 16, 18, 0, tzinfo=UTC)
    )

    assert isinstance(hint, CampaignHintV1)
    assert hint.model_dump(mode="json") == {
        "schema_version": "campaign_hint.v1",
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-a",
        "event_cursor": 41,
        "aggregate_version": 7,
        "event_type": "campaign:created",
        "correlation_id": "correlation-safe",
        "emitted_at": "2026-07-16T18:00:00Z",
    }


def test_campaign_hint_accepts_canonical_colon_identifiers() -> None:
    source = CampaignHintSource(
        cursor=1,
        workspace_id="workspace:a",
        campaign_id="campaign:a",
        aggregate_version=1,
        event_type="campaign:created",
        correlation_id="correlation:a",
    )

    hint = build_campaign_hint(source)

    assert hint.workspace_id == "workspace:a"
    assert hint.campaign_id == "campaign:a"
    assert hint.correlation_id == "correlation:a"


def test_workspace_hint_source_query_is_payload_free_and_cursor_bounded(tmp_path) -> None:
    repository = CampaignRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    created = repository.create_campaign(
        campaign(),
        revision(campaign()),
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="correlation-create",
        idempotency_key="create-campaign",
    )
    repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.VALIDATE,
        expected_version=created.campaign.version,
        actor_id="campaign-controller",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id="correlation-worker",
        idempotency_key="worker-validate",
    )

    sources = repository.list_workspace_campaign_hint_sources(
        "workspace-a", after_cursor=0, limit=10
    )

    assert [source.event_type for source in sources] == [
        "campaign:created",
        "campaign:validation-started",
    ]
    assert sources[-1].correlation_id == "correlation-worker"
    assert not hasattr(sources[-1], "payload")
    assert repository.latest_workspace_campaign_event_cursor("workspace-a") == sources[-1].cursor
    assert repository.list_workspace_campaign_hint_sources(
        "workspace-b", after_cursor=0, limit=10
    ) == ()


def test_live_ticket_requires_workspace_read_and_is_single_use(tmp_path) -> None:
    http, repository, _refresh = campaign_client(tmp_path)
    refresh = CampaignAuthService(repository).issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a", "workspace-b"),
    )
    token = exchange(http, refresh.raw_token)

    denied = http.post("/api/campaigns/live-ticket", json={"workspace_id": "workspace-a"})
    assert denied.status_code == 401
    response = http.post(
        "/api/campaigns/live-ticket",
        headers=bearer(token),
        json={"workspace_id": "workspace-a"},
    )

    assert response.status_code == 200
    assert set(response.json()) == {
        "schema_version",
        "ticket",
        "workspace_id",
        "after_cursor",
        "expires_at",
    }
    assert response.json()["schema_version"] == "campaign_live_ticket.v1"
    assert response.json()["workspace_id"] == "workspace-a"
    assert response.json()["ticket"].startswith("bgclt.")
    assert "bgca." not in response.text
    consumed = manager.consume_campaign_live_ticket(response.json()["ticket"])
    assert consumed is not None
    assert consumed.workspace_id == "workspace-a"
    assert manager.consume_campaign_live_ticket(response.json()["ticket"]) is None


def test_live_ticket_route_uses_canonical_workspace_identifier_grammar(tmp_path) -> None:
    http, repository, _refresh = campaign_client(tmp_path)
    refresh = CampaignAuthService(repository).issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace:a",),
    )
    token = exchange(http, refresh.raw_token)

    accepted = http.post(
        "/api/campaigns/live-ticket",
        headers=bearer(token),
        json={"workspace_id": "workspace:a"},
    )
    rejected = http.post(
        "/api/campaigns/live-ticket",
        headers=bearer(token),
        json={"workspace_id": "-workspace"},
    )

    assert accepted.status_code == 200
    assert accepted.json()["workspace_id"] == "workspace:a"
    assert rejected.status_code == 422
    assert "bgclt." not in rejected.text


def test_live_ticket_endpoint_rate_limits_one_credential_workspace_scope(tmp_path) -> None:
    http, repository, _refresh = campaign_client(tmp_path)
    refresh = CampaignAuthService(repository).issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    token = exchange(http, refresh.raw_token)

    responses = [
        http.post(
            "/api/campaigns/live-ticket",
            headers=bearer(token),
            json={"workspace_id": "workspace-a"},
        )
        for _ in range(31)
    ]

    assert [response.status_code for response in responses[:30]] == [200] * 30
    assert responses[-1].status_code == 429
    assert responses[-1].json()["detail"]["code"] == "campaign_live_ticket_rate_limited"
    assert "bgclt." not in responses[-1].text
    assert set(responses[-1].json()) == {"detail"}


class _InteractiveSocket:
    def __init__(self) -> None:
        self.cookies: dict[str, str] = {}
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.outgoing: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, message: str) -> None:
        await self.outgoing.put(json.loads(message))

    async def receive_text(self) -> str:
        message = await self.incoming.get()
        if message is None:
            raise WebSocketDisconnect()
        return message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "payload"),
    [
        ([], None),
        (None, None),
        ("primitive", None),
        ({"type": "campaign:subscribe", "payload": None}, None),
        ({"type": "campaign:subscribe", "payload": "ticket"}, None),
    ],
)
async def test_malformed_frames_are_non_disclosing_and_always_cleanup(
    frame, payload, monkeypatch
) -> None:
    del payload
    monkeypatch.setenv("BASHGYM_MODE", "desktop")
    manager.active_connections.clear()
    manager.campaign_subscriptions.clear()
    socket = _InteractiveSocket()
    task = asyncio.create_task(handle_websocket(socket))
    assert (await asyncio.wait_for(socket.outgoing.get(), timeout=1))["type"] == "connected"

    await socket.incoming.put(json.dumps(frame))
    error = await asyncio.wait_for(socket.outgoing.get(), timeout=1)
    assert error == {
        "type": "error",
        "payload": {"code": "invalid_frame"},
        "timestamp": error["timestamp"],
    }
    await socket.incoming.put(None)
    await asyncio.wait_for(task, timeout=1)
    assert socket not in manager.active_connections
    assert socket not in manager.campaign_subscriptions


@pytest.mark.asyncio
async def test_campaign_poller_recovers_after_one_projection_failure(monkeypatch) -> None:
    local_manager = ConnectionManager()
    socket = object()
    local_manager.campaign_subscriptions[socket] = {"workspace:a": object()}  # type: ignore[index]
    calls = 0

    async def poll_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("transient projection failure")
        local_manager.campaign_subscriptions.clear()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(local_manager, "poll_campaign_subscriptions_once", poll_once)
    monkeypatch.setattr("bashgym.api.websocket.asyncio.sleep", no_sleep)

    await local_manager._campaign_poll_loop()

    assert calls == 2


def test_outstanding_ticket_memory_replaces_prior_scope_ticket() -> None:
    local_manager = ConnectionManager()
    now = datetime.now(UTC)
    parent = SimpleNamespace(
        credential_id="credential-a",
        authorization_revision=3,
        workspace_ids=("workspace:a",),
        revoked_at=None,
        expires_at=now.replace(year=now.year + 1),
    )
    repository = SimpleNamespace(get_actor_credential=lambda _credential_id: parent)
    principal = SimpleNamespace(
        credential_id=parent.credential_id,
        authorization_revision=parent.authorization_revision,
        expires_at=parent.expires_at,
    )
    issued = [
        local_manager.issue_campaign_live_ticket(
            repository,
            principal,
            "workspace:a",
            after_cursor=index,
        )[0]
        for index in range(20)
    ]

    assert len(local_manager.campaign_tickets) == 1
    assert local_manager.consume_campaign_live_ticket(issued[0]) is None
    assert local_manager.consume_campaign_live_ticket(issued[-1]) is not None


@pytest.mark.asyncio
async def test_ticket_expiry_revision_and_live_revocation_fail_closed(monkeypatch) -> None:
    local_manager = ConnectionManager()
    now = datetime.now(UTC)
    parent = SimpleNamespace(
        credential_id="credential-a",
        authorization_revision=1,
        workspace_ids=("workspace:a",),
        revoked_at=None,
        expires_at=now + timedelta(hours=1),
    )
    repository = SimpleNamespace(get_actor_credential=lambda _credential_id: parent)
    principal = SimpleNamespace(
        credential_id=parent.credential_id,
        authorization_revision=parent.authorization_revision,
        expires_at=parent.expires_at,
    )

    expired, _ = local_manager.issue_campaign_live_ticket(
        repository,
        principal,
        "workspace:a",
        after_cursor=0,
        ttl=timedelta(seconds=-1),
    )
    assert local_manager.consume_campaign_live_ticket(expired) is None

    revised, _ = local_manager.issue_campaign_live_ticket(
        repository, principal, "workspace:a", after_cursor=0
    )
    parent.authorization_revision = 2
    assert local_manager.consume_campaign_live_ticket(revised) is None

    principal.authorization_revision = 2
    active, _ = local_manager.issue_campaign_live_ticket(
        repository, principal, "workspace:a", after_cursor=0
    )
    socket = _InteractiveSocket()
    monkeypatch.setattr(local_manager, "_ensure_campaign_poller", lambda: None)
    assert await local_manager.subscribe_campaign(socket, active) is True
    parent.revoked_at = datetime.now(UTC)
    await local_manager.poll_campaign_subscriptions_once()
    assert socket not in local_manager.campaign_subscriptions


@pytest.mark.asyncio
async def test_consumed_subscription_survives_ticket_ttl_expiry(monkeypatch) -> None:
    local_manager = ConnectionManager()
    now = datetime.now(UTC)
    parent = SimpleNamespace(
        credential_id="credential-a",
        authorization_revision=1,
        workspace_ids=("workspace:a",),
        revoked_at=None,
        expires_at=now + timedelta(hours=1),
    )
    source = CampaignHintSource(
        cursor=1,
        workspace_id="workspace:a",
        campaign_id="campaign:a",
        aggregate_version=1,
        event_type="campaign:created",
        correlation_id="correlation-a",
    )
    repository = SimpleNamespace(
        get_actor_credential=lambda _credential_id: parent,
        list_workspace_campaign_hint_sources=lambda workspace_id, after_cursor, limit: (source,),
    )
    principal = SimpleNamespace(
        credential_id=parent.credential_id,
        authorization_revision=parent.authorization_revision,
        expires_at=parent.expires_at,
    )
    ticket, binding = local_manager.issue_campaign_live_ticket(
        repository, principal, "workspace:a", after_cursor=0
    )
    socket = _InteractiveSocket()
    monkeypatch.setattr(local_manager, "_ensure_campaign_poller", lambda: None)
    assert await local_manager.subscribe_campaign(socket, ticket) is True
    assert (await asyncio.wait_for(socket.outgoing.get(), timeout=1))["type"] == (
        "campaign:subscribed"
    )

    object.__setattr__(binding, "expires_at", now - timedelta(seconds=1))
    await local_manager.poll_campaign_subscriptions_once()

    assert socket in local_manager.campaign_subscriptions
    assert "workspace:a" in local_manager.campaign_subscriptions[socket]
    hint = await asyncio.wait_for(socket.outgoing.get(), timeout=1)
    assert hint["type"] == "campaign:hint"
    assert hint["payload"]["event_type"] == "campaign:created"

    parent.expires_at = now - timedelta(seconds=1)
    await local_manager.poll_campaign_subscriptions_once()
    assert socket not in local_manager.campaign_subscriptions


@pytest.mark.asyncio
async def test_one_socket_can_hold_independent_workspace_subscriptions(tmp_path) -> None:
    http, repository, _refresh = campaign_client(tmp_path)
    refresh = CampaignAuthService(repository).issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a", "workspace-b"),
    )
    token = exchange(http, refresh.raw_token)
    tickets = [
        http.post(
            "/api/campaigns/live-ticket",
            headers=bearer(token),
            json={"workspace_id": workspace_id},
        ).json()["ticket"]
        for workspace_id in ("workspace-a", "workspace-b")
    ]
    socket = _InteractiveSocket()

    assert await manager.subscribe_campaign(socket, tickets[0]) is True
    assert await manager.subscribe_campaign(socket, tickets[1]) is True
    assert set(manager.campaign_subscriptions[socket]) == {"workspace-a", "workspace-b"}

    manager.unsubscribe_campaign(socket, "workspace-a")
    assert set(manager.campaign_subscriptions[socket]) == {"workspace-b"}
    manager.disconnect(socket)


@pytest.mark.asyncio
async def test_ticket_subscription_observes_direct_worker_commit_and_reconnects(
    tmp_path, monkeypatch
) -> None:
    """Exercise the real handler/poller twice without touching the running desktop app."""

    monkeypatch.setenv("BASHGYM_MODE", "desktop")
    http, repository, refresh = campaign_client(tmp_path)
    token = exchange(http, refresh.raw_token)
    created = repository.create_campaign(
        campaign(),
        revision(campaign()),
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="correlation-create",
        idempotency_key="create-campaign",
    )
    version = created.campaign.version
    manager.active_connections.clear()
    manager.campaign_subscriptions.clear()
    triggers = (CampaignTrigger.VALIDATE, CampaignTrigger.VALIDATION_PASSED)

    for generation, trigger in enumerate(triggers, start=1):
        ticket_response = http.post(
            "/api/campaigns/live-ticket",
            headers=bearer(token),
            json={"workspace_id": "workspace-a"},
        )
        assert ticket_response.status_code == 200
        socket = _InteractiveSocket()
        task = asyncio.create_task(handle_websocket(socket))
        assert (await asyncio.wait_for(socket.outgoing.get(), timeout=1))["type"] == "connected"
        await socket.incoming.put(json.dumps({
            "type": "campaign:subscribe",
            "payload": {"ticket": ticket_response.json()["ticket"]},
        }))
        subscribed = await asyncio.wait_for(socket.outgoing.get(), timeout=1)
        assert subscribed["type"] == "campaign:subscribed"
        assert set(subscribed["payload"]) == {"workspace_id", "accepted_cursor"}

        mutation = repository.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="campaign-controller",
            credential_kind=CredentialKind.CONTROLLER,
            correlation_id=f"correlation-worker-{generation}",
            idempotency_key=f"worker-{generation}",
        )
        version = mutation.campaign.version
        await manager.poll_campaign_subscriptions_once()
        streamed = await asyncio.wait_for(socket.outgoing.get(), timeout=1)
        assert streamed["type"] == "campaign:hint"
        assert set(streamed["payload"]) == {
            "schema_version",
            "workspace_id",
            "campaign_id",
            "event_cursor",
            "aggregate_version",
            "event_type",
            "correlation_id",
            "emitted_at",
        }
        assert streamed["payload"]["correlation_id"] == f"correlation-worker-{generation}"
        await manager.poll_campaign_subscriptions_once()
        assert socket.outgoing.empty()

        await socket.incoming.put(None)
        await asyncio.wait_for(task, timeout=1)
        assert socket not in manager.campaign_subscriptions
