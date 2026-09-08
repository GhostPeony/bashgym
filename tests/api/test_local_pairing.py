from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api import database
from bashgym.api.auth_routes import router


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "auth.db")
    database.init_db()
    app = FastAPI()
    app.include_router(router)
    with TestClient(app, base_url="http://localhost") as client:
        yield client


def test_pairing_single_use_http_only_session(client):
    code = database.issue_local_pairing("credential", 1)
    response = client.post(
        "/api/auth/local/pair",
        json={"code": code},
        headers={"Origin": "http://localhost", "X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 200
    assert "httponly" in response.headers["set-cookie"].lower()
    assert database.get_session_user(client.cookies.get("bashgym_session")) is not None
    assert (
        client.post(
            "/api/auth/local/pair",
            json={"code": code},
            headers={"Origin": "http://localhost", "X-Requested-With": "XMLHttpRequest"},
        ).status_code
        == 401
    )


def test_expired_code_fails(client):
    code = database.issue_local_pairing("credential", 1, ttl_seconds=-1)
    assert database.consume_local_pairing(code) is None


def test_pairing_rejects_cross_origin(client):
    code = database.issue_local_pairing("credential", 1)
    response = client.post(
        "/api/auth/local/pair",
        json={"code": code},
        headers={"Origin": "https://attacker.invalid", "X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 403
    assert database.consume_local_pairing(code) is not None


def test_concurrent_exchange_has_one_winner(client):
    code = database.issue_local_pairing("credential", 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(database.consume_local_pairing, [code, code]))
    assert sum(result is not None for result in results) == 1


def test_paired_browser_campaign_scope_and_revocation(tmp_path, monkeypatch):
    from bashgym.api.auth import AuthMiddleware
    from bashgym.campaigns.contracts import AutonomyProfile
    from tests.api.test_campaign_routes import campaign_client

    monkeypatch.setenv("BASHGYM_MODE", "headless")
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "auth.db")
    database.init_db()
    http, repository, refresh = campaign_client(tmp_path, profile=AutonomyProfile.DESKTOP_USER)
    http.app.include_router(router)
    http.app.add_middleware(AuthMiddleware)
    assert http.get("/api/campaign-auth/capabilities").status_code == 401
    code = database.issue_local_pairing(refresh.credential_id, 1)
    assert (
        http.post(
            "/api/auth/local/pair",
            json={"code": code},
            headers={"X-Requested-With": "XMLHttpRequest"},
        ).status_code
        == 200
    )
    capabilities = http.get("/api/campaign-auth/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["workspace_ids"] == ["workspace-a"]
    assert http.get("/api/campaigns", params={"workspace_id": "workspace-b"}).status_code == 403
    http.app.state.campaign_auth_service.revoke_credential(refresh.credential_id, reason="test")
    assert http.get("/api/campaign-auth/capabilities").status_code == 401


@pytest.mark.parametrize("change", ["revoke", "revision", "expire"])
def test_current_user_rechecks_local_authority_and_clears_stale_session(
    tmp_path, monkeypatch, change
):
    from datetime import timedelta

    from bashgym.campaigns import auth as campaign_auth
    from bashgym.campaigns.contracts import AutonomyProfile
    from tests.api.test_campaign_routes import campaign_client

    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "auth.db")
    database.init_db()
    http, _, refresh = campaign_client(tmp_path, profile=AutonomyProfile.DESKTOP_USER)
    http.app.include_router(router)
    token = database.consume_local_pairing(database.issue_local_pairing(refresh.credential_id, 1))
    http.cookies.set("bashgym_session", token)
    assert http.get("/api/auth/me").status_code == 200
    authority = http.app.state.campaign_auth_service
    if change == "revoke":
        authority.revoke_credential(refresh.credential_id, reason="test")
    elif change == "revision":
        authority.revise_credential_authorization(
            refresh.credential_id,
            autonomy_profile=AutonomyProfile.DESKTOP_USER,
            workspace_ids=("workspace-b",),
        )
    else:
        later = campaign_auth.utc_now() + timedelta(days=365)
        monkeypatch.setattr(campaign_auth, "utc_now", lambda: later)
    assert database.get_session_user(token) is not None
    response = http.get("/api/auth/me")
    assert response.status_code == 401
    assert "max-age=0" in response.headers["set-cookie"].lower()
    assert database.get_session_user(token) is None


def test_current_user_retains_github_session_behavior(client):
    user_id = database.upsert_user(github_id=123, username="fixture-user")
    client.cookies.set("bashgym_session", database.create_session(user_id))
    response = client.get("/api/auth/me")
    assert response.status_code == 200
    assert response.json()["github_id"] == 123


def test_headless_websocket_rejects_anonymous_client(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    from bashgym.api.websocket import handle_websocket

    monkeypatch.setenv("BASHGYM_MODE", "headless")
    app = FastAPI()
    app.websocket("/ws")(handle_websocket)
    with TestClient(app) as http:
        with pytest.raises(WebSocketDisconnect) as error:
            with http.websocket_connect("/ws"):
                pass
    assert error.value.code == 4401


def test_revoked_local_grant_cannot_open_websocket(tmp_path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    from bashgym.api.websocket import handle_websocket
    from bashgym.campaigns.contracts import AutonomyProfile
    from tests.api.test_campaign_routes import campaign_client

    monkeypatch.setenv("BASHGYM_MODE", "headless")
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "auth.db")
    database.init_db()
    http, repository, refresh = campaign_client(tmp_path, profile=AutonomyProfile.DESKTOP_USER)
    http.app.include_router(router)
    http.app.websocket("/ws")(handle_websocket)
    code = database.issue_local_pairing(refresh.credential_id, 1)
    assert (
        http.post(
            "/api/auth/local/pair",
            json={"code": code},
            headers={"X-Requested-With": "XMLHttpRequest"},
        ).status_code
        == 200
    )
    http.app.state.campaign_auth_service.revoke_credential(refresh.credential_id, reason="test")
    with pytest.raises(WebSocketDisconnect) as error:
        with http.websocket_connect("/ws"):
            pass
    assert error.value.code == 4401


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["personal", "broadcast", "topic"])
async def test_active_local_websocket_rechecks_revocation_before_sending(
    tmp_path, monkeypatch, channel
):
    from unittest.mock import AsyncMock, Mock

    from bashgym.api.websocket import ConnectionManager, WSMessage
    from bashgym.campaigns.contracts import AutonomyProfile
    from tests.api.test_campaign_routes import campaign_client

    monkeypatch.setenv("BASHGYM_MODE", "headless")
    monkeypatch.setattr(database, "_DB_PATH", tmp_path / "auth.db")
    database.init_db()
    http, repository, refresh = campaign_client(tmp_path, profile=AutonomyProfile.DESKTOP_USER)
    token = database.consume_local_pairing(database.issue_local_pairing(refresh.credential_id, 1))
    socket = Mock(app=http.app, cookies={"bashgym_session": token}, headers={})
    socket.accept = AsyncMock()
    socket.send_text = AsyncMock()
    socket.close = AsyncMock()
    manager = ConnectionManager()
    await manager.connect(socket)
    manager.subscribe(socket, "training")
    socket.send_text.reset_mock()
    http.app.state.campaign_auth_service.revoke_credential(refresh.credential_id, reason="test")
    message = WSMessage(type="training:log", payload={"log": "private training fixture"})
    if channel == "personal":
        await manager.send_personal(socket, message)
    elif channel == "broadcast":
        await manager.broadcast(message)
    else:
        await manager.broadcast_to_topic("training", message)
    socket.send_text.assert_not_awaited()
    socket.close.assert_awaited_once()
    assert socket not in manager.active_connections
