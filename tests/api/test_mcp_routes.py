from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api.mcp_routes import router
from bashgym.mcp.persistence import McpRepository
from bashgym.mcp.service import McpWorkbenchService


def _app(tmp_path: Path) -> FastAPI:
    app = FastAPI()
    service = McpWorkbenchService(
        McpRepository(tmp_path / "mcp.sqlite3"),
        workspace_root=Path.cwd(),
        secret_resolver=lambda _name: None,
    )
    service.initialize()
    app.state.mcp_workbench = service
    app.include_router(router)

    @app.on_event("shutdown")
    async def close_mcp() -> None:
        await service.aclose()

    return app


def _wait(client: TestClient, operation_id: str, workspace_id: str = "workspace-a") -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/mcp/operations/{operation_id}",
            params={"workspace_id": workspace_id},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"operation did not finish: {operation_id}")


def _create_reference_profile(client: TestClient) -> dict:
    response = client.post(
        "/api/mcp/profiles",
        json={
            "workspace_id": "workspace-a",
            "label": "Bundled MCP Reference",
            "transport": "stdio",
            "stdio": {
                "command": sys.executable,
                "args": ["-m", "bashgym.mcp.reference_server", "--transport", "stdio"],
                "cwd_policy": "workspace",
                "env_secret_refs": {},
                "sandbox_policy": "preferred",
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_profiles_are_secret_free_revisioned_and_workspace_scoped(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        created = client.post(
            "/api/mcp/profiles",
            json={
                "workspace_id": "workspace-a",
                "label": "Remote Search",
                "transport": "streamable_http",
                "remote": {
                    "url": "https://example.com/mcp",
                    "header_secret_refs": {"Authorization": "MCP_SEARCH_TOKEN"},
                },
            },
        )
        assert created.status_code == 201, created.text
        profile = created.json()
        serialized = created.text
        assert profile["profile_revision"] == 1
        assert profile["remote"]["header_secret_refs"] == {"Authorization": "MCP_SEARCH_TOKEN"}
        assert "Bearer " not in serialized
        assert "sk-" not in serialized

        oauth = client.get(
            f"/api/mcp/profiles/{profile['profile_id']}/oauth/status",
            params={"workspace_id": "workspace-a"},
        )
        assert oauth.status_code == 200, oauth.text
        assert oauth.json()["auth_mode"] == "auto"
        assert oauth.json()["interactive_oauth"] is True
        assert oauth.json()["has_tokens"] is False

        own = client.get("/api/mcp/profiles", params={"workspace_id": "workspace-a"})
        other = client.get("/api/mcp/profiles", params={"workspace_id": "workspace-b"})
        assert [item["profile_id"] for item in own.json()] == [profile["profile_id"]]
        assert other.json() == []

        updated = client.put(
            f"/api/mcp/profiles/{profile['profile_id']}",
            json={
                "workspace_id": "workspace-a",
                "expected_revision": 1,
                "label": "Remote Search v2",
                "transport": "streamable_http",
                "remote": {
                    "url": "https://example.com/mcp",
                    "header_secret_refs": {"Authorization": "MCP_SEARCH_TOKEN"},
                },
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["profile_revision"] == 2

        conflict = client.put(
            f"/api/mcp/profiles/{profile['profile_id']}",
            json={
                "workspace_id": "workspace-a",
                "expected_revision": 1,
                "label": "stale edit",
                "transport": "streamable_http",
                "remote": {"url": "https://example.com/mcp"},
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "revision_conflict"


def test_claude_config_preview_is_tolerant_and_never_echoes_raw_secrets(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        response = client.post(
            "/api/mcp/imports/claude/preview",
            json={
                "workspace_id": "workspace-a",
                "source_scope": "project",
                "config": {
                    "mcpServers": {
                        "safe": {
                            "type": "http",
                            "url": "https://example.test/mcp",
                            "headers": {"Authorization": "${MCP_SAFE_TOKEN}"},
                        },
                        "unsafe": {
                            "command": "python",
                            "env": {"TOKEN": "sk-do-not-echo"},
                        },
                    }
                },
            },
        )
        assert response.status_code == 200, response.text
        candidates = {item["server_name"]: item for item in response.json()}
        assert candidates["safe"]["supported"] is True
        assert candidates["safe"]["profile_input"]["remote"]["header_secret_refs"] == {
            "Authorization": "MCP_SAFE_TOKEN"
        }
        assert candidates["unsafe"]["supported"] is False
        assert "sk-do-not-echo" not in response.text


def test_stdio_reference_connect_snapshot_call_quick_test_and_cancel(
    tmp_path: Path,
) -> None:
    with TestClient(_app(tmp_path)) as client:
        profile = _create_reference_profile(client)
        profile_id = profile["profile_id"]
        revision = profile["profile_revision"]

        preview = client.get(
            f"/api/mcp/profiles/{profile_id}/stdio/preview",
            params={"workspace_id": "workspace-a", "profile_revision": revision},
        )
        assert preview.status_code == 200, preview.text
        preview_payload = preview.json()
        assert preview_payload["env_names"] == []
        approval = client.post(
            f"/api/mcp/profiles/{profile_id}/stdio/approve",
            json={
                "workspace_id": "workspace-a",
                "profile_revision": revision,
                "executable_sha256": preview_payload["executable"]["sha256"],
                "launch_fingerprint": preview_payload["launch_fingerprint"],
            },
        )
        assert approval.status_code == 200, approval.text

        accepted = client.post(
            f"/api/mcp/profiles/{profile_id}/connect",
            json={"workspace_id": "workspace-a", "profile_revision": revision},
        )
        assert accepted.status_code == 202, accepted.text
        connected = _wait(client, accepted.json()["operation_id"])
        assert connected["status"] == "succeeded", connected
        session_id = connected["result"]["session_id"]

        snapshot_response = client.get(
            f"/api/mcp/profiles/{profile_id}/snapshot",
            params={"workspace_id": "workspace-a"},
        )
        assert snapshot_response.status_code == 200, snapshot_response.text
        snapshot = snapshot_response.json()
        assert snapshot["negotiated_protocol_version"] == "2025-11-25"
        assert {tool["name"] for tool in snapshot["tools"]} >= {
            "read_fixture",
            "structured_sum",
            "slow_operation",
        }
        assert all(tool["policy"] == "allow" for tool in snapshot["tools"])

        called = client.post(
            f"/api/mcp/sessions/{session_id}/tools/read_fixture/call",
            json={"workspace_id": "workspace-a", "arguments": {"name": "alpha"}},
        )
        assert called.status_code == 202, called.text
        call_result = _wait(client, called.json()["operation_id"])
        assert call_result["status"] == "succeeded", call_result
        assert "Alpha is the first deterministic" in str(call_result["result"])

        quick = client.post(
            f"/api/mcp/profiles/{profile_id}/quick-test",
            json={"workspace_id": "workspace-a"},
        )
        assert quick.status_code == 202, quick.text
        quick_result = _wait(client, quick.json()["operation_id"])
        assert quick_result["status"] == "succeeded", quick_result
        assert quick_result["result"]["tool_count"] >= 5

        denied = client.get(
            f"/api/mcp/sessions/{session_id}",
            params={"workspace_id": "workspace-b"},
        )
        assert denied.status_code == 404

        slow_call = client.post(
            f"/api/mcp/sessions/{session_id}/tools/slow_operation/call",
            json={
                "workspace_id": "workspace-a",
                "arguments": {"delay_ms": 5000},
                "timeout_seconds": 10,
            },
        )
        assert slow_call.status_code == 202, slow_call.text
        time.sleep(0.2)
        cancelled = client.post(
            f"/api/mcp/operations/{slow_call.json()['operation_id']}/cancel",
            json={"workspace_id": "workspace-a"},
        )
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"

        lost_session = client.get(
            f"/api/mcp/sessions/{session_id}",
            params={"workspace_id": "workspace-a"},
        )
        assert lost_session.status_code == 200, lost_session.text
        assert lost_session.json()["state"] == "disconnected"
        assert lost_session.json()["stale"] is True

        stale_snapshot = client.get(
            f"/api/mcp/profiles/{profile_id}/snapshot",
            params={"workspace_id": "workspace-a"},
        )
        assert stale_snapshot.status_code == 200, stale_snapshot.text
        assert stale_snapshot.json()["stale"] is True


def test_unclassified_manual_tool_requires_explicit_one_time_approval(tmp_path: Path) -> None:
    app = _app(tmp_path)
    service: McpWorkbenchService = app.state.mcp_workbench
    # The policy decision itself is covered without creating an arbitrary external process.
    # A fake custom snapshot remains workspace-owned and cannot dispatch without approval.
    from bashgym.mcp.contracts import (
        McpCapabilitySnapshot,
        McpProfile,
        McpSession,
        SessionState,
        StreamableHttpTransport,
    )

    profile = McpProfile(
        profile_id="custom-profile",
        workspace_id="workspace-a",
        label="Custom MCP",
        transport=StreamableHttpTransport(url="https://example.com/mcp"),
    )
    service.repository.create_profile(profile)
    snapshot = McpCapabilitySnapshot(
        snapshot_id="custom-snapshot",
        workspace_id="workspace-a",
        profile_id=profile.profile_id,
        profile_revision=1,
        negotiated_protocol_version="2025-11-25",
        tools=[
            {
                "name": "custom_tool",
                "description": "A custom tool",
                "input_schema": {"type": "object"},
                "annotations": {},
                "policy": "ask",
            }
        ],
    )
    service.repository.save_snapshot(snapshot)
    session = McpSession(
        session_id="custom-session",
        workspace_id="workspace-a",
        profile_id=profile.profile_id,
        profile_revision=1,
        snapshot_id=snapshot.snapshot_id,
        state=SessionState.CONNECTED,
    )
    service.repository.create_session(session)

    with TestClient(app) as client:
        response = client.post(
            "/api/mcp/sessions/custom-session/tools/custom_tool/call",
            json={"workspace_id": "workspace-a", "arguments": {}},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "approval_required"
