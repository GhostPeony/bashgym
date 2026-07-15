from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from bashgym.campaigns.client import CampaignClientError
from bashgym.mcp.campaign_server import build_server
from bashgym.mcp.client_runtime import McpClientRuntime

ROOT = Path(__file__).resolve().parents[2]
PROHIBITED_SCOPE_FIELDS = {
    "workspace_id",
    "workspace",
    "credential_ref",
    "credential",
    "actor",
    "agent",
    "profile",
    "autonomy_profile",
    "capabilities",
}


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query=None,
        payload=None,
        headers=None,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "query": query,
                "payload": payload,
                "headers": headers,
            }
        )
        if path == "/campaigns":
            return {"campaigns": [{"campaign_id": f"campaign-{index}"} for index in range(5)]}
        if path.endswith("/events"):
            return {
                "items": [
                    {"cursor": index, "event": {"event_id": f"event-{index}"}}
                    for index in range(1, 5)
                ],
                "next_cursor": 4,
            }
        if path.endswith("/evidence"):
            return {
                "campaign_id": "campaign-1",
                "artifact_references": [f"evidence-{index}" for index in range(150)],
            }
        if path.endswith("/artifacts"):
            return {"artifacts": [{"artifact_id": f"artifact-{index}"} for index in range(5)]}
        if path.endswith("/proposals") and method == "GET":
            return {"proposals": [{"proposal_id": f"proposal-{index}"} for index in range(5)]}
        if "/studies/" in path and method == "GET":
            return {"study_id": path.rsplit("/", 1)[-1], "status": "running"}
        if path.endswith("/attempts"):
            return {"attempts": [{"attempt_id": f"attempt-{index}"} for index in range(5)]}
        if path.endswith("/comparisons"):
            return {
                "comparisons": [
                    {"comparison_id": f"comparison-{index}"} for index in range(5)
                ]
            }
        if "/manifest/" in path:
            return {"revision": 2, "manifest": {"budget_limits": {"GPU_HOURS": 4}}}
        if path.endswith("/metrics"):
            return {
                "metric_name": "loss",
                "source": "metrics.jsonl",
                "values": [{"step": index, "value": 1 / index} for index in range(1, 5)],
                "next_after_step": 4,
            }
        if method == "POST":
            return {
                "campaign": {"campaign_id": "campaign-1", "version": 3},
                "event": {"event_id": "event-transition"},
                "replayed": False,
            }
        return {"campaign_id": "campaign-1", "workspace_id": "workspace-a"}


async def call_tool(server, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return await server._tool_manager.call_tool(name, arguments, convert_result=False)


async def test_campaign_stdio_server_exposes_only_launch_scoped_contract():
    runtime = McpClientRuntime()
    connected = await runtime.connect_stdio(
        "campaigns",
        sys.executable,
        [
            "-m",
            "bashgym.mcp.campaign_server",
            "--workspace-id",
            "workspace-a",
            "--credential-ref",
            "BASHGYM_CAMPAIGN_REFRESH",
            "--agent",
            "codex",
        ],
        cwd=str(ROOT),
        environment={"PYTHONPATH": str(ROOT)},
    )
    try:
        tools = {tool["name"]: tool for tool in connected["inventory"]["tools"]}
        assert set(tools) == {
            "campaign_list",
            "campaign_inspect",
            "campaign_manifest",
            "campaign_evidence",
            "campaign_artifacts",
            "campaign_proposals",
            "campaign_studies",
            "campaign_study",
            "campaign_attempts",
            "campaign_comparisons",
            "campaign_events",
            "campaign_metrics",
            "campaign_create_from_template",
            "campaign_create",
            "campaign_revise",
            "campaign_propose_study",
            "campaign_withdraw_proposal",
            "campaign_start",
            "campaign_advance",
            "campaign_pause",
            "campaign_resume",
            "campaign_cancel",
            "campaign_conclude",
            "campaign_retry",
            "campaign_abandon_study",
            "campaign_amend_budget",
            "campaign_approve_source",
            "campaign_force_stop",
            "campaign_protected_lease",
            "campaign_protected_result",
            "campaign_promote",
            "campaign_export",
        }
        for tool in tools.values():
            properties = set(tool["inputSchema"].get("properties", {}))
            assert properties.isdisjoint(PROHIBITED_SCOPE_FIELDS)

        assert tools["campaign_list"]["annotations"]["readOnlyHint"] is True
        assert tools["campaign_metrics"]["annotations"]["openWorldHint"] is False
        assert tools["campaign_start"]["annotations"]["destructiveHint"] is False
        assert tools["campaign_start"]["annotations"]["openWorldHint"] is True
        assert tools["campaign_cancel"]["annotations"]["destructiveHint"] is True
        assert tools["campaign_cancel"]["annotations"]["idempotentHint"] is True
        assert tools["campaign_force_stop"]["annotations"]["destructiveHint"] is True
        assert "confirmed" in tools["campaign_force_stop"]["inputSchema"]["required"]
        assert "pid" not in tools["campaign_force_stop"]["inputSchema"]["properties"]
        assert "command" not in tools["campaign_force_stop"]["inputSchema"]["properties"]
        assert tools["campaign_list"]["inputSchema"]["properties"]["limit"]["maximum"] == 100
        assert tools["campaign_attempts"]["inputSchema"]["properties"]["limit"]["maximum"] == 100
        assert tools["campaign_comparisons"]["inputSchema"]["properties"]["limit"]["maximum"] == 100
        assert tools["campaign_events"]["inputSchema"]["properties"]["limit"]["maximum"] == 200
        assert tools["campaign_metrics"]["inputSchema"]["properties"]["limit"]["maximum"] == 1000
    finally:
        await runtime.aclose()


async def test_campaign_tools_bind_workspace_bound_arrays_and_mutation_headers():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    listed = await call_tool(server, "campaign_list", {"limit": 2})
    evidence = await call_tool(
        server,
        "campaign_evidence",
        {"campaign_id": "campaign-1"},
    )
    study = await call_tool(
        server,
        "campaign_study",
        {"campaign_id": "campaign-1", "study_id": "study-2"},
    )
    attempts = await call_tool(
        server,
        "campaign_attempts",
        {"campaign_id": "campaign-1", "limit": 2},
    )
    comparisons = await call_tool(
        server,
        "campaign_comparisons",
        {"campaign_id": "campaign-1", "limit": 2},
    )
    events = await call_tool(
        server,
        "campaign_events",
        {"campaign_id": "campaign-1", "after_cursor": 8, "limit": 2},
    )
    metrics = await call_tool(
        server,
        "campaign_metrics",
        {
            "campaign_id": "campaign-1",
            "attempt_id": "attempt-1",
            "metric_name": "loss",
            "source": "metrics.jsonl",
            "after_step": 12,
            "limit": 2,
        },
    )
    started = await call_tool(
        server,
        "campaign_start",
        {"campaign_id": "campaign-1", "expected_version": 2},
    )
    cancelled = await call_tool(
        server,
        "campaign_cancel",
        {
            "campaign_id": "campaign-1",
            "expected_version": 3,
            "reason": "Operator ended this bounded study.",
        },
    )

    assert listed == {
        "ok": True,
        "campaigns": [{"campaign_id": "campaign-0"}, {"campaign_id": "campaign-1"}],
        "count": 2,
        "truncated": True,
    }
    assert len(evidence["evidence"]["artifact_references"]) == 100
    assert study == {
        "ok": True,
        "study": {"study_id": "study-2", "status": "running"},
    }
    assert attempts == {
        "ok": True,
        "attempts": [{"attempt_id": "attempt-0"}, {"attempt_id": "attempt-1"}],
        "count": 2,
        "truncated": True,
    }
    assert comparisons == {
        "ok": True,
        "comparisons": [
            {"comparison_id": "comparison-0"},
            {"comparison_id": "comparison-1"},
        ],
        "count": 2,
        "truncated": True,
    }
    assert len(events["items"]) == 2 and events["truncated"] is True
    assert len(metrics["values"]) == 2 and metrics["truncated"] is True
    assert started["ok"] is True and cancelled["ok"] is True

    for call in client.calls:
        query = call["query"] or {}
        payload = call["payload"] or {}
        assert query.get("workspace_id", payload.get("workspace_id")) == "workspace-a"
        assert PROHIBITED_SCOPE_FIELDS.isdisjoint(
            set(query).difference({"workspace_id"}) | set(payload).difference({"workspace_id"})
        )

    start_call = next(call for call in client.calls if call["path"].endswith("/start"))
    assert start_call["payload"] == {"workspace_id": "workspace-a", "expected_version": 2}
    assert set(start_call["headers"]) == {"Idempotency-Key", "X-Correlation-ID"}
    assert "codex" not in " ".join(start_call["headers"].values())

    cancel_call = next(call for call in client.calls if call["path"].endswith("/cancel"))
    assert cancel_call["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 3,
        "stop_reason": "Operator ended this bounded study.",
    }


async def test_campaign_extended_tools_use_strict_paths_bodies_and_persisted_identity():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    await call_tool(
        server,
        "campaign_create_from_template",
        {"campaign_id": "campaign-2", "title": "Embedding cycle", "template_id": "embed-v1"},
    )
    await call_tool(
        server,
        "campaign_force_stop",
        {
            "campaign_id": "campaign-2",
            "action_id": "action-7",
            "expected_version": 9,
            "expected_remote_process_identity": {
                "compute_profile_id": "ssh-gpu-lab",
                "remote_run_id": "run-4",
                "pid": 812,
                "process_start_time": "2026-07-13T09:00:00Z",
                "command_hash": "a" * 64,
            },
            "reason": "Reconciled worker remained alive after cancellation.",
            "confirmed": True,
        },
    )
    await call_tool(
        server,
        "campaign_protected_result",
        {
            "campaign_id": "campaign-2",
            "expected_version": 10,
            "protected_epoch_id": "protected-1",
            "candidate_digest": "c" * 64,
            "passed": True,
            "metrics": {"recall_at_10": 0.84},
            "artifact_sha256": "d" * 64,
        },
    )
    await call_tool(
        server,
        "campaign_export",
        {
            "campaign_id": "campaign-2",
            "expected_version": 11,
            "formats": ["markdown", "csv", "pdf"],
        },
    )

    create_call, force_call, protected_call, export_call = client.calls
    assert create_call["path"] == "/campaigns/from-template"
    assert create_call["payload"] == {
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-2",
        "title": "Embedding cycle",
        "template_id": "embed-v1",
    }
    assert force_call["path"] == "/campaigns/campaign-2/actions/action-7/force-stop"
    assert set(force_call["payload"]) == {
        "workspace_id",
        "expected_version",
        "expected_remote_process_identity",
        "confirmed",
        "reason",
    }
    assert force_call["payload"]["confirmed"] is True
    assert protected_call["path"] == "/campaigns/campaign-2/protected-result"
    assert protected_call["payload"]["result"]["candidate_digest"] == "c" * 64
    assert export_call["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 11,
        "formats": ["markdown", "csv", "pdf"],
    }
    for call in client.calls:
        assert set(call["headers"]) == {"Idempotency-Key", "X-Correlation-ID"}


@pytest.mark.parametrize("scope_field", sorted(PROHIBITED_SCOPE_FIELDS))
async def test_campaign_tool_callers_cannot_inject_launch_scope(scope_field: str):
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(
        server,
        "campaign_start",
        {
            "campaign_id": "campaign-1",
            "expected_version": 2,
            scope_field: "injected",
        },
    )
    assert result["ok"] is True
    assert len(client.calls) == 1
    if scope_field in {"workspace_id", "workspace"}:
        assert client.calls[0]["payload"].get("workspace_id") == "workspace-a"
        assert "injected" not in client.calls[0]["payload"].values()
    else:
        assert scope_field not in client.calls[0]["payload"]
    assert scope_field not in client.calls[0]["headers"]


async def test_campaign_tools_return_secret_free_client_errors():
    class FailingClient(RecordingClient):
        def request_json(self, method: str, path: str, **kwargs) -> Any:
            raise CampaignClientError(
                "campaign_scope_denied",
                "Campaign operation is not permitted.",
                status_code=403,
            )

    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="hermes",
        client=FailingClient(),
    )
    result = await call_tool(server, "campaign_inspect", {"campaign_id": "campaign-1"})

    assert result == {
        "ok": False,
        "error": {
            "code": "campaign_scope_denied",
            "message": "Campaign operation is not permitted.",
            "retryable": False,
            "status_code": 403,
        },
    }
    assert "BASHGYM_CAMPAIGN_REFRESH" not in repr(result)


async def test_campaign_mutation_rejects_an_invalid_api_projection():
    class InvalidClient(RecordingClient):
        def request_json(self, method: str, path: str, **kwargs) -> Any:
            return ["not", "an", "object"]

    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=InvalidClient(),
    )
    result = await call_tool(
        server,
        "campaign_start",
        {"campaign_id": "campaign-1", "expected_version": 2},
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "campaign_response_invalid"


@pytest.mark.parametrize(
    ("workspace_id", "credential_ref", "agent"),
    [
        ("workspace with spaces", "BASHGYM_CAMPAIGN_REFRESH", "codex"),
        ("workspace-a", "bgcr.raw.token", "codex"),
        ("workspace-a", "BASHGYM_CAMPAIGN_REFRESH", "agent with spaces"),
    ],
)
def test_campaign_server_rejects_unsafe_launch_scope(
    workspace_id: str,
    credential_ref: str,
    agent: str,
):
    with pytest.raises((ValueError, RuntimeError)):
        build_server(
            workspace_id=workspace_id,
            credential_ref=credential_ref,
            agent=agent,
            client=RecordingClient(),
        )
