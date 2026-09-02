from __future__ import annotations

import json
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
        timeout=None,
    ) -> Any:
        call = {
            "method": method,
            "path": path,
            "query": query,
            "payload": payload,
            "headers": headers,
        }
        if timeout is not None:
            call["timeout"] = timeout
        self.calls.append(call)
        if path == "/campaigns":
            return {"campaigns": [{"campaign_id": f"campaign-{index}"} for index in range(5)]}
        if path.endswith("/events"):
            return {
                "items": [
                    {
                        "cursor": index,
                        "event": {
                            "schema_version": "public_campaign_event.v1",
                            "event_id": f"event-{index}",
                            "workspace_id": "workspace-a",
                            "campaign_id": "campaign-1",
                            "sequence": index,
                            "aggregate_version": index,
                            "event_type": "campaign:created",
                            "actor_id": "codex-agent",
                            "credential_kind": "access",
                            "created_at": "2026-07-16T00:00:00Z",
                        },
                    }
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
            return {
                "artifacts": [
                    {
                        "schema_version": "public_campaign_artifact.v1",
                        "workspace_id": "workspace-a",
                        "campaign_id": "campaign-1",
                        "artifact_id": f"artifact-{index}",
                        "producer_action_id": None,
                        "sha256": f"{index + 1:064x}",
                        "size_bytes": index,
                        "schema_name": "training_metrics_jsonl.v1",
                        "sealed": True,
                        "valid": True,
                        "created_at": "2026-07-16T00:00:00Z",
                    }
                    for index in range(5)
                ],
                "next_cursor": None,
                "has_more": False,
            }
        if path.endswith("/proposals") and method == "GET":
            return {"proposals": [{"proposal_id": f"proposal-{index}"} for index in range(5)]}
        if "/studies/" in path and method == "GET":
            return {"study_id": path.rsplit("/", 1)[-1], "status": "running"}
        if path.endswith("/attempts"):
            return {"attempts": [{"attempt_id": f"attempt-{index}"} for index in range(5)]}
        if path.endswith("/comparisons"):
            return {"comparisons": [{"comparison_id": f"comparison-{index}"} for index in range(5)]}
        if "/manifest/" in path:
            return {"revision": 2, "manifest": {"budget_limits": {"GPU_HOURS": 4}}}
        if path.endswith("/metrics"):
            return {
                "metric_name": "loss",
                "source": "metrics.jsonl",
                "values": [{"step": index, "value": 1 / index} for index in range(1, 5)],
                "next_after_step": 4,
            }
        if path.endswith("/research-wait"):
            return {
                "schema_version": "bashgym.research_wait.v1",
                "status": "changed",
                "after_cursor": 3,
                "next_cursor": 4,
                "research": {
                    "schema_version": "bashgym.research.v1",
                    "campaign_id": "campaign-1",
                    "workspace_id": "workspace-a",
                    "next_action": "propose_candidate",
                },
            }
        if path.endswith("/research-failures"):
            return {
                "schema_version": "bashgym.research_failures.v1",
                "campaign_id": "campaign-1",
                "reference": None,
                "candidate": None,
                "comparison": [],
                "truncated": False,
            }
        if method == "POST":
            return {
                "campaign": {"campaign_id": "campaign-1", "version": 3},
                "event": {"event_id": "event-transition"},
                "replayed": False,
            }
        return {"campaign_id": "campaign-1", "workspace_id": "workspace-a"}


class LeakyEventClient(RecordingClient):
    def request_json(self, method: str, path: str, **kwargs) -> Any:
        if path.endswith("/events"):
            return {
                "items": [
                    {
                        "cursor": 7,
                        "event": {
                            "schema_version": "campaign_event.v1",
                            "event_id": "event-7",
                            "workspace_id": "workspace-a",
                            "campaign_id": "campaign-1",
                            "sequence": 7,
                            "aggregate_version": 3,
                            "event_type": "campaign:protected-evaluation-completed",
                            "payload": {
                                "reference": "protected-epoch-canary",
                                "result": "candidate-map-canary",
                                "location": "C:/operator/restricted-result.json",
                            },
                            "actor_id": "campaign-controller",
                            "credential_kind": "controller",
                            "correlation_id": "protected-eval-correlation-canary",
                            "idempotency_key": "protected-eval-idempotency-canary",
                            "created_at": "2026-07-16T00:00:00Z",
                        },
                    }
                ],
                "next_cursor": 7,
            }
        return super().request_json(method, path, **kwargs)


class LeakyArtifactClient(RecordingClient):
    def request_json(self, method: str, path: str, **kwargs) -> Any:
        if path.endswith("/artifacts"):
            return {
                "artifacts": [
                    {
                        "schema_version": "campaign_artifact_record.v1",
                        "workspace_id": "workspace-a",
                        "campaign_id": "campaign-1",
                        "artifact_id": "artifact-1",
                        "producer_action_id": "action-1",
                        "uri": "C:/operator/restricted-result.json",
                        "sha256": "a" * 64,
                        "size_bytes": 10,
                        "schema_name": "training_metrics_jsonl.v1",
                        "sealed": True,
                        "valid": True,
                        "metadata": {
                            "reference": "candidate-map-canary",
                            "nested": {"ordinary": "protected-epoch-canary"},
                        },
                        "created_at": "2026-07-16T00:00:00Z",
                    }
                ],
                "next_cursor": None,
                "has_more": False,
            }
        return super().request_json(method, path, **kwargs)


class PaginatedArtifactClient(RecordingClient):
    def request_json(self, method: str, path: str, **kwargs) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "query": kwargs.get("query"),
                "payload": kwargs.get("payload"),
                "headers": kwargs.get("headers"),
            }
        )
        if path.endswith("/artifacts"):
            return {
                "artifacts": [
                    {
                        "schema_version": "public_campaign_artifact.v1",
                        "workspace_id": "workspace-a",
                        "campaign_id": "campaign-1",
                        "artifact_id": f"artifact-{index}",
                        "producer_action_id": None,
                        "sha256": f"{index:064x}",
                        "size_bytes": index,
                        "schema_name": "training_metrics_jsonl.v1",
                        "sealed": True,
                        "valid": True,
                        "created_at": "2026-07-16T00:00:00Z",
                    }
                    for index in (5, 6)
                ],
                "next_cursor": None,
                "has_more": False,
            }
        raise AssertionError(f"unexpected request: {method} {path}")


class OversizedResearchClient(RecordingClient):
    def request_json(self, method: str, path: str, **kwargs) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "query": kwargs.get("query"),
                "payload": kwargs.get("payload"),
                "headers": kwargs.get("headers"),
            }
        )
        if path.endswith("/research-state"):
            deeply_nested: Any = "leaf"
            for _ in range(20):
                deeply_nested = {"nested": deeply_nested}
            return {
                "long_summary": "x" * 10_000,
                "many_items": list(range(200)),
                "many_fields": {f"field-{index}": index for index in range(200)},
                "deeply_nested": deeply_nested,
            }
        if path == "/campaigns/setup/context":
            return {f"section-{index}": "y" * 10_000 for index in range(100)}
        if path == "/campaigns":
            return {
                "campaigns": [
                    {
                        "campaign_id": f"campaign-{index}",
                        "summary": "z" * 10_000,
                    }
                    for index in range(100)
                ]
            }
        return super().request_json(method, path, **kwargs)


async def call_tool(server, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return await server._tool_manager.call_tool(name, arguments, convert_result=False)


async def test_research_failures_delegates_to_the_canonical_read_route():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(server, "research_failures", {"campaign_id": "campaign-1"})

    assert result["ok"] is True
    assert result["failures"]["schema_version"] == "bashgym.research_failures.v1"
    assert client.calls == [
        {
            "method": "GET",
            "path": "/campaigns/campaign-1/research-failures",
            "query": {"workspace_id": "workspace-a"},
            "payload": None,
            "headers": None,
        }
    ]


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
            "research_prepare",
            "research_context",
            "research_state",
            "research_failures",
            "research_wait",
            "research_start",
            "research_submit_iteration",
            "research_conclude_hypothesis_family",
            "research_report",
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
            "campaign_prepare_code_lineage",
            "campaign_capture_code_lineage",
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
        assert tools["research_prepare"]["annotations"]["readOnlyHint"] is True
        assert tools["research_context"]["annotations"]["readOnlyHint"] is True
        assert tools["research_state"]["annotations"]["readOnlyHint"] is True
        assert tools["research_failures"]["annotations"]["readOnlyHint"] is True
        assert tools["research_wait"]["annotations"]["readOnlyHint"] is True
        assert tools["research_wait"]["inputSchema"]["properties"]["after_cursor"]["minimum"] == 0
        assert tools["research_wait"]["inputSchema"]["properties"]["timeout_seconds"] == {
            "default": 30,
            "maximum": 55,
            "minimum": 1,
            "title": "Timeout Seconds",
            "type": "integer",
        }
        assert tools["research_start"]["annotations"]["openWorldHint"] is True
        assert tools["research_submit_iteration"]["annotations"]["openWorldHint"] is True
        assert tools["research_conclude_hypothesis_family"]["annotations"]["openWorldHint"] is True
        assert tools["research_report"]["annotations"]["openWorldHint"] is True
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


async def test_research_facade_delegates_to_canonical_campaign_api():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )
    proposal = {
        "proposal_id": "candidate-2",
        "hypothesis": "Verified recovery traces improve held-out completion.",
        "evidence_references": ["evaluation-1"],
        "study_family": "verified-trajectory-sft",
        "primary_variable": "dataset_recipe",
        "controlled_variables": ["evaluation_recipe"],
        "expected_outcome": "Held-out completion improves.",
        "falsification_criterion": "Held-out completion does not improve.",
        "estimated_cost": 0.5,
        "dataset_recipe": {"source": "verified-bashgym-trajectories"},
        "training_recipe": {"method": "sft"},
        "evaluation_recipe": {"suite": "terminal-heldout-v1"},
        "stage_plan": {"stages": ["full_training", "evaluation"]},
        "rationale": "Change one bounded variable after reviewing the baseline.",
    }

    prepared = await call_tool(
        server,
        "research_prepare",
        {"session_id": "setupsess_0123456789abcdef0123456789abcdef"},
    )
    state = await call_tool(server, "research_state", {"campaign_id": "campaign-1"})
    waited = await call_tool(
        server,
        "research_wait",
        {"campaign_id": "campaign-1", "after_cursor": 3, "timeout_seconds": 55},
    )
    started = await call_tool(
        server,
        "research_start",
        {"campaign_id": "campaign-1", "expected_version": 2},
    )
    submitted = await call_tool(
        server,
        "research_submit_iteration",
        {
            "campaign_id": "campaign-1",
            "expected_version": 3,
            "role": "candidate",
            "proposal": proposal,
            "parent_proposal_id": "baseline-1",
        },
    )
    reported = await call_tool(
        server,
        "research_report",
        {
            "campaign_id": "campaign-1",
            "expected_version": 4,
            "formats": ["markdown", "json"],
        },
    )

    assert prepared["ok"] is True and prepared["context"]["campaign_id"] == "campaign-1"
    assert state["ok"] is True and state["research"]["campaign_id"] == "campaign-1"
    assert waited["ok"] is True
    assert waited["wait"]["status"] == "changed"
    assert waited["wait"]["next_cursor"] == 4
    assert started["ok"] is True
    assert submitted["ok"] is True
    assert reported["ok"] is True
    prepare_call, state_call, wait_call, start_call, submit_call, report_call = client.calls
    assert prepare_call == {
        "method": "GET",
        "path": "/campaigns/setup/context",
        "query": {
            "workspace_id": "workspace-a",
            "session_id": "setupsess_0123456789abcdef0123456789abcdef",
        },
        "payload": None,
        "headers": None,
    }
    assert state_call["method"] == "GET"
    assert state_call["path"] == "/campaigns/campaign-1/research-state"
    assert state_call["query"] == {"workspace_id": "workspace-a"}
    assert wait_call == {
        "method": "GET",
        "path": "/campaigns/campaign-1/research-wait",
        "query": {
            "workspace_id": "workspace-a",
            "after_cursor": 3,
            "timeout_seconds": 55,
        },
        "payload": None,
        "headers": None,
        "timeout": 60,
    }
    assert start_call["path"] == "/campaigns/campaign-1/start"
    assert start_call["payload"] == {"workspace_id": "workspace-a", "expected_version": 2}
    assert submit_call["path"] == "/campaigns/campaign-1/autoresearch/candidates"
    assert submit_call["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 3,
        **proposal,
        "parent_proposal_id": "baseline-1",
    }
    assert report_call["path"] == "/campaigns/campaign-1/export"
    assert report_call["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 4,
        "formats": ["markdown", "json"],
    }
    for call in (start_call, submit_call, report_call):
        assert set(call["headers"]) == {"Idempotency-Key", "X-Correlation-ID"}


async def test_research_state_bounds_nested_api_payloads():
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=OversizedResearchClient(),
    )

    result = await call_tool(server, "research_state", {"campaign_id": "campaign-1"})

    assert result["ok"] is True
    research = result["research"]
    assert research["long_summary"].endswith("[truncated]")
    assert len(research["long_summary"]) <= 4096
    assert len(research["many_items"]) <= 100
    assert research["many_items"][-1] == "[truncated]"
    assert len(research["many_fields"]) <= 100
    assert research["many_fields"]["_truncated"] == "101 entries omitted"
    nested = research["deeply_nested"]
    for _ in range(8):
        nested = nested["nested"]
    assert nested == "[truncated]"
    assert len(json.dumps(result, separators=(",", ":")).encode("utf-8")) <= 65_536


async def test_research_prepare_enforces_a_total_serialized_payload_limit():
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=OversizedResearchClient(),
    )

    result = await call_tool(server, "research_prepare", {})

    assert result == {
        "ok": True,
        "context": {
            "_truncated": "payload exceeded 65536 serialized bytes",
        },
    }
    assert len(json.dumps(result, separators=(",", ":")).encode("utf-8")) <= 65_536


async def test_campaign_list_enforces_one_total_serialized_payload_limit():
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=OversizedResearchClient(),
    )

    result = await call_tool(server, "campaign_list", {"limit": 100})

    assert result == {
        "ok": True,
        "campaigns": [
            {"_truncated": "payload exceeded 65536 serialized bytes"},
        ],
        "count": 1,
        "truncated": True,
    }
    assert len(json.dumps(result, separators=(",", ":")).encode("utf-8")) <= 65_536


async def test_research_submit_iteration_routes_explicit_baseline_without_parent():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(
        server,
        "research_submit_iteration",
        {
            "campaign_id": "campaign-1",
            "expected_version": 2,
            "role": "baseline",
            "proposal": {"proposal_id": "baseline-1"},
        },
    )

    assert result["ok"] is True
    assert client.calls[0]["path"] == "/campaigns/campaign-1/autoresearch/baseline"
    assert client.calls[0]["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 2,
        "proposal_id": "baseline-1",
    }


async def test_research_submit_iteration_preserves_exploratory_intervention_bundle():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(
        server,
        "research_submit_iteration",
        {
            "campaign_id": "campaign-1",
            "expected_version": 7,
            "role": "candidate",
            "parent_proposal_id": "candidate-parent",
            "proposal": {
                "proposal_id": "candidate-exploratory",
                "primary_variable": "training_recipe.learning_rate",
                "intervention_mode": "exploratory",
                "changed_variables": [
                    "training_recipe.learning_rate",
                    "training_recipe.seed",
                ],
                "hypothesis_family_id": "family-optimizer-schedule",
            },
        },
    )

    assert result["ok"] is True
    assert client.calls[0]["path"] == "/campaigns/campaign-1/autoresearch/candidates"
    assert client.calls[0]["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 7,
        "parent_proposal_id": "candidate-parent",
        "proposal_id": "candidate-exploratory",
        "primary_variable": "training_recipe.learning_rate",
        "intervention_mode": "exploratory",
        "changed_variables": [
            "training_recipe.learning_rate",
            "training_recipe.seed",
        ],
        "hypothesis_family_id": "family-optimizer-schedule",
    }


async def test_research_submit_iteration_routes_agent_designed_diagnostic():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(
        server,
        "research_submit_iteration",
        {
            "campaign_id": "campaign-1",
            "expected_version": 8,
            "role": "diagnostic",
            "parent_proposal_id": "candidate-parent",
            "proposal": {
                "proposal_id": "diagnostic-loss-slope",
                "evaluation_recipe": {
                    "schema_version": "bashgym.autoresearch_diagnostic_recipe.v1",
                    "probe_family": "loss_landscape",
                },
            },
        },
    )

    assert result["ok"] is True
    assert client.calls[0]["path"] == "/campaigns/campaign-1/autoresearch/diagnostics"
    assert client.calls[0]["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 8,
        "parent_proposal_id": "candidate-parent",
        "proposal_id": "diagnostic-loss-slope",
        "evaluation_recipe": {
            "schema_version": "bashgym.autoresearch_diagnostic_recipe.v1",
            "probe_family": "loss_landscape",
        },
    }


async def test_research_concludes_hypothesis_family_with_optional_follow_up():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(
        server,
        "research_conclude_hypothesis_family",
        {
            "campaign_id": "campaign-1",
            "expected_version": 8,
            "hypothesis_family_id": "family-longer-training",
            "disposition": "exhausted",
            "summary": "Longer continuation did not improve the fixed suite.",
            "follow_up_family_id": "family-data-coverage",
            "follow_up_hypothesis": "Increase coverage of residual failure clusters.",
        },
    )

    assert result["ok"] is True
    assert client.calls[0]["path"] == (
        "/campaigns/campaign-1/autoresearch/hypothesis-families/" "family-longer-training/conclude"
    )
    assert client.calls[0]["payload"] == {
        "workspace_id": "workspace-a",
        "expected_version": 8,
        "disposition": "exhausted",
        "summary": "Longer continuation did not improve the fixed suite.",
        "follow_up_family_id": "family-data-coverage",
        "follow_up_hypothesis": "Increase coverage of residual failure clusters.",
    }


async def test_research_context_delegates_bounded_search():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    await call_tool(
        server,
        "research_context",
        {
            "campaign_id": "campaign-1",
            "proposal_id": "proposal-1",
            "query": "information gain",
            "categories": ["research", "github"],
            "limit": 4,
        },
    )
    assert client.calls[0]["path"] == "/research/context"
    assert client.calls[0]["payload"] == {
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "proposal_id": "proposal-1",
        "query": "information gain",
        "categories": ["research", "github"],
        "limit": 4,
    }


@pytest.mark.parametrize(
    "server_owned_field",
    [
        "workspace_id",
        "campaign_id",
        "expected_version",
        "role",
        "parent_proposal_id",
        "planner_actor_id",
        "creation_sequence",
        "status",
        "actor",
        "profile",
        "autonomy_profile",
        "capabilities",
        "authorization",
    ],
)
async def test_research_submit_iteration_rejects_server_owned_proposal_fields(
    server_owned_field: str,
):
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(
        server,
        "research_submit_iteration",
        {
            "campaign_id": "campaign-1",
            "expected_version": 2,
            "role": "baseline",
            "proposal": {"proposal_id": "baseline-1", server_owned_field: "injected"},
        },
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "campaign_request_invalid"
    assert client.calls == []


@pytest.mark.parametrize(
    ("role", "parent_proposal_id"),
    [("candidate", None), ("baseline", "baseline-1")],
)
async def test_research_submit_iteration_rejects_invalid_parent_relationship(
    role: str,
    parent_proposal_id: str | None,
):
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )
    arguments = {
        "campaign_id": "campaign-1",
        "expected_version": 2,
        "role": role,
        "proposal": {"proposal_id": "proposal-1"},
    }
    if parent_proposal_id is not None:
        arguments["parent_proposal_id"] = parent_proposal_id

    result = await call_tool(server, "research_submit_iteration", arguments)

    assert result["ok"] is False
    assert result["error"]["code"] == "campaign_request_invalid"
    assert client.calls == []


async def test_campaign_events_tool_reprojects_untrusted_event_responses():
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=LeakyEventClient(),
    )

    result = await call_tool(
        server,
        "campaign_events",
        {"campaign_id": "campaign-1", "after_cursor": 0, "limit": 10},
    )

    assert result["ok"] is True
    event = result["items"][0]["event"]
    assert event["schema_version"] == "public_campaign_event.v1"
    assert "summary" not in event
    assert "payload" not in event
    serialized = repr(result)
    assert "protected-epoch-canary" not in serialized
    assert "candidate-map-canary" not in serialized
    assert "restricted-result.json" not in serialized
    assert "protected-eval-correlation-canary" not in serialized
    assert "protected-eval-idempotency-canary" not in serialized


async def test_campaign_artifacts_tool_reprojects_untrusted_artifact_responses():
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=LeakyArtifactClient(),
    )

    result = await call_tool(
        server,
        "campaign_artifacts",
        {"campaign_id": "campaign-1", "limit": 10},
    )

    assert result["ok"] is True
    artifact = result["artifacts"][0]
    assert artifact["schema_version"] == "public_campaign_artifact.v1"
    assert "uri" not in artifact
    assert "metadata" not in artifact
    serialized = repr(result)
    assert "restricted-result.json" not in serialized
    assert "candidate-map-canary" not in serialized
    assert "protected-epoch-canary" not in serialized


async def test_campaign_artifacts_tool_preserves_bounded_server_pagination():
    client = PaginatedArtifactClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    result = await call_tool(
        server,
        "campaign_artifacts",
        {
            "campaign_id": "campaign-1",
            "after_cursor": "a1.AAAAAAAAAAQ",
            "limit": 2,
        },
    )

    call = client.calls[-1]
    assert call["query"] == {
        "workspace_id": "workspace-a",
        "after_cursor": "a1.AAAAAAAAAAQ",
        "limit": 2,
    }
    assert result["next_cursor"] is None
    assert result["has_more"] is False
    assert result["count"] == 2


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
        {
            "campaign_id": "campaign-2",
            "title": "Embedding cycle",
            "template_id": "embed-v1",
            "stop_rules": {
                "schema_version": "autoresearch_stop_rules.v1",
                "max_attempts": 5,
                "budget_unit": "gpu_hours",
                "max_total_cost": 10.0,
                "minimum_improvement": 0.01,
            },
        },
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
        "stop_rules": {
            "schema_version": "autoresearch_stop_rules.v1",
            "max_attempts": 5,
            "budget_unit": "gpu_hours",
            "max_total_cost": 10.0,
            "minimum_improvement": 0.01,
        },
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


async def test_campaign_code_lineage_tools_bind_workspace_and_proposal_path():
    client = RecordingClient()
    server = build_server(
        workspace_id="workspace-a",
        credential_ref="BASHGYM_CAMPAIGN_REFRESH",
        agent="codex",
        client=client,
    )

    prepared = await call_tool(
        server,
        "campaign_prepare_code_lineage",
        {"campaign_id": "campaign-2", "proposal_id": "proposal-7"},
    )
    captured = await call_tool(
        server,
        "campaign_capture_code_lineage",
        {"campaign_id": "campaign-2", "proposal_id": "proposal-7"},
    )

    assert prepared["ok"] is True and captured["ok"] is True
    assert [call["path"] for call in client.calls] == [
        "/campaigns/campaign-2/proposals/proposal-7/code-lineage/prepare",
        "/campaigns/campaign-2/proposals/proposal-7/code-lineage/capture",
    ]
    assert all(call["payload"] == {"workspace_id": "workspace-a"} for call in client.calls)


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
