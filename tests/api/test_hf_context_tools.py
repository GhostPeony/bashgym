import json

import httpx
import pytest

from bashgym.agent.hf_context_tools import (
    HF_CONTEXT_TOOL_NAMES,
    HFContextToolClient,
    HFContextToolError,
    execute_hf_context_tool,
)
from bashgym.agent.tools import CORE_TOOLS


@pytest.mark.asyncio
async def test_search_tool_forwards_workspace_origin_and_intent():
    captured = {}

    async def handler(request: httpx.Request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"bundle_id": "hfctx_fixture", "version": 1})

    client = HFContextToolClient(
        api_base="http://testserver", transport=httpx.MockTransport(handler)
    )
    result = await execute_hf_context_tool(
        "hf_context_search",
        {
            "workspace_id": "workspace-a",
            "intent": "find coding datasets",
            "origin": {"panel_id": "terminal-1"},
        },
        client=client,
    )

    assert result["bundle_id"] == "hfctx_fixture"
    assert captured["path"] == "/api/hf/context/discover"
    assert captured["body"]["workspace_id"] == "workspace-a"
    assert captured["body"]["origin"] == {"panel_id": "terminal-1"}


@pytest.mark.asyncio
async def test_tool_preserves_stable_backend_error_code():
    async def handler(request: httpx.Request):
        return httpx.Response(
            409,
            json={"detail": {"code": "hf_bundle_conflict", "message": "stale version"}},
        )

    client = HFContextToolClient(
        api_base="http://testserver", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(HFContextToolError) as caught:
        await execute_hf_context_tool(
            "hf_context_pin",
            {
                "workspace_id": "workspace-a",
                "bundle_id": "hfctx_fixture",
                "version": 1,
                "expected_version": 1,
                "selected_evidence_ids": [],
            },
            client=client,
        )
    assert caught.value.code == "hf_bundle_conflict"


@pytest.mark.asyncio
async def test_tools_fail_closed_without_workspace_scope():
    with pytest.raises(HFContextToolError, match="workspace"):
        await execute_hf_context_tool("hf_context_deactivate", {})


def test_every_context_tool_is_in_the_core_agent_catalog():
    core_names = {tool["name"] for tool in CORE_TOOLS}
    assert HF_CONTEXT_TOOL_NAMES <= core_names
