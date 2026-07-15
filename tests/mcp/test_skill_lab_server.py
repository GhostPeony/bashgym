from __future__ import annotations

import sys
from pathlib import Path

from bashgym.mcp.client_runtime import McpClientRuntime

ROOT = Path(__file__).resolve().parents[2]


async def test_skill_lab_stdio_server_exposes_agent_contract():
    runtime = McpClientRuntime()
    connected = await runtime.connect_stdio(
        "skill-lab",
        sys.executable,
        [
            "-m",
            "bashgym.mcp.skill_lab_server",
            "--workspace-id",
            "test-workspace",
            "--origin-terminal-id",
            "terminal-1",
        ],
        cwd=str(ROOT),
        environment={"PYTHONPATH": str(ROOT)},
    )
    try:
        tools = {tool["name"]: tool for tool in connected["inventory"]["tools"]}
        names = set(tools)
        assert {
            "skill_lab_context",
            "skill_lab_list_skills",
            "skill_lab_inspect_skill",
            "skill_lab_prepare",
            "skill_lab_save_skill",
            "skill_lab_run",
            "skill_lab_status",
        } == names
        assert tools["skill_lab_context"]["annotations"]["readOnlyHint"] is True
        assert tools["skill_lab_save_skill"]["annotations"]["destructiveHint"] is True
        assert tools["skill_lab_run"]["annotations"]["openWorldHint"] is True
        resources = {item["uri"] for item in connected["inventory"]["resources"]}
        assert "bashgym://skill-lab/context" in resources
    finally:
        await runtime.aclose()
