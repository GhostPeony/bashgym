"""Workspace-scoped MCP bridge for BashGym Skill Lab."""

from __future__ import annotations

import argparse
import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from bashgym.agent.skill_lab_tools import SkillLabToolError, execute_skill_lab_tool


def build_server(
    *,
    workspace_id: str = "default",
    origin_terminal_id: str | None = None,
    origin_panel_id: str | None = None,
    agent: str = "terminal-agent",
    api_base: str | None = None,
) -> FastMCP:
    origin = {
        "kind": "agent",
        "terminal_id": origin_terminal_id,
        "panel_id": origin_panel_id,
        "agent": agent,
    }
    origin = {key: value for key, value in origin.items() if value}
    call_options = {
        "workspace_id": workspace_id,
        "origin": origin,
    }
    if api_base:
        call_options["api_base"] = api_base

    server = FastMCP(
        "bashgym-skill-lab",
        instructions=(
            "Use these tools when the user asks to build, inspect, test, or evaluate a skill. "
            "Preparing or inspecting a skill materializes the shared Skill Lab node on the "
            "BashGym canvas. File changes and model-call runs require explicit confirmation."
        ),
        json_response=True,
        log_level="ERROR",
    )

    async def call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return await execute_skill_lab_tool(name, arguments, **call_options)
        except SkillLabToolError as exc:
            return exc.as_dict()

    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    canvas_write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    file_write = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)
    model_calls = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)

    @server.tool(structured_output=True, annotations=read_only)
    async def skill_lab_context() -> dict[str, Any]:
        """Read sanitized workspace, linked-node, Skill Lab run, and action context."""

        return await call("skill_lab_context")

    @server.tool(structured_output=True, annotations=read_only)
    async def skill_lab_list_skills(
        query: str = "",
        source: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """List loaded skills and their immutable revisions."""

        return await call(
            "skill_lab_list_skills",
            {"query": query, "source": source, "limit": limit},
        )

    @server.tool(structured_output=True, annotations=canvas_write)
    async def skill_lab_inspect_skill(skill: str, source: str = "") -> dict[str, Any]:
        """Inspect a skill and open it in the workspace Skill Lab node."""

        return await call("skill_lab_inspect_skill", {"skill": skill, "source": source})

    @server.tool(structured_output=True, annotations=canvas_write)
    async def skill_lab_prepare(
        skill: str,
        source: str = "",
        endpoint_id: str = "hermes",
        cases: list[dict[str, Any]] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Focus Skill Lab and optionally save target and negative held-out cases."""

        return await call(
            "skill_lab_prepare",
            {
                "skill": skill,
                "source": source,
                "endpoint_id": endpoint_id,
                "cases": cases or [],
                "thresholds": thresholds or {},
            },
        )

    @server.tool(structured_output=True, annotations=file_write)
    async def skill_lab_save_skill(
        content: str,
        confirmed: bool,
        skill_id: str = "",
        name: str = "",
        description: str = "",
        expected_revision: str = "",
    ) -> dict[str, Any]:
        """Create or update SKILL.md after the user explicitly approves the file change."""

        return await call(
            "skill_lab_save_skill",
            {
                "content": content,
                "confirmed": confirmed,
                "skill_id": skill_id or None,
                "name": name or None,
                "description": description,
                "expected_revision": expected_revision or None,
            },
        )

    @server.tool(structured_output=True, annotations=model_calls)
    async def skill_lab_run(
        skill: str,
        endpoint_id: str,
        confirmed: bool,
        source: str = "",
        cases: list[dict[str, Any]] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Launch paired evaluation after the user approves its model-call preview."""

        return await call(
            "skill_lab_run",
            {
                "skill": skill,
                "source": source,
                "endpoint_id": endpoint_id,
                "confirmed": confirmed,
                "cases": cases or [],
                "thresholds": thresholds or {},
            },
        )

    @server.tool(structured_output=True, annotations=read_only)
    async def skill_lab_status(run_id: str = "", limit: int = 10) -> dict[str, Any]:
        """Read one paired run or list recent Skill Lab runs in this workspace."""

        return await call("skill_lab_status", {"run_id": run_id, "limit": limit})

    @server.resource(
        "bashgym://skill-lab/context",
        name="BashGym Skill Lab context",
        description="Sanitized current workspace and Skill Lab state.",
        mime_type="application/json",
    )
    async def skill_lab_context_resource() -> str:
        return json.dumps(await call("skill_lab_context"), indent=2)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--origin-terminal-id")
    parser.add_argument("--origin-panel-id")
    parser.add_argument("--agent", default="terminal-agent")
    parser.add_argument("--api-base")
    args = parser.parse_args(argv)
    build_server(
        workspace_id=args.workspace_id,
        origin_terminal_id=args.origin_terminal_id,
        origin_panel_id=args.origin_panel_id,
        agent=args.agent,
        api_base=args.api_base,
    ).run(transport="stdio")


if __name__ == "__main__":
    main()
