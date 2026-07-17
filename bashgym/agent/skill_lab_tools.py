"""Shared Skill Lab tool contract for Peony, Hermes, and local MCP clients."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import httpx

from bashgym.api_base import normalize_api_base

DEFAULT_API_BASE = normalize_api_base(
    os.environ.get("BASHGYM_API_BASE")
    or os.environ.get("BASHGYM_API_URL")
    or "http://127.0.0.1:8003/api"
).removesuffix("/api")


def _common_properties() -> dict[str, Any]:
    return {
        "workspace_id": {
            "type": "string",
            "description": "Canvas workspace id. The BashGym bridge supplies this automatically.",
        },
        "origin": {
            "type": "object",
            "description": "Source agent node. The BashGym bridge supplies this automatically.",
            "properties": {
                "panel_id": {"type": "string"},
                "terminal_id": {"type": "string"},
                "agent": {"type": "string"},
            },
        },
    }


_CASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "name": {"type": "string"},
        "prompt": {"type": "string"},
        "should_invoke": {"type": "boolean"},
        "expected_patterns": {"type": "array", "items": {"type": "string"}},
        "forbidden_patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "prompt", "should_invoke"],
}


SKILL_LAB_TOOLS: list[dict[str, Any]] = [
    {
        "name": "skill_lab_context",
        "description": (
            "Read the sanitized BashGym workspace, linked nodes, recent Skill Lab runs, "
            "and allowed actions before evaluating or building a skill."
        ),
        "input_schema": {
            "type": "object",
            "properties": {**_common_properties()},
            "required": [],
        },
    },
    {
        "name": "skill_lab_list_skills",
        "description": "List loaded skills and immutable revisions available to Skill Lab.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_common_properties(),
                "query": {"type": "string", "description": "Optional name or description filter."},
                "source": {"type": "string", "description": "Optional skill source filter."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": [],
        },
    },
    {
        "name": "skill_lab_inspect_skill",
        "description": (
            "Inspect one loaded skill, its exact revision, instructions, allowed tools, "
            "and current workspace success contract. Opens the skill in the canvas Skill Lab."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **_common_properties(),
                "skill": {"type": "string", "description": "Skill id or exact skill name."},
                "source": {"type": "string", "description": "Disambiguating source when using a name."},
            },
            "required": ["skill"],
        },
    },
    {
        "name": "skill_lab_prepare",
        "description": (
            "Create or focus the workspace Skill Lab for a skill and optionally save held-out "
            "target and negative routing cases. Use this while helping design an evaluation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **_common_properties(),
                "skill": {"type": "string", "description": "Skill id or exact skill name."},
                "source": {"type": "string"},
                "endpoint_id": {"type": "string", "description": "Agent endpoint used by paired runs."},
                "cases": {"type": "array", "items": _CASE_SCHEMA},
                "thresholds": {"type": "object", "additionalProperties": {"type": "number"}},
            },
            "required": ["skill"],
        },
    },
    {
        "name": "skill_lab_save_skill",
        "description": (
            "Create a workspace skill or update a loaded local SKILL.md at an expected revision. "
            "This changes files and requires explicit user confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **_common_properties(),
                "skill_id": {"type": "string", "description": "Existing skill id; omit to create."},
                "name": {"type": "string", "description": "New skill name when creating."},
                "description": {"type": "string", "description": "New skill description."},
                "content": {"type": "string", "description": "Complete SKILL.md content."},
                "expected_revision": {"type": "string", "description": "Revision being replaced."},
                "confirmed": {
                    "type": "boolean",
                    "description": "True only after the user explicitly approves the file change.",
                },
            },
            "required": ["content", "confirmed"],
        },
    },
    {
        "name": "skill_lab_run",
        "description": (
            "Launch the three-arm baseline, available, and forced Skill Lab evaluation. "
            "This makes model calls and requires explicit user confirmation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                **_common_properties(),
                "skill": {"type": "string", "description": "Skill id or exact skill name."},
                "source": {"type": "string"},
                "endpoint_id": {"type": "string"},
                "cases": {"type": "array", "items": _CASE_SCHEMA},
                "thresholds": {"type": "object", "additionalProperties": {"type": "number"}},
                "confirmed": {
                    "type": "boolean",
                    "description": "True only after the user approves the displayed model-call count.",
                },
            },
            "required": ["skill", "endpoint_id", "confirmed"],
        },
    },
    {
        "name": "skill_lab_status",
        "description": "Read one Skill Lab run or list recent runs for the current workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_common_properties(),
                "run_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": [],
        },
    },
]


class SkillLabToolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"error": self.code, "message": self.message}


class SkillLabToolClient:
    """Execute the shared tool contract against the local BashGym API."""

    def __init__(
        self,
        *,
        api_base: str = DEFAULT_API_BASE,
        workspace_id: str = "default",
        origin: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.workspace_id = workspace_id or "default"
        self.origin = {"kind": "agent", **(origin or {})}
        self._client = client

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        try:
            response = await client.request(method, f"{self.api_base}{path}", **kwargs)
            if response.status_code >= 400:
                try:
                    data = response.json()
                except ValueError:
                    data = {}
                detail = data.get("detail") if isinstance(data, dict) else None
                if not isinstance(detail, str):
                    detail = f"BashGym returned HTTP {response.status_code}"
                raise SkillLabToolError(f"http_{response.status_code}", detail[:500])
            return response.json()
        except httpx.HTTPError as exc:
            raise SkillLabToolError("api_unavailable", "The local BashGym API is unavailable.") from exc
        finally:
            if owns_client:
                await client.aclose()

    def _scope(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        workspace_id = str(arguments.get("workspace_id") or self.workspace_id or "default")
        supplied_origin = arguments.get("origin")
        origin = {**self.origin}
        if isinstance(supplied_origin, dict):
            origin.update(supplied_origin)
        origin.setdefault("agent", "bashgym-agent")
        return workspace_id, origin

    async def _resolve_skill(self, reference: str, source: str | None = None) -> dict[str, Any]:
        inventory = await self._request("GET", "/api/skill-lab/skills")
        skills = inventory.get("skills", []) if isinstance(inventory, dict) else []
        exact = [
            skill
            for skill in skills
            if isinstance(skill, dict)
            and (skill.get("skill_id") == reference or skill.get("name") == reference)
            and (not source or skill.get("source") == source)
        ]
        if not exact:
            raise SkillLabToolError("skill_not_found", f"No loaded skill matches {reference!r}.")
        by_id = [skill for skill in exact if skill.get("skill_id") == reference]
        if by_id:
            return by_id[0]
        if len(exact) > 1:
            choices = ", ".join(
                f"{item.get('skill_id')} ({item.get('source')}, {str(item.get('revision') or '')[:8]})"
                for item in exact[:8]
            )
            raise SkillLabToolError("skill_ambiguous", f"Choose a skill id: {choices}")
        return exact[0]

    async def _emit(
        self,
        *,
        event_type: str,
        workspace_id: str,
        origin: dict[str, Any],
        skill: dict[str, Any] | None = None,
        run: dict[str, Any] | None = None,
        summary: str,
    ) -> dict[str, Any]:
        correlation_id = str((run or {}).get("run_id") or f"skill-lab-{uuid4().hex[:12]}")
        config = {
            "selectedSkillId": (skill or {}).get("skill_id"),
            "selectedSkillName": (skill or {}).get("name"),
            "selectedSkillRevision": (skill or {}).get("revision"),
            "selectedSkillSource": (skill or {}).get("source"),
            "selectedSkillPath": (skill or {}).get("path"),
            "latestRunId": (run or {}).get("run_id"),
            "status": (run or {}).get("status"),
            "agentAction": event_type,
            "correlationId": correlation_id,
        }
        payload = {
            "type": event_type,
            "workspace_id": workspace_id,
            "source": origin,
            "correlation_id": correlation_id,
            "title": "Skill Lab",
            "summary": summary,
            "entity": {
                "kind": "skill_lab",
                "skill_id": (skill or {}).get("skill_id"),
                "skill_name": (skill or {}).get("name"),
                "skill_revision": (skill or {}).get("revision"),
                "run_id": (run or {}).get("run_id"),
                "status": (run or {}).get("status"),
            },
            "suggested_nodes": [
                {"recipe": "skill_lab", "title": "Skill Lab", "config": config}
            ],
            "relationships": [],
            "payload": {"tool": event_type},
        }
        return await self._request("POST", "/api/workspace/events", json=payload)

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        workspace_id, origin = self._scope(args)

        if name == "skill_lab_context":
            return await self._request(
                "GET", "/api/workspace/context", params={"workspace_id": workspace_id}
            )

        if name == "skill_lab_list_skills":
            data = await self._request("GET", "/api/skill-lab/skills")
            skills = data.get("skills", []) if isinstance(data, dict) else []
            query = str(args.get("query") or "").strip().casefold()
            source = str(args.get("source") or "").strip()
            filtered = [
                item
                for item in skills
                if isinstance(item, dict)
                and (not source or item.get("source") == source)
                and (
                    not query
                    or query in str(item.get("name") or "").casefold()
                    or query in str(item.get("description") or "").casefold()
                )
            ]
            limit = max(1, min(int(args.get("limit") or 50), 100))
            return {
                "skills": [
                    {
                        key: item.get(key)
                        for key in (
                            "skill_id",
                            "name",
                            "description",
                            "source",
                            "revision",
                            "allowed_tools",
                        )
                    }
                    for item in filtered[:limit]
                ],
                "matched": len(filtered),
                "returned": min(len(filtered), limit),
            }

        if name in {"skill_lab_inspect_skill", "skill_lab_prepare", "skill_lab_run"}:
            skill = await self._resolve_skill(
                str(args.get("skill") or ""), str(args.get("source") or "") or None
            )
        else:
            skill = None

        if name == "skill_lab_inspect_skill":
            detail = await self._request(
                "GET", f"/api/skill-lab/skills/{skill['skill_id']}"
            )
            try:
                contract = await self._request(
                    "GET",
                    f"/api/skill-lab/contracts/{skill['skill_id']}",
                    params={"workspace_id": workspace_id},
                )
            except SkillLabToolError as exc:
                if exc.code != "http_404":
                    raise
                contract = None
            await self._emit(
                event_type="skill_lab.inspected",
                workspace_id=workspace_id,
                origin=origin,
                skill=skill,
                summary=f"Inspecting {skill['name']} at revision {str(skill.get('revision') or '')[:8]}",
            )
            return {"skill": detail, "contract": contract}

        if name == "skill_lab_prepare":
            contract = None
            cases = args.get("cases")
            if isinstance(cases, list) and cases:
                contract = await self._request(
                    "PUT",
                    f"/api/skill-lab/contracts/{skill['skill_id']}",
                    params={"workspace_id": workspace_id},
                    json={
                        "endpoint_id": args.get("endpoint_id"),
                        "cases": cases,
                        "thresholds": args.get("thresholds") or {},
                    },
                )
            await self._emit(
                event_type="skill_lab.prepared",
                workspace_id=workspace_id,
                origin=origin,
                skill=skill,
                summary=f"Prepared {skill['name']} for paired evaluation",
            )
            return {
                "status": "prepared",
                "skill": skill,
                "contract": contract,
                "next": "Add target and negative routing cases, then request confirmation before running.",
            }

        if name == "skill_lab_save_skill":
            content = str(args.get("content") or "")
            if not content.strip():
                raise SkillLabToolError("invalid_content", "Complete SKILL.md content is required.")
            if not bool(args.get("confirmed")):
                return {
                    "status": "confirmation_required",
                    "action": "update" if args.get("skill_id") else "create",
                    "skill_id": args.get("skill_id"),
                    "name": args.get("name"),
                    "bytes": len(content.encode("utf-8")),
                    "message": "Ask the user to approve this SKILL.md file change, then call again with confirmed=true.",
                }
            if args.get("skill_id"):
                result = await self._request(
                    "PUT",
                    f"/api/skill-lab/skills/{args['skill_id']}",
                    json={
                        "content": content,
                        "expected_revision": args.get("expected_revision"),
                        "confirmed": True,
                    },
                )
            else:
                result = await self._request(
                    "POST",
                    "/api/skill-lab/skills",
                    json={
                        "name": args.get("name"),
                        "description": args.get("description"),
                        "content": content,
                        "confirmed": True,
                    },
                )
            await self._emit(
                event_type="skill_lab.skill.saved",
                workspace_id=workspace_id,
                origin=origin,
                skill=result,
                summary=f"Saved {result.get('name') or 'skill'}",
            )
            return {"status": "saved", "skill": result}

        if name == "skill_lab_run":
            cases = args.get("cases")
            if not isinstance(cases, list) or not cases:
                try:
                    contract = await self._request(
                        "GET",
                        f"/api/skill-lab/contracts/{skill['skill_id']}",
                        params={"workspace_id": workspace_id},
                    )
                except SkillLabToolError as exc:
                    if exc.code == "http_404":
                        raise SkillLabToolError(
                            "contract_required",
                            "Prepare target and negative routing cases before launching the eval.",
                        ) from exc
                    raise
                cases = contract.get("cases", [])
                thresholds = args.get("thresholds") or contract.get("thresholds", {})
            else:
                thresholds = args.get("thresholds") or {}
            call_count = len(cases) * 3
            if not bool(args.get("confirmed")):
                return {
                    "status": "confirmation_required",
                    "skill_id": skill["skill_id"],
                    "skill_name": skill["name"],
                    "cases": len(cases),
                    "model_calls": call_count,
                    "message": (
                        f"This paired evaluation will make {call_count} model calls. "
                        "Ask the user to approve it, then call again with confirmed=true."
                    ),
                }
            run = await self._request(
                "POST",
                "/api/skill-lab/runs",
                json={
                    "workspace_id": workspace_id,
                    "skill_id": skill["skill_id"],
                    "endpoint_id": args.get("endpoint_id"),
                    "cases": cases,
                    "thresholds": thresholds,
                },
            )
            await self._emit(
                event_type="skill_lab.run.started",
                workspace_id=workspace_id,
                origin=origin,
                skill=skill,
                run=run,
                summary=f"Started paired evaluation for {skill['name']}",
            )
            return run

        if name == "skill_lab_status":
            run_id = str(args.get("run_id") or "").strip()
            if run_id:
                return await self._request("GET", f"/api/skill-lab/runs/{run_id}")
            return {
                "runs": await self._request(
                    "GET",
                    "/api/skill-lab/runs",
                    params={
                        "workspace_id": workspace_id,
                        "limit": max(1, min(int(args.get("limit") or 10), 50)),
                    },
                )
            }

        raise SkillLabToolError("unknown_tool", f"Unknown Skill Lab tool: {name}")


async def execute_skill_lab_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    api_base: str = DEFAULT_API_BASE,
    workspace_id: str = "default",
    origin: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    bridge = SkillLabToolClient(
        api_base=api_base,
        workspace_id=workspace_id,
        origin=origin,
        client=client,
    )
    return await bridge.execute(name, arguments)


SKILL_LAB_TOOL_NAMES = frozenset(tool["name"] for tool in SKILL_LAB_TOOLS)
