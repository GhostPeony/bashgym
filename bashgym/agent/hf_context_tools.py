"""Safe Hugging Face context-pack tools shared by terminal agents."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_BASE = os.environ.get("BASHGYM_API_URL", "http://127.0.0.1:8003").rstrip("/")


def _workspace_properties() -> dict[str, Any]:
    return {
        "workspace_id": {
            "type": "string",
            "description": "Canvas workspace id supplied by the BashGym bridge.",
        },
        "origin": {
            "type": "object",
            "description": "Source terminal or agent node supplied by the canvas bridge.",
            "properties": {
                "panel_id": {"type": "string"},
                "terminal_id": {"type": "string"},
                "agent": {"type": "string"},
            },
        },
    }


HF_CONTEXT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "hf_context_search",
        "description": "Find source-linked Hugging Face models, datasets, and published evaluation evidence for a concrete task and prepare a workspace context bundle.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_workspace_properties(),
                "intent": {"type": "string"},
                "task": {"type": "string"},
                "target": {"type": "object"},
            },
            "required": ["intent"],
        },
    },
    {
        "name": "hf_context_inspect",
        "description": "Inspect one exact immutable Hugging Face context bundle version.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_workspace_properties(),
                "bundle_id": {"type": "string"},
                "version": {"type": "integer", "minimum": 1},
            },
            "required": ["bundle_id", "version"],
        },
    },
    {
        "name": "hf_context_pin",
        "description": "Create a new immutable bundle version containing the selected evidence IDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_workspace_properties(),
                "bundle_id": {"type": "string"},
                "version": {"type": "integer", "minimum": 1},
                "expected_version": {"type": "integer", "minimum": 1},
                "selected_evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["bundle_id", "version", "expected_version", "selected_evidence_ids"],
        },
    },
    {
        "name": "hf_context_activate",
        "description": "Make one exact ready bundle version the active Hugging Face context for the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_workspace_properties(),
                "bundle_id": {"type": "string"},
                "version": {"type": "integer", "minimum": 1},
            },
            "required": ["bundle_id", "version"],
        },
    },
    {
        "name": "hf_context_deactivate",
        "description": "Clear the workspace's active Hugging Face context pointer without deleting bundles.",
        "input_schema": {
            "type": "object",
            "properties": {**_workspace_properties()},
            "required": [],
        },
    },
    {
        "name": "hf_context_prepare_eval",
        "description": "Prepare a non-executing Eval recipe preview from one immutable Hugging Face context bundle.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_workspace_properties(),
                "bundle_id": {"type": "string"},
                "version": {"type": "integer", "minimum": 1},
            },
            "required": ["bundle_id", "version"],
        },
    },
]


class HFContextToolError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"error": self.code, "message": self.message}


class HFContextToolClient:
    def __init__(self, *, api_base: str = DEFAULT_API_BASE, transport: httpx.AsyncBaseTransport | None = None):
        self.api_base = api_base.rstrip("/")
        self.transport = transport

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        async with httpx.AsyncClient(timeout=35, transport=self.transport) as client:
            response = await client.request(
                method, f"{self.api_base}{path}", params=params, json=json_body
            )
        if response.is_error:
            try:
                detail = response.json().get("detail", {})
            except ValueError:
                detail = {}
            raise HFContextToolError(
                str(detail.get("code") or "hf_context_failed"),
                str(detail.get("message") or "Hugging Face context request failed."),
            )
        return response.json()


async def execute_hf_context_tool(
    name: str,
    tool_input: dict[str, Any],
    *,
    client: HFContextToolClient | None = None,
) -> Any:
    client = client or HFContextToolClient()
    workspace_id = str(tool_input.get("workspace_id") or "").strip()
    if not workspace_id:
        raise HFContextToolError("workspace_required", "A canvas workspace is required.")
    bundle_id = tool_input.get("bundle_id")
    version = tool_input.get("version")

    if name == "hf_context_search":
        return await client.request(
            "POST",
            "/api/hf/context/discover",
            json_body={
                "workspace_id": workspace_id,
                "intent": tool_input["intent"],
                "task": tool_input.get("task"),
                "target": tool_input.get("target") or {},
                "origin": tool_input.get("origin") or {},
            },
        )
    if name == "hf_context_inspect":
        return await client.request(
            "GET",
            f"/api/hf/context/bundles/{bundle_id}/versions/{version}",
            params={"workspace_id": workspace_id},
        )
    if name == "hf_context_pin":
        return await client.request(
            "POST",
            f"/api/hf/context/bundles/{bundle_id}/versions/{version}/pin",
            json_body={
                "workspace_id": workspace_id,
                "expected_version": tool_input["expected_version"],
                "selected_evidence_ids": tool_input["selected_evidence_ids"],
            },
        )
    if name == "hf_context_activate":
        return await client.request(
            "POST",
            f"/api/hf/context/bundles/{bundle_id}/versions/{version}/activate",
            json_body={"workspace_id": workspace_id},
        )
    if name == "hf_context_deactivate":
        return await client.request(
            "DELETE", "/api/hf/context/active", json_body={"workspace_id": workspace_id}
        )
    if name == "hf_context_prepare_eval":
        return await client.request(
            "POST",
            f"/api/hf/context/bundles/{bundle_id}/versions/{version}/actions/eval",
            json_body={"workspace_id": workspace_id},
        )
    raise HFContextToolError("unknown_tool", f"Unknown Hugging Face context tool: {name}")


HF_CONTEXT_TOOL_NAMES = frozenset(tool["name"] for tool in HF_CONTEXT_TOOLS)


__all__ = [
    "HF_CONTEXT_TOOLS",
    "HF_CONTEXT_TOOL_NAMES",
    "HFContextToolClient",
    "HFContextToolError",
    "execute_hf_context_tool",
]
