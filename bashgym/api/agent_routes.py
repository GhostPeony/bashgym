# bashgym/api/agent_routes.py
"""API routes for Peony — the botanical assistant for Bash Gym."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from bashgym import secrets as secret_store
from bashgym.agent.hf_context_tools import (
    HF_CONTEXT_TOOL_NAMES,
    HFContextToolError,
    execute_hf_context_tool,
)
from bashgym.agent.memory import PeonyMemory
from bashgym.agent.skill_lab_tools import (
    SKILL_LAB_TOOL_NAMES,
    SKILL_LAB_TOOLS,
    SkillLabToolError,
    execute_skill_lab_tool,
)
from bashgym.agent.skills.registry import SkillRegistry
from bashgym.agent.tools import AWARENESS_TOOLS, CORE_TOOLS, MEMORY_TOOLS, ToolRegistry
from bashgym.api.schemas import TrainingRequest
from bashgym.compute import normalize_training_target_payload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_skill_registry = SkillRegistry()
_tool_registry = ToolRegistry()
_memory = PeonyMemory()

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SAFE_SSH_TARGET = re.compile(r"^[A-Za-z0-9._@:-]{1,160}$")
_SAFE_TUNNEL_HOST = re.compile(r"^[A-Za-z0-9._-]{1,253}$")


def _validate_session_id(session_id: str) -> str:
    if not _SAFE_ID.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")
    return session_id


AGENT_ENDPOINTS_CONFIG = "agent_endpoints.json"
DEFAULT_AGENT_ENDPOINT_ID = "hermes"


def _validate_endpoint_id(endpoint_id: str) -> str:
    if not _SAFE_ID.match(endpoint_id):
        raise HTTPException(status_code=400, detail="Invalid endpoint ID")
    return endpoint_id


def _agent_endpoints_path() -> Path:
    from bashgym.config import get_bashgym_dir

    return get_bashgym_dir() / AGENT_ENDPOINTS_CONFIG


def _read_agent_endpoint_config() -> dict[str, dict[str, Any]]:
    path = _agent_endpoints_path()
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read agent endpoint config: %s", exc)
        return {}

    endpoints = data.get("endpoints", {})
    return endpoints if isinstance(endpoints, dict) else {}


def _write_agent_endpoint_config(endpoints: dict[str, dict[str, Any]]) -> None:
    path = _agent_endpoints_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"endpoints": endpoints}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalize_agent_base_url(raw_url: str) -> str:
    base_url = raw_url.strip().rstrip("/")
    if not base_url:
        raise HTTPException(status_code=400, detail="Endpoint URL is required")

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="Endpoint URL must be an http(s) URL",
        )
    if parsed.query or parsed.fragment:
        raise HTTPException(
            status_code=400,
            detail="Endpoint URL cannot include query strings or fragments",
        )

    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _default_agent_endpoint() -> dict[str, Any]:
    model = os.environ.get("HERMES_MODEL", "hermes-agent")
    return {
        "id": DEFAULT_AGENT_ENDPOINT_ID,
        "label": os.environ.get("HERMES_ENDPOINT_LABEL", "Hermes"),
        "kind": "hermes",
        "base_url": _normalize_agent_base_url(
            os.environ.get("HERMES_API_BASE", "http://127.0.0.1:8642/v1")
        ),
        "model": model,
        "model_options": _normalize_model_options(
            model,
            os.environ.get("HERMES_MODEL_OPTIONS", ""),
        ),
        "session_key": os.environ.get("HERMES_SESSION_KEY") or None,
        "enabled": True,
    }


def _normalize_model_options(model: str, raw_options: Any) -> list[str]:
    if isinstance(raw_options, str):
        candidates = [item.strip() for item in raw_options.split(",")]
    elif isinstance(raw_options, list):
        candidates = [str(item).strip() for item in raw_options]
    else:
        candidates = []

    options: list[str] = []
    for item in [model, *candidates]:
        if item and item not in options:
            options.append(item[:120])
    return options[:20]


def _agent_secret_key(endpoint_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", endpoint_id).upper()
    return f"AGENT_ENDPOINT_{normalized}_API_KEY"


def _agent_secret_env_keys(endpoint_id: str) -> list[str]:
    keys = [_agent_secret_key(endpoint_id)]
    if endpoint_id == DEFAULT_AGENT_ENDPOINT_ID:
        keys = ["HERMES_API_KEY", "HERMES_API_SERVER_KEY", *keys]
    return keys


def _get_agent_api_key(endpoint_id: str) -> str | None:
    for key in _agent_secret_env_keys(endpoint_id):
        value = secret_store.get_secret(key)
        if value:
            return value
    return None


def _set_agent_api_key(endpoint_id: str, api_key: str | None) -> None:
    if api_key is None:
        return
    trimmed = api_key.strip()
    if trimmed:
        secret_store.set_secret(_agent_secret_key(endpoint_id), trimmed)


def _delete_agent_api_key(endpoint_id: str) -> None:
    secret_store.delete_secret(_agent_secret_key(endpoint_id))


def _public_agent_profile(profile: dict[str, Any]) -> AgentEndpointProfile:
    endpoint_id = str(profile["id"])
    model = str(profile.get("model") or "hermes-agent")
    return AgentEndpointProfile(
        id=endpoint_id,
        label=str(profile.get("label") or "Hermes"),
        kind=str(profile.get("kind") or "hermes"),
        base_url=str(profile.get("base_url") or "http://127.0.0.1:8642/v1"),
        model=model,
        model_options=_normalize_model_options(model, profile.get("model_options") or []),
        session_key=profile.get("session_key") or None,
        enabled=bool(profile.get("enabled", True)),
        api_key_configured=_get_agent_api_key(endpoint_id) is not None,
    )


def _load_agent_endpoint_profiles() -> dict[str, dict[str, Any]]:
    endpoints = _read_agent_endpoint_config()
    if DEFAULT_AGENT_ENDPOINT_ID not in endpoints:
        endpoints[DEFAULT_AGENT_ENDPOINT_ID] = _default_agent_endpoint()
    return endpoints


def _get_agent_endpoint_profile(endpoint_id: str) -> dict[str, Any]:
    endpoint_id = _validate_endpoint_id(endpoint_id)
    endpoints = _load_agent_endpoint_profiles()
    profile = endpoints.get(endpoint_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Agent endpoint not found")
    profile["id"] = endpoint_id
    profile["base_url"] = _normalize_agent_base_url(
        str(profile.get("base_url") or "http://127.0.0.1:8642/v1")
    )
    profile["model"] = str(profile.get("model") or "hermes-agent")
    profile["model_options"] = _normalize_model_options(
        profile["model"],
        profile.get("model_options") or [],
    )
    return profile


def _endpoint_url(profile: dict[str, Any], path: str) -> str:
    base_url = _normalize_agent_base_url(str(profile["base_url"]))
    if path.lstrip("/").startswith("health") and base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return f"{base_url}/{path.lstrip('/')}"


def _agent_headers(profile: dict[str, Any], session_key: str | None = None) -> dict[str, str]:
    api_key = _get_agent_api_key(str(profile["id"]))
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    scoped_session_key = session_key or profile.get("session_key")
    if scoped_session_key:
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in str(scoped_session_key)):
            raise HTTPException(
                status_code=400,
                detail="Session key cannot contain control characters",
            )
        headers["x-hermes-session-key"] = str(scoped_session_key)[:256]

    return headers


def _sanitize_agent_error(message: str, endpoint_id: str) -> str:
    scrubbed = message
    for key in _agent_secret_env_keys(endpoint_id):
        value = secret_store.get_secret(key)
        if value:
            scrubbed = scrubbed.replace(value, "<redacted>")
    scrubbed = re.sub(
        r"Bearer\s+[A-Za-z0-9._~+/=-]+",
        "Bearer <redacted>",
        scrubbed,
        flags=re.IGNORECASE,
    )
    return scrubbed


def _agent_response_error(
    response: httpx.Response,
    endpoint_id: str,
    data: Any | None = None,
) -> str:
    """Return a safe, actionable error for an authenticated Hermes response."""
    if response.status_code in {401, 403}:
        return (
            "Hermes rejected the saved API server key. Open Hermes settings, "
            "update the endpoint API key, then test the connection again."
        )
    payload = _json_or_text(response) if data is None else data
    return _sanitize_agent_error(str(payload), endpoint_id)


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text[:1000]}


def _extract_responses_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for block in item.get("content", []) or []:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)

    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _extract_chat_completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
    return ""


def _openai_skill_lab_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        for tool in SKILL_LAB_TOOLS
    ]


def _chat_completion_skill_lab_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in SKILL_LAB_TOOLS
    ]


def _decode_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_responses_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        name = item.get("name")
        call_id = item.get("call_id") or item.get("id")
        if isinstance(name, str) and isinstance(call_id, str):
            calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": _decode_tool_arguments(item.get("arguments")),
                }
            )
    return calls


def _extract_chat_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message") or {}
    calls: list[dict[str, Any]] = []
    for item in message.get("tool_calls", []) or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        name = function.get("name")
        call_id = item.get("id")
        if isinstance(name, str) and isinstance(call_id, str):
            calls.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": _decode_tool_arguments(function.get("arguments")),
                }
            )
    return calls


def _skill_lab_intent(message: str) -> bool:
    normalized = message.casefold()
    return "skill" in normalized and any(
        word in normalized for word in ("build", "create", "edit", "evaluate", "eval", "test")
    )


def _explicit_action_approval(message: str) -> bool:
    normalized = message.casefold()
    if normalized.strip() in {"yes", "y", "approved", "confirm", "confirmed", "proceed"}:
        return True
    return any(
        phrase in normalized
        for phrase in (
            "approve",
            "confirmed",
            "do it",
            "go ahead",
            "launch it",
            "proceed",
            "run it",
            "save it",
            "start it",
            "yes,",
            "yes ",
        )
    )


def _encode_sse(event: str, payload: dict[str, Any]) -> str:
    """Encode one normalized SSE event for the desktop chat client."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _iter_sse_frames(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Parse SSE frames without assuming network chunks align to event boundaries."""
    event_name = "message"
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield event_name, "\n".join(data_lines)


def _stream_error_message(data: dict[str, Any]) -> str | None:
    error = data.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail")
        if isinstance(message, str):
            return message

    response = data.get("response")
    if isinstance(response, dict):
        nested = response.get("error")
        if isinstance(nested, dict) and isinstance(nested.get("message"), str):
            return nested["message"]
    return None


async def _iter_agent_stream_events(
    response: httpx.Response,
    protocol: str,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Normalize Hermes Responses/Chat Completions streams for the renderer."""
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        await response.aread()
        data = _json_or_text(response)
        if not isinstance(data, dict):
            yield "error", {"error": "Hermes returned an invalid response"}
            return
        text = (
            _extract_responses_text(data)
            if protocol == "responses"
            else _extract_chat_completion_text(data)
        )
        if text:
            yield "delta", {"delta": text}
            return
        yield "error", {"error": _stream_error_message(data) or "Hermes returned no text"}
        return

    async for event_name, raw_data in _iter_sse_frames(response):
        if raw_data == "[DONE]":
            return
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        event_type = str(data.get("type") or event_name)
        if event_type == "response.output_text.delta":
            delta = data.get("delta")
            if isinstance(delta, str) and delta:
                yield "delta", {"delta": delta}
            continue

        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield "delta", {"delta": content}
            continue

        if event_type == "hermes.tool.progress":
            label = data.get("tool") or data.get("name") or data.get("message")
            if isinstance(label, str) and label:
                yield "activity", {"label": label}
            continue

        if event_type == "response.output_item.added":
            item = data.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                label = item.get("name")
                if isinstance(label, str) and label:
                    yield "activity", {"label": label}
            continue

        if event_type in {"response.failed", "error"}:
            yield "error", {"error": _stream_error_message(data) or "Hermes stream failed"}
            return

        if event_type == "response.completed":
            completed = data.get("response")
            payload: dict[str, Any] = {}
            if isinstance(completed, dict):
                response_id = completed.get("id")
                if isinstance(response_id, str):
                    payload["response_id"] = response_id
                final_text = _extract_responses_text(completed)
                if final_text:
                    payload["final_text"] = final_text
            yield "terminal", payload


def _looks_like_agent_runtime_failure(text: str) -> bool:
    """Detect provider/runtime errors that Hermes may return as assistant text."""
    normalized = text.strip().lower()
    if not normalized:
        return False
    runtime_markers = (
        "api call failed",
        "api call failed after",
        "validationexception",
        "provided model identifier is invalid",
        "model identifier is invalid",
    )
    return any(marker in normalized for marker in runtime_markers)


# ---------------------------------------------------------------------------
# In-memory pending actions (shell confirmation gate)
# ---------------------------------------------------------------------------

PENDING_ACTIONS: dict[str, dict] = (
    {}
)  # token → {cmd, reason, messages, tool_use_id, expires_at, tools}


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] | None = None


class PendingAction(BaseModel):
    type: str  # "shell_command"
    command: str
    reason: str
    token: str


class ChatResponse(BaseModel):
    response: str
    context_used: list[str] = []
    pending_action: PendingAction | None = None


class ConfirmActionRequest(BaseModel):
    token: str
    approved: bool
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


class SessionMeta(BaseModel):
    session_id: str
    name: str
    created_at: str
    updated_at: str
    message_count: int = 0


class SessionMessage(BaseModel):
    id: str
    role: str
    content: str
    timestamp: float
    context_used: list[str] = []


class SaveSessionRequest(BaseModel):
    session_id: str
    name: str
    messages: list[SessionMessage]


# ---------------------------------------------------------------------------
# External agent endpoint models
# ---------------------------------------------------------------------------


class AgentEndpointUpdate(BaseModel):
    label: str = "Hermes"
    kind: str = "hermes"
    base_url: str = "http://127.0.0.1:8642/v1"
    model: str = "hermes-agent"
    model_options: list[str] = []
    session_key: str | None = None
    enabled: bool = True
    api_key: str | None = None
    clear_api_key: bool = False


class AgentEndpointProfile(BaseModel):
    id: str
    label: str
    kind: str = "hermes"
    base_url: str
    model: str
    model_options: list[str] = []
    session_key: str | None = None
    enabled: bool = True
    api_key_configured: bool = False


class AgentEndpointListResponse(BaseModel):
    endpoints: list[AgentEndpointProfile]


class AgentEndpointChatRequest(BaseModel):
    message: str
    context: str | None = None
    conversation: str | None = None
    session_key: str | None = None
    workspace_id: str = "default"
    origin: dict[str, Any] = Field(default_factory=dict)
    history: list[ChatMessage] = Field(default_factory=list)
    enable_skill_lab_tools: bool = True


class AgentEndpointChatResponse(BaseModel):
    response: str
    endpoint_id: str
    model: str
    response_id: str | None = None
    raw_status: str | None = None


class HermesSetupStatus(BaseModel):
    installed: bool
    command: str | None = None
    gateway_command: list[str] = []
    hermes_home: str
    config_path: str | None = None
    configured_model: str | None = None
    configured_provider: str | None = None
    env_path: str
    env_exists: bool = False
    env_api_enabled: bool = False
    env_key_present: bool = False
    gateway_url: str = "http://127.0.0.1:8642/v1"
    gateway_healthy: bool = False
    gateway_error: str | None = None
    profile: AgentEndpointProfile
    setup_needed: list[str] = []
    log_path: str | None = None


class HermesQuickSetupRequest(BaseModel):
    profile_id: str = DEFAULT_AGENT_ENDPOINT_ID
    label: str = "Hermes"
    base_url: str = "http://127.0.0.1:8642/v1"
    model: str = "hermes-agent"
    model_options: list[str] = []
    session_key: str | None = "bashgym-canvas"
    api_key: str | None = None
    write_env: bool = True
    start_gateway: bool = True


class HermesQuickSetupResponse(BaseModel):
    status: HermesSetupStatus
    actions: list[str]


class HermesTunnelRequest(BaseModel):
    endpoint_id: str = DEFAULT_AGENT_ENDPOINT_ID
    label: str = "Hermes"
    ssh_target: str
    remote_host: str = "127.0.0.1"
    remote_port: int = 8642
    local_port: int | None = None
    model: str = "hermes-agent"
    model_options: list[str] = []
    session_key: str | None = "bashgym-canvas"
    api_key: str | None = None
    save_profile: bool = True


class HermesTunnelDisconnectRequest(BaseModel):
    endpoint_id: str = DEFAULT_AGENT_ENDPOINT_ID


class HermesTunnelStatus(BaseModel):
    active: bool = False
    endpoint_id: str = DEFAULT_AGENT_ENDPOINT_ID
    ssh_target: str | None = None
    local_base_url: str | None = None
    local_port: int | None = None
    remote_host: str = "127.0.0.1"
    remote_port: int = 8642
    pid: int | None = None
    healthy: bool = False
    health_error: str | None = None
    profile: AgentEndpointProfile | None = None


_HERMES_TUNNELS: dict[str, dict[str, Any]] = {}


class ToolkitSkillResourceCounts(BaseModel):
    scripts: int = 0
    references: int = 0
    assets: int = 0


class ToolkitSkill(BaseModel):
    skill_id: str = ""
    name: str
    description: str = ""
    source: str
    available_sources: list[str] = Field(default_factory=list)
    path: str | None = None
    revision: str = ""
    content_revision: str = ""
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    shadowed_paths: list[str] = Field(default_factory=list)
    catalog_status: str = "active"
    canonical_skill_id: str | None = None
    quality_issues: list[str] = Field(default_factory=list)
    resource_counts: ToolkitSkillResourceCounts = ToolkitSkillResourceCounts()
    tool_count: int = 0


class ToolkitTool(BaseModel):
    name: str
    description: str = ""
    source: str
    required: list[str] = []


class ToolkitEndpointCapability(BaseModel):
    endpoint_id: str
    label: str
    kind: str = "hermes"
    enabled: bool = True
    ok: bool = False
    auth_configured: bool = False
    models: int = 0
    skills: int = 0
    toolsets: int = 0
    skill_names: list[str] = []
    toolset_names: list[str] = []
    warnings: list[str] = []


class ToolkitSkillRoot(BaseModel):
    label: str
    path: str
    exists: bool
    skill_count: int = 0


class ToolkitInventoryResponse(BaseModel):
    generated_at: str
    cached: bool = False
    cache_ttl_seconds: int
    counts: dict[str, int]
    skill_roots: list[ToolkitSkillRoot]
    skills: list[ToolkitSkill]
    tools: list[ToolkitTool]
    endpoint_capabilities: list[ToolkitEndpointCapability]
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_peony_logs_dir() -> Path:
    from bashgym.config import get_bashgym_dir

    d = get_bashgym_dir() / "peony_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_sessions_index_path() -> Path:
    return _get_peony_logs_dir() / "_sessions_index.json"


def _read_sessions_index() -> list[dict]:
    path = _get_sessions_index_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_sessions_index(index: list[dict]) -> None:
    path = _get_sessions_index_path()
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# System context gathering
# ---------------------------------------------------------------------------


async def _gather_system_context() -> str:
    """Gather current system state to give Peony full awareness."""
    sections = []

    # 1. System stats (traces, models, training)
    try:
        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        gold = (
            len(list((data_dir / "gold_traces").glob("*.json")))
            if (data_dir / "gold_traces").exists()
            else 0
        )
        silver = (
            len(list((data_dir / "silver_traces").glob("*.json")))
            if (data_dir / "silver_traces").exists()
            else 0
        )
        bronze = (
            len(list((data_dir / "bronze_traces").glob("*.json")))
            if (data_dir / "bronze_traces").exists()
            else 0
        )
        failed = (
            len(list((data_dir / "failed_traces").glob("*.json")))
            if (data_dir / "failed_traces").exists()
            else 0
        )
        models = len(list((data_dir / "models").iterdir())) if (data_dir / "models").exists() else 0
        pending = 0
        for trace_dir in [get_bashgym_dir() / "traces", data_dir / "traces"]:
            if trace_dir.exists():
                pending += len(list(trace_dir.glob("session_*.json")))
                pending += len(list(trace_dir.glob("session_*.jsonl")))
                pending += len(list(trace_dir.glob("imported_*.json")))
        sections.append(
            f"**System Stats:**\n"
            f"- Gold traces: {gold}\n"
            f"- Silver traces: {silver}\n"
            f"- Bronze traces: {bronze}\n"
            f"- Failed traces: {failed}\n"
            f"- Pending traces: {pending}\n"
            f"- Trained models: {models}\n"
            f"- Base model: {settings.training.base_model}"
        )
    except Exception as e:
        logger.debug(f"Could not gather system stats: {e}")

    # 2. GPU / hardware info
    try:
        from bashgym.api.system_info import get_system_info_service

        sysinfo = get_system_info_service().get_system_info()
        gpu_lines = []
        for gpu in sysinfo.gpus:
            gpu_lines.append(
                f"  - {gpu.model} | "
                f"VRAM: {gpu.vram_used:.1f}/{gpu.vram:.1f} GB | Util: {gpu.utilization}%"
            )
        sections.append(
            f"**Hardware:**\n"
            f"- Platform: {sysinfo.platform_name} ({sysinfo.arch})\n"
            f"- RAM: {sysinfo.available_ram:.1f}/{sysinfo.total_ram:.1f} GB available\n"
            f"- CUDA: {'Yes (' + sysinfo.cuda_version + ')' if sysinfo.cuda_available else 'No'}\n"
            f"- GPUs:\n" + ("\n".join(gpu_lines) if gpu_lines else "  - None detected")
        )
    except Exception as e:
        logger.debug(f"Could not gather hardware info: {e}")

    # 3. Trace repos
    try:
        from bashgym.trace_capture.importers.claude_history import ClaudeSessionImporter

        importer = ClaudeSessionImporter()
        projects_dir = importer.find_projects_dir()
        if projects_dir and projects_dir.exists():
            repos = [
                {"name": d.name, "trace_count": len(list(d.glob("*.jsonl")))}
                for d in projects_dir.iterdir()
                if d.is_dir()
            ]
            repos.sort(key=lambda r: r["trace_count"], reverse=True)
            if repos:
                repo_lines = [
                    f"  - {r['name']}: {r.get('trace_count', '?')} sessions" for r in repos[:10]
                ]
                sections.append(
                    f"**Trace Repositories ({len(repos)} total):**\n" + "\n".join(repo_lines)
                )
    except Exception as e:
        logger.debug(f"Could not gather trace repos: {e}")

    # 4. Training configuration
    try:
        from bashgym.config import get_settings

        settings = get_settings()
        training_defaults = TrainingRequest()
        sections.append(
            f"**Training Config:**\n"
            f"- Base model: {settings.training.base_model}\n"
            f"- Direct strategies: SFT, DPO, GRPO, RLVR, distillation, session distillation\n"
            f"- Default artifact retention: {training_defaults.artifact_retention.value}\n"
            f"- Default checkpoint limit: {training_defaults.checkpoint_limit}\n"
            f"- Default HF visibility: {'private' if training_defaults.hf_private else 'public'}\n"
            f"- Default HF upload artifact: {training_defaults.hf_upload_artifact.value}\n"
            f"- Auto-export GGUF: {settings.training.auto_export_gguf}\n"
            f"- Max sequence length: {settings.training.max_seq_length}"
        )
    except Exception as e:
        logger.debug(f"Could not gather training config: {e}")

    # 5. Active orchestration jobs
    try:
        from bashgym.api.orchestrator_routes import _jobs

        active = [j for j in _jobs.values() if j["status"] in ("decomposing", "executing")]
        if active:
            job_lines = [
                f"  - {j['id']}: {j['status']} ({j.get('spec', {}).title if j.get('spec') else 'untitled'})"
                for j in active
            ]
            sections.append(
                f"**Active Orchestration Jobs ({len(active)}):**\n" + "\n".join(job_lines)
            )
    except Exception as e:
        logger.debug(f"Could not gather orchestration status: {e}")

    return "\n\n".join(sections) if sections else "System context unavailable."


def _build_system_prompt(context: str, memory_prompt: str = "", skill_knowledge: str = "") -> str:
    sections = [
        # 1. Core identity
        "You are Peony — the botanical assistant for Bash Gym, a self-improving "
        "agentic development gym that captures coding sessions, transforms them into training "
        "data, and fine-tunes models.",
        # 2. Tool usage instructions
        "You have access to tools that let you take real actions in the system. Use them "
        "when the user asks you to do something actionable (import traces, search models, "
        "start training, etc.). For questions and analysis, respond directly.\n\n"
        "When you use a tool, briefly tell the user what you're doing before/after. "
        "Summarize tool results in natural language — don't dump raw JSON.\n\n"
        "For run_shell_command: only use this as a last resort when no structured tool covers "
        "the need. Always provide a clear reason.",
        # 3. Capabilities summary (always present)
        _tool_registry.capabilities_summary(),
    ]

    # 4. Memory prompt (if non-empty)
    if memory_prompt:
        sections.append(memory_prompt)

    # 5. Current system state
    sections.append(f"--- CURRENT SYSTEM STATE ---\n{context}\n--- END SYSTEM STATE ---")

    # 6. Skill knowledge (if non-empty)
    if skill_knowledge:
        sections.append(
            f"--- RELEVANT SKILL KNOWLEDGE ---\n{skill_knowledge}\n--- END SKILL KNOWLEDGE ---"
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


async def _execute_tool(name: str, tool_input: dict) -> str:
    """Execute a Peony tool and return a string result for the tool_result block."""

    if name in HF_CONTEXT_TOOL_NAMES:
        try:
            result = await execute_hf_context_tool(name, tool_input)
            return json.dumps(result)
        except HFContextToolError as exc:
            return json.dumps(exc.as_dict())

    if name in SKILL_LAB_TOOL_NAMES:
        try:
            result = await execute_skill_lab_tool(name, tool_input)
            return json.dumps(result)
        except SkillLabToolError as exc:
            return json.dumps(exc.as_dict())

    # ----- Memory tools (local, no HTTP needed) -----
    if name == "remember_fact":
        fact = _memory.remember_fact(tool_input["category"], tool_input["content"])
        return json.dumps({"status": "remembered", "fact": fact})

    if name == "recall_facts":
        facts = _memory.recall_facts(
            category=tool_input.get("category"),
            keyword=tool_input.get("keyword"),
        )
        return json.dumps({"facts": facts, "count": len(facts)})

    if name == "forget_fact":
        try:
            _memory.forget_fact(tool_input["fact_id"])
            return json.dumps({"status": "forgotten"})
        except ValueError as e:
            return json.dumps({"error": str(e)})

    if name == "update_user_profile":
        try:
            _memory.update_profile(tool_input["field"], tool_input["value"])
            profile = _memory.load_profile()
            return json.dumps({"status": "updated", "profile": profile})
        except ValueError as e:
            return json.dumps({"error": str(e)})

    # ----- Awareness tools (local, no HTTP needed) -----
    if name == "list_my_capabilities":
        result = _tool_registry.list_capabilities(category=tool_input.get("category"))
        return result

    # ----- Data collection tools (local, no HTTP needed) -----
    if name == "import_traces":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner

        scanner = ClaudeDataScanner()
        sources = tool_input.get("sources", ["all"])
        dry_run = tool_input.get("dry_run", False)
        project_filter = tool_input.get("project_filter")

        # Calculate 'since' from days parameter
        since = None
        days = tool_input.get("days")
        if days:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Resolve "all" or "sessions" — session import goes through existing HTTP endpoint
        source_list = None  # None means all in scanner
        if "all" not in sources:
            source_list = [s for s in sources if s != "sessions"]

        results = {}

        # Handle session import via existing endpoint if requested
        if source_list is None or "sessions" in sources:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post("http://localhost:8003/api/traces/import")
                    resp.raise_for_status()
                    results["sessions"] = resp.json()
            except Exception as e:
                results["sessions"] = {"error": str(e)}

        if dry_run:
            scan_results = scanner.scan_all(
                sources=source_list, since=since, project_filter=project_filter
            )
            for src, scan_result in scan_results.items():
                results[src] = {
                    "total_found": scan_result.total_found,
                    "already_collected": scan_result.already_collected,
                    "new_available": scan_result.new_available,
                }
        else:
            collect_results = scanner.collect_all(
                sources=source_list, since=since, project_filter=project_filter
            )
            for src, batch_result in collect_results.items():
                results[src] = {
                    "collected": batch_result.collected_count,
                    "skipped": batch_result.skipped_count,
                    "errors": batch_result.error_count,
                }

        return json.dumps(results)

    if name == "scan_claude_data":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner

        scanner = ClaudeDataScanner()
        scan_results = scanner.scan_all()
        results = {}
        for src, scan_result in scan_results.items():
            results[src] = {
                "total_found": scan_result.total_found,
                "already_collected": scan_result.already_collected,
                "new_available": scan_result.new_available,
            }
        return json.dumps(results)

    if name == "get_collection_status":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner

        scanner = ClaudeDataScanner()
        status = scanner.status()
        return json.dumps(status)

    # ----- HTTP-based tools -----
    import httpx

    base_url = "http://localhost:8003"

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if name == "get_trace_status":
                resp = await client.get(f"{base_url}/api/stats")
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "classify_pending_traces":
                dry_run = tool_input.get("dry_run", True)
                resp = await client.post(
                    f"{base_url}/api/traces/auto-classify",
                    params={
                        "dry_run": str(dry_run).lower(),
                        "auto_promote": str(not dry_run).lower(),
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "start_training":
                payload = {
                    "strategy": tool_input.get("strategy", "sft"),
                }
                training_config = tool_input.get("config") or {}
                if not isinstance(training_config, dict):
                    raise ValueError("start_training config must be an object")
                controlled_fields = {
                    "strategy",
                    "base_model",
                    "compute_target",
                    "correlation_id",
                    "origin",
                    "tracking",
                }
                blocked_fields = sorted(set(training_config).intersection(controlled_fields))
                if blocked_fields:
                    raise ValueError(
                        "start_training config must not override top-level fields: "
                        + ", ".join(blocked_fields)
                    )
                unknown_fields = sorted(
                    set(training_config).difference(TrainingRequest.model_fields)
                )
                if unknown_fields:
                    raise ValueError(
                        "start_training config contains unsupported fields: "
                        + ", ".join(unknown_fields)
                    )
                payload.update(training_config)
                if tool_input.get("model"):
                    payload["base_model"] = tool_input["model"]
                if tool_input.get("dataset_path"):
                    payload["dataset_path"] = tool_input["dataset_path"]
                if tool_input.get("correlation_id"):
                    payload["correlation_id"] = tool_input["correlation_id"]
                if tool_input.get("compute_target"):
                    payload["compute_target"] = tool_input["compute_target"]
                if tool_input.get("tracking_context"):
                    payload["tracking"] = tool_input["tracking_context"]
                payload["origin"] = {
                    "kind": "agent",
                    "agent": "hermes",
                    **(tool_input.get("origin") or {}),
                }
                payload = normalize_training_target_payload(payload)
                resp = await client.post(f"{base_url}/api/training/start", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "get_training_status":
                resp = await client.get(f"{base_url}/api/training")
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name in {
                "list_experiment_projects",
                "get_experiment_context",
                "get_experiment_run",
            }:
                from bashgym.config import get_bashgym_dir
                from bashgym.ledger.persistence import ExperimentLedgerRepository
                from bashgym.ledger.synthesis import build_project_context

                repository = ExperimentLedgerRepository(
                    get_bashgym_dir() / "campaigns" / "campaigns.sqlite3"
                )
                repository.initialize()
                workspace_id = str(tool_input.get("workspace_id") or "desktop-local")
                if name == "list_experiment_projects":
                    data = {
                        "schema_version": "experiment_projects.v1",
                        "workspace_id": workspace_id,
                        "projects": repository.list_projects(workspace_id),
                        "database_health": repository.database_health(workspace_id),
                    }
                elif name == "get_experiment_context":
                    data = build_project_context(
                        repository,
                        workspace_id,
                        str(tool_input["project_id"]),
                        recent_limit=int(tool_input.get("recent_limit") or 20),
                    )
                else:
                    data = repository.run_details(
                        workspace_id,
                        str(tool_input["project_id"]),
                        str(tool_input["run_id"]),
                    )
                return json.dumps(data)

            elif name == "start_data_designer":
                payload = {
                    "pipeline": tool_input["pipeline"],
                    "num_records": tool_input.get("num_records", 100),
                    "seed_type": tool_input.get("seed_type", "traces"),
                    "provider": tool_input.get("provider", "nvidia"),
                    "origin": {
                        "kind": "agent",
                        "agent": "hermes",
                        **(tool_input.get("origin") or {}),
                    },
                }
                if tool_input.get("seed_source"):
                    payload["seed_source"] = tool_input["seed_source"]
                if tool_input.get("model"):
                    payload["text_model"] = tool_input["model"]
                if tool_input.get("provider_endpoint"):
                    payload["provider_endpoint"] = tool_input["provider_endpoint"]
                resp = await client.post(f"{base_url}/api/factory/designer/create", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_search_models":
                params = {}
                if tool_input.get("task"):
                    params["task"] = tool_input["task"]
                if tool_input.get("sort"):
                    params["sort"] = tool_input["sort"]
                if tool_input.get("limit"):
                    params["limit"] = tool_input["limit"]
                resp = await client.get(f"{base_url}/api/hf/models/search", params=params)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_get_job_status":
                job_id = tool_input["job_id"]
                resp = await client.get(f"{base_url}/api/hf/jobs/{job_id}")
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_test_inference":
                payload = {
                    "model": tool_input["model_id"],
                    "prompt": tool_input["prompt"],
                }
                resp = await client.post(f"{base_url}/api/hf/inference/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            elif name == "hf_evaluate_model":
                payload = {
                    "model_id": tool_input["model_id"],
                    "metric": tool_input.get("metric", "accuracy"),
                }
                resp = await client.post(f"{base_url}/api/hf/evaluate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return json.dumps(data)

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except httpx.HTTPStatusError as e:
            return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# External agent endpoint routes
# ---------------------------------------------------------------------------


@router.get("/endpoints", response_model=AgentEndpointListResponse)
async def list_agent_endpoints():
    """List configured Hermes-compatible agent endpoints without secrets."""
    endpoints = _load_agent_endpoint_profiles()
    profiles = []
    for endpoint_id, profile in sorted(endpoints.items()):
        profile = {**profile, "id": endpoint_id}
        profiles.append(_public_agent_profile(profile))
    return AgentEndpointListResponse(endpoints=profiles)


@router.get("/hermes/setup-status", response_model=HermesSetupStatus)
async def get_hermes_setup_status(
    endpoint_id: str = DEFAULT_AGENT_ENDPOINT_ID,
    base_url: str = "http://127.0.0.1:8642/v1",
):
    """Inspect local Hermes gateway readiness for a fresh canvas node."""
    return await _hermes_setup_status(endpoint_id=endpoint_id, base_url=base_url)


@router.post("/hermes/quick-setup", response_model=HermesQuickSetupResponse)
async def quick_setup_hermes(request: HermesQuickSetupRequest):
    """Configure the local Hermes API server profile and optionally start the gateway."""
    endpoint_id = _validate_endpoint_id(request.profile_id or DEFAULT_AGENT_ENDPOINT_ID)
    base_url = _normalize_agent_base_url(request.base_url)
    actions: list[str] = []
    command = _hermes_command()

    env_path = _hermes_env_path(command)
    env_values = _parse_env_file(env_path)
    api_key = (
        request.api_key.strip()
        if request.api_key and request.api_key.strip()
        else _get_agent_api_key(endpoint_id)
        or env_values.get("API_SERVER_KEY")
        or secrets.token_urlsafe(32)
    )

    _save_agent_endpoint_profile(
        endpoint_id,
        label=request.label,
        base_url=base_url,
        model=request.model,
        model_options=request.model_options,
        session_key=request.session_key,
        api_key=api_key,
    )
    actions.append("Saved BashGym Hermes endpoint profile")

    if request.write_env:
        configured_with_cli = False
        if command:
            enabled_ok = _hermes_config_set(command, "API_SERVER_ENABLED", "true")
            configured_with_cli = enabled_ok
            if enabled_ok:
                actions.append("Enabled Hermes API server via hermes config")

        _upsert_env_file(
            env_path,
            {
                "API_SERVER_ENABLED": "true",
                "API_SERVER_KEY": api_key,
            },
        )
        actions.append(
            "Updated Hermes .env API server fallback"
            if configured_with_cli
            else "Updated Hermes .env API server settings"
        )

    healthy, _ = await _probe_hermes_health(base_url, api_key)
    log_path: str | None = None
    if request.start_gateway and not healthy:
        gateway_command = _hermes_gateway_command(command)
        if not gateway_command:
            actions.append("Hermes CLI not found; gateway was not started")
        else:
            try:
                log_path = _start_hermes_gateway(gateway_command)
                actions.append("Started Hermes gateway process")
                time.sleep(4)
            except Exception as exc:
                actions.append(f"Gateway start failed: {exc}")

    status = await _hermes_setup_status(
        endpoint_id=endpoint_id,
        base_url=base_url,
        log_path=log_path,
    )
    return HermesQuickSetupResponse(status=status, actions=actions)


@router.get("/hermes/tunnel/status", response_model=HermesTunnelStatus)
async def get_hermes_tunnel_status(endpoint_id: str = DEFAULT_AGENT_ENDPOINT_ID):
    """Return the local SSH tunnel status for a remote Hermes gateway."""
    return await _hermes_tunnel_status(endpoint_id)


@router.post("/hermes/tunnel/connect", response_model=HermesTunnelStatus)
async def connect_hermes_tunnel(request: HermesTunnelRequest):
    """Start a local SSH port-forward to a remote Hermes API server."""
    endpoint_id = _validate_endpoint_id(request.endpoint_id or DEFAULT_AGENT_ENDPOINT_ID)
    ssh_target = _validate_ssh_target(request.ssh_target)
    remote_host = _validate_tunnel_host(request.remote_host or "127.0.0.1")
    remote_port = _validate_port(request.remote_port, field="Remote port")

    _stop_hermes_tunnel(endpoint_id)
    if request.local_port is None:
        local_port = _find_available_local_port()
    else:
        local_port = _validate_port(request.local_port, field="Local port")
        if not _can_bind_local_port(local_port):
            raise HTTPException(
                status_code=409, detail=f"Local port {local_port} is already in use"
            )

    state = _start_hermes_ssh_tunnel(
        endpoint_id=endpoint_id,
        ssh_target=ssh_target,
        local_port=local_port,
        remote_host=remote_host,
        remote_port=remote_port,
    )
    local_base_url = str(state["local_base_url"])
    api_key = (
        request.api_key.strip()
        if request.api_key and request.api_key.strip()
        else _get_agent_api_key(endpoint_id)
    )

    profile: AgentEndpointProfile | None = None
    if request.save_profile:
        profile = _save_agent_endpoint_profile(
            endpoint_id,
            label=request.label or "Hermes",
            base_url=local_base_url,
            model=request.model or "hermes-agent",
            model_options=request.model_options,
            session_key=request.session_key,
            api_key=api_key,
        )

    healthy = False
    health_error: str | None = None
    process = state.get("process")
    for _ in range(20):
        if not _process_active(process):
            log_text = _tail_file(Path(str(state.get("log_path") or "")))
            _HERMES_TUNNELS.pop(endpoint_id, None)
            detail = log_text or "SSH tunnel process exited before the forward became available"
            raise HTTPException(status_code=502, detail=detail)
        healthy, health_error = await _probe_hermes_health(local_base_url, api_key)
        if healthy:
            break
        await asyncio.sleep(0.25)

    if profile is None:
        profile = _public_agent_profile(_get_agent_endpoint_profile(endpoint_id))

    return HermesTunnelStatus(
        active=_process_active(process),
        endpoint_id=endpoint_id,
        ssh_target=ssh_target,
        local_base_url=local_base_url,
        local_port=local_port,
        remote_host=remote_host,
        remote_port=remote_port,
        pid=int(state.get("pid") or 0) or None,
        healthy=healthy,
        health_error=health_error,
        profile=profile,
    )


@router.post("/hermes/tunnel/disconnect", response_model=HermesTunnelStatus)
async def disconnect_hermes_tunnel(request: HermesTunnelDisconnectRequest):
    """Stop a BashGym-managed local SSH port-forward for Hermes."""
    endpoint_id = _validate_endpoint_id(request.endpoint_id or DEFAULT_AGENT_ENDPOINT_ID)
    _stop_hermes_tunnel(endpoint_id)
    return await _hermes_tunnel_status(endpoint_id)


@router.put("/endpoints/{endpoint_id}", response_model=AgentEndpointProfile)
async def save_agent_endpoint(endpoint_id: str, request: AgentEndpointUpdate):
    """Create or update an agent endpoint profile."""
    endpoint_id = _validate_endpoint_id(endpoint_id)
    session_key = request.session_key.strip() if request.session_key else None
    if session_key and any(ord(ch) < 32 or ord(ch) == 127 for ch in session_key):
        raise HTTPException(
            status_code=400,
            detail="Session key cannot contain control characters",
        )

    profile = {
        "id": endpoint_id,
        "label": (request.label or "Hermes").strip()[:80],
        "kind": (request.kind or "hermes").strip().lower()[:40],
        "base_url": _normalize_agent_base_url(request.base_url),
        "model": (request.model or "hermes-agent").strip()[:120],
        "model_options": _normalize_model_options(
            (request.model or "hermes-agent").strip()[:120],
            request.model_options,
        ),
        "session_key": session_key[:256] if session_key else None,
        "enabled": bool(request.enabled),
    }

    endpoints = _read_agent_endpoint_config()
    endpoints[endpoint_id] = profile
    _write_agent_endpoint_config(endpoints)

    if request.clear_api_key:
        _delete_agent_api_key(endpoint_id)
    else:
        _set_agent_api_key(endpoint_id, request.api_key)

    return _public_agent_profile(profile)


@router.delete("/endpoints/{endpoint_id}")
async def delete_agent_endpoint(endpoint_id: str):
    """Delete an agent endpoint profile and its stored local secret."""
    endpoint_id = _validate_endpoint_id(endpoint_id)
    endpoints = _read_agent_endpoint_config()
    existed = endpoint_id in endpoints
    if existed:
        del endpoints[endpoint_id]
        _write_agent_endpoint_config(endpoints)
    _delete_agent_api_key(endpoint_id)
    return {"status": "ok", "endpoint_id": endpoint_id, "deleted": existed}


def _count_endpoint_items(data: Any, keys: tuple[str, ...]) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, dict):
                return len(value)
    return 0


def _extract_named_endpoint_items(data: Any, keys: tuple[str, ...], limit: int = 250) -> list[str]:
    candidates: Any = data
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, (list, dict)):
                candidates = value
                break

    if isinstance(candidates, dict):
        iterable = candidates.values()
    elif isinstance(candidates, list):
        iterable = candidates
    else:
        return []

    names: list[str] = []
    for item in iterable:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("id") or item.get("label") or "")
        else:
            name = ""
        if name:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _hermes_cli_path(command: str | None, *args: str) -> Path | None:
    if not command:
        return None
    try:
        result = subprocess.run(
            [command, "config", *args],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    return Path(value).expanduser() if value else None


def _hermes_env_path(command: str | None = None) -> Path:
    explicit = os.environ.get("HERMES_ENV_PATH")
    if explicit:
        return Path(explicit).expanduser()
    home_override = os.environ.get("HERMES_HOME")
    if home_override:
        return Path(home_override).expanduser() / ".env"

    discovered = _hermes_cli_path(command, "env-path")
    if discovered:
        return discovered

    return _hermes_home(command) / ".env"


def _hermes_config_path(command: str | None = None) -> Path | None:
    explicit = os.environ.get("HERMES_CONFIG_PATH")
    if explicit:
        return Path(explicit).expanduser()
    home_override = os.environ.get("HERMES_HOME")
    if home_override:
        return Path(home_override).expanduser() / "config.yaml"
    return _hermes_cli_path(command, "path")


def _hermes_home(command: str | None = None) -> Path:
    explicit = os.environ.get("HERMES_HOME")
    if explicit:
        return Path(explicit).expanduser()

    env_path = _hermes_cli_path(command, "env-path")
    if env_path:
        return env_path.parent

    return Path.home() / ".hermes"


def _default_hermes_home() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "hermes"
    return Path.home() / ".hermes"


def _active_hermes_skill_root() -> Path:
    explicit = os.environ.get("HERMES_HOME")
    if explicit:
        return Path(explicit).expanduser() / "skills"

    default_root = _default_hermes_home() / "skills"
    if default_root.exists():
        return default_root
    return _hermes_home(_hermes_command()) / "skills"


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _upsert_env_file(path: Path, updates: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = (
        path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    )
    seen: set[str] = set()
    output: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")
    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def _hermes_config_set(command: str | None, key: str, value: str) -> bool:
    if not command:
        return False
    try:
        result = subprocess.run(
            [command, "config", "set", key, value],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        logger.debug("Hermes config set failed for %s: %s", key, exc)
        return False
    if result.returncode != 0:
        logger.debug("Hermes config set failed for %s with exit %s", key, result.returncode)
        return False
    return True


def _truthy_config_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _hermes_api_server_enabled(env_values: dict[str, str], config_path: Path | None = None) -> bool:
    if _truthy_config_value(env_values.get("API_SERVER_ENABLED", "")):
        return True
    if not config_path or not config_path.exists():
        return False

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(config_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        data = None

    if not isinstance(data, dict):
        return False

    candidates = [
        data.get("api_server"),
        (
            (data.get("platforms") or {}).get("api_server")
            if isinstance(data.get("platforms"), dict)
            else None
        ),
        (
            (data.get("gateway") or {}).get("api_server")
            if isinstance(data.get("gateway"), dict)
            else None
        ),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and _truthy_config_value(candidate.get("enabled", "")):
            return True
    return False


def _hermes_config_summary(config_path: Path | None) -> tuple[str | None, str | None]:
    if not config_path or not config_path.exists():
        return None, None

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(config_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None, None

    if not isinstance(data, dict):
        return None, None

    model_config = data.get("model")
    if isinstance(model_config, str):
        return model_config, None
    if isinstance(model_config, dict):
        model = model_config.get("default") or model_config.get("model") or model_config.get("name")
        provider = model_config.get("provider")
        return (
            str(model) if model else None,
            str(provider) if provider else None,
        )
    return None, None


def _hermes_command() -> str | None:
    return shutil.which("hermes")


def _hermes_gateway_command(command: str | None) -> list[str]:
    if not command:
        return []
    try:
        result = subprocess.run(
            [command, "gateway", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode == 0:
            return [command, "gateway", "run"]
    except Exception:
        pass
    return [command, "gateway"]


async def _probe_hermes_health(
    base_url: str, api_key: str | None = None
) -> tuple[bool, str | None]:
    url = _normalize_agent_base_url(base_url)
    health_root = url[:-3] if url.endswith("/v1") else url
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    probe_url = f"{url}/capabilities" if api_key else f"{health_root}/health"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=2.0)) as client:
            response = await client.get(probe_url, headers=headers)
            if response.is_success:
                return True, None
            if response.status_code in {401, 403}:
                return False, "Saved API server key was rejected"
            return False, f"HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc) or exc.__class__.__name__


def _validate_ssh_target(target: str) -> str:
    value = target.strip()
    if not value or value.startswith("-") or not _SAFE_SSH_TARGET.match(value):
        raise HTTPException(
            status_code=400,
            detail="SSH target must be a host or SSH config alias without spaces",
        )
    return value


def _validate_tunnel_host(host: str) -> str:
    value = host.strip()
    if not value or value.startswith("-") or not _SAFE_TUNNEL_HOST.match(value):
        raise HTTPException(status_code=400, detail="Remote host is invalid")
    return value


def _validate_port(port: int, *, field: str) -> int:
    if port < 1 or port > 65535:
        raise HTTPException(status_code=400, detail=f"{field} must be between 1 and 65535")
    return port


def _can_bind_local_port(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def _find_available_local_port(preferred: int = 18642) -> int:
    start = _validate_port(preferred, field="Local port")
    for port in range(start, min(start + 100, 65536)):
        if _can_bind_local_port(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tunnel_log_path(endpoint_id: str) -> Path:
    from bashgym.config import get_bashgym_dir

    log_dir = get_bashgym_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"hermes-tunnel-{endpoint_id}.log"


def _tail_file(path: Path, limit: int = 1200) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:].strip()


def _forwarded_local_port(base_url: str) -> int | None:
    try:
        parsed = urlparse(_normalize_agent_base_url(base_url))
    except HTTPException:
        return None
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return None
    if parsed.port is None or parsed.port == 8642:
        return None
    return parsed.port


def _process_active(process: Any) -> bool:
    poll = getattr(process, "poll", None)
    return process is not None and callable(poll) and poll() is None


def _stop_forwarded_tunnel_process(port: int) -> bool:
    try:
        import psutil  # type: ignore
    except Exception:
        return False

    for conn in psutil.net_connections(kind="tcp"):
        laddr = getattr(conn, "laddr", None)
        conn_port = getattr(laddr, "port", None)
        if conn_port != port or not conn.pid:
            continue
        try:
            process = psutil.Process(conn.pid)
            name = process.name().lower()
            cmdline = " ".join(process.cmdline()).lower()
            if "ssh" not in name and "ssh" not in cmdline:
                continue
            process.terminate()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()
            return True
        except Exception:
            continue
    return False


def _stop_hermes_tunnel(endpoint_id: str) -> None:
    state = _HERMES_TUNNELS.pop(endpoint_id, None)
    process = state.get("process") if isinstance(state, dict) else None
    if _process_active(process):
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        return

    try:
        profile = _public_agent_profile(_get_agent_endpoint_profile(endpoint_id))
    except HTTPException:
        return
    local_port = _forwarded_local_port(profile.base_url)
    if local_port:
        _stop_forwarded_tunnel_process(local_port)


def _start_hermes_ssh_tunnel(
    *,
    endpoint_id: str,
    ssh_target: str,
    local_port: int,
    remote_host: str,
    remote_port: int,
) -> dict[str, Any]:
    ssh_command = shutil.which("ssh") or shutil.which("ssh.exe")
    if not ssh_command:
        raise HTTPException(status_code=503, detail="OpenSSH client was not found on this machine")

    log_path = _tunnel_log_path(endpoint_id)
    forward = f"127.0.0.1:{local_port}:{remote_host}:{remote_port}"
    command = [
        ssh_command,
        "-N",
        "-L",
        forward,
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=2",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        ssh_target,
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
    except Exception as exc:
        log_handle.close()
        raise HTTPException(status_code=502, detail=f"Failed to start SSH tunnel: {exc}") from exc
    log_handle.close()

    state = {
        "process": process,
        "endpoint_id": endpoint_id,
        "ssh_target": ssh_target,
        "local_port": local_port,
        "remote_host": remote_host,
        "remote_port": remote_port,
        "local_base_url": f"http://127.0.0.1:{local_port}/v1",
        "pid": getattr(process, "pid", None),
        "log_path": str(log_path),
    }
    _HERMES_TUNNELS[endpoint_id] = state
    return state


async def _hermes_tunnel_status(
    endpoint_id: str = DEFAULT_AGENT_ENDPOINT_ID,
) -> HermesTunnelStatus:
    endpoint_id = _validate_endpoint_id(endpoint_id)
    profile: AgentEndpointProfile | None = None
    try:
        profile = _public_agent_profile(_get_agent_endpoint_profile(endpoint_id))
    except HTTPException:
        profile = None

    state = _HERMES_TUNNELS.get(endpoint_id)
    if not state:
        if profile:
            local_port = _forwarded_local_port(profile.base_url)
            if local_port:
                healthy, health_error = await _probe_hermes_health(
                    profile.base_url,
                    _get_agent_api_key(endpoint_id),
                )
                return HermesTunnelStatus(
                    active=healthy,
                    endpoint_id=endpoint_id,
                    local_base_url=profile.base_url,
                    local_port=local_port,
                    healthy=healthy,
                    health_error=health_error,
                    profile=profile,
                )
        return HermesTunnelStatus(endpoint_id=endpoint_id, profile=profile)

    process = state.get("process")
    active = _process_active(process)
    local_base_url = str(state.get("local_base_url") or "")
    healthy = False
    health_error = "Tunnel process exited" if not active else None
    if active and local_base_url:
        healthy, health_error = await _probe_hermes_health(
            local_base_url,
            _get_agent_api_key(endpoint_id),
        )
    if not active:
        _HERMES_TUNNELS.pop(endpoint_id, None)

    return HermesTunnelStatus(
        active=active,
        endpoint_id=endpoint_id,
        ssh_target=str(state.get("ssh_target") or "") or None,
        local_base_url=local_base_url or None,
        local_port=int(state.get("local_port") or 0) or None,
        remote_host=str(state.get("remote_host") or "127.0.0.1"),
        remote_port=int(state.get("remote_port") or 8642),
        pid=int(state.get("pid") or 0) or None,
        healthy=healthy,
        health_error=health_error,
        profile=profile,
    )


def _save_agent_endpoint_profile(
    endpoint_id: str,
    *,
    label: str,
    base_url: str,
    model: str,
    model_options: list[str] | None = None,
    session_key: str | None,
    api_key: str | None,
) -> AgentEndpointProfile:
    endpoint_id = _validate_endpoint_id(endpoint_id)
    normalized_model = (model or "hermes-agent").strip()[:120]
    profile = {
        "id": endpoint_id,
        "label": (label or "Hermes").strip()[:80],
        "kind": "hermes",
        "base_url": _normalize_agent_base_url(base_url),
        "model": normalized_model,
        "model_options": _normalize_model_options(normalized_model, model_options or []),
        "session_key": (session_key or None),
        "enabled": True,
    }
    endpoints = _read_agent_endpoint_config()
    endpoints[endpoint_id] = profile
    _write_agent_endpoint_config(endpoints)
    _set_agent_api_key(endpoint_id, api_key)
    return _public_agent_profile(profile)


async def _hermes_setup_status(
    *,
    endpoint_id: str = DEFAULT_AGENT_ENDPOINT_ID,
    base_url: str = "http://127.0.0.1:8642/v1",
    log_path: str | None = None,
) -> HermesSetupStatus:
    endpoint_id = _validate_endpoint_id(endpoint_id)
    command = _hermes_command()
    env_path = _hermes_env_path(command)
    config_path = _hermes_config_path(command)
    env_values = _parse_env_file(env_path)
    api_server_enabled = _hermes_api_server_enabled(env_values, config_path)
    configured_model, configured_provider = _hermes_config_summary(config_path)
    profile = _get_agent_endpoint_profile(endpoint_id)
    public_profile = _public_agent_profile(profile)
    stored_api_key = _get_agent_api_key(endpoint_id)
    env_api_key = env_values.get("API_SERVER_KEY")
    parsed_base_url = urlparse(_normalize_agent_base_url(base_url))
    parsed_profile_url = urlparse(_normalize_agent_base_url(str(profile["base_url"])))
    is_local_gateway = (
        parsed_base_url.hostname in {"127.0.0.1", "localhost", "::1"}
        and parsed_base_url.port == 8642
        and parsed_profile_url.hostname in {"127.0.0.1", "localhost", "::1"}
        and parsed_profile_url.port == 8642
    )
    api_key = env_api_key if is_local_gateway and env_api_key else stored_api_key or env_api_key
    if is_local_gateway and env_api_key and stored_api_key != env_api_key:
        _set_agent_api_key(endpoint_id, env_api_key)
        public_profile = _public_agent_profile(profile)
    healthy, health_error = await _probe_hermes_health(base_url, api_key)
    setup_needed: list[str] = []

    if not command:
        setup_needed.append("Install Hermes CLI")
    if not api_server_enabled:
        setup_needed.append("Enable API server in Hermes config")
    if not env_values.get("API_SERVER_KEY") and not public_profile.api_key_configured:
        setup_needed.append("Create and save API server key")
    if not healthy:
        setup_needed.append("Start Hermes gateway")

    return HermesSetupStatus(
        installed=bool(command),
        command=command,
        gateway_command=_hermes_gateway_command(command),
        hermes_home=str(_hermes_home(command)),
        config_path=str(config_path) if config_path else None,
        configured_model=configured_model,
        configured_provider=configured_provider,
        env_path=str(env_path),
        env_exists=env_path.exists(),
        env_api_enabled=api_server_enabled,
        env_key_present=bool(env_values.get("API_SERVER_KEY")),
        gateway_url=_normalize_agent_base_url(base_url),
        gateway_healthy=healthy,
        gateway_error=health_error,
        profile=public_profile,
        setup_needed=setup_needed,
        log_path=log_path,
    )


def _start_hermes_gateway(command: list[str]) -> str | None:
    if not command:
        return None
    from bashgym.config import get_bashgym_dir

    log_dir = get_bashgym_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "hermes-gateway.log"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_handle = log_path.open("ab")
    try:
        subprocess.Popen(
            command,
            cwd=str(Path.home()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
    except Exception:
        log_handle.close()
        raise
    log_handle.close()
    return str(log_path)


async def _probe_agent_endpoint_profile(profile: dict[str, Any]) -> dict[str, Any]:
    endpoint_id = str(profile["id"])
    headers = _agent_headers(profile)
    probes: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
        for name, path, optional in (
            ("health", "health", False),
            ("health_detailed", "health/detailed", True),
            ("capabilities", "capabilities", False),
            ("models", "models", False),
            ("skills", "skills", False),
            ("toolsets", "toolsets", False),
        ):
            url = _endpoint_url(profile, path)
            try:
                response = await client.get(url, headers=headers)
                probes[name] = {
                    "ok": response.is_success,
                    "status_code": response.status_code,
                    "data": _json_or_text(response),
                }
                if not response.is_success and not optional:
                    warnings.append(f"{name} returned HTTP {response.status_code}")
            except Exception as exc:
                message = _sanitize_agent_error(str(exc), endpoint_id)
                probes[name] = {"ok": False, "error": message}
                if not optional:
                    warnings.append(f"{name} failed: {message}")

    models_data = probes.get("models", {}).get("data")
    skills_data = probes.get("skills", {}).get("data")
    toolsets_data = probes.get("toolsets", {}).get("data")
    authenticated_probe_names = ("capabilities", "models", "skills", "toolsets")
    rejected_auth = any(
        probes.get(name, {}).get("status_code") in {401, 403} for name in authenticated_probe_names
    )
    if rejected_auth:
        warnings.insert(
            0,
            "Hermes rejected the saved API server key. Update the endpoint API key and test again.",
        )
    ok = bool(probes.get("capabilities", {}).get("ok")) and not rejected_auth

    return {
        "ok": ok,
        "profile": _public_agent_profile(profile),
        "auth_configured": _get_agent_api_key(endpoint_id) is not None,
        "probes": probes,
        "summary": {
            "models": _count_endpoint_items(models_data, ("data", "models")),
            "skills": _count_endpoint_items(skills_data, ("skills", "data")),
            "toolsets": _count_endpoint_items(toolsets_data, ("toolsets", "data")),
        },
        "warnings": warnings,
    }


@router.post("/endpoints/{endpoint_id}/discover")
async def discover_agent_endpoint(endpoint_id: str):
    """Probe a Hermes-compatible endpoint for health and capabilities."""
    profile = _get_agent_endpoint_profile(endpoint_id)
    return await _probe_agent_endpoint_profile(profile)


TOOLKIT_CACHE_TTL_SECONDS = 60
_TOOLKIT_CACHE: dict[str, dict[str, Any]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _toolkit_skill_root_candidates() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    env_roots = os.environ.get("BASHGYM_SKILL_DIRS", "")
    for index, raw in enumerate([p for p in env_roots.split(os.pathsep) if p.strip()]):
        roots.append((f"env:{index + 1}", Path(raw).expanduser()))

    repo_root = _repo_root()
    roots.extend(
        [
            ("workspace", repo_root / "assistant" / "workspace" / "skills"),
            ("agents", Path.home() / ".agents" / "skills"),
            ("codex", Path.home() / ".codex" / "skills"),
            ("codex-system", Path.home() / ".codex" / "skills" / ".system"),
            ("claude", Path.home() / ".claude" / "skills"),
            ("hermes", _active_hermes_skill_root()),
        ]
    )

    seen: set[str] = set()
    unique: list[tuple[str, Path]] = []
    for label, path in roots:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique.append((label, path))
            seen.add(key)
    return unique


def _parse_frontmatter(skill_path: Path) -> tuple[dict[str, Any], str]:
    """Read SKILL.md frontmatter without making YAML a hard dependency."""
    try:
        content = skill_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, ""

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    end = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return {}, content

    raw_frontmatter = "\n".join(lines[1:end])
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(raw_frontmatter)
        return (parsed if isinstance(parsed, dict) else {}), content
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    except Exception:
        # Malformed optional YAML should not make the whole inventory disappear.
        pass

    parsed: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in raw_frontmatter.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") and current_list_key:
            parsed.setdefault(current_list_key, []).append(stripped[1:].strip().strip("\"'"))
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip().strip("\"'")
        if not value:
            parsed[key] = []
            current_list_key = key
        else:
            current_list_key = None
            if value.lower() in {"true", "false"}:
                parsed[key] = value.lower() == "true"
            elif value.startswith("[") and value.endswith("]"):
                parsed[key] = [
                    item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()
                ]
            else:
                parsed[key] = value
    return parsed, content


def _allowed_tools(frontmatter: dict[str, Any]) -> list[str]:
    raw_tools = frontmatter.get("allowed_tools", frontmatter.get("allowed-tools", []))
    if isinstance(raw_tools, str):
        raw_tools = raw_tools.replace(",", " ").split()
    if not isinstance(raw_tools, list):
        return []
    return list(dict.fromkeys(str(tool).strip() for tool in raw_tools if str(tool).strip()))


def _normalized_skill_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _skill_catalog_classification(
    frontmatter: dict[str, Any],
    description: str,
) -> tuple[str, list[str]]:
    status = str(frontmatter.get("status") or "").strip().casefold()
    deprecated = frontmatter.get("deprecated") is True or status in {
        "archived",
        "deprecated",
        "disabled",
        "retired",
    }
    if deprecated:
        return "deprecated", []

    issues: list[str] = []
    if not frontmatter:
        issues.append("missing_frontmatter")
    if not description.strip():
        issues.append("missing_description")
    return ("invalid" if issues else "active"), issues


def _stable_skill_id(name: str, source: str = "", path: str | Path | None = None) -> str:
    """Identify one loaded skill independently from its mutable content revision."""
    location = str(Path(path).resolve()) if path else name.strip().casefold()
    identity = f"{source.strip().casefold()}:{location.casefold()}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"skill-{digest[:20]}"


def _skill_content_revision(skill_path: Path) -> str:
    """Hash the skill instructions and every packaged runtime resource."""
    digest = hashlib.sha256()
    skill_dir = skill_path.parent
    included: list[Path] = [skill_path]
    for dirname in ("scripts", "references", "assets"):
        root = skill_dir / dirname
        if root.is_dir():
            included.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(set(included), key=lambda item: item.as_posix().casefold()):
        try:
            relative = path.relative_to(skill_dir).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        except OSError:
            continue
    return digest.hexdigest()


def _skill_metadata(skill_path: Path) -> tuple[str, str, dict[str, Any], str, list[str]]:
    frontmatter, content = _parse_frontmatter(skill_path)
    name = str(frontmatter.get("name") or skill_path.parent.name).strip() or skill_path.parent.name
    if not frontmatter:
        name = next(
            (
                line[2:].strip()
                for line in content.splitlines()[:20]
                if line.lower().startswith("# ") and line[2:].strip()
            ),
            name,
        )
    description = str(frontmatter.get("description") or "")
    revision = _skill_content_revision(skill_path)
    return name, description, frontmatter, revision, _allowed_tools(frontmatter)


def _read_skill_frontmatter(skill_path: Path) -> tuple[str, str]:
    name, description, _frontmatter, _revision, _allowed_tools_value = _skill_metadata(skill_path)
    return name, description


def _resource_counts(skill_dir: Path) -> ToolkitSkillResourceCounts:
    def count_files(dirname: str) -> int:
        path = skill_dir / dirname
        if not path.exists() or not path.is_dir():
            return 0
        return sum(1 for item in path.rglob("*") if item.is_file())

    return ToolkitSkillResourceCounts(
        scripts=count_files("scripts"),
        references=count_files("references"),
        assets=count_files("assets"),
    )


def _scan_skill_roots(
    max_skills: int = 400,
) -> tuple[list[ToolkitSkillRoot], list[ToolkitSkill], list[str]]:
    root_infos: list[ToolkitSkillRoot] = []
    skills: list[ToolkitSkill] = []
    warnings: list[str] = []
    seen_paths: set[str] = set()
    skills_by_name_revision: dict[tuple[str, str], ToolkitSkill] = {}
    skill_priorities: dict[str, tuple[int, int, str]] = {}

    skip_dirs = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "_archive",
        "archive",
        "deprecated",
    }
    embedded_host_dirs = {
        ".agents",
        ".claude",
        ".codex",
        ".cursor",
        ".factory",
        ".gbrain",
        ".hermes",
        ".kiro",
        ".openclaw",
        ".opencode",
        ".slate",
    }
    # Keep the cap per root. A global cap lets a large early root hide every
    # later root from the inventory.
    per_root_cap = max(1, max_skills)
    root_candidates = _toolkit_skill_root_candidates()
    for root_index, (label, root) in enumerate(root_candidates):
        root_count = 0
        exists = root.exists()
        if exists:
            try:
                for current, dirs, files in os.walk(root):
                    dirs[:] = [
                        d
                        for d in dirs
                        if d.casefold() not in skip_dirs and d.casefold() not in embedded_host_dirs
                    ]
                    dirs.sort()
                    files.sort()
                    if root_count >= per_root_cap:
                        dirs[:] = []
                        continue
                    depth = len(Path(current).relative_to(root).parts)
                    if depth > 6:
                        dirs[:] = []
                        continue
                    if "SKILL.md" not in files:
                        continue
                    skill_path = Path(current) / "SKILL.md"
                    key = str(skill_path.resolve())
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    name, description, frontmatter, revision, allowed_tools = _skill_metadata(
                        skill_path
                    )
                    catalog_status, quality_issues = _skill_catalog_classification(
                        frontmatter,
                        description,
                    )
                    root_count += 1
                    duplicate_key = (_normalized_skill_name(name), revision)
                    existing = skills_by_name_revision.get(duplicate_key)
                    if existing is not None:
                        existing.shadowed_paths.append(str(skill_path))
                        if label not in existing.available_sources:
                            existing.available_sources.append(label)
                        continue
                    skill = ToolkitSkill(
                        skill_id=_stable_skill_id(name, label, skill_path),
                        name=name,
                        description=description,
                        source=label,
                        available_sources=[label],
                        path=str(skill_path),
                        revision=revision,
                        content_revision=revision,
                        frontmatter=frontmatter,
                        allowed_tools=allowed_tools,
                        catalog_status=catalog_status,
                        quality_issues=quality_issues,
                        resource_counts=_resource_counts(skill_path.parent),
                    )
                    skills_by_name_revision[duplicate_key] = skill
                    skill_priorities[skill.skill_id] = (
                        root_index,
                        depth,
                        str(skill_path).casefold(),
                    )
                    skills.append(skill)
            except OSError as exc:
                warnings.append(f"Could not scan {label}: {exc}")
        root_infos.append(
            ToolkitSkillRoot(
                label=label,
                path=str(root),
                exists=exists,
                skill_count=root_count,
            )
        )
    peony_skills = _skill_registry.list_all()
    for skill in peony_skills:
        name = str(skill.get("name") or "unknown")
        revision = hashlib.sha256(
            json.dumps(skill, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        peony_skill = ToolkitSkill(
            skill_id=_stable_skill_id(name, "peony"),
            name=name,
            description=str(skill.get("description") or ""),
            source="peony",
            available_sources=["peony"],
            path=None,
            revision=revision,
            content_revision=revision,
            frontmatter={"name": name, "description": str(skill.get("description") or "")},
            allowed_tools=[
                str(tool.get("name") or "")
                for tool in skill.get("tools") or []
                if isinstance(tool, dict) and tool.get("name")
            ],
            resource_counts=ToolkitSkillResourceCounts(),
            tool_count=len(skill.get("tools") or []),
        )
        skill_priorities[peony_skill.skill_id] = (
            len(root_candidates),
            0,
            peony_skill.name.casefold(),
        )
        skills.append(peony_skill)

    active_by_name: dict[str, list[ToolkitSkill]] = {}
    for skill in skills:
        if skill.catalog_status != "active":
            continue
        key = _normalized_skill_name(skill.name)
        active_by_name.setdefault(key, []).append(skill)

    for candidates in active_by_name.values():
        canonical = min(
            candidates,
            key=lambda item: skill_priorities.get(item.skill_id, (999, 999, item.name)),
        )
        for skill in candidates:
            if skill.skill_id == canonical.skill_id:
                continue
            for source in skill.available_sources or [skill.source]:
                if source not in canonical.available_sources:
                    canonical.available_sources.append(source)
            skill.catalog_status = "alternate"
            skill.canonical_skill_id = canonical.skill_id
            if skill.path and skill.path not in canonical.shadowed_paths:
                canonical.shadowed_paths.append(skill.path)

    status_order = {"active": 0, "alternate": 1, "deprecated": 2, "invalid": 3}
    skills.sort(
        key=lambda item: (
            status_order.get(item.catalog_status, 9),
            item.name.casefold(),
            item.source,
        )
    )
    return root_infos, skills, warnings


def _tool_source_maps() -> tuple[dict[str, str], dict[str, list[str]]]:
    source_by_name: dict[str, str] = {}
    required_by_name: dict[str, list[str]] = {}
    for source, tools in (
        ("peony-core", CORE_TOOLS),
        ("peony-memory", MEMORY_TOOLS),
        ("peony-awareness", AWARENESS_TOOLS),
    ):
        for tool in tools:
            name = str(tool.get("name") or "")
            if name:
                source_by_name[name] = source
                required_by_name[name] = list(
                    (tool.get("input_schema") or {}).get("required") or []
                )

    for skill in _skill_registry.skills:
        skill_name = str(skill.get("name") or "skill")
        for tool in skill.get("tools", []) or []:
            name = str(tool.get("name") or "")
            if name and name not in source_by_name:
                source_by_name[name] = f"peony-skill:{skill_name}"
                required_by_name[name] = list(
                    (tool.get("input_schema") or {}).get("required") or []
                )

    return source_by_name, required_by_name


def _list_toolkit_tools() -> list[ToolkitTool]:
    skill_tools: list[dict] = []
    for skill in _skill_registry.skills:
        skill_tools.extend(skill.get("tools", []) or [])

    source_by_name, required_by_name = _tool_source_maps()
    tools: list[ToolkitTool] = []
    for tool in _tool_registry.build_tools(skill_tools=skill_tools):
        name = str(tool.get("name") or "")
        if not name:
            continue
        tools.append(
            ToolkitTool(
                name=name,
                description=str(tool.get("description") or ""),
                source=source_by_name.get(name, "peony"),
                required=required_by_name.get(name, []),
            )
        )
    tools.sort(key=lambda item: (item.source, item.name))
    return tools


async def _list_endpoint_capabilities(
    include_remote: bool,
) -> tuple[list[ToolkitEndpointCapability], list[str]]:
    capabilities: list[ToolkitEndpointCapability] = []
    warnings: list[str] = []

    for endpoint_id, raw_profile in sorted(_load_agent_endpoint_profiles().items()):
        profile = {**raw_profile, "id": endpoint_id}
        public_profile = _public_agent_profile(profile)
        item = ToolkitEndpointCapability(
            endpoint_id=endpoint_id,
            label=public_profile.label,
            kind=public_profile.kind,
            enabled=public_profile.enabled,
            auth_configured=public_profile.api_key_configured,
        )

        if not include_remote:
            item.warnings.append("Remote probing disabled for this request")
            capabilities.append(item)
            continue
        if not public_profile.enabled:
            item.warnings.append("Endpoint disabled")
            capabilities.append(item)
            continue
        if not public_profile.api_key_configured:
            item.warnings.append("API key not configured")
            capabilities.append(item)
            continue

        try:
            probe = await _probe_agent_endpoint_profile(profile)
            item.ok = bool(probe.get("ok"))
            summary = probe.get("summary") or {}
            item.models = int(summary.get("models") or 0)
            item.skills = int(summary.get("skills") or 0)
            item.toolsets = int(summary.get("toolsets") or 0)
            item.warnings = list(probe.get("warnings") or [])
            probes = probe.get("probes") or {}
            item.skill_names = _extract_named_endpoint_items(
                (probes.get("skills") or {}).get("data"),
                ("skills", "data"),
            )
            item.toolset_names = _extract_named_endpoint_items(
                (probes.get("toolsets") or {}).get("data"),
                ("toolsets", "data"),
            )
        except Exception as exc:
            message = _sanitize_agent_error(str(exc), endpoint_id)
            item.warnings.append(message)
            warnings.append(f"{endpoint_id}: {message}")
        capabilities.append(item)

    return capabilities, warnings


async def _build_toolkit_inventory(include_remote: bool) -> ToolkitInventoryResponse:
    root_infos, skills, scan_warnings = _scan_skill_roots()
    tools = _list_toolkit_tools()
    endpoint_capabilities, endpoint_warnings = await _list_endpoint_capabilities(include_remote)
    warnings = [*scan_warnings, *endpoint_warnings]

    return ToolkitInventoryResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        cached=False,
        cache_ttl_seconds=TOOLKIT_CACHE_TTL_SECONDS,
        counts={
            "skills": len(skills),
            "active_skills": sum(skill.catalog_status == "active" for skill in skills),
            "alternate_skills": sum(skill.catalog_status == "alternate" for skill in skills),
            "deprecated_skills": sum(skill.catalog_status == "deprecated" for skill in skills),
            "invalid_skills": sum(skill.catalog_status == "invalid" for skill in skills),
            "tools": len(tools),
            "skill_roots": len(root_infos),
            "endpoints": len(endpoint_capabilities),
            "endpoint_skills": sum(item.skills for item in endpoint_capabilities),
            "endpoint_toolsets": sum(item.toolsets for item in endpoint_capabilities),
        },
        skill_roots=root_infos,
        skills=skills,
        tools=tools,
        endpoint_capabilities=endpoint_capabilities,
        warnings=warnings,
    )


@router.get("/toolkit", response_model=ToolkitInventoryResponse)
async def get_toolkit_inventory(include_remote: bool = True, refresh: bool = False):
    """Return a cached capability inventory for local skills, tools, and agent endpoints."""
    cache_key = f"include_remote={include_remote}"
    now = datetime.now(timezone.utc)
    cached = _TOOLKIT_CACHE.get(cache_key)
    if (
        cached
        and not refresh
        and cached["expires_at"] > now
        and isinstance(cached.get("data"), ToolkitInventoryResponse)
    ):
        data = cached["data"].model_copy(deep=True)
        data.cached = True
        return data

    data = await _build_toolkit_inventory(include_remote)
    _TOOLKIT_CACHE[cache_key] = {
        "expires_at": now + timedelta(seconds=TOOLKIT_CACHE_TTL_SECONDS),
        "data": data,
    }
    return data


def _prepare_agent_endpoint_chat(
    endpoint_id: str,
    request: AgentEndpointChatRequest,
) -> dict[str, Any]:
    profile = _get_agent_endpoint_profile(endpoint_id)
    if not bool(profile.get("enabled", True)):
        raise HTTPException(status_code=400, detail="Agent endpoint is disabled")
    if not _get_agent_api_key(endpoint_id):
        raise HTTPException(status_code=400, detail="Agent endpoint API key is not configured")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    full_input = message
    if request.context and request.context.strip():
        full_input = (
            "<bashgym_authoritative_context>\n"
            f"{request.context.strip()}\n"
            "</bashgym_authoritative_context>\n\n"
            f"<user_request>\n{message}\n</user_request>"
        )

    model = str(profile.get("model") or "hermes-agent")
    session_key = request.session_key or profile.get("session_key")
    headers = _agent_headers(profile, session_key=session_key)
    instructions = (
        "You are Hermes connected to a BashGym workspace canvas. Use the supplied "
        "workspace context to help manage training runs, evals, datasets, models, "
        "and connected nodes. For current-state claims, precedence is live runtime, then "
        "the durable BashGym ledger, then the current workspace snapshot, then curated "
        "GBrain, then conversation memory. If history conflicts with supplied context, "
        "use the higher-authority evidence and explicitly report the conflict. Cite run, "
        "campaign, model, dataset, evaluation, and artifact IDs plus observation times "
        "when available. Never blend projects or experiments when their identities are "
        "missing. When terminal tools are available, use `bashgym training "
        "start` and `bashgym designer start` for executable work so the canvas can "
        "track it."
    )
    if request.enable_skill_lab_tools:
        instructions += (
            " You also have direct Skill Lab tools. When a user asks to build or "
            "evaluate a skill, inspect the workspace, prepare target and negative routing "
            "cases, and materialize Skill Lab through those tools. Never set confirmed=true "
            "for file changes or model-call evaluations until the user explicitly approves "
            "the preview. Be explicit about actions you can and cannot take."
        )
    else:
        instructions += " Do not call workspace tools for this response."
    conversation = request.conversation.strip() if request.conversation else None
    history = [
        {"role": item.role, "content": item.content}
        for item in request.history
        if item.role in {"user", "assistant"} and item.content.strip()
    ]
    responses_payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": full_input}],
            }
        ],
        "instructions": instructions,
        "user_message": message,
        "store": True,
    }
    if request.enable_skill_lab_tools:
        responses_payload["tools"] = _openai_skill_lab_tools()
    if conversation:
        responses_payload["conversation"] = conversation

    return {
        "profile": profile,
        "model": model,
        "headers": headers,
        "instructions": instructions,
        "full_input": full_input,
        "history": history,
        "workspace_id": request.workspace_id or "default",
        "origin": {
            "kind": "agent",
            "agent": endpoint_id,
            **request.origin,
        },
        "skill_lab_tools_enabled": request.enable_skill_lab_tools,
        "responses_payload": responses_payload,
    }


async def _execute_hermes_skill_lab_calls(
    prepared: dict[str, Any],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for call in calls:
        name = str(call.get("name") or "")
        arguments = call.get("arguments")
        if name not in SKILL_LAB_TOOL_NAMES:
            result: dict[str, Any] = {
                "error": "tool_not_allowed",
                "message": f"Hermes cannot call {name!r} through the BashGym bridge.",
            }
        else:
            scoped = dict(arguments) if isinstance(arguments, dict) else {}
            scoped.setdefault("workspace_id", prepared["workspace_id"])
            scoped.setdefault("origin", prepared["origin"])
            if (
                name in {"skill_lab_run", "skill_lab_save_skill"}
                and bool(scoped.get("confirmed"))
                and not _explicit_action_approval(prepared["user_message"])
            ):
                scoped["confirmed"] = False
            try:
                result = await execute_skill_lab_tool(
                    name,
                    scoped,
                    workspace_id=prepared["workspace_id"],
                    origin=prepared["origin"],
                )
            except SkillLabToolError as exc:
                result = exc.as_dict()
        outputs.append(
            {
                "call_id": str(call.get("id") or ""),
                "name": name,
                "output": json.dumps(result),
            }
        )
    return outputs


@router.post("/endpoints/{endpoint_id}/chat/stream")
async def stream_agent_endpoint_chat(
    endpoint_id: str,
    request: AgentEndpointChatRequest,
):
    """Stream a workspace-contextual Hermes response as normalized SSE."""
    if _skill_lab_intent(request.message):
        prepared_for_tools = _prepare_agent_endpoint_chat(endpoint_id, request)

        async def tool_aware_events() -> AsyncIterator[str]:
            yield _encode_sse(
                "meta",
                {"endpoint_id": endpoint_id, "model": prepared_for_tools["model"]},
            )
            yield _encode_sse("activity", {"label": "Skill Lab bridge"})
            try:
                result = await chat_with_agent_endpoint(endpoint_id, request)
            except HTTPException as exc:
                yield _encode_sse("error", {"error": str(exc.detail)})
                return
            yield _encode_sse("delta", {"delta": result.response})
            yield _encode_sse(
                "done",
                {
                    "endpoint_id": endpoint_id,
                    "model": result.model,
                    "response_id": result.response_id,
                },
            )

        return StreamingResponse(
            tool_aware_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    prepared = _prepare_agent_endpoint_chat(endpoint_id, request)
    profile = prepared["profile"]
    model = prepared["model"]
    headers = prepared["headers"]
    responses_url = _endpoint_url(profile, "responses")
    chat_url = _endpoint_url(profile, "chat/completions")

    async def stream_events() -> AsyncIterator[str]:
        yield _encode_sse("meta", {"endpoint_id": endpoint_id, "model": model})

        async def relay(response: httpx.Response, protocol: str) -> AsyncIterator[str]:
            text_parts: list[str] = []
            response_id: str | None = None
            async for event_type, payload in _iter_agent_stream_events(response, protocol):
                if event_type == "delta":
                    delta = payload.get("delta")
                    if isinstance(delta, str) and delta:
                        text_parts.append(delta)
                        yield _encode_sse("delta", {"delta": delta})
                elif event_type == "activity":
                    yield _encode_sse("activity", payload)
                elif event_type == "terminal":
                    candidate_id = payload.get("response_id")
                    if isinstance(candidate_id, str):
                        response_id = candidate_id
                    final_text = payload.get("final_text")
                    if isinstance(final_text, str) and final_text and not text_parts:
                        text_parts.append(final_text)
                        yield _encode_sse("delta", {"delta": final_text})
                elif event_type == "error":
                    detail = _sanitize_agent_error(
                        str(payload.get("error") or "Hermes stream failed"),
                        endpoint_id,
                    )
                    yield _encode_sse("error", {"error": detail})
                    return

            full_text = "".join(text_parts)
            if not full_text:
                yield _encode_sse("error", {"error": "Hermes returned no text"})
                return
            if _looks_like_agent_runtime_failure(full_text):
                detail = _sanitize_agent_error(full_text, endpoint_id)
                yield _encode_sse("error", {"error": detail})
                return
            yield _encode_sse(
                "done",
                {"endpoint_id": endpoint_id, "model": model, "response_id": response_id},
            )

        try:
            timeout = httpx.Timeout(120.0, connect=5.0, read=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                responses_payload = {**prepared["responses_payload"], "stream": True}
                async with client.stream(
                    "POST",
                    responses_url,
                    headers=headers,
                    json=responses_payload,
                ) as response:
                    fallback_to_chat = response.status_code in {404, 405}
                    if not fallback_to_chat:
                        if not response.is_success:
                            await response.aread()
                            detail = _agent_response_error(response, endpoint_id)
                            yield _encode_sse("error", {"error": detail})
                            return
                        async for event in relay(response, "responses"):
                            yield event
                        return

                chat_payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prepared["instructions"]},
                        *prepared["history"],
                        {"role": "user", "content": prepared["full_input"]},
                    ],
                    "stream": True,
                }
                async with client.stream(
                    "POST",
                    chat_url,
                    headers=headers,
                    json=chat_payload,
                ) as response:
                    if not response.is_success:
                        await response.aread()
                        detail = _agent_response_error(response, endpoint_id)
                        yield _encode_sse("error", {"error": detail})
                        return
                    async for event in relay(response, "chat.completions"):
                        yield event
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Hermes chat stream failed for endpoint %s", endpoint_id)
            detail = _sanitize_agent_error(
                str(exc).strip() or exc.__class__.__name__,
                endpoint_id,
            )
            yield _encode_sse("error", {"error": detail})

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/endpoints/{endpoint_id}/chat",
    response_model=AgentEndpointChatResponse,
)
async def chat_with_agent_endpoint(
    endpoint_id: str,
    request: AgentEndpointChatRequest,
):
    """Send a workspace-contextual message to a Hermes-compatible endpoint."""
    prepared = _prepare_agent_endpoint_chat(endpoint_id, request)
    profile = prepared["profile"]
    model = prepared["model"]
    headers = prepared["headers"]
    instructions = prepared["instructions"]
    full_input = prepared["full_input"]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        responses_payload = dict(prepared["responses_payload"])
        responses_url = _endpoint_url(profile, "responses")
        responses_runtime_failure: str | None = None
        fallback_to_chat = False
        responses_tools_disabled = False

        for _step in range(4):
            try:
                response = await client.post(
                    responses_url,
                    headers=headers,
                    json=responses_payload,
                )
            except Exception as exc:
                message = _sanitize_agent_error(str(exc), endpoint_id)
                raise HTTPException(status_code=502, detail=message)

            data = _json_or_text(response)
            if response.status_code in {404, 405}:
                fallback_to_chat = True
                break
            if (
                response.status_code in {400, 422}
                and "tools" in responses_payload
                and not responses_tools_disabled
            ):
                responses_payload.pop("tools", None)
                responses_tools_disabled = True
                continue
            if not response.is_success:
                detail = _agent_response_error(response, endpoint_id, data)
                raise HTTPException(status_code=502, detail=detail)
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="Agent endpoint returned non-JSON")

            tool_calls = _extract_responses_tool_calls(data)
            if tool_calls:
                outputs = await _execute_hermes_skill_lab_calls(prepared, tool_calls)
                response_id = data.get("id")
                if not isinstance(response_id, str):
                    raise HTTPException(status_code=502, detail="Agent tool response has no id")
                responses_payload = {
                    "model": model,
                    "previous_response_id": response_id,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": output["call_id"],
                            "output": output["output"],
                        }
                        for output in outputs
                    ],
                    "instructions": instructions,
                    "store": True,
                    "tools": _openai_skill_lab_tools(),
                }
                continue

            text = _extract_responses_text(data)
            if not text:
                raise HTTPException(status_code=502, detail="Agent endpoint returned no text")
            if _looks_like_agent_runtime_failure(text):
                responses_runtime_failure = _sanitize_agent_error(text, endpoint_id)
                fallback_to_chat = True
                break

            return AgentEndpointChatResponse(
                response=text,
                endpoint_id=endpoint_id,
                model=model,
                response_id=data.get("id") if isinstance(data.get("id"), str) else None,
                raw_status=data.get("status") if isinstance(data.get("status"), str) else None,
            )

        if not fallback_to_chat:
            raise HTTPException(
                status_code=502, detail="Agent exceeded the Skill Lab tool-call limit"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            *prepared["history"],
            {"role": "user", "content": full_input},
        ]
        chat_url = _endpoint_url(profile, "chat/completions")
        chat_tools_disabled = False
        for _step in range(4):
            chat_payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            if prepared["skill_lab_tools_enabled"] and not chat_tools_disabled:
                chat_payload["tools"] = _chat_completion_skill_lab_tools()
            try:
                chat_response = await client.post(chat_url, headers=headers, json=chat_payload)
            except Exception as exc:
                message = _sanitize_agent_error(str(exc), endpoint_id)
                raise HTTPException(status_code=502, detail=message)

            chat_data = _json_or_text(chat_response)
            if (
                chat_response.status_code in {400, 422}
                and prepared["skill_lab_tools_enabled"]
                and not chat_tools_disabled
            ):
                chat_tools_disabled = True
                continue
            if not chat_response.is_success:
                detail = _agent_response_error(chat_response, endpoint_id, chat_data)
                if responses_runtime_failure:
                    detail = f"{responses_runtime_failure}; chat fallback failed: {detail}"
                raise HTTPException(status_code=502, detail=detail)
            if not isinstance(chat_data, dict):
                raise HTTPException(status_code=502, detail="Agent endpoint returned non-JSON")

            tool_calls = _extract_chat_tool_calls(chat_data)
            if tool_calls:
                outputs = await _execute_hermes_skill_lab_calls(prepared, tool_calls)
                choices = chat_data.get("choices") or []
                assistant_message = choices[0].get("message") if choices else None
                if not isinstance(assistant_message, dict):
                    raise HTTPException(status_code=502, detail="Agent tool response is malformed")
                messages.append(assistant_message)
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": output["call_id"],
                        "content": output["output"],
                    }
                    for output in outputs
                )
                continue

            text = _extract_chat_completion_text(chat_data)
            if not text:
                detail = "Agent endpoint returned no text"
                if responses_runtime_failure:
                    detail = f"{responses_runtime_failure}; chat fallback returned no text"
                raise HTTPException(status_code=502, detail=detail)
            if _looks_like_agent_runtime_failure(text):
                detail = _sanitize_agent_error(text, endpoint_id)
                raise HTTPException(status_code=502, detail=detail)
            return AgentEndpointChatResponse(
                response=text,
                endpoint_id=endpoint_id,
                model=model,
                response_id=chat_data.get("id") if isinstance(chat_data.get("id"), str) else None,
                raw_status="chat.completions",
            )

        raise HTTPException(status_code=502, detail="Agent exceeded the Skill Lab tool-call limit")


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@router.post("/chat")
async def chat(request: ChatRequest):
    """Send a message to Peony, the botanical assistant."""
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503, detail="ANTHROPIC_API_KEY not configured. Peony requires an API key."
        )

    # Load memory
    memory_prompt = _memory.build_memory_prompt()

    # Match skills to user message
    matched_skills = _skill_registry.match(request.message)
    skill_tools = _skill_registry.get_tools(matched_skills)
    skill_knowledge = _skill_registry.get_knowledge(matched_skills)

    # Build dynamic tool list
    tools = _tool_registry.build_tools(skill_tools=skill_tools)

    # Gather live system context
    context = await _gather_system_context()
    context_used = [
        s.split(":**")[0].replace("**", "") for s in context.split("\n\n") if ":**" in s
    ]

    # Build system prompt with all sections
    system_prompt = _build_system_prompt(context, memory_prompt, skill_knowledge)

    # Build message list
    messages = []
    if request.history:
        for msg in request.history:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})
    else:
        messages.append({"role": "user", "content": request.message})

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # First Claude call with tools
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "tools": tools,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text[:200]}")
            raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.status_code}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="LLM request timed out")
        except Exception as e:
            logger.error(f"Peony chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    stop_reason = data.get("stop_reason")

    # --- Tool use path ---
    if stop_reason == "tool_use":
        tool_use_blocks = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
        tool_results = []
        pending_shell: dict | None = None

        for block in tool_use_blocks:
            tool_name = block["name"]
            tool_id = block["id"]
            tool_input = block.get("input", {})

            if tool_name == "run_shell_command":
                # Gate: require user confirmation
                token = secrets.token_urlsafe(16)
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
                PENDING_ACTIONS[token] = {
                    "command": tool_input["command"],
                    "reason": tool_input.get("reason", ""),
                    "messages": messages + [{"role": "assistant", "content": data["content"]}],
                    "tool_use_id": tool_id,
                    "expires_at": expires_at,
                    "system_prompt": system_prompt,
                    "context_used": context_used,
                    "headers": headers,
                    "tools": tools,
                }
                pending_shell = {
                    "type": "shell_command",
                    "command": tool_input["command"],
                    "reason": tool_input.get("reason", ""),
                    "token": token,
                }
                # Return immediately for shell commands — user must confirm
                # Emit a brief text response so the UI shows something
                return ChatResponse(
                    response="",
                    context_used=context_used,
                    pending_action=PendingAction(**pending_shell),
                )
            else:
                # Execute structured tools immediately
                result_str = await _execute_tool(tool_name, tool_input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    }
                )

        if not tool_results:
            # Only shell command was requested; already returned above
            return ChatResponse(response="", context_used=context_used)

        # Second Claude call with tool results
        follow_up_messages = messages + [
            {"role": "assistant", "content": data["content"]},
            {"role": "user", "content": tool_results},
        ]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client2:
                resp2 = await client2.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": "claude-sonnet-4-5-20250929",
                        "max_tokens": 1024,
                        "system": system_prompt,
                        "tools": tools,
                        "messages": follow_up_messages,
                    },
                )
                resp2.raise_for_status()
                data2 = resp2.json()
        except Exception as e:
            logger.error(f"Peony second call error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        response_text = ""
        for block in data2.get("content", []):
            if block.get("type") == "text":
                response_text += block["text"]

        return ChatResponse(
            response=response_text or "Done.",
            context_used=context_used,
        )

    # --- Normal end_turn path ---
    response_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            response_text += block["text"]

    if not response_text:
        response_text = "I couldn't generate a response. Please try again."

    return ChatResponse(response=response_text, context_used=context_used)


# ---------------------------------------------------------------------------
# Confirm action endpoint
# ---------------------------------------------------------------------------


@router.post("/confirm-action")
async def confirm_action(request: ConfirmActionRequest):
    """Approve or deny a pending shell command and resume the Claude conversation."""
    import httpx

    token = request.token
    action = PENDING_ACTIONS.pop(token, None)
    if not action:
        raise HTTPException(status_code=404, detail="Pending action not found or expired")

    # Check expiry
    if datetime.now(timezone.utc) > action["expires_at"]:
        raise HTTPException(status_code=410, detail="Pending action expired")

    tool_use_id = action["tool_use_id"]
    messages = action["messages"]
    system_prompt = action["system_prompt"]
    context_used = action["context_used"]
    action["headers"]
    action_tools = action.get("tools", _tool_registry.build_tools())

    if request.approved:
        # Run the command
        try:
            proc = subprocess.run(
                action["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = proc.stdout or ""
            if proc.stderr:
                output += f"\n[stderr]: {proc.stderr}"
            tool_result_content = output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            tool_result_content = "[Error]: Command timed out after 30 seconds"
        except Exception as e:
            tool_result_content = f"[Error]: {e}"
    else:
        tool_result_content = "User declined to run this command."

    # Resume Claude conversation with tool result
    follow_up_messages = messages + [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result_content,
                }
            ],
        }
    ]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "tools": action_tools,
                    "messages": follow_up_messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            response_text += block["text"]

    return ChatResponse(
        response=response_text or "Done.",
        context_used=context_used,
    )


# ---------------------------------------------------------------------------
# Session CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("/sessions")
async def list_sessions():
    """List all Peony chat sessions."""
    return _read_sessions_index()


@router.get("/sessions/{session_id}")
async def load_session(session_id: str):
    """Load messages for a specific session from its JSONL log."""
    session_id = _validate_session_id(session_id)
    log_path = _get_peony_logs_dir() / f"{session_id}.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    messages = []
    try:
        for line in log_path.read_text(encoding="utf-8").strip().splitlines():
            record = json.loads(line)
            if record.get("type") == "message":
                messages.append(
                    SessionMessage(
                        id=record["id"],
                        role=record["role"],
                        content=record["content"],
                        timestamp=record["timestamp"],
                        context_used=record.get("context_used", []),
                    )
                )
    except Exception as e:
        logger.error(f"Error reading session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read session")

    return messages


@router.post("/sessions")
async def save_session(req: SaveSessionRequest):
    """Save a session — writes JSONL file and updates the index."""
    _validate_session_id(req.session_id)
    logs_dir = _get_peony_logs_dir()
    log_path = logs_dir / f"{req.session_id}.jsonl"

    # Write JSONL
    now = datetime.now(timezone.utc).isoformat()

    lines = []
    # Meta line
    lines.append(
        json.dumps(
            {
                "type": "meta",
                "session_id": req.session_id,
                "name": req.name,
                "created_at": now,
                "updated_at": now,
            }
        )
    )
    # Message lines
    for msg in req.messages:
        lines.append(
            json.dumps(
                {
                    "type": "message",
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "context_used": msg.context_used,
                }
            )
        )

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update index
    index = _read_sessions_index()
    # Remove existing entry for this session
    index = [s for s in index if s.get("session_id") != req.session_id]
    index.append(
        {
            "session_id": req.session_id,
            "name": req.name,
            "created_at": now,
            "updated_at": now,
            "message_count": len(req.messages),
        }
    )
    _write_sessions_index(index)

    return {"status": "ok", "session_id": req.session_id}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session log and remove it from the index."""
    session_id = _validate_session_id(session_id)
    log_path = _get_peony_logs_dir() / f"{session_id}.jsonl"
    if log_path.exists():
        log_path.unlink()

    index = _read_sessions_index()
    index = [s for s in index if s.get("session_id") != session_id]
    _write_sessions_index(index)

    return {"status": "ok", "session_id": session_id}


# ---------------------------------------------------------------------------
# Session summarization
# ---------------------------------------------------------------------------


@router.post("/summarize-session/{session_id}")
async def summarize_session(session_id: str):
    """Generate and save an episode summary for a completed session."""
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")

    # Load session messages
    log_path = _get_peony_logs_dir() / f"{session_id}.jsonl"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    messages_text = []
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "message":
            messages_text.append(f"{record['role']}: {record['content']}")

    if not messages_text:
        raise HTTPException(status_code=400, detail="Session has no messages")

    transcript = "\n".join(messages_text[-20:])  # Last 20 messages max

    # Generate summary via Claude Haiku (fast, cheap)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Summarize this conversation in 2-3 sentences. "
                                "Focus on what the user wanted, what was accomplished, "
                                "and any decisions made.\n\n" + transcript
                            ),
                        }
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"LLM API error: {e.response.status_code}",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    summary_text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            summary_text += block["text"]

    if not summary_text:
        raise HTTPException(status_code=500, detail="Failed to generate summary")

    episode = _memory.save_episode(session_id, summary_text)
    return {"status": "ok", "episode": episode}
