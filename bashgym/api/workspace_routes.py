"""Workspace canvas context and intent routes.

These desktop-only endpoints give local agents a sanctioned way to read the
current canvas and emit semantic intents. The renderer owns layout; the backend
keeps a live sanitized snapshot and broadcasts intents to connected canvases.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from bashgym.api.websocket import (
    broadcast_workspace_canvas_intent,
    broadcast_workspace_context_updated,
)

router = APIRouter(prefix="/api/workspace", tags=["workspace"])

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)


def _now() -> str:
    return datetime.utcnow().isoformat()


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SECRET_KEY_PARTS)


def redact_workspace_value(value: Any) -> Any:
    """Return a recursively redacted JSON-compatible value."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(str(key)):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = redact_workspace_value(item)
        return redacted
    if isinstance(value, list):
        return [redact_workspace_value(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if value.startswith(("sk-", "ghp_", "hf_", "xoxb-", "xoxp-")):
            return "[redacted]"
        if "bearer " in lowered:
            return "[redacted]"
    return value


class WorkspacePanel(BaseModel):
    panel_id: str
    type: str
    title: str
    terminal_id: str | None = None
    adapter_config: dict[str, Any] | None = None
    visible: bool = True


class WorkspaceEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str | None = None
    data: dict[str, Any] | None = None


class WorkspaceTerminal(BaseModel):
    terminal_id: str
    panel_id: str | None = None
    title: str | None = None
    cwd: str | None = None
    agent_kind: str | None = None
    status: str | None = None
    current_tool: str | None = None


class WorkspaceCanvasSnapshot(BaseModel):
    schema_version: str = "bashgym.workspace.canvas.v1"
    updated_at: str = Field(default_factory=_now)
    panels: list[WorkspacePanel] = Field(default_factory=list)
    edges: list[WorkspaceEdge] = Field(default_factory=list)
    terminals: list[WorkspaceTerminal] = Field(default_factory=list)
    data_summaries: dict[str, Any] = Field(default_factory=dict)
    allowed_actions: list[str] = Field(
        default_factory=lambda: [
            "workspace.context.read",
            "workspace.canvas.intent.emit",
            "training.start",
        ]
    )


class WorkspaceEventSource(BaseModel):
    kind: str = "agent"
    terminal_id: str | None = None
    panel_id: str | None = None
    agent: str | None = None


class WorkspaceSuggestedNode(BaseModel):
    recipe: str
    title: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class WorkspaceRelationship(BaseModel):
    source: str
    target: str
    type: str


class WorkspaceEvent(BaseModel):
    type: str
    source: WorkspaceEventSource = Field(default_factory=WorkspaceEventSource)
    correlation_id: str | None = None
    title: str | None = None
    summary: str | None = None
    entity: dict[str, Any] = Field(default_factory=dict)
    suggested_nodes: list[WorkspaceSuggestedNode] = Field(default_factory=list)
    relationships: list[WorkspaceRelationship] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


def _state_snapshot(request: Request) -> WorkspaceCanvasSnapshot:
    snapshot = getattr(request.app.state, "workspace_canvas_snapshot", None)
    if isinstance(snapshot, WorkspaceCanvasSnapshot):
        return snapshot
    if isinstance(snapshot, dict):
        return WorkspaceCanvasSnapshot(**snapshot)
    return WorkspaceCanvasSnapshot()


def _state_events(request: Request) -> list[dict[str, Any]]:
    events = getattr(request.app.state, "workspace_events", None)
    if not isinstance(events, list):
        events = []
        request.app.state.workspace_events = events
    return events


def _training_runs(request: Request) -> list[dict[str, Any]]:
    runs = getattr(request.app.state, "training_runs", {}) or {}
    result: list[dict[str, Any]] = []
    for run in runs.values():
        if not isinstance(run, dict):
            continue
        result.append(
            redact_workspace_value(
                jsonable_encoder(
                    {
                        "run_id": run.get("run_id"),
                        "status": run.get("status"),
                        "strategy": run.get("strategy"),
                        "started_at": run.get("started_at"),
                        "completed_at": run.get("completed_at"),
                        "origin": run.get("origin"),
                        "correlation_id": run.get("correlation_id"),
                        "compute_target": run.get("compute_target"),
                        "config": run.get("config", {}),
                        "metrics": run.get("metrics"),
                        "error": run.get("error"),
                    }
                )
            )
        )
    return result


def _context_payload(request: Request) -> dict[str, Any]:
    snapshot = _state_snapshot(request)
    events = _state_events(request)
    return {
        "schema_version": "bashgym.workspace.context.v1",
        "generated_at": _now(),
        "canvas": redact_workspace_value(snapshot.dict()),
        "training_runs": _training_runs(request),
        "recent_events": events[-20:],
        "allowed_actions": snapshot.allowed_actions,
    }


def _as_markdown(payload: dict[str, Any]) -> str:
    canvas = payload.get("canvas", {})
    panels = canvas.get("panels", [])
    edges = canvas.get("edges", [])
    terminals = canvas.get("terminals", [])
    runs = payload.get("training_runs", [])
    events = payload.get("recent_events", [])

    lines = [
        "# BashGym Workspace Context",
        "",
        f"Generated: {payload.get('generated_at')}",
        "",
        "## Canvas",
        f"- panels: {len(panels)}",
        f"- edges: {len(edges)}",
        f"- terminals: {len(terminals)}",
    ]
    if panels:
        lines.extend(["", "### Panels"])
        for panel in panels:
            bits = [panel.get("type", "unknown"), panel.get("panel_id", "?")]
            if panel.get("terminal_id"):
                bits.append(f"terminal {panel['terminal_id']}")
            lines.append(f"- {panel.get('title') or panel.get('panel_id')}: {' | '.join(bits)}")
    if runs:
        lines.extend(["", "## Training Runs"])
        for run in runs:
            bits = [
                str(run.get("status") or "unknown"),
                str(run.get("strategy") or "unknown"),
            ]
            if run.get("compute_target"):
                bits.append(str(run["compute_target"]))
            lines.append(f"- {run.get('run_id')}: {' | '.join(bits)}")
    if events:
        lines.extend(["", "## Recent Workspace Events"])
        for event in events[-8:]:
            lines.append(
                f"- {event.get('type')}: {event.get('title') or event.get('summary') or event.get('event_id')}"
            )
    lines.extend(["", "## Allowed Actions"])
    for action in payload.get("allowed_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines)


@router.get("/context")
async def get_workspace_context(
    request: Request,
    format: str = Query("json", pattern="^(json|markdown)$"),
):
    """Return the latest sanitized workspace context for local agents."""
    payload = _context_payload(request)
    if format == "markdown":
        return PlainTextResponse(_as_markdown(payload), media_type="text/markdown")
    return payload


@router.put("/canvas/snapshot")
async def put_workspace_canvas_snapshot(request: Request, snapshot: WorkspaceCanvasSnapshot):
    """Store the renderer-owned sanitized canvas snapshot."""
    safe_snapshot = WorkspaceCanvasSnapshot(**redact_workspace_value(snapshot.dict()))
    safe_snapshot.updated_at = _now()
    request.app.state.workspace_canvas_snapshot = safe_snapshot
    await broadcast_workspace_context_updated(
        {
            "updated_at": safe_snapshot.updated_at,
            "panels": len(safe_snapshot.panels),
            "edges": len(safe_snapshot.edges),
            "terminals": len(safe_snapshot.terminals),
        }
    )
    return {"ok": True, "updated_at": safe_snapshot.updated_at}


@router.post("/events")
async def post_workspace_event(request: Request, event: WorkspaceEvent):
    """Accept a semantic workspace/canvas intent from an agent or local tool."""
    payload = redact_workspace_value(event.dict())
    payload["event_id"] = f"workspace_evt_{uuid4().hex[:12]}"
    payload["received_at"] = _now()
    if not payload.get("correlation_id"):
        payload["correlation_id"] = f"intent_{uuid4().hex[:12]}"

    events = _state_events(request)
    events.append(payload)
    del events[:-100]

    await broadcast_workspace_canvas_intent(payload)
    return {"ok": True, "event": payload}
