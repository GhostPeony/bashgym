"""
Trace Schema Validation Module

Defines a formal schema for trace events using the AgentTrace envelope pattern.
All trace records are validated at emission/import time against this schema,
ensuring data quality early in the pipeline rather than normalizing later.

Three surface types (from AgentTrace):
- operational: Tool call executions (args, returns, duration, errors)
- cognitive: LLM reasoning (thinking, plans, reflections, confidence)
- contextual: External I/O (file ops, git operations, HTTP calls)

Module: Trace Capture - Schema Validation
"""

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SurfaceType(str, Enum):
    """The three telemetry surfaces from AgentTrace."""
    OPERATIONAL = "operational"  # Tool calls, method executions
    COGNITIVE = "cognitive"      # Thinking, planning, reflection
    CONTEXTUAL = "contextual"    # File I/O, git, HTTP, DB


class CognitiveSchema(BaseModel):
    """Structured cognitive data from LLM reasoning."""
    thinking: Optional[str] = None
    plan: Optional[str] = None
    reflection: Optional[str] = None
    decision_rationale: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class OperationalBody(BaseModel):
    """Body schema for operational (tool call) events."""
    tool_name: str
    command: str  # Normalized to string (handles JSON-stringified objects)
    output: str = ""
    exit_code: Optional[int] = None
    success: Optional[bool] = None
    cwd: str = ""
    duration_ms: Optional[float] = None

    @field_validator("command", mode="before")
    @classmethod
    def normalize_command(cls, v: Any) -> str:
        """Normalize command field: JSON objects become stringified, strings pass through."""
        if isinstance(v, dict):
            return json.dumps(v)
        if isinstance(v, str):
            # Check if it's already a JSON-stringified object — leave as-is
            return v
        return str(v) if v is not None else ""


class CognitiveBody(BaseModel):
    """Body schema for cognitive (reasoning) events."""
    cognitive: CognitiveSchema
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class ContextualBody(BaseModel):
    """Body schema for contextual (external I/O) events."""
    operation_type: str  # "file_read", "file_write", "git", "http", etc.
    target: str = ""     # File path, URL, etc.
    details: Dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """Unified envelope schema for all trace events.

    Every event has a UUID, surface type, trace/span IDs for causal linking,
    a UTC timestamp, and a structured body. This mirrors AgentTrace's
    L(S:E:C) → R schema with consistency, causality, and fidelity guarantees.
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    surface_type: SurfaceType
    trace_id: str         # Groups related events in a session
    span_id: str          # Groups causally linked events (reasoning → action)
    parent_span_id: Optional[str] = None  # Hierarchical nesting
    timestamp: str        # UTC ISO-8601
    source_tool: str = "unknown"  # claude_code, opencode, aider, etc.
    body: Dict[str, Any]  # Validated per surface_type

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_iso(cls, v: Any) -> str:
        """Ensure timestamp is a valid ISO-8601 string."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                v = v.replace(tzinfo=timezone.utc)
            return v.isoformat()
        if isinstance(v, str):
            # Validate it parses
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"Invalid ISO-8601 timestamp: {v}")
            return v
        raise ValueError(f"Expected datetime or ISO string, got {type(v)}")


def validate_operational_event(event: TraceEvent) -> OperationalBody:
    """Validate the body of an operational event."""
    return OperationalBody(**event.body)


def validate_cognitive_event(event: TraceEvent) -> CognitiveBody:
    """Validate the body of a cognitive event."""
    return CognitiveBody(**event.body)


def validate_contextual_event(event: TraceEvent) -> ContextualBody:
    """Validate the body of a contextual event."""
    return ContextualBody(**event.body)


_BODY_VALIDATORS = {
    SurfaceType.OPERATIONAL: validate_operational_event,
    SurfaceType.COGNITIVE: validate_cognitive_event,
    SurfaceType.CONTEXTUAL: validate_contextual_event,
}


def validate_event(event: TraceEvent) -> TraceEvent:
    """Validate a TraceEvent including its surface-specific body.

    Raises ValidationError if the body doesn't match the surface type schema.
    """
    validator = _BODY_VALIDATORS.get(event.surface_type)
    if validator:
        validator(event)  # Raises on invalid body
    return event


def trace_step_to_events(
    step_dict: Dict[str, Any],
    trace_id: str,
    span_id: Optional[str] = None,
) -> List[TraceEvent]:
    """Convert a legacy TraceStep dict into validated TraceEvent(s).

    Produces an operational event for the tool call, and optionally a
    cognitive event if thinking/reasoning data is present.

    Args:
        step_dict: A serialized TraceStep dictionary.
        trace_id: The session-level trace ID.
        span_id: Optional span ID (generated if not provided).

    Returns:
        List of validated TraceEvent objects (1 operational + 0-1 cognitive).
    """
    span = span_id or uuid.uuid4().hex[:12]
    timestamp = step_dict.get("timestamp", datetime.now(timezone.utc).isoformat())
    source_tool = step_dict.get("source_tool", "unknown")
    meta = step_dict.get("metadata", {})
    events = []

    # Operational event (always)
    op_body = {
        "tool_name": step_dict.get("tool_name", "unknown"),
        "command": step_dict.get("command", ""),
        "output": step_dict.get("output", ""),
        "exit_code": step_dict.get("exit_code"),
        "success": step_dict.get("success"),
        "cwd": step_dict.get("cwd", ""),
    }
    op_event = TraceEvent(
        surface_type=SurfaceType.OPERATIONAL,
        trace_id=trace_id,
        span_id=span,
        timestamp=timestamp,
        source_tool=source_tool,
        body=op_body,
    )
    validate_event(op_event)
    events.append(op_event)

    # Cognitive event (if reasoning data present)
    cognitive_data = step_dict.get("cognitive") or meta.get("cognitive")
    if not cognitive_data:
        # Fall back to legacy fields
        thinking = meta.get("thinking_content")
        assistant_text = meta.get("assistant_text")
        if thinking or assistant_text:
            cognitive_data = {
                "thinking": thinking,
                "decision_rationale": assistant_text,
            }

    if cognitive_data:
        cog_body = {
            "cognitive": cognitive_data,
            "model": meta.get("model"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
        }
        cog_event = TraceEvent(
            surface_type=SurfaceType.COGNITIVE,
            trace_id=trace_id,
            span_id=span,
            parent_span_id=span,  # Cognitive nests inside operational
            timestamp=timestamp,
            source_tool=source_tool,
            body=cog_body,
        )
        validate_event(cog_event)
        events.append(cog_event)

    return events


def validate_session(session_dict: Dict[str, Any]) -> List[str]:
    """Validate a complete trace session dict and return a list of errors.

    Non-destructive: does not modify the input. Returns empty list if valid.
    """
    errors = []

    if not isinstance(session_dict, dict):
        return ["Session must be a dictionary"]

    # Check required top-level fields
    for field in ["session_id", "trace"]:
        if field not in session_dict:
            errors.append(f"Missing required field: {field}")

    trace = session_dict.get("trace", [])
    if not isinstance(trace, list):
        errors.append("'trace' must be a list")
        return errors

    trace_id = session_dict.get("session_id", "unknown")

    for i, step in enumerate(trace):
        if not isinstance(step, dict):
            errors.append(f"Step {i}: must be a dictionary")
            continue

        # Validate command field consistency
        command = step.get("command", "")
        if isinstance(command, dict):
            errors.append(
                f"Step {i}: 'command' is a dict (should be string). "
                "Use json.dumps() at capture time."
            )

        # Validate required fields
        if not step.get("tool_name"):
            errors.append(f"Step {i}: missing 'tool_name'")

        # Validate cognitive data structure if present
        cognitive = step.get("cognitive") or step.get("metadata", {}).get("cognitive")
        if cognitive and not isinstance(cognitive, dict):
            errors.append(f"Step {i}: 'cognitive' must be a dictionary")

    return errors
