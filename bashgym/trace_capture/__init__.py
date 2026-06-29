"""
Multi-Tool Trace Capture System

Supports automatic trace capture from multiple AI coding assistants:
- Claude Code (via hooks)
- OpenCode (via plugins)
- Generic (via shell wrapper - future)

Usage:
    from bashgym.trace_capture import setup_trace_capture, detect_tools

    # Auto-detect and setup
    tools = detect_tools()
    setup_trace_capture(tools)
"""

from .core import CognitiveData, TraceCapture, TraceSession, TraceStep
from .detector import detect_tools, get_tool_status
from .schema import (
    CognitiveBody,
    CognitiveSchema,
    ContextualBody,
    OperationalBody,
    SurfaceType,
    TraceEvent,
    trace_step_to_events,
    validate_event,
    validate_session,
)
from .setup import setup_trace_capture, uninstall_trace_capture
from .status_protocol import (
    AGENT_STATUS_MARKER,
    AGENT_STATUS_SCHEMA_VERSION,
    ALLOWED_AGENT_STATUSES,
    REPLAY_SCRUB_SCHEMA_VERSION,
    agent_status_to_event,
    format_agent_status_marker,
    normalize_agent_status,
    parse_agent_status_markers,
    scrub_trace_replay,
    scrub_trace_replay_file,
)

__all__ = [
    "TraceCapture",
    "TraceStep",
    "TraceSession",
    "CognitiveData",
    # Schema validation (AgentTrace-inspired)
    "TraceEvent",
    "SurfaceType",
    "CognitiveSchema",
    "OperationalBody",
    "CognitiveBody",
    "ContextualBody",
    "validate_event",
    "validate_session",
    "trace_step_to_events",
    "AGENT_STATUS_MARKER",
    "AGENT_STATUS_SCHEMA_VERSION",
    "ALLOWED_AGENT_STATUSES",
    "REPLAY_SCRUB_SCHEMA_VERSION",
    "normalize_agent_status",
    "format_agent_status_marker",
    "parse_agent_status_markers",
    "agent_status_to_event",
    "scrub_trace_replay",
    "scrub_trace_replay_file",
    # Detection & setup
    "detect_tools",
    "get_tool_status",
    "setup_trace_capture",
    "uninstall_trace_capture",
]
