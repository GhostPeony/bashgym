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
    # Detection & setup
    "detect_tools",
    "get_tool_status",
    "setup_trace_capture",
    "uninstall_trace_capture",
]
