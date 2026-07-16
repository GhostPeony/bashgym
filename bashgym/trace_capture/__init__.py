"""Multi-tool trace-capture APIs with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "bashgym.trace_capture.core": (
        "TraceCapture",
        "TraceStep",
        "TraceSession",
        "CognitiveData",
    ),
    "bashgym.trace_capture.schema": (
        "TraceEvent",
        "SurfaceType",
        "CognitiveSchema",
        "OperationalBody",
        "CognitiveBody",
        "ContextualBody",
        "validate_event",
        "validate_session",
        "trace_step_to_events",
    ),
    "bashgym.trace_capture.status_protocol": (
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
    ),
    "bashgym.trace_capture.detector": ("detect_tools", "get_tool_status"),
    "bashgym.trace_capture.setup": ("setup_trace_capture", "uninstall_trace_capture"),
}
_EXPORTS = {name: module_name for module_name, names in _MODULE_EXPORTS.items() for name in names}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
