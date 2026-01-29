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

from .core import TraceCapture, TraceStep, TraceSession
from .detector import detect_tools, get_tool_status
from .setup import setup_trace_capture, uninstall_trace_capture

__all__ = [
    'TraceCapture',
    'TraceStep',
    'TraceSession',
    'detect_tools',
    'get_tool_status',
    'setup_trace_capture',
    'uninstall_trace_capture',
]
