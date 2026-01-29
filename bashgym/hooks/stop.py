#!/usr/bin/env python3
"""
Stop Hook for Claude Code Instrumentation

Captures session interruption/stop events:
- User-initiated stops (Ctrl+C, escape)
- Error-induced stops
- Task completion status at time of stop
- Partial progress information

This helps understand why sessions ended and capture partial solutions.
"""

import os
import sys
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Cross-platform file locking
if platform.system() == 'Windows':
    import msvcrt

    def lock_file(f, exclusive=False):
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)

    def unlock_file(f):
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def lock_file(f, exclusive=False):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def unlock_file(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_bashgym_dir() -> Path:
    """Get the global Bash Gym directory (~/.bashgym/)."""
    if platform.system() == 'Windows':
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym"


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    session_file = get_bashgym_dir() / "current_session_id"
    if session_file.exists():
        return session_file.read_text().strip()
    return None


def get_traces_dir() -> Path:
    """Get the traces directory."""
    return get_bashgym_dir() / "traces"


def update_session_metadata(stop_reason: str, stop_details: Dict[str, Any]) -> None:
    """Update session metadata with stop information."""
    session_id = get_session_id()
    if not session_id:
        return

    traces_dir = get_traces_dir()
    metadata_file = traces_dir / f"session_{session_id}_metadata.json"

    if not metadata_file.exists():
        return

    try:
        with open(metadata_file, 'r') as f:
            lock_file(f, exclusive=False)
            metadata = json.load(f)
            unlock_file(f)

        metadata["stop_reason"] = stop_reason
        metadata["stop_details"] = stop_details
        metadata["stopped_at"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "stopped"

        with open(metadata_file, 'w') as f:
            lock_file(f, exclusive=True)
            json.dump(metadata, f, indent=2)
            unlock_file(f)

    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not update session metadata: {e}", file=sys.stderr)


def append_to_trace(stop_event: Dict[str, Any]) -> None:
    """Append stop event to the session trace."""
    session_id = get_session_id()
    if not session_id:
        return

    traces_dir = get_traces_dir()
    trace_file = traces_dir / f"session_{session_id}.json"

    if not trace_file.exists():
        return

    try:
        with open(trace_file, 'r') as f:
            lock_file(f, exclusive=False)
            traces = json.load(f)
            unlock_file(f)

        traces.append(stop_event)

        with open(trace_file, 'w') as f:
            lock_file(f, exclusive=True)
            json.dump(traces, f, indent=2)
            unlock_file(f)

    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not append to trace: {e}", file=sys.stderr)


def process_stop_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process the stop event and extract relevant information."""

    # Determine stop reason
    stop_reason = event.get("reason", "unknown")
    if stop_reason == "unknown":
        # Try to infer from event data
        if event.get("user_initiated"):
            stop_reason = "user_cancelled"
        elif event.get("error"):
            stop_reason = "error"
        elif event.get("timeout"):
            stop_reason = "timeout"

    stop_event = {
        "event_type": "stop",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": get_session_id(),
        "stop_reason": stop_reason,
        "user_initiated": event.get("user_initiated", False),
        "error_message": event.get("error", event.get("message", None)),
        "partial_output": event.get("partial_output", event.get("output", ""))[:5000],
        "tools_used_count": event.get("tools_used", 0),
        "last_tool": event.get("last_tool", None),
    }

    return stop_event


def main():
    """Main entry point for the stop hook."""
    event_json = os.environ.get("CLAUDE_HOOK_EVENT", "")

    if not event_json and not sys.stdin.isatty():
        event_json = sys.stdin.read()

    try:
        event = json.loads(event_json) if event_json else {}
    except json.JSONDecodeError:
        event = {}

    stop_event = process_stop_event(event)

    # Update session metadata
    update_session_metadata(
        stop_event["stop_reason"],
        {
            "error_message": stop_event.get("error_message"),
            "tools_used_count": stop_event.get("tools_used_count"),
            "last_tool": stop_event.get("last_tool"),
        }
    )

    # Append to trace
    append_to_trace(stop_event)

    reason = stop_event["stop_reason"]
    print(f"[BashGym] Session stopped: {reason}")

    if stop_event.get("error_message"):
        error_preview = stop_event["error_message"][:100]
        print(f"[BashGym] Error: {error_preview}")


if __name__ == "__main__":
    main()
