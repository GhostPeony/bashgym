#!/usr/bin/env python3
"""
Notification Hook for Claude Code Instrumentation

Captures notification events during sessions:
- Errors and warnings
- Rate limiting notices
- API issues
- Tool failures
- Important status updates

This helps identify issues and capture error-handling patterns.
"""

import os
import sys
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

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


def classify_notification(event: Dict[str, Any]) -> str:
    """Classify the notification type based on event content."""
    message = str(event.get("message", event.get("content", ""))).lower()
    level = event.get("level", event.get("type", "info")).lower()

    # Check for specific patterns
    if any(word in message for word in ["error", "failed", "failure", "exception"]):
        return "error"
    elif any(word in message for word in ["rate limit", "throttl", "too many requests"]):
        return "rate_limit"
    elif any(word in message for word in ["timeout", "timed out"]):
        return "timeout"
    elif any(word in message for word in ["warning", "warn", "caution"]):
        return "warning"
    elif any(word in message for word in ["api", "network", "connection"]):
        return "api_issue"
    elif any(word in message for word in ["permission", "denied", "unauthorized"]):
        return "permission"
    elif any(word in message for word in ["success", "complete", "done"]):
        return "success"
    elif level in ["error", "warning", "critical", "fatal"]:
        return level

    return "info"


def append_to_trace(notification_event: Dict[str, Any]) -> None:
    """Append notification event to the session trace."""
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

        traces.append(notification_event)

        with open(trace_file, 'w') as f:
            lock_file(f, exclusive=True)
            json.dump(traces, f, indent=2)
            unlock_file(f)

    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not append to trace: {e}", file=sys.stderr)


def update_session_errors(notification_type: str, message: str) -> None:
    """Track errors in session metadata for quality scoring."""
    if notification_type not in ["error", "rate_limit", "timeout", "api_issue"]:
        return

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

        # Initialize errors list if not present
        if "errors" not in metadata:
            metadata["errors"] = []

        # Add error summary (limit to prevent file bloat)
        if len(metadata["errors"]) < 50:
            metadata["errors"].append({
                "type": notification_type,
                "message": message[:500],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # Update error count
        metadata["error_count"] = len(metadata["errors"])

        with open(metadata_file, 'w') as f:
            lock_file(f, exclusive=True)
            json.dump(metadata, f, indent=2)
            unlock_file(f)

    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"Warning: Could not update session errors: {e}", file=sys.stderr)


def process_notification(event: Dict[str, Any]) -> Dict[str, Any]:
    """Process a notification event."""

    message = event.get("message", event.get("content", ""))
    if isinstance(message, dict):
        message = str(message)

    notification_type = classify_notification(event)

    notification_event = {
        "event_type": "notification",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": get_session_id(),
        "notification_type": notification_type,
        "level": event.get("level", event.get("type", "info")),
        "message": message[:2000] if message else None,
        "source": event.get("source", event.get("tool", "unknown")),
        "details": event.get("details", event.get("data", None)),
    }

    # Truncate details if too large
    if notification_event["details"]:
        details_str = json.dumps(notification_event["details"])
        if len(details_str) > 5000:
            notification_event["details"] = {"truncated": True, "preview": details_str[:1000]}

    return notification_event


def main():
    """Main entry point for the notification hook."""
    event_json = os.environ.get("CLAUDE_HOOK_EVENT", "")

    if not event_json and not sys.stdin.isatty():
        event_json = sys.stdin.read()

    if not event_json:
        return

    try:
        event = json.loads(event_json)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in notification hook: {e}", file=sys.stderr)
        return

    # Handle both single events and lists
    events = event if isinstance(event, list) else [event]

    for evt in events:
        notification = process_notification(evt)

        # Append to trace
        append_to_trace(notification)

        # Track errors in metadata
        update_session_errors(
            notification["notification_type"],
            notification.get("message", "")
        )

        # Log significant notifications
        if notification["notification_type"] in ["error", "rate_limit", "warning"]:
            msg_preview = (notification.get("message", "") or "")[:80]
            print(f"[BashGym] {notification['notification_type'].upper()}: {msg_preview}")


if __name__ == "__main__":
    main()
