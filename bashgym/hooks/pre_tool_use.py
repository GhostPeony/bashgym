#!/usr/bin/env python3
"""
Pre Tool Use Hook for Claude Code Instrumentation

Captures the model's reasoning and planning before each tool execution:
- What tool is about to be used
- The model's reasoning/thinking (if available)
- The planned action and expected outcome

This helps capture the "why" behind tool choices for better training data.
"""

import os
import sys
import json
import uuid
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

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


def get_pending_file() -> Path:
    """Get the path to the pending tool use file."""
    bashgym_dir = get_bashgym_dir()
    return bashgym_dir / "pending_tool_use.json"


def save_pending_tool_use(tool_info: Dict[str, Any]) -> None:
    """Save pending tool use info to be merged with post_tool_use."""
    pending_file = get_pending_file()

    try:
        with open(pending_file, 'w') as f:
            lock_file(f, exclusive=True)
            json.dump(tool_info, f, indent=2)
            unlock_file(f)
    except (IOError, OSError) as e:
        print(f"Warning: Could not save pending tool use: {e}", file=sys.stderr)


def extract_reasoning(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract reasoning and planning information from the event."""

    tool_name = event.get("tool_name", event.get("tool", "unknown"))
    tool_input = event.get("tool_input", event.get("input", {}))

    # Extract thinking/reasoning if available
    thinking = event.get("thinking", event.get("reasoning", ""))
    plan = event.get("plan", event.get("strategy", ""))

    # Extract the command/action being planned
    if isinstance(tool_input, str):
        planned_action = tool_input
    elif isinstance(tool_input, dict):
        planned_action = tool_input.get("command", tool_input.get("content", str(tool_input)))
    else:
        planned_action = str(tool_input)

    return {
        "pre_tool_id": f"{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "planned_action": planned_action[:5000],  # Truncate very long actions
        "thinking": thinking[:5000] if thinking else None,
        "plan": plan[:2000] if plan else None,
        "session_id": get_session_id(),
    }


def main():
    """Main entry point for the pre_tool_use hook."""
    event_json = os.environ.get("CLAUDE_HOOK_EVENT", "")

    if not event_json and not sys.stdin.isatty():
        event_json = sys.stdin.read()

    if not event_json:
        return

    try:
        event = json.loads(event_json)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in pre_tool_use hook: {e}", file=sys.stderr)
        return

    # Handle both single events and lists
    events = event if isinstance(event, list) else [event]

    for evt in events:
        tool_info = extract_reasoning(evt)

        # Save pending info for post_tool_use to pick up
        save_pending_tool_use(tool_info)

        tool_name = tool_info.get("tool_name", "unknown")
        action_preview = tool_info.get("planned_action", "")[:50]

        print(f"[BashGym] Planning: {tool_name} - {action_preview}...")


if __name__ == "__main__":
    main()
