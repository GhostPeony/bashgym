#!/usr/bin/env python3
"""
Post Tool Use Hook for Claude Code Instrumentation

This script intercepts Claude Code tool usage events and captures them
for the Golden Trace pipeline. It reads the CLAUDE_HOOK_EVENT environment
variable and appends relevant tool uses to the session trace.

Traces are stored globally in ~/.bashgym/traces/ with repo tagging,
allowing for both per-repo specialized training and cross-repo generalist training.

Module 1: Agent Instrumentation (The "Recorder")
"""

import os
import sys
import json
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# Cross-platform file locking
import platform
if platform.system() == 'Windows':
    import msvcrt

    def lock_file(f, exclusive=False):
        """Lock a file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)

    def unlock_file(f):
        """Unlock a file on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # File may not have been locked
else:
    import fcntl

    def lock_file(f, exclusive=False):
        """Lock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def unlock_file(f):
        """Unlock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_bashgym_dir() -> Path:
    """Get the global Bash Gym directory (~/.bashgym/)."""
    if platform.system() == 'Windows':
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym"


def get_repo_info() -> Dict[str, Any]:
    """Get information about the current git repository."""
    cwd = Path.cwd()

    repo_info = {
        "path": str(cwd),
        "name": cwd.name,
        "git_remote": None,
        "git_branch": None,
        "is_git_repo": False
    }

    try:
        # Check if we're in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=cwd, timeout=5
        )
        if result.returncode == 0:
            repo_info["is_git_repo"] = True

            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, cwd=cwd, timeout=5
            )
            if result.returncode == 0:
                repo_info["git_remote"] = result.stdout.strip()

            # Get current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=cwd, timeout=5
            )
            if result.returncode == 0:
                repo_info["git_branch"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # Git not available or timed out

    return repo_info


def get_session_id() -> str:
    """Get or create a session ID for the current session."""
    bashgym_dir = get_bashgym_dir()
    session_file = bashgym_dir / "current_session_id"

    # Create directory if needed
    bashgym_dir.mkdir(parents=True, exist_ok=True)

    if session_file.exists():
        return session_file.read_text().strip()

    # Generate new session ID
    session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_file.write_text(session_id)
    return session_id


def get_trace_file() -> Path:
    """Get the path to the current session's trace file."""
    bashgym_dir = get_bashgym_dir()
    traces_dir = bashgym_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    session_id = get_session_id()
    return traces_dir / f"session_{session_id}.json"


# Configuration - Capture ALL tools for comprehensive training data
RELEVANT_TOOLS = {
    # Core file operations
    "Bash", "Edit", "Write", "Read",
    "bash", "edit", "write", "read",
    # Search tools (important for understanding agent search patterns)
    "Glob", "Grep", "glob", "grep",
    # Web tools
    "WebFetch", "WebSearch", "webfetch", "websearch",
    # Task management
    "Task", "TodoWrite", "task", "todowrite",
    # User interaction
    "AskUserQuestion", "askuserquestion",
    # Notebook operations
    "NotebookEdit", "notebookedit",
    # Planning
    "EnterPlanMode", "ExitPlanMode", "enterplanmode", "exitplanmode",
    # Skills
    "Skill", "skill",
}


def get_timestamp() -> str:
    """Generate ISO format timestamp with timezone."""
    return datetime.now(timezone.utc).isoformat()


def generate_step_id() -> str:
    """Generate unique step identifier."""
    return f"{get_timestamp()}_{uuid.uuid4().hex[:8]}"


def safe_json_load(file_path: Path) -> List[Dict[str, Any]]:
    """Safely load JSON from file with file locking."""
    if not file_path.exists():
        return []

    try:
        with open(file_path, 'r') as f:
            lock_file(f, exclusive=False)
            content = f.read()
            unlock_file(f)

        if not content.strip():
            return []
        return json.loads(content)
    except (json.JSONDecodeError, IOError, OSError) as e:
        print(f"Warning: Could not load trace file: {e}", file=sys.stderr)
        return []


def safe_json_dump(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Safely write JSON to file with file locking."""
    try:
        with open(file_path, 'w') as f:
            lock_file(f, exclusive=True)
            json.dump(data, f, indent=2, ensure_ascii=False)
            unlock_file(f)
    except (IOError, OSError) as e:
        print(f"Error: Could not write trace file: {e}", file=sys.stderr)
        sys.exit(1)


def extract_tool_info(event: Dict[str, Any], repo_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract relevant information from a tool use event.

    Handles both single tool uses and batched tool uses.
    Includes repo information for hybrid storage approach.
    """
    tool_name = event.get("tool_name", event.get("tool", ""))

    # Check if this is a relevant tool
    if tool_name not in RELEVANT_TOOLS:
        return None

    # Extract command/input based on tool type
    tool_input = event.get("tool_input", event.get("input", {}))
    tool_output = event.get("tool_output", event.get("output", ""))

    # Handle different input formats
    if isinstance(tool_input, str):
        command = tool_input
    elif isinstance(tool_input, dict):
        command = tool_input.get("command", tool_input.get("content", str(tool_input)))
    else:
        command = str(tool_input)

    # Handle different output formats
    if isinstance(tool_output, dict):
        output = tool_output.get("stdout", "") + tool_output.get("stderr", "")
        exit_code = tool_output.get("exit_code", tool_output.get("exitCode", None))
    else:
        output = str(tool_output) if tool_output else ""
        exit_code = None

    return {
        "step_id": generate_step_id(),
        "timestamp": get_timestamp(),
        "tool_name": tool_name,
        "command": command,
        "output": output[:10000],  # Truncate very long outputs
        "exit_code": exit_code,
        "cwd": os.getcwd(),
        "success": exit_code == 0 if exit_code is not None else None,
        "repo": repo_info
    }


def process_hook_event(event_json: str) -> None:
    """
    Process a Claude Code hook event.

    The event is passed via CLAUDE_HOOK_EVENT environment variable
    and contains JSON data about the tool use.

    Traces are stored globally in ~/.bashgym/traces/ with repo metadata.
    """
    try:
        event = json.loads(event_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in hook event: {e}", file=sys.stderr)
        return

    # Handle both single events and lists of events
    events = event if isinstance(event, list) else [event]

    # Get repo info once for all events in this batch
    repo_info = get_repo_info()

    # Get the global trace file path
    trace_file = get_trace_file()

    # Load existing trace
    trace = safe_json_load(trace_file)

    # Process each event
    for evt in events:
        tool_info = extract_tool_info(evt, repo_info)
        if tool_info:
            trace.append(tool_info)
            print(f"[BashGym] Captured: {tool_info['tool_name']} - {tool_info['command'][:50]}... ({repo_info['name']})")

    # Save updated trace
    if trace:
        safe_json_dump(trace, trace_file)


def main():
    """Main entry point for the post_tool_use hook."""
    # Read event from environment variable
    event_json = os.environ.get("CLAUDE_HOOK_EVENT")

    if not event_json:
        # Also check stdin for piped input
        if not sys.stdin.isatty():
            event_json = sys.stdin.read()

    if not event_json:
        print("Warning: No CLAUDE_HOOK_EVENT found", file=sys.stderr)
        return

    process_hook_event(event_json)


if __name__ == "__main__":
    main()
