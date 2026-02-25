#!/usr/bin/env python3
"""
After Tool Hook for Gemini CLI Instrumentation

This script intercepts Gemini CLI tool usage events and captures them
for the Golden Trace pipeline. It reads JSON from stdin (as required by
Gemini CLI's hook system) and appends relevant tool uses to the session trace.

Traces are stored globally in ~/.bashgym/traces/ with repo tagging,
allowing for both per-repo specialized training and cross-repo generalist training.

Environment vars available from Gemini CLI:
  GEMINI_SESSION_ID - Current Gemini session ID
  GEMINI_PROJECT_DIR - Project directory
  GEMINI_CWD - Current working directory
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


def get_repo_info(cwd: str) -> Dict[str, Any]:
    """Get information about the current git repository."""
    cwd_path = Path(cwd)

    repo_info = {
        "path": str(cwd_path),
        "name": cwd_path.name,
        "git_remote": None,
        "git_branch": None,
        "is_git_repo": False
    }

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=cwd_path, timeout=5
        )
        if result.returncode == 0:
            repo_info["is_git_repo"] = True

            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, cwd=cwd_path, timeout=5
            )
            if result.returncode == 0:
                repo_info["git_remote"] = result.stdout.strip()

            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=cwd_path, timeout=5
            )
            if result.returncode == 0:
                repo_info["git_branch"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return repo_info


def get_session_id() -> str:
    """
    Get or create a session ID for the current session.

    Prefers GEMINI_SESSION_ID env var, prefixed with 'gemini_'.
    Falls back to generating a new ID.
    """
    gemini_session = os.environ.get("GEMINI_SESSION_ID")
    if gemini_session:
        return f"gemini_{gemini_session}"

    bashgym_dir = get_bashgym_dir()
    session_file = bashgym_dir / "current_gemini_session_id"

    bashgym_dir.mkdir(parents=True, exist_ok=True)

    if session_file.exists():
        return session_file.read_text().strip()

    session_id = f"gemini_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_file.write_text(session_id)
    return session_id


def get_trace_file() -> Path:
    """Get the path to the current session's trace file."""
    bashgym_dir = get_bashgym_dir()
    traces_dir = bashgym_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    session_id = get_session_id()
    return traces_dir / f"session_{session_id}.json"


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


def process_tool_event(event: Dict[str, Any]) -> None:
    """
    Process a Gemini CLI tool event.

    Gemini CLI passes tool events as JSON via stdin with fields:
      tool_name, tool_input, tool_output
    """
    tool_name = event.get("tool_name", "unknown")
    tool_input = event.get("tool_input", {})
    tool_output = event.get("tool_output", "")

    # Extract command from tool input
    if isinstance(tool_input, str):
        command = tool_input
    elif isinstance(tool_input, dict):
        command = tool_input.get("command", tool_input.get("content", json.dumps(tool_input)))
    else:
        command = str(tool_input)

    # Extract output
    if isinstance(tool_output, dict):
        output = tool_output.get("stdout", "") + tool_output.get("stderr", "")
        exit_code = tool_output.get("exit_code", tool_output.get("exitCode", None))
    else:
        output = str(tool_output) if tool_output else ""
        exit_code = None

    # Determine CWD - prefer Gemini env vars
    cwd = os.environ.get("GEMINI_CWD", os.getcwd())

    # Get repo info
    repo_info = get_repo_info(cwd)

    # Create trace step
    step = {
        "step_id": generate_step_id(),
        "timestamp": get_timestamp(),
        "tool_name": tool_name,
        "command": command[:10000],
        "output": output[:10000],
        "exit_code": exit_code,
        "success": exit_code == 0 if exit_code is not None else None,
        "cwd": cwd,
        "repo": repo_info,
        "source_tool": "gemini_cli",
        "metadata": {
            "gemini_session_id": os.environ.get("GEMINI_SESSION_ID"),
            "gemini_project_dir": os.environ.get("GEMINI_PROJECT_DIR"),
        }
    }

    # Append to trace file
    trace_file = get_trace_file()
    trace = safe_json_load(trace_file)
    trace.append(step)
    safe_json_dump(trace, trace_file)

    print(f"[BashGym] Captured: {tool_name} - {command[:50]}... ({repo_info['name']})",
          file=sys.stderr)


def main():
    """Main entry point for the gemini_after_tool hook."""
    # Gemini CLI passes JSON via stdin
    event_json = ""
    if not sys.stdin.isatty():
        event_json = sys.stdin.read()

    if not event_json:
        # Output required response and exit
        print(json.dumps({"decision": "allow"}))
        return

    try:
        event = json.loads(event_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON from Gemini CLI: {e}", file=sys.stderr)
        print(json.dumps({"decision": "allow"}))
        return

    process_tool_event(event)

    # Required response for Gemini CLI
    print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
