#!/usr/bin/env python3
"""
Session End Hook for Gemini CLI Instrumentation

Called when a Gemini CLI session ends. Clears the session ID file
so the next session gets a fresh ID.

Must print valid JSON to stdout: {"decision": "allow"}
"""

import json
import os
import platform
import sys
from pathlib import Path


def get_bashgym_dir() -> Path:
    """Get the global Bash Gym directory (~/.bashgym/)."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym"


def clear_session() -> None:
    """Clear the current Gemini session ID file."""
    session_file = get_bashgym_dir() / "current_gemini_session_id"

    if session_file.exists():
        try:
            session_id = session_file.read_text().strip()
            session_file.unlink()
            print(f"[BashGym] Gemini session ended: {session_id}", file=sys.stderr)
        except OSError as e:
            print(f"Warning: Could not clear session file: {e}", file=sys.stderr)


def main():
    """Main entry point for the gemini_session_end hook."""
    # Read stdin (Gemini CLI may pass event data)
    if not sys.stdin.isatty():
        _ = sys.stdin.read()  # Consume stdin but we don't need it

    clear_session()

    # Required response for Gemini CLI
    print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
