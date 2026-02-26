"""
Codex Adapter (Import Only)

Codex has no hook system for live trace capture.
This adapter provides session import functionality only,
converting Codex transcript files into BashGym trace format.
"""

import os
import json
import uuid
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional


def _get_codex_dir() -> Path:
    """Get the Codex config/data directory."""
    if platform.system() == 'Windows':
        home = Path(os.environ.get("USERPROFILE", ""))
    else:
        home = Path.home()
    return home / ".codex"


def _get_bashgym_dir() -> Path:
    """Get the global Bash Gym directory (~/.bashgym/)."""
    if platform.system() == 'Windows':
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym"


def _get_traces_dir() -> Path:
    """Get the BashGym traces directory."""
    traces_dir = _get_bashgym_dir() / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    return traces_dir


def install_codex_hooks() -> Tuple[bool, str]:
    """
    Codex has no hook system - return informational message.

    Returns:
        Tuple of (success, message)
    """
    return (
        True,
        "Codex uses session import (no live hooks). "
        "Use 'Import Sessions' to pull transcripts."
    )


def uninstall_codex_hooks() -> Tuple[bool, str]:
    """
    Codex has no hooks to uninstall.

    Returns:
        Tuple of (success, message)
    """
    return True, "Codex has no hooks to uninstall"


def _convert_tool_call(tool_call: Dict[str, Any], cwd: str) -> Dict[str, Any]:
    """Convert a Codex tool call to a BashGym trace step."""
    tool_name = tool_call.get("name", tool_call.get("tool", "unknown"))
    tool_input = tool_call.get("input", tool_call.get("arguments", {}))
    tool_output = tool_call.get("output", tool_call.get("result", ""))

    # Extract command from input
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

    timestamp = tool_call.get("timestamp", datetime.now(timezone.utc).isoformat())

    return {
        "step_id": f"{timestamp}_{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp,
        "tool_name": tool_name,
        "command": command[:10000],
        "output": output[:10000],
        "exit_code": exit_code,
        "success": exit_code == 0 if exit_code is not None else None,
        "cwd": cwd,
        "source_tool": "codex",
        "metadata": {}
    }


def _extract_user_text(msg: Dict[str, Any]) -> Optional[str]:
    """
    Extract user text from a message dict.

    Handles both plain string content and list-of-parts content.

    Returns:
        The user text string, or None if no text was found.
    """
    content = msg.get("content", msg.get("text", ""))
    if isinstance(content, str) and content:
        return content[:2000]
    if isinstance(content, list):
        text_parts = []
        for c in content:
            if isinstance(c, str):
                text_parts.append(c)
            elif isinstance(c, dict) and c.get("type") == "text":
                text_parts.append(c.get("text", ""))
        if text_parts:
            return "\n".join(text_parts)[:2000]
    return None


def parse_codex_transcript(
    transcript: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse a Codex transcript and extract trace steps plus conversation metadata.

    Args:
        transcript: Parsed JSON transcript (dict or list).

    Returns:
        Tuple of (trace_steps, metadata).
        metadata contains: conversation_turns, all_user_prompts,
        user_initial_prompt, models_used.
    """
    meta: Dict[str, Any] = {
        "conversation_turns": 0,
        "all_user_prompts": [],
        "user_initial_prompt": None,
        "models_used": [],
    }
    models_used: set = set()

    # Determine CWD from transcript metadata
    cwd = "unknown"
    if isinstance(transcript, dict):
        cwd = transcript.get("cwd", transcript.get("project_dir", "unknown"))

    # Extract session-level model
    if isinstance(transcript, dict):
        session_model = transcript.get("model", "")
        if session_model:
            models_used.add(session_model)

    # Extract tool calls and user messages from transcript
    tool_calls: List[Dict[str, Any]] = []
    if isinstance(transcript, dict):
        # Look for tool_calls in various locations
        tool_calls = transcript.get("tool_calls", [])
        if not tool_calls:
            tool_calls = transcript.get("steps", [])
        if not tool_calls:
            # Check messages array for tool use and user messages
            messages = transcript.get("messages", [])
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")

                # Capture user messages for metadata
                if role == "user":
                    user_text = _extract_user_text(msg)
                    if user_text:
                        meta["conversation_turns"] += 1
                        meta["all_user_prompts"].append({
                            "text": user_text,
                            "timestamp": msg.get("timestamp"),
                        })
                        if meta["user_initial_prompt"] is None:
                            meta["user_initial_prompt"] = user_text[:500]

                # Capture model from assistant messages
                if role == "assistant":
                    msg_model = msg.get("model", "")
                    if msg_model:
                        models_used.add(msg_model)
                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])

                elif msg.get("type") == "tool_call":
                    tool_calls.append(msg)
        else:
            # If tool_calls were found at the top level, still scan messages
            # for user prompts and models
            messages = transcript.get("messages", [])
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")

                if role == "user":
                    user_text = _extract_user_text(msg)
                    if user_text:
                        meta["conversation_turns"] += 1
                        meta["all_user_prompts"].append({
                            "text": user_text,
                            "timestamp": msg.get("timestamp"),
                        })
                        if meta["user_initial_prompt"] is None:
                            meta["user_initial_prompt"] = user_text[:500]

                if role == "assistant":
                    msg_model = msg.get("model", "")
                    if msg_model:
                        models_used.add(msg_model)

    elif isinstance(transcript, list):
        tool_calls = transcript

    # Convert tool calls to trace steps
    trace_steps: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            step = _convert_tool_call(tool_call, cwd)
            trace_steps.append(step)

    meta["models_used"] = sorted(models_used)
    return trace_steps, meta


def import_codex_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Import Codex session transcripts into BashGym trace format.

    Searches ~/.codex/ for JSON transcript files and converts
    tool_calls to BashGym trace steps.

    Args:
        limit: Maximum number of sessions to import.

    Returns:
        List of import result dicts with keys: session_file, trace_file, steps_imported, error
    """
    codex_dir = _get_codex_dir()
    traces_dir = _get_traces_dir()
    results = []

    if not codex_dir.exists():
        return results

    # Find JSON transcript files in ~/.codex/
    transcript_files = []
    for pattern in ["*.json", "**/*.json"]:
        transcript_files.extend(codex_dir.glob(pattern))

    # Deduplicate and sort by modification time (newest first)
    seen = set()
    unique_files = []
    for f in sorted(transcript_files, key=lambda p: p.stat().st_mtime, reverse=True):
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    # Apply limit
    unique_files = unique_files[:limit]

    for transcript_file in unique_files:
        result: Dict[str, Any] = {
            "session_file": str(transcript_file),
            "trace_file": None,
            "steps_imported": 0,
            "error": None,
        }

        try:
            with open(transcript_file, 'r') as f:
                transcript = json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            result["error"] = str(e)
            results.append(result)
            continue

        trace_steps, meta = parse_codex_transcript(transcript)

        if not trace_steps:
            # Skip files with no tool calls
            continue

        # Generate session ID from file name
        session_name = transcript_file.stem
        session_id = f"codex_{session_name}"
        trace_file = traces_dir / f"session_{session_id}.json"

        # Skip if already imported
        if trace_file.exists():
            result["trace_file"] = str(trace_file)
            result["error"] = "already imported"
            results.append(result)
            continue

        # Build enriched trace output with metadata
        trace_output = {
            "steps": trace_steps,
            "metadata": meta,
        }

        if trace_steps:
            try:
                with open(trace_file, 'w') as f:
                    json.dump(trace_output, f, indent=2, ensure_ascii=False)
                result["trace_file"] = str(trace_file)
                result["steps_imported"] = len(trace_steps)
                result["conversation_turns"] = meta.get("conversation_turns", 0)
                result["user_initial_prompt"] = meta.get("user_initial_prompt")
                result["models_used"] = meta.get("models_used", [])
            except (IOError, OSError) as e:
                result["error"] = str(e)

        results.append(result)

    return results


def get_install_command() -> str:
    """Get manual instructions for Codex integration."""
    return "No installation needed. Use 'Import Sessions' to pull Codex transcripts."
