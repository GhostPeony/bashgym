"""
OpenCode Session History Importer

Imports tool execution traces from OpenCode's session/message/part
storage files into BashGym's trace format.

OpenCode stores session data in a hierarchical structure:
  Linux primary:     ~/.local/share/opencode/storage/
  Linux alternative: ~/.config/opencode/storage/
  Windows:           %APPDATA%/opencode/storage/
                     or %LOCALAPPDATA%/opencode/storage/

Subdirectories:
  session/<projectID>/<sessionID>.json   - Session metadata
  message/<sessionID>/*.json             - Individual messages
  part/<messageID>/*.json                - Message parts (tool calls, responses)

Also supports CLI export: `opencode session list --format json`
                          `opencode export <sessionID>`
"""

import json
import os
import platform
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict

from ..core import TraceStep, TraceSession, TraceCapture


@dataclass
class OpenCodeImportResult:
    """Result of an OpenCode session import operation."""
    session_id: str
    source_file: Optional[Path] = None
    steps_imported: int = 0
    destination_file: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    import_method: str = "file"  # "file" or "cli"


class OpenCodeSessionImporter:
    """
    Import traces from OpenCode session files.

    Tries CLI export first if the opencode binary is available,
    then falls back to direct file scanning.
    """

    def __init__(self):
        self.trace_capture = TraceCapture()
        self.storage_dirs = self._get_storage_dirs()
        self.imported_sessions_file = (
            self.trace_capture.bashgym_dir / "imported_opencode_sessions.json"
        )
        self._imported_sessions: Optional[Set[str]] = None
        self._opencode_available: Optional[bool] = None

    @staticmethod
    def _get_storage_dirs() -> List[Path]:
        """
        Get all possible OpenCode storage directories.

        Returns directories in priority order (most likely first).
        """
        dirs: List[Path] = []

        if platform.system() == "Windows":
            base = Path(os.environ.get("USERPROFILE", ""))
            appdata = os.environ.get("APPDATA", "")
            localappdata = os.environ.get("LOCALAPPDATA", "")

            if appdata:
                dirs.append(Path(appdata) / "opencode" / "storage")
            if localappdata:
                dirs.append(Path(localappdata) / "opencode" / "storage")
            dirs.append(base / ".opencode" / "storage")
        else:
            home = Path.home()
            dirs.extend([
                home / ".local" / "share" / "opencode" / "storage",
                home / ".config" / "opencode" / "storage",
                home / ".opencode" / "storage",
            ])

        # Also check XDG_DATA_HOME if set
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            dirs.insert(0, Path(xdg_data) / "opencode" / "storage")

        return dirs

    def _find_active_storage_dir(self) -> Optional[Path]:
        """Find the first existing storage directory."""
        for d in self.storage_dirs:
            if d.exists() and d.is_dir():
                return d
        return None

    @property
    def opencode_available(self) -> bool:
        """Check if the opencode CLI binary is available."""
        if self._opencode_available is None:
            self._opencode_available = shutil.which("opencode") is not None
        return self._opencode_available

    @property
    def imported_sessions(self) -> Set[str]:
        """Get set of already imported session IDs."""
        if self._imported_sessions is None:
            self._imported_sessions = self._load_imported_sessions()
        return self._imported_sessions

    def _load_imported_sessions(self) -> Set[str]:
        """Load the list of already imported session IDs."""
        if not self.imported_sessions_file.exists():
            return set()
        try:
            with open(self.imported_sessions_file, "r") as f:
                data = json.load(f)
                return set(data.get("sessions", []))
        except (json.JSONDecodeError, IOError):
            return set()

    def _save_imported_session(self, session_id: str) -> None:
        """Mark a session as imported."""
        self.imported_sessions.add(session_id)
        try:
            self.imported_sessions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.imported_sessions_file, "w") as f:
                json.dump({"sessions": list(self.imported_sessions)}, f)
        except IOError as e:
            print(f"Warning: Could not save imported OpenCode sessions list: {e}")

    # ------------------------------------------------------------------
    # CLI-based import
    # ------------------------------------------------------------------

    def _export_session_via_cli(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export a session using the opencode CLI.

        Args:
            session_id: The session ID to export

        Returns:
            Parsed JSON dict of the exported session, or None on failure
        """
        if not self.opencode_available:
            return None

        try:
            result = subprocess.run(
                ["opencode", "export", session_id, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError):
            pass

        # Try alternative export command
        try:
            result = subprocess.run(
                ["opencode", "session", "export", session_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError):
            pass

        return None

    def _list_sessions_via_cli(self) -> List[Dict[str, Any]]:
        """
        List available sessions using the opencode CLI.

        Returns:
            List of session metadata dicts
        """
        if not self.opencode_available:
            return []

        try:
            result = subprocess.run(
                ["opencode", "session", "list", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get("sessions", [data])
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError, OSError):
            pass

        return []

    # ------------------------------------------------------------------
    # File-based import
    # ------------------------------------------------------------------

    def find_session_files(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Tuple[Path, datetime]]:
        """
        Find OpenCode session JSON files by scanning storage directories.

        Args:
            since: Only include sessions modified after this time
            until: Only include sessions modified before this time

        Returns:
            List of (file_path, modified_time) tuples sorted newest first
        """
        storage_dir = self._find_active_storage_dir()
        if not storage_dir:
            return []

        session_files: List[Tuple[Path, datetime]] = []
        session_dir = storage_dir / "session"

        if not session_dir.exists():
            return []

        try:
            # session/<projectID>/<sessionID>.json
            for project_dir in session_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                for json_file in project_dir.glob("*.json"):
                    try:
                        mtime = datetime.fromtimestamp(
                            json_file.stat().st_mtime, tz=timezone.utc
                        )
                    except OSError:
                        continue

                    if since and mtime < since:
                        continue
                    if until and mtime > until:
                        continue

                    session_files.append((json_file, mtime))
        except (PermissionError, OSError):
            pass

        session_files.sort(key=lambda x: x[1], reverse=True)
        return session_files

    def _load_messages_for_session(
        self, session_id: str, storage_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Load all message files for a given session.

        Args:
            session_id: The session ID
            storage_dir: Root storage directory

        Returns:
            List of message dicts, sorted by timestamp if available
        """
        messages_dir = storage_dir / "message" / session_id
        if not messages_dir.exists():
            # Try without subdirectory structure
            messages_dir = storage_dir / "message"
            if not messages_dir.exists():
                return []

        messages: List[Dict[str, Any]] = []

        try:
            for msg_file in messages_dir.glob("*.json"):
                try:
                    with open(msg_file, "r", encoding="utf-8") as f:
                        msg = json.load(f)
                    if isinstance(msg, dict):
                        msg["_source_file"] = str(msg_file)
                        messages.append(msg)
                    elif isinstance(msg, list):
                        for m in msg:
                            if isinstance(m, dict):
                                m["_source_file"] = str(msg_file)
                                messages.append(m)
                except (json.JSONDecodeError, IOError):
                    continue
        except (PermissionError, OSError):
            pass

        # Sort by timestamp if available
        def sort_key(m: Dict) -> str:
            return m.get("timestamp", m.get("created_at", ""))

        messages.sort(key=sort_key)
        return messages

    def _load_parts_for_message(
        self, message_id: str, storage_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Load all part files for a given message.

        Args:
            message_id: The message ID
            storage_dir: Root storage directory

        Returns:
            List of part dicts
        """
        parts_dir = storage_dir / "part" / message_id
        if not parts_dir.exists():
            return []

        parts: List[Dict[str, Any]] = []

        try:
            for part_file in parts_dir.glob("*.json"):
                try:
                    with open(part_file, "r", encoding="utf-8") as f:
                        part = json.load(f)
                    if isinstance(part, dict):
                        parts.append(part)
                    elif isinstance(part, list):
                        parts.extend(p for p in part if isinstance(p, dict))
                except (json.JSONDecodeError, IOError):
                    continue
        except (PermissionError, OSError):
            pass

        return parts

    def _extract_steps_from_parts(
        self,
        parts: List[Dict[str, Any]],
        session_id: str,
        session_file: Path,
        msg_idx: int,
    ) -> List[TraceStep]:
        """
        Extract TraceStep objects from message parts.

        Parts may contain:
          - tool_call: {"type": "tool-call", "name": "...", "input": {...}}
          - tool_result: {"type": "tool-result", "name": "...", "output": "..."}
          - text: {"type": "text", "content": "..."}
        """
        steps: List[TraceStep] = []

        for part_idx, part in enumerate(parts):
            part_type = part.get("type", "")
            timestamp = part.get(
                "timestamp",
                part.get("created_at", datetime.now(timezone.utc).isoformat()),
            )
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).isoformat()

            # Tool call part
            if part_type in ("tool-call", "tool_call", "toolCall"):
                tool_name = part.get("name", part.get("tool_name", "unknown"))
                tool_input = part.get("input", part.get("args", part.get("arguments", {})))

                command = (
                    json.dumps(tool_input)
                    if isinstance(tool_input, dict)
                    else str(tool_input)
                )

                step = TraceStep(
                    step_id=f"opencode_{session_id}_{msg_idx}_{part_idx}",
                    timestamp=timestamp,
                    tool_name=tool_name,
                    command=command,
                    output="",
                    exit_code=None,
                    success=None,
                    cwd="",
                    repo={"path": "", "name": "unknown"},
                    source_tool="opencode",
                    metadata={
                        "session_id": session_id,
                        "imported_from": str(session_file),
                        "part_type": part_type,
                        "tool_call_id": part.get("id", part.get("tool_call_id", "")),
                    },
                )
                steps.append(step)

            # Tool result part
            elif part_type in ("tool-result", "tool_result", "toolResult"):
                tool_name = part.get("name", part.get("tool_name", "unknown"))
                output = part.get("output", part.get("result", part.get("content", "")))
                is_error = part.get("is_error", part.get("isError", False))

                output_str = (
                    json.dumps(output) if isinstance(output, dict) else str(output)
                )[:10000]

                # Try to match to a preceding tool-call step
                tool_call_id = part.get("tool_call_id", part.get("id", ""))
                matched = False
                for existing_step in reversed(steps):
                    existing_tc_id = existing_step.metadata.get("tool_call_id", "")
                    if (
                        existing_tc_id
                        and existing_tc_id == tool_call_id
                        and not existing_step.output
                    ):
                        existing_step.output = output_str
                        existing_step.success = not is_error
                        existing_step.exit_code = 1 if is_error else 0
                        matched = True
                        break

                if not matched:
                    # Match by name if no ID match
                    for existing_step in reversed(steps):
                        if (
                            existing_step.tool_name == tool_name
                            and not existing_step.output
                        ):
                            existing_step.output = output_str
                            existing_step.success = not is_error
                            existing_step.exit_code = 1 if is_error else 0
                            matched = True
                            break

                if not matched:
                    # Standalone result step
                    step = TraceStep(
                        step_id=f"opencode_{session_id}_{msg_idx}_{part_idx}_result",
                        timestamp=timestamp,
                        tool_name=tool_name,
                        command="",
                        output=output_str,
                        exit_code=1 if is_error else 0,
                        success=not is_error,
                        cwd="",
                        repo={"path": "", "name": "unknown"},
                        source_tool="opencode",
                        metadata={
                            "session_id": session_id,
                            "imported_from": str(session_file),
                            "part_type": part_type,
                        },
                    )
                    steps.append(step)

        return steps

    def _extract_steps_from_message(
        self,
        msg: Dict[str, Any],
        session_id: str,
        session_file: Path,
        msg_idx: int,
        storage_dir: Optional[Path] = None,
    ) -> Tuple[List[TraceStep], Optional[str]]:
        """
        Extract TraceStep objects from a message dict.

        Also loads associated parts from storage if available.

        Returns:
            Tuple of (steps, user_text_or_None)
        """
        steps: List[TraceStep] = []
        user_text: Optional[str] = None
        role = msg.get("role", "")
        timestamp = msg.get(
            "timestamp",
            msg.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).isoformat()

        # Capture user text
        if role == "user":
            content = msg.get("content", msg.get("text", ""))
            if isinstance(content, str) and content:
                user_text = content[:2000]
            elif isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, str):
                        text_parts.append(c)
                    elif isinstance(c, dict) and c.get("type") == "text":
                        text_parts.append(c.get("text", ""))
                user_text = "\n".join(text_parts)[:2000] if text_parts else None

        # Extract inline tool calls from the message
        tool_calls = msg.get("tool_calls", msg.get("tools", []))
        if isinstance(tool_calls, list):
            for tc_idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue

                tool_name = tc.get("name", tc.get("type", "unknown"))
                tool_input = tc.get("input", tc.get("arguments", tc.get("args", {})))
                tool_output = tc.get("output", tc.get("result", ""))

                command = (
                    json.dumps(tool_input)
                    if isinstance(tool_input, dict)
                    else str(tool_input)
                )

                step = TraceStep(
                    step_id=f"opencode_{session_id}_{msg_idx}_{tc_idx}",
                    timestamp=timestamp,
                    tool_name=tool_name,
                    command=command,
                    output=str(tool_output)[:10000] if tool_output else "",
                    exit_code=0 if tool_output else None,
                    success=True if tool_output else None,
                    cwd="",
                    repo={"path": "", "name": "unknown"},
                    source_tool="opencode",
                    metadata={
                        "session_id": session_id,
                        "imported_from": str(session_file),
                    },
                )
                steps.append(step)

        # Load parts from storage if a message ID and storage dir are available
        message_id = msg.get("id", msg.get("message_id", ""))
        if message_id and storage_dir:
            parts = self._load_parts_for_message(str(message_id), storage_dir)
            if parts:
                part_steps = self._extract_steps_from_parts(
                    parts, session_id, session_file, msg_idx
                )
                steps.extend(part_steps)

        # Check for inline content blocks (similar to Claude format)
        content = msg.get("content", [])
        if isinstance(content, list):
            for c_idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")

                if block_type in ("tool_use", "tool-use", "toolUse"):
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    command = (
                        json.dumps(tool_input)
                        if isinstance(tool_input, dict)
                        else str(tool_input)
                    )

                    step = TraceStep(
                        step_id=f"opencode_{session_id}_{msg_idx}_c{c_idx}",
                        timestamp=timestamp,
                        tool_name=tool_name,
                        command=command,
                        output="",
                        exit_code=None,
                        success=None,
                        cwd="",
                        repo={"path": "", "name": "unknown"},
                        source_tool="opencode",
                        metadata={
                            "session_id": session_id,
                            "imported_from": str(session_file),
                            "tool_use_id": block.get("id", ""),
                        },
                    )
                    steps.append(step)

                elif block_type in ("tool_result", "tool-result", "toolResult"):
                    result_content = block.get("content", block.get("output", ""))
                    is_error = block.get("is_error", False)
                    tool_use_id = block.get("tool_use_id", "")

                    output_str = (
                        json.dumps(result_content)
                        if isinstance(result_content, (dict, list))
                        else str(result_content)
                    )[:10000]

                    # Try matching to a preceding tool_use step
                    matched = False
                    if tool_use_id:
                        for existing_step in reversed(steps):
                            if existing_step.metadata.get("tool_use_id") == tool_use_id:
                                existing_step.output = output_str
                                existing_step.success = not is_error
                                existing_step.exit_code = 1 if is_error else 0
                                matched = True
                                break

                    if not matched:
                        for existing_step in reversed(steps):
                            if not existing_step.output:
                                existing_step.output = output_str
                                existing_step.success = not is_error
                                existing_step.exit_code = 1 if is_error else 0
                                break

        return steps, user_text

    def parse_session_from_files(
        self, session_file: Path
    ) -> Tuple[List[TraceStep], Dict[str, Any]]:
        """
        Parse an OpenCode session by reading session, message, and part files.

        Args:
            session_file: Path to the session JSON file

        Returns:
            Tuple of (steps, session_metadata)
        """
        session_id = session_file.stem
        storage_dir = session_file.parent.parent.parent  # session/<projId>/<file>.json -> storage/

        meta: Dict[str, Any] = {
            "user_initial_prompt": None,
            "all_user_prompts": [],
            "conversation_turns": 0,
        }

        # Load session metadata
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading OpenCode session file {session_file}: {e}")
            return [], {"user_initial_prompt": None}

        if not isinstance(session_data, dict):
            return [], {"user_initial_prompt": None}

        # Extract session-level metadata
        meta["opencode_session_title"] = session_data.get("title", "")
        meta["opencode_project_id"] = session_data.get("project_id", "")

        steps: List[TraceStep] = []

        # Try loading messages from the message directory
        messages = self._load_messages_for_session(session_id, storage_dir)

        if messages:
            for msg_idx, msg in enumerate(messages):
                try:
                    msg_steps, user_text = self._extract_steps_from_message(
                        msg, session_id, session_file, msg_idx, storage_dir
                    )
                    steps.extend(msg_steps)

                    if user_text:
                        meta["conversation_turns"] += 1
                        meta["all_user_prompts"].append({
                            "text": user_text,
                            "timestamp": msg.get("timestamp"),
                        })
                        if meta["user_initial_prompt"] is None:
                            meta["user_initial_prompt"] = user_text[:500]
                except Exception:
                    continue
        else:
            # Fall back to inline messages in session data
            inline_messages = session_data.get(
                "messages", session_data.get("conversation", [])
            )
            if isinstance(inline_messages, list):
                for msg_idx, msg in enumerate(inline_messages):
                    if not isinstance(msg, dict):
                        continue
                    try:
                        msg_steps, user_text = self._extract_steps_from_message(
                            msg, session_id, session_file, msg_idx
                        )
                        steps.extend(msg_steps)

                        if user_text:
                            meta["conversation_turns"] += 1
                            meta["all_user_prompts"].append({
                                "text": user_text,
                                "timestamp": msg.get("timestamp"),
                            })
                            if meta["user_initial_prompt"] is None:
                                meta["user_initial_prompt"] = user_text[:500]
                    except Exception:
                        continue

        return steps, meta

    def parse_session_from_cli(
        self, session_data: Dict[str, Any], session_id: str
    ) -> Tuple[List[TraceStep], Dict[str, Any]]:
        """
        Parse an OpenCode session from CLI export output.

        Args:
            session_data: Parsed JSON from opencode export
            session_id: The session ID

        Returns:
            Tuple of (steps, session_metadata)
        """
        meta: Dict[str, Any] = {
            "user_initial_prompt": None,
            "all_user_prompts": [],
            "conversation_turns": 0,
            "opencode_session_title": session_data.get("title", ""),
        }

        steps: List[TraceStep] = []
        messages = session_data.get("messages", session_data.get("conversation", []))

        if not isinstance(messages, list):
            return [], {"user_initial_prompt": None}

        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            try:
                # Use a synthetic source file path for CLI exports
                source = Path(f"<opencode-cli-export:{session_id}>")
                msg_steps, user_text = self._extract_steps_from_message(
                    msg, session_id, source, msg_idx
                )
                steps.extend(msg_steps)

                if user_text:
                    meta["conversation_turns"] += 1
                    meta["all_user_prompts"].append({
                        "text": user_text,
                        "timestamp": msg.get("timestamp"),
                    })
                    if meta["user_initial_prompt"] is None:
                        meta["user_initial_prompt"] = user_text[:500]
            except Exception:
                continue

        return steps, meta

    def import_session(
        self, session_file: Path, force: bool = False
    ) -> OpenCodeImportResult:
        """
        Import a single OpenCode session file into BashGym format.

        Args:
            session_file: Path to the session JSON file
            force: Import even if already imported

        Returns:
            OpenCodeImportResult with import details
        """
        session_id = session_file.stem

        # Check if already imported
        if not force and session_id in self.imported_sessions:
            return OpenCodeImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="Already imported",
            )

        # Parse the session from files
        steps, session_meta = self.parse_session_from_files(session_file)

        if not steps:
            return OpenCodeImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="No relevant tool executions found",
            )

        # Create trace session
        user_initial_prompt = (
            session_meta.pop("user_initial_prompt", None)
            or "Imported OpenCode session"
        )
        session = TraceSession.from_steps(
            steps,
            source_tool="opencode",
            verification_passed=None,
            imported=True,
            import_source=str(session_file),
            user_initial_prompt=user_initial_prompt,
            **session_meta,
        )

        # Save to traces directory
        try:
            mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
        except OSError:
            mtime = datetime.now()
        timestamp = mtime.strftime("%Y%m%d_%H%M%S")
        filename = f"imported_opencode_{session_id[:8]}_{timestamp}.json"
        destination = self.trace_capture.traces_dir / filename

        try:
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
        except IOError as e:
            return OpenCodeImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                error=f"Failed to write trace: {e}",
            )

        # Mark as imported
        self._save_imported_session(session_id)

        return OpenCodeImportResult(
            session_id=session_id,
            source_file=session_file,
            steps_imported=len(steps),
            destination_file=destination,
            import_method="file",
        )

    def import_session_from_cli(
        self, session_id: str, force: bool = False
    ) -> OpenCodeImportResult:
        """
        Import a session using the opencode CLI export.

        Args:
            session_id: The session ID to export and import
            force: Import even if already imported

        Returns:
            OpenCodeImportResult with import details
        """
        # Check if already imported
        if not force and session_id in self.imported_sessions:
            return OpenCodeImportResult(
                session_id=session_id,
                steps_imported=0,
                skipped=True,
                skip_reason="Already imported",
                import_method="cli",
            )

        session_data = self._export_session_via_cli(session_id)
        if session_data is None:
            return OpenCodeImportResult(
                session_id=session_id,
                steps_imported=0,
                error="CLI export failed or returned no data",
                import_method="cli",
            )

        steps, session_meta = self.parse_session_from_cli(session_data, session_id)

        if not steps:
            return OpenCodeImportResult(
                session_id=session_id,
                steps_imported=0,
                skipped=True,
                skip_reason="No relevant tool executions found",
                import_method="cli",
            )

        # Create trace session
        user_initial_prompt = (
            session_meta.pop("user_initial_prompt", None)
            or "Imported OpenCode session"
        )
        session = TraceSession.from_steps(
            steps,
            source_tool="opencode",
            verification_passed=None,
            imported=True,
            import_source=f"opencode-cli:{session_id}",
            user_initial_prompt=user_initial_prompt,
            **session_meta,
        )

        # Save to traces directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imported_opencode_{session_id[:8]}_{timestamp}.json"
        destination = self.trace_capture.traces_dir / filename

        try:
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
        except IOError as e:
            return OpenCodeImportResult(
                session_id=session_id,
                steps_imported=0,
                error=f"Failed to write trace: {e}",
                import_method="cli",
            )

        # Mark as imported
        self._save_imported_session(session_id)

        return OpenCodeImportResult(
            session_id=session_id,
            steps_imported=len(steps),
            destination_file=destination,
            import_method="cli",
        )


def import_opencode_sessions(
    days: int = 60,
    limit: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    """
    Import recent OpenCode sessions.

    Tries CLI export first if opencode is installed, then falls back
    to direct file scanning.

    Args:
        days: Number of days to look back (default 60)
        limit: Maximum number of sessions to import
        verbose: Print progress messages

    Returns:
        List of import result dicts
    """
    importer = OpenCodeSessionImporter()
    results: List[Dict] = []
    imported_count = 0
    imported_ids: Set[str] = set()

    # ------------------------------------------------------------------
    # Strategy 1: Try CLI-based import first
    # ------------------------------------------------------------------
    if importer.opencode_available:
        if verbose:
            print("[BashGym] OpenCode CLI detected - trying CLI export")

        cli_sessions = importer._list_sessions_via_cli()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        for session_info in cli_sessions:
            if imported_count >= limit:
                break

            sid = session_info.get("id", session_info.get("session_id", ""))
            if not sid:
                continue

            # Check date if available
            created = session_info.get("created_at", session_info.get("timestamp", ""))
            if created:
                try:
                    if isinstance(created, (int, float)):
                        created_dt = datetime.fromtimestamp(created, tz=timezone.utc)
                    else:
                        created_dt = datetime.fromisoformat(
                            str(created).replace("Z", "+00:00")
                        )
                    if created_dt < cutoff:
                        continue
                except (ValueError, TypeError, OSError):
                    pass

            result = importer.import_session_from_cli(sid)

            result_dict = {
                "session_id": result.session_id,
                "source_file": str(result.source_file) if result.source_file else None,
                "steps_imported": result.steps_imported,
                "destination_file": str(result.destination_file) if result.destination_file else None,
                "error": result.error,
                "skipped": result.skipped,
                "skip_reason": result.skip_reason,
                "source_tool": "opencode",
                "import_method": result.import_method,
            }
            results.append(result_dict)

            if not result.skipped and not result.error:
                imported_count += 1
                imported_ids.add(sid)

            if verbose:
                if result.skipped:
                    print(f"  [-] {sid[:12]}: {result.skip_reason}")
                elif result.error:
                    print(f"  [!] {sid[:12]}: {result.error}")
                else:
                    print(
                        f"  [+] {sid[:12]}: "
                        f"{result.steps_imported} steps imported (CLI)"
                    )

    # ------------------------------------------------------------------
    # Strategy 2: File-based import (fallback or supplement)
    # ------------------------------------------------------------------
    storage_dir = importer._find_active_storage_dir()
    has_storage = storage_dir is not None

    if has_storage:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        session_files = importer.find_session_files(since=since)

        if verbose:
            method = "supplementing CLI" if importer.opencode_available else "scanning files"
            print(
                f"[BashGym] Found {len(session_files)} OpenCode session file(s) "
                f"from last {days} days ({method})"
            )

        for session_file, mtime in session_files:
            if imported_count >= limit:
                break

            # Skip if already imported via CLI
            if session_file.stem in imported_ids:
                continue

            result = importer.import_session(session_file)

            result_dict = {
                "session_id": result.session_id,
                "source_file": str(result.source_file) if result.source_file else None,
                "steps_imported": result.steps_imported,
                "destination_file": str(result.destination_file) if result.destination_file else None,
                "error": result.error,
                "skipped": result.skipped,
                "skip_reason": result.skip_reason,
                "source_tool": "opencode",
                "import_method": result.import_method,
            }
            results.append(result_dict)

            if not result.skipped and not result.error:
                imported_count += 1

            if verbose:
                if result.skipped:
                    print(f"  [-] {session_file.name}: {result.skip_reason}")
                elif result.error:
                    print(f"  [!] {session_file.name}: {result.error}")
                else:
                    print(
                        f"  [+] {session_file.name}: "
                        f"{result.steps_imported} steps imported (file)"
                    )

    if not has_storage and not importer.opencode_available:
        if verbose:
            print(
                "[BashGym] OpenCode not found - "
                "neither CLI nor storage directory detected"
            )

    return results
