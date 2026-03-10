"""
Gemini CLI Session History Importer

Imports tool execution traces from Gemini CLI's session JSON files
into BashGym's trace format.

Gemini CLI stores session transcripts at:
  Mac/Linux: ~/.gemini/tmp/<project_hash>/chats/session-*.json
  Windows:   C:\\Users\\<Username>\\.gemini\\tmp\\<project_hash>\\chats\\session-*.json

Each session file is a JSON array of conversation entries. Entries may
contain tool executions (with tool names, inputs, and outputs) as well
as the assistant's reasoning and text responses.
"""

import json
import os
import platform
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict

from ..core import TraceStep, TraceSession, TraceCapture


@dataclass
class GeminiImportResult:
    """Result of a Gemini session import operation."""
    session_id: str
    source_file: Path
    steps_imported: int
    destination_file: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class GeminiSessionImporter:
    """
    Import traces from Gemini CLI session files.

    Gemini CLI stores session transcripts as JSON files in:
    ~/.gemini/tmp/<project_hash>/chats/session-*.json
    """

    def __init__(self):
        self.trace_capture = TraceCapture()
        self.gemini_dir = self._get_gemini_dir()
        self.imported_sessions_file = (
            self.trace_capture.bashgym_dir / "imported_gemini_sessions.json"
        )
        self._imported_sessions: Optional[Set[str]] = None

    @staticmethod
    def _get_gemini_dir() -> Path:
        """Get Gemini CLI's data directory."""
        if platform.system() == "Windows":
            base = Path(os.environ.get("USERPROFILE", ""))
        else:
            base = Path.home()
        return base / ".gemini"

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
            print(f"Warning: Could not save imported Gemini sessions list: {e}")

    def find_session_files(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Tuple[Path, datetime]]:
        """
        Find Gemini CLI session JSON files, optionally filtered by date.

        Scans all project hash directories under ~/.gemini/tmp/ for
        chats/session-*.json files.

        Args:
            since: Only include sessions modified after this time
            until: Only include sessions modified before this time

        Returns:
            List of (file_path, modified_time) tuples sorted newest first
        """
        tmp_dir = self.gemini_dir / "tmp"
        if not tmp_dir.exists():
            return []

        session_files: List[Tuple[Path, datetime]] = []

        try:
            for project_dir in tmp_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                chats_dir = project_dir / "chats"
                if not chats_dir.exists():
                    continue

                for session_file in chats_dir.glob("session-*.json"):
                    try:
                        mtime = datetime.fromtimestamp(
                            session_file.stat().st_mtime, tz=timezone.utc
                        )
                    except OSError:
                        continue

                    if since and mtime < since:
                        continue
                    if until and mtime > until:
                        continue

                    session_files.append((session_file, mtime))
        except (PermissionError, OSError):
            pass

        session_files.sort(key=lambda x: x[1], reverse=True)
        return session_files

    @staticmethod
    def _extract_tool_steps_from_entry(
        entry: Dict[str, Any],
        session_id: str,
        session_file: Path,
        step_index: int,
    ) -> List[TraceStep]:
        """
        Extract TraceStep objects from a single conversation entry.

        Gemini CLI entries vary in structure. This method defensively
        handles several known formats:

        Format A (function call / response pairs):
          {"role": "model", "parts": [{"functionCall": {"name": ..., "args": {...}}}]}
          {"role": "tool", "parts": [{"functionResponse": {"name": ..., "response": {...}}}]}

        Format B (tool_code execution):
          {"role": "model", "parts": [{"executableCode": {"code": "...", "language": "..."}}]}
          followed by {"role": "model", "parts": [{"codeExecutionResult": {"output": "...", "outcome": "..."}}]}

        Format C (inline tool metadata):
          {"tool_name": "...", "tool_input": {...}, "tool_output": "..."}
        """
        steps: List[TraceStep] = []
        timestamp = entry.get(
            "timestamp",
            entry.get("createTime", datetime.now(timezone.utc).isoformat()),
        )

        # Normalize timestamp - Gemini sometimes uses epoch seconds
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

        # --- Format C: flat tool metadata fields ---
        tool_name = entry.get("tool_name")
        if tool_name:
            tool_input = entry.get("tool_input", {})
            tool_output = entry.get("tool_output", "")
            command = (
                json.dumps(tool_input)
                if isinstance(tool_input, dict)
                else str(tool_input)
            )
            output = str(tool_output)[:10000] if tool_output else ""

            step = TraceStep(
                step_id=f"gemini_{session_id}_{step_index}",
                timestamp=timestamp,
                tool_name=tool_name,
                command=command,
                output=output,
                exit_code=0 if output else None,
                success=True if output else None,
                cwd="",
                repo={"path": "", "name": "unknown"},
                source_tool="gemini_cli",
                metadata={
                    "session_id": session_id,
                    "imported_from": str(session_file),
                },
            )
            steps.append(step)
            return steps

        # --- Format A & B: parts-based entries ---
        parts = entry.get("parts", [])
        if not isinstance(parts, list):
            return steps

        for part_idx, part in enumerate(parts):
            if not isinstance(part, dict):
                continue

            # Format A: functionCall
            func_call = part.get("functionCall")
            if isinstance(func_call, dict):
                fname = func_call.get("name", "unknown_function")
                fargs = func_call.get("args", {})
                command = json.dumps(fargs) if isinstance(fargs, dict) else str(fargs)

                step = TraceStep(
                    step_id=f"gemini_{session_id}_{step_index}_{part_idx}",
                    timestamp=timestamp,
                    tool_name=fname,
                    command=command,
                    output="",  # Output comes in a later functionResponse entry
                    exit_code=None,
                    success=None,
                    cwd="",
                    repo={"path": "", "name": "unknown"},
                    source_tool="gemini_cli",
                    metadata={
                        "session_id": session_id,
                        "imported_from": str(session_file),
                        "call_type": "functionCall",
                    },
                )
                steps.append(step)

            # Format A: functionResponse (matches back to a functionCall step)
            func_resp = part.get("functionResponse")
            if isinstance(func_resp, dict):
                fname = func_resp.get("name", "unknown_function")
                response = func_resp.get("response", {})
                output_str = (
                    json.dumps(response)
                    if isinstance(response, dict)
                    else str(response)
                )[:10000]

                # Try to match to an existing step with the same function name
                # that doesn't have output yet
                matched = False
                for existing_step in reversed(steps):
                    if (
                        existing_step.tool_name == fname
                        and not existing_step.output
                    ):
                        existing_step.output = output_str
                        existing_step.success = True
                        existing_step.exit_code = 0
                        matched = True
                        break

                if not matched:
                    # Create a standalone step for unmatched responses
                    step = TraceStep(
                        step_id=f"gemini_{session_id}_{step_index}_{part_idx}_resp",
                        timestamp=timestamp,
                        tool_name=fname,
                        command="",
                        output=output_str,
                        exit_code=0,
                        success=True,
                        cwd="",
                        repo={"path": "", "name": "unknown"},
                        source_tool="gemini_cli",
                        metadata={
                            "session_id": session_id,
                            "imported_from": str(session_file),
                            "call_type": "functionResponse",
                        },
                    )
                    steps.append(step)

            # Format B: executableCode
            exec_code = part.get("executableCode")
            if isinstance(exec_code, dict):
                code = exec_code.get("code", "")
                language = exec_code.get("language", "PYTHON")

                step = TraceStep(
                    step_id=f"gemini_{session_id}_{step_index}_{part_idx}_exec",
                    timestamp=timestamp,
                    tool_name="code_execution",
                    command=code[:10000],
                    output="",
                    exit_code=None,
                    success=None,
                    cwd="",
                    repo={"path": "", "name": "unknown"},
                    source_tool="gemini_cli",
                    metadata={
                        "session_id": session_id,
                        "imported_from": str(session_file),
                        "language": language,
                        "call_type": "executableCode",
                    },
                )
                steps.append(step)

            # Format B: codeExecutionResult
            code_result = part.get("codeExecutionResult")
            if isinstance(code_result, dict):
                output_text = code_result.get("output", "")[:10000]
                outcome = code_result.get("outcome", "")
                is_success = outcome in ("OUTCOME_OK", "OK", "SUCCESS", "")

                # Match to the most recent code_execution step without output
                for existing_step in reversed(steps):
                    if (
                        existing_step.tool_name == "code_execution"
                        and not existing_step.output
                    ):
                        existing_step.output = output_text
                        existing_step.success = is_success
                        existing_step.exit_code = 0 if is_success else 1
                        break

        return steps

    def parse_session_file(
        self, session_file: Path
    ) -> Tuple[List[TraceStep], Dict[str, Any]]:
        """
        Parse a Gemini CLI session JSON file and extract tool steps.

        Args:
            session_file: Path to the session-*.json file

        Returns:
            Tuple of (steps, session_metadata)
        """
        steps: List[TraceStep] = []
        session_id = session_file.stem
        models_used: Set[str] = set()
        meta: Dict[str, Any] = {
            "user_initial_prompt": None,
            "all_user_prompts": [],
            "conversation_turns": 0,
        }

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading Gemini session file {session_file}: {e}")
            return [], {"user_initial_prompt": None}

        # Session data may be a list of entries or a dict with a conversation key
        entries: List[Dict[str, Any]] = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            # Try common keys where conversation entries may live
            for key in ("messages", "conversation", "entries", "history"):
                if key in data and isinstance(data[key], list):
                    entries = data[key]
                    break
            if not entries:
                # The whole dict might be a single entry
                entries = [data]
        else:
            return [], {"user_initial_prompt": None}

        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            try:
                role = entry.get("role", "")

                # Capture user prompts
                if role == "user":
                    meta["conversation_turns"] += 1
                    parts = entry.get("parts", [])
                    user_text = ""
                    if isinstance(parts, list):
                        for part in parts:
                            if isinstance(part, dict) and "text" in part:
                                user_text += part["text"]
                            elif isinstance(part, str):
                                user_text += part
                    elif isinstance(parts, str):
                        user_text = parts

                    # Also check for top-level "content" or "text" fields
                    if not user_text:
                        user_text = entry.get("content", entry.get("text", ""))
                        if isinstance(user_text, list):
                            user_text = " ".join(
                                str(x) for x in user_text if x
                            )

                    if user_text:
                        meta["all_user_prompts"].append({
                            "text": str(user_text)[:2000],
                            "timestamp": entry.get("timestamp"),
                        })
                        if meta["user_initial_prompt"] is None:
                            meta["user_initial_prompt"] = str(user_text)[:500]

                # Capture model version from model turns
                model_version = entry.get("modelVersion")
                if model_version and isinstance(model_version, str):
                    models_used.add(model_version)

                # Extract tool steps from any entry
                entry_steps = self._extract_tool_steps_from_entry(
                    entry, session_id, session_file, idx
                )
                steps.extend(entry_steps)

            except Exception:
                # Defensive: skip malformed entries
                continue

        # Finalize session metadata
        meta["models_used"] = sorted(models_used)

        return steps, meta

    def import_session(
        self, session_file: Path, force: bool = False
    ) -> GeminiImportResult:
        """
        Import a single Gemini session file into BashGym format.

        Args:
            session_file: Path to the session JSON file
            force: Import even if already imported

        Returns:
            GeminiImportResult with import details
        """
        session_id = session_file.stem

        # Check if already imported
        if not force and session_id in self.imported_sessions:
            return GeminiImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="Already imported",
            )

        # Parse the session
        steps, session_meta = self.parse_session_file(session_file)

        if not steps:
            return GeminiImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="No relevant tool executions found",
            )

        # Create trace session
        user_initial_prompt = (
            session_meta.pop("user_initial_prompt", None) or "Imported Gemini session"
        )
        session = TraceSession.from_steps(
            steps,
            source_tool="gemini_cli",
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
        filename = f"imported_gemini_{session_id[:8]}_{timestamp}.json"
        destination = self.trace_capture.traces_dir / filename

        try:
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
        except IOError as e:
            return GeminiImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                error=f"Failed to write trace: {e}",
            )

        # Mark as imported
        self._save_imported_session(session_id)

        return GeminiImportResult(
            session_id=session_id,
            source_file=session_file,
            steps_imported=len(steps),
            destination_file=destination,
        )


def import_gemini_sessions(
    days: int = 60,
    limit: int = 100,
    verbose: bool = True,
) -> List[Dict]:
    """
    Import recent Gemini CLI sessions.

    Args:
        days: Number of days to look back (default 60)
        limit: Maximum number of sessions to import
        verbose: Print progress messages

    Returns:
        List of import result dicts
    """
    importer = GeminiSessionImporter()

    if not importer.gemini_dir.exists():
        if verbose:
            print("[BashGym] Gemini CLI directory not found - skipping")
        return []

    since = datetime.now(timezone.utc) - timedelta(days=days)
    session_files = importer.find_session_files(since=since)

    if verbose:
        print(
            f"[BashGym] Found {len(session_files)} Gemini session(s) "
            f"from last {days} days"
        )

    results: List[Dict] = []
    imported_count = 0

    for session_file, mtime in session_files:
        if imported_count >= limit:
            break

        result = importer.import_session(session_file)

        result_dict = {
            "session_id": result.session_id,
            "source_file": str(result.source_file),
            "steps_imported": result.steps_imported,
            "destination_file": str(result.destination_file) if result.destination_file else None,
            "error": result.error,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "source_tool": "gemini_cli",
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
                    f"{result.steps_imported} steps imported"
                )

    return results
