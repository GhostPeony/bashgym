"""
Copilot CLI Session History Importer

Imports tool execution traces from GitHub Copilot CLI's session state
files into BashGym's trace format.

Copilot CLI stores session state at:
  Primary:  ~/.copilot/session-state/*.json
  Legacy:   ~/.copilot/history-session-state/*.json

Key insight: Copilot requires user approval before running commands,
creating accept/reject pairs that are valuable for DPO training.
Each proposed command may be accepted, rejected, or corrected by the user.
"""

import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..core import TraceCapture, TraceSession, TraceStep


@dataclass
class CopilotImportResult:
    """Result of a Copilot session import operation."""

    session_id: str
    source_file: Path
    steps_imported: int
    accepted_commands: int = 0
    rejected_commands: int = 0
    destination_file: Path | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class CopilotSessionImporter:
    """
    Import traces from Copilot CLI session state files.

    Copilot CLI stores session transcripts as JSON files in:
    ~/.copilot/session-state/*.json (current)
    ~/.copilot/history-session-state/*.json (legacy)
    """

    def __init__(self):
        self.trace_capture = TraceCapture()
        self.copilot_dir = self._get_copilot_dir()
        self.imported_sessions_file = (
            self.trace_capture.bashgym_dir / "imported_copilot_sessions.json"
        )
        self._imported_sessions: set[str] | None = None

    @staticmethod
    def _get_copilot_dir() -> Path:
        """Get Copilot CLI's data directory."""
        if platform.system() == "Windows":
            base = Path(os.environ.get("USERPROFILE", ""))
        else:
            base = Path.home()
        return base / ".copilot"

    @property
    def imported_sessions(self) -> set[str]:
        """Get set of already imported session IDs."""
        if self._imported_sessions is None:
            self._imported_sessions = self._load_imported_sessions()
        return self._imported_sessions

    def _load_imported_sessions(self) -> set[str]:
        """Load the list of already imported session IDs."""
        if not self.imported_sessions_file.exists():
            return set()
        try:
            with open(self.imported_sessions_file) as f:
                data = json.load(f)
                return set(data.get("sessions", []))
        except (OSError, json.JSONDecodeError):
            return set()

    def _save_imported_session(self, session_id: str) -> None:
        """Mark a session as imported."""
        self.imported_sessions.add(session_id)
        try:
            self.imported_sessions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.imported_sessions_file, "w") as f:
                json.dump({"sessions": list(self.imported_sessions)}, f)
        except OSError as e:
            print(f"Warning: Could not save imported Copilot sessions list: {e}")

    def _get_session_dirs(self) -> list[Path]:
        """Get all directories that may contain Copilot session files."""
        dirs = []
        for subdir in ("session-state", "history-session-state"):
            candidate = self.copilot_dir / subdir
            if candidate.exists() and candidate.is_dir():
                dirs.append(candidate)
        return dirs

    def find_session_files(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[tuple[Path, datetime]]:
        """
        Find Copilot CLI session JSON files, optionally filtered by date.

        Args:
            since: Only include sessions modified after this time
            until: Only include sessions modified before this time

        Returns:
            List of (file_path, modified_time) tuples sorted newest first
        """
        session_files: list[tuple[Path, datetime]] = []

        for session_dir in self._get_session_dirs():
            try:
                for json_file in session_dir.glob("*.json"):
                    try:
                        mtime = datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)
                    except OSError:
                        continue

                    if since and mtime < since:
                        continue
                    if until and mtime > until:
                        continue

                    session_files.append((json_file, mtime))
            except (PermissionError, OSError):
                continue

        session_files.sort(key=lambda x: x[1], reverse=True)
        return session_files

    @staticmethod
    def _extract_steps_from_session(
        data: dict[str, Any],
        session_id: str,
        session_file: Path,
    ) -> tuple[list[TraceStep], dict[str, Any]]:
        """
        Extract TraceStep objects and metadata from a Copilot session dict.

        Copilot session formats vary. This handles several known patterns:

        Pattern A - conversation with suggestions:
          {"messages": [...], "suggestions": [{"command": "...", "accepted": true/false}]}

        Pattern B - turn-based with tool results:
          {"turns": [{"role": "...", "content": "...", "tool_calls": [...]}]}

        Pattern C - flat command history:
          {"commands": [{"proposed": "...", "accepted": true, "executed": "...", "output": "..."}]}

        In all patterns, we look for accept/reject metadata for DPO training.
        """
        steps: list[TraceStep] = []
        models_seen: set[str] = set()
        meta: dict[str, Any] = {
            "user_initial_prompt": None,
            "all_user_prompts": [],
            "conversation_turns": 0,
            "accepted_commands": 0,
            "rejected_commands": 0,
            "models_used": [],
        }

        # Session-level model field
        session_model = data.get("model", data.get("model_name", ""))
        if session_model and isinstance(session_model, str):
            models_seen.add(session_model)

        timestamp_base = data.get(
            "timestamp",
            data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        if isinstance(timestamp_base, (int, float)):
            timestamp_base = datetime.fromtimestamp(timestamp_base, tz=timezone.utc).isoformat()

        # --- Pattern A: messages + suggestions ---
        messages = data.get("messages", [])
        if isinstance(messages, list):
            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role", "")
                content = msg.get("content", "")
                ts = msg.get("timestamp", timestamp_base)
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

                # Extract model name from message
                msg_model = msg.get("model", msg.get("model_name", ""))
                if msg_model and isinstance(msg_model, str):
                    models_seen.add(msg_model)

                if role == "user" and content:
                    content_str = content if isinstance(content, str) else str(content)
                    meta["conversation_turns"] += 1
                    meta["all_user_prompts"].append(
                        {
                            "text": content_str[:2000],
                            "timestamp": ts,
                        }
                    )
                    if meta["user_initial_prompt"] is None:
                        meta["user_initial_prompt"] = content_str[:500]

                # Extract tool_calls from assistant messages
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tc_idx, tc in enumerate(tool_calls):
                        if not isinstance(tc, dict):
                            continue

                        tool_name = tc.get(
                            "name",
                            tc.get("function", {}).get("name", "unknown"),
                        )
                        tool_args = tc.get(
                            "arguments",
                            tc.get("function", {}).get("arguments", {}),
                        )
                        tool_output = tc.get("output", tc.get("result", ""))

                        command = (
                            json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
                        )
                        output = str(tool_output)[:10000] if tool_output else ""

                        step = TraceStep(
                            step_id=f"copilot_{session_id}_{msg_idx}_{tc_idx}",
                            timestamp=ts,
                            tool_name=tool_name,
                            command=command,
                            output=output,
                            exit_code=0 if output else None,
                            success=True if output else None,
                            cwd="",
                            repo={"path": "", "name": "unknown"},
                            source_tool="copilot_cli",
                            metadata={
                                "session_id": session_id,
                                "imported_from": str(session_file),
                            },
                        )
                        steps.append(step)

        # --- Pattern A: suggestions (accept/reject pairs) ---
        suggestions = data.get("suggestions", [])
        if isinstance(suggestions, list):
            for sug_idx, suggestion in enumerate(suggestions):
                if not isinstance(suggestion, dict):
                    continue

                proposed = suggestion.get("command", suggestion.get("proposed", ""))
                sug_model = suggestion.get("model", suggestion.get("model_name", ""))
                if sug_model and isinstance(sug_model, str):
                    models_seen.add(sug_model)
                accepted = suggestion.get("accepted", None)
                executed = suggestion.get("executed", "")
                output = suggestion.get("output", "")[:10000]
                user_correction = suggestion.get(
                    "correction", suggestion.get("user_correction", "")
                )

                if not proposed:
                    continue

                user_accepted = bool(accepted) if accepted is not None else None
                if user_accepted is True:
                    meta["accepted_commands"] += 1
                elif user_accepted is False:
                    meta["rejected_commands"] += 1

                step = TraceStep(
                    step_id=f"copilot_{session_id}_sug_{sug_idx}",
                    timestamp=timestamp_base,
                    tool_name="command_suggestion",
                    command=str(proposed),
                    output=str(output) if output else "",
                    exit_code=0 if (user_accepted and output) else None,
                    success=user_accepted,
                    cwd="",
                    repo={"path": "", "name": "unknown"},
                    source_tool="copilot_cli",
                    metadata={
                        "session_id": session_id,
                        "imported_from": str(session_file),
                        "user_accepted": user_accepted,
                        "user_correction": str(user_correction) if user_correction else "",
                        "proposed_command": str(proposed),
                        "executed_command": str(executed) if executed else "",
                    },
                )
                steps.append(step)

        # --- Pattern B: turns-based format ---
        turns = data.get("turns", [])
        if isinstance(turns, list):
            for turn_idx, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue

                role = turn.get("role", "")
                content = turn.get("content", "")
                ts = turn.get("timestamp", timestamp_base)
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

                # Extract model name from turn
                turn_model = turn.get("model", turn.get("model_name", ""))
                if turn_model and isinstance(turn_model, str):
                    models_seen.add(turn_model)

                if role == "user" and content:
                    content_str = content if isinstance(content, str) else str(content)
                    meta["conversation_turns"] += 1
                    meta["all_user_prompts"].append(
                        {
                            "text": content_str[:2000],
                            "timestamp": ts,
                        }
                    )
                    if meta["user_initial_prompt"] is None:
                        meta["user_initial_prompt"] = content_str[:500]

                tool_calls = turn.get("tool_calls", turn.get("tools", []))
                if isinstance(tool_calls, list):
                    for tc_idx, tc in enumerate(tool_calls):
                        if not isinstance(tc, dict):
                            continue

                        tool_name = tc.get("name", tc.get("type", "unknown"))
                        tool_input = tc.get("input", tc.get("arguments", {}))
                        tool_output = tc.get("output", tc.get("result", ""))

                        command = (
                            json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input)
                        )

                        step = TraceStep(
                            step_id=f"copilot_{session_id}_turn_{turn_idx}_{tc_idx}",
                            timestamp=ts,
                            tool_name=tool_name,
                            command=command,
                            output=str(tool_output)[:10000] if tool_output else "",
                            exit_code=0 if tool_output else None,
                            success=True if tool_output else None,
                            cwd="",
                            repo={"path": "", "name": "unknown"},
                            source_tool="copilot_cli",
                            metadata={
                                "session_id": session_id,
                                "imported_from": str(session_file),
                            },
                        )
                        steps.append(step)

        # --- Pattern C: flat command history ---
        commands = data.get("commands", [])
        if isinstance(commands, list):
            for cmd_idx, cmd in enumerate(commands):
                if not isinstance(cmd, dict):
                    continue

                proposed = cmd.get("proposed", cmd.get("command", ""))
                cmd_model = cmd.get("model", cmd.get("model_name", ""))
                if cmd_model and isinstance(cmd_model, str):
                    models_seen.add(cmd_model)
                accepted = cmd.get("accepted", None)
                executed = cmd.get("executed", proposed)
                output = cmd.get("output", "")[:10000]
                user_correction = cmd.get("correction", "")

                if not proposed and not executed:
                    continue

                user_accepted = bool(accepted) if accepted is not None else None
                if user_accepted is True:
                    meta["accepted_commands"] += 1
                elif user_accepted is False:
                    meta["rejected_commands"] += 1

                ts = cmd.get("timestamp", timestamp_base)
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

                step = TraceStep(
                    step_id=f"copilot_{session_id}_cmd_{cmd_idx}",
                    timestamp=ts,
                    tool_name="bash",
                    command=str(executed or proposed),
                    output=str(output) if output else "",
                    exit_code=cmd.get("exit_code", 0 if output else None),
                    success=user_accepted if user_accepted is not None else (bool(output)),
                    cwd=cmd.get("cwd", ""),
                    repo={"path": cmd.get("cwd", ""), "name": "unknown"},
                    source_tool="copilot_cli",
                    metadata={
                        "session_id": session_id,
                        "imported_from": str(session_file),
                        "user_accepted": user_accepted,
                        "user_correction": str(user_correction) if user_correction else "",
                        "proposed_command": str(proposed),
                        "executed_command": str(executed) if executed else "",
                    },
                )
                steps.append(step)

        meta["models_used"] = sorted(models_seen)

        return steps, meta

    def parse_session_file(self, session_file: Path) -> tuple[list[TraceStep], dict[str, Any]]:
        """
        Parse a Copilot CLI session JSON file and extract tool steps.

        Args:
            session_file: Path to the session JSON file

        Returns:
            Tuple of (steps, session_metadata)
        """
        session_id = session_file.stem

        try:
            with open(session_file, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error reading Copilot session file {session_file}: {e}")
            return [], {"user_initial_prompt": None}

        if not isinstance(data, dict):
            # Try to handle if it is a list (array of sessions)
            if isinstance(data, list) and data:
                all_steps: list[TraceStep] = []
                combined_meta: dict[str, Any] = {
                    "user_initial_prompt": None,
                    "all_user_prompts": [],
                    "conversation_turns": 0,
                    "accepted_commands": 0,
                    "rejected_commands": 0,
                    "models_used": [],
                }
                all_models: set[str] = set()
                for entry in data:
                    if isinstance(entry, dict):
                        s, m = self._extract_steps_from_session(entry, session_id, session_file)
                        all_steps.extend(s)
                        if (
                            m.get("user_initial_prompt")
                            and not combined_meta["user_initial_prompt"]
                        ):
                            combined_meta["user_initial_prompt"] = m["user_initial_prompt"]
                        combined_meta["all_user_prompts"].extend(m.get("all_user_prompts", []))
                        combined_meta["conversation_turns"] += m.get("conversation_turns", 0)
                        combined_meta["accepted_commands"] += m.get("accepted_commands", 0)
                        combined_meta["rejected_commands"] += m.get("rejected_commands", 0)
                        all_models.update(m.get("models_used", []))
                combined_meta["models_used"] = sorted(all_models)
                return all_steps, combined_meta
            return [], {"user_initial_prompt": None}

        return self._extract_steps_from_session(data, session_id, session_file)

    def import_session(self, session_file: Path, force: bool = False) -> CopilotImportResult:
        """
        Import a single Copilot session file into BashGym format.

        Args:
            session_file: Path to the session JSON file
            force: Import even if already imported

        Returns:
            CopilotImportResult with import details
        """
        session_id = session_file.stem

        # Check if already imported
        if not force and session_id in self.imported_sessions:
            return CopilotImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="Already imported",
            )

        # Parse the session
        steps, session_meta = self.parse_session_file(session_file)

        if not steps:
            return CopilotImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="No relevant tool executions found",
            )

        accepted = session_meta.pop("accepted_commands", 0)
        rejected = session_meta.pop("rejected_commands", 0)

        # Create trace session
        user_initial_prompt = (
            session_meta.pop("user_initial_prompt", None) or "Imported Copilot session"
        )
        session = TraceSession.from_steps(
            steps,
            source_tool="copilot_cli",
            verification_passed=None,
            imported=True,
            import_source=str(session_file),
            user_initial_prompt=user_initial_prompt,
            accepted_commands=accepted,
            rejected_commands=rejected,
            **session_meta,
        )

        # Save to traces directory
        try:
            mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
        except OSError:
            mtime = datetime.now()
        timestamp = mtime.strftime("%Y%m%d_%H%M%S")
        filename = f"imported_copilot_{session_id[:8]}_{timestamp}.json"
        destination = self.trace_capture.traces_dir / filename

        try:
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
        except OSError as e:
            return CopilotImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                error=f"Failed to write trace: {e}",
            )

        # Mark as imported
        self._save_imported_session(session_id)

        return CopilotImportResult(
            session_id=session_id,
            source_file=session_file,
            steps_imported=len(steps),
            accepted_commands=accepted,
            rejected_commands=rejected,
            destination_file=destination,
        )


def import_copilot_sessions(
    days: int = 60,
    limit: int = 100,
    verbose: bool = True,
) -> list[dict]:
    """
    Import recent Copilot CLI sessions.

    Args:
        days: Number of days to look back (default 60)
        limit: Maximum number of sessions to import
        verbose: Print progress messages

    Returns:
        List of import result dicts
    """
    importer = CopilotSessionImporter()

    if not importer.copilot_dir.exists():
        if verbose:
            print("[BashGym] Copilot CLI directory not found - skipping")
        return []

    since = datetime.now(timezone.utc) - timedelta(days=days)
    session_files = importer.find_session_files(since=since)

    if verbose:
        print(f"[BashGym] Found {len(session_files)} Copilot session(s) " f"from last {days} days")

    results: list[dict] = []
    imported_count = 0

    for session_file, mtime in session_files:
        if imported_count >= limit:
            break

        result = importer.import_session(session_file)

        result_dict = {
            "session_id": result.session_id,
            "source_file": str(result.source_file),
            "steps_imported": result.steps_imported,
            "accepted_commands": result.accepted_commands,
            "rejected_commands": result.rejected_commands,
            "destination_file": str(result.destination_file) if result.destination_file else None,
            "error": result.error,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
            "source_tool": "copilot_cli",
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
                dpo_note = ""
                if result.accepted_commands or result.rejected_commands:
                    dpo_note = (
                        f" (accepted: {result.accepted_commands}, "
                        f"rejected: {result.rejected_commands})"
                    )
                print(
                    f"  [+] {session_file.name}: "
                    f"{result.steps_imported} steps imported{dpo_note}"
                )

    return results
