"""
Claude Code Session History Importer

Imports tool execution traces from Claude Code's session JSONL files
into BashGym's trace format.

Claude Code stores session transcripts at:
  ~/.claude/projects/<project-slug>/<session-id>.jsonl

Each line is a JSON object with types like:
  - "user" / "assistant" - conversation messages
  - "progress" - tool execution progress
  - "tool_use" embedded in assistant messages
  - "tool_result" embedded in progress messages

Instrumentation:
  - PII filtering on user prompts and tool outputs
  - Injection detection on user prompts
  - Profiling spans for import operations
"""

import json
import os
import platform
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict, field

from ..core import TraceStep, TraceSession, TraceCapture

# Import instrumentation (optional - graceful degradation if not available)
try:
    from bashgym.core import get_instrumentation, Instrumentation
    HAS_INSTRUMENTATION = True
except ImportError:
    HAS_INSTRUMENTATION = False
    Instrumentation = None


@dataclass
class ImportResult:
    """Result of an import operation."""
    session_id: str
    source_file: Path
    steps_imported: int
    destination_file: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    # Instrumentation stats
    pii_redactions: int = 0
    injection_blocked: bool = False
    sanitized_prompt: Optional[str] = None


class ClaudeSessionImporter:
    """
    Import traces from Claude Code session files.

    Claude Code stores session transcripts as JSONL files in:
    ~/.claude/projects/<project-slug>/<session-id>.jsonl
    """

    RELEVANT_TOOLS = {"Bash", "Edit", "Write", "Read", "Grep", "Glob"}

    def __init__(self):
        self.trace_capture = TraceCapture()
        self.claude_dir = self._get_claude_dir()
        self.imported_sessions_file = self.trace_capture.bashgym_dir / "imported_sessions.json"
        self._imported_sessions: Optional[Set[str]] = None

    @staticmethod
    def _get_claude_dir() -> Path:
        """Get Claude Code's data directory."""
        if platform.system() == 'Windows':
            base = Path(os.environ.get("USERPROFILE", ""))
        else:
            base = Path.home()
        return base / ".claude"

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
            with open(self.imported_sessions_file, 'r') as f:
                data = json.load(f)
                return set(data.get("sessions", []))
        except (json.JSONDecodeError, IOError):
            return set()

    def _save_imported_session(self, session_id: str) -> None:
        """Mark a session as imported."""
        self.imported_sessions.add(session_id)
        try:
            with open(self.imported_sessions_file, 'w') as f:
                json.dump({"sessions": list(self.imported_sessions)}, f)
        except IOError as e:
            print(f"Warning: Could not save imported sessions list: {e}")

    def find_projects_dir(self) -> Optional[Path]:
        """Find Claude Code's projects directory."""
        projects_dir = self.claude_dir / "projects"
        return projects_dir if projects_dir.exists() else None

    def find_session_files(
        self,
        project_filter: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[Tuple[Path, datetime]]:
        """
        Find session JSONL files, optionally filtered by date and project.

        Args:
            project_filter: Only include sessions from projects matching this substring
            since: Only include sessions modified after this time
            until: Only include sessions modified before this time

        Returns:
            List of (file_path, modified_time) tuples sorted by modified time (newest first)
        """
        projects_dir = self.find_projects_dir()
        if not projects_dir:
            return []

        session_files = []

        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Apply project filter
            if project_filter and project_filter.lower() not in project_dir.name.lower():
                continue

            for jsonl_file in project_dir.glob("*.jsonl"):
                # Get file modification time
                mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime, tz=timezone.utc)

                # Apply date filters
                if since and mtime < since:
                    continue
                if until and mtime > until:
                    continue

                session_files.append((jsonl_file, mtime))

        # Sort by modification time, newest first
        session_files.sort(key=lambda x: x[1], reverse=True)
        return session_files

    def parse_session_file(self, session_file: Path) -> Tuple[List[TraceStep], Optional[str]]:
        """
        Parse a Claude Code session JSONL file and extract tool execution steps.

        Args:
            session_file: Path to the .jsonl session file

        Returns:
            Tuple of (List of TraceStep objects, user's initial prompt or None)
        """
        steps = []
        session_id = session_file.stem
        cwd = None
        user_initial_prompt = None

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Track working directory
                    if "cwd" in event:
                        cwd = event["cwd"]

                    # Extract user's initial prompt (first user message)
                    if user_initial_prompt is None and event.get("type") == "user":
                        message = event.get("message", {})
                        content = message.get("content", [])
                        # Content can be a string or a list of content blocks
                        if isinstance(content, str):
                            user_initial_prompt = content[:500]  # Limit length
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    user_initial_prompt = item.get("text", "")[:500]
                                    break
                                elif isinstance(item, str):
                                    user_initial_prompt = item[:500]
                                    break

                    # Extract tool use from assistant messages
                    if event.get("type") == "assistant":
                        message = event.get("message", {})
                        content = message.get("content", [])
                        timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())

                        for item in content:
                            if item.get("type") == "tool_use":
                                tool_name = item.get("name", "")
                                tool_input = item.get("input", {})
                                tool_id = item.get("id", "")

                                if tool_name in self.RELEVANT_TOOLS:
                                    step = TraceStep(
                                        step_id=f"{session_id}_{tool_id}",
                                        timestamp=timestamp,
                                        tool_name=tool_name,
                                        command=json.dumps(tool_input),
                                        output="",  # Output comes in tool_result
                                        exit_code=None,
                                        success=None,
                                        cwd=cwd or "",
                                        repo={"path": cwd, "name": Path(cwd).name if cwd else "unknown"},
                                        source_tool="claude_code",
                                        metadata={
                                            "tool_use_id": tool_id,
                                            "session_id": session_id,
                                            "imported_from": str(session_file)
                                        }
                                    )
                                    steps.append(step)

                    # Extract tool results from user events (this is where Claude stores them)
                    elif event.get("type") == "user":
                        message = event.get("message", {})
                        content = message.get("content", [])

                        for item in content if isinstance(content, list) else []:
                            if item.get("type") == "tool_result":
                                tool_use_id = item.get("tool_use_id", "")
                                result_content = item.get("content", "")
                                is_error = item.get("is_error", False)

                                # Find matching step and update it
                                for step in steps:
                                    if step.metadata.get("tool_use_id") == tool_use_id:
                                        # Truncate large outputs
                                        if isinstance(result_content, str):
                                            step.output = result_content[:10000]
                                        elif isinstance(result_content, list):
                                            # Content can be a list of blocks
                                            text_parts = []
                                            for block in result_content:
                                                if isinstance(block, dict) and block.get("type") == "text":
                                                    text_parts.append(block.get("text", ""))
                                                elif isinstance(block, str):
                                                    text_parts.append(block)
                                            step.output = "\n".join(text_parts)[:10000]
                                        # Default to success if is_error not present
                                        step.success = not is_error
                                        step.exit_code = 1 if is_error else 0
                                        break

                    # Also check progress events for agent_progress (sub-agent results)
                    elif event.get("type") == "progress":
                        data = event.get("data", {})
                        if data.get("type") == "agent_progress":
                            msg = data.get("message", {})
                            msg_content = msg.get("message", {}).get("content", [])

                            for item in msg_content if isinstance(msg_content, list) else []:
                                if item.get("type") == "tool_result":
                                    tool_use_id = item.get("tool_use_id", "")
                                    result_content = item.get("content", "")
                                    is_error = item.get("is_error", False)

                                    for step in steps:
                                        if step.metadata.get("tool_use_id") == tool_use_id:
                                            if isinstance(result_content, str):
                                                step.output = result_content[:10000]
                                            step.success = not is_error
                                            step.exit_code = 1 if is_error else 0
                                            break

        except IOError as e:
            print(f"Error reading session file {session_file}: {e}")
            return [], None

        return steps, user_initial_prompt

    def import_session(
        self,
        session_file: Path,
        force: bool = False
    ) -> ImportResult:
        """
        Import a single session file into BashGym format.

        Args:
            session_file: Path to the .jsonl session file
            force: Import even if already imported

        Returns:
            ImportResult with import details
        """
        session_id = session_file.stem

        # Check if already imported
        if not force and session_id in self.imported_sessions:
            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="Already imported"
            )

        # Parse the session
        steps, user_initial_prompt = self.parse_session_file(session_file)

        if not steps:
            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                skipped=True,
                skip_reason="No relevant tool executions found"
            )

        # Create a trace session with user's initial prompt
        session = TraceSession.from_steps(
            steps,
            source_tool="claude_code",
            verification_passed=True,  # Assume successful for imports
            imported=True,
            import_source=str(session_file),
            user_initial_prompt=user_initial_prompt or "Imported session"
        )

        # Save to traces directory
        mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
        timestamp = mtime.strftime("%Y%m%d_%H%M%S")
        filename = f"imported_claude_{session_id[:8]}_{timestamp}.json"
        destination = self.trace_capture.traces_dir / filename

        try:
            with open(destination, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
        except IOError as e:
            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=0,
                error=f"Failed to write trace: {e}"
            )

        # Mark as imported
        self._save_imported_session(session_id)

        return ImportResult(
            session_id=session_id,
            source_file=session_file,
            steps_imported=len(steps),
            destination_file=destination
        )

    async def import_session_async(
        self,
        session_file: Path,
        force: bool = False,
        instrumentation: Optional["Instrumentation"] = None
    ) -> ImportResult:
        """
        Import a single session file with instrumentation (async).

        Applies PII filtering to user prompts and tool outputs,
        checks for injection attempts, and profiles the import.

        Args:
            session_file: Path to the .jsonl session file
            force: Import even if already imported
            instrumentation: Optional Instrumentation instance (uses global if None)

        Returns:
            ImportResult with import details and instrumentation stats
        """
        session_id = session_file.stem

        # Get instrumentation instance
        inst = instrumentation
        if inst is None and HAS_INSTRUMENTATION:
            inst = get_instrumentation()

        # Start profiling trace for this import
        trace_id = ""
        if inst and inst.profiler_enabled:
            trace_id = inst.start_trace(
                f"import:{session_id[:8]}",
                metadata={"source_file": str(session_file)}
            )

        pii_redactions = 0
        injection_blocked = False

        try:
            # Check if already imported
            if not force and session_id in self.imported_sessions:
                return ImportResult(
                    session_id=session_id,
                    source_file=session_file,
                    steps_imported=0,
                    skipped=True,
                    skip_reason="Already imported"
                )

            # Parse the session
            steps, user_initial_prompt = self.parse_session_file(session_file)

            if not steps:
                return ImportResult(
                    session_id=session_id,
                    source_file=session_file,
                    steps_imported=0,
                    skipped=True,
                    skip_reason="No relevant tool executions found"
                )

            sanitized_prompt = user_initial_prompt

            # Apply instrumentation if available
            if inst:
                # Check user prompt for injection
                if user_initial_prompt:
                    is_safe = await inst.check_injection(
                        user_initial_prompt,
                        location="import.user_prompt"
                    )
                    if not is_safe:
                        injection_blocked = True
                        # Still import but flag it
                        sanitized_prompt = "[INJECTION_DETECTED] " + user_initial_prompt

                    # Filter PII from user prompt
                    original_prompt = sanitized_prompt
                    sanitized_prompt = await inst.filter_pii(
                        sanitized_prompt,
                        location="import.user_prompt"
                    )
                    if sanitized_prompt != original_prompt:
                        pii_redactions += 1

                # Filter PII from tool outputs
                for step in steps:
                    if step.output:
                        original_output = step.output
                        step.output = await inst.filter_pii(
                            step.output,
                            location="import.tool_output"
                        )
                        if step.output != original_output:
                            pii_redactions += 1

                    # Also check command inputs for PII
                    if step.command:
                        original_command = step.command
                        step.command = await inst.filter_pii(
                            step.command,
                            location="import.tool_command"
                        )
                        if step.command != original_command:
                            pii_redactions += 1

            # Create a trace session with sanitized prompt
            session = TraceSession.from_steps(
                steps,
                source_tool="claude_code",
                verification_passed=True,
                imported=True,
                import_source=str(session_file),
                user_initial_prompt=sanitized_prompt or "Imported session"
            )

            # Add instrumentation metadata
            session.metadata["pii_redactions"] = pii_redactions
            session.metadata["injection_blocked"] = injection_blocked

            # Save to traces directory
            mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
            timestamp = mtime.strftime("%Y%m%d_%H%M%S")
            filename = f"imported_claude_{session_id[:8]}_{timestamp}.json"
            destination = self.trace_capture.traces_dir / filename

            try:
                with open(destination, 'w', encoding='utf-8') as f:
                    json.dump(asdict(session), f, indent=2, ensure_ascii=False)
            except IOError as e:
                return ImportResult(
                    session_id=session_id,
                    source_file=session_file,
                    steps_imported=0,
                    error=f"Failed to write trace: {e}",
                    pii_redactions=pii_redactions,
                    injection_blocked=injection_blocked
                )

            # Mark as imported
            self._save_imported_session(session_id)

            return ImportResult(
                session_id=session_id,
                source_file=session_file,
                steps_imported=len(steps),
                destination_file=destination,
                pii_redactions=pii_redactions,
                injection_blocked=injection_blocked,
                sanitized_prompt=sanitized_prompt
            )

        finally:
            # End profiling trace
            if inst and trace_id:
                inst.end_trace(trace_id)


async def import_today_async(
    project_filter: Optional[str] = None,
    verbose: bool = True
) -> List[ImportResult]:
    """
    Import today's Claude Code sessions with instrumentation (async).

    Args:
        project_filter: Only import from projects matching this substring
        verbose: Print progress messages

    Returns:
        List of ImportResult objects
    """
    importer = ClaudeSessionImporter()

    # Get start of today (local time)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_utc = today.astimezone(timezone.utc)

    session_files = importer.find_session_files(
        project_filter=project_filter,
        since=today_utc
    )

    if verbose:
        print(f"[BashGym] Found {len(session_files)} session(s) from today")

    results = []
    total_pii = 0
    total_injections = 0

    for session_file, mtime in session_files:
        result = await importer.import_session_async(session_file)
        results.append(result)
        total_pii += result.pii_redactions
        if result.injection_blocked:
            total_injections += 1

        if verbose:
            if result.skipped:
                print(f"  [-] {session_file.name}: {result.skip_reason}")
            elif result.error:
                print(f"  [!] {session_file.name}: {result.error}")
            else:
                pii_note = f" (PII: {result.pii_redactions})" if result.pii_redactions else ""
                inj_note = " [INJECTION]" if result.injection_blocked else ""
                print(f"  [+] {session_file.name}: {result.steps_imported} steps{pii_note}{inj_note}")

    if verbose and (total_pii or total_injections):
        print(f"[BashGym] Instrumentation: {total_pii} PII redactions, {total_injections} injection flags")

    return results


def import_today(
    project_filter: Optional[str] = None,
    verbose: bool = True
) -> List[ImportResult]:
    """
    Import today's Claude Code sessions.

    Args:
        project_filter: Only import from projects matching this substring
        verbose: Print progress messages

    Returns:
        List of ImportResult objects
    """
    importer = ClaudeSessionImporter()

    # Get start of today (local time)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_utc = today.astimezone(timezone.utc)

    session_files = importer.find_session_files(
        project_filter=project_filter,
        since=today_utc
    )

    if verbose:
        print(f"[BashGym] Found {len(session_files)} session(s) from today")

    results = []
    for session_file, mtime in session_files:
        result = importer.import_session(session_file)
        results.append(result)

        if verbose:
            if result.skipped:
                print(f"  [-] {session_file.name}: {result.skip_reason}")
            elif result.error:
                print(f"  [!] {session_file.name}: {result.error}")
            else:
                print(f"  [+] {session_file.name}: {result.steps_imported} steps imported")

    return results


def import_recent(
    days: int = 60,
    project_filter: Optional[str] = None,
    verbose: bool = True
) -> List[ImportResult]:
    """
    Import recent Claude Code sessions.

    Args:
        days: Number of days to look back (default 60)
        project_filter: Only import from projects matching this substring
        verbose: Print progress messages

    Returns:
        List of ImportResult objects
    """
    importer = ClaudeSessionImporter()

    # Calculate cutoff date
    since = datetime.now(timezone.utc) - timedelta(days=days)

    session_files = importer.find_session_files(
        project_filter=project_filter,
        since=since
    )

    if verbose:
        print(f"[BashGym] Found {len(session_files)} session(s) from last {days} days")

    results = []
    for session_file, mtime in session_files:
        result = importer.import_session(session_file)
        results.append(result)

        if verbose:
            if result.skipped:
                print(f"  [-] {session_file.name}: {result.skip_reason}")
            elif result.error:
                print(f"  [!] {session_file.name}: {result.error}")
            else:
                print(f"  [+] {session_file.name}: {result.steps_imported} steps imported")

    return results


def import_session(
    session_path: str,
    force: bool = False,
    verbose: bool = True
) -> ImportResult:
    """
    Import a specific session file.

    Args:
        session_path: Path to the session .jsonl file
        force: Import even if already imported
        verbose: Print progress messages

    Returns:
        ImportResult object
    """
    importer = ClaudeSessionImporter()
    session_file = Path(session_path)

    if not session_file.exists():
        return ImportResult(
            session_id="unknown",
            source_file=session_file,
            steps_imported=0,
            error=f"File not found: {session_path}"
        )

    result = importer.import_session(session_file, force=force)

    if verbose:
        if result.skipped:
            print(f"[BashGym] [-] Skipped: {result.skip_reason}")
        elif result.error:
            print(f"[BashGym] [!] Error: {result.error}")
        else:
            print(f"[BashGym] [+] Imported {result.steps_imported} steps to {result.destination_file}")

    return result
