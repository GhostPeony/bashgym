"""
Subagent Collector

Parses subagent JSONL files from Claude Code's local project storage:
    ~/.claude/projects/<project-slug>/<session-id>/subagents/agent-<id>.jsonl

Each subagent file is a full conversation transcript containing alternating
"user" and "assistant" type events.  The collector extracts structured
SubagentRecord objects with tool usage, token counts, and step details.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorScanResult,
    SubagentRecord,
)


class SubagentCollector(BaseCollector):
    """Collect structured records from Claude Code subagent JSONL files.

    Subagent files live at:
        <claude_dir>/projects/<slug>/<session-id>/subagents/agent-<id>.jsonl

    Each file is a full conversation transcript with alternating "user" and
    "assistant" events.  This collector parses those events and produces a
    SubagentRecord per file.
    """

    @property
    def source_type(self) -> str:
        return "subagents"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_subagent_files(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> List[Tuple[Path, str, str]]:
        """Find all subagent JSONL files under the projects directory.

        Parameters
        ----------
        since : str, optional
            ISO-8601 timestamp.  Only files modified after this time are
            returned.
        project_filter : str, optional
            Substring filter applied to the project directory name
            (case-insensitive).

        Returns
        -------
        list of (path, session_id, agent_id) tuples
        """
        projects_dir = self.claude_dir / "projects"
        if not projects_dir.exists():
            return []

        # Parse the optional since timestamp
        since_dt: Optional[datetime] = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None

        results: List[Tuple[Path, str, str]] = []

        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Apply project filter (case-insensitive substring match)
            if project_filter and project_filter.lower() not in project_dir.name.lower():
                continue

            # Iterate session directories inside the project
            for session_dir in project_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                subagents_dir = session_dir / "subagents"
                if not subagents_dir.is_dir():
                    continue

                session_id = session_dir.name

                for agent_file in subagents_dir.glob("agent-*.jsonl"):
                    # Apply date filter on file modification time
                    if since_dt is not None:
                        mtime = datetime.fromtimestamp(
                            agent_file.stat().st_mtime,
                            tz=timezone.utc,
                        )
                        if mtime < since_dt:
                            continue

                    # Extract agent_id from filename: "agent-<id>.jsonl" -> "<id>"
                    agent_id = agent_file.stem.removeprefix("agent-")
                    results.append((agent_file, session_id, agent_id))

        return results

    def _parse_subagent_file(
        self,
        filepath: Path,
        session_id: str,
        agent_id: str,
    ) -> Optional[SubagentRecord]:
        """Parse a single subagent JSONL file into a SubagentRecord.

        Parameters
        ----------
        filepath : Path
            Path to the agent-<id>.jsonl file.
        session_id : str
            The parent session ID (directory name).
        agent_id : str
            The agent identifier extracted from the filename.

        Returns
        -------
        SubagentRecord or None if the file is empty / unparseable.
        """
        models_used: Set[str] = set()
        tools_used: Set[str] = set()
        steps: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_tool_calls = 0
        user_prompt: Optional[str] = None
        slug: Optional[str] = None
        file_agent_id: Optional[str] = None
        timestamp: Optional[str] = None
        found_any_event = False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    found_any_event = True

                    # Extract session-level metadata from first occurrence
                    if file_agent_id is None and "agentId" in event:
                        file_agent_id = event["agentId"]
                    if slug is None and "slug" in event:
                        slug = event["slug"]
                    if timestamp is None and "timestamp" in event:
                        timestamp = event["timestamp"]

                    event_type = event.get("type")

                    # ----- USER events -----
                    if event_type == "user":
                        message = event.get("message", {})
                        content = message.get("content", "")
                        if user_prompt is None and content:
                            if isinstance(content, str):
                                user_prompt = content
                            elif isinstance(content, list):
                                # Extract text from content blocks
                                texts = []
                                for item in content:
                                    if isinstance(item, str):
                                        texts.append(item)
                                    elif isinstance(item, dict) and item.get("type") == "text":
                                        texts.append(item.get("text", ""))
                                if texts:
                                    user_prompt = "\n".join(texts)

                    # ----- ASSISTANT events -----
                    elif event_type == "assistant":
                        msg = event.get("message", {})
                        model = msg.get("model", "")
                        usage = msg.get("usage", {})

                        if model:
                            models_used.add(model)

                        total_input_tokens += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)

                        content = msg.get("content", [])
                        if not isinstance(content, list):
                            continue

                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "tool_use":
                                tool_name = item.get("name", "")
                                tool_input = item.get("input", {})
                                tool_id = item.get("id", "")

                                total_tool_calls += 1
                                if tool_name:
                                    tools_used.add(tool_name)

                                steps.append({
                                    "tool_name": tool_name,
                                    "input": tool_input,
                                    "tool_use_id": tool_id,
                                    "model": model,
                                })

        except (IOError, OSError):
            return None

        if not found_any_event:
            return None

        # Use the agent_id from the file contents if available, else from filename
        resolved_agent_id = file_agent_id or agent_id

        # Build timestamp: use the one from the file, or fall back to file mtime
        if timestamp is None:
            try:
                mtime = filepath.stat().st_mtime
                timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                timestamp = datetime.now(timezone.utc).isoformat()

        return SubagentRecord(
            session_id=f"{session_id}/{resolved_agent_id}",
            timestamp=timestamp,
            source_type=self.source_type,
            agent_id=resolved_agent_id,
            slug=slug or "",
            parent_session_id=session_id,
            steps=steps,
            models_used=sorted(models_used),
            tools_used=sorted(tools_used),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tool_calls=total_tool_calls,
            user_prompt=user_prompt or "",
        )

    # ------------------------------------------------------------------
    # Public interface (BaseCollector)
    # ------------------------------------------------------------------

    def scan(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorScanResult:
        """Scan for subagent files without collecting anything."""
        files = self._find_subagent_files(since=since, project_filter=project_filter)
        collected_ids = self._load_collected_ids()

        total_found = len(files)
        already_collected = 0
        estimated_bytes = 0

        for filepath, session_id, agent_id in files:
            record_id = f"{session_id}/{agent_id}"
            if record_id in collected_ids:
                already_collected += 1
            try:
                estimated_bytes += filepath.stat().st_size
            except OSError:
                pass

        return CollectorScanResult(
            source_type=self.source_type,
            total_found=total_found,
            already_collected=already_collected,
            new_available=total_found - already_collected,
            estimated_size_bytes=estimated_bytes,
        )

    def collect(self, session_id: str) -> List[SubagentRecord]:
        """Collect all subagent records for a specific session.

        Parameters
        ----------
        session_id : str
            The session UUID directory name to look for subagents in.

        Returns
        -------
        list of SubagentRecord
        """
        files = self._find_subagent_files()
        records: List[SubagentRecord] = []

        for filepath, file_session_id, agent_id in files:
            if file_session_id != session_id:
                continue

            record = self._parse_subagent_file(filepath, file_session_id, agent_id)
            if record is not None:
                records.append(record)

        return records

    def collect_all(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        """Collect all uncollected subagent records.

        Deduplicates using the "{session_id}/{agent_id}" composite key.
        """
        files = self._find_subagent_files(since=since, project_filter=project_filter)
        collected_ids = self._load_collected_ids()

        collected = 0
        skipped = 0
        errors: List[str] = []
        records: List[SubagentRecord] = []

        for filepath, session_id, agent_id in files:
            record_id = f"{session_id}/{agent_id}"

            if record_id in collected_ids:
                skipped += 1
                continue

            try:
                record = self._parse_subagent_file(filepath, session_id, agent_id)
                if record is not None:
                    records.append(record)
                    self._save_collected_id(record_id)
                    collected += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors.append(f"{filepath}: {exc}")

        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected,
            skipped=skipped,
            errors=errors,
            records=records,
        )
