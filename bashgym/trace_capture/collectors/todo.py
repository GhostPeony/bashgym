"""
Todo Collector

Imports task decomposition data from Claude Code's todos directory:
    ~/.claude/todos/<session-id>-agent-<agent-id>.json

Each todo file is a JSON array of task objects.  Most files contain an
empty array (``[]``) and are skipped.  Non-empty files contain objects
with ``subject`` and ``status`` fields representing task decomposition
used by Claude Code agents.

The collector deduplicates by the filename stem (the full
``<session-id>-agent-<agent-id>`` composite key).
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorScanResult,
    TodoRecord,
)

# Regex for the todo filename pattern:
#   <session-uuid>-agent-<agent-uuid>.json
# where UUIDs are 8-4-4-4-12 hex format.
_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_FILENAME_RE = re.compile(
    rf"^({_UUID_RE})-agent-({_UUID_RE})\.json$",
    re.IGNORECASE,
)


class TodoCollector(BaseCollector):
    """Collect structured records from Claude Code todo JSON files.

    Todo files live at:
        <claude_dir>/todos/<session-id>-agent-<agent-id>.json

    Each file is a JSON array of task objects with ``subject`` and
    ``status`` fields.  Files containing empty arrays are skipped since
    they carry no training value.
    """

    @property
    def source_type(self) -> str:
        return "todos"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_filename(
        self,
        filename: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract session_id and agent_id from a todo filename.

        Parameters
        ----------
        filename : str
            The filename to parse, e.g.
            ``abc12345-1234-5678-abcd-1234567890ab-agent-def67890-...json``

        Returns
        -------
        (session_id, agent_id) tuple, or (None, None) if the filename
        does not match the expected pattern.
        """
        match = _FILENAME_RE.match(filename)
        if not match:
            return None, None
        return match.group(1), match.group(2)

    def _find_todo_files(
        self,
        since: Optional[str] = None,
    ) -> List[Tuple[Path, str, str]]:
        """Find all non-empty todo JSON files in the todos directory.

        Parameters
        ----------
        since : str, optional
            ISO-8601 timestamp.  Only files modified after this time are
            returned.

        Returns
        -------
        list of (path, session_id, agent_id) tuples.  Files containing
        empty arrays are excluded.
        """
        todos_dir = self.claude_dir / "todos"
        if not todos_dir.exists():
            return []

        # Parse the optional since timestamp
        since_dt: Optional[datetime] = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None

        results: List[Tuple[Path, str, str]] = []

        for todo_file in todos_dir.glob("*.json"):
            if not todo_file.is_file():
                continue

            # Parse filename to extract session_id and agent_id
            session_id, agent_id = self._parse_filename(todo_file.name)
            if session_id is None or agent_id is None:
                continue

            # Apply date filter on file modification time
            if since_dt is not None:
                try:
                    mtime = datetime.fromtimestamp(
                        todo_file.stat().st_mtime,
                        tz=timezone.utc,
                    )
                    if mtime < since_dt:
                        continue
                except OSError:
                    continue

            # Skip files with empty arrays (no training value)
            try:
                content = todo_file.read_text(encoding="utf-8").strip()
                if not content or content == "[]":
                    continue
                data = json.loads(content)
                if not isinstance(data, list) or len(data) == 0:
                    continue
            except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError):
                continue

            results.append((todo_file, session_id, agent_id))

        return results

    def _parse_todo_file(
        self,
        filepath: Path,
        session_id: str,
        agent_id: str,
    ) -> Optional[TodoRecord]:
        """Parse a single todo JSON file into a TodoRecord.

        Parameters
        ----------
        filepath : Path
            Path to the todo JSON file.
        session_id : str
            The session UUID extracted from the filename.
        agent_id : str
            The agent UUID extracted from the filename.

        Returns
        -------
        TodoRecord or None if the file is empty, contains an empty array,
        or cannot be parsed.
        """
        try:
            content = filepath.read_text(encoding="utf-8")
            data = json.loads(content)
        except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError):
            return None

        if not isinstance(data, list) or len(data) == 0:
            return None

        # Count statuses
        total_tasks = len(data)
        completed_tasks = sum(
            1 for task in data
            if isinstance(task, dict) and task.get("status") == "completed"
        )
        pending_tasks = sum(
            1 for task in data
            if isinstance(task, dict) and task.get("status") == "pending"
        )

        # Build timestamp from file modification time
        try:
            mtime = filepath.stat().st_mtime
            timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            timestamp = datetime.now(timezone.utc).isoformat()

        return TodoRecord(
            session_id=session_id,
            timestamp=timestamp,
            source_type=self.source_type,
            agent_id=agent_id,
            tasks=data,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            pending_tasks=pending_tasks,
        )

    # ------------------------------------------------------------------
    # Public interface (BaseCollector)
    # ------------------------------------------------------------------

    def scan(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorScanResult:
        """Scan for non-empty todo files without collecting anything."""
        files = self._find_todo_files(since=since)
        collected_ids = self._load_collected_ids()

        total_found = len(files)
        already_collected = 0
        estimated_bytes = 0

        for filepath, session_id, agent_id in files:
            dedup_key = filepath.stem
            if dedup_key in collected_ids:
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

    def collect(self, session_id: str) -> List[TodoRecord]:
        """Collect todo records matching a specific session_id.

        Parameters
        ----------
        session_id : str
            The session UUID to filter by.

        Returns
        -------
        list of TodoRecord
        """
        files = self._find_todo_files()
        records: List[TodoRecord] = []

        for filepath, file_session_id, agent_id in files:
            if file_session_id != session_id:
                continue

            record = self._parse_todo_file(filepath, file_session_id, agent_id)
            if record is not None:
                records.append(record)

        return records

    def collect_all(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        """Collect all uncollected, non-empty todo records.

        Deduplicates by filename stem (the ``<session-id>-agent-<agent-id>``
        composite key).  Files with empty arrays are skipped.
        """
        files = self._find_todo_files(since=since)
        collected_ids = self._load_collected_ids()

        collected = 0
        skipped = 0
        errors: List[str] = []
        records: List[TodoRecord] = []

        for filepath, session_id, agent_id in files:
            dedup_key = filepath.stem

            if dedup_key in collected_ids:
                skipped += 1
                continue

            try:
                record = self._parse_todo_file(filepath, session_id, agent_id)
                if record is not None:
                    records.append(record)
                    self._save_collected_id(dedup_key)
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
