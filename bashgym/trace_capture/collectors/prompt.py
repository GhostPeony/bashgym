"""
Prompt Collector

Imports user prompts from Claude Code's history file:
    ~/.claude/history.jsonl

Each line in history.jsonl is a JSON object:
    {
        "display": "prompt text",
        "pastedContents": {"hash1": true},
        "timestamp": 1759817571796,
        "project": "C:\\Users\\Cade\\projects\\myapp"
    }

Pasted content is resolved from ~/.claude/paste-cache/<hash>.txt files.

The collector produces PromptRecord objects with prompt text, project name,
and any linked pasted content.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorScanResult,
    PromptRecord,
)

# Truncation limits
MAX_PROMPT_TEXT_LEN = 5000
MAX_PASTED_CONTENT_LEN = 10000

# Separator between multiple paste-cache entries
PASTE_SEPARATOR = "\n\n---\n\n"


class PromptCollector(BaseCollector):
    """Collect structured records from Claude Code's history.jsonl.

    History entries live at:
        <claude_dir>/history.jsonl

    Pasted content is resolved from:
        <claude_dir>/paste-cache/<hash>.txt

    Each line in the history file is one user prompt with an epoch-ms
    timestamp, a project path, and optional references to pasted content.
    """

    @property
    def source_type(self) -> str:
        return "prompts"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_paste_cache(self, paste_ids: dict[str, bool]) -> str:
        """Read paste-cache files for the given paste IDs.

        Parameters
        ----------
        paste_ids : dict
            Mapping of paste hashes to ``True`` (as stored in history.jsonl).

        Returns
        -------
        str
            Concatenated content of all resolved paste-cache files,
            joined by PASTE_SEPARATOR.  Empty string if none found.
        """
        paste_dir = self.claude_dir / "paste-cache"
        if not paste_dir.is_dir():
            return ""

        parts: list[str] = []
        for paste_id in paste_ids:
            paste_file = paste_dir / f"{paste_id}.txt"
            if paste_file.is_file():
                try:
                    content = paste_file.read_text(encoding="utf-8")
                    if content:
                        parts.append(content)
                except (OSError, UnicodeDecodeError):
                    pass

        return PASTE_SEPARATOR.join(parts)

    def _read_history_lines(self) -> list[str]:
        """Read all lines from history.jsonl.

        Returns
        -------
        list of str
            Raw lines (stripped).  Empty if file does not exist.
        """
        history_file = self.claude_dir / "history.jsonl"
        if not history_file.is_file():
            return []

        try:
            text = history_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        return [line.strip() for line in text.splitlines() if line.strip()]

    def _parse_entry(self, line: str) -> dict | None:
        """Parse a single history.jsonl line.

        Returns the parsed dict or None if invalid.
        """
        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None

    def _entry_matches(
        self,
        entry: dict,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> bool:
        """Check whether a history entry passes the given filters.

        Parameters
        ----------
        entry : dict
            Parsed history.jsonl entry.
        since : str, optional
            ISO-8601 timestamp.  Only entries with a timestamp after this
            value pass.
        project_filter : str, optional
            Case-insensitive substring match on the project path's final
            component (directory name).
        """
        # Timestamp filter
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None

            if since_dt is not None:
                ts_ms = entry.get("timestamp", 0)
                entry_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if entry_dt < since_dt:
                    return False

        # Project filter
        if project_filter:
            project_path = entry.get("project", "")
            if project_path:
                project_name = Path(project_path).name
            else:
                project_name = ""
            if project_filter.lower() not in project_name.lower():
                return False

        return True

    def _dedup_key(self, entry: dict) -> str:
        """Return the deduplication key for a history entry.

        Uses the string representation of the millisecond timestamp.
        """
        return str(entry.get("timestamp", ""))

    # ------------------------------------------------------------------
    # Public interface (BaseCollector)
    # ------------------------------------------------------------------

    def scan(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorScanResult:
        """Scan history.jsonl for available prompts without collecting."""
        lines = self._read_history_lines()
        collected_ids = self._load_collected_ids()

        total_found = 0
        already_collected = 0
        estimated_bytes = 0

        for line in lines:
            entry = self._parse_entry(line)
            if entry is None:
                continue

            if not self._entry_matches(entry, since=since, project_filter=project_filter):
                continue

            total_found += 1
            estimated_bytes += len(line.encode("utf-8"))

            dedup_key = self._dedup_key(entry)
            if dedup_key in collected_ids:
                already_collected += 1

        return CollectorScanResult(
            source_type=self.source_type,
            total_found=total_found,
            already_collected=already_collected,
            new_available=total_found - already_collected,
            estimated_size_bytes=estimated_bytes,
        )

    def collect(self, session_id: str) -> list[PromptRecord]:
        """Collect records for a single session.

        Prompts are not session-scoped, so this always returns an empty list.
        """
        return []

    def collect_all(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorBatchResult:
        """Collect all uncollected prompt records from history.jsonl.

        Reads history.jsonl line by line, filters by timestamp and project,
        deduplicates by timestamp string, reads paste-cache content, and
        produces PromptRecord objects.

        File naming: prompt_{timestamp}.json
        Truncation: prompt_text to 5000 chars, pasted_content to 10000 chars.
        """
        lines = self._read_history_lines()
        collected_ids = self._load_collected_ids()

        collected = 0
        skipped = 0
        errors: list[str] = []
        records: list[PromptRecord] = []

        for line in lines:
            entry = self._parse_entry(line)
            if entry is None:
                continue

            if not self._entry_matches(entry, since=since, project_filter=project_filter):
                continue

            dedup_key = self._dedup_key(entry)
            if dedup_key in collected_ids:
                skipped += 1
                continue

            try:
                ts_ms = entry.get("timestamp", 0)
                ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                timestamp_iso = ts_dt.isoformat()

                project_path = entry.get("project", "")
                project_name = Path(project_path).name if project_path else ""

                prompt_text = entry.get("display", "")[:MAX_PROMPT_TEXT_LEN]

                pasted_contents = entry.get("pastedContents", {})
                pasted_content = ""
                if pasted_contents:
                    pasted_content = self._read_paste_cache(pasted_contents)
                    pasted_content = pasted_content[:MAX_PASTED_CONTENT_LEN]

                record = PromptRecord(
                    session_id=f"prompt_{ts_ms}",
                    timestamp=timestamp_iso,
                    source_type=self.source_type,
                    project=project_name,
                    prompt_text=prompt_text,
                    pasted_content=pasted_content,
                )

                records.append(record)
                self._save_collected_id(dedup_key)
                collected += 1

            except Exception as exc:
                errors.append(f"prompt at {dedup_key}: {exc}")

        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected,
            skipped=skipped,
            errors=errors,
            records=records,
        )
