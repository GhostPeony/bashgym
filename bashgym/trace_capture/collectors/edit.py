"""
Edit Collector

Parses versioned file snapshots from Claude Code's file-history directory:
    ~/.claude/file-history/<session-id>/<content-hash>@v<version-number>

Each session directory contains pairs (or sequences) of file snapshots
representing before/after states of edits.  The collector groups snapshots
by content hash, sorts by version, reads content, and produces a unified
diff between the first and last version.
"""

import difflib
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorScanResult,
    EditRecord,
)

# Regex that matches filenames like "a1b2c3d4e5f6g7h8@v1"
VERSION_PATTERN = re.compile(r"^(.+)@v(\d+)$")


class EditCollector(BaseCollector):
    """Collect structured records from Claude Code file-history snapshots.

    File-history snapshots live at:
        <claude_dir>/file-history/<session-id>/<content-hash>@v<n>

    Each file contains the raw text of a file at a particular version.
    This collector groups snapshots by content hash, reads each version,
    and produces an EditRecord with a unified diff between the first and
    last versions.
    """

    @property
    def source_type(self) -> str:
        return "edits"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_edit_sessions(
        self,
        since: str | None = None,
    ) -> list[tuple[Path, str]]:
        """Find session directories under file-history/.

        Parameters
        ----------
        since : str, optional
            ISO-8601 timestamp.  Only directories with files modified after
            this time are returned.

        Returns
        -------
        list of (dir_path, session_id) tuples
        """
        file_history_dir = self.claude_dir / "file-history"
        if not file_history_dir.exists():
            return []

        # Parse the optional since timestamp
        since_dt: datetime | None = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None

        results: list[tuple[Path, str]] = []

        for session_dir in file_history_dir.iterdir():
            if not session_dir.is_dir():
                continue

            # Apply date filter on directory modification time
            if since_dt is not None:
                mtime = datetime.fromtimestamp(
                    session_dir.stat().st_mtime,
                    tz=timezone.utc,
                )
                if mtime < since_dt:
                    continue

            session_id = session_dir.name
            results.append((session_dir, session_id))

        return results

    def _parse_session_edits(
        self,
        session_dir: Path,
        session_id: str,
    ) -> list[EditRecord]:
        """Parse all versioned snapshots in a session directory.

        Groups files by content hash, sorts by version number, reads
        content from each version, and generates a unified diff between
        the first and last version of each hash group.

        Parameters
        ----------
        session_dir : Path
            Path to the session directory inside file-history/.
        session_id : str
            The session identifier (directory name).

        Returns
        -------
        list of EditRecord
        """
        # Group files by content hash: {hash: {version: path}}
        hash_versions: dict[str, dict[int, Path]] = defaultdict(dict)

        for entry in session_dir.iterdir():
            if not entry.is_file():
                continue

            match = VERSION_PATTERN.match(entry.name)
            if not match:
                continue

            content_hash = match.group(1)
            version_num = int(match.group(2))
            hash_versions[content_hash][version_num] = entry

        records: list[EditRecord] = []

        for content_hash, version_map in sorted(hash_versions.items()):
            sorted_versions = sorted(version_map.keys())
            versions_list: list[dict] = []

            for ver_num in sorted_versions:
                filepath = version_map[ver_num]
                try:
                    content = filepath.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    content = ""

                versions_list.append(
                    {
                        "version": ver_num,
                        "content": content,
                    }
                )

            # Generate unified diff between first and last version
            first_content = versions_list[0]["content"]
            last_content = versions_list[-1]["content"]
            diff_lines = list(
                difflib.unified_diff(
                    first_content.splitlines(keepends=True),
                    last_content.splitlines(keepends=True),
                    fromfile=f"{content_hash}@v{sorted_versions[0]}",
                    tofile=f"{content_hash}@v{sorted_versions[-1]}",
                )
            )
            diff_text = "".join(diff_lines)

            # Build timestamp from the latest version file's mtime
            latest_path = version_map[sorted_versions[-1]]
            try:
                mtime = latest_path.stat().st_mtime
                timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                timestamp = datetime.now(timezone.utc).isoformat()

            records.append(
                EditRecord(
                    session_id=session_id,
                    timestamp=timestamp,
                    source_type=self.source_type,
                    content_hash=content_hash,
                    versions=versions_list,
                    diff=diff_text,
                    total_versions=len(sorted_versions),
                )
            )

        return records

    # ------------------------------------------------------------------
    # Public interface (BaseCollector)
    # ------------------------------------------------------------------

    def scan(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorScanResult:
        """Scan for edit sessions without collecting anything."""
        sessions = self._find_edit_sessions(since=since)
        collected_ids = self._load_collected_ids()

        total_found = len(sessions)
        already_collected = 0
        estimated_bytes = 0

        for session_dir, session_id in sessions:
            if session_id in collected_ids:
                already_collected += 1
            try:
                for f in session_dir.iterdir():
                    if f.is_file():
                        estimated_bytes += f.stat().st_size
            except OSError:
                pass

        return CollectorScanResult(
            source_type=self.source_type,
            total_found=total_found,
            already_collected=already_collected,
            new_available=total_found - already_collected,
            estimated_size_bytes=estimated_bytes,
        )

    def collect(self, session_id: str) -> list[EditRecord]:
        """Collect all edit records for a specific session.

        Parameters
        ----------
        session_id : str
            The session UUID directory name to look for edits in.

        Returns
        -------
        list of EditRecord
        """
        sessions = self._find_edit_sessions()
        records: list[EditRecord] = []

        for session_dir, found_session_id in sessions:
            if found_session_id != session_id:
                continue

            records.extend(self._parse_session_edits(session_dir, found_session_id))

        return records

    def collect_all(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorBatchResult:
        """Collect all uncollected edit records.

        Deduplicates by session_id.  Each session directory is processed
        once; all edit records within it are collected together.
        """
        sessions = self._find_edit_sessions(since=since)
        collected_ids = self._load_collected_ids()

        collected = 0
        skipped = 0
        errors: list[str] = []
        records: list[EditRecord] = []

        for session_dir, session_id in sessions:
            if session_id in collected_ids:
                skipped += 1
                continue

            try:
                session_records = self._parse_session_edits(session_dir, session_id)
                if session_records:
                    records.extend(session_records)
                    self._save_collected_id(session_id)
                    collected += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors.append(f"{session_dir}: {exc}")

        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected,
            skipped=skipped,
            errors=errors,
            records=records,
        )
