"""
Environment Collector

Imports session environment data from Claude Code's local storage:
    ~/.claude/session-env/<session-id>/      — per-session environment directories
    ~/.claude/shell-snapshots/snapshot-bash-*.sh  — shell initialization snapshots

Session-env directories are named by session UUID and may contain environment
state files.  Shell snapshots are bash scripts capturing PATH, aliases, and
other shell initialization data.

This is the simplest collector — mostly metadata enrichment.  It captures
what's there without over-parsing.
"""

import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorScanResult,
    EnvironmentRecord,
)

# Regex for shell snapshot filenames: snapshot-bash-<timestamp>-<id>.sh
SNAPSHOT_PATTERN = re.compile(r"^snapshot-bash-(\d+)-(.+)\.sh$")

# Regex for extracting export statements like: export PATH=/usr/bin:/usr/local/bin
EXPORT_PATTERN = re.compile(r"^export\s+(\w+)=(.*)$")

# Regex for extracting alias statements like: alias ll='ls -la'
ALIAS_PATTERN = re.compile(r"""^alias\s+(\w[\w-]*)=(?:'([^']*)'|"([^"]*)"|(\S+))$""")


class EnvironmentCollector(BaseCollector):
    """Collect environment records from session-env dirs and shell snapshots.

    Session-env directories live at:
        <claude_dir>/session-env/<session-id>/

    Shell snapshots live at:
        <claude_dir>/shell-snapshots/snapshot-bash-<timestamp>-<id>.sh

    Each shell snapshot is a bash script containing exported variables and
    alias definitions.
    """

    @property
    def source_type(self) -> str:
        return "environments"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_session_env_dirs(
        self,
        since: str | None = None,
    ) -> list[tuple[Path, str]]:
        """Find session directories under session-env/.

        Parameters
        ----------
        since : str, optional
            ISO-8601 timestamp.  Only directories modified after this time
            are returned.

        Returns
        -------
        list of (dir_path, session_id) tuples
        """
        session_env_dir = self.claude_dir / "session-env"
        if not session_env_dir.exists():
            return []

        since_dt: datetime | None = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None

        results: list[tuple[Path, str]] = []

        for entry in session_env_dir.iterdir():
            if not entry.is_dir():
                continue

            if since_dt is not None:
                mtime = datetime.fromtimestamp(
                    entry.stat().st_mtime,
                    tz=timezone.utc,
                )
                if mtime < since_dt:
                    continue

            results.append((entry, entry.name))

        return results

    def _find_shell_snapshots(
        self,
        since: str | None = None,
    ) -> list[tuple[Path, str]]:
        """Find snapshot-bash-*.sh files under shell-snapshots/.

        Parameters
        ----------
        since : str, optional
            ISO-8601 timestamp.  Only files modified after this time
            are returned.

        Returns
        -------
        list of (filepath, snapshot_filename) tuples
        """
        snapshots_dir = self.claude_dir / "shell-snapshots"
        if not snapshots_dir.exists():
            return []

        since_dt: datetime | None = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None

        results: list[tuple[Path, str]] = []

        for entry in snapshots_dir.glob("snapshot-bash-*.sh"):
            if not entry.is_file():
                continue

            if since_dt is not None:
                mtime = datetime.fromtimestamp(
                    entry.stat().st_mtime,
                    tz=timezone.utc,
                )
                if mtime < since_dt:
                    continue

            results.append((entry, entry.name))

        return results

    def _parse_session_env(
        self,
        session_dir: Path,
        session_id: str,
    ) -> EnvironmentRecord:
        """Parse a session-env directory into an EnvironmentRecord.

        Lists the contents of the directory and captures basic metadata.
        Does not deeply parse any files found — just records what's there.

        Parameters
        ----------
        session_dir : Path
            Path to the session environment directory.
        session_id : str
            The session identifier (directory name).

        Returns
        -------
        EnvironmentRecord
        """
        # List contents of the session env directory
        contents: list[str] = []
        try:
            for entry in session_dir.iterdir():
                contents.append(entry.name)
        except OSError:
            pass

        # Build timestamp from directory mtime
        try:
            mtime = session_dir.stat().st_mtime
            timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            timestamp = datetime.now(timezone.utc).isoformat()

        return EnvironmentRecord(
            session_id=session_id,
            timestamp=timestamp,
            source_type=self.source_type,
            platform=platform.system(),
            metadata={"contents": contents} if contents else {},
        )

    def _parse_shell_snapshot(
        self,
        filepath: Path,
    ) -> EnvironmentRecord:
        """Parse a shell snapshot bash script into an EnvironmentRecord.

        Reads the bash script and extracts PATH and alias definitions.
        Keeps parsing minimal — just captures what's explicitly set.

        Parameters
        ----------
        filepath : Path
            Path to the snapshot-bash-*.sh file.

        Returns
        -------
        EnvironmentRecord
        """
        exports: dict[str, str] = {}
        aliases: dict[str, str] = {}
        raw_content = ""

        try:
            raw_content = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass

        for line in raw_content.splitlines():
            line = line.strip()

            # Match export VAR=value
            export_match = EXPORT_PATTERN.match(line)
            if export_match:
                exports[export_match.group(1)] = export_match.group(2)
                continue

            # Match alias name='value' or alias name="value" or alias name=value
            alias_match = ALIAS_PATTERN.match(line)
            if alias_match:
                name = alias_match.group(1)
                # Pick whichever group matched (single-quoted, double-quoted, or unquoted)
                value = alias_match.group(2) or alias_match.group(3) or alias_match.group(4) or ""
                aliases[name] = value

        # Extract snapshot id from filename
        snapshot_match = SNAPSHOT_PATTERN.match(filepath.name)
        snapshot_id = filepath.name
        if snapshot_match:
            snapshot_id = snapshot_match.group(2)

        # Build timestamp from file mtime
        try:
            mtime = filepath.stat().st_mtime
            timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            timestamp = datetime.now(timezone.utc).isoformat()

        shell_snapshot: dict[str, Any] = {}
        if exports:
            shell_snapshot["exports"] = exports
        if aliases:
            shell_snapshot["aliases"] = aliases

        return EnvironmentRecord(
            session_id=f"snapshot/{snapshot_id}",
            timestamp=timestamp,
            source_type=self.source_type,
            platform=platform.system(),
            shell="bash",
            shell_snapshot=shell_snapshot,
        )

    # ------------------------------------------------------------------
    # Public interface (BaseCollector)
    # ------------------------------------------------------------------

    def scan(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorScanResult:
        """Scan for session-env dirs and shell snapshots without collecting."""
        env_dirs = self._find_session_env_dirs(since=since)
        snapshots = self._find_shell_snapshots(since=since)
        collected_ids = self._load_collected_ids()

        total_found = len(env_dirs) + len(snapshots)
        already_collected = 0
        estimated_bytes = 0

        for _, session_id in env_dirs:
            if session_id in collected_ids:
                already_collected += 1

        for filepath, filename in snapshots:
            snapshot_match = SNAPSHOT_PATTERN.match(filename)
            snap_id = (
                f"snapshot/{snapshot_match.group(2)}" if snapshot_match else f"snapshot/{filename}"
            )
            if snap_id in collected_ids:
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

    def collect(self, session_id: str) -> list[EnvironmentRecord]:
        """Collect environment data for a specific session.

        Parameters
        ----------
        session_id : str
            The session UUID to look for in session-env/.

        Returns
        -------
        list of EnvironmentRecord
        """
        session_dir = self.claude_dir / "session-env" / session_id
        if not session_dir.is_dir():
            return []

        record = self._parse_session_env(session_dir, session_id)
        return [record]

    def collect_all(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorBatchResult:
        """Collect all uncollected environment records.

        Deduplicates by session_id for env dirs and by filename for snapshots.
        """
        env_dirs = self._find_session_env_dirs(since=since)
        snapshots = self._find_shell_snapshots(since=since)
        collected_ids = self._load_collected_ids()

        collected = 0
        skipped = 0
        errors: list[str] = []
        records: list[EnvironmentRecord] = []

        # Process session-env directories
        for session_dir, session_id in env_dirs:
            if session_id in collected_ids:
                skipped += 1
                continue

            try:
                record = self._parse_session_env(session_dir, session_id)
                records.append(record)
                self._save_collected_id(session_id)
                collected += 1
            except Exception as exc:
                errors.append(f"{session_dir}: {exc}")

        # Process shell snapshots
        for filepath, filename in snapshots:
            snapshot_match = SNAPSHOT_PATTERN.match(filename)
            snap_id = (
                f"snapshot/{snapshot_match.group(2)}" if snapshot_match else f"snapshot/{filename}"
            )

            if snap_id in collected_ids:
                skipped += 1
                continue

            try:
                record = self._parse_shell_snapshot(filepath)
                records.append(record)
                self._save_collected_id(snap_id)
                collected += 1
            except Exception as exc:
                errors.append(f"{filepath}: {exc}")

        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected,
            skipped=skipped,
            errors=errors,
            records=records,
        )
