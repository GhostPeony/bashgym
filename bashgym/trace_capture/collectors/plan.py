"""
Plan Collector

Imports implementation plans from Claude Code's plans directory:
    ~/.claude/plans/*.md

Plan files are markdown documents with alliterative names (e.g.,
``clever-jumping-fox.md``).  They contain structured implementation plans
but do not carry session IDs directly.  For Phase 1, records are stored
with an empty session_id.

The collector deduplicates by plan_name (the file stem).
"""

from datetime import datetime, timezone
from pathlib import Path

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorScanResult,
    PlanRecord,
)


class PlanCollector(BaseCollector):
    """Collect structured records from Claude Code plan markdown files.

    Plan files live at:
        <claude_dir>/plans/<alliterative-name>.md

    Each file is a Markdown document describing an implementation plan.
    Plans are not session-scoped, so ``collect(session_id)`` always returns
    an empty list.  Use ``collect_all()`` to gather all plans.
    """

    @property
    def source_type(self) -> str:
        return "plans"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_plan_files(
        self,
        since: str | None = None,
    ) -> list[Path]:
        """Find all plan markdown files in the plans directory.

        Parameters
        ----------
        since : str, optional
            ISO-8601 timestamp.  Only files modified after this time are
            returned.

        Returns
        -------
        list of Path
        """
        plans_dir = self.claude_dir / "plans"
        if not plans_dir.exists():
            return []

        # Parse the optional since timestamp
        since_dt: datetime | None = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except ValueError:
                since_dt = None

        results: list[Path] = []

        for plan_file in plans_dir.glob("*.md"):
            if not plan_file.is_file():
                continue

            # Apply date filter on file modification time
            if since_dt is not None:
                mtime = datetime.fromtimestamp(
                    plan_file.stat().st_mtime,
                    tz=timezone.utc,
                )
                if mtime < since_dt:
                    continue

            results.append(plan_file)

        return results

    def _parse_plan(self, filepath: Path) -> PlanRecord | None:
        """Parse a single plan markdown file into a PlanRecord.

        Parameters
        ----------
        filepath : Path
            Path to the .md plan file.

        Returns
        -------
        PlanRecord or None if the file cannot be read.
        """
        try:
            content = filepath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        plan_name = filepath.stem
        word_count = len(content.split())

        # Build timestamp from file modification time
        try:
            mtime = filepath.stat().st_mtime
            timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except OSError:
            timestamp = datetime.now(timezone.utc).isoformat()

        return PlanRecord(
            session_id="",
            timestamp=timestamp,
            source_type=self.source_type,
            plan_name=plan_name,
            content=content,
            word_count=word_count,
        )

    # ------------------------------------------------------------------
    # Public interface (BaseCollector)
    # ------------------------------------------------------------------

    def scan(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorScanResult:
        """Scan for plan files without collecting anything."""
        files = self._find_plan_files(since=since)
        collected_ids = self._load_collected_ids()

        total_found = len(files)
        already_collected = 0
        estimated_bytes = 0

        for filepath in files:
            plan_name = filepath.stem
            if plan_name in collected_ids:
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

    def collect(self, session_id: str) -> list[PlanRecord]:
        """Collect plan records for a specific session.

        Plans are not session-scoped, so this always returns an empty list.

        Parameters
        ----------
        session_id : str
            Ignored — plans have no session association.

        Returns
        -------
        empty list
        """
        return []

    def collect_all(
        self,
        since: str | None = None,
        project_filter: str | None = None,
    ) -> CollectorBatchResult:
        """Collect all uncollected plan records.

        Deduplicates by plan_name (the filename stem).
        """
        files = self._find_plan_files(since=since)
        collected_ids = self._load_collected_ids()

        collected = 0
        skipped = 0
        errors: list[str] = []
        records: list[PlanRecord] = []

        for filepath in files:
            plan_name = filepath.stem

            if plan_name in collected_ids:
                skipped += 1
                continue

            try:
                record = self._parse_plan(filepath)
                if record is not None:
                    records.append(record)
                    self._save_collected_id(plan_name)
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
