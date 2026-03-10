"""
Debug Collector

Parses Claude Code debug log files from:
    ~/.claude/debug/<session-id>.txt

Debug files are large (up to 30MB per session, 449MB total).  Each file
contains timestamped log entries covering:
  - Startup / plugin loading
  - API requests (``[API:request]`` tags)
  - Streaming responses (``Stream started - received first chunk``)
  - Token accounting (``autocompact: tokens=N ...``)
  - Model routing (``Tool search disabled for model 'MODEL'``)
  - System prompt configuration (``[SystemPrompt] ...``)
  - Error traces (``[ERROR]`` lines)
  - Attribution / billing headers

The collector extracts *metadata only* -- never raw user content.  This
keeps records lightweight and avoids storing PII.

Design constraints (from the task spec):
  - Do NOT store raw debug logs (they are huge).
  - Each debug file = one session (filename is session-id.txt).
  - Extract per-API-call metadata: timestamp, model, latency, token count.
  - Store only summary stats + limited samples.
  - For PII awareness: no user messages or file contents.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import (
    BaseCollector,
    CollectorBatchResult,
    CollectorScanResult,
    DebugRecord,
)


# ---------------------------------------------------------------------------
# Regex patterns for extracting data from debug log lines
# ---------------------------------------------------------------------------

# Timestamp at the start of every line: 2026-02-20T16:17:40.218Z
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")

# Log level tag: [DEBUG] or [ERROR] or [WARN]
_LEVEL_RE = re.compile(r"\[(DEBUG|ERROR|WARN|INFO)\]")

# API request initiation
_API_REQUEST_RE = re.compile(r"\[API:request\] Creating client")

# Stream response start (marks when first token arrives)
_STREAM_START_RE = re.compile(r"Stream started - received first chunk")

# Model name extraction from tool-search-disabled messages
_MODEL_RE = re.compile(r"Tool search disabled for model '([^']+)'")

# Token counts from autocompact
_AUTOCOMPACT_RE = re.compile(
    r"autocompact: tokens=(\d+) threshold=(\d+) effectiveWindow=(\d+)"
)

# System prompt configuration
_SYSTEM_PROMPT_RE = re.compile(r"\[SystemPrompt\]\s+(.*)")

# Error lines
_ERROR_RE = re.compile(r"\[ERROR\]\s+(.*)")

# Attribution / billing header
_ATTRIBUTION_RE = re.compile(
    r"attribution header x-anthropic-billing-header:\s+(.*)"
)


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string to a datetime object.

    Returns None if parsing fails.
    """
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class DebugCollector(BaseCollector):
    """Collect API-traffic metadata from Claude Code debug log files.

    Debug files live at:
        <claude_dir>/debug/<session-id>.txt

    Each file is a plain-text log with one timestamped entry per line.
    The collector parses these for structural metadata (API call counts,
    models, latencies, token accounting, errors) without storing any
    user content or raw payloads.
    """

    @property
    def source_type(self) -> str:
        return "debug"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _debug_dir(self) -> Path:
        """Path to the debug log directory."""
        return self.claude_dir / "debug"

    def _find_debug_files(
        self,
        since: Optional[str] = None,
    ) -> List[Path]:
        """Find all debug log .txt files, optionally filtered by date.

        Parameters
        ----------
        since : str, optional
            ISO-8601 timestamp.  Only files modified after this time are
            returned.

        Returns
        -------
        list of Path
        """
        debug_dir = self._debug_dir()
        if not debug_dir.exists():
            return []

        since_dt: Optional[datetime] = None
        if since:
            since_dt = _parse_timestamp(since)

        results: List[Path] = []
        for filepath in debug_dir.glob("*.txt"):
            if not filepath.is_file():
                continue

            if since_dt is not None:
                try:
                    mtime = datetime.fromtimestamp(
                        filepath.stat().st_mtime,
                        tz=timezone.utc,
                    )
                    if mtime < since_dt:
                        continue
                except OSError:
                    continue

            results.append(filepath)

        return results

    def _parse_debug_log(self, filepath: Path) -> Optional[DebugRecord]:
        """Parse a single debug log file into a DebugRecord.

        Extracts only structural metadata -- never stores raw user
        content or full log lines.

        Parameters
        ----------
        filepath : Path
            Path to the .txt debug log file.

        Returns
        -------
        DebugRecord or None if the file cannot be read.
        """
        session_id = filepath.stem

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except (IOError, OSError):
            return None

        # Accumulators
        api_call_count = 0
        models_used: Dict[str, bool] = {}
        token_snapshots: List[int] = []
        system_prompts: List[str] = []
        errors: List[str] = []
        latencies_ms: List[float] = []

        first_timestamp: Optional[str] = None
        last_timestamp: Optional[str] = None

        # Track pending API request timestamps to compute latency
        pending_api_request_ts: Optional[datetime] = None

        for line in content.splitlines():
            if not line:
                continue

            # Extract timestamp from line
            ts_match = _TS_RE.match(line)
            line_ts_str: Optional[str] = None
            line_ts: Optional[datetime] = None
            if ts_match:
                line_ts_str = ts_match.group(1)
                line_ts = _parse_timestamp(line_ts_str)
                if first_timestamp is None:
                    first_timestamp = line_ts_str
                last_timestamp = line_ts_str

            # API request detection
            if _API_REQUEST_RE.search(line):
                api_call_count += 1
                pending_api_request_ts = line_ts

            # Stream start -- compute latency from last API request
            if _STREAM_START_RE.search(line):
                if pending_api_request_ts is not None and line_ts is not None:
                    delta = (line_ts - pending_api_request_ts).total_seconds()
                    latencies_ms.append(round(delta * 1000))
                    pending_api_request_ts = None

            # Model extraction
            model_match = _MODEL_RE.search(line)
            if model_match:
                models_used[model_match.group(1)] = True

            # Token counts
            ac_match = _AUTOCOMPACT_RE.search(line)
            if ac_match:
                token_snapshots.append(int(ac_match.group(1)))

            # System prompt entries -- store the config string, not content
            sp_match = _SYSTEM_PROMPT_RE.search(line)
            if sp_match:
                system_prompts.append(sp_match.group(1).strip())

            # Error lines -- store the error message, truncated
            if _ERROR_RE.search(line):
                err_match = _ERROR_RE.search(line)
                if err_match:
                    error_msg = err_match.group(1).strip()
                    # Truncate long error messages
                    if len(error_msg) > 500:
                        error_msg = error_msg[:500] + "..."
                    errors.append(error_msg)

        # Compute summary stats
        total_latency_ms = sum(latencies_ms) if latencies_ms else 0
        max_tokens_seen = max(token_snapshots) if token_snapshots else 0

        # Build timestamp from first log entry or file modification time
        if first_timestamp:
            timestamp = first_timestamp + "Z" if not first_timestamp.endswith("Z") else first_timestamp
        else:
            try:
                mtime = filepath.stat().st_mtime
                timestamp = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:
                timestamp = datetime.now(timezone.utc).isoformat()

        return DebugRecord(
            session_id=session_id,
            timestamp=timestamp,
            source_type=self.source_type,
            metadata={
                "models_used": sorted(models_used.keys()),
                "max_tokens_seen": max_tokens_seen,
                "token_snapshots": token_snapshots,
                "latencies_ms": latencies_ms,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
            },
            system_prompts=system_prompts,
            full_thinking_blocks=[],  # No thinking blocks in current log format
            api_call_count=api_call_count,
            total_latency_ms=int(total_latency_ms),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Public interface (BaseCollector)
    # ------------------------------------------------------------------

    def scan(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorScanResult:
        """Scan for debug log files without collecting anything."""
        files = self._find_debug_files(since=since)
        collected_ids = self._load_collected_ids()

        total_found = len(files)
        already_collected = 0
        estimated_bytes = 0

        for filepath in files:
            session_id = filepath.stem
            if session_id in collected_ids:
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

    def collect(self, session_id: str) -> List[DebugRecord]:
        """Collect debug metadata for a single session.

        Parameters
        ----------
        session_id : str
            The session ID (filename stem) to collect.

        Returns
        -------
        list of DebugRecord (empty if the file doesn't exist)
        """
        filepath = self._debug_dir() / f"{session_id}.txt"
        if not filepath.exists():
            return []

        record = self._parse_debug_log(filepath)
        if record is None:
            return []

        return [record]

    def collect_all(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        """Collect debug metadata for all uncollected sessions.

        Deduplicates by session_id (the filename stem).
        """
        files = self._find_debug_files(since=since)
        collected_ids = self._load_collected_ids()

        collected = 0
        skipped = 0
        errors: List[str] = []
        records: List[DebugRecord] = []

        for filepath in files:
            session_id = filepath.stem

            if session_id in collected_ids:
                skipped += 1
                continue

            try:
                record = self._parse_debug_log(filepath)
                if record is not None:
                    records.append(record)
                    self._save_collected_id(session_id)
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
