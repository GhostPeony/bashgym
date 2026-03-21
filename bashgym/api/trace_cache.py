"""In-memory trace metadata index cache.

Parses all trace files once on first access, then serves from memory.
Only re-parses files that are new or modified (tracked by mtime).
Invalidated explicitly by promote/demote/classify operations.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class TraceIndexCache:
    """Caches parsed trace metadata to avoid re-reading every JSON file per request."""

    def __init__(self):
        # trace_id -> (TraceInfo dict, file_path, file_mtime, status_dir)
        self._entries: dict[str, dict] = {}
        self._file_mtimes: dict[str, float] = {}  # filepath -> mtime
        self._dir_snapshots: dict[str, set] = {}  # dir_path -> set of filenames
        self._initialized = False
        self._last_scan_time = 0.0
        self._min_scan_interval = 2.0  # seconds between full dir scans

    @property
    def initialized(self) -> bool:
        return self._initialized

    def build_index(
        self,
        tier_dirs: list[tuple[Path, str]],
        pending_dirs: list[Path],
        parse_tiered_fn,
        parse_pending_fn,
    ):
        """Full index build on startup. Called once."""
        start = time.time()
        count = 0

        for tier_dir, tier_status in tier_dirs:
            if not tier_dir.exists():
                continue
            files = list(tier_dir.glob("*.json"))
            self._dir_snapshots[str(tier_dir)] = {f.name for f in files}
            for trace_file in files:
                try:
                    mtime = trace_file.stat().st_mtime
                    with open(trace_file, encoding="utf-8") as f:
                        data = json.load(f)
                    info = parse_tiered_fn(trace_file, data, tier_status)
                    self._store(trace_file, info, mtime)
                    count += 1
                except Exception:
                    continue

        from bashgym.trace_capture.core import glob_pending_traces, load_trace_file

        seen = set()
        for pending_dir in pending_dirs:
            if not pending_dir.exists():
                continue
            files = glob_pending_traces(pending_dir)
            self._dir_snapshots[str(pending_dir)] = {f.name for f in files}
            for trace_file in files:
                if trace_file.name in seen:
                    continue
                seen.add(trace_file.name)
                try:
                    mtime = trace_file.stat().st_mtime
                    data = load_trace_file(trace_file)
                    info = parse_pending_fn(trace_file, data)
                    if info is not None:
                        self._store(trace_file, info, mtime)
                        count += 1
                except Exception:
                    continue

        self._initialized = True
        self._last_scan_time = time.time()
        elapsed = time.time() - start
        logger.info(f"Trace index built: {count} traces in {elapsed:.1f}s")

    def refresh(
        self,
        tier_dirs: list[tuple[Path, str]],
        pending_dirs: list[Path],
        parse_tiered_fn,
        parse_pending_fn,
    ):
        """Incremental refresh — only parse new/modified files, remove deleted ones."""
        now = time.time()
        if now - self._last_scan_time < self._min_scan_interval:
            return
        self._last_scan_time = now

        added = 0
        removed = 0

        for tier_dir, tier_status in tier_dirs:
            if not tier_dir.exists():
                continue
            current_files = {f.name: f for f in tier_dir.glob("*.json")}
            prev_files = self._dir_snapshots.get(str(tier_dir), set())

            # New files
            for name in current_files.keys() - prev_files:
                trace_file = current_files[name]
                try:
                    mtime = trace_file.stat().st_mtime
                    with open(trace_file, encoding="utf-8") as f:
                        data = json.load(f)
                    info = parse_tiered_fn(trace_file, data, tier_status)
                    self._store(trace_file, info, mtime)
                    added += 1
                except Exception:
                    continue

            # Deleted files
            for name in prev_files - current_files.keys():
                tid = Path(name).stem
                self._entries.pop(tid, None)
                removed += 1

            self._dir_snapshots[str(tier_dir)] = set(current_files.keys())

        from bashgym.trace_capture.core import glob_pending_traces, load_trace_file

        seen = set()
        for pending_dir in pending_dirs:
            if not pending_dir.exists():
                continue
            files = glob_pending_traces(pending_dir)
            current_files = {f.name: f for f in files}
            prev_files = self._dir_snapshots.get(str(pending_dir), set())

            for name in current_files.keys() - prev_files:
                if name in seen:
                    continue
                seen.add(name)
                trace_file = current_files[name]
                try:
                    mtime = trace_file.stat().st_mtime
                    data = load_trace_file(trace_file)
                    info = parse_pending_fn(trace_file, data)
                    if info is not None:
                        self._store(trace_file, info, mtime)
                        added += 1
                except Exception:
                    continue

            for name in prev_files - current_files.keys():
                tid = Path(name).stem
                self._entries.pop(tid, None)
                removed += 1

            self._dir_snapshots[str(pending_dir)] = set(current_files.keys())

        if added or removed:
            logger.info(f"Trace index refreshed: +{added} -{removed} (total: {len(self._entries)})")

    def invalidate(self, trace_id: str | None = None):
        """Invalidate a specific trace or the entire cache."""
        if trace_id:
            self._entries.pop(trace_id, None)
        else:
            self._entries.clear()
            self._dir_snapshots.clear()
            self._file_mtimes.clear()
            self._initialized = False

    def query(
        self,
        status: str | None = None,
        repo: str | None = None,
        source_tool: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Query the cache with filters, sorting, and pagination."""
        filtered = []
        counts = {"gold": 0, "silver": 0, "bronze": 0, "failed": 0, "pending": 0}

        for info in self._entries.values():
            s = info.status if hasattr(info, "status") else info.get("status")
            s_val = s.value if hasattr(s, "value") else str(s)

            # Count all traces by status (before filtering)
            if s_val in counts:
                counts[s_val] += 1

            # Apply status filter
            if status is not None:
                status_val = status.value if hasattr(status, "value") else str(status)
                if s_val != status_val:
                    continue

            # Apply repo filter
            if repo:
                r = info.repo if hasattr(info, "repo") else info.get("repo")
                if not _repo_matches(r, repo):
                    continue

            # Apply source_tool filter
            st = info.source_tool if hasattr(info, "source_tool") else info.get("source_tool")
            if source_tool and st != source_tool:
                continue

            filtered.append(info)

        # Sort by created_at descending
        def sort_key(t):
            ca = t.created_at if hasattr(t, "created_at") else t.get("created_at")
            return ca or ""

        filtered.sort(key=sort_key, reverse=True)

        total = len(filtered)
        page = filtered[offset : offset + limit]

        return {
            "traces": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "counts": counts,
        }

    def _store(self, trace_file: Path, info, mtime: float):
        tid = info.trace_id if hasattr(info, "trace_id") else info.get("trace_id")
        self._entries[tid] = info
        self._file_mtimes[str(trace_file)] = mtime


def _repo_matches(repo_info, filter_str: str) -> bool:
    """Check if a repo info object matches the filter string."""
    if repo_info is None:
        return False
    f = filter_str.lower()
    name = (
        repo_info.name
        if hasattr(repo_info, "name")
        else (repo_info.get("name") if isinstance(repo_info, dict) else "")
    )
    path = (
        repo_info.path
        if hasattr(repo_info, "path")
        else (repo_info.get("path") if isinstance(repo_info, dict) else "")
    )
    remote = (
        repo_info.git_remote
        if hasattr(repo_info, "git_remote")
        else (repo_info.get("git_remote") if isinstance(repo_info, dict) else "")
    )
    return (
        (name and f in str(name).lower())
        or (path and f in str(path).lower())
        or (remote and f in str(remote).lower())
    )
