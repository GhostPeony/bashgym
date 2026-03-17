"""
Per-Key Shared Memory for Orchestrator Workers

Provides fine-grained async-locked shared state so parallel workers
can share discoveries, detect conflicts, and coordinate in real-time.

Features:
- Per-key async locking (no global mutex bottleneck)
- Scoped views with read/write permissions per worker
- Change history for audit trail
- Conflict detection for overlapping output keys
- File-based IPC protocol for Claude Code worker communication

Uses a per-key locked SharedMemory pattern. Workers run in isolated git
worktrees but share state through this module. The orchestrator polls
worktree state files and merges discoveries back into the central store.

Module: Orchestrator
"""

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# File names for the IPC protocol
STATE_SNAPSHOT_FILE = "shared_state.json"
STATE_OUTBOX_FILE = "shared_state.jsonl"


@dataclass
class StateChange:
    """Record of a single state mutation."""
    key: str
    value: Any
    writer_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "writer_id": self.writer_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConflictInfo:
    """Describes a write-key overlap between parallel tasks."""
    key: str
    writers: List[str]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "writers": self.writers,
            "description": self.description,
        }


class ScopedView:
    """Permission-restricted view of SharedState.

    Shares the same underlying data store but enforces read/write
    permissions. A wildcard ("*") grants access to all keys.
    """

    def __init__(
        self,
        state: "SharedState",
        read_keys: Set[str],
        write_keys: Set[str],
    ):
        self._state = state
        self._read_keys = read_keys
        self._write_keys = write_keys

    async def read(self, key: str) -> Optional[Any]:
        """Read a single key, enforcing read permission."""
        if key not in self._read_keys and "*" not in self._read_keys:
            raise PermissionError(f"No read access to key: {key}")
        return await self._state.read(key)

    async def write(self, key: str, value: Any, writer_id: str) -> None:
        """Write a single key, enforcing write permission."""
        if key not in self._write_keys and "*" not in self._write_keys:
            raise PermissionError(f"No write access to key: {key}")
        await self._state.write(key, value, writer_id)

    async def read_all(self) -> Dict[str, Any]:
        """Read all keys this view has access to."""
        snapshot = self._state.snapshot()
        if "*" in self._read_keys:
            return snapshot
        return {k: v for k, v in snapshot.items() if k in self._read_keys}


class SharedState:
    """Per-key locked shared memory for parallel orchestrator workers.

    Features:
    - Per-key async locking (fine-grained, not global mutex)
    - Scoped views with read/write permissions per worker
    - Change history for audit trail
    - Conflict detection for overlapping output keys
    - File-based protocol for Claude Code worker IPC

    Usage:
        state = SharedState()

        # Direct read/write
        await state.write("api.schema", {"endpoints": [...]}, writer_id="task-1")
        schema = await state.read("api.schema")

        # Scoped views for workers
        view = state.scoped_view(
            read_keys={"api.schema", "shared.config"},
            write_keys={"models.user"},
        )
        await view.write("models.user", {...}, writer_id="task-2")

        # Conflict detection before dispatch
        conflicts = state.detect_conflicts({
            "task-1": {"api.schema", "api.routes"},
            "task-2": {"api.routes", "models.user"},
        })
        # => [ConflictInfo(key="api.routes", writers=["task-1", "task-2"], ...)]
    """

    def __init__(self, max_history: int = 1000):
        self._data: Dict[str, Any] = {}
        self._key_locks: Dict[str, asyncio.Lock] = {}
        self._lock_guard = asyncio.Lock()  # Protects _key_locks dict creation
        self._history: List[StateChange] = []
        self._max_history = max_history
        # Track which lines have been read per worktree to avoid re-processing
        self._poll_offsets: Dict[str, int] = {}

    # =========================================================================
    # Per-Key Locking
    # =========================================================================

    async def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a key (async-safe)."""
        if key in self._key_locks:
            return self._key_locks[key]
        async with self._lock_guard:
            # Double-checked: another coroutine may have created it
            if key not in self._key_locks:
                self._key_locks[key] = asyncio.Lock()
            return self._key_locks[key]

    # =========================================================================
    # Read / Write
    # =========================================================================

    async def read(self, key: str) -> Optional[Any]:
        """Read a value. No lock needed (eventual consistency is acceptable)."""
        return deepcopy(self._data.get(key))

    async def write(self, key: str, value: Any, writer_id: str) -> None:
        """Write with per-key lock. Records change in history and emits event."""
        lock = await self._get_lock(key)
        async with lock:
            self._data[key] = deepcopy(value)
            change = StateChange(key=key, value=value, writer_id=writer_id)
            self._history.append(change)

            # Trim history if over limit
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Emit event (non-blocking, outside the lock)
        self._emit_state_changed(change)

    async def write_batch(
        self,
        updates: Dict[str, Any],
        writer_id: str,
    ) -> None:
        """Write multiple keys atomically.

        Acquires all locks in sorted key order to prevent deadlocks.
        All writes succeed or none do (best-effort atomicity within
        a single asyncio event loop).
        """
        if not updates:
            return

        # Acquire locks in sorted order to prevent deadlocks
        sorted_keys = sorted(updates.keys())
        locks = [await self._get_lock(k) for k in sorted_keys]

        # Acquire all locks
        for lock in locks:
            await lock.acquire()

        try:
            changes = []
            for key in sorted_keys:
                value = updates[key]
                self._data[key] = deepcopy(value)
                change = StateChange(key=key, value=value, writer_id=writer_id)
                self._history.append(change)
                changes.append(change)

            # Trim history if over limit
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        finally:
            # Release all locks in reverse order
            for lock in reversed(locks):
                lock.release()

        # Emit events outside locks
        for change in changes:
            self._emit_state_changed(change)

    # =========================================================================
    # Scoped Views
    # =========================================================================

    def scoped_view(
        self,
        read_keys: Set[str],
        write_keys: Set[str],
    ) -> ScopedView:
        """Create a permission-restricted view for a worker."""
        return ScopedView(self, read_keys=read_keys, write_keys=write_keys)

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    def detect_conflicts(
        self,
        task_write_keys: Dict[str, Set[str]],
    ) -> List[ConflictInfo]:
        """Detect overlapping write keys between parallel tasks.

        Args:
            task_write_keys: Mapping of {task_id: set of keys this task writes}.

        Returns:
            List of ConflictInfo for keys written by multiple tasks.
        """
        # Invert: key -> list of writers
        key_writers: Dict[str, List[str]] = {}
        for task_id, keys in task_write_keys.items():
            for key in keys:
                key_writers.setdefault(key, []).append(task_id)

        conflicts = []
        for key, writers in sorted(key_writers.items()):
            if len(writers) > 1:
                conflicts.append(ConflictInfo(
                    key=key,
                    writers=sorted(writers),
                    description=(
                        f"Key '{key}' is written by {len(writers)} tasks: "
                        f"{', '.join(sorted(writers))}"
                    ),
                ))

        return conflicts

    # =========================================================================
    # History & Snapshots
    # =========================================================================

    def get_history(
        self,
        key: Optional[str] = None,
        writer_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[StateChange]:
        """Query change history with optional filters."""
        results = self._history

        if key is not None:
            results = [c for c in results if c.key == key]

        if writer_id is not None:
            results = [c for c in results if c.writer_id == writer_id]

        # Return most recent entries up to limit
        return list(results[-limit:])

    def snapshot(self) -> Dict[str, Any]:
        """Return a frozen (deep-copied) snapshot of all state."""
        return deepcopy(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API/WebSocket."""
        return {
            "data": self.snapshot(),
            "key_count": len(self._data),
            "history_length": len(self._history),
            "recent_changes": [
                c.to_dict() for c in self._history[-10:]
            ],
        }

    # =========================================================================
    # File-Based IPC Protocol
    # =========================================================================

    async def poll_worktree_state(
        self,
        worktree_path: str,
        task_id: str,
    ) -> int:
        """Poll a worktree's shared_state.jsonl file for new entries.

        Workers write discoveries as JSON lines to their worktree's
        STATE_OUTBOX_FILE. The orchestrator polls and merges new entries
        into the central SharedState.

        Each line is a JSON object: {"key": "...", "value": ...}

        Args:
            worktree_path: Path to the worker's git worktree
            task_id: ID of the task that owns this worktree

        Returns:
            Number of new entries merged
        """
        outbox = Path(worktree_path) / STATE_OUTBOX_FILE
        if not outbox.exists():
            return 0

        offset = self._poll_offsets.get(worktree_path, 0)
        merged = 0

        try:
            with open(outbox, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Process only lines we haven't seen
            new_lines = lines[offset:]
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry.get("key")
                    value = entry.get("value")
                    if key is not None:
                        await self.write(key, value, writer_id=task_id)
                        merged += 1
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON in {outbox} line {offset + merged}: "
                        f"{line[:100]}"
                    )

            self._poll_offsets[worktree_path] = len(lines)

        except OSError as e:
            logger.warning(f"Failed to read state outbox at {outbox}: {e}")

        if merged > 0:
            logger.debug(
                f"Merged {merged} state entries from {task_id} "
                f"({worktree_path})"
            )

        return merged

    async def write_state_file(self, worktree_path: str) -> None:
        """Write current state snapshot to worktree for worker to read.

        The worker can read STATE_SNAPSHOT_FILE at any point to see
        the full shared state from the orchestrator's perspective.
        """
        snapshot_path = Path(worktree_path) / STATE_SNAPSHOT_FILE
        snapshot = self.snapshot()

        try:
            content = json.dumps(snapshot, indent=2, default=str)
            snapshot_path.write_text(content, encoding="utf-8")
            logger.debug(
                f"Wrote state snapshot ({len(snapshot)} keys) "
                f"to {snapshot_path}"
            )
        except OSError as e:
            logger.warning(f"Failed to write state snapshot to {snapshot_path}: {e}")

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _emit_state_changed(self, change: StateChange) -> None:
        """Emit a state change event on the global EventBus."""
        try:
            from bashgym.events.bus import event_bus
            from bashgym.events.types import SharedStateChanged
            event_bus.emit(SharedStateChanged(
                key=change.key,
                writer_id=change.writer_id,
                value_preview=self._preview_value(change.value),
            ))
        except ImportError:
            pass  # Events module not available
        except Exception:
            logger.debug("Failed to emit state change event", exc_info=True)

    @staticmethod
    def _preview_value(value: Any, max_len: int = 200) -> str:
        """Create a short string preview of a value for event payloads."""
        try:
            s = json.dumps(value, default=str)
        except (TypeError, ValueError):
            s = repr(value)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s
