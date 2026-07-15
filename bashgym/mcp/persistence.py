"""SQLite persistence for the MCP Workbench control plane.

The repository deliberately uses composite ``(workspace_id, record_id)`` keys
and requires the workspace for every lookup.  Looking up another workspace's
identifier is therefore indistinguishable from looking up a missing record.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bashgym.mcp.contracts import (
    McpCapabilitySnapshot,
    McpOperation,
    McpProfile,
    McpProfileRevision,
    McpSession,
    McpStdioLaunchApproval,
    OperationState,
    RecoverySummary,
    SessionState,
    StdioTransport,
    utc_now,
)
from bashgym.mcp.operations import UNSET_OPERATION_RESULT, transitioned_operation


class McpPersistenceError(RuntimeError):
    """Base error for MCP repository failures."""


class RecordNotFoundError(McpPersistenceError):
    """Raised for both missing records and wrong-workspace lookups."""


class RecordAlreadyExistsError(McpPersistenceError):
    """Raised when an owned identifier is already present."""


class RevisionConflictError(McpPersistenceError):
    """Raised when an optimistic update targets a stale revision."""

    def __init__(self, record_type: str, expected: int, current: int):
        self.record_type = record_type
        self.expected = expected
        self.current = current
        super().__init__(f"{record_type} revision conflict: expected {expected}, current {current}")


class ProfileInUseError(McpPersistenceError):
    """Raised when a profile cannot be tombstoned while work is active."""


_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS mcp_profiles (
            workspace_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 1),
            label TEXT NOT NULL,
            transport_type TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT,
            PRIMARY KEY (workspace_id, profile_id)
        );

        CREATE TABLE IF NOT EXISTS mcp_profile_revisions (
            workspace_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 1),
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (workspace_id, profile_id, revision),
            FOREIGN KEY (workspace_id, profile_id)
                REFERENCES mcp_profiles(workspace_id, profile_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_mcp_profiles_workspace_updated
            ON mcp_profiles(workspace_id, updated_at DESC);
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS mcp_snapshots (
            workspace_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            profile_revision INTEGER NOT NULL CHECK (profile_revision >= 1),
            contract_hash TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
            drifted INTEGER NOT NULL DEFAULT 0 CHECK (drifted IN (0, 1)),
            PRIMARY KEY (workspace_id, snapshot_id),
            FOREIGN KEY (workspace_id, profile_id, profile_revision)
                REFERENCES mcp_profile_revisions(workspace_id, profile_id, revision)
                ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_mcp_snapshots_profile_captured
            ON mcp_snapshots(workspace_id, profile_id, captured_at DESC);

        CREATE TABLE IF NOT EXISTS mcp_sessions (
            workspace_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            profile_revision INTEGER NOT NULL CHECK (profile_revision >= 1),
            snapshot_id TEXT,
            state TEXT NOT NULL,
            stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
            revision INTEGER NOT NULL CHECK (revision >= 1),
            session_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            disconnected_at TEXT,
            PRIMARY KEY (workspace_id, session_id),
            FOREIGN KEY (workspace_id, profile_id, profile_revision)
                REFERENCES mcp_profile_revisions(workspace_id, profile_id, revision)
                ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id, snapshot_id)
                REFERENCES mcp_snapshots(workspace_id, snapshot_id) ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS idx_mcp_sessions_profile_state
            ON mcp_sessions(workspace_id, profile_id, state);

        CREATE TABLE IF NOT EXISTS mcp_operations (
            workspace_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            correlation_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            state TEXT NOT NULL,
            profile_id TEXT,
            session_id TEXT,
            idempotency_key TEXT,
            retry_of TEXT,
            revision INTEGER NOT NULL CHECK (revision >= 1),
            error_code TEXT,
            safe_message TEXT,
            operation_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            PRIMARY KEY (workspace_id, operation_id),
            FOREIGN KEY (workspace_id, profile_id)
                REFERENCES mcp_profiles(workspace_id, profile_id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id, session_id)
                REFERENCES mcp_sessions(workspace_id, session_id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id, retry_of)
                REFERENCES mcp_operations(workspace_id, operation_id) ON DELETE RESTRICT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_operations_idempotency
            ON mcp_operations(workspace_id, kind, idempotency_key)
            WHERE idempotency_key IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_mcp_operations_workspace_updated
            ON mcp_operations(workspace_id, updated_at DESC);
        """,
    ),
    (
        3,
        """
        ALTER TABLE mcp_operations ADD COLUMN result_json TEXT;
        """,
    ),
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS mcp_stdio_launch_approvals (
            workspace_id TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            profile_revision INTEGER NOT NULL CHECK (profile_revision >= 1),
            executable_fingerprint TEXT NOT NULL,
            launch_fingerprint TEXT NOT NULL,
            fingerprint_json TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            PRIMARY KEY (workspace_id, profile_id, profile_revision),
            FOREIGN KEY (workspace_id, profile_id, profile_revision)
                REFERENCES mcp_profile_revisions(workspace_id, profile_id, revision)
                ON DELETE RESTRICT
        );
        """,
    ),
)

_ACTIVE_OPERATION_STATES = (
    OperationState.QUEUED.value,
    OperationState.RUNNING.value,
    OperationState.WAITING_FOR_APPROVAL.value,
)
_LIVE_SESSION_STATES = (
    SessionState.CONNECTING.value,
    SessionState.CONNECTED.value,
    SessionState.DISCONNECTING.value,
)

_SESSION_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CONNECTING: frozenset(
        {SessionState.CONNECTED, SessionState.DISCONNECTED, SessionState.FAILED}
    ),
    SessionState.CONNECTED: frozenset(
        {SessionState.DISCONNECTING, SessionState.DISCONNECTED, SessionState.FAILED}
    ),
    SessionState.DISCONNECTING: frozenset({SessionState.DISCONNECTED, SessionState.FAILED}),
    SessionState.DISCONNECTED: frozenset(),
    SessionState.FAILED: frozenset(),
}


def _json(model: McpProfile | McpCapabilitySnapshot | McpSession | McpOperation) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


class McpRepository:
    """Workspace-scoped repository backed by one injected SQLite path."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._initialized = False

    @contextmanager
    def _connection(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> RecoverySummary:
        """Apply numbered migrations and make crash-interrupted state safe."""

        if self._initialized:
            return RecoverySummary(
                operations_interrupted=0,
                sessions_disconnected=0,
                snapshots_marked_stale=0,
            )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection(immediate=True) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS mcp_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """)
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM mcp_schema_migrations")
            }
            for version, sql in _MIGRATIONS:
                if version in applied:
                    continue
                connection.executescript(sql)
                connection.execute(
                    "INSERT INTO mcp_schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now().isoformat()),
                )
        self._initialized = True
        return self.recover_after_restart()

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("McpRepository.initialize() must be called first")

    def schema_versions(self) -> list[int]:
        self._require_initialized()
        with self._connection() as connection:
            return [
                row["version"]
                for row in connection.execute(
                    "SELECT version FROM mcp_schema_migrations ORDER BY version"
                )
            ]

    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> McpProfile:
        return McpProfile.model_validate_json(row["profile_json"])

    @staticmethod
    def _snapshot_from_row(row: sqlite3.Row) -> McpCapabilitySnapshot:
        payload = json.loads(row["snapshot_json"])
        payload["stale"] = bool(row["stale"])
        payload["drifted"] = bool(row["drifted"])
        return McpCapabilitySnapshot.model_validate(payload)

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> McpSession:
        payload = json.loads(row["session_json"])
        payload.update(
            {
                "state": row["state"],
                "stale": bool(row["stale"]),
                "revision": row["revision"],
                "updated_at": row["updated_at"],
                "disconnected_at": row["disconnected_at"],
            }
        )
        return McpSession.model_validate(payload)

    @staticmethod
    def _operation_from_row(row: sqlite3.Row) -> McpOperation:
        payload = json.loads(row["operation_json"])
        payload.update(
            {
                "state": row["state"],
                "revision": row["revision"],
                "error_code": row["error_code"],
                "safe_message": row["safe_message"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "updated_at": row["updated_at"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
            }
        )
        return McpOperation.model_validate(payload)

    @staticmethod
    def _approval_from_row(row: sqlite3.Row) -> McpStdioLaunchApproval:
        return McpStdioLaunchApproval.model_validate(
            {
                "workspace_id": row["workspace_id"],
                "profile_id": row["profile_id"],
                "profile_revision": row["profile_revision"],
                "executable_fingerprint": row["executable_fingerprint"],
                "launch_fingerprint": row["launch_fingerprint"],
                "fingerprint": json.loads(row["fingerprint_json"]),
                "approved_at": row["approved_at"],
            }
        )

    def create_profile(self, profile: McpProfile) -> McpProfile:
        self._require_initialized()
        if profile.revision != 1 or profile.deleted_at is not None:
            raise ValueError("A new profile must start at revision 1 and cannot be deleted")
        with self._connection(immediate=True) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO mcp_profiles(
                        workspace_id, profile_id, revision, label, transport_type,
                        profile_json, enabled, created_at, updated_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.workspace_id,
                        profile.profile_id,
                        profile.revision,
                        profile.label,
                        profile.transport.type,
                        _json(profile),
                        int(profile.enabled),
                        _iso(profile.created_at),
                        _iso(profile.updated_at),
                        None,
                    ),
                )
                self._insert_profile_revision(connection, profile)
            except sqlite3.IntegrityError as exc:
                raise RecordAlreadyExistsError("profile already exists") from exc
        return profile

    @staticmethod
    def _insert_profile_revision(connection: sqlite3.Connection, profile: McpProfile) -> None:
        connection.execute(
            """
            INSERT INTO mcp_profile_revisions(
                workspace_id, profile_id, revision, profile_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                profile.workspace_id,
                profile.profile_id,
                profile.revision,
                _json(profile),
                utc_now().isoformat(),
            ),
        )

    def get_profile(
        self, workspace_id: str, profile_id: str, *, include_deleted: bool = False
    ) -> McpProfile:
        self._require_initialized()
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._connection() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM mcp_profiles
                WHERE workspace_id = ? AND profile_id = ?{deleted_clause}
                """,
                (workspace_id, profile_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("profile not found")
        return self._profile_from_row(row)

    def list_profiles(
        self, workspace_id: str, *, include_deleted: bool = False
    ) -> list[McpProfile]:
        self._require_initialized()
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM mcp_profiles
                WHERE workspace_id = ?{deleted_clause}
                ORDER BY updated_at DESC, profile_id
                """,
                (workspace_id,),
            ).fetchall()
        return [self._profile_from_row(row) for row in rows]

    def get_profile_revision(
        self, workspace_id: str, profile_id: str, revision: int
    ) -> McpProfileRevision:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_profile_revisions
                WHERE workspace_id = ? AND profile_id = ? AND revision = ?
                """,
                (workspace_id, profile_id, revision),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("profile revision not found")
        profile = McpProfile.model_validate_json(row["profile_json"])
        return McpProfileRevision(
            workspace_id=workspace_id,
            profile_id=profile_id,
            revision=revision,
            profile=profile,
            created_at=row["created_at"],
        )

    def update_profile(self, profile: McpProfile, *, expected_revision: int) -> McpProfile:
        self._require_initialized()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_profiles
                WHERE workspace_id = ? AND profile_id = ? AND deleted_at IS NULL
                """,
                (profile.workspace_id, profile.profile_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("profile not found")
            current = self._profile_from_row(row)
            if current.revision != expected_revision:
                raise RevisionConflictError("profile", expected_revision, current.revision)
            updated = McpProfile.model_validate(
                profile.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "created_at": current.created_at,
                        "updated_at": utc_now(),
                        "deleted_at": None,
                    }
                ).model_dump()
            )
            cursor = connection.execute(
                """
                UPDATE mcp_profiles
                SET revision = ?, label = ?, transport_type = ?, profile_json = ?,
                    enabled = ?, updated_at = ?
                WHERE workspace_id = ? AND profile_id = ? AND revision = ?
                    AND deleted_at IS NULL
                """,
                (
                    updated.revision,
                    updated.label,
                    updated.transport.type,
                    _json(updated),
                    int(updated.enabled),
                    _iso(updated.updated_at),
                    updated.workspace_id,
                    updated.profile_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError("profile", expected_revision, current.revision)
            self._insert_profile_revision(connection, updated)
        return updated

    def tombstone_profile(
        self, workspace_id: str, profile_id: str, *, expected_revision: int
    ) -> McpProfile:
        self._require_initialized()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_profiles
                WHERE workspace_id = ? AND profile_id = ? AND deleted_at IS NULL
                """,
                (workspace_id, profile_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("profile not found")
            current = self._profile_from_row(row)
            if current.revision != expected_revision:
                raise RevisionConflictError("profile", expected_revision, current.revision)
            placeholders = ",".join("?" for _ in _ACTIVE_OPERATION_STATES)
            active_operation = connection.execute(
                f"""
                SELECT 1 FROM mcp_operations
                WHERE workspace_id = ? AND profile_id = ?
                    AND state IN ({placeholders}) LIMIT 1
                """,
                (workspace_id, profile_id, *_ACTIVE_OPERATION_STATES),
            ).fetchone()
            live_session_placeholders = ",".join("?" for _ in _LIVE_SESSION_STATES)
            live_session = connection.execute(
                f"""
                SELECT 1 FROM mcp_sessions
                WHERE workspace_id = ? AND profile_id = ?
                    AND state IN ({live_session_placeholders}) LIMIT 1
                """,
                (workspace_id, profile_id, *_LIVE_SESSION_STATES),
            ).fetchone()
            if active_operation or live_session:
                raise ProfileInUseError("profile has active sessions or operations")

            now = utc_now()
            deleted = McpProfile.model_validate(
                current.model_copy(
                    update={
                        "enabled": False,
                        "revision": current.revision + 1,
                        "updated_at": now,
                        "deleted_at": now,
                    }
                ).model_dump()
            )
            connection.execute(
                """
                UPDATE mcp_profiles
                SET revision = ?, enabled = 0, profile_json = ?, updated_at = ?, deleted_at = ?
                WHERE workspace_id = ? AND profile_id = ? AND revision = ?
                """,
                (
                    deleted.revision,
                    _json(deleted),
                    _iso(deleted.updated_at),
                    _iso(deleted.deleted_at),
                    workspace_id,
                    profile_id,
                    expected_revision,
                ),
            )
            self._insert_profile_revision(connection, deleted)
        return deleted

    def save_snapshot(self, snapshot: McpCapabilitySnapshot) -> McpCapabilitySnapshot:
        self._require_initialized()
        with self._connection(immediate=True) as connection:
            revision = connection.execute(
                """
                SELECT 1 FROM mcp_profile_revisions
                WHERE workspace_id = ? AND profile_id = ? AND revision = ?
                """,
                (snapshot.workspace_id, snapshot.profile_id, snapshot.profile_revision),
            ).fetchone()
            if revision is None:
                raise RecordNotFoundError("profile revision not found")
            try:
                connection.execute(
                    """
                    INSERT INTO mcp_snapshots(
                        workspace_id, snapshot_id, profile_id, profile_revision,
                        contract_hash, snapshot_json, captured_at, stale, drifted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.workspace_id,
                        snapshot.snapshot_id,
                        snapshot.profile_id,
                        snapshot.profile_revision,
                        snapshot.contract_hash,
                        _json(snapshot),
                        _iso(snapshot.captured_at),
                        int(snapshot.stale),
                        int(snapshot.drifted),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RecordAlreadyExistsError("snapshot already exists") from exc
        return snapshot

    def get_snapshot(self, workspace_id: str, snapshot_id: str) -> McpCapabilitySnapshot:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_snapshots
                WHERE workspace_id = ? AND snapshot_id = ?
                """,
                (workspace_id, snapshot_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("snapshot not found")
        return self._snapshot_from_row(row)

    def latest_snapshot(self, workspace_id: str, profile_id: str) -> McpCapabilitySnapshot:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_snapshots
                WHERE workspace_id = ? AND profile_id = ?
                ORDER BY captured_at DESC, snapshot_id DESC LIMIT 1
                """,
                (workspace_id, profile_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("snapshot not found")
        return self._snapshot_from_row(row)

    def create_session(self, session: McpSession) -> McpSession:
        self._require_initialized()
        if session.revision != 1:
            raise ValueError("A new session must start at revision 1")
        with self._connection(immediate=True) as connection:
            revision = connection.execute(
                """
                SELECT 1 FROM mcp_profile_revisions
                WHERE workspace_id = ? AND profile_id = ? AND revision = ?
                """,
                (session.workspace_id, session.profile_id, session.profile_revision),
            ).fetchone()
            if revision is None:
                raise RecordNotFoundError("profile revision not found")
            if session.snapshot_id is not None:
                snapshot = connection.execute(
                    """
                    SELECT 1 FROM mcp_snapshots
                    WHERE workspace_id = ? AND snapshot_id = ? AND profile_id = ?
                        AND profile_revision = ?
                    """,
                    (
                        session.workspace_id,
                        session.snapshot_id,
                        session.profile_id,
                        session.profile_revision,
                    ),
                ).fetchone()
                if snapshot is None:
                    raise RecordNotFoundError("snapshot not found")
            try:
                connection.execute(
                    """
                    INSERT INTO mcp_sessions(
                        workspace_id, session_id, profile_id, profile_revision, snapshot_id,
                        state, stale, revision, session_json, created_at, updated_at, disconnected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.workspace_id,
                        session.session_id,
                        session.profile_id,
                        session.profile_revision,
                        session.snapshot_id,
                        session.state.value,
                        int(session.stale),
                        session.revision,
                        _json(session),
                        _iso(session.created_at),
                        _iso(session.updated_at),
                        _iso(session.disconnected_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RecordAlreadyExistsError("session already exists") from exc
        return session

    def get_session(self, workspace_id: str, session_id: str) -> McpSession:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_sessions
                WHERE workspace_id = ? AND session_id = ?
                """,
                (workspace_id, session_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("session not found")
        return self._session_from_row(row)

    def list_sessions(
        self,
        workspace_id: str,
        *,
        profile_id: str | None = None,
        live_only: bool = False,
    ) -> list[McpSession]:
        """List workspace sessions, optionally narrowed to a profile or live states."""

        self._require_initialized()
        clauses = ["workspace_id = ?"]
        parameters: list[str] = [workspace_id]
        if profile_id is not None:
            clauses.append("profile_id = ?")
            parameters.append(profile_id)
        if live_only:
            clauses.append(f"state IN ({','.join('?' for _ in _LIVE_SESSION_STATES)})")
            parameters.extend(_LIVE_SESSION_STATES)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM mcp_sessions
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, session_id
                """,
                parameters,
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def update_session_snapshot(
        self,
        workspace_id: str,
        session_id: str,
        snapshot_id: str,
        *,
        expected_revision: int,
    ) -> McpSession:
        """Pin a session to a new snapshot from the same profile revision."""

        self._require_initialized()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_sessions
                WHERE workspace_id = ? AND session_id = ?
                """,
                (workspace_id, session_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("session not found")
            current = self._session_from_row(row)
            if current.revision != expected_revision:
                raise RevisionConflictError("session", expected_revision, current.revision)
            matching_snapshot = connection.execute(
                """
                SELECT 1 FROM mcp_snapshots
                WHERE workspace_id = ? AND snapshot_id = ? AND profile_id = ?
                    AND profile_revision = ?
                """,
                (
                    workspace_id,
                    snapshot_id,
                    current.profile_id,
                    current.profile_revision,
                ),
            ).fetchone()
            if matching_snapshot is None:
                raise RecordNotFoundError("snapshot not found")
            if current.snapshot_id == snapshot_id:
                return current
            updated = McpSession.model_validate(
                current.model_copy(
                    update={
                        "snapshot_id": snapshot_id,
                        "revision": current.revision + 1,
                        "updated_at": utc_now(),
                    }
                ).model_dump()
            )
            connection.execute(
                """
                UPDATE mcp_sessions
                SET snapshot_id = ?, revision = ?, session_json = ?, updated_at = ?
                WHERE workspace_id = ? AND session_id = ? AND revision = ?
                """,
                (
                    snapshot_id,
                    updated.revision,
                    _json(updated),
                    _iso(updated.updated_at),
                    workspace_id,
                    session_id,
                    expected_revision,
                ),
            )
        return updated

    def update_session_state(
        self,
        workspace_id: str,
        session_id: str,
        target: SessionState,
        *,
        expected_revision: int,
    ) -> McpSession:
        self._require_initialized()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_sessions
                WHERE workspace_id = ? AND session_id = ?
                """,
                (workspace_id, session_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("session not found")
            current = self._session_from_row(row)
            if current.revision != expected_revision:
                raise RevisionConflictError("session", expected_revision, current.revision)
            if target == current.state:
                return current
            if target not in _SESSION_TRANSITIONS[current.state]:
                raise ValueError(f"Cannot transition session from {current.state} to {target}")
            now = utc_now()
            updated = McpSession.model_validate(
                current.model_copy(
                    update={
                        "state": target,
                        "revision": current.revision + 1,
                        "updated_at": now,
                        "disconnected_at": (
                            now if target == SessionState.DISCONNECTED else current.disconnected_at
                        ),
                    }
                ).model_dump()
            )
            connection.execute(
                """
                UPDATE mcp_sessions
                SET state = ?, revision = ?, session_json = ?, updated_at = ?, disconnected_at = ?
                WHERE workspace_id = ? AND session_id = ? AND revision = ?
                """,
                (
                    updated.state.value,
                    updated.revision,
                    _json(updated),
                    _iso(updated.updated_at),
                    _iso(updated.disconnected_at),
                    workspace_id,
                    session_id,
                    expected_revision,
                ),
            )
        return updated

    def mark_session_lost(self, workspace_id: str, session_id: str) -> McpSession:
        """Fail closed after a live transport is aborted or disappears.

        The session becomes disconnected and stale, and its pinned snapshot is
        marked stale in the same transaction. Repeating the operation is safe.
        """

        self._require_initialized()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_sessions
                WHERE workspace_id = ? AND session_id = ?
                """,
                (workspace_id, session_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("session not found")
            current = self._session_from_row(row)
            if current.state == SessionState.DISCONNECTED and current.stale:
                return current
            now = utc_now()
            updated = McpSession.model_validate(
                current.model_copy(
                    update={
                        "state": SessionState.DISCONNECTED,
                        "stale": True,
                        "revision": current.revision + 1,
                        "updated_at": now,
                        "disconnected_at": current.disconnected_at or now,
                    }
                ).model_dump()
            )
            connection.execute(
                """
                UPDATE mcp_sessions
                SET state = ?, stale = 1, revision = ?, session_json = ?,
                    updated_at = ?, disconnected_at = ?
                WHERE workspace_id = ? AND session_id = ?
                """,
                (
                    updated.state.value,
                    updated.revision,
                    _json(updated),
                    _iso(updated.updated_at),
                    _iso(updated.disconnected_at),
                    workspace_id,
                    session_id,
                ),
            )
            if current.snapshot_id is not None:
                connection.execute(
                    """
                    UPDATE mcp_snapshots SET stale = 1
                    WHERE workspace_id = ? AND snapshot_id = ?
                    """,
                    (workspace_id, current.snapshot_id),
                )
        return updated

    def create_operation(self, operation: McpOperation) -> McpOperation:
        self._require_initialized()
        if operation.revision != 1:
            raise ValueError("A new operation must start at revision 1")
        with self._connection(immediate=True) as connection:
            if operation.idempotency_key is not None:
                existing = connection.execute(
                    """
                    SELECT * FROM mcp_operations
                    WHERE workspace_id = ? AND kind = ? AND idempotency_key = ?
                    """,
                    (
                        operation.workspace_id,
                        operation.kind.value,
                        operation.idempotency_key,
                    ),
                ).fetchone()
                if existing is not None:
                    return self._operation_from_row(existing)
            if operation.profile_id is not None:
                profile = connection.execute(
                    """
                    SELECT 1 FROM mcp_profiles
                    WHERE workspace_id = ? AND profile_id = ? AND deleted_at IS NULL
                    """,
                    (operation.workspace_id, operation.profile_id),
                ).fetchone()
                if profile is None:
                    raise RecordNotFoundError("profile not found")
            if operation.session_id is not None:
                session = connection.execute(
                    """
                    SELECT 1 FROM mcp_sessions
                    WHERE workspace_id = ? AND session_id = ?
                    """,
                    (operation.workspace_id, operation.session_id),
                ).fetchone()
                if session is None:
                    raise RecordNotFoundError("session not found")
            try:
                connection.execute(
                    """
                    INSERT INTO mcp_operations(
                        workspace_id, operation_id, correlation_id, kind, state,
                        profile_id, session_id, idempotency_key, retry_of, revision,
                        error_code, safe_message, operation_json, created_at, updated_at,
                        started_at, completed_at, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        operation.workspace_id,
                        operation.operation_id,
                        operation.correlation_id,
                        operation.kind.value,
                        operation.state.value,
                        operation.profile_id,
                        operation.session_id,
                        operation.idempotency_key,
                        operation.retry_of,
                        operation.revision,
                        operation.error_code,
                        operation.safe_message,
                        _json(operation),
                        _iso(operation.created_at),
                        _iso(operation.updated_at),
                        _iso(operation.started_at),
                        _iso(operation.completed_at),
                        (
                            json.dumps(operation.result, sort_keys=True, separators=(",", ":"))
                            if operation.result is not None
                            else None
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RecordAlreadyExistsError("operation already exists") from exc
        return operation

    def get_operation(self, workspace_id: str, operation_id: str) -> McpOperation:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_operations
                WHERE workspace_id = ? AND operation_id = ?
                """,
                (workspace_id, operation_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("operation not found")
        return self._operation_from_row(row)

    def update_operation_state(
        self,
        workspace_id: str,
        operation_id: str,
        target: OperationState,
        *,
        expected_revision: int,
        error_code: str | None = None,
        safe_message: str | None = None,
        result: dict | None | object = UNSET_OPERATION_RESULT,
    ) -> McpOperation:
        self._require_initialized()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_operations
                WHERE workspace_id = ? AND operation_id = ?
                """,
                (workspace_id, operation_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("operation not found")
            current = self._operation_from_row(row)
            if current.revision != expected_revision:
                raise RevisionConflictError("operation", expected_revision, current.revision)
            updated = transitioned_operation(
                current,
                target,
                error_code=error_code,
                safe_message=safe_message,
                result=result,
            )
            if updated is current:
                return current
            connection.execute(
                """
                UPDATE mcp_operations
                SET state = ?, revision = ?, error_code = ?, safe_message = ?,
                    operation_json = ?, updated_at = ?, started_at = ?, completed_at = ?,
                    result_json = ?
                WHERE workspace_id = ? AND operation_id = ? AND revision = ?
                """,
                (
                    updated.state.value,
                    updated.revision,
                    updated.error_code,
                    updated.safe_message,
                    _json(updated),
                    _iso(updated.updated_at),
                    _iso(updated.started_at),
                    _iso(updated.completed_at),
                    (
                        json.dumps(updated.result, sort_keys=True, separators=(",", ":"))
                        if updated.result is not None
                        else None
                    ),
                    workspace_id,
                    operation_id,
                    expected_revision,
                ),
            )
        return updated

    def save_stdio_launch_approval(
        self, approval: McpStdioLaunchApproval
    ) -> McpStdioLaunchApproval:
        """Persist an immutable, secret-free stdio launch approval."""

        self._require_initialized()
        with self._connection(immediate=True) as connection:
            revision_row = connection.execute(
                """
                SELECT profile_json FROM mcp_profile_revisions
                WHERE workspace_id = ? AND profile_id = ? AND revision = ?
                """,
                (
                    approval.workspace_id,
                    approval.profile_id,
                    approval.profile_revision,
                ),
            ).fetchone()
            if revision_row is None:
                raise RecordNotFoundError("profile revision not found")
            revision_profile = McpProfile.model_validate_json(revision_row["profile_json"])
            if not isinstance(revision_profile.transport, StdioTransport):
                raise ValueError("Launch approvals are valid only for stdio profiles")
            existing = connection.execute(
                """
                SELECT * FROM mcp_stdio_launch_approvals
                WHERE workspace_id = ? AND profile_id = ? AND profile_revision = ?
                """,
                (
                    approval.workspace_id,
                    approval.profile_id,
                    approval.profile_revision,
                ),
            ).fetchone()
            if existing is not None:
                saved = self._approval_from_row(existing)
                if saved == approval:
                    return saved
                raise RecordAlreadyExistsError("stdio launch approval already exists")
            connection.execute(
                """
                INSERT INTO mcp_stdio_launch_approvals(
                    workspace_id, profile_id, profile_revision, executable_fingerprint,
                    launch_fingerprint, fingerprint_json, approved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.workspace_id,
                    approval.profile_id,
                    approval.profile_revision,
                    approval.executable_fingerprint,
                    approval.launch_fingerprint,
                    json.dumps(
                        approval.fingerprint.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    _iso(approval.approved_at),
                ),
            )
        return approval

    def get_stdio_launch_approval(
        self, workspace_id: str, profile_id: str, profile_revision: int
    ) -> McpStdioLaunchApproval:
        """Return the exact revision's approval without cross-workspace leakage."""

        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM mcp_stdio_launch_approvals
                WHERE workspace_id = ? AND profile_id = ? AND profile_revision = ?
                """,
                (workspace_id, profile_id, profile_revision),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("stdio launch approval not found")
        return self._approval_from_row(row)

    def recover_after_restart(self) -> RecoverySummary:
        """Interrupt lost work and stale all metadata tied to lost live sessions."""

        self._require_initialized()
        now = utc_now().isoformat()
        with self._connection(immediate=True) as connection:
            live_session_rows = connection.execute(
                f"""
                SELECT workspace_id, session_id, snapshot_id FROM mcp_sessions
                WHERE state IN ({','.join('?' for _ in _LIVE_SESSION_STATES)})
                """,
                _LIVE_SESSION_STATES,
            ).fetchall()
            snapshot_keys = {
                (row["workspace_id"], row["snapshot_id"])
                for row in live_session_rows
                if row["snapshot_id"] is not None
            }
            for workspace_id, snapshot_id in snapshot_keys:
                connection.execute(
                    """
                    UPDATE mcp_snapshots SET stale = 1
                    WHERE workspace_id = ? AND snapshot_id = ?
                    """,
                    (workspace_id, snapshot_id),
                )
            sessions_cursor = connection.execute(
                f"""
                UPDATE mcp_sessions
                SET state = ?, stale = 1, revision = revision + 1,
                    updated_at = ?, disconnected_at = ?
                WHERE state IN ({','.join('?' for _ in _LIVE_SESSION_STATES)})
                """,
                (
                    SessionState.DISCONNECTED.value,
                    now,
                    now,
                    *_LIVE_SESSION_STATES,
                ),
            )
            operations_cursor = connection.execute(
                """
                UPDATE mcp_operations
                SET state = ?, revision = revision + 1, updated_at = ?, completed_at = ?,
                    error_code = COALESCE(error_code, 'backend_restarted'),
                    safe_message = COALESCE(safe_message, 'Operation interrupted by backend restart')
                WHERE state IN (?, ?)
                """,
                (
                    OperationState.INTERRUPTED.value,
                    now,
                    now,
                    OperationState.RUNNING.value,
                    OperationState.WAITING_FOR_APPROVAL.value,
                ),
            )
        return RecoverySummary(
            operations_interrupted=operations_cursor.rowcount,
            sessions_disconnected=sessions_cursor.rowcount,
            snapshots_marked_stale=len(snapshot_keys),
        )

    def journal_mode(self) -> str:
        """Expose the active SQLite journal mode for health checks and tests."""

        self._require_initialized()
        with self._connection() as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def foreign_keys_enabled(self) -> bool:
        """Expose per-connection foreign key enforcement for health checks and tests."""

        self._require_initialized()
        with self._connection() as connection:
            return bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])


__all__ = [
    "McpRepository",
    "McpPersistenceError",
    "ProfileInUseError",
    "RecordAlreadyExistsError",
    "RecordNotFoundError",
    "RevisionConflictError",
]
