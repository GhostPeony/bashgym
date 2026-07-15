"""Workspace-scoped SQLite persistence for immutable HF context bundles."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .context_contracts import CompletionOutcome, HFContextBundle, Lifecycle, utc_now

READY_VERSION_RETENTION = 20


class HFContextPersistenceError(RuntimeError):
    pass


class BundleNotFoundError(HFContextPersistenceError):
    pass


class BundleAlreadyExistsError(HFContextPersistenceError):
    pass


class ImmutableBundleError(HFContextPersistenceError):
    pass


class BundleRevisionConflictError(HFContextPersistenceError):
    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(f"bundle revision conflict: expected {expected}, current {current}")


_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS hf_context_lineages (
            workspace_id TEXT NOT NULL,
            bundle_id TEXT NOT NULL,
            head_version INTEGER NOT NULL CHECK (head_version >= 1),
            correlation_id TEXT,
            origin_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT,
            PRIMARY KEY (workspace_id, bundle_id)
        );

        CREATE TABLE IF NOT EXISTS hf_context_versions (
            workspace_id TEXT NOT NULL,
            bundle_id TEXT NOT NULL,
            version INTEGER NOT NULL CHECK (version >= 1),
            lifecycle TEXT NOT NULL CHECK (lifecycle IN ('collecting', 'ready')),
            completion_outcome TEXT,
            bundle_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            ready_at TEXT,
            stale_observed_at TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (workspace_id, bundle_id, version),
            FOREIGN KEY (workspace_id, bundle_id)
                REFERENCES hf_context_lineages(workspace_id, bundle_id)
                ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS hf_context_active (
            workspace_id TEXT PRIMARY KEY,
            bundle_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            activated_at TEXT NOT NULL,
            FOREIGN KEY (workspace_id, bundle_id, version)
                REFERENCES hf_context_versions(workspace_id, bundle_id, version)
                ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS hf_context_action_previews (
            workspace_id TEXT NOT NULL,
            bundle_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            action_hash TEXT NOT NULL,
            preview_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (workspace_id, bundle_id, version, action_hash),
            FOREIGN KEY (workspace_id, bundle_id, version)
                REFERENCES hf_context_versions(workspace_id, bundle_id, version)
                ON DELETE RESTRICT
        );

        CREATE INDEX IF NOT EXISTS hf_context_history_idx
            ON hf_context_lineages(workspace_id, deleted_at, updated_at DESC);
        CREATE INDEX IF NOT EXISTS hf_context_version_history_idx
            ON hf_context_versions(workspace_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS hf_context_collecting_idx
            ON hf_context_versions(workspace_id, lifecycle);
        """,
    ),
)


class HFContextRepository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self._initialized = False

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS hf_context_schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {
                int(row["version"])
                for row in connection.execute("SELECT version FROM hf_context_schema_migrations")
            }
            for version, sql in _MIGRATIONS:
                if version in applied:
                    continue
                connection.executescript(sql)
                connection.execute(
                    "INSERT INTO hf_context_schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now().isoformat()),
                )
        self._initialized = True

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise HFContextPersistenceError("repository is not initialized")

    @staticmethod
    def _bundle_json(bundle: HFContextBundle) -> str:
        return bundle.model_dump_json()

    @staticmethod
    def _bundle_from_row(row: sqlite3.Row) -> HFContextBundle:
        return HFContextBundle.model_validate_json(row["bundle_json"])

    def create_lineage(self, bundle: HFContextBundle) -> HFContextBundle:
        self._require_initialized()
        if bundle.version != 1:
            raise ValueError("new lineages must start at version 1")
        now = bundle.created_at.isoformat()
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO hf_context_lineages(
                        workspace_id, bundle_id, head_version, correlation_id,
                        origin_json, created_at, updated_at
                    ) VALUES (?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        bundle.workspace_id,
                        bundle.bundle_id,
                        bundle.correlation_id,
                        json.dumps(bundle.origin, sort_keys=True),
                        now,
                        now,
                    ),
                )
                self._insert_version(connection, bundle)
        except sqlite3.IntegrityError as exc:
            raise BundleAlreadyExistsError(bundle.bundle_id) from exc
        return bundle

    @staticmethod
    def _insert_version(connection: sqlite3.Connection, bundle: HFContextBundle) -> None:
        connection.execute(
            """
            INSERT INTO hf_context_versions(
                workspace_id, bundle_id, version, lifecycle, completion_outcome,
                bundle_json, content_hash, retrieved_at, ready_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle.workspace_id,
                bundle.bundle_id,
                bundle.version,
                bundle.lifecycle.value,
                bundle.completion_outcome.value if bundle.completion_outcome else None,
                bundle.model_dump_json(),
                bundle.content_hash,
                bundle.created_at.isoformat(),
                bundle.ready_at.isoformat() if bundle.ready_at else None,
                bundle.created_at.isoformat(),
            ),
        )

    def _lineage_head(
        self, connection: sqlite3.Connection, workspace_id: str, bundle_id: str
    ) -> int:
        row = connection.execute(
            """
            SELECT head_version FROM hf_context_lineages
            WHERE workspace_id = ? AND bundle_id = ? AND deleted_at IS NULL
            """,
            (workspace_id, bundle_id),
        ).fetchone()
        if row is None:
            raise BundleNotFoundError(bundle_id)
        return int(row["head_version"])

    def get_version(self, workspace_id: str, bundle_id: str, version: int) -> HFContextBundle:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT versions.bundle_json
                FROM hf_context_versions AS versions
                JOIN hf_context_lineages AS lineages
                  ON lineages.workspace_id = versions.workspace_id
                 AND lineages.bundle_id = versions.bundle_id
                WHERE versions.workspace_id = ? AND versions.bundle_id = ?
                  AND versions.version = ? AND lineages.deleted_at IS NULL
                """,
                (workspace_id, bundle_id, version),
            ).fetchone()
        if row is None:
            raise BundleNotFoundError(bundle_id)
        return self._bundle_from_row(row)

    def finalize_version(self, bundle: HFContextBundle) -> HFContextBundle:
        self._require_initialized()
        if bundle.lifecycle is not Lifecycle.READY:
            raise ValueError("finalized bundles must be ready")
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT versions.lifecycle
                FROM hf_context_versions AS versions
                JOIN hf_context_lineages AS lineages
                  ON lineages.workspace_id = versions.workspace_id
                 AND lineages.bundle_id = versions.bundle_id
                WHERE versions.workspace_id = ? AND versions.bundle_id = ?
                  AND versions.version = ? AND lineages.deleted_at IS NULL
                """,
                (bundle.workspace_id, bundle.bundle_id, bundle.version),
            ).fetchone()
            if row is None:
                raise BundleNotFoundError(bundle.bundle_id)
            if row["lifecycle"] != Lifecycle.COLLECTING.value:
                raise ImmutableBundleError(f"bundle {bundle.bundle_id} v{bundle.version} is ready")
            connection.execute(
                """
                UPDATE hf_context_versions
                SET lifecycle = ?, completion_outcome = ?, bundle_json = ?,
                    content_hash = ?, ready_at = ?
                WHERE workspace_id = ? AND bundle_id = ? AND version = ?
                """,
                (
                    bundle.lifecycle.value,
                    bundle.completion_outcome.value if bundle.completion_outcome else None,
                    self._bundle_json(bundle),
                    bundle.content_hash,
                    (bundle.ready_at or utc_now()).isoformat(),
                    bundle.workspace_id,
                    bundle.bundle_id,
                    bundle.version,
                ),
            )
            self._enforce_retention(connection, bundle.workspace_id)
        return bundle

    def update_collecting(self, bundle: HFContextBundle) -> HFContextBundle:
        """Persist a bounded discovery checkpoint without making it immutable."""

        self._require_initialized()
        if bundle.lifecycle is not Lifecycle.COLLECTING:
            raise ValueError("collecting checkpoints must remain collecting")
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT lifecycle FROM hf_context_versions
                WHERE workspace_id = ? AND bundle_id = ? AND version = ?
                """,
                (bundle.workspace_id, bundle.bundle_id, bundle.version),
            ).fetchone()
            if row is None:
                raise BundleNotFoundError(bundle.bundle_id)
            if row["lifecycle"] != Lifecycle.COLLECTING.value:
                raise ImmutableBundleError(
                    f"bundle {bundle.bundle_id} v{bundle.version} is ready"
                )
            connection.execute(
                """
                UPDATE hf_context_versions SET bundle_json = ?, content_hash = ?
                WHERE workspace_id = ? AND bundle_id = ? AND version = ?
                """,
                (
                    self._bundle_json(bundle),
                    bundle.content_hash,
                    bundle.workspace_id,
                    bundle.bundle_id,
                    bundle.version,
                ),
            )
        return bundle

    def cancel_version(
        self, workspace_id: str, bundle_id: str, version: int
    ) -> HFContextBundle:
        """Atomically finalize a collecting version as cancelled."""

        self._require_initialized()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT versions.bundle_json, versions.lifecycle
                FROM hf_context_versions AS versions
                JOIN hf_context_lineages AS lineages
                  ON lineages.workspace_id = versions.workspace_id
                 AND lineages.bundle_id = versions.bundle_id
                WHERE versions.workspace_id = ? AND versions.bundle_id = ?
                  AND versions.version = ? AND lineages.deleted_at IS NULL
                """,
                (workspace_id, bundle_id, version),
            ).fetchone()
            if row is None:
                raise BundleNotFoundError(bundle_id)
            current = self._bundle_from_row(row)
            if current.lifecycle is Lifecycle.READY:
                if current.completion_outcome is CompletionOutcome.CANCELLED:
                    return current
                raise ImmutableBundleError(
                    f"bundle {bundle_id} v{version} already completed"
                )
            ready_at = utc_now()
            cancelled = HFContextBundle.model_validate(
                {
                    **current.model_dump(mode="python", exclude={"content_hash"}),
                    "lifecycle": Lifecycle.READY,
                    "completion_outcome": CompletionOutcome.CANCELLED,
                    "ready_at": ready_at,
                }
            )
            connection.execute(
                """
                UPDATE hf_context_versions
                SET lifecycle = ?, completion_outcome = ?, bundle_json = ?,
                    content_hash = ?, ready_at = ?
                WHERE workspace_id = ? AND bundle_id = ? AND version = ?
                """,
                (
                    Lifecycle.READY.value,
                    CompletionOutcome.CANCELLED.value,
                    self._bundle_json(cancelled),
                    cancelled.content_hash,
                    ready_at.isoformat(),
                    workspace_id,
                    bundle_id,
                    version,
                ),
            )
            self._enforce_retention(connection, workspace_id)
        return cancelled

    def create_version(self, bundle: HFContextBundle, *, expected_head: int) -> HFContextBundle:
        self._require_initialized()
        if bundle.lifecycle is not Lifecycle.READY:
            raise ValueError("new immutable versions must be ready")
        with self._transaction() as connection:
            current = self._lineage_head(connection, bundle.workspace_id, bundle.bundle_id)
            if current != expected_head:
                raise BundleRevisionConflictError(expected_head, current)
            if bundle.version != current + 1:
                raise ValueError("new bundle version must follow the lineage head")
            self._insert_version(connection, bundle)
            connection.execute(
                """
                UPDATE hf_context_lineages SET head_version = ?, updated_at = ?
                WHERE workspace_id = ? AND bundle_id = ?
                """,
                (bundle.version, utc_now().isoformat(), bundle.workspace_id, bundle.bundle_id),
            )
            self._enforce_retention(connection, bundle.workspace_id)
        return bundle

    def create_collecting_version(
        self, bundle: HFContextBundle, *, expected_head: int
    ) -> HFContextBundle:
        self._require_initialized()
        if bundle.lifecycle is not Lifecycle.COLLECTING:
            raise ValueError("refresh versions must begin collecting")
        with self._transaction() as connection:
            current = self._lineage_head(connection, bundle.workspace_id, bundle.bundle_id)
            if current != expected_head:
                raise BundleRevisionConflictError(expected_head, current)
            if bundle.version != current + 1:
                raise ValueError("new bundle version must follow the lineage head")
            self._insert_version(connection, bundle)
            connection.execute(
                """
                UPDATE hf_context_lineages SET head_version = ?, updated_at = ?
                WHERE workspace_id = ? AND bundle_id = ?
                """,
                (bundle.version, utc_now().isoformat(), bundle.workspace_id, bundle.bundle_id),
            )
        return bundle

    @staticmethod
    def _enforce_retention(
        connection: sqlite3.Connection,
        workspace_id: str,
        *, keep: int = READY_VERSION_RETENTION,
    ) -> None:
        """Keep the newest ready versions while protecting active and collecting rows."""

        excess = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM hf_context_versions AS versions
                JOIN hf_context_lineages AS lineages
                  ON lineages.workspace_id = versions.workspace_id
                 AND lineages.bundle_id = versions.bundle_id
                WHERE versions.workspace_id = ? AND versions.lifecycle = 'ready'
                  AND lineages.deleted_at IS NULL
                """,
                (workspace_id,),
            ).fetchone()[0]
        ) - keep
        if excess <= 0:
            return
        candidates = connection.execute(
            """
            SELECT versions.bundle_id, versions.version
            FROM hf_context_versions AS versions
            JOIN hf_context_lineages AS lineages
              ON lineages.workspace_id = versions.workspace_id
             AND lineages.bundle_id = versions.bundle_id
            LEFT JOIN hf_context_active AS active
              ON active.workspace_id = versions.workspace_id
             AND active.bundle_id = versions.bundle_id
             AND active.version = versions.version
            WHERE versions.workspace_id = ? AND versions.lifecycle = 'ready'
              AND lineages.deleted_at IS NULL AND active.workspace_id IS NULL
            ORDER BY versions.created_at ASC, versions.bundle_id ASC, versions.version ASC
            LIMIT ?
            """,
            (workspace_id, excess),
        ).fetchall()
        for candidate in candidates:
            bundle_id = str(candidate["bundle_id"])
            version = int(candidate["version"])
            connection.execute(
                """
                DELETE FROM hf_context_action_previews
                WHERE workspace_id = ? AND bundle_id = ? AND version = ?
                """,
                (workspace_id, bundle_id, version),
            )
            connection.execute(
                """
                DELETE FROM hf_context_versions
                WHERE workspace_id = ? AND bundle_id = ? AND version = ?
                """,
                (workspace_id, bundle_id, version),
            )
            remaining = connection.execute(
                """
                SELECT MAX(version) AS head FROM hf_context_versions
                WHERE workspace_id = ? AND bundle_id = ?
                """,
                (workspace_id, bundle_id),
            ).fetchone()["head"]
            if remaining is None:
                connection.execute(
                    """
                    DELETE FROM hf_context_lineages
                    WHERE workspace_id = ? AND bundle_id = ?
                    """,
                    (workspace_id, bundle_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE hf_context_lineages SET head_version = ?, updated_at = ?
                    WHERE workspace_id = ? AND bundle_id = ?
                    """,
                    (int(remaining), utc_now().isoformat(), workspace_id, bundle_id),
                )

    def list_versions(self, workspace_id: str, *, limit: int = 20) -> list[HFContextBundle]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT versions.bundle_json
                FROM hf_context_versions AS versions
                JOIN hf_context_lineages AS lineages
                  ON lineages.workspace_id = versions.workspace_id
                 AND lineages.bundle_id = versions.bundle_id
                WHERE versions.workspace_id = ? AND lineages.deleted_at IS NULL
                  AND versions.lifecycle = 'ready'
                ORDER BY versions.created_at DESC, versions.bundle_id, versions.version DESC
                LIMIT ?
                """,
                (workspace_id, max(1, min(limit, 100))),
            ).fetchall()
        return [self._bundle_from_row(row) for row in rows]

    def activate(self, workspace_id: str, bundle_id: str, version: int) -> HFContextBundle:
        bundle = self.get_version(workspace_id, bundle_id, version)
        if bundle.lifecycle is not Lifecycle.READY:
            raise ImmutableBundleError("only ready bundle versions can be active")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO hf_context_active(workspace_id, bundle_id, version, activated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    bundle_id = excluded.bundle_id,
                    version = excluded.version,
                    activated_at = excluded.activated_at
                """,
                (workspace_id, bundle_id, version, utc_now().isoformat()),
            )
        return bundle

    def get_active(self, workspace_id: str) -> HFContextBundle | None:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT bundle_id, version FROM hf_context_active WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            return self.get_version(workspace_id, row["bundle_id"], int(row["version"]))
        except BundleNotFoundError:
            return None

    def deactivate(self, workspace_id: str) -> None:
        self._require_initialized()
        with self._transaction() as connection:
            connection.execute("DELETE FROM hf_context_active WHERE workspace_id = ?", (workspace_id,))

    def delete_lineage(self, workspace_id: str, bundle_id: str) -> None:
        self._require_initialized()
        with self._transaction() as connection:
            self._lineage_head(connection, workspace_id, bundle_id)
            connection.execute(
                "DELETE FROM hf_context_active WHERE workspace_id = ? AND bundle_id = ?",
                (workspace_id, bundle_id),
            )
            connection.execute(
                """
                UPDATE hf_context_lineages SET deleted_at = ?, updated_at = ?
                WHERE workspace_id = ? AND bundle_id = ?
                """,
                (utc_now().isoformat(), utc_now().isoformat(), workspace_id, bundle_id),
            )

    def put_eval_preview(
        self,
        workspace_id: str,
        bundle_id: str,
        version: int,
        action_hash: str,
        preview: dict[str, Any],
    ) -> dict[str, Any]:
        self.get_version(workspace_id, bundle_id, version)
        serialized = json.dumps(preview, sort_keys=True, separators=(",", ":"))
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO hf_context_action_previews(
                    workspace_id, bundle_id, version, action_hash, preview_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (workspace_id, bundle_id, version, action_hash, serialized, utc_now().isoformat()),
            )
            row = connection.execute(
                """
                SELECT preview_json FROM hf_context_action_previews
                WHERE workspace_id = ? AND bundle_id = ? AND version = ? AND action_hash = ?
                """,
                (workspace_id, bundle_id, version, action_hash),
            ).fetchone()
        assert row is not None
        return json.loads(row["preview_json"])

    def schema_version(self) -> int:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute("SELECT MAX(version) AS version FROM hf_context_schema_migrations").fetchone()
        return int(row["version"] or 0)

    def journal_mode(self) -> str:
        self._require_initialized()
        with self._connection() as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def foreign_keys_enabled(self) -> bool:
        self._require_initialized()
        with self._connection() as connection:
            return bool(connection.execute("PRAGMA foreign_keys").fetchone()[0])


__all__ = [
    "BundleAlreadyExistsError",
    "BundleNotFoundError",
    "BundleRevisionConflictError",
    "HFContextPersistenceError",
    "HFContextRepository",
    "ImmutableBundleError",
]
