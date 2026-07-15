"""Durable repository for project-isolated ML experiment history."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from typing import Any

from bashgym.campaigns.contracts import canonical_hash, utc_now
from bashgym.campaigns.persistence import RecordNotFoundError, _iso, _json
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.ledger.contracts import (
    ArtifactSpec,
    AttemptSpec,
    DatasetSpec,
    DatasetVersionSpec,
    DecisionSpec,
    EnvironmentSpec,
    EvaluationResultSpec,
    EvaluationSuiteSpec,
    ExperimentSpec,
    LedgerEventSpec,
    MetricPointSpec,
    ModelSpec,
    ModelVersionSpec,
    ProjectSpec,
    RunSpec,
    RunStatus,
    stable_ledger_id,
)


class LedgerPersistenceError(RuntimeError):
    """Base class for stable experiment-ledger failures."""


class LedgerConflictError(LedgerPersistenceError):
    code = "ledger_identity_conflict"


class LedgerTransitionError(LedgerPersistenceError):
    code = "ledger_invalid_transition"


_RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    RunStatus.QUEUED.value: frozenset(
        {RunStatus.PREPARING.value, RunStatus.RUNNING.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    ),
    RunStatus.PREPARING.value: frozenset(
        {RunStatus.RUNNING.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    ),
    RunStatus.RUNNING.value: frozenset(
        {
            RunStatus.PAUSED.value,
            RunStatus.UNKNOWN.value,
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        }
    ),
    RunStatus.PAUSED.value: frozenset(
        {RunStatus.RUNNING.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    ),
    RunStatus.UNKNOWN.value: frozenset(
        {RunStatus.RUNNING.value, RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    ),
    RunStatus.COMPLETED.value: frozenset(),
    RunStatus.FAILED.value: frozenset(),
    RunStatus.CANCELLED.value: frozenset(),
}
_TERMINAL = frozenset(
    {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)


def _decode_row(row: sqlite3.Row | None, *json_fields: str) -> dict[str, Any] | None:
    if row is None:
        return None
    value = dict(row)
    for field in json_fields:
        if field in value:
            value[field.removesuffix("_json")] = json.loads(value.pop(field))
    if "is_simulation" in value:
        value["is_simulation"] = bool(value["is_simulation"])
    return value


class ExperimentLedgerRepository(CampaignRuntimeRepository):
    """Experiment ledger stored in the authoritative campaign SQLite database."""

    @staticmethod
    def _identity_digest(spec: Any, *, exclude: set[str] | None = None) -> str:
        payload = spec.model_dump(mode="json")
        for field in exclude or set():
            payload.pop(field, None)
        return canonical_hash(payload)

    @staticmethod
    def _replay_or_conflict(
        row: sqlite3.Row | None, digest: str, *, entity: str
    ) -> bool:
        if row is None:
            return False
        if row["identity_digest"] != digest:
            raise LedgerConflictError(f"{entity} already exists with a different identity")
        return True

    @staticmethod
    def _require_row(
        connection: sqlite3.Connection, query: str, parameters: tuple[Any, ...], message: str
    ) -> sqlite3.Row:
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise RecordNotFoundError(message)
        return row

    def register_project(self, spec: ProjectSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"status", "created_at"})
        with self._connection(immediate=True) as connection:
            existing = connection.execute(
                "SELECT identity_digest FROM ledger_projects WHERE workspace_id = ? AND project_id = ?",
                (spec.workspace_id, spec.project_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="project")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_projects(
                        workspace_id, project_id, display_name, description, status,
                        owner_actor_id, tags_json, identity_digest, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.display_name,
                        spec.description,
                        spec.status.value,
                        spec.owner_actor_id,
                        _json(list(spec.tags)),
                        digest,
                        _iso(spec.created_at),
                        _iso(spec.created_at),
                    ),
                )
        return self.get_project(spec.workspace_id, spec.project_id), replayed

    def register_experiment(self, spec: ExperimentSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"status", "created_at"})
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_projects WHERE workspace_id = ? AND project_id = ?",
                (spec.workspace_id, spec.project_id),
                "ledger project not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_experiments
                   WHERE workspace_id = ? AND project_id = ? AND experiment_id = ?""",
                (spec.workspace_id, spec.project_id, spec.experiment_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="experiment")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_experiments(
                        workspace_id, project_id, experiment_id, name, objective, status,
                        campaign_id, parent_experiment_id, metadata_json, identity_digest,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.experiment_id,
                        spec.name,
                        spec.objective,
                        spec.status.value,
                        spec.campaign_id,
                        spec.parent_experiment_id,
                        _json(spec.metadata),
                        digest,
                        _iso(spec.created_at),
                        _iso(spec.created_at),
                    ),
                )
        return self.get_experiment(spec.workspace_id, spec.project_id, spec.experiment_id), replayed

    def register_model(self, spec: ModelSpec) -> tuple[dict[str, Any], bool]:
        digest = self._identity_digest(spec, exclude={"created_at"})
        return self._register_project_entity(
            table="ledger_models",
            id_field="model_id",
            id_value=spec.model_id,
            spec=spec,
            digest=digest,
            columns=("display_name", "task_type", "architecture", "metadata_json"),
            values=(spec.display_name, spec.task_type, spec.architecture, _json(spec.metadata)),
            json_fields=("metadata_json",),
        )

    def register_dataset(self, spec: DatasetSpec) -> tuple[dict[str, Any], bool]:
        digest = self._identity_digest(spec, exclude={"created_at"})
        return self._register_project_entity(
            table="ledger_datasets",
            id_field="dataset_id",
            id_value=spec.dataset_id,
            spec=spec,
            digest=digest,
            columns=("display_name", "task_type", "metadata_json"),
            values=(spec.display_name, spec.task_type, _json(spec.metadata)),
            json_fields=("metadata_json",),
        )

    def _register_project_entity(
        self,
        *,
        table: str,
        id_field: str,
        id_value: str,
        spec: Any,
        digest: str,
        columns: tuple[str, ...],
        values: tuple[Any, ...],
        json_fields: tuple[str, ...],
    ) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        allowed = {"ledger_models", "ledger_datasets"}
        if table not in allowed or id_field not in {"model_id", "dataset_id"}:
            raise ValueError("unsupported ledger entity")
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_projects WHERE workspace_id = ? AND project_id = ?",
                (spec.workspace_id, spec.project_id),
                "ledger project not found",
            )
            existing = connection.execute(
                f"SELECT identity_digest FROM {table} WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?",
                (spec.workspace_id, spec.project_id, id_value),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity=id_field)
            if not replayed:
                all_columns = ("workspace_id", "project_id", id_field, *columns, "identity_digest", "created_at")
                placeholders = ", ".join("?" for _ in all_columns)
                connection.execute(
                    f"INSERT INTO {table}({', '.join(all_columns)}) VALUES ({placeholders})",
                    (spec.workspace_id, spec.project_id, id_value, *values, digest, _iso(spec.created_at)),
                )
            row = connection.execute(
                f"SELECT * FROM {table} WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?",
                (spec.workspace_id, spec.project_id, id_value),
            ).fetchone()
        return _decode_row(row, *json_fields) or {}, replayed

    def register_model_version(self, spec: ModelVersionSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"created_at"})
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_models WHERE workspace_id = ? AND project_id = ? AND model_id = ?",
                (spec.workspace_id, spec.project_id, spec.model_id),
                "ledger model not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_model_versions
                   WHERE workspace_id = ? AND project_id = ? AND model_version_id = ?""",
                (spec.workspace_id, spec.project_id, spec.model_version_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="model version")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_model_versions(
                        workspace_id, project_id, model_id, model_version_id, source_uri,
                        source_revision, parent_model_version_id, config_digest, metadata_json,
                        identity_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.model_id,
                        spec.model_version_id,
                        spec.source_uri,
                        spec.source_revision,
                        spec.parent_model_version_id,
                        spec.config_digest,
                        _json(spec.metadata),
                        digest,
                        _iso(spec.created_at),
                    ),
                )
        return self.get_model_version(spec.workspace_id, spec.project_id, spec.model_version_id), replayed

    def register_dataset_version(self, spec: DatasetVersionSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"created_at"})
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_datasets WHERE workspace_id = ? AND project_id = ? AND dataset_id = ?",
                (spec.workspace_id, spec.project_id, spec.dataset_id),
                "ledger dataset not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_dataset_versions
                   WHERE workspace_id = ? AND project_id = ? AND dataset_version_id = ?""",
                (spec.workspace_id, spec.project_id, spec.dataset_version_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="dataset version")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_dataset_versions(
                        workspace_id, project_id, dataset_id, dataset_version_id, source_uri,
                        content_digest, split_manifest_json, row_counts_json, metadata_json,
                        identity_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.dataset_id,
                        spec.dataset_version_id,
                        spec.source_uri,
                        spec.content_digest,
                        _json(spec.split_manifest),
                        _json(spec.row_counts),
                        _json(spec.metadata),
                        digest,
                        _iso(spec.created_at),
                    ),
                )
        return self.get_dataset_version(spec.workspace_id, spec.project_id, spec.dataset_version_id), replayed

    def register_environment(self, spec: EnvironmentSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"created_at"})
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_projects WHERE workspace_id = ? AND project_id = ?",
                (spec.workspace_id, spec.project_id),
                "ledger project not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_environments
                   WHERE workspace_id = ? AND project_id = ? AND environment_id = ?""",
                (spec.workspace_id, spec.project_id, spec.environment_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="environment")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_environments(
                        workspace_id, project_id, environment_id, compute_target,
                        runtime_digest, hardware_json, metadata_json, identity_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.environment_id,
                        spec.compute_target,
                        spec.runtime_digest,
                        _json(spec.hardware),
                        _json(spec.metadata),
                        digest,
                        _iso(spec.created_at),
                    ),
                )
        return self.get_environment(spec.workspace_id, spec.project_id, spec.environment_id), replayed

    def register_run(self, spec: RunSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"status", "queued_at"})
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                """SELECT 1 FROM ledger_experiments
                   WHERE workspace_id = ? AND project_id = ? AND experiment_id = ?""",
                (spec.workspace_id, spec.project_id, spec.experiment_id),
                "ledger experiment not found",
            )
            for table, field, value, message in (
                ("ledger_model_versions", "model_version_id", spec.model_version_id, "ledger model version not found"),
                ("ledger_dataset_versions", "dataset_version_id", spec.dataset_version_id, "ledger dataset version not found"),
                ("ledger_environments", "environment_id", spec.environment_id, "ledger environment not found"),
            ):
                if value is not None:
                    self._require_row(
                        connection,
                        f"SELECT 1 FROM {table} WHERE workspace_id = ? AND project_id = ? AND {field} = ?",
                        (spec.workspace_id, spec.project_id, value),
                        message,
                    )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_runs
                   WHERE workspace_id = ? AND project_id = ? AND run_id = ?""",
                (spec.workspace_id, spec.project_id, spec.run_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="run")
            if not replayed:
                try:
                    connection.execute(
                        """
                        INSERT INTO ledger_runs(
                            workspace_id, project_id, experiment_id, run_id, source_system,
                            source_run_id, campaign_id, study_id, action_id, run_kind,
                            task_type, training_method, status, context_status, model_version_id,
                            dataset_version_id, environment_id, recipe_digest, config_json,
                            correlation_id, is_simulation, identity_digest, queued_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            spec.workspace_id,
                            spec.project_id,
                            spec.experiment_id,
                            spec.run_id,
                            spec.source_system,
                            spec.source_run_id,
                            spec.campaign_id,
                            spec.study_id,
                            spec.action_id,
                            spec.run_kind,
                            spec.task_type,
                            spec.training_method,
                            spec.status.value,
                            spec.context_status.value,
                            spec.model_version_id,
                            spec.dataset_version_id,
                            spec.environment_id,
                            spec.recipe_digest,
                            _json(spec.config),
                            spec.correlation_id,
                            int(spec.is_simulation),
                            digest,
                            _iso(spec.queued_at),
                            _iso(spec.queued_at),
                        ),
                    )
                    connection.execute(
                        """UPDATE ledger_projects SET updated_at = ?
                           WHERE workspace_id = ? AND project_id = ?""",
                        (_iso(spec.queued_at), spec.workspace_id, spec.project_id),
                    )
                    connection.execute(
                        """UPDATE ledger_experiments SET updated_at = ?
                           WHERE workspace_id = ? AND project_id = ? AND experiment_id = ?""",
                        (
                            _iso(spec.queued_at),
                            spec.workspace_id,
                            spec.project_id,
                            spec.experiment_id,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise LedgerConflictError(
                        "source_system/source_run_id is already assigned to another run"
                    ) from exc
        return self.get_run(spec.workspace_id, spec.project_id, spec.run_id), replayed

    def register_attempt(self, spec: AttemptSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"status", "created_at"})
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_runs WHERE workspace_id = ? AND project_id = ? AND run_id = ?",
                (spec.workspace_id, spec.project_id, spec.run_id),
                "ledger run not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_run_attempts
                   WHERE workspace_id = ? AND project_id = ? AND attempt_id = ?""",
                (spec.workspace_id, spec.project_id, spec.attempt_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="attempt")
            if not replayed:
                try:
                    connection.execute(
                        """
                        INSERT INTO ledger_run_attempts(
                            workspace_id, project_id, run_id, attempt_id, attempt_number,
                            source_attempt_id, status, metadata_json, identity_digest,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            spec.workspace_id,
                            spec.project_id,
                            spec.run_id,
                            spec.attempt_id,
                            spec.attempt_number,
                            spec.source_attempt_id,
                            spec.status.value,
                            _json(spec.metadata),
                            digest,
                            _iso(spec.created_at),
                            _iso(spec.created_at),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise LedgerConflictError("attempt number is already assigned") from exc
        return self.get_attempt_record(spec.workspace_id, spec.project_id, spec.attempt_id), replayed

    def transition_run(
        self,
        workspace_id: str,
        project_id: str,
        run_id: str,
        status: RunStatus,
        *,
        at: datetime | None = None,
    ) -> dict[str, Any]:
        return self._transition_record(
            table="ledger_runs",
            id_field="run_id",
            id_value=run_id,
            workspace_id=workspace_id,
            project_id=project_id,
            status=status,
            at=at,
        )

    def transition_attempt(
        self,
        workspace_id: str,
        project_id: str,
        attempt_id: str,
        status: RunStatus,
        *,
        at: datetime | None = None,
    ) -> dict[str, Any]:
        return self._transition_record(
            table="ledger_run_attempts",
            id_field="attempt_id",
            id_value=attempt_id,
            workspace_id=workspace_id,
            project_id=project_id,
            status=status,
            at=at,
        )

    def _transition_record(
        self,
        *,
        table: str,
        id_field: str,
        id_value: str,
        workspace_id: str,
        project_id: str,
        status: RunStatus,
        at: datetime | None,
    ) -> dict[str, Any]:
        self._require_initialized()
        if (table, id_field) not in {
            ("ledger_runs", "run_id"),
            ("ledger_run_attempts", "attempt_id"),
        }:
            raise ValueError("unsupported ledger transition")
        moment = at or utc_now()
        with self._connection(immediate=True) as connection:
            row = self._require_row(
                connection,
                f"SELECT status FROM {table} WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?",
                (workspace_id, project_id, id_value),
                f"ledger {id_field.removesuffix('_id')} not found",
            )
            current = str(row["status"])
            target = status.value
            if current != target and target not in _RUN_TRANSITIONS.get(current, frozenset()):
                raise LedgerTransitionError(f"cannot transition {current} to {target}")
            started_at = _iso(moment) if target == RunStatus.RUNNING.value else None
            completed_at = _iso(moment) if target in _TERMINAL else None
            connection.execute(
                f"""
                UPDATE {table}
                SET status = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = COALESCE(completed_at, ?),
                    updated_at = ?
                WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?
                """,
                (target, started_at, completed_at, _iso(moment), workspace_id, project_id, id_value),
            )
            if table == "ledger_runs":
                experiment_row = connection.execute(
                    """SELECT experiment_id FROM ledger_runs
                       WHERE workspace_id = ? AND project_id = ? AND run_id = ?""",
                    (workspace_id, project_id, id_value),
                ).fetchone()
                connection.execute(
                    """UPDATE ledger_projects SET updated_at = ?
                       WHERE workspace_id = ? AND project_id = ?""",
                    (_iso(moment), workspace_id, project_id),
                )
                if experiment_row is not None:
                    connection.execute(
                        """UPDATE ledger_experiments SET updated_at = ?
                           WHERE workspace_id = ? AND project_id = ? AND experiment_id = ?""",
                        (
                            _iso(moment),
                            workspace_id,
                            project_id,
                            experiment_row["experiment_id"],
                        ),
                    )
            updated = connection.execute(
                f"SELECT * FROM {table} WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?",
                (workspace_id, project_id, id_value),
            ).fetchone()
        json_fields = ("config_json",) if table == "ledger_runs" else ("metadata_json",)
        return _decode_row(updated, *json_fields) or {}

    def append_metric(self, spec: MetricPointSpec) -> bool:
        """Append one metric point; return True only when a new point is written."""

        self._require_initialized()
        if not math.isfinite(spec.metric_value):
            raise ValueError("metric_value must be finite")
        with self._connection(immediate=True) as connection:
            attempt = self._require_row(
                connection,
                """SELECT run_id FROM ledger_run_attempts
                   WHERE workspace_id = ? AND project_id = ? AND attempt_id = ?""",
                (spec.workspace_id, spec.project_id, spec.attempt_id),
                "ledger attempt not found",
            )
            if attempt["run_id"] != spec.run_id:
                raise LedgerConflictError("metric run_id does not match its attempt")
            existing = connection.execute(
                """
                SELECT metric_value, raw_sha256, context_json FROM ledger_metric_points
                WHERE workspace_id = ? AND project_id = ? AND attempt_id = ?
                  AND source = ? AND step = ? AND metric_name = ?
                """,
                (
                    spec.workspace_id,
                    spec.project_id,
                    spec.attempt_id,
                    spec.source,
                    spec.step,
                    spec.metric_name,
                ),
            ).fetchone()
            if existing is not None:
                if (
                    float(existing["metric_value"]) != spec.metric_value
                    or existing["raw_sha256"] != spec.raw_sha256
                    or existing["context_json"] != _json(spec.context)
                ):
                    raise LedgerConflictError("metric identity was replayed with different data")
                return False
            connection.execute(
                """
                INSERT INTO ledger_metric_points(
                    workspace_id, project_id, run_id, attempt_id, source, step,
                    metric_name, metric_value, raw_sha256, context_json, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.workspace_id,
                    spec.project_id,
                    spec.run_id,
                    spec.attempt_id,
                    spec.source,
                    spec.step,
                    spec.metric_name,
                    spec.metric_value,
                    spec.raw_sha256,
                    _json(spec.context),
                    _iso(spec.observed_at),
                ),
            )
        return True

    def record_artifact(self, spec: ArtifactSpec) -> tuple[dict[str, Any], bool]:
        return self._record_run_entity(
            table="ledger_artifacts",
            id_field="artifact_id",
            id_value=spec.artifact_id,
            spec=spec,
            columns=("attempt_id", "kind", "uri", "sha256", "size_bytes", "media_type", "metadata_json"),
            values=(spec.attempt_id, spec.kind, spec.uri, spec.sha256, spec.size_bytes, spec.media_type, _json(spec.metadata)),
            json_fields=("metadata_json",),
        )

    def register_evaluation_suite(
        self, spec: EvaluationSuiteSpec
    ) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec, exclude={"created_at"})
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_projects WHERE workspace_id = ? AND project_id = ?",
                (spec.workspace_id, spec.project_id),
                "ledger project not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_evaluation_suites
                   WHERE workspace_id = ? AND project_id = ? AND evaluation_suite_id = ?""",
                (spec.workspace_id, spec.project_id, spec.evaluation_suite_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="evaluation suite")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_evaluation_suites(
                        workspace_id, project_id, evaluation_suite_id, name, task_type,
                        dataset_version_id, metric_contract_json, code_digest, metadata_json,
                        identity_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.evaluation_suite_id,
                        spec.name,
                        spec.task_type,
                        spec.dataset_version_id,
                        _json(spec.metric_contract),
                        spec.code_digest,
                        _json(spec.metadata),
                        digest,
                        _iso(spec.created_at),
                    ),
                )
        return self.get_evaluation_suite(spec.workspace_id, spec.project_id, spec.evaluation_suite_id), replayed

    def record_evaluation_result(
        self, spec: EvaluationResultSpec
    ) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec)
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                """SELECT 1 FROM ledger_evaluation_suites
                   WHERE workspace_id = ? AND project_id = ? AND evaluation_suite_id = ?""",
                (spec.workspace_id, spec.project_id, spec.evaluation_suite_id),
                "ledger evaluation suite not found",
            )
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_runs WHERE workspace_id = ? AND project_id = ? AND run_id = ?",
                (spec.workspace_id, spec.project_id, spec.run_id),
                "ledger run not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_evaluation_results
                   WHERE workspace_id = ? AND project_id = ? AND evaluation_result_id = ?""",
                (spec.workspace_id, spec.project_id, spec.evaluation_result_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="evaluation result")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_evaluation_results(
                        workspace_id, project_id, evaluation_result_id, evaluation_suite_id,
                        run_id, attempt_id, model_version_id, status, metrics_json,
                        slice_metrics_json, artifact_id, compared_to_result_id, identity_digest,
                        started_at, completed_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.evaluation_result_id,
                        spec.evaluation_suite_id,
                        spec.run_id,
                        spec.attempt_id,
                        spec.model_version_id,
                        spec.status.value,
                        _json(spec.metrics),
                        _json(spec.slice_metrics),
                        spec.artifact_id,
                        spec.compared_to_result_id,
                        digest,
                        _iso(spec.started_at),
                        _iso(spec.completed_at) if spec.completed_at else None,
                        _iso(utc_now()),
                    ),
                )
        return self.get_evaluation_result(spec.workspace_id, spec.project_id, spec.evaluation_result_id), replayed

    def record_decision(self, spec: DecisionSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec)
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                """SELECT 1 FROM ledger_experiments
                   WHERE workspace_id = ? AND project_id = ? AND experiment_id = ?""",
                (spec.workspace_id, spec.project_id, spec.experiment_id),
                "ledger experiment not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_decisions
                   WHERE workspace_id = ? AND project_id = ? AND decision_id = ?""",
                (spec.workspace_id, spec.project_id, spec.decision_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="decision")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_decisions(
                        workspace_id, project_id, decision_id, experiment_id, run_id,
                        decision_type, outcome, rationale, evidence_refs_json, actor_id,
                        identity_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.workspace_id,
                        spec.project_id,
                        spec.decision_id,
                        spec.experiment_id,
                        spec.run_id,
                        spec.decision_type,
                        spec.outcome,
                        spec.rationale,
                        _json(list(spec.evidence_refs)),
                        spec.actor_id,
                        digest,
                        _iso(spec.created_at),
                    ),
                )
        return self.get_decision(spec.workspace_id, spec.project_id, spec.decision_id), replayed

    def append_event(self, spec: LedgerEventSpec) -> tuple[dict[str, Any], bool]:
        self._require_initialized()
        digest = self._identity_digest(spec)
        event_id = stable_ledger_id(
            "event", spec.workspace_id, spec.source_system, spec.source_event_id
        )
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_projects WHERE workspace_id = ? AND project_id = ?",
                (spec.workspace_id, spec.project_id),
                "ledger project not found",
            )
            existing = connection.execute(
                """SELECT identity_digest FROM ledger_events
                   WHERE workspace_id = ? AND source_system = ? AND source_event_id = ?""",
                (spec.workspace_id, spec.source_system, spec.source_event_id),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity="event")
            if not replayed:
                connection.execute(
                    """
                    INSERT INTO ledger_events(
                        event_id, workspace_id, project_id, experiment_id, run_id,
                        attempt_id, event_type, source_system, source_event_id, payload_json,
                        correlation_id, identity_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        spec.workspace_id,
                        spec.project_id,
                        spec.experiment_id,
                        spec.run_id,
                        spec.attempt_id,
                        spec.event_type,
                        spec.source_system,
                        spec.source_event_id,
                        _json(spec.payload),
                        spec.correlation_id,
                        digest,
                        _iso(spec.created_at),
                    ),
                )
            row = connection.execute(
                "SELECT * FROM ledger_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return _decode_row(row, "payload_json") or {}, replayed

    def _record_run_entity(
        self,
        *,
        table: str,
        id_field: str,
        id_value: str,
        spec: Any,
        columns: tuple[str, ...],
        values: tuple[Any, ...],
        json_fields: tuple[str, ...],
    ) -> tuple[dict[str, Any], bool]:
        if (table, id_field) != ("ledger_artifacts", "artifact_id"):
            raise ValueError("unsupported run entity")
        self._require_initialized()
        digest = self._identity_digest(spec)
        with self._connection(immediate=True) as connection:
            self._require_row(
                connection,
                "SELECT 1 FROM ledger_runs WHERE workspace_id = ? AND project_id = ? AND run_id = ?",
                (spec.workspace_id, spec.project_id, spec.run_id),
                "ledger run not found",
            )
            existing = connection.execute(
                f"SELECT identity_digest FROM {table} WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?",
                (spec.workspace_id, spec.project_id, id_value),
            ).fetchone()
            replayed = self._replay_or_conflict(existing, digest, entity=id_field)
            if not replayed:
                all_columns = (
                    "workspace_id", "project_id", id_field, "run_id", *columns,
                    "identity_digest", "created_at"
                )
                placeholders = ", ".join("?" for _ in all_columns)
                connection.execute(
                    f"INSERT INTO {table}({', '.join(all_columns)}) VALUES ({placeholders})",
                    (
                        spec.workspace_id,
                        spec.project_id,
                        id_value,
                        spec.run_id,
                        *values,
                        digest,
                        _iso(spec.created_at),
                    ),
                )
            row = connection.execute(
                f"SELECT * FROM {table} WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?",
                (spec.workspace_id, spec.project_id, id_value),
            ).fetchone()
        return _decode_row(row, *json_fields) or {}, replayed

    def get_project(self, workspace_id: str, project_id: str) -> dict[str, Any]:
        return self._get_one(
            "ledger_projects", "project_id", workspace_id, project_id, ("tags_json",), "ledger project not found"
        )

    def get_experiment(self, workspace_id: str, project_id: str, experiment_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_experiments", "experiment_id", workspace_id, project_id, experiment_id,
            ("metadata_json",), "ledger experiment not found"
        )

    def get_model_version(self, workspace_id: str, project_id: str, model_version_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_model_versions", "model_version_id", workspace_id, project_id, model_version_id,
            ("metadata_json",), "ledger model version not found"
        )

    def get_dataset_version(self, workspace_id: str, project_id: str, dataset_version_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_dataset_versions", "dataset_version_id", workspace_id, project_id, dataset_version_id,
            ("split_manifest_json", "row_counts_json", "metadata_json"), "ledger dataset version not found"
        )

    def get_environment(self, workspace_id: str, project_id: str, environment_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_environments", "environment_id", workspace_id, project_id, environment_id,
            ("hardware_json", "metadata_json"), "ledger environment not found"
        )

    def get_run(self, workspace_id: str, project_id: str, run_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_runs", "run_id", workspace_id, project_id, run_id,
            ("config_json",), "ledger run not found"
        )

    def get_attempt_record(self, workspace_id: str, project_id: str, attempt_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_run_attempts", "attempt_id", workspace_id, project_id, attempt_id,
            ("metadata_json",), "ledger attempt not found"
        )

    def get_evaluation_suite(self, workspace_id: str, project_id: str, suite_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_evaluation_suites", "evaluation_suite_id", workspace_id, project_id, suite_id,
            ("metric_contract_json", "metadata_json"), "ledger evaluation suite not found"
        )

    def get_evaluation_result(self, workspace_id: str, project_id: str, result_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_evaluation_results", "evaluation_result_id", workspace_id, project_id, result_id,
            ("metrics_json", "slice_metrics_json"), "ledger evaluation result not found"
        )

    def get_decision(self, workspace_id: str, project_id: str, decision_id: str) -> dict[str, Any]:
        return self._get_project_one(
            "ledger_decisions", "decision_id", workspace_id, project_id, decision_id,
            ("evidence_refs_json",), "ledger decision not found"
        )

    def _get_one(
        self, table: str, id_field: str, workspace_id: str, id_value: str,
        json_fields: tuple[str, ...], message: str
    ) -> dict[str, Any]:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE workspace_id = ? AND {id_field} = ?",
                (workspace_id, id_value),
            ).fetchone()
        value = _decode_row(row, *json_fields)
        if value is None:
            raise RecordNotFoundError(message)
        return value

    def _get_project_one(
        self, table: str, id_field: str, workspace_id: str, project_id: str, id_value: str,
        json_fields: tuple[str, ...], message: str
    ) -> dict[str, Any]:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM {table} WHERE workspace_id = ? AND project_id = ? AND {id_field} = ?",
                (workspace_id, project_id, id_value),
            ).fetchone()
        value = _decode_row(row, *json_fields)
        if value is None:
            raise RecordNotFoundError(message)
        return value

    def list_projects(self, workspace_id: str) -> list[dict[str, Any]]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM ledger_projects WHERE workspace_id = ? ORDER BY updated_at DESC, project_id",
                (workspace_id,),
            ).fetchall()
        return [_decode_row(row, "tags_json") or {} for row in rows]

    def list_experiments(self, workspace_id: str, project_id: str) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM ledger_experiments
                   WHERE workspace_id = ? AND project_id = ?
                   ORDER BY updated_at DESC, experiment_id""",
                (workspace_id, project_id),
            ).fetchall()
        return [_decode_row(row, "metadata_json") or {} for row in rows]

    def list_model_versions(self, workspace_id: str, project_id: str) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT mv.*, m.display_name AS model_display_name,
                       m.task_type AS model_task_type, m.architecture
                FROM ledger_model_versions mv
                JOIN ledger_models m
                  ON m.workspace_id = mv.workspace_id
                 AND m.project_id = mv.project_id
                 AND m.model_id = mv.model_id
                WHERE mv.workspace_id = ? AND mv.project_id = ?
                ORDER BY mv.created_at DESC, mv.model_version_id
                """,
                (workspace_id, project_id),
            ).fetchall()
        return [_decode_row(row, "metadata_json") or {} for row in rows]

    def list_dataset_versions(self, workspace_id: str, project_id: str) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT dv.*, d.display_name AS dataset_display_name,
                       d.task_type AS dataset_task_type
                FROM ledger_dataset_versions dv
                JOIN ledger_datasets d
                  ON d.workspace_id = dv.workspace_id
                 AND d.project_id = dv.project_id
                 AND d.dataset_id = dv.dataset_id
                WHERE dv.workspace_id = ? AND dv.project_id = ?
                ORDER BY dv.created_at DESC, dv.dataset_version_id
                """,
                (workspace_id, project_id),
            ).fetchall()
        return [
            _decode_row(row, "split_manifest_json", "row_counts_json", "metadata_json") or {}
            for row in rows
        ]

    def list_environments(self, workspace_id: str, project_id: str) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM ledger_environments
                   WHERE workspace_id = ? AND project_id = ?
                   ORDER BY created_at DESC, environment_id""",
                (workspace_id, project_id),
            ).fetchall()
        return [
            _decode_row(row, "hardware_json", "metadata_json") or {} for row in rows
        ]

    def list_evaluation_suites(
        self, workspace_id: str, project_id: str
    ) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM ledger_evaluation_suites
                   WHERE workspace_id = ? AND project_id = ?
                   ORDER BY created_at DESC, evaluation_suite_id""",
                (workspace_id, project_id),
            ).fetchall()
        return [
            _decode_row(row, "metric_contract_json", "metadata_json") or {} for row in rows
        ]

    def list_runs(
        self, workspace_id: str, project_id: str, *, limit: int = 100, status: str | None = None
    ) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        bounded = max(1, min(limit, 1000))
        query = "SELECT * FROM ledger_runs WHERE workspace_id = ? AND project_id = ?"
        parameters: list[Any] = [workspace_id, project_id]
        if status:
            query += " AND status = ?"
            parameters.append(status)
        query += " ORDER BY queued_at DESC, run_id LIMIT ?"
        parameters.append(bounded)
        with self._connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_decode_row(row, "config_json") or {} for row in rows]

    def list_attempt_records(self, workspace_id: str, project_id: str, run_id: str) -> list[dict[str, Any]]:
        self.get_run(workspace_id, project_id, run_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM ledger_run_attempts
                   WHERE workspace_id = ? AND project_id = ? AND run_id = ?
                   ORDER BY attempt_number""",
                (workspace_id, project_id, run_id),
            ).fetchall()
        return [_decode_row(row, "metadata_json") or {} for row in rows]

    def metric_series(
        self,
        workspace_id: str,
        project_id: str,
        *,
        metric_name: str,
        run_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        bounded = max(1, min(limit, 20_000))
        query = """SELECT * FROM ledger_metric_points
                   WHERE workspace_id = ? AND project_id = ? AND metric_name = ?"""
        parameters: list[Any] = [workspace_id, project_id, metric_name]
        if run_id:
            query += " AND run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY observed_at, run_id, step LIMIT ?"
        parameters.append(bounded)
        with self._connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_decode_row(row, "context_json") or {} for row in rows]

    def list_evaluation_results(
        self, workspace_id: str, project_id: str, *, run_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        query = """SELECT * FROM ledger_evaluation_results
                   WHERE workspace_id = ? AND project_id = ?"""
        parameters: list[Any] = [workspace_id, project_id]
        if run_id:
            query += " AND run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY created_at DESC, evaluation_result_id LIMIT ?"
        parameters.append(max(1, min(limit, 1000)))
        with self._connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_decode_row(row, "metrics_json", "slice_metrics_json") or {} for row in rows]

    def list_artifacts(
        self, workspace_id: str, project_id: str, *, run_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        query = "SELECT * FROM ledger_artifacts WHERE workspace_id = ? AND project_id = ?"
        parameters: list[Any] = [workspace_id, project_id]
        if run_id:
            query += " AND run_id = ?"
            parameters.append(run_id)
        query += " ORDER BY created_at DESC, artifact_id LIMIT ?"
        parameters.append(max(1, min(limit, 2000)))
        with self._connection() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return [_decode_row(row, "metadata_json") or {} for row in rows]

    def list_decisions(self, workspace_id: str, project_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM ledger_decisions
                   WHERE workspace_id = ? AND project_id = ?
                   ORDER BY created_at DESC, decision_id LIMIT ?""",
                (workspace_id, project_id, max(1, min(limit, 1000))),
            ).fetchall()
        return [_decode_row(row, "evidence_refs_json") or {} for row in rows]

    def list_events(
        self, workspace_id: str, project_id: str, *, after: int = 0, limit: int = 200
    ) -> list[dict[str, Any]]:
        self.get_project(workspace_id, project_id)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM ledger_events
                   WHERE workspace_id = ? AND project_id = ? AND cursor > ?
                   ORDER BY cursor LIMIT ?""",
                (workspace_id, project_id, max(0, after), max(1, min(limit, 1000))),
            ).fetchall()
        return [_decode_row(row, "payload_json") or {} for row in rows]

    def run_details(self, workspace_id: str, project_id: str, run_id: str) -> dict[str, Any]:
        return {
            "run": self.get_run(workspace_id, project_id, run_id),
            "attempts": self.list_attempt_records(workspace_id, project_id, run_id),
            "evaluations": self.list_evaluation_results(workspace_id, project_id, run_id=run_id),
            "artifacts": self.list_artifacts(workspace_id, project_id, run_id=run_id),
        }

    def database_health(self, workspace_id: str) -> dict[str, Any]:
        """Return a bounded operational health projection without exposing paths."""

        self._require_initialized()
        with self._connection() as connection:
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            counts = {
                "projects": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM ledger_projects WHERE workspace_id = ?",
                        (workspace_id,),
                    ).fetchone()[0]
                ),
                "runs": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM ledger_runs WHERE workspace_id = ?",
                        (workspace_id,),
                    ).fetchone()[0]
                ),
                "metric_points": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM ledger_metric_points WHERE workspace_id = ?",
                        (workspace_id,),
                    ).fetchone()[0]
                ),
                "evaluation_results": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM ledger_evaluation_results WHERE workspace_id = ?",
                        (workspace_id,),
                    ).fetchone()[0]
                ),
                "artifacts": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM ledger_artifacts WHERE workspace_id = ?",
                        (workspace_id,),
                    ).fetchone()[0]
                ),
            }
            last_cursor = int(
                connection.execute(
                    "SELECT COALESCE(MAX(cursor), 0) FROM ledger_events WHERE workspace_id = ?",
                    (workspace_id,),
                ).fetchone()[0]
            )
        wal_path = self.db_path.with_name(f"{self.db_path.name}-wal")
        return {
            "schema_version": "experiment_ledger_health.v1",
            "status": "healthy" if quick_check == "ok" else "unhealthy",
            "quick_check": quick_check,
            "journal_mode": self.journal_mode(),
            "foreign_keys_enabled": self.foreign_keys_enabled(),
            "migration_versions": self.schema_versions(),
            "database_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "wal_size_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            "last_event_cursor": last_cursor,
            "counts": counts,
        }
