"""Durable, portable authority for the AutoResearch guided setup flow.

Only logical binding identifiers cross this boundary. Installation-private paths,
hosts, controller lease keys, and credentials remain in their owning registries.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.autoresearch import AutoResearchTemplateDefinition
from bashgym.campaigns.contracts import (
    Campaign,
    CampaignEvent,
    CampaignStatus,
    CredentialKind,
    ManifestRevision,
    canonical_hash,
)
from bashgym.campaigns.persistence import (
    CampaignBudgetResourceLimitError,
    CampaignRepository,
    RecordAlreadyExistsError,
)
from bashgym.campaigns.readiness import AutoResearchDoctorReport


class GuidedSetupError(ValueError):
    """Base guided-setup error."""


class GuidedSetupConflictError(GuidedSetupError):
    """A durable setup authority is stale, altered, or already consumed."""


class GuidedSetupBindings(BaseModel):
    """Editable logical bindings; never physical locations or credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    data: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    compute: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    evaluation: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class GuidedSetupDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    template_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    installation_id: str = Field(pattern=r"^ins_[0-9a-f]{32}$")
    bindings: GuidedSetupBindings


_KIND_MAP: tuple[tuple[Literal["model", "data", "compute", "evaluation"], str], ...] = (
    ("model", "model"),
    ("data", "data"),
    ("compute", "compute"),
    ("evaluation", "evaluator"),
)
_SESSION_STEPS = ("template", "installation", "model", "data", "compute", "evaluation")
_SESSION_ID = re.compile(r"^setupsess_[0-9a-f]{32}$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
GUIDED_SETUP_MAX_TEMPLATES = 32
GUIDED_SETUP_MAX_INSTALLATIONS = 32
GUIDED_SETUP_MAX_BINDINGS_PER_KIND = 32


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _wire_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _expected_bindings(definition: AutoResearchTemplateDefinition) -> GuidedSetupBindings:
    policy = definition.policy
    evaluation = definition.manifest.evaluation_plan
    data = evaluation.get("dataset_binding_id")
    if not isinstance(data, str) or not data:
        if not definition.manifest.approved_data_scopes:
            raise GuidedSetupError("template data binding is unavailable")
        data = definition.manifest.approved_data_scopes[0]
    evaluation_id = (
        policy.evaluation_suite_id if policy is not None else evaluation.get("evaluation_suite_id")
    )
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise GuidedSetupError("template evaluation binding is unavailable")
    return GuidedSetupBindings(
        model=definition.target_model.target_contract_key,
        data=data,
        compute=definition.manifest.compute_profile_id,
        evaluation=evaluation_id,
    )


class GuidedSetupRepository:
    """Persist sealed validation receipts and the exact campaign binding references."""

    def __init__(self, db_path: str | Path, *, sealer: ArtifactSealer | None = None):
        self.db_path = Path(db_path)
        self.sealer = sealer
        self._initialized = False

    @contextmanager
    def _connection(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
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

    def initialize(self) -> None:
        with self._connection(immediate=True) as connection:
            required = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if (
                not {
                    "campaigns",
                    "campaign_recovery_installations",
                    "campaign_recovery_bindings",
                }
                <= required
            ):
                raise GuidedSetupError("campaign and installation registries must be initialized")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS campaign_guided_setup_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    definition_digest TEXT NOT NULL,
                    installation_id TEXT NOT NULL,
                    bindings_json TEXT NOT NULL,
                    binding_state_digest TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    response_seal TEXT NOT NULL,
                    seal_key_version TEXT NOT NULL,
                    ready INTEGER NOT NULL CHECK(ready IN (0,1)),
                    consumed_campaign_id TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(workspace_id, actor_id, idempotency_key),
                    FOREIGN KEY(installation_id) REFERENCES campaign_recovery_installations(installation_id)
                        ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS campaign_guided_setup_bindings (
                    workspace_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    validation_receipt_id TEXT NOT NULL,
                    installation_id TEXT NOT NULL,
                    model_binding_id TEXT NOT NULL,
                    data_binding_id TEXT NOT NULL,
                    compute_binding_id TEXT NOT NULL,
                    evaluation_binding_id TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, campaign_id),
                    FOREIGN KEY(workspace_id, campaign_id) REFERENCES campaigns(workspace_id, campaign_id)
                        ON DELETE RESTRICT,
                    FOREIGN KEY(validation_receipt_id) REFERENCES campaign_guided_setup_receipts(receipt_id)
                        ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS campaign_guided_setup_sessions (
                    workspace_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version >= 1),
                    state_json TEXT NOT NULL,
                    state_digest TEXT NOT NULL,
                    latest_receipt_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, actor_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS campaign_guided_setup_step_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version >= 1),
                    step TEXT NOT NULL CHECK(step IN (
                        'template','installation','model','data','compute','evaluation'
                    )),
                    selection_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    previous_receipt_id TEXT,
                    previous_receipt_digest TEXT,
                    state_json TEXT NOT NULL,
                    state_digest TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    response_seal TEXT NOT NULL,
                    seal_key_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(workspace_id, actor_id, idempotency_key),
                    UNIQUE(workspace_id, actor_id, session_id, version),
                    FOREIGN KEY(workspace_id, actor_id, session_id)
                        REFERENCES campaign_guided_setup_sessions(
                            workspace_id, actor_id, session_id
                        ) ON DELETE RESTRICT
                );
                """)
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(campaign_guided_setup_receipts)"
                ).fetchall()
            }
            if "seal_key_version" not in columns:
                connection.execute("""
                    ALTER TABLE campaign_guided_setup_receipts
                    ADD COLUMN seal_key_version TEXT NOT NULL DEFAULT ''
                    """)
            step_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(campaign_guided_setup_step_receipts)"
                ).fetchall()
            }
            if "previous_receipt_id" not in step_columns:
                connection.execute(
                    "ALTER TABLE campaign_guided_setup_step_receipts ADD COLUMN previous_receipt_id TEXT"
                )
            if "previous_receipt_digest" not in step_columns:
                connection.execute(
                    "ALTER TABLE campaign_guided_setup_step_receipts ADD COLUMN previous_receipt_digest TEXT"
                )
        self._initialized = True

    @classmethod
    def open_binding_registry(
        cls, db_path: str | Path, *, sealer: ArtifactSealer | None = None
    ) -> GuidedSetupRepository:
        """Open existing installation bindings without creating any schema or key."""

        value = cls(db_path, sealer=sealer)
        with value._connection() as connection:
            required = {
                str(row["name"])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        if (
            not {
                "campaigns",
                "campaign_recovery_installations",
                "campaign_recovery_bindings",
            }
            <= required
        ):
            raise GuidedSetupError("installation binding registry is unavailable")
        value._initialized = True
        return value

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise GuidedSetupError("guided setup repository is not initialized")

    def _require_sealer(self) -> ArtifactSealer:
        if self.sealer is None:
            raise GuidedSetupError("guided setup receipt authority is unavailable")
        return self.sealer

    def _seal(self, authority: dict[str, Any]) -> str:
        signature = self._require_sealer().sign_canonical_payload(
            authority,
            domain="bashgym.guided-setup-receipt.v1",
        )
        return f"sha256:{signature}"

    @staticmethod
    def _authority(
        row: sqlite3.Row | dict[str, Any], *, consumed: str | None = None
    ) -> dict[str, Any]:
        return {
            "receipt_id": str(row["receipt_id"]),
            "workspace_id": str(row["workspace_id"]),
            "actor_id": str(row["actor_id"]),
            "idempotency_key": str(row["idempotency_key"]),
            "request_hash": str(row["request_hash"]),
            "template_id": str(row["template_id"]),
            "definition_digest": str(row["definition_digest"]),
            "installation_id": str(row["installation_id"]),
            "bindings": json.loads(str(row["bindings_json"])),
            "binding_state_digest": str(row["binding_state_digest"]),
            "response": json.loads(str(row["response_json"])),
            "ready": bool(row["ready"]),
            "consumed_campaign_id": (
                consumed if consumed is not None else row["consumed_campaign_id"]
            ),
            "created_at": str(row["created_at"]),
        }

    def _verify_row(self, row: sqlite3.Row) -> None:
        sealer = self._require_sealer()
        if str(row["seal_key_version"]) != sealer.key_version or not hmac.compare_digest(
            self._seal(self._authority(row)), str(row["response_seal"])
        ):
            raise GuidedSetupConflictError("guided setup receipt seal is invalid")

    @staticmethod
    def template_summary(definition: AutoResearchTemplateDefinition) -> dict[str, Any]:
        expected = _expected_bindings(definition)
        return {
            "schema_version": "guided_setup_template.v1",
            "template_id": definition.template_id,
            "definition_digest": definition.definition_digest,
            "quality_claim_eligible": bool(
                definition.policy and definition.policy.quality_claim_eligible
            ),
            "required_bindings": expected.model_dump(mode="json"),
        }

    @staticmethod
    def _session_tables_available(connection: sqlite3.Connection) -> bool:
        tables = {str(row["name"]) for row in connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'campaign_guided_setup_sessions',
                    'campaign_guided_setup_step_receipts'
                )
                """).fetchall()}
        return tables == {
            "campaign_guided_setup_sessions",
            "campaign_guided_setup_step_receipts",
        }

    @staticmethod
    def bounded_template_summaries(
        definitions: dict[str, AutoResearchTemplateDefinition],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return a deterministic bounded public template projection."""

        ordered = sorted(definitions.items())[: GUIDED_SETUP_MAX_TEMPLATES + 1]
        return (
            [
                GuidedSetupRepository.template_summary(definition)
                for _template_id, definition in ordered[:GUIDED_SETUP_MAX_TEMPLATES]
            ],
            len(ordered) > GUIDED_SETUP_MAX_TEMPLATES,
        )

    @staticmethod
    def _discover_installations(
        connection: sqlite3.Connection,
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        installations: list[dict[str, Any]] = []
        installation_rows = connection.execute(
            """
            SELECT installation_id FROM campaign_recovery_installations
            ORDER BY installation_id LIMIT ?
            """,
            (GUIDED_SETUP_MAX_INSTALLATIONS + 1,),
        ).fetchall()
        installations_truncated = len(installation_rows) > GUIDED_SETUP_MAX_INSTALLATIONS
        any_bindings_truncated = False
        for installation in installation_rows[:GUIDED_SETUP_MAX_INSTALLATIONS]:
            installation_id = str(installation["installation_id"])
            bindings: dict[str, list[dict[str, Any]]] = {
                "model": [],
                "data": [],
                "compute": [],
                "evaluation": [],
            }
            storage_to_public = {storage: public for public, storage in _KIND_MAP}
            rows = connection.execute(
                """
                SELECT binding_kind, logical_id, availability, display_label,
                       integration_label, total_for_kind
                FROM (
                    SELECT binding_kind, logical_id, availability, display_label,
                           integration_label,
                           ROW_NUMBER() OVER (
                               PARTITION BY binding_kind ORDER BY logical_id
                           ) AS ordinal,
                           COUNT(*) OVER (PARTITION BY binding_kind) AS total_for_kind
                    FROM campaign_recovery_bindings
                    WHERE installation_id=?
                      AND binding_kind IN ('model', 'data', 'compute', 'evaluator')
                )
                WHERE ordinal <= ?
                ORDER BY binding_kind, logical_id
                """,
                (installation_id, GUIDED_SETUP_MAX_BINDINGS_PER_KIND),
            ).fetchall()
            truncated_kinds: set[str] = set()
            for row in rows:
                public_kind = storage_to_public.get(str(row["binding_kind"]))
                if public_kind is None:
                    continue
                if int(row["total_for_kind"]) > GUIDED_SETUP_MAX_BINDINGS_PER_KIND:
                    truncated_kinds.add(public_kind)
                availability = str(row["availability"])
                reason_codes = (
                    []
                    if availability == "reachable"
                    else [
                        (
                            "binding_inaccessible"
                            if availability == "inaccessible"
                            else "binding_availability_unknown"
                        )
                    ]
                )
                item: dict[str, Any] = {
                    "logical_id": str(row["logical_id"]),
                    "availability": availability,
                    "selectable": availability == "reachable",
                    "reason_codes": reason_codes,
                }
                if row["display_label"] is not None:
                    item["display_label"] = str(row["display_label"])
                if row["integration_label"] is not None:
                    item["integration_label"] = str(row["integration_label"])
                bindings[public_kind].append(item)
            reason_codes = [
                f"{kind}_binding_unavailable"
                for kind, items in bindings.items()
                if not any(item["selectable"] for item in items)
            ]
            installations.append(
                {
                    "installation_id": installation_id,
                    "ready": not reason_codes,
                    "reason_codes": reason_codes,
                    "bindings": bindings,
                    "truncation": {
                        "truncated": bool(truncated_kinds),
                        "reason_codes": ["bindings_truncated"] if truncated_kinds else [],
                        "limit_per_kind": GUIDED_SETUP_MAX_BINDINGS_PER_KIND,
                        "kinds": sorted(truncated_kinds),
                    },
                }
            )
            any_bindings_truncated = any_bindings_truncated or bool(truncated_kinds)
        return installations, installations_truncated, any_bindings_truncated

    def _step_authority(self, row: sqlite3.Row, response_base: dict[str, Any]) -> dict[str, Any]:
        return {
            "receipt_id": str(row["receipt_id"]),
            "workspace_id": str(row["workspace_id"]),
            "actor_id": str(row["actor_id"]),
            "session_id": str(row["session_id"]),
            "version": int(row["version"]),
            "step": str(row["step"]),
            "selection_id": str(row["selection_id"]),
            "idempotency_key": str(row["idempotency_key"]),
            "request_hash": str(row["request_hash"]),
            "previous_receipt_id": row["previous_receipt_id"],
            "previous_receipt_digest": row["previous_receipt_digest"],
            "state": json.loads(str(row["state_json"])),
            "state_digest": str(row["state_digest"]),
            "response": response_base,
            "created_at": str(row["created_at"]),
        }

    @staticmethod
    def _response_base(response: dict[str, Any]) -> dict[str, Any]:
        value = json.loads(_canonical(response))
        value["receipt"].pop("receipt_digest", None)
        value["session"]["latest_receipt"].pop("receipt_digest", None)
        return value

    def _verify_step_receipt(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            state = json.loads(str(row["state_json"]))
            response = json.loads(str(row["response_json"]))
            response_base = self._response_base(response)
            seal = str(row["response_seal"])
            if (
                canonical_hash(state) != str(row["state_digest"])
                or str(row["seal_key_version"]) != self._require_sealer().key_version
                or response["receipt"]["receipt_digest"] != seal
                or response["session"]["latest_receipt"]["receipt_digest"] != seal
                or response["receipt"].get("previous_receipt_id") != row["previous_receipt_id"]
                or response["receipt"].get("previous_receipt_digest")
                != row["previous_receipt_digest"]
                or not hmac.compare_digest(
                    self._seal_step(self._step_authority(row, response_base)), seal
                )
            ):
                raise ValueError("step receipt authority mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GuidedSetupConflictError("guided setup step receipt seal is invalid") from exc
        return response

    def _verify_receipt_chain(
        self, connection: sqlite3.Connection, latest: sqlite3.Row
    ) -> dict[str, Any]:
        current = latest
        latest_response: dict[str, Any] | None = None
        expected_version = int(latest["version"])
        while True:
            response = self._verify_step_receipt(current)
            if latest_response is None:
                latest_response = response
            if int(current["version"]) != expected_version:
                raise GuidedSetupConflictError("guided setup step receipt chain is invalid")
            previous_id = current["previous_receipt_id"]
            previous_digest = current["previous_receipt_digest"]
            if expected_version == 1:
                if previous_id is not None or previous_digest is not None:
                    raise GuidedSetupConflictError("guided setup step receipt chain is invalid")
                break
            if previous_id is None or previous_digest is None:
                raise GuidedSetupConflictError("guided setup step receipt chain is incomplete")
            previous = connection.execute(
                "SELECT * FROM campaign_guided_setup_step_receipts WHERE receipt_id=?",
                (str(previous_id),),
            ).fetchone()
            if (
                previous is None
                or str(previous["workspace_id"]) != str(latest["workspace_id"])
                or str(previous["actor_id"]) != str(latest["actor_id"])
                or str(previous["session_id"]) != str(latest["session_id"])
                or not hmac.compare_digest(str(previous["response_seal"]), str(previous_digest))
            ):
                raise GuidedSetupConflictError("guided setup step receipt chain is invalid")
            current = previous
            expected_version -= 1
        assert latest_response is not None
        return latest_response

    def _seal_step(self, authority: dict[str, Any]) -> str:
        signature = self._require_sealer().sign_canonical_payload(
            authority,
            domain="bashgym.guided-setup-step-receipt.v1",
        )
        return f"sha256:{signature}"

    def _verified_session_state(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        receipt = connection.execute(
            "SELECT * FROM campaign_guided_setup_step_receipts WHERE receipt_id=?",
            (str(row["latest_receipt_id"]),),
        ).fetchone()
        if receipt is None:
            raise GuidedSetupConflictError("guided setup step receipt is missing")
        response = self._verify_receipt_chain(connection, receipt)
        state = json.loads(str(row["state_json"]))
        if (
            int(row["version"]) != int(receipt["version"])
            or str(row["state_digest"]) != str(receipt["state_digest"])
            or canonical_hash(state) != str(row["state_digest"])
            or state != json.loads(str(receipt["state_json"]))
        ):
            raise GuidedSetupConflictError("guided setup session authority conflicts")
        return state, response

    def _project_session(
        self,
        connection: sqlite3.Connection,
        state: dict[str, Any],
        definitions: dict[str, AutoResearchTemplateDefinition],
        latest_receipt: dict[str, Any],
    ) -> dict[str, Any]:
        selections = state["selections"]
        template_id = selections["template_id"]
        installation_id = selections["installation_id"]
        selected_bindings = dict(selections["bindings"])
        reasons: list[str] = []
        definition = definitions.get(template_id) if isinstance(template_id, str) else None
        if template_id is None:
            reasons.append("template_not_selected")
        elif definition is None:
            reasons.append("template_unavailable")
        installation_exists = False
        if installation_id is None:
            reasons.append("installation_not_selected")
        else:
            installation_exists = (
                connection.execute(
                    "SELECT 1 FROM campaign_recovery_installations WHERE installation_id=?",
                    (installation_id,),
                ).fetchone()
                is not None
            )
            if not installation_exists:
                reasons.append("installation_unregistered")
        expected = _expected_bindings(definition) if definition is not None else None
        expected_values = expected.model_dump(mode="json") if expected is not None else {}
        for public_kind, storage_kind in _KIND_MAP:
            logical_id = selected_bindings.get(public_kind)
            if logical_id is None:
                reasons.append(f"{public_kind}_binding_not_selected")
                continue
            row = (
                connection.execute(
                    """
                    SELECT availability FROM campaign_recovery_bindings
                    WHERE installation_id=? AND binding_kind=? AND logical_id=?
                    """,
                    (installation_id, storage_kind, logical_id),
                ).fetchone()
                if installation_exists
                else None
            )
            if row is None:
                reasons.append(f"{public_kind}_binding_unregistered")
            elif str(row["availability"]) != "reachable":
                reasons.append(f"{public_kind}_binding_unavailable")
            if expected is not None and logical_id != expected_values[public_kind]:
                reasons.append(f"{public_kind}_binding_contract_mismatch")
        reasons = sorted(set(reasons))
        return {
            "schema_version": "guided_setup_session.v1",
            "workspace_id": str(state["workspace_id"]),
            "session_id": str(state["session_id"]),
            "version": int(state["version"]),
            "completed_steps": list(state["completed_steps"]),
            "selections": {
                "template_id": template_id,
                "installation_id": installation_id,
                "bindings": selected_bindings,
            },
            "ready_for_validation": not reasons,
            "reason_codes": reasons,
            "latest_receipt": latest_receipt,
            "updated_at": str(state["updated_at"]),
        }

    def context(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        definitions: dict[str, AutoResearchTemplateDefinition],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Project public discovery and an optional verified resumable session."""

        self._require_initialized()
        if _PUBLIC_ID.fullmatch(workspace_id) is None or _PUBLIC_ID.fullmatch(actor_id) is None:
            raise GuidedSetupError("guided setup context scope is invalid")
        if session_id is not None and _SESSION_ID.fullmatch(session_id) is None:
            raise GuidedSetupError("guided setup session identity is invalid")
        with self._connection() as connection:
            installations, installations_truncated, bindings_truncated = (
                self._discover_installations(connection)
            )
            templates, templates_truncated = self.bounded_template_summaries(definitions)
            session = None
            reason_codes = ["setup_session_not_started"]
            if session_id is not None:
                reason_codes = ["setup_session_not_found"]
                if self._session_tables_available(connection):
                    row = connection.execute(
                        """
                        SELECT * FROM campaign_guided_setup_sessions
                        WHERE workspace_id=? AND actor_id=? AND session_id=?
                        """,
                        (workspace_id, actor_id, session_id),
                    ).fetchone()
                    if row is not None:
                        state, response = self._verified_session_state(connection, row)
                        session = self._project_session(
                            connection,
                            state,
                            definitions,
                            response["receipt"],
                        )
                        reason_codes = list(session["reason_codes"])
            truncation_reasons = []
            if bindings_truncated:
                truncation_reasons.append("bindings_truncated")
            if installations_truncated:
                truncation_reasons.append("installations_truncated")
            if templates_truncated:
                truncation_reasons.append("templates_truncated")
            return {
                "schema_version": "guided_setup_context.v1",
                "workspace_id": workspace_id,
                "templates": templates,
                "installations": installations,
                "session": session,
                "reason_codes": reason_codes,
                "truncation": {
                    "truncated": bool(truncation_reasons),
                    "reason_codes": truncation_reasons,
                    "limits": {
                        "templates": GUIDED_SETUP_MAX_TEMPLATES,
                        "installations": GUIDED_SETUP_MAX_INSTALLATIONS,
                        "bindings_per_kind": GUIDED_SETUP_MAX_BINDINGS_PER_KIND,
                    },
                },
            }

    def advance_session(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        session_id: str,
        expected_version: int,
        step: str,
        selection_id: str,
        definitions: dict[str, AutoResearchTemplateDefinition],
        idempotency_key: str,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Persist one exact, ordered setup choice and its sealed receipt."""

        self._require_initialized()
        if (
            _PUBLIC_ID.fullmatch(workspace_id) is None
            or _PUBLIC_ID.fullmatch(actor_id) is None
            or _SESSION_ID.fullmatch(session_id) is None
            or _PUBLIC_ID.fullmatch(selection_id) is None
            or _PUBLIC_ID.fullmatch(idempotency_key) is None
            or step not in _SESSION_STEPS
            or expected_version < 0
        ):
            raise GuidedSetupError("guided setup session mutation is invalid")
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "actor_id": actor_id,
                "session_id": session_id,
                "expected_version": expected_version,
                "step": step,
                "selection_id": selection_id,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = connection.execute(
                """
                SELECT * FROM campaign_guided_setup_step_receipts
                WHERE workspace_id=? AND actor_id=? AND idempotency_key=?
                """,
                (workspace_id, actor_id, idempotency_key),
            ).fetchone()
            if replay is not None:
                if not hmac.compare_digest(str(replay["request_hash"]), request_hash):
                    raise GuidedSetupConflictError("guided setup step idempotency conflict")
                return self._verify_step_receipt(replay), True

            current = connection.execute(
                """
                SELECT * FROM campaign_guided_setup_sessions
                WHERE workspace_id=? AND actor_id=? AND session_id=?
                """,
                (workspace_id, actor_id, session_id),
            ).fetchone()
            previous_receipt_id: str | None = None
            previous_receipt_digest: str | None = None
            if current is None:
                if expected_version != 0 or step != "template":
                    raise GuidedSetupConflictError("guided setup step order conflicts")
                state = {
                    "schema_version": "guided_setup_session_state.v1",
                    "workspace_id": workspace_id,
                    "actor_id": actor_id,
                    "session_id": session_id,
                    "version": 0,
                    "completed_steps": [],
                    "selections": {
                        "template_id": None,
                        "installation_id": None,
                        "bindings": {},
                    },
                    "updated_at": _wire_time(now or datetime.now(UTC)),
                }
            else:
                state, previous_response = self._verified_session_state(connection, current)
                previous_receipt_id = str(previous_response["receipt"]["receipt_id"])
                previous_receipt_digest = str(previous_response["receipt"]["receipt_digest"])
                if int(current["version"]) != expected_version:
                    raise GuidedSetupConflictError("guided setup session version conflicts")
                if (
                    expected_version >= len(_SESSION_STEPS)
                    or step != _SESSION_STEPS[expected_version]
                ):
                    raise GuidedSetupConflictError("guided setup step order conflicts")

            selections = state["selections"]
            if step == "template":
                if selection_id not in definitions:
                    raise GuidedSetupConflictError("guided setup template is not registered")
                selections["template_id"] = selection_id
            elif step == "installation":
                registered = connection.execute(
                    "SELECT 1 FROM campaign_recovery_installations WHERE installation_id=?",
                    (selection_id,),
                ).fetchone()
                if registered is None:
                    raise GuidedSetupConflictError("guided setup installation is not registered")
                selections["installation_id"] = selection_id
            else:
                template_id = selections["template_id"]
                installation_id = selections["installation_id"]
                definition = definitions.get(template_id)
                if definition is None or installation_id is None:
                    raise GuidedSetupConflictError("guided setup step order conflicts")
                storage_kind = dict(_KIND_MAP)[step]
                registered = connection.execute(
                    """
                    SELECT availability FROM campaign_recovery_bindings
                    WHERE installation_id=? AND binding_kind=? AND logical_id=?
                    """,
                    (installation_id, storage_kind, selection_id),
                ).fetchone()
                if registered is None:
                    raise GuidedSetupConflictError("guided setup binding is not registered")
                if str(registered["availability"]) != "reachable":
                    raise GuidedSetupConflictError("guided setup binding is not reachable")
                expected = _expected_bindings(definition).model_dump(mode="json")
                if selection_id != expected[step]:
                    raise GuidedSetupConflictError("guided setup binding contract conflicts")
                selections["bindings"][step] = selection_id

            version = expected_version + 1
            created_at = _wire_time(now or datetime.now(UTC))
            state["version"] = version
            state["completed_steps"] = list(_SESSION_STEPS[:version])
            state["updated_at"] = created_at
            state_json = _canonical(state)
            state_digest = canonical_hash(state)
            receipt_id = (
                "setupstep_"
                + hashlib.sha256(
                    f"{workspace_id}\x1f{actor_id}\x1f{session_id}\x1f{idempotency_key}\x1f{request_hash}".encode()
                ).hexdigest()[:32]
            )
            receipt_base = {
                "schema_version": "guided_setup_step_receipt.v1",
                "receipt_id": receipt_id,
                "session_id": session_id,
                "version": version,
                "step": step,
                "selection_id": selection_id,
                "state_digest": state_digest,
                "previous_receipt_id": previous_receipt_id,
                "previous_receipt_digest": previous_receipt_digest,
                "created_at": created_at,
            }
            session_projection = self._project_session(connection, state, definitions, receipt_base)
            response_base = {
                "schema_version": "guided_setup_session_mutation.v1",
                "session": session_projection,
                "receipt": receipt_base,
            }
            authority = {
                "receipt_id": receipt_id,
                "workspace_id": workspace_id,
                "actor_id": actor_id,
                "session_id": session_id,
                "version": version,
                "step": step,
                "selection_id": selection_id,
                "idempotency_key": idempotency_key,
                "request_hash": request_hash,
                "previous_receipt_id": previous_receipt_id,
                "previous_receipt_digest": previous_receipt_digest,
                "state": state,
                "state_digest": state_digest,
                "response": response_base,
                "created_at": created_at,
            }
            seal = self._seal_step(authority)
            response = json.loads(_canonical(response_base))
            response["receipt"]["receipt_digest"] = seal
            response["session"]["latest_receipt"]["receipt_digest"] = seal
            connection.execute(
                """
                INSERT INTO campaign_guided_setup_sessions(
                    workspace_id, actor_id, session_id, version, state_json,
                    state_digest, latest_receipt_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, actor_id, session_id) DO UPDATE SET
                    version=excluded.version,
                    state_json=excluded.state_json,
                    state_digest=excluded.state_digest,
                    latest_receipt_id=excluded.latest_receipt_id,
                    updated_at=excluded.updated_at
                """,
                (
                    workspace_id,
                    actor_id,
                    session_id,
                    version,
                    state_json,
                    state_digest,
                    receipt_id,
                    created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_guided_setup_step_receipts(
                    receipt_id, workspace_id, actor_id, session_id, version, step,
                    selection_id, idempotency_key, request_hash, state_json,
                    previous_receipt_id, previous_receipt_digest, state_digest,
                    response_json, response_seal, seal_key_version,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    workspace_id,
                    actor_id,
                    session_id,
                    version,
                    step,
                    selection_id,
                    idempotency_key,
                    request_hash,
                    state_json,
                    previous_receipt_id,
                    previous_receipt_digest,
                    state_digest,
                    _canonical(response),
                    seal,
                    self._require_sealer().key_version,
                    created_at,
                ),
            )
            return response, False

    @staticmethod
    def _binding_rows(
        connection: sqlite3.Connection, draft: GuidedSetupDraft
    ) -> tuple[list[dict[str, str]], str]:
        installation = connection.execute(
            "SELECT 1 FROM campaign_recovery_installations WHERE installation_id=?",
            (draft.installation_id,),
        ).fetchone()
        references: list[dict[str, str]] = []
        values = draft.bindings.model_dump(mode="json")
        for public_kind, storage_kind in _KIND_MAP:
            logical_id = str(values[public_kind])
            row = (
                connection.execute(
                    """
                    SELECT availability FROM campaign_recovery_bindings
                    WHERE installation_id=? AND binding_kind=? AND logical_id=?
                    """,
                    (draft.installation_id, storage_kind, logical_id),
                ).fetchone()
                if installation is not None
                else None
            )
            references.append(
                {
                    "kind": public_kind,
                    "logical_id": logical_id,
                    "availability": str(row["availability"]) if row is not None else "missing",
                }
            )
        return references, canonical_hash(references)

    def _report(
        self,
        draft: GuidedSetupDraft,
        *,
        definition: AutoResearchTemplateDefinition,
        doctor: AutoResearchDoctorReport,
    ) -> tuple[dict[str, Any], str]:
        if (
            draft.template_id != definition.template_id
            or doctor.template_id != definition.template_id
        ):
            raise GuidedSetupError("guided setup template scope mismatch")
        if draft.workspace_id != doctor.workspace_id:
            raise GuidedSetupError("guided setup workspace scope mismatch")
        if doctor.definition_digest != definition.definition_digest:
            raise GuidedSetupConflictError("guided setup doctor is stale")
        expected = _expected_bindings(definition)
        with self._connection() as connection:
            references, binding_state_digest = self._binding_rows(connection, draft)
        submitted = draft.bindings.model_dump(mode="json")
        wanted = expected.model_dump(mode="json")
        # Campaign materialization is intentionally allowed while the resident
        # controller is offline; launch remains guarded by the ordinary Start
        # doctor. Every non-controller doctor failure still blocks creation.
        blocking = [
            check.code
            for check in doctor.checks
            if not check.ready and check.check_id != "controller"
        ]
        for kind, value in submitted.items():
            if value != wanted[kind]:
                blocking.append(f"{kind}_binding_contract_mismatch")
        for reference in references:
            if reference["availability"] != "reachable":
                blocking.append(f"{reference['kind']}_binding_unavailable")
        blocking = sorted(set(blocking))
        return (
            {
                "schema_version": "guided_setup_doctor.v1",
                "workspace_id": draft.workspace_id,
                "template_id": draft.template_id,
                "definition_digest": definition.definition_digest,
                "draft_digest": canonical_hash(draft.model_dump(mode="json")),
                "ready": doctor.materializable and not blocking,
                "blocking_codes": blocking,
                "binding_references": references,
            },
            binding_state_digest,
        )

    def doctor(
        self,
        draft: GuidedSetupDraft,
        *,
        definition: AutoResearchTemplateDefinition,
        doctor: AutoResearchDoctorReport,
    ) -> dict[str, Any]:
        """Run current checks without writing a receipt or changing campaign state."""

        self._require_initialized()
        report, _state = self._report(draft, definition=definition, doctor=doctor)
        return report

    def validate(
        self,
        draft: GuidedSetupDraft,
        *,
        definition: AutoResearchTemplateDefinition,
        doctor: AutoResearchDoctorReport,
        actor_id: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Seal the exact current draft and readiness state as durable authority."""

        self._require_initialized()
        if not actor_id or len(actor_id) > 160 or not idempotency_key or len(idempotency_key) > 160:
            raise GuidedSetupError("guided setup mutation identity is invalid")
        request_hash = canonical_hash(
            {
                "draft": draft.model_dump(mode="json"),
                "definition_digest": definition.definition_digest,
            }
        )
        with self._connection(immediate=True) as connection:
            previous = connection.execute(
                """
                SELECT *
                FROM campaign_guided_setup_receipts
                WHERE workspace_id=? AND actor_id=? AND idempotency_key=?
                """,
                (draft.workspace_id, actor_id, idempotency_key),
            ).fetchone()
            if previous is not None:
                if not hmac.compare_digest(str(previous["request_hash"]), request_hash):
                    raise GuidedSetupConflictError("guided setup idempotency conflict")
                self._verify_row(previous)
                return json.loads(str(previous["response_json"])), True

            report, binding_state_digest = self._report(draft, definition=definition, doctor=doctor)
            receipt_id = (
                "setuprcpt_"
                + hashlib.sha256(
                    f"{draft.workspace_id}\x1f{actor_id}\x1f{idempotency_key}\x1f{request_hash}".encode()
                ).hexdigest()[:32]
            )
            response = {**report, "receipt_id": receipt_id}
            response_json = _canonical(response)
            created_at = _wire_time(now or datetime.now(UTC))
            authority = {
                "receipt_id": receipt_id,
                "workspace_id": draft.workspace_id,
                "actor_id": actor_id,
                "idempotency_key": idempotency_key,
                "request_hash": request_hash,
                "template_id": draft.template_id,
                "definition_digest": definition.definition_digest,
                "installation_id": draft.installation_id,
                "bindings_json": _canonical(draft.bindings.model_dump(mode="json")),
                "binding_state_digest": binding_state_digest,
                "response_json": response_json,
                "ready": 1 if report["ready"] else 0,
                "consumed_campaign_id": None,
                "created_at": created_at,
            }
            connection.execute(
                """
                INSERT INTO campaign_guided_setup_receipts(
                    receipt_id, workspace_id, actor_id, idempotency_key, request_hash,
                    template_id, definition_digest, installation_id, bindings_json,
                    binding_state_digest, response_json, response_seal, seal_key_version, ready,
                    consumed_campaign_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    receipt_id,
                    draft.workspace_id,
                    actor_id,
                    idempotency_key,
                    request_hash,
                    draft.template_id,
                    definition.definition_digest,
                    draft.installation_id,
                    authority["bindings_json"],
                    binding_state_digest,
                    response_json,
                    self._seal(self._authority(authority)),
                    self._require_sealer().key_version,
                    1 if report["ready"] else 0,
                    created_at,
                ),
            )
            return response, False

    def authorize_creation(
        self,
        receipt_id: str,
        *,
        workspace_id: str,
        campaign_id: str,
        definition: AutoResearchTemplateDefinition,
        actor_id: str,
    ) -> GuidedSetupDraft:
        """Revalidate sealed setup authority immediately before campaign creation."""

        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_guided_setup_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            if row is None:
                raise GuidedSetupConflictError("guided setup receipt was not found")
            if str(row["workspace_id"]) != workspace_id or str(row["actor_id"]) != actor_id:
                raise GuidedSetupConflictError("guided setup receipt scope mismatch")
            consumed = row["consumed_campaign_id"]
            if consumed is not None and str(consumed) != campaign_id:
                raise GuidedSetupConflictError("guided setup receipt belongs to another campaign")
            self._verify_row(row)
            if not bool(row["ready"]):
                raise GuidedSetupConflictError("guided setup receipt is not ready")
            if not hmac.compare_digest(str(row["definition_digest"]), definition.definition_digest):
                raise GuidedSetupConflictError("guided setup template is stale")
            draft = GuidedSetupDraft(
                workspace_id=workspace_id,
                template_id=str(row["template_id"]),
                installation_id=str(row["installation_id"]),
                bindings=GuidedSetupBindings.model_validate_json(str(row["bindings_json"])),
            )
            _references, current_digest = self._binding_rows(connection, draft)
            if not hmac.compare_digest(str(row["binding_state_digest"]), current_digest):
                raise GuidedSetupConflictError("guided setup bindings are stale")
            return draft

    def receipt_template_id(self, receipt_id: str, *, workspace_id: str, actor_id: str) -> str:
        """Return the scope-bound template ID from an intact sealed receipt."""

        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_guided_setup_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            if row is None:
                raise GuidedSetupConflictError("guided setup receipt was not found")
            if str(row["workspace_id"]) != workspace_id or str(row["actor_id"]) != actor_id:
                raise GuidedSetupConflictError("guided setup receipt scope mismatch")
            self._verify_row(row)
            return str(row["template_id"])

    @staticmethod
    def _binding_authority(
        row: sqlite3.Row, receipt_id: str
    ) -> tuple[str, str, str, str, str, str]:
        bindings = GuidedSetupBindings.model_validate_json(str(row["bindings_json"]))
        return (
            receipt_id,
            str(row["installation_id"]),
            bindings.model,
            bindings.data,
            bindings.compute,
            bindings.evaluation,
        )

    @staticmethod
    def _binding_row(
        connection: sqlite3.Connection, workspace_id: str, campaign_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT validation_receipt_id, installation_id, model_binding_id,
                   data_binding_id, compute_binding_id, evaluation_binding_id
            FROM campaign_guided_setup_bindings
            WHERE workspace_id=? AND campaign_id=?
            """,
            (workspace_id, campaign_id),
        ).fetchone()

    def create_campaign_atomically(
        self,
        receipt_id: str,
        *,
        repository: CampaignRepository,
        campaign: Campaign,
        manifest_revision: ManifestRevision,
        definition: AutoResearchTemplateDefinition,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> GuidedSetupDraft:
        """Create the campaign and guided binding authority in one transaction."""

        self._require_initialized()
        repository._require_initialized()
        if repository.db_path.resolve() != self.db_path.resolve():
            raise GuidedSetupError("guided setup campaign repository scope mismatch")
        if len(manifest_revision.manifest.budget_limits) > 64:
            raise CampaignBudgetResourceLimitError()
        if campaign.status != CampaignStatus.DRAFT or campaign.version != 1:
            raise GuidedSetupError("guided setup campaign state is invalid")
        if (
            manifest_revision.workspace_id != campaign.workspace_id
            or manifest_revision.campaign_id != campaign.campaign_id
            or manifest_revision.revision != campaign.manifest_revision
        ):
            raise GuidedSetupError("guided setup manifest identity is invalid")

        request_hash = canonical_hash(
            {
                "campaign": campaign.model_dump(mode="json", exclude={"created_at", "updated_at"}),
                "manifest": manifest_revision.manifest.model_dump(mode="json"),
            }
        )
        mutation_kind = "campaign.create"
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM campaign_guided_setup_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            if row is None:
                raise GuidedSetupConflictError("guided setup receipt was not found")
            if (
                str(row["workspace_id"]) != campaign.workspace_id
                or str(row["actor_id"]) != actor_id
            ):
                raise GuidedSetupConflictError("guided setup receipt scope mismatch")
            self._verify_row(row)
            consumed = row["consumed_campaign_id"]
            if consumed is not None and str(consumed) != campaign.campaign_id:
                raise GuidedSetupConflictError("guided setup receipt belongs to another campaign")
            if not bool(row["ready"]):
                raise GuidedSetupConflictError("guided setup receipt is not ready")
            if str(row["template_id"]) != definition.template_id or not hmac.compare_digest(
                str(row["definition_digest"]), definition.definition_digest
            ):
                raise GuidedSetupConflictError("guided setup template is stale")
            draft = GuidedSetupDraft(
                workspace_id=campaign.workspace_id,
                template_id=str(row["template_id"]),
                installation_id=str(row["installation_id"]),
                bindings=GuidedSetupBindings.model_validate_json(str(row["bindings_json"])),
            )
            _references, current_digest = self._binding_rows(connection, draft)
            if not hmac.compare_digest(str(row["binding_state_digest"]), current_digest):
                raise GuidedSetupConflictError("guided setup bindings are stale")

            replay = repository._replay(
                connection,
                workspace_id=campaign.workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            expected_binding = self._binding_authority(row, receipt_id)
            if replay is not None:
                existing_binding = self._binding_row(
                    connection, campaign.workspace_id, campaign.campaign_id
                )
                if (
                    consumed != campaign.campaign_id
                    or existing_binding is None
                    or tuple(existing_binding) != expected_binding
                ):
                    raise GuidedSetupConflictError(
                        "guided setup campaign binding authority conflicts"
                    )
                return draft

            existing = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id=? AND campaign_id=?",
                (campaign.workspace_id, campaign.campaign_id),
            ).fetchone()
            if existing is not None:
                raise RecordAlreadyExistsError("campaign already exists")

            connection.execute(
                """
                UPDATE campaign_guided_setup_receipts
                SET consumed_campaign_id=?, response_seal=?
                WHERE receipt_id=?
                """,
                (
                    campaign.campaign_id,
                    self._seal(self._authority(row, consumed=campaign.campaign_id)),
                    receipt_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO campaigns(
                    workspace_id, campaign_id, title, kind, objective, target_model_json,
                    owner_actor_id, manifest_revision, status, prior_scheduling_status,
                    active_study_id, active_action_id, champion_ref,
                    best_development_candidate_ref, stop_reason, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign.workspace_id,
                    campaign.campaign_id,
                    campaign.title,
                    campaign.kind.value,
                    campaign.objective,
                    _canonical(campaign.target_model.model_dump(mode="json")),
                    campaign.owner_actor_id,
                    campaign.manifest_revision,
                    campaign.status.value,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    campaign.version,
                    campaign.created_at.isoformat(),
                    campaign.updated_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_manifest_revisions(
                    workspace_id, campaign_id, revision, manifest_json, manifest_hash,
                    actor_id, correlation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest_revision.workspace_id,
                    manifest_revision.campaign_id,
                    manifest_revision.revision,
                    _canonical(manifest_revision.manifest.model_dump(mode="json")),
                    manifest_revision.manifest_hash,
                    manifest_revision.actor_id,
                    manifest_revision.correlation_id,
                    manifest_revision.created_at.isoformat(),
                ),
            )
            event = CampaignEvent(
                event_id=(
                    "evt-"
                    + hashlib.sha256(
                        f"{campaign.campaign_id}:{idempotency_key}".encode()
                    ).hexdigest()[:24]
                ),
                workspace_id=campaign.workspace_id,
                campaign_id=campaign.campaign_id,
                sequence=1,
                aggregate_version=1,
                event_type="campaign:created",
                payload={"manifest_hash": manifest_revision.manifest_hash},
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
            CampaignRepository._insert_event(connection, event)
            CampaignRepository._insert_mutation(
                connection,
                workspace_id=campaign.workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=campaign,
                event=event,
            )
            connection.execute(
                """
                INSERT INTO campaign_guided_setup_bindings(
                    workspace_id, campaign_id, validation_receipt_id, installation_id,
                    model_binding_id, data_binding_id, compute_binding_id,
                    evaluation_binding_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign.workspace_id,
                    campaign.campaign_id,
                    *expected_binding,
                ),
            )
            return draft

    def record_creation(self, receipt_id: str, workspace_id: str, campaign_id: str) -> None:
        """Verify an atomically created campaign's sealed logical setup references."""

        self._require_initialized()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM campaign_guided_setup_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            if row is None or str(row["workspace_id"]) != workspace_id:
                raise GuidedSetupConflictError("guided setup receipt scope mismatch")
            self._verify_row(row)
            consumed = row["consumed_campaign_id"]
            if consumed is not None and str(consumed) != campaign_id:
                raise GuidedSetupConflictError("guided setup receipt belongs to another campaign")
            existing = self._binding_row(connection, workspace_id, campaign_id)
            if (
                consumed != campaign_id
                or existing is None
                or tuple(existing) != self._binding_authority(row, receipt_id)
            ):
                raise GuidedSetupConflictError("campaign setup binding authority conflicts")


__all__ = [
    "GuidedSetupBindings",
    "GuidedSetupConflictError",
    "GuidedSetupDraft",
    "GuidedSetupError",
    "GuidedSetupRepository",
]
