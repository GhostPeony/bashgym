"""Durable, installation-scoped recovery authority for AutoResearch campaigns.

The public projection is deliberately reconstructed from campaign state and the
current installation registry. Private transport details remain server-side and
are represented outside the trust boundary only by opaque logical identifiers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer


class CampaignRecoveryError(ValueError):
    """Base recovery authority error."""


class CampaignRecoveryConflictError(CampaignRecoveryError):
    """Recovery authority changed or does not match the submitted request."""

    code = "campaign_recovery_conflict"


class CampaignRecoveryNotFoundError(CampaignRecoveryError):
    """The scoped campaign or recovery registration does not exist."""

    code = "campaign_recovery_not_found"


class RecoveryAction(str, Enum):
    RESUME = "resume"
    REPAIR = "repair"
    TAKEOVER = "takeover"


class RecoveryRequest(BaseModel):
    """Exact client echo of current server-sealed recovery authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: RecoveryAction
    idempotency_key: str = Field(pattern=r"^idem_[0-9a-f]{32}$")
    workspace_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
    eligibility_receipt_id: str = Field(pattern=r"^rcpt_[0-9a-f]{32}$")
    doctor_evidence_id: str = Field(pattern=r"^evd_[0-9a-f]{32}$")
    expected_campaign_revision: int = Field(ge=1)
    expected_event_cursor: int = Field(ge=0)
    expected_aggregate_version: int = Field(ge=1)
    expected_controller_lease_id: str | None = Field(default=None, pattern=r"^lease_[0-9a-f]{32}$")
    checkpoint_id: str = Field(pattern=r"^ckpt_[0-9a-f]{32}$")
    artifact_id: str = Field(pattern=r"^art_[0-9a-f]{32}$")
    human_confirmed: Literal[True]


_BINDING_PREFIX = {
    "model": "mdl",
    "data": "dat",
    "evaluator": "evl",
    "compute": "cpu",
}
_PUBLIC_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+_-]{0,95}$")
_DURABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_LABEL_DENY = re.compile(
    r"(?:[/\\]|^[A-Za-z]:|[A-Za-z][A-Za-z0-9+.-]*:|password|secret|token|"
    r"api[ ._-]*key|private[ ._-]*key|(?:^|[ ._+-])(?:gh[pousr]_|github_pat_)|"
    r"(?:^|[ ._+-])sk[_-](?:(?:proj|live|test)[_-])?[A-Za-z0-9]|"
    r"macbook(?:[ ._+-]*(?:pro|air))?|imac|iphone|ipad|laptop|desktop|"
    r"workstation|localhost|hostname)",
    re.IGNORECASE,
)


def _wire_time(value: datetime) -> str:
    value = value.astimezone(UTC)
    milliseconds = value.microsecond // 1000
    base = value.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{milliseconds:03d}Z" if milliseconds else f"{base}Z"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _opaque(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()
    return f"{prefix}_{digest[:32]}"


@dataclass(frozen=True)
class RecoveryWorkClaim:
    """One fenced resident-worker claim over an accepted recovery request."""

    request_id: str
    workspace_id: str
    campaign_id: str
    action: RecoveryAction
    status: Literal["executing", "blocked"]
    expected_campaign_revision: int
    expected_event_cursor: int
    expected_aggregate_version: int
    claim_generation: int
    attempt_id: str | None = None
    outcome_code: str | None = None


class CampaignRecoveryRepository:
    """Recovery registry, sealed receipts, and optimistic request persistence."""

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
        """Create the additive recovery tables without changing campaign migrations."""

        with self._connection(immediate=True) as connection:
            campaign_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='campaigns'"
            ).fetchone()
            if campaign_table is None:
                raise CampaignRecoveryError("campaign repository must be initialized first")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS campaign_recovery_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_recovery_installations (
                    installation_id TEXT PRIMARY KEY,
                    controller_owner_id TEXT NOT NULL,
                    controller_lease_key TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS campaign_recovery_bindings (
                    installation_id TEXT NOT NULL,
                    binding_kind TEXT NOT NULL CHECK(binding_kind IN ('model','data','evaluator','compute')),
                    logical_id TEXT NOT NULL,
                    availability TEXT NOT NULL CHECK(availability IN ('reachable','inaccessible','unknown')),
                    display_label TEXT,
                    integration_label TEXT CHECK(integration_label IS NULL OR integration_label = 'NeMo'),
                    PRIMARY KEY(installation_id, binding_kind, logical_id),
                    FOREIGN KEY(installation_id) REFERENCES campaign_recovery_installations(installation_id)
                        ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS campaign_recovery_targets (
                    workspace_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    installation_id TEXT NOT NULL,
                    lineage_mode TEXT NOT NULL CHECK(lineage_mode IN ('clone','fork')),
                    parent_campaign_id TEXT,
                    checkpoint_source_id TEXT NOT NULL,
                    artifact_source_id TEXT NOT NULL,
                    campaign_revision INTEGER NOT NULL CHECK(campaign_revision >= 1),
                    schema_id TEXT NOT NULL,
                    schema_compatible INTEGER NOT NULL CHECK(schema_compatible IN (0,1)),
                    revision_compatible INTEGER NOT NULL CHECK(revision_compatible IN (0,1)),
                    PRIMARY KEY(workspace_id, campaign_id),
                    FOREIGN KEY(workspace_id, campaign_id) REFERENCES campaigns(workspace_id, campaign_id)
                        ON DELETE RESTRICT,
                    FOREIGN KEY(installation_id) REFERENCES campaign_recovery_installations(installation_id)
                        ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS campaign_recovery_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('doctor','eligibility','recovery')),
                    state_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    receipt_digest TEXT NOT NULL,
                    seal_key_version TEXT NOT NULL,
                    emitted_at TEXT NOT NULL,
                    expires_at TEXT,
                    FOREIGN KEY(workspace_id, campaign_id) REFERENCES campaigns(workspace_id, campaign_id)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_campaign_recovery_receipts_scope
                    ON campaign_recovery_receipts(workspace_id, campaign_id, emitted_at, receipt_id);
                CREATE TABLE IF NOT EXISTS campaign_recovery_mutations (
                    workspace_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, actor_id, idempotency_key),
                    FOREIGN KEY(workspace_id, campaign_id) REFERENCES campaigns(workspace_id, campaign_id)
                        ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS campaign_recovery_requests (
                    request_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('resume','repair')),
                    accepted_receipt_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('accepted','executing','completed','failed','blocked')),
                    claim_owner_id TEXT,
                    claim_generation INTEGER NOT NULL DEFAULT 0 CHECK(claim_generation >= 0),
                    claim_expires_at TEXT,
                    target_attempt_id TEXT,
                    outcome_code TEXT,
                    outcome_receipt_id TEXT,
                    authority_digest TEXT NOT NULL,
                    authority_seal_key_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(workspace_id, campaign_id, idempotency_key),
                    FOREIGN KEY(workspace_id, campaign_id) REFERENCES campaigns(workspace_id, campaign_id)
                        ON DELETE RESTRICT,
                    FOREIGN KEY(accepted_receipt_id) REFERENCES campaign_recovery_receipts(receipt_id)
                        ON DELETE RESTRICT,
                    FOREIGN KEY(outcome_receipt_id) REFERENCES campaign_recovery_receipts(receipt_id)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_campaign_recovery_requests_claim
                    ON campaign_recovery_requests(status, created_at, request_id);
                """)
            receipt_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(campaign_recovery_receipts)"
                ).fetchall()
            }
            if "seal_key_version" not in receipt_columns:
                connection.execute(
                    "ALTER TABLE campaign_recovery_receipts ADD COLUMN seal_key_version TEXT"
                )
            request_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(campaign_recovery_requests)"
                ).fetchall()
            }
            if "authority_digest" not in request_columns:
                connection.execute("""ALTER TABLE campaign_recovery_requests
                       ADD COLUMN authority_digest TEXT NOT NULL DEFAULT ''""")
            if "authority_seal_key_version" not in request_columns:
                connection.execute("""ALTER TABLE campaign_recovery_requests
                       ADD COLUMN authority_seal_key_version TEXT NOT NULL DEFAULT ''""")
            # Legacy colocated keys are deliberately retired. Existing receipts gain a
            # NULL key version during migration and therefore fail closed instead of
            # being silently trusted under unrelated external authority.
            connection.execute("DELETE FROM campaign_recovery_meta WHERE key='seal_key'")
        self._initialized = True

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise CampaignRecoveryError("recovery repository is not initialized")

    def _require_sealer(self) -> ArtifactSealer:
        if self.sealer is None:
            raise CampaignRecoveryError("recovery receipt seal authority is unavailable")
        return self.sealer

    def _receipt_seal(
        self,
        _connection: sqlite3.Connection,
        row: sqlite3.Row | dict[str, Any],
    ) -> str:
        """Authenticate every persisted field used to select or project a receipt."""

        sealer = self._require_sealer()
        key_version = row["seal_key_version"]
        if key_version is None or not hmac.compare_digest(str(key_version), sealer.key_version):
            raise CampaignRecoveryConflictError("recovery receipt seal key version is invalid")
        envelope = {
            "schema_version": "campaign_recovery_receipt_seal.v1",
            "seal_key_version": str(key_version),
            "receipt_id": str(row["receipt_id"]),
            "workspace_id": str(row["workspace_id"]),
            "campaign_id": str(row["campaign_id"]),
            "kind": str(row["kind"]),
            "state_digest": str(row["state_digest"]),
            "payload_json": str(row["payload_json"]),
            "emitted_at": str(row["emitted_at"]),
            "expires_at": str(row["expires_at"]) if row["expires_at"] is not None else None,
        }
        return "sha256_" + sealer.sign_canonical_payload(
            envelope,
            domain="bashgym.campaign-recovery.receipt.v1",
        )

    def verify_receipt(self, receipt_id: str) -> bool:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_recovery_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            return bool(
                row
                and hmac.compare_digest(
                    self._receipt_seal(connection, row),
                    str(row["receipt_digest"]),
                )
            )

    @staticmethod
    def _request_authority(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "campaign_recovery_execution_authority.v1",
            "request_id": str(row["request_id"]),
            "workspace_id": str(row["workspace_id"]),
            "campaign_id": str(row["campaign_id"]),
            "idempotency_key": str(row["idempotency_key"]),
            "action": str(row["action"]),
            "accepted_receipt_id": str(row["accepted_receipt_id"]),
            "status": str(row["status"]),
            "claim_owner_id": (
                str(row["claim_owner_id"]) if row["claim_owner_id"] is not None else None
            ),
            "claim_generation": int(row["claim_generation"]),
            "claim_expires_at": (
                str(row["claim_expires_at"]) if row["claim_expires_at"] is not None else None
            ),
            "target_attempt_id": (
                str(row["target_attempt_id"]) if row["target_attempt_id"] is not None else None
            ),
            "outcome_code": (str(row["outcome_code"]) if row["outcome_code"] is not None else None),
            "outcome_receipt_id": (
                str(row["outcome_receipt_id"]) if row["outcome_receipt_id"] is not None else None
            ),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def _request_seal(self, row: sqlite3.Row | dict[str, Any]) -> str:
        sealer = self._require_sealer()
        key_version = str(row["authority_seal_key_version"])
        if not hmac.compare_digest(key_version, sealer.key_version):
            raise CampaignRecoveryConflictError("recovery execution seal key version is invalid")
        return "sha256_" + sealer.sign_canonical_payload(
            self._request_authority(row),
            domain="bashgym.campaign-recovery.execution-authority.v1",
        )

    def _verify_request_row(self, row: sqlite3.Row | dict[str, Any]) -> None:
        if not hmac.compare_digest(self._request_seal(row), str(row["authority_digest"])):
            raise CampaignRecoveryConflictError("recovery execution authority seal is invalid")

    def _reseal_request(self, connection: sqlite3.Connection, request_id: str) -> None:
        row = connection.execute(
            "SELECT * FROM campaign_recovery_requests WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise CampaignRecoveryConflictError("recovery execution request is missing")
        key_version = self._require_sealer().key_version
        authority = dict(row)
        authority["authority_seal_key_version"] = key_version
        digest = self._request_seal(authority)
        connection.execute(
            """UPDATE campaign_recovery_requests
               SET authority_digest=?, authority_seal_key_version=?
               WHERE request_id=?""",
            (digest, key_version, request_id),
        )

    def register_installation(
        self,
        *,
        installation_id: str,
        controller_owner_id: str,
        controller_lease_key: str,
    ) -> None:
        """Register private controller authority for one local installation."""

        self._require_initialized()
        if not re.fullmatch(r"ins_[0-9a-f]{32}", installation_id):
            raise CampaignRecoveryError("invalid installation identifier")
        if not controller_owner_id or not controller_lease_key:
            raise CampaignRecoveryError("private controller registration is incomplete")
        with self._connection(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO campaign_recovery_installations(
                    installation_id, controller_owner_id, controller_lease_key
                ) VALUES (?, ?, ?)
                ON CONFLICT(installation_id) DO UPDATE SET
                    controller_owner_id=excluded.controller_owner_id,
                    controller_lease_key=excluded.controller_lease_key
                """,
                (installation_id, controller_owner_id, controller_lease_key),
            )

    def register_binding(
        self,
        *,
        installation_id: str,
        kind: Literal["model", "data", "evaluator", "compute"],
        logical_id: str,
        availability: Literal["reachable", "inaccessible", "unknown"],
        display_label: str | None = None,
        integration_label: Literal["NeMo"] | None = None,
    ) -> None:
        """Register a logical binding; no host, key, path, or credential is accepted."""

        self._require_initialized()
        if kind not in _BINDING_PREFIX or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", logical_id
        ):
            raise CampaignRecoveryError("invalid logical binding")
        if availability not in {"reachable", "inaccessible", "unknown"}:
            raise CampaignRecoveryError("invalid binding availability")
        if display_label is not None and (
            not _PUBLIC_LABEL.fullmatch(display_label) or _LABEL_DENY.search(display_label)
        ):
            raise CampaignRecoveryError("model display label is not public-safe")
        if kind != "model" and display_label is not None:
            raise CampaignRecoveryError("only model bindings have a display label")
        if kind != "compute" and integration_label is not None:
            raise CampaignRecoveryError("integration labels belong to compute bindings")
        with self._connection(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO campaign_recovery_bindings(
                    installation_id, binding_kind, logical_id, availability,
                    display_label, integration_label
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(installation_id, binding_kind, logical_id) DO UPDATE SET
                    availability=excluded.availability,
                    display_label=excluded.display_label,
                    integration_label=excluded.integration_label
                """,
                (
                    installation_id,
                    kind,
                    logical_id,
                    availability,
                    display_label,
                    integration_label,
                ),
            )

    def bind_campaign(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        installation_id: str,
        lineage_mode: Literal["clone", "fork"],
        parent_campaign_id: str | None,
        checkpoint_source_id: str,
        artifact_source_id: str,
        schema_compatible: bool,
        revision_compatible: bool,
    ) -> None:
        """Resolve portable lineage against the current installation.

        Source artifact identities may be installation-private. They are never
        copied into the public projection or mutation receipts.
        """

        self._require_initialized()
        if not _DURABLE_ID.fullmatch(workspace_id) or not _DURABLE_ID.fullmatch(campaign_id):
            raise CampaignRecoveryError("invalid campaign recovery scope")
        if lineage_mode not in {"clone", "fork"}:
            raise CampaignRecoveryError("invalid recovery lineage mode")
        if parent_campaign_id is not None and not _DURABLE_ID.fullmatch(parent_campaign_id):
            raise CampaignRecoveryError("parent campaign identifier is not public-safe")
        with self._connection(immediate=True) as connection:
            campaign = connection.execute(
                "SELECT manifest_revision FROM campaigns WHERE workspace_id=? AND campaign_id=?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise CampaignRecoveryNotFoundError("campaign not found")
            installation = connection.execute(
                "SELECT 1 FROM campaign_recovery_installations WHERE installation_id=?",
                (installation_id,),
            ).fetchone()
            if installation is None:
                raise CampaignRecoveryNotFoundError("installation not registered")
            schema_id = _opaque(
                "sch", workspace_id, campaign_id, checkpoint_source_id, artifact_source_id
            )
            connection.execute(
                """
                INSERT INTO campaign_recovery_targets(
                    workspace_id, campaign_id, installation_id, lineage_mode,
                    parent_campaign_id, checkpoint_source_id, artifact_source_id,
                    campaign_revision, schema_id, schema_compatible, revision_compatible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, campaign_id) DO UPDATE SET
                    installation_id=excluded.installation_id,
                    lineage_mode=excluded.lineage_mode,
                    parent_campaign_id=excluded.parent_campaign_id,
                    checkpoint_source_id=excluded.checkpoint_source_id,
                    artifact_source_id=excluded.artifact_source_id,
                    campaign_revision=excluded.campaign_revision,
                    schema_id=excluded.schema_id,
                    schema_compatible=excluded.schema_compatible,
                    revision_compatible=excluded.revision_compatible
                """,
                (
                    workspace_id,
                    campaign_id,
                    installation_id,
                    lineage_mode,
                    parent_campaign_id,
                    checkpoint_source_id,
                    artifact_source_id,
                    int(campaign["manifest_revision"]),
                    schema_id,
                    int(schema_compatible),
                    int(revision_compatible),
                ),
            )

    @staticmethod
    def _latest_receipt_time(
        connection: sqlite3.Connection, workspace_id: str, campaign_id: str
    ) -> datetime | None:
        row = connection.execute(
            """
            SELECT emitted_at FROM campaign_recovery_receipts
            WHERE workspace_id=? AND campaign_id=?
            ORDER BY emitted_at DESC, receipt_id DESC LIMIT 1
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        return _parse_time(str(row["emitted_at"])) if row else None

    @classmethod
    def _receipt_time(
        cls,
        connection: sqlite3.Connection,
        workspace_id: str,
        campaign_id: str,
        desired: datetime,
    ) -> datetime:
        desired = desired.astimezone(UTC).replace(microsecond=(desired.microsecond // 1000) * 1000)
        latest = cls._latest_receipt_time(connection, workspace_id, campaign_id)
        if latest is not None and desired <= latest:
            desired = latest + timedelta(milliseconds=1)
        return desired

    def _store_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        receipt_id: str,
        workspace_id: str,
        campaign_id: str,
        kind: str,
        state_digest: str,
        payload: dict[str, Any],
        emitted_at: datetime,
        expires_at: datetime | None = None,
    ) -> dict[str, Any]:
        payload_json = _canonical(payload)
        emitted_at_wire = _wire_time(emitted_at)
        expires_at_wire = _wire_time(expires_at) if expires_at else None
        receipt_row = {
            "receipt_id": receipt_id,
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "kind": kind,
            "state_digest": state_digest,
            "payload_json": payload_json,
            "seal_key_version": self._require_sealer().key_version,
            "emitted_at": emitted_at_wire,
            "expires_at": expires_at_wire,
        }
        digest = self._receipt_seal(connection, receipt_row)
        connection.execute(
            """
            INSERT INTO campaign_recovery_receipts(
                receipt_id, workspace_id, campaign_id, kind, state_digest,
                payload_json, receipt_digest, seal_key_version, emitted_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                workspace_id,
                campaign_id,
                kind,
                state_digest,
                payload_json,
                digest,
                receipt_row["seal_key_version"],
                emitted_at_wire,
                expires_at_wire,
            ),
        )
        return {
            "receipt_id": receipt_id,
            "kind": kind,
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "campaign_revision": payload["campaign_revision"],
            "event_cursor": payload["event_cursor"],
            "aggregate_version": payload["aggregate_version"],
            "emitted_at": _wire_time(emitted_at),
            "digest": digest,
        }

    @staticmethod
    def _binding_id(installation_id: str, kind: str, logical_id: str) -> str:
        return _opaque(_BINDING_PREFIX[kind], installation_id, kind, logical_id)

    def _compose(
        self,
        connection: sqlite3.Connection,
        workspace_id: str,
        campaign_id: str,
        *,
        now: datetime,
        eligibility_ttl: timedelta,
        issue_receipts: bool,
    ) -> dict[str, Any]:
        campaign = connection.execute(
            "SELECT * FROM campaigns WHERE workspace_id=? AND campaign_id=?",
            (workspace_id, campaign_id),
        ).fetchone()
        if campaign is None:
            raise CampaignRecoveryNotFoundError("campaign not found")
        target = connection.execute(
            """
            SELECT * FROM campaign_recovery_targets
            WHERE workspace_id=? AND campaign_id=?
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        if target is None:
            raise CampaignRecoveryConflictError("campaign recovery is not registered")
        installation = connection.execute(
            "SELECT * FROM campaign_recovery_installations WHERE installation_id=?",
            (target["installation_id"],),
        ).fetchone()
        if installation is None:
            raise CampaignRecoveryConflictError("installation recovery authority is unavailable")
        manifest_row = connection.execute(
            """
            SELECT manifest_json FROM campaign_manifest_revisions
            WHERE workspace_id=? AND campaign_id=? AND revision=?
            """,
            (workspace_id, campaign_id, campaign["manifest_revision"]),
        ).fetchone()
        if manifest_row is None:
            raise CampaignRecoveryConflictError("campaign manifest revision is unavailable")
        manifest = json.loads(str(manifest_row["manifest_json"]))
        model = json.loads(str(campaign["target_model_json"]))
        evaluation = dict(manifest.get("evaluation_plan") or {})
        approved_data = list(manifest.get("approved_data_scopes") or [])
        required = {
            "model": model.get("target_contract_key"),
            "data": evaluation.get("dataset_binding_id")
            or (approved_data[0] if approved_data else None),
            "evaluator": evaluation.get("evaluation_suite_id"),
            "compute": manifest.get("compute_profile_id"),
        }
        binding_rows: dict[str, sqlite3.Row | None] = {}
        bindings: dict[str, str | None] = {}
        for kind, logical_id in required.items():
            row = None
            if isinstance(logical_id, str):
                row = connection.execute(
                    """
                    SELECT * FROM campaign_recovery_bindings
                    WHERE installation_id=? AND binding_kind=? AND logical_id=?
                    """,
                    (installation["installation_id"], kind, logical_id),
                ).fetchone()
            binding_rows[kind] = row
            bindings[f"{kind}_id" if kind != "evaluator" else "evaluator_id"] = (
                self._binding_id(str(installation["installation_id"]), kind, str(logical_id))
                if row is not None
                else None
            )
        bindings["model_display_label"] = (
            str(binding_rows["model"]["display_label"])
            if binding_rows["model"] is not None
            and binding_rows["model"]["display_label"] is not None
            else None
        )
        # Match the renderer contract's exact field order/names.
        bindings = {
            "model_id": bindings.pop("model_id"),
            "model_display_label": bindings.pop("model_display_label"),
            "data_id": bindings.pop("data_id"),
            "evaluator_id": bindings.pop("evaluator_id"),
            "compute_id": bindings.pop("compute_id"),
        }

        artifact_rows: dict[str, sqlite3.Row | None] = {}
        for role, source_field in (
            ("checkpoint", "checkpoint_source_id"),
            ("artifact", "artifact_source_id"),
        ):
            artifact_rows[role] = connection.execute(
                """
                SELECT artifact_id FROM campaign_artifacts
                WHERE workspace_id=? AND campaign_id=? AND artifact_id=?
                  AND sealed=1 AND valid=1
                """,
                (workspace_id, campaign_id, target[source_field]),
            ).fetchone()

        cursor_row = connection.execute(
            """
            SELECT COALESCE(MAX(cursor), 0) AS event_cursor FROM campaign_events
            WHERE workspace_id=? AND campaign_id=?
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        scope = {
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "campaign_revision": int(campaign["manifest_revision"]),
            "event_cursor": int(cursor_row["event_cursor"]),
            "aggregate_version": int(campaign["version"]),
        }
        lineage = {
            "checkpoint_id": _opaque(
                "ckpt", workspace_id, campaign_id, target["checkpoint_source_id"]
            ),
            "artifact_id": _opaque("art", workspace_id, campaign_id, target["artifact_source_id"]),
            "parent_campaign_id": target["parent_campaign_id"],
            **{
                key: scope[key]
                for key in ("campaign_revision", "event_cursor", "aggregate_version")
            },
            "schema_version": str(target["schema_id"]),
        }

        lease = connection.execute(
            "SELECT * FROM campaign_scheduler_leases WHERE lease_key=?",
            (installation["controller_lease_key"],),
        ).fetchone()
        if lease is None:
            controller = {
                "state": "unowned",
                "owner_status": "unassigned",
                "lease_id": None,
            }
            controller_observation = {"state": "unowned", "lease_id": None, "version": 0}
        else:
            lease_id = _opaque("lease", installation["controller_lease_key"], lease["generation"])
            expiry = _parse_time(str(lease["expires_at"]))
            if expiry <= now:
                state, owner_status = "expired", "lease_expired"
            elif lease["owner_id"] == installation["controller_owner_id"]:
                state, owner_status = "owned", "current_controller"
            else:
                state, owner_status = "foreign_lease", "foreign_controller"
            controller = {
                "state": state,
                "owner_status": owner_status,
                "lease_id": lease_id,
            }
            controller_observation = {
                "state": state,
                "lease_id": lease_id,
                "version": int(lease["controller_observation_version"]),
                "generation": int(lease["generation"]),
                "expires_at": str(lease["expires_at"]),
            }

        compute_row = binding_rows["compute"]
        compute_availability = (
            str(compute_row["availability"]) if compute_row is not None else "unknown"
        )
        schema_compatible = bool(target["schema_compatible"])
        revision_compatible = bool(target["revision_compatible"]) and int(
            target["campaign_revision"]
        ) == int(campaign["manifest_revision"])
        checks = [
            {
                "check": "binding_verified",
                "status": "pass" if all(binding_rows.values()) else "fail",
            },
            {
                "check": "checkpoint_available",
                "status": "pass" if all(artifact_rows.values()) else "fail",
            },
            {
                "check": "compute_reachable",
                "status": "pass" if compute_availability == "reachable" else "fail",
            },
            {
                "check": "schema_compatible",
                "status": "pass" if schema_compatible and revision_compatible else "fail",
            },
            {
                "check": "lease_observed",
                "status": "fail" if controller["state"] == "foreign_lease" else "pass",
            },
        ]
        state_payload = {
            "installation_id": installation["installation_id"],
            "required": required,
            "binding_rows": {
                kind: (
                    {
                        "logical_id": row["logical_id"],
                        "availability": row["availability"],
                        "integration_label": row["integration_label"],
                    }
                    if row is not None
                    else None
                )
                for kind, row in binding_rows.items()
            },
            "lineage": lineage,
            "artifact_available": {key: value is not None for key, value in artifact_rows.items()},
            "schema_compatible": schema_compatible,
            "revision_compatible": revision_compatible,
            "controller": controller_observation,
        }
        state_digest = hashlib.sha256(_canonical(state_payload).encode()).hexdigest()
        evidence_id = _opaque("evd", workspace_id, campaign_id, state_digest)

        doctor_row = connection.execute(
            """
            SELECT * FROM campaign_recovery_receipts
            WHERE workspace_id=? AND campaign_id=? AND kind='doctor' AND state_digest=?
            ORDER BY emitted_at DESC LIMIT 1
            """,
            (workspace_id, campaign_id, state_digest),
        ).fetchone()
        doctor_payload = {
            **scope,
            "evidence_id": evidence_id,
            "checks": checks,
        }
        if doctor_row is None and issue_receipts:
            doctor_time = self._receipt_time(
                connection,
                workspace_id,
                campaign_id,
                now - timedelta(milliseconds=1),
            )
            doctor_receipt_id = _opaque(
                "rcpt", workspace_id, campaign_id, "doctor", state_digest, _wire_time(doctor_time)
            )
            self._store_receipt(
                connection,
                receipt_id=doctor_receipt_id,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                kind="doctor",
                state_digest=state_digest,
                payload=doctor_payload,
                emitted_at=doctor_time,
            )
            doctor_row = connection.execute(
                "SELECT * FROM campaign_recovery_receipts WHERE receipt_id=?",
                (doctor_receipt_id,),
            ).fetchone()
        if doctor_row is None:
            raise CampaignRecoveryConflictError("doctor evidence is not sealed")
        doctor_digest = self._receipt_seal(connection, doctor_row)
        if not hmac.compare_digest(doctor_digest, str(doctor_row["receipt_digest"])):
            raise CampaignRecoveryConflictError("doctor evidence receipt seal is invalid")

        eligible = all(check["status"] == "pass" for check in checks)
        if controller["state"] == "expired":
            allowed = [RecoveryAction.TAKEOVER.value] if eligible else []
        elif controller["state"] in {"owned", "unowned"}:
            allowed = [RecoveryAction.RESUME.value, RecoveryAction.REPAIR.value] if eligible else []
        else:
            allowed = []
        latest = connection.execute(
            """
            SELECT * FROM campaign_recovery_receipts
            WHERE workspace_id=? AND campaign_id=? ORDER BY emitted_at DESC, receipt_id DESC LIMIT 1
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        eligibility_row = None
        if (
            latest is not None
            and latest["kind"] == "eligibility"
            and latest["state_digest"] == state_digest
        ):
            if latest["expires_at"] and _parse_time(str(latest["expires_at"])) > now:
                eligibility_row = latest
        if eligibility_row is None and issue_receipts:
            issued_at = self._receipt_time(connection, workspace_id, campaign_id, now)
            expires_at = issued_at + eligibility_ttl
            receipt_id = _opaque(
                "rcpt",
                workspace_id,
                campaign_id,
                "eligibility",
                state_digest,
                _wire_time(issued_at),
            )
            eligibility_payload = {
                **scope,
                "receipt_id": receipt_id,
                "decision": "eligible" if eligible else "blocked",
                "allowed_actions": allowed,
                "doctor_evidence_id": evidence_id,
                "doctor_evidence_digest": doctor_digest,
                "issued_at": _wire_time(issued_at),
                "expires_at": _wire_time(expires_at),
            }
            self._store_receipt(
                connection,
                receipt_id=receipt_id,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                kind="eligibility",
                state_digest=state_digest,
                payload=eligibility_payload,
                emitted_at=issued_at,
                expires_at=expires_at,
            )
            eligibility_row = connection.execute(
                "SELECT * FROM campaign_recovery_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
        if eligibility_row is None:
            raise CampaignRecoveryConflictError("current eligibility is not sealed")
        eligibility_digest = self._receipt_seal(connection, eligibility_row)
        if not hmac.compare_digest(eligibility_digest, str(eligibility_row["receipt_digest"])):
            raise CampaignRecoveryConflictError("eligibility receipt seal is invalid")
        eligibility_payload = json.loads(str(eligibility_row["payload_json"]))
        eligibility = {
            "receipt_id": eligibility_payload["receipt_id"],
            "receipt_digest": eligibility_digest,
            "decision": eligibility_payload["decision"],
            "allowed_actions": eligibility_payload["allowed_actions"],
            **scope,
            "doctor_evidence_id": eligibility_payload["doctor_evidence_id"],
            "doctor_evidence_digest": eligibility_payload["doctor_evidence_digest"],
            "issued_at": eligibility_payload["issued_at"],
            "expires_at": eligibility_payload["expires_at"],
        }

        receipt_rows = connection.execute(
            """
            SELECT * FROM campaign_recovery_receipts
            WHERE workspace_id=? AND campaign_id=?
            ORDER BY emitted_at DESC, receipt_id DESC LIMIT 32
            """,
            (workspace_id, campaign_id),
        ).fetchall()
        receipt_rows = list(reversed(receipt_rows))
        receipts = []
        for row in receipt_rows:
            payload_json = str(row["payload_json"])
            if not hmac.compare_digest(
                self._receipt_seal(connection, row), str(row["receipt_digest"])
            ):
                raise CampaignRecoveryConflictError("historical receipt seal is invalid")
            payload = json.loads(payload_json)
            if (
                not re.fullmatch(r"rcpt_[0-9a-f]{32}", str(row["receipt_id"]))
                or row["kind"] not in {"doctor", "eligibility", "recovery"}
                or payload.get("workspace_id") != workspace_id
                or payload.get("campaign_id") != campaign_id
                or not isinstance(payload.get("campaign_revision"), int)
                or payload["campaign_revision"] < 1
                or not isinstance(payload.get("event_cursor"), int)
                or payload["event_cursor"] < 0
                or not isinstance(payload.get("aggregate_version"), int)
                or payload["aggregate_version"] < 1
            ):
                raise CampaignRecoveryConflictError("historical receipt binding is invalid")
            receipts.append(
                {
                    "receipt_id": row["receipt_id"],
                    "kind": row["kind"],
                    **{
                        key: payload[key]
                        for key in (
                            "workspace_id",
                            "campaign_id",
                            "campaign_revision",
                            "event_cursor",
                            "aggregate_version",
                        )
                    },
                    "emitted_at": row["emitted_at"],
                    "digest": row["receipt_digest"],
                }
            )
        key_rows = connection.execute(
            """
            SELECT idempotency_key FROM campaign_recovery_mutations
            WHERE workspace_id=? AND campaign_id=?
            ORDER BY created_at DESC, idempotency_key DESC LIMIT 64
            """,
            (workspace_id, campaign_id),
        ).fetchall()
        consumed_keys = [str(row["idempotency_key"]) for row in reversed(key_rows)]
        latest_request = connection.execute(
            """
            SELECT * FROM campaign_recovery_requests
            WHERE workspace_id=? AND campaign_id=?
            ORDER BY updated_at DESC, request_id DESC LIMIT 1
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        latest_execution = (
            self._execution_projection(connection, latest_request)
            if latest_request is not None
            else None
        )
        consumer_reason = {
            "owned": "ready",
            "unowned": "controller_unowned",
            "expired": "controller_lease_expired",
            "foreign_lease": "foreign_controller",
        }[str(controller["state"])]
        return {
            "schema_version": "campaign_recovery.v1",
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "installation": {
                "installation_id": installation["installation_id"],
                "lineage_mode": target["lineage_mode"],
            },
            "bindings": bindings,
            "lineage": lineage,
            "controller": controller,
            "compute": {
                "availability": compute_availability,
                "integration_label": (
                    compute_row["integration_label"] if compute_row is not None else None
                ),
            },
            "artifacts": {
                key: "available" if value is not None else "missing"
                for key, value in artifact_rows.items()
            },
            "compatibility": {
                "schema": "compatible" if schema_compatible else "incompatible",
                "revision": "compatible" if revision_compatible else "incompatible",
            },
            "doctor": {
                "evidence_id": evidence_id,
                "evidence_digest": doctor_digest,
                "checks": checks,
            },
            "receipts": receipts,
            "eligibility": eligibility,
            "execution_consumer": {
                "supported": True,
                "ready": controller["state"] == "owned",
                "reason_code": consumer_reason,
            },
            "latest_execution": latest_execution,
            "consumed_idempotency_keys": consumed_keys,
            "_state_digest": state_digest,
            "_controller_observation": controller_observation,
        }

    @staticmethod
    def _public(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in snapshot.items() if not key.startswith("_")}

    def project(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        now: datetime | None = None,
        eligibility_ttl: timedelta = timedelta(minutes=10),
    ) -> dict[str, Any]:
        self._require_initialized()
        if eligibility_ttl <= timedelta(0) or eligibility_ttl > timedelta(hours=1):
            raise CampaignRecoveryError("eligibility TTL is outside the bounded window")
        observed_at = (now or datetime.now(UTC)).astimezone(UTC)
        with self._connection(immediate=True) as connection:
            snapshot = self._compose(
                connection,
                workspace_id,
                campaign_id,
                now=observed_at,
                eligibility_ttl=eligibility_ttl,
                issue_receipts=True,
            )
        return self._public(snapshot)

    def _replay_outcome(
        self, connection: sqlite3.Connection, request: RecoveryRequest
    ) -> dict[str, Any]:
        receipt_ids = {
            "approval": _opaque(
                "rcpt",
                request.workspace_id,
                request.campaign_id,
                request.action.value,
                "approval",
                request.idempotency_key,
            ),
            "outcome": _opaque(
                "rcpt",
                request.workspace_id,
                request.campaign_id,
                request.action.value,
                request.idempotency_key,
            ),
        }
        rows = {}
        payloads = {}
        for role, receipt_id in receipt_ids.items():
            row = connection.execute(
                """
                SELECT * FROM campaign_recovery_receipts
                WHERE receipt_id=? AND workspace_id=? AND campaign_id=? AND kind='recovery'
                """,
                (receipt_id, request.workspace_id, request.campaign_id),
            ).fetchone()
            if row is None or not hmac.compare_digest(
                self._receipt_seal(connection, row),
                str(row["receipt_digest"]),
            ):
                raise CampaignRecoveryConflictError("recovery replay receipt seal is invalid")
            rows[role] = row
            payloads[role] = json.loads(str(row["payload_json"]))
        approval = payloads["approval"]
        outcome = payloads["outcome"]
        if set(approval) != {
            "workspace_id",
            "campaign_id",
            "campaign_revision",
            "event_cursor",
            "aggregate_version",
            "action",
            "confirmation",
            "lineage",
            "actor_category",
            "request",
        } or set(outcome) != {
            "workspace_id",
            "campaign_id",
            "campaign_revision",
            "event_cursor",
            "aggregate_version",
            "action",
            "outcome",
            "lineage",
            "actor_category",
            "request",
        }:
            raise CampaignRecoveryConflictError("recovery replay receipt shape is invalid")
        if (
            approval["workspace_id"] != request.workspace_id
            or approval["campaign_id"] != request.campaign_id
            or approval["action"] != request.action.value
            or approval["confirmation"] != "human_confirmed"
            or approval["actor_category"] != "human"
            or outcome["workspace_id"] != request.workspace_id
            or outcome["campaign_id"] != request.campaign_id
            or outcome["action"] != request.action.value
            or outcome["outcome"] != "accepted"
            or outcome["actor_category"] != "human"
            or outcome["lineage"] != approval["lineage"]
            or approval["request"] != request.model_dump(mode="json")
            or outcome["request"] != request.model_dump(mode="json")
        ):
            raise CampaignRecoveryConflictError("recovery replay receipt binding is invalid")

        def public_receipt(role: str) -> dict[str, Any]:
            row = rows[role]
            payload = payloads[role]
            return {
                "receipt_id": row["receipt_id"],
                "kind": "recovery",
                "workspace_id": payload["workspace_id"],
                "campaign_id": payload["campaign_id"],
                "campaign_revision": payload["campaign_revision"],
                "event_cursor": payload["event_cursor"],
                "aggregate_version": payload["aggregate_version"],
                "emitted_at": row["emitted_at"],
                "digest": row["receipt_digest"],
            }

        return {
            "schema_version": "campaign_recovery_outcome.v1",
            "workspace_id": request.workspace_id,
            "campaign_id": request.campaign_id,
            "action": request.action.value,
            "outcome": "accepted",
            "lineage": outcome["lineage"],
            "approval_receipt": public_receipt("approval"),
            "receipt": public_receipt("outcome"),
        }

    def request(
        self,
        request: RecoveryRequest,
        *,
        actor_id: str,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Accept one exact human-confirmed request under current sealed authority."""

        self._require_initialized()
        observed_at = (now or datetime.now(UTC)).astimezone(UTC)
        request_hash = hashlib.sha256(
            _canonical(request.model_dump(mode="json")).encode()
        ).hexdigest()
        with self._connection(immediate=True) as connection:
            replay = connection.execute(
                """
                SELECT request_hash, response_json FROM campaign_recovery_mutations
                WHERE workspace_id=? AND actor_id=? AND idempotency_key=?
                """,
                (request.workspace_id, actor_id, request.idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["request_hash"] != request_hash:
                    raise CampaignRecoveryConflictError("recovery idempotency conflict")
                return self._replay_outcome(connection, request), True

            submitted_receipt = connection.execute(
                """
                SELECT * FROM campaign_recovery_receipts
                WHERE receipt_id=? AND workspace_id=? AND campaign_id=? AND kind='eligibility'
                """,
                (
                    request.eligibility_receipt_id,
                    request.workspace_id,
                    request.campaign_id,
                ),
            ).fetchone()
            if submitted_receipt is None or not hmac.compare_digest(
                self._receipt_seal(connection, submitted_receipt),
                str(submitted_receipt["receipt_digest"]),
            ):
                raise CampaignRecoveryConflictError("recovery eligibility is unverified")
            if (
                submitted_receipt["expires_at"] is None
                or _parse_time(str(submitted_receipt["expires_at"])) <= observed_at
            ):
                raise CampaignRecoveryConflictError("recovery eligibility expired")

            snapshot = self._compose(
                connection,
                request.workspace_id,
                request.campaign_id,
                now=observed_at,
                eligibility_ttl=timedelta(minutes=10),
                issue_receipts=True,
            )
            eligibility = snapshot["eligibility"]
            if _parse_time(eligibility["expires_at"]) <= observed_at:
                raise CampaignRecoveryConflictError("recovery eligibility expired")
            expected = {
                "expected_controller_lease_id": snapshot["controller"]["lease_id"],
                "eligibility_receipt_id": eligibility["receipt_id"],
                "doctor_evidence_id": snapshot["doctor"]["evidence_id"],
                "expected_campaign_revision": snapshot["lineage"]["campaign_revision"],
                "expected_event_cursor": snapshot["lineage"]["event_cursor"],
                "expected_aggregate_version": snapshot["lineage"]["aggregate_version"],
                "checkpoint_id": snapshot["lineage"]["checkpoint_id"],
                "artifact_id": snapshot["lineage"]["artifact_id"],
            }
            submitted = request.model_dump(mode="json")
            for field, value in expected.items():
                if submitted[field] != value:
                    noun = "controller" if field == "expected_controller_lease_id" else "authority"
                    raise CampaignRecoveryConflictError(f"recovery {noun} changed")
            if (
                eligibility["decision"] != "eligible"
                or request.action.value not in eligibility["allowed_actions"]
            ):
                raise CampaignRecoveryConflictError("recovery action is not eligible")
            if snapshot["controller"]["state"] == "foreign_lease":
                raise CampaignRecoveryConflictError("active foreign lease cannot be taken over")

            approval_time = self._receipt_time(
                connection,
                request.workspace_id,
                request.campaign_id,
                observed_at,
            )
            approval_id = _opaque(
                "rcpt",
                request.workspace_id,
                request.campaign_id,
                request.action.value,
                "approval",
                request.idempotency_key,
            )
            approval_payload = {
                "workspace_id": request.workspace_id,
                "campaign_id": request.campaign_id,
                "campaign_revision": snapshot["lineage"]["campaign_revision"],
                "event_cursor": snapshot["lineage"]["event_cursor"],
                "aggregate_version": snapshot["lineage"]["aggregate_version"],
                "action": request.action.value,
                "confirmation": "human_confirmed",
                "lineage": snapshot["lineage"],
                "actor_category": "human",
                "request": request.model_dump(mode="json"),
            }
            approval_receipt = self._store_receipt(
                connection,
                receipt_id=approval_id,
                workspace_id=request.workspace_id,
                campaign_id=request.campaign_id,
                kind="recovery",
                state_digest=snapshot["_state_digest"],
                payload=approval_payload,
                emitted_at=approval_time,
            )

            if request.action == RecoveryAction.TAKEOVER:
                if snapshot["controller"]["state"] != "expired":
                    raise CampaignRecoveryConflictError("only an expired lease can be taken over")
                target = connection.execute(
                    """
                    SELECT i.controller_owner_id, i.controller_lease_key
                    FROM campaign_recovery_targets t
                    JOIN campaign_recovery_installations i ON i.installation_id=t.installation_id
                    WHERE t.workspace_id=? AND t.campaign_id=?
                    """,
                    (request.workspace_id, request.campaign_id),
                ).fetchone()
                lease = connection.execute(
                    "SELECT generation, expires_at FROM campaign_scheduler_leases WHERE lease_key=?",
                    (target["controller_lease_key"],),
                ).fetchone()
                if lease is None or _parse_time(str(lease["expires_at"])) > observed_at:
                    raise CampaignRecoveryConflictError("controller lease is no longer expired")
                changed = connection.execute(
                    """
                    UPDATE campaign_scheduler_leases SET
                        owner_id=?, generation=generation+1,
                        controller_observation_version=controller_observation_version+1,
                        expires_at=?, heartbeat_at=?
                    WHERE lease_key=? AND generation=? AND expires_at<=?
                    """,
                    (
                        target["controller_owner_id"],
                        _wire_time(observed_at + timedelta(seconds=30)),
                        _wire_time(observed_at),
                        target["controller_lease_key"],
                        int(lease["generation"]),
                        _wire_time(observed_at),
                    ),
                )
                if changed.rowcount != 1:
                    raise CampaignRecoveryConflictError("controller lease changed")

            receipt_time = self._receipt_time(
                connection,
                request.workspace_id,
                request.campaign_id,
                observed_at,
            )
            receipt_id = _opaque(
                "rcpt",
                request.workspace_id,
                request.campaign_id,
                request.action.value,
                request.idempotency_key,
            )
            receipt_payload = {
                "workspace_id": request.workspace_id,
                "campaign_id": request.campaign_id,
                "campaign_revision": snapshot["lineage"]["campaign_revision"],
                "event_cursor": snapshot["lineage"]["event_cursor"],
                "aggregate_version": snapshot["lineage"]["aggregate_version"],
                "action": request.action.value,
                "outcome": "accepted",
                "lineage": snapshot["lineage"],
                "actor_category": "human",
                "request": request.model_dump(mode="json"),
            }
            receipt = self._store_receipt(
                connection,
                receipt_id=receipt_id,
                workspace_id=request.workspace_id,
                campaign_id=request.campaign_id,
                kind="recovery",
                state_digest=snapshot["_state_digest"],
                payload=receipt_payload,
                emitted_at=receipt_time,
            )
            outcome = {
                "schema_version": "campaign_recovery_outcome.v1",
                "workspace_id": request.workspace_id,
                "campaign_id": request.campaign_id,
                "action": request.action.value,
                "outcome": "accepted",
                "lineage": snapshot["lineage"],
                "approval_receipt": approval_receipt,
                "receipt": receipt,
            }
            if request.action in {RecoveryAction.RESUME, RecoveryAction.REPAIR}:
                connection.execute(
                    """
                    INSERT INTO campaign_recovery_requests(
                        request_id, workspace_id, campaign_id, idempotency_key,
                        action, accepted_receipt_id, status,
                        authority_digest, authority_seal_key_version,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'accepted', '', '', ?, ?)
                    """,
                    (
                        receipt_id,
                        request.workspace_id,
                        request.campaign_id,
                        request.idempotency_key,
                        request.action.value,
                        receipt_id,
                        _wire_time(receipt_time),
                        _wire_time(receipt_time),
                    ),
                )
                self._reseal_request(connection, receipt_id)
            connection.execute(
                """
                INSERT INTO campaign_recovery_mutations(
                    workspace_id, campaign_id, actor_id, idempotency_key,
                    request_hash, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.workspace_id,
                    request.campaign_id,
                    actor_id,
                    request.idempotency_key,
                    request_hash,
                    _canonical(outcome),
                    _wire_time(receipt_time),
                ),
            )
            return outcome, False

    @staticmethod
    def _claim_from_row(
        row: sqlite3.Row, *, status: Literal["executing", "blocked"] | None = None
    ) -> RecoveryWorkClaim:
        return RecoveryWorkClaim(
            request_id=str(row["request_id"]),
            workspace_id=str(row["workspace_id"]),
            campaign_id=str(row["campaign_id"]),
            action=RecoveryAction(str(row["action"])),
            status=status or str(row["status"]),  # type: ignore[arg-type]
            expected_campaign_revision=int(row["expected_campaign_revision"]),
            expected_event_cursor=int(row["expected_event_cursor"]),
            expected_aggregate_version=int(row["expected_aggregate_version"]),
            claim_generation=int(row["claim_generation"]),
            attempt_id=(
                str(row["target_attempt_id"]) if row["target_attempt_id"] is not None else None
            ),
            outcome_code=str(row["outcome_code"]) if row["outcome_code"] is not None else None,
        )

    def _accepted_authority(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        receipt = connection.execute(
            "SELECT * FROM campaign_recovery_receipts WHERE receipt_id=? AND kind='recovery'",
            (row["accepted_receipt_id"],),
        ).fetchone()
        if receipt is None or not hmac.compare_digest(
            self._receipt_seal(connection, receipt), str(receipt["receipt_digest"])
        ):
            raise CampaignRecoveryConflictError("accepted recovery receipt seal is invalid")
        payload = json.loads(str(receipt["payload_json"]))
        request = payload.get("request")
        if (
            not isinstance(request, dict)
            or payload.get("workspace_id") != row["workspace_id"]
            or payload.get("campaign_id") != row["campaign_id"]
            or payload.get("action") != row["action"]
            or payload.get("outcome") != "accepted"
            or request.get("idempotency_key") != row["idempotency_key"]
        ):
            raise CampaignRecoveryConflictError("accepted recovery receipt binding is invalid")
        return payload

    def _settle_connection(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        status: Literal["completed", "failed", "blocked"],
        outcome_code: str,
        now: datetime,
        attempt_id: str | None = None,
    ) -> None:
        self._verify_request_row(row)
        payload = self._accepted_authority(connection, row)
        receipt_id = _opaque("rcpt", row["request_id"], "execution", status, outcome_code)
        receipt_payload = {
            "workspace_id": row["workspace_id"],
            "campaign_id": row["campaign_id"],
            "campaign_revision": payload["campaign_revision"],
            "event_cursor": payload["event_cursor"],
            "aggregate_version": payload["aggregate_version"],
            "action": row["action"],
            "execution_status": status,
            "outcome_code": outcome_code,
            "attempt_id": attempt_id,
            "request_id": row["request_id"],
        }
        existing = connection.execute(
            "SELECT * FROM campaign_recovery_receipts WHERE receipt_id=?", (receipt_id,)
        ).fetchone()
        if existing is None:
            receipt_time = self._receipt_time(
                connection, str(row["workspace_id"]), str(row["campaign_id"]), now
            )
            self._store_receipt(
                connection,
                receipt_id=receipt_id,
                workspace_id=str(row["workspace_id"]),
                campaign_id=str(row["campaign_id"]),
                kind="recovery",
                state_digest=str(
                    connection.execute(
                        "SELECT state_digest FROM campaign_recovery_receipts WHERE receipt_id=?",
                        (row["accepted_receipt_id"],),
                    ).fetchone()["state_digest"]
                ),
                payload=receipt_payload,
                emitted_at=receipt_time,
            )
        changed = connection.execute(
            """
            UPDATE campaign_recovery_requests SET
                status=?, outcome_code=?, outcome_receipt_id=?, target_attempt_id=COALESCE(?, target_attempt_id),
                claim_expires_at=NULL, updated_at=?
            WHERE request_id=? AND status='executing' AND claim_generation=?
            """,
            (
                status,
                outcome_code,
                receipt_id,
                attempt_id,
                _wire_time(now),
                row["request_id"],
                int(row["claim_generation"]),
            ),
        )
        if changed.rowcount != 1:
            raise CampaignRecoveryConflictError("recovery execution claim changed")
        self._reseal_request(connection, str(row["request_id"]))

    def claim_next(
        self,
        *,
        leader: Any,
        worker_id: str,
        ttl: timedelta,
        now: datetime,
    ) -> RecoveryWorkClaim | None:
        """Claim one accepted request under the exact resident scheduler fence."""

        self._require_initialized()
        if ttl <= timedelta(0) or ttl > timedelta(minutes=5):
            raise CampaignRecoveryError("recovery execution TTL is outside the bounded window")
        observed_at = now.astimezone(UTC)
        if worker_id != leader.owner_id:
            return None
        with self._connection(immediate=True) as connection:
            live_lease = connection.execute(
                """
                SELECT * FROM campaign_scheduler_leases
                WHERE lease_key=? AND owner_id=? AND generation=? AND expires_at>?
                """,
                (
                    leader.lease_key,
                    leader.owner_id,
                    leader.generation,
                    _wire_time(observed_at),
                ),
            ).fetchone()
            if live_lease is None:
                return None
            row = connection.execute(
                """
                SELECT r.*, NULL AS expected_campaign_revision,
                       NULL AS expected_event_cursor, NULL AS expected_aggregate_version
                FROM campaign_recovery_requests r
                JOIN campaign_recovery_targets t
                  ON t.workspace_id=r.workspace_id AND t.campaign_id=r.campaign_id
                JOIN campaign_recovery_installations i ON i.installation_id=t.installation_id
                WHERE i.controller_lease_key=? AND i.controller_owner_id=?
                  AND (r.status='accepted' OR (r.status='executing' AND r.claim_expires_at<=?))
                ORDER BY r.created_at, r.request_id LIMIT 1
                """,
                (leader.lease_key, worker_id, _wire_time(observed_at)),
            ).fetchone()
            if row is None:
                return None
            self._verify_request_row(row)
            payload = self._accepted_authority(connection, row)
            request = payload["request"]
            campaign = connection.execute(
                "SELECT manifest_revision, version FROM campaigns WHERE workspace_id=? AND campaign_id=?",
                (row["workspace_id"], row["campaign_id"]),
            ).fetchone()
            cursor = connection.execute(
                "SELECT COALESCE(MAX(cursor), 0) AS value FROM campaign_events WHERE workspace_id=? AND campaign_id=?",
                (row["workspace_id"], row["campaign_id"]),
            ).fetchone()
            target = connection.execute(
                """
                SELECT * FROM campaign_recovery_targets
                WHERE workspace_id=? AND campaign_id=?
                """,
                (row["workspace_id"], row["campaign_id"]),
            ).fetchone()
            current_lineage = (
                {
                    "checkpoint_id": _opaque(
                        "ckpt",
                        row["workspace_id"],
                        row["campaign_id"],
                        target["checkpoint_source_id"],
                    ),
                    "artifact_id": _opaque(
                        "art",
                        row["workspace_id"],
                        row["campaign_id"],
                        target["artifact_source_id"],
                    ),
                    "parent_campaign_id": target["parent_campaign_id"],
                    "campaign_revision": int(campaign["manifest_revision"]),
                    "event_cursor": int(cursor["value"]),
                    "aggregate_version": int(campaign["version"]),
                    "schema_version": str(target["schema_id"]),
                }
                if campaign is not None and target is not None
                else None
            )
            expected_controller = request.get("expected_controller_lease_id")
            current_controller = _opaque("lease", leader.lease_key, leader.generation)
            first_claim = row["status"] == "accepted"
            unbound_repair_retry = (
                row["status"] == "executing"
                and row["action"] == RecoveryAction.REPAIR.value
                and row["target_attempt_id"] is None
            )
            authority_matches = bool(
                campaign
                and int(campaign["manifest_revision"]) == int(request["expected_campaign_revision"])
                and int(campaign["version"]) == int(request["expected_aggregate_version"])
                and int(cursor["value"]) == int(request["expected_event_cursor"])
                and current_lineage == payload.get("lineage")
                and expected_controller in {None, current_controller}
            )
            generation = int(row["claim_generation"]) + 1
            enriched = dict(row)
            enriched.update(
                {
                    "expected_campaign_revision": int(request["expected_campaign_revision"]),
                    "expected_event_cursor": int(request["expected_event_cursor"]),
                    "expected_aggregate_version": int(request["expected_aggregate_version"]),
                    "claim_generation": generation,
                }
            )
            if (first_claim or unbound_repair_retry) and not authority_matches:
                changed = connection.execute(
                    """
                    UPDATE campaign_recovery_requests SET
                        status='executing', claim_owner_id=?, claim_generation=?,
                        claim_expires_at=?, updated_at=?
                    WHERE request_id=? AND (
                        status='accepted' OR (status='executing' AND claim_expires_at<=?)
                    )
                    """,
                    (
                        worker_id,
                        generation,
                        _wire_time(observed_at + ttl),
                        _wire_time(observed_at),
                        row["request_id"],
                        _wire_time(observed_at),
                    ),
                )
                if changed.rowcount != 1:
                    return None
                self._reseal_request(connection, str(row["request_id"]))
                executing = connection.execute(
                    "SELECT * FROM campaign_recovery_requests WHERE request_id=?",
                    (row["request_id"],),
                ).fetchone()
                enriched.update(dict(executing))
                enriched.update(
                    {
                        "expected_campaign_revision": int(request["expected_campaign_revision"]),
                        "expected_event_cursor": int(request["expected_event_cursor"]),
                        "expected_aggregate_version": int(request["expected_aggregate_version"]),
                    }
                )
                self._settle_connection(
                    connection,
                    enriched,
                    status="blocked",
                    outcome_code="authority_changed",
                    now=observed_at,
                )
                return self._claim_from_row(enriched, status="blocked")
            changed = connection.execute(
                """
                UPDATE campaign_recovery_requests SET
                    status='executing', claim_owner_id=?, claim_generation=?,
                    claim_expires_at=?, updated_at=?
                WHERE request_id=? AND (
                    status='accepted' OR (status='executing' AND claim_expires_at<=?)
                )
                """,
                (
                    worker_id,
                    generation,
                    _wire_time(observed_at + ttl),
                    _wire_time(observed_at),
                    row["request_id"],
                    _wire_time(observed_at),
                ),
            )
            if changed.rowcount != 1:
                return None
            self._reseal_request(connection, str(row["request_id"]))
            enriched.update(
                {
                    "status": "executing",
                    "claim_owner_id": worker_id,
                    "claim_expires_at": _wire_time(observed_at + ttl),
                }
            )
            return self._claim_from_row(enriched)

    def set_repair_target(self, claim: RecoveryWorkClaim, attempt_id: str) -> RecoveryWorkClaim:
        """Bind a repair claim once to one existing durable attempt identifier."""

        self._require_initialized()
        with self._connection(immediate=True) as connection:
            request_row = connection.execute(
                "SELECT * FROM campaign_recovery_requests WHERE request_id=?",
                (claim.request_id,),
            ).fetchone()
            if request_row is None:
                raise CampaignRecoveryConflictError("repair execution request is missing")
            self._verify_request_row(request_row)
            attempt = connection.execute(
                """
                SELECT 1 FROM campaign_attempts t JOIN campaign_actions a
                  ON a.workspace_id=t.workspace_id AND a.action_id=t.action_id
                WHERE t.workspace_id=? AND a.campaign_id=? AND t.attempt_id=?
                """,
                (claim.workspace_id, claim.campaign_id, attempt_id),
            ).fetchone()
            if attempt is None:
                raise CampaignRecoveryConflictError("repair attempt authority changed")
            changed = connection.execute(
                """
                UPDATE campaign_recovery_requests SET target_attempt_id=?
                WHERE request_id=? AND status='executing' AND claim_generation=?
                  AND (target_attempt_id IS NULL OR target_attempt_id=?)
                """,
                (attempt_id, claim.request_id, claim.claim_generation, attempt_id),
            )
            if changed.rowcount != 1:
                raise CampaignRecoveryConflictError("repair execution claim changed")
            self._reseal_request(connection, claim.request_id)
        return RecoveryWorkClaim(**{**claim.__dict__, "attempt_id": attempt_id})

    def settle(
        self,
        claim: RecoveryWorkClaim,
        *,
        status: Literal["completed", "failed", "blocked"],
        outcome_code: str,
        now: datetime,
    ) -> None:
        """Seal a bounded terminal result under the current claim generation."""

        self._require_initialized()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", outcome_code):
            raise CampaignRecoveryError("invalid recovery outcome code")
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM campaign_recovery_requests WHERE request_id=?",
                (claim.request_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "executing"
                or int(row["claim_generation"]) != claim.claim_generation
            ):
                raise CampaignRecoveryConflictError("recovery execution claim changed")
            self._verify_request_row(row)
            enriched = dict(row)
            payload = self._accepted_authority(connection, row)
            enriched.update(
                {
                    "expected_campaign_revision": payload["campaign_revision"],
                    "expected_event_cursor": payload["event_cursor"],
                    "expected_aggregate_version": payload["aggregate_version"],
                }
            )
            self._settle_connection(
                connection,
                enriched,
                status=status,
                outcome_code=outcome_code,
                now=now.astimezone(UTC),
                attempt_id=claim.attempt_id,
            )

    def execution_status(
        self, workspace_id: str, campaign_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        """Return one bounded secret-free lifecycle projection."""

        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM campaign_recovery_requests
                WHERE workspace_id=? AND campaign_id=? AND idempotency_key=?
                """,
                (workspace_id, campaign_id, idempotency_key),
            ).fetchone()
            if row is None:
                raise CampaignRecoveryNotFoundError("recovery request not found")
            return self._execution_projection(connection, row)

    def _execution_projection(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        """Verify and expose one bounded lifecycle without worker or lease identity."""

        self._verify_request_row(row)
        self._accepted_authority(connection, row)
        if row["outcome_receipt_id"] is not None:
            receipt = connection.execute(
                "SELECT * FROM campaign_recovery_receipts WHERE receipt_id=?",
                (row["outcome_receipt_id"],),
            ).fetchone()
            if receipt is None or not hmac.compare_digest(
                self._receipt_seal(connection, receipt), str(receipt["receipt_digest"])
            ):
                raise CampaignRecoveryConflictError("recovery execution receipt seal is invalid")
            result = json.loads(str(receipt["payload_json"]))
            if (
                result.get("request_id") != row["request_id"]
                or result.get("execution_status") != row["status"]
                or result.get("outcome_code") != row["outcome_code"]
                or result.get("attempt_id") != row["target_attempt_id"]
            ):
                raise CampaignRecoveryConflictError("recovery execution receipt binding is invalid")
        return {
            "schema_version": "campaign_recovery_execution.v1",
            "workspace_id": str(row["workspace_id"]),
            "campaign_id": str(row["campaign_id"]),
            "action": str(row["action"]),
            "status": str(row["status"]),
            "outcome_code": (str(row["outcome_code"]) if row["outcome_code"] is not None else None),
            "attempt_id": (
                str(row["target_attempt_id"]) if row["target_attempt_id"] is not None else None
            ),
        }


__all__ = [
    "CampaignRecoveryConflictError",
    "CampaignRecoveryError",
    "CampaignRecoveryNotFoundError",
    "CampaignRecoveryRepository",
    "RecoveryWorkClaim",
    "RecoveryAction",
    "RecoveryRequest",
]
