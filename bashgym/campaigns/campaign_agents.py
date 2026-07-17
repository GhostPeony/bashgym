"""Durable broker for campaign-scoped Codex and Hermes attachments.

Bearer material exists only at the service-to-broker callback boundary. Public
projections contain fixed message codes and allowlisted scalar fields only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.campaign_agent_contracts import (
    CampaignAgentActionContext,
    CampaignAgentAttachRequest,
    CampaignAgentCapability,
    CampaignAgentFamily,
    CampaignAgentGrantConfirmation,
    CampaignAgentGrantRequest,
    CampaignAgentHeartbeatRequest,
    CampaignAgentPublicViewQuery,
    CampaignAgentRevokeRequest,
    CampaignAgentScope,
)
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    Capability,
    CredentialKind,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.persistence import CampaignRepository, RecordNotFoundError


class CampaignAgentError(RuntimeError):
    code = "campaign_agent_error"


class CampaignAgentConflictError(CampaignAgentError):
    code = "campaign_agent_conflict"


class CampaignAgentAuthorizationError(PermissionError):
    code = "campaign_agent_authorization_denied"


class CampaignAgentCredentialError(PermissionError):
    code = "campaign_agent_credential_invalid"


class CampaignAgentBrokerUnavailableError(CampaignAgentError):
    code = "campaign_agent_broker_unavailable"


class CampaignAgentIntegrityError(CampaignAgentError):
    code = "campaign_agent_integrity_failed"


class CampaignAgentOriginVerifier(Protocol):
    def __call__(
        self,
        scope: CampaignAgentScope,
        family: CampaignAgentFamily,
        origin: str,
        session_id: str,
        agent_principal_id: str,
    ) -> bool: ...


@dataclass(frozen=True)
class BrokeredCampaignAgentCredential:
    """One-time secret delivery owned by Electron/main or another trusted broker.

    The callback is an acceptance boundary, not a transactional credential store:
    it must either install the credential and return, or raise without retaining it.
    A token from a callback that raises is never activated by campaign authority.
    """

    attachment_id: str
    credential_id: str
    raw_token: str = field(repr=False)
    workspace_id: str
    campaign_id: str
    agent_family: CampaignAgentFamily
    agent_origin: str
    session_id: str
    agent_principal_id: str
    granted_capabilities: tuple[CampaignAgentCapability, ...]
    authorization_revision: int
    issued_at: datetime
    expires_at: datetime


CredentialBroker = Callable[[BrokeredCampaignAgentCredential], None]

_GRANT_RECEIPT_DOMAIN = "bashgym.campaign-agent-grant-receipt.v1"
_GRANT_AUTHORITY_DOMAIN = "bashgym.campaign-agent-grant-authority.v1"
_TRANSITION_RECEIPT_DOMAIN = "bashgym.campaign-agent-transition-receipt.v1"
_ATTACHMENT_PROJECTION_DOMAIN = "bashgym.campaign-agent-attachment-projection.v1"
_AUDIT_PROJECTION_DOMAIN = "bashgym.campaign-agent-audit-projection.v1"
_PENDING_AUTHORITY_DOMAIN = "bashgym.campaign-agent-pending-authority.v1"


@dataclass(frozen=True)
class PreparedCampaignAgentAttachment:
    operation_id: str
    attachment_id: str
    attachment_version: int
    credential_issued_at: datetime


@dataclass(frozen=True)
class AuthorizedCampaignAgentAction:
    """Bearer-free authority passed from authentication to one fixed action adapter."""

    principal: ActorPrincipal
    scope: CampaignAgentScope
    required_capability: CampaignAgentCapability
    attachment_id: str
    attachment_version: int
    agent_family: CampaignAgentFamily
    agent_origin: str
    session_id: str

    def require_scope(self, workspace_id: str, campaign_id: str) -> None:
        if workspace_id != self.scope.workspace_id or campaign_id != self.scope.campaign_id:
            raise CampaignAgentAuthorizationError("protected campaign-agent action scope changed")


_CAPABILITY_BINDINGS: dict[CampaignAgentCapability, frozenset[Capability]] = {
    CampaignAgentCapability.CAMPAIGN_OBSERVE: frozenset({Capability.CAMPAIGN_READ}),
    CampaignAgentCapability.TRAINING_LAUNCH: frozenset(
        {Capability.COMPUTE_SMOKE, Capability.COMPUTE_TRAIN_WITHIN_BUDGET}
    ),
    CampaignAgentCapability.TRAINING_PAUSE_SELF: frozenset({Capability.CAMPAIGN_PAUSE}),
    CampaignAgentCapability.ARTIFACT_READ: frozenset({Capability.CAMPAIGN_READ}),
    CampaignAgentCapability.ARTIFACT_PROPOSE: frozenset({Capability.EXPERIMENT_LEDGER_WRITE}),
}

PROHIBITED_CAMPAIGN_AGENT_CAPABILITIES = frozenset(
    {
        Capability.DATA_APPROVE_EXTERNAL,
        Capability.COMPUTE_AMEND_BUDGET,
        Capability.COMPUTE_MANAGE_RESIDENT_SERVICES,
        Capability.COMPUTE_FORCE_STOP,
        Capability.EVAL_PROTECTED_ACQUIRE,
        Capability.EVAL_PROTECTED_EXECUTE,
        Capability.PROMOTION_DECIDE,
        Capability.PROMOTION_OVERRIDE,
        Capability.ARTIFACT_PUBLISH_HF,
        Capability.HANDOFF_EXTERNAL_PREPARE,
        Capability.HANDOFF_MEMEXAI_PREPARE,
    }
)


def mapped_campaign_capabilities(
    values: tuple[CampaignAgentCapability, ...],
) -> frozenset[Capability]:
    mapped = frozenset(capability for value in values for capability in _CAPABILITY_BINDINGS[value])
    if mapped & PROHIBITED_CAMPAIGN_AGENT_CAPABILITIES:
        raise CampaignAgentAuthorizationError("prohibited campaign-agent authority")
    return mapped


def capability_binding_digest(
    agent_principal_id: str,
    requested: tuple[CampaignAgentCapability, ...],
    granted: tuple[CampaignAgentCapability, ...],
) -> str:
    """Match the renderer's ASCII FNV-1a binding digest exactly."""

    payload = (
        f"principal={agent_principal_id};"
        f"requested={','.join(sorted(value.value for value in requested))};"
        f"granted={','.join(sorted(value.value for value in granted))}"
    )
    digest = 0x811C9DC5
    for value in payload.encode("ascii"):
        digest ^= value
        digest = (digest * 0x01000193) & 0xFFFFFFFF
    return f"fnv1a32:{digest:08x}"


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _grant_payload(
    request: CampaignAgentGrantRequest,
    *,
    human_principal_id: str,
    receipt_id: str,
    grant_revision: int,
    issued_at: datetime,
    expires_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": "campaign_agent_grant_confirmation.v1",
        "issuer": "campaign_authority",
        "receipt_id": receipt_id,
        "human_principal_id": human_principal_id,
        "scope": request.scope.model_dump(mode="json"),
        "agent_family": request.agent_family.value,
        "agent_origin": request.agent_origin,
        "agent_principal_id": request.agent_principal_id,
        "session_id": request.session_id,
        "requested_capabilities": [value.value for value in request.requested_capabilities],
        "granted_capabilities": [value.value for value in request.granted_capabilities],
        "capability_digest": capability_binding_digest(
            request.agent_principal_id,
            request.requested_capabilities,
            request.granted_capabilities,
        ),
        "grant_revision": grant_revision,
        "issued_at": _iso(issued_at),
        "expires_at": _iso(expires_at),
    }


def _grant_from_payload(payload: dict[str, Any], digest: str) -> CampaignAgentGrantConfirmation:
    return CampaignAgentGrantConfirmation.model_validate({**payload, "receipt_digest": digest})


def grant_confirmation_to_wire(
    receipt: CampaignAgentGrantConfirmation,
) -> dict[str, Any]:
    """Return the exact camelCase receipt shape consumed by the existing renderer."""

    return {
        "schemaVersion": receipt.schema_version,
        "issuer": receipt.issuer,
        "receiptId": receipt.receipt_id,
        "receiptDigest": receipt.receipt_digest,
        "humanPrincipal": {
            "principalId": receipt.human_principal_id,
            "principalType": "human",
        },
        "scope": {
            "workspaceId": receipt.scope.workspace_id,
            "campaignId": receipt.scope.campaign_id,
        },
        "agentFamily": receipt.agent_family.value,
        "agentOrigin": receipt.agent_origin,
        "agentPrincipalId": receipt.agent_principal_id,
        "sessionId": receipt.session_id,
        "requestedCapabilities": [value.value for value in receipt.requested_capabilities],
        "grantedCapabilities": [value.value for value in receipt.granted_capabilities],
        "capabilityDigest": receipt.capability_digest,
        "grantRevision": receipt.grant_revision,
        "issuedAt": _iso(receipt.issued_at),
        "expiresAt": _iso(receipt.expires_at),
    }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign_agent_grants (
    receipt_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    agent_principal_id TEXT NOT NULL,
    grant_revision INTEGER NOT NULL CHECK(grant_revision >= 1),
    request_hash TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed_attachment_version INTEGER,
    UNIQUE(workspace_id, campaign_id, agent_principal_id, grant_revision),
    UNIQUE(workspace_id, campaign_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS campaign_agent_attachments (
    workspace_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL UNIQUE,
    attachment_version INTEGER NOT NULL CHECK(attachment_version >= 1),
    status TEXT NOT NULL CHECK(status IN ('attached', 'revoked')),
    agent_family TEXT NOT NULL,
    agent_origin TEXT NOT NULL,
    session_id TEXT NOT NULL,
    agent_principal_id TEXT NOT NULL,
    requested_capabilities_json TEXT NOT NULL,
    granted_capabilities_json TEXT NOT NULL,
    attached_at TEXT NOT NULL,
    attached_by TEXT NOT NULL,
    grant_receipt_id TEXT NOT NULL,
    grant_receipt_digest TEXT NOT NULL,
    credential_id TEXT NOT NULL UNIQUE,
    credential_salt TEXT NOT NULL,
    credential_hash TEXT NOT NULL,
    credential_issued_at TEXT NOT NULL,
    credential_expires_at TEXT NOT NULL,
    credential_revocation_revision INTEGER NOT NULL CHECK(credential_revocation_revision >= 0),
    authorization_revision INTEGER NOT NULL CHECK(authorization_revision >= 1),
    last_seen_at TEXT,
    resume_cursor TEXT,
    revoked_at TEXT,
    revoked_by TEXT,
    PRIMARY KEY(workspace_id, campaign_id)
);

CREATE TABLE IF NOT EXISTS campaign_agent_receipts (
    receipt_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('attach', 'revoke')),
    actor_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    attachment_version INTEGER NOT NULL,
    receipt_digest TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    UNIQUE(workspace_id, campaign_id, idempotency_key),
    UNIQUE(workspace_id, campaign_id, attachment_version)
);

CREATE TABLE IF NOT EXISTS campaign_agent_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    workspace_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('attachment', 'revocation', 'credential', 'liveness')),
    occurred_at TEXT NOT NULL,
    message_code TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS campaign_agent_events_scope_idx
    ON campaign_agent_events(workspace_id, campaign_id, sequence);
"""
_SCHEMA_V2 = """
ALTER TABLE campaign_agent_attachments ADD COLUMN resume_sequence INTEGER;

CREATE TABLE campaign_agent_pending_attachments (
    operation_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    attachment_version INTEGER NOT NULL CHECK(attachment_version >= 1),
    request_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    human_actor_id TEXT NOT NULL,
    credential_id TEXT NOT NULL UNIQUE,
    credential_salt TEXT NOT NULL,
    credential_hash TEXT NOT NULL,
    credential_issued_at TEXT NOT NULL,
    credential_expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(workspace_id, campaign_id)
);
"""
_SCHEMA_V3 = """
ALTER TABLE campaign_agent_grants ADD COLUMN seal_key_version TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_grants ADD COLUMN authority_digest TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_grants ADD COLUMN authority_seal_key_version TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_receipts ADD COLUMN seal_key_version TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_attachments ADD COLUMN projection_digest TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_attachments ADD COLUMN projection_seal_key_version TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_pending_attachments ADD COLUMN authority_digest TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_pending_attachments ADD COLUMN authority_seal_key_version TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_events ADD COLUMN projection_digest TEXT NOT NULL DEFAULT '';
ALTER TABLE campaign_agent_events ADD COLUMN projection_seal_key_version TEXT NOT NULL DEFAULT '';
"""
_MIGRATIONS = (
    (1, "campaign_agent_broker_v1", _SCHEMA),
    (2, "campaign_agent_pending_activation_v2", _SCHEMA_V2),
    (3, "campaign_agent_authenticated_authority_v3", _SCHEMA_V3),
)


class CampaignAgentRepository:
    def __init__(self, db_path: str | Path, *, sealer: ArtifactSealer):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", sealer.key_version):
            raise CampaignAgentIntegrityError("campaign-agent seal key version is invalid")
        self.db_path = Path(db_path)
        self.sealer = sealer
        self.seal_key_version = sealer.key_version

    def _seal(self, payload: dict[str, Any], *, domain: str) -> str:
        return f"sha256:{self.sealer.sign_canonical_payload(payload, domain=domain)}"

    def _verify_seal(
        self,
        payload: dict[str, Any],
        *,
        digest: str,
        key_version: str,
        domain: str,
        label: str,
    ) -> None:
        if key_version != self.seal_key_version or not hmac.compare_digest(
            self._seal(payload, domain=domain), digest
        ):
            raise CampaignAgentIntegrityError(f"{label} seal is invalid")

    def _grant_authority_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema_version": "campaign_agent_grant_authority.v1",
            "seal_key_version": row["authority_seal_key_version"],
            "receipt_id": row["receipt_id"],
            "workspace_id": row["workspace_id"],
            "campaign_id": row["campaign_id"],
            "agent_principal_id": row["agent_principal_id"],
            "grant_revision": int(row["grant_revision"]),
            "request_hash": row["request_hash"],
            "receipt_digest": row["receipt_digest"],
            "payload": json.loads(row["payload_json"]),
            "idempotency_key": row["idempotency_key"],
            "issued_at": row["issued_at"],
            "expires_at": row["expires_at"],
            "consumed_attachment_version": row["consumed_attachment_version"],
        }

    def _refresh_grant_authority(self, connection: sqlite3.Connection, receipt_id: str) -> None:
        connection.execute(
            """UPDATE campaign_agent_grants
            SET authority_seal_key_version = ? WHERE receipt_id = ?""",
            (self.seal_key_version, receipt_id),
        )
        row = connection.execute(
            "SELECT * FROM campaign_agent_grants WHERE receipt_id = ?", (receipt_id,)
        ).fetchone()
        assert row is not None
        connection.execute(
            "UPDATE campaign_agent_grants SET authority_digest = ? WHERE receipt_id = ?",
            (
                self._seal(self._grant_authority_payload(row), domain=_GRANT_AUTHORITY_DOMAIN),
                receipt_id,
            ),
        )

    def _verify_grant(self, row: sqlite3.Row) -> CampaignAgentGrantConfirmation:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError) as exc:
            raise CampaignAgentIntegrityError("grant receipt payload is invalid") from exc
        self._verify_seal(
            {
                "schema_version": "campaign_agent_grant_receipt_seal.v1",
                "seal_key_version": row["seal_key_version"],
                "receipt": payload,
            },
            digest=row["receipt_digest"],
            key_version=row["seal_key_version"],
            domain=_GRANT_RECEIPT_DOMAIN,
            label="grant receipt",
        )
        self._verify_seal(
            self._grant_authority_payload(row),
            digest=row["authority_digest"],
            key_version=row["authority_seal_key_version"],
            domain=_GRANT_AUTHORITY_DOMAIN,
            label="grant authority",
        )
        try:
            receipt = _grant_from_payload(payload, row["receipt_digest"])
        except Exception as exc:
            raise CampaignAgentIntegrityError("grant receipt payload is invalid") from exc
        if (
            receipt.receipt_id != row["receipt_id"]
            or receipt.scope.workspace_id != row["workspace_id"]
            or receipt.scope.campaign_id != row["campaign_id"]
            or receipt.agent_principal_id != row["agent_principal_id"]
            or receipt.grant_revision != int(row["grant_revision"])
            or _iso(receipt.issued_at) != row["issued_at"]
            or _iso(receipt.expires_at) != row["expires_at"]
        ):
            raise CampaignAgentIntegrityError("grant receipt storage binding is invalid")
        return receipt

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS campaign_agent_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """)
            applied_rows = {
                int(row["version"]): row
                for row in connection.execute(
                    "SELECT version, name, checksum FROM campaign_agent_schema_migrations"
                ).fetchall()
            }
            if any(version > len(_MIGRATIONS) for version in applied_rows):
                raise CampaignAgentError("campaign-agent persistence contains an unknown migration")
            for version, name, script in _MIGRATIONS:
                checksum = hashlib.sha256(script.encode()).hexdigest()
                applied = applied_rows.get(version)
                if applied is not None:
                    if applied["name"] != name or applied["checksum"] != checksum:
                        raise CampaignAgentError(
                            "campaign-agent persistence migration checksum mismatch"
                        )
                    continue
                for statement in script.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute(
                    """INSERT INTO campaign_agent_schema_migrations(
                        version, name, checksum, applied_at
                    ) VALUES (?, ?, ?, ?)""",
                    (version, name, checksum, _iso(utc_now())),
                )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _latest_grant_revision(
        connection: sqlite3.Connection,
        workspace_id: str,
        campaign_id: str,
        agent_principal_id: str,
    ) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(grant_revision), 0) AS revision
            FROM campaign_agent_grants
            WHERE workspace_id = ? AND campaign_id = ? AND agent_principal_id = ?
            """,
            (workspace_id, campaign_id, agent_principal_id),
        ).fetchone()
        return int(row["revision"])

    @staticmethod
    def _monotonic_transition_time(
        connection: sqlite3.Connection,
        workspace_id: str,
        campaign_id: str,
        proposed: datetime,
    ) -> datetime:
        row = connection.execute(
            """
            SELECT occurred_at FROM campaign_agent_receipts
            WHERE workspace_id = ? AND campaign_id = ?
            ORDER BY attachment_version DESC LIMIT 1
            """,
            (workspace_id, campaign_id),
        ).fetchone()
        if row is None:
            return proposed
        latest = datetime.fromisoformat(row["occurred_at"].replace("Z", "+00:00"))
        return proposed if proposed > latest else latest + timedelta(seconds=1)

    def issue_grant(
        self,
        request: CampaignAgentGrantRequest,
        *,
        human_principal_id: str,
        now: datetime,
        ttl: timedelta,
    ) -> CampaignAgentGrantConfirmation:
        request_payload = request.model_dump(mode="json")
        request_hash = canonical_hash(
            {"request": request_payload, "human_principal_id": human_principal_id}
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                """
                SELECT *
                FROM campaign_agent_grants
                WHERE workspace_id = ? AND campaign_id = ? AND idempotency_key = ?
                """,
                (
                    request.scope.workspace_id,
                    request.scope.campaign_id,
                    request.idempotency_key,
                ),
            ).fetchone()
            if replay is not None:
                receipt = self._verify_grant(replay)
                if replay["request_hash"] != request_hash:
                    raise CampaignAgentConflictError("grant idempotency conflict")
                return receipt
            active = connection.execute(
                """
                SELECT * FROM campaign_agent_attachments
                WHERE workspace_id = ? AND campaign_id = ?
                  AND agent_principal_id = ? AND status = 'attached'
                """,
                (
                    request.scope.workspace_id,
                    request.scope.campaign_id,
                    request.agent_principal_id,
                ),
            ).fetchone()
            if active is not None:
                self._verify_attachment_projection(active)
                now = self._monotonic_transition_time(
                    connection,
                    request.scope.workspace_id,
                    request.scope.campaign_id,
                    now,
                )
            revision = (
                self._latest_grant_revision(
                    connection,
                    request.scope.workspace_id,
                    request.scope.campaign_id,
                    request.agent_principal_id,
                )
                + 1
            )
            receipt_id = f"cagr_{uuid4().hex}"
            expires_at = now + ttl
            payload = _grant_payload(
                request,
                human_principal_id=human_principal_id,
                receipt_id=receipt_id,
                grant_revision=revision,
                issued_at=now,
                expires_at=expires_at,
            )
            digest = self._seal(
                {
                    "schema_version": "campaign_agent_grant_receipt_seal.v1",
                    "seal_key_version": self.seal_key_version,
                    "receipt": payload,
                },
                domain=_GRANT_RECEIPT_DOMAIN,
            )
            connection.execute(
                """
                INSERT INTO campaign_agent_grants (
                    receipt_id, workspace_id, campaign_id, agent_principal_id,
                    grant_revision, request_hash, receipt_digest, payload_json,
                    idempotency_key, issued_at, expires_at, seal_key_version,
                    authority_digest, authority_seal_key_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '')
                """,
                (
                    receipt_id,
                    request.scope.workspace_id,
                    request.scope.campaign_id,
                    request.agent_principal_id,
                    revision,
                    request_hash,
                    digest,
                    _json(payload),
                    request.idempotency_key,
                    _iso(now),
                    _iso(expires_at),
                    self.seal_key_version,
                ),
            )
            self._refresh_grant_authority(connection, receipt_id)
            if active is not None:
                current_version = int(active["attachment_version"])
                next_version = current_version + 1
                connection.execute(
                    """
                    UPDATE campaign_agent_attachments SET
                        attachment_version = ?, status = 'revoked', revoked_at = ?,
                        revoked_by = ?, credential_revocation_revision = credential_revocation_revision + 1
                    WHERE workspace_id = ? AND campaign_id = ?
                    """,
                    (
                        next_version,
                        _iso(now),
                        human_principal_id,
                        request.scope.workspace_id,
                        request.scope.campaign_id,
                    ),
                )
                self._refresh_attachment_projection(
                    connection, request.scope.workspace_id, request.scope.campaign_id
                )
                revocation_key = (
                    "authrev-" + hashlib.sha256(request.idempotency_key.encode()).hexdigest()[:24]
                )
                revocation_id = f"car_{uuid4().hex}"
                transition_request_hash = canonical_hash(
                    {
                        "grant_request_hash": request_hash,
                        "invalidated_authorization_revision": revision - 1,
                    }
                )
                transition_digest = self._seal(
                    self._transition_seal_payload(
                        receipt_id=revocation_id,
                        workspace_id=request.scope.workspace_id,
                        campaign_id=request.scope.campaign_id,
                        attachment_id=active["attachment_id"],
                        kind="revoke",
                        actor_id=human_principal_id,
                        occurred_at=_iso(now),
                        idempotency_key=revocation_key,
                        attachment_version=next_version,
                        request_hash=transition_request_hash,
                    ),
                    domain=_TRANSITION_RECEIPT_DOMAIN,
                )
                connection.execute(
                    """
                    INSERT INTO campaign_agent_receipts (
                        receipt_id, workspace_id, campaign_id, attachment_id, kind,
                        actor_id, occurred_at, idempotency_key, attachment_version,
                        receipt_digest, request_hash, seal_key_version
                    ) VALUES (?, ?, ?, ?, 'revoke', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        revocation_id,
                        request.scope.workspace_id,
                        request.scope.campaign_id,
                        active["attachment_id"],
                        human_principal_id,
                        _iso(now),
                        revocation_key,
                        next_version,
                        transition_digest,
                        transition_request_hash,
                        self.seal_key_version,
                    ),
                )
                self._insert_event(
                    connection,
                    workspace_id=request.scope.workspace_id,
                    campaign_id=request.scope.campaign_id,
                    kind="revocation",
                    occurred_at=now,
                    message_code="agent_revoked",
                )
            return _grant_from_payload(payload, digest)

    def _transition_seal_payload(
        self,
        *,
        receipt_id: str,
        workspace_id: str,
        campaign_id: str,
        attachment_id: str,
        kind: str,
        actor_id: str,
        occurred_at: str,
        idempotency_key: str,
        attachment_version: int,
        request_hash: str,
        seal_key_version: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "campaign_agent_transition_receipt_seal.v1",
            "seal_key_version": seal_key_version or self.seal_key_version,
            "receipt_id": receipt_id,
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "attachment_id": attachment_id,
            "kind": kind,
            "actor_id": actor_id,
            "occurred_at": occurred_at,
            "idempotency_key": idempotency_key,
            "attachment_version": attachment_version,
            "request_hash": request_hash,
        }

    def _verify_transition(self, row: sqlite3.Row) -> None:
        self._verify_seal(
            self._transition_seal_payload(
                receipt_id=row["receipt_id"],
                workspace_id=row["workspace_id"],
                campaign_id=row["campaign_id"],
                attachment_id=row["attachment_id"],
                kind=row["kind"],
                actor_id=row["actor_id"],
                occurred_at=row["occurred_at"],
                idempotency_key=row["idempotency_key"],
                attachment_version=int(row["attachment_version"]),
                request_hash=row["request_hash"],
                seal_key_version=row["seal_key_version"],
            ),
            digest=row["receipt_digest"],
            key_version=row["seal_key_version"],
            domain=_TRANSITION_RECEIPT_DOMAIN,
            label="transition receipt",
        )

    @staticmethod
    def _attachment_projection_payload(row: sqlite3.Row) -> dict[str, Any]:
        fields = (
            "workspace_id",
            "campaign_id",
            "attachment_id",
            "attachment_version",
            "status",
            "agent_family",
            "agent_origin",
            "session_id",
            "agent_principal_id",
            "requested_capabilities_json",
            "granted_capabilities_json",
            "attached_at",
            "attached_by",
            "grant_receipt_id",
            "grant_receipt_digest",
            "credential_id",
            "credential_salt",
            "credential_hash",
            "credential_issued_at",
            "credential_expires_at",
            "credential_revocation_revision",
            "authorization_revision",
            "last_seen_at",
            "resume_cursor",
            "resume_sequence",
            "revoked_at",
            "revoked_by",
        )
        return {
            "schema_version": "campaign_agent_attachment_projection_seal.v1",
            "seal_key_version": row["projection_seal_key_version"],
            "attachment": {field: row[field] for field in fields},
        }

    def _refresh_attachment_projection(
        self, connection: sqlite3.Connection, workspace_id: str, campaign_id: str
    ) -> None:
        connection.execute(
            """UPDATE campaign_agent_attachments SET projection_seal_key_version = ?
            WHERE workspace_id = ? AND campaign_id = ?""",
            (self.seal_key_version, workspace_id, campaign_id),
        )
        row = connection.execute(
            """SELECT * FROM campaign_agent_attachments
            WHERE workspace_id = ? AND campaign_id = ?""",
            (workspace_id, campaign_id),
        ).fetchone()
        assert row is not None
        connection.execute(
            """UPDATE campaign_agent_attachments SET projection_digest = ?
            WHERE workspace_id = ? AND campaign_id = ?""",
            (
                self._seal(
                    self._attachment_projection_payload(row),
                    domain=_ATTACHMENT_PROJECTION_DOMAIN,
                ),
                workspace_id,
                campaign_id,
            ),
        )

    def _verify_attachment_projection(self, row: sqlite3.Row) -> None:
        self._verify_seal(
            self._attachment_projection_payload(row),
            digest=row["projection_digest"],
            key_version=row["projection_seal_key_version"],
            domain=_ATTACHMENT_PROJECTION_DOMAIN,
            label="attachment projection",
        )

    @staticmethod
    def _audit_projection_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema_version": "campaign_agent_audit_projection_seal.v1",
            "seal_key_version": row["projection_seal_key_version"],
            "event": {
                field: row[field]
                for field in (
                    "sequence",
                    "event_id",
                    "workspace_id",
                    "campaign_id",
                    "kind",
                    "occurred_at",
                    "message_code",
                )
            },
        }

    def _verify_audit_projection(self, row: sqlite3.Row) -> None:
        self._verify_seal(
            self._audit_projection_payload(row),
            digest=row["projection_digest"],
            key_version=row["projection_seal_key_version"],
            domain=_AUDIT_PROJECTION_DOMAIN,
            label="audit projection",
        )

    @staticmethod
    def _audit_public_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            field: row[field]
            for field in ("event_id", "sequence", "kind", "occurred_at", "message_code")
        }

    @staticmethod
    def _pending_authority_payload(row: sqlite3.Row) -> dict[str, Any]:
        fields = (
            "operation_id",
            "workspace_id",
            "campaign_id",
            "attachment_id",
            "attachment_version",
            "request_hash",
            "request_json",
            "human_actor_id",
            "credential_id",
            "credential_salt",
            "credential_hash",
            "credential_issued_at",
            "credential_expires_at",
            "created_at",
        )
        return {
            "schema_version": "campaign_agent_pending_authority_seal.v1",
            "seal_key_version": row["authority_seal_key_version"],
            "pending": {field: row[field] for field in fields},
        }

    def _refresh_pending_authority(self, connection: sqlite3.Connection, operation_id: str) -> None:
        connection.execute(
            """UPDATE campaign_agent_pending_attachments
            SET authority_seal_key_version = ? WHERE operation_id = ?""",
            (self.seal_key_version, operation_id),
        )
        row = connection.execute(
            "SELECT * FROM campaign_agent_pending_attachments WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        assert row is not None
        connection.execute(
            """UPDATE campaign_agent_pending_attachments
            SET authority_digest = ? WHERE operation_id = ?""",
            (
                self._seal(self._pending_authority_payload(row), domain=_PENDING_AUTHORITY_DOMAIN),
                operation_id,
            ),
        )

    def _verify_pending_authority(self, row: sqlite3.Row) -> None:
        self._verify_seal(
            self._pending_authority_payload(row),
            digest=row["authority_digest"],
            key_version=row["authority_seal_key_version"],
            domain=_PENDING_AUTHORITY_DOMAIN,
            label="pending attachment authority",
        )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        campaign_id: str,
        kind: str,
        occurred_at: datetime,
        message_code: str,
    ) -> None:
        event_id = f"cae_{uuid4().hex}"
        connection.execute(
            """
            INSERT INTO campaign_agent_events (
                event_id, workspace_id, campaign_id, kind, occurred_at, message_code
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                workspace_id,
                campaign_id,
                kind,
                _iso(occurred_at),
                message_code,
            ),
        )
        connection.execute(
            """UPDATE campaign_agent_events SET projection_seal_key_version = ?
            WHERE event_id = ?""",
            (self.seal_key_version, event_id),
        )
        row = connection.execute(
            "SELECT * FROM campaign_agent_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        assert row is not None
        connection.execute(
            "UPDATE campaign_agent_events SET projection_digest = ? WHERE event_id = ?",
            (
                self._seal(self._audit_projection_payload(row), domain=_AUDIT_PROJECTION_DOMAIN),
                event_id,
            ),
        )

    def prepare_attach(
        self,
        request: CampaignAgentAttachRequest,
        *,
        human_actor_id: str,
        credential_id: str,
        credential_salt: str,
        credential_hash: str,
        credential_expires_at: datetime,
        now: datetime,
    ) -> tuple[PreparedCampaignAgentAttachment | None, dict[str, Any] | None, bool]:
        """Persist a non-authoritative delivery outbox before invoking the broker."""

        request_hash = canonical_hash(
            {
                "request": request.model_dump(mode="json"),
                "human_actor_id": human_actor_id,
            }
        )
        workspace_id = request.scope.workspace_id
        campaign_id = request.scope.campaign_id
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                """
                SELECT request_hash FROM campaign_agent_receipts
                WHERE workspace_id = ? AND campaign_id = ? AND idempotency_key = ?
                """,
                (workspace_id, campaign_id, request.idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["request_hash"] != request_hash:
                    raise CampaignAgentConflictError("attach idempotency conflict")
                return None, self._public_view(connection, request.scope, now=now), True

            current = connection.execute(
                """SELECT * FROM campaign_agent_attachments
                WHERE workspace_id = ? AND campaign_id = ?""",
                (workspace_id, campaign_id),
            ).fetchone()
            if current is not None:
                self._verify_attachment_projection(current)
            current_version = int(current["attachment_version"]) if current else None
            if request.base_attachment_version != current_version:
                raise CampaignAgentConflictError("attachment version changed")
            if current is not None and current["status"] == "attached":
                raise CampaignAgentConflictError("a campaign agent is already attached")
            now = self._monotonic_transition_time(connection, workspace_id, campaign_id, now)

            grant = connection.execute(
                "SELECT * FROM campaign_agent_grants WHERE receipt_id = ?",
                (request.confirmation_receipt.receipt_id,),
            ).fetchone()
            if grant is None:
                raise CampaignAgentAuthorizationError("grant receipt is not server-owned")
            stored_receipt = self._verify_grant(grant)
            if stored_receipt != request.confirmation_receipt:
                raise CampaignAgentAuthorizationError("grant receipt binding mismatch")
            if grant["consumed_attachment_version"] is not None:
                raise CampaignAgentAuthorizationError("grant receipt has already been consumed")
            if request.confirmation_receipt.expires_at <= now:
                raise CampaignAgentAuthorizationError("grant receipt expired")
            expected_binding = (
                request.scope == stored_receipt.scope
                and request.agent_family == stored_receipt.agent_family
                and request.agent_origin == stored_receipt.agent_origin
                and request.agent_principal_id == stored_receipt.agent_principal_id
                and request.session_id == stored_receipt.session_id
                and request.requested_capabilities == stored_receipt.requested_capabilities
                and request.granted_capabilities == stored_receipt.granted_capabilities
                and stored_receipt.capability_digest
                == capability_binding_digest(
                    request.agent_principal_id,
                    request.requested_capabilities,
                    request.granted_capabilities,
                )
            )
            if not expected_binding:
                raise CampaignAgentAuthorizationError("grant receipt binding mismatch")
            latest_revision = self._latest_grant_revision(
                connection, workspace_id, campaign_id, request.agent_principal_id
            )
            if request.confirmation_receipt.grant_revision != latest_revision:
                raise CampaignAgentAuthorizationError("grant authorization revision changed")

            attachment_id = str(current["attachment_id"]) if current else f"caa_{uuid4().hex}"
            next_version = (current_version or 0) + 1
            pending = connection.execute(
                """SELECT * FROM campaign_agent_pending_attachments
                WHERE workspace_id = ? AND campaign_id = ?""",
                (workspace_id, campaign_id),
            ).fetchone()
            if (
                pending is not None
                and datetime.fromisoformat(pending["credential_expires_at"].replace("Z", "+00:00"))
                <= now
            ):
                connection.execute(
                    "DELETE FROM campaign_agent_pending_attachments WHERE operation_id = ?",
                    (pending["operation_id"],),
                )
                pending = None
            if pending is not None:
                self._verify_pending_authority(pending)
            if pending is not None and pending["request_hash"] != request_hash:
                raise CampaignAgentConflictError("another attachment delivery is pending")
            operation_id = str(pending["operation_id"]) if pending else f"caop_{uuid4().hex}"
            pending_values = (
                operation_id,
                workspace_id,
                campaign_id,
                attachment_id,
                next_version,
                request_hash,
                request.model_dump_json(),
                human_actor_id,
                credential_id,
                credential_salt,
                credential_hash,
                _iso(now),
                _iso(credential_expires_at),
                _iso(now),
            )
            if pending is None:
                connection.execute(
                    """INSERT INTO campaign_agent_pending_attachments(
                        operation_id, workspace_id, campaign_id, attachment_id,
                        attachment_version, request_hash, request_json, human_actor_id,
                        credential_id, credential_salt, credential_hash,
                        credential_issued_at, credential_expires_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    pending_values,
                )
            else:
                connection.execute(
                    """UPDATE campaign_agent_pending_attachments SET
                        attachment_id = ?, attachment_version = ?, request_json = ?,
                        human_actor_id = ?, credential_id = ?, credential_salt = ?,
                        credential_hash = ?, credential_issued_at = ?,
                        credential_expires_at = ?, created_at = ?
                    WHERE operation_id = ?""",
                    (
                        attachment_id,
                        next_version,
                        request.model_dump_json(),
                        human_actor_id,
                        credential_id,
                        credential_salt,
                        credential_hash,
                        _iso(now),
                        _iso(credential_expires_at),
                        _iso(now),
                        operation_id,
                    ),
                )
            self._refresh_pending_authority(connection, operation_id)
            return (
                PreparedCampaignAgentAttachment(
                    operation_id=operation_id,
                    attachment_id=attachment_id,
                    attachment_version=next_version,
                    credential_issued_at=now,
                ),
                None,
                False,
            )

    def reject_attach(self, operation_id: str, credential_id: str) -> None:
        """Durably discard exactly one delivery the trusted broker rejected.

        The broker callback has no compensating API. Deleting this prepared row
        guarantees that any token it may have observed remains unauthorized and
        that a corrected request is not blocked by stale delivery state.
        """

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM campaign_agent_pending_attachments
                WHERE operation_id = ? AND credential_id = ?""",
                (operation_id, credential_id),
            )

    def acknowledge_attach(
        self,
        operation_id: str,
        credential_id: str,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        """Atomically activate only the exact credential accepted by the broker."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending = connection.execute(
                "SELECT * FROM campaign_agent_pending_attachments WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if pending is None or pending["credential_id"] != credential_id:
                raise CampaignAgentConflictError("attachment activation acknowledgement changed")
            self._verify_pending_authority(pending)
            request = CampaignAgentAttachRequest.model_validate_json(pending["request_json"])
            workspace_id = request.scope.workspace_id
            campaign_id = request.scope.campaign_id
            expected_request_hash = canonical_hash(
                {
                    "request": request.model_dump(mode="json"),
                    "human_actor_id": pending["human_actor_id"],
                }
            )
            if pending["request_hash"] != expected_request_hash:
                raise CampaignAgentAuthorizationError("pending attachment binding integrity failed")
            current = connection.execute(
                """SELECT * FROM campaign_agent_attachments
                WHERE workspace_id = ? AND campaign_id = ?""",
                (workspace_id, campaign_id),
            ).fetchone()
            if current is not None:
                self._verify_attachment_projection(current)
            current_version = int(current["attachment_version"]) if current else None
            if request.base_attachment_version != current_version:
                raise CampaignAgentConflictError("attachment version changed before activation")
            if current is not None and current["status"] == "attached":
                raise CampaignAgentConflictError("a campaign agent is already attached")
            grant = connection.execute(
                "SELECT * FROM campaign_agent_grants WHERE receipt_id = ?",
                (request.confirmation_receipt.receipt_id,),
            ).fetchone()
            if grant is None or grant["consumed_attachment_version"] is not None:
                raise CampaignAgentAuthorizationError("grant receipt is no longer current")
            stored_receipt = self._verify_grant(grant)
            activation_time = datetime.fromisoformat(
                pending["credential_issued_at"].replace("Z", "+00:00")
            )
            if stored_receipt != request.confirmation_receipt or stored_receipt.expires_at <= now:
                raise CampaignAgentAuthorizationError("grant receipt expired or changed")
            if (
                datetime.fromisoformat(pending["credential_expires_at"].replace("Z", "+00:00"))
                <= now
            ):
                raise CampaignAgentAuthorizationError("pending credential expired")
            expected_binding = (
                request.scope == stored_receipt.scope
                and request.agent_family == stored_receipt.agent_family
                and request.agent_origin == stored_receipt.agent_origin
                and request.agent_principal_id == stored_receipt.agent_principal_id
                and request.session_id == stored_receipt.session_id
                and request.requested_capabilities == stored_receipt.requested_capabilities
                and request.granted_capabilities == stored_receipt.granted_capabilities
                and stored_receipt.capability_digest
                == capability_binding_digest(
                    request.agent_principal_id,
                    request.requested_capabilities,
                    request.granted_capabilities,
                )
            )
            if not expected_binding:
                raise CampaignAgentAuthorizationError("pending attachment grant binding changed")
            latest_revision = self._latest_grant_revision(
                connection, workspace_id, campaign_id, request.agent_principal_id
            )
            if stored_receipt.grant_revision != latest_revision:
                raise CampaignAgentAuthorizationError("grant authorization revision changed")
            next_version = int(pending["attachment_version"])
            attachment_id = str(pending["attachment_id"])
            revocation_revision = int(current["credential_revocation_revision"]) if current else 0
            values = (
                attachment_id,
                next_version,
                request.agent_family.value,
                request.agent_origin,
                request.session_id,
                request.agent_principal_id,
                _json([value.value for value in request.requested_capabilities]),
                _json([value.value for value in request.granted_capabilities]),
                _iso(activation_time),
                pending["human_actor_id"],
                stored_receipt.receipt_id,
                stored_receipt.receipt_digest,
                pending["credential_id"],
                pending["credential_salt"],
                pending["credential_hash"],
                _iso(activation_time),
                pending["credential_expires_at"],
                revocation_revision,
                stored_receipt.grant_revision,
                workspace_id,
                campaign_id,
            )
            if current is None:
                connection.execute(
                    """INSERT INTO campaign_agent_attachments(
                        attachment_id, attachment_version, status, agent_family,
                        agent_origin, session_id, agent_principal_id,
                        requested_capabilities_json, granted_capabilities_json,
                        attached_at, attached_by, grant_receipt_id, grant_receipt_digest,
                        credential_id, credential_salt, credential_hash,
                        credential_issued_at, credential_expires_at,
                        credential_revocation_revision, authorization_revision,
                        workspace_id, campaign_id
                    ) VALUES (?, ?, 'attached', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
            else:
                connection.execute(
                    """UPDATE campaign_agent_attachments SET
                        attachment_id = ?, attachment_version = ?, status = 'attached',
                        agent_family = ?, agent_origin = ?, session_id = ?,
                        agent_principal_id = ?, requested_capabilities_json = ?,
                        granted_capabilities_json = ?, attached_at = ?, attached_by = ?,
                        grant_receipt_id = ?, grant_receipt_digest = ?, credential_id = ?,
                        credential_salt = ?, credential_hash = ?, credential_issued_at = ?,
                        credential_expires_at = ?, credential_revocation_revision = ?,
                        authorization_revision = ?, last_seen_at = NULL, resume_cursor = NULL,
                        resume_sequence = NULL, revoked_at = NULL, revoked_by = NULL
                    WHERE workspace_id = ? AND campaign_id = ?""",
                    values,
                )
            self._refresh_attachment_projection(connection, workspace_id, campaign_id)
            connection.execute(
                """UPDATE campaign_agent_grants SET consumed_attachment_version = ?
                WHERE receipt_id = ?""",
                (next_version, stored_receipt.receipt_id),
            )
            self._refresh_grant_authority(connection, stored_receipt.receipt_id)
            receipt_id = f"car_{uuid4().hex}"
            transition_digest = self._seal(
                self._transition_seal_payload(
                    receipt_id=receipt_id,
                    workspace_id=workspace_id,
                    campaign_id=campaign_id,
                    attachment_id=attachment_id,
                    kind="attach",
                    actor_id=pending["human_actor_id"],
                    occurred_at=_iso(activation_time),
                    idempotency_key=request.idempotency_key,
                    attachment_version=next_version,
                    request_hash=pending["request_hash"],
                ),
                domain=_TRANSITION_RECEIPT_DOMAIN,
            )
            connection.execute(
                """
                INSERT INTO campaign_agent_receipts (
                    receipt_id, workspace_id, campaign_id, attachment_id, kind,
                    actor_id, occurred_at, idempotency_key, attachment_version,
                    receipt_digest, request_hash, seal_key_version
                ) VALUES (?, ?, ?, ?, 'attach', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    workspace_id,
                    campaign_id,
                    attachment_id,
                    pending["human_actor_id"],
                    _iso(activation_time),
                    request.idempotency_key,
                    next_version,
                    transition_digest,
                    pending["request_hash"],
                    self.seal_key_version,
                ),
            )
            self._insert_event(
                connection,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                kind="attachment",
                occurred_at=activation_time,
                message_code="agent_attached",
            )
            if current is not None:
                self._insert_event(
                    connection,
                    workspace_id=workspace_id,
                    campaign_id=campaign_id,
                    kind="credential",
                    occurred_at=activation_time,
                    message_code="credential_renewed",
                )
            connection.execute(
                "DELETE FROM campaign_agent_pending_attachments WHERE operation_id = ?",
                (operation_id,),
            )
            observed_at = max(now, activation_time)
            return self._public_view(connection, request.scope, now=observed_at)

    def revoke(
        self,
        request: CampaignAgentRevokeRequest,
        *,
        human_actor_id: str,
        now: datetime,
    ) -> tuple[dict[str, Any], bool]:
        request_hash = canonical_hash(
            {
                "request": request.model_dump(mode="json"),
                "human_actor_id": human_actor_id,
            }
        )
        workspace_id = request.scope.workspace_id
        campaign_id = request.scope.campaign_id
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                """SELECT request_hash FROM campaign_agent_receipts
                WHERE workspace_id = ? AND campaign_id = ? AND idempotency_key = ?""",
                (workspace_id, campaign_id, request.idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["request_hash"] != request_hash:
                    raise CampaignAgentConflictError("revoke idempotency conflict")
                return self._public_view(connection, request.scope, now=now), True
            current = connection.execute(
                """SELECT * FROM campaign_agent_attachments
                WHERE workspace_id = ? AND campaign_id = ?""",
                (workspace_id, campaign_id),
            ).fetchone()
            if current is None:
                raise CampaignAgentConflictError("no campaign agent is attached")
            self._verify_attachment_projection(current)
            if current["attachment_id"] != request.attachment_id:
                raise CampaignAgentAuthorizationError("attachment scope mismatch")
            if int(current["attachment_version"]) != request.attachment_version:
                raise CampaignAgentConflictError("attachment version changed")
            if current["status"] != "attached":
                raise CampaignAgentConflictError("campaign agent is already revoked")
            now = self._monotonic_transition_time(connection, workspace_id, campaign_id, now)
            next_version = request.attachment_version + 1
            connection.execute(
                """
                UPDATE campaign_agent_attachments SET
                    attachment_version = ?, status = 'revoked', revoked_at = ?,
                    revoked_by = ?, credential_revocation_revision = credential_revocation_revision + 1
                WHERE workspace_id = ? AND campaign_id = ?
                """,
                (next_version, _iso(now), human_actor_id, workspace_id, campaign_id),
            )
            self._refresh_attachment_projection(connection, workspace_id, campaign_id)
            receipt_id = f"car_{uuid4().hex}"
            transition_digest = self._seal(
                self._transition_seal_payload(
                    receipt_id=receipt_id,
                    workspace_id=workspace_id,
                    campaign_id=campaign_id,
                    attachment_id=request.attachment_id,
                    kind="revoke",
                    actor_id=human_actor_id,
                    occurred_at=_iso(now),
                    idempotency_key=request.idempotency_key,
                    attachment_version=next_version,
                    request_hash=request_hash,
                ),
                domain=_TRANSITION_RECEIPT_DOMAIN,
            )
            connection.execute(
                """
                INSERT INTO campaign_agent_receipts (
                    receipt_id, workspace_id, campaign_id, attachment_id, kind,
                    actor_id, occurred_at, idempotency_key, attachment_version,
                    receipt_digest, request_hash, seal_key_version
                ) VALUES (?, ?, ?, ?, 'revoke', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    workspace_id,
                    campaign_id,
                    request.attachment_id,
                    human_actor_id,
                    _iso(now),
                    request.idempotency_key,
                    next_version,
                    transition_digest,
                    request_hash,
                    self.seal_key_version,
                ),
            )
            self._insert_event(
                connection,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                kind="revocation",
                occurred_at=now,
                message_code="agent_revoked",
            )
            return self._public_view(connection, request.scope, now=now), False

    def _credential_attachment(
        self,
        connection: sqlite3.Connection,
        raw_token: str,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        try:
            prefix, credential_id, secret = raw_token.split(".", 2)
        except ValueError as exc:
            raise CampaignAgentCredentialError("credential invalid") from exc
        if prefix != "bgag" or len(secret) < 32:
            raise CampaignAgentCredentialError("credential invalid")
        row = connection.execute(
            "SELECT * FROM campaign_agent_attachments WHERE credential_id = ?",
            (credential_id,),
        ).fetchone()
        if row is None:
            raise CampaignAgentCredentialError("credential invalid")
        self._verify_attachment_projection(row)
        expected_hash = hashlib.sha256(f"{row['credential_salt']}:{secret}".encode()).hexdigest()
        grant = connection.execute(
            "SELECT * FROM campaign_agent_grants WHERE receipt_id = ?",
            (row["grant_receipt_id"],),
        ).fetchone()
        if grant is None:
            raise CampaignAgentIntegrityError("attachment grant receipt is missing")
        verified_grant = self._verify_grant(grant)
        latest_grant = self._latest_grant_revision(
            connection,
            row["workspace_id"],
            row["campaign_id"],
            row["agent_principal_id"],
        )
        if (
            not hmac.compare_digest(expected_hash, row["credential_hash"])
            or row["status"] != "attached"
            or datetime.fromisoformat(row["credential_expires_at"].replace("Z", "+00:00")) <= now
            or int(row["authorization_revision"]) != latest_grant
            or verified_grant.grant_revision != latest_grant
            or verified_grant.receipt_digest != row["grant_receipt_digest"]
            or grant["consumed_attachment_version"] != row["attachment_version"]
        ):
            raise CampaignAgentCredentialError("credential invalid")
        return row

    def _authenticated_attachment(
        self,
        connection: sqlite3.Connection,
        raw_token: str,
        context: CampaignAgentActionContext | CampaignAgentHeartbeatRequest,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        row = self._credential_attachment(connection, raw_token, now=now)
        exact_binding = (
            row["workspace_id"] == context.scope.workspace_id
            and row["campaign_id"] == context.scope.campaign_id
            and row["agent_family"] == context.agent_family.value
            and row["agent_origin"] == context.agent_origin
            and row["session_id"] == context.session_id
            and row["agent_principal_id"] == context.agent_principal_id
        )
        if not exact_binding:
            raise CampaignAgentCredentialError("credential invalid")
        return row

    @staticmethod
    def _action_authorization(
        row: sqlite3.Row,
        required_capability: CampaignAgentCapability,
    ) -> AuthorizedCampaignAgentAction:
        granted = tuple(
            CampaignAgentCapability(value) for value in json.loads(row["granted_capabilities_json"])
        )
        if required_capability not in granted:
            raise CampaignAgentAuthorizationError(
                "required campaign-agent capability was not granted"
            )
        family = CampaignAgentFamily(row["agent_family"])
        profile = (
            AutonomyProfile.CODEX_TRUSTED
            if family is CampaignAgentFamily.CODEX
            else AutonomyProfile.HERMES_BOUNDED
        )
        return AuthorizedCampaignAgentAction(
            principal=ActorPrincipal(
                actor_id=row["agent_principal_id"],
                autonomy_profile=profile,
                credential_id=row["credential_id"],
                credential_kind=CredentialKind.ACCESS,
                workspace_ids=(row["workspace_id"],),
                capabilities=mapped_campaign_capabilities((required_capability,)),
                expires_at=datetime.fromisoformat(
                    row["credential_expires_at"].replace("Z", "+00:00")
                ),
            ),
            scope=CampaignAgentScope(
                workspace_id=row["workspace_id"], campaign_id=row["campaign_id"]
            ),
            required_capability=required_capability,
            attachment_id=row["attachment_id"],
            attachment_version=int(row["attachment_version"]),
            agent_family=family,
            agent_origin=row["agent_origin"],
            session_id=row["session_id"],
        )

    def authorize_action(
        self,
        raw_token: str,
        context: CampaignAgentActionContext,
        *,
        required_capability: CampaignAgentCapability,
        now: datetime,
    ) -> AuthorizedCampaignAgentAction:
        """Authenticate one exact protected action without projecting bearer material."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._authenticated_attachment(connection, raw_token, context, now=now)
            return self._action_authorization(row, required_capability)

    def authorize_bearer_action(
        self,
        raw_token: str,
        *,
        required_capability: CampaignAgentCapability,
        now: datetime,
    ) -> AuthorizedCampaignAgentAction:
        """Authorize a fixed action from credential-bound persisted provenance only."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._credential_attachment(connection, raw_token, now=now)
            if row["last_seen_at"] is None:
                raise CampaignAgentAuthorizationError("campaign-agent heartbeat is not current")
            last_seen_at = datetime.fromisoformat(row["last_seen_at"].replace("Z", "+00:00"))
            if now - last_seen_at > timedelta(minutes=2):
                raise CampaignAgentAuthorizationError("campaign-agent heartbeat is not current")
            return self._action_authorization(row, required_capability)

    def authenticate_and_heartbeat(
        self,
        raw_token: str,
        request: CampaignAgentHeartbeatRequest,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._authenticated_attachment(connection, raw_token, request, now=now)
            prior_seen = (
                datetime.fromisoformat(row["last_seen_at"].replace("Z", "+00:00"))
                if row["last_seen_at"] is not None
                else None
            )
            observed_at = max(now, prior_seen) if prior_seen is not None else now
            if (
                datetime.fromisoformat(row["credential_expires_at"].replace("Z", "+00:00"))
                <= observed_at
            ):
                raise CampaignAgentCredentialError("credential invalid")
            if row["last_seen_at"] is None:
                prior_liveness = "offline"
            else:
                assert prior_seen is not None
                prior_age = observed_at - prior_seen
                prior_liveness = (
                    "live"
                    if prior_age <= timedelta(minutes=2)
                    else "idle" if prior_age <= timedelta(minutes=10) else "offline"
                )
            cursor_advance = request.resume_cursor is not None
            if cursor_advance:
                current_cursor = row["resume_cursor"]
                current_sequence = row["resume_sequence"]
                assert request.resume_sequence is not None
                cursor_replay = (
                    current_sequence is not None
                    and request.resume_sequence == int(current_sequence)
                    and request.resume_cursor == current_cursor
                )
                if not cursor_replay and request.expected_resume_cursor != current_cursor:
                    raise CampaignAgentConflictError("resume cursor CAS conflict")
                if (
                    not cursor_replay
                    and current_sequence is not None
                    and (
                        request.resume_sequence < int(current_sequence)
                        or (
                            request.resume_sequence == int(current_sequence)
                            and request.resume_cursor != current_cursor
                        )
                    )
                ):
                    raise CampaignAgentConflictError("resume cursor cannot move backward")
            connection.execute(
                """UPDATE campaign_agent_attachments
                SET last_seen_at = ?,
                    resume_cursor = CASE WHEN ? THEN ? ELSE resume_cursor END,
                    resume_sequence = CASE WHEN ? THEN ? ELSE resume_sequence END
                WHERE workspace_id = ? AND campaign_id = ?""",
                (
                    _iso(observed_at),
                    cursor_advance,
                    request.resume_cursor,
                    cursor_advance,
                    request.resume_sequence,
                    request.scope.workspace_id,
                    request.scope.campaign_id,
                ),
            )
            self._refresh_attachment_projection(
                connection, request.scope.workspace_id, request.scope.campaign_id
            )
            if prior_liveness != "live":
                self._insert_event(
                    connection,
                    workspace_id=request.scope.workspace_id,
                    campaign_id=request.scope.campaign_id,
                    kind="liveness",
                    occurred_at=observed_at,
                    message_code="liveness_changed",
                )
            return self._public_view(connection, request.scope, now=observed_at)

    def public_view(self, query: CampaignAgentPublicViewQuery, *, now: datetime) -> dict[str, Any]:
        from bashgym.campaigns.campaign_agent_contracts import CampaignAgentScope

        with self._connect() as connection:
            return self._public_view(
                connection,
                CampaignAgentScope(workspace_id=query.workspace_id, campaign_id=query.campaign_id),
                now=now,
                after_sequence=query.after_sequence,
                limit=query.limit,
            )

    @staticmethod
    def _page_result(
        rows: list[Any], *, limit: int, cursor_field: str, after: int
    ) -> dict[str, Any]:
        has_more = len(rows) > limit
        visible = rows[:limit]
        return {
            "items": [dict(row) for row in visible],
            "next_cursor": int(visible[-1][cursor_field]) if visible else after,
            "has_more": has_more,
        }

    @staticmethod
    def _transition_public_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            field: row[field]
            for field in (
                "receipt_id",
                "kind",
                "actor_id",
                "occurred_at",
                "idempotency_key",
                "attachment_version",
                "receipt_digest",
            )
        }

    def audit_page(
        self,
        scope,
        *,
        after_sequence: int,
        limit: int,
    ) -> dict[str, Any]:
        bounded_limit = min(max(limit, 1), 50)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM campaign_agent_events
                WHERE workspace_id = ? AND campaign_id = ? AND sequence > ?
                ORDER BY sequence ASC LIMIT ?
                """,
                (
                    scope.workspace_id,
                    scope.campaign_id,
                    after_sequence,
                    bounded_limit + 1,
                ),
            ).fetchall()
            for row in rows:
                self._verify_audit_projection(row)
            public_rows = [self._audit_public_row(row) for row in rows]
        return self._page_result(
            public_rows, limit=bounded_limit, cursor_field="sequence", after=after_sequence
        )

    def receipt_page(
        self,
        scope,
        *,
        after_version: int,
        limit: int,
    ) -> dict[str, Any]:
        bounded_limit = min(max(limit, 1), 50)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM campaign_agent_receipts
                WHERE workspace_id = ? AND campaign_id = ? AND attachment_version > ?
                ORDER BY attachment_version ASC LIMIT ?
                """,
                (
                    scope.workspace_id,
                    scope.campaign_id,
                    after_version,
                    bounded_limit + 1,
                ),
            ).fetchall()
            for row in rows:
                self._verify_transition(row)
            public_rows = [self._transition_public_row(row) for row in rows]
        return self._page_result(
            public_rows,
            limit=bounded_limit,
            cursor_field="attachment_version",
            after=after_version,
        )

    def _public_view(
        self,
        connection: sqlite3.Connection,
        scope,
        *,
        now: datetime,
        after_sequence: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        row = connection.execute(
            """SELECT * FROM campaign_agent_attachments
            WHERE workspace_id = ? AND campaign_id = ?""",
            (scope.workspace_id, scope.campaign_id),
        ).fetchone()
        attachment = None
        if row is not None:
            self._verify_attachment_projection(row)
            grant = connection.execute(
                "SELECT * FROM campaign_agent_grants WHERE receipt_id = ?",
                (row["grant_receipt_id"],),
            ).fetchone()
            if grant is None:
                raise CampaignAgentIntegrityError("attachment grant receipt is missing")
            verified_grant = self._verify_grant(grant)
            if verified_grant.receipt_digest != row["grant_receipt_digest"]:
                raise CampaignAgentIntegrityError("attachment grant receipt binding is invalid")
            receipt_limit = 19 if row["status"] == "attached" else 20
            receipts = connection.execute(
                """
                SELECT *
                FROM campaign_agent_receipts
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY attachment_version DESC LIMIT ?
                """,
                (scope.workspace_id, scope.campaign_id, receipt_limit),
            ).fetchall()
            receipts = list(reversed(receipts))
            if not receipts:
                raise CampaignAgentIntegrityError("attachment transition history is missing")
            for receipt in receipts:
                self._verify_transition(receipt)
            expires_at = datetime.fromisoformat(row["credential_expires_at"].replace("Z", "+00:00"))
            if row["status"] == "revoked":
                credential_status, liveness = "revoked", "revoked"
            elif expires_at <= now:
                credential_status, liveness = "expired", "expired"
            elif row["last_seen_at"] is None:
                credential_status, liveness = "active", "offline"
            else:
                last_seen = datetime.fromisoformat(row["last_seen_at"].replace("Z", "+00:00"))
                age = now - last_seen
                credential_status = "active"
                liveness = (
                    "live"
                    if age <= timedelta(minutes=2)
                    else ("idle" if age <= timedelta(minutes=10) else "offline")
                )
            attachment = {
                "schema_version": "campaign_agent_public_attachment.v1",
                "attachment_id": row["attachment_id"],
                "attachment_version": int(row["attachment_version"]),
                "status": row["status"],
                "requested_capabilities": json.loads(row["requested_capabilities_json"]),
                "granted_capabilities": json.loads(row["granted_capabilities_json"]),
                "receipt_window": {
                    "from_version": int(receipts[0]["attachment_version"]),
                    "through_version": int(receipts[-1]["attachment_version"]),
                    "has_earlier": int(receipts[0]["attachment_version"]) > 1,
                },
                "provenance": {
                    "agent_family": row["agent_family"],
                    "agent_origin": row["agent_origin"],
                    "agent_origin_status": "verified",
                    "session_id": row["session_id"],
                    "agent_principal_id": row["agent_principal_id"],
                    "attached_at": row["attached_at"],
                    "attached_by": row["attached_by"],
                    "grant_receipt_id": row["grant_receipt_id"],
                    "grant_receipt_digest": row["grant_receipt_digest"],
                    "credential_issued_at": row["credential_issued_at"],
                    "credential_expires_at": row["credential_expires_at"],
                    "credential_status": credential_status,
                    "credential_revocation_revision": int(row["credential_revocation_revision"]),
                    "credential_status_source": "campaign_authority",
                    "liveness": liveness,
                    "resume_cursor": row["resume_cursor"],
                    "revoked_at": row["revoked_at"],
                    "revoked_by": row["revoked_by"],
                },
                "receipts": [self._transition_public_row(receipt) for receipt in receipts],
            }
        events = connection.execute(
            """
            SELECT *
            FROM campaign_agent_events
            WHERE workspace_id = ? AND campaign_id = ? AND sequence > ?
            ORDER BY sequence ASC LIMIT ?
            """,
            (scope.workspace_id, scope.campaign_id, after_sequence, min(limit, 50)),
        ).fetchall()
        for event in events:
            self._verify_audit_projection(event)
        return {
            "schema_version": "campaign_agent_public_view.v1",
            "observed_at": _iso(now),
            "scope": {
                "workspace_id": scope.workspace_id,
                "campaign_id": scope.campaign_id,
            },
            "attachment": attachment,
            "audit_events": [self._audit_public_row(event) for event in events],
        }


class CampaignAgentService:
    def __init__(
        self,
        campaign_repository: CampaignRepository,
        repository: CampaignAgentRepository,
        *,
        origin_verifier: CampaignAgentOriginVerifier,
        credential_broker: CredentialBroker | None = None,
    ):
        self.campaign_repository = campaign_repository
        self.repository = repository
        self.origin_verifier = origin_verifier
        self.credential_broker = credential_broker

    @staticmethod
    def _require_human(principal: ActorPrincipal, workspace_id: str) -> None:
        principal.require(workspace_id, Capability.CAMPAIGN_REVISE)
        if principal.autonomy_profile != AutonomyProfile.DESKTOP_USER:
            raise CampaignAgentAuthorizationError("a human campaign grant is required")

    def _require_campaign(self, workspace_id: str, campaign_id: str) -> None:
        try:
            self.campaign_repository.get_campaign(workspace_id, campaign_id)
        except RecordNotFoundError:
            raise

    def _verify_origin(
        self,
        scope: CampaignAgentScope,
        family: CampaignAgentFamily,
        origin: str,
        session_id: str,
        principal_id: str,
    ) -> None:
        try:
            verified = self.origin_verifier(scope, family, origin, session_id, principal_id)
        except Exception:
            verified = False
        if not verified:
            raise CampaignAgentAuthorizationError("agent origin is not verified")

    def issue_grant(
        self,
        principal: ActorPrincipal,
        request: CampaignAgentGrantRequest,
        *,
        now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=10),
    ) -> CampaignAgentGrantConfirmation:
        if ttl <= timedelta(0) or ttl > timedelta(minutes=30):
            raise ValueError("grant TTL must be positive and at most 30 minutes")
        self._require_human(principal, request.scope.workspace_id)
        self._require_campaign(request.scope.workspace_id, request.scope.campaign_id)
        self._verify_origin(
            request.scope,
            request.agent_family,
            request.agent_origin,
            request.session_id,
            request.agent_principal_id,
        )
        if request.agent_principal_id == principal.actor_id:
            raise CampaignAgentAuthorizationError("human and agent principals must differ")
        mapped_campaign_capabilities(request.granted_capabilities)
        return self.repository.issue_grant(
            request,
            human_principal_id=principal.actor_id,
            now=(now or utc_now()).astimezone(UTC).replace(microsecond=0),
            ttl=ttl,
        )

    def attach(
        self,
        principal: ActorPrincipal,
        request: CampaignAgentAttachRequest,
        *,
        now: datetime | None = None,
        credential_ttl: timedelta = timedelta(hours=1),
    ) -> tuple[dict[str, Any], bool]:
        if credential_ttl <= timedelta(0) or credential_ttl > timedelta(hours=24):
            raise ValueError("credential TTL must be positive and at most 24 hours")
        self._require_human(principal, request.scope.workspace_id)
        self._require_campaign(request.scope.workspace_id, request.scope.campaign_id)
        if request.confirmation_receipt.human_principal_id != principal.actor_id:
            raise CampaignAgentAuthorizationError(
                "grant receipt is bound to another human principal"
            )
        self._verify_origin(
            request.scope,
            request.agent_family,
            request.agent_origin,
            request.session_id,
            request.agent_principal_id,
        )
        mapped_campaign_capabilities(request.granted_capabilities)
        broker = self.credential_broker
        if broker is None:
            raise CampaignAgentBrokerUnavailableError(
                "trusted campaign-agent credential broker is unavailable"
            )
        observed_at = (now or utc_now()).astimezone(UTC).replace(microsecond=0)
        credential_id = f"cag_{uuid4().hex}"
        secret = secrets.token_urlsafe(32)
        salt = secrets.token_hex(16)
        token_hash = hashlib.sha256(f"{salt}:{secret}".encode()).hexdigest()
        raw_token = f"bgag.{credential_id}.{secret}"
        expires_at = observed_at + credential_ttl

        prepared, replay_view, replayed = self.repository.prepare_attach(
            request,
            human_actor_id=principal.actor_id,
            credential_id=credential_id,
            credential_salt=salt,
            credential_hash=token_hash,
            credential_expires_at=expires_at,
            now=observed_at,
        )
        if replayed:
            assert replay_view is not None
            return replay_view, True
        assert prepared is not None
        try:
            broker(
                BrokeredCampaignAgentCredential(
                    attachment_id=prepared.attachment_id,
                    credential_id=credential_id,
                    raw_token=raw_token,
                    workspace_id=request.scope.workspace_id,
                    campaign_id=request.scope.campaign_id,
                    agent_family=request.agent_family,
                    agent_origin=request.agent_origin,
                    session_id=request.session_id,
                    agent_principal_id=request.agent_principal_id,
                    granted_capabilities=request.granted_capabilities,
                    authorization_revision=request.confirmation_receipt.grant_revision,
                    issued_at=prepared.credential_issued_at,
                    expires_at=expires_at,
                )
            )
        except Exception as exc:
            self.repository.reject_attach(prepared.operation_id, credential_id)
            raise CampaignAgentBrokerUnavailableError(
                "trusted campaign-agent credential broker rejected delivery"
            ) from exc
        return (
            self.repository.acknowledge_attach(
                prepared.operation_id,
                credential_id,
                now=(
                    utc_now().astimezone(UTC).replace(microsecond=0) if now is None else observed_at
                ),
            ),
            False,
        )

    def revoke(
        self,
        principal: ActorPrincipal,
        request: CampaignAgentRevokeRequest,
        *,
        now: datetime | None = None,
    ) -> tuple[dict[str, Any], bool]:
        self._require_human(principal, request.scope.workspace_id)
        self._require_campaign(request.scope.workspace_id, request.scope.campaign_id)
        return self.repository.revoke(
            request,
            human_actor_id=principal.actor_id,
            now=(now or utc_now()).astimezone(UTC).replace(microsecond=0),
        )

    def heartbeat(
        self,
        raw_token: str,
        request: CampaignAgentHeartbeatRequest,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        self._require_campaign(request.scope.workspace_id, request.scope.campaign_id)
        return self.repository.authenticate_and_heartbeat(
            raw_token,
            request,
            now=(now or utc_now()).astimezone(UTC).replace(microsecond=0),
        )

    def authorize_action(
        self,
        raw_token: str,
        context: CampaignAgentActionContext,
        *,
        required_capability: CampaignAgentCapability,
        now: datetime | None = None,
    ) -> AuthorizedCampaignAgentAction:
        """Authorize one adapter-owned action against live provenance and grant state.

        This is an internal adapter boundary. Raw campaign-agent credentials must
        never be accepted from renderer-facing REST bodies or returned in public
        projections. The action adapter, not the agent, selects the required
        capability for its fixed operation.
        """

        self._require_campaign(context.scope.workspace_id, context.scope.campaign_id)
        self._verify_origin(
            context.scope,
            context.agent_family,
            context.agent_origin,
            context.session_id,
            context.agent_principal_id,
        )
        return self.repository.authorize_action(
            raw_token,
            context,
            required_capability=required_capability,
            now=(now or utc_now()).astimezone(UTC).replace(microsecond=0),
        )

    def authorize_bearer_action(
        self,
        raw_token: str,
        *,
        required_capability: CampaignAgentCapability,
        now: datetime | None = None,
    ) -> AuthorizedCampaignAgentAction:
        """Derive exact action provenance from the persisted bearer attachment."""

        authorization = self.repository.authorize_bearer_action(
            raw_token,
            required_capability=required_capability,
            now=(now or utc_now()).astimezone(UTC).replace(microsecond=0),
        )
        self._require_campaign(authorization.scope.workspace_id, authorization.scope.campaign_id)
        self._verify_origin(
            authorization.scope,
            authorization.agent_family,
            authorization.agent_origin,
            authorization.session_id,
            authorization.principal.actor_id,
        )
        return authorization

    def public_view(
        self,
        principal: ActorPrincipal,
        query: CampaignAgentPublicViewQuery,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        principal.require(query.workspace_id, Capability.CAMPAIGN_READ)
        self._require_campaign(query.workspace_id, query.campaign_id)
        return self.repository.public_view(
            query,
            now=(now or utc_now()).astimezone(UTC).replace(microsecond=0),
        )

    def audit_page(
        self,
        principal: ActorPrincipal,
        scope,
        *,
        after_sequence: int,
        limit: int,
    ) -> dict[str, Any]:
        principal.require(scope.workspace_id, Capability.CAMPAIGN_READ)
        self._require_campaign(scope.workspace_id, scope.campaign_id)
        return self.repository.audit_page(scope, after_sequence=after_sequence, limit=limit)

    def receipt_page(
        self,
        principal: ActorPrincipal,
        scope,
        *,
        after_version: int,
        limit: int,
    ) -> dict[str, Any]:
        principal.require(scope.workspace_id, Capability.CAMPAIGN_READ)
        self._require_campaign(scope.workspace_id, scope.campaign_id)
        return self.repository.receipt_page(scope, after_version=after_version, limit=limit)


__all__ = [
    "AuthorizedCampaignAgentAction",
    "BrokeredCampaignAgentCredential",
    "CampaignAgentAuthorizationError",
    "CampaignAgentBrokerUnavailableError",
    "CampaignAgentConflictError",
    "CampaignAgentCredentialError",
    "CampaignAgentIntegrityError",
    "CampaignAgentRepository",
    "CampaignAgentService",
    "PROHIBITED_CAMPAIGN_AGENT_CAPABILITIES",
    "capability_binding_digest",
    "grant_confirmation_to_wire",
    "mapped_campaign_capabilities",
]
