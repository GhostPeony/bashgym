"""Sealed desktop host attestations and one-time encrypted credential delivery.

This module does not assert that a PTY is alive.  It records a short-lived
attestation made by an authenticated desktop user.  Electron/main remains
responsible for proving PTY liveness before registration and revoking the
registration when that PTY exits.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import Field

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.campaign_agent_contracts import (
    CampaignAgentFamily,
    CampaignAgentHostSessionRegistrationRequest,
    CampaignAgentIdentifier,
    CampaignAgentScope,
)
from bashgym.campaigns.campaign_agents import BrokeredCampaignAgentCredential
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    AutonomyProfile,
    Capability,
    FrozenContractModel,
    utc_now,
)
from bashgym.campaigns.persistence import CampaignRepository, RecordNotFoundError

_REGISTRATION_DOMAIN = "bashgym.campaign-agent-host-registration.v1"
_DELIVERY_DOMAIN = "bashgym.campaign-agent-encrypted-delivery.v1"
_HKDF_INFO_PREFIX = "bashgym.campaign-agent-delivery-key.v1"
_ALGORITHM = "X25519-HKDF-SHA256+CHACHA20-POLY1305"


class CampaignAgentSessionError(RuntimeError):
    code = "campaign_agent_session_error"


class CampaignAgentSessionConflictError(CampaignAgentSessionError):
    code = "campaign_agent_session_conflict"


class CampaignAgentSessionAuthorizationError(PermissionError):
    code = "campaign_agent_session_authorization_denied"


class CampaignAgentSessionIntegrityError(CampaignAgentSessionError):
    code = "campaign_agent_session_integrity_failed"


class CampaignAgentSessionReceipt(FrozenContractModel):
    schema_version: str = "campaign_agent_host_session.v1"
    registration_id: CampaignAgentIdentifier
    scope: CampaignAgentScope
    agent_family: CampaignAgentFamily
    agent_origin: CampaignAgentIdentifier
    agent_principal_id: CampaignAgentIdentifier
    session_id: CampaignAgentIdentifier
    public_key_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    registered_at: datetime
    expires_at: datetime
    status: str


class CampaignAgentDeliveryEnvelope(FrozenContractModel):
    schema_version: str = "campaign_agent_delivery_envelope.v1"
    envelope_id: CampaignAgentIdentifier
    registration_id: CampaignAgentIdentifier
    credential_id: CampaignAgentIdentifier
    algorithm: str
    ephemeral_public_key: str
    hkdf_salt: str
    hkdf_info: str
    nonce: str
    ciphertext: str
    aad_json: str
    created_at: datetime


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class CampaignAgentSessionRepository:
    def __init__(self, db_path: str | Path, *, sealer: ArtifactSealer):
        self.db_path = Path(db_path)
        self.sealer = sealer

    @contextmanager
    def _connection(self, *, write: bool = False):
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except Exception:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection(write=True) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS campaign_agent_host_sessions (
                    registration_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    agent_family TEXT NOT NULL,
                    agent_origin TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    agent_principal_id TEXT NOT NULL,
                    public_key TEXT NOT NULL,
                    public_key_digest TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('live', 'revoked')),
                    registered_by TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    seal_digest TEXT NOT NULL,
                    seal_key_version TEXT NOT NULL,
                    UNIQUE (registered_by, idempotency_key),
                    UNIQUE (public_key_digest)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_agent_live_exact_session
                ON campaign_agent_host_sessions (
                    workspace_id, campaign_id, agent_family, agent_origin,
                    session_id, agent_principal_id
                ) WHERE status = 'live';

                CREATE TABLE IF NOT EXISTS campaign_agent_delivery_envelopes (
                    envelope_id TEXT PRIMARY KEY,
                    registration_id TEXT NOT NULL REFERENCES campaign_agent_host_sessions(registration_id),
                    credential_id TEXT NOT NULL UNIQUE,
                    algorithm TEXT NOT NULL,
                    ephemeral_public_key TEXT NOT NULL,
                    hkdf_salt TEXT NOT NULL,
                    hkdf_info TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    ciphertext TEXT NOT NULL,
                    aad_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'claimed', 'revoked')),
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    claimed_at TEXT,
                    claimed_by TEXT,
                    seal_digest TEXT NOT NULL,
                    seal_key_version TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_agent_one_pending_delivery
                ON campaign_agent_delivery_envelopes (registration_id)
                WHERE state = 'pending';
                """)

    @staticmethod
    def _registration_payload(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "registration_id",
                "workspace_id",
                "campaign_id",
                "agent_family",
                "agent_origin",
                "session_id",
                "agent_principal_id",
                "public_key",
                "public_key_digest",
                "status",
                "registered_by",
                "registered_at",
                "expires_at",
                "revoked_at",
                "idempotency_key",
                "request_hash",
            )
        }

    @staticmethod
    def _delivery_payload(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "envelope_id",
                "registration_id",
                "credential_id",
                "algorithm",
                "ephemeral_public_key",
                "hkdf_salt",
                "hkdf_info",
                "nonce",
                "ciphertext",
                "aad_json",
                "state",
                "created_at",
                "expires_at",
                "claimed_at",
                "claimed_by",
            )
        }

    def _seal(self, payload: dict[str, Any], domain: str) -> str:
        return self.sealer.sign_canonical_payload(payload, domain=domain)

    def _verify_registration(self, row: sqlite3.Row) -> None:
        if row["seal_key_version"] != self.sealer.key_version or not secrets.compare_digest(
            row["seal_digest"], self._seal(self._registration_payload(row), _REGISTRATION_DOMAIN)
        ):
            raise CampaignAgentSessionIntegrityError("host session registration seal is invalid")

    def _verify_delivery(self, row: sqlite3.Row) -> None:
        if row["seal_key_version"] != self.sealer.key_version or not secrets.compare_digest(
            row["seal_digest"], self._seal(self._delivery_payload(row), _DELIVERY_DOMAIN)
        ):
            raise CampaignAgentSessionIntegrityError("credential delivery seal is invalid")

    @staticmethod
    def _receipt(row: sqlite3.Row) -> CampaignAgentSessionReceipt:
        return CampaignAgentSessionReceipt(
            registration_id=row["registration_id"],
            scope=CampaignAgentScope(
                workspace_id=row["workspace_id"], campaign_id=row["campaign_id"]
            ),
            agent_family=CampaignAgentFamily(row["agent_family"]),
            agent_origin=row["agent_origin"],
            agent_principal_id=row["agent_principal_id"],
            session_id=row["session_id"],
            public_key_digest=row["public_key_digest"],
            registered_at=_time(row["registered_at"]),
            expires_at=_time(row["expires_at"]),
            status=row["status"],
        )

    def _revoke_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        self._verify_registration(row)
        if row["status"] == "revoked":
            return row
        values = dict(row)
        values["status"] = "revoked"
        values["revoked_at"] = _iso(now)
        values["seal_digest"] = self._seal(self._registration_payload(values), _REGISTRATION_DOMAIN)
        connection.execute(
            """UPDATE campaign_agent_host_sessions
               SET status='revoked', revoked_at=?, seal_digest=?
               WHERE registration_id=? AND status='live'""",
            (values["revoked_at"], values["seal_digest"], row["registration_id"]),
        )
        pending = connection.execute(
            """SELECT * FROM campaign_agent_delivery_envelopes
               WHERE registration_id=? AND state='pending'""",
            (row["registration_id"],),
        ).fetchall()
        for envelope in pending:
            self._verify_delivery(envelope)
            delivery = dict(envelope)
            delivery["state"] = "revoked"
            delivery["seal_digest"] = self._seal(self._delivery_payload(delivery), _DELIVERY_DOMAIN)
            connection.execute(
                """UPDATE campaign_agent_delivery_envelopes
                   SET state='revoked', seal_digest=? WHERE envelope_id=?""",
                (delivery["seal_digest"], envelope["envelope_id"]),
            )
        updated = connection.execute(
            "SELECT * FROM campaign_agent_host_sessions WHERE registration_id=?",
            (row["registration_id"],),
        ).fetchone()
        assert updated is not None
        return updated

    def register(
        self,
        request: CampaignAgentHostSessionRegistrationRequest,
        *,
        actor_id: str,
        now: datetime,
    ) -> CampaignAgentSessionReceipt:
        now = now.astimezone(UTC).replace(microsecond=0)
        request_payload = request.model_dump(mode="json")
        request_hash = "sha256:" + hashlib.sha256(_canonical(request_payload).encode()).hexdigest()
        public_key_digest = (
            "sha256:" + hashlib.sha256(_decode(request.ephemeral_public_key)).hexdigest()
        )
        with self._connection(write=True) as connection:
            replay = connection.execute(
                "SELECT * FROM campaign_agent_host_sessions WHERE registered_by = ? AND idempotency_key = ?",
                (actor_id, request.idempotency_key),
            ).fetchone()
            if replay is not None:
                self._verify_registration(replay)
                if replay["request_hash"] != request_hash:
                    raise CampaignAgentSessionConflictError("registration replay changed")
                return self._receipt(replay)
            exact = connection.execute(
                """SELECT * FROM campaign_agent_host_sessions
                   WHERE workspace_id = ? AND campaign_id = ? AND agent_family = ?
                     AND agent_origin = ? AND session_id = ? AND agent_principal_id = ?
                     AND status = 'live'""",
                (
                    request.scope.workspace_id,
                    request.scope.campaign_id,
                    request.agent_family.value,
                    request.agent_origin,
                    request.session_id,
                    request.agent_principal_id,
                ),
            ).fetchone()
            if exact is not None:
                self._verify_registration(exact)
                if exact["registered_by"] != actor_id:
                    raise CampaignAgentSessionAuthorizationError(
                        "live session belongs to another desktop user"
                    )
                if exact["public_key_digest"] == public_key_digest:
                    raise CampaignAgentSessionConflictError(
                        "session renewal requires a fresh public key"
                    )
                self._revoke_row(connection, exact, now=now)
            key_reuse = connection.execute(
                "SELECT * FROM campaign_agent_host_sessions WHERE public_key_digest = ?",
                (public_key_digest,),
            ).fetchone()
            if key_reuse is not None:
                self._verify_registration(key_reuse)
                raise CampaignAgentSessionConflictError("session public key was already registered")
            values: dict[str, Any] = {
                "registration_id": f"cags_{uuid4().hex}",
                "workspace_id": request.scope.workspace_id,
                "campaign_id": request.scope.campaign_id,
                "agent_family": request.agent_family.value,
                "agent_origin": request.agent_origin,
                "session_id": request.session_id,
                "agent_principal_id": request.agent_principal_id,
                "public_key": request.ephemeral_public_key,
                "public_key_digest": public_key_digest,
                "status": "live",
                "registered_by": actor_id,
                "registered_at": _iso(now),
                "expires_at": _iso(now + timedelta(seconds=request.ttl_seconds)),
                "revoked_at": None,
                "idempotency_key": request.idempotency_key,
                "request_hash": request_hash,
            }
            values["seal_digest"] = self._seal(
                self._registration_payload(values), _REGISTRATION_DOMAIN
            )
            values["seal_key_version"] = self.sealer.key_version
            columns = tuple(values)
            connection.execute(
                f"INSERT INTO campaign_agent_host_sessions ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(values[column] for column in columns),
            )
            row = connection.execute(
                "SELECT * FROM campaign_agent_host_sessions WHERE registration_id = ?",
                (values["registration_id"],),
            ).fetchone()
            assert row is not None
            return self._receipt(row)

    def get_registration(self, registration_id: str) -> sqlite3.Row:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_agent_host_sessions WHERE registration_id = ?",
                (registration_id,),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign agent host session not found")
        self._verify_registration(row)
        return row

    def get_live_exact(
        self, request: CampaignAgentHostSessionRegistrationRequest, *, now: datetime
    ) -> CampaignAgentSessionReceipt:
        row = self.find_live_exact(
            request.scope,
            request.agent_family,
            request.agent_origin,
            request.session_id,
            request.agent_principal_id,
            now=now,
        )
        if row is None:
            raise CampaignAgentSessionAuthorizationError("host session is not live")
        return self._receipt(row)

    def find_live_exact(
        self,
        scope: CampaignAgentScope,
        family: CampaignAgentFamily,
        origin: str,
        session_id: str,
        agent_principal_id: str,
        *,
        now: datetime,
    ) -> sqlite3.Row | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT * FROM campaign_agent_host_sessions
                   WHERE workspace_id = ? AND campaign_id = ? AND agent_family = ?
                     AND agent_origin = ? AND session_id = ? AND agent_principal_id = ?
                     AND status = 'live'
                   ORDER BY registered_at DESC, registration_id DESC
                   LIMIT 1""",
                (
                    scope.workspace_id,
                    scope.campaign_id,
                    family.value,
                    origin,
                    session_id,
                    agent_principal_id,
                ),
            ).fetchone()
        if row is None:
            return None
        self._verify_registration(row)
        if row["status"] != "live" or _time(row["expires_at"]) <= now.astimezone(UTC):
            return None
        return row

    def revoke(
        self, registration_id: str, *, actor_id: str, now: datetime
    ) -> CampaignAgentSessionReceipt:
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM campaign_agent_host_sessions WHERE registration_id = ?",
                (registration_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign agent host session not found")
            self._verify_registration(row)
            if row["registered_by"] != actor_id:
                raise CampaignAgentSessionAuthorizationError("host session belongs to another user")
            if row["status"] == "revoked":
                return self._receipt(row)
            return self._receipt(self._revoke_row(connection, row, now=now))

    def store_delivery(
        self,
        registration: sqlite3.Row,
        credential: BrokeredCampaignAgentCredential,
        *,
        now: datetime,
    ) -> None:
        self._verify_registration(registration)
        recipient = X25519PublicKey.from_public_bytes(_decode(registration["public_key"]))
        ephemeral_private = X25519PrivateKey.generate()
        ephemeral_public = _b64(
            ephemeral_private.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        )
        envelope_id = f"cage_{uuid4().hex}"
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(12)
        hkdf_info = (
            f"{_HKDF_INFO_PREFIX}:{registration['registration_id']}:{credential.credential_id}"
        )
        aad = {
            "schema_version": "campaign_agent_delivery_aad.v1",
            "envelope_id": envelope_id,
            "registration_id": registration["registration_id"],
            "credential_id": credential.credential_id,
            "attachment_id": credential.attachment_id,
            "workspace_id": credential.workspace_id,
            "campaign_id": credential.campaign_id,
            "agent_family": credential.agent_family.value,
            "agent_origin": credential.agent_origin,
            "session_id": credential.session_id,
            "agent_principal_id": credential.agent_principal_id,
            "public_key_digest": registration["public_key_digest"],
            "issued_at": _iso(credential.issued_at),
            "expires_at": _iso(credential.expires_at),
        }
        aad_json = _canonical(aad)
        plaintext = _canonical(
            {
                "schema_version": "campaign_agent_credential_delivery.v1",
                "registration_id": registration["registration_id"],
                "credential_id": credential.credential_id,
                "raw_token": credential.raw_token,
            }
        ).encode()
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=hkdf_info.encode()).derive(
            ephemeral_private.exchange(recipient)
        )
        ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad_json.encode())
        values: dict[str, Any] = {
            "envelope_id": envelope_id,
            "registration_id": registration["registration_id"],
            "credential_id": credential.credential_id,
            "algorithm": _ALGORITHM,
            "ephemeral_public_key": ephemeral_public,
            "hkdf_salt": _b64(salt),
            "hkdf_info": hkdf_info,
            "nonce": _b64(nonce),
            "ciphertext": _b64(ciphertext),
            "aad_json": aad_json,
            "state": "pending",
            "created_at": _iso(credential.issued_at),
            "expires_at": _iso(credential.expires_at),
            "claimed_at": None,
            "claimed_by": None,
        }
        values["seal_digest"] = self._seal(self._delivery_payload(values), _DELIVERY_DOMAIN)
        values["seal_key_version"] = self.sealer.key_version
        with self._connection(write=True) as connection:
            current = connection.execute(
                "SELECT * FROM campaign_agent_host_sessions WHERE registration_id=?",
                (registration["registration_id"],),
            ).fetchone()
            if current is None:
                raise CampaignAgentSessionAuthorizationError("host session disappeared")
            self._verify_registration(current)
            if current["status"] != "live" or _time(current["expires_at"]) <= now:
                raise CampaignAgentSessionAuthorizationError("host session is not live")
            columns = tuple(values)
            try:
                connection.execute(
                    f"INSERT INTO campaign_agent_delivery_envelopes ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    tuple(values[column] for column in columns),
                )
            except sqlite3.IntegrityError as exc:
                raise CampaignAgentSessionConflictError(
                    "credential delivery already exists"
                ) from exc

    def claim(
        self, registration_id: str, *, actor_id: str, now: datetime
    ) -> CampaignAgentDeliveryEnvelope:
        with self._connection(write=True) as connection:
            registration = connection.execute(
                "SELECT * FROM campaign_agent_host_sessions WHERE registration_id=?",
                (registration_id,),
            ).fetchone()
            if registration is None:
                raise RecordNotFoundError("campaign agent host session not found")
            self._verify_registration(registration)
            if registration["registered_by"] != actor_id:
                raise CampaignAgentSessionAuthorizationError("delivery belongs to another user")
            if registration["status"] != "live" or _time(registration["expires_at"]) <= now:
                raise CampaignAgentSessionAuthorizationError("host session is not live")
            row = connection.execute(
                """SELECT * FROM campaign_agent_delivery_envelopes
                   WHERE registration_id=? AND state='pending'
                   ORDER BY created_at, envelope_id LIMIT 1""",
                (registration_id,),
            ).fetchone()
            if row is None:
                raise CampaignAgentSessionConflictError("no pending credential delivery")
            self._verify_delivery(row)
            if _time(row["expires_at"]) <= now:
                raise CampaignAgentSessionAuthorizationError("credential delivery expired")
            values = dict(row)
            values["state"] = "claimed"
            values["claimed_at"] = _iso(now)
            values["claimed_by"] = actor_id
            values["seal_digest"] = self._seal(self._delivery_payload(values), _DELIVERY_DOMAIN)
            updated = connection.execute(
                """UPDATE campaign_agent_delivery_envelopes
                   SET state='claimed', claimed_at=?, claimed_by=?, seal_digest=?
                   WHERE envelope_id=? AND state='pending'""",
                (values["claimed_at"], actor_id, values["seal_digest"], row["envelope_id"]),
            )
            if updated.rowcount != 1:
                raise CampaignAgentSessionConflictError("credential delivery was already claimed")
            return CampaignAgentDeliveryEnvelope(
                envelope_id=row["envelope_id"],
                registration_id=row["registration_id"],
                credential_id=row["credential_id"],
                algorithm=row["algorithm"],
                ephemeral_public_key=row["ephemeral_public_key"],
                hkdf_salt=row["hkdf_salt"],
                hkdf_info=row["hkdf_info"],
                nonce=row["nonce"],
                ciphertext=row["ciphertext"],
                aad_json=row["aad_json"],
                created_at=_time(row["created_at"]),
            )


class CampaignAgentSessionService:
    def __init__(
        self,
        repository: CampaignAgentSessionRepository,
        campaign_repository: CampaignRepository | None = None,
    ):
        self.repository = repository
        self.campaign_repository = campaign_repository

    def _authorize(self, principal: ActorPrincipal, workspace_id: str) -> None:
        principal.require(workspace_id, Capability.CAMPAIGN_REVISE)
        if principal.autonomy_profile != AutonomyProfile.DESKTOP_USER:
            raise CampaignAgentSessionAuthorizationError("desktop user authority is required")

    def register(
        self,
        principal: ActorPrincipal,
        request: CampaignAgentHostSessionRegistrationRequest,
        *,
        now: datetime | None = None,
    ) -> CampaignAgentSessionReceipt:
        self._authorize(principal, request.scope.workspace_id)
        if self.campaign_repository is not None:
            self.campaign_repository.get_campaign(
                request.scope.workspace_id, request.scope.campaign_id
            )
        return self.repository.register(
            request, actor_id=principal.actor_id, now=(now or utc_now())
        )

    def revoke(
        self,
        principal: ActorPrincipal,
        registration_id: str,
        *,
        now: datetime | None = None,
    ) -> CampaignAgentSessionReceipt:
        row = self.repository.get_registration(registration_id)
        self._authorize(principal, row["workspace_id"])
        return self.repository.revoke(
            registration_id, actor_id=principal.actor_id, now=(now or utc_now())
        )

    def claim(
        self,
        principal: ActorPrincipal,
        registration_id: str,
        *,
        now: datetime | None = None,
    ) -> CampaignAgentDeliveryEnvelope:
        row = self.repository.get_registration(registration_id)
        self._authorize(principal, row["workspace_id"])
        return self.repository.claim(
            registration_id,
            actor_id=principal.actor_id,
            now=(now or utc_now()).astimezone(UTC).replace(microsecond=0),
        )


class CampaignAgentSessionOriginVerifier:
    def __init__(
        self,
        repository: CampaignAgentSessionRepository,
        *,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.repository = repository
        self.clock = clock

    def __call__(
        self,
        scope: CampaignAgentScope,
        family: CampaignAgentFamily,
        origin: str,
        session_id: str,
        agent_principal_id: str,
    ) -> bool:
        try:
            return (
                self.repository.find_live_exact(
                    scope,
                    family,
                    origin,
                    session_id,
                    agent_principal_id,
                    now=self.clock(),
                )
                is not None
            )
        except Exception:
            return False


class EncryptedCampaignAgentCredentialBroker:
    def __init__(
        self,
        repository: CampaignAgentSessionRepository,
        *,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.repository = repository
        self.clock = clock

    def __call__(self, credential: BrokeredCampaignAgentCredential) -> None:
        scope = CampaignAgentScope(
            workspace_id=credential.workspace_id, campaign_id=credential.campaign_id
        )
        registration = self.repository.find_live_exact(
            scope,
            credential.agent_family,
            credential.agent_origin,
            credential.session_id,
            credential.agent_principal_id,
            now=self.clock(),
        )
        if registration is None:
            raise CampaignAgentSessionAuthorizationError("exact host session is not live")
        self.repository.store_delivery(registration, credential, now=self.clock())


__all__ = [
    "CampaignAgentDeliveryEnvelope",
    "CampaignAgentSessionAuthorizationError",
    "CampaignAgentSessionConflictError",
    "CampaignAgentSessionIntegrityError",
    "CampaignAgentSessionOriginVerifier",
    "CampaignAgentSessionReceipt",
    "CampaignAgentSessionRepository",
    "CampaignAgentSessionService",
    "EncryptedCampaignAgentCredentialBroker",
]
