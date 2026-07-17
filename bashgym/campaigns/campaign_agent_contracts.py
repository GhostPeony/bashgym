"""Fail-closed contracts for brokered, campaign-scoped coding agents."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from bashgym.campaigns.contracts import FrozenContractModel

CampaignAgentIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]


class CampaignAgentFamily(str, Enum):
    CODEX = "codex"
    HERMES = "hermes"


class CampaignAgentCapability(str, Enum):
    """Small product vocabulary; each value has one fixed server capability mapping."""

    CAMPAIGN_OBSERVE = "campaign_observe"
    TRAINING_LAUNCH = "training_launch"
    TRAINING_PAUSE_SELF = "training_pause_self"
    ARTIFACT_READ = "artifact_read"
    ARTIFACT_PROPOSE = "artifact_propose"


class CampaignAgentScope(FrozenContractModel):
    workspace_id: CampaignAgentIdentifier
    campaign_id: CampaignAgentIdentifier


class CampaignAgentGrantRequest(FrozenContractModel):
    scope: CampaignAgentScope
    agent_family: CampaignAgentFamily
    agent_origin: CampaignAgentIdentifier
    agent_principal_id: CampaignAgentIdentifier
    session_id: CampaignAgentIdentifier
    requested_capabilities: tuple[CampaignAgentCapability, ...]
    granted_capabilities: tuple[CampaignAgentCapability, ...]
    idempotency_key: CampaignAgentIdentifier

    @model_validator(mode="after")
    def validate_capabilities(self) -> CampaignAgentGrantRequest:
        requested = tuple(sorted(self.requested_capabilities, key=str))
        granted = tuple(sorted(self.granted_capabilities, key=str))
        if not requested:
            raise ValueError("at least one explicit campaign-agent capability is required")
        if len(set(requested)) != len(requested) or len(set(granted)) != len(granted):
            raise ValueError("campaign-agent capabilities must be unique")
        if not set(granted).issubset(requested):
            raise ValueError("granted capabilities must be a subset of requested capabilities")
        object.__setattr__(self, "requested_capabilities", requested)
        object.__setattr__(self, "granted_capabilities", granted)
        return self


class CampaignAgentGrantConfirmation(FrozenContractModel):
    schema_version: Literal["campaign_agent_grant_confirmation.v1"] = (
        "campaign_agent_grant_confirmation.v1"
    )
    issuer: Literal["campaign_authority"] = "campaign_authority"
    receipt_id: CampaignAgentIdentifier
    receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    human_principal_id: CampaignAgentIdentifier
    scope: CampaignAgentScope
    agent_family: CampaignAgentFamily
    agent_origin: CampaignAgentIdentifier
    agent_principal_id: CampaignAgentIdentifier
    session_id: CampaignAgentIdentifier
    requested_capabilities: tuple[CampaignAgentCapability, ...]
    granted_capabilities: tuple[CampaignAgentCapability, ...]
    capability_digest: str = Field(pattern=r"^fnv1a32:[0-9a-f]{8}$")
    grant_revision: int = Field(ge=1)
    issued_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def validate_times(self) -> CampaignAgentGrantConfirmation:
        if self.issued_at >= self.expires_at:
            raise ValueError("grant confirmation must be issued before it expires")
        if self.human_principal_id == self.agent_principal_id:
            raise ValueError("grant confirmation requires a distinct human principal")
        return self


class CampaignAgentAttachRequest(FrozenContractModel):
    scope: CampaignAgentScope
    agent_family: CampaignAgentFamily
    agent_origin: CampaignAgentIdentifier
    agent_principal_id: CampaignAgentIdentifier
    session_id: CampaignAgentIdentifier
    requested_capabilities: tuple[CampaignAgentCapability, ...]
    granted_capabilities: tuple[CampaignAgentCapability, ...]
    confirmation_receipt: CampaignAgentGrantConfirmation
    base_attachment_version: int | None = Field(default=None, ge=1)
    idempotency_key: CampaignAgentIdentifier

    @model_validator(mode="after")
    def validate_declarations(self) -> CampaignAgentAttachRequest:
        declared = CampaignAgentGrantRequest(
            scope=self.scope,
            agent_family=self.agent_family,
            agent_origin=self.agent_origin,
            agent_principal_id=self.agent_principal_id,
            session_id=self.session_id,
            requested_capabilities=self.requested_capabilities,
            granted_capabilities=self.granted_capabilities,
            idempotency_key=self.idempotency_key,
        )
        object.__setattr__(self, "requested_capabilities", declared.requested_capabilities)
        object.__setattr__(self, "granted_capabilities", declared.granted_capabilities)
        return self


class CampaignAgentRevokeRequest(FrozenContractModel):
    scope: CampaignAgentScope
    attachment_id: CampaignAgentIdentifier
    attachment_version: int = Field(ge=1)
    idempotency_key: CampaignAgentIdentifier


class CampaignAgentHeartbeatRequest(FrozenContractModel):
    scope: CampaignAgentScope
    agent_family: CampaignAgentFamily
    agent_origin: CampaignAgentIdentifier
    agent_principal_id: CampaignAgentIdentifier
    session_id: CampaignAgentIdentifier
    resume_cursor: CampaignAgentIdentifier | None = None
    resume_sequence: int | None = Field(default=None, ge=0)
    expected_resume_cursor: CampaignAgentIdentifier | None = None

    @model_validator(mode="after")
    def validate_cursor_advance(self) -> CampaignAgentHeartbeatRequest:
        if (self.resume_cursor is None) != (self.resume_sequence is None):
            raise ValueError("resume cursor and sequence must be supplied together")
        if self.resume_cursor is None and self.expected_resume_cursor is not None:
            raise ValueError("cursor CAS expectation requires a cursor advance")
        return self


class CampaignAgentActionContext(FrozenContractModel):
    """Server-consumed provenance for one protected campaign action.

    The required capability is intentionally not caller-controlled. Each action
    adapter supplies its fixed capability when asking campaign authority to
    authenticate this exact session provenance.
    """

    scope: CampaignAgentScope
    agent_family: CampaignAgentFamily
    agent_origin: CampaignAgentIdentifier
    agent_principal_id: CampaignAgentIdentifier
    session_id: CampaignAgentIdentifier


class CampaignAgentHostSessionRegistrationRequest(FrozenContractModel):
    """Exact, short-lived host attestation supplied by the desktop authority."""

    scope: CampaignAgentScope
    agent_family: CampaignAgentFamily
    agent_origin: CampaignAgentIdentifier
    agent_principal_id: CampaignAgentIdentifier
    session_id: CampaignAgentIdentifier
    ephemeral_public_key: str = Field(min_length=43, max_length=43, pattern=r"^[A-Za-z0-9_-]{43}$")
    ttl_seconds: int = Field(ge=30, le=600)
    idempotency_key: CampaignAgentIdentifier

    @field_validator("ephemeral_public_key")
    @classmethod
    def canonical_x25519_key(cls, value: str) -> str:
        import base64

        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey,
            X25519PublicKey,
        )

        try:
            decoded = base64.urlsafe_b64decode(value + "=")
        except Exception as exc:
            raise ValueError("canonical X25519 public key required") from exc
        if len(decoded) != 32 or base64.urlsafe_b64encode(decoded).decode().rstrip("=") != value:
            raise ValueError("canonical X25519 public key required")
        try:
            probe_private = X25519PrivateKey.from_private_bytes(b"\x01" * 32)
            probe_private.exchange(X25519PublicKey.from_public_bytes(decoded))
        except ValueError as exc:
            raise ValueError("usable X25519 public key required") from exc
        return value


class CampaignAgentPublicViewQuery(FrozenContractModel):
    workspace_id: CampaignAgentIdentifier
    campaign_id: CampaignAgentIdentifier
    after_sequence: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)


__all__ = [
    "CampaignAgentActionContext",
    "CampaignAgentAttachRequest",
    "CampaignAgentCapability",
    "CampaignAgentFamily",
    "CampaignAgentGrantConfirmation",
    "CampaignAgentGrantRequest",
    "CampaignAgentHostSessionRegistrationRequest",
    "CampaignAgentHeartbeatRequest",
    "CampaignAgentPublicViewQuery",
    "CampaignAgentRevokeRequest",
    "CampaignAgentScope",
]
