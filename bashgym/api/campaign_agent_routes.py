"""Authenticated REST boundary for brokered campaign-agent attachments."""

from __future__ import annotations

import hmac
import os
from datetime import UTC
from typing import Annotated, Any, Literal, Never

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bashgym.campaigns.auth import CampaignAuthenticationError, CampaignAuthService
from bashgym.campaigns.campaign_agent_contracts import (
    CampaignAgentAttachRequest,
    CampaignAgentCapability,
    CampaignAgentFamily,
    CampaignAgentGrantConfirmation,
    CampaignAgentGrantRequest,
    CampaignAgentHeartbeatRequest,
    CampaignAgentHostSessionRegistrationRequest,
    CampaignAgentPublicViewQuery,
    CampaignAgentRevokeRequest,
    CampaignAgentScope,
)
from bashgym.campaigns.campaign_agent_sessions import (
    CampaignAgentSessionConflictError,
    CampaignAgentSessionIntegrityError,
    CampaignAgentSessionOriginVerifier,
    CampaignAgentSessionRepository,
    CampaignAgentSessionService,
    EncryptedCampaignAgentCredentialBroker,
)
from bashgym.campaigns.campaign_agents import (
    CampaignAgentAuthorizationError,
    CampaignAgentBrokerUnavailableError,
    CampaignAgentConflictError,
    CampaignAgentCredentialError,
    CampaignAgentIntegrityError,
    CampaignAgentRepository,
    CampaignAgentService,
    grant_confirmation_to_wire,
)
from bashgym.campaigns.persistence import RecordNotFoundError
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService

campaign_agent_router = APIRouter(prefix="/api/campaigns", tags=["campaign-agents"])
campaign_agent_credential_router = APIRouter(
    prefix="/api/campaign-agent", tags=["campaign-agent-credential"]
)

IdentifierField = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
    ),
]


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(value.title() for value in tail)


class TransportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", alias_generator=_camel, populate_by_name=True)


class ScopeInput(TransportModel):
    workspace_id: IdentifierField
    campaign_id: IdentifierField

    def contract(self) -> CampaignAgentScope:
        return CampaignAgentScope(workspace_id=self.workspace_id, campaign_id=self.campaign_id)


class GrantInput(TransportModel):
    scope: ScopeInput
    agent_family: CampaignAgentFamily
    agent_origin: IdentifierField
    agent_principal_id: IdentifierField
    session_id: IdentifierField
    requested_capabilities: tuple[CampaignAgentCapability, ...]
    granted_capabilities: tuple[CampaignAgentCapability, ...]
    idempotency_key: IdentifierField

    def contract(self) -> CampaignAgentGrantRequest:
        return CampaignAgentGrantRequest(
            scope=self.scope.contract(),
            agent_family=self.agent_family,
            agent_origin=self.agent_origin,
            agent_principal_id=self.agent_principal_id,
            session_id=self.session_id,
            requested_capabilities=self.requested_capabilities,
            granted_capabilities=self.granted_capabilities,
            idempotency_key=self.idempotency_key,
        )


class HumanPrincipalInput(TransportModel):
    principal_id: IdentifierField
    principal_type: Literal["human"]


class GrantConfirmationInput(TransportModel):
    schema_version: Literal["campaign_agent_grant_confirmation.v1"]
    issuer: Literal["campaign_authority"]
    receipt_id: IdentifierField
    receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    human_principal: HumanPrincipalInput
    scope: ScopeInput
    agent_family: CampaignAgentFamily
    agent_origin: IdentifierField
    agent_principal_id: IdentifierField
    session_id: IdentifierField
    requested_capabilities: tuple[CampaignAgentCapability, ...]
    granted_capabilities: tuple[CampaignAgentCapability, ...]
    capability_digest: str = Field(pattern=r"^fnv1a32:[0-9a-f]{8}$")
    grant_revision: int = Field(ge=1)
    issued_at: str
    expires_at: str

    @field_validator("issued_at", "expires_at")
    @classmethod
    def canonical_timestamp(cls, value: str) -> str:
        from datetime import datetime

        if len(value) != 20 or not value.endswith("Z"):
            raise ValueError("canonical UTC second timestamp required")
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ValueError("canonical UTC second timestamp required") from exc
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("canonical UTC second timestamp required")
        return value

    def contract(self) -> CampaignAgentGrantConfirmation:
        return CampaignAgentGrantConfirmation(
            schema_version=self.schema_version,
            issuer=self.issuer,
            receipt_id=self.receipt_id,
            receipt_digest=self.receipt_digest,
            human_principal_id=self.human_principal.principal_id,
            scope=self.scope.contract(),
            agent_family=self.agent_family,
            agent_origin=self.agent_origin,
            agent_principal_id=self.agent_principal_id,
            session_id=self.session_id,
            requested_capabilities=self.requested_capabilities,
            granted_capabilities=self.granted_capabilities,
            capability_digest=self.capability_digest,
            grant_revision=self.grant_revision,
            issued_at=self.issued_at,
            expires_at=self.expires_at,
        )


class AttachInput(TransportModel):
    action: Literal["attach"]
    scope: ScopeInput
    agent_family: CampaignAgentFamily
    agent_origin: IdentifierField
    agent_principal_id: IdentifierField
    session_id: IdentifierField
    requested_capabilities: tuple[CampaignAgentCapability, ...]
    granted_capabilities: tuple[CampaignAgentCapability, ...]
    confirmation_receipt: GrantConfirmationInput
    base_attachment_version: int | None = Field(default=None, ge=1)
    idempotency_key: IdentifierField

    def contract(self) -> CampaignAgentAttachRequest:
        return CampaignAgentAttachRequest(
            scope=self.scope.contract(),
            agent_family=self.agent_family,
            agent_origin=self.agent_origin,
            agent_principal_id=self.agent_principal_id,
            session_id=self.session_id,
            requested_capabilities=self.requested_capabilities,
            granted_capabilities=self.granted_capabilities,
            confirmation_receipt=self.confirmation_receipt.contract(),
            base_attachment_version=self.base_attachment_version,
            idempotency_key=self.idempotency_key,
        )


class RevokeInput(TransportModel):
    action: Literal["revoke"]
    attachment_id: IdentifierField
    attachment_version: int = Field(ge=1)
    scope: ScopeInput
    actor_id: IdentifierField
    idempotency_key: IdentifierField

    def contract(self) -> CampaignAgentRevokeRequest:
        return CampaignAgentRevokeRequest(
            scope=self.scope.contract(),
            attachment_id=self.attachment_id,
            attachment_version=self.attachment_version,
            idempotency_key=self.idempotency_key,
        )


class HeartbeatInput(TransportModel):
    scope: ScopeInput
    agent_family: CampaignAgentFamily
    agent_origin: IdentifierField
    agent_principal_id: IdentifierField
    session_id: IdentifierField
    resume_cursor: IdentifierField | None = None
    resume_sequence: int | None = Field(default=None, ge=0)
    expected_resume_cursor: IdentifierField | None = None

    def contract(self) -> CampaignAgentHeartbeatRequest:
        return CampaignAgentHeartbeatRequest(
            scope=self.scope.contract(),
            agent_family=self.agent_family,
            agent_origin=self.agent_origin,
            agent_principal_id=self.agent_principal_id,
            session_id=self.session_id,
            resume_cursor=self.resume_cursor,
            resume_sequence=self.resume_sequence,
            expected_resume_cursor=self.expected_resume_cursor,
        )


class HostSessionRegistrationInput(TransportModel):
    scope: ScopeInput
    agent_family: CampaignAgentFamily
    agent_origin: IdentifierField
    agent_principal_id: IdentifierField
    session_id: IdentifierField
    ephemeral_public_key: str = Field(min_length=43, max_length=43, pattern=r"^[A-Za-z0-9_-]{43}$")
    ttl_seconds: int = Field(ge=30, le=600)
    idempotency_key: IdentifierField

    def contract(self) -> CampaignAgentHostSessionRegistrationRequest:
        return CampaignAgentHostSessionRegistrationRequest(
            scope=self.scope.contract(),
            agent_family=self.agent_family,
            agent_origin=self.agent_origin,
            agent_principal_id=self.agent_principal_id,
            session_id=self.session_id,
            ephemeral_public_key=self.ephemeral_public_key,
            ttl_seconds=self.ttl_seconds,
            idempotency_key=self.idempotency_key,
        )


def _bearer(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token.strip():
        raise CampaignAuthenticationError()
    return token.strip()


def _campaign_services(
    request: Request,
) -> tuple[CampaignRuntimeRepository, CampaignAuthService]:
    repository = getattr(request.app.state, "campaign_repository", None)
    auth = getattr(request.app.state, "campaign_auth_service", None)
    if not isinstance(repository, CampaignRuntimeRepository) or not isinstance(
        auth, CampaignAuthService
    ):
        # Reuse the campaign surface's installation-aware lazy initialization.
        from bashgym.api.campaign_routes import _services

        repository, auth, _campaign_service = _services(request)
    return repository, auth


def _service(request: Request) -> tuple[CampaignAgentService, CampaignAuthService]:
    campaigns, auth = _campaign_services(request)
    repository = getattr(request.app.state, "campaign_agent_repository", None)
    if not isinstance(repository, CampaignAgentRepository):
        from bashgym.api.campaign_routes import _campaign_authority_sealer

        try:
            sealer = _campaign_authority_sealer(request)
        except Exception as exc:
            raise CampaignAgentAuthorityUnavailableError from exc
        repository = CampaignAgentRepository(campaigns.db_path, sealer=sealer)
        repository.initialize()
        request.app.state.campaign_agent_repository = repository
    service = getattr(request.app.state, "campaign_agent_service", None)
    origin_verifier = getattr(
        request.app.state, "campaign_agent_origin_verifier", lambda *_args: False
    )
    broker = getattr(request.app.state, "campaign_agent_credential_broker", None)
    if not isinstance(service, CampaignAgentService):
        service = CampaignAgentService(
            campaigns,
            repository,
            origin_verifier=origin_verifier,
            credential_broker=broker,
        )
        request.app.state.campaign_agent_service = service
    else:
        # App-state bindings may become ready after the first read-only request.
        service.origin_verifier = origin_verifier
        service.credential_broker = broker
    return service, auth


def _human_principal(request: Request):
    _agent_service, auth = _service(request)
    return auth.authenticate_access(_bearer(request))


def _session_service(request: Request) -> CampaignAgentSessionService:
    """Initialize storage without enabling host origin or credential bindings.

    Registration is only a desktop claim about a session.  The Python API must
    not infer PTY liveness from that claim, so the campaign-agent service remains
    fail closed until Electron/main explicitly installs both trusted bindings.
    """

    campaigns, _auth = _campaign_services(request)
    repository = getattr(request.app.state, "campaign_agent_session_repository", None)
    if not isinstance(repository, CampaignAgentSessionRepository):
        from bashgym.api.campaign_routes import _campaign_authority_sealer

        try:
            sealer = _campaign_authority_sealer(request)
        except Exception as exc:
            raise CampaignAgentAuthorityUnavailableError from exc
        repository = CampaignAgentSessionRepository(campaigns.db_path, sealer=sealer)
        repository.initialize()
        request.app.state.campaign_agent_session_repository = repository
    service = getattr(request.app.state, "campaign_agent_session_service", None)
    if not isinstance(service, CampaignAgentSessionService):
        service = CampaignAgentSessionService(repository, campaigns)
        request.app.state.campaign_agent_session_service = service
    return service


def _scope_wire(scope: CampaignAgentScope) -> dict[str, str]:
    return {"workspaceId": scope.workspace_id, "campaignId": scope.campaign_id}


def _session_receipt_wire(receipt) -> dict[str, Any]:
    return {
        "schemaVersion": receipt.schema_version,
        "registrationId": receipt.registration_id,
        "scope": _scope_wire(receipt.scope),
        "agentFamily": receipt.agent_family.value,
        "agentOrigin": receipt.agent_origin,
        "agentPrincipalId": receipt.agent_principal_id,
        "sessionId": receipt.session_id,
        "publicKeyDigest": receipt.public_key_digest,
        "registeredAt": receipt.registered_at,
        "expiresAt": receipt.expires_at,
        "status": receipt.status,
    }


def _delivery_wire(envelope) -> dict[str, Any]:
    return {
        "schemaVersion": envelope.schema_version,
        "envelopeId": envelope.envelope_id,
        "registrationId": envelope.registration_id,
        "credentialId": envelope.credential_id,
        "algorithm": envelope.algorithm,
        "ephemeralPublicKey": envelope.ephemeral_public_key,
        "hkdfSalt": envelope.hkdf_salt,
        "hkdfInfo": envelope.hkdf_info,
        "nonce": envelope.nonce,
        "ciphertext": envelope.ciphertext,
        "aadJson": envelope.aad_json,
        "createdAt": envelope.created_at,
    }


class CampaignAgentAuthorityUnavailableError(RuntimeError):
    """Raised when the installation-held campaign signing authority is unavailable."""


def activate_managed_desktop_campaign_agent_bindings(
    request: Request,
    *,
    authenticated_bootstrap: str,
) -> bool:
    """Install trusted session bindings after the current desktop bootstrap authenticates.

    ``campaign_worker_managed`` is established by the backend startup path after
    it validates desktop mode, campaign enablement, and the Electron launch
    bootstrap. The current launch secret is then authenticated by
    ``CampaignAuthService`` before this function is called. No request body or
    renderer-supplied trust claim participates in this decision.
    """

    current_bootstrap = os.environ.get("BASHGYM_DESKTOP_BOOTSTRAP_SECRET", "").strip()
    if (
        not bool(getattr(request.app.state, "campaign_worker_managed", False))
        or not bool(getattr(request.app.state, "campaign_desktop_bootstrap_checked", False))
        or not current_bootstrap
        or not hmac.compare_digest(current_bootstrap, authenticated_bootstrap)
    ):
        return False

    try:
        session_service = _session_service(request)
        agent_service, _auth = _service(request)
        verifier = CampaignAgentSessionOriginVerifier(session_service.repository)
        broker = EncryptedCampaignAgentCredentialBroker(session_service.repository)
    except Exception as exc:
        raise CampaignAgentAuthorityUnavailableError from exc

    request.app.state.campaign_agent_origin_verifier = verifier
    request.app.state.campaign_agent_credential_broker = broker
    agent_service.origin_verifier = verifier
    agent_service.credential_broker = broker
    return True


def _raise_api(exc: Exception) -> Never:
    if isinstance(exc, ValidationError):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "campaign_agent_request_invalid",
                "message": "Campaign-agent request validation failed.",
            },
        ) from exc
    if isinstance(exc, (CampaignAuthenticationError, CampaignAgentCredentialError)):
        raise HTTPException(
            status_code=401,
            detail={"code": "campaign_auth_required", "message": "Authentication required."},
        ) from exc
    if isinstance(exc, (CampaignAgentAuthorizationError, PermissionError)):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "campaign_agent_authorization_denied",
                "message": "Campaign-agent operation is not permitted.",
            },
        ) from exc
    if isinstance(exc, CampaignAgentSessionConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": "Campaign-agent host session state changed. Reconcile before retrying.",
            },
        ) from exc
    if isinstance(exc, RecordNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": "campaign_not_found", "message": "Campaign record not found."},
        ) from exc
    if isinstance(exc, CampaignAgentConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": "Campaign-agent state changed. Reconcile before retrying.",
            },
        ) from exc
    if isinstance(exc, CampaignAgentIntegrityError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "campaign_agent_integrity_failed",
                "message": "Campaign-agent authority state could not be verified.",
            },
        ) from exc
    if isinstance(exc, CampaignAgentSessionIntegrityError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": "Campaign-agent host session state could not be verified.",
            },
        ) from exc
    if isinstance(exc, CampaignAgentAuthorityUnavailableError):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "campaign_agent_authority_unavailable",
                "message": "The campaign authority is unavailable.",
            },
        ) from exc
    if isinstance(exc, CampaignAgentBrokerUnavailableError):
        raise HTTPException(
            status_code=503,
            detail={
                "code": exc.code,
                "message": "The trusted campaign-agent credential broker is unavailable.",
            },
        ) from exc
    raise exc


def _validated(model_type, body: Any):
    """Validate without allowing FastAPI/Pydantic to echo rejected secret canaries."""

    try:
        return model_type.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "campaign_agent_request_invalid",
                "message": "Campaign-agent request validation failed.",
            },
        ) from exc


@campaign_agent_router.post("/{campaign_id}/agent-grant")
def issue_agent_grant(campaign_id: str, body: Annotated[Any, Body()], request: Request):
    try:
        body = _validated(GrantInput, body)
        if body.scope.campaign_id != campaign_id:
            raise CampaignAgentAuthorizationError("campaign route scope mismatch")
        service, _auth = _service(request)
        receipt = service.issue_grant(_human_principal(request), body.contract())
        return grant_confirmation_to_wire(receipt)
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_router.post("/{campaign_id}/agent-attachment")
def attach_agent(
    campaign_id: str,
    body: Annotated[Any, Body()],
    request: Request,
    response: Response,
):
    try:
        body = _validated(AttachInput, body)
        if body.scope.campaign_id != campaign_id:
            raise CampaignAgentAuthorizationError("campaign route scope mismatch")
        service, _auth = _service(request)
        view, replayed = service.attach(_human_principal(request), body.contract())
        response.headers["X-BashGym-Replayed"] = "true" if replayed else "false"
        return view
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_router.post("/{campaign_id}/agent-attachment/{attachment_id}/revoke")
def revoke_agent(
    campaign_id: str,
    attachment_id: str,
    body: Annotated[Any, Body()],
    request: Request,
    response: Response,
):
    try:
        body = _validated(RevokeInput, body)
        if body.scope.campaign_id != campaign_id or body.attachment_id != attachment_id:
            raise CampaignAgentAuthorizationError("attachment route scope mismatch")
        principal = _human_principal(request)
        if body.actor_id != principal.actor_id:
            raise CampaignAgentAuthorizationError("client actor provenance is not authority")
        service, _auth = _service(request)
        view, replayed = service.revoke(principal, body.contract())
        response.headers["X-BashGym-Replayed"] = "true" if replayed else "false"
        return view
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_router.get("/{campaign_id}/agent-attachment")
def get_agent_attachment(
    campaign_id: str,
    request: Request,
    workspace_id: IdentifierField = Query(),
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
):
    try:
        service, _auth = _service(request)
        return service.public_view(
            _human_principal(request),
            CampaignAgentPublicViewQuery(
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                after_sequence=after_sequence,
                limit=limit,
            ),
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_router.get("/{campaign_id}/agent-attachment/audit")
def get_agent_attachment_audit(
    campaign_id: str,
    request: Request,
    workspace_id: IdentifierField = Query(),
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
):
    try:
        service, _auth = _service(request)
        scope = CampaignAgentScope(workspace_id=workspace_id, campaign_id=campaign_id)
        return service.audit_page(
            _human_principal(request),
            scope,
            after_sequence=after_sequence,
            limit=limit,
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_router.get("/{campaign_id}/agent-attachment/receipts")
def get_agent_attachment_receipts(
    campaign_id: str,
    request: Request,
    workspace_id: IdentifierField = Query(),
    after_version: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
):
    try:
        service, _auth = _service(request)
        scope = CampaignAgentScope(workspace_id=workspace_id, campaign_id=campaign_id)
        return service.receipt_page(
            _human_principal(request),
            scope,
            after_version=after_version,
            limit=limit,
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_credential_router.post("/heartbeat")
def campaign_agent_heartbeat(body: Annotated[Any, Body()], request: Request):
    try:
        body = _validated(HeartbeatInput, body)
        service, _auth = _service(request)
        return service.heartbeat(_bearer(request), body.contract())
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_credential_router.get("/actions/observe")
def observe_campaign_as_agent(request: Request):
    """Return a bounded status projection for the bearer-bound campaign only."""

    try:
        service, _auth = _service(request)
        authorization = service.authorize_bearer_action(
            _bearer(request),
            required_capability=CampaignAgentCapability.CAMPAIGN_OBSERVE,
        )
        scope = authorization.scope
        state = service.campaign_repository.read_control_room_snapshot(
            scope.workspace_id, scope.campaign_id
        )
        authorization.require_scope(state.campaign.workspace_id, state.campaign.campaign_id)
        return {
            "schemaVersion": "campaign_agent_observation.v1",
            "scope": {
                "workspaceId": scope.workspace_id,
                "campaignId": scope.campaign_id,
            },
            "campaign": {
                "status": state.campaign.status.value,
                "version": state.campaign.version,
                "manifestRevision": state.campaign.manifest_revision,
                "activeStudyId": state.campaign.active_study_id,
                "activeActionId": state.campaign.active_action_id,
                "latestEventCursor": state.latest_event_cursor,
            },
            "agent": {
                "attachmentId": authorization.attachment_id,
                "attachmentVersion": authorization.attachment_version,
                "agentFamily": authorization.agent_family.value,
                "agentPrincipalId": authorization.principal.actor_id,
                "authorizedCapability": authorization.required_capability.value,
            },
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_credential_router.get("/actions/artifacts")
def list_campaign_artifacts_as_agent(
    request: Request,
    after_cursor: str | None = Query(
        default=None,
        min_length=14,
        max_length=14,
        pattern=r"^a1\.[A-Za-z0-9_-]{11}$",
    ),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Read one bounded, URI-free artifact page for the bearer-bound campaign."""

    try:
        service, _auth = _service(request)
        authorization = service.authorize_bearer_action(
            _bearer(request),
            required_capability=CampaignAgentCapability.ARTIFACT_READ,
        )
        scope = authorization.scope
        artifacts, next_cursor, has_more = CampaignService(service.campaign_repository).artifacts(
            scope.workspace_id,
            scope.campaign_id,
            authorization.principal,
            after_cursor=after_cursor,
            limit=limit,
        )
        return {
            "schemaVersion": "campaign_agent_artifact_page.v1",
            "scope": {
                "workspaceId": scope.workspace_id,
                "campaignId": scope.campaign_id,
            },
            "items": [
                {
                    "artifactId": artifact.artifact_id,
                    "producerActionId": artifact.producer_action_id,
                    "sha256": artifact.sha256,
                    "sizeBytes": artifact.size_bytes,
                    "schemaName": artifact.schema_name,
                    "sealed": artifact.sealed,
                    "valid": artifact.valid,
                    "createdAt": artifact.created_at.astimezone(UTC)
                    .replace(microsecond=0)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                for artifact in artifacts
            ],
            "nextCursor": next_cursor,
            "hasMore": has_more,
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_credential_router.post("/sessions", status_code=201)
def register_campaign_agent_host_session(body: Annotated[Any, Body()], request: Request):
    try:
        registration = _validated(HostSessionRegistrationInput, body).contract()
        principal = _human_principal(request)
        return _session_receipt_wire(_session_service(request).register(principal, registration))
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_credential_router.post("/sessions/{registration_id}/deliveries/claim")
def claim_campaign_agent_delivery(registration_id: IdentifierField, request: Request):
    try:
        return _delivery_wire(
            _session_service(request).claim(_human_principal(request), registration_id)
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_agent_credential_router.post("/sessions/{registration_id}/revoke")
def revoke_campaign_agent_host_session(registration_id: IdentifierField, request: Request):
    try:
        return _session_receipt_wire(
            _session_service(request).revoke(_human_principal(request), registration_id)
        )
    except Exception as exc:
        _raise_api(exc)


__all__ = [
    "activate_managed_desktop_campaign_agent_bindings",
    "campaign_agent_credential_router",
    "campaign_agent_router",
]
