"""Authenticated REST boundary for durable campaign recovery authority."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Never

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthenticationError, CampaignAuthService
from bashgym.campaigns.campaign_recovery import (
    CampaignRecoveryConflictError,
    CampaignRecoveryError,
    CampaignRecoveryNotFoundError,
    CampaignRecoveryRepository,
    RecoveryAction,
    RecoveryRequest,
)
from bashgym.campaigns.contracts import AutonomyProfile, Capability
from bashgym.campaigns.human_oversight import HumanOversightIntegrityError
from bashgym.campaigns.runtime import CampaignRuntimeRepository

campaign_recovery_router = APIRouter(prefix="/api/campaigns", tags=["campaign-recovery"])

IdentifierField = Annotated[
    str,
    Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$",
    ),
]


def _recovery_sealer(request: Request) -> ArtifactSealer:
    from bashgym.api.campaign_routes import _campaign_authority_sealer

    try:
        return _campaign_authority_sealer(request)
    except (HumanOversightIntegrityError, ValueError) as exc:
        raise CampaignRecoveryError("recovery receipt seal authority is unavailable") from exc


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(value.title() for value in tail)


class RecoveryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", alias_generator=_camel, populate_by_name=True)

    action: RecoveryAction
    idempotency_key: str = Field(pattern=r"^idem_[0-9a-f]{32}$")
    workspace_id: IdentifierField
    campaign_id: IdentifierField
    eligibility_receipt_id: str = Field(pattern=r"^rcpt_[0-9a-f]{32}$")
    doctor_evidence_id: str = Field(pattern=r"^evd_[0-9a-f]{32}$")
    expected_campaign_revision: int = Field(ge=1)
    expected_event_cursor: int = Field(ge=0)
    expected_aggregate_version: int = Field(ge=1)
    expected_controller_lease_id: str | None = Field(default=None, pattern=r"^lease_[0-9a-f]{32}$")
    checkpoint_id: str = Field(pattern=r"^ckpt_[0-9a-f]{32}$")
    artifact_id: str = Field(pattern=r"^art_[0-9a-f]{32}$")
    human_confirmed: Literal[True]

    def contract(self) -> RecoveryRequest:
        # The authority contract uses Literal[True], so false never reaches the repository.
        return RecoveryRequest.model_validate(self.model_dump())


def _bearer(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token.strip():
        raise CampaignAuthenticationError()
    return token.strip()


def _services(
    request: Request,
) -> tuple[CampaignRuntimeRepository, CampaignAuthService, CampaignRecoveryRepository]:
    campaigns = getattr(request.app.state, "campaign_repository", None)
    auth = getattr(request.app.state, "campaign_auth_service", None)
    if not isinstance(campaigns, CampaignRuntimeRepository) or not isinstance(
        auth, CampaignAuthService
    ):
        from bashgym.api.campaign_routes import _services as campaign_services

        campaigns, auth, _campaign_service = campaign_services(request)
    recovery = getattr(request.app.state, "campaign_recovery_repository", None)
    if not isinstance(recovery, CampaignRecoveryRepository):
        recovery = CampaignRecoveryRepository(
            campaigns.db_path,
            sealer=_recovery_sealer(request),
        )
        recovery.initialize()
        request.app.state.campaign_recovery_repository = recovery
    return campaigns, auth, recovery


def _principal(request: Request):
    _campaigns, auth, _recovery = _services(request)
    return auth.authenticate_access(_bearer(request))


def _raise_api(exc: Exception) -> Never:
    if isinstance(exc, CampaignAuthenticationError):
        raise HTTPException(
            status_code=401,
            detail={"code": "campaign_auth_required", "message": "Authentication required."},
        ) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "campaign_recovery_authorization_denied",
                "message": "Campaign recovery is not permitted for this principal.",
            },
        ) from exc
    if isinstance(exc, CampaignRecoveryNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={
                "code": exc.code,
                "message": "Campaign recovery record was not found.",
            },
        ) from exc
    if isinstance(exc, CampaignRecoveryConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": "Recovery authority changed. Reconcile before retrying.",
            },
        ) from exc
    if isinstance(exc, CampaignRecoveryError):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "campaign_recovery_request_invalid",
                "message": "Campaign recovery request is invalid.",
            },
        ) from exc
    raise exc


def _validated(body: Any) -> RecoveryInput:
    try:
        if not isinstance(body, dict):
            raise ValueError("recovery body must be an object")
        return RecoveryInput.model_validate(body)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "campaign_recovery_request_invalid",
                "message": "Campaign recovery request validation failed.",
            },
        ) from exc


def _route_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", value):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "campaign_recovery_request_invalid",
                "message": "Campaign recovery route validation failed.",
            },
        )
    return value


def _route_action(value: str) -> RecoveryAction:
    try:
        return RecoveryAction(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "campaign_recovery_request_invalid",
                "message": "Campaign recovery route validation failed.",
            },
        ) from exc


@campaign_recovery_router.get("/{campaign_id}/recovery")
def get_campaign_recovery(
    campaign_id: str,
    request: Request,
    workspace_id: str = Query(alias="workspaceId"),
):
    try:
        campaign_id = _route_identifier(campaign_id)
        workspace_id = _route_identifier(workspace_id)
        principal = _principal(request)
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        _campaigns, _auth, recovery = _services(request)
        return recovery.project(workspace_id, campaign_id)
    except Exception as exc:
        _raise_api(exc)


@campaign_recovery_router.post("/{campaign_id}/recovery/{action}")
def request_campaign_recovery(
    campaign_id: str,
    action: str,
    body: Annotated[Any, Body()],
    request: Request,
    response: Response,
):
    try:
        campaign_id = _route_identifier(campaign_id)
        action_contract = _route_action(action)
        value = _validated(body)
        if value.campaign_id != campaign_id or value.action != action_contract:
            raise PermissionError("recovery route scope mismatch")
        principal = _principal(request)
        principal.require(value.workspace_id, Capability.CAMPAIGN_RESUME)
        if principal.autonomy_profile != AutonomyProfile.DESKTOP_USER:
            raise PermissionError("human confirmation requires the desktop user principal")
        _campaigns, _auth, recovery = _services(request)
        outcome, replayed = recovery.request(value.contract(), actor_id=principal.actor_id)
        response.headers["X-BashGym-Replayed"] = "true" if replayed else "false"
        return outcome
    except Exception as exc:
        _raise_api(exc)


__all__ = ["campaign_recovery_router"]
