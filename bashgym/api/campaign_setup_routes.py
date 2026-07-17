"""Authenticated API for the portable AutoResearch guided-setup flow."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Never, TypeVar

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bashgym.api.campaign_routes import (
    CampaignTemplateCreateInput,
    _autoresearch_definitions,
    _autoresearch_doctor_report,
    _principal,
    _raise_api,
    _services,
    _templates,
    create_campaign_from_template,
)
from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
from bashgym.campaigns.contracts import Campaign, Capability, ManifestRevision
from bashgym.campaigns.guided_setup import (
    GUIDED_SETUP_MAX_TEMPLATES,
    GuidedSetupBindings,
    GuidedSetupConflictError,
    GuidedSetupDraft,
    GuidedSetupError,
    GuidedSetupRepository,
)

campaign_setup_router = APIRouter(prefix="/api/campaigns/setup", tags=["campaign-setup"])

_MAX_SETUP_BODY_BYTES = 12_000
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class SetupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuidedSetupDraftInput(SetupModel):
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

    def contract(self) -> GuidedSetupDraft:
        return GuidedSetupDraft.model_validate(self.model_dump(mode="json"))


class GuidedSetupCreateInput(SetupModel):
    workspace_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    campaign_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    title: str = Field(min_length=1, max_length=240)
    validation_receipt_id: str = Field(pattern=r"^setuprcpt_[0-9a-f]{32}$")


class GuidedSetupSessionInput(SetupModel):
    workspace_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    session_id: str = Field(pattern=r"^setupsess_[0-9a-f]{32}$")
    expected_version: int = Field(ge=0, le=6)
    step: Literal["template", "installation", "model", "data", "compute", "evaluation"]
    selection_id: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )


def _setup_repository(request: Request) -> GuidedSetupRepository:
    campaigns, _auth, _service = _services(request)
    # Guided setup and recovery share the established external, versioned
    # campaign authority. The secret is never persisted in the campaign DB.
    from bashgym.api.campaign_routes import _campaign_authority_sealer

    authority = _campaign_authority_sealer(request)
    recovery = getattr(request.app.state, "campaign_recovery_repository", None)
    if not isinstance(recovery, CampaignRecoveryRepository):
        recovery = CampaignRecoveryRepository(campaigns.db_path, sealer=authority)
        recovery.initialize()
        request.app.state.campaign_recovery_repository = recovery
    setup = getattr(request.app.state, "campaign_guided_setup_repository", None)
    if (
        not isinstance(setup, GuidedSetupRepository)
        or setup.sealer is None
        or setup.sealer.key_version != authority.key_version
    ):
        setup = GuidedSetupRepository(campaigns.db_path, sealer=authority)
        setup.initialize()
        request.app.state.campaign_guided_setup_repository = setup
    return setup


def _definition(request: Request, template_id: str):
    try:
        return _autoresearch_definitions(request)[template_id]
    except KeyError as exc:
        raise GuidedSetupError("guided setup template was not found") from exc


def _raise_setup_api(exc: Exception) -> Never:
    if isinstance(exc, GuidedSetupConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "campaign_guided_setup_conflict",
                "message": "Guided setup authority changed. Run doctor and validate again.",
            },
        ) from exc
    if isinstance(exc, GuidedSetupError):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "campaign_guided_setup_invalid",
                "message": "Guided setup request is invalid.",
            },
        ) from exc
    _raise_api(exc)


def _safe_invalid() -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "code": "campaign_guided_setup_invalid",
            "message": "Guided setup request is invalid.",
        },
    )


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise _safe_invalid()
    return value


def _header(request: Request, name: str, *, default: str | None = None) -> str:
    value = request.headers.get(name, default)
    if value is None:
        raise _safe_invalid()
    return _identifier(value)


def _query_identifier(request: Request, name: str) -> str:
    values = request.query_params.getlist(name)
    if len(values) != 1:
        raise _safe_invalid()
    return _identifier(values[0])


def _optional_query_identifier(request: Request, name: str) -> str | None:
    values = request.query_params.getlist(name)
    if not values:
        return None
    if len(values) != 1:
        raise _safe_invalid()
    return _identifier(values[0])


async def _body_model(request: Request, model: type[_ModelT]) -> _ModelT:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_SETUP_BODY_BYTES:
                raise _safe_invalid()
        except ValueError as exc:
            raise _safe_invalid() from exc
    raw = await request.body()
    if not raw or len(raw) > _MAX_SETUP_BODY_BYTES:
        raise _safe_invalid()
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("setup body must be an object")
        return model.model_validate(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise _safe_invalid() from exc


@campaign_setup_router.get("/templates")
def list_guided_setup_templates(request: Request):
    try:
        workspace_id = _query_identifier(request, "workspace_id")
        principal = _principal(request)
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        templates, truncated = GuidedSetupRepository.bounded_template_summaries(
            _autoresearch_definitions(request)
        )
        return {
            "schema_version": "guided_setup_templates.v1",
            "templates": templates,
            "truncation": {
                "truncated": truncated,
                "reason_codes": ["templates_truncated"] if truncated else [],
                "limit": GUIDED_SETUP_MAX_TEMPLATES,
            },
        }
    except Exception as exc:
        _raise_setup_api(exc)


@campaign_setup_router.get("/context")
def get_guided_setup_context(request: Request):
    """Discover public logical bindings and optionally resume a sealed session."""

    try:
        workspace_id = _query_identifier(request, "workspace_id")
        session_id = _optional_query_identifier(request, "session_id")
        campaigns, _auth, _service = _services(request)
        principal = _principal(request)
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        authority = None
        if session_id is not None:
            from bashgym.api.campaign_routes import _campaign_authority_sealer

            authority = _campaign_authority_sealer(request)
        setup = GuidedSetupRepository.open_binding_registry(campaigns.db_path, sealer=authority)
        return setup.context(
            workspace_id=workspace_id,
            actor_id=principal.actor_id,
            session_id=session_id,
            definitions=_autoresearch_definitions(request),
        )
    except Exception as exc:
        _raise_setup_api(exc)


@campaign_setup_router.post("/session")
async def advance_guided_setup_session(request: Request, response: Response):
    """Persist one ordered logical selection as a sealed resumable receipt."""

    try:
        body = await _body_model(request, GuidedSetupSessionInput)
        idempotency_key = _header(request, "Idempotency-Key")
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.CAMPAIGN_CREATE_FROM_TEMPLATE)
        result, replayed = _setup_repository(request).advance_session(
            workspace_id=body.workspace_id,
            actor_id=principal.actor_id,
            session_id=body.session_id,
            expected_version=body.expected_version,
            step=body.step,
            selection_id=body.selection_id,
            definitions=_autoresearch_definitions(request),
            idempotency_key=idempotency_key,
        )
        response.headers["X-BashGym-Replayed"] = "true" if replayed else "false"
        return result
    except Exception as exc:
        _raise_setup_api(exc)


@campaign_setup_router.post("/doctor")
async def doctor_guided_setup(request: Request):
    """Run setup readiness without mutating receipts, bindings, or campaign state."""

    try:
        body = await _body_model(request, GuidedSetupDraftInput)
        campaigns, _auth, _service = _services(request)
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.CAMPAIGN_READ)
        definition = _definition(request, body.template_id)
        doctor = _autoresearch_doctor_report(request, campaigns, definition, body.workspace_id)
        # Doctor is a read-only projection: it must not create setup tables,
        # registration rows, or sealing keys as a side effect.
        setup = getattr(request.app.state, "campaign_guided_setup_repository", None)
        if not isinstance(setup, GuidedSetupRepository):
            setup = GuidedSetupRepository.open_binding_registry(campaigns.db_path)
        return setup.doctor(body.contract(), definition=definition, doctor=doctor)
    except Exception as exc:
        _raise_setup_api(exc)


@campaign_setup_router.post("/validate")
async def validate_guided_setup(request: Request, response: Response):
    try:
        body = await _body_model(request, GuidedSetupDraftInput)
        idempotency_key = _header(request, "Idempotency-Key")
        campaigns, _auth, _service = _services(request)
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.CAMPAIGN_CREATE_FROM_TEMPLATE)
        definition = _definition(request, body.template_id)
        doctor = _autoresearch_doctor_report(request, campaigns, definition, body.workspace_id)
        result, replayed = _setup_repository(request).validate(
            body.contract(),
            definition=definition,
            doctor=doctor,
            actor_id=principal.actor_id,
            idempotency_key=idempotency_key,
        )
        response.headers["X-BashGym-Replayed"] = "true" if replayed else "false"
        return result
    except Exception as exc:
        _raise_setup_api(exc)


@campaign_setup_router.post("/create")
async def create_guided_setup_campaign(request: Request):
    try:
        body = await _body_model(request, GuidedSetupCreateInput)
        idempotency_key = _header(request, "Idempotency-Key")
        correlation_id = _header(request, "X-Correlation-ID", default="campaign-guided-setup")
        campaigns, _auth, _service = _services(request)
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.CAMPAIGN_CREATE_FROM_TEMPLATE)
        setup = _setup_repository(request)
        template_id = setup.receipt_template_id(
            body.validation_receipt_id,
            workspace_id=body.workspace_id,
            actor_id=principal.actor_id,
        )
        definition = _definition(request, template_id)
        template = _templates(request)[template_id]
        campaign = Campaign(
            campaign_id=body.campaign_id,
            workspace_id=body.workspace_id,
            title=body.title,
            kind=template.kind,
            objective=template.objective,
            target_model=template.target_model,
            owner_actor_id=principal.actor_id,
        )
        manifest_revision = ManifestRevision(
            workspace_id=body.workspace_id,
            campaign_id=body.campaign_id,
            revision=1,
            manifest=template.manifest,
            actor_id=principal.actor_id,
            correlation_id=correlation_id,
        )
        receipt = setup.create_campaign_atomically(
            body.validation_receipt_id,
            repository=campaigns,
            campaign=campaign,
            manifest_revision=manifest_revision,
            definition=definition,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )
        payload = create_campaign_from_template(
            CampaignTemplateCreateInput(
                workspace_id=body.workspace_id,
                campaign_id=body.campaign_id,
                title=body.title,
                template_id=receipt.template_id,
            ),
            request,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
        )
        setup.record_creation(body.validation_receipt_id, body.workspace_id, body.campaign_id)
        payload["setup"] = {
            "schema_version": "guided_setup_creation.v1",
            "validation_receipt_id": body.validation_receipt_id,
            "binding_references": receipt.bindings.model_dump(mode="json"),
        }
        return payload
    except Exception as exc:
        _raise_setup_api(exc)


__all__ = ["campaign_setup_router"]
