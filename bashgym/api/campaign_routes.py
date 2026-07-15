"""Explicitly authenticated REST surface for durable experiment campaigns."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Never

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bashgym.campaigns.auth import CampaignAuthenticationError, CampaignAuthService
from bashgym.campaigns.autoresearch import (
    AutoResearchBudgetError,
    AutoResearchCampaignCore,
    AutoResearchConflictError,
    AutoResearchInvariantError,
    AutoResearchRepository,
    AutoResearchResult,
    AutoResearchTemplateDefinition,
    autoresearch_spec_for_template,
    build_autoresearch_template_registry,
    builtin_autoresearch_template_definitions,
    load_autoresearch_template_definitions,
)
from bashgym.campaigns.contracts import (
    ActorPrincipal,
    Campaign,
    CampaignKind,
    CampaignManifest,
    CampaignStatus,
    CampaignTrigger,
    Capability,
    ProtectedEvaluationResult,
    StagePlan,
    StudyProposalSubmission,
    TargetModelContract,
    canonical_hash,
)
from bashgym.campaigns.lineage import (
    ApprovedSourceRepositoryProfile,
    GitHypothesisLineageManager,
    GitLineageError,
)
from bashgym.campaigns.persistence import (
    BudgetExceededError,
    BudgetInvariantError,
    IdempotencyConflictError,
    InvalidProposalTransitionError,
    PromotionGateFailedError,
    ProtectedLeaseDeniedError,
    RecordAlreadyExistsError,
    RecordNotFoundError,
    RevisionConflictError,
)
from bashgym.campaigns.readiness import (
    AutoResearchDoctorReport,
    doctor_autoresearch_template,
)
from bashgym.campaigns.remote import ApprovedRemoteExecutorProfile, RemoteRunIdentity
from bashgym.campaigns.runtime import (
    ActionClaimConflictError,
    ActionIdentityMismatchError,
    CampaignRuntimeRepository,
)
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.transitions import InvalidCampaignTransitionError
from bashgym.campaigns.worker_service import (
    CONTROLLER_OFFLINE_GUIDANCE,
    DesktopWorkerStatusProjection,
    WorkerServiceError,
    load_approved_remote_profiles,
    load_approved_source_profiles,
    project_controller_status,
    read_worker_config,
)
from bashgym.config import get_bashgym_dir, get_settings
from bashgym.ledger.persistence import ExperimentLedgerRepository

campaign_router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])
campaign_auth_router = APIRouter(prefix="/api/campaign-auth", tags=["campaign-auth"])


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CampaignCreateInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=160)
    campaign_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    kind: CampaignKind
    objective: str = Field(min_length=1, max_length=4000)
    target_model: TargetModelContract
    manifest: CampaignManifest


class CampaignTemplate(ApiModel):
    kind: CampaignKind
    objective: str = Field(min_length=1, max_length=4000)
    target_model: TargetModelContract
    manifest: CampaignManifest


class CampaignTemplateCreateInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=160)
    campaign_id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    template_id: str = Field(min_length=1, max_length=160)


class CampaignTransitionInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=160)
    expected_version: int = Field(ge=1)
    stop_reason: str | None = Field(default=None, max_length=2000)


class CampaignExpectedVersionInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=160)
    expected_version: int = Field(ge=1)


class CampaignProposalCreateInput(CampaignExpectedVersionInput):
    proposal_id: str = Field(min_length=1, max_length=160)
    hypothesis: str = Field(min_length=1, max_length=4000)
    evidence_references: tuple[str, ...] = ()
    study_family: str = Field(min_length=1, max_length=160)
    primary_variable: str = Field(min_length=1, max_length=1000)
    controlled_variables: tuple[str, ...] = ()
    expected_outcome: str = Field(min_length=1, max_length=2000)
    falsification_criterion: str = Field(min_length=1, max_length=2000)
    estimated_cost: float = Field(ge=0)
    priority: int = Field(default=50, ge=0, le=100)
    prerequisite_study_ids: tuple[str, ...] = ()
    dataset_recipe: dict[str, Any]
    training_recipe: dict[str, Any]
    evaluation_recipe: dict[str, Any]
    required_capabilities: frozenset[Capability] = frozenset()
    stage_plan: StagePlan
    rationale: str = Field(min_length=1, max_length=4000)


class AutoResearchCandidateCreateInput(CampaignProposalCreateInput):
    parent_proposal_id: str = Field(min_length=1, max_length=160)


class AutoResearchResultInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=160)
    result: AutoResearchResult


class AutoResearchEvaluationIngestInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=160)
    project_id: str = Field(min_length=1, max_length=160)
    evaluation_result_id: str = Field(min_length=1, max_length=160)


class CodeLineageInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=160)


class CampaignManifestReviseInput(CampaignExpectedVersionInput):
    manifest: CampaignManifest
    reason: str = Field(min_length=1, max_length=2000)


class CampaignReasonInput(CampaignExpectedVersionInput):
    reason: str = Field(min_length=1, max_length=2000)


class CampaignBudgetAmendInput(CampaignReasonInput):
    resource: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    delta: float

    @field_validator("delta")
    @classmethod
    def _delta_nonzero(cls, value: float) -> float:
        if value == 0:
            raise ValueError("budget amendment delta cannot be zero")
        return value


class SourceApprovalEvidence(ApiModel):
    provenance: str = Field(min_length=1, max_length=2000)
    license: str = Field(min_length=1, max_length=240)
    privacy_review: str = Field(min_length=1, max_length=2000)
    contamination_review: str = Field(min_length=1, max_length=2000)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    notes: str | None = Field(default=None, max_length=2000)


class CampaignSourceApprovalInput(CampaignExpectedVersionInput):
    evidence: SourceApprovalEvidence


class CampaignForceStopInput(CampaignReasonInput):
    expected_remote_process_identity: RemoteRunIdentity
    confirmed: Literal[True]


class CampaignPromotionInput(CampaignExpectedVersionInput):
    override_reason: str | None = Field(default=None, min_length=1, max_length=2000)


class CampaignProtectedResultInput(CampaignExpectedVersionInput):
    result: ProtectedEvaluationResult


class CampaignExportInput(CampaignExpectedVersionInput):
    formats: tuple[Literal["markdown", "json", "csv", "png", "docx", "pdf"], ...]

    @field_validator("formats")
    @classmethod
    def _formats_unique(cls, value):
        if not value or len(set(value)) != len(value):
            raise ValueError("export formats must be non-empty and unique")
        return value


def _services(
    request: Request,
) -> tuple[CampaignRuntimeRepository, CampaignAuthService, CampaignService]:
    if not get_settings().campaigns_enabled:
        raise HTTPException(
            status_code=404,
            detail={"code": "campaigns_disabled", "message": "Campaigns are disabled."},
        )
    repository = getattr(request.app.state, "campaign_repository", None)
    if not isinstance(repository, CampaignRuntimeRepository):
        repository = AutoResearchRepository(
            get_bashgym_dir() / "campaigns" / "campaigns.sqlite3"
        )
        repository.initialize()
        request.app.state.campaign_repository = repository
    auth = getattr(request.app.state, "campaign_auth_service", None)
    if not isinstance(auth, CampaignAuthService):
        auth = CampaignAuthService(repository)
        request.app.state.campaign_auth_service = auth
    if not getattr(request.app.state, "campaign_desktop_bootstrap_checked", False):
        bootstrap = os.environ.get("BASHGYM_DESKTOP_BOOTSTRAP_SECRET", "")
        if bootstrap:
            auth.install_desktop_bootstrap(bootstrap)
        request.app.state.campaign_desktop_bootstrap_checked = True
    service = getattr(request.app.state, "campaign_service", None)
    if not isinstance(service, CampaignService):
        templates = _templates(request)
        service = CampaignService(
            repository,
            approved_template_hashes={
                template_id: canonical_hash(template.manifest.model_dump(mode="json"))
                for template_id, template in templates.items()
            },
        )
        request.app.state.campaign_service = service
    else:
        service.approved_template_hashes.update(
            {
                template_id: canonical_hash(template.manifest.model_dump(mode="json"))
                for template_id, template in _templates(request).items()
            }
        )
    return repository, auth, service


def _autoresearch_definitions(request: Request) -> dict[str, AutoResearchTemplateDefinition]:
    builtins = {
        definition.template_id: definition
        for definition in builtin_autoresearch_template_definitions()
    }
    installation_directory = getattr(
        request.app.state,
        "campaign_autoresearch_template_directory",
        get_bashgym_dir() / "campaigns" / "autoresearch-templates",
    )
    installed: dict[str, AutoResearchTemplateDefinition] = {}
    installation_path = os.fspath(installation_directory)
    if os.path.exists(installation_path):
        if not os.path.isdir(installation_path):
            raise ValueError("AutoResearch installation template path must be a directory")
        installed = {
            definition.template_id: definition
            for definition in load_autoresearch_template_definitions(
                Path(installation_path)
            )
        }
    raw = getattr(request.app.state, "campaign_autoresearch_templates", {})
    configured = {
        str(template_id): (
            definition
            if isinstance(definition, AutoResearchTemplateDefinition)
            else AutoResearchTemplateDefinition.model_validate(definition)
        )
        for template_id, definition in dict(raw).items()
    }
    for template_id, definition in {**installed, **configured}.items():
        if template_id != definition.template_id:
            raise ValueError("configured AutoResearch template identity mismatch")
        if template_id in builtins and definition != builtins[template_id]:
            raise ValueError(f"built-in AutoResearch template is reserved: {template_id}")
        if (
            template_id in installed
            and template_id in configured
            and installed[template_id] != configured[template_id]
        ):
            raise ValueError("installed AutoResearch template identity conflict")
    return {**builtins, **installed, **configured}


def _templates(request: Request) -> dict[str, CampaignTemplate]:
    autoresearch = {
        template_id: CampaignTemplate.model_validate(payload)
        for template_id, payload in build_autoresearch_template_registry(
            _autoresearch_definitions(request).values()
        ).items()
    }
    raw = getattr(request.app.state, "campaign_templates", {})
    configured = {
        str(template_id): (
            template
            if isinstance(template, CampaignTemplate)
            else CampaignTemplate.model_validate(template)
        )
        for template_id, template in dict(raw).items()
    }
    for template_id, template in configured.items():
        if template_id in autoresearch and template != autoresearch[template_id]:
            raise ValueError(f"AutoResearch campaign template is reserved: {template_id}")
    return {**autoresearch, **configured}


def _approved_executor_profiles(
    request: Request,
) -> dict[tuple[str, str], ApprovedRemoteExecutorProfile]:
    raw = getattr(request.app.state, "campaign_executor_profiles", None)
    if raw is None:
        config_path = Path(
            getattr(
                request.app.state,
                "campaign_worker_config_path",
                get_bashgym_dir() / "campaigns" / "worker-config.v1.json",
            )
        )
        if not config_path.is_file():
            return {}
        try:
            return load_approved_remote_profiles(read_worker_config(config_path))
        except (WorkerServiceError, ValueError):
            return {}
    values = raw.values() if isinstance(raw, Mapping) else raw
    profiles: dict[tuple[str, str], ApprovedRemoteExecutorProfile] = {}
    for value in values:
        profile = (
            value
            if isinstance(value, ApprovedRemoteExecutorProfile)
            else ApprovedRemoteExecutorProfile.model_validate(value)
        )
        key = (profile.compute_profile_id, profile.target_contract_key)
        if key in profiles:
            raise ValueError("duplicate campaign executor profile binding")
        profiles[key] = profile
    return profiles


def _approved_source_profiles(
    request: Request,
) -> dict[str, ApprovedSourceRepositoryProfile]:
    raw = getattr(request.app.state, "campaign_source_profiles", None)
    if raw is None:
        config_path = Path(
            getattr(
                request.app.state,
                "campaign_worker_config_path",
                get_bashgym_dir() / "campaigns" / "worker-config.v1.json",
            )
        )
        if not config_path.is_file():
            return {}
        try:
            return load_approved_source_profiles(read_worker_config(config_path))
        except (WorkerServiceError, ValueError):
            return {}
    values = raw.values() if isinstance(raw, Mapping) else raw
    profiles: dict[str, ApprovedSourceRepositoryProfile] = {}
    for value in values:
        profile = (
            value
            if isinstance(value, ApprovedSourceRepositoryProfile)
            else ApprovedSourceRepositoryProfile.model_validate(value)
        )
        if profile.profile_id in profiles:
            raise ValueError("duplicate campaign source profile binding")
        profiles[profile.profile_id] = profile
    return profiles


def _lineage_manager(request: Request) -> GitHypothesisLineageManager:
    root = Path(
        getattr(
            request.app.state,
            "campaign_lineage_worktree_root",
            get_bashgym_dir() / "campaigns" / "source-worktrees",
        )
    )
    return GitHypothesisLineageManager(root)


def _autoresearch_doctor_report(
    request: Request,
    repository: CampaignRuntimeRepository,
    definition: AutoResearchTemplateDefinition,
    workspace_id: str,
) -> AutoResearchDoctorReport:
    ledger = ExperimentLedgerRepository(repository.db_path)
    ledger.initialize()
    return doctor_autoresearch_template(
        definition,
        workspace_id=workspace_id,
        ledger=ledger,
        executor_profiles=_approved_executor_profiles(request),
        controller=project_controller_status(repository, get_bashgym_dir()),
        source_profiles=_approved_source_profiles(request),
    )


def _autoresearch_core(repository: CampaignRuntimeRepository) -> AutoResearchCampaignCore:
    if isinstance(repository, AutoResearchRepository):
        autoresearch_repository = repository
    else:
        autoresearch_repository = AutoResearchRepository(repository.db_path)
        autoresearch_repository.initialize()
    return AutoResearchCampaignCore(autoresearch_repository)


def _proposal_submission(campaign_id: str, body: CampaignProposalCreateInput):
    return StudyProposalSubmission(
        proposal_id=body.proposal_id,
        workspace_id=body.workspace_id,
        campaign_id=campaign_id,
        hypothesis=body.hypothesis,
        evidence_references=body.evidence_references,
        study_family=body.study_family,
        primary_variable=body.primary_variable,
        controlled_variables=body.controlled_variables,
        expected_outcome=body.expected_outcome,
        falsification_criterion=body.falsification_criterion,
        estimated_cost=body.estimated_cost,
        priority=body.priority,
        prerequisite_study_ids=body.prerequisite_study_ids,
        dataset_recipe=body.dataset_recipe,
        training_recipe=body.training_recipe,
        evaluation_recipe=body.evaluation_recipe,
        required_capabilities=body.required_capabilities,
        stage_plan=body.stage_plan,
        rationale=body.rationale,
    )


def _bearer(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token.strip():
        raise CampaignAuthenticationError()
    return token.strip()


def _principal(request: Request) -> ActorPrincipal:
    _repository, auth, _service = _services(request)
    return auth.authenticate_access(_bearer(request))


def _desktop_worker_status(
    request: Request,
    repository: CampaignRuntimeRepository,
) -> DesktopWorkerStatusProjection:
    """Return authenticated, secret-free worker readiness for the renderer."""

    supervisor = getattr(request.app.state, "campaign_worker_supervisor", None)
    if supervisor is not None:
        try:
            return supervisor.status(repository=repository)
        except Exception:
            failure_code = "campaign_worker_status_unavailable"
            managed = True
    else:
        failure_code = getattr(
            request.app.state,
            "campaign_worker_bootstrap_failure_code",
            None,
        )
        managed = bool(getattr(request.app.state, "campaign_worker_managed", False))
    observed_at = datetime.now(UTC)
    controller = project_controller_status(
        repository,
        get_bashgym_dir(),
        now=observed_at,
    )
    return DesktopWorkerStatusProjection(
        managed=managed,
        online=False,
        state="failed" if failure_code else "offline",
        code=failure_code or ("worker_offline" if managed else "worker_not_managed"),
        observed_at=observed_at,
        thread_alive=False,
        controller=controller,
        guidance=CONTROLLER_OFFLINE_GUIDANCE,
    )


def _raise_api(exc: Exception) -> Never:
    if isinstance(exc, CampaignAuthenticationError):
        raise HTTPException(
            status_code=401,
            detail={"code": exc.code, "message": "Campaign authentication required."},
        ) from exc
    if isinstance(exc, PermissionError):
        code = (
            "campaign_scope_denied"
            if "workspace" in str(exc) or "scope" in str(exc)
            else "campaign_capability_required"
        )
        raise HTTPException(
            status_code=403,
            detail={"code": code, "message": "Campaign operation is not permitted."},
        ) from exc
    if isinstance(exc, GitLineageError):
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": "Git lineage policy rejected the operation."},
        ) from exc
    if isinstance(exc, RecordNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": "campaign_not_found", "message": "Campaign record not found."},
        ) from exc
    if isinstance(exc, RecordAlreadyExistsError):
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": "Campaign record already exists."},
        ) from exc
    if isinstance(exc, RevisionConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "campaign_version_conflict",
                "message": "Campaign version changed.",
                "expected": exc.expected,
                "current": exc.current,
            },
        ) from exc
    if isinstance(exc, IdempotencyConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "campaign_idempotency_conflict",
                "message": "Idempotency key was reused with a different request.",
            },
        ) from exc
    if isinstance(exc, (InvalidCampaignTransitionError, InvalidProposalTransitionError)):
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if isinstance(exc, (ActionClaimConflictError, ActionIdentityMismatchError)):
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": "Campaign action state changed."},
        ) from exc
    if isinstance(exc, AutoResearchConflictError):
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if isinstance(exc, (AutoResearchBudgetError, AutoResearchInvariantError)):
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if isinstance(
        exc,
        (
            BudgetExceededError,
            BudgetInvariantError,
            ProtectedLeaseDeniedError,
            PromotionGateFailedError,
        ),
    ):
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": "Campaign policy gate rejected the operation."},
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            status_code=422,
            detail={"code": "campaign_invalid", "message": str(exc)},
        ) from exc
    raise exc


def _mutation_payload(mutation) -> dict[str, Any]:
    return {
        "campaign": mutation.campaign.model_dump(mode="json"),
        "event": mutation.event.model_dump(mode="json"),
        "replayed": mutation.replayed,
    }


def _proposal_mutation_payload(mutation) -> dict[str, Any]:
    return {
        **_mutation_payload(mutation),
        "record": mutation.record.model_dump(mode="json"),
    }


def _operation_mutation_payload(mutation) -> dict[str, Any]:
    payload = _mutation_payload(mutation)
    if hasattr(mutation, "details"):
        payload["details"] = mutation.details
    if hasattr(mutation, "entry"):
        payload["entry"] = mutation.entry.model_dump(mode="json")
    return payload


@campaign_auth_router.post("/exchange")
def exchange_campaign_refresh(request: Request):
    try:
        _repository, auth, _service = _services(request)
        bearer = _bearer(request)
        credential = (
            auth.exchange_desktop_bootstrap(bearer)
            if bearer.startswith("bgcb.")
            else auth.exchange_refresh(bearer)
        )
        return credential.model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@campaign_auth_router.get("/capabilities")
def campaign_capabilities(request: Request):
    try:
        repository, auth, _service = _services(request)
        principal = auth.authenticate_access(_bearer(request))
        return {
            "actor_id": principal.actor_id,
            "autonomy_profile": principal.autonomy_profile.value,
            "workspace_ids": list(principal.workspace_ids),
            "capabilities": sorted(item.value for item in principal.capabilities),
            "expires_at": principal.expires_at.isoformat(),
            "worker": _desktop_worker_status(request, repository).model_dump(mode="json"),
        }
    except Exception as exc:
        _raise_api(exc)


def _campaign_ledger_projection(
    repository: CampaignRuntimeRepository,
    workspace_id: str,
    campaign_id: str,
) -> dict[str, Any]:
    """Return bounded experiment-ledger evidence linked to one durable campaign."""

    ledger = ExperimentLedgerRepository(repository.db_path)
    ledger.initialize()
    projects: list[dict[str, Any]] = []
    for project in ledger.list_projects(workspace_id):
        project_id = str(project["project_id"])
        experiments = [
            item
            for item in ledger.list_experiments(workspace_id, project_id)
            if item.get("campaign_id") == campaign_id
        ]
        runs = [
            item
            for item in ledger.list_runs(workspace_id, project_id, limit=500)
            if item.get("campaign_id") == campaign_id
        ]
        if not experiments and not runs:
            continue
        run_ids = {str(item["run_id"]) for item in runs}
        evaluations = [
            item
            for item in ledger.list_evaluation_results(workspace_id, project_id, limit=1000)
            if str(item.get("run_id")) in run_ids
        ]
        artifacts = [
            {key: value for key, value in item.items() if key != "uri"}
            for item in ledger.list_artifacts(workspace_id, project_id, limit=1000)
            if str(item.get("run_id")) in run_ids
        ]
        experiment_ids = {str(item["experiment_id"]) for item in experiments}
        decisions = [
            item
            for item in ledger.list_decisions(workspace_id, project_id, limit=1000)
            if str(item.get("experiment_id")) in experiment_ids
            or str(item.get("run_id")) in run_ids
        ]
        projects.append(
            {
                "project": project,
                "experiments": experiments[:20],
                "runs": runs[:50],
                "evaluations": evaluations[:100],
                "artifacts": artifacts[:100],
                "decisions": decisions[:50],
                "evidence": {
                    "experiment_ids": [item["experiment_id"] for item in experiments[:20]],
                    "run_ids": [item["run_id"] for item in runs[:50]],
                    "evaluation_result_ids": [
                        item["evaluation_result_id"] for item in evaluations[:100]
                    ],
                    "artifact_ids": [item["artifact_id"] for item in artifacts[:100]],
                    "decision_ids": [item["decision_id"] for item in decisions[:50]],
                },
            }
        )
    projection = {
        "schema_version": "bashgym.campaign-ledger-projection.v1",
        "workspace_id": workspace_id,
        "campaign_id": campaign_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "linked": bool(projects),
        "projects": projects,
    }
    try:
        core = _autoresearch_core(repository)
        spec = core.repository.get_autoresearch_spec(workspace_id, campaign_id)
        projection["autoresearch"] = {
            "spec": spec.model_dump(mode="json"),
            "state": core.state(workspace_id, campaign_id).model_dump(mode="json"),
            "proposals": [
                item.model_dump(mode="json")
                for item in core.repository.list_autoresearch_proposals(
                    workspace_id, campaign_id
                )
            ],
            "outcomes": [
                item.model_dump(mode="json")
                for item in core.repository.list_autoresearch_outcomes(
                    workspace_id, campaign_id
                )
            ],
            "code_lineages": [
                item.model_dump(mode="json")
                for item in core.repository.list_code_lineages(workspace_id, campaign_id)
            ],
        }
    except RecordNotFoundError:
        projection["autoresearch"] = None
    return projection


@campaign_router.post("")
def create_campaign(
    body: CampaignCreateInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-create", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        principal = _principal(request)
        campaign = Campaign(
            campaign_id=body.campaign_id,
            workspace_id=body.workspace_id,
            title=body.title,
            kind=body.kind,
            objective=body.objective,
            target_model=body.target_model,
            owner_actor_id=principal.actor_id,
        )
        return _mutation_payload(
            service.create(
                campaign,
                body.manifest,
                principal=principal,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/from-template")
def create_campaign_from_template(
    body: CampaignTemplateCreateInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-template", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        principal = _principal(request)
        try:
            template = _templates(request)[body.template_id]
        except KeyError as exc:
            raise ValueError("campaign_template_not_approved") from exc
        definition = _autoresearch_definitions(request).get(body.template_id)
        if definition is not None:
            readiness = _autoresearch_doctor_report(
                request, _repository, definition, body.workspace_id
            )
            if not readiness.materializable:
                raise AutoResearchInvariantError(
                    "autoresearch_installation_bindings_unresolved:"
                    + ",".join(readiness.blocking_codes)
                )
        campaign = Campaign(
            campaign_id=body.campaign_id,
            workspace_id=body.workspace_id,
            title=body.title,
            kind=template.kind,
            objective=template.objective,
            target_model=template.target_model,
            owner_actor_id=principal.actor_id,
        )
        mutation = service.create(
            campaign,
            template.manifest,
            principal=principal,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            approved_template_id=body.template_id,
        )
        payload = _mutation_payload(mutation)
        spec = autoresearch_spec_for_template(
            body.template_id,
            workspace_id=body.workspace_id,
            campaign_id=body.campaign_id,
            definitions=_autoresearch_definitions(request).values(),
        )
        if spec is not None:
            core = _autoresearch_core(_repository)
            core.register(spec)
            setup_key = canonical_hash(
                [body.workspace_id, body.campaign_id, body.template_id]
            )[:24]
            prepared = core.prepare(
                body.workspace_id,
                body.campaign_id,
                controller_id="autoresearch-controller",
                correlation_id=correlation_id,
                idempotency_prefix=f"autoresearch-prepare-{setup_key}",
            )
            payload["campaign"] = prepared.model_dump(mode="json")
            payload["autoresearch"] = core.state(
                body.workspace_id, body.campaign_id
            ).model_dump(mode="json")
        return payload
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("")
def list_campaigns(
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
    kind: CampaignKind | None = None,
    status: CampaignStatus | None = None,
):
    try:
        repository, _auth, service = _services(request)
        principal = _principal(request)
        return {
            "campaigns": [
                item.model_dump(mode="json")
                for item in service.list(workspace_id, principal, kind=kind, status=status)
            ],
            "controller": project_controller_status(
                repository, get_bashgym_dir()
            ).model_dump(mode="json"),
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/templates")
def list_campaign_templates(
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        principal = _principal(request)
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        return {
            "templates": [
                {"template_id": template_id, **template.model_dump(mode="json")}
                for template_id, template in sorted(_templates(request).items())
            ]
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/templates/{template_id}/doctor")
def doctor_campaign_template(
    template_id: str,
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        repository, _auth, _service = _services(request)
        principal = _principal(request)
        principal.require(workspace_id, Capability.CAMPAIGN_READ)
        try:
            definition = _autoresearch_definitions(request)[template_id]
        except KeyError as exc:
            raise RecordNotFoundError("AutoResearch template not found") from exc
        return _autoresearch_doctor_report(
            request, repository, definition, workspace_id
        ).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}")
def get_campaign(campaign_id: str, request: Request, workspace_id: str = Query(...)):
    try:
        _repository, _auth, service = _services(request)
        return service.get(workspace_id, campaign_id, _principal(request)).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/autoresearch")
def get_autoresearch_campaign(
    campaign_id: str,
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        repository, _auth, service = _services(request)
        service.get(workspace_id, campaign_id, _principal(request))
        core = _autoresearch_core(repository)
        return {
            "spec": core.repository.get_autoresearch_spec(
                workspace_id, campaign_id
            ).model_dump(mode="json"),
            "state": core.state(workspace_id, campaign_id).model_dump(mode="json"),
            "proposals": [
                item.model_dump(mode="json")
                for item in core.repository.list_autoresearch_proposals(
                    workspace_id, campaign_id
                )
            ],
            "outcomes": [
                item.model_dump(mode="json")
                for item in core.repository.list_autoresearch_outcomes(
                    workspace_id, campaign_id
                )
            ],
            "code_lineages": [
                item.model_dump(mode="json")
                for item in core.repository.list_code_lineages(workspace_id, campaign_id)
            ],
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/manifest/{revision}")
def get_campaign_manifest_revision(
    campaign_id: str,
    revision: int,
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        _repository, _auth, service = _services(request)
        return service.manifest(
            workspace_id, campaign_id, revision, _principal(request)
        ).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/manifest/revise")
def revise_campaign_manifest(
    campaign_id: str,
    body: CampaignManifestReviseInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-manifest-revise", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.revise_manifest(
                body.workspace_id,
                campaign_id,
                body.manifest,
                reason=body.reason,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/proposals")
def submit_campaign_proposal(
    campaign_id: str,
    body: CampaignProposalCreateInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-proposal-submit",
        alias="X-Correlation-ID",
        max_length=160,
    ),
):
    try:
        repository, _auth, service = _services(request)
        core = _autoresearch_core(repository)
        try:
            core.repository.get_autoresearch_spec(body.workspace_id, campaign_id)
        except RecordNotFoundError:
            pass
        else:
            raise AutoResearchInvariantError(
                "autoresearch_proposal_requires_explicit_baseline_or_candidate_role"
            )
        submission = _proposal_submission(campaign_id, body)
        return _proposal_mutation_payload(
            service.submit_proposal(
                submission,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/autoresearch/baseline")
def submit_autoresearch_baseline(
    campaign_id: str,
    body: CampaignProposalCreateInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-autoresearch-baseline",
        alias="X-Correlation-ID",
        max_length=160,
    ),
):
    try:
        repository, _auth, _service = _services(request)
        mutation = _autoresearch_core(repository).submit_baseline(
            _proposal_submission(campaign_id, body),
            expected_version=body.expected_version,
            principal=_principal(request),
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )
        return _proposal_mutation_payload(mutation)
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/autoresearch/candidates")
def submit_autoresearch_candidate(
    campaign_id: str,
    body: AutoResearchCandidateCreateInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-autoresearch-candidate",
        alias="X-Correlation-ID",
        max_length=160,
    ),
):
    try:
        repository, _auth, _service = _services(request)
        mutation = _autoresearch_core(repository).submit_controlled_candidate(
            _proposal_submission(campaign_id, body),
            parent_proposal_id=body.parent_proposal_id,
            changed_variable=body.primary_variable,
            expected_version=body.expected_version,
            principal=_principal(request),
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )
        return _proposal_mutation_payload(mutation)
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/autoresearch/results")
def record_autoresearch_result(
    campaign_id: str,
    body: AutoResearchResultInput,
    request: Request,
):
    try:
        repository, _auth, service = _services(request)
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.EXPERIMENT_LEDGER_WRITE)
        service.get(body.workspace_id, campaign_id, principal)
        if (
            body.result.workspace_id != body.workspace_id
            or body.result.campaign_id != campaign_id
        ):
            raise AutoResearchInvariantError("autoresearch_result_identity_mismatch")
        if body.result.provenance.value == "real":
            raise AutoResearchInvariantError(
                "autoresearch_real_result_requires_authoritative_evaluation"
            )
        return _autoresearch_core(repository).record_result(body.result).model_dump(
            mode="json"
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post(
    "/{campaign_id}/proposals/{proposal_id}/code-lineage/prepare"
)
def prepare_campaign_code_lineage(
    campaign_id: str,
    proposal_id: str,
    body: CodeLineageInput,
    request: Request,
):
    """Prepare a private worktree; only the authorized caller receives its path."""

    try:
        repository, _auth, service = _services(request)
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.EXPERIMENT_CODE_MUTATE)
        service.get(body.workspace_id, campaign_id, principal)
        record = repository.get_code_lineage(body.workspace_id, proposal_id)
        if record.campaign_id != campaign_id:
            raise RecordNotFoundError("campaign code lineage not found")
        profile = _approved_source_profiles(request).get(
            record.source_repository_profile_id
        )
        if profile is None:
            raise GitLineageError("campaign_git_lineage_source_profile_unavailable")
        receipt = _lineage_manager(request).prepare(profile, record)
        saved = repository.advance_code_lineage(receipt.record)
        return {
            "record": saved.model_dump(mode="json"),
            "worktree_path": str(receipt.worktree_path),
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post(
    "/{campaign_id}/proposals/{proposal_id}/code-lineage/capture"
)
def capture_campaign_code_lineage(
    campaign_id: str,
    proposal_id: str,
    body: CodeLineageInput,
    request: Request,
):
    """Capture exactly one approved commit without returning private path material."""

    try:
        repository, _auth, service = _services(request)
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.EXPERIMENT_CODE_MUTATE)
        service.get(body.workspace_id, campaign_id, principal)
        record = repository.get_code_lineage(body.workspace_id, proposal_id)
        if record.campaign_id != campaign_id:
            raise RecordNotFoundError("campaign code lineage not found")
        profile = _approved_source_profiles(request).get(
            record.source_repository_profile_id
        )
        if profile is None:
            raise GitLineageError("campaign_git_lineage_source_profile_unavailable")
        captured = _lineage_manager(request).capture(profile, record)
        return {"record": repository.advance_code_lineage(captured).model_dump(mode="json")}
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/autoresearch/ingest-evaluation")
def ingest_autoresearch_evaluation(
    campaign_id: str,
    body: AutoResearchEvaluationIngestInput,
    request: Request,
):
    try:
        repository, _auth, service = _services(request)
        principal = _principal(request)
        principal.require(body.workspace_id, Capability.EXPERIMENT_LEDGER_WRITE)
        service.get(body.workspace_id, campaign_id, principal)
        return _autoresearch_core(repository).ingest_evaluation_result(
            workspace_id=body.workspace_id,
            campaign_id=campaign_id,
            project_id=body.project_id,
            evaluation_result_id=body.evaluation_result_id,
        ).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/proposals")
def list_campaign_proposals(
    campaign_id: str,
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        _repository, _auth, service = _services(request)
        return {
            "proposals": [
                item.model_dump(mode="json")
                for item in service.proposals(workspace_id, campaign_id, _principal(request))
            ]
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/studies")
def list_campaign_studies(
    campaign_id: str,
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        _repository, _auth, service = _services(request)
        return {
            "studies": [
                item.model_dump(mode="json")
                for item in service.studies(
                    workspace_id, campaign_id, _principal(request)
                )
            ]
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/studies/{study_id}")
def get_campaign_study(
    campaign_id: str,
    study_id: str,
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        _repository, _auth, service = _services(request)
        return service.study(
            workspace_id, campaign_id, study_id, _principal(request)
        ).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/proposals/{proposal_id}/withdraw")
def withdraw_campaign_proposal(
    campaign_id: str,
    proposal_id: str,
    body: CampaignExpectedVersionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-proposal-withdraw",
        alias="X-Correlation-ID",
        max_length=160,
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _proposal_mutation_payload(
            service.withdraw_proposal(
                body.workspace_id,
                campaign_id,
                proposal_id,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/evidence")
def get_campaign_evidence(
    campaign_id: str,
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=160),
):
    try:
        _repository, _auth, service = _services(request)
        return service.evidence(workspace_id, campaign_id, _principal(request)).model_dump(
            mode="json"
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/advance")
def request_campaign_advance(
    campaign_id: str,
    body: CampaignExpectedVersionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-advance", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _mutation_payload(
            service.request_advance(
                body.workspace_id,
                campaign_id,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/actions/{action_id}/retry")
def retry_campaign_action(
    campaign_id: str,
    action_id: str,
    body: CampaignExpectedVersionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-action-retry", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.retry_action(
                body.workspace_id,
                campaign_id,
                action_id,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/studies/{study_id}/abandon")
def abandon_campaign_study(
    campaign_id: str,
    study_id: str,
    body: CampaignReasonInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-study-abandon", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.abandon_study(
                body.workspace_id,
                campaign_id,
                study_id,
                reason=body.reason,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/budget/amend")
def amend_campaign_budget(
    campaign_id: str,
    body: CampaignBudgetAmendInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-budget-amend", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.amend_budget(
                body.workspace_id,
                campaign_id,
                body.resource,
                body.delta,
                reason=body.reason,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/sources/{source_id}/approve")
def approve_campaign_source(
    campaign_id: str,
    source_id: str,
    body: CampaignSourceApprovalInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-source-approve", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.approve_source(
                body.workspace_id,
                campaign_id,
                source_id,
                body.evidence.model_dump(mode="json"),
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/actions/{action_id}/force-stop")
def force_stop_campaign_action(
    campaign_id: str,
    action_id: str,
    body: CampaignForceStopInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-force-stop", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.request_force_stop(
                body.workspace_id,
                campaign_id,
                action_id,
                body.expected_remote_process_identity,
                reason=body.reason,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/protected-lease")
def acquire_campaign_protected_lease(
    campaign_id: str,
    body: CampaignExpectedVersionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-protected-lease", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.acquire_protected_lease(
                body.workspace_id,
                campaign_id,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/protected-result")
def record_campaign_protected_result(
    campaign_id: str,
    body: CampaignProtectedResultInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-protected-result", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.record_protected_evaluation(
                body.workspace_id,
                campaign_id,
                body.result,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/promotion")
def promote_campaign_candidate(
    campaign_id: str,
    body: CampaignPromotionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-promotion", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.promote(
                body.workspace_id,
                campaign_id,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                override_reason=body.override_reason,
            )
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/export")
def export_campaign(
    campaign_id: str,
    body: CampaignExportInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-export", alias="X-Correlation-ID", max_length=160
    ),
):
    try:
        _repository, _auth, service = _services(request)
        return _operation_mutation_payload(
            service.export(
                body.workspace_id,
                campaign_id,
                body.formats,
                expected_version=body.expected_version,
                principal=_principal(request),
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
        )
    except Exception as exc:
        _raise_api(exc)


def _transition(
    campaign_id: str,
    trigger: CampaignTrigger,
    body: CampaignTransitionInput,
    request: Request,
    idempotency_key: str,
    correlation_id: str,
):
    _repository, _auth, service = _services(request)
    return _mutation_payload(
        service.transition(
            body.workspace_id,
            campaign_id,
            trigger,
            expected_version=body.expected_version,
            principal=_principal(request),
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            stop_reason=body.stop_reason,
        )
    )


def _transition_endpoint(
    campaign_id: str,
    trigger: CampaignTrigger,
    body: CampaignTransitionInput,
    request: Request,
    idempotency_key: str,
    correlation_id: str,
):
    try:
        return _transition(
            campaign_id,
            trigger,
            body,
            request,
            idempotency_key,
            correlation_id,
        )
    except Exception as exc:
        _raise_api(exc)


@campaign_router.post("/{campaign_id}/start")
def start_campaign(
    campaign_id: str,
    body: CampaignTransitionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-start", alias="X-Correlation-ID", max_length=160
    ),
):
    return _transition_endpoint(
        campaign_id,
        CampaignTrigger.START,
        body,
        request,
        idempotency_key,
        correlation_id,
    )


@campaign_router.post("/{campaign_id}/pause")
def pause_campaign(
    campaign_id: str,
    body: CampaignTransitionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-pause", alias="X-Correlation-ID", max_length=160
    ),
):
    return _transition_endpoint(
        campaign_id,
        CampaignTrigger.PAUSE,
        body,
        request,
        idempotency_key,
        correlation_id,
    )


@campaign_router.post("/{campaign_id}/resume")
def resume_campaign(
    campaign_id: str,
    body: CampaignTransitionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-resume", alias="X-Correlation-ID", max_length=160
    ),
):
    return _transition_endpoint(
        campaign_id,
        CampaignTrigger.RESUME,
        body,
        request,
        idempotency_key,
        correlation_id,
    )


@campaign_router.post("/{campaign_id}/cancel")
def cancel_campaign(
    campaign_id: str,
    body: CampaignTransitionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-cancel", alias="X-Correlation-ID", max_length=160
    ),
):
    return _transition_endpoint(
        campaign_id,
        CampaignTrigger.CANCEL,
        body,
        request,
        idempotency_key,
        correlation_id,
    )


@campaign_router.post("/{campaign_id}/conclude")
def conclude_campaign(
    campaign_id: str,
    body: CampaignTransitionInput,
    request: Request,
    idempotency_key: str = Header(..., alias="Idempotency-Key", max_length=160),
    correlation_id: str = Header(
        default="campaign-rest-conclude", alias="X-Correlation-ID", max_length=160
    ),
):
    return _transition_endpoint(
        campaign_id,
        CampaignTrigger.CONCLUDE,
        body,
        request,
        idempotency_key,
        correlation_id,
    )


@campaign_router.get("/{campaign_id}/events")
def campaign_events(
    campaign_id: str,
    request: Request,
    workspace_id: str = Query(...),
    after_cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
):
    try:
        _repository, _auth, service = _services(request)
        items = service.events(
            workspace_id,
            campaign_id,
            _principal(request),
            after_cursor=after_cursor,
            limit=limit,
        )
        return {
            "items": [
                {"cursor": cursor, "event": event.model_dump(mode="json")}
                for cursor, event in items
            ],
            "next_cursor": items[-1][0] if items else after_cursor,
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/artifacts")
def campaign_artifacts(campaign_id: str, request: Request, workspace_id: str = Query(...)):
    try:
        _repository, _auth, service = _services(request)
        return {
            "artifacts": [
                item.model_dump(mode="json", exclude={"uri"})
                for item in service.artifacts(workspace_id, campaign_id, _principal(request))
            ]
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/attempts")
def campaign_attempts(campaign_id: str, request: Request, workspace_id: str = Query(...)):
    try:
        _repository, _auth, service = _services(request)
        return {
            "attempts": [
                item.model_dump(mode="json", exclude={"sealed_result_uri"})
                for item in service.attempts(workspace_id, campaign_id, _principal(request))
            ]
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/comparisons")
def campaign_comparisons(campaign_id: str, request: Request, workspace_id: str = Query(...)):
    try:
        _repository, _auth, service = _services(request)
        return {
            "comparisons": [
                item.model_dump(mode="json")
                for item in service.comparisons(workspace_id, campaign_id, _principal(request))
            ]
        }
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/ledger")
def campaign_ledger(campaign_id: str, request: Request, workspace_id: str = Query(...)):
    try:
        repository, _auth, service = _services(request)
        principal = _principal(request)
        service.get(workspace_id, campaign_id, principal)
        return _campaign_ledger_projection(repository, workspace_id, campaign_id)
    except Exception as exc:
        _raise_api(exc)


@campaign_router.get("/{campaign_id}/attempts/{attempt_id}/metrics")
def campaign_metrics(
    campaign_id: str,
    attempt_id: str,
    request: Request,
    workspace_id: str = Query(...),
    metric_name: str = Query(..., min_length=1, max_length=160),
    source: str = Query(..., min_length=1, max_length=240),
    after_step: int = Query(default=-1, ge=-1),
    limit: int = Query(default=2000, ge=1, le=5000),
):
    try:
        _repository, _auth, service = _services(request)
        values = service.metrics(
            workspace_id,
            campaign_id,
            attempt_id,
            metric_name,
            _principal(request),
            source=source,
            after_step=after_step,
            limit=limit,
        )
        return {
            "metric_name": metric_name,
            "source": source,
            "values": [item.model_dump() for item in values],
            "next_after_step": values[-1].step if values else after_step,
        }
    except Exception as exc:
        _raise_api(exc)


__all__ = [
    "CampaignTemplate",
    "campaign_auth_router",
    "campaign_router",
]
