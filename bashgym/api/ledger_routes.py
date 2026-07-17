"""Authenticated read and ingestion surface for the experiment ledger."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from typing_extensions import Never

from bashgym.api.campaign_routes import _autoresearch_core, _principal, _raise_api
from bashgym.campaigns.autoresearch import AutoResearchError
from bashgym.campaigns.contracts import Capability
from bashgym.campaigns.persistence import RecordNotFoundError
from bashgym.config import get_bashgym_dir
from bashgym.ledger.contracts import (
    ArtifactSpec,
    DecisionSpec,
    EvaluationResultSpec,
    EvaluationSuiteSpec,
    LedgerEventSpec,
)
from bashgym.ledger.persistence import (
    ExperimentLedgerRepository,
    LedgerConflictError,
    LedgerTransitionError,
)
from bashgym.ledger.synthesis import (
    build_project_context,
    build_sync_envelope,
    compare_runs,
    metric_trend,
)

router = APIRouter(prefix="/api/ledger", tags=["experiment-ledger"])
logger = logging.getLogger(__name__)


def _repository(request: Request) -> ExperimentLedgerRepository:
    repository = getattr(request.app.state, "experiment_ledger_repository", None)
    if isinstance(repository, ExperimentLedgerRepository):
        return repository
    campaign_repository = getattr(request.app.state, "campaign_repository", None)
    database_path = getattr(campaign_repository, "db_path", None)
    repository = ExperimentLedgerRepository(
        database_path or get_bashgym_dir() / "campaigns" / "campaigns.sqlite3"
    )
    repository.initialize()
    request.app.state.experiment_ledger_repository = repository
    return repository


def _authorize(request: Request, workspace_id: str) -> None:
    try:
        _principal(request).require(workspace_id, Capability.CAMPAIGN_READ)
    except Exception as exc:
        _raise_api(exc)


def _authorize_write(request: Request, workspace_id: str) -> None:
    try:
        _principal(request).require(workspace_id, Capability.EXPERIMENT_LEDGER_WRITE)
    except Exception as exc:
        _raise_api(exc)


def _require_route_scope(project_id: str, workspace_id: str, spec) -> None:
    if spec.workspace_id != workspace_id or spec.project_id != project_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "ledger_scope_mismatch",
                "message": "Path, query, and record workspace/project identities must match.",
            },
        )


def _raise_ledger(exc: Exception) -> Never:
    if isinstance(exc, RecordNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": "ledger_not_found", "message": "Ledger record not found."},
        ) from exc
    if isinstance(exc, (LedgerConflictError, LedgerTransitionError)):
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    _raise_api(exc)


def _mutation(record: dict, replayed: bool) -> dict:
    return {
        "schema_version": "experiment_ledger_mutation.v1",
        "record": record,
        "replayed": replayed,
    }


def _automatic_autoresearch_ingestion(
    request: Request,
    repository: ExperimentLedgerRepository,
    evaluation: dict,
) -> dict:
    """Attempt derived campaign ingestion without invalidating the ledger write."""

    projection = {
        "schema_version": "autoresearch_evaluation_ingestion.v1",
        "status": "not_applicable",
        "code": "autoresearch_evaluation_not_completed",
        "campaign_id": None,
        "outcome": None,
    }
    if evaluation["status"] != "completed":
        return projection
    run = repository.get_run(
        evaluation["workspace_id"], evaluation["project_id"], evaluation["run_id"]
    )
    campaign_id = run.get("campaign_id")
    projection["campaign_id"] = campaign_id
    if run.get("source_system") != "bashgym" or not campaign_id:
        projection["code"] = "autoresearch_campaign_lineage_not_present"
        return projection
    if (evaluation.get("slice_metrics") or {}).get("autoresearch_role") == "checkpoint":
        projection["code"] = "autoresearch_checkpoint_evaluation"
        return projection
    campaign_repository = getattr(request.app.state, "campaign_repository", repository)
    try:
        outcome = _autoresearch_core(campaign_repository).ingest_evaluation_result(
            workspace_id=evaluation["workspace_id"],
            campaign_id=campaign_id,
            project_id=evaluation["project_id"],
            evaluation_result_id=evaluation["evaluation_result_id"],
        )
    except RecordNotFoundError:
        projection["code"] = "autoresearch_campaign_not_registered"
        return projection
    except AutoResearchError as exc:
        projection["status"] = "deferred"
        projection["code"] = str(exc).split(":", 1)[0] or exc.code
        return projection
    except Exception:  # noqa: BLE001 - keep the authoritative ledger write durable
        logger.exception("automatic AutoResearch evaluation ingestion failed")
        projection["status"] = "deferred"
        projection["code"] = "autoresearch_ingestion_internal_error"
        return projection
    projection["status"] = "ingested"
    projection["code"] = (
        "autoresearch_evaluation_replayed"
        if outcome.replayed
        else "autoresearch_evaluation_ingested"
    )
    projection["outcome"] = outcome.model_dump(mode="json")
    return projection


@router.get("/health")
def ledger_health(request: Request, workspace_id: str = Query(min_length=1, max_length=160)):
    _authorize(request, workspace_id)
    return _repository(request).database_health(workspace_id)


@router.get("/projects")
def list_projects(request: Request, workspace_id: str = Query(min_length=1, max_length=160)):
    _authorize(request, workspace_id)
    return {
        "schema_version": "experiment_projects.v1",
        "projects": _repository(request).list_projects(workspace_id),
    }


@router.get("/projects/{project_id}")
def get_project(
    project_id: str, request: Request, workspace_id: str = Query(min_length=1, max_length=160)
):
    _authorize(request, workspace_id)
    try:
        return _repository(request).get_project(workspace_id, project_id)
    except Exception as exc:
        _raise_ledger(exc)


@router.get("/projects/{project_id}/context")
def project_context(
    project_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
    recent_limit: int = Query(default=20, ge=1, le=100),
):
    _authorize(request, workspace_id)
    try:
        return build_project_context(
            _repository(request), workspace_id, project_id, recent_limit=recent_limit
        )
    except Exception as exc:
        _raise_ledger(exc)


@router.get("/projects/{project_id}/runs")
def list_runs(
    project_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    _authorize(request, workspace_id)
    try:
        return {
            "schema_version": "experiment_runs.v1",
            "project_id": project_id,
            "runs": _repository(request).list_runs(
                workspace_id, project_id, status=status, limit=limit
            ),
        }
    except Exception as exc:
        _raise_ledger(exc)


@router.get("/projects/{project_id}/runs/{run_id}")
def run_details(
    project_id: str,
    run_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
):
    _authorize(request, workspace_id)
    try:
        return _repository(request).run_details(workspace_id, project_id, run_id)
    except Exception as exc:
        _raise_ledger(exc)


@router.get("/projects/{project_id}/compare")
def compare_project_runs(
    project_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
    run_id: list[str] = Query(min_length=2, max_length=20),
):
    _authorize(request, workspace_id)
    try:
        return compare_runs(_repository(request), workspace_id, project_id, run_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "ledger_comparison_invalid", "message": str(exc)},
        ) from exc
    except Exception as exc:
        _raise_ledger(exc)


@router.get("/projects/{project_id}/metrics/{metric_name}/trend")
def project_metric_trend(
    project_id: str,
    metric_name: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
    run_id: str | None = None,
    limit: int = Query(default=5000, ge=1, le=20_000),
):
    _authorize(request, workspace_id)
    try:
        points = _repository(request).metric_series(
            workspace_id,
            project_id,
            metric_name=metric_name,
            run_id=run_id,
            limit=limit,
        )
        by_run: dict[str, list[dict]] = {}
        for point in points:
            by_run.setdefault(str(point["run_id"]), []).append(point)
        return {
            "schema_version": "experiment_metric_trend.v1",
            "project_id": project_id,
            "run_id": run_id,
            "metric_name": metric_name,
            "trend": metric_trend(points) if run_id else None,
            "trends_by_run": {
                item_run_id: metric_trend(run_points)
                for item_run_id, run_points in sorted(by_run.items())
            },
            "points": points,
        }
    except Exception as exc:
        _raise_ledger(exc)


@router.get("/projects/{project_id}/evaluations")
def list_evaluations(
    project_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
    run_id: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
):
    _authorize(request, workspace_id)
    try:
        return {
            "schema_version": "experiment_evaluations.v1",
            "project_id": project_id,
            "evaluations": _repository(request).list_evaluation_results(
                workspace_id, project_id, run_id=run_id, limit=limit
            ),
        }
    except Exception as exc:
        _raise_ledger(exc)


@router.post("/projects/{project_id}/evaluation-suites")
def register_evaluation_suite(
    project_id: str,
    body: EvaluationSuiteSpec,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
):
    _authorize_write(request, workspace_id)
    _require_route_scope(project_id, workspace_id, body)
    try:
        record, replayed = _repository(request).register_evaluation_suite(body)
        return _mutation(record, replayed)
    except Exception as exc:
        _raise_ledger(exc)


@router.post("/projects/{project_id}/evaluation-results")
def record_evaluation_result(
    project_id: str,
    body: EvaluationResultSpec,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
):
    _authorize_write(request, workspace_id)
    _require_route_scope(project_id, workspace_id, body)
    try:
        repository = _repository(request)
        record, replayed = repository.record_evaluation_result(body)
        mutation = _mutation(record, replayed)
        mutation["autoresearch_ingestion"] = _automatic_autoresearch_ingestion(
            request, repository, record
        )
        return mutation
    except Exception as exc:
        _raise_ledger(exc)


@router.post("/projects/{project_id}/artifacts")
def record_artifact(
    project_id: str,
    body: ArtifactSpec,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
):
    _authorize_write(request, workspace_id)
    _require_route_scope(project_id, workspace_id, body)
    try:
        record, replayed = _repository(request).record_artifact(body)
        return _mutation(record, replayed)
    except Exception as exc:
        _raise_ledger(exc)


@router.post("/projects/{project_id}/decisions")
def record_decision(
    project_id: str,
    body: DecisionSpec,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
):
    _authorize_write(request, workspace_id)
    _require_route_scope(project_id, workspace_id, body)
    try:
        record, replayed = _repository(request).record_decision(body)
        return _mutation(record, replayed)
    except Exception as exc:
        _raise_ledger(exc)


@router.get("/projects/{project_id}/events")
def project_events(
    project_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
    after_cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
):
    _authorize(request, workspace_id)
    try:
        return build_sync_envelope(
            _repository(request),
            workspace_id,
            project_id,
            after_cursor=after_cursor,
            limit=limit,
        )
    except Exception as exc:
        _raise_ledger(exc)


@router.post("/projects/{project_id}/events")
def append_project_event(
    project_id: str,
    body: LedgerEventSpec,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=160),
):
    _authorize_write(request, workspace_id)
    _require_route_scope(project_id, workspace_id, body)
    try:
        record, replayed = _repository(request).append_event(body)
        return _mutation(record, replayed)
    except Exception as exc:
        _raise_ledger(exc)
