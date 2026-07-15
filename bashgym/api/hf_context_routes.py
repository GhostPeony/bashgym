"""Workspace-scoped Hugging Face context-pack routes."""

from __future__ import annotations

import asyncio
from typing import Any, Never

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from bashgym.api.workspace_routes import WorkspaceEvent, WorkspaceEventSource, post_workspace_event
from bashgym.config import get_bashgym_dir
from bashgym.integrations.huggingface.context_persistence import (
    BundleAlreadyExistsError,
    BundleNotFoundError,
    BundleRevisionConflictError,
    HFContextRepository,
    ImmutableBundleError,
)
from bashgym.integrations.huggingface.context_service import HFContextService

router = APIRouter(prefix="/api/hf/context", tags=["huggingface-context"])


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiscoverInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=200)
    intent: str = Field(min_length=1, max_length=2000)
    task: str | None = Field(default=None, max_length=200)
    target: dict[str, Any] = Field(default_factory=dict)
    origin: dict[str, str] = Field(default_factory=dict)


class PinInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=200)
    expected_version: int = Field(ge=1)
    selected_evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class WorkspaceInput(ApiModel):
    workspace_id: str = Field(min_length=1, max_length=200)


class RefreshInput(WorkspaceInput):
    expected_version: int = Field(ge=1)


def _service(request: Request) -> HFContextService:
    current = getattr(request.app.state, "hf_context_service", None)
    if isinstance(current, HFContextService):
        return current
    repository = HFContextRepository(get_bashgym_dir() / "hf_context" / "hf_context.sqlite3")
    repository.initialize()
    current = HFContextService(repository)
    request.app.state.hf_context_service = current
    return current


def _raise_api(exc: Exception) -> Never:
    if isinstance(exc, BundleNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": "hf_bundle_not_found", "message": "Context bundle not found."},
        ) from exc
    if isinstance(exc, BundleRevisionConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "hf_bundle_conflict",
                "message": str(exc),
                "expected": exc.expected,
                "current": exc.current,
            },
        ) from exc
    if isinstance(exc, (BundleAlreadyExistsError, ImmutableBundleError)):
        raise HTTPException(
            status_code=409,
            detail={"code": "hf_bundle_conflict", "message": str(exc)},
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(
            status_code=422,
            detail={"code": "hf_context_invalid", "message": str(exc)},
        ) from exc
    raise exc


async def _publish_bundle_event(
    request: Request,
    event_type: str,
    *,
    workspace_id: str,
    bundle_id: str | None = None,
    version: int | None = None,
    title: str,
    summary: str,
    evidence_count: int | None = None,
) -> None:
    entity: dict[str, Any] = {}
    if bundle_id is not None:
        entity["bundle_id"] = bundle_id
    if version is not None:
        entity["version"] = version
    if evidence_count is not None:
        entity["evidence_count"] = evidence_count
    await post_workspace_event(
        request,
        WorkspaceEvent(
            type=event_type,
            workspace_id=workspace_id,
            source=WorkspaceEventSource(kind="huggingface"),
            title=title,
            summary=summary,
            entity=entity,
            payload={
                "idempotency_key": f"{event_type}:{workspace_id}:{bundle_id or 'active'}:{version or 0}"
            },
        ),
    )


async def _run_discovery_task(
    request: Request,
    *,
    workspace_id: str,
    bundle_id: str,
    version: int,
) -> None:
    bundle = await asyncio.to_thread(
        _service(request).run_discovery, workspace_id, bundle_id, version
    )
    if bundle.completion_outcome is None or bundle.completion_outcome.value == "cancelled":
        return
    await _publish_bundle_event(
        request,
        "hf-context:discovery-completed",
        workspace_id=workspace_id,
        bundle_id=bundle.bundle_id,
        version=bundle.version,
        title="Hugging Face evidence ready",
        summary=f"Prepared {len(bundle.evidence)} source-linked evidence records.",
        evidence_count=len(bundle.evidence),
    )


@router.post("/discover", status_code=status.HTTP_202_ACCEPTED)
async def discover_context(
    request: Request, body: DiscoverInput, background: BackgroundTasks
):
    try:
        bundle = _service(request).begin_discovery(
            workspace_id=body.workspace_id,
            intent=body.intent,
            task=body.task,
            target=body.target,
            origin=body.origin,
        )
        await _publish_bundle_event(
            request,
            "hf-context:discovery-started",
            workspace_id=body.workspace_id,
            bundle_id=bundle.bundle_id,
            version=bundle.version,
            title="Hugging Face discovery started",
            summary="Collecting source-linked model, dataset, and evaluation evidence.",
            evidence_count=0,
        )
        background.add_task(
            _run_discovery_task,
            request,
            workspace_id=body.workspace_id,
            bundle_id=bundle.bundle_id,
            version=bundle.version,
        )
        return bundle.model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.get("/bundles")
async def list_bundles(
    request: Request,
    workspace_id: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=100),
):
    try:
        service = _service(request)
        return {
            "bundles": [item.model_dump(mode="json") for item in service.history(workspace_id, limit=limit)],
            "active": service.active_summary(workspace_id),
        }
    except Exception as exc:
        _raise_api(exc)


@router.get("/bundles/{bundle_id}/versions/{version}")
async def get_bundle(
    request: Request,
    bundle_id: str,
    version: int,
    workspace_id: str = Query(..., min_length=1, max_length=200),
):
    try:
        return _service(request).get(workspace_id, bundle_id, version).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.post("/bundles/{bundle_id}/versions/{version}/pin", status_code=status.HTTP_201_CREATED)
async def pin_bundle(request: Request, bundle_id: str, version: int, body: PinInput):
    try:
        bundle = (
            _service(request)
            .pin(
                body.workspace_id,
                bundle_id,
                version,
                selected_evidence_ids=body.selected_evidence_ids,
                expected_version=body.expected_version,
            )
        )
        await _publish_bundle_event(
            request,
            "hf-context:pinned",
            workspace_id=body.workspace_id,
            bundle_id=bundle.bundle_id,
            version=bundle.version,
            title="Hugging Face context pinned",
            summary=f"Pinned {len(bundle.selected_evidence_ids)} evidence records.",
            evidence_count=len(bundle.selected_evidence_ids),
        )
        return bundle.model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.post(
    "/bundles/{bundle_id}/versions/{version}/refresh",
    status_code=status.HTTP_202_ACCEPTED,
)
async def refresh_bundle(
    request: Request,
    bundle_id: str,
    version: int,
    body: RefreshInput,
    background: BackgroundTasks,
):
    try:
        collecting, _previous = _service(request).begin_refresh(
            body.workspace_id,
            bundle_id,
            version,
            expected_version=body.expected_version,
        )
        await _publish_bundle_event(
            request,
            "hf-context:discovery-started",
            workspace_id=body.workspace_id,
            bundle_id=bundle_id,
            version=collecting.version,
            title="Hugging Face refresh started",
            summary="Refreshing the exact context lineage with current Hub evidence.",
            evidence_count=len(collecting.evidence),
        )
        background.add_task(
            _run_discovery_task,
            request,
            workspace_id=body.workspace_id,
            bundle_id=bundle_id,
            version=collecting.version,
        )
        return collecting.model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.post("/bundles/{bundle_id}/versions/{version}/cancel")
async def cancel_bundle(
    request: Request, bundle_id: str, version: int, body: WorkspaceInput
):
    try:
        bundle = _service(request).cancel(body.workspace_id, bundle_id, version)
        await _publish_bundle_event(
            request,
            "hf-context:discovery-cancelled",
            workspace_id=body.workspace_id,
            bundle_id=bundle_id,
            version=version,
            title="Hugging Face discovery cancelled",
            summary=f"Kept {len(bundle.evidence)} usable evidence records collected so far.",
            evidence_count=len(bundle.evidence),
        )
        return bundle.model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.post("/bundles/{bundle_id}/versions/{version}/activate")
async def activate_bundle(
    request: Request, bundle_id: str, version: int, body: WorkspaceInput
):
    try:
        bundle = (
            _service(request)
            .activate(body.workspace_id, bundle_id, version)
        )
        await _publish_bundle_event(
            request,
            "hf-context:activated",
            workspace_id=body.workspace_id,
            bundle_id=bundle.bundle_id,
            version=bundle.version,
            title="Hugging Face context activated",
            summary="This immutable bundle is now active for workspace agents.",
            evidence_count=len(bundle.evidence),
        )
        return bundle.model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.delete("/active")
async def deactivate_bundle(request: Request, body: WorkspaceInput):
    try:
        _service(request).deactivate(body.workspace_id)
        await _publish_bundle_event(
            request,
            "hf-context:deactivated",
            workspace_id=body.workspace_id,
            title="Hugging Face context deactivated",
            summary="Cleared the active workspace context pointer.",
        )
        return {"status": "deactivated", "workspace_id": body.workspace_id}
    except Exception as exc:
        _raise_api(exc)


@router.delete("/bundles/{bundle_id}")
async def delete_bundle(request: Request, bundle_id: str, body: WorkspaceInput):
    try:
        _service(request).delete(body.workspace_id, bundle_id)
        return {"status": "deleted", "bundle_id": bundle_id}
    except Exception as exc:
        _raise_api(exc)


@router.get("/bundles/{bundle_id}/versions/{version}/markdown")
async def get_markdown(
    request: Request,
    bundle_id: str,
    version: int,
    workspace_id: str = Query(..., min_length=1, max_length=200),
):
    try:
        return _service(request).markdown(workspace_id, bundle_id, version).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.post("/bundles/{bundle_id}/versions/{version}/actions/eval")
async def prepare_eval(request: Request, bundle_id: str, version: int, body: WorkspaceInput):
    try:
        preview = _service(request).prepare_eval(body.workspace_id, bundle_id, version)
        await _publish_bundle_event(
            request,
            "hf-context:eval-prepared",
            workspace_id=body.workspace_id,
            bundle_id=bundle_id,
            version=version,
            title="Hugging Face Eval preview prepared",
            summary=f"Prepared {len(preview.get('tasks') or [])} evaluation tasks without executing them.",
        )
        return preview
    except Exception as exc:
        _raise_api(exc)


__all__ = ["router"]
