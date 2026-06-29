"""API routes for curated public training/evaluation sources."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from bashgym.sources import (
    SourceUse,
    get_source,
    list_sources,
    prepare_source_artifacts,
    prepare_source_manifest,
    recommend_sources,
)
from bashgym.sources.catalog import validate_catalog

router = APIRouter(prefix="/api/sources", tags=["sources"])


class SourceRecommendRequest(BaseModel):
    domain: str | None = None
    goal: SourceUse | None = None
    include_eval_only: bool = False


class SourcePrepareRequest(BaseModel):
    goal: SourceUse = SourceUse.EVALUATION
    output_dir: str | None = None
    input_path: str | None = Field(
        default=None,
        description="Optional local JSON/JSONL source records to convert into artifacts.",
    )
    limit: int | None = Field(default=None, ge=1, le=100000)
    allow_eval_only: bool = False
    override_reason: str | None = Field(default=None, max_length=500)


@router.get("")
async def list_source_cards() -> dict[str, Any]:
    """List curated source cards and registry validation status."""

    errors = validate_catalog()
    return {
        "ok": not errors,
        "schema_version": "bashgym.source_catalog.v1",
        "count": len(list_sources()),
        "sources": [card.to_dict() for card in list_sources()],
        "validation_errors": errors,
    }


@router.get("/{source_id}")
async def inspect_source(source_id: str) -> dict[str, Any]:
    """Inspect one source card."""

    try:
        card = get_source(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown source {source_id!r}") from exc
    return {
        "ok": True,
        "schema_version": "bashgym.source_card.v1",
        "source": card.to_dict(),
        "validation_errors": card.validation_errors(),
    }


@router.post("/recommend")
async def recommend_source_cards(body: SourceRecommendRequest) -> dict[str, Any]:
    """Recommend source cards for a domain and training/eval goal."""

    return {
        "ok": True,
        "schema_version": "bashgym.source_recommendations.v1",
        "domain": body.domain,
        "goal": body.goal.value if body.goal else None,
        "recommendations": recommend_sources(
            domain=body.domain,
            goal=body.goal,
            include_eval_only=body.include_eval_only,
        ),
    }


@router.post("/{source_id}/prepare")
async def prepare_source(source_id: str, body: SourcePrepareRequest) -> dict[str, Any]:
    """Prepare a source manifest or local converted source artifacts."""

    try:
        card = get_source(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown source {source_id!r}") from exc
    if body.input_path:
        if not body.output_dir:
            raise HTTPException(
                status_code=400,
                detail="input_path artifact conversion requires output_dir",
            )
        artifact_report = prepare_source_artifacts(
            card,
            goal=body.goal,
            input_path=body.input_path,
            output_dir=body.output_dir,
            allow_eval_only=body.allow_eval_only,
            override_reason=body.override_reason,
            limit=body.limit,
        )
        if not artifact_report["ok"]:
            raise HTTPException(status_code=400, detail=artifact_report)
        return artifact_report
    manifest = prepare_source_manifest(
        card,
        goal=body.goal,
        output_dir=body.output_dir,
        allow_eval_only=body.allow_eval_only,
        override_reason=body.override_reason,
    )
    status_code = 200 if manifest["use_verdict"]["ok"] else 400
    if status_code == 400:
        raise HTTPException(status_code=400, detail=manifest)
    return {"ok": True, **manifest}
