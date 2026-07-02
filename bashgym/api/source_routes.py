"""API routes for curated public training/evaluation sources."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from bashgym.sources import (
    DEFAULT_SOURCE_FETCH_LIMIT,
    SourceUse,
    fetch_source_records,
    get_source,
    list_sources,
    prepare_source_artifacts,
    prepare_source_manifest,
    recommend_sources,
)
from bashgym.sources.catalog import validate_catalog

router = APIRouter(prefix="/api/sources", tags=["sources"])


def _source_allowed_roots() -> list[Path]:
    """Directories a source fetch/prepare may write to or read from.

    Defaults to the project ``data/`` and ``.bashgym/`` dirs plus ``~/.bashgym``.
    Extra roots can be added via the ``BASHGYM_SOURCE_ROOTS`` env var (os.pathsep
    separated) — used by tests and by operators with non-default data layouts.
    """

    roots = [
        Path("data").resolve(),
        Path(".bashgym").resolve(),
        (Path.home() / ".bashgym").resolve(),
    ]
    for entry in os.environ.get("BASHGYM_SOURCE_ROOTS", "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry).expanduser().resolve())
    return roots


def _within_allowed_roots(resolved: Path) -> bool:
    return any(
        resolved == root or resolved.is_relative_to(root) for root in _source_allowed_roots()
    )


def _resolve_source_output_dir(raw: str | None) -> Path:
    """Resolve a caller-supplied output directory, confined to allowed roots.

    Prevents an unauthenticated caller from writing source_records.jsonl (and
    creating directories) anywhere on the host. The directory need not exist yet
    (fetch creates it), so existence is not required — only containment.
    """

    if not raw or not str(raw).strip():
        raise HTTPException(status_code=400, detail="output_dir is required")
    resolved = Path(str(raw)).expanduser().resolve()
    if not _within_allowed_roots(resolved):
        raise HTTPException(
            status_code=400,
            detail="output_dir is outside allowed directories (data/, .bashgym/)",
        )
    return resolved


def _resolve_source_input_path(raw: str) -> Path:
    """Resolve a caller-supplied input file, confined to allowed roots."""

    resolved = Path(str(raw)).expanduser().resolve()
    if not _within_allowed_roots(resolved):
        raise HTTPException(
            status_code=400,
            detail="input_path is outside allowed directories (data/, .bashgym/)",
        )
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=400, detail="input_path must be an existing file")
    return resolved


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
    fetch: bool = Field(
        default=False,
        description="Fetch Hugging Face source records into output_dir before artifact conversion.",
    )
    split: str = Field(default="train", max_length=100)
    subset: str | None = Field(default=None, max_length=200)
    revision: str | None = Field(default=None, max_length=200)
    limit: int | None = Field(default=None, ge=1, le=100000)
    fetch_approval_reason: str | None = Field(default=None, max_length=500)
    force_refresh: bool = False
    allow_eval_only: bool = False
    override_reason: str | None = Field(default=None, max_length=500)


class SourceFetchRequest(BaseModel):
    output_dir: str
    split: str = Field(default="train", max_length=100)
    subset: str | None = Field(default=None, max_length=200)
    revision: str | None = Field(default=None, max_length=200)
    limit: int | None = Field(default=DEFAULT_SOURCE_FETCH_LIMIT, ge=1, le=100000)
    approval_reason: str | None = Field(default=None, max_length=500)
    force_refresh: bool = False


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


@router.post("/{source_id}/fetch")
async def fetch_source(source_id: str, body: SourceFetchRequest) -> dict[str, Any]:
    """Fetch Hugging Face-backed source records into local JSONL."""

    try:
        card = get_source(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown source {source_id!r}") from exc
    output_dir = str(_resolve_source_output_dir(body.output_dir))
    fetch_report = fetch_source_records(
        card,
        output_dir=output_dir,
        split=body.split,
        subset=body.subset,
        revision=body.revision,
        limit=body.limit,
        approval_reason=body.approval_reason,
        force_refresh=body.force_refresh,
    )
    if not fetch_report["ok"]:
        raise HTTPException(status_code=400, detail=fetch_report)
    return fetch_report


@router.post("/{source_id}/prepare")
async def prepare_source(source_id: str, body: SourcePrepareRequest) -> dict[str, Any]:
    """Prepare a source manifest or local converted source artifacts."""

    try:
        card = get_source(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown source {source_id!r}") from exc
    input_path = body.input_path
    if input_path:
        input_path = str(_resolve_source_input_path(input_path))
    fetch_report = None
    if body.fetch:
        if input_path:
            raise HTTPException(
                status_code=400,
                detail="fetch cannot be combined with input_path; choose remote fetch or local input",
            )
        output_dir = str(_resolve_source_output_dir(body.output_dir))
        fetch_report = fetch_source_records(
            card,
            output_dir=output_dir,
            split=body.split,
            subset=body.subset,
            revision=body.revision,
            limit=body.limit if body.limit is not None else DEFAULT_SOURCE_FETCH_LIMIT,
            approval_reason=body.fetch_approval_reason,
            force_refresh=body.force_refresh,
        )
        if not fetch_report["ok"]:
            raise HTTPException(status_code=400, detail=fetch_report)
        input_path = fetch_report["records_path"]

    if input_path:
        output_dir = str(_resolve_source_output_dir(body.output_dir))
        artifact_report = prepare_source_artifacts(
            card,
            goal=body.goal,
            input_path=input_path,
            output_dir=output_dir,
            allow_eval_only=body.allow_eval_only,
            override_reason=body.override_reason,
            limit=body.limit,
        )
        if fetch_report:
            artifact_report["fetch_report"] = fetch_report
        if not artifact_report["ok"]:
            raise HTTPException(status_code=400, detail=artifact_report)
        return artifact_report
    manifest_output_dir = (
        str(_resolve_source_output_dir(body.output_dir)) if body.output_dir else None
    )
    manifest = prepare_source_manifest(
        card,
        goal=body.goal,
        output_dir=manifest_output_dir,
        allow_eval_only=body.allow_eval_only,
        override_reason=body.override_reason,
    )
    status_code = 200 if manifest["use_verdict"]["ok"] else 400
    if status_code == 400:
        raise HTTPException(status_code=400, detail=manifest)
    return {"ok": True, **manifest}
