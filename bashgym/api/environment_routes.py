"""API routes for executable terminal environment import and curation."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from bashgym.environments.builder import materialize_environment
from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.decontaminate import filter_contaminated_environments
from bashgym.environments.loader import environment_from_record
from bashgym.environments.metrics import summarize_environment_mix
from bashgym.environments.tmax_importer import TMAX_HF_DATASETS, TMaxImporter

router = APIRouter(prefix="/api/environments", tags=["environments"])


class NormalizeEnvironmentsRequest(BaseModel):
    """Normalize raw TMax/Harbor/DataDesigner-like records into EnvironmentSpec."""

    records: list[dict[str, Any]]
    source: str = "external"
    source_uri: str | None = None
    preserve_raw: bool = True


class ImportEnvironmentsRequest(BaseModel):
    """Import a local JSON/JSONL file into EnvironmentSpec records."""

    path: str
    source: str = "tmax"
    preserve_raw: bool = True


class MaterializeEnvironmentRequest(BaseModel):
    """Write one EnvironmentSpec to a local executable environment bundle."""

    environment: dict[str, Any]
    output_dir: str
    overwrite: bool = False


class DecontaminateEnvironmentsRequest(BaseModel):
    """Drop environments overlapping benchmark text."""

    environments: list[dict[str, Any]]
    benchmark_texts: list[str]
    big_n: int = Field(default=13, ge=1)
    small_n: int = Field(default=3, ge=1)
    jaccard_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


def _report_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _env_response(envs: list[EnvironmentSpec], errors: list[dict[str, Any]] | None = None) -> dict:
    return {
        "environments": [env.to_dict() for env in envs],
        "report": summarize_environment_mix(envs).to_dict(),
        "errors": errors or [],
    }


def _environment_from_payload(payload: dict[str, Any]) -> EnvironmentSpec:
    spec = EnvironmentSpec.from_dict(payload)
    errors = spec.validation_errors()
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})
    return spec


@router.get("/pipelines")
async def environment_pipelines() -> dict:
    """Expose environment-generation affordances and known external sources."""
    try:
        from bashgym.factory.designer_pipelines import DATA_DESIGNER_AVAILABLE, PIPELINES
    except ImportError:
        data_designer_available = False
        pipeline_names: list[str] = []
    else:
        data_designer_available = DATA_DESIGNER_AVAILABLE
        pipeline_names = sorted(PIPELINES)

    return {
        "available": True,
        "data_designer_available": data_designer_available,
        "pipelines": [
            {
                "name": "terminal_env_generation",
                "available": "terminal_env_generation" in pipeline_names,
                "description": "Generate executable, verifiable terminal tasks with environment axes.",
                "outputs": ["EnvironmentSpec", "env.json", "workspace files", "verifier"],
            }
        ],
        "registered_pipelines": pipeline_names,
        "external_sources": TMAX_HF_DATASETS,
    }


@router.post("/normalize")
async def normalize_environments(request: NormalizeEnvironmentsRequest) -> dict:
    """Normalize raw environment rows while preserving per-row validation issues."""
    envs: list[EnvironmentSpec] = []
    errors: list[dict[str, Any]] = []
    for idx, record in enumerate(request.records):
        try:
            env = environment_from_record(
                record,
                source=request.source,
                source_uri=request.source_uri,
                preserve_raw=request.preserve_raw,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"index": idx, "error": str(exc)})
            continue
        validation_errors = env.validation_errors()
        if validation_errors:
            errors.append({"index": idx, "id": env.id, "validation_errors": validation_errors})
        envs.append(env)
    return _env_response(envs, errors)


@router.post("/import-jsonl")
async def import_jsonl(request: ImportEnvironmentsRequest) -> dict:
    """Import a local JSON or JSONL environment file."""
    path = Path(request.path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"environment file not found: {path}")
    importer = TMaxImporter(preserve_raw=request.preserve_raw)
    try:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            envs = importer.from_jsonl(path, source=request.source)
        elif suffix == ".json":
            envs = importer.from_json(path, source=request.source)
        else:
            raise ValueError("supported formats are .json, .jsonl, and .ndjson")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    errors = [
        {"index": idx, "id": env.id, "validation_errors": env.validation_errors()}
        for idx, env in enumerate(envs)
        if env.validation_errors()
    ]
    return _env_response(envs, errors)


@router.post("/decontaminate")
async def decontaminate_environments(request: DecontaminateEnvironmentsRequest) -> dict:
    """Filter imported/generated environments against benchmark text snippets."""
    envs = [EnvironmentSpec.from_dict(payload) for payload in request.environments]
    kept, report = filter_contaminated_environments(
        envs,
        request.benchmark_texts,
        big_n=request.big_n,
        small_n=request.small_n,
        jaccard_threshold=request.jaccard_threshold,
    )
    return {
        "environments": [env.to_dict() for env in kept],
        "report": _report_dict(report),
        "mix_report": summarize_environment_mix(kept).to_dict(),
    }


@router.post("/materialize")
async def materialize_environment_bundle(request: MaterializeEnvironmentRequest) -> dict:
    """Materialize an environment bundle without executing its verifier/build."""
    spec = _environment_from_payload(request.environment)
    try:
        result = materialize_environment(spec, request.output_dir, overwrite=request.overwrite)
    except (OSError, ValueError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"build": result.to_dict()}
