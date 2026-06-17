"""Advanced eval API routes: held-out trace gate + external benchmark wiring.

Surfaces the model-agnostic eval modules (``bashgym.eval``) over HTTP:

- ``POST /api/eval/heldout``      run the base-vs-candidate held-out gate (async)
- ``GET  /api/eval/heldout/{id}`` poll one held-out run (full report)
- ``GET  /api/eval/heldout``      list recent held-out runs
- ``GET  /api/eval/verdict/{model_id}`` latest ship/no-ship verdict from the registry
- ``GET  /api/eval/benchmark-commands`` argv for the external harnesses (run on host)
- ``POST /api/eval/benchmarks/ingest``  fold lm-eval results into forgetting + record

The held-out run hits a served, OpenAI-compatible endpoint (a connected provider
or an explicit ``base_url``/``model``); the heavy external harnesses run in the
serving venv on the host, so this only builds their commands and ingests results.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from bashgym.eval import service
from bashgym.eval.heldout import _METRICS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["eval"])


# =============================================================================
# Persistence
# =============================================================================


def _jobs_file() -> Path:
    from bashgym.config import get_bashgym_dir

    d = get_bashgym_dir() / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d / "heldout_jobs.json"


def _load_jobs() -> dict[str, Any]:
    f = _jobs_file()
    if f.exists():
        try:
            with open(f, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load held-out jobs: %s", e)
    return {}


def _save_jobs(jobs: dict[str, Any]) -> None:
    try:
        with open(_jobs_file(), "w", encoding="utf-8") as fh:
            json.dump(jobs, fh, indent=2)
    except OSError as e:
        logger.error("Failed to save held-out jobs: %s", e)


def _jobs(request: Request) -> dict[str, Any]:
    """The in-memory held-out job store, lazily hydrated from disk."""
    if getattr(request.app.state, "heldout_jobs", None) is None:
        request.app.state.heldout_jobs = _load_jobs()
    return request.app.state.heldout_jobs


# =============================================================================
# Schemas
# =============================================================================


class EndpointSpec(BaseModel):
    """A served model to evaluate: a connected provider or an explicit endpoint."""

    provider: str | None = Field(None, description="Connected provider_type (see /api/providers)")
    base_url: str | None = Field(None, description="OpenAI-compatible base URL (e.g. .../v1)")
    model: str = Field("", description="Model name to request (overrides provider default)")
    api_key: str | None = Field(None, description="Override key; omit to reuse the provider's")


class HeldoutEvalRequest(BaseModel):
    model_id: str = Field(..., description="Registry model_id the verdict is recorded against")
    dataset_path: str = Field(..., description="Path to the frozen held-out .jsonl")
    candidate: EndpointSpec
    base: EndpointSpec
    metric: str = Field("exact_match", description=f"One of {list(_METRICS)}")
    limit: int | None = Field(None, ge=1, description="Cap examples for a fast smoke eval")
    min_trace_delta: float | None = None
    max_forgetting_drop: float | None = None
    require_ci_excludes_zero: bool | None = None
    forgetting_drops: dict[str, float] | None = Field(
        None, description="Pre-computed {task: base-candidate} drops to fold into the gate"
    )
    n_resamples: int = Field(1000, ge=100, le=10000)
    seed: int = 0


class HeldoutJobResponse(BaseModel):
    job_id: str
    model_id: str
    metric: str
    status: str  # running, completed, failed
    report: dict | None = None
    error: str | None = None
    created_at: str | None = None


class BenchmarkIngestRequest(BaseModel):
    model_id: str
    base_results: dict = Field(..., description="lm-eval/NeMo-Evaluator results JSON for the base")
    candidate_results: dict = Field(..., description="...and for the candidate")
    max_forgetting_drop: float | None = None


# =============================================================================
# Held-out trace gate
# =============================================================================


def _run_heldout_job(
    jobs: dict[str, Any],
    job_id: str,
    req: HeldoutEvalRequest,
    base_cfg: service.EndpointConfig,
    cand_cfg: service.EndpointConfig,
) -> None:
    """Background worker: load held-out set, run the gate, record the verdict.

    Sync (FastAPI runs it in a threadpool) because the predictors make blocking
    HTTP calls to the served endpoint.
    """
    try:
        examples = service.load_jsonl_examples(req.dataset_path, limit=req.limit)
        thresholds = service.thresholds_from(
            min_trace_delta=req.min_trace_delta,
            max_forgetting_drop=req.max_forgetting_drop,
            require_ci_excludes_zero=req.require_ci_excludes_zero,
        )
        report = service.run_heldout(
            examples,
            base_cfg,
            cand_cfg,
            metric=req.metric,
            thresholds=thresholds,
            forgetting_drops=req.forgetting_drops,
            n_resamples=req.n_resamples,
            seed=req.seed,
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["report"] = report.to_dict()
        _record_verdict(req.model_id, report.to_dict())
    except Exception as e:  # noqa: BLE001 - any failure marks the job failed, not crashes
        logger.exception("Held-out eval job %s failed", job_id)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    _save_jobs(jobs)


def _record_verdict(model_id: str, report: dict) -> None:
    try:
        from bashgym.models import get_registry

        get_registry().record_heldout_eval(model_id, report)
    except Exception as e:  # noqa: BLE001 - recording is best-effort; the run still stands
        logger.warning("Could not record held-out verdict for %s: %s", model_id, e)


def _resolve(request: Request, spec: EndpointSpec, label: str) -> service.EndpointConfig:
    registry = getattr(request.app.state, "provider_registry", None)
    try:
        return service.resolve_endpoint(
            provider_registry=registry,
            provider=spec.provider,
            base_url=spec.base_url,
            model=spec.model,
            api_key=spec.api_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"{label} endpoint: {e}") from e


@router.post("/heldout", response_model=HeldoutJobResponse)
async def run_heldout(request: Request, req: HeldoutEvalRequest, background: BackgroundTasks):
    """Start a held-out base-vs-candidate gate run against served endpoints."""
    if req.metric not in _METRICS:
        raise HTTPException(status_code=400, detail=f"metric must be one of {list(_METRICS)}")
    if not Path(req.dataset_path).exists():
        raise HTTPException(status_code=400, detail=f"dataset not found: {req.dataset_path}")

    base_cfg = _resolve(request, req.base, "base")
    cand_cfg = _resolve(request, req.candidate, "candidate")

    job_id = f"heldout_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    job = {
        "job_id": job_id,
        "model_id": req.model_id,
        "metric": req.metric,
        "status": "running",
        "report": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    jobs = _jobs(request)
    jobs[job_id] = job
    _save_jobs(jobs)

    background.add_task(_run_heldout_job, jobs, job_id, req, base_cfg, cand_cfg)
    return HeldoutJobResponse(**job)


@router.get("/heldout/{job_id}", response_model=HeldoutJobResponse)
async def get_heldout(request: Request, job_id: str):
    job = _jobs(request).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="held-out job not found")
    return HeldoutJobResponse(**job)


@router.get("/heldout", response_model=list[HeldoutJobResponse])
async def list_heldout(request: Request, limit: int = 20):
    jobs = list(_jobs(request).values())
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return [HeldoutJobResponse(**j) for j in jobs[:limit]]


@router.get("/verdict/{model_id}")
async def get_verdict(model_id: str):
    """Latest ship/no-ship held-out verdict recorded on a model profile."""
    try:
        from bashgym.models import get_registry

        profile = get_registry().get(model_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"registry error: {e}") from e
    if profile is None:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id}")
    return {
        "model_id": model_id,
        "display_name": getattr(profile, "display_name", model_id),
        "latest_heldout_eval": profile.latest_heldout_eval,
        "n_heldout_evals": len(getattr(profile, "heldout_evals", [])),
    }


# =============================================================================
# External benchmarks (run on the serving host; ingest results here)
# =============================================================================


@router.get("/benchmark-commands")
async def benchmark_commands(
    base_url: str,
    model: str,
    tasks: str | None = None,
    include: str = "forgetting,terminal_bench,bfcl,swebench",
):
    """argv for each external benchmark harness against a served endpoint."""
    task_tuple = tuple(t.strip() for t in tasks.split(",") if t.strip()) if tasks else None
    include_tuple = tuple(t.strip() for t in include.split(",") if t.strip())
    cmds = service.benchmark_commands(
        base_url, model, forgetting_tasks=task_tuple, include=include_tuple
    )
    return {"commands": cmds}


@router.post("/benchmarks/ingest")
async def ingest_benchmarks(req: BenchmarkIngestRequest):
    """Diff base vs candidate lm-eval results into forgetting drops and record scores."""
    report = service.ingest_forgetting(req.base_results, req.candidate_results)
    recorded: list[str] = []
    try:
        from bashgym.models import get_registry

        recorded = service.record_forgetting(get_registry(), req.model_id, report)
    except Exception as e:  # noqa: BLE001 - recording is best-effort
        logger.warning("Could not record forgetting scores for %s: %s", req.model_id, e)

    max_drop = req.max_forgetting_drop if req.max_forgetting_drop is not None else 0.05
    worst = report.worst
    forgetting_ok = not any(d > max_drop for d in report.drops.values())
    return {
        "model_id": req.model_id,
        "forgetting": report.to_dict(),
        "recorded": recorded,
        "max_forgetting_drop": max_drop,
        "forgetting_ok": forgetting_ok,
        "worst": list(worst) if worst else None,
    }
