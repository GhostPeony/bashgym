"""Advanced eval API routes: held-out trace gate + external benchmark wiring.

Surfaces the model-agnostic eval modules (``bashgym.eval``) over HTTP:

- ``POST /api/eval/heldout``      run the base-vs-candidate held-out gate (async)
- ``GET  /api/eval/heldout/{id}`` poll one held-out run (full report)
- ``GET  /api/eval/heldout``      list recent held-out runs
- ``GET  /api/eval/verdict/{model_id}`` latest ship/no-ship verdict from the registry
- ``GET  /api/eval/benchmark-commands`` argv for the external harnesses (run on host)
- ``POST /api/eval/benchmarks/ingest``  fold lm-eval results into forgetting + record
- ``POST /api/eval/benchmarks/external-ingest`` normalize and record public harness scores

The held-out run hits a served, OpenAI-compatible endpoint (a connected provider
or an explicit ``base_url``/``model``); the heavy external harnesses run in the
serving venv on the host, so this only builds their commands and ingests results.
"""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.eval import service
from bashgym.eval.dppo_replay import enrich_dppo_replay_jsonl, write_dppo_replay_jsonl
from bashgym.eval.heldout import _METRICS
from bashgym.eval.release_gate import combine_release_gate_evidence
from bashgym.gym.dppo_launcher import DPPOSmokeLaunchConfig, prepare_dppo_smoke_launch

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


class HeldoutEnvironmentEvidence(BaseModel):
    passk: dict[str, Any] | None = Field(None, description="Optional environment pass@k report")
    holdout_gate: dict[str, Any] | None = Field(
        None,
        description="Optional environment holdout gate result or endpoint response",
    )
    holdout_comparison: dict[str, Any] | None = Field(
        None,
        description="Optional base-vs-candidate environment holdout comparison result",
    )
    spurious_reward_control: dict[str, Any] | None = Field(
        None,
        description="Optional spurious-reward negative-control result",
    )
    external_benchmarks: dict[str, Any] | None = Field(
        None,
        description="Optional normalized external benchmark report or ingest response",
    )
    world_model_quality: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional ECHO/RWML prediction-quality metrics. Diagnostic release "
            "evidence only until correlated with held-out pass@k."
        ),
    )
    external_benchmark_min_scores: dict[str, float] | None = Field(
        None,
        description="Optional minimum score thresholds by external benchmark name",
    )
    external_benchmarks_required: bool = Field(
        False,
        description="Block release unless external benchmark evidence is supplied",
    )
    required: bool = Field(
        False,
        description="Block release unless at least one environment gate result is supplied",
    )


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
    environment_evidence: HeldoutEnvironmentEvidence | None = Field(
        None,
        description="Precomputed executable-environment metrics/gates to fold into release verdict",
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


class ExternalBenchmarkIngestRequest(BaseModel):
    model_id: str
    results: Any = Field(
        ...,
        description="External harness result JSON, result map, or trial list",
    )
    benchmark_name: str | None = Field(
        None,
        description="Optional registry benchmark name, e.g. harbor_terminal_bench",
    )
    source: str | None = Field(None, description="Optional source/harness label")
    record_to_registry: bool = True


class EnvironmentAttemptSpec(BaseModel):
    environment_id: str
    attempt_index: int = Field(..., ge=0)
    passed: bool
    reward: float | None = None
    verifier_status: str | None = None
    timeout: bool = False
    tool_calls: int | None = Field(None, ge=0)
    tokens: int | None = Field(None, ge=0)
    action_tokens: int | None = Field(None, ge=0)
    observation_tokens: int | None = Field(None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentPassKRequest(BaseModel):
    model_id: str | None = Field(None, description="Optional model id to record metrics against")
    environments: list[dict[str, Any]] = Field(..., min_length=1)
    attempts: list[EnvironmentAttemptSpec] = Field(..., min_length=1)
    k_values: list[int] = Field(default_factory=lambda: [1, 4, 8], min_length=1)
    record_to_registry: bool = True


class EnvironmentPassKResponse(BaseModel):
    model_id: str | None = None
    report: dict
    recorded: list[str] = Field(default_factory=list)


class EnvironmentHoldoutGateRequest(BaseModel):
    model_id: str | None = Field(None, description="Optional model id to record the gate against")
    environments: list[dict[str, Any]] = Field(..., min_length=1)
    attempts: list[EnvironmentAttemptSpec] = Field(..., min_length=1)
    split_by: str = Field("task_family", description="domain, source, source_uri, repo, generator_seed, or task_family")
    holdout_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    seed: int = 0
    k_values: list[int] = Field(default_factory=lambda: [1, 4, 8], min_length=1)
    min_pass_at_1: float = Field(0.0, ge=0.0, le=1.0)
    max_timeout_rate: float = Field(0.25, ge=0.0, le=1.0)
    max_tamper_rate: float = Field(0.0, ge=0.0, le=1.0)
    require_no_contamination: bool = True
    record_to_registry: bool = True


class EnvironmentHoldoutGateResponse(BaseModel):
    model_id: str | None = None
    result: dict
    recorded: list[str] = Field(default_factory=list)


class EnvironmentHoldoutComparisonRequest(BaseModel):
    environments: list[dict[str, Any]] = Field(..., min_length=1)
    base_attempts: list[EnvironmentAttemptSpec] = Field(..., min_length=1)
    candidate_attempts: list[EnvironmentAttemptSpec] = Field(..., min_length=1)
    split_by: str = Field("task_family", description="domain, source, source_uri, repo, generator_seed, or task_family")
    cluster_by: str = Field("task_family", description="Bootstrap cluster key")
    holdout_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    seed: int = 0
    k_values: list[int] = Field(default_factory=lambda: [1, 4, 8], min_length=1)
    compare_k: int = Field(1, ge=1)
    min_delta: float = Field(0.0, ge=-1.0, le=1.0)
    min_candidate_pass_at_1: float = Field(0.0, ge=0.0, le=1.0)
    require_ci_excludes_zero: bool = True
    max_candidate_timeout_rate: float = Field(0.25, ge=0.0, le=1.0)
    max_candidate_tamper_rate: float = Field(0.0, ge=0.0, le=1.0)
    require_no_contamination: bool = True
    n_resamples: int = Field(1000, ge=1, le=10000)


class EnvironmentHoldoutComparisonResponse(BaseModel):
    result: dict


class EnvironmentSpuriousRewardControlRequest(BaseModel):
    environments: list[dict[str, Any]] = Field(..., min_length=1)
    attempts: list[EnvironmentAttemptSpec] = Field(..., min_length=1)
    control_attempts: list[EnvironmentAttemptSpec] | None = None
    split_by: str = Field("task_family", description="domain, source, source_uri, repo, generator_seed, or task_family")
    holdout_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    seed: int = 0
    k_values: list[int] = Field(default_factory=lambda: [1, 4, 8], min_length=1)
    n_trials: int = Field(200, ge=1, le=5000)
    random_pass_probability: float = Field(0.05, ge=0.0, le=1.0)
    min_observed_pass_at_1: float = Field(0.0, ge=0.0, le=1.0)
    max_control_pass_at_1: float = Field(0.25, ge=0.0, le=1.0)
    min_lift_over_control: float = Field(0.0, ge=-1.0, le=1.0)
    require_no_contamination: bool = True


class EnvironmentSpuriousRewardControlResponse(BaseModel):
    result: dict


class EnvironmentCommandAttemptSpec(BaseModel):
    environment_id: str
    attempt_index: int = Field(..., ge=0)
    commands: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentRolloutPassKRequest(BaseModel):
    model_id: str | None = Field(None, description="Optional model id to record metrics against")
    environments: list[dict[str, Any]] = Field(..., min_length=1)
    command_attempts: list[EnvironmentCommandAttemptSpec] = Field(..., min_length=1)
    k_values: list[int] = Field(default_factory=lambda: [1, 4, 8], min_length=1)
    workspace_root: str | None = Field(
        None,
        description="Local workspace root. Defaults to a temp BashGym rollout directory.",
    )
    keep_workspace: bool = True
    allow_dangerous_commands: bool = False
    stop_on_error: bool = True
    record_to_registry: bool = True


class EnvironmentRolloutPassKResponse(BaseModel):
    model_id: str | None = None
    report: dict
    attempts: list[dict]
    rollouts: list[dict]
    recorded: list[str] = Field(default_factory=list)
    sampling_report: dict[str, Any] | None = None
    dppo_report: dict[str, Any] | None = None
    dppo_replay: dict[str, Any] | None = None


class EnvironmentCanarySuiteRequest(BaseModel):
    categories: list[str] = Field(default_factory=list)
    workspace_root: str | None = Field(
        None,
        description="Local workspace root. Defaults to a temp BashGym canary directory.",
    )
    keep_workspace: bool = True


class EnvironmentCanarySuiteResponse(BaseModel):
    summary: dict
    canaries: list[dict]
    rollouts: list[dict]


class EnvironmentModelRolloutPassKRequest(BaseModel):
    model_id: str | None = Field(None, description="Optional model id to record metrics against")
    endpoint: EndpointSpec
    environments: list[dict[str, Any]] = Field(..., min_length=1)
    attempts_per_environment: int = Field(1, ge=1, le=32)
    k_values: list[int] = Field(default_factory=lambda: [1, 4, 8], min_length=1)
    workspace_root: str | None = Field(
        None,
        description="Local workspace root. Defaults to a temp BashGym rollout directory.",
    )
    keep_workspace: bool = True
    allow_dangerous_commands: bool = False
    stop_on_error: bool = False
    max_tool_calls: int | None = Field(None, ge=1)
    max_observation_chars: int = Field(6000, ge=500, le=50000)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=16, le=8192)
    request_timeout: float = Field(60.0, ge=1.0, le=600.0)
    use_tool_calling: bool = True
    capture_logprobs: bool = False
    top_logprobs: int | None = Field(None, ge=0, le=20)
    filter_zero_std_groups: bool = False
    active_sampling: bool = False
    target_prompt_groups: int | None = Field(None, ge=1, le=128)
    dppo_replay_output_path: str | None = Field(
        None,
        description="Optional JSONL path for sampled DPPO replay records.",
    )
    include_world_model_replay: bool = Field(
        False,
        description="Attach ECHO/RWML world-model payloads to DPPO replay records.",
    )
    rwml_history_window: int = Field(
        4,
        description="Prior command/observation pairs per RWML replay transition.",
        ge=0,
        le=64,
    )
    record_to_registry: bool = True


class DPPOTrainLogprobsSpec(BaseModel):
    environment_id: str
    attempt_index: int = Field(..., ge=0)
    token_logprobs: list[float] = Field(..., min_length=1)
    tokens: list[str] | None = None
    model: str | None = None
    base_url: str | None = None


class DPPOReplayEnrichRequest(BaseModel):
    input_path: str = Field(..., description="Existing DPPO replay JSONL artifact")
    output_path: str = Field(..., description="Output path for replay records with train logprobs")
    train_logprobs: list[DPPOTrainLogprobsSpec] = Field(..., min_length=1)
    divergence: Literal["binary_tv", "binary_kl"] = Field(
        "binary_tv", description="binary_tv or binary_kl"
    )
    threshold: float | None = Field(None, ge=0.0)


class DPPOSmokeLaunchPlanRequest(BaseModel):
    replay_path: str = Field(..., description="DPPO replay or scored replay JSONL")
    output_dir: str = Field(..., description="Directory for backend smoke artifacts")
    base_model: str = Field(..., min_length=1)
    backend: str = Field("auto", description="auto, verl, skyrl, tmax_open_instruct")
    max_steps: int = Field(1, ge=1, le=100)
    n_gpus_per_node: int = Field(1, ge=1, le=64)
    write_script: bool = True
    command_template: str | None = Field(
        None,
        description="Optional command template with {replay_path}, {output_dir}, {base_model}.",
    )
    echo_enabled: bool = False
    echo_aux_lambda: float = Field(0.05, ge=0.0)
    rwml_enabled: bool = False
    rwml_distance_threshold: float = Field(0.2, gt=0.0, le=2.0)
    rwml_easy_pass_rate_threshold: float = Field(0.8, ge=0.0, le=1.0)
    rwml_easy_keep_probability: float = Field(0.1, ge=0.0, le=1.0)
    rwml_history_window: int = Field(4, ge=0, le=64)
    rwml_embedding_model: str = ""
    rwml_kl_beta: float = Field(0.0, ge=0.0)


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
        report_dict = report.to_dict()
        if req.environment_evidence is not None:
            report_dict = combine_release_gate_evidence(
                report_dict,
                req.environment_evidence.model_dump(exclude_none=True),
            )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["report"] = report_dict
        _record_verdict(req.model_id, report_dict)
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
        "latest_environment_holdout_eval": profile.latest_environment_holdout_eval,
        "n_environment_holdout_evals": len(getattr(profile, "environment_holdout_evals", [])),
    }


# =============================================================================
# External benchmarks (run on the serving host; ingest results here)
# =============================================================================


@router.get("/benchmark-commands")
async def benchmark_commands(
    base_url: str,
    model: str,
    tasks: str | None = None,
    include: str = "forgetting,terminal_bench,harbor_terminal_bench,bfcl,swebench",
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


@router.post("/benchmarks/external-ingest")
async def ingest_external_benchmarks(req: ExternalBenchmarkIngestRequest):
    """Normalize standalone public benchmark results and record their scores."""

    try:
        report = service.ingest_external_benchmarks(
            req.results,
            benchmark_name=req.benchmark_name,
            source=req.source,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    recorded: list[str] = []
    if req.record_to_registry:
        try:
            from bashgym.models import get_registry

            recorded = service.record_external_benchmarks(get_registry(), req.model_id, report)
        except Exception as e:  # noqa: BLE001 - recording is best-effort
            logger.warning("Could not record external benchmark scores for %s: %s", req.model_id, e)

    return {
        "model_id": req.model_id,
        "report": report.to_dict(),
        "recorded": recorded,
    }


# =============================================================================
# BashGym executable environment pass@k
# =============================================================================


@router.post("/environments/passk", response_model=EnvironmentPassKResponse)
async def environment_passk(req: EnvironmentPassKRequest):
    """Compute verifier-backed pass@k from collected environment rollout attempts."""
    if any(k <= 0 for k in req.k_values):
        raise HTTPException(status_code=400, detail="k_values must be positive")
    try:
        report = service.run_environment_passk(
            req.environments,
            [attempt.model_dump() for attempt in req.attempts],
            k_values=req.k_values,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    recorded: list[str] = []
    if req.model_id and req.record_to_registry:
        try:
            from bashgym.models import get_registry

            recorded = service.record_environment_passk(get_registry(), req.model_id, report)
        except Exception as e:  # noqa: BLE001 - recording is best-effort
            logger.warning("Could not record environment pass@k for %s: %s", req.model_id, e)

    return EnvironmentPassKResponse(
        model_id=req.model_id,
        report=report.to_dict(),
        recorded=recorded,
    )


@router.post("/environments/holdout-gate", response_model=EnvironmentHoldoutGateResponse)
async def environment_holdout_gate(req: EnvironmentHoldoutGateRequest):
    """Compute a contamination-aware environment holdout pass@k gate."""

    if any(k <= 0 for k in req.k_values):
        raise HTTPException(status_code=400, detail="k_values must be positive")
    try:
        result = service.run_environment_holdout_gate(
            req.environments,
            [attempt.model_dump() for attempt in req.attempts],
            split_by=req.split_by,
            holdout_fraction=req.holdout_fraction,
            seed=req.seed,
            k_values=req.k_values,
            min_pass_at_1=req.min_pass_at_1,
            max_timeout_rate=req.max_timeout_rate,
            max_tamper_rate=req.max_tamper_rate,
            require_no_contamination=req.require_no_contamination,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    recorded: list[str] = []
    if req.model_id and req.record_to_registry:
        try:
            from bashgym.models import get_registry

            recorded = service.record_environment_holdout_gate(
                get_registry(), req.model_id, result
            )
        except Exception as e:  # noqa: BLE001 - recording is best-effort
            logger.warning("Could not record environment holdout gate for %s: %s", req.model_id, e)
    return EnvironmentHoldoutGateResponse(
        model_id=req.model_id,
        result=result,
        recorded=recorded,
    )


@router.post(
    "/environments/holdout-comparison",
    response_model=EnvironmentHoldoutComparisonResponse,
)
async def environment_holdout_comparison(req: EnvironmentHoldoutComparisonRequest):
    """Compare base vs candidate attempts on a contamination-aware environment holdout."""

    if any(k <= 0 for k in req.k_values):
        raise HTTPException(status_code=400, detail="k_values must be positive")
    try:
        result = service.run_environment_holdout_comparison_gate(
            req.environments,
            [attempt.model_dump() for attempt in req.base_attempts],
            [attempt.model_dump() for attempt in req.candidate_attempts],
            split_by=req.split_by,
            cluster_by=req.cluster_by,
            holdout_fraction=req.holdout_fraction,
            seed=req.seed,
            k_values=req.k_values,
            compare_k=req.compare_k,
            min_delta=req.min_delta,
            min_candidate_pass_at_1=req.min_candidate_pass_at_1,
            require_ci_excludes_zero=req.require_ci_excludes_zero,
            max_candidate_timeout_rate=req.max_candidate_timeout_rate,
            max_candidate_tamper_rate=req.max_candidate_tamper_rate,
            require_no_contamination=req.require_no_contamination,
            n_resamples=req.n_resamples,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return EnvironmentHoldoutComparisonResponse(result=result)


@router.post(
    "/environments/spurious-reward-control",
    response_model=EnvironmentSpuriousRewardControlResponse,
)
async def environment_spurious_reward_control(req: EnvironmentSpuriousRewardControlRequest):
    """Run a spurious-reward negative-control audit on the environment holdout."""

    if any(k <= 0 for k in req.k_values):
        raise HTTPException(status_code=400, detail="k_values must be positive")
    try:
        result = service.run_environment_spurious_reward_control(
            req.environments,
            [attempt.model_dump() for attempt in req.attempts],
            control_attempts=(
                [attempt.model_dump() for attempt in req.control_attempts]
                if req.control_attempts is not None
                else None
            ),
            split_by=req.split_by,
            holdout_fraction=req.holdout_fraction,
            seed=req.seed,
            k_values=req.k_values,
            n_trials=req.n_trials,
            random_pass_probability=req.random_pass_probability,
            min_observed_pass_at_1=req.min_observed_pass_at_1,
            max_control_pass_at_1=req.max_control_pass_at_1,
            min_lift_over_control=req.min_lift_over_control,
            require_no_contamination=req.require_no_contamination,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return EnvironmentSpuriousRewardControlResponse(result=result)


@router.post("/environments/local-rollout-passk", response_model=EnvironmentRolloutPassKResponse)
async def environment_local_rollout_passk(req: EnvironmentRolloutPassKRequest):
    """Run local command-script environment attempts, then compute verifier-backed pass@k."""
    if any(k <= 0 for k in req.k_values):
        raise HTTPException(status_code=400, detail="k_values must be positive")
    workspace_root = req.workspace_root or str(Path(tempfile.gettempdir()) / "bashgym_env_rollouts")
    try:
        report, rollouts = service.run_local_environment_rollout_passk(
            req.environments,
            [attempt.model_dump() for attempt in req.command_attempts],
            workspace_root=workspace_root,
            k_values=req.k_values,
            keep_workspace=req.keep_workspace,
            allow_dangerous_commands=req.allow_dangerous_commands,
            stop_on_error=req.stop_on_error,
        )
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    recorded: list[str] = []
    if req.model_id and req.record_to_registry:
        try:
            from bashgym.models import get_registry

            recorded = service.record_environment_passk(get_registry(), req.model_id, report)
        except Exception as e:  # noqa: BLE001 - recording is best-effort
            logger.warning("Could not record local rollout pass@k for %s: %s", req.model_id, e)

    return EnvironmentRolloutPassKResponse(
        model_id=req.model_id,
        report=report.to_dict(),
        attempts=[rollout.attempt.to_dict() for rollout in rollouts],
        rollouts=[rollout.to_dict() for rollout in rollouts],
        recorded=recorded,
        sampling_report=None,
        dppo_report=None,
    )


@router.post(
    "/environments/reward-hacking-canaries",
    response_model=EnvironmentCanarySuiteResponse,
)
async def environment_reward_hacking_canaries(req: EnvironmentCanarySuiteRequest):
    """Run built-in adversarial canaries against environment rollout guardrails."""

    workspace_root = req.workspace_root or str(
        Path(tempfile.gettempdir()) / "bashgym_env_canaries"
    )
    categories = req.categories or None
    try:
        canaries, rollouts, summary = service.run_reward_hacking_canary_suite(
            workspace_root=workspace_root,
            categories=categories,
            keep_workspace=req.keep_workspace,
        )
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return EnvironmentCanarySuiteResponse(
        summary=summary,
        canaries=[canary.to_dict() for canary in canaries],
        rollouts=[rollout.to_dict() for rollout in rollouts],
    )


@router.post("/environments/model-rollout-passk", response_model=EnvironmentRolloutPassKResponse)
async def environment_model_rollout_passk(
    request: Request, req: EnvironmentModelRolloutPassKRequest
):
    """Run served-model environment attempts, then compute verifier-backed pass@k."""
    if any(k <= 0 for k in req.k_values):
        raise HTTPException(status_code=400, detail="k_values must be positive")
    endpoint = _resolve(request, req.endpoint, "model")
    workspace_root = req.workspace_root or str(Path(tempfile.gettempdir()) / "bashgym_env_rollouts")
    try:
        report, rollouts, sampling_report = service.run_model_environment_rollout_passk(
            req.environments,
            endpoint,
            workspace_root=workspace_root,
            attempts_per_environment=req.attempts_per_environment,
            k_values=req.k_values,
            keep_workspace=req.keep_workspace,
            allow_dangerous_commands=req.allow_dangerous_commands,
            stop_on_error=req.stop_on_error,
            max_tool_calls=req.max_tool_calls,
            max_observation_chars=req.max_observation_chars,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            request_timeout=req.request_timeout,
            use_tool_calling=req.use_tool_calling,
            capture_logprobs=req.capture_logprobs,
            top_logprobs=req.top_logprobs,
            filter_zero_std_groups=req.filter_zero_std_groups,
            active_sampling=req.active_sampling,
            target_prompt_groups=req.target_prompt_groups,
        )
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    recorded: list[str] = []
    if req.model_id and req.record_to_registry:
        try:
            from bashgym.models import get_registry

            recorded = service.record_environment_passk(get_registry(), req.model_id, report)
        except Exception as e:  # noqa: BLE001 - recording is best-effort
            logger.warning("Could not record model rollout pass@k for %s: %s", req.model_id, e)

    dppo_replay = None
    if req.dppo_replay_output_path:
        specs = [EnvironmentSpec.from_dict(environment) for environment in req.environments]
        dppo_replay = write_dppo_replay_jsonl(
            req.dppo_replay_output_path,
            specs,
            rollouts,
            include_world_model=req.include_world_model_replay,
            history_window=req.rwml_history_window,
        )

    return EnvironmentRolloutPassKResponse(
        model_id=req.model_id,
        report=report.to_dict(),
        attempts=[rollout.attempt.to_dict() for rollout in rollouts],
        rollouts=[rollout.to_dict() for rollout in rollouts],
        recorded=recorded,
        sampling_report=sampling_report,
        dppo_report=service.summarize_dppo_readiness(rollouts),
        dppo_replay=dppo_replay,
    )


@router.post("/environments/dppo-replay/enrich")
async def enrich_environment_dppo_replay(req: DPPOReplayEnrichRequest):
    """Attach train-policy logprobs to a DPPO replay JSONL artifact."""

    scored_by_attempt = {
        (spec.environment_id, spec.attempt_index): {
            "token_logprobs": spec.token_logprobs,
            "tokens": spec.tokens,
            "model": spec.model,
            "base_url": spec.base_url,
        }
        for spec in req.train_logprobs
    }

    def scorer(record: dict[str, Any]) -> dict[str, Any]:
        key = (str(record.get("environment_id")), int(record.get("attempt_index") or 0))
        if key not in scored_by_attempt:
            raise ValueError(
                "missing train logprobs for "
                f"environment_id={key[0]!r} attempt_index={key[1]}"
            )
        return scored_by_attempt[key]

    try:
        summary = enrich_dppo_replay_jsonl(
            req.input_path,
            req.output_path,
            scorer,
            divergence=req.divergence,
            threshold=req.threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"dppo_replay": summary}


@router.post("/environments/dppo-replay/smoke-plan")
async def plan_environment_dppo_smoke(req: DPPOSmokeLaunchPlanRequest):
    """Prepare a backend-specific DPPO smoke-launch command/script."""

    try:
        config = DPPOSmokeLaunchConfig(
            replay_path=Path(req.replay_path),
            output_dir=Path(req.output_dir),
            base_model=req.base_model,
            backend=req.backend,
            max_steps=req.max_steps,
            n_gpus_per_node=req.n_gpus_per_node,
            write_script=req.write_script,
            command_template=req.command_template,
            echo_enabled=req.echo_enabled,
            echo_aux_lambda=req.echo_aux_lambda,
            rwml_enabled=req.rwml_enabled,
            rwml_distance_threshold=req.rwml_distance_threshold,
            rwml_easy_pass_rate_threshold=req.rwml_easy_pass_rate_threshold,
            rwml_easy_keep_probability=req.rwml_easy_keep_probability,
            rwml_history_window=req.rwml_history_window,
            rwml_embedding_model=req.rwml_embedding_model,
            rwml_kl_beta=req.rwml_kl_beta,
        )
        plan = prepare_dppo_smoke_launch(config)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"plan": plan.to_dict()}
