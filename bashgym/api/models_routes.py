"""
Model Registry API Routes

Provides REST API endpoints for:
- Listing and querying trained models
- Getting model profiles
- Updating model metadata
- Comparing models
- Leaderboard and trends
- Custom evaluation set management
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from bashgym.models import (
    get_registry,
    ModelProfile,
    get_eval_generator,
    CustomEvalRunner,
    CustomEvalSet,
)


# Pydantic models for API

class ModelSummary(BaseModel):
    """Summary of a model for list views."""
    model_id: str
    display_name: str
    description: str
    tags: List[str]
    starred: bool
    base_model: str
    training_strategy: str
    status: str
    created_at: str
    # Quick stats
    custom_eval_pass_rate: Optional[float] = None
    benchmark_avg_score: Optional[float] = None
    model_size_display: str
    inference_latency_ms: Optional[float] = None
    training_duration_display: str


class ModelProfileResponse(BaseModel):
    """Full model profile response."""
    # Identity
    model_id: str
    run_id: str
    display_name: str
    description: str
    tags: List[str]
    starred: bool
    created_at: str

    # Lineage
    base_model: str
    training_strategy: str
    teacher_model: Optional[str] = None
    training_traces: List[str]
    parent_model: Optional[str] = None
    training_repos: List[str]

    # Training
    config: Dict[str, Any]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float
    training_duration_display: str
    loss_curve: List[Dict[str, Any]]
    final_metrics: Dict[str, float]

    # Artifacts
    artifacts: Dict[str, Any]
    model_dir: str

    # Evaluations
    benchmarks: Dict[str, Any]
    custom_evals: Dict[str, Any]
    evaluation_history: List[Dict[str, Any]]

    # Operational
    model_size_bytes: int
    model_size_display: str
    model_size_params: Optional[str] = None
    inference_latency_ms: Optional[float] = None
    status: str
    deployed_to: Optional[str] = None

    # Computed
    custom_eval_pass_rate: Optional[float] = None
    benchmark_avg_score: Optional[float] = None


class ModelListResponse(BaseModel):
    """Response for model list endpoint."""
    models: List[ModelSummary]
    total: int


class ModelUpdateRequest(BaseModel):
    """Request to update model metadata."""
    display_name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    starred: Optional[bool] = None


class CompareRequest(BaseModel):
    """Request to compare multiple models."""
    model_ids: List[str]
    metrics: Optional[List[str]] = None


class CompareResponse(BaseModel):
    """Response for model comparison."""
    models: Dict[str, Dict[str, Any]]


class LeaderboardEntry(BaseModel):
    """Entry in the leaderboard."""
    rank: int
    model_id: str
    display_name: str
    value: float
    base_model: str
    strategy: str


class LeaderboardResponse(BaseModel):
    """Response for leaderboard endpoint."""
    metric: str
    entries: List[LeaderboardEntry]


class TrendDataPoint(BaseModel):
    """Data point for trend charts."""
    timestamp: str
    model_id: str
    display_name: str
    value: float


class TrendsResponse(BaseModel):
    """Response for trends endpoint."""
    metric: str
    data: List[TrendDataPoint]


# Helper functions

def profile_to_summary(profile: ModelProfile) -> ModelSummary:
    """Convert ModelProfile to ModelSummary."""
    return ModelSummary(
        model_id=profile.model_id,
        display_name=profile.display_name,
        description=profile.description,
        tags=profile.tags,
        starred=profile.starred,
        base_model=profile.base_model,
        training_strategy=profile.training_strategy,
        status=profile.status,
        created_at=profile.created_at.isoformat(),
        custom_eval_pass_rate=profile.custom_eval_pass_rate,
        benchmark_avg_score=profile.benchmark_avg_score,
        model_size_display=profile.model_size_display,
        inference_latency_ms=profile.inference_latency_ms,
        training_duration_display=profile.training_duration_display,
    )


def profile_to_response(profile: ModelProfile) -> ModelProfileResponse:
    """Convert ModelProfile to full API response."""
    return ModelProfileResponse(
        model_id=profile.model_id,
        run_id=profile.run_id,
        display_name=profile.display_name,
        description=profile.description,
        tags=profile.tags,
        starred=profile.starred,
        created_at=profile.created_at.isoformat(),
        base_model=profile.base_model,
        training_strategy=profile.training_strategy,
        teacher_model=profile.teacher_model,
        training_traces=profile.training_traces,
        parent_model=profile.parent_model,
        training_repos=profile.training_repos,
        config=profile.config,
        started_at=profile.started_at.isoformat() if profile.started_at else None,
        completed_at=profile.completed_at.isoformat() if profile.completed_at else None,
        duration_seconds=profile.duration_seconds,
        training_duration_display=profile.training_duration_display,
        loss_curve=profile.loss_curve,
        final_metrics=profile.final_metrics,
        artifacts=profile.artifacts.to_dict(),
        model_dir=profile.model_dir,
        benchmarks={k: v.to_dict() for k, v in profile.benchmarks.items()},
        custom_evals={k: v.to_dict() for k, v in profile.custom_evals.items()},
        evaluation_history=[e.to_dict() for e in profile.evaluation_history],
        model_size_bytes=profile.model_size_bytes,
        model_size_display=profile.model_size_display,
        model_size_params=profile.model_size_params,
        inference_latency_ms=profile.inference_latency_ms,
        status=profile.status,
        deployed_to=profile.deployed_to,
        custom_eval_pass_rate=profile.custom_eval_pass_rate,
        benchmark_avg_score=profile.benchmark_avg_score,
    )


# Router

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models(
    strategy: Optional[str] = Query(None, description="Filter by training strategy"),
    base_model: Optional[str] = Query(None, description="Filter by base model (partial match)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    starred: bool = Query(False, description="Only show starred models"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List trained models with filtering and sorting.

    Returns model summaries with quick stats for the model browser.
    """
    registry = get_registry()

    # Parse tags
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]

    profiles = registry.list(
        strategy=strategy,
        base_model=base_model,
        status=status,
        tags=tag_list,
        starred_only=starred,
        sort_by=sort_by,
        sort_order=sort_order,  # type: ignore
        limit=limit,
        offset=offset,
    )

    # Get total count (without pagination)
    total = len(registry.list(
        strategy=strategy,
        base_model=base_model,
        status=status,
        tags=tag_list,
        starred_only=starred,
    ))

    return ModelListResponse(
        models=[profile_to_summary(p) for p in profiles],
        total=total,
    )


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    metric: str = Query("custom_eval_pass_rate", description="Metric to rank by"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get ranked leaderboard of models by a metric.

    Available metrics: custom_eval_pass_rate, benchmark_avg_score, benchmark_<name>
    """
    registry = get_registry()
    entries = registry.leaderboard(metric=metric, limit=limit)

    return LeaderboardResponse(
        metric=metric,
        entries=[LeaderboardEntry(**e) for e in entries],
    )


@router.get("/trends", response_model=TrendsResponse)
async def get_trends(
    metric: str = Query("benchmark_avg_score", description="Metric to track"),
    days: int = Query(30, ge=1, le=365),
):
    """
    Get metric trends over time for charting.
    """
    registry = get_registry()
    data = registry.trends(metric=metric, days=days)

    return TrendsResponse(
        metric=metric,
        data=[TrendDataPoint(**d) for d in data],
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """
    Compare multiple models side-by-side.

    Returns metrics for each model for comparison tables/charts.
    """
    registry = get_registry()
    result = registry.compare(
        model_ids=request.model_ids,
        metrics=request.metrics,
    )

    return CompareResponse(models=result)


@router.get("/{model_id}", response_model=ModelProfileResponse)
async def get_model(model_id: str):
    """
    Get full profile for a specific model.

    Returns all model details including training config, artifacts, and evaluations.
    """
    registry = get_registry()
    profile = registry.get(model_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    return profile_to_response(profile)


@router.post("/{model_id}", response_model=ModelProfileResponse)
async def update_model(model_id: str, request: ModelUpdateRequest):
    """
    Update model metadata.

    Editable fields: display_name, description, tags, starred
    """
    registry = get_registry()

    updates = {}
    if request.display_name is not None:
        updates["display_name"] = request.display_name
    if request.description is not None:
        updates["description"] = request.description
    if request.tags is not None:
        updates["tags"] = request.tags
    if request.starred is not None:
        updates["starred"] = request.starred

    profile = registry.update(model_id, updates)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    return profile_to_response(profile)


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    archive: bool = Query(True, description="Archive instead of delete"),
):
    """
    Delete or archive a model.

    By default, archives the model (marks as archived but keeps files).
    Set archive=false to remove from registry entirely.
    """
    registry = get_registry()
    success = registry.delete(model_id, archive=archive)

    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"status": "ok", "archived": archive}


@router.post("/{model_id}/star")
async def star_model(
    model_id: str,
    starred: bool = Query(True, description="Star or unstar"),
):
    """
    Star or unstar a model.

    Starred models appear at the top of lists.
    """
    registry = get_registry()
    profile = registry.star(model_id, starred=starred)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"status": "ok", "starred": profile.starred}


@router.post("/{model_id}/evaluate")
async def trigger_evaluation(model_id: str):
    """
    Trigger evaluation of a model.

    Runs all configured benchmarks and custom evaluations.
    Returns immediately - results available via WebSocket or polling.
    """
    registry = get_registry()
    profile = registry.get(model_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    # TODO: Implement async evaluation trigger
    # This would queue an evaluation job that runs benchmarks
    # and updates the profile when complete

    return {
        "status": "queued",
        "model_id": model_id,
        "message": "Evaluation queued - results will be available shortly"
    }


@router.get("/{model_id}/artifacts")
async def list_artifacts(model_id: str):
    """
    List all artifacts for a model.

    Returns paths to checkpoints, merged model, GGUF exports, etc.
    """
    registry = get_registry()
    profile = registry.get(model_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "model_id": model_id,
        "model_dir": profile.model_dir,
        "artifacts": profile.artifacts.to_dict(),
    }


@router.post("/{model_id}/rescan")
async def rescan_model(model_id: str):
    """
    Rescan a model's directory to update artifacts and metadata.

    Useful after manual changes or new exports.
    """
    registry = get_registry()
    profile = registry.get(model_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    # Force rescan of the entire registry
    registry.scan(force_rescan=True)

    # Get updated profile
    updated = registry.get(model_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Model not found after rescan")

    return profile_to_response(updated)


# ==============================================================================
# Custom Evaluation Set Endpoints
# ==============================================================================

class EvalSetSummary(BaseModel):
    """Summary of a custom eval set."""
    eval_set_id: str
    name: str
    description: str
    num_cases: int
    generation_mode: str  # replay, variation, or manual
    source_traces: List[str]
    created_at: str


class EvalCaseResponse(BaseModel):
    """An eval case in the set."""
    case_id: str
    name: str
    description: str
    system_prompt: Optional[str] = None
    user_prompt: str
    expected_behavior: str
    verification: Dict[str, Any]
    source_trace_id: Optional[str] = None


class EvalSetResponse(BaseModel):
    """Full eval set response."""
    eval_set_id: str
    name: str
    description: str
    cases: List[EvalCaseResponse]
    generation_mode: str
    source_traces: List[str]
    created_at: str


class GenerateEvalSetRequest(BaseModel):
    """Request to generate an eval set from gold traces."""
    name: str
    description: Optional[str] = None
    trace_ids: Optional[List[str]] = None  # None = use all gold traces
    mode: str = "both"  # replay, variation, or both
    max_cases: int = 50
    include_failed_traces: bool = False


class RunEvalRequest(BaseModel):
    """Request to run an eval set against a model."""
    eval_set_id: str
    max_tokens: int = 4096
    temperature: float = 0.0


class RunEvalResponse(BaseModel):
    """Response from running an eval."""
    model_id: str
    eval_set_id: str
    status: str
    passed: int
    failed: int
    total: int
    pass_rate: float
    results: List[Dict[str, Any]]


# Eval sets router (nested under /api/models)
eval_router = APIRouter(prefix="/eval-sets", tags=["eval-sets"])


@eval_router.get("", response_model=List[EvalSetSummary])
async def list_eval_sets():
    """
    List all custom evaluation sets.
    """
    generator = get_eval_generator()
    eval_sets = generator.list_eval_sets()

    return [
        EvalSetSummary(
            eval_set_id=es.eval_set_id,
            name=es.name,
            description=es.description,
            num_cases=len(es.cases),
            generation_mode=es.generation_mode,
            source_traces=es.source_traces,
            created_at=es.created_at.isoformat(),
        )
        for es in eval_sets
    ]


@eval_router.get("/{eval_set_id}", response_model=EvalSetResponse)
async def get_eval_set(eval_set_id: str):
    """
    Get a specific eval set with all its cases.
    """
    generator = get_eval_generator()
    eval_set = generator.get_eval_set(eval_set_id)

    if not eval_set:
        raise HTTPException(status_code=404, detail="Eval set not found")

    return EvalSetResponse(
        eval_set_id=eval_set.eval_set_id,
        name=eval_set.name,
        description=eval_set.description,
        cases=[
            EvalCaseResponse(
                case_id=case.case_id,
                name=case.name,
                description=case.description,
                system_prompt=case.system_prompt,
                user_prompt=case.user_prompt,
                expected_behavior=case.expected_behavior,
                verification=case.verification.to_dict() if hasattr(case.verification, 'to_dict') else {
                    "method": case.verification.method,
                    "expected_output": case.verification.expected_output,
                    "test_command": case.verification.test_command,
                    "check_files": case.verification.check_files,
                    "llm_criteria": case.verification.llm_criteria,
                },
                source_trace_id=case.source_trace_id,
            )
            for case in eval_set.cases
        ],
        generation_mode=eval_set.generation_mode,
        source_traces=eval_set.source_traces,
        created_at=eval_set.created_at.isoformat(),
    )


@eval_router.post("/generate")
async def generate_eval_set(request: GenerateEvalSetRequest, background_tasks: BackgroundTasks):
    """
    Generate a new eval set from gold traces.

    Modes:
    - replay: Exact task replay (verify same inputs produce similar outputs)
    - variation: Generate variations of tasks (test generalization)
    - both: Include both replay and variation cases
    """
    generator = get_eval_generator()

    # Generate the eval set (this may take a while)
    try:
        eval_set = generator.generate_from_traces(
            name=request.name,
            description=request.description or f"Auto-generated eval set from {request.mode} mode",
            trace_ids=request.trace_ids,
            mode=request.mode,
            max_cases=request.max_cases,
            include_failed_traces=request.include_failed_traces,
        )

        return {
            "status": "created",
            "eval_set_id": eval_set.eval_set_id,
            "name": eval_set.name,
            "num_cases": len(eval_set.cases),
            "message": f"Generated {len(eval_set.cases)} eval cases"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@eval_router.delete("/{eval_set_id}")
async def delete_eval_set(eval_set_id: str):
    """
    Delete an eval set.
    """
    generator = get_eval_generator()
    success = generator.delete_eval_set(eval_set_id)

    if not success:
        raise HTTPException(status_code=404, detail="Eval set not found")

    return {"status": "deleted", "eval_set_id": eval_set_id}


# Add run-eval endpoint to model routes
@router.post("/{model_id}/run-eval", response_model=RunEvalResponse)
async def run_eval_on_model(model_id: str, request: RunEvalRequest):
    """
    Run an evaluation set against a model.

    Returns pass/fail results for each case.
    """
    registry = get_registry()
    profile = registry.get(model_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    generator = get_eval_generator()
    eval_set = generator.get_eval_set(request.eval_set_id)

    if not eval_set:
        raise HTTPException(status_code=404, detail="Eval set not found")

    # Run the evaluation
    runner = CustomEvalRunner(model_path=profile.model_dir)

    try:
        results = runner.run_eval_set(
            eval_set=eval_set,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Calculate summary stats
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0.0

        # Update model profile with eval results
        from bashgym.models import CustomEvalResult
        eval_result = CustomEvalResult(
            eval_set_id=request.eval_set_id,
            eval_type=eval_set.generation_mode,
            passed=passed,
            total=len(results),
            pass_rate=pass_rate,
            evaluated_at=datetime.now(),
            case_results=[r.to_dict() if hasattr(r, 'to_dict') else {
                "case_id": r.case_id,
                "passed": r.passed,
                "error": r.error,
            } for r in results],
        )
        profile.custom_evals[request.eval_set_id] = eval_result
        profile.save()

        return RunEvalResponse(
            model_id=model_id,
            eval_set_id=request.eval_set_id,
            status="completed",
            passed=passed,
            failed=failed,
            total=len(results),
            pass_rate=pass_rate,
            results=[{
                "case_id": r.case_id,
                "case_name": next((c.name for c in eval_set.cases if c.case_id == r.case_id), r.case_id),
                "passed": r.passed,
                "output": r.output[:500] if r.output else None,  # Truncate long outputs
                "error": r.error,
            } for r in results],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ==============================================================================
# Deploy and Download Endpoints
# ==============================================================================

class DeployOllamaRequest(BaseModel):
    """Request to deploy a model to Ollama."""
    model_name: Optional[str] = None  # Default: use display_name
    quantization: str = "q4_k_m"


@router.post("/{model_id}/deploy-ollama")
async def deploy_to_ollama(model_id: str, request: DeployOllamaRequest):
    """
    Deploy a model to local Ollama.

    Creates a GGUF export if needed, then registers with Ollama.
    """
    import subprocess
    import tempfile

    registry = get_registry()
    profile = registry.get(model_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    # Find or create GGUF
    gguf_path = None
    for export in profile.artifacts.gguf_exports:
        if export.get("quantization") == request.quantization:
            gguf_path = export.get("path")
            break

    if not gguf_path:
        # Check if there's a merged model to convert
        if profile.artifacts.merged_path:
            # TODO: Trigger GGUF export
            raise HTTPException(
                status_code=400,
                detail=f"No GGUF export found with quantization {request.quantization}. Export first."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="No merged model found. Complete training first."
            )

    # Verify GGUF file exists
    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        raise HTTPException(status_code=404, detail=f"GGUF file not found: {gguf_path}")

    # Create Modelfile
    model_name = request.model_name or profile.display_name.lower().replace(" ", "-")
    modelfile_content = f'''FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER num_ctx 8192

SYSTEM """You are a helpful coding assistant trained with Bash Gym."""
'''

    try:
        # Write Modelfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name

        # Create model in Ollama
        result = subprocess.run(
            ['ollama', 'create', model_name, '-f', modelfile_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Clean up
        Path(modelfile_path).unlink(missing_ok=True)

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama create failed: {result.stderr}"
            )

        # Update profile
        profile.deployed_to = f"ollama:{model_name}"
        profile.save()

        return {
            "status": "deployed",
            "model_name": model_name,
            "message": f"Model deployed to Ollama as '{model_name}'. Run with: ollama run {model_name}"
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Ollama create timed out")
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not installed. Install from https://ollama.ai"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/download")
async def download_artifact(
    model_id: str,
    path: str = Query(..., description="Path to the artifact file"),
):
    """
    Get download info for a model artifact.

    Returns the file path that can be used for download.
    """
    from fastapi.responses import FileResponse

    registry = get_registry()
    profile = registry.get(model_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Model not found")

    # Validate path is within model directory (security)
    artifact_path = Path(path)
    model_dir = Path(profile.model_dir)

    # Normalize paths for comparison
    try:
        artifact_resolved = artifact_path.resolve()
        model_resolved = model_dir.resolve()

        # Check if artifact is within model directory or its subdirectories
        if not str(artifact_resolved).startswith(str(model_resolved)):
            raise HTTPException(status_code=403, detail="Access denied: path outside model directory")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(
        path=str(artifact_path),
        filename=artifact_path.name,
        media_type="application/octet-stream"
    )


# Register eval router
router.include_router(eval_router)
