# bashgym/api/factory_routes.py
"""API routes for synthetic data generation and the Data Factory."""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/factory", tags=["factory"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SyntheticGenerateRequest(BaseModel):
    """Request to start synthetic data generation."""
    strategy: str = "trace_seeded"
    repo_filter: str = "all"
    selected_repos: List[str] = []
    preset: str = "balanced"
    target_examples: Optional[int] = None
    multiplier: Optional[int] = None
    provider: str = "nim"
    merge_mode: str = "mixed"


class SyntheticGenerateResponse(BaseModel):
    """Response from starting synthetic generation."""
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response with job status and progress."""
    job_id: str
    status: str
    progress: Dict[str, int]
    config: Optional[Dict] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# In-Memory Job Storage
# =============================================================================

# Job tracking (would use Redis/DB in production)
generation_jobs: Dict[str, Dict] = {}


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/synthetic/generate", response_model=SyntheticGenerateResponse)
async def start_synthetic_generation(
    request: SyntheticGenerateRequest,
    background_tasks: BackgroundTasks
):
    """Start a synthetic data generation job.

    Accepts configuration for how to generate synthetic training data
    and returns a job ID that can be polled for status.
    """
    job_id = f"gen_{uuid.uuid4().hex[:8]}"

    generation_jobs[job_id] = {
        "status": "queued",
        "progress": {"current": 0, "total": 0},
        "config": request.model_dump()
    }

    # Add background task for async generation
    background_tasks.add_task(run_generation_job, job_id, request)

    return SyntheticGenerateResponse(job_id=job_id, status="queued")


@router.get("/synthetic/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a synthetic generation job."""
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = generation_jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        config=job.get("config"),
        output_dir=job.get("output_dir"),
        error=job.get("error")
    )


@router.get("/synthetic/jobs", response_model=List[JobStatusResponse])
async def list_jobs():
    """List all synthetic generation jobs."""
    return [
        JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            config=job.get("config"),
            output_dir=job.get("output_dir"),
            error=job.get("error")
        )
        for job_id, job in generation_jobs.items()
    ]


@router.get("/synthetic/presets")
async def get_presets():
    """Get available generation presets."""
    from bashgym.factory.synthetic_generator import PRESETS

    return {
        name: {
            "label": preset.label,
            "description": preset.description,
            "target_examples": preset.target_examples,
        }
        for name, preset in PRESETS.items()
    }


# =============================================================================
# Background Task
# =============================================================================

async def run_generation_job(job_id: str, request: SyntheticGenerateRequest):
    """Background task to run synthetic data generation."""
    from bashgym.factory.synthetic_generator import SyntheticGenerator, PRESETS
    from bashgym.factory.pattern_extractor import PatternExtractor

    try:
        generation_jobs[job_id]["status"] = "running"

        # Load traces based on repo filter
        traces = await load_traces_for_generation(
            repo_filter=request.repo_filter,
            selected_repos=request.selected_repos
        )

        if not traces:
            logger.warning(f"No traces found for job {job_id}")
            generation_jobs[job_id]["status"] = "completed"
            generation_jobs[job_id]["progress"] = {"current": 0, "total": 0}
            return

        # Extract patterns from traces
        extractor = PatternExtractor()
        repo_name = request.selected_repos[0] if request.selected_repos else "all"
        patterns = extractor.extract_patterns(traces, repo_name=repo_name)

        # Calculate target count
        preset = PRESETS.get(request.preset)
        target = request.target_examples or (preset.target_examples if preset else 500)

        generation_jobs[job_id]["progress"]["total"] = target

        # Extract seed prompts from traces
        seed_prompts = extract_seed_prompts(traces)

        # Generate synthetic tasks
        generator = SyntheticGenerator()

        def on_progress(current: int, total: int):
            generation_jobs[job_id]["progress"]["current"] = current

        if request.strategy == "trace_seeded":
            tasks = await generator.generate_batch(
                patterns=patterns,
                seed_prompts=seed_prompts,
                count=target,
                provider=request.provider,
                on_progress=on_progress
            )
        elif request.strategy == "augmented":
            seed_examples = [{"prompt": p, "repo": repo_name} for p in seed_prompts]
            tasks = await generator.generate_augmented(
                seed_examples=seed_examples,
                variations_per_seed=request.multiplier or 3,
                provider=request.provider
            )
        elif request.strategy == "schema_driven":
            # Build schema from trace data
            repo_schema = {
                "name": repo_name,
                "structure": {},
                "frameworks": list(patterns.framework_hints)
            }
            tasks = await generator.generate_from_schema(
                repo_schema=repo_schema,
                count=target,
                provider=request.provider
            )
        else:
            raise ValueError(f"Unknown strategy: {request.strategy}")

        # Export to NeMo format
        output_dir = Path(f"data/synthetic/{job_id}")
        generator.export_to_nemo(tasks, output_dir)

        generation_jobs[job_id]["status"] = "completed"
        generation_jobs[job_id]["output_dir"] = str(output_dir)
        generation_jobs[job_id]["progress"]["current"] = len(tasks)

        logger.info(f"Job {job_id} completed: {len(tasks)} tasks generated")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        generation_jobs[job_id]["status"] = "failed"
        generation_jobs[job_id]["error"] = str(e)


async def load_traces_for_generation(
    repo_filter: str,
    selected_repos: List[str]
) -> List[Dict]:
    """Load traces from gold_traces directory based on filter."""
    import json
    from pathlib import Path

    traces = []
    traces_dir = Path("data/gold_traces")

    if not traces_dir.exists():
        logger.warning(f"Traces directory not found: {traces_dir}")
        return traces

    for trace_file in traces_dir.glob("*.json"):
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace = json.load(f)

            # Filter by repo if needed
            if repo_filter == "single" and selected_repos:
                repo = trace.get("primary_repo", {}).get("name", "")
                if repo not in selected_repos:
                    continue
            elif repo_filter == "selected" and selected_repos:
                repo = trace.get("primary_repo", {}).get("name", "")
                if repo not in selected_repos:
                    continue

            traces.append(trace)

        except Exception as e:
            logger.warning(f"Failed to load trace {trace_file}: {e}")

    return traces


def extract_seed_prompts(traces: List[Dict]) -> List[str]:
    """Extract prompts from traces to use as seeds."""
    prompts = []
    for trace in traces:
        # Try different prompt locations
        prompt = trace.get("initial_prompt", "")
        if not prompt:
            prompt = trace.get("prompt", "")
        if not prompt and "messages" in trace:
            # Look for first user message
            for msg in trace.get("messages", []):
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break
        if prompt:
            prompts.append(prompt)
    return prompts


# =============================================================================
# DataDesigner Integration Endpoints
# =============================================================================

class DesignerPreviewRequest(BaseModel):
    """Request to preview DataDesigner pipeline output."""
    pipeline: str = "coding_agent_sft"
    num_records: int = 5
    provider: str = "nvidia"
    provider_endpoint: str = "https://integrate.api.nvidia.com/v1"
    text_model: Optional[str] = None
    code_model: Optional[str] = None
    judge_model: Optional[str] = None


class DesignerCreateRequest(BaseModel):
    """Request to start full DataDesigner generation."""
    pipeline: str = "coding_agent_sft"
    num_records: int = 100
    seed_source: Optional[str] = None
    seed_type: str = "traces"  # traces, huggingface, file, unstructured
    column_mapping: Optional[Dict[str, str]] = None
    provider: str = "nvidia"
    provider_endpoint: str = "https://integrate.api.nvidia.com/v1"
    text_model: Optional[str] = None
    code_model: Optional[str] = None
    judge_model: Optional[str] = None
    output_dir: Optional[str] = None
    export_nemo: bool = True
    train_val_split: float = 0.9


class DesignerValidateRequest(BaseModel):
    """Request to validate a DataDesigner pipeline config."""
    pipeline: str = "coding_agent_sft"


class DesignerJobResponse(BaseModel):
    """Response from a DataDesigner job."""
    job_id: str
    status: str
    pipeline: str
    num_records: int
    progress: Optional[Dict[str, int]] = None
    output_dir: Optional[str] = None
    export_result: Optional[Dict] = None
    error: Optional[str] = None


class DesignerPipelineInfo(BaseModel):
    """Info about an available DataDesigner pipeline."""
    name: str
    description: str
    columns: List[str]


class DesignerHuggingFaceRequest(BaseModel):
    """Request to generate from a HuggingFace dataset."""
    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    num_records: int = 100
    pipeline: str = "coding_agent_sft"
    column_mapping: Optional[Dict[str, str]] = None
    provider: str = "nvidia"


class DesignerPushToHubRequest(BaseModel):
    """Request to publish dataset to HuggingFace Hub."""
    job_id: str
    repo_id: str
    private: bool = True


# DataDesigner job tracking
designer_jobs: Dict[str, Dict] = {}


@router.post("/designer/preview")
async def designer_preview(request: DesignerPreviewRequest):
    """Preview generated data with any pipeline config.

    Returns a small sample of generated records for inspection
    before committing to a full generation run.
    """
    try:
        from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig

        config = PipelineConfig(
            pipeline=request.pipeline,
            provider=request.provider,
            provider_endpoint=request.provider_endpoint,
            num_records=request.num_records,
        )
        if request.text_model:
            config.text_model = request.text_model
        if request.code_model:
            config.code_model = request.code_model
        if request.judge_model:
            config.judge_model = request.judge_model

        pipeline = DataDesignerPipeline(config)
        df = pipeline.preview(num_records=request.num_records)

        return {
            "records": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "count": len(df),
        }
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Designer preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/designer/create", response_model=DesignerJobResponse)
async def designer_create(
    request: DesignerCreateRequest,
    background_tasks: BackgroundTasks,
):
    """Start a full DataDesigner generation job.

    Runs in the background. Poll /designer/jobs/{job_id} for status.
    """
    job_id = f"dd_{uuid.uuid4().hex[:8]}"

    designer_jobs[job_id] = {
        "status": "queued",
        "pipeline": request.pipeline,
        "num_records": request.num_records,
        "progress": {"current": 0, "total": request.num_records},
        "config": request.model_dump(),
    }

    background_tasks.add_task(run_designer_job, job_id, request)

    return DesignerJobResponse(
        job_id=job_id,
        status="queued",
        pipeline=request.pipeline,
        num_records=request.num_records,
    )


@router.get("/designer/jobs/{job_id}", response_model=DesignerJobResponse)
async def designer_job_status(job_id: str):
    """Get DataDesigner generation job progress."""
    if job_id not in designer_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = designer_jobs[job_id]
    return DesignerJobResponse(
        job_id=job_id,
        status=job["status"],
        pipeline=job["pipeline"],
        num_records=job["num_records"],
        progress=job.get("progress"),
        output_dir=job.get("output_dir"),
        export_result=job.get("export_result"),
        error=job.get("error"),
    )


@router.get("/designer/pipelines")
async def list_designer_pipelines():
    """List available DataDesigner pipeline builders."""
    try:
        from bashgym.factory.designer_pipelines import PIPELINES
    except ImportError:
        return {"pipelines": [], "available": False}

    pipelines = []
    for name, builder_fn in PIPELINES.items():
        doc = builder_fn.__doc__ or ""
        first_line = doc.strip().split("\n")[0] if doc.strip() else name
        pipelines.append(DesignerPipelineInfo(
            name=name,
            description=first_line,
            columns=[],  # Would need to introspect builder
        ))

    return {"pipelines": [p.model_dump() for p in pipelines], "available": True}


@router.post("/designer/validate")
async def designer_validate(request: DesignerValidateRequest):
    """Validate a DataDesigner pipeline config without running generation."""
    try:
        from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig

        config = PipelineConfig(pipeline=request.pipeline)
        pipeline = DataDesignerPipeline(config)
        result = pipeline.validate()
        return result
    except ImportError as e:
        return {"valid": False, "errors": [str(e)], "columns": []}
    except Exception as e:
        return {"valid": False, "errors": [str(e)], "columns": []}


@router.post("/designer/from-hf")
async def designer_from_huggingface(
    request: DesignerHuggingFaceRequest,
    background_tasks: BackgroundTasks,
):
    """Generate training data from a HuggingFace dataset.

    Starts a background job that downloads the HF dataset, uses it as
    seed data for the DataDesigner pipeline, and exports results.
    """
    job_id = f"dd_hf_{uuid.uuid4().hex[:8]}"

    designer_jobs[job_id] = {
        "status": "queued",
        "pipeline": request.pipeline,
        "num_records": request.num_records,
        "progress": {"current": 0, "total": request.num_records},
        "config": request.model_dump(),
    }

    background_tasks.add_task(run_designer_hf_job, job_id, request)

    return DesignerJobResponse(
        job_id=job_id,
        status="queued",
        pipeline=request.pipeline,
        num_records=request.num_records,
    )


@router.post("/designer/push-to-hub")
async def designer_push_to_hub(request: DesignerPushToHubRequest):
    """Publish a generated dataset to HuggingFace Hub."""
    if request.job_id not in designer_jobs:
        raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found")

    job = designer_jobs[request.job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before publishing")

    try:
        import pandas as pd
        from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig

        output_dir = job.get("output_dir")
        if not output_dir:
            raise HTTPException(status_code=400, detail="No output data found for this job")

        # Load the generated data
        train_path = Path(output_dir) / "train.jsonl"
        if train_path.exists():
            df = pd.read_json(train_path, lines=True)
        else:
            raise HTTPException(status_code=400, detail="No generated data found")

        pipeline = DataDesignerPipeline(PipelineConfig())
        url = pipeline.push_to_hub(df, request.repo_id, request.private)

        return {"url": url, "repo_id": request.repo_id}
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DataDesigner Background Tasks
# =============================================================================

async def run_designer_job(job_id: str, request: DesignerCreateRequest):
    """Background task for DataDesigner generation."""
    try:
        from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig

        designer_jobs[job_id]["status"] = "running"

        config = PipelineConfig(
            pipeline=request.pipeline,
            num_records=request.num_records,
            provider=request.provider,
            provider_endpoint=request.provider_endpoint,
            output_dir=Path(request.output_dir or f"data/designer_output/{job_id}"),
            train_val_split=request.train_val_split,
        )
        if request.text_model:
            config.text_model = request.text_model
        if request.code_model:
            config.code_model = request.code_model
        if request.judge_model:
            config.judge_model = request.judge_model

        pipeline = DataDesignerPipeline(config)

        # Route to appropriate entry point
        seed = request.seed_source or "data/gold_traces"

        if request.seed_type == "traces":
            df = pipeline.from_traces(Path(seed), request.num_records)
        elif request.seed_type == "huggingface":
            df = pipeline.from_dataset(
                seed, request.num_records,
                column_mapping=request.column_mapping,
            )
        elif request.seed_type == "file":
            df = pipeline.from_dataset(seed, request.num_records)
        elif request.seed_type == "unstructured":
            df = pipeline.from_unstructured(Path(seed), request.num_records)
        else:
            raise ValueError(f"Unknown seed_type: {request.seed_type}")

        designer_jobs[job_id]["progress"]["current"] = len(df)

        # Export to NeMo format if requested
        export_result = None
        if request.export_nemo:
            export_result = pipeline.export_nemo(df)
            designer_jobs[job_id]["export_result"] = export_result

        designer_jobs[job_id]["status"] = "completed"
        designer_jobs[job_id]["output_dir"] = str(config.output_dir)

        logger.info(f"Designer job {job_id} completed: {len(df)} records generated")

    except Exception as e:
        logger.error(f"Designer job {job_id} failed: {e}")
        designer_jobs[job_id]["status"] = "failed"
        designer_jobs[job_id]["error"] = str(e)


async def run_designer_hf_job(job_id: str, request: DesignerHuggingFaceRequest):
    """Background task for DataDesigner HuggingFace generation."""
    try:
        from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig

        designer_jobs[job_id]["status"] = "running"

        config = PipelineConfig(
            pipeline=request.pipeline,
            num_records=request.num_records,
            provider=request.provider,
            output_dir=Path(f"data/designer_output/{job_id}"),
        )

        pipeline = DataDesignerPipeline(config)
        df = pipeline.from_dataset(
            request.dataset,
            request.num_records,
            column_mapping=request.column_mapping,
            subset=request.subset,
            split=request.split,
        )

        designer_jobs[job_id]["progress"]["current"] = len(df)

        # Export to NeMo format
        export_result = pipeline.export_nemo(df)
        designer_jobs[job_id]["export_result"] = export_result

        designer_jobs[job_id]["status"] = "completed"
        designer_jobs[job_id]["output_dir"] = str(config.output_dir)

        logger.info(f"Designer HF job {job_id} completed: {len(df)} records")

    except Exception as e:
        logger.error(f"Designer HF job {job_id} failed: {e}")
        designer_jobs[job_id]["status"] = "failed"
        designer_jobs[job_id]["error"] = str(e)
