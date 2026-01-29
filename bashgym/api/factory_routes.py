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
