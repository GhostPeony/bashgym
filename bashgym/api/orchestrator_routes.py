# bashgym/api/orchestrator_routes.py
"""API routes for the multi-agent orchestration system."""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orchestrate", tags=["orchestrator"])


# =============================================================================
# Request/Response Models
# =============================================================================

class LLMConfigRequest(BaseModel):
    """LLM provider configuration for spec decomposition."""
    provider: str = "anthropic"  # anthropic, openai, gemini, ollama
    model: Optional[str] = None  # Uses provider default if not set
    api_key: Optional[str] = None  # Falls back to env vars
    base_url: Optional[str] = None  # For Ollama or custom endpoints
    temperature: float = 0.3
    max_tokens: int = 4096


class SpecRequest(BaseModel):
    """Request to submit a development spec for decomposition."""
    title: str
    description: str
    constraints: List[str] = []
    acceptance_criteria: List[str] = []
    repository: Optional[str] = None
    base_branch: str = "main"
    max_budget_usd: float = 10.0
    max_workers: int = 5
    llm_config: Optional[LLMConfigRequest] = None


class ApproveRequest(BaseModel):
    """Request to approve a decomposed plan."""
    base_branch: str = "main"


class RetryRequest(BaseModel):
    """Request to retry a failed task."""
    modified_prompt: Optional[str] = None


# In-memory job storage (would be persistent in production)
_jobs: Dict[str, dict] = {}


# =============================================================================
# Helper: build orchestrator from config
# =============================================================================

def _build_llm_config(req: Optional[LLMConfigRequest] = None):
    """Build LLMConfig from request, with provider defaults."""
    from bashgym.orchestrator.models import LLMConfig, LLMProvider

    if not req:
        return LLMConfig()

    provider_map = {
        "anthropic": LLMProvider.ANTHROPIC,
        "openai": LLMProvider.OPENAI,
        "gemini": LLMProvider.GEMINI,
        "ollama": LLMProvider.OLLAMA,
    }
    provider = provider_map.get(req.provider.lower())
    if not provider:
        raise ValueError(
            f"Unknown provider '{req.provider}'. "
            f"Supported: anthropic, openai, gemini, ollama"
        )

    config = LLMConfig(
        provider=provider,
        model=req.model or "",  # Empty = auto-resolve from provider
        api_key=req.api_key or "",
        base_url=req.base_url,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    return config


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/submit")
async def submit_spec(request: SpecRequest, background_tasks: BackgroundTasks):
    """Submit a development spec for decomposition.

    Returns a job ID and starts async decomposition using the configured
    LLM provider (defaults to Anthropic Claude Opus).

    The decomposed TaskDAG must be approved via /approve before execution.
    """
    from bashgym.orchestrator.models import OrchestratorSpec
    from bashgym.orchestrator.agent import OrchestrationAgent

    job_id = str(uuid.uuid4())[:8]

    try:
        llm_config = _build_llm_config(request.llm_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    spec = OrchestratorSpec(
        title=request.title,
        description=request.description,
        constraints=request.constraints,
        acceptance_criteria=request.acceptance_criteria,
        repository=request.repository,
        base_branch=request.base_branch,
        max_budget_usd=request.max_budget_usd,
        max_workers=request.max_workers,
    )

    _jobs[job_id] = {
        "id": job_id,
        "status": "decomposing",
        "spec": spec,
        "llm_config": llm_config,
        "agent": None,
        "dag": None,
        "results": [],
        "error": None,
    }

    background_tasks.add_task(_decompose_spec, job_id, spec, llm_config)

    return {
        "job_id": job_id,
        "status": "decomposing",
        "provider": llm_config.provider.value,
        "model": llm_config.model,
    }


async def _decompose_spec(job_id: str, spec, llm_config):
    """Background task: decompose spec into TaskDAG."""
    from bashgym.orchestrator.agent import OrchestrationAgent
    from pathlib import Path

    try:
        repo_path = Path(spec.repository) if spec.repository else None
        agent = OrchestrationAgent(
            llm_config=llm_config,
            max_workers=spec.max_workers,
            repo_path=repo_path,
            use_worktrees=repo_path is not None,
        )

        dag = await agent.submit_spec(spec)

        _jobs[job_id]["agent"] = agent
        _jobs[job_id]["dag"] = dag
        _jobs[job_id]["status"] = "awaiting_approval"

    except Exception as e:
        logger.error(f"Decomposition failed for job {job_id}: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


@router.post("/{job_id}/approve")
async def approve_plan(job_id: str, request: ApproveRequest, background_tasks: BackgroundTasks):
    """Approve the decomposed plan and start execution.

    Workers are spawned as Claude Code CLI subprocesses in isolated
    git worktrees.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    if job["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job['status']}', expected 'awaiting_approval'"
        )

    job["status"] = "executing"
    background_tasks.add_task(_execute_dag, job_id, request.base_branch)

    dag = job["dag"]
    return {
        "status": "executing",
        "task_count": len(dag.nodes),
        "stats": dag.stats,
    }


async def _execute_dag(job_id: str, base_branch: str):
    """Background task: execute approved DAG."""
    job = _jobs[job_id]
    agent = job["agent"]

    try:
        results = await agent.execute(dag=job["dag"], base_branch=base_branch)
        job["results"] = results
        job["status"] = "completed"
    except Exception as e:
        logger.error(f"Execution failed for job {job_id}: {e}")
        job["status"] = "failed"
        job["error"] = str(e)


@router.get("/{job_id}/status")
async def get_status(job_id: str):
    """Get orchestration job status.

    Returns task-level progress, cost, and timing information.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    dag = job.get("dag")

    response = {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
    }

    if dag:
        response["dag"] = dag.to_dict()
        response["stats"] = dag.stats

    results = job.get("results", [])
    if results:
        response["total_cost"] = sum(r.cost_usd for r in results)
        response["total_time"] = sum(r.duration_seconds for r in results)
        response["completed"] = sum(1 for r in results if r.success)
        response["failed"] = sum(1 for r in results if not r.success)

    return response


@router.post("/{job_id}/task/{task_id}/retry")
async def retry_task(
    job_id: str,
    task_id: str,
    request: RetryRequest,
    background_tasks: BackgroundTasks,
):
    """Retry a failed task with an optionally modified prompt."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    dag = job.get("dag")
    if not dag or task_id not in dag.nodes:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    from bashgym.orchestrator.models import TaskStatus

    task = dag.nodes[task_id]
    if task.status != TaskStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Task status is '{task.status.value}', expected 'failed'"
        )

    # Reset task for retry
    task.status = TaskStatus.PENDING
    task.retry_count += 1
    if request.modified_prompt:
        task.worker_prompt = request.modified_prompt

    # Re-execute just this task
    if job["status"] != "executing":
        job["status"] = "executing"
        background_tasks.add_task(
            _execute_dag, job_id, job["spec"].base_branch
        )

    return {
        "status": "retrying",
        "task_id": task_id,
        "retry_count": task.retry_count,
    }


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel all workers and clean up worktrees."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    agent = job.get("agent")

    if agent and agent.pool:
        await agent.pool.cancel_all()

    if agent and agent.worktrees:
        await agent.worktrees.cleanup_all()

    job["status"] = "cancelled"

    return {"status": "cancelled", "job_id": job_id}


@router.get("/providers")
async def list_providers():
    """List available LLM providers and their default models."""
    from bashgym.orchestrator.models import LLMConfig

    config = LLMConfig()
    providers = []
    for provider_name, defaults in config.PROVIDER_DEFAULTS.items():
        providers.append({
            "provider": provider_name,
            "default_model": defaults["model"],
            "env_key": defaults["env_key"],
            "base_url": defaults["base_url"],
        })

    return {"providers": providers}


@router.get("/jobs")
async def list_jobs():
    """List all orchestration jobs."""
    jobs = []
    for job_id, job in _jobs.items():
        entry = {
            "job_id": job_id,
            "status": job["status"],
            "title": job["spec"].title if job.get("spec") else None,
        }
        dag = job.get("dag")
        if dag:
            entry["task_count"] = len(dag.nodes)
            entry["stats"] = dag.stats
        jobs.append(entry)

    return {"jobs": jobs}
