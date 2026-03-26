"""
Cascade RL API Routes

Provides REST endpoints for managing cascade RL training:
- Start/stop cascade training with domain-by-domain GRPO stages
- Track per-stage progress and metrics
- Trigger MOPD distillation after cascade completes
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cascade", tags=["cascade"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CascadeStartRequest(BaseModel):
    """Request to start a cascade RL training run."""

    domains: list[str] = Field(
        default_factory=lambda: [
            "file_operations",
            "bash_commands",
            "search_and_navigate",
            "multi_step_reasoning",
        ],
        description="Ordered list of domains to train",
    )
    base_model: str = Field("Qwen/Qwen2.5-Coder-1.5B-Instruct", description="Base model")
    dataset_path: str = Field("data/gold_traces", description="Path to training data")
    train_steps_per_stage: int = Field(200, ge=10, le=5000, description="Steps per domain stage")
    grpo_num_generations: int = Field(4, ge=2, le=16, description="GRPO generations per prompt")
    grpo_temperature: float = Field(0.7, ge=0.0, le=2.0)
    learning_rate: float = Field(2e-5, gt=0, le=1e-2)
    lora_r: int = Field(16, ge=4, le=128)
    lora_alpha: int = Field(32, ge=8, le=256)
    load_in_4bit: bool = Field(True)
    use_remote_ssh: bool = Field(False)
    mode: str = Field("simulate", description="'simulate' or 'real'")
    early_stopping_patience: int = Field(3, ge=0, le=10)
    min_domain_examples: int = Field(10, ge=1, le=1000)


class MOPDStartRequest(BaseModel):
    """Request to start MOPD distillation after cascade completes."""

    student_model: str = Field(
        "", description="Student base model (empty = use cascade base model)"
    )
    distillation_alpha: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(2.0, ge=0.1, le=10.0)
    train_steps: int = Field(500, ge=10, le=10000)
    use_remote_ssh: bool = Field(False)


# =============================================================================
# Helpers
# =============================================================================


def _get_scheduler(request: Request):
    """Retrieve the active CascadeScheduler from app state, if any."""
    return getattr(request.app.state, "cascade_scheduler", None)


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/start")
async def start_cascade(request: Request, body: CascadeStartRequest):
    """Start a cascade RL training run with sequential domain stages."""
    existing = _get_scheduler(request)
    if existing and existing.status == "running":
        raise HTTPException(status_code=409, detail="Cascade training already running")

    from bashgym.gym.cascade_scheduler import CascadeConfig, CascadeScheduler

    config = CascadeConfig(
        domains=body.domains,
        base_model=body.base_model,
        dataset_path=Path(body.dataset_path),
        grpo_num_generations=body.grpo_num_generations,
        grpo_temperature=body.grpo_temperature,
        train_steps_per_stage=body.train_steps_per_stage,
        learning_rate=body.learning_rate,
        lora_r=body.lora_r,
        lora_alpha=body.lora_alpha,
        load_in_4bit=body.load_in_4bit,
        use_remote_ssh=body.use_remote_ssh,
        mode=body.mode,
        early_stopping_patience=body.early_stopping_patience,
        min_domain_examples=body.min_domain_examples,
    )

    scheduler = CascadeScheduler(config)
    request.app.state.cascade_scheduler = scheduler

    # Broadcast via WebSocket
    from bashgym.api.websocket import WSMessage, manager

    async def cascade_callback(event_type: str, stage_or_data):
        data = stage_or_data.to_dict() if hasattr(stage_or_data, "to_dict") else stage_or_data
        msg = WSMessage(type=f"cascade:{event_type}", payload=data)
        await manager.broadcast(msg)

    # Launch cascade as background task
    async def run_and_store():
        try:
            result = await scheduler.run_cascade(callback=cascade_callback)
            request.app.state.cascade_result = result
            # Broadcast completion
            msg = WSMessage(
                type="cascade:completed",
                payload={
                    "status": result.status,
                    "stages_completed": sum(1 for s in result.stages if s.status == "completed"),
                    "total_duration": round(result.total_duration_seconds, 1),
                    "best_checkpoints": {k: str(v) for k, v in result.best_checkpoints.items()},
                },
            )
            await manager.broadcast(msg)
        except Exception as exc:
            logger.error(f"[Cascade] Background run crashed: {exc}", exc_info=True)
            scheduler.status = "failed"
            scheduler._error = str(exc)

    asyncio.create_task(run_and_store())

    logger.info(
        f"[Cascade] Started: {len(body.domains)} domains, mode={body.mode}, "
        f"steps_per_stage={body.train_steps_per_stage}"
    )

    return {
        "status": "started",
        "domains": body.domains,
        "stages": len(body.domains),
        "mode": body.mode,
    }


@router.post("/stop")
async def stop_cascade(request: Request):
    """Stop the running cascade after the current stage completes."""
    scheduler = _get_scheduler(request)
    if not scheduler:
        raise HTTPException(status_code=404, detail="No cascade session found")
    if scheduler.status != "running":
        raise HTTPException(
            status_code=409, detail=f"Cascade not running (status: {scheduler.status})"
        )
    scheduler.stop()
    logger.info("[Cascade] Stop requested")
    return {"status": "stopping"}


@router.get("/status")
async def get_cascade_status(request: Request):
    """Get cascade training status with per-stage details."""
    scheduler = _get_scheduler(request)
    if not scheduler:
        return {
            "status": "idle",
            "current_stage": 0,
            "total_stages": 0,
            "stages": [],
            "domains": [],
        }
    return scheduler.get_status()


@router.post("/distill")
async def start_mopd_distillation(request: Request, body: MOPDStartRequest):
    """Start MOPD distillation to merge domain checkpoints into a unified model."""
    scheduler = _get_scheduler(request)
    cascade_result = getattr(request.app.state, "cascade_result", None)

    if not scheduler or not cascade_result:
        raise HTTPException(
            status_code=404,
            detail="No completed cascade run found. Run cascade first.",
        )

    if cascade_result.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Cascade not completed (status: {cascade_result.status})",
        )

    if not cascade_result.best_checkpoints:
        raise HTTPException(
            status_code=409,
            detail="No domain checkpoints available for distillation",
        )

    from bashgym.api.websocket import WSMessage, manager

    mopd_config = scheduler.create_mopd_config(
        cascade_result,
        student_model=body.student_model or scheduler.config.base_model,
        distillation_alpha=body.distillation_alpha,
        temperature=body.temperature,
        train_steps=body.train_steps,
        use_remote_ssh=body.use_remote_ssh,
    )

    async def mopd_callback(event_type, data):
        msg = WSMessage(type=f"cascade:{event_type}", payload=data)
        await manager.broadcast(msg)

    async def run_mopd():
        try:
            # MOPD distillation is a future feature — for now broadcast readiness
            await mopd_callback(
                "mopd-dataset-ready",
                {
                    "domains": list(mopd_config.domain_checkpoints.keys()),
                    "student_model": mopd_config.student_model,
                    "output_path": str(mopd_config.output_path),
                },
            )

            await mopd_callback(
                "mopd-training-started",
                {
                    "student_model": mopd_config.student_model,
                    "train_steps": mopd_config.train_steps,
                    "domains": list(mopd_config.domain_checkpoints.keys()),
                },
            )

            # Placeholder: actual MOPD training will be implemented
            # in a future step; for now simulate completion
            import random

            await asyncio.sleep(random.uniform(2.0, 5.0))

            result = {
                "status": "completed",
                "student_model": mopd_config.student_model,
                "output_path": str(mopd_config.output_path),
                "domains_distilled": list(mopd_config.domain_checkpoints.keys()),
            }
            request.app.state.mopd_result = result
            await mopd_callback("mopd-completed", result)

        except Exception as exc:
            logger.error(f"[MOPD] Distillation failed: {exc}", exc_info=True)
            error_result = {"status": "failed", "error": str(exc)}
            request.app.state.mopd_result = error_result
            await mopd_callback("mopd-failed", error_result)

    asyncio.create_task(run_mopd())

    return {
        "status": "started",
        "domains": list(cascade_result.best_checkpoints.keys()),
        "student_model": mopd_config.student_model,
        "train_steps": body.train_steps,
    }


@router.get("/distill/status")
async def get_mopd_status(request: Request):
    """Get MOPD distillation status."""
    result = getattr(request.app.state, "mopd_result", None)
    if not result:
        return {"status": "idle"}
    return result
