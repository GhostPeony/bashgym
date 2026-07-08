"""Development-only routes for exercising UI surfaces with realistic synthetic data.

These endpoints never touch real training infrastructure. They emit WebSocket
events with the same payload shapes the trainer produces so frontend surfaces
can be built and reviewed without a GPU or a live run. Desktop mode only.
"""

import asyncio
import logging
import math
import random
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from bashgym.api.websocket import MessageType, WSMessage, manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dev", tags=["dev"])

_sim_task: asyncio.Task | None = None
_sim_run_id: str | None = None


class TrainingSimRequest(BaseModel):
    """Config for a simulated training run stream."""

    total_steps: int = Field(300, ge=10, le=100_000)
    interval_s: float = Field(0.6, ge=0.05, le=10.0)
    strategy: str = "sft"
    compute_target: str = Field(
        "local", description="Label shown in the UI, e.g. 'local', 'ssh:<device>', 'cloud'"
    )
    fail_at_step: int | None = Field(None, description="Emit training:failed at this step")


def _eta(steps_left: int, interval_s: float) -> str:
    secs = int(steps_left * interval_s)
    return f"{secs // 60}m {secs % 60}s"


async def _log(run_id: str, line: str, level: str = "info") -> None:
    await manager.broadcast(
        WSMessage(
            type=MessageType.TRAINING_LOG,
            payload={"run_id": run_id, "message": line, "level": level},
        )
    )


async def _run_simulation(run_id: str, req: TrainingSimRequest) -> None:
    rng = random.Random(run_id)
    steps_per_epoch = max(1, req.total_steps // 3)
    base_lr = 2e-4
    warmup = max(1, int(req.total_steps * 0.05))

    await _log(run_id, f"[sim] Loading base model for {req.strategy} run {run_id}")
    await _log(run_id, "[sim] Dataset: 1,024 examples | batch 4 | grad-accum 4")
    await _log(run_id, f"[sim] Compute target: {req.compute_target}")

    samples = 0
    payload: dict = {}
    try:
        for step in range(1, req.total_steps + 1):
            await asyncio.sleep(req.interval_s)

            progress = step / req.total_steps
            loss = 2.2 * math.exp(-3.0 * progress) + 0.35 + rng.uniform(-0.04, 0.04)
            if step < warmup:
                lr = base_lr * (step / warmup)
            else:
                decay = (step - warmup) / max(1, req.total_steps - warmup)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * decay))
            samples += 16
            payload = {
                "run_id": run_id,
                "step": step,
                "total_steps": req.total_steps,
                "epoch": round(step / steps_per_epoch, 3),
                "loss": round(loss, 4),
                "learning_rate": lr,
                "grad_norm": round(abs(rng.gauss(0.7, 0.25)), 3),
                "eta": _eta(req.total_steps - step, req.interval_s),
                "simulation": True,
                "samples_processed": samples,
                "tokens_per_second": round(rng.gauss(2400, 120), 1),
                "gpu_memory_gb": round(rng.gauss(10.6, 0.2), 2),
                "gpu_utilization": round(min(100.0, max(0.0, rng.gauss(93, 4))), 1),
                "compute_target": req.compute_target,
            }
            if step % 20 == 0:
                payload["eval_loss"] = round(loss + rng.uniform(0.02, 0.12), 4)

            await manager.broadcast(WSMessage(type=MessageType.TRAINING_PROGRESS, payload=payload))

            if step % 10 == 0:
                await _log(
                    run_id,
                    f"{{'loss': {payload['loss']}, 'grad_norm': {payload['grad_norm']}, "
                    f"'learning_rate': {lr:.2e}, 'epoch': {payload['epoch']}, 'step': {step}}}",
                )

            if req.fail_at_step and step >= req.fail_at_step:
                await _log(
                    run_id, "[sim] CUDA error: device-side assert triggered (simulated)", "error"
                )
                await manager.broadcast(
                    WSMessage(
                        type=MessageType.TRAINING_FAILED,
                        payload={"run_id": run_id, "error": "Simulated failure for UI review"},
                    )
                )
                return

        await _log(run_id, f"[sim] Training complete - final loss {payload.get('loss')}")
        await manager.broadcast(
            WSMessage(
                type=MessageType.TRAINING_COMPLETE,
                payload={
                    "run_id": run_id,
                    "metrics": {"final_loss": payload.get("loss"), "steps": req.total_steps},
                },
            )
        )
    except asyncio.CancelledError:
        await _log(run_id, "[sim] Simulation stopped by user")
        await manager.broadcast(
            WSMessage(
                type=MessageType.TRAINING_COMPLETE,
                payload={"run_id": run_id, "metrics": {"stopped_at_step": payload.get("step", 0)}},
            )
        )
        raise


@router.post("/training-sim/start")
async def start_training_sim(request: TrainingSimRequest):
    """Start a simulated training run streamed over the real WebSocket hub."""
    global _sim_task, _sim_run_id
    if _sim_task and not _sim_task.done():
        raise HTTPException(status_code=409, detail=f"Simulation {_sim_run_id} already running")

    _sim_run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_sim"
    _sim_task = asyncio.create_task(_run_simulation(_sim_run_id, request))
    logger.info(f"[dev] Started training simulation {_sim_run_id}")
    return {
        "run_id": _sim_run_id,
        "total_steps": request.total_steps,
        "interval_s": request.interval_s,
        "compute_target": request.compute_target,
    }


@router.post("/training-sim/stop")
async def stop_training_sim():
    """Stop the running training simulation, if any."""
    global _sim_task
    if not _sim_task or _sim_task.done():
        return {"stopped": False, "detail": "No simulation running"}
    _sim_task.cancel()
    return {"stopped": True, "run_id": _sim_run_id}
