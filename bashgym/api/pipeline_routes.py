"""
Pipeline API routes for Bash Gym.

Provides REST endpoints for managing the auto-import pipeline:
- Pipeline configuration (get/update)
- Pipeline status (stage counts, watcher state)
- Manual stage triggers
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

# Pipeline singleton - initialized on first access
_pipeline = None


def build_cascade_start_payload(config, gold_dir: Path, gold_count: int) -> dict[str, Any]:
    """Build the CascadeStartRequest-compatible payload for a pipeline trigger."""
    mode = getattr(config, "cascade_mode", "simulate")
    if mode not in {"simulate", "real"}:
        raise ValueError("cascade_mode must be 'simulate' or 'real'")
    base_model = getattr(config, "cascade_base_model", "")
    if mode == "real" and not base_model:
        raise ValueError("cascade_base_model is required when cascade_mode='real'")
    repo_domains_enabled = bool(getattr(config, "cascade_repo_domains_enabled", False))
    return {
        "base_model": base_model,
        "dataset_path": str(gold_dir),
        "mode": mode,
        "train_steps_per_stage": int(getattr(config, "cascade_train_steps_per_stage", 200)),
        "min_domain_examples": int(getattr(config, "cascade_min_domain_examples", 10)),
        "use_remote_ssh": bool(getattr(config, "cascade_use_remote_ssh", False)),
        "repo_domains_enabled": repo_domains_enabled,
        "repo_domains_dir": str(gold_dir) if repo_domains_enabled else "",
        "trigger": {
            "source": "pipeline",
            "gold_count": gold_count,
            "gold_threshold": getattr(config, "cascade_gold_threshold", 200),
        },
    }


async def _start_pipeline_cascade(app_state: Any, payload: dict[str, Any]) -> dict[str, Any]:
    existing = getattr(app_state, "cascade_scheduler", None)
    if existing and existing.status == "running":
        return {"status": "skipped", "reason": "cascade already running", "request": payload}

    from bashgym.api.websocket import WSMessage, manager
    from bashgym.gym.cascade_scheduler import CascadeConfig, CascadeScheduler

    scheduler = CascadeScheduler(
        CascadeConfig(
            base_model=payload["base_model"],
            dataset_path=Path(payload["dataset_path"]),
            train_steps_per_stage=payload["train_steps_per_stage"],
            min_domain_examples=payload["min_domain_examples"],
            use_remote_ssh=payload["use_remote_ssh"],
            mode=payload["mode"],
            repo_domains_enabled=payload["repo_domains_enabled"],
            repo_domains_dir=payload["repo_domains_dir"],
        )
    )
    app_state.cascade_scheduler = scheduler
    app_state.cascade_auto_trigger = payload["trigger"]

    async def cascade_callback(event_type: str, stage_or_data):
        data = stage_or_data.to_dict() if hasattr(stage_or_data, "to_dict") else stage_or_data
        await manager.broadcast(WSMessage(type=f"cascade:{event_type}", payload=data))

    async def run_and_store():
        try:
            result = await scheduler.run_cascade(callback=cascade_callback)
            app_state.cascade_result = result
            await manager.broadcast(
                WSMessage(
                    type="cascade:completed",
                    payload={
                        "status": result.status,
                        "stages_completed": sum(
                            1 for stage in result.stages if stage.status == "completed"
                        ),
                        "total_duration": round(result.total_duration_seconds, 1),
                        "best_checkpoints": {
                            key: str(path) for key, path in result.best_checkpoints.items()
                        },
                        "trigger": payload["trigger"],
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - background task must capture failure
            logger.error("[Pipeline] Auto-cascade failed: %s", exc, exc_info=True)
            scheduler.status = "failed"
            scheduler._error = str(exc)
            app_state.cascade_result = {"status": "failed", "error": str(exc)}

    asyncio.create_task(run_and_store())
    return {"status": "queued", "request": payload}


def _wire_cascade_trigger(pipeline, app_state: Any) -> None:
    if getattr(pipeline, "_cascade_trigger_app_state_id", None) == id(app_state):
        return

    loop = asyncio.get_running_loop()

    def trigger(gold_count: int) -> dict[str, Any]:
        payload = build_cascade_start_payload(
            pipeline.config,
            pipeline._trace_capture.gold_traces_dir,
            gold_count,
        )
        existing = getattr(app_state, "cascade_scheduler", None)
        if existing and existing.status == "running":
            return {"status": "skipped", "reason": "cascade already running", "request": payload}
        asyncio.run_coroutine_threadsafe(_start_pipeline_cascade(app_state, payload), loop)
        return {"status": "queued", "request": payload}

    pipeline.cascade_trigger = trigger
    pipeline._cascade_trigger_app_state_id = id(app_state)


def _get_pipeline(app_state: Any | None = None):
    """Get or create the Pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from bashgym.api.websocket import MessageType, broadcast_pipeline_event
        from bashgym.pipeline.orchestrator import Pipeline

        def on_pipeline_event(event_type: str, payload):
            type_map = {
                "pipeline:import": MessageType.PIPELINE_IMPORT,
                "pipeline:classified": MessageType.PIPELINE_CLASSIFIED,
                "pipeline:threshold_reached": MessageType.PIPELINE_THRESHOLD_REACHED,
                "pipeline:stage_started": MessageType.PIPELINE_STAGE_STARTED,
            }
            msg_type = type_map.get(event_type)
            if msg_type:
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        broadcast_pipeline_event(msg_type, payload),
                    )
                except RuntimeError:
                    pass  # No event loop running

        _pipeline = Pipeline(on_event=on_pipeline_event)
    if app_state is not None:
        _wire_cascade_trigger(_pipeline, app_state)
    return _pipeline


@router.get("/config")
async def get_pipeline_config(request: Request):
    """Get current pipeline configuration."""
    pipeline = _get_pipeline(request.app.state)
    return pipeline.config.to_dict()


@router.put("/config")
async def update_pipeline_config(updates: dict[str, Any], request: Request):
    """Update pipeline configuration. Hot-reloads all components."""
    pipeline = _get_pipeline(request.app.state)
    new_config = pipeline.save_config(updates)
    return new_config.to_dict()


@router.get("/status")
async def get_pipeline_status(request: Request):
    """Get current pipeline status including stage counts."""
    pipeline = _get_pipeline(request.app.state)
    status = pipeline.get_status()
    status["cascade_auto_trigger"] = getattr(request.app.state, "cascade_auto_trigger", None)
    return status


@router.post("/trigger/{stage}")
async def trigger_pipeline_stage(stage: str, request: Request):
    """Manually trigger a pipeline stage."""
    pipeline = _get_pipeline(request.app.state)
    if stage == "import":
        from bashgym.trace_capture.importers.claude_history import ClaudeSessionImporter

        importer = ClaudeSessionImporter()
        results = importer.import_recent(days=60, verbose=False)
        imported = [r for r in results if not r.skipped and not r.error]
        return {"imported": len(imported), "total": len(results)}
    elif stage == "classify":
        from bashgym.trace_capture.core import TraceCapture

        trace_capture = TraceCapture()
        count = 0
        for trace_file in list(trace_capture.traces_dir.glob("*.json")) + list(
            trace_capture.traces_dir.glob("*.jsonl")
        ):
            result = pipeline.handle_session_file(trace_file)
            if result:
                count += 1
        return {"classified": count}
    elif stage == "cascade":
        gold_count = len(list(pipeline._trace_capture.gold_traces_dir.glob("*.json")))
        try:
            payload = build_cascade_start_payload(
                pipeline.config,
                pipeline._trace_capture.gold_traces_dir,
                gold_count,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _start_pipeline_cascade(request.app.state, payload)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")


def start_pipeline_watcher(app_state: Any | None = None):
    """Start the pipeline filesystem watcher. Call from app startup."""
    try:
        pipeline = _get_pipeline(app_state)
        pipeline.start_watcher()
        logger.info("Pipeline watcher started")
    except Exception as e:
        logger.warning(f"Failed to start pipeline watcher: {e}")


def stop_pipeline_watcher():
    """Stop the pipeline filesystem watcher. Call from app shutdown."""
    global _pipeline
    if _pipeline:
        _pipeline.stop_watcher()
        logger.info("Pipeline watcher stopped")
