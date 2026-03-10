"""
Pipeline API routes for Bash Gym.

Provides REST endpoints for managing the auto-import pipeline:
- Pipeline configuration (get/update)
- Pipeline status (stage counts, watcher state)
- Manual stage triggers
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

# Pipeline singleton - initialized on first access
_pipeline = None


def _get_pipeline():
    """Get or create the Pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from bashgym.pipeline.orchestrator import Pipeline
        from bashgym.api.websocket import broadcast_pipeline_event, MessageType

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
    return _pipeline


@router.get("/config")
async def get_pipeline_config():
    """Get current pipeline configuration."""
    pipeline = _get_pipeline()
    return pipeline.config.to_dict()


@router.put("/config")
async def update_pipeline_config(updates: Dict[str, Any]):
    """Update pipeline configuration. Hot-reloads all components."""
    pipeline = _get_pipeline()
    new_config = pipeline.save_config(updates)
    return new_config.to_dict()


@router.get("/status")
async def get_pipeline_status():
    """Get current pipeline status including stage counts."""
    pipeline = _get_pipeline()
    return pipeline.get_status()


@router.post("/trigger/{stage}")
async def trigger_pipeline_stage(stage: str):
    """Manually trigger a pipeline stage."""
    pipeline = _get_pipeline()
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
        for trace_file in list(trace_capture.traces_dir.glob("*.json")) + list(trace_capture.traces_dir.glob("*.jsonl")):
            result = pipeline.handle_session_file(trace_file)
            if result:
                count += 1
        return {"classified": count}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")


def start_pipeline_watcher():
    """Start the pipeline filesystem watcher. Call from app startup."""
    try:
        pipeline = _get_pipeline()
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
