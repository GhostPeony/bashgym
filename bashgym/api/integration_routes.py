"""
Bashbros Integration API routes for Bash Gym.

Provides REST endpoints for managing the bashbros integration:
- Integration status and health
- Settings management
- Trace management (from bashbros)
- Model export and rollback
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/integration", tags=["integration"])


# =============================================================================
# Schemas
# =============================================================================

class IntegrationStatusResponse(BaseModel):
    """Integration status response."""
    enabled: bool = Field(description="Whether integration is enabled")
    linked: bool = Field(description="Whether bashbros is linked")
    linked_at: Optional[str] = Field(default=None, description="When integration was linked")
    bashbros_connected: bool = Field(description="Whether bashbros is actively connected")
    bashgym_connected: bool = Field(description="Whether bashgym is running")
    pending_traces: int = Field(description="Number of pending traces")
    processed_traces: int = Field(description="Number of processed traces")
    current_model_version: Optional[str] = Field(default=None, description="Current sidekick model version")
    training_in_progress: bool = Field(description="Whether training is running")


class IntegrationSettingsResponse(BaseModel):
    """Integration settings response."""
    version: str
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None

    # Integration
    enabled: bool = False
    linked_at: Optional[str] = None

    # Capture
    capture_mode: str = "successful_only"
    auto_stream: bool = True

    # Training
    auto_training_enabled: bool = False
    quality_threshold: int = 50
    trigger: str = "quality_based"

    # Security
    bashbros_primary: bool = True
    policy_path: Optional[str] = None

    # Model sync
    auto_export_ollama: bool = True
    ollama_model_name: str = "bashgym-sidekick"
    notify_on_update: bool = True


class UpdateSettingsRequest(BaseModel):
    """Request to update integration settings."""
    capture: Optional[Dict[str, Any]] = Field(default=None, description="Capture settings")
    training: Optional[Dict[str, Any]] = Field(default=None, description="Training settings")
    security: Optional[Dict[str, Any]] = Field(default=None, description="Security settings")
    model_sync: Optional[Dict[str, Any]] = Field(default=None, description="Model sync settings")


class TraceInfoResponse(BaseModel):
    """Info about a trace from bashbros."""
    filename: str
    task: str
    source: str
    verified: bool
    steps: int


class ModelVersionResponse(BaseModel):
    """Model version info."""
    version: str
    created: str
    traces_used: int
    quality_avg: float
    is_latest: bool
    gguf_available: bool


class ExportModelRequest(BaseModel):
    """Request to export model to GGUF."""
    run_id: str = Field(description="Training run ID to export")
    quantization: str = Field(default="q4_k_m", description="GGUF quantization level")
    traces_used: int = Field(default=0, description="Number of traces used")
    quality_avg: float = Field(default=0.0, description="Average quality score")


class ExportModelResponse(BaseModel):
    """Response from model export."""
    success: bool
    version: Optional[str] = None
    gguf_path: Optional[str] = None
    ollama_registered: bool = False
    error: Optional[str] = None


class RollbackRequest(BaseModel):
    """Request to rollback to a previous model version."""
    version: str = Field(description="Version to rollback to (e.g., 'v2')")


class LinkResponse(BaseModel):
    """Response from link/unlink operations."""
    success: bool
    linked: bool
    linked_at: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status", response_model=IntegrationStatusResponse)
async def get_integration_status():
    """Get current integration status."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        status = integration.get_status()
        settings = integration.get_settings()

        return IntegrationStatusResponse(
            enabled=settings.enabled,
            linked=integration.is_linked(),
            linked_at=settings.linked_at,
            bashbros_connected=status.bashbros_connected,
            bashgym_connected=status.bashgym_connected,
            pending_traces=status.pending_traces,
            processed_traces=status.processed_traces,
            current_model_version=status.current_model_version,
            training_in_progress=status.training_in_progress,
        )

    except Exception as e:
        logger.error(f"Failed to get integration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings", response_model=IntegrationSettingsResponse)
async def get_integration_settings():
    """Get current integration settings."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()
        settings = integration.get_settings()

        return IntegrationSettingsResponse(
            version=settings.version,
            updated_at=settings.updated_at,
            updated_by=settings.updated_by,
            enabled=settings.enabled,
            linked_at=settings.linked_at,
            capture_mode=settings.capture_mode.value,
            auto_stream=settings.auto_stream,
            auto_training_enabled=settings.auto_training_enabled,
            quality_threshold=settings.quality_threshold,
            trigger=settings.trigger.value,
            bashbros_primary=settings.bashbros_primary,
            policy_path=settings.policy_path,
            auto_export_ollama=settings.auto_export_ollama,
            ollama_model_name=settings.ollama_model_name,
            notify_on_update=settings.notify_on_update,
        )

    except Exception as e:
        logger.error(f"Failed to get integration settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/settings", response_model=IntegrationSettingsResponse)
async def update_integration_settings(request: UpdateSettingsRequest):
    """Update integration settings."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        updates = {}
        if request.capture:
            updates["capture"] = request.capture
        if request.training:
            updates["training"] = request.training
        if request.security:
            updates["security"] = request.security
        if request.model_sync:
            updates["model_sync"] = request.model_sync

        settings = integration.update_settings(updates)

        return IntegrationSettingsResponse(
            version=settings.version,
            updated_at=settings.updated_at,
            updated_by=settings.updated_by,
            enabled=settings.enabled,
            linked_at=settings.linked_at,
            capture_mode=settings.capture_mode.value,
            auto_stream=settings.auto_stream,
            auto_training_enabled=settings.auto_training_enabled,
            quality_threshold=settings.quality_threshold,
            trigger=settings.trigger.value,
            bashbros_primary=settings.bashbros_primary,
            policy_path=settings.policy_path,
            auto_export_ollama=settings.auto_export_ollama,
            ollama_model_name=settings.ollama_model_name,
            notify_on_update=settings.notify_on_update,
        )

    except Exception as e:
        logger.error(f"Failed to update integration settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/link", response_model=LinkResponse)
async def link_integration():
    """Link bashgym with bashbros."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        success = integration.link()
        settings = integration.get_settings()

        return LinkResponse(
            success=success,
            linked=integration.is_linked(),
            linked_at=settings.linked_at,
        )

    except Exception as e:
        logger.error(f"Failed to link integration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unlink", response_model=LinkResponse)
async def unlink_integration():
    """Unlink bashgym from bashbros."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        success = integration.unlink()

        return LinkResponse(
            success=success,
            linked=False,
            linked_at=None,
        )

    except Exception as e:
        logger.error(f"Failed to unlink integration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/pending", response_model=List[TraceInfoResponse])
async def list_pending_traces():
    """List pending traces from bashbros."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        traces = integration.list_pending_traces()
        return [TraceInfoResponse(**t) for t in traces]

    except Exception as e:
        logger.error(f"Failed to list pending traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/traces/process")
async def process_pending_traces(background_tasks: BackgroundTasks):
    """Trigger processing of pending traces."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        # Process in background
        def do_process():
            count = integration._process_pending_traces()
            logger.info(f"Processed {count} pending traces")

        background_tasks.add_task(do_process)

        return {
            "status": "processing",
            "pending_count": integration.get_pending_count(),
        }

    except Exception as e:
        logger.error(f"Failed to process traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/versions", response_model=List[ModelVersionResponse])
async def list_model_versions():
    """List available model versions."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        versions = integration.get_model_versions()
        return [ModelVersionResponse(**v) for v in versions]

    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/export", response_model=ExportModelResponse)
async def export_model(request: ExportModelRequest, background_tasks: BackgroundTasks):
    """Export a trained model to GGUF and register with Ollama."""
    try:
        from bashgym.integrations.bashbros import get_integration
        from bashgym.config import get_settings

        integration = get_integration()
        settings = get_settings()

        # Get the model path from training run
        model_dir = Path(settings.models_dir) / request.run_id / "merged"
        if not model_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found for run {request.run_id}"
            )

        # Export to GGUF
        gguf_path = integration.export_to_gguf(
            model_path=model_dir,
            quantization=request.quantization,
            traces_used=request.traces_used,
            quality_avg=request.quality_avg,
        )

        if gguf_path:
            manifest = integration._load_manifest()
            return ExportModelResponse(
                success=True,
                version=manifest.latest,
                gguf_path=str(gguf_path),
                ollama_registered=integration.get_settings().auto_export_ollama,
            )
        else:
            return ExportModelResponse(
                success=False,
                error="GGUF export failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        return ExportModelResponse(
            success=False,
            error=str(e)
        )


@router.post("/models/rollback", response_model=ExportModelResponse)
async def rollback_model(request: RollbackRequest):
    """Rollback to a previous model version."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        success = integration.rollback_model(request.version)

        if success:
            return ExportModelResponse(
                success=True,
                version=request.version,
                ollama_registered=integration.get_settings().auto_export_ollama,
            )
        else:
            return ExportModelResponse(
                success=False,
                error=f"Failed to rollback to version {request.version}"
            )

    except Exception as e:
        logger.error(f"Failed to rollback model: {e}")
        return ExportModelResponse(
            success=False,
            error=str(e)
        )


@router.post("/watcher/start")
async def start_trace_watcher():
    """Start watching for traces from bashbros."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()
        integration.start_watching()

        return {"status": "watching", "message": "Trace watcher started"}

    except Exception as e:
        logger.error(f"Failed to start watcher: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watcher/stop")
async def stop_trace_watcher():
    """Stop watching for traces."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()
        integration.stop_watching()

        return {"status": "stopped", "message": "Trace watcher stopped"}

    except Exception as e:
        logger.error(f"Failed to stop watcher: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/policy")
async def get_security_policy():
    """Get the bashbros security policy if available."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        policy_path = integration.get_bashbros_policy_path()

        if policy_path and policy_path.exists():
            with open(policy_path, 'r') as f:
                content = f.read()

            return {
                "available": True,
                "path": str(policy_path),
                "content": content,
                "bashbros_primary": integration.should_use_bashbros_security(),
            }
        else:
            return {
                "available": False,
                "path": None,
                "content": None,
                "bashbros_primary": False,
            }

    except Exception as e:
        logger.error(f"Failed to get security policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/directory")
async def get_integration_directory():
    """Get the integration directory path and structure."""
    try:
        from bashgym.integrations.bashbros import get_integration
        integration = get_integration()

        return {
            "base_path": str(integration.integration_dir),
            "directories": {
                "traces_pending": str(integration.pending_dir),
                "traces_processed": str(integration.processed_dir),
                "traces_failed": str(integration.failed_dir),
                "models": str(integration.models_dir),
                "config": str(integration.config_dir),
                "status": str(integration.status_dir),
            },
            "exists": integration.integration_dir.exists(),
        }

    except Exception as e:
        logger.error(f"Failed to get directory info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
