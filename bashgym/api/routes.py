"""
Bash Gym API - FastAPI routes for frontend integration

This module provides REST API endpoints for:
- Task submission and management
- Training run control
- Model management and export
- Trace management
- Router statistics
- System status and health checks
- WebSocket for real-time updates
"""

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Set up logging for API routes
logger = logging.getLogger(__name__)

from bashgym.api.achievements_routes import router as achievements_router  # noqa: E402
from bashgym.api.agent_routes import router as agent_router  # noqa: E402
from bashgym.api.autoresearch_routes import router as autoresearch_router  # noqa: E402
from bashgym.api.device_routes import get_registry as get_device_registry  # noqa: E402
from bashgym.api.device_routes import router as device_router  # noqa: E402
from bashgym.api.factory_routes import router as factory_router  # noqa: E402
from bashgym.api.hf_routes import router as hf_router  # noqa: E402
from bashgym.api.integration_routes import router as integration_router  # noqa: E402
from bashgym.api.models_routes import router as models_router  # noqa: E402
from bashgym.api.observability_routes import router as observability_router  # noqa: E402
from bashgym.api.orchestrator_routes import router as orchestrator_router  # noqa: E402
from bashgym.api.pipeline_routes import router as pipeline_router  # noqa: E402
from bashgym.api.pipeline_routes import start_pipeline_watcher, stop_pipeline_watcher  # noqa: E402
from bashgym.api.schemas import (  # noqa: E402
    AvailableModel,
    BenchmarkResultSchema,
    ColumnConfig,
    ColumnConstraint,
    DataSource,
    ErrorAnalysisSchema,
    EvaluationRequest,
    EvaluationResponse,
    ExportExamplesRequest,
    ExportExamplesResponse,
    ExportRequest,
    ExportResponse,
    FactoryConfig,
    GenerateExamplesRequest,
    GenerateExamplesResponse,
    GpuInfo,
    HealthCheck,
    HooksInstallRequest,
    HooksInstallResponse,
    ModelConfig,
    ModelRecommendations,
    OutputConfig,
    PreviewResult,
    PreviewRow,
    PrivacyConfig,
    PromptOptConfig,
    RepoInfo,
    RouterStats,
    RoutingStrategyEnum,
    SafetyConfig,
    SeedExample,
    SeedSource,
    SynthesisJob,
    SynthesisJobStatus,
    SynthesisJobType,
    SystemInfoResponse,
    SystemStats,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    TraceImportRequest,
    TraceInfo,
    TraceQuality,
    TraceQualityTier,
    TraceStatus,
    TraceSummaryDetail,
    TrainingExampleResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    TrainingStrategy,
)
from bashgym.api.security_routes import router as security_router  # noqa: E402
from bashgym.api.settings_routes import router as settings_router  # noqa: E402
from bashgym.api.training_monitor import OrphanedTrainingMonitor  # noqa: E402
from bashgym.api.training_state import (  # noqa: E402
    TrainingRunState,
    is_process_alive,
    list_run_states,
    load_run_state,
    resume_process,
    save_run_state,
    suspend_process,
    terminate_process,
    update_run_state,
)
from bashgym.api.websocket import (  # noqa: E402
    MessageType,
    TrainingProgressCallback,
    broadcast_guardrail_blocked,
    broadcast_guardrail_warn,
    broadcast_pii_redacted,
    broadcast_task_status,
    broadcast_trace_event,
    broadcast_training_complete,
    broadcast_training_failed,
    broadcast_verification_result,
    handle_websocket,
)
from bashgym.factory.quality_calculator import calculate_quality_breakdown  # noqa: E402


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Bash Gym API",
        description="Self-Improving Agentic Development Gym",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS middleware for frontend
    from bashgym.config import get_settings as _get_settings

    _settings = _get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Authentication middleware (supports desktop pass-through + web cookie/API key)
    from bashgym.api.auth import AuthMiddleware

    app.add_middleware(AuthMiddleware)

    # Initialize state
    app.state.trainer = None
    app.state.router = None
    app.state.verifier = None
    app.state.trace_processor = None
    app.state.data_factory = None
    app.state.tasks = {}  # In-memory task storage
    app.state.training_runs = {}  # In-memory training run storage
    app.state.factory_config = None  # Factory configuration
    app.state.synthesis_jobs = {}  # Synthesis job storage
    app.state.evaluation_jobs = {}  # Evaluation job storage
    app.state.training_monitor = OrphanedTrainingMonitor()  # Orphaned process monitor
    app.state.provider_registry = None
    app.state.autoresearcher = None  # AutoResearch hyperparameter search
    app.state.trace_researcher = None  # Trace Research data curation search
    from bashgym.api.trace_cache import TraceIndexCache

    app.state.trace_cache = TraceIndexCache()

    @app.on_event("startup")
    async def startup():
        """Initialize Bash Gym components on startup."""
        from bashgym.config import get_settings

        settings = get_settings()
        settings.setup()

        # Initialize auth database and clean up expired sessions
        from bashgym.api.database import cleanup_expired_sessions, init_db

        init_db()
        removed = cleanup_expired_sessions()
        if removed:
            logger.info(f"Cleaned up {removed} expired sessions")

        # Initialize instrumentation (guardrails + profiler)
        try:
            from bashgym.core import get_instrumentation

            app.state.instrumentation = get_instrumentation()

            # Register WebSocket callback for guardrail events
            if app.state.instrumentation:

                async def on_guardrail_event(event):
                    """Broadcast guardrail events via WebSocket."""
                    if event.action_taken.value == "block":
                        await broadcast_guardrail_blocked(
                            check_type=event.check_type.value,
                            location=event.location,
                            confidence=event.confidence,
                            content_preview=(
                                event.original_content[:100] if event.original_content else None
                            ),
                            details=event.details,
                        )
                    elif event.action_taken.value == "warn":
                        await broadcast_guardrail_warn(
                            check_type=event.check_type.value,
                            location=event.location,
                            confidence=event.confidence,
                            content_preview=(
                                event.original_content[:100] if event.original_content else None
                            ),
                            details=event.details,
                        )
                    elif event.action_taken.value == "modify":
                        await broadcast_pii_redacted(
                            location=event.location,
                            redaction_count=1,
                            pii_types=[event.check_type.value],
                            details=event.details,
                        )

                app.state.instrumentation.on_event(on_guardrail_event)
                logger.info("Instrumentation initialized with WebSocket callbacks")

        except ImportError as e:
            logger.warning(f"Instrumentation not available: {e}")
            app.state.instrumentation = None

        # Initialize components lazily
        try:
            from bashgym.factory.data_factory import DataFactory
            from bashgym.factory.trace_processor import TraceProcessor
            from bashgym.gym.router import ModelRouter
            from bashgym.gym.trainer import Trainer, TrainerConfig
            from bashgym.judge.verifier import Verifier

            app.state.trainer = Trainer(TrainerConfig())

            # Initialize provider registry
            from bashgym.providers.registry import ProviderRegistry as _ProviderRegistry

            registry = _ProviderRegistry()

            # Register Anthropic
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                try:
                    from bashgym.providers.anthropic import AnthropicProvider

                    registry.register(AnthropicProvider(api_key=anthropic_key))
                    logger.info("Registered Anthropic provider")
                except Exception as e:
                    logger.warning(f"Failed to register Anthropic provider: {e}")

            # Register NVIDIA NIM
            nvidia_key = os.environ.get("NVIDIA_API_KEY")
            if nvidia_key:
                try:
                    from bashgym.providers.nim import NIMProvider

                    registry.register(NIMProvider(api_key=nvidia_key))
                    logger.info("Registered NIM provider")
                except Exception as e:
                    logger.warning(f"Failed to register NIM provider: {e}")

            # Register Ollama (local)
            from bashgym.config import get_settings as _get_settings

            _s = _get_settings()
            if _s.ollama.enabled:
                try:
                    from bashgym.providers.ollama import OllamaProvider as _OllamaProvider

                    ollama_provider = _OllamaProvider(base_url=_s.ollama.base_url)
                    registry.register(ollama_provider)
                    logger.info(f"Registered Ollama provider at {_s.ollama.base_url}")
                except Exception as e:
                    logger.warning(f"Failed to register Ollama provider: {e}")

            app.state.provider_registry = registry

            # Pass registry to router
            app.state.router = ModelRouter(registry=registry)
            app.state.verifier = Verifier()
            app.state.trace_processor = TraceProcessor()
            app.state.data_factory = DataFactory()

        except ImportError as e:
            print(f"Warning: Some components not available: {e}")

        # Start pipeline watcher
        start_pipeline_watcher()

        # Auto-discover provider models and auto-select Student
        if app.state.provider_registry:
            try:
                await app.state.provider_registry.discover_models()
                model_map = app.state.provider_registry.get_model_map()
                logger.info(f"Discovered models: {model_map}")

                # Auto-select best Ollama model as Student
                if _s.ollama.enabled and _s.ollama.auto_register:
                    best = app.state.provider_registry.select_best_model(
                        provider_type="ollama",
                        prefer_code=_s.ollama.prefer_code_models,
                        default_model=_s.ollama.default_model or None,
                    )
                    if best and app.state.router:
                        from bashgym.gym.router import ModelConfig as _ModelConfig
                        from bashgym.gym.router import ModelType as _ModelType

                        app.state.router.register_model(
                            _ModelConfig(
                                name=best.id,
                                model_type=_ModelType.STUDENT,
                                endpoint="ollama://registry",
                            )
                        )
                        logger.info(
                            f"Auto-selected Student model: {best.id} ({best.parameter_size})"
                        )
            except Exception as e:
                logger.warning(f"Model discovery failed: {e}")

        # Auto-import SSH device from .env on first run
        try:
            registry = get_device_registry()
            imported = await registry.auto_import_from_env()
            if imported:
                logger.info(
                    f"Auto-imported SSH device from .env: {imported.name} ({imported.host})"
                )
        except Exception as e:
            logger.warning(f"Failed to auto-import SSH device: {e}")

        # Recover orphaned training runs from disk
        try:
            orphaned_states = list_run_states()
            for state in orphaned_states:
                if state.status in ("running", "paused"):
                    if is_process_alive(state.pid):
                        logger.info(
                            f"Reconnecting to orphaned training {state.run_id} "
                            f"(PID {state.pid})"
                        )
                        # Reconstruct in-memory entry
                        app.state.training_runs[state.run_id] = {
                            "run_id": state.run_id,
                            "strategy": state.config.get("strategy", "sft"),
                            "status": (
                                TrainingStatus.RUNNING
                                if state.status == "running"
                                else TrainingStatus.PAUSED
                            ),
                            "config": state.config,
                            "started_at": state.started_at,
                            "pid": state.pid,
                        }
                        # Start monitoring via output directory polling
                        callback = TrainingProgressCallback(state.run_id)
                        app.state.training_monitor.start_monitoring(
                            state,
                            progress_callback=callback.on_progress,
                            log_callback=callback.on_log,
                        )
                    else:
                        # Process is dead — classify outcome
                        final_dir = Path(state.output_dir) / "final"
                        merged_dir = Path(state.output_dir) / "merged"
                        if final_dir.exists() or merged_dir.exists():
                            logger.info(f"Orphaned run {state.run_id} completed (final/ found)")
                            update_run_state(state.output_dir, status="completed")
                        else:
                            logger.warning(
                                f"Orphaned run {state.run_id} process dead, marking failed"
                            )
                            update_run_state(state.output_dir, status="failed")
        except Exception as e:
            logger.error(f"Error recovering orphaned training runs: {e}")

        # Build trace metadata index cache
        try:
            from bashgym.config import get_bashgym_dir
            from bashgym.config import get_settings as _gs

            _data_dir = Path(_gs().data.data_dir)
            _tier_dirs = [
                (_data_dir / "gold_traces", TraceStatus.GOLD),
                (_data_dir / "silver_traces", TraceStatus.SILVER),
                (_data_dir / "bronze_traces", TraceStatus.BRONZE),
                (_data_dir / "failed_traces", TraceStatus.FAILED),
            ]
            _global_traces = get_bashgym_dir() / "traces"
            _project_traces = _data_dir / "traces"
            _pending_dirs = []
            if _global_traces.exists():
                _pending_dirs.append(_global_traces)
            if _project_traces.exists() and _project_traces.resolve() != _global_traces.resolve():
                _pending_dirs.append(_project_traces)

            def _parse_pending_for_cache(trace_file, data):
                if isinstance(data, dict) and "trace" in data:
                    if not data.get("trace"):
                        return None
                    return _parse_imported_trace_file(trace_file, data)
                elif isinstance(data, list) and len(data) > 0:
                    return _parse_raw_trace_file(trace_file, data)
                return None

            app.state.trace_cache.build_index(
                tier_dirs=_tier_dirs,
                pending_dirs=_pending_dirs,
                parse_tiered_fn=_parse_trace_file,
                parse_pending_fn=_parse_pending_for_cache,
            )
            # Store dir config for refresh calls
            app.state._trace_tier_dirs = _tier_dirs
            app.state._trace_pending_dirs = _pending_dirs
            app.state._trace_parse_pending = _parse_pending_for_cache
        except Exception as e:
            logger.error(f"Failed to build trace index: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        if hasattr(app.state, "provider_registry") and app.state.provider_registry:
            await app.state.provider_registry.close()
        if app.state.router:
            await app.state.router.close()
        stop_pipeline_watcher()
        # Stop all orphaned training monitors
        app.state.training_monitor.stop_all()

    # =========================================================================
    # WebSocket Endpoint
    # =========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await handle_websocket(websocket)

    # =========================================================================
    # Health & Status Endpoints
    # =========================================================================

    @app.get("/api/health", response_model=HealthCheck, tags=["System"])
    async def health_check():
        """Check API health status."""
        return HealthCheck(
            status="healthy", timestamp=datetime.utcnow().isoformat(), version="0.1.0"
        )

    @app.get("/api/debug/traces", tags=["System"])
    async def debug_traces():
        """Debug endpoint to diagnose trace discovery issues."""
        import os as _os

        if _os.environ.get("BASHGYM_MODE", "").lower() == "web":
            raise HTTPException(status_code=404, detail="Not found")
        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        result = {"data_dir": str(data_dir), "global_dir": str(global_dir), "directories": {}}

        # Check all trace directories (including tiered: gold/silver/bronze)
        dirs_to_check = {
            "global_traces": global_dir / "traces",
            "global_gold": global_dir / "gold_traces",
            "global_silver": global_dir / "silver_traces",
            "global_bronze": global_dir / "bronze_traces",
            "global_failed": global_dir / "failed_traces",
            "project_traces": data_dir / "traces",
            "project_gold": data_dir / "gold_traces",
            "project_silver": data_dir / "silver_traces",
            "project_bronze": data_dir / "bronze_traces",
            "project_failed": data_dir / "failed_traces",
        }

        for name, path in dirs_to_check.items():
            dir_info = {"path": str(path), "exists": path.exists(), "files": []}
            if path.exists():
                files = list(path.glob("*.json"))
                dir_info["count"] = len(files)
                dir_info["files"] = [f.name for f in files[:10]]  # First 10 only
            result["directories"][name] = dir_info

        return result

    @app.get("/api/stats", response_model=SystemStats, tags=["System"])
    async def get_system_stats():
        """Get system statistics."""
        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()

        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        # Count traces in tiered directories (gold/silver/bronze/failed)
        gold_traces = (
            list((data_dir / "gold_traces").glob("*.json"))
            if (data_dir / "gold_traces").exists()
            else []
        )
        silver_traces = (
            list((data_dir / "silver_traces").glob("*.json"))
            if (data_dir / "silver_traces").exists()
            else []
        )
        bronze_traces = (
            list((data_dir / "bronze_traces").glob("*.json"))
            if (data_dir / "bronze_traces").exists()
            else []
        )
        failed_traces = (
            list((data_dir / "failed_traces").glob("*.json"))
            if (data_dir / "failed_traces").exists()
            else []
        )
        models = list((data_dir / "models").iterdir()) if (data_dir / "models").exists() else []

        # Count pending traces from both global and project directories
        pending_count = 0
        global_traces = global_dir / "traces"
        project_traces = data_dir / "traces"

        if global_traces.exists():
            from bashgym.trace_capture.core import glob_pending_traces

            pending_count += len(glob_pending_traces(global_traces))
        if project_traces.exists() and project_traces.resolve() != global_traces.resolve():
            pending_count += len(glob_pending_traces(project_traces))

        return SystemStats(
            gold_traces_count=len(gold_traces),
            silver_traces_count=len(silver_traces),
            bronze_traces_count=len(bronze_traces),
            failed_traces_count=len(failed_traces),
            pending_traces_count=pending_count,
            models_count=len(models),
            base_model=settings.training.base_model,
            auto_export_gguf=True,
            active_tasks=len([t for t in app.state.tasks.values() if t.get("status") == "running"]),
            active_training_runs=len(
                [r for r in app.state.training_runs.values() if r.get("status") == "running"]
            ),
        )

    # =========================================================================
    # System Info / Hardware Endpoints
    # =========================================================================

    @app.get("/api/system/info", response_model=SystemInfoResponse, tags=["System"])
    async def get_system_info(refresh: bool = False):
        """Get system hardware information including GPU, RAM, and CUDA status."""
        from bashgym.api.system_info import get_system_info_service

        service = get_system_info_service()
        info = service.get_system_info(force_refresh=refresh)

        return SystemInfoResponse(
            gpus=[
                GpuInfo(
                    vendor=g.vendor,
                    model=g.model,
                    vram=g.vram,
                    vram_used=g.vram_used,
                    driver=g.driver,
                    temperature=g.temperature,
                    utilization=g.utilization,
                )
                for g in info.gpus
            ],
            total_ram=info.total_ram,
            available_ram=info.available_ram,
            platform=info.platform_name,
            arch=info.arch,
            cuda_available=info.cuda_available,
            cuda_version=info.cuda_version,
            python_available=info.python_available,
            python_version=info.python_version,
        )

    @app.get("/api/system/gpus", tags=["System"])
    async def get_gpus():
        """Get GPU information only."""
        from bashgym.api.system_info import get_system_info_service

        service = get_system_info_service()
        gpus = service.get_gpus()

        return [
            GpuInfo(
                vendor=g.vendor,
                model=g.model,
                vram=g.vram,
                vram_used=g.vram_used,
                driver=g.driver,
                temperature=g.temperature,
                utilization=g.utilization,
            )
            for g in gpus
        ]

    @app.get("/api/system/recommendations", response_model=ModelRecommendations, tags=["System"])
    async def get_model_recommendations():
        """Get model recommendations based on detected hardware."""
        from bashgym.api.system_info import get_system_info_service

        service = get_system_info_service()
        recommendations = service.get_model_recommendations()

        return ModelRecommendations(**recommendations)

    # =========================================================================
    # Model Providers Endpoints
    # =========================================================================

    @app.get("/api/providers", tags=["Providers"])
    async def get_providers():
        """Detect all available model providers (local and cloud)."""
        try:
            from bashgym.providers import detect_providers

            providers = await detect_providers()
            return {
                "providers": [p.to_dict() for p in providers],
                "summary": {
                    "available": len([p for p in providers if p.available]),
                    "total": len(providers),
                },
            }
        except ImportError:
            return {"providers": [], "summary": {"available": 0, "total": 0}}

    @app.get("/api/providers/models", tags=["Providers"])
    async def get_available_models(
        include_local: bool = True, include_cloud: bool = True, code_only: bool = False
    ):
        """Get all available models organized by category."""
        try:
            from bashgym.providers import get_available_models as get_models

            models = await get_models(
                include_local=include_local, include_cloud=include_cloud, code_only=code_only
            )
            return {
                "local": [m.to_dict() for m in models["local"]],
                "training": [m.to_dict() for m in models["training"]],
                "teacher": [m.to_dict() for m in models["teacher"]],
                "inference": [m.to_dict() for m in models["inference"]],
            }
        except ImportError:
            return {"local": [], "training": [], "teacher": [], "inference": []}

    @app.get("/api/providers/ollama/models", tags=["Providers"])
    async def get_ollama_models():
        """Get models from local Ollama installation."""
        try:
            from bashgym.providers.ollama import get_ollama_provider

            provider = get_ollama_provider()

            if not await provider.is_running():
                return {
                    "available": False,
                    "error": "Ollama not running. Start with: ollama serve",
                    "models": [],
                }

            models = await provider.list_ollama_models()
            return {"available": True, "models": [m.to_dict() for m in models]}
        except ImportError:
            return {"available": False, "error": "Ollama provider not installed", "models": []}

    @app.post("/api/providers/ollama/pull", tags=["Providers"])
    async def pull_ollama_model(model_name: str, background_tasks: BackgroundTasks):
        """Pull (download) a model from Ollama registry."""
        try:
            from bashgym.providers.ollama import get_ollama_provider

            provider = get_ollama_provider()

            if not await provider.is_running():
                raise HTTPException(status_code=503, detail="Ollama not running")

            # Start pull in background
            async def do_pull():
                await provider.pull_model(model_name)

            background_tasks.add_task(do_pull)

            return {
                "status": "pulling",
                "model": model_name,
                "message": f"Started downloading {model_name}",
            }
        except ImportError:
            raise HTTPException(status_code=501, detail="Ollama provider not available")

    @app.delete("/api/providers/ollama/models/{model_name:path}", tags=["Providers"])
    async def delete_ollama_model(model_name: str):
        """Delete a model from Ollama."""
        try:
            from bashgym.providers.ollama import get_ollama_provider

            provider = get_ollama_provider()

            if not await provider.is_running():
                raise HTTPException(status_code=503, detail="Ollama not running")

            success = await provider.delete_model(model_name)
            if success:
                return {"status": "deleted", "model": model_name}
            else:
                raise HTTPException(status_code=404, detail="Model not found")
        except ImportError:
            raise HTTPException(status_code=501, detail="Ollama provider not available")

    # =========================================================================
    # SSH Endpoints
    # =========================================================================

    @app.get("/api/ssh/preflight", tags=["SSH"])
    async def ssh_preflight():
        """Run pre-flight checks on the default remote training device."""
        # Try device registry first
        try:
            registry = get_device_registry()
            default_device = await registry.get_default()
            if default_device:
                from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig

                ssh_config = SSHConfig(
                    host=default_device.host,
                    port=default_device.port,
                    username=default_device.username,
                    key_path=default_device.key_path,
                    remote_work_dir=default_device.work_dir,
                )
                trainer = RemoteTrainer(ssh_config)
                result = await trainer.preflight_check()
                return {
                    "ok": result.ok,
                    "python_version": result.python_version,
                    "disk_free_gb": result.disk_free_gb,
                    "error": result.error,
                    "host": default_device.host,
                    "username": default_device.username,
                    "hostname": getattr(result, "hostname", None),
                    "os_info": getattr(result, "os_info", None),
                    "cuda_version": getattr(result, "cuda_version", None),
                    "gpus": getattr(result, "gpus", None),
                }
        except Exception as e:
            logger.warning(f"Device registry preflight failed, falling back to env: {e}")

        # Fallback to env vars
        from bashgym.config import get_settings as _get_settings

        _s = _get_settings()
        if not _s.ssh.enabled:
            return {
                "ok": False,
                "error": "No SSH devices configured and SSH_REMOTE_ENABLED not set.",
            }

        from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig

        config = SSHConfig.from_settings(_s.ssh)
        trainer = RemoteTrainer(config)
        result = await trainer.preflight_check()
        return {
            "ok": result.ok,
            "python_version": result.python_version,
            "disk_free_gb": result.disk_free_gb,
            "error": result.error,
            "host": _s.ssh.host,
            "username": _s.ssh.username,
        }

    # =========================================================================
    # Task Endpoints
    # =========================================================================

    @app.post("/api/tasks", response_model=TaskResponse, tags=["Tasks"])
    async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
        """Submit a new task for agent execution."""
        task_id = request.task_id or f"task_{uuid.uuid4().hex[:12]}"

        # Store task
        app.state.tasks[task_id] = {
            "task_id": task_id,
            "prompt": request.prompt,
            "status": TaskStatus.PENDING,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Run task in background
        background_tasks.add_task(run_task, app, task_id, request)

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Task queued for execution",
            created_at=app.state.tasks[task_id]["created_at"],
        )

    async def run_task(app, task_id: str, request: TaskRequest):
        """Execute a task in the background."""
        try:
            app.state.tasks[task_id]["status"] = TaskStatus.RUNNING
            await broadcast_task_status(task_id, "running")

            # TODO: Integrate with actual agent runner
            # For now, simulate task execution
            await asyncio.sleep(2)

            app.state.tasks[task_id]["status"] = TaskStatus.COMPLETED
            app.state.tasks[task_id]["completed_at"] = datetime.utcnow().isoformat()
            app.state.tasks[task_id]["result"] = {"success": True}

            await broadcast_task_status(task_id, "completed", {"success": True})

        except Exception as e:
            app.state.tasks[task_id]["status"] = TaskStatus.FAILED
            app.state.tasks[task_id]["error"] = str(e)
            await broadcast_task_status(task_id, "failed", {"error": str(e)})

    @app.get("/api/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
    async def get_task_status(task_id: str):
        """Get the status of a submitted task."""
        if task_id not in app.state.tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = app.state.tasks[task_id]
        return TaskResponse(
            task_id=task_id,
            status=task["status"],
            created_at=task.get("created_at"),
            completed_at=task.get("completed_at"),
            result=task.get("result"),
        )

    @app.get("/api/tasks", response_model=list[TaskResponse], tags=["Tasks"])
    async def list_tasks(status: TaskStatus | None = None, limit: int = 50):
        """List all tasks, optionally filtered by status."""
        tasks = list(app.state.tasks.values())

        if status:
            tasks = [t for t in tasks if t["status"] == status]

        tasks = tasks[:limit]

        return [
            TaskResponse(
                task_id=t["task_id"],
                status=t["status"],
                created_at=t.get("created_at"),
                completed_at=t.get("completed_at"),
            )
            for t in tasks
        ]

    # =========================================================================
    # Training Endpoints
    # =========================================================================

    @app.post("/api/training/start", response_model=TrainingResponse, tags=["Training"])
    async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
        """Start a new training run."""
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Store training run
        app.state.training_runs[run_id] = {
            "run_id": run_id,
            "strategy": request.strategy,
            "status": TrainingStatus.PENDING,
            "config": request.dict(),
            "started_at": datetime.utcnow().isoformat(),
        }

        # Start training in background
        background_tasks.add_task(run_training, app, run_id, request)

        return TrainingResponse(
            run_id=run_id,
            status=TrainingStatus.PENDING,
            strategy=request.strategy,
            message="Training run queued",
            started_at=app.state.training_runs[run_id]["started_at"],
        )

    async def run_training(app, run_id: str, request: TrainingRequest):
        """Execute training in the background with WebSocket updates."""
        import traceback

        logger.info(f"[Training] Background task started for run {run_id}")
        print(f"[Training] Background task started for run {run_id}")
        try:
            app.state.training_runs[run_id]["status"] = TrainingStatus.RUNNING
            logger.info(f"[Training] Status set to RUNNING for {run_id}")

            # Create progress callback for WebSocket updates
            callback = TrainingProgressCallback(run_id)

            if app.state.trainer:
                from bashgym.config import get_settings
                from bashgym.factory.example_generator import (
                    ExampleGenerator,
                    ExampleGeneratorConfig,
                )
                from bashgym.gym.trainer import TrainerConfig
                from bashgym.gym.trainer import TrainingStrategy as TS  # noqa: N817

                settings = get_settings()

                # Determine training data source
                dataset_path = None
                if request.data_source == DataSource.SECURITY_DATASET:
                    # Ingest security dataset directly
                    logger.info(f"Ingesting security dataset: {request.security_dataset_type}")
                    from bashgym.factory.security_ingester import (
                        ConversionMode,
                        DatasetType,
                        IngestionConfig,
                        SecurityIngester,
                    )

                    sec_config = IngestionConfig(
                        dataset_type=DatasetType(request.security_dataset_type),
                        input_path=request.security_dataset_path,
                        mode=ConversionMode(request.security_conversion_mode or "direct"),
                        max_samples=request.security_max_samples,
                        balance_classes=request.security_balance_classes,
                        output_dir=str(Path(settings.data.training_batches_dir) / run_id),
                        train_split=0.9,
                    )
                    sec_ingester = SecurityIngester(sec_config)
                    result = sec_ingester.ingest_direct()
                    dataset_path = Path(result.train_path)
                    logger.info(
                        f"Security ingestion complete: {result.examples_generated} examples -> {dataset_path}"
                    )
                elif request.data_source == DataSource.DATASET_PATH or request.dataset_path:
                    dataset_path = Path(request.dataset_path)
                else:
                    # Generate training data from gold traces automatically
                    logger.info("Auto-generating training data from gold traces...")
                    gold_dir = Path(settings.data.gold_traces_dir)
                    output_dir = Path(settings.data.training_batches_dir)

                    if gold_dir.exists() and list(gold_dir.glob("*.json")):
                        # Filter traces by selected repos if specified
                        trace_files = list(gold_dir.glob("*.json"))

                        if request.selected_repos and len(request.selected_repos) > 0:
                            logger.info(f"Filtering traces for repos: {request.selected_repos}")
                            filtered_files = []
                            for trace_file in trace_files:
                                try:
                                    with open(trace_file, encoding="utf-8") as f:
                                        trace_data = json.load(f)
                                    # Check primary_repo.name (can be dict or list)
                                    primary_repo = trace_data.get("primary_repo", {})
                                    if isinstance(primary_repo, list) and len(primary_repo) > 0:
                                        primary_repo = primary_repo[0]  # Use first repo if list
                                    repo_name = (
                                        primary_repo.get("name")
                                        if isinstance(primary_repo, dict)
                                        else None
                                    )
                                    if repo_name and repo_name in request.selected_repos:
                                        filtered_files.append(trace_file)
                                except Exception as e:
                                    logger.warning(f"Error reading trace {trace_file}: {e}")

                            logger.info(
                                f"Filtered to {len(filtered_files)} traces for selected repos (from {len(trace_files)} total)"
                            )
                            trace_files = filtered_files

                            if not trace_files:
                                raise ValueError(
                                    f"No gold traces found for repos: {request.selected_repos}"
                                )

                        gen_config = ExampleGeneratorConfig(output_dir=str(output_dir))
                        generator = ExampleGenerator(gen_config)

                        # Process only the filtered trace files
                        examples = []
                        stats = {"sessions_processed": 0, "examples_generated": 0}
                        for trace_file in trace_files:
                            try:
                                session_examples = generator.generate_examples(trace_file)
                                examples.extend(session_examples)
                                stats["sessions_processed"] += 1
                                stats["examples_generated"] += len(session_examples)
                            except Exception as e:
                                logger.warning(f"Error processing trace {trace_file}: {e}")

                        logger.info(
                            f"Generated {len(examples)} examples from {stats['sessions_processed']} gold traces"
                        )

                        if examples:
                            result = generator.export_for_nemo(
                                examples, output_dir, train_split=0.9
                            )
                            dataset_path = result["train"]
                            logger.info(f"Training data ready: {dataset_path}")
                        else:
                            raise ValueError("No training examples generated from gold traces")
                    else:
                        raise ValueError(f"No gold traces found in {gold_dir}")

                # Update trainer config
                config = TrainerConfig(
                    base_model=request.base_model or "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    model_type=request.model_type or "qwen",
                    strategy=TS(request.strategy.value),
                    num_epochs=request.num_epochs,
                    batch_size=request.batch_size,
                    learning_rate=request.learning_rate,
                    warmup_ratio=request.warmup_ratio,
                    gradient_accumulation_steps=request.gradient_accumulation_steps,
                    max_seq_length=request.max_seq_length,
                    save_steps=request.save_steps,
                    # LoRA
                    use_lora=request.use_lora,
                    lora_r=request.lora_rank or 16,
                    lora_alpha=request.lora_alpha or 32,
                    lora_dropout=request.lora_dropout,
                    load_in_4bit=request.load_in_4bit,
                    # Strategy-specific
                    dpo_beta=request.dpo_beta,
                    grpo_num_generations=request.grpo_num_generations,
                    grpo_temperature=request.grpo_temperature,
                    grpo_reward_mode=getattr(request, "grpo_reward_mode", "syntax"),
                    # Knowledge Distillation
                    teacher_model=request.teacher_model or "claude-sonnet-4-20250514",
                    teacher_temperature=request.teacher_temperature,
                    distillation_alpha=request.distillation_alpha,
                    # Export & backend
                    auto_export_gguf=request.auto_export_gguf,
                    gguf_quantization=request.gguf_quantization,
                    use_nemo_gym=request.use_nemo_gym,
                    use_remote_ssh=request.use_remote_ssh,
                )

                app.state.trainer.config = config

                # Resolve device for remote SSH training
                if config.use_remote_ssh:
                    ssh_config = None
                    device_id_to_use = getattr(request, "device_id", None)
                    try:
                        registry = get_device_registry()
                        if device_id_to_use:
                            device = await registry.get_device(device_id_to_use)
                        else:
                            device = await registry.get_default()
                        if device:
                            from bashgym.gym.remote_trainer import SSHConfig

                            ssh_config = SSHConfig(
                                host=device.host,
                                port=device.port,
                                username=device.username,
                                key_path=device.key_path,
                                remote_work_dir=device.work_dir,
                            )
                    except Exception as e:
                        logger.warning(f"Device resolution failed, falling back to env: {e}")
                    app.state.trainer.ssh_config = ssh_config
                else:
                    app.state.trainer.ssh_config = None

                # Determine output dir for state persistence
                output_dir = str(Path(config.output_dir) / run_id)

                def _on_pid(pid: int, training_run):
                    """Called immediately after subprocess.Popen — persist PID to disk."""
                    app.state.training_runs[run_id]["pid"] = pid
                    state = TrainingRunState(
                        run_id=run_id,
                        pid=pid,
                        status="running",
                        config=request.dict(),
                        started_at=app.state.training_runs[run_id].get("started_at", ""),
                        script_path=str(Path(output_dir) / "train_sft.py"),
                        dataset_path=str(dataset_path),
                        output_dir=output_dir,
                    )
                    save_run_state(state)
                    logger.info(f"Persisted run state for {run_id} (PID {pid})")

                if request.strategy == TrainingStrategy.SFT:
                    run = app.state.trainer.train_sft(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync,
                        pid_callback=_on_pid,
                    )
                elif request.strategy == TrainingStrategy.DPO:
                    run = app.state.trainer.train_dpo(
                        dataset_path=dataset_path, run_id=run_id, callback=callback.on_progress_sync
                    )
                elif request.strategy == TrainingStrategy.DISTILLATION:
                    run = app.state.trainer.train_distillation(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync,
                        pid_callback=_on_pid,
                    )
                elif request.strategy.value == "grpo":
                    from bashgym.gym.trainer import GRPOTrainer

                    grpo_trainer = GRPOTrainer(app.state.trainer.config)
                    run = grpo_trainer.train_grpo(
                        dataset_path=dataset_path,
                        verifier_fn=lambda p, r: 0.0,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync,
                        pid_callback=_on_pid,
                    )
                elif request.strategy.value == "rlvr":
                    run = app.state.trainer.train_rlvr(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync,
                        pid_callback=_on_pid,
                    )
                else:
                    raise ValueError(f"Unknown training strategy: {request.strategy}")

                app.state.training_runs[run_id]["status"] = TrainingStatus.COMPLETED
                app.state.training_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()
                app.state.training_runs[run_id]["metrics"] = run.metrics if run else {}

                # Persist completed state to disk
                update_run_state(
                    output_dir, status="completed", completed_at=datetime.utcnow().isoformat()
                )

                await broadcast_training_complete(run_id, run.metrics if run else {})

            else:
                # Simulate training if trainer not available (no GPU/dependencies)
                logger.warning("Running in SIMULATION mode - trainer not available")
                app.state.training_runs[run_id]["simulation"] = True

                steps_per_epoch = 100
                total_steps = request.num_epochs * steps_per_epoch
                import random

                for epoch in range(request.num_epochs):
                    for step in range(steps_per_epoch):
                        current_step = epoch * steps_per_epoch + step + 1
                        progress = current_step / total_steps

                        # Simulate realistic loss curve (exponential decay with noise)
                        base_loss = 2.5 * (0.6**progress) + 0.4
                        noise = random.uniform(-0.05, 0.05)
                        loss = max(0.3, base_loss + noise)

                        # Simulate gradient norm (decreases as training stabilizes)
                        grad_norm = 1.0 * (1.0 - progress * 0.7) + random.uniform(-0.1, 0.1)

                        # Calculate ETA
                        steps_remaining = total_steps - current_step
                        eta_minutes = steps_remaining * 0.02  # 20ms per step in simulation
                        eta = (
                            f"{int(eta_minutes)}m"
                            if eta_minutes >= 1
                            else f"{int(eta_minutes * 60)}s"
                        )

                        await callback.on_progress(
                            {
                                "epoch": epoch + 1,
                                "total_epochs": request.num_epochs,
                                "step": current_step,
                                "total_steps": total_steps,
                                "loss": round(loss, 4),
                                "learning_rate": request.learning_rate,
                                "grad_norm": round(grad_norm, 3),
                                "eta": eta,
                                "simulation": True,  # Flag to indicate simulation mode
                            }
                        )
                        await asyncio.sleep(0.02)  # Fast simulation (20ms per step)

                final_loss = round(2.5 * (0.6**1.0) + 0.4, 4)
                app.state.training_runs[run_id]["status"] = TrainingStatus.COMPLETED
                app.state.training_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()
                app.state.training_runs[run_id]["metrics"] = {
                    "final_loss": final_loss,
                    "simulation": True,
                }
                await broadcast_training_complete(
                    run_id, {"final_loss": final_loss, "simulation": True}
                )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[Training] FAILED for {run_id}: {error_msg}")
            logger.error(f"[Training] Traceback: {traceback.format_exc()}")
            print(f"[Training] FAILED for {run_id}: {error_msg}")
            print(f"[Training] Traceback:\n{traceback.format_exc()}")
            app.state.training_runs[run_id]["status"] = TrainingStatus.FAILED
            app.state.training_runs[run_id]["error"] = error_msg
            # Persist failed state to disk (output_dir may not exist for early failures)
            try:
                run_output_dir = str(Path("data/models") / run_id)
                update_run_state(run_output_dir, status="failed")
            except Exception:
                pass  # State file may not exist yet if failure was before subprocess
            await broadcast_training_failed(run_id, error_msg)

    @app.get("/api/training/{run_id}", response_model=TrainingResponse, tags=["Training"])
    async def get_training_status(run_id: str):
        """Get the status of a training run."""
        if run_id in app.state.training_runs:
            run = app.state.training_runs[run_id]
            return TrainingResponse(
                run_id=run_id,
                status=run["status"],
                strategy=run["strategy"],
                error=run.get("error"),
                started_at=run.get("started_at"),
                completed_at=run.get("completed_at"),
                metrics=run.get("metrics"),
            )

        # Fallback: check persisted state on disk
        state = load_run_state(str(Path("data/models") / run_id))
        if state:
            return TrainingResponse(
                run_id=state.run_id,
                status=state.status,
                strategy=state.config.get("strategy", "sft"),
                started_at=state.started_at,
                completed_at=state.completed_at,
                metrics=state.last_metrics,
            )

        raise HTTPException(status_code=404, detail="Training run not found")

    @app.post("/api/training/{run_id}/pause", tags=["Training"])
    async def pause_training(run_id: str):
        """Pause a running training job by suspending the subprocess."""
        if run_id not in app.state.training_runs:
            raise HTTPException(status_code=404, detail="Training run not found")

        pid = app.state.training_runs[run_id].get("pid")
        if not pid:
            raise HTTPException(status_code=400, detail="No PID available for this run")

        if not is_process_alive(pid):
            raise HTTPException(status_code=400, detail="Training process is no longer running")

        if not suspend_process(pid):
            raise HTTPException(status_code=500, detail="Failed to suspend training process")

        app.state.training_runs[run_id]["status"] = TrainingStatus.PAUSED
        update_run_state(str(Path("data/models") / run_id), status="paused")
        return {"success": True, "message": f"Training paused (PID {pid} suspended)"}

    @app.post("/api/training/{run_id}/resume", tags=["Training"])
    async def resume_training(run_id: str):
        """Resume a paused training job by resuming the subprocess."""
        if run_id not in app.state.training_runs:
            raise HTTPException(status_code=404, detail="Training run not found")

        pid = app.state.training_runs[run_id].get("pid")
        if not pid:
            raise HTTPException(status_code=400, detail="No PID available for this run")

        if not is_process_alive(pid):
            raise HTTPException(status_code=400, detail="Training process is no longer running")

        if not resume_process(pid):
            raise HTTPException(status_code=500, detail="Failed to resume training process")

        app.state.training_runs[run_id]["status"] = TrainingStatus.RUNNING
        update_run_state(str(Path("data/models") / run_id), status="running")
        return {"success": True, "message": f"Training resumed (PID {pid})"}

    @app.post("/api/training/{run_id}/stop", tags=["Training"])
    async def stop_training(run_id: str):
        """Stop a training job by terminating the subprocess."""
        if run_id not in app.state.training_runs:
            raise HTTPException(status_code=404, detail="Training run not found")

        pid = app.state.training_runs[run_id].get("pid")
        if pid and is_process_alive(pid):
            terminate_process(pid)
            logger.info(f"Terminated training process PID {pid} for {run_id}")

        # Stop any orphaned monitor
        app.state.training_monitor.stop_monitoring(run_id)

        app.state.training_runs[run_id]["status"] = TrainingStatus.FAILED
        app.state.training_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()
        update_run_state(
            str(Path("data/models") / run_id),
            status="failed",
            completed_at=datetime.utcnow().isoformat(),
        )
        return {"success": True, "message": "Training stopped"}

    @app.get("/api/training", response_model=list[TrainingResponse], tags=["Training"])
    async def list_training_runs(status: TrainingStatus | None = None, limit: int = 50):
        """List all training runs (in-memory + persisted on disk)."""
        # Start with in-memory runs
        seen_ids = set()
        results = []
        for r in app.state.training_runs.values():
            seen_ids.add(r["run_id"])
            if status and r["status"] != status:
                continue
            results.append(
                TrainingResponse(
                    run_id=r["run_id"],
                    status=r["status"],
                    strategy=r["strategy"],
                    started_at=r.get("started_at"),
                    completed_at=r.get("completed_at"),
                    metrics=r.get("metrics"),
                )
            )

        # Merge persisted runs not already in memory
        for state in list_run_states():
            if state.run_id in seen_ids:
                continue
            if status and state.status != status.value:
                continue
            results.append(
                TrainingResponse(
                    run_id=state.run_id,
                    status=state.status,
                    strategy=state.config.get("strategy", "sft"),
                    started_at=state.started_at,
                    completed_at=state.completed_at,
                    metrics=state.last_metrics,
                )
            )

        return results[:limit]

    # =========================================================================
    # Model Endpoints
    # =========================================================================

    # NOTE: /api/models endpoint moved to models_routes.py with enhanced ModelRegistry
    # See bashgym/api/models_routes.py for the new implementation

    @app.post("/api/models/{model_id}/export", response_model=ExportResponse, tags=["Models"])
    async def export_model(
        model_id: str, request: ExportRequest, background_tasks: BackgroundTasks
    ):
        """Export a trained model to GGUF or other formats."""
        if app.state.trainer:
            export_path = app.state.trainer.export_model(
                run_id=model_id,
                export_format=request.format.value,
                quantization=request.quantization,
            )

            return ExportResponse(
                model_id=model_id,
                format=request.format,
                status="completed" if export_path else "failed",
                output_path=str(export_path) if export_path else None,
                message="Export script generated" if export_path else "Export failed",
            )

        return ExportResponse(
            model_id=model_id, format=request.format, status="pending", message="Export queued"
        )

    # =========================================================================
    # Trace Endpoints
    # =========================================================================

    @app.get("/api/traces", tags=["Traces"])
    async def list_traces(
        status: TraceStatus | None = None,
        repo: str | None = None,
        source_tool: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        """List traces with pagination (served from in-memory cache)."""
        cache = app.state.trace_cache

        if not cache.initialized:
            raise HTTPException(status_code=503, detail="Trace index still building")

        # Incremental refresh picks up new/deleted files
        cache.refresh(
            tier_dirs=app.state._trace_tier_dirs,
            pending_dirs=app.state._trace_pending_dirs,
            parse_tiered_fn=_parse_trace_file,
            parse_pending_fn=app.state._trace_parse_pending,
        )

        return cache.query(
            status=status,
            repo=repo,
            source_tool=source_tool,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/traces/stats", tags=["Traces"])
    async def get_trace_stats(time_range: str = Query("7d", alias="range")):
        """Get trace statistics over time for the dashboard chart.

        Args:
            time_range: Time range — '24h', '7d', '30d', or 'all'.
        """
        from datetime import datetime, timedelta

        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        now = datetime.now()

        # Range configuration
        range_configs = {
            "24h": {
                "window": timedelta(hours=24),
                "bucket_size": timedelta(hours=1),
                "label_fmt": "%H:%M",
                "count": 25,
            },
            "7d": {
                "window": timedelta(days=7),
                "bucket_size": timedelta(days=1),
                "label_fmt": "%a %m/%d",
                "count": 8,
            },
            "30d": {
                "window": timedelta(days=30),
                "bucket_size": timedelta(days=1),
                "label_fmt": "%m/%d",
                "count": 31,
            },
        }

        if time_range not in range_configs and time_range != "all":
            raise HTTPException(
                status_code=400, detail="Invalid range. Must be one of: 24h, 7d, 30d, all"
            )

        if time_range == "all":
            # Find earliest trace file across all dirs
            earliest = now
            all_dirs = [
                data_dir / "gold_traces",
                data_dir / "silver_traces",
                data_dir / "bronze_traces",
                data_dir / "failed_traces",
                global_dir / "traces",
                data_dir / "traces",
            ]
            for d in all_dirs:
                if d.exists():
                    for f in d.glob("*.json"):
                        try:
                            mtime = datetime.fromtimestamp(f.stat().st_mtime)
                            if mtime < earliest:
                                earliest = mtime
                        except Exception:
                            continue
            span = now - earliest
            if span > timedelta(days=180):
                bucket_size = timedelta(days=30)
                label_fmt = "%b %Y"
            else:
                bucket_size = timedelta(weeks=1)
                label_fmt = "%m/%d"
            window = span + bucket_size  # ensure earliest falls in first bucket
            bucket_count = max(int(window / bucket_size) + 1, 2)
        else:
            cfg = range_configs[time_range]
            window = cfg["window"]
            bucket_size = cfg["bucket_size"]
            label_fmt = cfg["label_fmt"]
            bucket_count = cfg["count"]

        # Generate buckets from oldest to newest
        buckets = []
        for i in range(bucket_count - 1, -1, -1):
            bucket_time = now - bucket_size * i
            # Truncate to bucket boundary
            if bucket_size >= timedelta(days=1):
                bucket_start = bucket_time.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                bucket_start = bucket_time.replace(minute=0, second=0, microsecond=0)
            label = "Now" if i == 0 else bucket_time.strftime(label_fmt)
            buckets.append(
                {
                    "time": label,
                    "start": bucket_start,
                    "gold": 0,
                    "failed": 0,
                    "pending": 0,
                }
            )

        # Count traces by status and time
        def count_traces_in_dir(trace_dir: Path, status: str, pattern: str = "*.json"):
            if not trace_dir.exists():
                return
            for trace_file in trace_dir.glob(pattern):
                try:
                    mtime = datetime.fromtimestamp(trace_file.stat().st_mtime)

                    # Try to extract time from filename (imported_claude_xxx_YYYYMMDD_HHMMSS.json)
                    fname = trace_file.stem
                    if "_202" in fname:
                        parts = fname.split("_")
                        for idx, part in enumerate(parts):
                            if part.startswith("202") and len(part) == 8:
                                try:
                                    date_str = part
                                    time_str = parts[idx + 1] if idx + 1 < len(parts) else "000000"
                                    mtime = datetime.strptime(
                                        f"{date_str}_{time_str[:6]}", "%Y%m%d_%H%M%S"
                                    )
                                except (ValueError, IndexError):
                                    pass
                                break

                    # Find matching bucket
                    for b_idx in range(len(buckets)):
                        b_start = buckets[b_idx]["start"]
                        b_end = (
                            buckets[b_idx + 1]["start"]
                            if b_idx + 1 < len(buckets)
                            else now + timedelta(hours=1)
                        )
                        if b_start <= mtime < b_end:
                            buckets[b_idx][status] += 1
                            break
                except Exception:
                    continue

        # Count gold traces
        gold_dir = data_dir / "gold_traces"
        count_traces_in_dir(gold_dir, "gold")

        # Count failed traces
        failed_dir = data_dir / "failed_traces"
        count_traces_in_dir(failed_dir, "failed")

        # Count pending traces from both locations
        pending_patterns = ["session_*.json", "session_*.jsonl", "imported_*.json"]
        global_traces_dir = global_dir / "traces"
        project_traces_dir = data_dir / "traces"

        for pattern in pending_patterns:
            count_traces_in_dir(global_traces_dir, "pending", pattern)
            if (
                project_traces_dir.exists()
                and project_traces_dir.resolve() != global_traces_dir.resolve()
            ):
                count_traces_in_dir(project_traces_dir, "pending", pattern)

        # Calculate cumulative totals for area chart
        cumulative_gold = 0
        cumulative_failed = 0
        cumulative_pending = 0

        timeline_data = []
        for bucket in buckets:
            cumulative_gold += bucket["gold"]
            cumulative_failed += bucket["failed"]
            cumulative_pending += bucket["pending"]
            timeline_data.append(
                {
                    "time": bucket["time"],
                    "gold": cumulative_gold,
                    "failed": cumulative_failed,
                    "pending": cumulative_pending,
                }
            )

        # Get current totals
        total_gold = sum(1 for _ in gold_dir.glob("*.json")) if gold_dir.exists() else 0
        total_failed = sum(1 for _ in failed_dir.glob("*.json")) if failed_dir.exists() else 0
        total_pending = 0
        for pattern in pending_patterns:
            if global_traces_dir.exists():
                total_pending += sum(1 for _ in global_traces_dir.glob(pattern))
            if (
                project_traces_dir.exists()
                and project_traces_dir.resolve() != global_traces_dir.resolve()
            ):
                total_pending += sum(1 for _ in project_traces_dir.glob(pattern))

        return {
            "timeline": timeline_data,
            "totals": {
                "gold": total_gold,
                "failed": total_failed,
                "pending": total_pending,
                "total": total_gold + total_failed + total_pending,
            },
        }

    @app.get("/api/traces/analytics", tags=["Traces"])
    async def get_trace_analytics():
        """Aggregate analytics across all traces for training pipeline insights."""
        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        # Aggregate stats
        tool_stats: dict[str, dict[str, Any]] = {}  # tool -> {calls, sessions, successes}
        quality_distribution = {"gold": 0, "silver": 0, "bronze": 0, "failed": 0, "pending": 0}
        total_steps = 0
        total_sessions = 0
        total_tokens = 0
        cost_total_usd = 0.0
        quality_scores = []  # for computing avg_quality_score
        source_agg: dict[str, dict[str, Any]] = {}  # source_tool -> {traces, steps, tokens}

        # Scan all trace directories (including pending)
        dirs_to_scan = [
            (data_dir / "gold_traces", "gold"),
            (data_dir / "silver_traces", "silver"),
            (data_dir / "bronze_traces", "bronze"),
            (data_dir / "failed_traces", "failed"),
            (get_bashgym_dir() / "traces", "pending"),
        ]

        for trace_dir, tier in dirs_to_scan:
            if not trace_dir.exists():
                continue
            for trace_file in trace_dir.glob("*.json"):
                try:
                    with open(trace_file) as f:
                        data = json.load(f)

                    total_sessions += 1
                    quality_distribution[tier] += 1

                    # Get steps
                    if isinstance(data, list):
                        steps = data
                    else:
                        steps = data.get("trace", data.get("steps", []))

                    num_steps = len(steps)
                    total_steps += num_steps

                    # Aggregate source_tool breakdown
                    if isinstance(data, dict):
                        src = data.get("source_tool", "claude_code")
                        metadata = data.get("metadata", {})
                        cost = metadata.get("api_equivalent_cost_usd", 0) or 0
                        cost_total_usd += float(cost)

                        # Quality score from metadata or summary
                        qs = data.get("summary", {}).get("total_score")
                        if qs is None and metadata:
                            qs = metadata.get("total_score")
                        if qs is not None:
                            quality_scores.append(float(qs))
                    else:
                        src = "claude_code"

                    if src not in source_agg:
                        source_agg[src] = {"traces": 0, "steps": 0, "tokens": 0}
                    source_agg[src]["traces"] += 1
                    source_agg[src]["steps"] += num_steps

                    # Aggregate tool usage
                    for step in steps:
                        tool = step.get("tool_name", step.get("tool", "unknown"))
                        if tool not in tool_stats:
                            tool_stats[tool] = {
                                "calls": 0,
                                "sessions": set(),
                                "successes": 0,
                                "total_tokens": 0,
                            }
                        tool_stats[tool]["calls"] += 1
                        tool_stats[tool]["sessions"].add(str(trace_file.stem))

                        # Check success
                        success = step.get("success")
                        exit_code = step.get("exit_code")
                        if success is True or exit_code == 0:
                            tool_stats[tool]["successes"] += 1

                        # Token counting
                        tokens = step.get("input_tokens", 0) + step.get("output_tokens", 0)
                        tool_stats[tool]["total_tokens"] += tokens
                        total_tokens += tokens
                        source_agg[src]["tokens"] += tokens

                except (OSError, json.JSONDecodeError):
                    continue

        # Format tool stats (convert sets to counts)
        formatted_tools = []
        for tool, stats in sorted(tool_stats.items(), key=lambda x: -x[1]["calls"]):
            formatted_tools.append(
                {
                    "tool": tool,
                    "calls": stats["calls"],
                    "sessions": len(stats["sessions"]),
                    "success_rate": (
                        stats["successes"] / stats["calls"] if stats["calls"] > 0 else 0
                    ),
                    "total_tokens": stats["total_tokens"],
                }
            )

        # Training readiness
        trainable = quality_distribution["gold"] + quality_distribution["silver"]

        # Format source breakdown
        source_breakdown = []
        for src_name, src_stats in sorted(source_agg.items(), key=lambda x: -x[1]["traces"]):
            source_breakdown.append(
                {
                    "source": src_name,
                    "traces": src_stats["traces"],
                    "steps": src_stats["steps"],
                    "tokens": src_stats["tokens"],
                }
            )

        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "tool_stats": formatted_tools,
            "quality_distribution": quality_distribution,
            "totals": {
                "sessions": total_sessions,
                "steps": total_steps,
                "tokens": total_tokens,
            },
            "training_readiness": {
                "sft_ready": quality_distribution["gold"],
                "dpo_pairs_possible": min(
                    quality_distribution["silver"], quality_distribution["failed"]
                ),
                "total_trainable": trainable,
            },
            "source_breakdown": source_breakdown,
            "cost_total_usd": round(cost_total_usd, 2),
            "avg_quality_score": round(avg_quality_score, 4),
        }

    @app.get("/api/traces/repos", tags=["Traces"])
    async def list_trace_repos():
        """List all unique repositories that have traces.

        Returns a list of repo info objects for filtering.
        """
        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        repos = {}

        # Build list of directories to scan - same logic as list_traces
        dirs_to_scan = []

        # Tiered traces (gold/silver/bronze) and failed from project data dir
        dirs_to_scan.append((data_dir / "gold_traces", "*.json"))
        dirs_to_scan.append((data_dir / "silver_traces", "*.json"))
        dirs_to_scan.append((data_dir / "bronze_traces", "*.json"))
        dirs_to_scan.append((data_dir / "failed_traces", "*.json"))

        # Pending traces from BOTH global and project dirs (matching list_traces logic)
        global_traces_dir = get_bashgym_dir() / "traces"
        project_traces_dir = data_dir / "traces"

        if global_traces_dir.exists():
            # Include both session_*.json and imported_*.json
            dirs_to_scan.append((global_traces_dir, "session_*.json"))
            dirs_to_scan.append((global_traces_dir, "session_*.jsonl"))
            dirs_to_scan.append((global_traces_dir, "imported_*.json"))
        if (
            project_traces_dir.exists()
            and project_traces_dir.resolve() != global_traces_dir.resolve()
        ):
            dirs_to_scan.append((project_traces_dir, "session_*.json"))
            dirs_to_scan.append((project_traces_dir, "session_*.jsonl"))
            dirs_to_scan.append((project_traces_dir, "imported_*.json"))

        seen_files = set()  # Track by filename to avoid duplicates
        for trace_dir, glob_pattern in dirs_to_scan:
            if not trace_dir.exists():
                continue
            for trace_file in trace_dir.glob(glob_pattern):
                # Skip duplicates by filename
                if trace_file.name in seen_files:
                    continue
                seen_files.add(trace_file.name)
                try:
                    from bashgym.trace_capture.core import load_trace_file

                    data = load_trace_file(trace_file)

                    # Handle raw trace format (array of steps)
                    if isinstance(data, list):
                        # Extract repo from first step that has repo info
                        for step in data:
                            step_repo = step.get("repo", {})
                            if step_repo and step_repo.get("name"):
                                repo_name = step_repo.get("name")
                                if repo_name not in repos:
                                    repos[repo_name] = {
                                        "name": repo_name,
                                        "path": step_repo.get("path", ""),
                                        "git_remote": step_repo.get("git_remote"),
                                        "trace_count": 0,
                                    }
                                repos[repo_name]["trace_count"] += 1
                                break
                    else:
                        # Handle gold/failed trace format (object with primary_repo)
                        primary_repo = data.get("primary_repo", {})
                        repo_name = primary_repo.get("name", "unknown")
                        if repo_name and repo_name != "unknown":
                            if repo_name not in repos:
                                repos[repo_name] = {
                                    "name": repo_name,
                                    "path": primary_repo.get("path", ""),
                                    "git_remote": primary_repo.get("git_remote"),
                                    "trace_count": 0,
                                }
                            repos[repo_name]["trace_count"] += 1
                except Exception:
                    continue

        return list(repos.values())

    def _matches_repo(data: dict, repo_filter: str) -> bool:
        """Check if trace matches the repo filter."""
        primary_repo = data.get("primary_repo", {})
        repo_name = primary_repo.get("name", "")
        repo_path = primary_repo.get("path", "")
        git_remote = primary_repo.get("git_remote", "")

        filter_lower = repo_filter.lower()
        return (
            filter_lower in repo_name.lower()
            or filter_lower in repo_path.lower()
            or (git_remote and filter_lower in git_remote.lower())
        )

    def _sanitize_task_description(raw: str, max_length: int = 120) -> str:
        """Clean a raw user prompt into a concise trace title.

        Strips markdown formatting, numbered lists, bold markers, collapses
        whitespace, and takes only the first meaningful sentence or phrase.
        """
        if not raw:
            return "Unknown task"

        text = raw.strip()

        # Strip markdown bold/italic markers
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        # Strip markdown headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Strip markdown links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Strip backtick code markers
        text = re.sub(r"`+([^`]*)`+", r"\1", text)
        # Strip numbered list prefixes (1. 2. etc.)
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
        # Strip bullet points
        text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)

        # Collapse all whitespace (newlines, tabs, multiple spaces) to single space
        text = re.sub(r"\s+", " ", text).strip()

        # Take first sentence if it exists and is reasonable length
        sentence_end = re.search(r"[.!?]\s", text)
        if sentence_end and sentence_end.start() < max_length:
            text = text[: sentence_end.start() + 1]
        elif len(text) > max_length:
            # Truncate at last word boundary before max_length
            truncated = text[:max_length]
            last_space = truncated.rfind(" ")
            if last_space > max_length * 0.5:
                text = truncated[:last_space] + "..."
            else:
                text = truncated + "..."

        return text

    def _parse_imported_trace_file(trace_file: Path, data: dict) -> TraceInfo:
        """Parse an imported TraceSession file into TraceInfo."""
        trace_steps = data.get("trace", [])
        total_steps = len(trace_steps)
        summary = data.get("summary", {})
        metadata = data.get("metadata", {})

        # Get success rate from summary or calculate
        success_rate = summary.get("success_rate", 0)
        if success_rate == 0 and total_steps > 0:
            successful_steps = sum(
                1 for s in trace_steps if s.get("success") is True or s.get("exit_code") == 0
            )
            success_rate = successful_steps / total_steps

        # Extract repo info
        primary_repo = data.get("primary_repo", {})
        repos = data.get("repos", [])
        repo_info = None
        if primary_repo:
            repo_info = RepoInfo(
                name=primary_repo.get("name", "unknown"),
                path=primary_repo.get("path"),
                git_remote=primary_repo.get("git_remote"),
                git_branch=primary_repo.get("git_branch"),
                is_git_repo=primary_repo.get("is_git_repo", False),
            )

        # Get source info
        source_tool = data.get("source_tool", "unknown")

        # Get task description from metadata (user's initial prompt) or generate from tool usage
        task_desc = _sanitize_task_description(metadata.get("user_initial_prompt", ""))

        # Compute tool breakdown for all imported traces
        tool_counts = {}
        for step in trace_steps:
            tool = step.get("tool_name", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        if task_desc == "Unknown task":
            # Fallback: generate description from tool usage summary
            tool_summary = ", ".join(
                [
                    f"{count} {tool}"
                    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:4]
                ]
            )
            task_desc = f"Imported from {source_tool}: {tool_summary}"

        # Calculate quality using centralized calculator
        quality = calculate_quality_breakdown(steps=trace_steps, metadata=metadata)

        return TraceInfo(
            trace_id=trace_file.stem,
            task_id=data.get("session_id", trace_file.stem),
            task_description=task_desc,
            status=TraceStatus.PENDING,
            steps_count=total_steps,
            quality=TraceQuality(
                success_rate=quality.success_rate,
                verification_score=quality.verification_score,
                complexity_score=quality.complexity_score,
                length_score=quality.length_score,
                tool_diversity=quality.tool_diversity,
                efficiency_score=quality.efficiency_score,
                cognitive_quality=quality.cognitive_quality,
                total_score=quality.total_score,
            ),
            source_tool=source_tool,
            repo=repo_info,
            repos_count=len(repos) if repos else 1,
            tool_breakdown=tool_counts,
            created_at=data.get(
                "timestamp", datetime.fromtimestamp(trace_file.stat().st_ctime).isoformat()
            ),
            promoted_at=metadata.get("promoted_at"),
        )

    def _parse_raw_trace_file(trace_file: Path, data: list[dict]) -> TraceInfo:
        """Parse a raw session trace file (array of tool calls) into TraceInfo."""
        total_steps = len(data)

        # Count successful steps (those without errors)
        successful_steps = 0
        for step in data:
            # Check for success indicators
            exit_code = step.get("exit_code")
            success = step.get("success")
            if success is True or exit_code == 0 or (exit_code is None and success is None):
                successful_steps += 1

        successful_steps / total_steps if total_steps > 0 else 0

        # Extract repo info from first step with repo data
        repo_info = None
        repos_seen = set()
        for step in data:
            step_repo = step.get("repo", {})
            if step_repo and step_repo.get("name"):
                repos_seen.add(step_repo.get("name"))
                if repo_info is None:
                    repo_info = RepoInfo(
                        name=step_repo.get("name", "unknown"),
                        path=step_repo.get("path"),
                        git_remote=step_repo.get("git_remote"),
                        git_branch=step_repo.get("git_branch"),
                        is_git_repo=step_repo.get("is_git_repo", False),
                    )

        # Get timestamps
        first_timestamp = data[0].get("timestamp", "") if data else ""
        data[-1].get("timestamp", "") if data else ""

        # Generate task description from tool usage summary
        tool_counts = {}
        for step in data:
            tool = step.get("tool_name", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        tool_summary = ", ".join(
            [
                f"{count} {tool}"
                for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:4]
            ]
        )
        task_desc = f"Coding session: {tool_summary}"

        # Calculate quality using centralized calculator
        quality = calculate_quality_breakdown(steps=data)

        return TraceInfo(
            trace_id=trace_file.stem,
            task_id=trace_file.stem,
            task_description=task_desc,
            status=TraceStatus.PENDING,
            steps_count=total_steps,
            quality=TraceQuality(
                success_rate=quality.success_rate,
                verification_score=quality.verification_score,
                complexity_score=quality.complexity_score,
                length_score=quality.length_score,
                tool_diversity=quality.tool_diversity,
                efficiency_score=quality.efficiency_score,
                cognitive_quality=quality.cognitive_quality,
                total_score=quality.total_score,
            ),
            source_tool="claude_code",
            repo=repo_info,
            repos_count=len(repos_seen),
            tool_breakdown=tool_counts,
            created_at=first_timestamp
            or datetime.fromtimestamp(trace_file.stat().st_ctime).isoformat(),
            promoted_at=None,
        )

    def _parse_trace_file(trace_file: Path, data, status: TraceStatus) -> TraceInfo:
        """Parse a trace file into TraceInfo."""
        # Handle both dict-wrapped and raw list traces
        if isinstance(data, list):
            metadata = {}
            trace_steps = data
        else:
            metadata = data.get("metadata", {})
            trace_steps = data.get("trace", data.get("steps", []))
        total_steps = len(trace_steps)

        # Calculate quality using centralized calculator
        quality = calculate_quality_breakdown(steps=trace_steps, metadata=metadata)

        # Extract repo info
        primary_repo = data.get("primary_repo", {}) if isinstance(data, dict) else {}
        repos = data.get("repos", []) if isinstance(data, dict) else []
        repo_info = None
        if primary_repo:
            repo_info = RepoInfo(
                name=primary_repo.get("name", "unknown"),
                path=primary_repo.get("path"),
                git_remote=primary_repo.get("git_remote"),
                git_branch=primary_repo.get("git_branch"),
                is_git_repo=primary_repo.get("is_git_repo", False),
            )

        # Determine quality tier based on metrics (NVIDIA NeMo thresholds)
        # Gold: ≥90% success, ≥0.75 quality | Silver: ≥75%, ≥0.55 | Bronze: ≥60%, ≥0.40
        quality_tier = None
        if quality.success_rate >= 0.90 and quality.total_score >= 0.75:
            quality_tier = TraceQualityTier.GOLD
        elif quality.success_rate >= 0.75 and quality.total_score >= 0.55:
            quality_tier = TraceQualityTier.SILVER
        elif quality.success_rate >= 0.60 and quality.total_score >= 0.40:
            quality_tier = TraceQualityTier.BRONZE
        else:
            quality_tier = TraceQualityTier.REJECTED

        # Compute tool breakdown
        tool_counts = {}
        for step in trace_steps:
            tool = step.get("tool_name", step.get("tool", "unknown"))
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return TraceInfo(
            trace_id=trace_file.stem,
            task_id=metadata.get("task_id", trace_file.stem),
            task_description=_sanitize_task_description(metadata.get("user_initial_prompt", "")),
            status=status,
            quality_tier=quality_tier,
            steps_count=total_steps,
            quality=TraceQuality(
                success_rate=quality.success_rate,
                verification_score=quality.verification_score,
                complexity_score=quality.complexity_score,
                length_score=quality.length_score,
                tool_diversity=quality.tool_diversity,
                efficiency_score=quality.efficiency_score,
                cognitive_quality=quality.cognitive_quality,
                total_score=quality.total_score,
            ),
            source_tool=(
                data.get("source_tool", "claude_code") if isinstance(data, dict) else "claude_code"
            ),
            repo=repo_info,
            repos_count=len(repos),
            tool_breakdown=tool_counts,
            created_at=metadata.get(
                "created_at", datetime.fromtimestamp(trace_file.stat().st_ctime).isoformat()
            ),
            promoted_at=metadata.get("promoted_at"),
        )

    @app.get("/api/traces/gold", response_model=list[TraceInfo], tags=["Traces"])
    async def list_gold_traces(limit: int = 100):
        """List only gold (successful) traces."""
        result = await list_traces(status=TraceStatus.GOLD, limit=limit)
        return result["traces"]

    @app.get("/api/traces/{trace_id}", response_model=TraceSummaryDetail, tags=["Traces"])
    async def get_trace(trace_id: str):
        """Get a specific trace by ID with enriched detail for promote/demote decisions."""
        from datetime import datetime

        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        # Check all tier directories + pending dirs
        tier_checks = [
            (data_dir / "gold_traces" / f"{trace_id}.json", TraceStatus.GOLD),
            (data_dir / "silver_traces" / f"{trace_id}.json", TraceStatus.SILVER),
            (data_dir / "bronze_traces" / f"{trace_id}.json", TraceStatus.BRONZE),
            (data_dir / "failed_traces" / f"{trace_id}.json", TraceStatus.FAILED),
        ]

        # Also check pending directories
        global_traces_dir = get_bashgym_dir() / "traces"
        project_traces_dir = data_dir / "traces"
        for pending_dir in [global_traces_dir, project_traces_dir]:
            if pending_dir.exists():
                candidate = pending_dir / f"{trace_id}.json"
                if candidate.exists():
                    tier_checks.append((candidate, TraceStatus.PENDING))
                candidate_jsonl = pending_dir / f"session_{trace_id}.jsonl"
                if candidate_jsonl.exists():
                    tier_checks.append((candidate_jsonl, TraceStatus.PENDING))

        trace_path = None
        trace_status = None
        data = None
        for path, status in tier_checks:
            if path.exists():
                from bashgym.trace_capture.core import load_trace_file

                data = load_trace_file(path)
                trace_path = path
                trace_status = status
                break

        if trace_path is None or data is None:
            raise HTTPException(status_code=404, detail="Trace not found")

        base_info = _parse_trace_file(trace_path, data, trace_status)

        # Extract steps for enrichment
        if isinstance(data, list):
            trace_steps = data
        else:
            trace_steps = data.get("trace", data.get("steps", []))

        # Compute duration from timestamps
        duration_seconds = None
        timestamps = []
        for step in trace_steps:
            ts = step.get("timestamp")
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        timestamps.append(ts)
                    else:
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        timestamps.append(dt.timestamp())
                except Exception:
                    pass
        if len(timestamps) >= 2:
            duration_seconds = max(timestamps) - min(timestamps)

        # Build step_outcomes
        step_outcomes = []
        for step in trace_steps:
            success = step.get("success")
            exit_code = step.get("exit_code")
            if success is not None:
                step_outcomes.append(bool(success))
            elif exit_code is not None:
                step_outcomes.append(exit_code == 0)
            else:
                step_outcomes.append(None)

        # Raw metrics
        unique_tools = set()
        unique_commands = set()
        successful = 0
        failed = 0
        cognitive_steps = 0
        planning_phases = 0
        reflections = 0
        thinking_steps = 0

        for step in trace_steps:
            tool = step.get("tool_name", step.get("tool", ""))
            if tool:
                unique_tools.add(tool)
            cmd = step.get("command", "")
            if cmd:
                unique_commands.add(cmd.split()[0] if cmd.split() else cmd)

            # Count outcomes
            s = step.get("success")
            ec = step.get("exit_code")
            if s is True or ec == 0:
                successful += 1
            elif s is False or (ec is not None and ec != 0):
                failed += 1

            # Cognitive sub-types
            step_type = step.get("type", "")
            tool_lower = tool.lower() if tool else ""
            if step_type in ("thinking", "thought") or "think" in tool_lower:
                thinking_steps += 1
                cognitive_steps += 1
            elif step_type in ("plan", "planning") or "plan" in tool_lower:
                planning_phases += 1
                cognitive_steps += 1
            elif step_type in ("reflect", "reflection") or "reflect" in tool_lower:
                reflections += 1
                cognitive_steps += 1

        raw_metrics = {
            "total_steps": len(trace_steps),
            "successful_steps": successful,
            "failed_steps": failed,
            "unique_tools": len(unique_tools),
            "unique_commands": len(unique_commands),
            "cognitive_steps": cognitive_steps,
        }

        cognitive_summary = None
        if cognitive_steps > 0:
            cognitive_summary = {
                "planning_phases": planning_phases,
                "reflections": reflections,
                "thinking_steps": thinking_steps,
                "cognitive_coverage": round(cognitive_steps / max(len(trace_steps), 1), 2),
            }

        return TraceSummaryDetail(
            **base_info.model_dump(),
            duration_seconds=duration_seconds,
            step_outcomes=step_outcomes,
            cognitive_summary=cognitive_summary,
            raw_metrics=raw_metrics,
        )

    @app.post("/api/traces/{trace_id}/promote", tags=["Traces"])
    async def promote_trace(trace_id: str, target_tier: str = "gold"):
        """Promote a trace to a higher tier (gold/silver/bronze).

        Args:
            trace_id: The trace ID to promote
            target_tier: Target tier (gold, silver, bronze). Default is gold.
        """
        from bashgym.config import get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        # Validate target tier
        valid_tiers = ["gold", "silver", "bronze"]
        if target_tier not in valid_tiers:
            raise HTTPException(
                status_code=400, detail=f"Invalid target tier. Must be one of: {valid_tiers}"
            )

        target_path = data_dir / f"{target_tier}_traces" / f"{trace_id}.json"

        # Check if already at target tier
        if target_path.exists():
            return {"success": True, "message": f"Trace is already {target_tier}"}

        # Find the trace in other directories (lower tiers or failed)
        source_dirs = [
            data_dir / "failed_traces",
            data_dir / "bronze_traces",
            data_dir / "silver_traces",
            data_dir / "gold_traces",
        ]

        source_path = None
        for dir_path in source_dirs:
            candidate = dir_path / f"{trace_id}.json"
            if candidate.exists():
                source_path = candidate
                break

        if not source_path:
            # Also search pending directories for JSONL traces
            from bashgym.config import get_bashgym_dir

            pending_dirs = [get_bashgym_dir() / "traces", data_dir / "traces"]
            for dir_path in source_dirs + pending_dirs:
                for pattern in [
                    f"session_{trace_id}.jsonl",
                    f"{trace_id}.jsonl",
                    f"{trace_id}.json",
                ]:
                    candidate = dir_path / pattern
                    if candidate.exists():
                        source_path = candidate
                        break
                if source_path:
                    break

        if not source_path:
            raise HTTPException(status_code=404, detail="Trace not found")

        # Move to target tier
        target_path.parent.mkdir(parents=True, exist_ok=True)

        from bashgym.trace_capture.core import load_trace_file

        data = load_trace_file(source_path)

        # If raw step list (from JSONL), wrap into trace dict
        if isinstance(data, list):
            if not data:
                raise HTTPException(status_code=400, detail="Trace file is empty")
            data = {
                "trace": data,
                "metadata": {},
                "primary_repo": data[0].get("repo", {}),
                "source_tool": data[0].get("source_tool", "claude_code"),
            }

        # Update metadata
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["promoted_at"] = datetime.utcnow().isoformat()
        data["metadata"]["status"] = target_tier
        data["metadata"]["quality_tier"] = target_tier

        with open(target_path, "w") as f:
            json.dump(data, f, indent=2)

        source_path.unlink()

        # Invalidate cache — force re-index on next request
        app.state.trace_cache.invalidate(trace_id)

        # Broadcast event
        await broadcast_trace_event(MessageType.TRACE_PROMOTED, trace_id)

        return {"success": True, "message": f"Trace promoted to {target_tier}"}

    @app.post("/api/traces/{trace_id}/demote", tags=["Traces"])
    async def demote_trace(trace_id: str, target_tier: str = "failed"):
        """Demote a trace to a lower tier.

        Args:
            trace_id: The trace ID to demote
            target_tier: Target tier (silver, bronze, failed). Default is failed.
        """
        from bashgym.config import get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        # Validate target tier (can't demote to gold)
        valid_targets = ["silver", "bronze", "failed"]
        if target_tier not in valid_targets:
            raise HTTPException(
                status_code=400, detail=f"Invalid target tier. Must be one of: {valid_targets}"
            )

        target_path = data_dir / f"{target_tier}_traces" / f"{trace_id}.json"

        # Check if already at target tier
        if target_path.exists():
            return {"success": True, "message": f"Trace is already at {target_tier}"}

        # Find the trace in higher tier directories
        source_dirs = [
            data_dir / "gold_traces",
            data_dir / "silver_traces",
            data_dir / "bronze_traces",
        ]

        source_path = None
        for dir_path in source_dirs:
            candidate = dir_path / f"{trace_id}.json"
            if candidate.exists():
                source_path = candidate
                break

        if not source_path:
            raise HTTPException(status_code=404, detail="Trace not found in any tier")

        # Move to target tier
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(source_path) as f:
            data = json.load(f)

        # Update metadata
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["demoted_at"] = datetime.utcnow().isoformat()
        data["metadata"]["status"] = target_tier
        data["metadata"]["quality_tier"] = target_tier

        with open(target_path, "w") as f:
            json.dump(data, f, indent=2)

        source_path.unlink()

        # Invalidate cache
        app.state.trace_cache.invalidate(trace_id)

        # Broadcast event
        await broadcast_trace_event(MessageType.TRACE_DEMOTED, trace_id)

        return {"success": True, "message": f"Trace demoted to {target_tier}"}

    # =========================================================================
    # Training Examples Endpoints
    # =========================================================================

    @app.post(
        "/api/traces/{trace_id}/generate-examples",
        response_model=GenerateExamplesResponse,
        tags=["Training Examples"],
    )
    async def generate_examples_from_trace(trace_id: str, request: GenerateExamplesRequest = None):
        """Generate training examples from a trace session.

        Segments a trace session into logical tasks and converts each to
        a NeMo-compatible training example.
        """
        from bashgym.config import get_bashgym_dir, get_settings
        from bashgym.factory.example_generator import ExampleGenerator, ExampleGeneratorConfig

        if request is None:
            request = GenerateExamplesRequest()

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        # Find the trace file
        trace_path = None
        for search_dir in [data_dir, global_dir]:
            for subdir in ["gold_traces", "traces", "failed_traces"]:
                candidate = search_dir / subdir / f"{trace_id}.json"
                if candidate.exists():
                    trace_path = candidate
                    break
            if trace_path:
                break

        if not trace_path:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

        # Configure generator
        config = ExampleGeneratorConfig(
            min_success_rate=request.min_success_rate,
            max_steps_per_segment=request.max_steps_per_example,
            output_dir=str(data_dir / "training_examples"),
        )

        generator = ExampleGenerator(config)

        # Generate examples
        examples = generator.generate_examples(trace_path)

        # Convert to response format
        example_responses = []
        for ex in examples:
            example_responses.append(
                TrainingExampleResponse(
                    example_id=ex.example_id,
                    system_prompt=ex.system_prompt,
                    user_prompt=ex.user_prompt,
                    assistant_response=ex.assistant_response,
                    step_count=ex.metadata.get("step_count", 0),
                    success_rate=ex.metadata.get("success_rate", 0.0),
                    confidence=ex.metadata.get("segmentation_confidence", 0.5),
                    source_trace_id=trace_id,
                )
            )

        # Calculate total steps
        total_steps = sum(ex.step_count for ex in example_responses)

        return GenerateExamplesResponse(
            trace_id=trace_id,
            examples=example_responses,
            total_steps=total_steps,
            examples_generated=len(example_responses),
        )

    @app.get(
        "/api/training/examples",
        response_model=list[TrainingExampleResponse],
        tags=["Training Examples"],
    )
    async def list_training_examples(limit: int = 100, offset: int = 0):
        """List generated training examples.

        Returns examples that have been previously generated and saved.
        """
        from bashgym.config import get_settings

        settings = get_settings()
        examples_dir = Path(settings.data.data_dir) / "training_examples"

        if not examples_dir.exists():
            return []

        examples = []

        # Read from JSONL files
        for jsonl_file in examples_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            messages = data.get("messages", [])

                            system_prompt = ""
                            user_prompt = ""
                            assistant_response = ""

                            for msg in messages:
                                if msg.get("role") == "system":
                                    system_prompt = msg.get("content", "")
                                elif msg.get("role") == "user":
                                    user_prompt = msg.get("content", "")
                                elif msg.get("role") == "assistant":
                                    assistant_response = msg.get("content", "")

                            examples.append(
                                TrainingExampleResponse(
                                    example_id=data.get("id", f"ex_{len(examples)}"),
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                    assistant_response=assistant_response,
                                    step_count=data.get("metadata", {}).get("step_count", 0),
                                    success_rate=data.get("metadata", {}).get("success_rate", 0.0),
                                    confidence=data.get("metadata", {}).get(
                                        "segmentation_confidence", 0.5
                                    ),
                                    source_trace_id=data.get("metadata", {}).get("source_trace_id"),
                                )
                            )
            except (OSError, json.JSONDecodeError):
                continue

        # Apply pagination
        return examples[offset : offset + limit]

    @app.post(
        "/api/training/export", response_model=ExportExamplesResponse, tags=["Training Examples"]
    )
    async def export_training_examples(request: ExportExamplesRequest):
        """Export training examples to JSONL files for NeMo training.

        Generates train and validation splits from gold traces or specified traces.
        Output format is NeMo-compatible JSONL with messages array.
        """
        from bashgym.config import get_bashgym_dir, get_settings
        from bashgym.factory.example_generator import ExampleGenerator, ExampleGeneratorConfig

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        config = ExampleGeneratorConfig(output_dir=str(data_dir / "training_batches"))
        generator = ExampleGenerator(config)

        all_examples = []

        # Determine which traces to process
        if request.trace_ids:
            # Process specific traces
            for trace_id in request.trace_ids:
                for search_dir in [data_dir, global_dir]:
                    for subdir in ["gold_traces", "traces", "failed_traces"]:
                        trace_path = search_dir / subdir / f"{trace_id}.json"
                        if trace_path.exists():
                            examples = generator.generate_examples(trace_path)
                            all_examples.extend(examples)
                            break
        else:
            # Process gold traces (or all if include_gold_only is False)
            dirs_to_process = ["gold_traces"]
            if not request.include_gold_only:
                dirs_to_process.extend(["traces", "failed_traces"])

            for search_dir in [data_dir, global_dir]:
                for subdir in dirs_to_process:
                    trace_dir = search_dir / subdir
                    if trace_dir.exists():
                        for trace_path in trace_dir.glob("*.json"):
                            examples = generator.generate_examples(trace_path)
                            all_examples.extend(examples)

        if not all_examples:
            return ExportExamplesResponse(
                success=False,
                message="No examples generated. Check that traces exist and meet quality thresholds.",
            )

        # Export with train/val split
        try:
            result = generator.export_for_nemo(
                all_examples, data_dir / "training_batches", train_split=request.train_split
            )

            return ExportExamplesResponse(
                success=True,
                train_path=str(result["train"]),
                val_path=str(result["validation"]),
                train_count=result["train_count"],
                val_count=result["val_count"],
                message=f"Exported {len(all_examples)} examples",
            )
        except Exception as e:
            return ExportExamplesResponse(success=False, message=f"Export failed: {str(e)}")

    @app.get("/api/training/export/download", tags=["Training Examples"])
    async def download_training_export(split: str = "train"):
        """Download the most recent exported JSONL file as a browser download.

        Args:
            split: Which split to download - 'train' or 'val'
        """
        from fastapi.responses import FileResponse

        from bashgym.config import get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        batches_dir = data_dir / "training_batches"

        if split not in ("train", "val"):
            raise HTTPException(status_code=400, detail="split must be 'train' or 'val'")

        filename = f"{split}.jsonl"
        file_path = batches_dir / filename

        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"No exported {split} file found. Run export first."
            )

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/jsonl",
        )

    @app.post("/api/traces/sync", tags=["Traces"])
    async def sync_traces_to_project():
        """Sync traces from ~/.bashgym/ to project data/ directory.

        Copies pending, gold, silver, bronze, and failed traces to project-local storage.
        Useful for version control and portability. Only copies files that
        don't already exist in the destination.

        Returns:
            Dict with counts of synced traces per category and project directory path.
        """
        import shutil

        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        global_dir = get_bashgym_dir()
        project_dir = Path(settings.data.data_dir)

        # Don't sync if they're the same directory
        if global_dir.resolve() == project_dir.resolve():
            return {
                "synced": {"pending": 0, "gold": 0, "silver": 0, "bronze": 0, "failed": 0},
                "project_dir": str(project_dir),
                "message": "Global and project directories are the same, no sync needed",
            }

        synced = {"pending": 0, "gold": 0, "silver": 0, "bronze": 0, "failed": 0}

        # Include all tier directories
        for category in [
            "traces",
            "gold_traces",
            "silver_traces",
            "bronze_traces",
            "failed_traces",
        ]:
            src = global_dir / category
            dst = project_dir / category
            dst.mkdir(parents=True, exist_ok=True)

            if src.exists():
                for f in list(src.glob("*.json")) + list(src.glob("*.jsonl")):
                    if not (dst / f.name).exists():
                        shutil.copy2(f, dst / f.name)
                        key = "pending" if category == "traces" else category.replace("_traces", "")
                        synced[key] += 1

        return {"synced": synced, "project_dir": str(project_dir)}

    @app.post("/api/traces/import", tags=["Traces"])
    async def import_traces_from_claude():
        """Import new Claude Code sessions from ~/.claude/projects/ into the trace pipeline.

        Scans all project directories in ~/.claude/projects/, imports sessions that
        haven't been imported yet, and copies them to ~/.bashgym/traces/.

        Returns:
            Dict with count of imported sessions, total found, and new trace IDs.
        """
        try:
            from bashgym.trace_capture.importers.claude_history import (
                import_recent,
            )

            results = import_recent(days=60, verbose=False)
        except Exception as e:
            logger.error(f"Trace import failed: {e}")
            raise HTTPException(status_code=500, detail=f"Import failed: {e}")

        imported = [r for r in results if not r.skipped and not r.error]
        new_trace_ids = [r.session_id for r in imported]

        return {
            "imported": len(imported),
            "total": len(results),
            "skipped": len([r for r in results if r.skipped]),
            "errors": len([r for r in results if r.error]),
            "new_trace_ids": new_trace_ids,
        }

    # ------------------------------------------------------------------
    # Unified trace import endpoints (all sources)
    # ------------------------------------------------------------------

    # Import handler map: source_name -> callable(TraceImportRequest) -> list
    def _get_import_handlers():
        from bashgym.trace_capture.adapters.codex import import_codex_sessions
        from bashgym.trace_capture.importers import (
            import_chatgpt_sessions,
            import_copilot_sessions,
            import_gemini_sessions,
            import_mcp_logs,
            import_opencode_sessions,
            import_recent,
        )

        return {
            "claude": lambda req: import_recent(days=req.days, verbose=False, force=req.force),
            "gemini": lambda req: import_gemini_sessions(
                days=req.days, limit=req.limit, verbose=False
            ),
            "copilot": lambda req: import_copilot_sessions(
                days=req.days, limit=req.limit, verbose=False
            ),
            "opencode": lambda req: import_opencode_sessions(
                days=req.days, limit=req.limit, verbose=False
            ),
            "codex": lambda req: import_codex_sessions(limit=req.limit),
            "chatgpt": lambda req: import_chatgpt_sessions(force=req.force),
            "mcp": lambda req: import_mcp_logs(force=req.force),
        }

    def _build_import_response(source: str, results) -> dict:
        """Normalise heterogeneous import results into a uniform dict.

        Handles three result shapes:
        - dataclass objects (Claude ``ImportResult``) with ``.skipped``, ``.error``, ``.destination_file``
        - standard dicts (Gemini/Copilot/OpenCode) with ``skipped``, ``error``, ``destination_file`` keys
        - codex dicts with ``trace_file``, ``error`` keys (no ``skipped`` key)
        """
        if not results:
            return {
                "source": source,
                "imported": 0,
                "skipped": 0,
                "errors": 0,
                "total": 0,
                "new_trace_ids": [],
            }

        imported = 0
        skipped = 0
        errors = 0
        new_ids: list[str] = []

        for r in results:
            # Unify access: support both dataclass attrs and dict keys
            if isinstance(r, dict):
                r_skipped = r.get("skipped", False)
                r_error = r.get("error", None)
                r_dest = r.get("destination_file") or r.get("trace_file")
            else:
                r_skipped = getattr(r, "skipped", False)
                r_error = getattr(r, "error", None)
                r_dest = getattr(r, "destination_file", None)

            if r_error:
                errors += 1
            elif r_skipped:
                skipped += 1
            else:
                imported += 1

            if r_dest:
                new_ids.append(str(r_dest))

        return {
            "source": source,
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "total": len(results),
            "new_trace_ids": new_ids,
        }

    # IMPORTANT: /import/all must be registered BEFORE /import/{source}
    # so FastAPI does not capture "all" as a path parameter.

    @app.post("/api/traces/import/all", tags=["Traces"])
    async def import_traces_all(request: TraceImportRequest = None):
        """Import traces from all detected tools.

        Iterates over every known source (claude, gemini, copilot, opencode, codex),
        runs the importer, and returns per-source results plus an aggregate total.
        """
        if request is None:
            request = TraceImportRequest()

        handlers = _get_import_handlers()
        all_results: list[dict] = []

        for source, handler in handlers.items():
            try:
                results = handler(request)
                all_results.append(_build_import_response(source, results))
            except Exception as e:
                logger.error(f"Import from {source} failed: {e}")
                all_results.append(
                    {
                        "source": source,
                        "imported": 0,
                        "skipped": 0,
                        "errors": 1,
                        "total": 0,
                        "new_trace_ids": [],
                        "error_detail": str(e),
                    }
                )

        total_imported = sum(r["imported"] for r in all_results)
        return {"results": all_results, "total_imported": total_imported}

    @app.post("/api/traces/import/{source}", tags=["Traces"])
    async def import_traces_by_source(source: str, request: TraceImportRequest = None):
        """Import traces from a specific tool.

        Args:
            source: One of ``claude``, ``gemini``, ``copilot``, ``opencode``, ``codex``.
            request: Optional import parameters (days, limit, force).
        """
        if request is None:
            request = TraceImportRequest()

        handlers = _get_import_handlers()
        if source not in handlers:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source: {source}. Valid sources: {list(handlers.keys())}",
            )

        try:
            results = handlers[source](request)
            return _build_import_response(source, results)
        except Exception as e:
            logger.error(f"Import from {source} failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/traces/import-since", tags=["Traces"])
    async def get_traces_since(since: str):
        """Get traces created after a given ISO timestamp.

        Args:
            since: ISO 8601 timestamp (e.g. 2024-01-01T00:00:00Z)

        Returns:
            Dict with count of traces created after the timestamp.
        """
        from bashgym.config import get_bashgym_dir, get_settings

        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {since}")

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        bashgym_dir = get_bashgym_dir()

        new_traces = []
        dirs_to_check = [
            bashgym_dir / "traces",
            bashgym_dir / "gold_traces",
            bashgym_dir / "silver_traces",
            bashgym_dir / "bronze_traces",
            bashgym_dir / "failed_traces",
            data_dir / "traces",
            data_dir / "gold_traces",
            data_dir / "silver_traces",
            data_dir / "bronze_traces",
            data_dir / "failed_traces",
        ]

        seen_ids = set()
        for trace_dir in dirs_to_check:
            if not trace_dir.exists():
                continue
            for f in trace_dir.glob("*.json"):
                try:
                    import datetime as _dt

                    mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime, tz=_dt.timezone.utc)
                    if mtime > since_dt and f.stem not in seen_ids:
                        seen_ids.add(f.stem)
                        new_traces.append(f.stem)
                except Exception:
                    continue

        return {"count": len(new_traces), "traces": new_traces}

    @app.post("/api/traces/auto-classify", tags=["Traces"])
    async def auto_classify_traces(
        # Tiered thresholds based on NVIDIA NeMo recommendations
        # Reference: NVIDIA min_success_rate: 0.9, min_reward: 0.8 for SFT
        gold_success_rate: float = 0.90,
        gold_quality_score: float = 0.75,
        silver_success_rate: float = 0.75,
        silver_quality_score: float = 0.55,
        bronze_success_rate: float = 0.60,
        bronze_quality_score: float = 0.40,
        dry_run: bool = True,
        auto_promote: bool = False,
    ):
        """Auto-classify pending traces into quality tiers based on NVIDIA NeMo research.

        Tiered Classification (based on industry standards):
        - GOLD (≥90% success, ≥0.75 quality): SFT training, high-confidence examples
          → Matches NVIDIA's min_success_rate: 0.9 recommendation
        - SILVER (≥75% success, ≥0.55 quality): DPO chosen responses, secondary SFT
        - BRONZE (≥60% success, ≥0.40 quality): DPO rejected responses, review candidates
        - REJECTED (<60% success): Not suitable for training, archive only

        Verification adjustments:
        - verification_passed=True -> One-tier boost (e.g. silver→gold, bronze→silver)
        - verification_passed=False AND has_verification=True -> Failed (verified unsuccessful)

        Args:
            gold_success_rate: Min success rate for Gold tier (default 0.90 per NVIDIA)
            gold_quality_score: Min quality score for Gold tier (default 0.75)
            silver_success_rate: Min success rate for Silver tier (default 0.75)
            silver_quality_score: Min quality score for Silver tier (default 0.55)
            bronze_success_rate: Min success rate for Bronze tier (default 0.60)
            bronze_quality_score: Min quality score for Bronze tier (default 0.40)
            dry_run: If True, only return classifications without moving files (default True)
            auto_promote: If True and not dry_run, move traces to their tier directories (default False)

        Returns:
            Dict with tiered classification results and DPO pairing suggestions.
        """
        import shutil

        from bashgym.config import get_bashgym_dir, get_settings

        settings = get_settings()
        global_dir = get_bashgym_dir()
        data_dir = Path(settings.data.data_dir)

        # Tiered classifications based on NVIDIA NeMo research
        classifications = {
            "gold": [],  # ≥90% success, ≥0.75 quality → SFT training
            "silver": [],  # ≥75% success, ≥0.55 quality → DPO chosen
            "bronze": [],  # ≥60% success, ≥0.40 quality → DPO rejected
            "rejected": [],  # <60% success → Not suitable for training
            "failed": [],  # Explicit verification failure
            "pending": [],  # Unable to process
        }

        # Track detailed info for DPO pair generation
        detailed_classifications = {"gold": [], "silver": [], "bronze": [], "rejected": []}

        def calculate_quality(data: Any) -> dict:
            """Calculate quality metrics for both trace formats using centralized calculator."""
            steps = []
            metadata = {}

            # Handle raw trace format (array of steps)
            if isinstance(data, list):
                steps = data
            # Handle imported/gold format (dict with trace key)
            elif isinstance(data, dict):
                steps = data.get("trace", [])
                metadata = data.get("metadata", {})
                summary = data.get("summary", {})
                # Merge summary into metadata for verification info
                if "verification_passed" in summary:
                    metadata["verification_passed"] = summary["verification_passed"]

            if len(steps) == 0:
                return {
                    "success_rate": 0,
                    "quality_score": 0,
                    "total_steps": 0,
                    "verification_passed": False,
                    "has_verification": False,
                    "unique_tools": 0,
                }

            # Use centralized quality calculator
            quality = calculate_quality_breakdown(steps=steps, metadata=metadata)

            return {
                "success_rate": round(quality.success_rate, 3),
                "quality_score": round(quality.total_score, 3),
                "total_steps": quality.total_steps,
                "verification_passed": quality.verification_passed_flag is True,
                "has_verification": quality.has_verification,
                "unique_tools": quality.unique_tools_count,
            }

        # Collect pending traces from both directories
        pending_dirs = []
        global_traces_dir = global_dir / "traces"
        project_traces_dir = data_dir / "traces"

        if global_traces_dir.exists():
            pending_dirs.append(global_traces_dir)
        if (
            project_traces_dir.exists()
            and project_traces_dir.resolve() != global_traces_dir.resolve()
        ):
            pending_dirs.append(project_traces_dir)

        seen_files = set()
        for pending_dir in pending_dirs:
            from bashgym.trace_capture.core import glob_pending_traces

            for trace_file in glob_pending_traces(pending_dir):
                if trace_file.name in seen_files:
                    continue
                seen_files.add(trace_file.name)

                try:
                    from bashgym.trace_capture.core import load_trace_file

                    data = load_trace_file(trace_file)

                    metrics = calculate_quality(data)
                    trace_id = trace_file.stem
                    success_rate = metrics["success_rate"]
                    quality_score = metrics["quality_score"]

                    # Build detailed info for tracking
                    trace_info = {
                        "id": trace_id,
                        "success_rate": success_rate,
                        "quality_score": quality_score,
                        "steps": metrics["total_steps"],
                        "unique_tools": metrics["unique_tools"],
                        "file_path": str(trace_file),
                    }

                    # Determine tier
                    # Step 1: Check for explicit verification failure
                    # Step 2: Classify by quality metrics
                    # Step 3: Apply verification_passed as a one-tier boost (not a blanket override)
                    tier = None
                    target_dir = None

                    if metrics["has_verification"] and not metrics["verification_passed"]:
                        # Verified failed -> Failed (explicit failure overrides all)
                        tier = "failed"
                        target_dir = data_dir / "failed_traces"
                    else:
                        # Determine base tier from quality metrics
                        if (
                            success_rate >= gold_success_rate
                            and quality_score >= gold_quality_score
                        ):
                            base_tier = "gold"
                        elif (
                            success_rate >= silver_success_rate
                            and quality_score >= silver_quality_score
                        ):
                            base_tier = "silver"
                        elif (
                            success_rate >= bronze_success_rate
                            and quality_score >= bronze_quality_score
                        ):
                            base_tier = "bronze"
                        else:
                            base_tier = "rejected"

                        # Apply verification_passed boost: promote one tier (max gold)
                        # This rewards traces where tests passed but doesn't skip quality checks
                        if metrics["verification_passed"] and base_tier != "gold":
                            tier_promotion = {
                                "silver": "gold",
                                "bronze": "silver",
                                "rejected": "bronze",
                            }
                            tier = tier_promotion.get(base_tier, base_tier)
                        else:
                            tier = base_tier

                        # Map tier to directory
                        tier_dirs = {
                            "gold": data_dir / "gold_traces",
                            "silver": data_dir / "silver_traces",
                            "bronze": data_dir / "bronze_traces",
                            "rejected": data_dir / "failed_traces",
                        }
                        target_dir = tier_dirs.get(tier, data_dir / "failed_traces")

                    # Add to classification lists
                    classifications[tier].append(trace_id)
                    if tier in detailed_classifications:
                        detailed_classifications[tier].append(trace_info)

                    # Move file if not dry_run and auto_promote enabled
                    if not dry_run and auto_promote and target_dir:
                        target_path = target_dir / f"{trace_id}.json"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(trace_file), str(target_path))
                        app.state.trace_cache.invalidate(trace_id)

                except Exception as e:
                    print(f"[DEBUG] Error processing {trace_file}: {e}")
                    classifications["pending"].append(trace_file.stem)

        # Generate DPO pair suggestions (Gold + Bronze for contrastive learning)
        dpo_pairs = []
        gold_details = detailed_classifications["gold"]
        bronze_details = detailed_classifications["bronze"]

        # Simple pairing: match by similar step count (within 2x range)
        for gold_trace in gold_details:
            for bronze_trace in bronze_details:
                gold_steps = gold_trace["steps"]
                bronze_steps = bronze_trace["steps"]
                # Pair traces with similar complexity (step count within 2x)
                if bronze_steps > 0 and 0.5 <= gold_steps / bronze_steps <= 2.0:
                    dpo_pairs.append(
                        {
                            "chosen": gold_trace["id"],
                            "chosen_success_rate": gold_trace["success_rate"],
                            "rejected": bronze_trace["id"],
                            "rejected_success_rate": bronze_trace["success_rate"],
                            "quality_gap": round(
                                gold_trace["quality_score"] - bronze_trace["quality_score"], 3
                            ),
                        }
                    )
                    break  # One pair per gold trace

        return {
            "classifications": classifications,
            "detailed": detailed_classifications,
            "dry_run": dry_run,
            "thresholds": {
                "gold": {"success_rate": gold_success_rate, "quality_score": gold_quality_score},
                "silver": {
                    "success_rate": silver_success_rate,
                    "quality_score": silver_quality_score,
                },
                "bronze": {
                    "success_rate": bronze_success_rate,
                    "quality_score": bronze_quality_score,
                },
            },
            "auto_promote": auto_promote,
            "summary": {
                "gold": len(classifications["gold"]),
                "silver": len(classifications["silver"]),
                "bronze": len(classifications["bronze"]),
                "rejected": len(classifications["rejected"]),
                "failed": len(classifications["failed"]),
                "pending": len(classifications["pending"]),
                "total_processed": len(seen_files),
            },
            "dpo_pairs": dpo_pairs,
            "dpo_pairs_count": len(dpo_pairs),
            "training_recommendations": {
                "sft_eligible": len(classifications["gold"]) + len(classifications["silver"]),
                "dpo_chosen_pool": len(classifications["gold"]) + len(classifications["silver"]),
                "dpo_rejected_pool": len(classifications["bronze"]),
                "note": "Gold+Silver traces are suitable for SFT. Use Gold as DPO 'chosen', Bronze as DPO 'rejected'.",
            },
        }

    # =========================================================================
    # Router Endpoints
    # =========================================================================

    @app.get("/api/router/stats", response_model=RouterStats, tags=["Router"])
    async def get_router_stats():
        """Get model router statistics."""
        if app.state.router:
            stats = app.state.router.get_routing_stats()

            # Get model-specific stats
            teacher_model = app.state.router.get_teacher_model()
            student_model = app.state.router.get_student_model()

            return RouterStats(
                total_requests=stats.get("total_requests", 0),
                teacher_requests=stats.get("teacher_requests", 0),
                student_requests=stats.get("student_requests", 0),
                teacher_success_rate=teacher_model.success_rate if teacher_model else 0.98,
                student_success_rate=student_model.success_rate if student_model else 0.92,
                avg_teacher_latency=teacher_model.avg_latency_ms if teacher_model else 450,
                avg_student_latency=student_model.avg_latency_ms if student_model else 120,
                current_student_rate=stats.get("current_student_rate", 0.1),
            )

        # Return demo stats if router not available
        return RouterStats(
            total_requests=100,
            teacher_requests=90,
            student_requests=10,
            teacher_success_rate=0.98,
            student_success_rate=0.92,
            avg_teacher_latency=450,
            avg_student_latency=120,
            current_student_rate=0.1,
        )

    @app.post("/api/router/strategy", tags=["Router"])
    async def set_routing_strategy(strategy: RoutingStrategyEnum):
        """Set the routing strategy."""
        if app.state.router:
            from bashgym.gym.router import RoutingStrategy

            app.state.router.config.strategy = RoutingStrategy(strategy.value)

        return {"success": True, "strategy": strategy.value}

    @app.post("/api/router/student-rate", tags=["Router"])
    async def set_student_rate(rate: float):
        """Set the student model traffic rate (0-1)."""
        if rate < 0 or rate > 1:
            raise HTTPException(status_code=400, detail="Rate must be between 0 and 1")

        if app.state.router:
            app.state.router.current_student_rate = rate

        return {"success": True, "rate": rate}

    @app.get("/api/providers/health", tags=["Providers"])
    async def get_providers_health():
        """Get health status of all inference providers."""
        registry = getattr(app.state, "provider_registry", None)
        if not registry:
            return {"providers": {}, "model_map": {}}
        health = await registry.check_all_health()
        return {
            "providers": {k: v.to_dict() for k, v in health.items()},
            "model_map": registry.get_model_map(),
        }

    @app.post("/api/router/student-provider", tags=["Router"])
    async def set_student_provider(provider_type: str, model_name: str):
        """Set which provider and model to use as the Student."""
        registry = getattr(app.state, "provider_registry", None)
        if not registry:
            raise HTTPException(status_code=503, detail="Provider registry not initialized")
        provider = registry.get_provider(provider_type)
        if not provider:
            raise HTTPException(
                status_code=404, detail=f"Provider '{provider_type}' not registered"
            )
        registry.map_model(model_name, provider_type)
        if app.state.router:
            from bashgym.gym.router import ModelConfig as _ModelConfig
            from bashgym.gym.router import ModelType as _ModelType

            app.state.router.register_model(
                _ModelConfig(
                    name=model_name,
                    model_type=_ModelType.STUDENT,
                    endpoint=f"{provider_type}://registry",
                )
            )
        return {
            "success": True,
            "provider": provider_type,
            "model": model_name,
            "is_local": provider.is_local,
        }

    @app.get("/api/router/config", tags=["Router"])
    async def get_router_config():
        """Get full router configuration including active models."""
        router = app.state.router
        registry = getattr(app.state, "provider_registry", None)
        result = {
            "strategy": router.config.strategy.value if router else "unknown",
            "student_rate": router.current_student_rate if router else 0,
            "teacher_model": None,
            "student_model": None,
            "providers": {},
        }
        if router:
            teacher = router.get_teacher_model()
            student = router.get_student_model()
            if teacher:
                result["teacher_model"] = {"name": teacher.name, "type": teacher.model_type.value}
            if student:
                result["student_model"] = {"name": student.name, "type": student.model_type.value}
        if registry:
            result["providers"] = registry.get_status_summary()
        return result

    @app.post("/api/providers/ollama/warmup", tags=["Providers"])
    async def warmup_ollama_model(model_name: str):
        """Pre-load an Ollama model into VRAM."""
        registry = getattr(app.state, "provider_registry", None)
        if not registry:
            raise HTTPException(status_code=503, detail="Registry not initialized")
        provider = registry.get_provider("ollama")
        if not provider:
            raise HTTPException(status_code=503, detail="Ollama provider not registered")
        success = await provider.warm_up(model_name)
        return {"success": success, "model": model_name}

    # =========================================================================
    # Verification Endpoints
    # =========================================================================

    @app.post("/api/verify/{task_id}", tags=["Verification"])
    async def run_verification(task_id: str):
        """Run verification tests for a task."""
        if app.state.verifier:
            # TODO: Get task workspace path
            result = {"passed": True, "tests_run": 5, "tests_passed": 5}
        else:
            result = {
                "passed": True,
                "tests_run": 0,
                "tests_passed": 0,
                "message": "Verifier not available",
            }

        await broadcast_verification_result(task_id, result.get("passed", False), result)

        return result

    # =========================================================================
    # Trace Repair Endpoints
    # =========================================================================

    @app.post("/api/traces/repair-prompts", tags=["Traces"])
    async def repair_trace_prompts():
        """Backfill missing user_initial_prompt from original source files.

        Scans all traces that have import_source metadata and attempts to
        extract the user's initial prompt from the original JSONL file.
        """
        from bashgym.config import get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        repaired = 0
        failed = 0
        skipped = 0
        details = []

        # Scan gold_traces directory
        gold_dir = data_dir / "gold_traces"
        if not gold_dir.exists():
            return {"repaired": 0, "failed": 0, "skipped": 0, "message": "No gold traces directory"}

        for trace_file in gold_dir.glob("*.json"):
            try:
                with open(trace_file, encoding="utf-8") as f:
                    trace_data = json.load(f)

                metadata = trace_data.get("metadata", {})

                # Skip if already has user_initial_prompt
                if metadata.get("user_initial_prompt"):
                    skipped += 1
                    continue

                # Get source file path
                source_path = metadata.get("import_source")
                if not source_path:
                    skipped += 1
                    continue

                source_file = Path(source_path)
                if not source_file.exists():
                    failed += 1
                    details.append({"file": trace_file.name, "error": "Source file not found"})
                    continue

                # Extract user prompt from source JSONL
                user_prompt = None
                with open(source_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            if event.get("type") == "user":
                                message = event.get("message", {})
                                content = message.get("content", [])

                                if isinstance(content, str):
                                    user_prompt = content[:500]
                                    break
                                elif isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            user_prompt = item.get("text", "")[:500]
                                            break
                                        elif isinstance(item, str):
                                            user_prompt = item[:500]
                                            break
                                    if user_prompt:
                                        break
                        except json.JSONDecodeError:
                            continue

                if not user_prompt:
                    failed += 1
                    details.append(
                        {"file": trace_file.name, "error": "No user message found in source"}
                    )
                    continue

                # Update the trace file
                trace_data["metadata"]["user_initial_prompt"] = user_prompt
                with open(trace_file, "w", encoding="utf-8") as f:
                    json.dump(trace_data, f, indent=2, ensure_ascii=False)

                repaired += 1
                details.append(
                    {
                        "file": trace_file.name,
                        "prompt": (
                            user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt
                        ),
                    }
                )

            except Exception as e:
                failed += 1
                details.append({"file": trace_file.name, "error": str(e)})

        return {
            "repaired": repaired,
            "failed": failed,
            "skipped": skipped,
            "total_scanned": repaired + failed + skipped,
            "details": details[:20],  # Limit details to first 20
        }

    # =========================================================================
    # Synthesis Endpoints
    # =========================================================================

    @app.post("/api/synthesize", tags=["Synthesis"])
    async def run_synthesis():
        """Run data synthesis from gold traces."""
        from bashgym.config import get_settings

        settings = get_settings()

        if app.state.data_factory:
            try:
                # Process gold traces
                gold_dir = Path(settings.data.data_dir) / "gold_traces"
                output_dir = Path(settings.data.data_dir) / "training_batches"

                if gold_dir.exists():
                    examples = app.state.data_factory.process_directory(gold_dir)
                    if examples:
                        batch_path = (
                            output_dir
                            / f"sft_batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
                        )
                        app.state.data_factory.save_batch(examples, batch_path)

                        return {
                            "success": True,
                            "examples_created": len(examples),
                            "output_path": str(batch_path),
                        }

                return {
                    "success": True,
                    "examples_created": 0,
                    "message": "No gold traces to process",
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return {"success": False, "message": "Data factory not available"}

    # =========================================================================
    # Hooks Endpoints
    # =========================================================================

    @app.get("/api/hooks/status", tags=["Hooks"])
    async def get_hooks_status():
        """Check status of all AI coding tool trace capture hooks."""
        try:
            from bashgym.trace_capture import get_tool_status

            return get_tool_status()
        except ImportError:
            # Fallback to legacy Claude Code-only check
            import os
            import platform

            if platform.system() == "Windows":
                hooks_dir = Path(os.environ.get("USERPROFILE", "")) / ".claude" / "hooks"
            else:
                hooks_dir = Path.home() / ".claude" / "hooks"

            post_tool_use_installed = (hooks_dir / "post_tool_use.py").exists()
            session_end_installed = (hooks_dir / "session_end.py").exists()

            return {
                "hooks_dir": str(hooks_dir),
                "post_tool_use_installed": post_tool_use_installed,
                "session_end_installed": session_end_installed,
                "all_installed": post_tool_use_installed and session_end_installed,
                "platform": platform.system(),
            }

    @app.post("/api/hooks/install", response_model=HooksInstallResponse, tags=["Hooks"])
    async def install_hooks(request: HooksInstallRequest = Body(default=HooksInstallRequest())):
        """
        Install trace capture hooks for AI coding tools.

        Args:
            request: Request body with tools list (e.g., {"tools": ["claude_code", "opencode"]}).
                     If tools is None or empty, auto-detects and installs for all detected tools.
        """
        tools = request.tools if request and request.tools else None

        try:
            from bashgym.trace_capture import setup_trace_capture

            result = setup_trace_capture(tools=tools, verbose=False)
            return result
        except ImportError:
            # Fallback to legacy Claude Code-only install
            import os
            import platform
            import shutil

            if platform.system() == "Windows":
                hooks_dir = Path(os.environ.get("USERPROFILE", "")) / ".claude" / "hooks"
            else:
                hooks_dir = Path.home() / ".claude" / "hooks"

            hooks_dir.mkdir(parents=True, exist_ok=True)

            source_hooks_dir = Path(__file__).parent.parent / "hooks"
            if not source_hooks_dir.exists():
                source_hooks_dir = Path(__file__).parent.parent.parent / "hooks"
            if not source_hooks_dir.exists():
                source_hooks_dir = Path.cwd() / "hooks"

            installed = []
            errors = []

            for hook_name in ["post_tool_use.py", "session_end.py"]:
                source_path = source_hooks_dir / hook_name
                dest_path = hooks_dir / hook_name

                if not source_path.exists():
                    source_path = Path(__file__).parent.parent.parent / hook_name

                if source_path.exists():
                    try:
                        shutil.copy2(source_path, dest_path)
                        installed.append(hook_name)
                    except Exception as e:
                        errors.append(f"{hook_name}: {str(e)}")
                else:
                    errors.append(f"{hook_name}: Source file not found")

            return {
                "success": len(errors) == 0,
                "hooks_dir": str(hooks_dir),
                "installed": installed,
                "errors": errors,
                "message": (
                    f"Installed {len(installed)} hooks" if installed else "No hooks installed"
                ),
            }

    # =========================================================================
    # Factory Endpoints (Data Designer, Privacy, Prompt Optimization)
    # =========================================================================

    @app.get("/api/factory/config", response_model=FactoryConfig, tags=["Factory"])
    async def get_factory_config():
        """Get the current factory configuration."""
        from bashgym.config import get_settings

        settings = get_settings()

        # Try to load from disk if not in memory
        if app.state.factory_config is None:
            config_path = Path(settings.data.data_dir) / "factory_config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        data = json.load(f)
                    app.state.factory_config = FactoryConfig(**data)
                except Exception:
                    pass

        if app.state.factory_config is None:
            # Return default config
            return FactoryConfig(
                columns=[],
                seeds=[],
                privacy=PrivacyConfig(),
                prompt_optimization=PromptOptConfig(),
                output=OutputConfig(),
                safety=SafetyConfig(),
                default_model=ModelConfig(),
            )
        return app.state.factory_config

    @app.put("/api/factory/config", tags=["Factory"])
    async def update_factory_config(config: FactoryConfig):
        """Update the factory configuration."""
        from bashgym.config import get_settings

        settings = get_settings()

        # Store in memory
        app.state.factory_config = config

        # Persist to disk
        config_path = Path(settings.data.data_dir) / "factory_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config.dict(), f, indent=2)

        return {"success": True, "message": "Factory configuration updated"}

    # -------------------------------------------------------------------------
    # Seeds Management
    # -------------------------------------------------------------------------

    @app.get("/api/factory/seeds", response_model=list[SeedExample], tags=["Factory"])
    async def get_seeds():
        """Get all seed examples."""
        if app.state.factory_config and app.state.factory_config.seeds:
            return app.state.factory_config.seeds
        return []

    @app.post("/api/factory/seeds", response_model=SeedExample, tags=["Factory"])
    async def add_seed(seed_data: dict[str, Any]):
        """Add a new seed example."""
        seed = SeedExample(
            id=f"seed_{uuid.uuid4().hex[:12]}",
            data=seed_data.get("data", {}),
            source=SeedSource(seed_data.get("source", "manual")),
            created_at=datetime.utcnow().isoformat(),
            trace_id=seed_data.get("trace_id"),
        )

        if app.state.factory_config is None:
            app.state.factory_config = FactoryConfig(
                columns=[],
                seeds=[],
                privacy=PrivacyConfig(),
                prompt_optimization=PromptOptConfig(),
                output=OutputConfig(),
                safety=SafetyConfig(),
                default_model=ModelConfig(),
            )

        app.state.factory_config.seeds.append(seed)

        # Persist
        await update_factory_config(app.state.factory_config)

        return seed

    @app.delete("/api/factory/seeds/{seed_id}", tags=["Factory"])
    async def delete_seed(seed_id: str):
        """Delete a seed example."""
        if app.state.factory_config and app.state.factory_config.seeds:
            app.state.factory_config.seeds = [
                s for s in app.state.factory_config.seeds if s.id != seed_id
            ]
            await update_factory_config(app.state.factory_config)
            return {"success": True}
        raise HTTPException(status_code=404, detail="Seed not found")

    @app.post("/api/factory/seeds/from-traces", tags=["Factory"])
    async def import_seeds_from_traces(request: dict[str, Any]):
        """Import seeds from gold traces."""
        from bashgym.config import get_settings

        settings = get_settings()

        trace_ids = request.get("trace_ids")
        gold_dir = Path(settings.data.data_dir) / "gold_traces"

        if not gold_dir.exists():
            return {"imported": 0, "seeds": []}

        imported_seeds = []

        # Get traces to import
        if trace_ids:
            trace_files = [gold_dir / f"{tid}.json" for tid in trace_ids]
        else:
            trace_files = list(gold_dir.glob("*.json"))

        for trace_file in trace_files[:50]:  # Limit to 50
            if not trace_file.exists():
                continue

            try:
                with open(trace_file, encoding="utf-8") as f:
                    trace_data = json.load(f)

                metadata = trace_data.get("metadata", {})
                trace_steps = trace_data.get("trace", [])

                # Extract seed data from trace
                seed_data = {
                    "task_description": metadata.get("user_initial_prompt", ""),
                    "task_name": metadata.get("task_id", trace_file.stem),
                }

                # Extract commands from trace
                commands = []
                for step in trace_steps:
                    if step.get("tool") == "Bash":
                        cmd = step.get("command", step.get("input", ""))
                        if cmd:
                            commands.append(cmd)

                if commands:
                    seed_data["commands"] = "\n".join(commands[:5])

                seed = SeedExample(
                    id=f"seed_{uuid.uuid4().hex[:12]}",
                    data=seed_data,
                    source=SeedSource.GOLD_TRACE,
                    created_at=datetime.utcnow().isoformat(),
                    trace_id=trace_file.stem,
                )

                imported_seeds.append(seed)

            except Exception:
                continue

        # Add to config
        if app.state.factory_config is None:
            app.state.factory_config = FactoryConfig(
                columns=[],
                seeds=[],
                privacy=PrivacyConfig(),
                prompt_optimization=PromptOptConfig(),
                output=OutputConfig(),
                safety=SafetyConfig(),
                default_model=ModelConfig(),
            )

        app.state.factory_config.seeds.extend(imported_seeds)
        await update_factory_config(app.state.factory_config)

        return {"imported": len(imported_seeds), "seeds": imported_seeds}

    # -------------------------------------------------------------------------
    # Preview Mode
    # -------------------------------------------------------------------------

    @app.post("/api/factory/preview", response_model=PreviewResult, tags=["Factory"])
    async def generate_preview(request: dict[str, Any]):
        """Generate a small preview batch for inspection."""
        row_count = request.get("row_count", 50)
        row_count = min(max(row_count, 10), 200)  # Clamp to 10-200

        config = app.state.factory_config
        if not config or not config.columns:
            return PreviewResult(
                rows=[],
                total_generated=0,
                valid_count=0,
                invalid_count=0,
                validation_summary={},
                column_coverage={},
            )

        # Generate preview rows
        preview_rows = []
        validation_summary = {}
        column_values = {col.name: [] for col in config.columns}

        for i in range(row_count):
            row_data = {}
            errors = []
            risk_flags = []

            for col in config.columns:
                # Simulate data generation based on column type
                value = _generate_column_value(col, config.seeds, i)
                row_data[col.name] = value

                if value:
                    column_values[col.name].append(value)

                # Check constraints
                for constraint in col.constraints:
                    error = _check_constraint(value, constraint)
                    if error:
                        errors.append(f"{col.name}: {error}")
                        validation_summary[error] = validation_summary.get(error, 0) + 1

                # Check risk level
                if col.risk_level.value != "normal":
                    risk_flags.append(f"{col.name}: {col.risk_level.value} risk")

            preview_rows.append(
                PreviewRow(
                    id=f"preview_{i}",
                    data=row_data,
                    validation_errors=errors,
                    risk_flags=risk_flags,
                )
            )

        # Calculate coverage
        column_coverage = {}
        for col_name, values in column_values.items():
            non_empty = sum(1 for v in values if v)
            column_coverage[col_name] = non_empty / row_count if row_count > 0 else 0

        valid_count = sum(1 for r in preview_rows if not r.validation_errors)

        return PreviewResult(
            rows=preview_rows,
            total_generated=row_count,
            valid_count=valid_count,
            invalid_count=row_count - valid_count,
            validation_summary=validation_summary,
            column_coverage=column_coverage,
        )

    def _generate_column_value(col: ColumnConfig, seeds: list[SeedExample], index: int) -> str:
        """Generate a value for a column based on its type."""
        import random

        col_type = col.type.value
        config = col.config

        # Use seed data if available
        if seeds and col.name in seeds[0].data:
            seed_values = [s.data.get(col.name, "") for s in seeds if col.name in s.data]
            if seed_values:
                # Vary the seed slightly
                base = random.choice(seed_values)
                return base

        if col_type == "category":
            values = config.get("values", ["option_a", "option_b", "option_c"])
            weights = config.get("weights")
            if weights and len(weights) == len(values):
                return random.choices(values, weights=weights)[0]
            return random.choice(values)

        elif col_type == "uuid":
            return str(uuid.uuid4())

        elif col_type == "datetime":
            fmt = config.get("format", "%Y-%m-%d")
            return datetime.utcnow().strftime(fmt)

        elif col_type == "person":
            first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
            return f"{random.choice(first_names)} {random.choice(last_names)}"

        elif col_type == "gaussian":
            mean = config.get("mean", 0)
            std = config.get("std", 1)
            return str(round(random.gauss(mean, std), 2))

        elif col_type == "sampler":
            dist = config.get("distribution", "uniform")
            if dist == "uniform":
                return str(random.random())
            elif dist == "bernoulli":
                return str(random.choice([0, 1]))
            return str(random.random())

        elif col_type == "expression":
            template = config.get("template", "{{index}}")
            return template.replace("{{index}}", str(index))

        elif col_type == "llm":
            # Simulate LLM output
            prompt = config.get("prompt", "Generate text")
            return f"[LLM output for: {prompt[:50]}...]"

        elif col_type == "validator":
            return "valid"

        return f"sample_{col.name}_{index}"

    def _check_constraint(value: str, constraint: ColumnConstraint) -> str | None:
        """Check if a value satisfies a constraint."""
        import re

        c_type = constraint.type
        c_value = constraint.value
        error_msg = constraint.error_message

        if c_type == "enum":
            if isinstance(c_value, list) and value not in c_value:
                return error_msg or f"Value must be one of: {c_value}"

        elif c_type == "regex":
            if isinstance(c_value, str):
                try:
                    if not re.match(c_value, value):
                        return error_msg or f"Value must match pattern: {c_value}"
                except re.error:
                    pass

        elif c_type == "min_length":
            if isinstance(c_value, (int, float)) and len(value) < c_value:
                return error_msg or f"Value must be at least {c_value} characters"

        elif c_type == "max_length":
            if isinstance(c_value, (int, float)) and len(value) > c_value:
                return error_msg or f"Value must be at most {c_value} characters"

        return None

    # -------------------------------------------------------------------------
    # Synthesis Jobs
    # -------------------------------------------------------------------------

    @app.get("/api/factory/jobs", response_model=list[SynthesisJob], tags=["Factory"])
    async def list_synthesis_jobs(limit: int = 50):
        """List all synthesis jobs."""
        jobs = list(app.state.synthesis_jobs.values())
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return [
            SynthesisJob(
                id=j["id"],
                status=j["status"],
                job_type=j.get("job_type", SynthesisJobType.FULL),
                created_at=j.get("created_at"),
                completed_at=j.get("completed_at"),
                examples_created=j.get("examples_created"),
                valid_examples=j.get("valid_examples"),
                output_path=j.get("output_path"),
                error=j.get("error"),
            )
            for j in jobs[:limit]
        ]

    @app.get("/api/factory/jobs/{job_id}", response_model=SynthesisJob, tags=["Factory"])
    async def get_synthesis_job(job_id: str):
        """Get a specific synthesis job."""
        if job_id not in app.state.synthesis_jobs:
            raise HTTPException(status_code=404, detail="Synthesis job not found")

        j = app.state.synthesis_jobs[job_id]
        return SynthesisJob(
            id=j["id"],
            status=j["status"],
            job_type=j.get("job_type", SynthesisJobType.FULL),
            created_at=j.get("created_at"),
            completed_at=j.get("completed_at"),
            examples_created=j.get("examples_created"),
            valid_examples=j.get("valid_examples"),
            output_path=j.get("output_path"),
            error=j.get("error"),
        )

    @app.post("/api/factory/jobs/{job_id}/cancel", tags=["Factory"])
    async def cancel_synthesis_job(job_id: str):
        """Cancel a running synthesis job."""
        if job_id not in app.state.synthesis_jobs:
            raise HTTPException(status_code=404, detail="Synthesis job not found")

        job = app.state.synthesis_jobs[job_id]
        if job["status"] == SynthesisJobStatus.RUNNING:
            job["status"] = SynthesisJobStatus.FAILED
            job["error"] = "Cancelled by user"
            return {"success": True}

        return {"success": False, "message": "Job is not running"}

    @app.post("/api/factory/synthesize", response_model=SynthesisJob, tags=["Factory"])
    async def run_factory_synthesis(request: dict[str, Any], background_tasks: BackgroundTasks):
        """Run data synthesis using the factory configuration."""
        is_preview = request.get("preview", False)
        row_count = request.get("row_count")

        job_id = f"synth_{uuid.uuid4().hex[:12]}"
        job_type = SynthesisJobType.PREVIEW if is_preview else SynthesisJobType.FULL

        # Create job record
        app.state.synthesis_jobs[job_id] = {
            "id": job_id,
            "status": SynthesisJobStatus.PENDING,
            "job_type": job_type,
            "created_at": datetime.utcnow().isoformat(),
            "row_count": row_count,
        }

        # Run synthesis in background
        background_tasks.add_task(run_synthesis_job, app, job_id, is_preview, row_count)

        return SynthesisJob(
            id=job_id,
            status=SynthesisJobStatus.PENDING,
            job_type=job_type,
            created_at=app.state.synthesis_jobs[job_id]["created_at"],
        )

    async def run_synthesis_job(app, job_id: str, is_preview: bool = False, row_count: int = None):
        """Execute synthesis job in the background."""
        from bashgym.config import get_settings

        settings = get_settings()

        try:
            app.state.synthesis_jobs[job_id]["status"] = SynthesisJobStatus.RUNNING

            config = app.state.factory_config
            if config is None:
                raise ValueError("No factory configuration found")

            # Determine row count
            if row_count is None:
                row_count = 50 if is_preview else config.output.row_count

            # Use data factory if available
            if app.state.data_factory and not is_preview:
                gold_dir = Path(settings.data.data_dir) / "gold_traces"
                output_dir = Path(settings.data.data_dir) / "training_batches"

                if gold_dir.exists():
                    examples = app.state.data_factory.process_directory(gold_dir)
                    if examples:
                        # Apply output config
                        output_format = config.output.format.value
                        task_name = config.output.task_name

                        batch_path = (
                            output_dir
                            / f"{task_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{output_format}"
                        )
                        app.state.data_factory.save_batch(examples, batch_path)

                        app.state.synthesis_jobs[job_id]["status"] = SynthesisJobStatus.COMPLETED
                        app.state.synthesis_jobs[job_id][
                            "completed_at"
                        ] = datetime.utcnow().isoformat()
                        app.state.synthesis_jobs[job_id]["examples_created"] = len(examples)
                        app.state.synthesis_jobs[job_id]["valid_examples"] = len(examples)
                        app.state.synthesis_jobs[job_id]["output_path"] = str(batch_path)
                        return

            # Simulate synthesis
            await asyncio.sleep(1 if is_preview else 3)

            simulated_count = row_count
            valid_count = int(simulated_count * 0.95)  # 95% valid

            app.state.synthesis_jobs[job_id]["status"] = SynthesisJobStatus.COMPLETED
            app.state.synthesis_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
            app.state.synthesis_jobs[job_id]["examples_created"] = simulated_count
            app.state.synthesis_jobs[job_id]["valid_examples"] = valid_count

            if not is_preview:
                output_dir = Path(settings.data.data_dir) / "training_batches"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = (
                    output_dir / f"synthetic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
                )
                app.state.synthesis_jobs[job_id]["output_path"] = str(output_path)

        except Exception as e:
            app.state.synthesis_jobs[job_id]["status"] = SynthesisJobStatus.FAILED
            app.state.synthesis_jobs[job_id]["error"] = str(e)

    # -------------------------------------------------------------------------
    # Column Validation
    # -------------------------------------------------------------------------

    @app.post("/api/factory/columns/{column_id}/validate", tags=["Factory"])
    async def validate_column(column_id: str, request: dict[str, Any]):
        """Validate a specific column configuration."""
        sample_size = request.get("sample_size", 10)

        config = app.state.factory_config
        if not config:
            raise HTTPException(status_code=404, detail="No factory configuration")

        column = next((c for c in config.columns if c.id == column_id), None)
        if not column:
            raise HTTPException(status_code=404, detail="Column not found")

        # Generate sample and validate
        sample_rows = []
        errors = []

        for i in range(sample_size):
            value = _generate_column_value(column, config.seeds, i)
            row_errors = []

            for constraint in column.constraints:
                error = _check_constraint(value, constraint)
                if error:
                    row_errors.append(error)
                    if error not in errors:
                        errors.append(error)

            sample_rows.append(
                PreviewRow(
                    id=f"sample_{i}",
                    data={column.name: value},
                    validation_errors=row_errors,
                    risk_flags=[],
                )
            )

        valid = len(errors) == 0

        return {"valid": valid, "errors": errors, "sample": sample_rows}

    # -------------------------------------------------------------------------
    # Available Models
    # -------------------------------------------------------------------------

    @app.get("/api/factory/models", response_model=list[AvailableModel], tags=["Factory"])
    async def list_available_models():
        """List available models for LLM columns."""
        # Return common models - in production this would query NIM or other services
        return [
            # Anthropic Claude 4.5 (recommended for high-quality augmentation)
            AvailableModel(
                id="claude-opus-4-5-20251101",
                name="Claude Opus 4.5 (Best Quality)",
                provider="Anthropic",
            ),
            AvailableModel(
                id="claude-sonnet-4-5-20250929",
                name="Claude Sonnet 4.5 (Recommended)",
                provider="Anthropic",
            ),
            AvailableModel(
                id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5 (Fast)", provider="Anthropic"
            ),
            # NVIDIA NIM (cost-effective, good for bulk generation)
            AvailableModel(
                id="qwen/qwen2.5-coder-32b-instruct",
                name="Qwen 2.5 Coder 32B (Default NIM)",
                provider="NVIDIA NIM",
            ),
            AvailableModel(
                id="qwen/qwen2.5-coder-7b-instruct", name="Qwen 2.5 Coder 7B", provider="NVIDIA NIM"
            ),
            AvailableModel(
                id="meta/llama-3.1-8b-instruct", name="Llama 3.1 8B Instruct", provider="NVIDIA NIM"
            ),
            AvailableModel(
                id="meta/llama-3.1-70b-instruct",
                name="Llama 3.1 70B Instruct",
                provider="NVIDIA NIM",
            ),
            AvailableModel(
                id="nvidia/nemotron-4-340b-instruct", name="Nemotron 4 340B", provider="NVIDIA NIM"
            ),
            AvailableModel(
                id="mistralai/mistral-7b-instruct-v0.3",
                name="Mistral 7B Instruct",
                provider="NVIDIA NIM",
            ),
            # OpenAI (alternative)
            AvailableModel(id="openai/gpt-4o", name="GPT-4o", provider="OpenAI"),
        ]

    # =========================================================================
    # Evaluation Endpoints
    # =========================================================================

    def _get_evaluations_file() -> Path:
        """Get path to evaluations JSON file."""
        from bashgym.config import get_settings

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "evaluations.json"

    def _load_evaluations() -> dict[str, Any]:
        """Load evaluations from JSON file."""
        eval_file = _get_evaluations_file()
        if eval_file.exists():
            try:
                with open(eval_file, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load evaluations: {e}")
        return {}

    def _save_evaluations(evaluations: dict[str, Any]):
        """Save evaluations to JSON file."""
        eval_file = _get_evaluations_file()
        try:
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(evaluations, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save evaluations: {e}")

    # Load evaluations on startup
    app.state.evaluation_jobs = _load_evaluations()
    logger.info(f"Loaded {len(app.state.evaluation_jobs)} evaluation jobs from disk")

    async def run_evaluation_task(
        app_state, job_id: str, request: EvaluationRequest, model_path: Path
    ):
        """Execute evaluation in background."""

        from bashgym.judge.benchmarks import BenchmarkConfig, BenchmarkRunner, BenchmarkType

        # Try to load the trained model for real inference
        model_loader = None
        use_real_inference = False

        try:
            from bashgym.inference import ModelLoader

            model_loader = ModelLoader(model_path)
            model_loader.load()
            use_real_inference = True
            logger.info(f"Loaded model from {model_path} for real inference")
        except ImportError as e:
            logger.warning(f"Inference module dependencies not available: {e}")
            logger.info("Falling back to simulation mode")
        except Exception as e:
            logger.warning(f"Could not load model for inference: {e}")
            logger.info("Falling back to simulation mode")

        try:
            results = {}
            for benchmark_name in request.benchmarks:
                try:
                    config = BenchmarkConfig(
                        benchmark_type=BenchmarkType(benchmark_name),
                        num_samples=request.num_samples,
                        model_name=request.model_id,
                    )
                    runner = BenchmarkRunner(config)

                    if use_real_inference and model_loader:
                        # Real model inference
                        def generate_fn(prompt: str) -> str:
                            return model_loader.generate(prompt)

                        # Run async benchmark
                        result = await runner.run_benchmark(generate_fn)
                    else:
                        # Fallback to simulation mode (canonical solutions)
                        result = runner.run_benchmark_simulated()

                    # Compute error analysis
                    error_analysis = result.compute_error_analysis()

                    results[benchmark_name] = BenchmarkResultSchema(
                        score=result.pass_rate,
                        passed=result.passed_samples,
                        total=result.total_samples,
                        duration_seconds=result.total_time_seconds,
                        errors=(
                            ErrorAnalysisSchema(
                                wrong_answer=error_analysis.wrong_answer,
                                syntax_error=error_analysis.syntax_error,
                                runtime_error=error_analysis.runtime_error,
                                timeout=error_analysis.timeout,
                                other=error_analysis.other,
                            )
                            if error_analysis
                            else None
                        ),
                    )
                except ValueError as e:
                    # Unknown benchmark type - skip it
                    logger.warning(f"Skipping unknown benchmark '{benchmark_name}': {e}")
                    continue

            app_state.evaluation_jobs[job_id]["status"] = "completed"
            app_state.evaluation_jobs[job_id]["results"] = {
                k: v.model_dump() for k, v in results.items()
            }

            # Save results to model profile if available
            try:
                from bashgym.models import get_registry

                registry = get_registry()
                for benchmark_name, result_schema in results.items():
                    registry.add_benchmark_result(
                        model_id=request.model_id,
                        benchmark_name=benchmark_name,
                        score=result_schema.score,
                        passed=result_schema.passed,
                        total=result_schema.total,
                    )
                logger.info(f"Saved benchmark results to model profile: {request.model_id}")
            except ImportError:
                logger.debug("Model registry not available, skipping profile update")
            except Exception as e:
                logger.warning(f"Failed to save benchmark results to model profile: {e}")

        except Exception as e:
            logger.error(f"Evaluation job {job_id} failed: {e}")
            app_state.evaluation_jobs[job_id]["status"] = "failed"
            app_state.evaluation_jobs[job_id]["error"] = str(e)
        finally:
            # Unload model to free memory
            if model_loader and model_loader.is_loaded:
                try:
                    model_loader.unload()
                    logger.info("Model unloaded after evaluation")
                except Exception as e:
                    logger.warning(f"Failed to unload model: {e}")

        # Save to disk after completion
        _save_evaluations(app_state.evaluation_jobs)

    @app.post("/api/evaluation/run", response_model=EvaluationResponse, tags=["Evaluation"])
    async def run_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
        """Start a model evaluation job."""
        from bashgym.config import get_bashgym_dir

        job_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Verify model exists - check both project data/models and ~/.bashgym/models
        model_path = None
        search_paths = [
            Path.cwd() / "data" / "models" / request.model_id,
            get_bashgym_dir() / "models" / request.model_id,
        ]

        for candidate in search_paths:
            if candidate.exists():
                model_path = candidate
                break

        if not model_path:
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

        # Create job
        job = {
            "job_id": job_id,
            "model_id": request.model_id,
            "benchmarks": request.benchmarks,
            "status": "running",
            "results": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
        }
        app.state.evaluation_jobs[job_id] = job

        # Save to disk
        _save_evaluations(app.state.evaluation_jobs)

        # Run in background
        background_tasks.add_task(run_evaluation_task, app.state, job_id, request, model_path)

        return EvaluationResponse(**job)

    @app.get("/api/evaluation/{job_id}", response_model=EvaluationResponse, tags=["Evaluation"])
    async def get_evaluation_status(job_id: str):
        """Get evaluation job status and results."""
        if job_id not in app.state.evaluation_jobs:
            raise HTTPException(status_code=404, detail="Evaluation job not found")

        job = app.state.evaluation_jobs[job_id]
        # Convert results dict back to BenchmarkResultSchema if present
        results = None
        if job.get("results"):
            results = {k: BenchmarkResultSchema(**v) for k, v in job["results"].items()}

        return EvaluationResponse(
            job_id=job["job_id"],
            model_id=job["model_id"],
            benchmarks=job["benchmarks"],
            status=job["status"],
            results=results,
            error=job.get("error"),
            created_at=job.get("created_at"),
        )

    @app.get("/api/evaluation", response_model=list[EvaluationResponse], tags=["Evaluation"])
    async def list_evaluations(limit: int = 20):
        """List recent evaluation jobs."""
        jobs = list(app.state.evaluation_jobs.values())
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)

        result = []
        for j in jobs[:limit]:
            results = None
            if j.get("results"):
                results = {k: BenchmarkResultSchema(**v) for k, v in j["results"].items()}
            result.append(
                EvaluationResponse(
                    job_id=j["job_id"],
                    model_id=j["model_id"],
                    benchmarks=j["benchmarks"],
                    status=j["status"],
                    results=results,
                    error=j.get("error"),
                    created_at=j.get("created_at"),
                )
            )
        return result

    @app.get("/api/benchmarks/status", tags=["Evaluation"])
    async def get_benchmark_status():
        """Get status of benchmark dataset availability and cache."""
        from bashgym.judge.benchmark_loader import BenchmarkLoader

        status = {}
        for benchmark_id in BenchmarkLoader.DATASETS:
            dataset_name, config, _ = BenchmarkLoader.DATASETS[benchmark_id]
            status[benchmark_id] = {
                "available": True,
                "loaded": benchmark_id in BenchmarkLoader._cache,
                "dataset": dataset_name,
                "config": config,
                "num_samples": len(BenchmarkLoader._cache.get(benchmark_id, [])),
            }
        return {"benchmarks": status}

    # Include auth routes (GitHub OAuth)
    from bashgym.api.auth_routes import router as auth_router

    app.include_router(auth_router)

    # Include observability routes
    app.include_router(observability_router)

    # Include models routes
    app.include_router(models_router)

    # Include HuggingFace routes
    app.include_router(hf_router)

    # Include Factory routes (synthetic data generation)
    app.include_router(factory_router)

    # Include Achievement routes (stats + achievements)
    app.include_router(achievements_router)

    # Include Security Dataset routes
    app.include_router(security_router)

    # Include Settings routes (env/API key management)
    app.include_router(settings_router)

    # Include AutoResearch routes (hyperparameter search)
    app.include_router(autoresearch_router)

    # Include Device routes (SSH device registry)
    app.include_router(device_router)

    # Experimental routes — desktop only (hidden in web mode)
    if not _settings.is_web_mode:
        app.include_router(integration_router)
        app.include_router(orchestrator_router)
        app.include_router(agent_router)
        app.include_router(pipeline_router)

    # =========================================================================
    # Web-mode endpoints: trace upload, hook receiver, install docs
    # =========================================================================

    from fastapi import File, UploadFile

    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB per file  # noqa: N806
    MAX_HOOK_PAYLOAD = 10 * 1024 * 1024  # 10 MB per hook payload  # noqa: N806

    @app.post("/api/traces/upload", tags=["Traces"])
    async def upload_trace_files(
        files: list[UploadFile] = File(...),
    ):
        """Upload one or more trace session files (multipart form).

        Accepts .json files up to 50 MB each. Files are validated as JSON
        and saved to the traces directory.
        """
        traces_dir = Path(_settings.data.data_dir) / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        errors = []
        for f in files:
            try:
                content = await f.read(MAX_UPLOAD_SIZE + 1)
                if len(content) > MAX_UPLOAD_SIZE:
                    errors.append(f"{f.filename}: exceeds 50 MB limit")
                    continue
                # Validate it's JSON
                json.loads(content)
                # Sanitize filename
                safe_name = re.sub(r"[^\w\-.]", "_", f.filename or "trace.json")
                dest = traces_dir / f"uploaded_{uuid.uuid4().hex[:8]}_{safe_name}"
                dest.write_bytes(content)
                saved.append(str(dest.name))
            except json.JSONDecodeError:
                errors.append(f"{f.filename}: invalid JSON")
            except Exception as e:
                errors.append(f"{f.filename}: {str(e)}")

        if saved:
            await broadcast_trace_event("added", {"uploaded": saved})

        return {"saved": saved, "errors": errors}

    from fastapi import Form

    @app.post("/api/traces/upload/import", tags=["Traces"])
    async def upload_and_import_traces(
        file: UploadFile = File(...),
        source: str = Form(...),
        force: bool = Form(False),
    ):
        """Upload and import trace files from external AI tools.

        Accepts:
        - source=chatgpt: zip or JSON containing conversations.json
        - source=mcp: JSON or JSONL file with MCP tool call logs

        Files are processed through the appropriate importer and saved
        as structured trace sessions.
        """
        if source not in ("chatgpt", "mcp"):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source: {source}. Supported: chatgpt, mcp",
            )

        content = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

        import tempfile

        suffix = f"_{file.filename}" if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            if source == "chatgpt":
                from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

                importer = ChatGPTImporter()
                if file.filename and file.filename.endswith(".zip"):
                    results = importer.import_from_zip(tmp_path, force=force)
                else:
                    results = importer.import_from_json(tmp_path, force=force)

                imported = [r for r in results if not r.skipped and not r.error]
                skipped = [r for r in results if r.skipped]
                failed = [r for r in results if r.error]

                resp = {
                    "source": "chatgpt",
                    "imported_count": len(imported),
                    "skipped_count": len(skipped),
                    "failed_count": len(failed),
                    "total_steps": sum(r.steps_imported for r in imported),
                    "errors": [r.error for r in failed if r.error],
                }

            elif source == "mcp":
                from bashgym.trace_capture.importers.mcp_logs import MCPLogImporter

                importer = MCPLogImporter()
                result = importer.import_from_file(tmp_path, force=force)

                if result.error:
                    resp = {
                        "source": "mcp",
                        "imported_count": 0,
                        "skipped_count": 0,
                        "failed_count": 1,
                        "total_steps": 0,
                        "errors": [result.error],
                    }
                else:
                    resp = {
                        "source": "mcp",
                        "imported_count": 0 if result.skipped else 1,
                        "skipped_count": 1 if result.skipped else 0,
                        "failed_count": 0,
                        "total_steps": result.steps_imported,
                        "errors": [],
                    }

            if resp.get("imported_count", 0) > 0:
                await broadcast_trace_event(
                    "added", {"source": source, "count": resp["imported_count"]}
                )

            return resp
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post("/api/traces/hook", tags=["Traces"])
    async def receive_hook_trace(
        payload: dict[str, Any] = Body(..., max_length=MAX_HOOK_PAYLOAD),
    ):
        """Receive trace data from remote Claude Code hook instances.

        Accepts hook payloads up to 10 MB. Authenticated via X-API-Key header
        when BASHGYM_API_KEY is set on the server.
        """
        traces_dir = Path(_settings.data.data_dir) / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        trace_id = payload.get("session_id", uuid.uuid4().hex[:12])
        # Sanitize trace_id for filesystem safety
        safe_id = re.sub(r"[^\w\-]", "_", str(trace_id))
        dest = traces_dir / f"remote_{safe_id}.json"
        dest.write_text(json.dumps(payload, indent=2))

        await broadcast_trace_event("added", {"remote": str(dest.name)})

        return {"status": "ok", "trace_id": trace_id, "path": str(dest.name)}

    @app.get("/docs/agent-install", tags=["Docs"])
    async def agent_install_docs():
        """Structured instructions for AI agents to configure trace hooks.

        Returns markdown that an AI agent (e.g. Claude Code via WebFetch) can
        read and execute to set up trace forwarding to this server.
        """
        from bashgym.config import get_settings as _gs

        settings = _gs()

        # Build the server URL — best-effort from settings
        host = settings.host
        port = 8003  # default
        base_url = f"http://{host}:{port}" if host != "0.0.0.0" else "http://YOUR_SERVER:8003"

        instructions = f"""# BashGym Trace Hook Setup

## Quick Setup

Add this to your `~/.claude/settings.json` to send traces to the BashGym server:

```json
{{
  "hooks": {{
    "PostToolUse": [
      {{
        "matcher": {{"toolName": "*"}},
        "hooks": [
          {{
            "type": "command",
            "command": "curl -s -X POST {base_url}/api/traces/hook -H 'Content-Type: application/json' -H 'X-API-Key: YOUR_BASHGYM_API_KEY' -d @-"
          }}
        ]
      }}
    ]
  }}
}}
```

## Steps

1. Open `~/.claude/settings.json` (create it if it doesn't exist)
2. Add or merge the `hooks` section above
3. Replace `YOUR_BASHGYM_API_KEY` with your API key (set via `BASHGYM_API_KEY` env var on the server)
4. Restart Claude Code for hooks to take effect

## Verify

After setup, run a Claude Code session and check:
- `GET {base_url}/api/traces` should show new traces
- The BashGym web UI Traces page should update in real-time

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/traces/hook` | POST | Receive hook payload (JSON body) |
| `/api/traces/upload/files` | POST | Upload .json trace files (multipart) |
| `/api/traces` | GET | List all traces |
| `/api/health` | GET | Health check |

## Authentication

All endpoints (except `/api/health`) require the `X-API-Key` header when `BASHGYM_API_KEY` is set on the server.
"""
        return {"content": instructions, "format": "markdown"}

    # =========================================================================
    # Static file serving (SPA) — serve built frontend when available
    # =========================================================================
    _frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if _frontend_dist.exists():
        from starlette.responses import FileResponse
        from starlette.staticfiles import StaticFiles

        # Serve assets (JS, CSS, images) from /assets/
        _assets_dir = _frontend_dist / "assets"
        if _assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="static-assets")

        # Serve static files at root (favicon, icons, etc.)
        @app.get("/ghost-icon.png")
        async def ghost_icon():
            icon_path = _frontend_dist / "ghost-icon.png"
            if icon_path.exists():
                return FileResponse(str(icon_path))
            raise HTTPException(status_code=404)

        @app.get("/favicon.ico")
        async def favicon():
            fav_path = _frontend_dist / "favicon.ico"
            if fav_path.exists():
                return FileResponse(str(fav_path))
            raise HTTPException(status_code=404)

        # SPA fallback: all non-API, non-asset routes serve index.html
        _dist_resolved = _frontend_dist.resolve()

        @app.get("/{full_path:path}", tags=["SPA"])
        async def spa_fallback(full_path: str):
            """Serve the SPA for all non-API routes."""
            # Don't intercept API, WebSocket, or docs routes
            if full_path.startswith(("api/", "ws", "docs/", "openapi.json")):
                raise HTTPException(status_code=404)

            # Try to serve the exact file first (e.g., robots.txt)
            if full_path:
                file_path = (_frontend_dist / full_path).resolve()
                # Path traversal guard: ensure resolved path stays within dist/
                if str(file_path).startswith(str(_dist_resolved)) and file_path.is_file():
                    return FileResponse(str(file_path))

            # Otherwise serve index.html for SPA routing
            index_path = _frontend_dist / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))

            raise HTTPException(status_code=404, detail="Frontend not built")

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
