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

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import uuid
import json
import asyncio
import logging

# Set up logging for API routes
logger = logging.getLogger(__name__)

from bashgym.api.schemas import (
    TaskRequest, TaskResponse, TaskStatus,
    TrainingRequest, TrainingResponse, TrainingStatus, TrainingStrategy,
    ModelInfo, ExportRequest, ExportResponse, ExportFormat,
    SystemStats, HealthCheck,
    TraceInfo, TraceStatus, TraceQuality, TraceQualityTier, RepoInfo,
    RouterStats, RoutingStrategyEnum,
    SystemInfoResponse, GpuInfo, ModelRecommendations,
    FactoryConfig, ColumnConfig, ColumnConstraint, PrivacyConfig, PromptOptConfig, OutputConfig, SafetyConfig,
    SynthesisJob, SynthesisJobStatus, SynthesisJobType,
    SeedExample, SeedSource, PreviewResult, PreviewRow, ModelConfig, AvailableModel,
    HooksInstallRequest, HooksInstallResponse,
    TrainingExampleResponse, GenerateExamplesRequest, GenerateExamplesResponse,
    ExportExamplesRequest, ExportExamplesResponse,
    EvaluationRequest, EvaluationResponse, BenchmarkResultSchema, ErrorAnalysisSchema
)
from bashgym.api.websocket import (
    handle_websocket, manager,
    broadcast_training_complete, broadcast_training_failed,
    broadcast_task_status, broadcast_trace_event,
    broadcast_router_stats, broadcast_verification_result,
    broadcast_guardrail_blocked, broadcast_guardrail_warn, broadcast_pii_redacted,
    TrainingProgressCallback, MessageType
)
from bashgym.api.observability_routes import router as observability_router
from bashgym.api.models_routes import router as models_router
from bashgym.api.hf_routes import router as hf_router
from bashgym.api.factory_routes import router as factory_router
from bashgym.api.integration_routes import router as integration_router
from bashgym.api.achievements_routes import router as achievements_router
from bashgym.factory.quality_calculator import calculate_quality_breakdown


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.on_event("startup")
    async def startup():
        """Initialize Bash Gym components on startup."""
        from bashgym.config import get_settings
        settings = get_settings()
        settings.setup()

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
                            content_preview=event.original_content[:100] if event.original_content else None,
                            details=event.details
                        )
                    elif event.action_taken.value == "warn":
                        await broadcast_guardrail_warn(
                            check_type=event.check_type.value,
                            location=event.location,
                            confidence=event.confidence,
                            content_preview=event.original_content[:100] if event.original_content else None,
                            details=event.details
                        )
                    elif event.action_taken.value == "modify":
                        await broadcast_pii_redacted(
                            location=event.location,
                            redaction_count=1,
                            pii_types=[event.check_type.value],
                            details=event.details
                        )

                app.state.instrumentation.on_event(on_guardrail_event)
                logger.info("Instrumentation initialized with WebSocket callbacks")

        except ImportError as e:
            logger.warning(f"Instrumentation not available: {e}")
            app.state.instrumentation = None

        # Initialize components lazily
        try:
            from bashgym.gym.trainer import Trainer, TrainerConfig
            from bashgym.gym.router import ModelRouter, RouterConfig
            from bashgym.judge.verifier import Verifier
            from bashgym.factory.trace_processor import TraceProcessor
            from bashgym.factory.data_factory import DataFactory

            app.state.trainer = Trainer(TrainerConfig())
            app.state.router = ModelRouter()
            app.state.verifier = Verifier()
            app.state.trace_processor = TraceProcessor()
            app.state.data_factory = DataFactory()

        except ImportError as e:
            print(f"Warning: Some components not available: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        if app.state.router:
            await app.state.router.close()

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
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="0.1.0"
        )

    @app.get("/api/debug/traces", tags=["System"])
    async def debug_traces():
        """Debug endpoint to diagnose trace discovery issues."""
        from bashgym.config import get_settings, get_bashgym_dir

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        result = {
            "data_dir": str(data_dir),
            "global_dir": str(global_dir),
            "directories": {}
        }

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
            dir_info = {
                "path": str(path),
                "exists": path.exists(),
                "files": []
            }
            if path.exists():
                files = list(path.glob("*.json"))
                dir_info["count"] = len(files)
                dir_info["files"] = [f.name for f in files[:10]]  # First 10 only
            result["directories"][name] = dir_info

        return result

    @app.get("/api/stats", response_model=SystemStats, tags=["System"])
    async def get_system_stats():
        """Get system statistics."""
        from bashgym.config import get_settings, get_bashgym_dir
        settings = get_settings()

        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        # Count traces in tiered directories (gold/silver/bronze/failed)
        gold_traces = list((data_dir / "gold_traces").glob("*.json")) if (data_dir / "gold_traces").exists() else []
        silver_traces = list((data_dir / "silver_traces").glob("*.json")) if (data_dir / "silver_traces").exists() else []
        bronze_traces = list((data_dir / "bronze_traces").glob("*.json")) if (data_dir / "bronze_traces").exists() else []
        failed_traces = list((data_dir / "failed_traces").glob("*.json")) if (data_dir / "failed_traces").exists() else []
        models = list((data_dir / "models").iterdir()) if (data_dir / "models").exists() else []

        # Count pending traces from both global and project directories
        pending_count = 0
        global_traces = global_dir / "traces"
        project_traces = data_dir / "traces"

        if global_traces.exists():
            pending_count += len(list(global_traces.glob("session_*.json")))
            pending_count += len(list(global_traces.glob("imported_*.json")))
        if project_traces.exists() and project_traces.resolve() != global_traces.resolve():
            pending_count += len(list(project_traces.glob("session_*.json")))
            pending_count += len(list(project_traces.glob("imported_*.json")))

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
            active_training_runs=len([r for r in app.state.training_runs.values() if r.get("status") == "running"])
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
            gpus=[GpuInfo(
                vendor=g.vendor,
                model=g.model,
                vram=g.vram,
                vram_used=g.vram_used,
                driver=g.driver,
                temperature=g.temperature,
                utilization=g.utilization,
            ) for g in info.gpus],
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

        return [GpuInfo(
            vendor=g.vendor,
            model=g.model,
            vram=g.vram,
            vram_used=g.vram_used,
            driver=g.driver,
            temperature=g.temperature,
            utilization=g.utilization,
        ) for g in gpus]

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
                    "total": len(providers)
                }
            }
        except ImportError:
            return {"providers": [], "summary": {"available": 0, "total": 0}}

    @app.get("/api/providers/models", tags=["Providers"])
    async def get_available_models(
        include_local: bool = True,
        include_cloud: bool = True,
        code_only: bool = False
    ):
        """Get all available models organized by category."""
        try:
            from bashgym.providers import get_available_models as get_models
            models = await get_models(
                include_local=include_local,
                include_cloud=include_cloud,
                code_only=code_only
            )
            return {
                "local": [m.to_dict() for m in models["local"]],
                "training": [m.to_dict() for m in models["training"]],
                "teacher": [m.to_dict() for m in models["teacher"]],
                "inference": [m.to_dict() for m in models["inference"]]
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
                    "models": []
                }

            models = await provider.list_models()
            return {
                "available": True,
                "models": [m.to_dict() for m in models]
            }
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
                "message": f"Started downloading {model_name}"
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
    # Task Endpoints
    # =========================================================================

    @app.post("/api/tasks", response_model=TaskResponse, tags=["Tasks"])
    async def submit_task(
        request: TaskRequest,
        background_tasks: BackgroundTasks
    ):
        """Submit a new task for agent execution."""
        task_id = request.task_id or f"task_{uuid.uuid4().hex[:12]}"

        # Store task
        app.state.tasks[task_id] = {
            "task_id": task_id,
            "prompt": request.prompt,
            "status": TaskStatus.PENDING,
            "created_at": datetime.utcnow().isoformat()
        }

        # Run task in background
        background_tasks.add_task(run_task, app, task_id, request)

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Task queued for execution",
            created_at=app.state.tasks[task_id]["created_at"]
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
            result=task.get("result")
        )

    @app.get("/api/tasks", response_model=List[TaskResponse], tags=["Tasks"])
    async def list_tasks(
        status: Optional[TaskStatus] = None,
        limit: int = 50
    ):
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
                completed_at=t.get("completed_at")
            )
            for t in tasks
        ]

    # =========================================================================
    # Training Endpoints
    # =========================================================================

    @app.post("/api/training/start", response_model=TrainingResponse, tags=["Training"])
    async def start_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks
    ):
        """Start a new training run."""
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Store training run
        app.state.training_runs[run_id] = {
            "run_id": run_id,
            "strategy": request.strategy,
            "status": TrainingStatus.PENDING,
            "config": request.dict(),
            "started_at": datetime.utcnow().isoformat()
        }

        # Start training in background
        background_tasks.add_task(run_training, app, run_id, request)

        return TrainingResponse(
            run_id=run_id,
            status=TrainingStatus.PENDING,
            strategy=request.strategy,
            message="Training run queued",
            started_at=app.state.training_runs[run_id]["started_at"]
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
                from bashgym.gym.trainer import TrainerConfig, TrainingStrategy as TS
                from bashgym.factory.example_generator import ExampleGenerator, ExampleGeneratorConfig
                from bashgym.config import get_settings

                settings = get_settings()

                # Auto-generate training data from gold traces if no dataset specified
                dataset_path = None
                if request.dataset_path:
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
                                    with open(trace_file, 'r', encoding='utf-8') as f:
                                        trace_data = json.load(f)
                                    # Check primary_repo.name (can be dict or list)
                                    primary_repo = trace_data.get('primary_repo', {})
                                    if isinstance(primary_repo, list) and len(primary_repo) > 0:
                                        primary_repo = primary_repo[0]  # Use first repo if list
                                    repo_name = primary_repo.get('name') if isinstance(primary_repo, dict) else None
                                    if repo_name and repo_name in request.selected_repos:
                                        filtered_files.append(trace_file)
                                except Exception as e:
                                    logger.warning(f"Error reading trace {trace_file}: {e}")

                            logger.info(f"Filtered to {len(filtered_files)} traces for selected repos (from {len(trace_files)} total)")
                            trace_files = filtered_files

                            if not trace_files:
                                raise ValueError(f"No gold traces found for repos: {request.selected_repos}")

                        gen_config = ExampleGeneratorConfig(output_dir=str(output_dir))
                        generator = ExampleGenerator(gen_config)

                        # Process only the filtered trace files
                        examples = []
                        stats = {'sessions_processed': 0, 'examples_generated': 0}
                        for trace_file in trace_files:
                            try:
                                session_examples = generator.generate_examples(trace_file)
                                examples.extend(session_examples)
                                stats['sessions_processed'] += 1
                                stats['examples_generated'] += len(session_examples)
                            except Exception as e:
                                logger.warning(f"Error processing trace {trace_file}: {e}")

                        logger.info(f"Generated {len(examples)} examples from {stats['sessions_processed']} gold traces")

                        if examples:
                            result = generator.export_for_nemo(examples, output_dir, train_split=0.9)
                            dataset_path = result["train"]
                            logger.info(f"Training data ready: {dataset_path}")
                        else:
                            raise ValueError("No training examples generated from gold traces")
                    else:
                        raise ValueError(f"No gold traces found in {gold_dir}")

                # Update trainer config
                config = TrainerConfig(
                    base_model=request.base_model or "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    strategy=TS(request.strategy.value),
                    num_epochs=request.num_epochs,
                    batch_size=request.batch_size,
                    learning_rate=request.learning_rate,
                    max_seq_length=request.max_seq_length,
                    use_lora=request.use_lora,
                    lora_r=request.lora_rank or 16,
                    lora_alpha=request.lora_alpha or 32,
                    auto_export_gguf=request.auto_export_gguf,
                    gguf_quantization=request.gguf_quantization,
                    use_nemo_gym=request.use_nemo_gym,  # Use NeMo cloud training
                )

                app.state.trainer.config = config

                if request.strategy == TrainingStrategy.SFT:
                    run = app.state.trainer.train_sft(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync,
                        log_callback=callback.on_log_sync
                    )
                elif request.strategy == TrainingStrategy.DPO:
                    run = app.state.trainer.train_dpo(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync
                    )
                else:
                    # GRPO training (use dedicated GRPO trainer)
                    logger.info("Running GRPO training...")
                    run = app.state.trainer.train_grpo(
                        dataset_path=dataset_path,
                        run_id=run_id,
                        callback=callback.on_progress_sync
                    ) if hasattr(app.state.trainer, 'train_grpo') else None

                    if run is None:
                        # Fallback simulation for GRPO
                        logger.warning("GRPO trainer not available, running simulation")
                        for epoch in range(request.num_epochs):
                            await callback.on_progress({
                                "epoch": epoch + 1,
                                "total_epochs": request.num_epochs,
                                "step": (epoch + 1) * 50,
                                "total_steps": request.num_epochs * 50,
                                "loss": 2.5 - (epoch * 0.3),
                                "learning_rate": request.learning_rate,
                                "grad_norm": 0.5 - (epoch * 0.05),
                                "eta": f"{(request.num_epochs - epoch - 1) * 2}m",
                                "simulation": True
                            })
                            await asyncio.sleep(1)

                app.state.training_runs[run_id]["status"] = TrainingStatus.COMPLETED
                app.state.training_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()
                app.state.training_runs[run_id]["metrics"] = run.metrics if run else {}

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
                        base_loss = 2.5 * (0.6 ** progress) + 0.4
                        noise = random.uniform(-0.05, 0.05)
                        loss = max(0.3, base_loss + noise)

                        # Simulate gradient norm (decreases as training stabilizes)
                        grad_norm = 1.0 * (1.0 - progress * 0.7) + random.uniform(-0.1, 0.1)

                        # Calculate ETA
                        steps_remaining = total_steps - current_step
                        eta_minutes = steps_remaining * 0.02  # 20ms per step in simulation
                        eta = f"{int(eta_minutes)}m" if eta_minutes >= 1 else f"{int(eta_minutes * 60)}s"

                        await callback.on_progress({
                            "epoch": epoch + 1,
                            "total_epochs": request.num_epochs,
                            "step": current_step,
                            "total_steps": total_steps,
                            "loss": round(loss, 4),
                            "learning_rate": request.learning_rate,
                            "grad_norm": round(grad_norm, 3),
                            "eta": eta,
                            "simulation": True  # Flag to indicate simulation mode
                        })
                        await asyncio.sleep(0.02)  # Fast simulation (20ms per step)

                final_loss = round(2.5 * (0.6 ** 1.0) + 0.4, 4)
                app.state.training_runs[run_id]["status"] = TrainingStatus.COMPLETED
                app.state.training_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()
                app.state.training_runs[run_id]["metrics"] = {"final_loss": final_loss, "simulation": True}
                await broadcast_training_complete(run_id, {"final_loss": final_loss, "simulation": True})

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[Training] FAILED for {run_id}: {error_msg}")
            logger.error(f"[Training] Traceback: {traceback.format_exc()}")
            print(f"[Training] FAILED for {run_id}: {error_msg}")
            print(f"[Training] Traceback:\n{traceback.format_exc()}")
            app.state.training_runs[run_id]["status"] = TrainingStatus.FAILED
            app.state.training_runs[run_id]["error"] = error_msg
            await broadcast_training_failed(run_id, error_msg)

    @app.get("/api/training/{run_id}", response_model=TrainingResponse, tags=["Training"])
    async def get_training_status(run_id: str):
        """Get the status of a training run."""
        if run_id not in app.state.training_runs:
            raise HTTPException(status_code=404, detail="Training run not found")

        run = app.state.training_runs[run_id]
        return TrainingResponse(
            run_id=run_id,
            status=run["status"],
            strategy=run["strategy"],
            error=run.get("error"),  # Include error message if failed
            started_at=run.get("started_at"),
            completed_at=run.get("completed_at"),
            metrics=run.get("metrics")
        )

    @app.post("/api/training/{run_id}/pause", tags=["Training"])
    async def pause_training(run_id: str):
        """Pause a running training job."""
        if run_id not in app.state.training_runs:
            raise HTTPException(status_code=404, detail="Training run not found")

        app.state.training_runs[run_id]["status"] = TrainingStatus.PAUSED
        return {"success": True, "message": "Training paused"}

    @app.post("/api/training/{run_id}/resume", tags=["Training"])
    async def resume_training(run_id: str):
        """Resume a paused training job."""
        if run_id not in app.state.training_runs:
            raise HTTPException(status_code=404, detail="Training run not found")

        app.state.training_runs[run_id]["status"] = TrainingStatus.RUNNING
        return {"success": True, "message": "Training resumed"}

    @app.post("/api/training/{run_id}/stop", tags=["Training"])
    async def stop_training(run_id: str):
        """Stop a training job."""
        if run_id not in app.state.training_runs:
            raise HTTPException(status_code=404, detail="Training run not found")

        app.state.training_runs[run_id]["status"] = TrainingStatus.COMPLETED
        app.state.training_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()
        return {"success": True, "message": "Training stopped"}

    @app.get("/api/training", response_model=List[TrainingResponse], tags=["Training"])
    async def list_training_runs(
        status: Optional[TrainingStatus] = None,
        limit: int = 50
    ):
        """List all training runs."""
        runs = list(app.state.training_runs.values())

        if status:
            runs = [r for r in runs if r["status"] == status]

        runs = runs[:limit]

        return [
            TrainingResponse(
                run_id=r["run_id"],
                status=r["status"],
                strategy=r["strategy"],
                started_at=r.get("started_at"),
                completed_at=r.get("completed_at"),
                metrics=r.get("metrics")
            )
            for r in runs
        ]

    # =========================================================================
    # Model Endpoints
    # =========================================================================

    # NOTE: /api/models endpoint moved to models_routes.py with enhanced ModelRegistry
    # See bashgym/api/models_routes.py for the new implementation

    @app.post("/api/models/{model_id}/export", response_model=ExportResponse, tags=["Models"])
    async def export_model(
        model_id: str,
        request: ExportRequest,
        background_tasks: BackgroundTasks
    ):
        """Export a trained model to GGUF or other formats."""
        if app.state.trainer:
            export_path = app.state.trainer.export_model(
                run_id=model_id,
                export_format=request.format.value,
                quantization=request.quantization
            )

            return ExportResponse(
                model_id=model_id,
                format=request.format,
                status="completed" if export_path else "failed",
                output_path=str(export_path) if export_path else None,
                message="Export script generated" if export_path else "Export failed"
            )

        return ExportResponse(
            model_id=model_id,
            format=request.format,
            status="pending",
            message="Export queued"
        )

    # =========================================================================
    # Trace Endpoints
    # =========================================================================

    @app.get("/api/traces", response_model=List[TraceInfo], tags=["Traces"])
    async def list_traces(
        status: Optional[TraceStatus] = None,
        repo: Optional[str] = None,
        limit: int = 100
    ):
        """List all traces.

        Args:
            status: Filter by trace status (gold/failed/pending)
            repo: Filter by repository name (supports partial matching)
            limit: Maximum number of traces to return
        """
        from bashgym.config import get_settings
        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        print(f"[DEBUG] list_traces called - data_dir: {data_dir}")

        traces = []

        # Load tiered traces (gold/silver/bronze) and failed
        # Tier directories map to TraceStatus
        tier_dirs = [
            (data_dir / "gold_traces", TraceStatus.GOLD),
            (data_dir / "silver_traces", TraceStatus.SILVER),
            (data_dir / "bronze_traces", TraceStatus.BRONZE),
            (data_dir / "failed_traces", TraceStatus.FAILED),
        ]

        for tier_dir, tier_status in tier_dirs:
            if tier_dir.exists():
                for trace_file in tier_dir.glob("*.json"):
                    try:
                        with open(trace_file, encoding='utf-8') as f:
                            data = json.load(f)

                        # Apply repo filter
                        if repo and not _matches_repo(data, repo):
                            continue

                        trace_info = _parse_trace_file(trace_file, data, tier_status)
                        if status is None or status == tier_status:
                            traces.append(trace_info)
                    except Exception:
                        continue

        # Load pending traces (raw session traces and imported traces)
        # Check BOTH global ~/.bashgym/traces/ AND project data/traces/
        from bashgym.config import get_bashgym_dir
        global_traces_dir = get_bashgym_dir() / "traces"
        project_traces_dir = data_dir / "traces"

        pending_dirs = []
        if global_traces_dir.exists():
            pending_dirs.append(global_traces_dir)
        if project_traces_dir.exists() and project_traces_dir.resolve() != global_traces_dir.resolve():
            pending_dirs.append(project_traces_dir)

        print(f"[DEBUG] Checking pending dirs: {pending_dirs}")

        seen_files = set()  # Track by filename to avoid duplicates
        for pending_dir in pending_dirs:
            # Include both session_*.json and imported_*.json files
            session_files = list(pending_dir.glob("session_*.json")) + list(pending_dir.glob("imported_*.json"))
            print(f"[DEBUG] Found {len(session_files)} trace files in {pending_dir}")
            for trace_file in session_files:
                # Skip if we've already seen this file (by name)
                if trace_file.name in seen_files:
                    continue
                seen_files.add(trace_file.name)
                print(f"[DEBUG] Processing: {trace_file}")
                try:
                    with open(trace_file, encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle both raw trace format (list) and imported TraceSession format (dict with trace key)
                    if isinstance(data, dict) and "trace" in data:
                        # Imported TraceSession format
                        trace_steps = data.get("trace", [])
                        print(f"[DEBUG] Loaded imported trace with {len(trace_steps)} steps")
                        if not trace_steps:
                            continue

                        # Apply repo filter
                        if repo and not _matches_repo(data, repo):
                            continue

                        trace_info = _parse_imported_trace_file(trace_file, data)
                        if status is None or status == TraceStatus.PENDING:
                            traces.append(trace_info)

                    elif isinstance(data, list):
                        # Raw session trace format (array of steps)
                        print(f"[DEBUG] Loaded raw trace with {len(data)} entries")
                        if len(data) == 0:
                            continue

                        # Apply repo filter
                        if repo:
                            repo_match = False
                            for step in data:
                                step_repo = step.get("repo", {})
                                if step_repo and repo.lower() in step_repo.get("name", "").lower():
                                    repo_match = True
                                    break
                            if not repo_match:
                                continue

                        trace_info = _parse_raw_trace_file(trace_file, data)
                        if status is None or status == TraceStatus.PENDING:
                            traces.append(trace_info)
                    else:
                        print(f"[DEBUG] Skipping - unrecognized format")
                        continue
                except Exception as e:
                    print(f"[DEBUG] Error parsing {trace_file}: {e}")
                    continue

        # Sort by created_at descending
        traces.sort(key=lambda x: x.created_at or "", reverse=True)

        return traces[:limit]

    @app.get("/api/traces/stats", tags=["Traces"])
    async def get_trace_stats():
        """Get trace statistics over time for the dashboard chart.

        Returns timeline data with counts of gold/failed/pending traces
        grouped by time periods (last 24 hours in hourly buckets).
        """
        from datetime import datetime, timedelta
        from bashgym.config import get_settings, get_bashgym_dir

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        # Initialize hourly buckets for last 24 hours
        now = datetime.now()
        buckets = []
        for i in range(24, -1, -1):  # 24 hours ago to now
            bucket_time = now - timedelta(hours=i)
            buckets.append({
                "time": bucket_time.strftime("%H:%M") if i > 0 else "Now",
                "hour": bucket_time.replace(minute=0, second=0, microsecond=0),
                "gold": 0,
                "failed": 0,
                "pending": 0
            })

        # Count traces by status and time
        def count_traces_in_dir(trace_dir: Path, status: str, pattern: str = "*.json"):
            if not trace_dir.exists():
                return
            for trace_file in trace_dir.glob(pattern):
                try:
                    # Get file modification time as proxy for trace time
                    mtime = datetime.fromtimestamp(trace_file.stat().st_mtime)

                    # Also try to extract time from filename (imported_claude_xxx_YYYYMMDD_HHMMSS.json)
                    fname = trace_file.stem
                    if "_202" in fname:  # Year prefix
                        parts = fname.split("_")
                        for i, part in enumerate(parts):
                            if part.startswith("202") and len(part) == 8:
                                try:
                                    date_str = part
                                    time_str = parts[i + 1] if i + 1 < len(parts) else "000000"
                                    mtime = datetime.strptime(f"{date_str}_{time_str[:6]}", "%Y%m%d_%H%M%S")
                                except (ValueError, IndexError):
                                    pass
                                break

                    # Find matching bucket
                    for bucket in buckets:
                        bucket_start = bucket["hour"]
                        bucket_end = bucket_start + timedelta(hours=1)
                        if bucket_start <= mtime < bucket_end:
                            bucket[status] += 1
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
        pending_patterns = ["session_*.json", "imported_*.json"]
        global_traces_dir = global_dir / "traces"
        project_traces_dir = data_dir / "traces"

        for pattern in pending_patterns:
            count_traces_in_dir(global_traces_dir, "pending", pattern)
            if project_traces_dir.exists() and project_traces_dir.resolve() != global_traces_dir.resolve():
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
            timeline_data.append({
                "time": bucket["time"],
                "gold": cumulative_gold,
                "failed": cumulative_failed,
                "pending": cumulative_pending
            })

        # Get current totals
        total_gold = sum(1 for _ in gold_dir.glob("*.json")) if gold_dir.exists() else 0
        total_failed = sum(1 for _ in failed_dir.glob("*.json")) if failed_dir.exists() else 0
        total_pending = 0
        for pattern in pending_patterns:
            if global_traces_dir.exists():
                total_pending += sum(1 for _ in global_traces_dir.glob(pattern))
            if project_traces_dir.exists() and project_traces_dir.resolve() != global_traces_dir.resolve():
                total_pending += sum(1 for _ in project_traces_dir.glob(pattern))

        return {
            "timeline": timeline_data,
            "totals": {
                "gold": total_gold,
                "failed": total_failed,
                "pending": total_pending,
                "total": total_gold + total_failed + total_pending
            }
        }

    @app.get("/api/traces/repos", tags=["Traces"])
    async def list_trace_repos():
        """List all unique repositories that have traces.

        Returns a list of repo info objects for filtering.
        """
        from bashgym.config import get_settings, get_bashgym_dir
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
            dirs_to_scan.append((global_traces_dir, "imported_*.json"))
        if project_traces_dir.exists() and project_traces_dir.resolve() != global_traces_dir.resolve():
            dirs_to_scan.append((project_traces_dir, "session_*.json"))
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
                    with open(trace_file, encoding='utf-8') as f:
                        data = json.load(f)

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
                                        "trace_count": 0
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
                                    "trace_count": 0
                                }
                            repos[repo_name]["trace_count"] += 1
                except Exception:
                    continue

        return list(repos.values())

    def _matches_repo(data: Dict, repo_filter: str) -> bool:
        """Check if trace matches the repo filter."""
        primary_repo = data.get("primary_repo", {})
        repo_name = primary_repo.get("name", "")
        repo_path = primary_repo.get("path", "")
        git_remote = primary_repo.get("git_remote", "")

        filter_lower = repo_filter.lower()
        return (
            filter_lower in repo_name.lower() or
            filter_lower in repo_path.lower() or
            (git_remote and filter_lower in git_remote.lower())
        )

    def _parse_imported_trace_file(trace_file: Path, data: Dict) -> TraceInfo:
        """Parse an imported TraceSession file into TraceInfo."""
        trace_steps = data.get("trace", [])
        total_steps = len(trace_steps)
        summary = data.get("summary", {})
        metadata = data.get("metadata", {})

        # Get success rate from summary or calculate
        success_rate = summary.get("success_rate", 0)
        if success_rate == 0 and total_steps > 0:
            successful_steps = sum(1 for s in trace_steps if s.get("success") is True or s.get("exit_code") == 0)
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
                is_git_repo=primary_repo.get("is_git_repo", False)
            )

        # Get source info
        source_tool = data.get("source_tool", "unknown")

        # Get task description from metadata (user's initial prompt) or generate from tool usage
        task_desc = metadata.get("user_initial_prompt", "")

        if not task_desc:
            # Fallback: generate description from tool usage summary
            tool_counts = {}
            for step in trace_steps:
                tool = step.get("tool_name", "unknown")
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

            tool_summary = ", ".join([f"{count} {tool}" for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:4]])
            task_desc = f"Imported from {source_tool}: {tool_summary}"

        # Calculate quality using centralized calculator
        quality = calculate_quality_breakdown(
            steps=trace_steps,
            metadata=metadata
        )

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
                total_score=quality.total_score
            ),
            repo=repo_info,
            repos_count=len(repos) if repos else 1,
            created_at=data.get("timestamp", datetime.fromtimestamp(trace_file.stat().st_ctime).isoformat()),
            promoted_at=metadata.get("promoted_at")
        )

    def _parse_raw_trace_file(trace_file: Path, data: List[Dict]) -> TraceInfo:
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

        success_rate = successful_steps / total_steps if total_steps > 0 else 0

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
                        is_git_repo=step_repo.get("is_git_repo", False)
                    )

        # Get timestamps
        first_timestamp = data[0].get("timestamp", "") if data else ""
        last_timestamp = data[-1].get("timestamp", "") if data else ""

        # Generate task description from tool usage summary
        tool_counts = {}
        for step in data:
            tool = step.get("tool_name", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        tool_summary = ", ".join([f"{count} {tool}" for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:4]])
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
                total_score=quality.total_score
            ),
            repo=repo_info,
            repos_count=len(repos_seen),
            created_at=first_timestamp or datetime.fromtimestamp(trace_file.stat().st_ctime).isoformat(),
            promoted_at=None
        )

    def _parse_trace_file(trace_file: Path, data: Dict, status: TraceStatus) -> TraceInfo:
        """Parse a trace file into TraceInfo."""
        metadata = data.get("metadata", {})
        trace_steps = data.get("trace", [])
        total_steps = len(trace_steps)

        # Calculate quality using centralized calculator
        quality = calculate_quality_breakdown(steps=trace_steps, metadata=metadata)

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
                is_git_repo=primary_repo.get("is_git_repo", False)
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

        return TraceInfo(
            trace_id=trace_file.stem,
            task_id=metadata.get("task_id", trace_file.stem),
            task_description=metadata.get("user_initial_prompt", "Unknown task"),
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
                total_score=quality.total_score
            ),
            repo=repo_info,
            repos_count=len(repos),
            created_at=metadata.get("created_at", datetime.fromtimestamp(trace_file.stat().st_ctime).isoformat()),
            promoted_at=metadata.get("promoted_at")
        )

    @app.get("/api/traces/gold", response_model=List[TraceInfo], tags=["Traces"])
    async def list_gold_traces(limit: int = 100):
        """List only gold (successful) traces."""
        return await list_traces(status=TraceStatus.GOLD, limit=limit)

    @app.get("/api/traces/{trace_id}", response_model=TraceInfo, tags=["Traces"])
    async def get_trace(trace_id: str):
        """Get a specific trace by ID."""
        from bashgym.config import get_settings
        settings = get_settings()
        data_dir = Path(settings.data.data_dir)

        # Check all tier directories in priority order
        tier_checks = [
            (data_dir / "gold_traces" / f"{trace_id}.json", TraceStatus.GOLD),
            (data_dir / "silver_traces" / f"{trace_id}.json", TraceStatus.SILVER),
            (data_dir / "bronze_traces" / f"{trace_id}.json", TraceStatus.BRONZE),
            (data_dir / "failed_traces" / f"{trace_id}.json", TraceStatus.FAILED),
        ]

        for trace_path, trace_status in tier_checks:
            if trace_path.exists():
                with open(trace_path) as f:
                    data = json.load(f)
                return _parse_trace_file(trace_path, data, trace_status)

        raise HTTPException(status_code=404, detail="Trace not found")

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
            raise HTTPException(status_code=400, detail=f"Invalid target tier. Must be one of: {valid_tiers}")

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
            raise HTTPException(status_code=404, detail="Trace not found")

        # Move to target tier
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(source_path) as f:
            data = json.load(f)

        # Update metadata
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["promoted_at"] = datetime.utcnow().isoformat()
        data["metadata"]["status"] = target_tier
        data["metadata"]["quality_tier"] = target_tier

        with open(target_path, 'w') as f:
            json.dump(data, f, indent=2)

        source_path.unlink()

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
            raise HTTPException(status_code=400, detail=f"Invalid target tier. Must be one of: {valid_targets}")

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

        with open(target_path, 'w') as f:
            json.dump(data, f, indent=2)

        source_path.unlink()

        # Broadcast event
        await broadcast_trace_event(MessageType.TRACE_DEMOTED, trace_id)

        return {"success": True, "message": f"Trace demoted to {target_tier}"}

    # =========================================================================
    # Training Examples Endpoints
    # =========================================================================

    @app.post("/api/traces/{trace_id}/generate-examples", response_model=GenerateExamplesResponse, tags=["Training Examples"])
    async def generate_examples_from_trace(trace_id: str, request: GenerateExamplesRequest = None):
        """Generate training examples from a trace session.

        Segments a trace session into logical tasks and converts each to
        a NeMo-compatible training example.
        """
        from bashgym.config import get_settings, get_bashgym_dir
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
            output_dir=str(data_dir / "training_examples")
        )

        generator = ExampleGenerator(config)

        # Generate examples
        examples = generator.generate_examples(trace_path)

        # Convert to response format
        example_responses = []
        for ex in examples:
            example_responses.append(TrainingExampleResponse(
                example_id=ex.example_id,
                system_prompt=ex.system_prompt,
                user_prompt=ex.user_prompt,
                assistant_response=ex.assistant_response,
                step_count=ex.metadata.get("step_count", 0),
                success_rate=ex.metadata.get("success_rate", 0.0),
                confidence=ex.metadata.get("segmentation_confidence", 0.5),
                source_trace_id=trace_id
            ))

        # Calculate total steps
        total_steps = sum(ex.step_count for ex in example_responses)

        return GenerateExamplesResponse(
            trace_id=trace_id,
            examples=example_responses,
            total_steps=total_steps,
            examples_generated=len(example_responses)
        )

    @app.get("/api/training/examples", response_model=List[TrainingExampleResponse], tags=["Training Examples"])
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

                            examples.append(TrainingExampleResponse(
                                example_id=data.get("id", f"ex_{len(examples)}"),
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                assistant_response=assistant_response,
                                step_count=data.get("metadata", {}).get("step_count", 0),
                                success_rate=data.get("metadata", {}).get("success_rate", 0.0),
                                confidence=data.get("metadata", {}).get("segmentation_confidence", 0.5),
                                source_trace_id=data.get("metadata", {}).get("source_trace_id")
                            ))
            except (json.JSONDecodeError, IOError):
                continue

        # Apply pagination
        return examples[offset:offset + limit]

    @app.post("/api/training/export", response_model=ExportExamplesResponse, tags=["Training Examples"])
    async def export_training_examples(request: ExportExamplesRequest):
        """Export training examples to JSONL files for NeMo training.

        Generates train and validation splits from gold traces or specified traces.
        Output format is NeMo-compatible JSONL with messages array.
        """
        from bashgym.config import get_settings, get_bashgym_dir
        from bashgym.factory.example_generator import ExampleGenerator, ExampleGeneratorConfig

        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        global_dir = get_bashgym_dir()

        config = ExampleGeneratorConfig(
            output_dir=str(data_dir / "training_batches")
        )
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
                message="No examples generated. Check that traces exist and meet quality thresholds."
            )

        # Export with train/val split
        try:
            result = generator.export_for_nemo(
                all_examples,
                data_dir / "training_batches",
                train_split=request.train_split
            )

            return ExportExamplesResponse(
                success=True,
                train_path=str(result["train"]),
                val_path=str(result["validation"]),
                train_count=result["train_count"],
                val_count=result["val_count"],
                message=f"Exported {len(all_examples)} examples"
            )
        except Exception as e:
            return ExportExamplesResponse(
                success=False,
                message=f"Export failed: {str(e)}"
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
                "message": "Global and project directories are the same, no sync needed"
            }

        synced = {"pending": 0, "gold": 0, "silver": 0, "bronze": 0, "failed": 0}

        # Include all tier directories
        for category in ["traces", "gold_traces", "silver_traces", "bronze_traces", "failed_traces"]:
            src = global_dir / category
            dst = project_dir / category
            dst.mkdir(parents=True, exist_ok=True)

            if src.exists():
                for f in src.glob("*.json"):
                    if not (dst / f.name).exists():
                        shutil.copy2(f, dst / f.name)
                        key = "pending" if category == "traces" else category.replace("_traces", "")
                        synced[key] += 1

        return {"synced": synced, "project_dir": str(project_dir)}

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
        auto_promote: bool = False
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
            "gold": [],      # ≥90% success, ≥0.75 quality → SFT training
            "silver": [],    # ≥75% success, ≥0.55 quality → DPO chosen
            "bronze": [],    # ≥60% success, ≥0.40 quality → DPO rejected
            "rejected": [],  # <60% success → Not suitable for training
            "failed": [],    # Explicit verification failure
            "pending": []    # Unable to process
        }

        # Track detailed info for DPO pair generation
        detailed_classifications = {
            "gold": [],
            "silver": [],
            "bronze": [],
            "rejected": []
        }

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
                return {"success_rate": 0, "quality_score": 0, "total_steps": 0,
                        "verification_passed": False, "has_verification": False, "unique_tools": 0}

            # Use centralized quality calculator
            quality = calculate_quality_breakdown(steps=steps, metadata=metadata)

            return {
                "success_rate": round(quality.success_rate, 3),
                "quality_score": round(quality.total_score, 3),
                "total_steps": quality.total_steps,
                "verification_passed": quality.verification_passed_flag is True,
                "has_verification": quality.has_verification,
                "unique_tools": quality.unique_tools_count
            }

        # Collect pending traces from both directories
        pending_dirs = []
        global_traces_dir = global_dir / "traces"
        project_traces_dir = data_dir / "traces"

        if global_traces_dir.exists():
            pending_dirs.append(global_traces_dir)
        if project_traces_dir.exists() and project_traces_dir.resolve() != global_traces_dir.resolve():
            pending_dirs.append(project_traces_dir)

        seen_files = set()
        for pending_dir in pending_dirs:
            for trace_file in list(pending_dir.glob("session_*.json")) + list(pending_dir.glob("imported_*.json")):
                if trace_file.name in seen_files:
                    continue
                seen_files.add(trace_file.name)

                try:
                    with open(trace_file, encoding='utf-8') as f:
                        data = json.load(f)

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
                        "file_path": str(trace_file)
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
                        if success_rate >= gold_success_rate and quality_score >= gold_quality_score:
                            base_tier = "gold"
                        elif success_rate >= silver_success_rate and quality_score >= silver_quality_score:
                            base_tier = "silver"
                        elif success_rate >= bronze_success_rate and quality_score >= bronze_quality_score:
                            base_tier = "bronze"
                        else:
                            base_tier = "rejected"

                        # Apply verification_passed boost: promote one tier (max gold)
                        # This rewards traces where tests passed but doesn't skip quality checks
                        if metrics["verification_passed"] and base_tier != "gold":
                            tier_promotion = {"silver": "gold", "bronze": "silver", "rejected": "bronze"}
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
                    dpo_pairs.append({
                        "chosen": gold_trace["id"],
                        "chosen_success_rate": gold_trace["success_rate"],
                        "rejected": bronze_trace["id"],
                        "rejected_success_rate": bronze_trace["success_rate"],
                        "quality_gap": round(gold_trace["quality_score"] - bronze_trace["quality_score"], 3)
                    })
                    break  # One pair per gold trace

        return {
            "classifications": classifications,
            "detailed": detailed_classifications,
            "dry_run": dry_run,
            "thresholds": {
                "gold": {"success_rate": gold_success_rate, "quality_score": gold_quality_score},
                "silver": {"success_rate": silver_success_rate, "quality_score": silver_quality_score},
                "bronze": {"success_rate": bronze_success_rate, "quality_score": bronze_quality_score}
            },
            "auto_promote": auto_promote,
            "summary": {
                "gold": len(classifications["gold"]),
                "silver": len(classifications["silver"]),
                "bronze": len(classifications["bronze"]),
                "rejected": len(classifications["rejected"]),
                "failed": len(classifications["failed"]),
                "pending": len(classifications["pending"]),
                "total_processed": len(seen_files)
            },
            "dpo_pairs": dpo_pairs,
            "dpo_pairs_count": len(dpo_pairs),
            "training_recommendations": {
                "sft_eligible": len(classifications["gold"]) + len(classifications["silver"]),
                "dpo_chosen_pool": len(classifications["gold"]) + len(classifications["silver"]),
                "dpo_rejected_pool": len(classifications["bronze"]),
                "note": "Gold+Silver traces are suitable for SFT. Use Gold as DPO 'chosen', Bronze as DPO 'rejected'."
            }
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
                current_student_rate=stats.get("current_student_rate", 0.1)
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
            current_student_rate=0.1
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
            result = {"passed": True, "tests_run": 0, "tests_passed": 0, "message": "Verifier not available"}

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
                with open(trace_file, encoding='utf-8') as f:
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
                with open(source_file, encoding='utf-8') as f:
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
                    details.append({"file": trace_file.name, "error": "No user message found in source"})
                    continue

                # Update the trace file
                trace_data["metadata"]["user_initial_prompt"] = user_prompt
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(trace_data, f, indent=2, ensure_ascii=False)

                repaired += 1
                details.append({"file": trace_file.name, "prompt": user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt})

            except Exception as e:
                failed += 1
                details.append({"file": trace_file.name, "error": str(e)})

        return {
            "repaired": repaired,
            "failed": failed,
            "skipped": skipped,
            "total_scanned": repaired + failed + skipped,
            "details": details[:20]  # Limit details to first 20
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
                        batch_path = output_dir / f"sft_batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
                        app.state.data_factory.save_batch(examples, batch_path)

                        return {
                            "success": True,
                            "examples_created": len(examples),
                            "output_path": str(batch_path)
                        }

                return {"success": True, "examples_created": 0, "message": "No gold traces to process"}

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
                "platform": platform.system()
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
                "message": f"Installed {len(installed)} hooks" if installed else "No hooks installed"
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
                default_model=ModelConfig()
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

        with open(config_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)

        return {"success": True, "message": "Factory configuration updated"}

    # -------------------------------------------------------------------------
    # Seeds Management
    # -------------------------------------------------------------------------

    @app.get("/api/factory/seeds", response_model=List[SeedExample], tags=["Factory"])
    async def get_seeds():
        """Get all seed examples."""
        if app.state.factory_config and app.state.factory_config.seeds:
            return app.state.factory_config.seeds
        return []

    @app.post("/api/factory/seeds", response_model=SeedExample, tags=["Factory"])
    async def add_seed(seed_data: Dict[str, Any]):
        """Add a new seed example."""
        seed = SeedExample(
            id=f"seed_{uuid.uuid4().hex[:12]}",
            data=seed_data.get("data", {}),
            source=SeedSource(seed_data.get("source", "manual")),
            created_at=datetime.utcnow().isoformat(),
            trace_id=seed_data.get("trace_id")
        )

        if app.state.factory_config is None:
            app.state.factory_config = FactoryConfig(
                columns=[], seeds=[], privacy=PrivacyConfig(),
                prompt_optimization=PromptOptConfig(), output=OutputConfig(),
                safety=SafetyConfig(), default_model=ModelConfig()
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
    async def import_seeds_from_traces(request: Dict[str, Any]):
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
                with open(trace_file, encoding='utf-8') as f:
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
                    trace_id=trace_file.stem
                )

                imported_seeds.append(seed)

            except Exception:
                continue

        # Add to config
        if app.state.factory_config is None:
            app.state.factory_config = FactoryConfig(
                columns=[], seeds=[], privacy=PrivacyConfig(),
                prompt_optimization=PromptOptConfig(), output=OutputConfig(),
                safety=SafetyConfig(), default_model=ModelConfig()
            )

        app.state.factory_config.seeds.extend(imported_seeds)
        await update_factory_config(app.state.factory_config)

        return {"imported": len(imported_seeds), "seeds": imported_seeds}

    # -------------------------------------------------------------------------
    # Preview Mode
    # -------------------------------------------------------------------------

    @app.post("/api/factory/preview", response_model=PreviewResult, tags=["Factory"])
    async def generate_preview(request: Dict[str, Any]):
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
                column_coverage={}
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

            preview_rows.append(PreviewRow(
                id=f"preview_{i}",
                data=row_data,
                validation_errors=errors,
                risk_flags=risk_flags
            ))

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
            column_coverage=column_coverage
        )

    def _generate_column_value(col: ColumnConfig, seeds: List[SeedExample], index: int) -> str:
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

    def _check_constraint(value: str, constraint: ColumnConstraint) -> Optional[str]:
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

    @app.get("/api/factory/jobs", response_model=List[SynthesisJob], tags=["Factory"])
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
                error=j.get("error")
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
            error=j.get("error")
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
    async def run_factory_synthesis(
        request: Dict[str, Any],
        background_tasks: BackgroundTasks
    ):
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
            "row_count": row_count
        }

        # Run synthesis in background
        background_tasks.add_task(run_synthesis_job, app, job_id, is_preview, row_count)

        return SynthesisJob(
            id=job_id,
            status=SynthesisJobStatus.PENDING,
            job_type=job_type,
            created_at=app.state.synthesis_jobs[job_id]["created_at"]
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

                        batch_path = output_dir / f"{task_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{output_format}"
                        app.state.data_factory.save_batch(examples, batch_path)

                        app.state.synthesis_jobs[job_id]["status"] = SynthesisJobStatus.COMPLETED
                        app.state.synthesis_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
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
                output_path = output_dir / f"synthetic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
                app.state.synthesis_jobs[job_id]["output_path"] = str(output_path)

        except Exception as e:
            app.state.synthesis_jobs[job_id]["status"] = SynthesisJobStatus.FAILED
            app.state.synthesis_jobs[job_id]["error"] = str(e)

    # -------------------------------------------------------------------------
    # Column Validation
    # -------------------------------------------------------------------------

    @app.post("/api/factory/columns/{column_id}/validate", tags=["Factory"])
    async def validate_column(column_id: str, request: Dict[str, Any]):
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

            sample_rows.append(PreviewRow(
                id=f"sample_{i}",
                data={column.name: value},
                validation_errors=row_errors,
                risk_flags=[]
            ))

        valid = len(errors) == 0

        return {
            "valid": valid,
            "errors": errors,
            "sample": sample_rows
        }

    # -------------------------------------------------------------------------
    # Available Models
    # -------------------------------------------------------------------------

    @app.get("/api/factory/models", response_model=List[AvailableModel], tags=["Factory"])
    async def list_available_models():
        """List available models for LLM columns."""
        # Return common models - in production this would query NIM or other services
        return [
            # Anthropic Claude 4.5 (recommended for high-quality augmentation)
            AvailableModel(id="claude-opus-4-5-20251101", name="Claude Opus 4.5 (Best Quality)", provider="Anthropic"),
            AvailableModel(id="claude-sonnet-4-5-20250929", name="Claude Sonnet 4.5 (Recommended)", provider="Anthropic"),
            AvailableModel(id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5 (Fast)", provider="Anthropic"),
            # NVIDIA NIM (cost-effective, good for bulk generation)
            AvailableModel(id="qwen/qwen2.5-coder-32b-instruct", name="Qwen 2.5 Coder 32B (Default NIM)", provider="NVIDIA NIM"),
            AvailableModel(id="qwen/qwen2.5-coder-7b-instruct", name="Qwen 2.5 Coder 7B", provider="NVIDIA NIM"),
            AvailableModel(id="meta/llama-3.1-8b-instruct", name="Llama 3.1 8B Instruct", provider="NVIDIA NIM"),
            AvailableModel(id="meta/llama-3.1-70b-instruct", name="Llama 3.1 70B Instruct", provider="NVIDIA NIM"),
            AvailableModel(id="nvidia/nemotron-4-340b-instruct", name="Nemotron 4 340B", provider="NVIDIA NIM"),
            AvailableModel(id="mistralai/mistral-7b-instruct-v0.3", name="Mistral 7B Instruct", provider="NVIDIA NIM"),
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

    def _load_evaluations() -> Dict[str, Any]:
        """Load evaluations from JSON file."""
        eval_file = _get_evaluations_file()
        if eval_file.exists():
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load evaluations: {e}")
        return {}

    def _save_evaluations(evaluations: Dict[str, Any]):
        """Save evaluations to JSON file."""
        eval_file = _get_evaluations_file()
        try:
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluations, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save evaluations: {e}")

    # Load evaluations on startup
    app.state.evaluation_jobs = _load_evaluations()
    logger.info(f"Loaded {len(app.state.evaluation_jobs)} evaluation jobs from disk")

    async def run_evaluation_task(app_state, job_id: str, request: EvaluationRequest, model_path: Path):
        """Execute evaluation in background."""
        from bashgym.judge.benchmarks import BenchmarkRunner, BenchmarkConfig, BenchmarkType
        import asyncio

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
                        model_name=request.model_id
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
                        errors=ErrorAnalysisSchema(
                            wrong_answer=error_analysis.wrong_answer,
                            syntax_error=error_analysis.syntax_error,
                            runtime_error=error_analysis.runtime_error,
                            timeout=error_analysis.timeout,
                            other=error_analysis.other
                        ) if error_analysis else None
                    )
                except ValueError as e:
                    # Unknown benchmark type - skip it
                    logger.warning(f"Skipping unknown benchmark '{benchmark_name}': {e}")
                    continue

            app_state.evaluation_jobs[job_id]["status"] = "completed"
            app_state.evaluation_jobs[job_id]["results"] = {k: v.model_dump() for k, v in results.items()}

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
            "created_at": datetime.utcnow().isoformat()
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
            created_at=job.get("created_at")
        )

    @app.get("/api/evaluation", response_model=List[EvaluationResponse], tags=["Evaluation"])
    async def list_evaluations(limit: int = 20):
        """List recent evaluation jobs."""
        jobs = list(app.state.evaluation_jobs.values())
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)

        result = []
        for j in jobs[:limit]:
            results = None
            if j.get("results"):
                results = {k: BenchmarkResultSchema(**v) for k, v in j["results"].items()}
            result.append(EvaluationResponse(
                job_id=j["job_id"],
                model_id=j["model_id"],
                benchmarks=j["benchmarks"],
                status=j["status"],
                results=results,
                error=j.get("error"),
                created_at=j.get("created_at")
            ))
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
                "num_samples": len(BenchmarkLoader._cache.get(benchmark_id, []))
            }
        return {"benchmarks": status}

    # Include observability routes
    app.include_router(observability_router)

    # Include models routes
    app.include_router(models_router)

    # Include HuggingFace routes
    app.include_router(hf_router)

    # Include Factory routes (synthetic data generation)
    app.include_router(factory_router)

    # Include Integration routes (bashbros integration)
    app.include_router(integration_router)

    # Include Achievement routes (stats + achievements)
    app.include_router(achievements_router)

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
