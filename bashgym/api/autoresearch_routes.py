"""
AutoResearch API Routes

Provides REST endpoints for managing automated hyperparameter search
and trace-based data curation research:

Hyperparameter search:
- Start/stop/pause/resume the search loop
- Query status and experiment history

Trace research (data-centric):
- Start/stop/pause/resume the trace mining loop
- Query status and experiment history
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from bashgym.api.schemas import (
    AutoResearchRequest,
    AutoResearchStatusResponse,
    ExperimentResultSchema,
    SchemaResearchRequest,
    TraceExperimentResultSchema,
    TraceResearchRequest,
    TraceResearchStatusResponse,
)
from bashgym.api.websocket import WSMessage, manager
from bashgym.gym.autoresearch import (
    AutoResearchConfig,
    AutoResearcher,
    AutoResearchStatus,
    ExperimentResult,
)
from bashgym.gym.trace_researcher import (
    DataPipelineConfig,
    TraceExperimentResult,
    TraceResearchConfig,
    TraceResearcher,
    TraceResearchStatus,
)
from bashgym.gym.trainer import TrainerConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/autoresearch", tags=["autoresearch"])


# =============================================================================
# Helpers
# =============================================================================


def _get_researcher(request: Request) -> AutoResearcher | None:
    """Retrieve the active AutoResearcher from app state, if any."""
    return getattr(request.app.state, "autoresearcher", None)


async def _broadcast_experiment(
    result: ExperimentResult,
    best_config: TrainerConfig,
    best_metric: float,
    total_experiments: int,
    search_params: list,
):
    """Broadcast an experiment result via WebSocket."""
    message = WSMessage(
        type="autoresearch:experiment",
        payload={
            "experiment_id": result.experiment_id,
            "total_experiments": total_experiments,
            "config_snapshot": result.config_snapshot,
            "metric_value": result.metric_value,
            "best_metric": round(best_metric, 6),
            "improved": result.improved,
            "duration_seconds": result.duration_seconds,
        },
    )
    await manager.broadcast(message)


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/start")
async def start_autoresearch(body: AutoResearchRequest, request: Request):
    """Start an autoresearch hyperparameter search."""
    existing = _get_researcher(request)
    if existing and existing.status == AutoResearchStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="AutoResearch is already running. Stop it first.",
        )

    # Build AutoResearchConfig
    mode = getattr(body, "mode", "simulate") or "simulate"
    ar_config = AutoResearchConfig(
        search_params=body.search_params,
        max_experiments=body.max_experiments,
        train_steps=body.train_steps,
        dataset_subset_ratio=body.dataset_subset_ratio,
        eval_metric=body.eval_metric,
        mutation_rate=body.mutation_rate,
        mutation_scale=body.mutation_scale,
        mode=mode,
    )

    # Build base TrainerConfig from request overrides
    base_config = TrainerConfig()
    if body.base_config:
        for key, value in body.base_config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

    # Resolve dataset paths for real mode
    dataset_path = Path("data/gold_traces")
    val_dataset_path = None

    if mode == "real":
        # Generate training data from gold traces (same logic as run_training)
        try:
            from bashgym.config import get_settings
            from bashgym.factory.example_generator import (
                ExampleGenerator,
                ExampleGeneratorConfig,
            )

            settings = get_settings()
            gold_dir = Path(settings.data.gold_traces_dir)
            output_dir = Path(settings.data.training_batches_dir)

            if gold_dir.exists() and list(gold_dir.glob("*.json")):
                gen_config = ExampleGeneratorConfig(output_dir=str(output_dir))
                generator = ExampleGenerator(gen_config)
                examples, _stats = generator.process_directory(gold_dir)

                if examples:
                    result = generator.export_for_nemo(examples, output_dir, train_split=0.9)
                    dataset_path = Path(result["train"])
                    val_dataset_path = (
                        Path(result["validation"]) if result.get("validation") else None
                    )
                    logger.info(
                        f"[AutoResearch] Real mode: train={dataset_path}, val={val_dataset_path}"
                    )
                else:
                    logger.warning(
                        "[AutoResearch] No examples from gold traces, falling back to simulate"
                    )
                    ar_config.mode = "simulate"
            else:
                logger.warning("[AutoResearch] No gold traces found, falling back to simulate")
                ar_config.mode = "simulate"
        except Exception as e:
            logger.error(f"[AutoResearch] Failed to generate dataset for real mode: {e}")
            ar_config.mode = "simulate"

    researcher = AutoResearcher(
        ar_config,
        base_config,
        dataset_path=dataset_path,
        val_dataset_path=val_dataset_path,
    )
    request.app.state.autoresearcher = researcher

    # Callback that broadcasts each experiment via WebSocket
    total_experiments = ar_config.max_experiments
    search_params = ar_config.search_params

    async def on_experiment(result, best_config, best_metric):
        await _broadcast_experiment(
            result,
            best_config,
            best_metric,
            total_experiments,
            search_params,
        )

    # Launch the loop as a background asyncio task
    async def _run_wrapper():
        try:
            await researcher.run_loop(dataset_path, callback=on_experiment)
        except Exception as exc:
            logger.error(f"[AutoResearch] Background loop crashed: {exc}", exc_info=True)
            researcher.status = AutoResearchStatus.FAILED
            researcher._error = str(exc)

        # Broadcast completion/failure
        final_type = (
            "autoresearch:complete"
            if researcher.status in (AutoResearchStatus.COMPLETED, AutoResearchStatus.STOPPED)
            else "autoresearch:failed"
        )
        msg = WSMessage(
            type=final_type,
            payload={
                "status": researcher.status,
                "total_experiments": len(researcher.experiments),
                "best_metric": (
                    round(researcher.best_metric, 6)
                    if researcher.best_metric < float("inf")
                    else None
                ),
                "error": researcher._error,
            },
        )
        await manager.broadcast(msg)

    asyncio.create_task(_run_wrapper())

    logger.info(
        f"[AutoResearch] Started: {ar_config.max_experiments} experiments, "
        f"params={ar_config.search_params}"
    )

    return {
        "status": "started",
        "max_experiments": ar_config.max_experiments,
        "search_params": ar_config.search_params,
    }


@router.post("/stop")
async def stop_autoresearch(request: Request):
    """Stop the running autoresearch loop."""
    researcher = _get_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No autoresearch session found")

    if researcher.status not in (AutoResearchStatus.RUNNING, AutoResearchStatus.PAUSED):
        raise HTTPException(
            status_code=409,
            detail=f"AutoResearch is not running (status: {researcher.status})",
        )

    researcher.stop()
    logger.info("[AutoResearch] Stop requested")
    return {"status": "stopping", "completed_experiments": len(researcher.experiments)}


@router.post("/pause")
async def pause_autoresearch(request: Request):
    """Pause the running autoresearch loop."""
    researcher = _get_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No autoresearch session found")

    if researcher.status != AutoResearchStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"AutoResearch is not running (status: {researcher.status})",
        )

    researcher.pause()
    logger.info("[AutoResearch] Paused")
    return {"status": "paused", "completed_experiments": len(researcher.experiments)}


@router.post("/resume")
async def resume_autoresearch(request: Request):
    """Resume a paused autoresearch loop."""
    researcher = _get_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No autoresearch session found")

    if researcher.status != AutoResearchStatus.PAUSED:
        raise HTTPException(
            status_code=409,
            detail=f"AutoResearch is not paused (status: {researcher.status})",
        )

    researcher.resume()
    logger.info("[AutoResearch] Resumed")
    return {"status": "running", "completed_experiments": len(researcher.experiments)}


@router.get("/status", response_model=AutoResearchStatusResponse)
async def get_autoresearch_status(request: Request):
    """Get current autoresearch status, experiments, and best config."""
    researcher = _get_researcher(request)
    if not researcher:
        return AutoResearchStatusResponse(
            status=AutoResearchStatus.IDLE,
            total_experiments=0,
            completed_experiments=0,
            best_metric=None,
            best_config={},
            search_params=[],
            experiments=[],
            started_at=None,
            completed_at=None,
            error=None,
        )

    experiments = [
        ExperimentResultSchema(
            experiment_id=e.experiment_id,
            config_snapshot=e.config_snapshot,
            metric_value=e.metric_value,
            improved=e.improved,
            duration_seconds=e.duration_seconds,
            timestamp=e.timestamp,
        )
        for e in researcher.experiments
    ]

    return AutoResearchStatusResponse(
        status=researcher.status,
        total_experiments=researcher.config.max_experiments,
        completed_experiments=len(researcher.experiments),
        best_metric=(
            round(researcher.best_metric, 6) if researcher.best_metric < float("inf") else None
        ),
        best_config={
            param: getattr(researcher.best_config, param)
            for param in researcher.config.search_params
            if hasattr(researcher.best_config, param)
        },
        search_params=researcher.config.search_params,
        experiments=experiments,
        started_at=researcher._started_at,
        completed_at=researcher._completed_at,
        error=researcher._error,
    )


# =============================================================================
# Trace Research Helpers
# =============================================================================


def _get_trace_researcher(request: Request) -> TraceResearcher | None:
    """Retrieve the active TraceResearcher from app state, if any."""
    return getattr(request.app.state, "trace_researcher", None)


async def _broadcast_trace_experiment(
    result: TraceExperimentResult,
    best_pipeline: DataPipelineConfig,
    best_metric: float,
    best_data_stats: dict,
    total_experiments: int,
    search_params: list,
):
    """Broadcast a trace research experiment result via WebSocket."""
    message = WSMessage(
        type="autoresearch:trace-experiment",
        payload={
            "experiment_id": result.experiment_id,
            "total_experiments": total_experiments,
            "config_snapshot": result.config_snapshot,
            "examples_generated": result.examples_generated,
            "unique_repos": result.unique_repos,
            "avg_example_length": result.avg_example_length,
            "metric_value": result.metric_value,
            "best_metric": round(best_metric, 4),
            "improved": result.improved,
            "duration_seconds": result.duration_seconds,
        },
    )
    await manager.broadcast(message)


# =============================================================================
# Trace Research Endpoints
# =============================================================================


@router.post("/trace-research/start")
async def start_trace_research(body: TraceResearchRequest, request: Request):
    """Start a trace data research loop."""
    existing = _get_trace_researcher(request)
    if existing and existing.status == TraceResearchStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="Trace research is already running. Stop it first.",
        )

    # Build TraceResearchConfig
    mode = getattr(body, "mode", "simulate") or "simulate"
    tr_config = TraceResearchConfig(
        search_params=body.search_params,
        max_experiments=body.max_experiments,
        mutation_rate=body.mutation_rate,
        mutation_scale=body.mutation_scale,
        mode=mode,
        train_steps=body.train_steps,
    )

    # Resolve gold_traces_dir for real mode
    if mode == "real":
        try:
            from bashgym.config import get_settings

            settings = get_settings()
            gold_dir = Path(settings.data.gold_traces_dir)

            if gold_dir.exists() and list(gold_dir.glob("*.json")):
                tr_config.gold_traces_dir = str(gold_dir)
                logger.info(f"[TraceResearch] Real mode: gold_traces_dir={gold_dir}")
            else:
                logger.warning("[TraceResearch] No gold traces found, falling back to simulate")
                tr_config.mode = "simulate"
        except Exception as e:
            logger.error(f"[TraceResearch] Failed to resolve gold_traces_dir: {e}")
            tr_config.mode = "simulate"

    researcher = TraceResearcher(tr_config)
    request.app.state.trace_researcher = researcher

    # Callback that broadcasts each experiment via WebSocket
    total_experiments = tr_config.max_experiments
    search_params = tr_config.search_params

    async def on_experiment(result, best_pipeline, best_metric, best_data_stats):
        await _broadcast_trace_experiment(
            result,
            best_pipeline,
            best_metric,
            best_data_stats,
            total_experiments,
            search_params,
        )

    # Launch the loop as a background asyncio task
    async def _run_wrapper():
        try:
            await researcher.run_loop(callback=on_experiment)
        except Exception as exc:
            logger.error(f"[TraceResearch] Background loop crashed: {exc}", exc_info=True)
            researcher.status = TraceResearchStatus.FAILED
            researcher._error = str(exc)

        # Broadcast completion/failure
        final_type = (
            "autoresearch:trace-research-complete"
            if researcher.status in (TraceResearchStatus.COMPLETED, TraceResearchStatus.STOPPED)
            else "autoresearch:trace-research-failed"
        )
        msg = WSMessage(
            type=final_type,
            payload={
                "status": researcher.status,
                "total_experiments": len(researcher.experiments),
                "best_metric": (
                    round(researcher.best_metric, 4)
                    if researcher.best_metric < float("inf")
                    else None
                ),
                "best_data_stats": researcher.best_data_stats,
                "error": researcher._error,
            },
        )
        await manager.broadcast(msg)

    asyncio.create_task(_run_wrapper())

    logger.info(
        f"[TraceResearch] Started: {tr_config.max_experiments} experiments, "
        f"params={tr_config.search_params}"
    )

    return {
        "status": "started",
        "max_experiments": tr_config.max_experiments,
        "search_params": tr_config.search_params,
    }


@router.post("/trace-research/stop")
async def stop_trace_research(request: Request):
    """Stop the running trace research loop."""
    researcher = _get_trace_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No trace research session found")

    if researcher.status not in (TraceResearchStatus.RUNNING, TraceResearchStatus.PAUSED):
        raise HTTPException(
            status_code=409,
            detail=f"Trace research is not running (status: {researcher.status})",
        )

    researcher.stop()
    logger.info("[TraceResearch] Stop requested")
    return {"status": "stopping", "completed_experiments": len(researcher.experiments)}


@router.post("/trace-research/pause")
async def pause_trace_research(request: Request):
    """Pause the running trace research loop."""
    researcher = _get_trace_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No trace research session found")

    if researcher.status != TraceResearchStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Trace research is not running (status: {researcher.status})",
        )

    researcher.pause()
    logger.info("[TraceResearch] Paused")
    return {"status": "paused", "completed_experiments": len(researcher.experiments)}


@router.post("/trace-research/resume")
async def resume_trace_research(request: Request):
    """Resume a paused trace research loop."""
    researcher = _get_trace_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No trace research session found")

    if researcher.status != TraceResearchStatus.PAUSED:
        raise HTTPException(
            status_code=409,
            detail=f"Trace research is not paused (status: {researcher.status})",
        )

    researcher.resume()
    logger.info("[TraceResearch] Resumed")
    return {"status": "running", "completed_experiments": len(researcher.experiments)}


@router.get("/trace-research/status", response_model=TraceResearchStatusResponse)
async def get_trace_research_status(request: Request):
    """Get current trace research status, experiments, and best pipeline."""
    researcher = _get_trace_researcher(request)
    if not researcher:
        return TraceResearchStatusResponse(
            status=TraceResearchStatus.IDLE,
            total_experiments=0,
            completed_experiments=0,
            best_metric=None,
            best_pipeline={},
            best_data_stats={},
            search_params=[],
            experiments=[],
            started_at=None,
            completed_at=None,
            error=None,
        )

    experiments = [
        TraceExperimentResultSchema(
            experiment_id=e.experiment_id,
            config_snapshot=e.config_snapshot,
            examples_generated=e.examples_generated,
            unique_repos=e.unique_repos,
            avg_example_length=e.avg_example_length,
            metric_value=e.metric_value,
            improved=e.improved,
            duration_seconds=e.duration_seconds,
            timestamp=e.timestamp,
        )
        for e in researcher.experiments
    ]

    return TraceResearchStatusResponse(
        status=researcher.status,
        total_experiments=researcher.config.max_experiments,
        completed_experiments=len(researcher.experiments),
        best_metric=(
            round(researcher.best_metric, 4) if researcher.best_metric < float("inf") else None
        ),
        best_pipeline={
            p: getattr(researcher.best_pipeline, p) for p in researcher.config.search_params
        },
        best_data_stats=researcher.best_data_stats,
        search_params=researcher.config.search_params,
        experiments=experiments,
        started_at=researcher._started_at,
        completed_at=researcher._completed_at,
        error=researcher._error,
    )


# =============================================================================
# Schema Research Helpers
# =============================================================================


def _get_schema_researcher(request: Request) -> AutoResearcher | None:
    """Retrieve the active schema AutoResearcher from app state, if any."""
    return getattr(request.app.state, "schema_researcher", None)


# =============================================================================
# Schema Research Endpoints (Data Designer schema evolution)
# =============================================================================


@router.post("/schema-research/start")
async def start_schema_research(body: SchemaResearchRequest, request: Request):
    """Start schema research -- evolutionary search over Data Designer pipeline configs."""
    existing = _get_schema_researcher(request)
    if existing and existing.status == AutoResearchStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail="Schema research is already running. Stop it first.",
        )

    from bashgym.gym.schema_search_space import SchemaSearchSpace

    # Create search space
    search_space = SchemaSearchSpace(
        base_pipeline_name=body.base_template,
        mutation_rate=body.mutation_rate,
        mutation_scale=body.mutation_scale,
        stage1_examples=body.stage1_examples,
        stage1_judge_threshold=body.stage1_judge_threshold,
        stage2_train_steps=body.stage2_train_steps,
    )

    # Create AutoResearcher with schema search space
    ar_config = AutoResearchConfig(
        max_experiments=body.max_experiments,
        mode=body.mode,
        mutation_rate=body.mutation_rate,
        mutation_scale=body.mutation_scale,
    )

    # Use default genome as the "base config"
    base_genome = SchemaSearchSpace.create_default_genome(body.base_template)

    researcher = AutoResearcher(
        config=ar_config,
        base_trainer_config=base_genome,  # genome dict acts as config
        search_space=search_space,
    )
    request.app.state.schema_researcher = researcher

    # Callback that broadcasts each experiment via WebSocket
    total_experiments = ar_config.max_experiments

    async def on_experiment(result, best_config, best_metric):
        msg = WSMessage(
            type="schema-research:experiment",
            payload={
                "experiment_id": result.experiment_id,
                "total_experiments": total_experiments,
                "config_snapshot": result.config_snapshot,
                "metric_value": result.metric_value,
                "best_metric": round(best_metric, 6),
                "improved": result.improved,
                "duration_seconds": result.duration_seconds,
            },
        )
        await manager.broadcast(msg)

    # Launch the loop as a background asyncio task
    dataset_path = Path("data/gold_traces")  # placeholder

    async def _run_wrapper():
        try:
            await researcher.run_loop(dataset_path, callback=on_experiment)
        except Exception as exc:
            logger.error(f"[SchemaResearch] Background loop crashed: {exc}", exc_info=True)
            researcher.status = AutoResearchStatus.FAILED
            researcher._error = str(exc)

        # Broadcast completion/failure
        final_type = (
            "schema-research:complete"
            if researcher.status in (AutoResearchStatus.COMPLETED, AutoResearchStatus.STOPPED)
            else "schema-research:failed"
        )
        msg = WSMessage(
            type=final_type,
            payload={
                "status": researcher.status,
                "total_experiments": len(researcher.experiments),
                "best_metric": (
                    round(researcher.best_metric, 6)
                    if researcher.best_metric < float("inf")
                    else None
                ),
                "error": researcher._error,
            },
        )
        await manager.broadcast(msg)

    asyncio.create_task(_run_wrapper())

    # Broadcast status
    status_msg = WSMessage(
        type="schema-research:status",
        payload={
            "status": "running",
            "template": body.base_template,
            "max_experiments": ar_config.max_experiments,
        },
    )
    await manager.broadcast(status_msg)

    logger.info(
        f"[SchemaResearch] Started: {ar_config.max_experiments} experiments, "
        f"template={body.base_template}"
    )

    return {
        "status": "started",
        "template": body.base_template,
        "max_experiments": ar_config.max_experiments,
    }


@router.post("/schema-research/stop")
async def stop_schema_research(request: Request):
    """Stop the running schema research loop."""
    researcher = _get_schema_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No schema research session found")

    if researcher.status not in (AutoResearchStatus.RUNNING, AutoResearchStatus.PAUSED):
        raise HTTPException(
            status_code=409,
            detail=f"Schema research is not running (status: {researcher.status})",
        )

    researcher.stop()
    logger.info("[SchemaResearch] Stop requested")
    return {"status": "stopping", "completed_experiments": len(researcher.experiments)}


@router.post("/schema-research/pause")
async def pause_schema_research(request: Request):
    """Pause the running schema research loop."""
    researcher = _get_schema_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No schema research session found")

    if researcher.status != AutoResearchStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"Schema research is not running (status: {researcher.status})",
        )

    researcher.pause()
    logger.info("[SchemaResearch] Paused")
    return {"status": "paused", "completed_experiments": len(researcher.experiments)}


@router.post("/schema-research/resume")
async def resume_schema_research(request: Request):
    """Resume a paused schema research loop."""
    researcher = _get_schema_researcher(request)
    if not researcher:
        raise HTTPException(status_code=404, detail="No schema research session found")

    if researcher.status != AutoResearchStatus.PAUSED:
        raise HTTPException(
            status_code=409,
            detail=f"Schema research is not paused (status: {researcher.status})",
        )

    researcher.resume()
    logger.info("[SchemaResearch] Resumed")
    return {"status": "running", "completed_experiments": len(researcher.experiments)}


@router.get("/schema-research/status")
async def get_schema_research_status(request: Request):
    """Get schema research status and experiment history."""
    researcher = _get_schema_researcher(request)
    if not researcher:
        return {
            "status": "idle",
            "total_experiments": 0,
            "completed_experiments": 0,
            "best_metric": None,
            "best_config": {},
            "experiments": [],
        }
    return researcher.get_status()


@router.get("/schema-research/quality")
async def get_schema_research_quality(request: Request):
    """Get quality analytics for the current/last schema research run."""
    researcher = _get_schema_researcher(request)
    if not researcher or not researcher.experiments:
        return {
            "experiments_count": 0,
            "improvements": 0,
            "best_metric": None,
            "score_distribution": {},
            "template": None,
        }

    experiments = researcher.experiments
    improvements = sum(1 for e in experiments if e.improved)

    # Score distribution for quality dashboard
    scores = [e.metric_value for e in experiments]
    buckets: dict[str, int] = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    for s in scores:
        if s < 1.5:
            buckets["excellent"] += 1
        elif s < 2.0:
            buckets["good"] += 1
        elif s < 3.0:
            buckets["fair"] += 1
        else:
            buckets["poor"] += 1

    return {
        "experiments_count": len(experiments),
        "improvements": improvements,
        "best_metric": (
            round(researcher.best_metric, 6) if researcher.best_metric < float("inf") else None
        ),
        "score_distribution": buckets,
        "template": (
            getattr(researcher.search_space, "base_pipeline_name", None)
            if hasattr(researcher, "search_space")
            else None
        ),
    }
