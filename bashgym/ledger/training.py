"""Training lifecycle adapter from direct BashGym runs into the experiment ledger."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bashgym.campaigns.contracts import DESKTOP_LOCAL_SCOPE, canonical_hash, utc_now
from bashgym.ledger.contracts import (
    UNASSIGNED_EXPERIMENT_ID,
    UNASSIGNED_PROJECT_ID,
    AttemptSpec,
    ContextStatus,
    DatasetSpec,
    DatasetVersionSpec,
    EnvironmentSpec,
    ExperimentSpec,
    LedgerEventSpec,
    MetricPointSpec,
    ModelSpec,
    ModelVersionSpec,
    ProjectSpec,
    RunSpec,
    RunStatus,
    ensure_safe_payload,
    stable_ledger_id,
)
from bashgym.ledger.persistence import ExperimentLedgerRepository


@dataclass(frozen=True)
class TrainingLedgerHandle:
    workspace_id: str
    project_id: str
    experiment_id: str
    run_id: str
    attempt_id: str
    correlation_id: str
    verified: bool

    def as_dict(self) -> dict[str, str]:
        return {
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "context_status": "verified" if self.verified else "unassigned",
        }


def _safe_training_config(payload: dict[str, Any]) -> dict[str, Any]:
    excluded = {"origin", "tracking"}
    safe = {key: value for key, value in payload.items() if key not in excluded}
    return ensure_safe_payload(safe, field_name="training config")


def prepare_training_run(
    repository: ExperimentLedgerRepository,
    *,
    run_id: str,
    request_payload: dict[str, Any],
    is_simulation: bool,
) -> TrainingLedgerHandle:
    """Register project lineage, a queued run, attempt, and durable event."""

    tracking = request_payload.get("tracking")
    correlation_id = str(request_payload.get("correlation_id") or f"training:{run_id}")
    strategy = str(request_payload.get("strategy") or "unknown")
    if tracking:
        workspace_id = str(tracking["workspace_id"])
        project_id = str(tracking["project_id"])
        experiment_id = str(tracking["experiment_id"])
        verified = True
        repository.register_project(
            ProjectSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                display_name=str(tracking["project_display_name"]),
                description=str(tracking.get("project_description") or ""),
                owner_actor_id=str(tracking.get("owner_actor_id") or "bashgym"),
                tags=tuple(sorted(set(tracking.get("tags") or ()))),
            )
        )
        repository.register_experiment(
            ExperimentSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                experiment_id=experiment_id,
                name=str(tracking["experiment_name"]),
                objective=str(tracking["objective"]),
                campaign_id=tracking.get("campaign_id"),
                metadata=dict(tracking.get("metadata") or {}),
            )
        )
        repository.register_model(
            ModelSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                model_id=str(tracking["model_id"]),
                display_name=str(request_payload.get("base_model") or tracking["model_id"]),
                task_type=str(tracking["task_type"]),
                architecture=str(request_payload.get("model_type") or ""),
            )
        )
        repository.register_model_version(
            ModelVersionSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                model_id=str(tracking["model_id"]),
                model_version_id=str(tracking["model_version_id"]),
                source_uri=str(tracking["model_source_uri"]),
                source_revision=str(tracking.get("model_source_revision") or ""),
                config_digest=str(tracking["model_config_digest"]),
            )
        )
        repository.register_dataset(
            DatasetSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                dataset_id=str(tracking["dataset_id"]),
                display_name=str(tracking["dataset_id"]),
                task_type=str(tracking["task_type"]),
            )
        )
        repository.register_dataset_version(
            DatasetVersionSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                dataset_id=str(tracking["dataset_id"]),
                dataset_version_id=str(tracking["dataset_version_id"]),
                source_uri=str(tracking["dataset_source_uri"]),
                content_digest=str(tracking["dataset_content_digest"]),
                split_manifest=dict(tracking.get("dataset_split_manifest") or {}),
                row_counts=dict(tracking.get("dataset_row_counts") or {}),
            )
        )
        repository.register_environment(
            EnvironmentSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                environment_id=str(tracking["environment_id"]),
                compute_target=str(request_payload.get("compute_target") or "local"),
                runtime_digest=str(tracking["environment_runtime_digest"]),
                hardware=dict(tracking.get("environment_hardware") or {}),
            )
        )
        model_version_id = str(tracking["model_version_id"])
        dataset_version_id = str(tracking["dataset_version_id"])
        environment_id = str(tracking["environment_id"])
        task_type = str(tracking["task_type"])
        campaign_id = tracking.get("campaign_id")
        study_id = tracking.get("study_id")
        action_id = tracking.get("action_id")
    else:
        workspace_id = DESKTOP_LOCAL_SCOPE
        project_id = UNASSIGNED_PROJECT_ID
        experiment_id = UNASSIGNED_EXPERIMENT_ID
        verified = False
        repository.register_project(
            ProjectSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                display_name="Unassigned training runs",
                description=(
                    "Legacy or ad-hoc runs retained without guessing project, model, dataset, "
                    "or environment lineage. Assign context before using them for decisions."
                ),
                owner_actor_id="bashgym",
                tags=("needs-context",),
            )
        )
        repository.register_experiment(
            ExperimentSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                experiment_id=experiment_id,
                name="Unassigned direct training",
                objective="Retain run evidence until verified project context is supplied.",
            )
        )
        model_version_id = None
        dataset_version_id = None
        environment_id = None
        task_type = "unknown"
        campaign_id = None
        study_id = None
        action_id = None

    safe_config = _safe_training_config(request_payload)
    attempt_id = stable_ledger_id("attempt", workspace_id, project_id, run_id, 1)
    repository.register_run(
        RunSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run_id,
            source_system="bashgym-direct-training",
            source_run_id=run_id,
            campaign_id=campaign_id,
            study_id=study_id,
            action_id=action_id,
            run_kind="training",
            task_type=task_type,
            training_method=strategy,
            context_status=ContextStatus.VERIFIED if verified else ContextStatus.UNASSIGNED,
            model_version_id=model_version_id,
            dataset_version_id=dataset_version_id,
            environment_id=environment_id,
            recipe_digest=canonical_hash(safe_config),
            config=safe_config,
            correlation_id=correlation_id,
            is_simulation=is_simulation,
        )
    )
    repository.register_attempt(
        AttemptSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            run_id=run_id,
            attempt_id=attempt_id,
            attempt_number=1,
            source_attempt_id=run_id,
            metadata={"executor": "simulation" if is_simulation else "trainer"},
        )
    )
    repository.append_event(
        LedgerEventSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type="training-queued",
            source_system="bashgym-direct-training",
            source_event_id=f"{run_id}:queued",
            correlation_id=correlation_id,
            payload={
                "strategy": strategy,
                "context_status": "verified" if verified else "unassigned",
                "simulation": is_simulation,
            },
        )
    )
    return TrainingLedgerHandle(
        workspace_id=workspace_id,
        project_id=project_id,
        experiment_id=experiment_id,
        run_id=run_id,
        attempt_id=attempt_id,
        correlation_id=correlation_id,
        verified=verified,
    )


def mark_training_running(
    repository: ExperimentLedgerRepository, handle: TrainingLedgerHandle
) -> None:
    moment = utc_now()
    repository.transition_run(
        handle.workspace_id, handle.project_id, handle.run_id, RunStatus.RUNNING, at=moment
    )
    repository.transition_attempt(
        handle.workspace_id, handle.project_id, handle.attempt_id, RunStatus.RUNNING, at=moment
    )
    repository.append_event(
        LedgerEventSpec(
            workspace_id=handle.workspace_id,
            project_id=handle.project_id,
            experiment_id=handle.experiment_id,
            run_id=handle.run_id,
            attempt_id=handle.attempt_id,
            event_type="training-started",
            source_system="bashgym-direct-training",
            source_event_id=f"{handle.run_id}:started",
            correlation_id=handle.correlation_id,
        )
    )


_METRIC_NAMES = {
    "loss": "train.loss",
    "learning_rate": "train.learning_rate",
    "grad_norm": "train.grad_norm",
    "eval_loss": "eval.loss",
    "tokens_per_second": "system.tokens_per_second",
    "gpu_memory_gb": "system.gpu_memory_gb",
    "gpu_utilization": "system.gpu_utilization",
    "session_distillation_loss": "train.session_distillation_loss",
    "session_distillation_kl": "train.session_distillation_kl",
    "session_distillation_ce": "train.session_distillation_ce",
    "session_distillation_masked_tokens": "train.session_distillation_masked_tokens",
}


def record_training_progress(
    repository: ExperimentLedgerRepository,
    handle: TrainingLedgerHandle,
    metrics: dict[str, Any],
) -> int:
    """Persist canonical numeric metrics from one progress callback."""

    step = metrics.get("step")
    if not isinstance(step, int) or step < 0:
        return 0
    context = {
        key: metrics[key]
        for key in ("epoch", "total_epochs", "total_steps", "simulation")
        if metrics.get(key) is not None
    }
    written = 0
    for source_name, metric_name in _METRIC_NAMES.items():
        value = metrics.get(source_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        if repository.append_metric(
            MetricPointSpec(
                workspace_id=handle.workspace_id,
                project_id=handle.project_id,
                run_id=handle.run_id,
                attempt_id=handle.attempt_id,
                source="training-callback",
                step=step,
                metric_name=metric_name,
                metric_value=numeric,
                raw_sha256=canonical_hash(
                    {
                        "step": step,
                        "source_name": source_name,
                        "value": numeric,
                        "context": context,
                    }
                ),
                context=context,
            )
        ):
            written += 1
    return written


def finalize_training_run(
    repository: ExperimentLedgerRepository,
    handle: TrainingLedgerHandle,
    *,
    status: RunStatus,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
        raise ValueError("training final status must be terminal")
    moment = utc_now()
    repository.transition_attempt(
        handle.workspace_id, handle.project_id, handle.attempt_id, status, at=moment
    )
    repository.transition_run(
        handle.workspace_id, handle.project_id, handle.run_id, status, at=moment
    )
    numeric_metrics = {
        key: float(value)
        for key, value in (metrics or {}).items()
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    }
    repository.append_event(
        LedgerEventSpec(
            workspace_id=handle.workspace_id,
            project_id=handle.project_id,
            experiment_id=handle.experiment_id,
            run_id=handle.run_id,
            attempt_id=handle.attempt_id,
            event_type=f"training-{status.value}",
            source_system="bashgym-direct-training",
            source_event_id=f"{handle.run_id}:{status.value}",
            correlation_id=handle.correlation_id,
            payload={
                "final_metrics": dict(sorted(numeric_metrics.items())[:50]),
                "error_summary": (error or "")[:2000],
            },
        )
    )
