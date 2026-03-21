"""
Typed event definitions for the Bash Gym EventBus.

Each event is a @dataclass with domain-specific fields and an `event_type`
class variable that maps to the corresponding MessageType string for
backward compatibility with the WebSocket wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# =============================================================================
# Pipeline Events
# =============================================================================


@dataclass
class TraceImported:
    """A trace was imported into the pipeline."""

    event_type: str = field(default="pipeline:import", init=False)
    trace_id: str = ""
    filename: str = ""
    source: str = ""
    steps: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TraceClassified:
    """A trace was classified (gold / failed / pending)."""

    event_type: str = field(default="pipeline:classified", init=False)
    trace_id: str = ""
    classification: str = ""  # "gold", "failed", "pending"
    quality_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThresholdReached:
    """Pipeline threshold was reached, triggering next stage."""

    event_type: str = field(default="pipeline:threshold_reached", init=False)
    gold_count: int = 0
    threshold: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PipelineStageStarted:
    """A pipeline stage has started."""

    event_type: str = field(default="pipeline:stage_started", init=False)
    stage: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Training Events
# =============================================================================


@dataclass
class TrainingStarted:
    """A training run has started."""

    event_type: str = field(default="training:started", init=False)
    run_id: str = ""
    strategy: str = ""
    base_model: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingProgress:
    """Training progress update (step/epoch)."""

    event_type: str = field(default="training:progress", init=False)
    run_id: str = ""
    epoch: float | None = None
    total_epochs: int | None = None
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float = 0.0
    eta: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingComplete:
    """Training run completed successfully."""

    event_type: str = field(default="training:complete", init=False)
    run_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingFailed:
    """Training run failed."""

    event_type: str = field(default="training:failed", init=False)
    run_id: str = ""
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingLog:
    """A raw training log line."""

    event_type: str = field(default="training:log", init=False)
    run_id: str = ""
    message: str = ""
    level: str = "info"
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# AutoResearch Events
# =============================================================================


@dataclass
class ExperimentStarted:
    """An auto-research experiment has started."""

    event_type: str = field(default="autoresearch:experiment:started", init=False)
    experiment_id: str = ""
    hypothesis: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExperimentCompleted:
    """An auto-research experiment has completed."""

    event_type: str = field(default="autoresearch:experiment:completed", init=False)
    experiment_id: str = ""
    result: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AutoResearchComplete:
    """All auto-research experiments have finished."""

    event_type: str = field(default="autoresearch:complete", init=False)
    total_experiments: int = 0
    successful: int = 0
    failed: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GoalProgressed:
    """Training goal progress was updated after an experiment."""

    event_type: str = field(default="autoresearch:goal:progressed", init=False)
    experiment_id: str = ""
    weighted_score: float = 0.0
    recommendation: str = ""  # "continue" | "adjust" | "complete"
    criteria_scores: dict[str, float] = field(default_factory=dict)
    constraints_status: dict[str, str] = field(default_factory=dict)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Judge Events
# =============================================================================


@dataclass
class JudgeVerdict:
    """Semantic judge has rendered a verdict on a solution."""

    event_type: str = field(default="judge:verdict", init=False)
    task_id: str = ""
    passed: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Orchestration Events
# =============================================================================


@dataclass
class TaskStarted:
    """An orchestration task worker was spawned."""

    event_type: str = field(default="orchestration:task:started", init=False)
    job_id: str = ""
    task_id: str = ""
    task_title: str = ""
    active_workers: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskCompleted:
    """An orchestration task completed successfully."""

    event_type: str = field(default="orchestration:task:completed", init=False)
    job_id: str = ""
    task_id: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    newly_unblocked: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskFailed:
    """An orchestration task failed."""

    event_type: str = field(default="orchestration:task:failed", init=False)
    job_id: str = ""
    task_id: str = ""
    error: str = ""
    will_retry: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrchestrationComplete:
    """Entire orchestration job finished."""

    event_type: str = field(default="orchestration:complete", init=False)
    job_id: str = ""
    completed: int = 0
    failed: int = 0
    total_cost_usd: float = 0.0
    total_time_seconds: float = 0.0
    merge_successes: int = 0
    merge_failures: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BudgetUpdate:
    """Budget status update during orchestration."""

    event_type: str = field(default="orchestration:budget:update", init=False)
    job_id: str = ""
    spent_usd: float = 0.0
    budget_usd: float = 0.0
    remaining_usd: float = 0.0
    tasks_completed: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Shared State Events
# =============================================================================


@dataclass
class SharedStateChanged:
    """A shared state key was written by a worker."""

    event_type: str = field(default="orchestration:state:changed", init=False)
    key: str = ""
    writer_id: str = ""
    value_preview: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Router Events
# =============================================================================


@dataclass
class RouterStatsUpdated:
    """Router statistics have been updated."""

    event_type: str = field(default="router:stats", init=False)
    stats: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# System / Verification Events
# =============================================================================


@dataclass
class VerificationResult:
    """Verification tests completed."""

    event_type: str = field(default="verification:result", init=False)
    task_id: str = ""
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskStatus:
    """Task status update."""

    event_type: str = field(default="task:status", init=False)
    task_id: str = ""
    status: str = ""
    result: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Trace Events (add/promote/demote)
# =============================================================================


@dataclass
class TraceAdded:
    """A trace was added."""

    event_type: str = field(default="trace:added", init=False)
    trace_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TracePromoted:
    """A trace was promoted to gold."""

    event_type: str = field(default="trace:promoted", init=False)
    trace_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TraceDemoted:
    """A trace was demoted to failed."""

    event_type: str = field(default="trace:demoted", init=False)
    trace_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Guardrail Events
# =============================================================================


@dataclass
class GuardrailBlocked:
    """Content was blocked by guardrails."""

    event_type: str = field(default="guardrail:blocked", init=False)
    check_type: str = ""
    location: str = ""
    action: str = "block"
    confidence: float = 0.0
    content_preview: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GuardrailWarn:
    """Content triggered a guardrail warning."""

    event_type: str = field(default="guardrail:warn", init=False)
    check_type: str = ""
    location: str = ""
    action: str = "warn"
    confidence: float = 0.0
    content_preview: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GuardrailPiiRedacted:
    """PII was redacted from content."""

    event_type: str = field(default="guardrail:pii_redacted", init=False)
    location: str = ""
    redaction_count: int = 0
    pii_types: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# HuggingFace Events
# =============================================================================


@dataclass
class HFJobStarted:
    """A HuggingFace training job has started."""

    event_type: str = field(default="hf:job:started", init=False)
    job_id: str = ""
    hardware: str = ""
    repo_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HFJobLog:
    """A HuggingFace job log line."""

    event_type: str = field(default="hf:job:log", init=False)
    job_id: str = ""
    log: str = ""
    level: str = "info"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HFJobCompleted:
    """A HuggingFace job completed successfully."""

    event_type: str = field(default="hf:job:completed", init=False)
    job_id: str = ""
    model_repo: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HFJobFailed:
    """A HuggingFace job failed."""

    event_type: str = field(default="hf:job:failed", init=False)
    job_id: str = ""
    error: str = ""
    logs: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HFJobMetrics:
    """Real-time training metrics from a HuggingFace cloud job."""

    event_type: str = field(default="hf:job:metrics", init=False)
    job_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HFSpaceReady:
    """A HuggingFace Space is ready."""

    event_type: str = field(default="hf:space:ready", init=False)
    space_name: str = ""
    url: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HFSpaceError:
    """A HuggingFace Space encountered an error."""

    event_type: str = field(default="hf:space:error", init=False)
    space_name: str = ""
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Integration Events (Bashbros)
# =============================================================================


@dataclass
class IntegrationTraceReceived:
    """A new trace was received from bashbros."""

    event_type: str = field(default="integration:trace:received", init=False)
    filename: str = ""
    task: str = ""
    source: str = "bashbros"
    steps: int = 0
    verified: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntegrationTraceProcessed:
    """A trace has been processed by bashbros integration."""

    event_type: str = field(default="integration:trace:processed", init=False)
    filename: str = ""
    success: bool = False
    trace_id: str | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntegrationModelExported:
    """A model was exported to GGUF for bashbros sidekick."""

    event_type: str = field(default="integration:model:exported", init=False)
    version: str = ""
    gguf_path: str = ""
    ollama_registered: bool = False
    traces_used: int = 0
    quality_avg: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntegrationModelRollback:
    """Model was rolled back to a previous version."""

    event_type: str = field(default="integration:model:rollback", init=False)
    version: str = ""
    previous_version: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntegrationLinked:
    """Bashbros integration was linked."""

    event_type: str = field(default="integration:linked", init=False)
    linked: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntegrationUnlinked:
    """Bashbros integration was unlinked."""

    event_type: str = field(default="integration:unlinked", init=False)
    linked: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IntegrationTrainingTriggered:
    """Auto-training was triggered by bashbros integration."""

    event_type: str = field(default="integration:training:triggered", init=False)
    gold_traces: int = 0
    threshold: int = 0
    run_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Prompt Evolution Events
# =============================================================================


@dataclass
class PromptEvolved:
    """A prompt variant was generated by the fast-loop evolver."""

    event_type: str = field(default="prompt:evolved", init=False)
    variant_id: str = ""
    generation: int = 0
    improvement_delta: float = 0.0
    patterns_addressed: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
