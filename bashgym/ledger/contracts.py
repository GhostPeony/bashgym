"""Stable, secret-free contracts for the project-isolated experiment ledger."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from bashgym.campaigns.contracts import (
    ContractModel,
    HexDigest,
    Identifier,
    canonical_hash,
    utc_now,
)

UNASSIGNED_PROJECT_ID = "unassigned"
UNASSIGNED_EXPERIMENT_ID = "unassigned"


class LedgerStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContextStatus(str, Enum):
    VERIFIED = "verified"
    UNASSIGNED = "unassigned"
    STALE = "stale"


def stable_ledger_id(prefix: str, *parts: Any) -> str:
    """Return a readable deterministic ID for an immutable source identity."""

    digest = canonical_hash(list(parts))[:24]
    safe_prefix = re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-") or "item"
    return f"{safe_prefix}_{digest}"


_FORBIDDEN_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "credential",
        "credentials",
        "dataset_rows",
        "examples",
        "password",
        "private_key",
        "raw_data",
        "raw_dataset",
        "refresh_token",
        "secret",
        "token",
    }
)


def ensure_safe_payload(value: Any, *, field_name: str = "metadata") -> Any:
    """Reject secret-shaped or raw-data-shaped content before durable storage."""

    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in _FORBIDDEN_KEYS or normalized.endswith("_password"):
                    raise ValueError(f"{field_name} contains forbidden key at {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, field_name)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > 131_072:
        raise ValueError(f"{field_name} exceeds the 128 KiB ledger limit")
    return value


def payload_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class SafeContract(ContractModel):
    @field_validator("metadata", "config", "payload", "hardware", check_fields=False)
    @classmethod
    def validate_safe_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_safe_payload(value)


class ProjectSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    display_name: str = Field(min_length=1, max_length=240)
    description: str = Field(default="", max_length=4000)
    status: LedgerStatus = LedgerStatus.ACTIVE
    owner_actor_id: Identifier
    tags: tuple[Identifier, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


class ExperimentSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    experiment_id: Identifier
    name: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=8000)
    status: LedgerStatus = LedgerStatus.ACTIVE
    campaign_id: Identifier | None = None
    parent_experiment_id: Identifier | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ModelSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    model_id: Identifier
    display_name: str = Field(min_length=1, max_length=240)
    task_type: Identifier
    architecture: str = Field(default="", max_length=240)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ModelVersionSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    model_id: Identifier
    model_version_id: Identifier
    source_uri: str = Field(min_length=1, max_length=2000)
    source_revision: str = Field(default="", max_length=500)
    parent_model_version_id: Identifier | None = None
    config_digest: HexDigest
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DatasetSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    dataset_id: Identifier
    display_name: str = Field(min_length=1, max_length=240)
    task_type: Identifier
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DatasetVersionSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    dataset_id: Identifier
    dataset_version_id: Identifier
    source_uri: str = Field(min_length=1, max_length=2000)
    content_digest: HexDigest
    split_manifest: dict[str, Any] = Field(default_factory=dict)
    row_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("split_manifest", "row_counts")
    @classmethod
    def validate_dataset_summary(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_safe_payload(value, field_name="dataset summary")


class EnvironmentSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    environment_id: Identifier
    compute_target: str = Field(min_length=1, max_length=500)
    runtime_digest: HexDigest
    hardware: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RunSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    experiment_id: Identifier
    run_id: Identifier
    source_system: Identifier
    source_run_id: Identifier
    campaign_id: Identifier | None = None
    study_id: Identifier | None = None
    action_id: Identifier | None = None
    run_kind: Identifier
    task_type: Identifier
    training_method: Identifier
    status: RunStatus = RunStatus.QUEUED
    context_status: ContextStatus = ContextStatus.VERIFIED
    model_version_id: Identifier | None = None
    dataset_version_id: Identifier | None = None
    environment_id: Identifier | None = None
    recipe_digest: HexDigest
    config: dict[str, Any] = Field(default_factory=dict)
    correlation_id: Identifier
    is_simulation: bool = False
    queued_at: datetime = Field(default_factory=utc_now)


class AttemptSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    run_id: Identifier
    attempt_id: Identifier
    attempt_number: int = Field(ge=1)
    source_attempt_id: Identifier | None = None
    status: RunStatus = RunStatus.QUEUED
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MetricPointSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    run_id: Identifier
    attempt_id: Identifier
    source: Identifier
    step: int = Field(ge=0)
    metric_name: Identifier
    metric_value: float
    raw_sha256: HexDigest
    context: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)

    @field_validator("context")
    @classmethod
    def validate_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_safe_payload(value, field_name="metric context")


class ArtifactSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    artifact_id: Identifier
    run_id: Identifier
    attempt_id: Identifier | None = None
    kind: Identifier
    uri: str = Field(min_length=1, max_length=2000)
    sha256: HexDigest
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=240)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationSuiteSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    evaluation_suite_id: Identifier
    name: str = Field(min_length=1, max_length=240)
    task_type: Identifier
    dataset_version_id: Identifier | None = None
    metric_contract: dict[str, Any]
    code_digest: HexDigest
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("metric_contract")
    @classmethod
    def validate_metric_contract(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_safe_payload(value, field_name="metric contract")


class EvaluationResultSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    evaluation_result_id: Identifier
    evaluation_suite_id: Identifier
    run_id: Identifier
    attempt_id: Identifier | None = None
    model_version_id: Identifier | None = None
    status: RunStatus
    metrics: dict[str, float]
    slice_metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_id: Identifier | None = None
    compared_to_result_id: Identifier | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @field_validator("slice_metrics")
    @classmethod
    def validate_slice_metrics(cls, value: dict[str, Any]) -> dict[str, Any]:
        return ensure_safe_payload(value, field_name="slice metrics")


class DecisionSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    decision_id: Identifier
    experiment_id: Identifier
    run_id: Identifier | None = None
    decision_type: Identifier
    outcome: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=12000)
    evidence_refs: tuple[str, ...] = ()
    actor_id: Identifier
    created_at: datetime = Field(default_factory=utc_now)


class LedgerEventSpec(SafeContract):
    workspace_id: Identifier
    project_id: Identifier
    event_type: Identifier
    source_system: Identifier
    source_event_id: Identifier
    correlation_id: Identifier
    experiment_id: Identifier | None = None
    run_id: Identifier | None = None
    attempt_id: Identifier | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
