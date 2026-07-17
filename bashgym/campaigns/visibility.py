"""Central fail-closed visibility projections for durable campaign records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from .contracts import (
    CANONICAL_CAMPAIGN_EVENT_TYPES,
    PUBLIC_CAMPAIGN_ARTIFACT_SCHEMA_NAMES,
    PUBLIC_CAMPAIGN_BLOCKER_CODES,
    CampaignEvent,
    PublicCampaignArtifactV1,
    PublicCampaignAttemptV1,
    PublicCampaignEventSummaryV1,
    PublicCampaignEventV1,
    StageKind,
)

PUBLIC_CAMPAIGN_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "workspace_id",
        "campaign_id",
        "sequence",
        "aggregate_version",
        "event_type",
        "summary",
        "actor_id",
        "credential_kind",
        "created_at",
    }
)
PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES = MappingProxyType(
    {
        "schema_version": "public_metadata",
        "event_id": "public_metadata",
        "workspace_id": "workspace_safe",
        "campaign_id": "workspace_safe",
        "sequence": "workspace_safe",
        "aggregate_version": "workspace_safe",
        "event_type": "workspace_safe",
        "summary": "workspace_safe",
        "actor_id": "workspace_safe",
        "credential_kind": "workspace_safe",
        "created_at": "workspace_safe",
    }
)

PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "action_id",
        "attempt_id",
        "study_id",
        "proposal_id",
        "entry_id",
        "stage",
        "code",
        "manifest_revision",
        "stage_index",
        "next_stage_index",
        "claim_generation",
        "cursor_end",
        "alert_count",
        "study_completed",
    }
)

_IDENTITY_FIELDS = frozenset(
    {
        "action_id",
        "attempt_id",
        "study_id",
        "proposal_id",
        "entry_id",
    }
)
_ENUM_FIELDS = frozenset({"stage", "code"})
_INTEGER_FIELDS = frozenset(
    {
        "manifest_revision",
        "stage_index",
        "next_stage_index",
        "claim_generation",
        "cursor_end",
        "alert_count",
    }
)
PUBLIC_EVENT_SUMMARY_FIELD_CLASSES = MappingProxyType(
    {
        "schema_version": "public_metadata",
        **{
            field: "workspace_safe"
            for field in (_IDENTITY_FIELDS | _ENUM_FIELDS | _INTEGER_FIELDS | {"study_completed"})
        },
    }
)

PUBLIC_CAMPAIGN_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "workspace_id",
        "campaign_id",
        "artifact_id",
        "producer_action_id",
        "sha256",
        "size_bytes",
        "schema_name",
        "sealed",
        "valid",
        "created_at",
    }
)
PUBLIC_CAMPAIGN_ARTIFACT_FIELD_CLASSES = MappingProxyType(
    {
        "schema_version": "public_metadata",
        "workspace_id": "workspace_safe",
        "campaign_id": "workspace_safe",
        "artifact_id": "workspace_safe",
        "producer_action_id": "workspace_safe",
        "sha256": "workspace_safe",
        "size_bytes": "workspace_safe",
        "schema_name": "workspace_safe",
        "sealed": "workspace_safe",
        "valid": "workspace_safe",
        "created_at": "workspace_safe",
    }
)

PUBLIC_CAMPAIGN_ATTEMPT_FIELDS = frozenset(
    {
        "schema_version",
        "workspace_id",
        "campaign_id",
        "study_id",
        "action_id",
        "attempt_id",
        "attempt_number",
        "claim_generation",
        "status",
        "stage",
        "manifest_revision",
        "input_digest",
        "candidate_digest",
        "executor_kind",
        "created_at",
        "updated_at",
    }
)
PUBLIC_CAMPAIGN_ATTEMPT_FIELD_CLASSES = MappingProxyType(
    {
        "schema_version": "public_metadata",
        "workspace_id": "workspace_safe",
        "campaign_id": "workspace_safe",
        "study_id": "workspace_safe",
        "action_id": "workspace_safe",
        "attempt_id": "workspace_safe",
        "attempt_number": "workspace_safe",
        "claim_generation": "workspace_safe",
        "status": "workspace_safe",
        "stage": "workspace_safe",
        "manifest_revision": "workspace_safe",
        "input_digest": "workspace_safe",
        "candidate_digest": "workspace_safe",
        "executor_kind": "workspace_safe",
        "created_at": "workspace_safe",
        "updated_at": "workspace_safe",
    }
)


def _fields(*names: str) -> frozenset[str]:
    return frozenset(names)


PUBLIC_EVENT_TYPE_FIELDS = MappingProxyType(
    {
        "campaign:created": _fields(),
        "campaign:validation-started": _fields(),
        "campaign:validation-failed": _fields(),
        "campaign:ready": _fields(),
        "campaign:started": _fields(),
        "campaign:paused": _fields(),
        "campaign:resumed": _fields(),
        "campaign:authority-required": _fields(),
        "campaign:authority-satisfied": _fields(),
        "campaign:cancelling": _fields(),
        "campaign:cancelled": _fields(),
        "campaign:completed": _fields(),
        "campaign:failed": _fields(),
        "campaign:exhausted": _fields(),
        "campaign:proposal-submitted": _fields("proposal_id"),
        "campaign:proposal-rejected": _fields("proposal_id"),
        "campaign:proposal-withdrawn": _fields("proposal_id"),
        "campaign:proposal-accepted": _fields("proposal_id", "study_id"),
        "campaign:advance-requested": _fields(),
        "campaign:manifest-revised": _fields("manifest_revision"),
        "campaign:source-approved": _fields(),
        "campaign:study-abandoned": _fields("study_id"),
        "campaign:action-blocked": _fields("study_id", "stage_index", "stage", "code"),
        "campaign:stages-skipped": _fields("study_id", "next_stage_index", "study_completed"),
        "campaign:action-retry-scheduled": _fields("action_id", "attempt_id", "study_id", "stage"),
        "campaign:force-stop-requested": _fields("action_id", "attempt_id"),
        "campaign:training-metrics-appended": _fields(
            "action_id",
            "attempt_id",
            "cursor_end",
            "alert_count",
        ),
        "campaign:remote-run-registered": _fields("action_id", "attempt_id", "claim_generation"),
        "campaign:remote-run-adopted": _fields("action_id", "attempt_id", "claim_generation"),
        "campaign:remote-capacity-blocked": _fields("action_id", "attempt_id", "claim_generation"),
        "campaign:action-scheduled": _fields("action_id", "attempt_id", "study_id", "stage"),
        "campaign:action-claimed": _fields("action_id", "attempt_id", "claim_generation"),
        "campaign:action-unknown": _fields("action_id", "attempt_id"),
        "campaign:action-failed": _fields("action_id", "attempt_id", "study_id", "stage"),
        "campaign:action-cancelled": _fields("action_id", "attempt_id", "study_id", "stage"),
        "campaign:action-completed": _fields("action_id", "attempt_id", "study_id", "stage"),
        "campaign:action-force-stopped": _fields("action_id", "attempt_id", "study_id", "stage"),
        "campaign:budget-recorded": _fields("entry_id"),
        "campaign:budget-overrun": _fields("entry_id"),
        # Presence is visible, but protected/candidate/result identities are not.
        "campaign:protected-lease-acquired": _fields(),
        "campaign:protected-evaluation-completed": _fields(),
        "campaign:promotion-committed": _fields(),
        "campaign:export-completed": _fields(),
        # Human-review events expose presence only. Work, receipt, rationale, sample,
        # decision, and promotion bindings remain in the authenticated queue.
        "campaign:human-work-claimed": _fields(),
        "campaign:human-work-enqueued": _fields(),
        "campaign:human-work-submitted": _fields(),
        "campaign:human-promotion-held": _fields(),
        "campaign:human-promotion-approved": _fields(),
    }
)

if frozenset(PUBLIC_EVENT_TYPE_FIELDS) != CANONICAL_CAMPAIGN_EVENT_TYPES:
    raise RuntimeError("campaign public event registry is incomplete")


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_STAGES = frozenset(item.value for item in StageKind)


def _safe_identifier(value: Any) -> str | None:
    if isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value):
        return value
    return None


def _safe_summary(
    event_type: str, payload: Mapping[str, Any]
) -> PublicCampaignEventSummaryV1 | None:
    allowed = PUBLIC_EVENT_TYPE_FIELDS.get(event_type)
    if not allowed:
        return None
    projected: dict[str, Any] = {}
    for field in allowed:
        value = payload.get(field)
        if field in _IDENTITY_FIELDS:
            safe_value = _safe_identifier(value)
            if safe_value is not None:
                projected[field] = safe_value
        elif field in _ENUM_FIELDS:
            allowed_values = {
                "stage": _STAGES,
                "code": PUBLIC_CAMPAIGN_BLOCKER_CODES,
            }[field]
            if isinstance(value, str) and value in allowed_values:
                projected[field] = value
        elif field in _INTEGER_FIELDS:
            minimum = 1 if field == "manifest_revision" else 0
            if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
                projected[field] = value
        elif field == "study_completed":
            if isinstance(value, bool):
                projected[field] = value
    if not projected:
        return None
    try:
        return PublicCampaignEventSummaryV1.model_validate(projected)
    except ValidationError:
        return None


def project_public_campaign_event(
    event: CampaignEvent | Mapping[str, Any],
) -> PublicCampaignEventV1:
    """Project raw or untrusted event-shaped input through the public allowlist."""

    raw = event.model_dump(mode="json") if isinstance(event, CampaignEvent) else dict(event)
    event_type = str(raw.get("event_type", ""))
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        payload = raw.get("summary")
    if not isinstance(payload, Mapping):
        payload = {}

    return PublicCampaignEventV1(
        event_id=raw["event_id"],
        workspace_id=raw["workspace_id"],
        campaign_id=raw["campaign_id"],
        sequence=raw["sequence"],
        aggregate_version=raw["aggregate_version"],
        event_type=event_type,
        summary=_safe_summary(event_type, payload),
        actor_id=raw["actor_id"],
        credential_kind=raw["credential_kind"],
        created_at=raw["created_at"],
    )


def project_public_campaign_attempt(attempt: Any) -> PublicCampaignAttemptV1:
    """Project raw or untrusted attempt-shaped input without executor configuration."""

    raw = attempt.model_dump(mode="json") if hasattr(attempt, "model_dump") else dict(attempt)
    executor = raw.get("executor")
    executor_kind = executor.get("kind") if isinstance(executor, Mapping) else None
    return PublicCampaignAttemptV1(
        workspace_id=raw["workspace_id"],
        campaign_id=raw["campaign_id"],
        study_id=raw["study_id"],
        action_id=raw["action_id"],
        attempt_id=raw["attempt_id"],
        attempt_number=raw["attempt_number"],
        claim_generation=raw["claim_generation"],
        status=raw["status"],
        stage=raw["stage"],
        manifest_revision=raw["manifest_revision"],
        input_digest=raw["input_digest"],
        candidate_digest=raw["candidate_digest"],
        executor_kind=_safe_identifier(executor_kind),
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
    )


def project_public_campaign_artifact(artifact: Any) -> PublicCampaignArtifactV1:
    """Project raw or untrusted artifact-shaped input without URI or metadata."""

    raw = artifact.model_dump(mode="json") if hasattr(artifact, "model_dump") else dict(artifact)
    schema_name = raw.get("schema_name")
    if schema_name not in PUBLIC_CAMPAIGN_ARTIFACT_SCHEMA_NAMES:
        schema_name = "unclassified_artifact.v1"
    return PublicCampaignArtifactV1(
        workspace_id=raw["workspace_id"],
        campaign_id=raw["campaign_id"],
        artifact_id=raw["artifact_id"],
        producer_action_id=raw.get("producer_action_id"),
        sha256=raw["sha256"],
        size_bytes=raw["size_bytes"],
        schema_name=schema_name,
        sealed=raw["sealed"],
        valid=raw["valid"],
        created_at=raw["created_at"],
    )


__all__ = [
    "PUBLIC_CAMPAIGN_ARTIFACT_FIELD_CLASSES",
    "PUBLIC_CAMPAIGN_ARTIFACT_FIELDS",
    "PUBLIC_CAMPAIGN_ATTEMPT_FIELD_CLASSES",
    "PUBLIC_CAMPAIGN_ATTEMPT_FIELDS",
    "PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES",
    "PUBLIC_CAMPAIGN_EVENT_FIELDS",
    "PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS",
    "PUBLIC_EVENT_SUMMARY_FIELD_CLASSES",
    "PUBLIC_EVENT_TYPE_FIELDS",
    "project_public_campaign_artifact",
    "project_public_campaign_attempt",
    "project_public_campaign_event",
]
