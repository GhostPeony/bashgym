"""Central fail-closed visibility projections for durable campaign records."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from .contracts import (
    CampaignEvent,
    PublicCampaignEventSummaryV1,
    PublicCampaignEventV1,
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
        "correlation_identity",
        "idempotency_identity",
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
        "correlation_identity": "workspace_safe",
        "idempotency_identity": "workspace_safe",
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
        "source_id",
        "entry_id",
        "stage",
        "status",
        "code",
        "trigger",
        "outcome",
        "unit",
        "kind",
        "manifest_revision",
        "stage_index",
        "next_stage_index",
        "claim_generation",
        "cursor_end",
        "alert_count",
        "study_completed",
        "reserved",
        "actual",
        "effective_limit",
        "reason_codes",
        "metric_names",
        "evidence_ids",
        "artifact_ids",
    }
)

_IDENTITY_FIELDS = frozenset(
    {
        "action_id",
        "attempt_id",
        "study_id",
        "proposal_id",
        "source_id",
        "entry_id",
        "stage",
        "status",
        "code",
        "trigger",
        "outcome",
        "unit",
        "kind",
    }
)
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
_NUMBER_FIELDS = frozenset({"reserved", "actual", "effective_limit"})
_IDENTITY_LIST_FIELDS = frozenset(
    {"reason_codes", "metric_names", "evidence_ids", "artifact_ids"}
)
PUBLIC_EVENT_SUMMARY_FIELD_CLASSES = MappingProxyType(
    {
        "schema_version": "public_metadata",
        **{
            field: "workspace_safe"
            for field in (
                _IDENTITY_FIELDS
                | _INTEGER_FIELDS
                | _NUMBER_FIELDS
                | _IDENTITY_LIST_FIELDS
                | {"study_completed"}
            )
        },
    }
)


def _fields(*names: str) -> frozenset[str]:
    return frozenset(names)


PUBLIC_EVENT_TYPE_FIELDS = MappingProxyType(
    {
        "campaign:created": _fields(),
        "campaign:validation-started": _fields("trigger"),
        "campaign:ready": _fields("trigger"),
        "campaign:started": _fields("trigger"),
        "campaign:paused": _fields("trigger"),
        "campaign:resumed": _fields("trigger"),
        "campaign:authority-required": _fields("trigger"),
        "campaign:authority-satisfied": _fields("trigger"),
        "campaign:cancelling": _fields("trigger"),
        "campaign:cancelled": _fields("trigger"),
        "campaign:completed": _fields("trigger"),
        "campaign:failed": _fields("trigger"),
        "campaign:exhausted": _fields("trigger"),
        "campaign:proposal-submitted": _fields(
            "proposal_id", "status", "reason_codes"
        ),
        "campaign:proposal-rejected": _fields(
            "proposal_id", "status", "reason_codes"
        ),
        "campaign:proposal-withdrawn": _fields("proposal_id", "status"),
        "campaign:proposal-accepted": _fields("proposal_id", "study_id"),
        "campaign:advance-requested": _fields(),
        "campaign:manifest-revised": _fields("manifest_revision"),
        "campaign:source-approved": _fields("source_id"),
        "campaign:study-abandoned": _fields("study_id", "status"),
        "campaign:action-blocked": _fields(
            "study_id", "stage_index", "stage", "code"
        ),
        "campaign:stages-skipped": _fields(
            "study_id", "next_stage_index", "study_completed"
        ),
        "campaign:action-retry-scheduled": _fields(
            "action_id", "attempt_id", "study_id", "stage"
        ),
        "campaign:force-stop-requested": _fields("action_id", "attempt_id"),
        "campaign:training-metrics-appended": _fields(
            "action_id",
            "attempt_id",
            "cursor_end",
            "metric_names",
            "alert_count",
        ),
        "campaign:remote-run-registered": _fields(
            "action_id", "attempt_id", "claim_generation"
        ),
        "campaign:remote-run-adopted": _fields(
            "action_id", "attempt_id", "claim_generation"
        ),
        "campaign:remote-capacity-blocked": _fields(
            "action_id", "attempt_id", "claim_generation"
        ),
        "campaign:action-scheduled": _fields(
            "action_id", "attempt_id", "study_id", "stage"
        ),
        "campaign:action-claimed": _fields(
            "action_id", "attempt_id", "claim_generation"
        ),
        "campaign:action-unknown": _fields("action_id", "attempt_id"),
        "campaign:action-succeeded": _fields(
            "action_id", "attempt_id", "study_id", "stage", "outcome"
        ),
        "campaign:action-failed": _fields(
            "action_id", "attempt_id", "study_id", "stage", "outcome"
        ),
        "campaign:action-cancelled": _fields(
            "action_id", "attempt_id", "study_id", "stage", "outcome"
        ),
        "campaign:action-completed": _fields(
            "action_id", "attempt_id", "study_id", "stage", "outcome"
        ),
        "campaign:budget-recorded": _fields(
            "entry_id", "unit", "kind", "reserved", "actual", "effective_limit"
        ),
        "campaign:budget-overrun": _fields(
            "entry_id", "unit", "kind", "reserved", "actual", "effective_limit"
        ),
        # Presence is visible, but protected/candidate/result identities are not.
        "campaign:protected-lease-acquired": _fields(),
        "campaign:protected-evaluation-completed": _fields(),
        "campaign:promotion-committed": _fields(),
        "campaign:export-completed": _fields(),
    }
)


def _opaque_identity(value: Any, *, workspace_id: Any, campaign_id: Any) -> str:
    encoded = f"{workspace_id}\0{campaign_id}\0{value}".encode(
        "utf-8", errors="replace"
    )
    return hashlib.sha256(encoded).hexdigest()


def _safe_summary(event_type: str, payload: Mapping[str, Any]) -> PublicCampaignEventSummaryV1 | None:
    allowed = PUBLIC_EVENT_TYPE_FIELDS.get(event_type)
    if not allowed:
        return None
    projected: dict[str, Any] = {}
    for field in allowed:
        value = payload.get(field)
        if field in _IDENTITY_FIELDS:
            if isinstance(value, str):
                projected[field] = value
        elif field in _INTEGER_FIELDS:
            if isinstance(value, int) and not isinstance(value, bool):
                projected[field] = value
        elif field in _NUMBER_FIELDS:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                projected[field] = value
        elif field == "study_completed":
            if isinstance(value, bool):
                projected[field] = value
        elif field in _IDENTITY_LIST_FIELDS:
            if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                projected[field] = tuple(value[:100])
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

    correlation_identity = raw.get("correlation_identity")
    if not isinstance(correlation_identity, str) or len(correlation_identity) != 64:
        correlation_identity = _opaque_identity(
            raw.get("correlation_id", ""),
            workspace_id=raw.get("workspace_id", ""),
            campaign_id=raw.get("campaign_id", ""),
        )
    idempotency_identity = raw.get("idempotency_identity")
    if not isinstance(idempotency_identity, str) or len(idempotency_identity) != 64:
        idempotency_identity = _opaque_identity(
            raw.get("idempotency_key", ""),
            workspace_id=raw.get("workspace_id", ""),
            campaign_id=raw.get("campaign_id", ""),
        )

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
        correlation_identity=correlation_identity,
        idempotency_identity=idempotency_identity,
        created_at=raw["created_at"],
    )


__all__ = [
    "PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES",
    "PUBLIC_CAMPAIGN_EVENT_FIELDS",
    "PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS",
    "PUBLIC_EVENT_SUMMARY_FIELD_CLASSES",
    "PUBLIC_EVENT_TYPE_FIELDS",
    "project_public_campaign_event",
]
