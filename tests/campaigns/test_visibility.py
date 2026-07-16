import json
from datetime import UTC, datetime

from bashgym.campaigns.contracts import (
    CampaignEvent,
    CredentialKind,
    PublicCampaignEventSummaryV1,
    PublicCampaignEventV1,
)
from bashgym.campaigns.visibility import (
    PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES,
    PUBLIC_CAMPAIGN_EVENT_FIELDS,
    PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS,
    PUBLIC_EVENT_SUMMARY_FIELD_CLASSES,
    PUBLIC_EVENT_TYPE_FIELDS,
    project_public_campaign_event,
)


def raw_event(*, event_type: str, payload: dict) -> CampaignEvent:
    return CampaignEvent(
        event_id="event-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        sequence=1,
        aggregate_version=2,
        event_type=event_type,
        payload=payload,
        actor_id="campaign-controller",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id="caller-controlled-correlation-canary",
        idempotency_key="caller-controlled-idempotency-canary",
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )


def test_public_event_contract_has_an_exact_recursive_visibility_allowlist():
    assert set(PublicCampaignEventV1.model_fields) == PUBLIC_CAMPAIGN_EVENT_FIELDS
    assert set(PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES) == PUBLIC_CAMPAIGN_EVENT_FIELDS
    assert (
        set(PublicCampaignEventSummaryV1.model_fields)
        == PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS
    )
    assert (
        set(PUBLIC_EVENT_SUMMARY_FIELD_CLASSES)
        == PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS
    )
    assert set(PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES.values()) <= {
        "public_metadata",
        "workspace_safe",
    }
    assert set(PUBLIC_EVENT_SUMMARY_FIELD_CLASSES.values()) <= {
        "public_metadata",
        "workspace_safe",
    }
    classified_summary_fields = set().union(*PUBLIC_EVENT_TYPE_FIELDS.values())
    assert classified_summary_fields <= (
        PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS - {"schema_version"}
    )


def test_unknown_events_and_fields_fail_closed_to_safe_identity():
    projected = project_public_campaign_event(
        raw_event(
            event_type="campaign:future-protected-operation",
            payload={
                "ordinary": "future-mapping-canary",
                "location": "C:/operator/private.json",
                "nested": {"reference": ["future-result-canary"]},
            },
        )
    )

    assert projected.summary is None
    serialized = json.dumps(projected.model_dump(mode="json"), sort_keys=True)
    assert "future-mapping-canary" not in serialized
    assert "private.json" not in serialized
    assert "future-result-canary" not in serialized
    assert "caller-controlled-correlation-canary" not in serialized
    assert "caller-controlled-idempotency-canary" not in serialized


def test_known_event_keeps_only_registered_typed_summary_fields():
    projected = project_public_campaign_event(
        raw_event(
            event_type="campaign:action-blocked",
            payload={
                "study_id": "study-1",
                "stage_index": 2,
                "stage": "full_training",
                "code": "compute_capacity_unavailable",
                "message": "operator-error-canary",
                "location": "C:/operator/private.json",
                "result": {"candidate": "candidate-map-canary"},
            },
        )
    )

    assert projected.summary is not None
    assert projected.summary.model_dump(mode="json", exclude_none=True) == {
        "schema_version": "public_campaign_event_summary.v1",
        "study_id": "study-1",
        "stage": "full_training",
        "code": "compute_capacity_unavailable",
        "stage_index": 2,
    }
    serialized = json.dumps(projected.model_dump(mode="json"), sort_keys=True)
    assert "operator-error-canary" not in serialized
    assert "private.json" not in serialized
    assert "candidate-map-canary" not in serialized


def test_protected_event_types_never_expose_payload_shape_or_count():
    sparse = project_public_campaign_event(
        raw_event(
            event_type="campaign:protected-evaluation-completed",
            payload={"result": "one-protected-canary"},
        )
    )
    dense = project_public_campaign_event(
        raw_event(
            event_type="campaign:protected-evaluation-completed",
            payload={
                "reference": "protected-epoch-canary",
                "result": "candidate-map-canary",
                "rows": ["row-canary"] * 100,
            },
        )
    )

    assert sparse.summary is dense.summary is None
    assert set(sparse.model_dump(mode="json")) == set(dense.model_dump(mode="json"))
