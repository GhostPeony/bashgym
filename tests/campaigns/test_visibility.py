import hashlib
import json
from datetime import datetime

from bashgym._compat import UTC
from bashgym.campaigns.contracts import (
    CANONICAL_CAMPAIGN_EVENT_TYPES,
    CampaignEvent,
    CredentialKind,
    PublicCampaignArtifactV1,
    PublicCampaignEventSummaryV1,
    PublicCampaignEventV1,
)
from bashgym.campaigns.visibility import (
    PUBLIC_CAMPAIGN_ARTIFACT_FIELD_CLASSES,
    PUBLIC_CAMPAIGN_ARTIFACT_FIELDS,
    PUBLIC_CAMPAIGN_ATTEMPT_FIELDS,
    PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES,
    PUBLIC_CAMPAIGN_EVENT_FIELDS,
    PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS,
    PUBLIC_EVENT_SUMMARY_FIELD_CLASSES,
    PUBLIC_EVENT_TYPE_FIELDS,
    project_public_campaign_artifact,
    project_public_campaign_attempt,
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
    assert set(PublicCampaignEventSummaryV1.model_fields) == PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS
    assert set(PUBLIC_EVENT_SUMMARY_FIELD_CLASSES) == PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS
    assert set(PUBLIC_CAMPAIGN_EVENT_FIELD_CLASSES.values()) <= {
        "public_metadata",
        "workspace_safe",
    }
    assert set(PUBLIC_EVENT_SUMMARY_FIELD_CLASSES.values()) <= {
        "public_metadata",
        "workspace_safe",
    }
    classified_summary_fields = set().union(*PUBLIC_EVENT_TYPE_FIELDS.values())
    assert classified_summary_fields <= (PUBLIC_EVENT_SUMMARY_CONTRACT_FIELDS - {"schema_version"})
    assert set(PUBLIC_EVENT_TYPE_FIELDS) == CANONICAL_CAMPAIGN_EVENT_TYPES
    assert "correlation_identity" not in PUBLIC_CAMPAIGN_EVENT_FIELDS
    assert "idempotency_identity" not in PUBLIC_CAMPAIGN_EVENT_FIELDS
    assert set(PublicCampaignArtifactV1.model_fields) == PUBLIC_CAMPAIGN_ARTIFACT_FIELDS
    assert set(PUBLIC_CAMPAIGN_ARTIFACT_FIELD_CLASSES) == PUBLIC_CAMPAIGN_ARTIFACT_FIELDS


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
    assert "correlation_identity" not in serialized
    assert "idempotency_identity" not in serialized
    guessed = hashlib.sha256(
        b"workspace-a\0campaign-1\0caller-controlled-correlation-canary"
    ).hexdigest()
    assert guessed not in serialized


def test_known_event_keeps_only_registered_typed_summary_fields():
    projected = project_public_campaign_event(
        raw_event(
            event_type="campaign:action-blocked",
            payload={
                "study_id": "study-1",
                "stage_index": 2,
                "stage": "full_training",
                "code": "campaign_remote_profile_unavailable",
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
        "code": "campaign_remote_profile_unavailable",
        "stage_index": 2,
    }
    serialized = json.dumps(projected.model_dump(mode="json"), sort_keys=True)
    assert "operator-error-canary" not in serialized
    assert "private.json" not in serialized
    assert "candidate-map-canary" not in serialized


def test_allowed_field_names_reject_untrusted_values_and_non_finite_numbers():
    projected = project_public_campaign_event(
        raw_event(
            event_type="campaign:action-blocked",
            payload={
                "study_id": "study-1",
                "stage_index": float("inf"),
                "stage": "candidate-map-canary",
                "code": "private-case-identity",
            },
        )
    )

    assert projected.summary is not None
    assert projected.summary.model_dump(mode="json", exclude_none=True) == {
        "schema_version": "public_campaign_event_summary.v1",
        "study_id": "study-1",
    }

    metrics = project_public_campaign_event(
        raw_event(
            event_type="campaign:training-metrics-appended",
            payload={
                "action_id": "action-1",
                "attempt_id": "attempt-1",
                "metric_names": ["candidate-map-canary"],
                "reason_codes": ["private-case-identity"],
            },
        )
    )
    serialized = json.dumps(metrics.model_dump(mode="json"), sort_keys=True)
    assert "candidate-map-canary" not in serialized
    assert "private-case-identity" not in serialized


def test_metadata_only_inventory_includes_validation_and_all_terminal_actions():
    assert PUBLIC_EVENT_TYPE_FIELDS["campaign:validation-failed"] == frozenset()
    assert PUBLIC_EVENT_TYPE_FIELDS["campaign:action-completed"]
    assert PUBLIC_EVENT_TYPE_FIELDS["campaign:action-failed"]
    assert PUBLIC_EVENT_TYPE_FIELDS["campaign:action-cancelled"]
    assert PUBLIC_EVENT_TYPE_FIELDS["campaign:action-force-stopped"]


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


def test_public_attempt_projection_drops_executor_configuration_and_lease_identity():
    projected = project_public_campaign_attempt(
        {
            "schema_version": "campaign_action_attempt.v1",
            "attempt_id": "attempt-1",
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "study_id": "study-1",
            "action_id": "action-1",
            "attempt_number": 1,
            "claim_generation": 2,
            "status": "running",
            "input_digest": "a" * 64,
            "candidate_digest": "b" * 64,
            "manifest_revision": 3,
            "stage": "full_training",
            "lease_owner": "worker-owner-canary",
            "lease_expires_at": "2026-07-17T00:00:30Z",
            "heartbeat_at": "2026-07-17T00:00:10Z",
            "executor": {
                "kind": "ssh_remote",
                "script_path": "C:/operator/restricted/train_stage.py",
                "python_executable": "/home/operator/venv-canary/bin/python",
                "input_files": ["C:/operator/restricted/dataset-canary.jsonl"],
                "output_paths": ["logs"],
            },
            "sealed_result_uri": "file:///C:/operator/restricted/sealed-result-canary",
            "created_at": "2026-07-17T00:00:00Z",
            "updated_at": "2026-07-17T00:00:20Z",
        }
    )

    assert set(projected.model_dump(mode="json")) == PUBLIC_CAMPAIGN_ATTEMPT_FIELDS
    assert projected.executor_kind == "ssh_remote"
    serialized = json.dumps(projected.model_dump(mode="json"), sort_keys=True)
    assert "restricted" not in serialized
    assert "train_stage.py" not in serialized
    assert "venv-canary" not in serialized
    assert "dataset-canary" not in serialized
    assert "sealed-result-canary" not in serialized
    assert "worker-owner-canary" not in serialized


def test_public_attempt_projection_normalizes_untrusted_executor_kind():
    base = {
        "attempt_id": "attempt-1",
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "study_id": "study-1",
        "action_id": "action-1",
        "attempt_number": 1,
        "claim_generation": 0,
        "status": "scheduled",
        "input_digest": "a" * 64,
        "candidate_digest": "b" * 64,
        "manifest_revision": 1,
        "stage": "full_training",
        "created_at": "2026-07-17T00:00:00Z",
        "updated_at": "2026-07-17T00:00:00Z",
    }
    unsafe = project_public_campaign_attempt(
        {**base, "executor": {"kind": "C:/unsafe path canary"}}
    )
    missing = project_public_campaign_attempt(base)

    assert unsafe.executor_kind is None
    assert "canary" not in json.dumps(unsafe.model_dump(mode="json"))
    assert missing.executor_kind is None


def test_public_artifact_projection_drops_uri_metadata_and_runtime_extras():
    projected = project_public_campaign_artifact(
        {
            "schema_version": "campaign_artifact_record.v1",
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "artifact_id": "artifact-1",
            "producer_action_id": "action-1",
            "uri": "C:/operator/restricted-result.json",
            "sha256": "a" * 64,
            "size_bytes": 10,
            "schema_name": "training_metrics_jsonl.v1",
            "sealed": True,
            "valid": True,
            "metadata": {
                "reference": "candidate-map-canary",
                "nested": {"ordinary": "protected-epoch-canary"},
            },
            "created_at": "2026-07-16T00:00:00Z",
        }
    )

    assert set(projected.model_dump(mode="json")) == PUBLIC_CAMPAIGN_ARTIFACT_FIELDS
    serialized = json.dumps(projected.model_dump(mode="json"), sort_keys=True)
    assert "restricted-result.json" not in serialized
    assert "candidate-map-canary" not in serialized
    assert "protected-epoch-canary" not in serialized


def test_public_artifact_projection_replaces_unproven_schema_names():
    projected = project_public_campaign_artifact(
        {
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "artifact_id": "artifact-1",
            "producer_action_id": None,
            "sha256": "a" * 64,
            "size_bytes": 10,
            "schema_name": "candidate-map-canary",
            "sealed": True,
            "valid": True,
            "created_at": "2026-07-16T00:00:00Z",
        }
    )

    assert projected.schema_name == "unclassified_artifact.v1"
