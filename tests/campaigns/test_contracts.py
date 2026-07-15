"""Executable campaign-contract and state-machine tests."""

from itertools import product

import pytest
from pydantic import ValidationError

from bashgym.campaigns.contracts import (
    HERMES_CAPABILITIES,
    AutonomyProfile,
    CampaignEvent,
    CampaignStatus,
    CampaignTrigger,
    Capability,
    CredentialKind,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
)
from bashgym.campaigns.transitions import (
    InvalidCampaignTransitionError,
    allowed_triggers,
    transition_campaign,
)


def test_campaign_transition_matrix_is_total_and_fail_closed():
    for status, trigger in product(CampaignStatus, CampaignTrigger):
        if trigger in allowed_triggers(status):
            prior = (
                CampaignStatus.PAUSED if trigger == CampaignTrigger.AUTHORITY_SATISFIED else None
            )
            result = transition_campaign(status, trigger, prior_scheduling_status=prior)
            assert isinstance(result.status, CampaignStatus)
            assert result.event_type.startswith("campaign:")
        else:
            with pytest.raises(InvalidCampaignTransitionError) as error:
                transition_campaign(status, trigger)
            assert error.value.code == "campaign_invalid_transition"


def test_authority_round_trip_preserves_prior_pause_intent():
    blocked = transition_campaign(CampaignStatus.PAUSED, CampaignTrigger.AUTHORITY_MISSING)
    resumed = transition_campaign(
        blocked.status,
        CampaignTrigger.AUTHORITY_SATISFIED,
        prior_scheduling_status=blocked.prior_scheduling_status,
    )

    assert blocked.status == CampaignStatus.AWAITING_AUTHORITY
    assert resumed.status == CampaignStatus.PAUSED
    assert resumed.prior_scheduling_status is None


def test_terminal_campaigns_have_no_mutating_triggers():
    for status in (
        CampaignStatus.COMPLETED,
        CampaignStatus.EXHAUSTED,
        CampaignStatus.FAILED,
        CampaignStatus.CANCELLED,
    ):
        assert allowed_triggers(status) == frozenset()


def test_stage_plan_is_immutable_unique_and_explicit():
    plan = StagePlan(
        items=(
            StagePlanItem(
                stage=StageKind.DATA_BUILD,
                disposition=StageDisposition.REQUIRED,
                reason="Dataset must be sealed.",
            ),
            StagePlanItem(
                stage=StageKind.PROTECTED_EVALUATION,
                disposition=StageDisposition.NOT_APPLICABLE,
                reason="Development-only campaign.",
            ),
        )
    )
    assert plan.items[1].disposition == StageDisposition.NOT_APPLICABLE
    with pytest.raises(ValidationError):
        plan.items[0].reason = "mutated"

    with pytest.raises(ValidationError, match="cannot repeat"):
        StagePlan(items=(plan.items[0], plan.items[0]))


def test_campaign_event_rejects_secret_shaped_payload_fields():
    with pytest.raises(ValidationError, match="secret-like"):
        CampaignEvent(
            event_id="event-1",
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            sequence=1,
            aggregate_version=1,
            event_type="campaign:created",
            payload={"nested": {"access_token": "must-not-persist"}},
            actor_id="actor-a",
            credential_kind=CredentialKind.ACCESS,
            correlation_id="correlation-1",
            idempotency_key="request-1",
        )


def test_hermes_profile_cannot_cross_privileged_boundaries():
    assert Capability.COMPUTE_TRAIN_WITHIN_BUDGET in HERMES_CAPABILITIES
    assert Capability.EVAL_DEVELOPMENT in HERMES_CAPABILITIES
    assert Capability.EXPERIMENT_LEDGER_WRITE in HERMES_CAPABILITIES
    assert Capability.EXPERIMENT_CODE_MUTATE not in HERMES_CAPABILITIES
    assert Capability.COMPUTE_AMEND_BUDGET not in HERMES_CAPABILITIES
    assert Capability.EVAL_PROTECTED_ACQUIRE not in HERMES_CAPABILITIES
    assert Capability.PROMOTION_DECIDE not in HERMES_CAPABILITIES
    assert Capability.ARTIFACT_PUBLISH_HF not in HERMES_CAPABILITIES
    assert AutonomyProfile.HERMES_BOUNDED.value == "hermes_bounded"
