"""Pure campaign transition rules shared by every transport."""

from __future__ import annotations

from dataclasses import dataclass

from bashgym.campaigns.contracts import TERMINAL_CAMPAIGN_STATES, CampaignStatus, CampaignTrigger


class InvalidCampaignTransitionError(ValueError):
    """Raised before persistence when a campaign trigger is illegal."""

    code = "campaign_invalid_transition"

    def __init__(self, status: CampaignStatus, trigger: CampaignTrigger):
        self.status = status
        self.trigger = trigger
        allowed = sorted(item.value for item in allowed_triggers(status))
        super().__init__(
            f"{self.code}: {trigger.value} is not allowed from {status.value}; "
            f"allowed triggers: {', '.join(allowed) or 'none'}"
        )


@dataclass(frozen=True)
class TransitionResult:
    status: CampaignStatus
    prior_scheduling_status: CampaignStatus | None
    event_type: str


@dataclass(frozen=True)
class PromotionGateEvaluation:
    eligible: bool
    blocking_codes: tuple[str, ...]


def evaluate_promotion_gate(
    *,
    active_action_id: str | None,
    comparison_verdict: str | None,
    candidate_digest: str | None,
    protected_required: bool,
    protected_passed: bool,
    human_work_complete: bool,
) -> PromotionGateEvaluation:
    """Evaluate the promotion evidence gate shared by reads and mutations."""

    blockers: list[str] = []
    if active_action_id is not None:
        blockers.append("campaign_active_action_present")
    if comparison_verdict != "passed" or not candidate_digest:
        blockers.append("campaign_development_gate_not_passed")
    if protected_required and not protected_passed:
        blockers.append("campaign_protected_gate_not_passed")
    if not human_work_complete:
        blockers.append("campaign_human_work_incomplete")
    return PromotionGateEvaluation(eligible=not blockers, blocking_codes=tuple(blockers))


_FIXED_TRANSITIONS: dict[CampaignTrigger, tuple[frozenset[CampaignStatus], CampaignStatus, str]] = {
    CampaignTrigger.VALIDATE: (
        frozenset({CampaignStatus.DRAFT}),
        CampaignStatus.VALIDATING,
        "campaign:validation-started",
    ),
    CampaignTrigger.VALIDATION_PASSED: (
        frozenset({CampaignStatus.VALIDATING}),
        CampaignStatus.READY,
        "campaign:ready",
    ),
    CampaignTrigger.VALIDATION_FAILED: (
        frozenset({CampaignStatus.VALIDATING}),
        CampaignStatus.DRAFT,
        "campaign:validation-failed",
    ),
    CampaignTrigger.START: (
        frozenset({CampaignStatus.READY}),
        CampaignStatus.ACTIVE,
        "campaign:started",
    ),
    CampaignTrigger.PAUSE: (
        frozenset({CampaignStatus.ACTIVE}),
        CampaignStatus.PAUSED,
        "campaign:paused",
    ),
    CampaignTrigger.RESUME: (
        frozenset({CampaignStatus.PAUSED}),
        CampaignStatus.ACTIVE,
        "campaign:resumed",
    ),
    CampaignTrigger.STOPPING_RULE_MET: (
        frozenset({CampaignStatus.ACTIVE}),
        CampaignStatus.EXHAUSTED,
        "campaign:exhausted",
    ),
    CampaignTrigger.CONCLUDE: (
        frozenset({CampaignStatus.ACTIVE, CampaignStatus.AWAITING_AUTHORITY}),
        CampaignStatus.COMPLETED,
        "campaign:completed",
    ),
    CampaignTrigger.PROMOTION_COMMITTED: (
        frozenset({CampaignStatus.ACTIVE, CampaignStatus.AWAITING_AUTHORITY}),
        CampaignStatus.COMPLETED,
        "campaign:completed",
    ),
    CampaignTrigger.CANCELLATION_SETTLED: (
        frozenset({CampaignStatus.CANCELLING}),
        CampaignStatus.CANCELLED,
        "campaign:cancelled",
    ),
}


def allowed_triggers(status: CampaignStatus) -> frozenset[CampaignTrigger]:
    if status in TERMINAL_CAMPAIGN_STATES:
        return frozenset()
    values = {
        trigger
        for trigger, (sources, _destination, _event) in _FIXED_TRANSITIONS.items()
        if status in sources
    }
    values.update({CampaignTrigger.INVARIANT_FAILURE, CampaignTrigger.CANCEL})
    if status in {CampaignStatus.ACTIVE, CampaignStatus.PAUSED}:
        values.add(CampaignTrigger.AUTHORITY_MISSING)
    if status == CampaignStatus.AWAITING_AUTHORITY:
        values.add(CampaignTrigger.AUTHORITY_SATISFIED)
    if status == CampaignStatus.CANCELLING:
        values.discard(CampaignTrigger.CANCEL)
    return frozenset(values)


def transition_campaign(
    status: CampaignStatus,
    trigger: CampaignTrigger,
    *,
    prior_scheduling_status: CampaignStatus | None = None,
) -> TransitionResult:
    """Return the exact legal destination without mutating persistence."""

    if trigger not in allowed_triggers(status):
        raise InvalidCampaignTransitionError(status, trigger)

    if trigger == CampaignTrigger.AUTHORITY_MISSING:
        return TransitionResult(
            status=CampaignStatus.AWAITING_AUTHORITY,
            prior_scheduling_status=status,
            event_type="campaign:authority-required",
        )
    if trigger == CampaignTrigger.AUTHORITY_SATISFIED:
        if prior_scheduling_status not in {CampaignStatus.ACTIVE, CampaignStatus.PAUSED}:
            raise InvalidCampaignTransitionError(status, trigger)
        return TransitionResult(
            status=prior_scheduling_status,
            prior_scheduling_status=None,
            event_type="campaign:authority-satisfied",
        )
    if trigger == CampaignTrigger.INVARIANT_FAILURE:
        return TransitionResult(CampaignStatus.FAILED, None, "campaign:failed")
    if trigger == CampaignTrigger.CANCEL:
        return TransitionResult(CampaignStatus.CANCELLING, None, "campaign:cancelling")

    _sources, destination, event_type = _FIXED_TRANSITIONS[trigger]
    return TransitionResult(destination, None, event_type)


__all__ = [
    "InvalidCampaignTransitionError",
    "PromotionGateEvaluation",
    "TransitionResult",
    "allowed_triggers",
    "evaluate_promotion_gate",
    "transition_campaign",
]
