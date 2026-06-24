import math

import pytest

from bashgym.gym.dppo import (
    DPPO_BINARY_KL_THRESHOLD,
    DPPO_BINARY_TV_THRESHOLD,
    DPPOToken,
    analyze_dppo_batch,
    binary_kl_divergence,
    binary_tv_divergence,
    dppo_mask_decision,
    policy_ratio,
    probability_from_logprob,
)


def _lp(probability: float) -> float:
    return math.log(probability)


def test_binary_tv_is_anchored_to_behavior_policy_probability():
    assert binary_tv_divergence(_lp(0.2), _lp(0.3)) == pytest.approx(0.1)
    assert binary_tv_divergence(_lp(0.2), _lp(0.01)) == pytest.approx(0.19)


def test_binary_kl_matches_two_bucket_partition():
    value = binary_kl_divergence(_lp(0.2), _lp(0.3))
    expected = 0.2 * math.log(0.2 / 0.3) + 0.8 * math.log(0.8 / 0.7)
    assert value == pytest.approx(expected)


def test_policy_ratio_kept_for_diagnostics_not_mask_definition():
    assert policy_ratio(_lp(0.001), _lp(0.01)) == pytest.approx(10.0)
    # The same ratio can be much larger/smaller in probability-space impact.
    assert binary_tv_divergence(_lp(0.001), _lp(0.01)) == pytest.approx(0.009)


def test_positive_advantage_masks_only_when_policy_increases_too_far():
    inside = dppo_mask_decision(DPPOToken(_lp(0.2), _lp(0.3), advantage=1.0))
    outside = dppo_mask_decision(DPPOToken(_lp(0.2), _lp(0.5), advantage=1.0))
    opposite_direction = dppo_mask_decision(DPPOToken(_lp(0.2), _lp(0.01), advantage=1.0))

    assert inside.masked is False
    assert outside.masked is True
    assert outside.reason == "positive_advantage_above_trust_region"
    assert opposite_direction.masked is False


def test_negative_advantage_masks_only_when_policy_decreases_too_far():
    inside = dppo_mask_decision(DPPOToken(_lp(0.5), _lp(0.4), advantage=-1.0))
    outside = dppo_mask_decision(DPPOToken(_lp(0.5), _lp(0.2), advantage=-1.0))
    opposite_direction = dppo_mask_decision(DPPOToken(_lp(0.5), _lp(0.8), advantage=-1.0))

    assert inside.masked is False
    assert outside.masked is True
    assert outside.reason == "negative_advantage_above_trust_region"
    assert opposite_direction.masked is False


def test_binary_kl_threshold_path_uses_kl_score():
    decision = dppo_mask_decision(
        DPPOToken(_lp(0.2), _lp(0.5), advantage=1.0),
        divergence="binary_kl",
        threshold=DPPO_BINARY_KL_THRESHOLD,
    )

    assert decision.masked is True
    assert decision.binary_kl > DPPO_BINARY_KL_THRESHOLD


def test_analyze_dppo_batch_reports_mask_and_mismatch_telemetry():
    decisions, telemetry = analyze_dppo_batch(
        [
            DPPOToken(_lp(0.2), _lp(0.5), advantage=1.0, token="good"),
            DPPOToken(_lp(0.5), _lp(0.2), advantage=-1.0, token="bad"),
            DPPOToken(_lp(0.2), _lp(0.25), advantage=1.0, token="ok"),
        ],
        collapse_min_tokens=3,
        collapse_masked_fraction=0.6,
    )

    assert [decision.masked for decision in decisions] == [True, True, False]
    assert telemetry.n_tokens == 3
    assert telemetry.masked_updates == 2
    assert telemetry.masked_fraction == pytest.approx(2 / 3)
    assert telemetry.max_binary_tv == pytest.approx(0.3)
    assert telemetry.threshold == DPPO_BINARY_TV_THRESHOLD
    assert telemetry.collapse_warning is True
    assert telemetry.to_dict()["divergence"] == "binary_tv"


def test_probability_from_logprob_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        probability_from_logprob(float("nan"))
