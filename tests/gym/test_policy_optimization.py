"""Tests for provider-neutral GRPO/DAPO/GDPO math."""

from __future__ import annotations

import math

import pytest

from bashgym.gym.policy_optimization import (
    clipped_policy_objective,
    dapo_overlong_reward,
    gdpo_advantages,
    global_normalized_microbatch_sum,
)


def test_clipped_policy_objective_supports_asymmetric_clip_higher():
    result = clipped_policy_objective(
        current_logprob=math.log(1.25),
        behavior_logprob=0.0,
        advantage=2.0,
        ratio_clip_min=0.2,
        ratio_clip_max=0.3,
    )

    assert result.probability_ratio == pytest.approx(1.25)
    assert result.objective == pytest.approx(2.5)
    assert result.ratio_was_clipped is False
    assert result.loss == pytest.approx(-2.5)


def test_clipped_policy_objective_applies_negative_advantage_dual_clip():
    result = clipped_policy_objective(
        current_logprob=math.log(10.0),
        behavior_logprob=0.0,
        advantage=-2.0,
        ratio_clip_min=0.2,
        ratio_clip_max=0.2,
        dual_clip=3.0,
    )

    assert result.unclipped_objective == pytest.approx(-20.0)
    assert result.clipped_objective == pytest.approx(-2.4)
    assert result.objective == pytest.approx(-6.0)
    assert result.dual_clipped is True


def test_gdpo_normalizes_each_component_within_each_prompt_group():
    result = gdpo_advantages(
        group_ids=["prompt-a", "prompt-a", "prompt-b", "prompt-b"],
        reward_components={
            "correctness": [0.0, 1.0, 1.0, 0.0],
            "format": [1.0, 1.0, 0.0, 1.0],
        },
    )

    assert result.combined_advantages == pytest.approx(
        (-1 / math.sqrt(2), 1 / math.sqrt(2), 0.0, 0.0),
        abs=2e-6,
    )
    assert sum(result.advantages) == pytest.approx(0.0, abs=1e-9)
    advantage_mean = sum(result.advantages) / len(result.advantages)
    advantage_std = math.sqrt(
        sum((value - advantage_mean) ** 2 for value in result.advantages)
        / (len(result.advantages) - 1)
    )
    assert advantage_std == pytest.approx(1.0, abs=2e-6)
    assert ("prompt-a", "format") in result.zero_variance_groups


def test_gdpo_zero_variance_components_are_safe_and_visible():
    result = gdpo_advantages(
        group_ids=[1, 1],
        reward_components={"correctness": [1.0, 1.0], "format": [0.5, 0.5]},
        weights={"format": 0.25},
    )

    assert result.advantages == (0.0, 0.0)
    assert result.zero_variance_groups == ((1, "correctness"), (1, "format"))


def test_dapo_overlong_reward_penalizes_only_the_buffer_region():
    before_buffer = dapo_overlong_reward(
        reward=1.0,
        response_tokens=80,
        max_response_tokens=100,
        overlong_buffer_tokens=20,
    )
    inside_buffer = dapo_overlong_reward(
        reward=1.0,
        response_tokens=90,
        max_response_tokens=100,
        overlong_buffer_tokens=20,
    )
    at_limit = dapo_overlong_reward(
        reward=1.0,
        response_tokens=100,
        max_response_tokens=100,
        overlong_buffer_tokens=20,
    )

    assert before_buffer.shaped_reward == 1.0
    assert inside_buffer.penalty == -0.5
    assert inside_buffer.shaped_reward == 0.5
    assert at_limit.shaped_reward == 0.0


def test_global_normalization_is_equivalent_across_microbatches():
    first = global_normalized_microbatch_sum([2.0, 4.0], [1, 1], total_valid_items=3)
    second = global_normalized_microbatch_sum([6.0, 100.0], [1, 0], total_valid_items=3)

    assert first + second == pytest.approx((2.0 + 4.0 + 6.0) / 3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ratio_clip_min": 1.0}, "ratio_clip_min"),
        ({"dual_clip": 1.0}, "dual_clip"),
    ],
)
def test_clipped_policy_objective_rejects_unsafe_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        clipped_policy_objective(
            current_logprob=0.0,
            behavior_logprob=0.0,
            advantage=1.0,
            **kwargs,
        )
