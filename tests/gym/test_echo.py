"""Tests for the ECHO environment-prediction auxiliary-loss layer.

ECHO (arXiv:2605.24517) adds an auxiliary cross-entropy loss on environment
observation tokens that shares the policy forward pass with GRPO:
    L_ECHO = L_GRPO(action tokens) + lambda * L_env(observation tokens)
with lambda = 0.05 and L_env = -(1/Z) * sum_{t in O'} log p(x_t | x_<t),
Z = |O| (total observation length). This module is the pure, framework-free
contract a terminal-RL trainer backend consumes; it mirrors how
``bashgym.gym.dppo`` implements the math layer before any trainer backend.
"""

import math

import pytest

from bashgym.gym.echo import (
    ECHO_DEFAULT_LAMBDA,
    EchoConfig,
    EchoSegment,
    build_echo_masks,
    combine_echo_loss,
    environment_prediction_loss,
)


def test_echo_config_defaults_match_paper():
    config = EchoConfig()

    assert config.enabled is False
    assert config.aux_lambda == ECHO_DEFAULT_LAMBDA == 0.05
    assert config.exclude_token_ids == ()
    assert config.to_dict() == {
        "enabled": False,
        "aux_lambda": 0.05,
        "exclude_token_ids": [],
    }


def test_build_echo_masks_separates_action_and_observation_positions():
    masks = build_echo_masks(
        [
            EchoSegment("system", [1, 2]),
            EchoSegment("prompt", [3, 4, 5]),
            EchoSegment("action", [6, 7]),
            EchoSegment("observation", [8, 9, 10]),
            EchoSegment("action", [11]),
            EchoSegment("observation", [12]),
        ]
    )

    assert masks.input_ids == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    # action tokens -> policy-gradient; observation tokens -> env-prediction CE
    assert masks.action_mask == (
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
    )
    assert masks.observation_mask == (
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
    )
    # system + prompt tokens are ignored by both losses
    assert masks.total_observation_tokens == 4


def test_build_echo_masks_excludes_warning_tokens_from_observation_loss():
    masks = build_echo_masks(
        [
            EchoSegment("action", [6]),
            EchoSegment("observation", [8, 99, 10, 99]),
        ],
        exclude_token_ids=(99,),
    )

    # excluded "warning" token id 99 is dropped from the observation loss mask
    assert masks.observation_mask == (False, True, False, True, False)
    # but Z still normalizes by the full observation length |O| (paper: Z = |O|)
    assert masks.total_observation_tokens == 4


def test_build_echo_masks_rejects_unknown_role():
    with pytest.raises(ValueError, match="role"):
        build_echo_masks([EchoSegment("tool", [1])])


def test_environment_prediction_loss_is_neg_mean_logprob_normalized_by_total_observation():
    # kept O' log-probs sum to -2.0; normalize by |O| = 4 (not |O'| = 2)
    loss = environment_prediction_loss([-1.5, -0.5], total_observation_tokens=4)

    assert loss == pytest.approx(2.0 / 4)


def test_environment_prediction_loss_zero_when_no_observation_tokens():
    assert environment_prediction_loss([], total_observation_tokens=0) == 0.0


def test_combine_echo_loss_adds_scaled_auxiliary_term():
    total = combine_echo_loss(grpo_loss=1.0, env_loss=2.0, aux_lambda=0.05)

    assert total == pytest.approx(1.0 + 0.05 * 2.0)


def test_combine_echo_loss_defaults_to_paper_lambda():
    assert combine_echo_loss(1.0, 2.0) == pytest.approx(1.0 + ECHO_DEFAULT_LAMBDA * 2.0)


def test_environment_prediction_loss_matches_explicit_formula():
    logprobs = [-0.2, -0.3, -1.0]
    z = 5
    expected = -sum(logprobs) / z

    assert environment_prediction_loss(logprobs, total_observation_tokens=z) == pytest.approx(
        expected
    )
    assert not math.isnan(expected)
