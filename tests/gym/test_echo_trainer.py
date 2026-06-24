"""Torch-level ECHO loss the terminal-RL backend applies inside compute_loss.

Runs only where torch is installed (skipped otherwise). On this repo that's the
Python 3.12 CUDA env; the math is CPU-tensor based so it needs no GPU to verify.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from bashgym.gym.echo_trainer import (  # noqa: E402 - after importorskip
    echo_augmented_loss,
    environment_prediction_loss_from_logits,
)


def test_env_loss_from_uniform_logits_equals_neg_log_vocab():
    vocab = 5
    # all-zero logits -> uniform softmax -> per-token logprob = -ln(vocab)
    logits = torch.zeros((4, vocab))
    input_ids = torch.tensor([0, 1, 2, 3])
    observation_mask = torch.tensor([False, False, True, True])

    loss = environment_prediction_loss_from_logits(
        logits, input_ids, observation_mask, total_observation_tokens=2
    )

    # two observation targets, each -ln(5); -(sum)/Z with Z=2 -> ln(5)
    assert float(loss) == pytest.approx(math.log(vocab), abs=1e-5)


def test_env_loss_zero_when_no_observation_tokens():
    logits = torch.zeros((3, 4))
    input_ids = torch.tensor([0, 1, 2])
    mask = torch.tensor([False, False, False])

    loss = environment_prediction_loss_from_logits(
        logits, input_ids, mask, total_observation_tokens=0
    )

    assert float(loss) == 0.0


def test_env_loss_normalizer_uses_total_observation_not_kept_count():
    # one kept observation target, but Z=4 (paper: Z = |O|, all observation tokens)
    vocab = 5
    logits = torch.zeros((4, vocab))
    input_ids = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([False, False, False, True])

    loss = environment_prediction_loss_from_logits(
        logits, input_ids, mask, total_observation_tokens=4
    )

    assert float(loss) == pytest.approx(math.log(vocab) / 4, abs=1e-5)


def test_echo_augmented_loss_adds_scaled_env_term():
    vocab = 5
    logits = torch.zeros((4, vocab))
    input_ids = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([False, False, True, True])
    base = torch.tensor(1.0)

    total = echo_augmented_loss(
        base, logits, input_ids, mask, total_observation_tokens=2, aux_lambda=0.05
    )

    assert float(total) == pytest.approx(1.0 + 0.05 * math.log(vocab), abs=1e-5)


def test_echo_augmented_loss_keeps_gradient():
    vocab = 4
    logits = torch.zeros((3, vocab), requires_grad=True)
    input_ids = torch.tensor([0, 1, 2])
    mask = torch.tensor([False, True, True])
    base = torch.zeros(())

    total = echo_augmented_loss(base, logits, input_ids, mask, total_observation_tokens=2)
    total.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
