"""ECHO loss the terminal-RL trainer applies inside ``compute_loss``.

This is the torch-facing companion to the framework-free ``bashgym.gym.echo``:
it turns model logits into the ECHO environment-prediction loss term and combines
it with the policy-gradient (GRPO) loss. ``torch`` is imported lazily so importing
this module never requires torch — only calling the functions does.

Integration contract (what the backend's trainer does per step):

    # one forward pass produces `logits` for the full rollout transcript
    masks = build_echo_masks(segments)            # from bashgym.gym.echo, with the tokenizer
    loss = compute_loss(...)                       # the backend's GRPO loss on action tokens
    loss = echo_augmented_loss(
        loss, logits, input_ids,
        observation_mask=torch.tensor(masks.observation_mask),
        total_observation_tokens=masks.total_observation_tokens,
        aux_lambda=config.echo_aux_lambda,         # 0.05
    )

The action-token GRPO loss and the observation-token ECHO loss share that single
forward pass (ECHO, arXiv:2605.24517) — no extra rollouts or forward passes.
"""

from __future__ import annotations

from typing import Any

from bashgym.gym.echo import ECHO_DEFAULT_LAMBDA


def environment_prediction_loss_from_logits(
    logits: Any,
    input_ids: Any,
    observation_mask: Any,
    total_observation_tokens: int,
) -> Any:
    """ECHO environment loss from logits: ``-(1/Z) * sum_{t in O'} log p(x_t)``.

    ``logits`` is ``(seq_len, vocab)`` for one sequence, ``input_ids`` is
    ``(seq_len,)``, and ``observation_mask`` is a boolean ``(seq_len,)`` marking
    the kept observation target tokens (O', warning tokens already excluded).
    ``total_observation_tokens`` is Z = |O| (every observation token) so the
    normalizer stays comparable across kept subsets. Returns a scalar tensor;
    ``0`` when there are no observation tokens. Standard next-token shift is
    applied (logits[t] predicts token t+1).
    """

    import torch.nn.functional as functional

    if total_observation_tokens <= 0:
        return logits.new_zeros(())

    shift_logits = logits[:-1, :]
    shift_labels = input_ids[1:]
    shift_obs = observation_mask[1:].bool()

    token_logprobs = (
        functional.log_softmax(shift_logits, dim=-1)
        .gather(-1, shift_labels.long().unsqueeze(-1))
        .squeeze(-1)
    )
    selected = token_logprobs[shift_obs]
    return -selected.sum() / total_observation_tokens


def echo_augmented_loss(
    base_loss: Any,
    logits: Any,
    input_ids: Any,
    observation_mask: Any,
    total_observation_tokens: int,
    aux_lambda: float = ECHO_DEFAULT_LAMBDA,
) -> Any:
    """Combine a policy-gradient loss with the scaled ECHO environment loss.

    ``base_loss + aux_lambda * env_loss``; differentiable through ``logits`` so the
    auxiliary term trains the policy to predict its own environment observations.
    """

    env_loss = environment_prediction_loss_from_logits(
        logits, input_ids, observation_mask, total_observation_tokens
    )
    return base_loss + aux_lambda * env_loss
