"""ECHO environment-prediction auxiliary loss for terminal-agent RL.

ECHO ("Terminal Agents Learn World Models for Free", arXiv:2605.24517) augments
the GRPO policy-gradient loss with a dense auxiliary loss that trains the policy
to predict the environment observation tokens caused by its own actions:

    L_ECHO(theta) = L_GRPO(theta; A) + lambda * L_Env(theta; O')
    L_Env(theta; O') = -(1/Z) * sum_{t in O'} log p_theta(x_t | x_<t),   Z = |O|

where A is the set of action (assistant) token positions, O is every observation
(terminal output) token position, O' subset O is the kept subset (low-entropy
"warning" tokens excluded), and lambda = 0.05. The two losses share a single
actor forward pass: the same logits are gathered through an action mask (GRPO)
and an observation mask (environment prediction).

This module is deliberately the framework-free math/segmentation layer that a
terminal-RL trainer backend consumes, mirroring how ``bashgym.gym.dppo``
implements the trust-region math before any trainer backend. It carries no torch
or GPU dependency, so it is hardware-agnostic by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

ECHO_DEFAULT_LAMBDA = 0.05

ACTION_ROLE = "action"
OBSERVATION_ROLE = "observation"
_IGNORED_ROLES = frozenset({"system", "prompt"})
_VALID_ROLES = frozenset({ACTION_ROLE, OBSERVATION_ROLE}) | _IGNORED_ROLES


@dataclass(frozen=True)
class EchoConfig:
    """Resolved ECHO knobs for a terminal-RL run."""

    enabled: bool = False
    aux_lambda: float = ECHO_DEFAULT_LAMBDA
    exclude_token_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "aux_lambda": self.aux_lambda,
            "exclude_token_ids": list(self.exclude_token_ids),
        }


@dataclass
class EchoSegment:
    """One contiguous span of the transcript with its supervision role.

    ``role`` is one of ``system``/``prompt`` (ignored by both losses),
    ``action`` (GRPO policy-gradient target), or ``observation`` (environment
    prediction target). The caller maps a rollout transcript onto these roles.
    """

    role: str
    token_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class EchoMasks:
    """Per-token supervision masks for one transcript.

    ``observation_mask`` already excludes "warning" token ids, while
    ``total_observation_tokens`` counts every observation token (|O|) so the
    environment loss normalizer Z stays comparable across kept subsets.
    """

    input_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    observation_mask: tuple[bool, ...]
    total_observation_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_ids": list(self.input_ids),
            "action_mask": list(self.action_mask),
            "observation_mask": list(self.observation_mask),
            "total_observation_tokens": self.total_observation_tokens,
        }


def build_echo_masks(
    segments: Iterable[EchoSegment],
    *,
    exclude_token_ids: Iterable[int] = (),
) -> EchoMasks:
    """Flatten labeled transcript segments into ECHO supervision masks."""

    excluded = set(exclude_token_ids)
    input_ids: list[int] = []
    action_mask: list[bool] = []
    observation_mask: list[bool] = []
    total_observation_tokens = 0

    for segment in segments:
        if segment.role not in _VALID_ROLES:
            raise ValueError(
                f"EchoSegment role={segment.role!r} must be one of {sorted(_VALID_ROLES)}"
            )
        is_action = segment.role == ACTION_ROLE
        is_observation = segment.role == OBSERVATION_ROLE
        for token_id in segment.token_ids:
            input_ids.append(token_id)
            action_mask.append(is_action)
            observation_mask.append(is_observation and token_id not in excluded)
            if is_observation:
                total_observation_tokens += 1

    return EchoMasks(
        input_ids=tuple(input_ids),
        action_mask=tuple(action_mask),
        observation_mask=tuple(observation_mask),
        total_observation_tokens=total_observation_tokens,
    )


def environment_prediction_loss(
    kept_token_logprobs: Sequence[float],
    total_observation_tokens: int,
) -> float:
    """ECHO environment loss: -(1/Z) * sum_{t in O'} log p(x_t), Z = |O|.

    ``kept_token_logprobs`` are the per-token log-probabilities of the kept
    observation targets (O'); ``total_observation_tokens`` is |O| (every
    observation token, including excluded warning tokens) so different kept
    subsets remain comparable. Returns 0.0 when there are no observation tokens.
    """

    if total_observation_tokens <= 0:
        return 0.0
    return -sum(kept_token_logprobs) / total_observation_tokens


def combine_echo_loss(
    grpo_loss: float,
    env_loss: float,
    aux_lambda: float = ECHO_DEFAULT_LAMBDA,
) -> float:
    """Combine the GRPO loss with the scaled environment-prediction loss."""

    return grpo_loss + aux_lambda * env_loss
