"""RWML (Reinforcement World Model Learning) reward layer for terminal agents.

RWML (arXiv:2602.05842) trains an action-conditioned world model as a
self-supervised pre-RL stage before task-success RL. The agent predicts the next
environment state from history+action, and is rewarded by an embedding-space
binary signal optimized with GRPO:

    r_WM(s_hat_{t+1}, s_{t+1}) = 1 if d < tau_d else 0
    d(s_hat, s) = 1 - cos(E(s_hat), E(s))
    A = (r_WM - mean(r_WM)) / std(r_WM)        # group-relative advantage

where E is an off-the-shelf text embedding model. Rollouts are converted into
<s_<=t, a_t, s_{t+1}> triplets, and "easy" transitions (consistently predicted)
are sub-sampled so training concentrates on harder ones.

This module is the pure, embedding-provider-agnostic contract: the embedding
model is injected as ``embed_fn``, so the layer carries no torch/GPU or provider
dependency and is hardware-agnostic by construction. It mirrors how
``bashgym.gym.dppo`` ships the math layer before any trainer backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

RWML_DEFAULT_DISTANCE_THRESHOLD = 0.2
RWML_DEFAULT_EASY_KEEP_PROBABILITY = 0.1
RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD = 0.8
RWML_DEFAULT_HISTORY_WINDOW = 4

EmbedFn = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class RWMLConfig:
    """Resolved RWML knobs for a terminal-RL pre-RL stage."""

    enabled: bool = False
    distance_threshold: float = RWML_DEFAULT_DISTANCE_THRESHOLD
    easy_pass_rate_threshold: float = RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD
    easy_keep_probability: float = RWML_DEFAULT_EASY_KEEP_PROBABILITY
    history_window: int = RWML_DEFAULT_HISTORY_WINDOW
    embedding_model: str = ""
    kl_beta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "distance_threshold": self.distance_threshold,
            "easy_pass_rate_threshold": self.easy_pass_rate_threshold,
            "easy_keep_probability": self.easy_keep_probability,
            "history_window": self.history_window,
            "embedding_model": self.embedding_model,
            "kl_beta": self.kl_beta,
        }


@dataclass(frozen=True)
class WorldModelTransition:
    """One <history, action, next_state> world-model training triplet."""

    action: str
    next_state: str
    instruction: str = ""
    prior: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "prior": [list(pair) for pair in self.prior],
            "action": self.action,
            "next_state": self.next_state,
        }


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def world_model_reward(
    predicted_state: str,
    actual_state: str,
    embed_fn: EmbedFn,
    *,
    distance_threshold: float = RWML_DEFAULT_DISTANCE_THRESHOLD,
) -> float:
    """Binary embedding-space world-model reward: 1.0 if d < tau_d else 0.0."""

    distance = 1.0 - cosine_similarity(embed_fn(predicted_state), embed_fn(actual_state))
    return 1.0 if distance < distance_threshold else 0.0


def extract_transitions(
    steps: Iterable[dict[str, str]],
    *,
    instruction: str = "",
    history_window: int = RWML_DEFAULT_HISTORY_WINDOW,
) -> list[WorldModelTransition]:
    """Convert a rollout's (action, resulting state) steps into triplets.

    ``steps`` is an ordered sequence of dicts with ``action`` and ``state`` keys,
    where ``state`` is the environment observation produced by executing
    ``action``. Each transition carries up to ``history_window`` prior
    (action, state) pairs plus the originating instruction.
    """

    if history_window < 0:
        raise ValueError("history_window must be non-negative")

    pairs = [(str(step["action"]), str(step["state"])) for step in steps]
    transitions: list[WorldModelTransition] = []
    for index, (action, next_state) in enumerate(pairs):
        start = max(0, index - history_window)
        prior = tuple(pairs[start:index])
        transitions.append(
            WorldModelTransition(
                action=action,
                next_state=next_state,
                instruction=instruction,
                prior=prior,
            )
        )
    return transitions


def group_relative_advantages(rewards: Sequence[float]) -> list[float]:
    """Standardize a group's rewards: (r - mean) / std. Zero variance -> zeros."""

    values = [float(reward) for reward in rewards]
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def build_world_model_reward_fn(
    embed_fn: EmbedFn,
    *,
    distance_threshold: float = RWML_DEFAULT_DISTANCE_THRESHOLD,
) -> Callable[[Sequence[str], Sequence[str]], list[float]]:
    """Build the RWML reward function for the pre-RL GRPO stage.

    Returns ``reward_fn(predicted_states, actual_states) -> list[float]``: the
    binary embedding-space world-model reward for each (predicted, actual) next
    state pair, using the injected ``embed_fn``. The trainer backend calls this
    with the model's predicted next states and the rollout's real next states.
    """

    def reward_fn(predicted_states: Sequence[str], actual_states: Sequence[str]) -> list[float]:
        return [
            world_model_reward(predicted, actual, embed_fn, distance_threshold=distance_threshold)
            for predicted, actual in zip(predicted_states, actual_states, strict=True)
        ]

    return reward_fn


def is_easy_sample(pass_rate: float, threshold: float) -> bool:
    """A transition is "easy" when it is predicted correctly often enough."""

    return pass_rate >= threshold


def keep_easy_sample(
    pass_rate: float,
    *,
    threshold: float = RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD,
    keep_probability: float = RWML_DEFAULT_EASY_KEEP_PROBABILITY,
    draw: float,
) -> bool:
    """Decide whether to keep a transition under easy-sample sub-sampling.

    Hard transitions (``pass_rate`` below ``threshold``) are always kept. Easy
    transitions are kept only when the supplied uniform ``draw`` in [0, 1) falls
    below ``keep_probability``; ``draw`` is injected so the decision is
    deterministic and testable (the caller owns the RNG).
    """

    if not is_easy_sample(pass_rate, threshold):
        return True
    return draw < keep_probability
