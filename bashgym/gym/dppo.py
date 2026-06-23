"""DPPO trust-region primitives for terminal-agent RL.

This module deliberately implements the math/telemetry layer before any trainer
backend. DPPO is only useful for BashGym once rollout behavior logprobs and
trainer logprobs can be compared against the same sampled tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DPPO_BINARY_TV_THRESHOLD = 0.15
DPPO_BINARY_KL_THRESHOLD = 0.05
DPPO_COLLAPSE_MASKED_FRACTION = 0.95
DPPO_COLLAPSE_MIN_TOKENS = 200
MIN_PROBABILITY = 1e-12
MAX_LOGPROB = 0.0

DPPODivergence = Literal["binary_tv", "binary_kl"]


def probability_from_logprob(logprob: float) -> float:
    """Convert a logprob into a safe probability for divergence math."""

    if not math.isfinite(logprob):
        raise ValueError("logprob must be finite")
    return min(1.0 - MIN_PROBABILITY, max(MIN_PROBABILITY, math.exp(min(logprob, MAX_LOGPROB))))


def binary_tv_divergence(behavior_logprob: float, train_logprob: float) -> float:
    """Binary-TV divergence for the partition {sampled token, all other tokens}."""

    behavior_probability = probability_from_logprob(behavior_logprob)
    train_probability = probability_from_logprob(train_logprob)
    return abs(train_probability - behavior_probability)


def binary_kl_divergence(behavior_logprob: float, train_logprob: float) -> float:
    """Binary KL divergence anchored to the rollout/behavior policy."""

    behavior_probability = probability_from_logprob(behavior_logprob)
    train_probability = probability_from_logprob(train_logprob)
    behavior_tail = 1.0 - behavior_probability
    train_tail = 1.0 - train_probability
    return behavior_probability * math.log(behavior_probability / train_probability) + (
        behavior_tail * math.log(behavior_tail / train_tail)
    )


def policy_ratio(behavior_logprob: float, train_logprob: float) -> float:
    """Sample-token probability ratio used for comparison telemetry."""

    if not math.isfinite(behavior_logprob) or not math.isfinite(train_logprob):
        raise ValueError("logprobs must be finite")
    return math.exp(train_logprob - behavior_logprob)


@dataclass(frozen=True)
class DPPOToken:
    """One sampled token with behavior and train-policy logprobs."""

    behavior_logprob: float
    train_logprob: float
    advantage: float
    token: str | None = None


@dataclass(frozen=True)
class DPPOMaskDecision:
    """Trust-region mask decision for one sampled token."""

    token: str | None
    advantage: float
    ratio: float
    binary_tv: float
    binary_kl: float
    abs_logprob_diff: float
    abs_policy_mismatch: float
    masked: bool
    reason: str


@dataclass(frozen=True)
class DPPOTelemetry:
    """Batch-level DPPO validation metrics."""

    n_tokens: int
    masked_updates: int
    masked_fraction: float
    mean_binary_tv: float
    max_binary_tv: float
    mean_binary_kl: float
    max_binary_kl: float
    mean_abs_logprob_diff: float
    max_abs_logprob_diff: float
    mean_abs_policy_mismatch: float
    max_abs_policy_mismatch: float
    collapse_warning: bool
    divergence: DPPODivergence
    threshold: float

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "n_tokens": self.n_tokens,
            "masked_updates": self.masked_updates,
            "masked_fraction": self.masked_fraction,
            "mean_binary_tv": self.mean_binary_tv,
            "max_binary_tv": self.max_binary_tv,
            "mean_binary_kl": self.mean_binary_kl,
            "max_binary_kl": self.max_binary_kl,
            "mean_abs_logprob_diff": self.mean_abs_logprob_diff,
            "max_abs_logprob_diff": self.max_abs_logprob_diff,
            "mean_abs_policy_mismatch": self.mean_abs_policy_mismatch,
            "max_abs_policy_mismatch": self.max_abs_policy_mismatch,
            "collapse_warning": self.collapse_warning,
            "divergence": self.divergence,
            "threshold": self.threshold,
        }


def dppo_mask_decision(
    token: DPPOToken,
    *,
    divergence: DPPODivergence = "binary_tv",
    threshold: float | None = None,
) -> DPPOMaskDecision:
    """Return whether DPPO would mask this sampled-token update.

    The trust region is anchored to behavior logprobs from rollout collection,
    not a recomputed reference policy. Positive-advantage tokens are masked only
    when the train policy has already increased them beyond the divergence
    threshold; negative-advantage tokens are masked only when it has already
    decreased them too far.
    """

    if divergence == "binary_tv":
        active_threshold = DPPO_BINARY_TV_THRESHOLD if threshold is None else threshold
    elif divergence == "binary_kl":
        active_threshold = DPPO_BINARY_KL_THRESHOLD if threshold is None else threshold
    else:
        raise ValueError(f"unsupported DPPO divergence: {divergence}")
    if active_threshold < 0:
        raise ValueError("threshold must be non-negative")

    behavior_probability = probability_from_logprob(token.behavior_logprob)
    train_probability = probability_from_logprob(token.train_logprob)
    tv = abs(train_probability - behavior_probability)
    kl = binary_kl_divergence(token.behavior_logprob, token.train_logprob)
    score = tv if divergence == "binary_tv" else kl
    update_moves_probability_up = train_probability > behavior_probability
    update_moves_probability_down = train_probability < behavior_probability
    positive_mask = token.advantage > 0 and update_moves_probability_up and score > active_threshold
    negative_mask = token.advantage < 0 and update_moves_probability_down and score > active_threshold
    masked = positive_mask or negative_mask
    if positive_mask:
        reason = "positive_advantage_above_trust_region"
    elif negative_mask:
        reason = "negative_advantage_above_trust_region"
    else:
        reason = "inside_trust_region"

    return DPPOMaskDecision(
        token=token.token,
        advantage=token.advantage,
        ratio=policy_ratio(token.behavior_logprob, token.train_logprob),
        binary_tv=tv,
        binary_kl=kl,
        abs_logprob_diff=abs(token.train_logprob - token.behavior_logprob),
        abs_policy_mismatch=abs(train_probability - behavior_probability),
        masked=masked,
        reason=reason,
    )


def analyze_dppo_batch(
    tokens: list[DPPOToken],
    *,
    divergence: DPPODivergence = "binary_tv",
    threshold: float | None = None,
    collapse_min_tokens: int = DPPO_COLLAPSE_MIN_TOKENS,
    collapse_masked_fraction: float = DPPO_COLLAPSE_MASKED_FRACTION,
) -> tuple[list[DPPOMaskDecision], DPPOTelemetry]:
    """Analyze DPPO trust-region telemetry for a sampled-token batch."""

    if not tokens:
        raise ValueError("tokens must not be empty")
    decisions = [
        dppo_mask_decision(token, divergence=divergence, threshold=threshold) for token in tokens
    ]
    masked_updates = sum(1 for decision in decisions if decision.masked)
    masked_fraction = masked_updates / len(decisions)
    active_threshold = (
        (DPPO_BINARY_TV_THRESHOLD if divergence == "binary_tv" else DPPO_BINARY_KL_THRESHOLD)
        if threshold is None
        else threshold
    )

    telemetry = DPPOTelemetry(
        n_tokens=len(decisions),
        masked_updates=masked_updates,
        masked_fraction=masked_fraction,
        mean_binary_tv=_mean([decision.binary_tv for decision in decisions]),
        max_binary_tv=max(decision.binary_tv for decision in decisions),
        mean_binary_kl=_mean([decision.binary_kl for decision in decisions]),
        max_binary_kl=max(decision.binary_kl for decision in decisions),
        mean_abs_logprob_diff=_mean([decision.abs_logprob_diff for decision in decisions]),
        max_abs_logprob_diff=max(decision.abs_logprob_diff for decision in decisions),
        mean_abs_policy_mismatch=_mean([decision.abs_policy_mismatch for decision in decisions]),
        max_abs_policy_mismatch=max(decision.abs_policy_mismatch for decision in decisions),
        collapse_warning=(
            len(decisions) >= collapse_min_tokens and masked_fraction >= collapse_masked_fraction
        ),
        divergence=divergence,
        threshold=active_threshold,
    )
    return decisions, telemetry


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
