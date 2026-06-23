"""Terminal-agent RL recipe helpers.

The profile constants capture the TMax/DAPO-style stability defaults BashGym
uses for terminal environments. The active-sampling helpers are pure so the
rollout and trainer backends can share the same variance filtering semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from typing import Any

DEFAULT_TRAINING_PROFILE = "default"
TERMINAL_RL_TMAX_LIKE_PROFILE = "terminal_rl_tmax_like"


@dataclass(frozen=True)
class TerminalRLDefaults:
    """Default knobs for a TMax-like terminal-agent RL run."""

    training_profile: str = TERMINAL_RL_TMAX_LIKE_PROFILE
    grpo_group_size: int = 32
    prompts_per_rollout_batch: int = 8
    max_tool_calls_per_episode: int = 64
    grpo_loss_type: str = "dapo"
    token_level_loss: bool = True
    filter_zero_std_groups: bool = True
    active_sampling: bool = True
    lm_head_fp32: bool = True
    interleaved_thinking: bool = True
    sft_warm_start_policy: str = "weak_models_only"


TMAX_LIKE_DEFAULTS = TerminalRLDefaults()


def normalize_training_profile(profile: str | None) -> str:
    """Normalize an API/UI profile string into a stable internal key."""

    normalized = (profile or DEFAULT_TRAINING_PROFILE).strip().lower()
    if normalized in {"", "none", DEFAULT_TRAINING_PROFILE}:
        return DEFAULT_TRAINING_PROFILE
    if normalized in {"tmax", "tmax_like", "terminal_rl", TERMINAL_RL_TMAX_LIKE_PROFILE}:
        return TERMINAL_RL_TMAX_LIKE_PROFILE
    raise ValueError(
        f"training_profile={profile!r} must be one of "
        f"{[DEFAULT_TRAINING_PROFILE, TERMINAL_RL_TMAX_LIKE_PROFILE]}"
    )


def is_terminal_rl_profile(profile: str | None) -> bool:
    """Return whether a profile string resolves to the terminal-RL recipe."""

    return normalize_training_profile(profile) == TERMINAL_RL_TMAX_LIKE_PROFILE


def reward_group_std(rewards: Iterable[float]) -> float:
    """Population standard deviation for a completion group's rewards."""

    values = [float(reward) for reward in rewards]
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((reward - mean) ** 2 for reward in values) / len(values)
    return sqrt(variance)


@dataclass(frozen=True)
class RewardGroup:
    """Rewards sampled for one prompt group, plus optional caller payload."""

    prompt_id: str
    rewards: tuple[float, ...]
    payload: Any = None

    @property
    def std(self) -> float:
        return reward_group_std(self.rewards)

    def has_variance(self, epsilon: float = 1e-8) -> bool:
        return self.std > epsilon

    def is_all_zero(self, epsilon: float = 1e-8) -> bool:
        return all(abs(reward) <= epsilon for reward in self.rewards)

    def is_all_one(self, epsilon: float = 1e-8) -> bool:
        return all(abs(reward - 1.0) <= epsilon for reward in self.rewards)


@dataclass(frozen=True)
class ActiveSamplingResult:
    """Selection report for zero-std filtering plus refill-by-sampling."""

    selected: tuple[RewardGroup, ...]
    dropped: tuple[RewardGroup, ...]
    requested_groups: int
    candidate_groups: int

    @property
    def effective_groups(self) -> int:
        return len(self.selected)

    @property
    def zero_std_groups_dropped(self) -> int:
        return len(self.dropped)

    @property
    def all_zero_groups_dropped(self) -> int:
        return sum(1 for group in self.dropped if group.is_all_zero())

    @property
    def all_one_groups_dropped(self) -> int:
        return sum(1 for group in self.dropped if group.is_all_one())

    @property
    def active_sampling_refills(self) -> int:
        return max(0, len(self.selected) + len(self.dropped) - self.requested_groups)

    @property
    def maintained_batch(self) -> bool:
        return self.effective_groups == self.requested_groups

    def telemetry(self) -> dict[str, int | bool]:
        """Return frontend/log-friendly counters."""

        return {
            "active_sampling_refills": self.active_sampling_refills,
            "zero_std_groups_dropped": self.zero_std_groups_dropped,
            "all_zero_groups_dropped": self.all_zero_groups_dropped,
            "all_one_groups_dropped": self.all_one_groups_dropped,
            "effective_prompt_groups": self.effective_groups,
            "requested_prompt_groups": self.requested_groups,
            "candidate_prompt_groups": self.candidate_groups,
            "maintained_batch": self.maintained_batch,
        }


def active_sample_groups(
    candidate_groups: Iterable[RewardGroup],
    *,
    target_groups: int,
    epsilon: float = 1e-8,
) -> ActiveSamplingResult:
    """Drop zero-variance groups while refilling from later candidates.

    Candidate groups are examined in order. Non-zero-std groups are kept until
    ``target_groups`` is reached; zero-std groups encountered before that point
    are dropped and counted as refills.
    """

    if target_groups < 1:
        raise ValueError("target_groups must be >= 1")

    candidates = tuple(candidate_groups)
    selected: list[RewardGroup] = []
    dropped: list[RewardGroup] = []
    for group in candidates:
        if len(selected) >= target_groups:
            break
        if group.has_variance(epsilon):
            selected.append(group)
        else:
            dropped.append(group)

    return ActiveSamplingResult(
        selected=tuple(selected),
        dropped=tuple(dropped),
        requested_groups=target_groups,
        candidate_groups=len(candidates),
    )
