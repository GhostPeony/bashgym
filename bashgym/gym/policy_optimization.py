"""Dependency-free policy-optimization math shared by training adapters.

These are independent implementations of published GRPO/DAPO/GDPO formulas.
They intentionally do not import NeMo RL, Ray, Torch, TRL, or a model backend so
the same contracts can be used by BashGym-native and optional training engines.

References:
- https://docs.nvidia.com/nemo/rl/0.6.0/guides/grpo.html
- https://docs.nvidia.com/nemo/rl/0.6.0/design-docs/loss-functions.html
- https://docs.nvidia.com/nemo/rl/0.5.0/guides/dapo.html
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sample_std(values: Sequence[float], mean: float) -> float:
    """Return sample standard deviation, or zero when fewer than two values exist."""

    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


@dataclass(frozen=True)
class ClippedPolicyObjective:
    """One PPO/GRPO-style objective contribution before loss sign inversion."""

    probability_ratio: float
    unclipped_objective: float
    clipped_objective: float
    objective: float
    ratio_was_clipped: bool
    dual_clipped: bool

    @property
    def loss(self) -> float:
        """Return the minimization loss corresponding to this maximization objective."""

        return -self.objective


def clipped_policy_objective(
    *,
    current_logprob: float,
    behavior_logprob: float,
    advantage: float,
    ratio_clip_min: float = 0.2,
    ratio_clip_max: float = 0.2,
    dual_clip: float | None = None,
) -> ClippedPolicyObjective:
    """Compute a symmetric or DAPO-style asymmetric clipped policy objective.

    ``ratio_clip_min`` and ``ratio_clip_max`` are distances around 1.0, so a
    value of 0.2 produces a lower ratio bound of 0.8 or upper bound of 1.2.
    ``dual_clip`` applies only to negative advantages and must be greater than 1.
    """

    current = _finite_float(current_logprob, name="current_logprob")
    behavior = _finite_float(behavior_logprob, name="behavior_logprob")
    advantage_value = _finite_float(advantage, name="advantage")
    clip_min = _finite_float(ratio_clip_min, name="ratio_clip_min")
    clip_max = _finite_float(ratio_clip_max, name="ratio_clip_max")
    if not 0 <= clip_min < 1:
        raise ValueError("ratio_clip_min must be in [0, 1)")
    if clip_max < 0:
        raise ValueError("ratio_clip_max must be non-negative")
    if dual_clip is not None:
        dual_clip = _finite_float(dual_clip, name="dual_clip")
        if dual_clip <= 1:
            raise ValueError("dual_clip must be greater than 1")

    try:
        ratio = math.exp(current - behavior)
    except OverflowError as exc:
        raise ValueError("current_logprob - behavior_logprob is too large") from exc
    if not math.isfinite(ratio):
        raise ValueError("policy probability ratio must be finite")

    lower_bound = 1.0 - clip_min
    upper_bound = 1.0 + clip_max
    clipped_ratio = min(max(ratio, lower_bound), upper_bound)
    unclipped = ratio * advantage_value
    clipped = clipped_ratio * advantage_value
    objective = min(unclipped, clipped)
    dual_clipped = False
    if dual_clip is not None and advantage_value < 0:
        dual_bound = dual_clip * advantage_value
        if dual_bound > objective:
            objective = dual_bound
            dual_clipped = True

    return ClippedPolicyObjective(
        probability_ratio=ratio,
        unclipped_objective=unclipped,
        clipped_objective=clipped,
        objective=objective,
        ratio_was_clipped=not math.isclose(ratio, clipped_ratio, rel_tol=0.0, abs_tol=1e-15),
        dual_clipped=dual_clipped,
    )


@dataclass(frozen=True)
class GDPOAdvantageResult:
    """Per-sample GDPO advantages plus diagnostics needed by campaign evidence."""

    advantages: tuple[float, ...]
    combined_advantages: tuple[float, ...]
    component_advantages: Mapping[str, tuple[float, ...]]
    zero_variance_groups: tuple[tuple[Hashable, str], ...]


def gdpo_advantages(
    *,
    group_ids: Sequence[Hashable],
    reward_components: Mapping[str, Sequence[float]],
    weights: Mapping[str, float] | None = None,
    epsilon: float = 1e-6,
    normalize_combined: bool = True,
) -> GDPOAdvantageResult:
    """Normalize each reward per prompt group, combine it, then normalize the batch.

    This preserves the training signal from differently scaled reward components.
    A component with zero variance in one prompt group contributes zero advantage
    for that group and is reported for active-sampling diagnostics.
    """

    if not group_ids:
        raise ValueError("group_ids must not be empty")
    if len(reward_components) < 2:
        raise ValueError("GDPO requires at least two reward components")
    epsilon_value = _finite_float(epsilon, name="epsilon")
    if epsilon_value <= 0:
        raise ValueError("epsilon must be positive")

    sample_count = len(group_ids)
    component_values: dict[str, tuple[float, ...]] = {}
    for raw_name, values in reward_components.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("reward component names must be non-empty")
        if name in component_values:
            raise ValueError(f"duplicate reward component name {name!r}")
        if len(values) != sample_count:
            raise ValueError(
                f"reward component {name!r} has {len(values)} values; expected {sample_count}"
            )
        component_values[name] = tuple(
            _finite_float(value, name=f"reward_components[{name!r}]") for value in values
        )

    configured_weights = dict(weights or {})
    unknown_weights = sorted(set(configured_weights) - set(component_values))
    if unknown_weights:
        raise ValueError(f"weights contain unknown reward components: {unknown_weights}")
    resolved_weights = {
        name: _finite_float(configured_weights.get(name, 1.0), name=f"weights[{name!r}]")
        for name in component_values
    }

    group_indices: dict[Hashable, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        try:
            group_indices.setdefault(group_id, []).append(index)
        except TypeError as exc:
            raise ValueError("group_ids must contain hashable values") from exc

    normalized_components: dict[str, list[float]] = {
        name: [0.0] * sample_count for name in component_values
    }
    zero_variance: list[tuple[Hashable, str]] = []
    for name, values in component_values.items():
        for group_id, indices in group_indices.items():
            group_rewards = [values[index] for index in indices]
            group_mean = sum(group_rewards) / len(group_rewards)
            group_std = _sample_std(group_rewards, group_mean)
            if group_std == 0:
                zero_variance.append((group_id, name))
            denominator = group_std + epsilon_value
            for index in indices:
                normalized_components[name][index] = (values[index] - group_mean) / denominator

    combined = [
        sum(
            resolved_weights[name] * normalized_components[name][index]
            for name in normalized_components
        )
        for index in range(sample_count)
    ]
    advantages = list(combined)
    if normalize_combined:
        batch_mean = sum(combined) / len(combined)
        batch_std = _sample_std(combined, batch_mean)
        denominator = batch_std + epsilon_value
        advantages = [(value - batch_mean) / denominator for value in combined]

    return GDPOAdvantageResult(
        advantages=tuple(advantages),
        combined_advantages=tuple(combined),
        component_advantages={
            name: tuple(values) for name, values in normalized_components.items()
        },
        zero_variance_groups=tuple(zero_variance),
    )


@dataclass(frozen=True)
class RewardShapingResult:
    """Result of DAPO-style overlong reward shaping for one response."""

    original_reward: float
    shaped_reward: float
    penalty: float
    response_tokens: int
    penalty_start_tokens: int


def dapo_overlong_reward(
    *,
    reward: float,
    response_tokens: int,
    max_response_tokens: int,
    overlong_buffer_tokens: int,
    max_penalty: float = 1.0,
) -> RewardShapingResult:
    """Apply the linear DAPO penalty inside the configured response-length buffer."""

    original_reward = _finite_float(reward, name="reward")
    penalty_scale = _finite_float(max_penalty, name="max_penalty")
    if response_tokens < 0:
        raise ValueError("response_tokens must be non-negative")
    if max_response_tokens <= 0:
        raise ValueError("max_response_tokens must be positive")
    if not 0 < overlong_buffer_tokens <= max_response_tokens:
        raise ValueError("overlong_buffer_tokens must be in (0, max_response_tokens]")
    if response_tokens > max_response_tokens:
        raise ValueError("response_tokens must not exceed max_response_tokens")
    if penalty_scale < 0:
        raise ValueError("max_penalty must be non-negative")

    penalty_start = max_response_tokens - overlong_buffer_tokens
    excess_tokens = max(response_tokens - penalty_start, 0)
    penalty = -(excess_tokens / overlong_buffer_tokens) * penalty_scale
    return RewardShapingResult(
        original_reward=original_reward,
        shaped_reward=original_reward + penalty,
        penalty=penalty,
        response_tokens=response_tokens,
        penalty_start_tokens=penalty_start,
    )


def global_normalized_microbatch_sum(
    values: Sequence[float],
    mask: Sequence[bool | int],
    *,
    total_valid_items: int,
) -> float:
    """Return a microbatch loss contribution normalized by the global valid count.

    Summing this result across microbatches is equivalent to taking one masked
    mean over the unsplit global batch. ``total_valid_items`` can represent valid
    tokens for token-level losses or valid sequences for sequence-level losses.
    """

    if len(values) != len(mask):
        raise ValueError("values and mask must have the same length")
    if total_valid_items <= 0:
        raise ValueError("total_valid_items must be positive")
    normalized_mask: list[int] = []
    for item in mask:
        if item not in (0, 1, False, True):
            raise ValueError("mask values must be boolean or 0/1")
        normalized_mask.append(int(bool(item)))
    if sum(normalized_mask) > total_valid_items:
        raise ValueError("microbatch valid count cannot exceed total_valid_items")

    finite_values = [
        _finite_float(value, name=f"values[{index}]") for index, value in enumerate(values)
    ]
    numerator = sum(
        value for value, include in zip(finite_values, normalized_mask, strict=True) if include
    )
    return numerator / total_valid_items
