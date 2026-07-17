"""Typed adapter between named BashGym rewards and NeMo RL GDPO batches.

The adapter is deliberately backend-light: it validates and binds the reward
contract before an optional NeMo RL process is started, while the reference
advantages come from BashGym's dependency-free GDPO implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bashgym.environments.contracts import RewardComponentSpec
from bashgym.gym.policy_optimization import GDPOAdvantageResult, gdpo_advantages


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: float, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class GDPOComponent:
    """One ordered reward component in a GDPO training contract."""

    name: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("GDPO reward component names must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "weight",
            _finite(self.weight, field=f"weight for reward component {name!r}"),
        )


@dataclass(frozen=True)
class NemoGDPOConfig:
    """Exact NeMo RL switches required to select the GDPO estimator."""

    name: str = "gdpo"
    normalize_rewards: bool = True
    use_leave_one_out_baseline: bool = False

    def __post_init__(self) -> None:
        if self.name != "gdpo":
            raise ValueError("NeMo GDPO advantage estimator name must be 'gdpo'")
        if self.normalize_rewards is not True:
            raise ValueError("NeMo GDPO must normalize reward components")
        if self.use_leave_one_out_baseline is not False:
            raise ValueError("NeMo GDPO must not use a leave-one-out baseline")

    def to_nested_dict(self) -> dict[str, Any]:
        """Return the shape consumed by a NeMo RL GRPO recipe."""

        return {
            "grpo": {
                "adv_estimator": {
                    "name": self.name,
                    "normalize_rewards": self.normalize_rewards,
                    "use_leave_one_out_baseline": self.use_leave_one_out_baseline,
                }
            }
        }

    def to_flat_dict(self) -> dict[str, str | bool]:
        """Return command/config override keys without losing exact semantics."""

        return {
            "grpo.adv_estimator.name": self.name,
            "grpo.adv_estimator.normalize_rewards": self.normalize_rewards,
            "grpo.adv_estimator.use_leave_one_out_baseline": (self.use_leave_one_out_baseline),
        }


@dataclass(frozen=True)
class GDPOBindingReceipt:
    """Deterministic, secret-free receipt for one validated reward binding."""

    schema_version: int
    component_order: tuple[str, ...]
    nemo_reward_keys: tuple[str, ...]
    sample_count: int
    group_count: int
    contract_digest: str
    batch_digest: str
    advantages_digest: str
    config_digest: str
    receipt_digest: str


@dataclass(frozen=True)
class NemoGDPOBatch:
    """Validated NeMo-shaped reward columns and reference GDPO advantages."""

    group_ids: tuple[str | int, ...]
    reward_columns: Mapping[str, tuple[float, ...]]
    total_reward: tuple[float, ...]
    component_order: tuple[str, ...]
    component_to_nemo_key: Mapping[str, str]
    reference_advantages: GDPOAdvantageResult
    config: NemoGDPOConfig
    receipt: GDPOBindingReceipt

    def to_nemo_batch(self) -> dict[str, list[str | int | float]]:
        """Return JSON-compatible columns using NeMo RL's reward-key convention."""

        batch: dict[str, list[str | int | float]] = {"group_ids": list(self.group_ids)}
        for key in self.component_to_nemo_key.values():
            batch[key] = list(self.reward_columns[key])
        batch["total_reward"] = list(self.total_reward)
        return batch


@dataclass(frozen=True)
class NamedRewardGDPOAdapter:
    """Bind an ordered named-reward contract to NeMo RL GDPO inputs."""

    components: tuple[GDPOComponent, ...]
    config: NemoGDPOConfig = NemoGDPOConfig()
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if len(components) < 2:
            raise ValueError("GDPO requires at least two declared reward components")
        names = [component.name for component in components]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate GDPO reward components: {duplicates}")
        if all(component.weight == 0 for component in components):
            raise ValueError("at least one GDPO reward component weight must be non-zero")
        object.__setattr__(self, "components", components)
        epsilon = _finite(self.epsilon, field="epsilon")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "epsilon", epsilon)

    @classmethod
    def from_reward_specs(
        cls,
        reward_components: Sequence[RewardComponentSpec],
        *,
        config: NemoGDPOConfig | None = None,
        epsilon: float = 1e-6,
    ) -> NamedRewardGDPOAdapter:
        """Preserve the environment contract's declared order and weights."""

        return cls(
            components=tuple(
                GDPOComponent(name=component.name, weight=component.weight)
                for component in reward_components
            ),
            config=config or NemoGDPOConfig(),
            epsilon=epsilon,
        )

    @property
    def component_order(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    @property
    def weights(self) -> dict[str, float]:
        return {component.name: component.weight for component in self.components}

    def assert_contract(
        self,
        *,
        component_order: Sequence[str],
        weights: Mapping[str, float],
    ) -> None:
        """Fail closed if a trainer-side declaration drifted from the environment."""

        observed_order = tuple(str(name).strip() for name in component_order)
        if observed_order != self.component_order:
            raise ValueError(
                "GDPO component order mismatch: "
                f"expected {self.component_order}, got {observed_order}"
            )
        if set(weights) != set(self.weights):
            raise ValueError("GDPO component weight names do not match the reward contract")
        for name, expected in self.weights.items():
            observed = _finite(weights[name], field=f"weight for reward component {name!r}")
            if observed != expected:
                raise ValueError(
                    f"GDPO weight mismatch for {name!r}: expected {expected}, got {observed}"
                )

    def bind(
        self,
        *,
        group_ids: Sequence[Hashable],
        reward_rows: Sequence[Mapping[str, float]],
        component_order: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> NemoGDPOBatch:
        """Validate rows, build NeMo columns, and calculate reference advantages."""

        if component_order is not None or weights is not None:
            if component_order is None or weights is None:
                raise ValueError("component_order and weights must be supplied together")
            self.assert_contract(component_order=component_order, weights=weights)

        if not reward_rows:
            raise ValueError("reward_rows must not be empty")
        if len(group_ids) != len(reward_rows):
            raise ValueError("group_ids and reward_rows must have the same length")

        normalized_group_ids: list[str | int] = []
        for index, group_id in enumerate(group_ids):
            if isinstance(group_id, bool) or not isinstance(group_id, (str, int)):
                raise ValueError(f"group_ids[{index}] must be a string or integer")
            if isinstance(group_id, str) and not group_id.strip():
                raise ValueError(f"group_ids[{index}] must be non-empty")
            normalized_group_ids.append(group_id)
        group_sizes = Counter(normalized_group_ids)
        singleton_groups = sorted(
            (repr(group_id) for group_id, count in group_sizes.items() if count < 2)
        )
        if singleton_groups:
            raise ValueError(
                "each GDPO group must contain at least two samples; singleton groups: "
                + ", ".join(singleton_groups)
            )

        component_values: dict[str, list[float]] = {name: [] for name in self.component_order}
        expected_names = set(self.component_order)
        totals: list[float] = []
        for row_index, row in enumerate(reward_rows):
            observed_names = set(row)
            if observed_names != expected_names:
                missing = sorted(expected_names - observed_names)
                extra = sorted(observed_names - expected_names)
                raise ValueError(
                    f"reward_rows[{row_index}] component mismatch: missing={missing}, extra={extra}"
                )
            total = 0.0
            for component in self.components:
                value = _finite(
                    row[component.name],
                    field=f"reward_rows[{row_index}][{component.name!r}]",
                )
                component_values[component.name].append(value)
                total += component.weight * value
            totals.append(_finite(total, field=f"total_reward[{row_index}]"))

        reference = gdpo_advantages(
            group_ids=normalized_group_ids,
            reward_components=component_values,
            weights=self.weights,
            epsilon=self.epsilon,
            normalize_combined=self.config.normalize_rewards,
        )
        component_to_nemo_key = {
            name: f"reward{index}" for index, name in enumerate(self.component_order, start=1)
        }
        reward_columns = {
            component_to_nemo_key[name]: tuple(component_values[name])
            for name in self.component_order
        }
        contract_payload = {
            "schema_version": 1,
            "components": [
                {"name": component.name, "weight": component.weight}
                for component in self.components
            ],
        }
        batch_payload = {
            "group_ids": normalized_group_ids,
            **{key: list(values) for key, values in reward_columns.items()},
            "total_reward": totals,
        }
        advantages_payload = {
            "advantages": list(reference.advantages),
            "combined_advantages": list(reference.combined_advantages),
            "component_advantages": {
                name: list(values)
                for name, values in sorted(reference.component_advantages.items())
            },
            "zero_variance_groups": [
                [group_id, name] for group_id, name in reference.zero_variance_groups
            ],
        }
        receipt_payload = {
            "schema_version": 1,
            "component_order": list(self.component_order),
            "nemo_reward_keys": list(reward_columns),
            "sample_count": len(reward_rows),
            "group_count": len(group_sizes),
            "contract_digest": _canonical_hash(contract_payload),
            "batch_digest": _canonical_hash(batch_payload),
            "advantages_digest": _canonical_hash(advantages_payload),
            "config_digest": _canonical_hash(self.config.to_nested_dict()),
        }
        receipt = GDPOBindingReceipt(
            **receipt_payload,
            receipt_digest=_canonical_hash(receipt_payload),
        )
        return NemoGDPOBatch(
            group_ids=tuple(normalized_group_ids),
            reward_columns=reward_columns,
            total_reward=tuple(totals),
            component_order=self.component_order,
            component_to_nemo_key=component_to_nemo_key,
            reference_advantages=reference,
            config=self.config,
            receipt=receipt,
        )


__all__ = [
    "GDPOBindingReceipt",
    "GDPOComponent",
    "NamedRewardGDPOAdapter",
    "NemoGDPOBatch",
    "NemoGDPOConfig",
]
