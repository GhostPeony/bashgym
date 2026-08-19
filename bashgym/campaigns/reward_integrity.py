"""Bounded reward-design evidence for verifier-based AutoResearch methods.

The executable verifier remains installation-owned.  This module projects its
named reward components, aggregate rollout distributions, hard-constraint
violations, and existing exploit-canary results into agent-safe evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import FrozenContractModel, HexDigest, Identifier, canonical_hash
from bashgym.environments.contracts import EnvironmentSpec

_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class RewardConstraint(FrozenContractModel):
    schema_version: Literal["bashgym.reward_constraint.v1"] = "bashgym.reward_constraint.v1"
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def finite_ordered_bound(self) -> RewardConstraint:
        if self.minimum is None and self.maximum is None:
            raise ValueError("reward constraint requires at least one bound")
        if any(
            value is not None and not math.isfinite(value) for value in (self.minimum, self.maximum)
        ):
            raise ValueError("reward constraint bounds must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("reward constraint minimum cannot exceed maximum")
        return self


class RewardComponentContract(FrozenContractModel):
    name: Identifier
    weight: float
    hard_constraint: RewardConstraint | None = None

    @field_validator("weight")
    @classmethod
    def finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reward weight must be finite")
        return value


class AutoResearchRewardSpec(FrozenContractModel):
    schema_version: Literal["bashgym.autoresearch_reward_spec.v1"] = (
        "bashgym.autoresearch_reward_spec.v1"
    )
    environment_id: Identifier
    verifier_digest: HexDigest
    components: tuple[RewardComponentContract, ...] = Field(min_length=1, max_length=100)
    reward_spec_digest: str = ""

    @model_validator(mode="after")
    def exact_digest_and_unique_components(self) -> AutoResearchRewardSpec:
        names = tuple(item.name for item in self.components)
        if len(names) != len(set(names)):
            raise ValueError("reward spec component names must be unique")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"reward_spec_digest"}))
        if self.reward_spec_digest and self.reward_spec_digest != expected:
            raise ValueError("reward spec digest mismatch")
        if not self.reward_spec_digest:
            object.__setattr__(self, "reward_spec_digest", expected)
        if not _HEX_DIGEST.fullmatch(self.reward_spec_digest):
            raise ValueError("reward spec digest must be sha256")
        return self


class RewardComponentDistribution(FrozenContractModel):
    name: Identifier
    count: int = Field(ge=1)
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float = Field(ge=0)
    hard_constraint_violations: int = Field(ge=0)
    hard_constraint_violation_rate: float = Field(ge=0, le=1)

    @field_validator("minimum", "maximum", "mean", "standard_deviation")
    @classmethod
    def finite_statistics(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reward component statistics must be finite")
        return value

    @model_validator(mode="after")
    def consistent_distribution(self) -> RewardComponentDistribution:
        if not self.minimum <= self.mean <= self.maximum:
            raise ValueError("reward component mean must lie within its range")
        if self.hard_constraint_violations > self.count:
            raise ValueError("reward component violations cannot exceed count")
        expected = self.hard_constraint_violations / self.count
        if not math.isclose(
            self.hard_constraint_violation_rate,
            expected,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("reward component violation rate does not match count")
        return self


class RewardCanaryEvidence(FrozenContractModel):
    schema_version: Literal["bashgym.reward_canary_evidence.v1"] = (
        "bashgym.reward_canary_evidence.v1"
    )
    total: int = Field(ge=0)
    guarded: int = Field(ge=0)
    failed: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    categories: tuple[Identifier, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def consistent_counts(self) -> RewardCanaryEvidence:
        if self.guarded + self.failed != self.total:
            raise ValueError("reward canary counts do not match total")
        expected = self.failed / self.total if self.total else 0.0
        if not math.isclose(self.failure_rate, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("reward canary failure rate does not match counts")
        if tuple(sorted(set(self.categories))) != self.categories:
            raise ValueError("reward canary categories must be sorted and unique")
        return self


class AutoResearchRewardIntegrityEvidence(FrozenContractModel):
    schema_version: Literal["bashgym.autoresearch_reward_integrity.v1"] = (
        "bashgym.autoresearch_reward_integrity.v1"
    )
    reward_spec: AutoResearchRewardSpec
    rollout_count: int = Field(ge=1)
    components: tuple[RewardComponentDistribution, ...] = Field(min_length=1, max_length=100)
    hard_constraint_violations: int = Field(ge=0)
    hard_constraint_observations: int = Field(ge=0)
    hard_constraint_violation_rate: float = Field(ge=0, le=1)
    canaries: RewardCanaryEvidence

    @model_validator(mode="after")
    def consistent_aggregate(self) -> AutoResearchRewardIntegrityEvidence:
        spec_names = tuple(item.name for item in self.reward_spec.components)
        observed_names = tuple(item.name for item in self.components)
        if observed_names != spec_names:
            raise ValueError("reward integrity components do not match reward spec")
        if any(item.count != self.rollout_count for item in self.components):
            raise ValueError("reward integrity component counts do not match rollouts")
        expected_violations = sum(item.hard_constraint_violations for item in self.components)
        expected_observations = self.rollout_count * sum(
            1 for item in self.reward_spec.components if item.hard_constraint is not None
        )
        if (
            self.hard_constraint_violations != expected_violations
            or self.hard_constraint_observations != expected_observations
        ):
            raise ValueError("reward integrity hard-constraint totals are inconsistent")
        expected_rate = (
            expected_violations / expected_observations if expected_observations else 0.0
        )
        if not math.isclose(
            self.hard_constraint_violation_rate,
            expected_rate,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("reward integrity hard-constraint rate is inconsistent")
        return self

    def method_evidence(self) -> dict[str, bool | int | float]:
        """Return only the scalar measurements consumed by method readiness."""

        return {
            "reward_spec_verified": True,
            "reward_canary_cases": self.canaries.total,
            "reward_canary_failure_rate": self.canaries.failure_rate,
            "hard_constraint_violation_rate": self.hard_constraint_violation_rate,
        }


def build_reward_spec(environment: EnvironmentSpec) -> AutoResearchRewardSpec:
    errors = environment.validation_errors()
    if errors:
        raise ValueError("invalid reward environment: " + "; ".join(errors))
    if not environment.verifier.reward_components:
        raise ValueError("reward environment requires named reward components")
    return AutoResearchRewardSpec(
        environment_id=environment.id,
        verifier_digest=canonical_hash(environment.verifier.to_dict()),
        components=tuple(
            RewardComponentContract(
                name=item.name,
                weight=item.weight,
                hard_constraint=(
                    RewardConstraint(
                        minimum=item.hard_constraint.minimum,
                        maximum=item.hard_constraint.maximum,
                    )
                    if item.hard_constraint is not None
                    else None
                ),
            )
            for item in environment.verifier.reward_components
        ),
    )


def _canary_evidence(summary: Mapping[str, Any]) -> RewardCanaryEvidence:
    try:
        total = int(summary["total"])
        guarded = int(summary["guarded"])
        failed = int(summary["failed"])
        guard_rate = float(summary["guard_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reward canary summary is incomplete") from exc
    expected_guard_rate = guarded / total if total else 0.0
    if (
        total < 0
        or guarded < 0
        or failed < 0
        or guarded + failed != total
        or not math.isfinite(guard_rate)
        or not math.isclose(guard_rate, expected_guard_rate, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("reward canary summary is inconsistent")
    categories_value = summary.get("categories") or {}
    if not isinstance(categories_value, Mapping):
        raise ValueError("reward canary summary categories are invalid")
    categories = tuple(sorted(str(key) for key in categories_value))
    return RewardCanaryEvidence(
        total=total,
        guarded=guarded,
        failed=failed,
        failure_rate=failed / total if total else 0.0,
        categories=categories,
    )


def build_reward_integrity_evidence(
    environment: EnvironmentSpec,
    *,
    reward_rows: Sequence[Mapping[str, float]],
    canary_summary: Mapping[str, Any],
) -> AutoResearchRewardIntegrityEvidence:
    """Project bounded reward integrity evidence from exact rollout components."""

    if not reward_rows:
        raise ValueError("reward integrity evidence requires at least one rollout")
    spec = build_reward_spec(environment)
    expected = {item.name for item in spec.components}
    values: dict[str, list[float]] = {name: [] for name in expected}
    for row in reward_rows:
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"reward component mismatch: missing={missing}, unexpected={unexpected}"
            )
        for name in expected:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError("reward component values must be finite")
            values[name].append(value)

    distributions: list[RewardComponentDistribution] = []
    total_violations = 0
    constrained_observations = 0
    for component in spec.components:
        observed = values[component.name]
        mean = sum(observed) / len(observed)
        std = math.sqrt(sum((item - mean) ** 2 for item in observed) / len(observed))
        constraint = component.hard_constraint
        violations = 0
        if constraint is not None:
            violations = sum(
                1
                for item in observed
                if (constraint.minimum is not None and item < constraint.minimum)
                or (constraint.maximum is not None and item > constraint.maximum)
            )
            constrained_observations += len(observed)
            total_violations += violations
        distributions.append(
            RewardComponentDistribution(
                name=component.name,
                count=len(observed),
                minimum=min(observed),
                maximum=max(observed),
                mean=mean,
                standard_deviation=std,
                hard_constraint_violations=violations,
                hard_constraint_violation_rate=violations / len(observed),
            )
        )

    return AutoResearchRewardIntegrityEvidence(
        reward_spec=spec,
        rollout_count=len(reward_rows),
        components=tuple(distributions),
        hard_constraint_violations=total_violations,
        hard_constraint_observations=constrained_observations,
        hard_constraint_violation_rate=(
            total_violations / constrained_observations if constrained_observations else 0.0
        ),
        canaries=_canary_evidence(canary_summary),
    )


__all__ = [
    "AutoResearchRewardIntegrityEvidence",
    "AutoResearchRewardSpec",
    "RewardCanaryEvidence",
    "RewardComponentContract",
    "RewardComponentDistribution",
    "RewardConstraint",
    "build_reward_integrity_evidence",
    "build_reward_spec",
]
