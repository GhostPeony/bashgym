"""Persisted advisory evidence thresholds for AutoResearch method selection."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import FrozenContractModel


class AutoResearchMethodThresholds(FrozenContractModel):
    """Evidence criteria supplied by campaign setup, never method authorization."""

    schema_version: Literal["autoresearch_method_thresholds.v1"] = (
        "autoresearch_method_thresholds.v1"
    )
    min_demonstration_examples: int | None = Field(default=None, ge=1)
    min_target_slice_coverage: float | None = Field(default=None, ge=0, le=1)
    max_contamination_rate: float | None = Field(default=None, ge=0, le=1)
    min_preference_pairs: int | None = Field(default=None, ge=1)
    min_preference_agreement_lower_bound: float | None = Field(default=None, ge=0, le=1)
    max_ambiguous_pair_rate: float | None = Field(default=None, ge=0, le=1)
    max_preference_position_bias_rate: float | None = Field(default=None, ge=0, le=1)
    max_preference_label_conflict_rate: float | None = Field(default=None, ge=0, le=1)
    max_preference_contamination_rate: float | None = Field(default=None, ge=0, le=1)
    min_rollout_groups: int | None = Field(default=None, ge=1)
    min_rollout_success_rate: float | None = Field(default=None, ge=0, le=1)
    max_rollout_success_rate: float | None = Field(default=None, ge=0, le=1)
    max_zero_std_group_fraction: float | None = Field(default=None, ge=0, le=1)
    max_verifier_error_rate: float | None = Field(default=None, ge=0, le=1)
    min_reward_canary_cases: int | None = Field(default=None, ge=1)
    max_reward_canary_failure_rate: float | None = Field(default=None, ge=0, le=1)
    max_hard_constraint_violation_rate: float | None = Field(default=None, ge=0, le=1)
    min_teacher_metric_gap: float | None = None
    min_teacher_output_acceptance_rate: float | None = Field(default=None, ge=0, le=1)
    min_recovery_traces: int | None = Field(default=None, ge=1)
    min_recovery_lift_lower_bound: float | None = None

    @field_validator("min_teacher_metric_gap", "min_recovery_lift_lower_bound")
    @classmethod
    def finite_unbounded_thresholds(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("method threshold must be finite")
        return value

    @model_validator(mode="after")
    def ordered_rollout_success(self) -> AutoResearchMethodThresholds:
        if (
            self.min_rollout_success_rate is not None
            and self.max_rollout_success_rate is not None
            and self.min_rollout_success_rate > self.max_rollout_success_rate
        ):
            raise ValueError("minimum rollout success cannot exceed maximum")
        return self


__all__ = ["AutoResearchMethodThresholds"]
