"""Typed, evaluator-authored experiment-power evidence for AutoResearch."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, model_validator

from bashgym.campaigns.contracts import FrozenContractModel, HexDigest, Identifier

AUTORESEARCH_EVALUATION_POWER_KEY = "autoresearch_evaluation_power"
_MAX_SAMPLE_COUNT = 100_000_000


class AutoResearchSequentialStoppingEvidence(FrozenContractModel):
    """A predeclared sequential design result produced by the fixed evaluator."""

    schema_version: Literal["bashgym.autoresearch_sequential_stopping.v1"] = (
        "bashgym.autoresearch_sequential_stopping.v1"
    )
    plan_digest: HexDigest
    method: Identifier
    looks_completed: int = Field(ge=1, le=100_000)
    maximum_sample_count: int = Field(ge=1, le=_MAX_SAMPLE_COUNT)
    stopping_reason: Literal[
        "precision_reached",
        "effect_detected",
        "futility",
        "maximum_samples_reached",
    ]


class AutoResearchEvaluationPowerEvidence(FrozenContractModel):
    """Bounded statistical evidence carried by one exact evaluation result."""

    schema_version: Literal["bashgym.autoresearch_evaluation_power.v1"] = (
        "bashgym.autoresearch_evaluation_power.v1"
    )
    sample_count: int = Field(ge=1, le=_MAX_SAMPLE_COUNT)
    comparison_design: Literal["paired", "unpaired", "single_model"]
    uncertainty_method: Identifier
    confidence_level: float | None = Field(default=None, gt=0.0, lt=1.0)
    interval_lower: float | None = None
    interval_upper: float | None = None
    maximum_interval_width: float | None = Field(default=None, gt=0.0)
    minimum_detectable_effect: float | None = Field(default=None, ge=0.0)
    target_power: float | None = Field(default=None, gt=0.0, le=1.0)
    estimated_power: float | None = Field(default=None, ge=0.0, le=1.0)
    sequential_stopping: AutoResearchSequentialStoppingEvidence | None = None

    @model_validator(mode="after")
    def validate_statistical_contract(self) -> AutoResearchEvaluationPowerEvidence:
        numeric = (
            self.confidence_level,
            self.interval_lower,
            self.interval_upper,
            self.maximum_interval_width,
            self.minimum_detectable_effect,
            self.target_power,
            self.estimated_power,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("experiment power evidence must contain finite values")
        interval_fields = (self.interval_lower, self.interval_upper)
        if (interval_fields[0] is None) != (interval_fields[1] is None):
            raise ValueError("experiment power interval bounds must be supplied together")
        if self.interval_lower is not None:
            if self.confidence_level is None:
                raise ValueError("experiment power interval requires a confidence level")
            if self.interval_lower > self.interval_upper:  # type: ignore[operator]
                raise ValueError("experiment power interval bounds are reversed")
        if self.maximum_interval_width is not None and self.interval_lower is None:
            raise ValueError("maximum interval width requires observed interval bounds")
        if (self.target_power is None) != (self.estimated_power is None):
            raise ValueError("target and estimated power must be supplied together")
        sequential = self.sequential_stopping
        if sequential is not None and sequential.maximum_sample_count < self.sample_count:
            raise ValueError("sequential maximum sample count is below the observed sample count")
        return self


def _positive_count(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer")
    if not 1 <= value <= _MAX_SAMPLE_COUNT:
        raise ValueError(f"{field_name} is outside the supported range")
    return value


def _resolved_evaluation(
    outcome: Mapping[str, Any], evaluations: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        return None
    references = result.get("evidence_references")
    if isinstance(references, (str, bytes, bytearray)) or not isinstance(references, Sequence):
        return None
    by_id = {
        str(item.get("evaluation_result_id")): item
        for item in evaluations
        if item.get("evaluation_result_id")
    }
    for reference in references:
        evaluation = by_id.get(str(reference))
        if evaluation is not None:
            return evaluation
    return None


def _evaluation_projection(
    evaluation: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], AutoResearchEvaluationPowerEvidence | None]:
    result_id = str(evaluation.get("evaluation_result_id") or "") if evaluation else None
    slices = evaluation.get("slice_metrics") if evaluation else None
    slices = slices if isinstance(slices, Mapping) else {}
    example_count = _positive_count(
        slices.get("example_count"), field_name="slice_metrics.example_count"
    )
    sample_count = _positive_count(
        slices.get("sample_count"), field_name="slice_metrics.sample_count"
    )
    if example_count is not None and sample_count is not None and example_count != sample_count:
        raise ValueError("evaluation sample count fields disagree")
    observed_count = example_count if example_count is not None else sample_count
    observed_source = (
        "slice_metrics.example_count"
        if example_count is not None
        else "slice_metrics.sample_count" if sample_count is not None else None
    )

    raw_evidence = slices.get(AUTORESEARCH_EVALUATION_POWER_KEY)
    evidence = (
        AutoResearchEvaluationPowerEvidence.model_validate(raw_evidence)
        if raw_evidence is not None
        else None
    )
    if evidence is not None:
        if observed_count is not None and observed_count != evidence.sample_count:
            raise ValueError("evaluation power sample count does not match slice metrics")
        observed_count = evidence.sample_count
        observed_source = f"slice_metrics.{AUTORESEARCH_EVALUATION_POWER_KEY}.sample_count"

    criteria: list[dict[str, Any]] = []
    uncertainty = None
    comparison_design = None
    if evidence is not None:
        comparison_design = evidence.comparison_design
        uncertainty = {
            "method": evidence.uncertainty_method,
            "confidence_level": evidence.confidence_level,
            "interval_lower": evidence.interval_lower,
            "interval_upper": evidence.interval_upper,
            "minimum_detectable_effect": evidence.minimum_detectable_effect,
        }
        if evidence.maximum_interval_width is not None:
            interval_width = evidence.interval_upper - evidence.interval_lower  # type: ignore[operator]
            criteria.append(
                {
                    "criterion": "maximum_interval_width",
                    "observed": interval_width,
                    "target": evidence.maximum_interval_width,
                    "passed": interval_width <= evidence.maximum_interval_width,
                }
            )
        if evidence.target_power is not None:
            criteria.append(
                {
                    "criterion": "target_power",
                    "observed": evidence.estimated_power,
                    "target": evidence.target_power,
                    "passed": evidence.estimated_power >= evidence.target_power,  # type: ignore[operator]
                }
            )

    sufficiency_status = (
        "unavailable"
        if observed_count is None
        else (
            "sufficient"
            if criteria and all(item["passed"] for item in criteria)
            else "insufficient" if criteria else "not_assessed"
        )
    )
    return (
        {
            "evaluation_result_id": result_id,
            "sample_count": observed_count,
            "sample_count_source": observed_source,
            "comparison_design": comparison_design,
            "uncertainty": uncertainty,
            "sufficiency": {
                "status": sufficiency_status,
                "criteria": criteria,
            },
        },
        evidence,
    )


def _seed_uncertainty(hypothesis_family: Mapping[str, Any] | None) -> dict[str, Any]:
    if hypothesis_family is None:
        status = "not_grouped"
        completed = 0
        distinct_seeds = 0
        summary: Mapping[str, Any] = {}
    else:
        status = (
            "replicated"
            if hypothesis_family.get("status") == "replicated"
            else ("single_observation")
        )
        completed = int(hypothesis_family.get("completed_real_results") or 0)
        seeds = hypothesis_family.get("training_seeds")
        distinct_seeds = len(set(seeds)) if isinstance(seeds, list) else 0
        raw_summary = hypothesis_family.get("primary_metric_summary")
        summary = raw_summary if isinstance(raw_summary, Mapping) else {}
    replicated = status == "replicated"
    return {
        "status": status,
        "completed_real_results": completed,
        "distinct_training_seeds": distinct_seeds,
        "sample_standard_deviation": (
            summary.get("sample_standard_deviation") if replicated else None
        ),
        "standard_error": summary.get("standard_error") if replicated else None,
        "uncertainty_method": summary.get("uncertainty_method") if replicated else None,
        "limitation": "Between-run variation is not a per-example confidence interval.",
    }


def build_experiment_power_projection(
    *,
    outcome: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    hypothesis_family: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project exact power evidence without inventing a statistical conclusion."""

    evaluation, evidence = _evaluation_projection(_resolved_evaluation(outcome, evaluations))
    sequential = evidence.sequential_stopping if evidence is not None else None
    limitations = []
    if evaluation["sample_count"] is None:
        limitations.append("The exact evaluation sample count was not recorded.")
    if evaluation["sufficiency"]["status"] == "not_assessed":
        limitations.append("An observed sample count is not evidence of adequate power.")
    limitations.append(
        "Repeated fixed-sample evaluations are not sequential-stopping evidence without a "
        "predeclared design."
    )
    return {
        "schema_version": "bashgym.autoresearch_experiment_power.v1",
        "evaluation": evaluation,
        "seed_uncertainty": _seed_uncertainty(hypothesis_family),
        "sequential_stopping": {
            "status": "predeclared" if sequential is not None else "not_predeclared",
            "evidence": sequential.model_dump(mode="json") if sequential is not None else None,
        },
        "limitations": limitations,
    }


__all__ = [
    "AUTORESEARCH_EVALUATION_POWER_KEY",
    "AutoResearchEvaluationPowerEvidence",
    "AutoResearchSequentialStoppingEvidence",
    "build_experiment_power_projection",
]
