"""Bounded evaluator-authored behavioral failures for AutoResearch decisions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, field_validator

from bashgym.campaigns.contracts import FrozenContractModel, Identifier

AUTORESEARCH_FAILURE_OBSERVATIONS_KEY = "autoresearch_failure_observations"
MAX_AUTORESEARCH_FAILURE_OBSERVATIONS = 12

_PRIVATE_TEXT_PATTERNS = (
    re.compile(r"(?i)\b(?:https?|file|ssh)://"),
    re.compile(r"(?i)(?:[a-z]:\\|/(?:users|home|var|tmp|etc)/)"),
    re.compile(r"(?i)\b(?:api[_-]?(?:key|token)|password|secret|access[_-]?token|token)\s*[:=]"),
)


def _public_summary(value: str) -> str:
    normalized = value.strip()
    if any(pattern.search(normalized) for pattern in _PRIVATE_TEXT_PATTERNS):
        raise ValueError("failure observation contains private or secret-like text")
    return normalized


class AutoResearchFailureObservation(FrozenContractModel):
    """One aggregate failure category; never a raw evaluation example."""

    schema_version: Literal["autoresearch_failure_observation.v1"] = (
        "autoresearch_failure_observation.v1"
    )
    observation_id: Identifier
    category: Identifier
    summary: str = Field(min_length=1, max_length=1000)
    slice_path: str | None = Field(default=None, min_length=1, max_length=1000)
    checkpoint_step: int | None = Field(default=None, ge=1, le=10_000_000)
    count: int = Field(ge=1, le=1_000_000)

    @field_validator("summary", "slice_path")
    @classmethod
    def reject_private_text(cls, value: str | None) -> str | None:
        return None if value is None else _public_summary(value)


def validated_failure_observations(value: Any) -> tuple[AutoResearchFailureObservation, ...]:
    """Parse one bounded sequence and reject ambiguous duplicate identities."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("failure observations must be a sequence")
    if len(value) > MAX_AUTORESEARCH_FAILURE_OBSERVATIONS:
        raise ValueError("failure observations may contain at most 12 items")
    observations = tuple(AutoResearchFailureObservation.model_validate(item) for item in value)
    identifiers = tuple(item.observation_id for item in observations)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("failure observation IDs must be unique")
    return observations


def _outcome_evaluation(
    outcome: Mapping[str, Any] | None,
    evaluations_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]] | None:
    if outcome is None:
        return None
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        return None
    references = result.get("evidence_references")
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        return None
    for reference in references:
        evaluation = evaluations_by_id.get(str(reference))
        if evaluation is not None:
            return str(result.get("proposal_id") or ""), evaluation
    return None


def _project_evaluation_failures(
    resolved: tuple[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    if resolved is None:
        return None, {}
    proposal_id, evaluation = resolved
    slices = evaluation.get("slice_metrics")
    raw = (
        slices.get(AUTORESEARCH_FAILURE_OBSERVATIONS_KEY, ()) if isinstance(slices, Mapping) else ()
    )
    observations = validated_failure_observations(raw)
    counts: dict[str, int] = {}
    for observation in observations:
        counts[observation.category] = counts.get(observation.category, 0) + observation.count
    return (
        {
            "proposal_id": proposal_id,
            "evaluation_result_id": str(evaluation.get("evaluation_result_id") or ""),
            "observations": [item.model_dump(mode="json") for item in observations],
        },
        counts,
    )


def build_research_failure_packet(
    *,
    campaign_id: str,
    reference_outcome: Mapping[str, Any] | None,
    candidate_outcome: Mapping[str, Any] | None,
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare aggregate failure categories from exact outcome evidence references."""

    evaluations_by_id = {
        str(item.get("evaluation_result_id")): item
        for item in evaluations
        if item.get("evaluation_result_id")
    }
    reference, reference_counts = _project_evaluation_failures(
        _outcome_evaluation(reference_outcome, evaluations_by_id)
    )
    candidate, candidate_counts = _project_evaluation_failures(
        _outcome_evaluation(candidate_outcome, evaluations_by_id)
    )
    categories = sorted(set(reference_counts) | set(candidate_counts))
    selected = categories[:MAX_AUTORESEARCH_FAILURE_OBSERVATIONS]
    comparison = []
    for category in selected:
        reference_count = reference_counts.get(category, 0)
        candidate_count = candidate_counts.get(category, 0)
        delta = candidate_count - reference_count
        comparison.append(
            {
                "category": category,
                "reference_count": reference_count,
                "candidate_count": candidate_count,
                "delta": delta,
                "status": "improved" if delta < 0 else "regressed" if delta > 0 else "unchanged",
            }
        )
    return {
        "schema_version": "bashgym.research_failures.v1",
        "campaign_id": campaign_id,
        "reference": reference,
        "candidate": candidate,
        "comparison": comparison,
        "truncated": len(categories) > len(selected),
    }


__all__ = [
    "AUTORESEARCH_FAILURE_OBSERVATIONS_KEY",
    "MAX_AUTORESEARCH_FAILURE_OBSERVATIONS",
    "AutoResearchFailureObservation",
    "build_research_failure_packet",
    "validated_failure_observations",
]
