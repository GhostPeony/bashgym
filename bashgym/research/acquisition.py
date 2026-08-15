"""Research-context and expected-information-gain contracts.

These values explain why a host agent selected an experiment.  They are not
evaluation evidence and have no authority over KEEP/DISCARD decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_ALLOWED_CATEGORIES = frozenset({"research", "github", "pdf"})
_PROBABILITY_TOLERANCE = 1e-9


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def _validate_identifier(value: str) -> str:
    if not 1 <= len(value) <= 160 or not _IDENTIFIER.fullmatch(value):
        raise ValueError("research_identifier_invalid")
    return value


class ResearchContextSource(_FrozenModel):
    schema_version: Literal["bashgym.research_context_source.v1"] = (
        "bashgym.research_context_source.v1"
    )
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2_000)
    summary: str = Field(default="", max_length=1_000)
    source_type: Literal["research", "github", "pdf"]
    published_at: str | None = Field(default=None, max_length=100)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("research_context_source_url_invalid")
        return value


class ResearchContextBundle(_FrozenModel):
    schema_version: Literal["bashgym.research_context.v1"] = "bashgym.research_context.v1"
    workspace_id: str
    campaign_id: str
    proposal_id: str
    query: str = Field(min_length=1, max_length=1_000)
    categories: tuple[Literal["research", "github", "pdf"], ...]
    status: Literal["available", "unavailable"]
    code: str | None = Field(default=None, max_length=160)
    sources: tuple[ResearchContextSource, ...] = Field(default=(), max_length=10)
    retrieval_digest: str = ""

    @field_validator("workspace_id", "campaign_id", "proposal_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("categories")
    @classmethod
    def validate_categories(
        cls, value: tuple[Literal["research", "github", "pdf"], ...]
    ) -> tuple[Literal["research", "github", "pdf"], ...]:
        if not value or len(value) != len(set(value)) or not set(value) <= _ALLOWED_CATEGORIES:
            raise ValueError("research_context_categories_invalid")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> ResearchContextBundle:
        urls = tuple(source.url for source in self.sources)
        if len(urls) != len(set(urls)):
            raise ValueError("research_context_sources_not_unique")
        if self.status == "unavailable" and not self.code:
            raise ValueError("research_context_unavailable_code_required")
        if self.status == "available" and self.code:
            raise ValueError("research_context_available_code_forbidden")
        projection = self.model_dump(mode="json", exclude={"retrieval_digest"})
        expected = _canonical_hash(projection)
        if self.retrieval_digest and self.retrieval_digest != expected:
            raise ValueError("research_context_digest_mismatch")
        if not self.retrieval_digest:
            object.__setattr__(self, "retrieval_digest", expected)
        return self


class CompetingHypothesis(_FrozenModel):
    schema_version: Literal["bashgym.competing_hypothesis.v1"] = "bashgym.competing_hypothesis.v1"
    hypothesis_id: str
    statement: str = Field(min_length=1, max_length=2_000)
    prior_probability: float = Field(ge=0.0, le=1.0)

    @field_validator("hypothesis_id")
    @classmethod
    def validate_hypothesis_id(cls, value: str) -> str:
        return _validate_identifier(value)


class PredictedOutcome(_FrozenModel):
    schema_version: Literal["bashgym.predicted_outcome.v1"] = "bashgym.predicted_outcome.v1"
    outcome_id: str
    label: str = Field(min_length=1, max_length=1_000)

    @field_validator("outcome_id")
    @classmethod
    def validate_outcome_id(cls, value: str) -> str:
        return _validate_identifier(value)


class ExperimentAcquisition(_FrozenModel):
    """Host-authored beliefs with scores recomputed by BashGym."""

    schema_version: Literal["bashgym.experiment_acquisition.v1"] = (
        "bashgym.experiment_acquisition.v1"
    )
    workspace_id: str
    campaign_id: str
    proposal_id: str
    selection_mode: Literal["expected_improvement", "information_gain", "hybrid"]
    hypotheses: tuple[CompetingHypothesis, ...] = Field(min_length=2, max_length=12)
    outcomes: tuple[PredictedOutcome, ...] = Field(min_length=2, max_length=12)
    conditional_outcome_probabilities: dict[str, dict[str, float]]
    expected_cost: float = Field(gt=0.0)
    expected_information_gain: float = Field(default=0.0, ge=0.0)
    cost_normalized_information_gain: float = Field(default=0.0, ge=0.0)

    @field_validator("workspace_id", "campaign_id", "proposal_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("expected_cost", mode="before")
    @classmethod
    def validate_expected_cost(cls, value: Any) -> Any:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 1e-9
        ):
            raise ValueError("acquisition_expected_cost_invalid")
        return value

    @field_validator("expected_information_gain", "cost_normalized_information_gain", mode="before")
    @classmethod
    def validate_supplied_score(cls, value: Any) -> Any:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("acquisition_score_invalid")
        return value

    @model_validator(mode="after")
    def compute_scores(self) -> ExperimentAcquisition:
        hypothesis_ids = tuple(item.hypothesis_id for item in self.hypotheses)
        outcome_ids = tuple(item.outcome_id for item in self.outcomes)
        if (
            len(hypothesis_ids) != len(set(hypothesis_ids))
            or len(outcome_ids) != len(set(outcome_ids))
            or set(self.conditional_outcome_probabilities) != set(hypothesis_ids)
        ):
            raise ValueError("acquisition_probability_contract_invalid")
        priors = [item.prior_probability for item in self.hypotheses]
        if not math.isclose(sum(priors), 1.0, abs_tol=_PROBABILITY_TOLERANCE):
            raise ValueError("acquisition_probability_contract_invalid")
        matrix: list[list[float]] = []
        for hypothesis_id in hypothesis_ids:
            row = self.conditional_outcome_probabilities[hypothesis_id]
            if set(row) != set(outcome_ids):
                raise ValueError("acquisition_probability_contract_invalid")
            values = [row[outcome_id] for outcome_id in outcome_ids]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in values
            ) or not math.isclose(sum(values), 1.0, abs_tol=_PROBABILITY_TOLERANCE):
                raise ValueError("acquisition_probability_contract_invalid")
            matrix.append([float(value) for value in values])

        prior_entropy = _entropy(priors)
        expected_posterior_entropy = 0.0
        for outcome_index in range(len(outcome_ids)):
            probability = sum(
                priors[hypothesis_index] * matrix[hypothesis_index][outcome_index]
                for hypothesis_index in range(len(hypothesis_ids))
            )
            if probability <= 0:
                continue
            posterior = [
                priors[hypothesis_index] * matrix[hypothesis_index][outcome_index] / probability
                for hypothesis_index in range(len(hypothesis_ids))
            ]
            expected_posterior_entropy += probability * _entropy(posterior)
        information_gain = max(0.0, prior_entropy - expected_posterior_entropy)
        normalized = information_gain / self.expected_cost

        if (
            "expected_information_gain" in self.model_fields_set
            and not math.isclose(
                self.expected_information_gain,
                information_gain,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ) or (
            "cost_normalized_information_gain" in self.model_fields_set
            and not math.isclose(
                self.cost_normalized_information_gain,
                normalized,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("acquisition_score_mismatch")
        object.__setattr__(self, "expected_information_gain", information_gain)
        object.__setattr__(self, "cost_normalized_information_gain", normalized)
        return self


__all__ = [
    "CompetingHypothesis",
    "ExperimentAcquisition",
    "PredictedOutcome",
    "ResearchContextBundle",
    "ResearchContextSource",
]
