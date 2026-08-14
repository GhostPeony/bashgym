"""Information-gain contracts for agent-authored experiments."""

import math

import pytest
from pydantic import ValidationError

from bashgym.research.acquisition import (
    CompetingHypothesis,
    ExperimentAcquisition,
    PredictedOutcome,
    ResearchContextBundle,
    ResearchContextSource,
)


def context(**updates):
    values = {
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "proposal_id": "proposal-1",
        "query": "which intervention distinguishes the failure hypotheses?",
        "categories": ("research",),
        "status": "available",
        "sources": (
            ResearchContextSource(
                title="Model Discovery Agent",
                url="https://arxiv.org/abs/2608.09696",
                summary="Bayesian experiment design for mechanistic model discovery.",
                source_type="research",
            ),
        ),
    }
    values.update(updates)
    return ResearchContextBundle(**values)


def acquisition(**updates):
    values = {
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "proposal_id": "proposal-1",
        "selection_mode": "information_gain",
        "hypotheses": (
            CompetingHypothesis(
                hypothesis_id="optimization",
                statement="The optimizer is under-updating.",
                prior_probability=0.5,
            ),
            CompetingHypothesis(
                hypothesis_id="data",
                statement="The dataset lacks recovery examples.",
                prior_probability=0.5,
            ),
        ),
        "outcomes": (
            PredictedOutcome(outcome_id="improves", label="Held-out score improves"),
            PredictedOutcome(outcome_id="flat", label="Held-out score is flat"),
        ),
        "conditional_outcome_probabilities": {
            "optimization": {"improves": 0.9, "flat": 0.1},
            "data": {"improves": 0.2, "flat": 0.8},
        },
        "expected_cost": 2.0,
    }
    values.update(updates)
    return ExperimentAcquisition(**values)


def test_context_digest_is_canonical_and_sources_are_unique():
    first = context()
    second = ResearchContextBundle.model_validate(first.model_dump(mode="json"))
    assert first.retrieval_digest == second.retrieval_digest

    with pytest.raises(ValidationError, match="research_context_sources_not_unique"):
        context(sources=(first.sources[0], first.sources[0]))


def test_information_gain_and_cost_normalization_are_recomputed():
    result = acquisition()
    # Symmetric prior entropy is one bit. This experiment is informative but not perfect.
    assert result.expected_information_gain == pytest.approx(0.39731260974948646)
    assert result.cost_normalized_information_gain == pytest.approx(
        result.expected_information_gain / 2.0
    )
    assert math.isfinite(result.expected_information_gain)

    with pytest.raises(ValidationError, match="acquisition_score_mismatch"):
        acquisition(expected_information_gain=0.99)


def test_zero_information_experiment_scores_zero():
    result = acquisition(
        conditional_outcome_probabilities={
            "optimization": {"improves": 0.5, "flat": 0.5},
            "data": {"improves": 0.5, "flat": 0.5},
        }
    )
    assert result.expected_information_gain == pytest.approx(0.0, abs=1e-12)
    assert result.cost_normalized_information_gain == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "hypotheses": (
                CompetingHypothesis(
                    hypothesis_id="optimization", statement="A", prior_probability=0.7
                ),
                CompetingHypothesis(hypothesis_id="data", statement="B", prior_probability=0.7),
            )
        },
        {
            "conditional_outcome_probabilities": {
                "optimization": {"improves": 0.8, "flat": 0.8},
                "data": {"improves": 0.2, "flat": 0.8},
            }
        },
    ],
)
def test_probability_contracts_must_be_complete_and_normalized(updates):
    with pytest.raises(ValidationError, match="acquisition_probability_contract_invalid"):
        acquisition(**updates)
