import pytest
from pydantic import ValidationError

from bashgym.campaigns.failure_observations import (
    AutoResearchFailureObservation,
    build_research_failure_packet,
    validated_failure_observations,
)


def _observation(identifier: str = "count-error", **overrides):
    value = {
        "observation_id": identifier,
        "category": "count_error",
        "summary": "The predicted object counts were incorrect.",
        "slice_path": "behavior.counting",
        "count": 7,
    }
    value.update(overrides)
    return value


def test_failure_observations_accept_only_bounded_behavioral_summaries():
    observations = validated_failure_observations([_observation()])

    assert observations == (
        AutoResearchFailureObservation(
            observation_id="count-error",
            category="count_error",
            summary="The predicted object counts were incorrect.",
            slice_path="behavior.counting",
            count=7,
        ),
    )


def test_failure_observations_reject_duplicate_ids_and_unbounded_lists():
    with pytest.raises(ValueError, match="unique"):
        validated_failure_observations([_observation(), _observation()])

    with pytest.raises(ValueError, match="at most 12"):
        validated_failure_observations([_observation(f"failure-{index}") for index in range(13)])


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": "raw held-out input"},
        {"summary": "Inspect https://example.test/case/4"},
        {"summary": r"Read C:\\Users\\operator\\case.json"},
        {"summary": "/home/operator/case.json"},
        {"summary": "api_token=do-not-persist"},
        {"slice_path": "file:///tmp/eval.json"},
    ],
)
def test_failure_observations_reject_raw_examples_locations_and_secrets(overrides):
    with pytest.raises((ValidationError, ValueError)):
        validated_failure_observations([_observation(**overrides)])


def _outcome(proposal_id: str, evaluation_id: str):
    return {
        "result": {
            "proposal_id": proposal_id,
            "evidence_references": [evaluation_id],
        }
    }


def _evaluation(evaluation_id: str, observations: list[dict]):
    return {
        "evaluation_result_id": evaluation_id,
        "slice_metrics": {"autoresearch_failure_observations": observations},
    }


def test_failure_packet_compares_categories_from_exact_outcome_evidence():
    packet = build_research_failure_packet(
        campaign_id="campaign-a",
        reference_outcome=_outcome("baseline", "evaluation-baseline"),
        candidate_outcome=_outcome("candidate-1", "evaluation-candidate"),
        evaluations=(
            _evaluation(
                "evaluation-baseline",
                [_observation(count=7), _observation("format", category="format_error", count=2)],
            ),
            _evaluation(
                "evaluation-candidate",
                [
                    _observation(count=3),
                    _observation("combined", category="count_and_format_error", count=4),
                ],
            ),
        ),
    )

    assert packet == {
        "schema_version": "bashgym.research_failures.v1",
        "campaign_id": "campaign-a",
        "reference": {
            "proposal_id": "baseline",
            "evaluation_result_id": "evaluation-baseline",
            "observations": [
                {
                    "schema_version": "autoresearch_failure_observation.v1",
                    "observation_id": "count-error",
                    "category": "count_error",
                    "summary": "The predicted object counts were incorrect.",
                    "slice_path": "behavior.counting",
                    "checkpoint_step": None,
                    "count": 7,
                },
                {
                    "schema_version": "autoresearch_failure_observation.v1",
                    "observation_id": "format",
                    "category": "format_error",
                    "summary": "The predicted object counts were incorrect.",
                    "slice_path": "behavior.counting",
                    "checkpoint_step": None,
                    "count": 2,
                },
            ],
        },
        "candidate": {
            "proposal_id": "candidate-1",
            "evaluation_result_id": "evaluation-candidate",
            "observations": [
                {
                    "schema_version": "autoresearch_failure_observation.v1",
                    "observation_id": "count-error",
                    "category": "count_error",
                    "summary": "The predicted object counts were incorrect.",
                    "slice_path": "behavior.counting",
                    "checkpoint_step": None,
                    "count": 3,
                },
                {
                    "schema_version": "autoresearch_failure_observation.v1",
                    "observation_id": "combined",
                    "category": "count_and_format_error",
                    "summary": "The predicted object counts were incorrect.",
                    "slice_path": "behavior.counting",
                    "checkpoint_step": None,
                    "count": 4,
                },
            ],
        },
        "comparison": [
            {
                "category": "count_and_format_error",
                "reference_count": 0,
                "candidate_count": 4,
                "delta": 4,
                "status": "regressed",
            },
            {
                "category": "count_error",
                "reference_count": 7,
                "candidate_count": 3,
                "delta": -4,
                "status": "improved",
            },
            {
                "category": "format_error",
                "reference_count": 2,
                "candidate_count": 0,
                "delta": -2,
                "status": "improved",
            },
        ],
        "truncated": False,
    }


def test_failure_packet_is_sparse_and_bounds_category_union():
    observations = [
        _observation(f"failure-{index}", category=f"category_{index:02d}", count=index + 1)
        for index in range(12)
    ]
    candidate = [
        _observation(f"other-{index}", category=f"other_{index:02d}", count=index + 1)
        for index in range(12)
    ]

    packet = build_research_failure_packet(
        campaign_id="campaign-a",
        reference_outcome=_outcome("baseline", "evaluation-baseline"),
        candidate_outcome=_outcome("candidate", "evaluation-candidate"),
        evaluations=(
            _evaluation("evaluation-baseline", observations),
            _evaluation("evaluation-candidate", candidate),
        ),
    )

    assert len(packet["comparison"]) == 12
    assert packet["truncated"] is True
    empty = build_research_failure_packet(
        campaign_id="campaign-a",
        reference_outcome=None,
        candidate_outcome=None,
        evaluations=(),
    )
    assert empty["reference"] is None
    assert empty["candidate"] is None
    assert empty["comparison"] == []
