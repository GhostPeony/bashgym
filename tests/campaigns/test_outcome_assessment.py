import pytest

from bashgym.campaigns.outcome_assessment import build_outcome_assessment


def _analysis(*statuses: tuple[str, str]) -> dict:
    return {
        "schema_version": "bashgym.research_failures.v1",
        "comparison": [
            {
                "category": category,
                "reference_count": 4,
                "candidate_count": 3,
                "delta": -1,
                "status": status,
            }
            for category, status in statuses
        ],
    }


@pytest.mark.parametrize(
    (
        "outcome",
        "provenance",
        "decision",
        "reason",
        "analysis",
        "classification",
        "is_failure",
        "kind",
    ),
    [
        (
            "crashed",
            "real",
            "crash",
            "experiment_crashed",
            None,
            "invalid_execution",
            True,
            "execution",
        ),
        (
            "completed",
            "real",
            "discard",
            "candidate_failed_protected_metric_gate",
            _analysis(("format_errors", "improved")),
            "unacceptable_regression",
            True,
            "scientific_guardrail",
        ),
        (
            "completed",
            "real",
            "keep",
            "candidate_improved_primary_metric",
            _analysis(("format_errors", "regressed")),
            "acceptable_tradeoff",
            False,
            None,
        ),
        (
            "completed",
            "real",
            "discard",
            "candidate_did_not_clear_improvement_gate",
            _analysis(("tool_errors", "improved"), ("format_errors", "regressed")),
            "mixed_evidence",
            False,
            None,
        ),
        (
            "completed",
            "real",
            "discard",
            "candidate_did_not_clear_improvement_gate",
            _analysis(("tool_errors", "unchanged")),
            "no_demonstrated_gain",
            False,
            None,
        ),
        (
            "completed",
            "real",
            "keep",
            "candidate_improved_primary_metric",
            _analysis(("tool_errors", "improved")),
            "clear_improvement",
            False,
            None,
        ),
    ],
)
def test_assessment_reserves_failure_for_execution_and_predeclared_guardrails(
    outcome,
    provenance,
    decision,
    reason,
    analysis,
    classification,
    is_failure,
    kind,
):
    assessment = build_outcome_assessment(
        outcome=outcome,
        provenance=provenance,
        decision=decision,
        reason_code=reason,
        failure_analysis=analysis,
    )

    assert assessment["classification"] == classification
    assert assessment["is_failure"] is is_failure
    assert assessment["failure_kind"] == kind
    assert assessment["evidence_strength"] == "single_observation"


def test_assessment_exposes_bounded_tradeoffs_and_improvements():
    assessment = build_outcome_assessment(
        outcome="completed",
        provenance="real",
        decision="discard",
        reason_code="candidate_did_not_clear_improvement_gate",
        failure_analysis=_analysis(
            ("tool_errors", "improved"),
            ("format_errors", "regressed"),
            ("style_errors", "unchanged"),
        ),
        evidence_strength="replicated",
    )

    assert assessment == {
        "schema_version": "bashgym.autoresearch_outcome_assessment.v1",
        "classification": "mixed_evidence",
        "is_failure": False,
        "failure_kind": None,
        "decision": "discard",
        "reason_code": "mixed_evidence_not_retained",
        "observed_tradeoffs": ["format_errors"],
        "observed_improvements": ["tool_errors"],
        "evidence_strength": "replicated",
    }


def test_assessment_marks_contradictory_completed_evidence_inconclusive():
    assessment = build_outcome_assessment(
        outcome="completed",
        provenance="real",
        decision="crash",
        reason_code="unexpected",
        failure_analysis=None,
    )

    assert assessment["classification"] == "inconclusive"
    assert assessment["is_failure"] is False
    assert assessment["reason_code"] == "completed_evidence_inconclusive"
