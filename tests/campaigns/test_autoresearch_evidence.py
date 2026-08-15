"""Compact contract coverage for sealed AutoResearch evaluation evidence."""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from bashgym._compat import UTC
from bashgym.campaigns.autoresearch_evidence import (
    AutoResearchEvaluationContext,
    AutoResearchEvaluationEvidence,
    AutoResearchEvaluatorReadiness,
    evaluation_context_bytes,
    validate_baseline_evaluator_readiness,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
MODEL_DIGEST = "a" * 64
READINESS_CONTRACT = {
    "known_good_case_id": "known-good",
    "known_bad_case_id": "known-bad",
    "baseline_repeat_count": 3,
    "maximum_baseline_spread": 0.02,
}


def _context(**updates):
    return AutoResearchEvaluationContext(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        study_id="study-a",
        action_id="evaluate-a",
        attempt_id="attempt-a",
        candidate_digest="b" * 64,
        evaluation_suite_id="suite-held-out",
        evaluation_code_digest="c" * 64,
        dataset_version_id="dataset-held-out-v1",
        dataset_content_digest="d" * 64,
        evaluated_model_manifest_digest=MODEL_DIGEST,
        **updates,
    )


def _evidence(**updates):
    values = {
        "campaign_id": "campaign-a",
        "study_id": "study-a",
        "action_id": "evaluate-a",
        "attempt_id": "attempt-a",
        "candidate_digest": "b" * 64,
        "evaluation_suite_id": "suite-held-out",
        "evaluation_code_digest": "c" * 64,
        "dataset_version_id": "dataset-held-out-v1",
        "evaluated_model_manifest_digest": MODEL_DIGEST,
        "metrics": {"score": 0.75},
        "started_at": NOW,
        "completed_at": NOW + timedelta(seconds=1),
    }
    values.update(updates)
    return AutoResearchEvaluationEvidence(**values)


def _readiness(**updates):
    values = {
        "known_good_case_id": "known-good",
        "known_good_passed": True,
        "known_bad_case_id": "known-bad",
        "known_bad_rejected": True,
        "baseline_scores": (0.74, 0.75, 0.76),
    }
    values.update(updates)
    return AutoResearchEvaluatorReadiness(**values)


def test_evaluation_only_baseline_context_binds_the_registered_model_digest():
    context = _context()

    assert context.evaluated_model_manifest_digest == MODEL_DIGEST
    assert b"dataset-held-out-v1" in evaluation_context_bytes(context)


def test_evidence_binds_exact_campaign_study_proposal_attempt_inputs():
    evidence = _evidence()

    assert (
        evidence.campaign_id,
        evidence.study_id,
        evidence.action_id,
        evidence.attempt_id,
        evidence.candidate_digest,
    ) == ("campaign-a", "study-a", "evaluate-a", "attempt-a", "b" * 64)


@pytest.mark.parametrize("metric", (float("nan"), float("inf"), float("-inf")))
def test_evidence_rejects_nonfinite_primary_metrics(metric):
    with pytest.raises(ValidationError, match="finite"):
        _evidence(metrics={"score": metric})


def test_evidence_rejects_reversed_evaluation_timestamps():
    with pytest.raises(ValidationError, match="completed_at"):
        _evidence(started_at=NOW + timedelta(seconds=1), completed_at=NOW)


def test_baseline_readiness_accepts_matching_canaries_and_stable_repeats():
    evidence = _evidence(
        metrics={"score": 0.75},
        evaluator_readiness=_readiness(),
    )

    validate_baseline_evaluator_readiness(
        evidence,
        primary_metric="score",
        readiness_contract=READINESS_CONTRACT,
    )


def test_baseline_readiness_is_optional_when_suite_does_not_declare_it():
    validate_baseline_evaluator_readiness(
        _evidence(metrics={"score": 0.75}),
        primary_metric="score",
        readiness_contract=None,
    )


@pytest.mark.parametrize(
    ("evidence", "contract", "code"),
    (
        (
            _evidence(metrics={"score": 0.75}),
            READINESS_CONTRACT,
            "autoresearch_evaluator_readiness_missing",
        ),
        (
            _evidence(
                metrics={"score": 0.75},
                evaluator_readiness=_readiness(known_bad_rejected=False),
            ),
            READINESS_CONTRACT,
            "autoresearch_evaluator_canary_failed",
        ),
        (
            _evidence(
                metrics={"score": 0.75},
                evaluator_readiness=_readiness(baseline_scores=(0.70, 0.75, 0.80)),
            ),
            READINESS_CONTRACT,
            "autoresearch_baseline_unstable",
        ),
        (
            _evidence(
                metrics={"score": 0.80},
                evaluator_readiness=_readiness(),
            ),
            READINESS_CONTRACT,
            "autoresearch_baseline_repeat_mean_mismatch",
        ),
        (
            _evidence(
                metrics={"score": 0.75},
                evaluator_readiness=_readiness(known_good_case_id="different-case"),
            ),
            READINESS_CONTRACT,
            "autoresearch_evaluator_canary_failed",
        ),
    ),
)
def test_baseline_readiness_rejects_untrustworthy_evaluator_observations(evidence, contract, code):
    with pytest.raises(ValueError, match=code):
        validate_baseline_evaluator_readiness(
            evidence,
            primary_metric="score",
            readiness_contract=contract,
        )
