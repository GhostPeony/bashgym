"""Compact contract coverage for sealed AutoResearch evaluation evidence."""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from bashgym._compat import UTC
from bashgym.campaigns.autoresearch_evidence import (
    AutoResearchEvaluationContext,
    AutoResearchEvaluationEvidence,
    evaluation_context_bytes,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
MODEL_DIGEST = "a" * 64


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
