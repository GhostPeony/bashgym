"""Compact contract coverage for sealed AutoResearch evaluation evidence."""

import hashlib
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.autoresearch import ExperimentRole
from bashgym.campaigns.autoresearch_evidence import (
    AutoResearchCheckpointObservation,
    AutoResearchEvaluationContext,
    AutoResearchEvaluationEvidence,
    AutoResearchEvaluatorReadiness,
    CampaignEvaluationProjector,
    SealedEvaluationReader,
    evaluation_context_bytes,
    validate_baseline_evaluator_readiness,
    validate_checkpoint_observations,
)
from bashgym.campaigns.contracts import (
    ActionAttempt,
    ArtifactOutput,
    AttemptStatus,
    SealedActionResult,
    StageKind,
)
from bashgym.campaigns.diagnostic_actions import (
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA,
    AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN,
    AutoResearchDiagnosticEvidence,
    AutoResearchDiagnosticRecipe,
    diagnostic_recipe_digest,
    public_diagnostic_projection,
)
from bashgym.campaigns.failure_observations import AutoResearchFailureObservation
from bashgym.campaigns.lineage import canonical_model_manifest_digest
from bashgym.campaigns.runtime import CampaignArtifactRecord

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


def test_evidence_carries_only_typed_failure_observations():
    evidence = _evidence(
        failure_observations=(
            AutoResearchFailureObservation(
                observation_id="count-error",
                category="count_error",
                summary="The predicted object counts were incorrect.",
                count=5,
            ),
        )
    )

    serialized = evidence.model_dump(mode="json")
    assert serialized["failure_observations"] == [
        {
            "schema_version": "autoresearch_failure_observation.v1",
            "observation_id": "count-error",
            "category": "count_error",
            "summary": "The predicted object counts were incorrect.",
            "slice_path": None,
            "checkpoint_step": None,
            "count": 5,
        }
    ]
    assert "prediction" not in str(serialized).lower()


@pytest.mark.parametrize("metric", (float("nan"), float("inf"), float("-inf")))
def test_evidence_rejects_nonfinite_primary_metrics(metric):
    with pytest.raises(ValidationError, match="finite"):
        _evidence(metrics={"score": metric})


def test_evidence_rejects_reversed_evaluation_timestamps():
    with pytest.raises(ValidationError, match="completed_at"):
        _evidence(started_at=NOW + timedelta(seconds=1), completed_at=NOW)


def _checkpoint_artifact(step: int, relative_path: str, digest: str, size: int):
    return SimpleNamespace(
        schema_name="huggingface_checkpoint_file.v1",
        sha256=digest,
        size_bytes=size,
        metadata={
            "checkpoint_step": step,
            "relative_path": relative_path,
        },
    )


def test_checkpoint_observations_require_sorted_unique_steps_and_finite_metrics():
    first = AutoResearchCheckpointObservation(
        checkpoint_step=10,
        evaluated_model_manifest_digest="1" * 64,
        metrics={"score": 0.60},
        started_at=NOW,
        completed_at=NOW + timedelta(seconds=1),
    )
    second = first.model_copy(
        update={
            "checkpoint_step": 20,
            "evaluated_model_manifest_digest": "2" * 64,
        }
    )

    evidence = _evidence(checkpoint_observations=(first, second))

    assert [item.checkpoint_step for item in evidence.checkpoint_observations] == [10, 20]
    with pytest.raises(ValidationError, match="sorted and unique"):
        _evidence(checkpoint_observations=(second, first))
    with pytest.raises(ValidationError, match="finite"):
        AutoResearchCheckpointObservation(
            checkpoint_step=10,
            evaluated_model_manifest_digest="1" * 64,
            metrics={"score": float("nan")},
            started_at=NOW,
            completed_at=NOW,
        )


def test_checkpoint_observations_must_match_sealed_training_inventory():
    files = (
        _checkpoint_artifact(10, "adapter_config.json", "1" * 64, 10),
        _checkpoint_artifact(10, "adapter_model.safetensors", "2" * 64, 20),
    )
    expected_digest = canonical_model_manifest_digest(files)
    evidence = _evidence(
        checkpoint_observations=(
            AutoResearchCheckpointObservation(
                checkpoint_step=10,
                evaluated_model_manifest_digest=expected_digest,
                metrics={"score": 0.60},
                started_at=NOW,
                completed_at=NOW + timedelta(seconds=1),
            ),
        )
    )

    validate_checkpoint_observations(evidence, checkpoint_artifacts=files)

    tampered = evidence.model_copy(
        update={
            "checkpoint_observations": (
                evidence.checkpoint_observations[0].model_copy(
                    update={"evaluated_model_manifest_digest": "f" * 64}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="checkpoint manifest mismatch"):
        validate_checkpoint_observations(tampered, checkpoint_artifacts=files)


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


def test_projector_authenticates_remote_diagnostic_projection_before_recording():
    recipe = AutoResearchDiagnosticRecipe(
        probe_family="loss_landscape",
        question="Does held-out loss rise after the retained checkpoint?",
        hypothesis="The longer continuation moved beyond the useful region.",
        informs_methods=("sft", "data_redesign"),
        measurements=(
            {
                "name": "heldout_loss_slope",
                "interpretation": "minimize",
                "unit": "loss_per_step",
            },
        ),
        sample_limit=64,
        seed=17,
        data_scope_ids=("approved-development",),
    )
    evidence = AutoResearchDiagnosticEvidence(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        proposal_id="diagnostic-a",
        study_id="study-a",
        action_id="diagnostic-action-a",
        attempt_id="diagnostic-attempt-a",
        recipe_digest=diagnostic_recipe_digest(recipe),
        runner_id="diagnostic-runner",
        runner_version="1",
        status="completed",
        measurements=(
            {
                "name": "heldout_loss_slope",
                "value": 0.031,
                "sample_count": 64,
                "unit": "loss_per_step",
            },
        ),
        resource_usage=(
            {
                "unit": "gpu_hours",
                "amount": 0.02,
                "source": "runner",
                "confidence": "measured",
            },
        ),
    )
    normalized = {
        "evidence": evidence.model_dump(mode="json"),
        "projection": public_diagnostic_projection(recipe, evidence),
    }
    payload = evidence.model_dump_json().encode()
    output = ArtifactOutput(
        path=AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_name=AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA,
    )
    attempt = ActionAttempt(
        attempt_id=evidence.attempt_id,
        workspace_id=evidence.workspace_id,
        campaign_id=evidence.campaign_id,
        study_id=evidence.study_id,
        action_id=evidence.action_id,
        attempt_number=1,
        claim_generation=1,
        status=AttemptStatus.COMPLETED,
        input_digest="1" * 64,
        candidate_digest="2" * 64,
        manifest_revision=1,
        stage=StageKind.CONTRACT_EVALUATION,
        executor={
            "kind": "ssh_remote",
            "stage": "contract_evaluation",
            "compute_profile_id": "registered-compute",
            "seal_executor_id": "ssh-remote",
            "seal_executor_version": "1",
            "diagnostic_proposal_id": evidence.proposal_id,
            "diagnostic_recipe": recipe.model_dump(mode="json"),
            "diagnostic_contract": {
                "runner_id": evidence.runner_id,
                "runner_version": evidence.runner_version,
            },
        },
        sealed_result_uri="bashgym-remote-seal://placeholder",
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=1),
    )
    manifest = SealedActionResult(
        workspace_id=attempt.workspace_id,
        campaign_id=attempt.campaign_id,
        study_id=attempt.study_id,
        action_id=attempt.action_id,
        attempt_id=attempt.attempt_id,
        manifest_revision=attempt.manifest_revision,
        candidate_digest=attempt.candidate_digest,
        input_digest=attempt.input_digest,
        claim_generation=attempt.claim_generation,
        executor_id="ssh-remote",
        executor_version="1",
        compute_profile_id="registered-compute",
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
        outcome="completed",
        exit_code=0,
        exit_reason="completed",
        outputs=(output,),
    )
    sealer = ArtifactSealer(b"d" * 32, key_version="diagnostic-test-v1")
    attempt = attempt.model_copy(
        update={
            "sealed_result_uri": (
                "bashgym-remote-seal://registered-compute/diagnostic-attempt-a/sha256/"
                + hashlib.sha256(sealer.envelope_bytes(manifest)).hexdigest()
            )
        }
    )
    artifact = CampaignArtifactRecord(
        workspace_id=attempt.workspace_id,
        campaign_id=attempt.campaign_id,
        artifact_id="artifact-diagnostic-a",
        producer_action_id=attempt.action_id,
        uri="bashgym-remote-artifact://registered-compute/diagnostic-attempt-a/output",
        sha256=output.sha256,
        size_bytes=output.size_bytes,
        schema_name=output.schema_name,
        sealed=True,
        valid=True,
        metadata={
            "attempt_id": attempt.attempt_id,
            "normalized_diagnostic": normalized,
            "projection_key_version": sealer.key_version,
            "projection_signature": sealer.sign_canonical_payload(
                normalized,
                domain=AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN,
            ),
        },
        created_at=NOW,
    )

    class Repository:
        recorded = None

        def get_autoresearch_proposal(self, *_args):
            return SimpleNamespace(role=ExperimentRole.DIAGNOSTIC)

        def get_proposal(self, *_args):
            return SimpleNamespace(study_id="study-a")

        def list_study_attempts(self, *_args):
            return (attempt,)

        def get_attempt_result_manifest(self, *_args):
            return manifest

        def list_action_artifacts(self, *_args):
            return (artifact,)

        def get_autoresearch_spec(self, *_args):
            return SimpleNamespace(stop_rules=SimpleNamespace(budget_unit="gpu_hours"))

        def record_autoresearch_diagnostic_result(self, result):
            self.recorded = result
            return result

    repository = Repository()
    projector = CampaignEvaluationProjector.__new__(CampaignEvaluationProjector)
    projector.repository = repository
    projector.reader = SealedEvaluationReader(sealer)

    result = projector.project_diagnostic_and_ingest("workspace-a", "campaign-a", "diagnostic-a")

    assert result.projection["measurements"][0]["value"] == 0.031
    assert result.actual_cost == pytest.approx(0.02)
    assert repository.recorded == result
