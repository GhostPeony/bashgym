"""Standard, fail-closed evidence emitted by a pinned AutoResearch evaluator."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.artifacts import ArtifactSealer, ArtifactSealError
from bashgym.campaigns.contracts import (
    AUTORESEARCH_EVALUATION_SCHEMA,
    ActionAttempt,
    FrozenContractModel,
    HexDigest,
    Identifier,
    SealedActionResult,
    StageKind,
)
from bashgym.campaigns.diagnostic_actions import (
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA,
    AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN,
    MAX_AUTORESEARCH_DIAGNOSTIC_BYTES,
    AutoResearchDiagnosticEvidence,
    AutoResearchDiagnosticRecipe,
    public_diagnostic_projection,
)
from bashgym.campaigns.failure_observations import (
    AUTORESEARCH_FAILURE_OBSERVATIONS_KEY,
    AutoResearchFailureObservation,
)
from bashgym.campaigns.lineage import canonical_model_manifest_digest

if TYPE_CHECKING:
    from bashgym.campaigns.autoresearch import (
        AutoResearchOutcomeRecord,
        AutoResearchRepository,
    )
    from bashgym.campaigns.runtime import CampaignArtifactRecord
    from bashgym.ledger.contracts import DatasetVersionSpec, EvaluationSuiteSpec
    from bashgym.ledger.persistence import ExperimentLedgerRepository


AUTORESEARCH_EVALUATION_FILENAME = "autoresearch_evaluation.json"
AUTORESEARCH_EVALUATION_CONTEXT_FILENAME = "autoresearch_evaluation_context.json"
AUTORESEARCH_EVALUATION_CONTEXT_SCHEMA = "autoresearch_evaluation_context.v1"
MAX_AUTORESEARCH_EVALUATION_BYTES = 1024 * 1024
AUTORESEARCH_NORMALIZED_EVALUATION_DOMAIN = "bashgym.autoresearch.normalized-evaluation.v1"
AUTORESEARCH_CHECKPOINT_TRAJECTORY_KEY = "autoresearch_checkpoint_trajectory"


def _require_finite(value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evaluation metrics must be finite")
    if isinstance(value, dict):
        for nested in value.values():
            _require_finite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _require_finite(nested)


class AutoResearchEvaluationContext(FrozenContractModel):
    """Server-generated evaluator input binding identity and pinned materials."""

    schema_version: Literal["autoresearch_evaluation_context.v1"] = (
        AUTORESEARCH_EVALUATION_CONTEXT_SCHEMA
    )
    workspace_id: Identifier
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    candidate_digest: HexDigest
    evaluation_suite_id: Identifier
    evaluation_code_digest: HexDigest
    dataset_version_id: Identifier
    dataset_content_digest: HexDigest
    evaluated_model_manifest_digest: HexDigest


class AutoResearchEvaluatorReadiness(FrozenContractModel):
    """Evaluator-authored baseline canaries and repeated-score observations."""

    schema_version: Literal["autoresearch_evaluator_readiness.v1"] = (
        "autoresearch_evaluator_readiness.v1"
    )
    known_good_case_id: Identifier
    known_good_passed: bool
    known_bad_case_id: Identifier
    known_bad_rejected: bool
    baseline_scores: tuple[float, ...] = Field(min_length=3, max_length=10)

    @field_validator("baseline_scores")
    @classmethod
    def finite_baseline_scores(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        _require_finite(value)
        return value

    @model_validator(mode="after")
    def distinct_canary_cases(self) -> AutoResearchEvaluatorReadiness:
        if self.known_good_case_id == self.known_bad_case_id:
            raise ValueError("evaluator readiness canary cases must be distinct")
        return self


class AutoResearchCheckpointObservation(FrozenContractModel):
    """One diagnostic fixed-suite observation of a sealed training checkpoint."""

    schema_version: Literal["autoresearch_checkpoint_observation.v1"] = (
        "autoresearch_checkpoint_observation.v1"
    )
    observation_id: Identifier | None = None
    checkpoint_step: int = Field(ge=1, le=10_000_000)
    evaluated_model_manifest_digest: HexDigest
    metrics: dict[Identifier, float] = Field(min_length=1)
    slice_metrics: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime

    @field_validator("metrics", "slice_metrics")
    @classmethod
    def finite_metrics(cls, value):
        _require_finite(value)
        return value

    @model_validator(mode="after")
    def validate_identity_and_time(self) -> AutoResearchCheckpointObservation:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        if self.observation_id is None:
            object.__setattr__(
                self,
                "observation_id",
                f"checkpoint-step-{self.checkpoint_step}-{self.evaluated_model_manifest_digest[:12]}",
            )
        return self


def evaluation_context_bytes(context: AutoResearchEvaluationContext) -> bytes:
    """Return the one canonical byte representation used for launch and adoption."""

    return json.dumps(
        context.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class AutoResearchEvaluationEvidence(FrozenContractModel):
    """Sealed, evaluator-authored development-quality evidence."""

    schema_version: Literal[AUTORESEARCH_EVALUATION_SCHEMA] = AUTORESEARCH_EVALUATION_SCHEMA
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    candidate_digest: HexDigest
    evaluation_suite_id: Identifier
    evaluation_code_digest: HexDigest
    dataset_version_id: Identifier
    evaluated_model_manifest_digest: HexDigest
    metrics: dict[Identifier, float] = Field(min_length=1)
    slice_metrics: dict[str, Any] = Field(default_factory=dict)
    checkpoint_observations: tuple[AutoResearchCheckpointObservation, ...] = Field(
        default=(), max_length=8
    )
    failure_observations: tuple[AutoResearchFailureObservation, ...] = Field(
        default=(), max_length=12
    )
    evaluator_readiness: AutoResearchEvaluatorReadiness | None = None
    started_at: datetime
    completed_at: datetime

    @field_validator("metrics", "slice_metrics")
    @classmethod
    def finite_metrics(cls, value):
        _require_finite(value)
        return value

    @model_validator(mode="after")
    def ordered_timestamps(self) -> AutoResearchEvaluationEvidence:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        steps = tuple(item.checkpoint_step for item in self.checkpoint_observations)
        if steps != tuple(sorted(set(steps))):
            raise ValueError("checkpoint observation steps must be sorted and unique")
        if AUTORESEARCH_CHECKPOINT_TRAJECTORY_KEY in self.slice_metrics:
            raise ValueError("checkpoint trajectory is reserved for verified projection")
        if AUTORESEARCH_FAILURE_OBSERVATIONS_KEY in self.slice_metrics:
            raise ValueError("failure observations are reserved for verified projection")
        observation_ids = tuple(item.observation_id for item in self.failure_observations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("failure observation IDs must be unique")
        return self


def validate_checkpoint_observations(
    evidence: AutoResearchEvaluationEvidence,
    *,
    checkpoint_artifacts: Iterable[CampaignArtifactRecord],
) -> None:
    """Bind evaluator observations to exact checkpoint files from the training seal."""

    grouped: dict[int, list[CampaignArtifactRecord]] = {}
    for artifact in checkpoint_artifacts:
        metadata = getattr(artifact, "metadata", {})
        step = metadata.get("checkpoint_step") if isinstance(metadata, dict) else None
        relative_path = metadata.get("relative_path") if isinstance(metadata, dict) else None
        if (
            getattr(artifact, "schema_name", None) != "huggingface_checkpoint_file.v1"
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 1
            or not isinstance(relative_path, str)
            or not relative_path
        ):
            raise ValueError("checkpoint artifact inventory is invalid")
        grouped.setdefault(step, []).append(artifact)

    for observation in evidence.checkpoint_observations:
        files = grouped.get(observation.checkpoint_step)
        if not files:
            raise ValueError("checkpoint manifest mismatch")
        try:
            digest = canonical_model_manifest_digest(files)
        except ValueError as exc:
            raise ValueError("checkpoint artifact inventory is invalid") from exc
        if digest != observation.evaluated_model_manifest_digest:
            raise ValueError("checkpoint manifest mismatch")


def validate_evaluator_readiness_contract(readiness_contract: Any) -> dict[str, Any]:
    """Validate the small baseline-readiness policy registered with an evaluator."""

    if not isinstance(readiness_contract, dict):
        raise ValueError("autoresearch_evaluator_readiness_contract_invalid")
    good_id = readiness_contract.get("known_good_case_id")
    bad_id = readiness_contract.get("known_bad_case_id")
    repeat_count = readiness_contract.get("baseline_repeat_count")
    maximum_spread = readiness_contract.get("maximum_baseline_spread")
    if (
        not isinstance(good_id, str)
        or not good_id
        or not isinstance(bad_id, str)
        or not bad_id
        or good_id == bad_id
        or isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 3
        or repeat_count > 10
        or isinstance(maximum_spread, bool)
        or not isinstance(maximum_spread, (int, float))
        or not math.isfinite(float(maximum_spread))
        or float(maximum_spread) < 0
    ):
        raise ValueError("autoresearch_evaluator_readiness_contract_invalid")
    return {
        "known_good_case_id": good_id,
        "known_bad_case_id": bad_id,
        "baseline_repeat_count": repeat_count,
        "maximum_baseline_spread": float(maximum_spread),
    }


def validate_baseline_evaluator_readiness(
    evidence: AutoResearchEvaluationEvidence,
    *,
    primary_metric: str,
    readiness_contract: Any,
) -> None:
    """Enforce sealed baseline canaries when the evaluation suite declares them."""

    if readiness_contract is None:
        return

    contract = validate_evaluator_readiness_contract(readiness_contract)
    good_id = contract["known_good_case_id"]
    bad_id = contract["known_bad_case_id"]
    repeat_count = contract["baseline_repeat_count"]
    maximum_spread = contract["maximum_baseline_spread"]
    readiness = evidence.evaluator_readiness
    if readiness is None:
        raise ValueError("autoresearch_evaluator_readiness_missing")
    if (
        readiness.known_good_case_id != good_id
        or not readiness.known_good_passed
        or readiness.known_bad_case_id != bad_id
        or not readiness.known_bad_rejected
    ):
        raise ValueError("autoresearch_evaluator_canary_failed")
    if len(readiness.baseline_scores) != repeat_count:
        raise ValueError("autoresearch_baseline_repeat_count_mismatch")
    observed_spread = max(readiness.baseline_scores) - min(readiness.baseline_scores)
    if observed_spread > float(maximum_spread) and not math.isclose(
        observed_spread,
        float(maximum_spread),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("autoresearch_baseline_unstable")
    primary_value = evidence.metrics.get(primary_metric)
    if primary_value is None or not math.isclose(
        float(primary_value),
        statistics.fmean(readiness.baseline_scores),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("autoresearch_baseline_repeat_mean_mismatch")


def _read_bounded_file(path: Path, *, max_bytes: int) -> tuple[bytes, str, int]:
    """Read and hash one bounded file from the same handle and byte snapshot."""

    with path.open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("evaluation evidence exceeds the 1 MiB limit")
    return payload, hashlib.sha256(payload).hexdigest(), len(payload)


def _validate_remote_seal_execution(
    attempt: ActionAttempt,
    manifest: Any,
    *,
    campaign_compute_profile_id: str | None = None,
) -> None:
    """Bind one authenticated seal to its durable remote executor authority."""

    executor = attempt.executor
    compute_profile_id = executor.get("compute_profile_id")
    persisted_executor_id = executor.get("seal_executor_id")
    persisted_executor_version = executor.get("seal_executor_version")
    if (
        executor.get("kind") != "ssh_remote"
        or executor.get("stage") != attempt.stage.value
        or not isinstance(persisted_executor_id, str)
        or not persisted_executor_id
        or not isinstance(persisted_executor_version, str)
        or not persisted_executor_version
        or not isinstance(compute_profile_id, str)
        or not compute_profile_id
        or manifest.executor_id != persisted_executor_id
        or manifest.executor_version != persisted_executor_version
        or manifest.compute_profile_id != compute_profile_id
        or (
            campaign_compute_profile_id is not None
            and compute_profile_id != campaign_compute_profile_id
        )
    ):
        raise ValueError("sealed result executor or compute identity mismatch")


class SealedEvaluationReader:
    """Verify one sealed evaluator output without mutating campaign or ledger state."""

    def __init__(self, sealer: ArtifactSealer) -> None:
        self.sealer = sealer

    def read(
        self,
        *,
        attempt: ActionAttempt,
        artifacts: Iterable[CampaignArtifactRecord],
        sealed_directory: Path | None,
        evaluation_context: AutoResearchEvaluationContext,
        evaluation_context_sha256: HexDigest,
        evaluation_suite: EvaluationSuiteSpec,
        dataset_version: DatasetVersionSpec,
        evaluated_model_digest: HexDigest | None = None,
        training_artifacts: Iterable[CampaignArtifactRecord],
        checkpoint_artifacts: Iterable[CampaignArtifactRecord] = (),
        remote_manifest: SealedActionResult | None = None,
    ) -> tuple[AutoResearchEvaluationEvidence, CampaignArtifactRecord]:
        """Return verified typed evidence and its exact public artifact record."""

        if attempt.stage != StageKind.DEVELOPMENT_EVALUATION:
            raise ValueError("evaluation evidence requires a development-evaluation action")
        expected_seal_identity = {
            "expected_workspace_id": attempt.workspace_id,
            "expected_campaign_id": attempt.campaign_id,
            "expected_study_id": attempt.study_id,
            "expected_action_id": attempt.action_id,
            "expected_attempt_id": attempt.attempt_id,
            "expected_manifest_revision": attempt.manifest_revision,
            "expected_candidate_digest": attempt.candidate_digest,
            "expected_input_digest": attempt.input_digest,
            "expected_claim_generation": attempt.claim_generation,
        }
        try:
            if remote_manifest is None:
                if sealed_directory is None:
                    raise ValueError("local seal directory is missing")
                manifest = self.sealer.verify(sealed_directory, **expected_seal_identity)
            else:
                if sealed_directory is not None or not attempt.sealed_result_uri:
                    raise ValueError("remote seal reference is invalid")
                envelope = self.sealer.envelope_bytes(remote_manifest)
                manifest = self.sealer.verify_envelope_bytes(envelope, **expected_seal_identity)
                prefix = (
                    f"bashgym-remote-seal://{manifest.compute_profile_id}/"
                    f"{attempt.attempt_id}/sha256/"
                )
                if not attempt.sealed_result_uri.startswith(prefix):
                    raise ValueError("remote seal reference is invalid")
                persisted_digest = attempt.sealed_result_uri.removeprefix(prefix)
                if (
                    len(persisted_digest) != 64
                    or hashlib.sha256(envelope).hexdigest() != persisted_digest
                ):
                    raise ValueError("remote seal digest mismatch")
        except (ArtifactSealError, ValueError) as exc:
            raise ValueError("autoresearch evaluation seal is invalid") from exc
        if manifest.outcome != "completed":
            raise ValueError("evaluation evidence requires a completed action seal")
        try:
            _validate_remote_seal_execution(attempt, manifest)
        except ValueError as exc:
            raise ValueError("evaluation seal executor or compute identity mismatch") from exc

        matching_outputs = tuple(
            output
            for output in manifest.outputs
            if output.schema_name == AUTORESEARCH_EVALUATION_SCHEMA
        )
        matching_artifacts = tuple(
            artifact
            for artifact in artifacts
            if artifact.schema_name == AUTORESEARCH_EVALUATION_SCHEMA
            and artifact.producer_action_id == attempt.action_id
        )
        if len(matching_outputs) != 1 or len(matching_artifacts) != 1:
            raise ValueError(
                "evaluation seal must contain exactly one standardized evidence artifact"
            )
        output = matching_outputs[0]
        artifact = matching_artifacts[0]
        if output.path != AUTORESEARCH_EVALUATION_FILENAME:
            raise ValueError("evaluation evidence output filename is not standardized")
        if (
            artifact.workspace_id != attempt.workspace_id
            or artifact.campaign_id != attempt.campaign_id
            or artifact.metadata.get("attempt_id") != attempt.attempt_id
            or not artifact.sealed
            or not artifact.valid
            or artifact.sha256 != output.sha256
            or artifact.size_bytes != output.size_bytes
        ):
            raise ValueError("evaluation artifact identity does not match its action seal")

        if output.size_bytes > MAX_AUTORESEARCH_EVALUATION_BYTES:
            raise ValueError("evaluation evidence exceeds the 1 MiB limit")
        evidence: AutoResearchEvaluationEvidence | None = None
        payload: bytes | None = None
        if remote_manifest is not None:
            assert attempt.sealed_result_uri is not None
            expected_uri = (
                "bashgym-remote-artifact://"
                + attempt.sealed_result_uri.removeprefix("bashgym-remote-seal://")
                + "/"
                + output.path
            )
            normalized = artifact.metadata.get("normalized_evaluation")
            signature = artifact.metadata.get("projection_signature")
            if (
                artifact.uri != expected_uri
                or artifact.metadata.get("projection_key_version") != self.sealer.key_version
                or not isinstance(normalized, dict)
                or not isinstance(signature, str)
                or not hmac.compare_digest(
                    signature,
                    self.sealer.sign_canonical_payload(
                        normalized,
                        domain=AUTORESEARCH_NORMALIZED_EVALUATION_DOMAIN,
                    ),
                )
            ):
                raise ValueError("remote evaluation projection signature is invalid")
            evidence = AutoResearchEvaluationEvidence.model_validate(normalized)
        else:
            assert sealed_directory is not None
            root = sealed_directory.resolve()
            path = sealed_directory / output.path
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise ValueError("evaluation artifact path escapes its action seal") from exc
            if path.is_symlink() or not path.is_file():
                raise ValueError("evaluation artifact path must be a regular non-symlink file")
            try:
                artifact_path = Path(artifact.uri).resolve(strict=True)
            except OSError as exc:
                raise ValueError("evaluation artifact path is unavailable") from exc
            if artifact_path != resolved:
                raise ValueError("evaluation artifact path does not name the sealed output")
            payload, digest, size = _read_bounded_file(
                path, max_bytes=MAX_AUTORESEARCH_EVALUATION_BYTES
            )
            if digest != output.sha256 or size != output.size_bytes:
                raise ValueError("evaluation artifact digest does not match its action seal")

        context_bytes = evaluation_context_bytes(evaluation_context)
        if hashlib.sha256(context_bytes).hexdigest() != evaluation_context_sha256:
            raise ValueError("evaluation context digest mismatch")
        expected_context_identity = (
            attempt.workspace_id,
            attempt.campaign_id,
            attempt.study_id,
            attempt.action_id,
            attempt.attempt_id,
            attempt.candidate_digest,
        )
        actual_context_identity = (
            evaluation_context.workspace_id,
            evaluation_context.campaign_id,
            evaluation_context.study_id,
            evaluation_context.action_id,
            evaluation_context.attempt_id,
            evaluation_context.candidate_digest,
        )
        if actual_context_identity != expected_context_identity:
            raise ValueError("evaluation context identity mismatch")
        if (
            evaluation_suite.workspace_id != attempt.workspace_id
            or dataset_version.workspace_id != attempt.workspace_id
            or evaluation_suite.project_id != dataset_version.project_id
            or evaluation_suite.dataset_version_id != dataset_version.dataset_version_id
            or evaluation_context.evaluation_suite_id != evaluation_suite.evaluation_suite_id
            or evaluation_context.evaluation_code_digest != evaluation_suite.code_digest
            or evaluation_context.dataset_version_id != dataset_version.dataset_version_id
            or evaluation_context.dataset_content_digest != dataset_version.content_digest
        ):
            raise ValueError("evaluation context does not match registered evaluator inputs")

        model_artifacts = tuple(training_artifacts)
        if model_artifacts:
            model_manifest_digest = canonical_model_manifest_digest(model_artifacts)
        elif evaluated_model_digest is not None:
            model_manifest_digest = evaluated_model_digest
        else:
            raise ValueError("evaluation evidence requires an evaluated model identity")
        if evaluation_context.evaluated_model_manifest_digest != model_manifest_digest:
            raise ValueError("evaluation context model manifest mismatch")

        if evidence is None:
            try:
                evidence = AutoResearchEvaluationEvidence.model_validate_json(payload)
            except (OSError, ValueError) as exc:
                raise ValueError("autoresearch evaluation evidence is invalid") from exc
        evidence_identity = (
            evidence.campaign_id,
            evidence.study_id,
            evidence.action_id,
            evidence.attempt_id,
        )
        if evidence_identity != expected_context_identity[1:5]:
            raise ValueError("evaluation evidence identity mismatch")
        if evidence.candidate_digest != evaluation_context.candidate_digest:
            raise ValueError("evaluation evidence candidate digest mismatch")
        if (
            evidence.evaluation_suite_id != evaluation_context.evaluation_suite_id
            or evidence.evaluation_code_digest != evaluation_context.evaluation_code_digest
            or evidence.dataset_version_id != evaluation_context.dataset_version_id
        ):
            raise ValueError("evaluation evidence registered-input identity mismatch")
        if evidence.evaluated_model_manifest_digest != model_manifest_digest:
            raise ValueError("evaluation evidence model manifest mismatch")
        validate_checkpoint_observations(
            evidence,
            checkpoint_artifacts=checkpoint_artifacts,
        )
        return evidence, artifact


@dataclass(frozen=True)
class _ResolvedProjection:
    campaign: Any
    spec: Any
    control: Any
    proposal: Any
    study: Any
    study_attempts: tuple[ActionAttempt, ...]
    evaluation_attempt: ActionAttempt
    evaluation_suite: Any
    dataset_version: Any
    evidence: AutoResearchEvaluationEvidence
    evaluation_artifact: Any
    project_id: str
    experiment_id: str
    run_id: str
    attempt_id: str
    model_id: str
    model_version_id: str
    environment_id: str
    artifact_id: str
    evaluation_result_id: str
    evaluated_model_digest: str
    model_source_uri: str
    model_source_revision: str
    model_source_created_at: datetime
    model_source_metadata: dict[str, str]
    profile_id: str
    profile_revision: int
    profile_digest: str
    compute_profile_id: str
    recipe_digest: str
    actual_cost: float
    model_spec: Any
    model_version_spec: Any
    environment_spec: Any
    experiment_spec: Any
    run_spec: Any
    attempt_spec: Any
    artifact_spec: Any
    evaluation_spec: Any


class CampaignEvaluationProjector:
    """Deterministically project one sealed campaign evaluation into the ledger."""

    def __init__(
        self,
        repository: AutoResearchRepository,
        ledger: ExperimentLedgerRepository,
        reader: SealedEvaluationReader,
    ) -> None:
        if Path(repository.db_path).resolve() != Path(ledger.db_path).resolve():
            raise ValueError("autoresearch projector repositories must share one database")
        self.repository = repository
        self.ledger = ledger
        self.reader = reader

    @staticmethod
    def _invariant(code: str) -> None:
        from bashgym.campaigns.autoresearch import AutoResearchInvariantError

        raise AutoResearchInvariantError(code)

    @staticmethod
    def _is_hex_digest(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def project_diagnostic_and_ingest(
        self,
        workspace_id: str,
        campaign_id: str,
        proposal_id: str,
    ) -> Any:
        """Verify one sealed diagnostic artifact and persist its aggregate projection."""

        from bashgym.campaigns.autoresearch import (
            AutoResearchDiagnosticResult,
            ExperimentRole,
        )

        control = self.repository.get_autoresearch_proposal(workspace_id, campaign_id, proposal_id)
        proposal = self.repository.get_proposal(workspace_id, campaign_id, proposal_id)
        if control.role != ExperimentRole.DIAGNOSTIC or proposal.study_id is None:
            self._invariant("autoresearch_diagnostic_lineage_mismatch")
        attempts = self.repository.list_study_attempts(workspace_id, campaign_id, proposal.study_id)
        matching = tuple(
            attempt
            for attempt in attempts
            if attempt.stage == StageKind.CONTRACT_EVALUATION
            and attempt.status.value == "completed"
        )
        if len(matching) != 1:
            self._invariant("autoresearch_exact_completed_diagnostic_required")
        attempt = matching[0]
        executor = attempt.executor
        recipe = AutoResearchDiagnosticRecipe.model_validate(executor.get("diagnostic_recipe"))
        contract = executor.get("diagnostic_contract")
        if (
            executor.get("kind") != "ssh_remote"
            or executor.get("stage") != StageKind.CONTRACT_EVALUATION.value
            or executor.get("diagnostic_proposal_id") != proposal_id
            or not isinstance(contract, dict)
            or not attempt.sealed_result_uri
            or not attempt.sealed_result_uri.startswith("bashgym-remote-seal://")
        ):
            self._invariant("autoresearch_diagnostic_executor_mismatch")
        manifest = self.repository.get_attempt_result_manifest(workspace_id, attempt.attempt_id)
        try:
            _validate_remote_seal_execution(attempt, manifest)
        except ValueError:
            self._invariant("autoresearch_diagnostic_seal_invalid")
        outputs = tuple(
            output
            for output in manifest.outputs
            if output.schema_name == AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA
            and output.path == AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME
        )
        artifacts = tuple(
            artifact
            for artifact in self.repository.list_action_artifacts(
                workspace_id, campaign_id, attempt.action_id
            )
            if artifact.schema_name == AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA
            and artifact.producer_action_id == attempt.action_id
            and artifact.metadata.get("attempt_id") == attempt.attempt_id
        )
        if len(outputs) != 1 or len(artifacts) != 1:
            self._invariant("autoresearch_diagnostic_artifact_invalid")
        output, artifact = outputs[0], artifacts[0]
        normalized = artifact.metadata.get("normalized_diagnostic")
        signature = artifact.metadata.get("projection_signature")
        if (
            output.size_bytes > MAX_AUTORESEARCH_DIAGNOSTIC_BYTES
            or artifact.sha256 != output.sha256
            or artifact.size_bytes != output.size_bytes
            or not artifact.sealed
            or not artifact.valid
            or artifact.metadata.get("projection_key_version") != self.reader.sealer.key_version
            or not isinstance(normalized, dict)
            or not isinstance(signature, str)
            or not hmac.compare_digest(
                signature,
                self.reader.sealer.sign_canonical_payload(
                    normalized,
                    domain=AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN,
                ),
            )
        ):
            self._invariant("autoresearch_diagnostic_artifact_invalid")
        evidence = AutoResearchDiagnosticEvidence.model_validate(normalized.get("evidence"))
        projection = normalized.get("projection")
        if (
            not isinstance(projection, dict)
            or projection != public_diagnostic_projection(recipe, evidence)
            or (
                evidence.workspace_id,
                evidence.campaign_id,
                evidence.proposal_id,
                evidence.study_id,
                evidence.action_id,
                evidence.attempt_id,
            )
            != (
                workspace_id,
                campaign_id,
                proposal_id,
                proposal.study_id,
                attempt.action_id,
                attempt.attempt_id,
            )
        ):
            self._invariant("autoresearch_diagnostic_evidence_mismatch")
        budget_unit = self.repository.get_autoresearch_spec(
            workspace_id, campaign_id
        ).stop_rules.budget_unit
        actual_cost = sum(
            usage.amount for usage in evidence.resource_usage if usage.unit == budget_unit
        )
        return self.repository.record_autoresearch_diagnostic_result(
            AutoResearchDiagnosticResult(
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                proposal_id=proposal_id,
                study_id=proposal.study_id,
                attempt_id=attempt.attempt_id,
                status=evidence.status,
                projection=projection,
                actual_cost=actual_cost,
                recorded_at=attempt.updated_at,
            )
        )

    def _build_result(self, resolved: _ResolvedProjection) -> Any:
        """Derive the sole authoritative AutoResearch result from sealed projection."""

        from bashgym.campaigns.autoresearch import (
            AutoResearchResult,
            ExperimentOutcome,
            ExperimentProvenance,
        )
        from bashgym.campaigns.contracts import canonical_hash

        if len(resolved.study_attempts) > 100:
            self._invariant("autoresearch_result_reference_limit_exceeded")
        metric_value = resolved.evidence.metrics.get(resolved.spec.primary_metric)
        if metric_value is None or not math.isfinite(float(metric_value)):
            self._invariant("autoresearch_primary_metric_missing")
        return AutoResearchResult(
            result_id=(
                "autoresearch-result-"
                + canonical_hash([resolved.project_id, resolved.evaluation_result_id])[:32]
            ),
            workspace_id=resolved.campaign.workspace_id,
            campaign_id=resolved.campaign.campaign_id,
            proposal_id=resolved.control.proposal_id,
            study_id=resolved.study.study_id,
            role=resolved.control.role,
            provenance=ExperimentProvenance.REAL,
            outcome=ExperimentOutcome.COMPLETED,
            metric_name=resolved.spec.primary_metric,
            metric_value=float(metric_value),
            metrics={name: float(value) for name, value in resolved.evidence.metrics.items()},
            actual_cost=resolved.actual_cost,
            attempt_ids=tuple(attempt.attempt_id for attempt in resolved.study_attempts),
            evidence_references=(
                resolved.evaluation_result_id,
                resolved.run_id,
                resolved.artifact_id,
                resolved.evaluation_artifact.artifact_id,
            ),
            recorded_at=resolved.evidence.completed_at,
        )

    def _resolve(
        self,
        workspace_id: str,
        campaign_id: str,
        proposal_id: str,
    ) -> _ResolvedProjection:
        from bashgym.campaigns.artifacts import ArtifactSealError
        from bashgym.campaigns.autoresearch import AutoResearchCampaignCore
        from bashgym.campaigns.contracts import AttemptStatus, canonical_hash
        from bashgym.campaigns.remote import (
            RemoteResidentModelSource,
            SealedStageArtifactInput,
            SealedStageArtifactSource,
        )
        from bashgym.ledger.contracts import (
            ArtifactSpec,
            AttemptSpec,
            ContextStatus,
            EnvironmentSpec,
            EvaluationResultSpec,
            ExperimentSpec,
            ModelSpec,
            ModelVersionSpec,
            RunSpec,
            RunStatus,
            stable_ledger_id,
        )

        campaign = self.repository.get_campaign(workspace_id, campaign_id)
        spec = self.repository.get_autoresearch_spec(workspace_id, campaign_id)
        if not spec.require_sealed_artifact:
            self._invariant("autoresearch_sealed_projection_required")
        control = self.repository.get_autoresearch_proposal(workspace_id, campaign_id, proposal_id)
        proposal = self.repository.get_proposal(workspace_id, campaign_id, proposal_id)
        if proposal.study_id is None:
            self._invariant("autoresearch_projection_study_required")
        study = self.repository.get_study(workspace_id, campaign_id, proposal.study_id)
        if (
            campaign.workspace_id != workspace_id
            or campaign.campaign_id != campaign_id
            or spec.workspace_id != workspace_id
            or spec.campaign_id != campaign_id
            or control.workspace_id != workspace_id
            or control.campaign_id != campaign_id
            or control.proposal_id != proposal_id
            or proposal.proposal.workspace_id != workspace_id
            or proposal.proposal.campaign_id != campaign_id
            or proposal.proposal.proposal_id != proposal_id
            or study.workspace_id != workspace_id
            or study.campaign_id != campaign_id
            or study.proposal_id != proposal_id
            or study.status not in AutoResearchCampaignCore._SUCCESS_STUDY_STATES
        ):
            self._invariant("autoresearch_projection_lineage_mismatch")
        if AutoResearchCampaignCore._proposal_is_simulated(proposal.proposal):
            self._invariant("autoresearch_fake_executor_cannot_claim_real_result")

        study_attempts = self.repository.list_study_attempts(
            workspace_id, campaign_id, study.study_id
        )
        terminal = {
            AttemptStatus.COMPLETED,
            AttemptStatus.FAILED,
            AttemptStatus.FORCE_STOPPED,
            AttemptStatus.CANCELLED,
        }
        if not study_attempts or any(attempt.status not in terminal for attempt in study_attempts):
            self._invariant("autoresearch_study_attempts_not_terminal")
        completed_evaluations = tuple(
            attempt
            for attempt in study_attempts
            if attempt.stage == StageKind.DEVELOPMENT_EVALUATION
            and attempt.status == AttemptStatus.COMPLETED
        )
        if len(completed_evaluations) != 1:
            self._invariant("autoresearch_exact_completed_evaluation_required")
        evaluation_attempt = completed_evaluations[0]
        if evaluation_attempt.candidate_digest != study.candidate_digest:
            self._invariant("autoresearch_evaluation_candidate_mismatch")
        if not evaluation_attempt.sealed_result_uri:
            self._invariant("autoresearch_evaluation_seal_required")

        if spec.ledger_project_id is None or spec.evaluation_suite_id is None:
            self._invariant("autoresearch_evaluation_binding_required")
        project_id = spec.ledger_project_id
        executor = evaluation_attempt.executor
        binding = executor.get("evaluation_binding")
        if (
            executor.get("kind") != "ssh_remote"
            or executor.get("stage") != StageKind.DEVELOPMENT_EVALUATION.value
            or not isinstance(binding, dict)
            or binding.get("ledger_project_id") != project_id
            or binding.get("evaluation_suite_id") != spec.evaluation_suite_id
        ):
            self._invariant("autoresearch_evaluation_executor_binding_mismatch")
        evaluation_suite = self.repository.get_evaluation_suite_spec(
            workspace_id, project_id, spec.evaluation_suite_id
        )
        if evaluation_suite.dataset_version_id is None:
            self._invariant("autoresearch_evaluation_dataset_required")
        dataset_version = self.repository.get_dataset_version_spec(
            workspace_id, project_id, evaluation_suite.dataset_version_id
        )
        if (
            evaluation_suite.workspace_id != workspace_id
            or evaluation_suite.project_id != project_id
            or dataset_version.workspace_id != workspace_id
            or dataset_version.project_id != project_id
            or binding.get("evaluation_code_digest") != evaluation_suite.code_digest
            or binding.get("dataset_version_id") != dataset_version.dataset_version_id
            or binding.get("dataset_content_digest") != dataset_version.content_digest
            or executor.get("expected_script_sha256") != evaluation_suite.code_digest
            or evaluation_suite.metric_contract.get("primary_metric") != spec.primary_metric
        ):
            self._invariant("autoresearch_registered_evaluator_binding_mismatch")
        target_model_digest = canonical_hash(campaign.target_model.model_dump(mode="json"))
        profile_id = executor.get("profile_id")
        profile_revision = executor.get("profile_revision")
        profile_digest = executor.get("profile_digest")
        compute_profile_id = executor.get("compute_profile_id")
        seal_executor_id = executor.get("seal_executor_id")
        seal_executor_version = executor.get("seal_executor_version")
        evaluation_manifest_revision = self.repository.get_manifest_revision(
            workspace_id, campaign_id, evaluation_attempt.manifest_revision
        )
        if (
            executor.get("target_contract_key") != campaign.target_model.target_contract_key
            or executor.get("target_model_digest") != target_model_digest
            or not isinstance(profile_id, str)
            or not profile_id
            or not isinstance(profile_revision, int)
            or profile_revision < 1
            or not self._is_hex_digest(profile_digest)
            or not isinstance(compute_profile_id, str)
            or not compute_profile_id
            or compute_profile_id != evaluation_manifest_revision.manifest.compute_profile_id
        ):
            self._invariant("autoresearch_evaluation_environment_binding_mismatch")

        checkpoint_artifacts: tuple[Any, ...] = ()
        try:
            registered_base = executor.get("registered_base_model")
            if registered_base is not None:
                from bashgym.campaigns.remote import RegisteredRemoteModelSource

                source = RegisteredRemoteModelSource.model_validate(registered_base)
                if (
                    control.role.value != "baseline"
                    or executor.get("source_training") is not None
                    or executor.get("sealed_stage_artifact_inputs")
                    or source.compute_profile_id != compute_profile_id
                    or source.target_contract_key != campaign.target_model.target_contract_key
                ):
                    self._invariant("autoresearch_evaluated_model_source_invalid")
                evaluated_model_digest = source.physical_model_digest
                training_artifacts: tuple[Any, ...] = ()
                model_source_uri = f"autoresearch-registered-model://{source.source_id}"
                model_source_revision = source.source_id
                model_source_created_at = evaluation_attempt.created_at
                model_source_metadata = {
                    "source_kind": "registered_base_model",
                    "target_model_digest": source.model_digest,
                }
                if source.artifact_receipt is not None:
                    model_source_metadata.update(
                        {
                            "model_id": source.artifact_receipt.model_id,
                            "model_revision": source.artifact_receipt.revision,
                            "artifact_manifest_sha256": (
                                source.artifact_receipt.artifact_manifest_sha256
                            ),
                        }
                    )
            elif executor.get("remote_resident_model") is not None:
                source = RemoteResidentModelSource.model_validate(
                    executor.get("remote_resident_model")
                )
                if control.role.value != "candidate":
                    self._invariant("autoresearch_evaluated_model_source_invalid")
                training_attempt, source_stage_index = (
                    self.repository.get_immediately_preceding_training_attempt(
                        workspace_id,
                        campaign_id,
                        study.study_id,
                        evaluation_attempt.action_id,
                    )
                )
                expected_source = self.repository.remote_resident_full_training_source(
                    workspace_id,
                    campaign_id,
                    study.study_id,
                    source_stage_index + 1,
                )
                if (
                    source != expected_source
                    or training_attempt.candidate_digest != study.candidate_digest
                    or training_attempt.status.value != "completed"
                    or training_attempt.stage != StageKind.FULL_TRAINING
                    or not training_attempt.sealed_result_uri
                ):
                    self._invariant("autoresearch_training_checkpoint_lineage_mismatch")
                training_artifacts = tuple(
                    artifact
                    for artifact in self.repository.list_action_artifacts(
                        workspace_id, campaign_id, training_attempt.action_id
                    )
                    if artifact.schema_name == "huggingface_model_file.v1"
                )
                checkpoint_artifacts = tuple(
                    artifact
                    for artifact in self.repository.list_action_artifacts(
                        workspace_id, campaign_id, training_attempt.action_id
                    )
                    if artifact.schema_name == "huggingface_checkpoint_file.v1"
                    and artifact.metadata.get("attempt_id") == training_attempt.attempt_id
                )
                if (
                    not training_artifacts
                    or canonical_model_manifest_digest(training_artifacts) != source.model_digest
                ):
                    self._invariant("autoresearch_evaluated_model_artifact_mismatch")
                training_manifest = self.repository.get_attempt_result_manifest(
                    workspace_id, training_attempt.attempt_id
                )
                training_envelope = self.reader.sealer.envelope_bytes(training_manifest)
                training_prefix = (
                    f"bashgym-remote-seal://{source.compute_profile_id}/"
                    f"{training_attempt.attempt_id}/sha256/"
                )
                if not training_attempt.sealed_result_uri.startswith(
                    training_prefix
                ) or hashlib.sha256(
                    training_envelope
                ).hexdigest() != training_attempt.sealed_result_uri.removeprefix(
                    training_prefix
                ):
                    self._invariant("autoresearch_training_seal_invalid")
                evaluated_model_digest = source.model_digest
                model_source_uri = (
                    "autoresearch-remote-checkpoint://sha256/" + evaluated_model_digest
                )
                model_source_revision = ""
                model_source_created_at = training_attempt.updated_at
                model_source_metadata = {
                    "source_kind": "remote_resident_training_output",
                    "training_attempt_id": training_attempt.attempt_id,
                }
            else:
                source = SealedStageArtifactSource.model_validate(executor.get("source_training"))
                sealed_inputs = tuple(
                    SealedStageArtifactInput.model_validate(item)
                    for item in executor.get("sealed_stage_artifact_inputs", ())
                )
                if not sealed_inputs or control.role.value != "candidate":
                    self._invariant("autoresearch_evaluated_model_source_invalid")
                training_attempt, source_stage_index = (
                    self.repository.get_immediately_preceding_training_attempt(
                        workspace_id,
                        campaign_id,
                        study.study_id,
                        evaluation_attempt.action_id,
                    )
                )
                expected_source = SealedStageArtifactSource(
                    campaign_id=training_attempt.campaign_id,
                    study_id=training_attempt.study_id,
                    action_id=training_attempt.action_id,
                    attempt_id=training_attempt.attempt_id,
                    stage_index=source_stage_index,
                )
                if (
                    source != expected_source
                    or training_attempt.candidate_digest != study.candidate_digest
                    or training_attempt.status.value != "completed"
                    or training_attempt.stage != StageKind.FULL_TRAINING
                    or not training_attempt.sealed_result_uri
                ):
                    self._invariant("autoresearch_training_checkpoint_lineage_mismatch")
                training_artifacts = tuple(
                    self.repository.get_artifact(
                        workspace_id, campaign_id, item.campaign_artifact_id
                    )
                    for item in sealed_inputs
                )
                checkpoint_artifacts = tuple(
                    artifact
                    for artifact in self.repository.list_action_artifacts(
                        workspace_id, campaign_id, training_attempt.action_id
                    )
                    if artifact.schema_name == "huggingface_checkpoint_file.v1"
                    and artifact.metadata.get("attempt_id") == training_attempt.attempt_id
                )
                if any(
                    artifact.producer_action_id != training_attempt.action_id
                    or artifact.metadata.get("attempt_id") != training_attempt.attempt_id
                    or not artifact.sealed
                    or not artifact.valid
                    for artifact in training_artifacts
                ):
                    self._invariant("autoresearch_evaluated_model_artifact_mismatch")
                try:
                    training_manifest = self.reader.sealer.verify(
                        Path(training_attempt.sealed_result_uri),
                        expected_workspace_id=workspace_id,
                        expected_campaign_id=campaign_id,
                        expected_study_id=study.study_id,
                        expected_action_id=training_attempt.action_id,
                        expected_attempt_id=training_attempt.attempt_id,
                        expected_manifest_revision=training_attempt.manifest_revision,
                        expected_candidate_digest=training_attempt.candidate_digest,
                        expected_input_digest=training_attempt.input_digest,
                        expected_claim_generation=training_attempt.claim_generation,
                    )
                except ArtifactSealError:
                    self._invariant("autoresearch_training_seal_invalid")
                expected_outputs = {
                    (f"final/{item.remote_relative_path.removeprefix('model/')}", item.sha256)
                    for item in sealed_inputs
                }
                sealed_outputs = {
                    (output.path, output.sha256)
                    for output in training_manifest.outputs
                    if output.schema_name == "huggingface_model_file.v1"
                }
                if training_manifest.outcome != "completed" or sealed_outputs != expected_outputs:
                    self._invariant("autoresearch_training_seal_model_set_mismatch")
                evaluated_model_digest = canonical_model_manifest_digest(sealed_inputs)
                model_source_uri = "autoresearch-checkpoint://sha256/" + evaluated_model_digest
                model_source_revision = ""
                model_source_created_at = training_attempt.updated_at
                model_source_metadata = {
                    "source_kind": "sealed_training_output",
                    "training_attempt_id": training_attempt.attempt_id,
                }
        except (TypeError, ValueError):
            self._invariant("autoresearch_evaluated_model_source_invalid")
        if executor.get("evaluated_model_digest") != evaluated_model_digest:
            self._invariant("autoresearch_evaluated_model_source_invalid")
        context = AutoResearchEvaluationContext(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            study_id=study.study_id,
            action_id=evaluation_attempt.action_id,
            attempt_id=evaluation_attempt.attempt_id,
            candidate_digest=study.candidate_digest,
            evaluation_suite_id=evaluation_suite.evaluation_suite_id,
            evaluation_code_digest=evaluation_suite.code_digest,
            dataset_version_id=dataset_version.dataset_version_id,
            dataset_content_digest=dataset_version.content_digest,
            evaluated_model_manifest_digest=evaluated_model_digest,
        )
        context_sha256 = hashlib.sha256(evaluation_context_bytes(context)).hexdigest()
        if executor.get("evaluation_context_sha256") != context_sha256:
            self._invariant("autoresearch_evaluation_context_digest_mismatch")
        campaign_artifacts = self.repository.list_action_artifacts(
            workspace_id, campaign_id, evaluation_attempt.action_id
        )
        evaluation_candidates = tuple(
            artifact
            for artifact in campaign_artifacts
            if artifact.producer_action_id == evaluation_attempt.action_id
            and artifact.schema_name == AUTORESEARCH_EVALUATION_SCHEMA
        )
        if len(evaluation_candidates) == 1:
            if evaluation_attempt.sealed_result_uri.startswith("bashgym-remote-seal://"):
                evidence_size = evaluation_candidates[0].size_bytes
            else:
                try:
                    evidence_size = Path(evaluation_candidates[0].uri).stat().st_size
                except OSError:
                    self._invariant("autoresearch_evaluation_evidence_invalid")
            if evidence_size > MAX_AUTORESEARCH_EVALUATION_BYTES:
                self._invariant("autoresearch_evaluation_evidence_exceeds_limit")
        remote_evaluation = evaluation_attempt.sealed_result_uri.startswith(
            "bashgym-remote-seal://"
        )
        try:
            if remote_evaluation:
                evaluation_manifest = self.repository.get_attempt_result_manifest(
                    workspace_id, evaluation_attempt.attempt_id
                )
            else:
                evaluation_manifest = self.reader.sealer.verify(
                    Path(evaluation_attempt.sealed_result_uri),
                    expected_workspace_id=workspace_id,
                    expected_campaign_id=campaign_id,
                    expected_study_id=study.study_id,
                    expected_action_id=evaluation_attempt.action_id,
                    expected_attempt_id=evaluation_attempt.attempt_id,
                    expected_manifest_revision=evaluation_attempt.manifest_revision,
                    expected_candidate_digest=evaluation_attempt.candidate_digest,
                    expected_input_digest=evaluation_attempt.input_digest,
                    expected_claim_generation=evaluation_attempt.claim_generation,
                )
        except (ArtifactSealError, ValueError):
            self._invariant("autoresearch_evaluation_seal_invalid")
        try:
            _validate_remote_seal_execution(
                evaluation_attempt,
                evaluation_manifest,
                campaign_compute_profile_id=(
                    evaluation_manifest_revision.manifest.compute_profile_id
                ),
            )
        except ValueError:
            self._invariant("autoresearch_evaluation_seal_executor_binding_mismatch")
        try:
            evidence, evaluation_artifact = self.reader.read(
                attempt=evaluation_attempt,
                artifacts=campaign_artifacts,
                sealed_directory=(
                    None if remote_evaluation else Path(evaluation_attempt.sealed_result_uri)
                ),
                evaluation_context=context,
                evaluation_context_sha256=context_sha256,
                evaluation_suite=evaluation_suite,
                dataset_version=dataset_version,
                evaluated_model_digest=evaluated_model_digest,
                training_artifacts=training_artifacts,
                checkpoint_artifacts=checkpoint_artifacts,
                remote_manifest=(evaluation_manifest if remote_evaluation else None),
            )
        except (OSError, TypeError, ValueError):
            self._invariant("autoresearch_evaluation_evidence_invalid")
        from bashgym.campaigns.autoresearch import ExperimentRole

        if control.role == ExperimentRole.BASELINE:
            try:
                validate_baseline_evaluator_readiness(
                    evidence,
                    primary_metric=spec.primary_metric,
                    readiness_contract=evaluation_suite.metric_contract.get("evaluator_readiness"),
                )
            except ValueError as exc:
                self._invariant(str(exc))

        usage = self.repository.study_budget_usage(
            workspace_id, campaign_id, study.study_id, spec.stop_rules.budget_unit
        )
        if not math.isfinite(float(usage["reserved"])) or abs(usage["reserved"]) > 1e-9:
            self._invariant("autoresearch_study_budget_not_settled")
        if (
            not math.isfinite(float(usage["actual"]))
            or usage["actual"] < 0
            or usage["actual"] > spec.stop_rules.max_total_cost
        ):
            self._invariant("autoresearch_study_budget_actual_invalid")

        recipe_digest = executor.get("recipe_digest")
        if not self._is_hex_digest(recipe_digest):
            self._invariant("autoresearch_evaluation_recipe_digest_invalid")

        experiment_id = stable_ledger_id(
            "autoresearch-experiment", workspace_id, campaign_id, proposal_id
        )
        run_id = stable_ledger_id(
            "autoresearch-run",
            workspace_id,
            campaign_id,
            study.study_id,
            evaluation_attempt.action_id,
        )
        attempt_id = stable_ledger_id(
            "autoresearch-attempt",
            workspace_id,
            campaign_id,
            evaluation_attempt.attempt_id,
        )
        model_id = stable_ledger_id(
            "autoresearch-model",
            campaign.target_model.target_contract_key,
            target_model_digest,
        )
        model_version_id = stable_ledger_id(
            "autoresearch-model-version",
            target_model_digest,
            study.candidate_digest,
            evaluated_model_digest,
        )
        environment_id = stable_ledger_id(
            "autoresearch-environment",
            compute_profile_id,
            profile_digest,
            seal_executor_id,
            seal_executor_version,
        )
        artifact_id = stable_ledger_id(
            "autoresearch-artifact",
            evaluation_artifact.artifact_id,
            evaluation_artifact.sha256,
        )
        evaluation_result_id = stable_ledger_id(
            "autoresearch-evaluation",
            campaign_id,
            proposal_id,
            evaluation_suite.evaluation_suite_id,
            evaluation_artifact.sha256,
        )
        try:
            model_spec = ModelSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                model_id=model_id,
                display_name=campaign.target_model.target_contract_key,
                task_type=campaign.target_model.task,
                architecture=campaign.target_model.base_model_ref,
                metadata={"target_contract_key": campaign.target_model.target_contract_key},
                created_at=campaign.created_at,
            )
            model_version_spec = ModelVersionSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                model_id=model_id,
                model_version_id=model_version_id,
                source_uri=model_source_uri,
                source_revision=model_source_revision,
                parent_model_version_id=None,
                config_digest=study.candidate_digest,
                metadata={
                    "evaluated_model_digest": evaluated_model_digest,
                    **model_source_metadata,
                },
                created_at=model_source_created_at,
            )
            environment_spec = EnvironmentSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                environment_id=environment_id,
                compute_target=compute_profile_id,
                runtime_digest=profile_digest,
                metadata={
                    "executor_profile_id": profile_id,
                    "executor_profile_revision": profile_revision,
                    "seal_executor_id": seal_executor_id,
                    "seal_executor_version": seal_executor_version,
                },
                created_at=evaluation_attempt.created_at,
            )
            experiment_spec = ExperimentSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                experiment_id=experiment_id,
                name=f"AutoResearch {control.role.value}: {proposal.proposal.proposal_id}",
                objective=proposal.proposal.hypothesis,
                campaign_id=campaign_id,
                metadata={"proposal_id": proposal.proposal.proposal_id},
                created_at=control.created_at,
            )
            run_spec = RunSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                experiment_id=experiment_id,
                run_id=run_id,
                source_system="bashgym",
                source_run_id=evaluation_attempt.action_id,
                campaign_id=campaign_id,
                study_id=study.study_id,
                action_id=evaluation_attempt.action_id,
                run_kind="evaluation",
                task_type=campaign.target_model.task,
                training_method="autoresearch",
                status=RunStatus.COMPLETED,
                context_status=ContextStatus.VERIFIED,
                model_version_id=model_version_id,
                dataset_version_id=dataset_version.dataset_version_id,
                environment_id=environment_id,
                recipe_digest=recipe_digest,
                config={
                    "proposal_id": proposal.proposal.proposal_id,
                    "candidate_digest": study.candidate_digest,
                    "compute_profile_id": compute_profile_id,
                    "executor_profile_id": profile_id,
                    "executor_profile_revision": profile_revision,
                    "executor_profile_digest": profile_digest,
                    "seal_executor_id": seal_executor_id,
                    "seal_executor_version": seal_executor_version,
                    "evaluation_artifact_id": evaluation_artifact.artifact_id,
                },
                correlation_id=evaluation_result_id,
                queued_at=evaluation_attempt.created_at,
            )
            attempt_spec = AttemptSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_number=evaluation_attempt.attempt_number,
                source_attempt_id=evaluation_attempt.attempt_id,
                status=RunStatus.COMPLETED,
                metadata={
                    "candidate_digest": study.candidate_digest,
                    "campaign_action_id": evaluation_attempt.action_id,
                },
                created_at=evaluation_attempt.created_at,
            )
            artifact_spec = ArtifactSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                artifact_id=artifact_id,
                run_id=run_id,
                attempt_id=attempt_id,
                kind=AUTORESEARCH_EVALUATION_SCHEMA,
                uri=evaluation_artifact.uri,
                sha256=evaluation_artifact.sha256,
                size_bytes=evaluation_artifact.size_bytes,
                media_type="application/json",
                metadata={
                    "campaign_artifact_id": evaluation_artifact.artifact_id,
                    "campaign_action_id": evaluation_attempt.action_id,
                    "campaign_attempt_id": evaluation_attempt.attempt_id,
                    "schema_name": evaluation_artifact.schema_name,
                    "sealed": evaluation_artifact.sealed,
                    "valid": evaluation_artifact.valid,
                },
                created_at=evaluation_artifact.created_at,
            )
            projected_slice_metrics = dict(evidence.slice_metrics)
            if evidence.checkpoint_observations:
                projected_slice_metrics[AUTORESEARCH_CHECKPOINT_TRAJECTORY_KEY] = [
                    item.model_dump(mode="json") for item in evidence.checkpoint_observations
                ]
            if evidence.failure_observations:
                projected_slice_metrics[AUTORESEARCH_FAILURE_OBSERVATIONS_KEY] = [
                    item.model_dump(mode="json") for item in evidence.failure_observations
                ]
            evaluation_spec = EvaluationResultSpec(
                workspace_id=workspace_id,
                project_id=project_id,
                evaluation_result_id=evaluation_result_id,
                evaluation_suite_id=evaluation_suite.evaluation_suite_id,
                run_id=run_id,
                attempt_id=attempt_id,
                model_version_id=model_version_id,
                status=RunStatus.COMPLETED,
                metrics=evidence.metrics,
                slice_metrics=projected_slice_metrics,
                artifact_id=artifact_id,
                compared_to_result_id=None,
                started_at=evidence.started_at,
                completed_at=evidence.completed_at,
            )
        except (TypeError, ValueError):
            self._invariant("autoresearch_projection_spec_invalid")
        return _ResolvedProjection(
            campaign=campaign,
            spec=spec,
            control=control,
            proposal=proposal,
            study=study,
            study_attempts=study_attempts,
            evaluation_attempt=evaluation_attempt,
            evaluation_suite=evaluation_suite,
            dataset_version=dataset_version,
            evidence=evidence,
            evaluation_artifact=evaluation_artifact,
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run_id,
            attempt_id=attempt_id,
            model_id=model_id,
            model_version_id=model_version_id,
            environment_id=environment_id,
            artifact_id=artifact_id,
            evaluation_result_id=evaluation_result_id,
            evaluated_model_digest=evaluated_model_digest,
            model_source_uri=model_source_uri,
            model_source_revision=model_source_revision,
            model_source_created_at=model_source_created_at,
            model_source_metadata=model_source_metadata,
            profile_id=profile_id,
            profile_revision=profile_revision,
            profile_digest=profile_digest,
            compute_profile_id=compute_profile_id,
            recipe_digest=recipe_digest,
            actual_cost=float(usage["actual"]),
            model_spec=model_spec,
            model_version_spec=model_version_spec,
            environment_spec=environment_spec,
            experiment_spec=experiment_spec,
            run_spec=run_spec,
            attempt_spec=attempt_spec,
            artifact_spec=artifact_spec,
            evaluation_spec=evaluation_spec,
        )

    def _write(self, resolved: _ResolvedProjection) -> None:
        self.ledger.register_model(resolved.model_spec)
        self.ledger.register_model_version(resolved.model_version_spec)
        self.ledger.register_environment(resolved.environment_spec)
        self.ledger.register_experiment(resolved.experiment_spec)
        self.ledger.register_run(resolved.run_spec)
        self.ledger.register_attempt(resolved.attempt_spec)
        self.ledger.record_artifact(resolved.artifact_spec)
        self.ledger.record_evaluation_result(resolved.evaluation_spec)

    def _project_and_ingest(
        self,
        workspace_id: str,
        campaign_id: str,
        proposal_id: str,
        expected_evaluation_result_id: str | None,
    ) -> AutoResearchOutcomeRecord:
        from bashgym.campaigns.autoresearch import AutoResearchLedgerCommitContext

        resolved = self._resolve(workspace_id, campaign_id, proposal_id)
        if (
            expected_evaluation_result_id is not None
            and expected_evaluation_result_id != resolved.evaluation_result_id
        ):
            self._invariant("autoresearch_expected_evaluation_result_mismatch")
        self._write(resolved)
        return self.repository._record_autoresearch_result(
            self._build_result(resolved),
            ledger_context=AutoResearchLedgerCommitContext(
                project_id=resolved.project_id,
                experiment_id=resolved.experiment_id,
                run_id=resolved.run_id,
                attempt_id=resolved.attempt_id,
                correlation_id=resolved.evaluation_result_id,
            ),
        )

    def project_and_ingest(
        self,
        workspace_id: str,
        campaign_id: str,
        proposal_id: str,
        *,
        expected_evaluation_result_id: str | None = None,
    ) -> AutoResearchOutcomeRecord:
        return self._project_and_ingest(
            workspace_id,
            campaign_id,
            proposal_id,
            expected_evaluation_result_id=expected_evaluation_result_id,
        )


def ingest_completed_evaluation(
    projector: CampaignEvaluationProjector,
    *,
    workspace_id: str,
    campaign_id: str,
    proposal_id: str,
    expected_evaluation_result_id: str | None = None,
) -> AutoResearchOutcomeRecord:
    """Reconcile a completed evaluation exclusively through sealed projection."""

    return projector._project_and_ingest(
        workspace_id,
        campaign_id,
        proposal_id,
        expected_evaluation_result_id=expected_evaluation_result_id,
    )


__all__ = [
    "AUTORESEARCH_EVALUATION_CONTEXT_FILENAME",
    "AUTORESEARCH_EVALUATION_CONTEXT_SCHEMA",
    "AUTORESEARCH_EVALUATION_FILENAME",
    "AUTORESEARCH_EVALUATION_SCHEMA",
    "AUTORESEARCH_CHECKPOINT_TRAJECTORY_KEY",
    "AUTORESEARCH_NORMALIZED_EVALUATION_DOMAIN",
    "AutoResearchEvaluationContext",
    "AutoResearchEvaluationEvidence",
    "AutoResearchCheckpointObservation",
    "AutoResearchEvaluatorReadiness",
    "CampaignEvaluationProjector",
    "MAX_AUTORESEARCH_EVALUATION_BYTES",
    "SealedEvaluationReader",
    "evaluation_context_bytes",
    "validate_baseline_evaluator_readiness",
    "validate_checkpoint_observations",
    "validate_evaluator_readiness_contract",
    "ingest_completed_evaluation",
]
