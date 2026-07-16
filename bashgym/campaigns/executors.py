"""Executor protocol and deterministic fake used to prove the durable loop."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.contracts import (
    ActionAttempt,
    ContractModel,
    ResourceUsage,
    SealedActionResult,
    utc_now,
)
from bashgym.campaigns.evaluation import (
    DevelopmentComparison,
    DevelopmentDatasetContract,
    DevelopmentGateContract,
    RetrievalEvaluationArtifact,
    compare_development_evaluations,
    load_retrieval_evaluation_artifact,
)
from bashgym.campaigns.nemo_gym_evidence import (
    NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
    NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA,
    load_nemo_gym_campaign_evidence,
)
from bashgym.campaigns.remote import RemoteObservation, RemoteRunIdentity


@dataclass(frozen=True)
class FakeExecutionRequest:
    workspace_id: str
    campaign_id: str
    study_id: str
    action_id: str
    attempt_id: str
    manifest_revision: int
    candidate_digest: str
    input_digest: str
    claim_generation: int
    compute_profile_id: str = "fake-local"
    steps: int = 8


class FakeExecutor:
    """Write deterministic loss/evidence artifacts without invoking model training."""

    executor_id = "campaign-fake-executor"
    executor_version = "1"

    def __init__(self, artifact_root: Path, sealer: ArtifactSealer):
        self.artifact_root = artifact_root
        self.sealer = sealer
        self.execution_count = 0

    def execute(self, request: FakeExecutionRequest) -> tuple[Path, SealedActionResult]:
        if request.steps < 2:
            raise ValueError("fake execution requires at least two metric steps")
        self.execution_count += 1
        started_at = utc_now()
        temporary = (
            self.artifact_root / ".tmp" / f"{request.action_id}.{request.attempt_id}.{uuid4().hex}"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        metrics_path = temporary / "training_metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
            for step in range(1, request.steps + 1):
                loss = round(1.0 / (1.0 + step / 2.0), 8)
                handle.write(json.dumps({"step": step, "loss": loss}, sort_keys=True) + "\n")
        summary = {
            "action_id": request.action_id,
            "attempt_id": request.attempt_id,
            "final_loss": round(1.0 / (1.0 + request.steps / 2.0), 8),
            "steps": request.steps,
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        outputs = self.sealer.describe_outputs(
            temporary,
            {
                "summary.json": "campaign_fake_summary.v1",
                "training_metrics.jsonl": "training_metrics_jsonl.v1",
            },
        )
        ended_at = utc_now()
        manifest = SealedActionResult(
            workspace_id=request.workspace_id,
            campaign_id=request.campaign_id,
            study_id=request.study_id,
            action_id=request.action_id,
            attempt_id=request.attempt_id,
            manifest_revision=request.manifest_revision,
            candidate_digest=request.candidate_digest,
            input_digest=request.input_digest,
            claim_generation=request.claim_generation,
            executor_id=self.executor_id,
            executor_version=self.executor_version,
            compute_profile_id=request.compute_profile_id,
            remote_process_identity={"kind": "fake", "run_id": request.attempt_id},
            started_at=started_at,
            ended_at=ended_at,
            outcome="completed",
            exit_code=0,
            exit_reason="fake execution completed",
            resource_usage=(
                ResourceUsage(
                    unit="step_count",
                    amount=float(request.steps),
                    source="fake_executor",
                    confidence="measured",
                ),
            ),
            outputs=outputs,
        )
        sealed = (
            self.artifact_root
            / request.workspace_id
            / request.campaign_id
            / request.study_id
            / request.action_id
            / request.attempt_id
        )
        self.sealer.seal(temporary, sealed, manifest)
        return sealed, manifest


class RemoteOutputSealer:
    """Convert a closed, downloaded remote run into the common evidence seal."""

    executor_id = "campaign-ssh-remote-executor"
    executor_version = "1"

    def __init__(self, artifact_root: Path, sealer: ArtifactSealer):
        self.artifact_root = artifact_root.resolve()
        self.sealer = sealer

    def seal_completed(
        self,
        attempt: ActionAttempt,
        identity: RemoteRunIdentity,
        observation: RemoteObservation,
        temporary: Path,
    ) -> tuple[Path, SealedActionResult]:
        if observation.exit_code != 0 or observation.state.value != "completed":
            raise ValueError("remote completion must have a proven zero exit code")
        evidence_path = temporary / NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME
        if evidence_path.exists():
            load_nemo_gym_campaign_evidence(evidence_path, expected_attempt=attempt)
        schemas = {
            path.relative_to(temporary).as_posix(): self._schema_for(path, temporary)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        if not schemas:
            raise ValueError("remote completion has no downloaded evidence")
        outputs = self.sealer.describe_outputs(temporary, schemas)
        elapsed_seconds = max(0.0, (observation.observed_at - identity.launched_at).total_seconds())
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
            executor_id=self.executor_id,
            executor_version=self.executor_version,
            compute_profile_id=identity.compute_profile_id,
            remote_process_identity=identity.model_dump(mode="json"),
            started_at=identity.launched_at,
            ended_at=observation.observed_at,
            outcome="completed",
            exit_code=0,
            exit_reason=observation.safe_reason,
            resource_usage=(
                ResourceUsage(
                    unit="wall_clock_seconds",
                    amount=elapsed_seconds,
                    source="remote_supervisor",
                    confidence="measured",
                ),
            ),
            outputs=outputs,
        )
        sealed = (
            self.artifact_root
            / attempt.workspace_id
            / attempt.campaign_id
            / attempt.study_id
            / attempt.action_id
            / attempt.attempt_id
        )
        self.sealer.seal(temporary, sealed, manifest)
        return sealed, manifest

    def seal_terminal(
        self,
        attempt: ActionAttempt,
        identity: RemoteRunIdentity,
        observation: RemoteObservation,
        temporary: Path,
        *,
        outcome: Literal["failed", "cancelled", "force_stopped"],
    ) -> tuple[Path, SealedActionResult]:
        """Seal failure/cancellation evidence without pretending it is a model result."""

        if observation.state.value != "failed" or observation.exit_code in {0, None}:
            raise ValueError("remote terminal evidence requires a proven failing exit code")
        schemas = {
            path.relative_to(temporary).as_posix(): self._schema_for(path, temporary)
            for path in temporary.rglob("*")
            if path.is_file()
        }
        required = {"training.log", "exit_code", "launch_manifest.json"}
        if not required.issubset(schemas):
            raise ValueError("remote terminal evidence is incomplete")
        outputs = self.sealer.describe_outputs(temporary, schemas)
        elapsed_seconds = max(0.0, (observation.observed_at - identity.launched_at).total_seconds())
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
            executor_id=self.executor_id,
            executor_version=self.executor_version,
            compute_profile_id=identity.compute_profile_id,
            remote_process_identity=identity.model_dump(mode="json"),
            started_at=identity.launched_at,
            ended_at=observation.observed_at,
            outcome=outcome,
            exit_code=observation.exit_code,
            exit_reason=observation.safe_reason,
            resource_usage=(
                ResourceUsage(
                    unit="wall_clock_seconds",
                    amount=elapsed_seconds,
                    source="remote_supervisor",
                    confidence="measured",
                ),
            ),
            log_reference="training.log",
            outputs=outputs,
        )
        sealed = (
            self.artifact_root
            / attempt.workspace_id
            / attempt.campaign_id
            / attempt.study_id
            / attempt.action_id
            / attempt.attempt_id
        )
        self.sealer.seal(temporary, sealed, manifest)
        return sealed, manifest

    def seal_unlaunched_cancelled(
        self,
        attempt: ActionAttempt,
        *,
        compute_profile_id: str,
    ) -> tuple[Path, SealedActionResult]:
        """Seal a cancellation that won the race before any remote side effect."""

        observed_at = utc_now()
        temporary = (
            self.artifact_root / ".tmp" / f"{attempt.action_id}.{attempt.attempt_id}.{uuid4().hex}"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        evidence = {
            "schema_version": "campaign_unlaunched_cancellation.v1",
            "attempt_id": attempt.attempt_id,
            "reason": "campaign_cancelled_before_remote_launch",
        }
        (temporary / "cancellation.json").write_text(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        outputs = self.sealer.describe_outputs(
            temporary, {"cancellation.json": "campaign_unlaunched_cancellation.v1"}
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
            executor_id=self.executor_id,
            executor_version=self.executor_version,
            compute_profile_id=compute_profile_id,
            remote_process_identity={"kind": "unlaunched"},
            started_at=observed_at,
            ended_at=observed_at,
            outcome="cancelled",
            exit_reason="campaign_cancelled_before_remote_launch",
            outputs=outputs,
        )
        sealed = (
            self.artifact_root
            / attempt.workspace_id
            / attempt.campaign_id
            / attempt.study_id
            / attempt.action_id
            / attempt.attempt_id
        )
        self.sealer.seal(temporary, sealed, manifest)
        return sealed, manifest

    @staticmethod
    def _schema_for(path: Path, root: Path) -> str:
        relative = path.relative_to(root).as_posix()
        if relative == NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME:
            return NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA
        if relative == "training_metrics.jsonl":
            return "training_metrics_jsonl.v1"
        if relative == "training_manifest.json":
            return "embedding_training_manifest.v1"
        if relative == "launch_manifest.json":
            return "campaign_remote_launch_manifest.v2"
        if relative == "training.log":
            return "campaign_training_log.v1"
        if relative == "exit_code":
            return "campaign_remote_exit_code.v1"
        if relative.startswith("final/"):
            return "huggingface_model_file.v1"
        return "campaign_remote_output.v1"


class DevelopmentScorerConfig(ContractModel):
    """Hash-pinned inputs for invoking the MemexAI development scorer."""

    schema_version: str = "campaign_development_scorer_config.v1"
    scorer_script_path: Path
    expected_scorer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model_path: Path
    corpus_path: Path
    expected_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_embedding_matrix: Path
    expected_matrix_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_embedding_chunk_ids: Path
    expected_chunk_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_prefix_mode: Literal["raw", "qwen_retrieval", "memexai_youtube"] = "memexai_youtube"
    embedding_device: str = Field(default="cuda", min_length=1, max_length=80)
    embedding_batch_size: int = Field(default=32, ge=1, le=4096)
    latency_repetitions: int = Field(default=3, ge=1, le=100)
    truncate_dim: int = Field(default=768, ge=1, le=8192)
    timeout_seconds: int = Field(default=3600, ge=1, le=86400)


class DevelopmentEvaluationConfig(ContractModel):
    schema_version: str = "campaign_development_evaluation_config.v1"
    development_path: Path
    expected_development_sha256: str
    protected_hashes: frozenset[str] = frozenset()
    protected_path_fragments: tuple[str, ...] = ()
    scored_rows_path: Path | None = None
    scorer: DevelopmentScorerConfig | None = None
    corpus_sha256: str
    representation_contract: dict[str, object]
    median_latency_ms: float | None = None
    model_footprint_bytes: int | None = None
    champion_evaluation_id: str | None = None
    gate_contract: DevelopmentGateContract = Field(default_factory=DevelopmentGateContract)

    @model_validator(mode="after")
    def require_one_score_source(self) -> DevelopmentEvaluationConfig:
        if (self.scored_rows_path is None) == (self.scorer is None):
            raise ValueError("exactly one of scored_rows_path or scorer is required")
        if self.scorer is not None:
            configured_mode = self.representation_contract.get("query_prefix_mode")
            if configured_mode != self.scorer.query_prefix_mode:
                raise ValueError("campaign_development_query_prefix_contract_mismatch")
            if self.corpus_sha256 != self.scorer.expected_corpus_sha256:
                raise ValueError("campaign_development_corpus_contract_mismatch")
            if self.median_latency_ms is not None or self.model_footprint_bytes is not None:
                raise ValueError("scorer-derived measurements cannot be supplied explicitly")
        return self


@dataclass(frozen=True)
class DevelopmentEvaluationExecution:
    sealed_path: Path
    manifest: SealedActionResult
    evaluation: RetrievalEvaluationArtifact
    comparison: DevelopmentComparison | None


class DevelopmentEvaluationExecutor:
    """Seal already-scored dev-only rows under the campaign's exact contracts."""

    executor_id = "campaign-development-evaluation-executor"
    executor_version = "1"

    def __init__(self, artifact_root: Path, sealer: ArtifactSealer):
        self.artifact_root = artifact_root.resolve()
        self.sealer = sealer

    @staticmethod
    def _require_file_hash(path: Path, expected: str, label: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"campaign_development_{label}_missing")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"campaign_development_{label}_hash_mismatch")
        return resolved

    def _score(
        self,
        config: DevelopmentEvaluationConfig,
        temporary: Path,
    ) -> tuple[Path, float, int]:
        scorer = config.scorer
        if scorer is None:
            raise ValueError("campaign_development_scorer_missing")
        script = self._require_file_hash(
            scorer.scorer_script_path, scorer.expected_scorer_sha256, "scorer"
        )
        corpus = self._require_file_hash(
            scorer.corpus_path, scorer.expected_corpus_sha256, "corpus"
        )
        matrix = self._require_file_hash(
            scorer.corpus_embedding_matrix, scorer.expected_matrix_sha256, "matrix"
        )
        chunk_ids = self._require_file_hash(
            scorer.corpus_embedding_chunk_ids,
            scorer.expected_chunk_ids_sha256,
            "chunk_ids",
        )
        model_path = scorer.embedding_model_path.resolve()
        if not model_path.exists():
            raise ValueError("campaign_development_model_missing")
        scoring_directory = temporary / "scoring"
        command = [
            sys.executable,
            str(script),
            "--queries-jsonl",
            str(config.development_path.resolve()),
            "--corpus-jsonl",
            str(corpus),
            "--output-dir",
            str(scoring_directory),
            "--embedding-model-path",
            str(model_path),
            "--embedding-device",
            scorer.embedding_device,
            "--embedding-batch-size",
            str(scorer.embedding_batch_size),
            "--latency-repetitions",
            str(scorer.latency_repetitions),
            "--corpus-embedding-matrix",
            str(matrix),
            "--corpus-embedding-chunk-ids",
            str(chunk_ids),
            "--truncate-dim",
            str(scorer.truncate_dim),
            "--splits",
            "dev",
            "--require-exclusive-split",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=scorer.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("campaign_development_scorer_timed_out") from exc
        if completed.returncode != 0:
            raise RuntimeError("campaign_development_scorer_failed")
        manifest_path = scoring_directory / "query_format_ablation_manifest.json"
        rows_path = scoring_directory / f"{scorer.query_prefix_mode}-retrieval_eval_queries.jsonl"
        if not manifest_path.is_file() or not rows_path.is_file():
            raise RuntimeError("campaign_development_scorer_outputs_missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run = manifest["runs"][scorer.query_prefix_mode]
            latency = float(run["median_query_latency_ms"])
            footprint = int(manifest["model_footprint_bytes"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("campaign_development_scorer_manifest_invalid") from exc
        if latency < 0 or footprint < 0:
            raise RuntimeError("campaign_development_scorer_manifest_invalid")
        return rows_path, latency, footprint

    def execute(
        self,
        attempt: ActionAttempt,
        config: DevelopmentEvaluationConfig,
        *,
        champion: RetrievalEvaluationArtifact | None = None,
    ) -> DevelopmentEvaluationExecution:
        started_at = utc_now()
        dataset = DevelopmentDatasetContract(
            expected_sha256=config.expected_development_sha256,
            protected_hashes=config.protected_hashes,
            protected_path_fragments=config.protected_path_fragments,
            minimum_queries=config.gate_contract.minimum_queries,
            minimum_videos=config.gate_contract.minimum_videos,
        ).validate_file(config.development_path)
        temporary = (
            self.artifact_root / ".tmp" / f"{attempt.action_id}.{attempt.attempt_id}.{uuid4().hex}"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        scored_rows_path = config.scored_rows_path
        median_latency_ms = config.median_latency_ms
        model_footprint_bytes = config.model_footprint_bytes
        if config.scorer is not None:
            scored_rows_path, median_latency_ms, model_footprint_bytes = self._score(
                config, temporary
            )
        if scored_rows_path is None:
            raise ValueError("campaign_development_scored_rows_missing")
        evaluation = load_retrieval_evaluation_artifact(
            scored_rows_path,
            candidate_digest=attempt.candidate_digest,
            corpus_sha256=config.corpus_sha256,
            development_sha256=dataset.sha256,
            representation_contract=config.representation_contract,
            median_latency_ms=median_latency_ms,
            model_footprint_bytes=model_footprint_bytes,
        )
        if {row.eval_id for row in evaluation.rows} != set(dataset.eval_ids):
            raise ValueError("campaign_development_evaluation_ids_do_not_match_dataset")
        if (config.champion_evaluation_id is None) != (champion is None):
            raise ValueError("campaign_champion_evaluation_contract_incomplete")
        comparison = (
            compare_development_evaluations(champion, evaluation, config.gate_contract)
            if champion is not None
            else None
        )

        (temporary / "evaluation.json").write_text(
            evaluation.model_dump_json(indent=2), encoding="utf-8"
        )
        (temporary / "validated_development_dataset.json").write_text(
            dataset.model_dump_json(indent=2), encoding="utf-8"
        )
        schemas = {
            "evaluation.json": "campaign_retrieval_evaluation.v1",
            "validated_development_dataset.json": "campaign_validated_dev_dataset.v1",
        }
        if config.scorer is not None:
            schemas.update(
                {
                    "scoring/query_format_ablation_manifest.json": (
                        "memexai_query_format_ablation_manifest.v1"
                    ),
                    f"scoring/{config.scorer.query_prefix_mode}-retrieval_eval_queries.jsonl": (
                        "campaign_scored_development_rows.v1"
                    ),
                }
            )
        if comparison is not None:
            (temporary / "comparison.json").write_text(
                comparison.model_dump_json(indent=2), encoding="utf-8"
            )
            schemas["comparison.json"] = "campaign_development_comparison.v1"
        outputs = self.sealer.describe_outputs(temporary, schemas)
        ended_at = utc_now()
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
            executor_id=self.executor_id,
            executor_version=self.executor_version,
            compute_profile_id="local-evaluation",
            remote_process_identity={"kind": "local_evaluation"},
            started_at=started_at,
            ended_at=ended_at,
            outcome="completed",
            exit_code=0,
            exit_reason="development evaluation evidence validated",
            resource_usage=(
                ResourceUsage(
                    unit="evaluation_queries",
                    amount=float(dataset.row_count),
                    source="validated_development_dataset",
                    confidence="measured",
                ),
            ),
            outputs=outputs,
        )
        sealed = (
            self.artifact_root
            / attempt.workspace_id
            / attempt.campaign_id
            / attempt.study_id
            / attempt.action_id
            / attempt.attempt_id
        )
        self.sealer.seal(temporary, sealed, manifest)
        return DevelopmentEvaluationExecution(sealed, manifest, evaluation, comparison)


def fake_digest(value: str) -> str:
    """Create readable deterministic fixture digests for fake campaign work."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "FakeExecutionRequest",
    "FakeExecutor",
    "DevelopmentEvaluationConfig",
    "DevelopmentScorerConfig",
    "DevelopmentEvaluationExecution",
    "DevelopmentEvaluationExecutor",
    "RemoteOutputSealer",
    "fake_digest",
]
