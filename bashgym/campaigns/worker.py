"""Foreground resident campaign worker with reconcile-before-claim semantics."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from bashgym.campaigns.artifacts import ArtifactSealer, ArtifactSealError
from bashgym.campaigns.autoresearch_dataset import (
    AUTORESEARCH_DATASET_FILE_SCHEMA,
    AUTORESEARCH_DATASET_RECEIPT_FILENAME,
    AUTORESEARCH_DATASET_RECEIPT_SCHEMA,
    MAX_AUTORESEARCH_DATASET_RECEIPT_BYTES,
    AutoResearchDatasetReceipt,
    build_dataset_ledger_specs,
)
from bashgym.campaigns.autoresearch_evidence import (
    AUTORESEARCH_EVALUATION_CONTEXT_FILENAME,
    AUTORESEARCH_EVALUATION_FILENAME,
    AUTORESEARCH_NORMALIZED_EVALUATION_DOMAIN,
    MAX_AUTORESEARCH_EVALUATION_BYTES,
    AutoResearchEvaluationContext,
    AutoResearchEvaluationEvidence,
    evaluation_context_bytes,
)
from bashgym.campaigns.campaign_recovery import (
    CampaignRecoveryRepository,
    RecoveryAction,
    RecoveryWorkClaim,
)
from bashgym.campaigns.contracts import (
    AUTORESEARCH_EVALUATION_SCHEMA,
    ActionAttempt,
    AttemptStatus,
    CampaignStatus,
    CampaignTrigger,
    CredentialKind,
    SealedActionResult,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.diagnostic_actions import (
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA,
    AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME,
    AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN,
    MAX_AUTORESEARCH_DIAGNOSTIC_BYTES,
    AutoResearchDiagnosticRecipe,
    AutoResearchDiagnosticRequest,
    diagnostic_request_bytes,
    public_diagnostic_projection,
    validated_diagnostic_evidence,
)
from bashgym.campaigns.evaluation import (
    DevelopmentComparison,
    RetrievalEvaluationArtifact,
)
from bashgym.campaigns.executor_registry import ExecutorAdapter, ExecutorRegistry
from bashgym.campaigns.executors import (
    DevelopmentEvaluationConfig,
    DevelopmentEvaluationExecutor,
    FakeExecutionRequest,
    FakeExecutor,
    RemoteOutputSealer,
)
from bashgym.campaigns.human_oversight import HumanOversightRepository
from bashgym.campaigns.lineage import (
    ApprovedSourceRepositoryProfile,
    CodeLineageSnapshotReceipt,
    GitHypothesisLineageManager,
    GitLineageError,
)
from bashgym.campaigns.persistence import (
    CampaignPersistenceError,
    LeaseBusyError,
    LeaseLostError,
    LeaseRecord,
    RecordNotFoundError,
    RevisionConflictError,
)
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
    CodeLineageLaunchSnapshot,
    RegisteredRemoteEvaluationDatasetSource,
    RegisteredRemoteModelSource,
    RemoteCapacityPolicy,
    RemoteLaunchRequest,
    RemoteResidentDatasetSource,
    RemoteResidentModelSource,
    RemoteRunState,
    RemoteTrainingAdapter,
    SealedStageArtifactInput,
    SealedStageArtifactSource,
    remote_executor_config,
)
from bashgym.campaigns.result_reuse import (
    REUSABLE_STAGES,
    REUSED_FROM_ACTION_KEY,
    REUSED_FROM_ATTEMPT_KEY,
)
from bashgym.campaigns.runtime import (
    CampaignRuntimeRepository,
    ReusableCompletion,
    _default_registry,
)
from bashgym.campaigns.transitions import InvalidCampaignTransitionError

if TYPE_CHECKING:
    from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator

logger = logging.getLogger(__name__)

UNREGISTERED_EXECUTOR_KIND_CODE = "campaign_executor_kind_not_registered"


class SimulatedWorkerCrashError(RuntimeError):
    """Test-only fault boundary after external side effect and before DB commit."""


class UnregisteredExecutorKindError(RuntimeError):
    """Raised when an attempt names an executor kind absent from this registry."""


def scheduler_lease_key(data_directory: Path) -> str:
    """Return the stable leader key for one canonical BashGym data directory."""

    canonical = str(data_directory.resolve()).casefold()
    directory_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"scheduler:{directory_digest}"


def _controller_selection_idempotency_key(workspace_id: str, campaign_id: str, version: int) -> str:
    identity = canonical_hash([workspace_id, campaign_id, version])[:32]
    return f"controller-select-{identity}"


class CampaignWorker:
    """Own the global scheduler lease and execute durable actions one at a time."""

    def __init__(
        self,
        repository: CampaignRuntimeRepository,
        artifact_root: Path,
        sealer: ArtifactSealer,
        *,
        data_directory: Path,
        worker_id: str | None = None,
        leader_ttl: timedelta = timedelta(seconds=15),
        action_ttl: timedelta = timedelta(seconds=15),
        remote_adapters: dict[str, RemoteTrainingAdapter] | None = None,
        remote_executor_profiles: (
            Mapping[tuple[str, str], ApprovedRemoteExecutorProfile] | None
        ) = None,
        source_repository_profiles: Mapping[str, ApprovedSourceRepositoryProfile] | None = None,
        lineage_manager: GitHypothesisLineageManager | None = None,
        autoresearch_loop: AutoResearchLoopCoordinator | None = None,
        executor_registry: ExecutorRegistry | None = None,
    ):
        self.repository = repository
        self.artifact_root = artifact_root.resolve()
        self.sealer = sealer
        self.worker_id = worker_id or f"worker-{uuid4().hex}"
        self.leader_key = scheduler_lease_key(data_directory)
        self.leader_ttl = leader_ttl
        self.action_ttl = action_ttl
        self._leader: LeaseRecord | None = None
        self._stop_requested = False
        self.executor_registry = executor_registry or _default_registry()
        self.executor = FakeExecutor(self.artifact_root, sealer)
        self.remote_output_sealer = RemoteOutputSealer(self.artifact_root, sealer)
        self.development_evaluation_executor = DevelopmentEvaluationExecutor(
            self.artifact_root, sealer
        )
        self.human_oversight = HumanOversightRepository(repository, sealer=sealer)
        self.recovery = CampaignRecoveryRepository(repository.db_path, sealer=sealer)
        self.recovery.initialize()
        self.remote_adapters = dict(remote_adapters or {})
        self.remote_executor_profiles = dict(remote_executor_profiles or {})
        self.source_repository_profiles = dict(source_repository_profiles or {})
        self.autoresearch_loop = autoresearch_loop
        self.lineage_manager = lineage_manager or GitHypothesisLineageManager(
            data_directory.resolve() / "campaigns" / "source-worktrees"
        )
        self.lineage_snapshot_root = data_directory.resolve() / "campaigns" / "source-snapshots"
        self.evaluation_context_root = (
            data_directory.resolve() / "campaigns" / "evaluation-contexts"
        )
        self._lineage_snapshots: dict[str, CodeLineageSnapshotReceipt] = {}

    @property
    def leader(self) -> LeaseRecord | None:
        return self._leader

    def request_stop(self) -> None:
        """Stop taking new claims after reconciliation of already-owned work."""

        self._stop_requested = True

    def _ensure_leader(self, now: datetime) -> LeaseRecord:
        if self._leader is None:
            self._leader = self.repository.acquire_lease(
                self.leader_key,
                self.worker_id,
                ttl=self.leader_ttl,
                now=now,
            )
        else:
            try:
                self._leader = self.repository.heartbeat_lease(
                    self._leader.lease_key,
                    self._leader.owner_id,
                    self._leader.generation,
                    ttl=self.leader_ttl,
                    now=now,
                )
            except LeaseLostError:
                self._leader = None
                self._leader = self.repository.acquire_lease(
                    self.leader_key,
                    self.worker_id,
                    ttl=self.leader_ttl,
                    now=now,
                )
        return self._leader

    def sealed_path(self, attempt: ActionAttempt) -> Path:
        return (
            self.artifact_root
            / attempt.workspace_id
            / attempt.campaign_id
            / attempt.study_id
            / attempt.action_id
            / attempt.attempt_id
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @classmethod
    def _write_evaluation_context(cls, destination: Path, payload: bytes) -> None:
        """Durably publish canonical context without exposing partial final bytes."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.read_bytes() != payload
            ):
                raise RuntimeError("campaign_remote_evaluation_context_mismatch")
            return
        temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        cls._fsync_directory(destination.parent)

    def _verify(self, attempt: ActionAttempt, sealed_path: Path):
        return self.sealer.verify(
            sealed_path,
            expected_workspace_id=attempt.workspace_id,
            expected_campaign_id=attempt.campaign_id,
            expected_study_id=attempt.study_id,
            expected_action_id=attempt.action_id,
            expected_attempt_id=attempt.attempt_id,
            expected_manifest_revision=attempt.manifest_revision,
            expected_candidate_digest=attempt.candidate_digest,
            expected_input_digest=attempt.input_digest,
            expected_claim_generation=attempt.claim_generation,
        )

    def _ingest_sealed_metrics(
        self, attempt: ActionAttempt, sealed_path: Path, *, now: datetime
    ) -> None:
        metrics_path = sealed_path / "training_metrics.jsonl"
        if not metrics_path.is_file() or metrics_path.is_symlink():
            return
        payload = metrics_path.read_text(encoding="utf-8")
        self.repository.append_remote_metrics(
            attempt,
            tuple(payload.splitlines()),
            source="training_metrics.jsonl",
            cursor_end=len(payload.encode("utf-8")),
            now=now,
        )

    def _persist_development_evaluation_evidence(
        self,
        attempt: ActionAttempt,
        sealed_path: Path,
        *,
        now: datetime,
    ) -> None:
        try:
            evaluation = RetrievalEvaluationArtifact.model_validate_json(
                (sealed_path / "evaluation.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise CampaignPersistenceError("campaign_development_evaluation_seal_invalid") from exc
        self.repository.store_retrieval_evaluation(
            attempt.workspace_id,
            attempt.campaign_id,
            evaluation,
            now=now,
        )
        comparison_path = sealed_path / "comparison.json"
        if not comparison_path.is_file() or comparison_path.is_symlink():
            return
        try:
            comparison = DevelopmentComparison.model_validate_json(
                comparison_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise CampaignPersistenceError("campaign_development_comparison_seal_invalid") from exc
        manifest = self.repository.get_manifest_revision(
            attempt.workspace_id,
            attempt.campaign_id,
            attempt.manifest_revision,
        ).manifest
        requires_human_review = bool(
            manifest.promotion_gates.get(
                "requires_human_review",
                manifest.promotion_gates.get("quality_claim_eligible", False),
            )
        )
        if not requires_human_review:
            self.repository.store_development_comparison(
                attempt.workspace_id,
                attempt.campaign_id,
                comparison,
                now=now,
            )
            return
        config = DevelopmentEvaluationConfig.model_validate(
            {key: value for key, value in attempt.executor.items() if key != "kind"}
        )
        if not config.champion_evaluation_id:
            raise CampaignPersistenceError("campaign_human_review_champion_missing")
        champion = self.repository.get_retrieval_evaluation(
            attempt.workspace_id,
            config.champion_evaluation_id,
        )
        self.human_oversight.enqueue_development_comparison(
            workspace_id=attempt.workspace_id,
            campaign_id=attempt.campaign_id,
            campaign_revision=attempt.manifest_revision,
            comparison=comparison,
            champion=champion,
            candidate=evaluation,
            now=now,
        )

    def _adapter_for(self, attempt: ActionAttempt) -> ExecutorAdapter:
        """Resolve the registered adapter for one attempt, or fail with its public code."""

        kind = str(attempt.executor.get("kind", "fake"))
        try:
            return self.executor_registry.get(kind)
        except KeyError as exc:
            raise UnregisteredExecutorKindError(UNREGISTERED_EXECUTOR_KIND_CODE) from exc

    def _reconcile_remote(self, attempt: ActionAttempt, *, now: datetime) -> str | None:
        """Adopt an unowned remote attempt, then advance its registered remote run."""

        if (
            attempt.lease_owner != self.worker_id
            or attempt.lease_expires_at is None
            or attempt.lease_expires_at <= now
        ):
            try:
                attempt = self.repository.adopt_remote_attempt(
                    attempt,
                    self._leader,
                    ttl=self.action_ttl,
                    now=now,
                )
            except LeaseBusyError:
                # Another live worker still holds this attempt's lease.
                # Leave it for the owner (or for adoption after expiry)
                # instead of failing the whole reconcile pass.
                return None
        return asyncio.run(self._remote_tick(attempt, now=now))

    def reconcile_once(self, *, now: datetime) -> str | None:
        """Register sealed results before marking expired uncertain work."""

        for attempt in self.repository.list_unfinished_attempts():
            try:
                adapter = self._adapter_for(attempt)
            except UnregisteredExecutorKindError:
                logger.warning(
                    "%s: skipping attempt %s with executor kind %s",
                    UNREGISTERED_EXECUTOR_KIND_CODE,
                    attempt.attempt_id,
                    attempt.executor.get("kind"),
                )
                continue
            short_circuit = adapter.reconcile(self, attempt, now=now)
            if short_circuit is not None:
                return short_circuit
            if not adapter.repair_allowed():
                continue
            sealed_path = self.sealed_path(attempt)
            if sealed_path.is_dir():
                manifest = self._verify(attempt, sealed_path)
                self._ingest_sealed_metrics(attempt, sealed_path, now=now)
                if attempt.executor.get("kind") == "development_evaluation":
                    self._persist_development_evaluation_evidence(
                        attempt,
                        sealed_path,
                        now=now,
                    )
                self.repository.complete_from_seal(
                    manifest,
                    sealed_path,
                    worker_id=self.worker_id,
                    reconcile=True,
                    now=now,
                )
                return "reconciled"
            if (
                attempt.status == AttemptStatus.RUNNING
                and attempt.lease_expires_at is not None
                and attempt.lease_expires_at <= now
            ):
                self.repository.mark_expired_unknown(attempt, now=now)
                return "unknown"
        return None

    def _repair_allowed(self, attempt: ActionAttempt) -> bool:
        """Report whether the attempt's registered executor supports local repair."""

        return self._adapter_for(attempt).repair_allowed()

    def _settle_unregistered_kind(self, claim: RecoveryWorkClaim, *, now: datetime) -> str:
        """Block a recovery claim whose attempt names an executor kind this worker lacks."""

        self.recovery.settle(
            claim,
            status="blocked",
            outcome_code=UNREGISTERED_EXECUTOR_KIND_CODE,
            now=now,
        )
        return "recovery_blocked"

    def _repair_recovery(self, claim: RecoveryWorkClaim, *, now: datetime) -> str:
        """Reconcile only one exact existing sealed local attempt, or block safely."""

        attempt = None
        if claim.attempt_id is not None:
            try:
                attempt = self.repository.get_attempt(claim.workspace_id, claim.attempt_id)
            except RecordNotFoundError:
                self.recovery.settle(
                    claim, status="blocked", outcome_code="needs_operator", now=now
                )
                return "recovery_blocked"
            if attempt.campaign_id != claim.campaign_id:
                self.recovery.settle(
                    claim, status="blocked", outcome_code="authority_changed", now=now
                )
                return "recovery_blocked"
            if attempt.status == AttemptStatus.COMPLETED:
                self.recovery.settle(
                    claim, status="completed", outcome_code="attempt_reconciled", now=now
                )
                return "recovery_repaired"
        else:
            try:
                candidates = tuple(
                    value
                    for value in self.repository.list_attempts(
                        claim.workspace_id, claim.campaign_id
                    )
                    if value.status in {AttemptStatus.RUNNING, AttemptStatus.UNKNOWN}
                    and self._repair_allowed(value)
                    and self.sealed_path(value).is_dir()
                )
            except UnregisteredExecutorKindError:
                return self._settle_unregistered_kind(claim, now=now)
            if len(candidates) != 1:
                self.recovery.settle(
                    claim, status="blocked", outcome_code="needs_operator", now=now
                )
                return "recovery_blocked"
            attempt = candidates[0]
            claim = self.recovery.set_repair_target(claim, attempt.attempt_id)

        sealed_path = self.sealed_path(attempt)
        try:
            repairable = self._repair_allowed(attempt)
        except UnregisteredExecutorKindError:
            return self._settle_unregistered_kind(claim, now=now)
        if not sealed_path.is_dir() or not repairable:
            self.recovery.settle(claim, status="blocked", outcome_code="needs_operator", now=now)
            return "recovery_blocked"
        try:
            manifest = self._verify(attempt, sealed_path)
            self._ingest_sealed_metrics(attempt, sealed_path, now=now)
            if attempt.executor.get("kind") == "development_evaluation":
                self._persist_development_evaluation_evidence(attempt, sealed_path, now=now)
            self.repository.complete_from_seal(
                manifest,
                sealed_path,
                worker_id=self.worker_id,
                reconcile=True,
                now=now,
            )
        except (CampaignPersistenceError, OSError, ValueError):
            self.recovery.settle(claim, status="blocked", outcome_code="needs_operator", now=now)
            return "recovery_blocked"
        self.recovery.settle(claim, status="completed", outcome_code="attempt_reconciled", now=now)
        return "recovery_repaired"

    def _consume_recovery_once(self, leader: LeaseRecord, *, now: datetime) -> str | None:
        """Execute one durable recovery request beneath the resident leader fence."""

        claim = self.recovery.claim_next(
            leader=leader,
            worker_id=self.worker_id,
            ttl=self.action_ttl,
            now=now,
        )
        if claim is None:
            return None
        if claim.status == "blocked":
            return "recovery_blocked"
        if claim.action == RecoveryAction.REPAIR:
            return self._repair_recovery(claim, now=now)
        if claim.action != RecoveryAction.RESUME:
            self.recovery.settle(claim, status="blocked", outcome_code="needs_operator", now=now)
            return "recovery_blocked"
        try:
            self.repository.transition_campaign(
                claim.workspace_id,
                claim.campaign_id,
                CampaignTrigger.RESUME,
                expected_version=claim.expected_aggregate_version,
                actor_id="campaign-recovery-worker",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=f"recovery-{claim.request_id}",
                idempotency_key=f"recovery-resume-{claim.request_id}",
                payload={"recovery_request_id": claim.request_id},
            )
        except (InvalidCampaignTransitionError, RevisionConflictError):
            self.recovery.settle(claim, status="blocked", outcome_code="authority_changed", now=now)
            return "recovery_blocked"
        self.recovery.settle(claim, status="completed", outcome_code="campaign_resumed", now=now)
        return "recovery_resumed"

    def controller_once(
        self,
        leader: LeaseRecord,
        *,
        now: datetime,
        excluded_campaign_keys: frozenset[tuple[str, str]] = frozenset(),
    ) -> str | None:
        """Select one proposal and schedule its next safe stage under the leader fence."""

        campaign = self.repository.next_controller_campaign(
            excluded_campaign_keys=excluded_campaign_keys
        )
        if campaign is None:
            return None
        if campaign.active_study_id is None:
            selected = self.repository.select_next_proposal_as_controller(
                campaign.workspace_id,
                campaign.campaign_id,
                expected_version=campaign.version,
                controller_id="campaign-controller",
                correlation_id=f"worker-{self.worker_id}",
                idempotency_key=_controller_selection_idempotency_key(
                    campaign.workspace_id, campaign.campaign_id, campaign.version
                ),
            )
            if selected is None:
                return None
            campaign = selected.campaign
        if campaign.active_study_id is None:
            return "proposal_selected"
        if self.repository.skip_not_applicable_stages_under_leader(
            campaign.workspace_id,
            campaign.campaign_id,
            campaign.active_study_id,
            leader,
            expected_campaign_version=campaign.version,
            now=now,
        ):
            return "stage_skipped"
        try:
            spec = self.repository.next_action_spec(
                campaign.workspace_id,
                campaign.campaign_id,
                campaign.active_study_id,
                executor_profiles=self.remote_executor_profiles,
            )
        except CampaignPersistenceError as exc:
            self.repository.record_controller_blocker_under_leader(
                campaign.workspace_id,
                campaign.campaign_id,
                campaign.active_study_id,
                leader,
                code=getattr(type(exc), "code", "campaign_controller_action_blocked"),
                now=now,
            )
            return "action_blocked"
        self.repository.schedule_action_under_leader(
            spec,
            leader,
            expected_campaign_version=campaign.version,
            now=now,
        )
        return "action_scheduled"

    def _verify_sealed_data_build(
        self,
        data_attempt: ActionAttempt,
        data_manifest: SealedActionResult,
        *,
        compute_profile_id: str,
    ) -> None:
        """Bind one data build's remote seal reference to its own sealed identity."""

        envelope = self.sealer.envelope_bytes(data_manifest)
        prefix = f"bashgym-remote-seal://{compute_profile_id}/{data_attempt.attempt_id}/sha256/"
        if (
            not data_attempt.sealed_result_uri
            or not data_attempt.sealed_result_uri.startswith(prefix)
            or hashlib.sha256(envelope).hexdigest()
            != data_attempt.sealed_result_uri.removeprefix(prefix)
        ):
            raise ValueError("remote dataset seal mismatch")
        self.sealer.verify_envelope_bytes(
            envelope,
            expected_workspace_id=data_attempt.workspace_id,
            expected_campaign_id=data_attempt.campaign_id,
            expected_study_id=data_attempt.study_id,
            expected_action_id=data_attempt.action_id,
            expected_attempt_id=data_attempt.attempt_id,
            expected_manifest_revision=data_attempt.manifest_revision,
            expected_candidate_digest=data_attempt.candidate_digest,
            expected_input_digest=data_attempt.input_digest,
            expected_claim_generation=data_attempt.claim_generation,
        )

    def _remote_request(self, attempt: ActionAttempt) -> RemoteLaunchRequest:
        executor = attempt.executor
        required = {
            "compute_profile_id",
            "script_path",
            "input_files",
            "script_args",
            "recipe_digest",
            "profile_id",
            "profile_revision",
            "profile_digest",
            "target_contract_key",
            "target_model_digest",
        }
        if not required.issubset(executor):
            raise RuntimeError("campaign_remote_executor_contract_incomplete")
        profile_key = (executor["compute_profile_id"], executor["target_contract_key"])
        try:
            profile = self.remote_executor_profiles[profile_key]
        except KeyError as exc:
            raise RuntimeError("campaign_remote_executor_profile_unavailable") from exc
        try:
            study = self.repository.get_study(
                attempt.workspace_id, attempt.campaign_id, attempt.study_id
            )
        except RecordNotFoundError as exc:
            raise RuntimeError("campaign_remote_study_unavailable") from exc
        try:
            code_lineage = self.repository.get_code_lineage(attempt.workspace_id, study.proposal_id)
        except RecordNotFoundError as exc:
            if "code_lineage_execution" in executor:
                raise RuntimeError("campaign_code_lineage_execution_record_unavailable") from exc
            code_lineage = None
        sealed_stage_inputs: tuple[SealedStageArtifactInput, ...] = ()
        source_training = None
        registered_base_model = None
        registered_evaluation_dataset = None
        remote_resident_model = None
        remote_resident_dataset = None
        evaluation_suite = None
        dataset_version = None
        evaluation_context_path = None
        diagnostic_recipe = None
        diagnostic_request_path = None
        if attempt.stage.value == "contract_evaluation":
            try:
                diagnostic_recipe = AutoResearchDiagnosticRecipe.model_validate(
                    executor["diagnostic_recipe"]
                )
                diagnostic_contract = profile.stage_profile(attempt.stage).diagnostic_contract
                if diagnostic_contract is None or executor.get(
                    "diagnostic_contract"
                ) != diagnostic_contract.model_dump(mode="json"):
                    raise ValueError("diagnostic contract mismatch")
                campaign = self.repository.get_campaign(attempt.workspace_id, attempt.campaign_id)
                manifest = self.repository.get_manifest_revision(
                    attempt.workspace_id,
                    attempt.campaign_id,
                    campaign.manifest_revision,
                ).manifest
            except (KeyError, RecordNotFoundError, ValueError) as exc:
                raise RuntimeError("campaign_remote_diagnostic_contract_invalid") from exc
        if (
            attempt.stage.value == "full_training"
            and executor.get("training_base_model") is not None
        ):
            try:
                registered_base_model = RegisteredRemoteModelSource.model_validate(
                    executor["training_base_model"]
                )
            except ValueError as exc:
                raise RuntimeError("campaign_remote_training_base_invalid") from exc
            if profile.registered_base_model != registered_base_model:
                raise RuntimeError("campaign_remote_training_base_invalid")
        if (
            attempt.stage.value == "full_training"
            and executor.get("remote_resident_model") is not None
        ):
            try:
                remote_resident_model = RemoteResidentModelSource.model_validate(
                    executor["remote_resident_model"]
                )
            except ValueError as exc:
                raise RuntimeError("campaign_remote_training_checkpoint_invalid") from exc
        if attempt.stage.value == "full_training":
            try:
                expected_parent_model = self.repository.remote_resident_training_parent_source(
                    attempt.workspace_id,
                    attempt.campaign_id,
                    attempt.study_id,
                )
            except (CampaignPersistenceError, RecordNotFoundError) as exc:
                raise RuntimeError("campaign_remote_training_checkpoint_invalid") from exc
            if expected_parent_model is not None:
                if (
                    registered_base_model is not None
                    or remote_resident_model != expected_parent_model
                ):
                    raise RuntimeError("campaign_remote_training_checkpoint_invalid")
                self._verify_remote_resident_model_source(
                    attempt.workspace_id, remote_resident_model
                )
            elif remote_resident_model is not None:
                raise RuntimeError("campaign_remote_training_checkpoint_invalid")
        if (
            attempt.stage.value == "full_training"
            and executor.get("remote_resident_dataset") is not None
        ):
            try:
                remote_resident_dataset = RemoteResidentDatasetSource.model_validate(
                    executor["remote_resident_dataset"]
                )
                actual_dataset_source = self.repository.remote_resident_data_build_source(
                    attempt.workspace_id,
                    attempt.campaign_id,
                    attempt.study_id,
                    remote_resident_dataset.stage_index + 1,
                )
                data_attempt = self.repository.completed_data_build_attempt(
                    attempt.workspace_id,
                    attempt.campaign_id,
                    attempt.study_id,
                    remote_resident_dataset.stage_index,
                )
                data_manifest = self.repository.get_attempt_result_manifest(
                    attempt.workspace_id,
                    data_attempt.attempt_id,
                )
                if (
                    actual_dataset_source != remote_resident_dataset
                    or remote_resident_dataset.stage_index + 1 != attempt.stage_index
                    or data_attempt.status.value != "completed"
                    or data_attempt.stage.value != "data_build"
                    or data_attempt.candidate_digest != attempt.candidate_digest
                ):
                    raise ValueError("remote dataset source mismatch")
                self._verify_sealed_data_build(
                    data_attempt,
                    data_manifest,
                    compute_profile_id=remote_resident_dataset.compute_profile_id,
                )
                reuse_chain = self.repository.reuse_source_chain(data_attempt)
                source_attempt = reuse_chain[-1][0] if reuse_chain else data_attempt
                if (
                    source_attempt.attempt_id != remote_resident_dataset.attempt_id
                    or source_attempt.action_id != remote_resident_dataset.action_id
                ):
                    raise ValueError("remote dataset source mismatch")
                for reused_attempt, reused_manifest in reuse_chain:
                    self._verify_sealed_data_build(
                        reused_attempt,
                        reused_manifest,
                        compute_profile_id=remote_resident_dataset.compute_profile_id,
                    )
                sealed_files = {
                    (output.path, output.sha256, output.size_bytes)
                    for output in data_manifest.outputs
                    if output.schema_name == AUTORESEARCH_DATASET_FILE_SCHEMA
                }
                resident_files = {
                    (
                        "dataset/" + item.remote_relative_path,
                        item.sha256,
                        item.size_bytes,
                    )
                    for item in remote_resident_dataset.files
                }
                if sealed_files != resident_files:
                    raise ValueError("remote dataset inventory mismatch")
            except (
                ArtifactSealError,
                CampaignPersistenceError,
                KeyError,
                RecordNotFoundError,
                ValueError,
            ) as exc:
                raise RuntimeError("campaign_remote_training_dataset_invalid") from exc
        if attempt.stage.value == "development_evaluation":
            try:
                binding = executor["evaluation_binding"]
                evaluation_suite = self.repository.get_evaluation_suite_spec(
                    attempt.workspace_id,
                    binding["ledger_project_id"],
                    binding["evaluation_suite_id"],
                )
                dataset_version = self.repository.get_dataset_version_spec(
                    attempt.workspace_id,
                    binding["ledger_project_id"],
                    binding["dataset_version_id"],
                )
                registered_evaluation_dataset = (
                    RegisteredRemoteEvaluationDatasetSource.model_validate(
                        executor["registered_evaluation_dataset"]
                    )
                )
                if (
                    profile.registered_evaluation_dataset != registered_evaluation_dataset
                    or binding.get("dataset_remote_path")
                    != registered_evaluation_dataset.remote_dataset_path
                ):
                    raise RuntimeError("campaign_remote_evaluation_dataset_invalid")
                if executor.get("registered_base_model") is not None:
                    registered_base_model = RegisteredRemoteModelSource.model_validate(
                        executor["registered_base_model"]
                    )
                    if (
                        executor.get("source_training") is not None
                        or executor.get("sealed_stage_artifact_inputs")
                        or executor.get("remote_resident_model") is not None
                    ):
                        raise RuntimeError("campaign_remote_evaluated_model_source_invalid")
                elif executor.get("remote_resident_model") is not None:
                    remote_resident_model = RemoteResidentModelSource.model_validate(
                        executor["remote_resident_model"]
                    )
                    if executor.get("source_training") is not None or executor.get(
                        "sealed_stage_artifact_inputs"
                    ):
                        raise RuntimeError("campaign_remote_evaluated_model_source_invalid")
                    try:
                        training_attempt, source_stage_index = (
                            self.repository.get_immediately_preceding_training_attempt(
                                attempt.workspace_id,
                                attempt.campaign_id,
                                attempt.study_id,
                                attempt.action_id,
                            )
                        )
                        actual_remote_source = self.repository.remote_resident_full_training_source(
                            attempt.workspace_id,
                            attempt.campaign_id,
                            attempt.study_id,
                            source_stage_index + 1,
                        )
                    except (CampaignPersistenceError, RecordNotFoundError) as exc:
                        raise RuntimeError("campaign_remote_training_checkpoint_invalid") from exc
                    if (
                        actual_remote_source != remote_resident_model
                        or training_attempt.status.value != "completed"
                        or training_attempt.candidate_digest != attempt.candidate_digest
                    ):
                        raise RuntimeError("campaign_remote_training_checkpoint_invalid")
                    try:
                        training_manifest = self.repository.get_attempt_result_manifest(
                            attempt.workspace_id, training_attempt.attempt_id
                        )
                        training_envelope = self.sealer.envelope_bytes(training_manifest)
                        training_prefix = (
                            f"bashgym-remote-seal://{remote_resident_model.compute_profile_id}/"
                            f"{training_attempt.attempt_id}/sha256/"
                        )
                        if (
                            not training_attempt.sealed_result_uri
                            or not training_attempt.sealed_result_uri.startswith(training_prefix)
                            or hashlib.sha256(training_envelope).hexdigest()
                            != training_attempt.sealed_result_uri.removeprefix(training_prefix)
                        ):
                            raise ValueError("training seal reference mismatch")
                        self.sealer.verify_envelope_bytes(
                            training_envelope,
                            expected_workspace_id=training_attempt.workspace_id,
                            expected_campaign_id=training_attempt.campaign_id,
                            expected_study_id=training_attempt.study_id,
                            expected_action_id=training_attempt.action_id,
                            expected_attempt_id=training_attempt.attempt_id,
                            expected_manifest_revision=training_attempt.manifest_revision,
                            expected_candidate_digest=training_attempt.candidate_digest,
                            expected_input_digest=training_attempt.input_digest,
                            expected_claim_generation=training_attempt.claim_generation,
                        )
                        sealed_models = {
                            (output.path, output.sha256, output.size_bytes)
                            for output in training_manifest.outputs
                            if output.schema_name == "huggingface_model_file.v1"
                        }
                        resident_models = {
                            (
                                "final/" + item.remote_relative_path.removeprefix("model/"),
                                item.sha256,
                                item.size_bytes,
                            )
                            for item in remote_resident_model.files
                        }
                        if sealed_models != resident_models:
                            raise ValueError("training model inventory mismatch")
                    except (ArtifactSealError, CampaignPersistenceError, ValueError) as exc:
                        raise RuntimeError("campaign_remote_training_checkpoint_invalid") from exc
                else:
                    source_training = SealedStageArtifactSource.model_validate(
                        executor["source_training"]
                    )
                    sealed_stage_inputs = tuple(
                        SealedStageArtifactInput.model_validate(item)
                        for item in executor["sealed_stage_artifact_inputs"]
                    )
                    try:
                        training_attempt, source_stage_index = (
                            self.repository.get_immediately_preceding_training_attempt(
                                attempt.workspace_id,
                                attempt.campaign_id,
                                attempt.study_id,
                                attempt.action_id,
                            )
                        )
                    except RecordNotFoundError as exc:
                        raise RuntimeError("campaign_remote_training_checkpoint_invalid") from exc
                    actual_source = SealedStageArtifactSource(
                        campaign_id=training_attempt.campaign_id,
                        study_id=training_attempt.study_id,
                        action_id=training_attempt.action_id,
                        attempt_id=training_attempt.attempt_id,
                        stage_index=source_stage_index,
                    )
                    if actual_source != source_training:
                        raise RuntimeError("campaign_remote_training_checkpoint_invalid")
                    for sealed_input in sealed_stage_inputs:
                        artifact = self.repository.get_artifact(
                            attempt.workspace_id,
                            attempt.campaign_id,
                            sealed_input.campaign_artifact_id,
                        )
                        relative_path = sealed_input.remote_relative_path.removeprefix("model/")
                        if (
                            artifact.sha256 != sealed_input.sha256
                            or artifact.workspace_id != attempt.workspace_id
                            or artifact.campaign_id != source_training.campaign_id
                            or artifact.producer_action_id != source_training.action_id
                            or artifact.size_bytes != sealed_input.size_bytes
                            or artifact.schema_name != sealed_input.schema_name
                            or Path(artifact.uri).resolve() != sealed_input.local_sealed_path
                            or artifact.metadata.get("relative_path") != relative_path
                            or artifact.metadata.get("attempt_id") != source_training.attempt_id
                            or not artifact.sealed
                            or not artifact.valid
                        ):
                            raise RuntimeError("campaign_remote_sealed_stage_artifact_mismatch")
                    if (
                        training_attempt.stage.value != "full_training"
                        or training_attempt.status.value != "completed"
                        or training_attempt.candidate_digest != attempt.candidate_digest
                        or training_attempt.sealed_result_uri is None
                    ):
                        raise RuntimeError("campaign_remote_training_checkpoint_invalid")
                    training_manifest = self.sealer.verify(
                        Path(training_attempt.sealed_result_uri),
                        expected_workspace_id=training_attempt.workspace_id,
                        expected_campaign_id=training_attempt.campaign_id,
                        expected_study_id=training_attempt.study_id,
                        expected_action_id=training_attempt.action_id,
                        expected_attempt_id=training_attempt.attempt_id,
                        expected_manifest_revision=training_attempt.manifest_revision,
                        expected_candidate_digest=training_attempt.candidate_digest,
                        expected_input_digest=training_attempt.input_digest,
                        expected_claim_generation=training_attempt.claim_generation,
                    )
                    sealed_output_paths = {
                        output.path
                        for output in training_manifest.outputs
                        if output.schema_name == "huggingface_model_file.v1"
                    }
                    requested_paths = {
                        f"final/{item.remote_relative_path.removeprefix('model/')}"
                        for item in sealed_stage_inputs
                    }
                    if sealed_output_paths != requested_paths:
                        raise RuntimeError("campaign_remote_training_checkpoint_invalid")
            except RuntimeError:
                raise
            except (KeyError, OSError, ValueError) as exc:
                raise RuntimeError("campaign_remote_sealed_stage_artifact_invalid") from exc
        try:
            expected = remote_executor_config(
                profile,
                attempt.stage,
                recipe_digest=executor["recipe_digest"],
                recipe_script_args=tuple(executor.get("recipe_script_args", ())),
                code_lineage=code_lineage,
                sealed_stage_artifact_inputs=sealed_stage_inputs,
                evaluation_suite=evaluation_suite,
                dataset_version=dataset_version,
                source_training=(
                    source_training if attempt.stage.value == "development_evaluation" else None
                ),
                remote_resident_model=(
                    remote_resident_model
                    if attempt.stage.value in {"full_training", "development_evaluation"}
                    else None
                ),
                remote_resident_dataset=(
                    remote_resident_dataset if attempt.stage.value == "full_training" else None
                ),
                bind_registered_training_base=(
                    attempt.stage.value == "full_training" and registered_base_model is not None
                ),
                evaluate_registered_base_model=registered_base_model is not None,
                diagnostic_recipe=diagnostic_recipe,
                approved_data_scopes=(
                    frozenset(manifest.approved_data_scopes)
                    if diagnostic_recipe is not None
                    else frozenset()
                ),
            )
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError("campaign_remote_executor_material_invalid") from exc
        persisted_context_sha = executor.get("evaluation_context_sha256")
        expected_executor = {"kind": "ssh_remote", **expected}
        if persisted_context_sha is not None:
            expected_executor["evaluation_context_sha256"] = persisted_context_sha
        persisted_diagnostic_sha = executor.get("diagnostic_request_sha256")
        if diagnostic_recipe is not None:
            expected_executor["diagnostic_proposal_id"] = executor.get("diagnostic_proposal_id")
            expected_executor["diagnostic_request_sha256"] = persisted_diagnostic_sha
        if executor != expected_executor:
            raise RuntimeError("campaign_remote_executor_profile_mismatch")
        script_args = tuple(executor["script_args"])
        training_model_path = None
        if attempt.stage.value == "full_training":
            training_model_path = (
                registered_base_model.remote_model_path
                if registered_base_model is not None
                else (
                    remote_resident_model.remote_model_path
                    if remote_resident_model is not None
                    else None
                )
            )
        if training_model_path is not None:
            if any(
                argument == "--model-dir" or argument.startswith("--model-dir=")
                for argument in script_args
            ):
                raise RuntimeError("campaign_remote_training_base_argument_conflict")
            script_args = (
                *script_args,
                "--model-dir",
                training_model_path,
            )
        if remote_resident_dataset is not None:
            if any(
                argument == "--dataset-dir" or argument.startswith("--dataset-dir=")
                for argument in script_args
            ):
                raise RuntimeError("campaign_remote_training_dataset_argument_conflict")
            script_args = (
                *script_args,
                "--dataset-dir",
                remote_resident_dataset.remote_dataset_path,
            )
        input_files = tuple(Path(value) for value in executor["input_files"])
        if diagnostic_recipe is not None:
            diagnostic_contract = profile.stage_profile(attempt.stage).diagnostic_contract
            assert diagnostic_contract is not None
            request = AutoResearchDiagnosticRequest(
                workspace_id=attempt.workspace_id,
                campaign_id=attempt.campaign_id,
                proposal_id=executor["diagnostic_proposal_id"],
                study_id=attempt.study_id,
                action_id=attempt.action_id,
                attempt_id=attempt.attempt_id,
                recipe=diagnostic_recipe,
                recipe_digest=executor["recipe_digest"],
                runner_id=diagnostic_contract.runner_id,
                runner_version=diagnostic_contract.runner_version,
            )
            request_bytes = diagnostic_request_bytes(request)
            if persisted_diagnostic_sha != hashlib.sha256(request_bytes).hexdigest():
                raise RuntimeError("campaign_remote_diagnostic_request_digest_mismatch")
            request_directory = self.evaluation_context_root / attempt.attempt_id
            diagnostic_request_path = request_directory / AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME
            self._write_evaluation_context(diagnostic_request_path, request_bytes)
            input_files = (diagnostic_request_path, *input_files)
            script_args = (
                *script_args,
                "--request",
                AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME,
                "--output",
                AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
            )
        if evaluation_suite is not None and dataset_version is not None:
            context = AutoResearchEvaluationContext(
                workspace_id=attempt.workspace_id,
                campaign_id=attempt.campaign_id,
                study_id=attempt.study_id,
                action_id=attempt.action_id,
                attempt_id=attempt.attempt_id,
                candidate_digest=attempt.candidate_digest,
                evaluation_suite_id=evaluation_suite.evaluation_suite_id,
                evaluation_code_digest=evaluation_suite.code_digest,
                dataset_version_id=dataset_version.dataset_version_id,
                dataset_content_digest=dataset_version.content_digest,
                evaluated_model_manifest_digest=executor["evaluated_model_digest"],
            )
            context_bytes = evaluation_context_bytes(context)
            context_sha256 = hashlib.sha256(context_bytes).hexdigest()
            if persisted_context_sha != context_sha256:
                raise RuntimeError("campaign_remote_evaluation_context_digest_mismatch")
            context_directory = self.evaluation_context_root / attempt.attempt_id
            evaluation_context_path = context_directory / AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
            self._write_evaluation_context(evaluation_context_path, context_bytes)
            binding = executor["evaluation_binding"]
            input_files = (evaluation_context_path, *input_files)
            script_args = (
                *script_args,
                "--context",
                AUTORESEARCH_EVALUATION_CONTEXT_FILENAME,
                "--model-dir",
                (
                    registered_base_model.remote_model_path
                    if registered_base_model is not None
                    else (
                        remote_resident_model.remote_model_path
                        if remote_resident_model is not None
                        else "model"
                    )
                ),
                "--dataset",
                binding["dataset_remote_path"],
                "--output",
                AUTORESEARCH_EVALUATION_FILENAME,
            )
        source_snapshot = None
        if code_lineage is not None:
            binding = profile.stage_profile(attempt.stage).code_lineage_binding
            if binding is None:
                raise RuntimeError("campaign_code_lineage_execution_binding_unavailable")
            try:
                source_profile = self.source_repository_profiles[
                    binding.source_repository_profile_id
                ]
            except KeyError as exc:
                raise RuntimeError("campaign_code_lineage_source_profile_unavailable") from exc
            try:
                receipt = self._lineage_snapshots.get(code_lineage.record_digest)
                if receipt is None:
                    receipt = self.lineage_manager.materialize_snapshot(
                        source_profile,
                        code_lineage,
                        self.lineage_snapshot_root,
                        entrypoint_path=binding.entrypoint_path,
                        max_archive_bytes=binding.max_archive_bytes,
                    )
                    self._lineage_snapshots[code_lineage.record_digest] = receipt
                else:
                    self.lineage_manager.verify_snapshot_receipt(
                        receipt,
                        code_lineage,
                        max_archive_bytes=binding.max_archive_bytes,
                    )
            except GitLineageError as exc:
                raise RuntimeError("campaign_code_lineage_snapshot_invalid") from exc
            source_snapshot = CodeLineageLaunchSnapshot(
                binding_id=binding.binding_id,
                binding_revision=binding.binding_revision,
                binding_digest=binding.binding_digest,
                source_repository_profile_id=receipt.source_repository_profile_id,
                lineage_id=receipt.lineage_id,
                record_digest=receipt.record_digest,
                commit_sha=receipt.commit_sha,
                patch_sha256=receipt.patch_sha256,
                entrypoint_path=receipt.entrypoint_path,
                working_directory=binding.working_directory,
                archive_path=receipt.archive_path,
                archive_sha256=receipt.archive_sha256,
                archive_size_bytes=receipt.archive_size_bytes,
            )
        return RemoteLaunchRequest(
            compute_profile_id=executor["compute_profile_id"],
            run_id=attempt.attempt_id,
            script_path=Path(executor["script_path"]),
            input_files=input_files,
            script_args=script_args,
            python_executable=executor["python_executable"],
            recipe_digest=executor["recipe_digest"],
            output_paths=tuple(
                executor.get(
                    "output_paths",
                    ("final", "training_manifest.json", "training_metrics.jsonl"),
                )
            ),
            source_snapshot=source_snapshot,
            sealed_stage_artifact_inputs=sealed_stage_inputs,
            source_training=(
                source_training if attempt.stage.value == "development_evaluation" else None
            ),
            registered_base_model=registered_base_model,
            registered_evaluation_dataset=registered_evaluation_dataset,
            remote_resident_model=remote_resident_model,
            remote_resident_dataset=remote_resident_dataset,
            evaluation_context_sha256=persisted_context_sha,
        )

    def _verify_remote_envelope(
        self, attempt: ActionAttempt, envelope: bytes
    ) -> SealedActionResult:
        return self.sealer.verify_envelope_bytes(
            envelope,
            expected_workspace_id=attempt.workspace_id,
            expected_campaign_id=attempt.campaign_id,
            expected_study_id=attempt.study_id,
            expected_action_id=attempt.action_id,
            expected_attempt_id=attempt.attempt_id,
            expected_manifest_revision=attempt.manifest_revision,
            expected_candidate_digest=attempt.candidate_digest,
            expected_input_digest=attempt.input_digest,
            expected_claim_generation=attempt.claim_generation,
        )

    def _verify_remote_resident_model_source(
        self, workspace_id: str, source: RemoteResidentModelSource
    ) -> SealedActionResult:
        try:
            training_attempt = self.repository.get_attempt(
                workspace_id,
                source.attempt_id,
            )
            training_manifest = self.repository.get_attempt_result_manifest(
                training_attempt.workspace_id,
                training_attempt.attempt_id,
            )
            training_envelope = self.sealer.envelope_bytes(training_manifest)
            training_prefix = (
                f"bashgym-remote-seal://{source.compute_profile_id}/"
                f"{training_attempt.attempt_id}/sha256/"
            )
            remote_run = self.repository.get_remote_run(
                training_attempt.workspace_id,
                training_attempt.attempt_id,
            )
            if (
                training_attempt.campaign_id != source.campaign_id
                or training_attempt.study_id != source.study_id
                or training_attempt.action_id != source.action_id
                or training_attempt.stage.value != "full_training"
                or training_attempt.status.value != "completed"
                or not training_attempt.sealed_result_uri
                or not training_attempt.sealed_result_uri.startswith(training_prefix)
                or hashlib.sha256(training_envelope).hexdigest()
                != training_attempt.sealed_result_uri.removeprefix(training_prefix)
                or remote_run is None
                or remote_run.identity.run_id != training_attempt.attempt_id
                or remote_run.identity.compute_profile_id != source.compute_profile_id
                or source.remote_model_path != f"{remote_run.identity.remote_run_directory}/final"
            ):
                raise ValueError("training source identity mismatch")
            verified = self.sealer.verify_envelope_bytes(
                training_envelope,
                expected_workspace_id=training_attempt.workspace_id,
                expected_campaign_id=training_attempt.campaign_id,
                expected_study_id=training_attempt.study_id,
                expected_action_id=training_attempt.action_id,
                expected_attempt_id=training_attempt.attempt_id,
                expected_manifest_revision=training_attempt.manifest_revision,
                expected_candidate_digest=training_attempt.candidate_digest,
                expected_input_digest=training_attempt.input_digest,
                expected_claim_generation=training_attempt.claim_generation,
            )
            sealed_models = {
                (output.path, output.sha256, output.size_bytes)
                for output in verified.outputs
                if output.schema_name == "huggingface_model_file.v1"
            }
            resident_models = {
                (
                    "final/" + item.remote_relative_path.removeprefix("model/"),
                    item.sha256,
                    item.size_bytes,
                )
                for item in source.files
            }
            if (
                verified.outcome != "completed"
                or verified.compute_profile_id != source.compute_profile_id
                or verified.remote_process_identity != remote_run.identity.model_dump(mode="json")
                or sealed_models != resident_models
            ):
                raise ValueError("training source manifest mismatch")
            return verified
        except (
            ArtifactSealError,
            CampaignPersistenceError,
            RecordNotFoundError,
            ValueError,
        ) as exc:
            raise RuntimeError("campaign_remote_training_checkpoint_invalid") from exc

    async def _verified_terminal_envelope(
        self,
        attempt: ActionAttempt,
        adapter: RemoteTrainingAdapter,
        identity,
        expected_manifest: SealedActionResult,
    ) -> tuple[bytes, SealedActionResult]:
        envelope = await adapter.read_action_seal(identity)
        if envelope is None:
            verified = expected_manifest
        else:
            verified = self.sealer.verify_envelope_bytes(
                envelope,
                expected_workspace_id=attempt.workspace_id,
                expected_campaign_id=attempt.campaign_id,
                expected_study_id=attempt.study_id,
                expected_action_id=attempt.action_id,
                expected_attempt_id=attempt.attempt_id,
                expected_manifest_revision=attempt.manifest_revision,
                expected_candidate_digest=attempt.candidate_digest,
                expected_input_digest=attempt.input_digest,
            )
            if verified.claim_generation > attempt.claim_generation:
                raise ArtifactSealError(f"{ArtifactSealError.code}: claim generation mismatch")
        if (
            verified.outputs != expected_manifest.outputs
            or verified.outcome != expected_manifest.outcome
            or verified.compute_profile_id != identity.compute_profile_id
            or verified.remote_process_identity != identity.model_dump(mode="json")
        ):
            raise RuntimeError("campaign_remote_action_seal_mismatch")
        if envelope is None or verified.claim_generation < attempt.claim_generation:
            verified = verified.model_copy(update={"claim_generation": attempt.claim_generation})
            envelope = self.sealer.envelope_bytes(verified)
            await adapter.persist_action_seal(identity, envelope)
            persisted_envelope = await adapter.read_action_seal(identity)
            if persisted_envelope != envelope:
                raise RuntimeError("campaign_remote_action_seal_persistence_failed")
        return envelope, self._verify_remote_envelope(attempt, envelope)

    async def _remote_tick(self, attempt: ActionAttempt, *, now: datetime) -> str:
        request = self._remote_request(attempt)
        try:
            adapter = self.remote_adapters[request.compute_profile_id]
        except KeyError as exc:
            raise RuntimeError("campaign_remote_compute_profile_unavailable") from exc
        record = self.repository.get_remote_run(attempt.workspace_id, attempt.attempt_id)
        campaign = self.repository.get_campaign(attempt.workspace_id, attempt.campaign_id)
        if record is None:
            identity = await adapter.discover(request)
            if identity is None:
                if campaign.status == CampaignStatus.CANCELLING:
                    manifest = self.remote_output_sealer.unlaunched_cancelled_manifest(
                        attempt, compute_profile_id=request.compute_profile_id
                    )
                    envelope = self.sealer.envelope_bytes(manifest)
                    verified = self.sealer.verify_envelope_bytes(
                        envelope,
                        expected_workspace_id=attempt.workspace_id,
                        expected_campaign_id=attempt.campaign_id,
                        expected_study_id=attempt.study_id,
                        expected_action_id=attempt.action_id,
                        expected_attempt_id=attempt.attempt_id,
                        expected_manifest_revision=attempt.manifest_revision,
                        expected_candidate_digest=attempt.candidate_digest,
                        expected_input_digest=attempt.input_digest,
                        expected_claim_generation=attempt.claim_generation,
                    )
                    sealed_reference = (
                        f"bashgym-controller-state://{attempt.attempt_id}/sha256/"
                        f"{hashlib.sha256(envelope).hexdigest()}"
                    )
                    self.repository.settle_terminal_from_seal(
                        verified,
                        sealed_reference,
                        worker_id=self.worker_id,
                        now=now,
                    )
                    return "remote_cancelled"
                capacity_config = attempt.executor.get("capacity_policy", {})
                capacity = await adapter.capacity_preflight(
                    RemoteCapacityPolicy.model_validate(capacity_config)
                )
                if not capacity.admitted:
                    self.repository.defer_unlaunched_remote_attempt(
                        attempt,
                        worker_id=self.worker_id,
                        reasons=capacity.blocking_reasons,
                        now=now,
                    )
                    return "remote_capacity_blocked"
                identity = await adapter.launch(request)
            record = self.repository.register_remote_identity(attempt, identity, now=now)

        force_stop_request = self.repository.pending_force_stop_request(
            attempt.workspace_id, attempt.action_id, record.identity
        )
        if force_stop_request is not None:
            executed = await adapter.force_stop(record.identity)
            self.repository.settle_force_stop_request(
                attempt.workspace_id, force_stop_request, executed=executed
            )
            if executed:
                return "remote_force_stopping"

        observation = await adapter.observe(record.identity)
        metric_cursor = record.metric_cursor
        # Raw logs remain canonical run evidence. The worker streams only
        # the typed metrics needed for compact campaign projections.
        log_cursor = record.log_cursor
        metric_lines: list[str] = []
        try:
            chunk = await adapter.read_stream(
                record.identity,
                "training_metrics.jsonl",
                metric_cursor,
            )
        except RuntimeError:
            pass
        else:
            metric_cursor = chunk.next_cursor
            metric_lines.extend(chunk.complete_lines)
        self.repository.append_remote_metrics(
            attempt,
            tuple(metric_lines),
            source="training_metrics.jsonl",
            cursor_end=metric_cursor.byte_offset,
            now=now,
        )
        collection_ttl = (
            timedelta(hours=1) if observation.state == RemoteRunState.COMPLETED else self.action_ttl
        )
        record = self.repository.update_remote_run(
            record,
            observation,
            metric_cursor=metric_cursor,
            log_cursor=log_cursor,
            worker_id=self.worker_id,
            lease_ttl=collection_ttl,
            now=now,
        )
        if campaign.status == CampaignStatus.CANCELLING and observation.state in {
            RemoteRunState.RUNNING,
            RemoteRunState.PAUSED,
        }:
            await adapter.terminate(record.identity)
            return "remote_cancelling"
        if observation.state == RemoteRunState.RUNNING:
            return "remote_running"
        if observation.state == RemoteRunState.PAUSED:
            return "remote_paused"
        if observation.state == RemoteRunState.UNKNOWN:
            return "remote_unknown"
        if observation.state == RemoteRunState.FAILED:
            inventory = await adapter.inventory_terminal_evidence(
                record.identity, observation=observation
            )
            outcome = "cancelled" if campaign.status == CampaignStatus.CANCELLING else "failed"
            expected_manifest = self.remote_output_sealer.terminal_manifest(
                attempt,
                record.identity,
                observation,
                inventory,
                outcome=outcome,
            )
            envelope, verified = await self._verified_terminal_envelope(
                attempt, adapter, record.identity, expected_manifest
            )
            sealed_reference = (
                f"bashgym-remote-seal://{record.identity.compute_profile_id}/"
                f"{record.identity.run_id}/sha256/{hashlib.sha256(envelope).hexdigest()}"
            )
            self.repository.settle_terminal_from_seal(
                verified,
                sealed_reference,
                worker_id=self.worker_id,
                now=now,
            )
            return "remote_cancelled" if outcome == "cancelled" else "remote_failed"

        inventory = await adapter.inventory_outputs(
            record.identity, request, observation=observation
        )
        expected_manifest = self.remote_output_sealer.completed_manifest(
            attempt, record.identity, observation, inventory
        )
        envelope, verified = await self._verified_terminal_envelope(
            attempt, adapter, record.identity, expected_manifest
        )
        sealed_reference = (
            f"bashgym-remote-seal://{record.identity.compute_profile_id}/{record.identity.run_id}"
            f"/sha256/{hashlib.sha256(envelope).hexdigest()}"
        )
        artifact_metadata: dict[str, dict[str, object]] = {}
        generated_dataset = None
        generated_dataset_version = None
        if attempt.stage.value == "data_build":
            receipt_outputs = tuple(
                output
                for output in verified.outputs
                if output.schema_name == AUTORESEARCH_DATASET_RECEIPT_SCHEMA
                and output.path == AUTORESEARCH_DATASET_RECEIPT_FILENAME
            )
            dataset_outputs = tuple(
                output
                for output in verified.outputs
                if output.schema_name == AUTORESEARCH_DATASET_FILE_SCHEMA
                and output.path.startswith("dataset/")
            )
            if len(receipt_outputs) != 1 or not dataset_outputs:
                raise RuntimeError("campaign_remote_dataset_output_invalid")
            receipt_output = receipt_outputs[0]
            receipt_payload = await adapter.read_output_bytes(
                record.identity,
                receipt_output.path,
                expected_sha256=receipt_output.sha256,
                expected_size_bytes=receipt_output.size_bytes,
                max_bytes=MAX_AUTORESEARCH_DATASET_RECEIPT_BYTES,
            )
            try:
                receipt = AutoResearchDatasetReceipt.model_validate_json(receipt_payload)
            except ValueError as exc:
                raise RuntimeError("campaign_remote_dataset_output_invalid") from exc
            expected_files = {(item.path, item.sha256, item.size_bytes) for item in receipt.files}
            actual_files = {
                (output.path, output.sha256, output.size_bytes) for output in dataset_outputs
            }
            if expected_files != actual_files:
                raise RuntimeError("campaign_remote_dataset_output_invalid")
            manifest_revision = self.repository.get_manifest_revision(
                attempt.workspace_id,
                attempt.campaign_id,
                attempt.manifest_revision,
            )
            project_id = manifest_revision.manifest.evaluation_plan.get("ledger_project_id")
            if not isinstance(project_id, str) or not project_id:
                raise RuntimeError("campaign_remote_dataset_project_invalid")
            generated_dataset, generated_dataset_version = build_dataset_ledger_specs(
                attempt,
                receipt,
                project_id=project_id,
                task_type=campaign.target_model.task,
                created_at=now,
            )
            normalized_receipt = receipt.model_dump(mode="json")
            artifact_metadata[receipt_output.path] = {
                "normalized_dataset_receipt": normalized_receipt,
                "ledger_project_id": project_id,
                "dataset_id": generated_dataset.dataset_id,
                "dataset_version_id": generated_dataset_version.dataset_version_id,
                "content_digest": generated_dataset_version.content_digest,
            }
            receipt_files = {item.path: item for item in receipt.files}
            for output in dataset_outputs:
                item = receipt_files[output.path]
                artifact_metadata[output.path] = {
                    "relative_path": output.path.removeprefix("dataset/"),
                    "ledger_project_id": project_id,
                    "dataset_id": generated_dataset.dataset_id,
                    "dataset_version_id": generated_dataset_version.dataset_version_id,
                    "content_digest": generated_dataset_version.content_digest,
                    "split": item.split,
                    "row_count": item.row_count,
                }
        if attempt.stage.value == "development_evaluation":
            evaluation_outputs = tuple(
                output
                for output in verified.outputs
                if output.schema_name == AUTORESEARCH_EVALUATION_SCHEMA
                and output.path == AUTORESEARCH_EVALUATION_FILENAME
            )
            if len(evaluation_outputs) != 1:
                raise RuntimeError("campaign_remote_evaluation_output_invalid")
            output = evaluation_outputs[0]
            payload = await adapter.read_output_bytes(
                record.identity,
                output.path,
                expected_sha256=output.sha256,
                expected_size_bytes=output.size_bytes,
                max_bytes=MAX_AUTORESEARCH_EVALUATION_BYTES,
            )
            try:
                evidence = AutoResearchEvaluationEvidence.model_validate_json(payload)
            except ValueError as exc:
                raise RuntimeError("campaign_remote_evaluation_output_invalid") from exc
            expected_identity = (
                attempt.campaign_id,
                attempt.study_id,
                attempt.action_id,
                attempt.attempt_id,
                attempt.candidate_digest,
                attempt.executor.get("evaluated_model_digest"),
            )
            actual_identity = (
                evidence.campaign_id,
                evidence.study_id,
                evidence.action_id,
                evidence.attempt_id,
                evidence.candidate_digest,
                evidence.evaluated_model_manifest_digest,
            )
            if actual_identity != expected_identity:
                raise RuntimeError("campaign_remote_evaluation_output_invalid")
            normalized = evidence.model_dump(mode="json")
            artifact_metadata[output.path] = {
                "normalized_evaluation": normalized,
                "projection_key_version": self.sealer.key_version,
                "projection_signature": self.sealer.sign_canonical_payload(
                    normalized,
                    domain=AUTORESEARCH_NORMALIZED_EVALUATION_DOMAIN,
                ),
            }
        if attempt.stage.value == "contract_evaluation":
            diagnostic_outputs = tuple(
                output
                for output in verified.outputs
                if output.schema_name == AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA
                and output.path == AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME
            )
            if len(diagnostic_outputs) != 1:
                raise RuntimeError("campaign_remote_diagnostic_output_invalid")
            output = diagnostic_outputs[0]
            payload = await adapter.read_output_bytes(
                record.identity,
                output.path,
                expected_sha256=output.sha256,
                expected_size_bytes=output.size_bytes,
                max_bytes=MAX_AUTORESEARCH_DIAGNOSTIC_BYTES,
            )
            recipe = AutoResearchDiagnosticRecipe.model_validate(
                attempt.executor.get("diagnostic_recipe")
            )
            contract = attempt.executor.get("diagnostic_contract")
            if not isinstance(contract, dict):
                raise RuntimeError("campaign_remote_diagnostic_output_invalid")
            try:
                evidence = validated_diagnostic_evidence(
                    payload,
                    recipe=recipe,
                    expected_identity={
                        "workspace_id": attempt.workspace_id,
                        "campaign_id": attempt.campaign_id,
                        "proposal_id": str(attempt.executor.get("diagnostic_proposal_id", "")),
                        "study_id": attempt.study_id,
                        "action_id": attempt.action_id,
                        "attempt_id": attempt.attempt_id,
                    },
                    expected_runner_id=str(contract.get("runner_id", "")),
                    expected_runner_version=str(contract.get("runner_version", "")),
                )
            except ValueError as exc:
                raise RuntimeError("campaign_remote_diagnostic_output_invalid") from exc
            normalized = {
                "evidence": evidence.model_dump(mode="json"),
                "projection": public_diagnostic_projection(recipe, evidence),
            }
            artifact_metadata[output.path] = {
                "normalized_diagnostic": normalized,
                "projection_key_version": self.sealer.key_version,
                "projection_signature": self.sealer.sign_canonical_payload(
                    normalized,
                    domain=AUTORESEARCH_NORMALIZED_DIAGNOSTIC_DOMAIN,
                ),
            }
        self.repository.complete_from_seal(
            verified,
            sealed_reference,
            worker_id=self.worker_id,
            artifact_metadata_by_path=artifact_metadata,
            dataset_spec=generated_dataset,
            dataset_version_spec=generated_dataset_version,
            now=now,
        )
        return "completed"

    def _development_evaluation_tick(self, attempt: ActionAttempt, *, now: datetime) -> str:
        config = DevelopmentEvaluationConfig.model_validate(
            {key: value for key, value in attempt.executor.items() if key != "kind"}
        )
        champion = (
            self.repository.get_retrieval_evaluation(
                attempt.workspace_id, config.champion_evaluation_id
            )
            if config.champion_evaluation_id
            else None
        )
        execution = self.development_evaluation_executor.execute(attempt, config, champion=champion)
        verified = self._verify(attempt, execution.sealed_path)
        self._persist_development_evaluation_evidence(
            attempt,
            execution.sealed_path,
            now=now,
        )
        self.repository.complete_from_seal(
            verified,
            execution.sealed_path,
            worker_id=self.worker_id,
            now=now,
        )
        return "completed"

    def _resolve_reusable_completion(
        self, attempt: ActionAttempt
    ) -> tuple[ReusableCompletion, tuple[tuple[ActionAttempt, SealedActionResult], ...]] | None:
        """Match one content-identical completion, treating a damaged row as a miss.

        A reuse link written by another study can be unresolvable: its source row was
        removed, its action later failed, or the chain is cyclic. That is a cache miss,
        not a reason to fail the claimed action, so the worker reports the damaged
        attempt and executes the stage for real. Integrity of a resolved source is a
        separate question, decided by `_verify_reuse_source`, and it fails closed.
        """

        source = None
        try:
            source = self.repository.find_reusable_completion(
                attempt.workspace_id,
                attempt.result_key or "",
                stage=attempt.stage,
                exclude_action_id=attempt.action_id,
            )
            if source is None:
                return None
            return source, self.repository.reuse_source_chain(source.attempt)
        except (CampaignPersistenceError, RecordNotFoundError) as exc:
            logger.warning(
                "campaign reuse skipped for attempt %s: matched attempt %s is unresolvable: %s",
                attempt.attempt_id,
                source.attempt.attempt_id if source is not None else "none",
                exc,
            )
            return None

    def _verify_reuse_source(
        self, source_attempt: ActionAttempt, source_manifest: SealedActionResult
    ) -> SealedActionResult:
        """Bind one stored producer row to its sealed bytes and return the bound manifest.

        Reuse re-signs a producer's content under the consuming identity, so the row
        the resolution read is never trusted for having been stored. A remote seal
        binds through the digest inside its seal reference; a local seal binds through
        the signed envelope on disk, and the manifest that envelope carries is what the
        caller must build from.
        """

        source_uri = str(source_attempt.sealed_result_uri or "")
        if source_uri.startswith("bashgym-remote-seal://"):
            self._verify_sealed_data_build(
                source_attempt,
                source_manifest,
                compute_profile_id=source_manifest.compute_profile_id,
            )
            return source_manifest
        sealed_manifest = self._verify(source_attempt, Path(source_uri))
        if sealed_manifest != source_manifest:
            raise ArtifactSealError(
                f"{ArtifactSealError.code}: stored result does not match the sealed manifest"
            )
        return sealed_manifest

    def _reuse_tick(
        self,
        attempt: ActionAttempt,
        source: ReusableCompletion,
        chain: tuple[tuple[ActionAttempt, SealedActionResult], ...],
        *,
        now: datetime,
    ) -> str:
        """Complete the claimed attempt from a content-identical sealed result.

        The match may itself be a reusing attempt. The link is written against the
        attempt that executed, so every recorded link is one hop and repeated reuse
        of one content key cannot grow a chain.
        """

        verified_source = self._verify_reuse_source(source.attempt, source.manifest)
        verified_chain = tuple(
            (hop_attempt, self._verify_reuse_source(hop_attempt, hop_manifest))
            for hop_attempt, hop_manifest in chain
        )
        producer, producer_manifest = (
            verified_chain[-1] if verified_chain else (source.attempt, verified_source)
        )
        provenance = {
            "kind": "reused",
            REUSED_FROM_ATTEMPT_KEY: producer.attempt_id,
            REUSED_FROM_ACTION_KEY: producer.action_id,
            "compute_profile_id": producer_manifest.compute_profile_id,
        }
        derived = producer_manifest.model_copy(
            update={
                "workspace_id": attempt.workspace_id,
                "campaign_id": attempt.campaign_id,
                "study_id": attempt.study_id,
                "action_id": attempt.action_id,
                "attempt_id": attempt.attempt_id,
                "manifest_revision": attempt.manifest_revision,
                "candidate_digest": attempt.candidate_digest,
                "input_digest": attempt.input_digest,
                "claim_generation": attempt.claim_generation,
                "remote_process_identity": provenance,
                "log_reference": None,
                "started_at": now,
                "ended_at": now,
                "exit_reason": f"reused sealed result from {producer.attempt_id}",
                "resource_usage": (),
            }
        )
        source_uri = str(source.attempt.sealed_result_uri or "")
        if source_uri.startswith("bashgym-remote-seal://"):
            envelope = self.sealer.envelope_bytes(derived)
            sealed_reference: Path | str = (
                f"bashgym-remote-seal://{derived.compute_profile_id}/"
                f"{attempt.attempt_id}/sha256/{hashlib.sha256(envelope).hexdigest()}"
            )
        else:
            source_directory = Path(source_uri)
            sealed_directory = self.sealed_path(attempt)
            temporary = sealed_directory.with_name(sealed_directory.name + ".reuse-tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            for output in derived.outputs:
                destination = temporary / output.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_directory / output.path, destination)
                except OSError:
                    shutil.copyfile(source_directory / output.path, destination)
            sealed_reference = self.sealer.seal(temporary, sealed_directory, derived)
            self._verify(attempt, Path(sealed_reference))
        self.repository.complete_from_seal(
            derived,
            sealed_reference,
            worker_id=self.worker_id,
            artifact_metadata_by_path=source.artifact_metadata_by_path,
            now=now,
        )
        return "reused"

    def _fake_tick(
        self, attempt: ActionAttempt, *, now: datetime, crash_after_seal: bool = False
    ) -> str:
        request = FakeExecutionRequest(
            workspace_id=attempt.workspace_id,
            campaign_id=attempt.campaign_id,
            study_id=attempt.study_id,
            action_id=attempt.action_id,
            attempt_id=attempt.attempt_id,
            manifest_revision=attempt.manifest_revision,
            candidate_digest=attempt.candidate_digest,
            input_digest=attempt.input_digest,
            claim_generation=attempt.claim_generation,
            steps=int(attempt.executor.get("steps", 8)),
        )
        sealed_path, _manifest = self.executor.execute(request)
        if crash_after_seal:
            raise SimulatedWorkerCrashError(
                "simulated crash after seal and before completion commit"
            )
        verified = self._verify(attempt, sealed_path)
        self._ingest_sealed_metrics(attempt, sealed_path, now=now)
        self.repository.complete_from_seal(verified, sealed_path, worker_id=self.worker_id, now=now)
        return "completed"

    def run_once(
        self,
        *,
        now: datetime | None = None,
        crash_after_seal: bool = False,
    ) -> str:
        """Reconcile, claim, execute outside SQLite, then commit the sealed result."""

        tick_at = now or utc_now()
        try:
            leader = self._ensure_leader(tick_at)
        except LeaseBusyError:
            return "not_leader"
        recovery_result = self._consume_recovery_once(leader, now=tick_at)
        if recovery_result is not None:
            return recovery_result
        reconciled = self.reconcile_once(now=tick_at)
        if reconciled is not None:
            return reconciled
        if self._stop_requested:
            return "stopped"
        deferred_autoresearch_status = None
        excluded_campaign_keys: frozenset[tuple[str, str]] = frozenset()
        if self.autoresearch_loop is not None:
            loop_result = self.autoresearch_loop.tick(now=tick_at)
            if loop_result.effect_performed:
                return f"autoresearch_{loop_result.status}"
            if loop_result.agent_action_required:
                deferred_autoresearch_status = f"autoresearch_{loop_result.status}"
                if loop_result.workspace_id is not None and loop_result.campaign_id is not None:
                    excluded_campaign_keys = frozenset(
                        {(loop_result.workspace_id, loop_result.campaign_id)}
                    )
        controller_result = self.controller_once(
            leader,
            now=tick_at,
            excluded_campaign_keys=excluded_campaign_keys,
        )
        attempt = self.repository.claim_next_action(
            leader,
            ttl=self.action_ttl,
            now=tick_at,
        )
        if attempt is None:
            return controller_result or deferred_autoresearch_status or "idle"
        if attempt.result_key is not None and attempt.stage in REUSABLE_STAGES:
            resolved = self._resolve_reusable_completion(attempt)
            if resolved is not None:
                return self._reuse_tick(attempt, *resolved, now=tick_at)
        adapter = self._adapter_for(attempt)
        if crash_after_seal:
            return self._fake_tick(attempt, now=tick_at, crash_after_seal=True)
        return adapter.tick(self, attempt, now=tick_at)

    def run_forever(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = utc_now,
        heartbeat_seconds: float = 5.0,
        ready_poll_seconds: float = 2.0,
        idle_poll_seconds: float = 30.0,
    ) -> None:
        """Maintain the leader heartbeat while backing off full work checks when idle."""

        if heartbeat_seconds <= 0 or ready_poll_seconds <= 0 or idle_poll_seconds <= 0:
            raise ValueError("worker polling intervals must be positive")
        next_work_check: datetime | None = None
        try:
            while not self._stop_requested:
                now = clock()
                try:
                    self._ensure_leader(now)
                except LeaseBusyError:
                    sleep(heartbeat_seconds)
                    continue
                if next_work_check is None or now >= next_work_check:
                    result = self.run_once(now=now)
                    interval = (
                        ready_poll_seconds
                        if result
                        in {
                            "completed",
                            "reused",
                            "reconciled",
                            "unknown",
                            "remote_running",
                            "remote_paused",
                            "remote_unknown",
                            "remote_failed",
                            "remote_cancelling",
                            "remote_cancelled",
                            "remote_capacity_blocked",
                            "remote_force_stopping",
                            "stage_skipped",
                        }
                        else idle_poll_seconds
                    )
                    next_work_check = now + timedelta(seconds=interval)
                remaining = max(0.0, (next_work_check - now).total_seconds())
                sleep(min(heartbeat_seconds, remaining or heartbeat_seconds))
        finally:
            if self._leader is not None:
                try:
                    self.repository.release_lease(
                        self._leader.lease_key,
                        self._leader.owner_id,
                        self._leader.generation,
                        now=clock(),
                    )
                finally:
                    self._leader = None


__all__ = ["CampaignWorker", "SimulatedWorkerCrashError", "scheduler_lease_key"]
