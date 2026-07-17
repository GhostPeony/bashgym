"""Durable action queue and fenced completion transactions for campaign workers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from bashgym.campaigns.contracts import (
    PUBLIC_CAMPAIGN_BLOCKER_CODES,
    ActionAttempt,
    ActionStatus,
    AttemptStatus,
    BudgetEntryKind,
    BudgetLedgerEntry,
    Campaign,
    CampaignEvent,
    CampaignManifest,
    CampaignStatus,
    CodeLineageState,
    ContractModel,
    CredentialKind,
    FrozenContractModel,
    SealedActionResult,
    StageKind,
    StagePlan,
    StudyStatus,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.evaluation import (
    DevelopmentComparison,
    RetrievalEvaluationArtifact,
)
from bashgym.campaigns.lineage import code_mutation_kind_for_variable
from bashgym.campaigns.metrics import (
    MetricSeriesValue,
    TrainingAlert,
    TrainingMetricPoint,
    detect_training_alerts,
    parse_metric_lines,
)
from bashgym.campaigns.nemo_gym_evidence import (
    NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA,
    load_nemo_gym_campaign_evidence,
)
from bashgym.campaigns.persistence import (
    BudgetExceededError,
    CampaignPersistenceError,
    CampaignRepository,
    LeaseBusyError,
    LeaseLostError,
    LeaseRecord,
    OperationMutation,
    RecordNotFoundError,
    RevisionConflictError,
    _dt,
    _iso,
    _json,
)
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
    RemoteObservation,
    RemoteRunIdentity,
    RemoteRunState,
    RemoteStreamCursor,
    remote_executor_config,
)


class ActionClaimConflictError(CampaignPersistenceError):
    code = "campaign_action_claim_conflict"


class ActionIdentityMismatchError(CampaignPersistenceError):
    code = "campaign_action_identity_mismatch"


class ActionSpec(ContractModel):
    """Immutable logical stage input scheduled under the global leader fence."""

    schema_version: str = "campaign_action_spec.v1"
    workspace_id: str
    campaign_id: str
    study_id: str
    stage_index: int = Field(ge=0)
    stage: StageKind
    input_contract: dict[str, Any]
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_revision: int = Field(ge=1)
    budget_unit: str
    budget_reservation: float = Field(gt=0)
    executor_kind: Literal["fake", "ssh_remote", "development_evaluation"] = "fake"
    executor_config: dict[str, Any] = Field(default_factory=dict)
    fake_steps: int = Field(default=8, ge=2, le=10000)

    def model_post_init(self, __context: Any) -> None:
        if self.executor_kind == "ssh_remote" and self.stage not in {
            StageKind.SMOKE_TRAINING,
            StageKind.FULL_TRAINING,
            StageKind.DEVELOPMENT_EVALUATION,
        }:
            raise ValueError("remote executor is restricted to approved compute stages")
        if (
            self.executor_kind == "development_evaluation"
            and self.stage != StageKind.DEVELOPMENT_EVALUATION
        ):
            raise ValueError("development evaluation executor is restricted to its approved stage")

    @property
    def input_digest(self) -> str:
        return canonical_hash(
            {
                "stage": self.stage.value,
                "input_contract": self.input_contract,
                "candidate_digest": self.candidate_digest,
                "manifest_revision": self.manifest_revision,
                "executor_kind": self.executor_kind,
                "executor_config": self.executor_config,
            }
        )

    @property
    def action_key(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


@dataclass(frozen=True)
class RuntimeCompletion:
    attempt: ActionAttempt
    campaign_version: int
    event: CampaignEvent
    replayed: bool = False


class RemoteRunRecord(FrozenContractModel):
    schema_version: str = "campaign_remote_run_record.v1"
    workspace_id: str
    attempt_id: str
    claim_generation: int = Field(ge=1)
    identity: RemoteRunIdentity
    state: RemoteRunState
    metric_cursor: RemoteStreamCursor = Field(default_factory=RemoteStreamCursor)
    log_cursor: RemoteStreamCursor = Field(default_factory=RemoteStreamCursor)
    last_observation: RemoteObservation | None = None
    created_at: datetime
    updated_at: datetime


class CampaignArtifactRecord(FrozenContractModel):
    schema_version: str = "campaign_artifact_record.v1"
    workspace_id: str
    campaign_id: str
    artifact_id: str
    producer_action_id: str | None = None
    uri: str
    sha256: str
    size_bytes: int = Field(ge=0)
    schema_name: str
    sealed: bool
    valid: bool
    metadata: dict[str, Any]
    created_at: datetime


_ARTIFACT_CURSOR_PREFIX = "a1."


def _encode_artifact_cursor(sequence: int) -> str:
    payload = base64.urlsafe_b64encode(sequence.to_bytes(8, "big")).decode("ascii")
    return f"{_ARTIFACT_CURSOR_PREFIX}{payload.rstrip('=')}"


def _decode_artifact_cursor(cursor: str) -> int:
    if not cursor.startswith(_ARTIFACT_CURSOR_PREFIX):
        raise ValueError("invalid artifact continuation cursor")
    payload = cursor.removeprefix(_ARTIFACT_CURSOR_PREFIX)
    try:
        raw = base64.b64decode(f"{payload}=", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid artifact continuation cursor") from exc
    if len(raw) != 8:
        raise ValueError("invalid artifact continuation cursor")
    sequence = int.from_bytes(raw, "big")
    if sequence < 1:
        raise ValueError("invalid artifact continuation cursor")
    return sequence


def _study_status_for_stage(stage: StageKind) -> StudyStatus:
    return {
        StageKind.DATA_BUILD: StudyStatus.DATA_BUILDING,
        StageKind.CONTRACT_EVALUATION: StudyStatus.CONTRACT_EVALUATING,
        StageKind.SMOKE_TRAINING: StudyStatus.SMOKE_TRAINING,
        StageKind.FULL_TRAINING: StudyStatus.FULL_TRAINING,
        StageKind.DEVELOPMENT_EVALUATION: StudyStatus.DEVELOPMENT_EVALUATING,
        StageKind.COMPARISON: StudyStatus.COMPARING,
        StageKind.RECIPE_LOCK: StudyStatus.RECIPE_LOCKED,
        StageKind.PROTECTED_EVALUATION: StudyStatus.PROTECTED_EVALUATING,
        StageKind.PROMOTION: StudyStatus.PROMOTION_DECIDING,
    }[stage]


class CampaignRuntimeRepository(CampaignRepository):
    """Campaign repository extension used only by the resident controller."""

    @staticmethod
    def _require_leader(connection: sqlite3.Connection, leader: LeaseRecord, now: datetime) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM campaign_scheduler_leases
            WHERE lease_key = ? AND owner_id = ? AND generation = ? AND expires_at > ?
            """,
            (leader.lease_key, leader.owner_id, leader.generation, _iso(now)),
        ).fetchone()
        if row is None:
            raise LeaseLostError(LeaseLostError.code)

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> ActionAttempt:
        return ActionAttempt(
            attempt_id=row["attempt_id"],
            workspace_id=row["workspace_id"],
            campaign_id=row["campaign_id"],
            study_id=row["study_id"],
            action_id=row["action_id"],
            attempt_number=row["attempt_number"],
            claim_generation=row["claim_generation"],
            status=row["attempt_status"],
            input_digest=row["input_digest"],
            candidate_digest=row["candidate_digest"],
            manifest_revision=row["manifest_revision"],
            stage=row["stage_kind"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            executor=json.loads(row["executor_json"]),
            sealed_result_uri=row["sealed_result_uri"],
            created_at=row["attempt_created_at"],
            updated_at=row["attempt_updated_at"],
        )

    @staticmethod
    def _attempt_select() -> str:
        return """
            SELECT a.workspace_id, a.campaign_id, a.study_id, a.action_id,
                   a.stage_kind, a.input_digest, a.candidate_digest,
                   a.manifest_revision, a.sealed_result_uri,
                   t.attempt_id, t.attempt_number, t.claim_generation,
                   t.status AS attempt_status, t.lease_owner, t.lease_expires_at,
                   t.heartbeat_at, t.executor_json,
                   t.created_at AS attempt_created_at, t.updated_at AS attempt_updated_at
            FROM campaign_actions a
            JOIN campaign_attempts t
              ON t.workspace_id = a.workspace_id AND t.action_id = a.action_id
        """

    def get_attempt(self, workspace_id: str, attempt_id: str) -> ActionAttempt:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (workspace_id, attempt_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign attempt not found")
        return self._attempt_from_row(row)

    def list_attempts(self, workspace_id: str, campaign_id: str) -> tuple[ActionAttempt, ...]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                self._attempt_select()
                + """
                WHERE a.workspace_id = ? AND a.campaign_id = ?
                ORDER BY t.created_at, t.attempt_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(self._attempt_from_row(row) for row in rows)

    def next_controller_campaign(self) -> Campaign | None:
        """Return one deterministic active aggregate that can make controller progress."""

        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT c.* FROM campaigns c
                WHERE c.status = ? AND c.active_action_id IS NULL
                  AND (
                    c.active_study_id IS NOT NULL
                    OR EXISTS (
                        SELECT 1 FROM campaign_proposals p
                        WHERE p.workspace_id = c.workspace_id
                          AND p.campaign_id = c.campaign_id
                          AND p.status = ?
                    )
                  )
                ORDER BY c.updated_at, c.workspace_id, c.campaign_id
                LIMIT 1
                """,
                (CampaignStatus.ACTIVE.value, "submitted"),
            ).fetchone()
        return self._campaign_from_row(row) if row is not None else None

    def next_action_spec(
        self,
        workspace_id: str,
        campaign_id: str,
        study_id: str,
        *,
        executor_profiles: Mapping[tuple[str, str], ApprovedRemoteExecutorProfile] | None = None,
    ) -> ActionSpec:
        """Build a safe controller action from immutable study/manifest evidence.

        Actor-authored recipes may influence bounded fake-run steps and reservations,
        but never supply executable commands or remote process configuration. Live
        executors remain registered server-side by the approved compute profile.
        """

        self._require_initialized()
        with self._connection() as connection:
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            study_row = connection.execute(
                """
                SELECT * FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (workspace_id, campaign_id, study_id),
            ).fetchone()
            if campaign_row is None or study_row is None:
                raise RecordNotFoundError("campaign study not found")
            campaign = self._campaign_from_row(campaign_row)
            study = self._study_from_row(study_row)
            plan = study.stage_plan
            if study.current_stage_index >= len(plan.items):
                raise CampaignPersistenceError("campaign_stage_cursor_exhausted")
            item = plan.items[study.current_stage_index]
            proposal_row = connection.execute(
                """
                SELECT proposal_json FROM campaign_proposals
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (workspace_id, campaign_id, study.proposal_id),
            ).fetchone()
            lineage_row = connection.execute(
                """
                SELECT * FROM campaign_code_lineages
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (workspace_id, campaign_id, study.proposal_id),
            ).fetchone()
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (workspace_id, campaign_id, campaign.manifest_revision),
            ).fetchone()
        proposal = json.loads(proposal_row["proposal_json"])
        manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
        required_lineage_kind = code_mutation_kind_for_variable(
            str(proposal.get("primary_variable", ""))
        )
        code_lineage = None
        if lineage_row is not None:
            code_lineage = self._code_lineage_from_row(lineage_row)
            if code_lineage.state != CodeLineageState.CAPTURED:
                raise CampaignPersistenceError("campaign_code_lineage_not_captured")
        if required_lineage_kind is not None:
            if code_lineage is None:
                raise CampaignPersistenceError("campaign_code_lineage_not_registered")
            if code_lineage.mutation_kind != required_lineage_kind:
                raise CampaignPersistenceError("campaign_code_lineage_mutation_kind_mismatch")
        if item.stage == StageKind.DATA_BUILD:
            recipe = dict(proposal["dataset_recipe"])
        elif item.stage in {StageKind.SMOKE_TRAINING, StageKind.FULL_TRAINING}:
            recipe = dict(proposal["training_recipe"])
        else:
            recipe = dict(proposal["evaluation_recipe"])
        runtime = recipe.get("runtime", {})
        if not isinstance(runtime, dict):
            raise CampaignPersistenceError("campaign_recipe_runtime_invalid")
        executor_kind = runtime.get("executor_kind", "fake")
        executor_config: dict[str, Any] = {}
        if executor_kind in {"registered_compute", "registered_training", "ssh_remote"}:
            if item.stage not in {
                StageKind.SMOKE_TRAINING,
                StageKind.FULL_TRAINING,
                StageKind.DEVELOPMENT_EVALUATION,
            }:
                raise CampaignPersistenceError("campaign_remote_stage_not_allowed")
            profile_key = (
                manifest.compute_profile_id,
                campaign.target_model.target_contract_key,
            )
            profile = (executor_profiles or {}).get(profile_key)
            if profile is None:
                raise CampaignPersistenceError("campaign_remote_profile_unavailable")
            target_model_digest = canonical_hash(campaign.target_model.model_dump(mode="json"))
            if profile.target_model_digest != target_model_digest:
                raise CampaignPersistenceError("campaign_remote_target_model_mismatch")
            try:
                configured_stage = profile.stage_profile(item.stage)
            except KeyError as exc:
                raise CampaignPersistenceError("campaign_remote_profile_material_invalid") from exc
            if code_lineage is not None:
                binding = configured_stage.code_lineage_binding
                if binding is None:
                    raise CampaignPersistenceError(
                        "campaign_code_lineage_execution_binding_required"
                    )
                if (
                    binding.source_repository_profile_id
                    != code_lineage.source_repository_profile_id
                ):
                    raise CampaignPersistenceError(
                        "campaign_code_lineage_execution_binding_mismatch"
                    )
            recipe_digest = canonical_hash(
                {
                    "training_recipe": proposal["training_recipe"],
                    "profile_digest": profile.profile_digest,
                    "stage": item.stage.value,
                    "code_lineage_record_digest": (
                        code_lineage.record_digest if code_lineage is not None else None
                    ),
                }
            )
            try:
                executor_config = remote_executor_config(
                    profile,
                    item.stage,
                    recipe_digest=recipe_digest,
                    code_lineage=code_lineage,
                )
            except (KeyError, OSError, ValueError) as exc:
                raise CampaignPersistenceError("campaign_remote_profile_material_invalid") from exc
            budget_unit = configured_stage.budget_unit
            reservation = configured_stage.budget_reservation
        elif executor_kind == "fake":
            budget_unit = str(
                runtime.get(
                    "budget_unit",
                    (
                        "gpu_hours"
                        if "gpu_hours" in manifest.budget_limits
                        else sorted(manifest.budget_limits)[0]
                    ),
                )
            )
            reservation = float(runtime.get("budget_reservation", 0.01))
        else:
            raise CampaignPersistenceError("campaign_executor_kind_not_registered")
        if budget_unit not in manifest.budget_limits:
            raise CampaignPersistenceError("campaign_budget_unit_not_approved")
        fake_steps = int(runtime.get("fake_steps", 8))
        input_contract = {
            "stage_input": item.input_contract,
            "dataset_recipe_digest": canonical_hash(proposal["dataset_recipe"]),
            "training_recipe_digest": canonical_hash(proposal["training_recipe"]),
            "evaluation_recipe_digest": canonical_hash(proposal["evaluation_recipe"]),
        }
        if code_lineage is not None:
            input_contract["code_lineage"] = {
                "lineage_id": code_lineage.lineage_id,
                "record_digest": code_lineage.record_digest,
                "mutation_kind": code_lineage.mutation_kind.value,
                "source_repository_profile_id": (code_lineage.source_repository_profile_id),
                "base_commit": code_lineage.base_commit,
                "commit_sha": code_lineage.commit_sha,
                "patch_sha256": code_lineage.patch_sha256,
                "changed_paths": list(code_lineage.changed_paths),
            }
        return ActionSpec(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            study_id=study_id,
            stage_index=study.current_stage_index,
            stage=item.stage,
            input_contract=input_contract,
            candidate_digest=study.candidate_digest,
            manifest_revision=campaign.manifest_revision,
            budget_unit=budget_unit,
            budget_reservation=reservation,
            executor_kind=(
                "ssh_remote"
                if executor_kind
                in {"registered_compute", "registered_training", "ssh_remote"}
                else executor_kind
            ),
            executor_config=executor_config,
            fake_steps=fake_steps,
        )

    def record_controller_blocker_under_leader(
        self,
        workspace_id: str,
        campaign_id: str,
        study_id: str,
        leader: LeaseRecord,
        *,
        code: str,
        now: datetime | None = None,
    ) -> bool:
        """Append one bounded blocker event without reserving budget or scheduling."""

        self._require_initialized()
        if code not in PUBLIC_CAMPAIGN_BLOCKER_CODES:
            code = "campaign_controller_action_blocked"
        observed_at = now or utc_now()
        with self._connection(immediate=True) as connection:
            self._require_leader(connection, leader, observed_at)
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            study_row = connection.execute(
                """
                SELECT current_stage_index, stage_plan_json
                FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (workspace_id, campaign_id, study_id),
            ).fetchone()
            if campaign_row is None or study_row is None:
                raise RecordNotFoundError("campaign study not found")
            campaign = self._campaign_from_row(campaign_row)
            plan = StagePlan.model_validate_json(study_row["stage_plan_json"])
            cursor = int(study_row["current_stage_index"])
            stage = plan.items[cursor].stage.value if cursor < len(plan.items) else "exhausted"
            event_seed = canonical_hash(
                {
                    "workspace_id": workspace_id,
                    "campaign_id": campaign_id,
                    "study_id": study_id,
                    "manifest_revision": campaign.manifest_revision,
                    "stage_index": cursor,
                    "stage": stage,
                    "code": code,
                }
            )
            event_id = f"evt-{event_seed[:24]}"
            exists = connection.execute(
                "SELECT 1 FROM campaign_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if exists is not None:
                return False
            event = CampaignEvent(
                event_id=event_id,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=campaign.version,
                event_type="campaign:action-blocked",
                payload={
                    "study_id": study_id,
                    "stage_index": cursor,
                    "stage": stage,
                    "code": code,
                },
                actor_id="campaign-controller",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=f"controller-blocked-{event_seed[:16]}",
                idempotency_key=f"controller-blocked-{event_seed[:24]}",
                created_at=observed_at,
            )
            self._insert_event(connection, event)
        return True

    def skip_not_applicable_stages_under_leader(
        self,
        workspace_id: str,
        campaign_id: str,
        study_id: str,
        leader: LeaseRecord,
        *,
        expected_campaign_version: int,
        now: datetime | None = None,
    ) -> bool:
        """Advance consecutive policy-marked stages without creating an action."""

        self._require_initialized()
        skipped_at = now or utc_now()
        with self._connection(immediate=True) as connection:
            self._require_leader(connection, leader, skipped_at)
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            study_row = connection.execute(
                """
                SELECT * FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (workspace_id, campaign_id, study_id),
            ).fetchone()
            if campaign_row is None or study_row is None:
                raise RecordNotFoundError("campaign study not found")
            campaign = self._campaign_from_row(campaign_row)
            if campaign.version != expected_campaign_version:
                raise RevisionConflictError(expected_campaign_version, campaign.version)
            if (
                campaign.status != CampaignStatus.ACTIVE
                or campaign.active_study_id != study_id
                or campaign.active_action_id is not None
            ):
                raise CampaignPersistenceError("campaign_not_schedulable")
            plan = StagePlan.model_validate_json(study_row["stage_plan_json"])
            cursor = int(study_row["current_stage_index"])
            skipped: list[dict[str, Any]] = []
            while cursor < len(plan.items):
                item = plan.items[cursor]
                if item.disposition.value == "required":
                    break
                skipped.append({"stage_index": cursor, "stage": item.stage.value})
                cursor += 1
            if not skipped:
                return False
            finished = cursor >= len(plan.items)
            next_status = (
                StudyStatus.COMPLETED
                if finished
                else _study_status_for_stage(plan.items[cursor].stage)
            )
            connection.execute(
                """
                UPDATE campaign_studies SET status = ?, current_stage_index = ?,
                    version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (
                    next_status.value,
                    cursor,
                    _iso(skipped_at),
                    workspace_id,
                    campaign_id,
                    study_id,
                ),
            )
            updated = connection.execute(
                """
                UPDATE campaigns SET active_study_id = ?, version = version + 1,
                    updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    None if finished else study_id,
                    _iso(skipped_at),
                    workspace_id,
                    campaign_id,
                    expected_campaign_version,
                ),
            )
            if updated.rowcount != 1:
                raise RevisionConflictError(expected_campaign_version, campaign.version)
            event_seed = canonical_hash(
                {
                    "campaign_id": campaign_id,
                    "study_id": study_id,
                    "campaign_version": campaign.version + 1,
                    "skipped": skipped,
                }
            )
            event = CampaignEvent(
                event_id=f"evt-{event_seed[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=campaign.version + 1,
                event_type="campaign:stages-skipped",
                payload={
                    "study_id": study_id,
                    "skipped": skipped,
                    "next_stage_index": cursor,
                    "study_completed": finished,
                },
                actor_id="campaign-controller",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=f"controller-skip-v{campaign.version}",
                idempotency_key=f"controller-skip-{event_seed[:24]}",
                created_at=skipped_at,
            )
            self._insert_event(connection, event)
        return True

    @staticmethod
    def _remote_run_from_row(row: sqlite3.Row) -> RemoteRunRecord:
        return RemoteRunRecord(
            workspace_id=row["workspace_id"],
            attempt_id=row["attempt_id"],
            claim_generation=row["claim_generation"],
            identity=RemoteRunIdentity.model_validate_json(row["identity_json"]),
            state=row["state"],
            metric_cursor=RemoteStreamCursor.model_validate_json(row["metric_cursor_json"]),
            log_cursor=RemoteStreamCursor.model_validate_json(row["log_cursor_json"]),
            last_observation=(
                RemoteObservation.model_validate_json(row["last_observation_json"])
                if row["last_observation_json"]
                else None
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_remote_run(self, workspace_id: str, attempt_id: str) -> RemoteRunRecord | None:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_remote_runs WHERE workspace_id = ? AND attempt_id = ?",
                (workspace_id, attempt_id),
            ).fetchone()
        return self._remote_run_from_row(row) if row is not None else None

    def retry_action(
        self,
        workspace_id: str,
        campaign_id: str,
        action_id: str,
        *,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Schedule one new attempt for a terminal failed action under the same hashes."""

        self._require_initialized()
        mutation_kind = "campaign.action.retry"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "action_id": action_id,
                "expected_version": expected_version,
            }
        )
        retryable = {
            ActionStatus.FAILED.value,
            ActionStatus.CANCELLED.value,
            ActionStatus.FORCE_STOPPED.value,
        }
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            campaign = self._campaign_from_row(campaign_row)
            if campaign.version != expected_version:
                raise RevisionConflictError(expected_version, campaign.version)
            if campaign.status != CampaignStatus.ACTIVE or campaign.active_action_id is not None:
                raise ActionClaimConflictError("campaign_action_not_retryable")
            action = connection.execute(
                """
                SELECT * FROM campaign_actions
                WHERE workspace_id = ? AND campaign_id = ? AND action_id = ?
                """,
                (workspace_id, campaign_id, action_id),
            ).fetchone()
            if action is None:
                raise RecordNotFoundError("campaign action not found")
            if action["status"] not in retryable:
                raise ActionClaimConflictError("campaign_action_not_retryable")
            latest = connection.execute(
                """
                SELECT * FROM campaign_attempts
                WHERE workspace_id = ? AND action_id = ?
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (workspace_id, action_id),
            ).fetchone()
            attempt_number = int(latest["attempt_number"]) + 1
            reservation = json.loads(action["reservation_json"])
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (workspace_id, campaign_id, int(action["manifest_revision"])),
            ).fetchone()
            manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
            unit = str(reservation["unit"])
            amount = float(reservation["amount"])
            totals = connection.execute(
                """
                SELECT COALESCE(SUM(reserved_delta), 0) AS reserved,
                       COALESCE(SUM(actual_delta), 0) AS actual,
                       COALESCE(SUM(limit_delta), 0) AS limit_delta
                FROM campaign_budget_ledger
                WHERE workspace_id = ? AND campaign_id = ? AND unit = ?
                """,
                (workspace_id, campaign_id, unit),
            ).fetchone()
            effective_limit = float(manifest.budget_limits[unit]) + float(totals["limit_delta"])
            if float(totals["reserved"]) + float(totals["actual"]) + amount > effective_limit:
                raise BudgetExceededError(BudgetExceededError.code)
            scheduled_at = utc_now()
            attempt_id = f"attempt-{hashlib.sha256(f'{action_id}:{attempt_number}'.encode()).hexdigest()[:24]}-{attempt_number}"
            connection.execute(
                """
                INSERT INTO campaign_budget_ledger(
                    workspace_id, campaign_id, entry_id, unit, entry_kind,
                    reserved_delta, actual_delta, limit_delta, action_id,
                    evidence_json, actor_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    f"budget-reserve-{action_id}-attempt-{attempt_number}",
                    unit,
                    BudgetEntryKind.RESERVE.value,
                    amount,
                    action_id,
                    _json({"attempt_id": attempt_id, "retry": True}),
                    actor_id,
                    _iso(scheduled_at),
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_attempts(
                    workspace_id, action_id, attempt_id, attempt_number,
                    claim_generation, status, executor_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    action_id,
                    attempt_id,
                    attempt_number,
                    AttemptStatus.SCHEDULED.value,
                    latest["executor_json"],
                    _iso(scheduled_at),
                    _iso(scheduled_at),
                ),
            )
            connection.execute(
                """
                UPDATE campaign_actions SET status = ?, sealed_result_uri = NULL,
                    version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND action_id = ?
                """,
                (ActionStatus.SCHEDULED.value, _iso(scheduled_at), workspace_id, action_id),
            )
            connection.execute(
                """
                UPDATE campaign_studies SET status = ?, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND study_id = ?
                """,
                (
                    _study_status_for_stage(StageKind(action["stage_kind"])).value,
                    _iso(scheduled_at),
                    workspace_id,
                    action["study_id"],
                ),
            )
            updated = campaign.model_copy(
                update={
                    "active_study_id": str(action["study_id"]),
                    "active_action_id": action_id,
                    "version": campaign.version + 1,
                    "updated_at": scheduled_at,
                }
            )
            connection.execute(
                """
                UPDATE campaigns SET active_study_id = ?, active_action_id = ?,
                    version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    updated.active_study_id,
                    action_id,
                    updated.version,
                    _iso(scheduled_at),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            details = {
                "action_id": action_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "input_digest": str(action["input_digest"]),
                "candidate_digest": str(action["candidate_digest"]),
            }
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'retry:{action_id}:{idempotency_key}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:action-retry-scheduled",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=scheduled_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def request_force_stop(
        self,
        workspace_id: str,
        campaign_id: str,
        action_id: str,
        expected_identity: RemoteRunIdentity,
        *,
        reason: str,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> OperationMutation:
        """Queue a worker force-stop only when the entire persisted identity matches."""

        self._require_initialized()
        identity_digest = canonical_hash(expected_identity.model_dump(mode="json"))
        mutation_kind = "campaign.action.force_stop"
        request_hash = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "action_id": action_id,
                "identity_digest": identity_digest,
                "reason": reason,
                "expected_version": expected_version,
            }
        )
        with self._connection(immediate=True) as connection:
            replay = self._replay_operation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
            )
            if replay is not None:
                return replay
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            campaign = self._campaign_from_row(campaign_row)
            if campaign.version != expected_version:
                raise RevisionConflictError(expected_version, campaign.version)
            if campaign.active_action_id != action_id:
                raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
            remote = connection.execute(
                """
                SELECT r.identity_json, t.attempt_id
                FROM campaign_actions a
                JOIN campaign_attempts t ON t.workspace_id = a.workspace_id AND t.action_id = a.action_id
                JOIN campaign_remote_runs r ON r.workspace_id = t.workspace_id AND r.attempt_id = t.attempt_id
                WHERE a.workspace_id = ? AND a.campaign_id = ? AND a.action_id = ?
                  AND t.status IN (?, ?)
                ORDER BY t.attempt_number DESC LIMIT 1
                """,
                (
                    workspace_id,
                    campaign_id,
                    action_id,
                    AttemptStatus.RUNNING.value,
                    AttemptStatus.UNKNOWN.value,
                ),
            ).fetchone()
            if remote is None:
                raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
            persisted = RemoteRunIdentity.model_validate_json(remote["identity_json"])
            if persisted != expected_identity:
                raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
            requested_at = utc_now()
            request_id = f"control-{hashlib.sha256(f'{action_id}:{idempotency_key}'.encode()).hexdigest()[:24]}"
            connection.execute(
                """
                INSERT INTO campaign_action_control_requests(
                    workspace_id, campaign_id, action_id, request_id, control,
                    expected_identity_digest, reason, state, actor_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'force_stop', ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    action_id,
                    request_id,
                    identity_digest,
                    reason,
                    actor_id,
                    _iso(requested_at),
                    _iso(requested_at),
                ),
            )
            updated = campaign.model_copy(
                update={"version": campaign.version + 1, "updated_at": requested_at}
            )
            connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (updated.version, _iso(requested_at), workspace_id, campaign_id, expected_version),
            )
            details = {
                "action_id": action_id,
                "attempt_id": str(remote["attempt_id"]),
                "request_id": request_id,
                "expected_identity_digest": identity_digest,
                "state": "pending",
            }
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'force-stop:{request_id}'.encode()).hexdigest()[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=updated.version,
                event_type="campaign:force-stop-requested",
                payload=details,
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=requested_at,
            )
            self._insert_event(connection, event)
            self._insert_mutation(
                connection,
                workspace_id=workspace_id,
                actor_id=actor_id,
                mutation_kind=mutation_kind,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                campaign=updated,
                event=event,
                response_json=_json(
                    {"campaign": updated.model_dump(mode="json"), "details": details}
                ),
            )
        return OperationMutation(updated, event, details)

    def pending_force_stop_request(
        self, workspace_id: str, action_id: str, identity: RemoteRunIdentity
    ) -> str | None:
        identity_digest = canonical_hash(identity.model_dump(mode="json"))
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT request_id FROM campaign_action_control_requests
                WHERE workspace_id = ? AND action_id = ? AND control = 'force_stop'
                  AND expected_identity_digest = ? AND state = 'pending'
                ORDER BY created_at LIMIT 1
                """,
                (workspace_id, action_id, identity_digest),
            ).fetchone()
        return str(row["request_id"]) if row is not None else None

    def settle_force_stop_request(
        self, workspace_id: str, request_id: str, *, executed: bool
    ) -> None:
        settled_at = utc_now()
        with self._connection(immediate=True) as connection:
            connection.execute(
                """
                UPDATE campaign_action_control_requests SET state = ?, updated_at = ?
                WHERE workspace_id = ? AND request_id = ? AND state = 'pending'
                """,
                (
                    "executed" if executed else "identity_mismatch",
                    _iso(settled_at),
                    workspace_id,
                    request_id,
                ),
            )

    def append_remote_metrics(
        self,
        attempt: ActionAttempt,
        lines: tuple[str, ...],
        *,
        source: str,
        cursor_end: int,
        now: datetime | None = None,
    ) -> tuple[tuple[TrainingMetricPoint, ...], tuple[TrainingAlert, ...]]:
        """Idempotently persist complete metric lines before advancing their cursor."""

        if not lines:
            return (), ()
        observed_at = now or utc_now()
        points = parse_metric_lines(lines)
        alerts = detect_training_alerts(points)
        with self._connection(immediate=True) as connection:
            for point in points:
                for name, value in point.values.items():
                    existing = connection.execute(
                        """
                        SELECT metric_value FROM campaign_metric_points
                        WHERE workspace_id = ? AND attempt_id = ? AND source = ?
                          AND step = ? AND metric_name = ?
                        """,
                        (
                            attempt.workspace_id,
                            attempt.attempt_id,
                            source,
                            point.step,
                            name,
                        ),
                    ).fetchone()
                    if existing is not None and float(existing["metric_value"]) != value:
                        raise CampaignPersistenceError("campaign_metric_value_conflict")
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO campaign_metric_points(
                            workspace_id, attempt_id, source, step, metric_name,
                            metric_value, raw_sha256, observed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            attempt.workspace_id,
                            attempt.attempt_id,
                            source,
                            point.step,
                            name,
                            value,
                            point.raw_sha256,
                            _iso(observed_at),
                        ),
                    )
            for alert in alerts:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO campaign_training_alerts(
                        workspace_id, attempt_id, alert_code, step, metric_name,
                        severity, metric_value, message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.workspace_id,
                        attempt.attempt_id,
                        alert.code,
                        alert.step,
                        alert.metric_name,
                        alert.severity.value,
                        alert.metric_value,
                        alert.message,
                        _iso(observed_at),
                    ),
                )
            campaign_row = connection.execute(
                "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (attempt.workspace_id, attempt.campaign_id),
            ).fetchone()
            event_identity = f"metrics:{attempt.attempt_id}:{source}:{cursor_end}"
            event_id = f"evt-{hashlib.sha256(event_identity.encode()).hexdigest()[:24]}"
            if (
                connection.execute(
                    "SELECT 1 FROM campaign_events WHERE event_id = ?", (event_id,)
                ).fetchone()
                is None
            ):
                self._insert_event(
                    connection,
                    CampaignEvent(
                        event_id=event_id,
                        workspace_id=attempt.workspace_id,
                        campaign_id=attempt.campaign_id,
                        sequence=self._next_event_sequence(
                            connection, attempt.workspace_id, attempt.campaign_id
                        ),
                        aggregate_version=int(campaign_row["version"]),
                        event_type="campaign:training-metrics-appended",
                        payload={
                            "action_id": attempt.action_id,
                            "attempt_id": attempt.attempt_id,
                            "source": source,
                            "cursor_end": cursor_end,
                            "steps": sorted({point.step for point in points}),
                            "metric_names": sorted(
                                {name for point in points for name in point.values}
                            ),
                            "alert_count": len(alerts),
                        },
                        actor_id="campaign-controller",
                        credential_kind=CredentialKind.CONTROLLER,
                        correlation_id="worker-metric-stream",
                        idempotency_key=f"metrics-{attempt.attempt_id}-{cursor_end}",
                        created_at=observed_at,
                    ),
                )
        return points, alerts

    def get_metric_series(
        self,
        workspace_id: str,
        attempt_id: str,
        metric_name: str,
        *,
        source: str | None = None,
        after_step: int = -1,
        limit: int = 2000,
    ) -> tuple[MetricSeriesValue, ...]:
        self._require_initialized()
        if after_step < -1 or limit < 1 or limit > 5000:
            raise ValueError("campaign_metric_pagination_invalid")
        with self._connection() as connection:
            if source is None:
                rows = connection.execute(
                    """
                    SELECT source, step, metric_value, observed_at FROM campaign_metric_points
                    WHERE workspace_id = ? AND attempt_id = ? AND metric_name = ?
                      AND step > ?
                    ORDER BY source, step
                    LIMIT ?
                    """,
                    (workspace_id, attempt_id, metric_name, after_step, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT source, step, metric_value, observed_at FROM campaign_metric_points
                    WHERE workspace_id = ? AND attempt_id = ? AND metric_name = ? AND source = ?
                      AND step > ?
                    ORDER BY step
                    LIMIT ?
                    """,
                    (workspace_id, attempt_id, metric_name, source, after_step, limit),
                ).fetchall()
        return tuple(
            MetricSeriesValue(
                step=row["step"],
                source=row["source"],
                value=row["metric_value"],
                observed_at=row["observed_at"],
            )
            for row in rows
        )

    def list_artifacts(
        self, workspace_id: str, campaign_id: str
    ) -> tuple[CampaignArtifactRecord, ...]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM campaign_artifacts
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, artifact_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(
            CampaignArtifactRecord(
                workspace_id=row["workspace_id"],
                campaign_id=row["campaign_id"],
                artifact_id=row["artifact_id"],
                producer_action_id=row["producer_action_id"],
                uri=row["uri"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                schema_name=row["schema_name"],
                sealed=bool(row["sealed"]),
                valid=bool(row["valid"]),
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        )

    def list_artifact_page(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        after_cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[tuple[CampaignArtifactRecord, ...], str | None, bool]:
        """Return an append-stable artifact page with an opaque continuation cursor."""

        self._require_initialized()
        limit = max(1, min(limit, 200))
        after_sequence = _decode_artifact_cursor(after_cursor) if after_cursor else 0
        with self._connection() as connection:
            campaign = connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign is None:
                raise RecordNotFoundError("campaign not found")
            rows = connection.execute(
                """
                SELECT rowid AS artifact_sequence, * FROM campaign_artifacts
                WHERE workspace_id = ? AND campaign_id = ? AND rowid > ?
                ORDER BY rowid LIMIT ?
                """,
                (workspace_id, campaign_id, after_sequence, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        artifacts = tuple(
            CampaignArtifactRecord(
                workspace_id=row["workspace_id"],
                campaign_id=row["campaign_id"],
                artifact_id=row["artifact_id"],
                producer_action_id=row["producer_action_id"],
                uri=row["uri"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                schema_name=row["schema_name"],
                sealed=bool(row["sealed"]),
                valid=bool(row["valid"]),
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in page_rows
        )
        next_cursor = (
            _encode_artifact_cursor(int(page_rows[-1]["artifact_sequence"]))
            if has_more and page_rows
            else None
        )
        return artifacts, next_cursor, has_more

    def store_retrieval_evaluation(
        self,
        workspace_id: str,
        campaign_id: str,
        artifact: RetrievalEvaluationArtifact,
        *,
        now: datetime | None = None,
    ) -> str:
        """Store immutable evaluation evidence with content-addressed identity."""

        created_at = now or utc_now()
        evaluation_id = f"eval-{canonical_hash(artifact.model_dump(mode='json'))[:24]}"
        payload = _json(artifact.model_dump(mode="json"))
        with self._connection(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT evaluation_json FROM campaign_evaluations
                WHERE workspace_id = ? AND evaluation_id = ?
                """,
                (workspace_id, evaluation_id),
            ).fetchone()
            if existing is not None:
                if existing["evaluation_json"] != payload:
                    raise CampaignPersistenceError("campaign_evaluation_identity_conflict")
                return evaluation_id
            connection.execute(
                """
                INSERT INTO campaign_evaluations(
                    workspace_id, campaign_id, evaluation_id, evaluation_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (workspace_id, campaign_id, evaluation_id, payload, _iso(created_at)),
            )
        return evaluation_id

    def get_retrieval_evaluation(
        self, workspace_id: str, evaluation_id: str
    ) -> RetrievalEvaluationArtifact:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT evaluation_json FROM campaign_evaluations
                WHERE workspace_id = ? AND evaluation_id = ?
                """,
                (workspace_id, evaluation_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign evaluation not found")
        return RetrievalEvaluationArtifact.model_validate_json(row["evaluation_json"])

    def store_development_comparison(
        self,
        workspace_id: str,
        campaign_id: str,
        comparison: DevelopmentComparison,
        *,
        now: datetime | None = None,
    ) -> str:
        """Persist a deterministic gate result without granting promotion authority."""

        created_at = now or utc_now()
        decision_id = f"gate-{comparison.comparison_digest[:24]}"
        payload = _json(comparison.model_dump(mode="json"))
        with self._connection(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT decision_json FROM campaign_gate_decisions
                WHERE workspace_id = ? AND decision_id = ?
                """,
                (workspace_id, decision_id),
            ).fetchone()
            if existing is not None:
                if existing["decision_json"] != payload:
                    raise CampaignPersistenceError("campaign_gate_identity_conflict")
                return decision_id
            blocking_human_work = connection.execute(
                """
                SELECT 1
                FROM campaign_human_work AS work
                JOIN campaigns AS campaign
                  ON campaign.workspace_id = work.workspace_id
                 AND campaign.campaign_id = work.campaign_id
                WHERE work.workspace_id = ? AND work.campaign_id = ?
                  AND work.campaign_revision = campaign.manifest_revision
                  AND work.blocking = 1
                  AND work.state NOT IN ('submitted', 'replaced')
                LIMIT 1
                """,
                (workspace_id, campaign_id),
            ).fetchone()
            if blocking_human_work is not None:
                raise CampaignPersistenceError("campaign_human_work_incomplete")
            connection.execute(
                """
                INSERT INTO campaign_gate_decisions(
                    workspace_id, campaign_id, decision_id, decision_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (workspace_id, campaign_id, decision_id, payload, _iso(created_at)),
            )
        return decision_id

    def get_development_comparison(
        self, workspace_id: str, decision_id: str
    ) -> DevelopmentComparison:
        self._require_initialized()
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT decision_json FROM campaign_gate_decisions
                WHERE workspace_id = ? AND decision_id = ?
                """,
                (workspace_id, decision_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("campaign gate decision not found")
        return DevelopmentComparison.model_validate_json(row["decision_json"])

    def list_development_comparisons(
        self, workspace_id: str, campaign_id: str
    ) -> tuple[DevelopmentComparison, ...]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT decision_json FROM campaign_gate_decisions
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, decision_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(
            DevelopmentComparison.model_validate_json(row["decision_json"]) for row in rows
        )

    def register_remote_identity(
        self,
        attempt: ActionAttempt,
        identity: RemoteRunIdentity,
        *,
        now: datetime | None = None,
    ) -> RemoteRunRecord:
        """Persist launch identity transactionally before another remote operation."""

        registered_at = now or utc_now()
        if attempt.claim_generation < 1 or attempt.status != AttemptStatus.RUNNING:
            raise ActionClaimConflictError(ActionClaimConflictError.code)
        with self._connection(immediate=True) as connection:
            attempt_row = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (attempt.workspace_id, attempt.attempt_id),
            ).fetchone()
            if attempt_row is None:
                raise RecordNotFoundError("campaign attempt not found")
            current = self._attempt_from_row(attempt_row)
            if (
                current.status != AttemptStatus.RUNNING
                or current.claim_generation != attempt.claim_generation
            ):
                raise ActionClaimConflictError(ActionClaimConflictError.code)
            existing = connection.execute(
                "SELECT * FROM campaign_remote_runs WHERE workspace_id = ? AND attempt_id = ?",
                (attempt.workspace_id, attempt.attempt_id),
            ).fetchone()
            if existing is not None:
                record = self._remote_run_from_row(existing)
                if (
                    record.claim_generation != attempt.claim_generation
                    or record.identity != identity
                ):
                    raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
                return record
            connection.execute(
                """
                INSERT INTO campaign_remote_runs(
                    workspace_id, attempt_id, claim_generation, identity_json, state,
                    metric_cursor_json, log_cursor_json, last_observation_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    attempt.workspace_id,
                    attempt.attempt_id,
                    attempt.claim_generation,
                    _json(identity.model_dump(mode="json")),
                    RemoteRunState.RUNNING.value,
                    _json(RemoteStreamCursor().model_dump(mode="json")),
                    _json(RemoteStreamCursor().model_dump(mode="json")),
                    _iso(registered_at),
                    _iso(registered_at),
                ),
            )
            campaign_row = connection.execute(
                "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (attempt.workspace_id, attempt.campaign_id),
            ).fetchone()
            event_identity = f"remote:{attempt.attempt_id}:{attempt.claim_generation}"
            self._insert_event(
                connection,
                CampaignEvent(
                    event_id=f"evt-{hashlib.sha256(event_identity.encode()).hexdigest()[:24]}",
                    workspace_id=attempt.workspace_id,
                    campaign_id=attempt.campaign_id,
                    sequence=self._next_event_sequence(
                        connection, attempt.workspace_id, attempt.campaign_id
                    ),
                    aggregate_version=int(campaign_row["version"]),
                    event_type="campaign:remote-run-registered",
                    payload={
                        "action_id": attempt.action_id,
                        "attempt_id": attempt.attempt_id,
                        "compute_profile_id": identity.compute_profile_id,
                        "run_id": identity.run_id,
                        "claim_generation": attempt.claim_generation,
                    },
                    actor_id="campaign-controller",
                    credential_kind=CredentialKind.CONTROLLER,
                    correlation_id="worker-remote-launch",
                    idempotency_key=f"remote-{attempt.attempt_id}-{attempt.claim_generation}",
                    created_at=registered_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM campaign_remote_runs WHERE workspace_id = ? AND attempt_id = ?",
                (attempt.workspace_id, attempt.attempt_id),
            ).fetchone()
        return self._remote_run_from_row(row)

    def adopt_remote_attempt(
        self,
        attempt: ActionAttempt,
        leader: LeaseRecord,
        *,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> ActionAttempt:
        """Fence and adopt an expired remote process without relaunching it."""

        adopted_at = now or utc_now()
        expires_at = adopted_at + ttl
        with self._connection(immediate=True) as connection:
            self._require_leader(connection, leader, adopted_at)
            lease_key = f"action:{attempt.action_id}"
            lease = connection.execute(
                "SELECT * FROM campaign_scheduler_leases WHERE lease_key = ?",
                (lease_key,),
            ).fetchone()
            if lease is None or _dt(lease["expires_at"]) > adopted_at:
                raise LeaseBusyError(LeaseBusyError.code)
            generation = int(lease["generation"]) + 1
            connection.execute(
                """
                UPDATE campaign_scheduler_leases SET owner_id = ?, generation = ?,
                    expires_at = ?, heartbeat_at = ? WHERE lease_key = ?
                """,
                (
                    leader.owner_id,
                    generation,
                    _iso(expires_at),
                    _iso(adopted_at),
                    lease_key,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE campaign_attempts SET status = ?, claim_generation = ?,
                    lease_owner = ?, lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND claim_generation = ?
                  AND status IN (?, ?)
                """,
                (
                    AttemptStatus.RUNNING.value,
                    generation,
                    leader.owner_id,
                    _iso(expires_at),
                    _iso(adopted_at),
                    _iso(adopted_at),
                    attempt.workspace_id,
                    attempt.attempt_id,
                    attempt.claim_generation,
                    AttemptStatus.RUNNING.value,
                    AttemptStatus.UNKNOWN.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ActionClaimConflictError(ActionClaimConflictError.code)
            connection.execute(
                """
                UPDATE campaign_actions SET status = ?, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND action_id = ? AND status IN (?, ?)
                """,
                (
                    ActionStatus.RUNNING.value,
                    _iso(adopted_at),
                    attempt.workspace_id,
                    attempt.action_id,
                    ActionStatus.RUNNING.value,
                    ActionStatus.UNKNOWN.value,
                ),
            )
            connection.execute(
                """
                UPDATE campaign_remote_runs SET claim_generation = ?, updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND claim_generation = ?
                """,
                (
                    generation,
                    _iso(adopted_at),
                    attempt.workspace_id,
                    attempt.attempt_id,
                    attempt.claim_generation,
                ),
            )
            campaign_row = connection.execute(
                "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (attempt.workspace_id, attempt.campaign_id),
            ).fetchone()
            event_identity = f"adopt:{attempt.attempt_id}:{generation}"
            self._insert_event(
                connection,
                CampaignEvent(
                    event_id=f"evt-{hashlib.sha256(event_identity.encode()).hexdigest()[:24]}",
                    workspace_id=attempt.workspace_id,
                    campaign_id=attempt.campaign_id,
                    sequence=self._next_event_sequence(
                        connection, attempt.workspace_id, attempt.campaign_id
                    ),
                    aggregate_version=int(campaign_row["version"]),
                    event_type="campaign:remote-run-adopted",
                    payload={
                        "action_id": attempt.action_id,
                        "attempt_id": attempt.attempt_id,
                        "claim_generation": generation,
                    },
                    actor_id="campaign-controller",
                    credential_kind=CredentialKind.CONTROLLER,
                    correlation_id=f"worker-{leader.owner_id}",
                    idempotency_key=f"adopt-{attempt.attempt_id}-{generation}",
                    created_at=adopted_at,
                ),
            )
            updated = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (attempt.workspace_id, attempt.attempt_id),
            ).fetchone()
        return self._attempt_from_row(updated)

    def defer_unlaunched_remote_attempt(
        self,
        attempt: ActionAttempt,
        *,
        worker_id: str,
        reasons: tuple[str, ...],
        now: datetime | None = None,
    ) -> ActionAttempt:
        """Return a capacity-blocked, never-launched claim to the durable queue."""

        deferred_at = now or utc_now()
        with self._connection(immediate=True) as connection:
            if connection.execute(
                "SELECT 1 FROM campaign_remote_runs WHERE workspace_id = ? AND attempt_id = ?",
                (attempt.workspace_id, attempt.attempt_id),
            ).fetchone():
                raise CampaignPersistenceError("campaign_remote_run_already_launched")
            lease_key = f"action:{attempt.action_id}"
            lease = connection.execute(
                """
                UPDATE campaign_scheduler_leases SET expires_at = ?, heartbeat_at = ?
                WHERE lease_key = ? AND owner_id = ? AND generation = ?
                """,
                (
                    _iso(deferred_at),
                    _iso(deferred_at),
                    lease_key,
                    worker_id,
                    attempt.claim_generation,
                ),
            )
            if lease.rowcount != 1:
                raise LeaseLostError(LeaseLostError.code)
            cursor = connection.execute(
                """
                UPDATE campaign_attempts SET status = ?, claim_generation = 0,
                    lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND status = ?
                  AND lease_owner = ? AND claim_generation = ?
                """,
                (
                    AttemptStatus.SCHEDULED.value,
                    _iso(deferred_at),
                    attempt.workspace_id,
                    attempt.attempt_id,
                    AttemptStatus.RUNNING.value,
                    worker_id,
                    attempt.claim_generation,
                ),
            )
            if cursor.rowcount != 1:
                raise ActionClaimConflictError(ActionClaimConflictError.code)
            connection.execute(
                """
                UPDATE campaign_actions SET status = ?, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND action_id = ? AND status = ?
                """,
                (
                    ActionStatus.SCHEDULED.value,
                    _iso(deferred_at),
                    attempt.workspace_id,
                    attempt.action_id,
                    ActionStatus.RUNNING.value,
                ),
            )
            campaign_row = connection.execute(
                "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (attempt.workspace_id, attempt.campaign_id),
            ).fetchone()
            event_identity = f"capacity:{attempt.attempt_id}:{attempt.claim_generation}"
            self._insert_event(
                connection,
                CampaignEvent(
                    event_id=f"evt-{hashlib.sha256(event_identity.encode()).hexdigest()[:24]}",
                    workspace_id=attempt.workspace_id,
                    campaign_id=attempt.campaign_id,
                    sequence=self._next_event_sequence(
                        connection, attempt.workspace_id, attempt.campaign_id
                    ),
                    aggregate_version=int(campaign_row["version"]),
                    event_type="campaign:remote-capacity-blocked",
                    payload={
                        "action_id": attempt.action_id,
                        "attempt_id": attempt.attempt_id,
                        "claim_generation": attempt.claim_generation,
                        "reasons": list(reasons),
                    },
                    actor_id="campaign-controller",
                    credential_kind=CredentialKind.CONTROLLER,
                    correlation_id="worker-capacity-preflight",
                    idempotency_key=f"capacity-{attempt.attempt_id}-{attempt.claim_generation}",
                    created_at=deferred_at,
                ),
            )
            updated = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (attempt.workspace_id, attempt.attempt_id),
            ).fetchone()
        return self._attempt_from_row(updated)

    def update_remote_run(
        self,
        record: RemoteRunRecord,
        observation: RemoteObservation,
        *,
        metric_cursor: RemoteStreamCursor | None = None,
        log_cursor: RemoteStreamCursor | None = None,
        worker_id: str | None = None,
        lease_ttl: timedelta | None = None,
        now: datetime | None = None,
    ) -> RemoteRunRecord:
        """CAS-update observation and append-only cursors for the exact identity."""

        observed_at = now or utc_now()
        if observation.identity != record.identity:
            raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM campaign_remote_runs WHERE workspace_id = ? AND attempt_id = ?",
                (record.workspace_id, record.attempt_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign remote run not found")
            current = self._remote_run_from_row(row)
            if current.identity != record.identity or current.updated_at != record.updated_at:
                raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
            next_metric = metric_cursor or current.metric_cursor
            next_log = log_cursor or current.log_cursor
            if (
                next_metric.byte_offset < current.metric_cursor.byte_offset
                or next_log.byte_offset < current.log_cursor.byte_offset
            ):
                raise CampaignPersistenceError("campaign_remote_cursor_regression")
            if (worker_id is None) != (lease_ttl is None):
                raise ValueError("worker_id and lease_ttl must be provided together")
            if worker_id is not None and lease_ttl is not None:
                attempt_row = connection.execute(
                    self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                    (record.workspace_id, record.attempt_id),
                ).fetchone()
                if attempt_row is None:
                    raise RecordNotFoundError("campaign attempt not found")
                attempt = self._attempt_from_row(attempt_row)
                lease_key = f"action:{attempt.action_id}"
                expires_at = observed_at + lease_ttl
                lease = connection.execute(
                    """
                    UPDATE campaign_scheduler_leases SET expires_at = ?, heartbeat_at = ?
                    WHERE lease_key = ? AND owner_id = ? AND generation = ? AND expires_at > ?
                    """,
                    (
                        _iso(expires_at),
                        _iso(observed_at),
                        lease_key,
                        worker_id,
                        record.claim_generation,
                        _iso(observed_at),
                    ),
                )
                if lease.rowcount != 1:
                    raise LeaseLostError(LeaseLostError.code)
                attempt_cursor = connection.execute(
                    """
                    UPDATE campaign_attempts SET lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                    WHERE workspace_id = ? AND attempt_id = ? AND lease_owner = ?
                      AND claim_generation = ? AND status = ?
                    """,
                    (
                        _iso(expires_at),
                        _iso(observed_at),
                        _iso(observed_at),
                        record.workspace_id,
                        record.attempt_id,
                        worker_id,
                        record.claim_generation,
                        AttemptStatus.RUNNING.value,
                    ),
                )
                if attempt_cursor.rowcount != 1:
                    raise LeaseLostError(LeaseLostError.code)
            connection.execute(
                """
                UPDATE campaign_remote_runs SET state = ?, metric_cursor_json = ?,
                    log_cursor_json = ?, last_observation_json = ?, updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND updated_at = ?
                """,
                (
                    observation.state.value,
                    _json(next_metric.model_dump(mode="json")),
                    _json(next_log.model_dump(mode="json")),
                    _json(observation.model_dump(mode="json")),
                    _iso(observed_at),
                    record.workspace_id,
                    record.attempt_id,
                    _iso(record.updated_at),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM campaign_remote_runs WHERE workspace_id = ? AND attempt_id = ?",
                (record.workspace_id, record.attempt_id),
            ).fetchone()
        return self._remote_run_from_row(updated)

    def schedule_action_under_leader(
        self,
        spec: ActionSpec,
        leader: LeaseRecord,
        *,
        expected_campaign_version: int,
        now: datetime | None = None,
    ) -> ActionAttempt:
        """Atomically reserve budget, insert action/attempt, update state, and emit."""

        self._require_initialized()
        scheduled_at = now or utc_now()
        action_id = f"action-{spec.action_key[:24]}"
        attempt_id = f"attempt-{spec.action_key[:24]}-1"
        with self._connection(immediate=True) as connection:
            self._require_leader(connection, leader, scheduled_at)
            existing = connection.execute(
                self._attempt_select() + " WHERE a.workspace_id = ? AND a.action_key = ?",
                (spec.workspace_id, spec.action_key),
            ).fetchone()
            if existing is not None:
                return self._attempt_from_row(existing)
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (spec.workspace_id, spec.campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            campaign = self._campaign_from_row(campaign_row)
            if campaign.version != expected_campaign_version:
                raise RevisionConflictError(expected_campaign_version, campaign.version)
            if campaign.status != CampaignStatus.ACTIVE or campaign.active_action_id is not None:
                raise CampaignPersistenceError("campaign_not_schedulable")
            study_row = connection.execute(
                """
                SELECT * FROM campaign_studies
                WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
                """,
                (spec.workspace_id, spec.campaign_id, spec.study_id),
            ).fetchone()
            if study_row is None:
                raise RecordNotFoundError("campaign study not found")
            if int(study_row["current_stage_index"]) != spec.stage_index:
                raise CampaignPersistenceError("campaign_stage_cursor_mismatch")
            stage_plan = StagePlan.model_validate_json(study_row["stage_plan_json"])
            if spec.stage_index >= len(stage_plan.items):
                raise CampaignPersistenceError("campaign_stage_cursor_exhausted")
            item = stage_plan.items[spec.stage_index]
            if item.stage != spec.stage or item.disposition.value != "required":
                raise CampaignPersistenceError("campaign_stage_plan_mismatch")
            manifest_row = connection.execute(
                """
                SELECT manifest_json FROM campaign_manifest_revisions
                WHERE workspace_id = ? AND campaign_id = ? AND revision = ?
                """,
                (spec.workspace_id, spec.campaign_id, spec.manifest_revision),
            ).fetchone()
            if manifest_row is None:
                raise RecordNotFoundError("campaign manifest revision not found")
            manifest = CampaignManifest.model_validate_json(manifest_row["manifest_json"])
            base_limit = manifest.budget_limits.get(spec.budget_unit)
            if base_limit is None:
                raise CampaignPersistenceError("campaign_budget_unit_not_approved")
            totals = connection.execute(
                """
                SELECT COALESCE(SUM(reserved_delta), 0) AS reserved,
                       COALESCE(SUM(actual_delta), 0) AS actual,
                       COALESCE(SUM(limit_delta), 0) AS limit_delta
                FROM campaign_budget_ledger
                WHERE workspace_id = ? AND campaign_id = ? AND unit = ?
                """,
                (spec.workspace_id, spec.campaign_id, spec.budget_unit),
            ).fetchone()
            effective_limit = float(base_limit) + float(totals["limit_delta"])
            if (
                float(totals["reserved"]) + float(totals["actual"]) + spec.budget_reservation
                > effective_limit
            ):
                raise BudgetExceededError(BudgetExceededError.code)
            reservation = BudgetLedgerEntry(
                entry_id=f"budget-reserve-{action_id}",
                workspace_id=spec.workspace_id,
                campaign_id=spec.campaign_id,
                unit=spec.budget_unit,
                kind=BudgetEntryKind.RESERVE,
                reserved_delta=spec.budget_reservation,
                action_id=action_id,
                evidence={"action_key": spec.action_key},
                actor_id="campaign-controller",
                created_at=scheduled_at,
            )
            connection.execute(
                """
                INSERT INTO campaign_budget_ledger(
                    workspace_id, campaign_id, entry_id, unit, entry_kind,
                    reserved_delta, actual_delta, limit_delta, action_id,
                    evidence_json, actor_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?)
                """,
                (
                    reservation.workspace_id,
                    reservation.campaign_id,
                    reservation.entry_id,
                    reservation.unit,
                    reservation.kind.value,
                    reservation.reserved_delta,
                    action_id,
                    _json(reservation.evidence),
                    reservation.actor_id,
                    _iso(scheduled_at),
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_actions(
                    workspace_id, campaign_id, study_id, action_id, stage_index,
                    stage_kind, input_digest, candidate_digest, manifest_revision,
                    action_key, reservation_json, status, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    spec.workspace_id,
                    spec.campaign_id,
                    spec.study_id,
                    action_id,
                    spec.stage_index,
                    spec.stage.value,
                    spec.input_digest,
                    spec.candidate_digest,
                    spec.manifest_revision,
                    spec.action_key,
                    _json({"unit": spec.budget_unit, "amount": spec.budget_reservation}),
                    ActionStatus.SCHEDULED.value,
                    _iso(scheduled_at),
                    _iso(scheduled_at),
                ),
            )
            connection.execute(
                """
                INSERT INTO campaign_attempts(
                    workspace_id, action_id, attempt_id, attempt_number,
                    claim_generation, status, executor_json, created_at, updated_at
                ) VALUES (?, ?, ?, 1, 0, ?, ?, ?, ?)
                """,
                (
                    spec.workspace_id,
                    action_id,
                    attempt_id,
                    AttemptStatus.SCHEDULED.value,
                    _json(
                        {
                            "kind": spec.executor_kind,
                            **spec.executor_config,
                            **({"steps": spec.fake_steps} if spec.executor_kind == "fake" else {}),
                        }
                    ),
                    _iso(scheduled_at),
                    _iso(scheduled_at),
                ),
            )
            connection.execute(
                """
                UPDATE campaign_studies SET status = ?, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND study_id = ?
                """,
                (
                    _study_status_for_stage(spec.stage).value,
                    _iso(scheduled_at),
                    spec.workspace_id,
                    spec.study_id,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE campaigns SET active_study_id = ?, active_action_id = ?,
                    version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ? AND status = ?
                """,
                (
                    spec.study_id,
                    action_id,
                    _iso(scheduled_at),
                    spec.workspace_id,
                    spec.campaign_id,
                    expected_campaign_version,
                    CampaignStatus.ACTIVE.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RevisionConflictError(expected_campaign_version, campaign.version)
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'schedule:{action_id}'.encode()).hexdigest()[:24]}",
                workspace_id=spec.workspace_id,
                campaign_id=spec.campaign_id,
                sequence=self._next_event_sequence(connection, spec.workspace_id, spec.campaign_id),
                aggregate_version=campaign.version + 1,
                event_type="campaign:action-scheduled",
                payload={
                    "action_id": action_id,
                    "attempt_id": attempt_id,
                    "study_id": spec.study_id,
                    "stage": spec.stage.value,
                },
                actor_id="campaign-controller",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=f"worker-{leader.owner_id}",
                idempotency_key=f"schedule-{spec.action_key[:32]}",
                created_at=scheduled_at,
            )
            self._insert_event(connection, event)
            row = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (spec.workspace_id, attempt_id),
            ).fetchone()
        return self._attempt_from_row(row)

    def claim_next_action(
        self,
        leader: LeaseRecord,
        *,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> ActionAttempt | None:
        """Claim one scheduled action with a monotonic action fencing generation."""

        claimed_at = now or utc_now()
        expires_at = claimed_at + ttl
        with self._connection(immediate=True) as connection:
            self._require_leader(connection, leader, claimed_at)
            row = connection.execute(
                self._attempt_select()
                + """
                  JOIN campaigns c ON c.workspace_id = a.workspace_id AND c.campaign_id = a.campaign_id
                  WHERE a.status = ? AND t.status = ? AND c.status = ?
                  ORDER BY a.created_at, a.action_id LIMIT 1
                  """,
                (
                    ActionStatus.SCHEDULED.value,
                    AttemptStatus.SCHEDULED.value,
                    CampaignStatus.ACTIVE.value,
                ),
            ).fetchone()
            if row is None:
                return None
            lease_key = f"action:{row['action_id']}"
            lease = connection.execute(
                "SELECT * FROM campaign_scheduler_leases WHERE lease_key = ?",
                (lease_key,),
            ).fetchone()
            if lease is None:
                generation = 1
                connection.execute(
                    """
                    INSERT INTO campaign_scheduler_leases(
                        lease_key, owner_id, generation, expires_at, heartbeat_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (lease_key, leader.owner_id, generation, _iso(expires_at), _iso(claimed_at)),
                )
            elif _dt(lease["expires_at"]) <= claimed_at:
                generation = int(lease["generation"]) + 1
                connection.execute(
                    """
                    UPDATE campaign_scheduler_leases
                    SET owner_id = ?, generation = ?, expires_at = ?, heartbeat_at = ?
                    WHERE lease_key = ?
                    """,
                    (leader.owner_id, generation, _iso(expires_at), _iso(claimed_at), lease_key),
                )
            else:
                raise LeaseBusyError(LeaseBusyError.code)
            attempt_cursor = connection.execute(
                """
                UPDATE campaign_attempts SET status = ?, claim_generation = ?,
                    lease_owner = ?, lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND status = ? AND claim_generation = 0
                """,
                (
                    AttemptStatus.RUNNING.value,
                    generation,
                    leader.owner_id,
                    _iso(expires_at),
                    _iso(claimed_at),
                    _iso(claimed_at),
                    row["workspace_id"],
                    row["attempt_id"],
                    AttemptStatus.SCHEDULED.value,
                ),
            )
            if attempt_cursor.rowcount != 1:
                raise ActionClaimConflictError(ActionClaimConflictError.code)
            connection.execute(
                """
                UPDATE campaign_actions SET status = ?, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND action_id = ? AND status = ?
                """,
                (
                    ActionStatus.RUNNING.value,
                    _iso(claimed_at),
                    row["workspace_id"],
                    row["action_id"],
                    ActionStatus.SCHEDULED.value,
                ),
            )
            campaign_row = connection.execute(
                "SELECT version, campaign_id FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (row["workspace_id"], row["campaign_id"]),
            ).fetchone()
            claim_identity = f"claim:{row['attempt_id']}:{generation}"
            self._insert_event(
                connection,
                CampaignEvent(
                    event_id=f"evt-{hashlib.sha256(claim_identity.encode()).hexdigest()[:24]}",
                    workspace_id=row["workspace_id"],
                    campaign_id=row["campaign_id"],
                    sequence=self._next_event_sequence(
                        connection, row["workspace_id"], row["campaign_id"]
                    ),
                    aggregate_version=int(campaign_row["version"]),
                    event_type="campaign:action-claimed",
                    payload={
                        "action_id": row["action_id"],
                        "attempt_id": row["attempt_id"],
                        "claim_generation": generation,
                    },
                    actor_id="campaign-controller",
                    credential_kind=CredentialKind.CONTROLLER,
                    correlation_id=f"worker-{leader.owner_id}",
                    idempotency_key=f"claim-{row['attempt_id']}-{generation}",
                    created_at=claimed_at,
                ),
            )
            claimed = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (row["workspace_id"], row["attempt_id"]),
            ).fetchone()
        return self._attempt_from_row(claimed)

    def list_unfinished_attempts(self) -> list[ActionAttempt]:
        self._require_initialized()
        with self._connection() as connection:
            rows = connection.execute(
                self._attempt_select()
                + " WHERE t.status IN (?, ?) ORDER BY t.updated_at, t.attempt_id",
                (AttemptStatus.RUNNING.value, AttemptStatus.UNKNOWN.value),
            ).fetchall()
        return [self._attempt_from_row(row) for row in rows]

    def mark_expired_unknown(self, attempt: ActionAttempt, *, now: datetime) -> ActionAttempt:
        """Preserve uncertain work and its reservation instead of retrying it."""

        if attempt.lease_expires_at is None or attempt.lease_expires_at > now:
            return attempt
        with self._connection(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE campaign_attempts SET status = ?, updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND status = ?
                """,
                (
                    AttemptStatus.UNKNOWN.value,
                    _iso(now),
                    attempt.workspace_id,
                    attempt.attempt_id,
                    AttemptStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount == 1:
                connection.execute(
                    """
                    UPDATE campaign_actions SET status = ?, version = version + 1, updated_at = ?
                    WHERE workspace_id = ? AND action_id = ? AND status = ?
                    """,
                    (
                        ActionStatus.UNKNOWN.value,
                        _iso(now),
                        attempt.workspace_id,
                        attempt.action_id,
                        ActionStatus.RUNNING.value,
                    ),
                )
                campaign_row = connection.execute(
                    "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                    (attempt.workspace_id, attempt.campaign_id),
                ).fetchone()
                self._insert_event(
                    connection,
                    CampaignEvent(
                        event_id=f"evt-{hashlib.sha256(f'unknown:{attempt.attempt_id}'.encode()).hexdigest()[:24]}",
                        workspace_id=attempt.workspace_id,
                        campaign_id=attempt.campaign_id,
                        sequence=self._next_event_sequence(
                            connection, attempt.workspace_id, attempt.campaign_id
                        ),
                        aggregate_version=int(campaign_row["version"]),
                        event_type="campaign:action-unknown",
                        payload={
                            "action_id": attempt.action_id,
                            "attempt_id": attempt.attempt_id,
                        },
                        actor_id="campaign-controller",
                        credential_kind=CredentialKind.CONTROLLER,
                        correlation_id="worker-reconciliation",
                        idempotency_key=f"unknown-{attempt.attempt_id}",
                        created_at=now,
                    ),
                )
            updated = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (attempt.workspace_id, attempt.attempt_id),
            ).fetchone()
        return self._attempt_from_row(updated)

    def settle_terminal_from_seal(
        self,
        manifest: SealedActionResult,
        sealed_directory: Path,
        *,
        worker_id: str,
        now: datetime | None = None,
    ) -> RuntimeCompletion:
        """Atomically preserve terminal evidence, settle budget, and release the action."""

        settled_at = now or utc_now()
        if manifest.outcome == "completed":
            raise ValueError("terminal settlement cannot accept a completed manifest")
        status_map = {
            "failed": (AttemptStatus.FAILED, ActionStatus.FAILED),
            "cancelled": (AttemptStatus.CANCELLED, ActionStatus.CANCELLED),
            "force_stopped": (AttemptStatus.FORCE_STOPPED, ActionStatus.FORCE_STOPPED),
        }
        attempt_status, action_status = status_map[manifest.outcome]
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (manifest.workspace_id, manifest.attempt_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign attempt not found")
            attempt = self._attempt_from_row(row)
            expected = (
                attempt.workspace_id,
                attempt.campaign_id,
                attempt.study_id,
                attempt.action_id,
                attempt.attempt_id,
                attempt.manifest_revision,
                attempt.candidate_digest,
                attempt.input_digest,
                attempt.claim_generation,
            )
            actual = (
                manifest.workspace_id,
                manifest.campaign_id,
                manifest.study_id,
                manifest.action_id,
                manifest.attempt_id,
                manifest.manifest_revision,
                manifest.candidate_digest,
                manifest.input_digest,
                manifest.claim_generation,
            )
            if actual != expected:
                raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
            if attempt.status == attempt_status:
                event_row = connection.execute(
                    """
                    SELECT * FROM campaign_events
                    WHERE workspace_id = ? AND campaign_id = ?
                      AND json_extract(payload_json, '$.attempt_id') = ?
                    ORDER BY cursor DESC LIMIT 1
                    """,
                    (manifest.workspace_id, manifest.campaign_id, manifest.attempt_id),
                ).fetchone()
                campaign_row = connection.execute(
                    "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                    (manifest.workspace_id, manifest.campaign_id),
                ).fetchone()
                return RuntimeCompletion(
                    attempt,
                    int(campaign_row["version"]),
                    self._event_from_row(event_row),
                    replayed=True,
                )
            if attempt.status not in {AttemptStatus.RUNNING, AttemptStatus.UNKNOWN}:
                raise ActionClaimConflictError(ActionClaimConflictError.code)
            lease = connection.execute(
                """
                SELECT 1 FROM campaign_scheduler_leases
                WHERE lease_key = ? AND owner_id = ? AND generation = ? AND expires_at > ?
                """,
                (
                    f"action:{manifest.action_id}",
                    worker_id,
                    manifest.claim_generation,
                    _iso(settled_at),
                ),
            ).fetchone()
            if attempt.lease_owner != worker_id or lease is None:
                raise LeaseLostError(LeaseLostError.code)
            action_row = connection.execute(
                "SELECT * FROM campaign_actions WHERE workspace_id = ? AND action_id = ?",
                (manifest.workspace_id, manifest.action_id),
            ).fetchone()
            reservation = json.loads(action_row["reservation_json"])
            for output in manifest.outputs:
                artifact_id = f"artifact-{hashlib.sha256(f'{manifest.attempt_id}:{output.path}'.encode()).hexdigest()[:24]}"
                connection.execute(
                    """
                    INSERT OR IGNORE INTO campaign_artifacts(
                        workspace_id, campaign_id, artifact_id, producer_action_id,
                        uri, sha256, size_bytes, schema_name, sealed, valid,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                    """,
                    (
                        manifest.workspace_id,
                        manifest.campaign_id,
                        artifact_id,
                        manifest.action_id,
                        str(sealed_directory / output.path),
                        output.sha256,
                        output.size_bytes,
                        output.schema_name,
                        _json(
                            {
                                "attempt_id": manifest.attempt_id,
                                "outcome": manifest.outcome,
                            }
                        ),
                        _iso(settled_at),
                    ),
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO campaign_budget_ledger(
                    workspace_id, campaign_id, entry_id, unit, entry_kind,
                    reserved_delta, actual_delta, limit_delta, action_id,
                    evidence_json, actor_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    manifest.workspace_id,
                    manifest.campaign_id,
                    (
                        f"budget-terminal-{manifest.action_id}"
                        if attempt.attempt_number == 1
                        else f"budget-terminal-{manifest.action_id}-attempt-{attempt.attempt_number}"
                    ),
                    reservation["unit"],
                    BudgetEntryKind.SETTLE.value,
                    -float(reservation["amount"]),
                    float(reservation["amount"]),
                    manifest.action_id,
                    _json(
                        {
                            "seal_uri": str(sealed_directory),
                            "outcome": manifest.outcome,
                            "resource_usage": [
                                item.model_dump(mode="json") for item in manifest.resource_usage
                            ],
                        }
                    ),
                    "campaign-controller",
                    _iso(settled_at),
                ),
            )
            connection.execute(
                """
                UPDATE campaign_attempts SET status = ?, result_json = ?, updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND claim_generation = ?
                """,
                (
                    attempt_status.value,
                    _json(manifest.model_dump(mode="json")),
                    _iso(settled_at),
                    manifest.workspace_id,
                    manifest.attempt_id,
                    manifest.claim_generation,
                ),
            )
            connection.execute(
                """
                UPDATE campaign_actions SET status = ?, sealed_result_uri = ?,
                    version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND action_id = ?
                """,
                (
                    action_status.value,
                    str(sealed_directory),
                    _iso(settled_at),
                    manifest.workspace_id,
                    manifest.action_id,
                ),
            )
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (manifest.workspace_id, manifest.campaign_id),
            ).fetchone()
            campaign = self._campaign_from_row(campaign_row)
            cancellation_settled = campaign.status == CampaignStatus.CANCELLING
            study_status = (
                StudyStatus.CANCELLED
                if cancellation_settled or manifest.outcome != "failed"
                else StudyStatus.EXECUTION_FAILED
            )
            connection.execute(
                """
                UPDATE campaign_studies SET status = ?, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND study_id = ?
                """,
                (
                    study_status.value,
                    _iso(settled_at),
                    manifest.workspace_id,
                    manifest.study_id,
                ),
            )
            next_campaign_status = (
                CampaignStatus.CANCELLED if cancellation_settled else campaign.status
            )
            connection.execute(
                """
                UPDATE campaigns SET status = ?, active_action_id = NULL,
                    active_study_id = NULL, version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    next_campaign_status.value,
                    _iso(settled_at),
                    manifest.workspace_id,
                    manifest.campaign_id,
                    campaign.version,
                ),
            )
            connection.execute(
                """
                UPDATE campaign_scheduler_leases SET expires_at = ?, heartbeat_at = ?
                WHERE lease_key = ? AND generation = ?
                """,
                (
                    _iso(settled_at),
                    _iso(settled_at),
                    f"action:{manifest.action_id}",
                    manifest.claim_generation,
                ),
            )
            event_type = (
                "campaign:cancelled"
                if cancellation_settled
                else f"campaign:action-{manifest.outcome.replace('_', '-')}"
            )
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'terminal:{manifest.attempt_id}:{manifest.outcome}'.encode()).hexdigest()[:24]}",
                workspace_id=manifest.workspace_id,
                campaign_id=manifest.campaign_id,
                sequence=self._next_event_sequence(
                    connection, manifest.workspace_id, manifest.campaign_id
                ),
                aggregate_version=campaign.version + 1,
                event_type=event_type,
                payload={
                    "action_id": manifest.action_id,
                    "attempt_id": manifest.attempt_id,
                    "study_id": manifest.study_id,
                    "stage": attempt.stage.value,
                    "outcome": manifest.outcome,
                    "exit_reason": manifest.exit_reason,
                },
                actor_id="campaign-controller",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=f"worker-{worker_id}",
                idempotency_key=f"terminal-{manifest.attempt_id}-{manifest.outcome}",
                created_at=settled_at,
            )
            self._insert_event(connection, event)
            updated = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (manifest.workspace_id, manifest.attempt_id),
            ).fetchone()
        return RuntimeCompletion(self._attempt_from_row(updated), campaign.version + 1, event)

    def complete_from_seal(
        self,
        manifest: SealedActionResult,
        sealed_directory: Path,
        *,
        worker_id: str,
        reconcile: bool = False,
        now: datetime | None = None,
    ) -> RuntimeCompletion:
        """Register artifacts, settle budget, and advance state exactly once."""

        completed_at = now or utc_now()
        with self._connection(immediate=True) as connection:
            row = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (manifest.workspace_id, manifest.attempt_id),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("campaign attempt not found")
            attempt = self._attempt_from_row(row)
            expected = (
                attempt.workspace_id,
                attempt.campaign_id,
                attempt.study_id,
                attempt.action_id,
                attempt.attempt_id,
                attempt.manifest_revision,
                attempt.candidate_digest,
                attempt.input_digest,
                attempt.claim_generation,
            )
            actual = (
                manifest.workspace_id,
                manifest.campaign_id,
                manifest.study_id,
                manifest.action_id,
                manifest.attempt_id,
                manifest.manifest_revision,
                manifest.candidate_digest,
                manifest.input_digest,
                manifest.claim_generation,
            )
            if actual != expected:
                raise ActionIdentityMismatchError(ActionIdentityMismatchError.code)
            if attempt.status == AttemptStatus.COMPLETED:
                event_row = connection.execute(
                    """
                    SELECT * FROM campaign_events
                    WHERE workspace_id = ? AND campaign_id = ? AND event_type = ?
                      AND json_extract(payload_json, '$.attempt_id') = ?
                    ORDER BY cursor DESC LIMIT 1
                    """,
                    (
                        manifest.workspace_id,
                        manifest.campaign_id,
                        "campaign:action-completed",
                        manifest.attempt_id,
                    ),
                ).fetchone()
                campaign_row = connection.execute(
                    "SELECT version FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                    (manifest.workspace_id, manifest.campaign_id),
                ).fetchone()
                return RuntimeCompletion(
                    attempt,
                    int(campaign_row["version"]),
                    self._event_from_row(event_row),
                    replayed=True,
                )
            if attempt.status not in {AttemptStatus.RUNNING, AttemptStatus.UNKNOWN}:
                raise ActionClaimConflictError(ActionClaimConflictError.code)
            if not reconcile:
                lease = connection.execute(
                    """
                    SELECT 1 FROM campaign_scheduler_leases
                    WHERE lease_key = ? AND owner_id = ? AND generation = ? AND expires_at > ?
                    """,
                    (
                        f"action:{manifest.action_id}",
                        worker_id,
                        manifest.claim_generation,
                        _iso(completed_at),
                    ),
                ).fetchone()
                if attempt.lease_owner != worker_id or lease is None:
                    raise LeaseLostError(LeaseLostError.code)
            action_row = connection.execute(
                "SELECT * FROM campaign_actions WHERE workspace_id = ? AND action_id = ?",
                (manifest.workspace_id, manifest.action_id),
            ).fetchone()
            reservation = json.loads(action_row["reservation_json"])
            for output in manifest.outputs:
                artifact_id = f"artifact-{hashlib.sha256(f'{manifest.attempt_id}:{output.path}'.encode()).hexdigest()[:24]}"
                metadata: dict[str, Any] = {"attempt_id": manifest.attempt_id}
                if output.schema_name == NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA:
                    evidence = load_nemo_gym_campaign_evidence(
                        sealed_directory / output.path,
                        expected_attempt=attempt,
                    )
                    metadata["nemo_gym"] = evidence.bounded_reference(
                        artifact_id=artifact_id,
                        artifact_sha256=output.sha256,
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO campaign_artifacts(
                        workspace_id, campaign_id, artifact_id, producer_action_id,
                        uri, sha256, size_bytes, schema_name, sealed, valid,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                    """,
                    (
                        manifest.workspace_id,
                        manifest.campaign_id,
                        artifact_id,
                        manifest.action_id,
                        str(sealed_directory / output.path),
                        output.sha256,
                        output.size_bytes,
                        output.schema_name,
                        _json(metadata),
                        _iso(completed_at),
                    ),
                )
            settlement = BudgetLedgerEntry(
                entry_id=(
                    f"budget-settle-{manifest.action_id}"
                    if attempt.attempt_number == 1
                    else f"budget-settle-{manifest.action_id}-attempt-{attempt.attempt_number}"
                ),
                workspace_id=manifest.workspace_id,
                campaign_id=manifest.campaign_id,
                unit=reservation["unit"],
                kind=BudgetEntryKind.SETTLE,
                reserved_delta=-float(reservation["amount"]),
                actual_delta=float(reservation["amount"]),
                action_id=manifest.action_id,
                evidence={
                    "seal_uri": str(sealed_directory),
                    "resource_usage": [
                        item.model_dump(mode="json") for item in manifest.resource_usage
                    ],
                },
                actor_id="campaign-controller",
                created_at=completed_at,
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO campaign_budget_ledger(
                    workspace_id, campaign_id, entry_id, unit, entry_kind,
                    reserved_delta, actual_delta, limit_delta, action_id,
                    evidence_json, actor_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    settlement.workspace_id,
                    settlement.campaign_id,
                    settlement.entry_id,
                    settlement.unit,
                    settlement.kind.value,
                    settlement.reserved_delta,
                    settlement.actual_delta,
                    settlement.action_id,
                    _json(settlement.evidence),
                    settlement.actor_id,
                    _iso(completed_at),
                ),
            )
            connection.execute(
                """
                UPDATE campaign_attempts SET status = ?, result_json = ?, updated_at = ?
                WHERE workspace_id = ? AND attempt_id = ? AND claim_generation = ?
                """,
                (
                    AttemptStatus.COMPLETED.value,
                    _json(manifest.model_dump(mode="json")),
                    _iso(completed_at),
                    manifest.workspace_id,
                    manifest.attempt_id,
                    manifest.claim_generation,
                ),
            )
            connection.execute(
                """
                UPDATE campaign_actions SET status = ?, sealed_result_uri = ?,
                    version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND action_id = ?
                """,
                (
                    ActionStatus.COMPLETED.value,
                    str(sealed_directory),
                    _iso(completed_at),
                    manifest.workspace_id,
                    manifest.action_id,
                ),
            )
            study_row = connection.execute(
                "SELECT * FROM campaign_studies WHERE workspace_id = ? AND study_id = ?",
                (manifest.workspace_id, manifest.study_id),
            ).fetchone()
            plan = StagePlan.model_validate_json(study_row["stage_plan_json"])
            next_index = int(study_row["current_stage_index"]) + 1
            finished = next_index >= len(plan.items)
            next_status = (
                StudyStatus.COMPLETED
                if finished
                else _study_status_for_stage(plan.items[next_index].stage)
            )
            connection.execute(
                """
                UPDATE campaign_studies SET status = ?, current_stage_index = ?,
                    version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND study_id = ?
                """,
                (
                    next_status.value,
                    next_index,
                    _iso(completed_at),
                    manifest.workspace_id,
                    manifest.study_id,
                ),
            )
            campaign_row = connection.execute(
                "SELECT * FROM campaigns WHERE workspace_id = ? AND campaign_id = ?",
                (manifest.workspace_id, manifest.campaign_id),
            ).fetchone()
            campaign = self._campaign_from_row(campaign_row)
            connection.execute(
                """
                UPDATE campaigns SET active_action_id = NULL, active_study_id = ?,
                    version = version + 1, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    None if finished else manifest.study_id,
                    _iso(completed_at),
                    manifest.workspace_id,
                    manifest.campaign_id,
                    campaign.version,
                ),
            )
            lease_key = f"action:{manifest.action_id}"
            connection.execute(
                """
                UPDATE campaign_scheduler_leases SET expires_at = ?, heartbeat_at = ?
                WHERE lease_key = ? AND generation = ?
                """,
                (_iso(completed_at), _iso(completed_at), lease_key, manifest.claim_generation),
            )
            event = CampaignEvent(
                event_id=f"evt-{hashlib.sha256(f'complete:{manifest.attempt_id}'.encode()).hexdigest()[:24]}",
                workspace_id=manifest.workspace_id,
                campaign_id=manifest.campaign_id,
                sequence=self._next_event_sequence(
                    connection, manifest.workspace_id, manifest.campaign_id
                ),
                aggregate_version=campaign.version + 1,
                event_type="campaign:action-completed",
                payload={
                    "action_id": manifest.action_id,
                    "attempt_id": manifest.attempt_id,
                    "study_id": manifest.study_id,
                    "stage": attempt.stage.value,
                    "sealed_result_uri": str(sealed_directory),
                },
                actor_id="campaign-controller",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=f"worker-{worker_id}",
                idempotency_key=f"complete-{manifest.attempt_id}",
                created_at=completed_at,
            )
            self._insert_event(connection, event)
            completed = connection.execute(
                self._attempt_select() + " WHERE t.workspace_id = ? AND t.attempt_id = ?",
                (manifest.workspace_id, manifest.attempt_id),
            ).fetchone()
        return RuntimeCompletion(self._attempt_from_row(completed), campaign.version + 1, event)


__all__ = [
    "ActionClaimConflictError",
    "ActionIdentityMismatchError",
    "ActionSpec",
    "CampaignRuntimeRepository",
    "CampaignArtifactRecord",
    "RuntimeCompletion",
    "RemoteRunRecord",
]
