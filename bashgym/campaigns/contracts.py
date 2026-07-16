"""Stable, secret-free contracts for durable experiment campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
HexDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitObjectId = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
]


class ContractModel(BaseModel):
    """Base for mutable public contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenContractModel(BaseModel):
    """Base for immutable evidence contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value with a deterministic representation."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CampaignStatus(str, Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    AWAITING_AUTHORITY = "awaiting_authority"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_CAMPAIGN_STATES = frozenset(
    {
        CampaignStatus.COMPLETED,
        CampaignStatus.EXHAUSTED,
        CampaignStatus.FAILED,
        CampaignStatus.CANCELLED,
    }
)


class CampaignTrigger(str, Enum):
    VALIDATE = "validate"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    AUTHORITY_MISSING = "authority_missing"
    AUTHORITY_SATISFIED = "authority_satisfied"
    STOPPING_RULE_MET = "stopping_rule_met"
    CONCLUDE = "conclude"
    PROMOTION_COMMITTED = "promotion_committed"
    INVARIANT_FAILURE = "invariant_failure"
    CANCEL = "cancel"
    CANCELLATION_SETTLED = "cancellation_settled"


class CampaignKind(str, Enum):
    EMBEDDING_RETRIEVAL = "embedding_retrieval"
    GENERAL = "general"


class ProposalStatus(str, Enum):
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class StudyStatus(str, Enum):
    VALIDATED = "validated"
    DATA_BUILDING = "data_building"
    CONTRACT_EVALUATING = "contract_evaluating"
    SMOKE_TRAINING = "smoke_training"
    FULL_TRAINING = "full_training"
    DEVELOPMENT_EVALUATING = "development_evaluating"
    COMPARING = "comparing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    DEVELOPMENT_PASSED = "development_passed"
    RECIPE_LOCKED = "recipe_locked"
    PROTECTED_EVALUATING = "protected_evaluating"
    PROMOTION_DECIDING = "promotion_deciding"
    PROMOTED = "promoted"
    FINAL_REJECTED = "final_rejected"
    EXECUTION_FAILED = "execution_failed"
    ABANDONED = "abandoned"
    CANCELLED = "cancelled"


class StageKind(str, Enum):
    DATA_BUILD = "data_build"
    CONTRACT_EVALUATION = "contract_evaluation"
    SMOKE_TRAINING = "smoke_training"
    FULL_TRAINING = "full_training"
    DEVELOPMENT_EVALUATION = "development_evaluation"
    COMPARISON = "comparison"
    RECIPE_LOCK = "recipe_lock"
    PROTECTED_EVALUATION = "protected_evaluation"
    PROMOTION = "promotion"


class StageDisposition(str, Enum):
    REQUIRED = "required"
    NOT_APPLICABLE = "not_applicable"


class ActionStatus(str, Enum):
    SCHEDULED = "scheduled"
    CLAIMED = "claimed"
    RUNNING = "running"
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    FAILED = "failed"
    FORCE_STOPPED = "force_stopped"
    CANCELLED = "cancelled"


class AttemptStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    FAILED = "failed"
    FORCE_STOPPED = "force_stopped"
    CANCELLED = "cancelled"


class BudgetEntryKind(str, Enum):
    RESERVE = "reserve"
    SETTLE = "settle"
    RELEASE = "release"
    AMEND = "amend"
    CORRECTION = "correction"


class CodeMutationKind(str, Enum):
    """Operator-approved experiment-code surface changed by a hypothesis."""

    TRAINER = "trainer"
    GYM = "gym"
    REWARD = "reward"
    EVALUATOR = "evaluator"


class CodeLineageState(str, Enum):
    """Monotonic lifecycle for one isolated hypothesis branch."""

    REQUIRED = "required"
    PREPARED = "prepared"
    CAPTURED = "captured"


class AutonomyProfile(str, Enum):
    DESKTOP_USER = "desktop_user"
    HERMES_BOUNDED = "hermes_bounded"
    CODEX_TRUSTED = "codex_trusted"


class CredentialKind(str, Enum):
    DESKTOP_BOOTSTRAP = "desktop_bootstrap"
    REFRESH = "refresh"
    ACCESS = "access"
    CONTROLLER = "controller"


class Capability(str, Enum):
    CAMPAIGN_READ = "campaign.read"
    CAMPAIGN_CREATE_FROM_TEMPLATE = "campaign.create_from_template"
    CAMPAIGN_CREATE = "campaign.create"
    CAMPAIGN_REVISE = "campaign.revise"
    CAMPAIGN_START = "campaign.start"
    CAMPAIGN_PAUSE = "campaign.pause"
    CAMPAIGN_RESUME = "campaign.resume"
    CAMPAIGN_CANCEL = "campaign.cancel"
    CAMPAIGN_COMPLETE = "campaign.complete"
    STUDY_PROPOSE = "study.propose"
    STUDY_RETRY = "study.retry"
    STUDY_ABANDON = "study.abandon"
    DATA_USE_APPROVED = "data.use_approved"
    DATA_APPROVE_EXTERNAL = "data.approve_external"
    DATA_BUILD = "data.build"
    COMPUTE_SMOKE = "compute.smoke"
    COMPUTE_TRAIN_WITHIN_BUDGET = "compute.train_within_budget"
    COMPUTE_AMEND_BUDGET = "compute.amend_budget"
    COMPUTE_MANAGE_RESIDENT_SERVICES = "compute.manage_resident_services"
    COMPUTE_FORCE_STOP = "compute.force_stop"
    EVAL_DEVELOPMENT = "eval.development"
    EVAL_PROTECTED_ACQUIRE = "eval.protected_acquire"
    EVAL_PROTECTED_EXECUTE = "eval.protected_execute"
    EXPERIMENT_LEDGER_WRITE = "experiment.ledger_write"
    EXPERIMENT_CODE_MUTATE = "experiment.code_mutate"
    PROMOTION_DECIDE = "promotion.decide"
    PROMOTION_OVERRIDE = "promotion.override"
    ARTIFACT_PUBLISH_HF = "artifact.publish_hf"
    HANDOFF_MEMEXAI_PREPARE = "handoff.memexai_prepare"


HERMES_CAPABILITIES = frozenset(
    {
        Capability.CAMPAIGN_READ,
        Capability.CAMPAIGN_CREATE_FROM_TEMPLATE,
        Capability.CAMPAIGN_START,
        Capability.CAMPAIGN_PAUSE,
        Capability.CAMPAIGN_RESUME,
        Capability.CAMPAIGN_CANCEL,
        Capability.STUDY_PROPOSE,
        Capability.STUDY_RETRY,
        Capability.DATA_USE_APPROVED,
        Capability.DATA_BUILD,
        Capability.COMPUTE_SMOKE,
        Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
        Capability.EVAL_DEVELOPMENT,
        Capability.EXPERIMENT_LEDGER_WRITE,
    }
)
CODEX_CAPABILITIES = frozenset(Capability)
DESKTOP_LOCAL_SCOPE = "desktop-local"


class TargetModelContract(FrozenContractModel):
    """Identity and compatibility boundary for a model campaign."""

    schema_version: Literal["target_model.v1"] = "target_model.v1"
    target_contract_key: Identifier
    base_model_ref: str = Field(min_length=1, max_length=1000)
    task: str = Field(min_length=1, max_length=160)
    representation_contract: dict[str, Any] = Field(default_factory=dict)


class CampaignManifest(FrozenContractModel):
    """Immutable approved scope for one campaign revision."""

    schema_version: Literal["campaign_manifest.v1"] = "campaign_manifest.v1"
    approved_data_scopes: tuple[Identifier, ...]
    compute_profile_id: Identifier
    budget_limits: dict[Identifier, float]
    evaluation_plan: dict[str, Any]
    promotion_gates: dict[str, Any]
    protected_artifact_refs: tuple[str, ...] = ()
    max_proposal_rounds: int = Field(default=5, ge=1, le=100)
    retention_days_failed: int = Field(default=90, ge=1)
    allow_hf_publication: bool = False
    allow_memexai_handoff: bool = False

    @field_validator("approved_data_scopes")
    @classmethod
    def validate_data_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("approved_data_scopes must be non-empty, sorted, and unique")
        return value

    @field_validator("budget_limits")
    @classmethod
    def validate_budgets(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(amount < 0 for amount in value.values()):
            raise ValueError("budget_limits must be non-empty and non-negative")
        return value


class ManifestRevision(FrozenContractModel):
    schema_version: Literal["campaign_manifest_revision.v1"] = "campaign_manifest_revision.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    revision: int = Field(ge=1)
    manifest: CampaignManifest
    manifest_hash: HexDigest = ""
    actor_id: Identifier
    correlation_id: Identifier
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def verify_manifest_hash(self) -> ManifestRevision:
        expected = canonical_hash(self.manifest.model_dump(mode="json"))
        if self.manifest_hash and self.manifest_hash != expected:
            raise ValueError("manifest_hash does not match manifest")
        if not self.manifest_hash:
            object.__setattr__(self, "manifest_hash", expected)
        return self


class Campaign(ContractModel):
    schema_version: Literal["campaign.v1"] = "campaign.v1"
    campaign_id: Identifier
    workspace_id: Identifier
    title: str = Field(min_length=1, max_length=240)
    kind: CampaignKind
    objective: str = Field(min_length=1, max_length=4000)
    target_model: TargetModelContract
    owner_actor_id: Identifier
    manifest_revision: int = Field(default=1, ge=1)
    status: CampaignStatus = CampaignStatus.DRAFT
    prior_scheduling_status: Literal[CampaignStatus.ACTIVE, CampaignStatus.PAUSED] | None = None
    active_study_id: Identifier | None = None
    active_action_id: Identifier | None = None
    champion_ref: str | None = Field(default=None, max_length=2000)
    best_development_candidate_ref: str | None = Field(default=None, max_length=2000)
    stop_reason: str | None = Field(default=None, max_length=2000)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class StagePlanItem(FrozenContractModel):
    schema_version: Literal["campaign_stage_plan_item.v1"] = "campaign_stage_plan_item.v1"
    stage: StageKind
    disposition: StageDisposition
    reason: str = Field(min_length=1, max_length=2000)
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)


class StagePlan(FrozenContractModel):
    schema_version: Literal["campaign_stage_plan.v1"] = "campaign_stage_plan.v1"
    items: tuple[StagePlanItem, ...]

    @field_validator("items")
    @classmethod
    def validate_items(cls, value: tuple[StagePlanItem, ...]) -> tuple[StagePlanItem, ...]:
        if not value:
            raise ValueError("stage plan cannot be empty")
        stages = [item.stage for item in value]
        if len(set(stages)) != len(stages):
            raise ValueError("stage plan cannot repeat a stage")
        return value


class StudyProposalSubmission(FrozenContractModel):
    """Actor-authored proposal fields before server identity and queue sequencing."""

    schema_version: Literal["campaign_study_proposal_submission.v1"] = (
        "campaign_study_proposal_submission.v1"
    )
    proposal_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    hypothesis: str = Field(min_length=1, max_length=4000)
    evidence_references: tuple[str, ...] = ()
    study_family: Identifier
    primary_variable: str = Field(min_length=1, max_length=1000)
    controlled_variables: tuple[str, ...] = ()
    expected_outcome: str = Field(min_length=1, max_length=2000)
    falsification_criterion: str = Field(min_length=1, max_length=2000)
    estimated_cost: float = Field(ge=0)
    priority: int = Field(default=50, ge=0, le=100)
    prerequisite_study_ids: tuple[Identifier, ...] = ()
    dataset_recipe: dict[str, Any]
    training_recipe: dict[str, Any]
    evaluation_recipe: dict[str, Any]
    required_capabilities: frozenset[Capability] = frozenset()
    stage_plan: StagePlan
    rationale: str = Field(min_length=1, max_length=4000)


class StudyProposal(FrozenContractModel):
    """Immutable scientist proposal; rationale is evidence, never instruction."""

    schema_version: Literal["campaign_study_proposal.v1"] = "campaign_study_proposal.v1"
    proposal_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    hypothesis: str = Field(min_length=1, max_length=4000)
    evidence_references: tuple[str, ...] = ()
    study_family: Identifier
    primary_variable: str = Field(min_length=1, max_length=1000)
    controlled_variables: tuple[str, ...] = ()
    expected_outcome: str = Field(min_length=1, max_length=2000)
    falsification_criterion: str = Field(min_length=1, max_length=2000)
    estimated_cost: float = Field(ge=0)
    priority: int = Field(default=50, ge=0, le=100)
    prerequisite_study_ids: tuple[Identifier, ...] = ()
    dataset_recipe: dict[str, Any]
    training_recipe: dict[str, Any]
    evaluation_recipe: dict[str, Any]
    required_capabilities: frozenset[Capability] = frozenset()
    stage_plan: StagePlan
    planner_actor_id: Identifier
    rationale: str = Field(min_length=1, max_length=4000)
    status: ProposalStatus = ProposalStatus.SUBMITTED
    creation_sequence: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class ProposalValidation(FrozenContractModel):
    schema_version: Literal["campaign_proposal_validation.v1"] = "campaign_proposal_validation.v1"
    valid: bool
    reason_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_reasons(self) -> ProposalValidation:
        if self.valid and self.reason_codes:
            raise ValueError("valid proposal cannot contain rejection reasons")
        if not self.valid and not self.reason_codes:
            raise ValueError("invalid proposal requires rejection reasons")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("proposal rejection reasons must be sorted and unique")
        return self


class ProposalRecord(FrozenContractModel):
    schema_version: Literal["campaign_proposal_record.v1"] = "campaign_proposal_record.v1"
    proposal: StudyProposal
    validation: ProposalValidation
    study_id: Identifier | None = None
    updated_at: datetime


class CodeLineageRecord(FrozenContractModel):
    """Secret-free, durable Git evidence for a code-mutating proposal."""

    schema_version: Literal["campaign_code_lineage.v1"] = "campaign_code_lineage.v1"
    lineage_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    proposal_id: Identifier
    mutation_kind: CodeMutationKind
    source_repository_profile_id: Identifier
    state: CodeLineageState
    base_commit: GitObjectId | None = None
    branch_name: str | None = Field(default=None, min_length=1, max_length=240)
    commit_sha: GitObjectId | None = None
    changed_paths: tuple[str, ...] = ()
    patch_sha256: HexDigest | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    captured_at: datetime | None = None

    @field_validator("branch_name")
    @classmethod
    def validate_branch_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value.startswith("bashgym/autoresearch/")
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) is None
            or ".." in value
            or "//" in value
            or value.endswith(("/", ".", ".lock"))
            or any(part.startswith(".") for part in value.split("/"))
        ):
            raise ValueError("code lineage branch name is not safe")
        return value

    @field_validator("changed_paths")
    @classmethod
    def validate_changed_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("code lineage changed paths must be sorted and unique")
        for raw_path in value:
            path = PurePosixPath(raw_path)
            if (
                not raw_path
                or raw_path.startswith("/")
                or "\\" in raw_path
                or path.as_posix() != raw_path
                or any(part in {"", ".", ".."} for part in path.parts)
                or any(ord(character) < 32 for character in raw_path)
            ):
                raise ValueError("code lineage changed path is not a safe relative path")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> CodeLineageRecord:
        has_preparation_evidence = self.base_commit is not None or self.branch_name is not None
        prepared = self.base_commit is not None and self.branch_name is not None
        captured = (
            self.commit_sha is not None
            and self.patch_sha256 is not None
            and bool(self.changed_paths)
            and self.captured_at is not None
        )
        if self.updated_at < self.created_at:
            raise ValueError("code lineage updated_at cannot precede created_at")
        if self.captured_at is not None and self.captured_at < self.created_at:
            raise ValueError("code lineage captured_at cannot precede created_at")
        if self.state == CodeLineageState.REQUIRED and (
            has_preparation_evidence
            or captured
            or self.commit_sha is not None
            or self.patch_sha256 is not None
            or self.changed_paths
            or self.captured_at is not None
        ):
            raise ValueError("required code lineage cannot contain Git evidence")
        if self.state == CodeLineageState.PREPARED and (
            not prepared
            or self.commit_sha is not None
            or self.patch_sha256 is not None
            or self.changed_paths
            or self.captured_at is not None
        ):
            raise ValueError("prepared code lineage requires only base and branch evidence")
        if self.state == CodeLineageState.CAPTURED and (not prepared or not captured):
            raise ValueError("captured code lineage requires complete Git evidence")
        return self

    @property
    def record_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class CampaignArtifactReference(FrozenContractModel):
    schema_version: Literal["campaign_artifact_reference.v1"] = "campaign_artifact_reference.v1"
    artifact_id: Identifier
    sha256: HexDigest
    size_bytes: int = Field(ge=0)
    schema_name: str = Field(min_length=1, max_length=240)
    valid: bool


class NemoGymEvidenceReference(FrozenContractModel):
    """Bounded NeMo Gym identity exposed to planners without raw rollout content."""

    schema_version: Literal["campaign_nemo_gym_evidence_reference.v1"] = (
        "campaign_nemo_gym_evidence_reference.v1"
    )
    artifact_id: Identifier
    artifact_sha256: HexDigest
    bundle_digest: HexDigest
    environment_id: Identifier
    environment_digest: HexDigest
    rollout_batch_digest: HexDigest
    token_evidence_digest: HexDigest
    refit_receipt_digest: HexDigest
    rollout_count: int = Field(ge=1, le=4096)
    mean_total_reward: float
    training_step: int = Field(ge=0)
    policy_revision: int = Field(ge=0)

    @field_validator("mean_total_reward")
    @classmethod
    def finite_reward(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("NeMo Gym mean total reward must be finite")
        return value


class CompletedHypothesisSummary(FrozenContractModel):
    schema_version: Literal["campaign_completed_hypothesis_summary.v1"] = (
        "campaign_completed_hypothesis_summary.v1"
    )
    proposal_id: Identifier
    study_id: Identifier | None = None
    study_family: Identifier
    status: ProposalStatus


class CampaignEvidenceSnapshot(FrozenContractModel):
    """Bounded planner context with no protected rows, raw excerpts, or artifact URIs."""

    schema_version: Literal["campaign_evidence_snapshot.v1"] = "campaign_evidence_snapshot.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    campaign_version: int = Field(ge=1)
    manifest_revision: int = Field(ge=1)
    status: CampaignStatus
    objective: str = Field(min_length=1, max_length=4000)
    champion_ref: str | None = Field(default=None, max_length=2000)
    best_development_candidate_ref: str | None = Field(default=None, max_length=2000)
    approved_data_scopes: tuple[Identifier, ...]
    compute_profile_id: Identifier
    budget_remaining: dict[Identifier, float]
    proposal_counts: dict[ProposalStatus, int]
    completed_hypotheses: tuple[CompletedHypothesisSummary, ...] = Field(max_length=50)
    artifact_references: tuple[CampaignArtifactReference, ...] = Field(max_length=100)
    nemo_gym_evidence_references: tuple[NemoGymEvidenceReference, ...] = Field(
        default=(), max_length=100
    )
    available_executors: tuple[Identifier, ...]
    active_study_id: Identifier | None = None
    active_action_id: Identifier | None = None
    snapshot_digest: HexDigest = ""
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def populate_digest(self) -> CampaignEvidenceSnapshot:
        payload = self.model_dump(mode="json", exclude={"snapshot_digest", "created_at"})
        digest = canonical_hash(payload)
        if self.snapshot_digest and self.snapshot_digest != digest:
            raise ValueError("snapshot_digest does not match evidence snapshot")
        if not self.snapshot_digest:
            object.__setattr__(self, "snapshot_digest", digest)
        return self


class ControlRoomCampaignV1(FrozenContractModel):
    """Safe campaign identity and lifecycle fields for the control room."""

    schema_version: Literal["control_room_campaign.v1"] = "control_room_campaign.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    title: str = Field(min_length=1, max_length=240)
    kind: CampaignKind
    objective: str = Field(min_length=1, max_length=4000)
    manifest_revision: int = Field(ge=1)
    status: CampaignStatus
    active_study_id: Identifier | None = None
    active_action_id: Identifier | None = None
    stop_reason: str | None = Field(default=None, max_length=2000)
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ControlRoomStatusCountV1(FrozenContractModel):
    schema_version: Literal["control_room_status_count.v1"] = "control_room_status_count.v1"
    status: Identifier
    count: int = Field(ge=1)


class ControlRoomCollectionSummaryV1(FrozenContractModel):
    """Bounded status histogram; never embeds collection rows."""

    schema_version: Literal["control_room_collection_summary.v1"] = (
        "control_room_collection_summary.v1"
    )
    total: int = Field(ge=0)
    by_status: tuple[ControlRoomStatusCountV1, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_counts(self) -> ControlRoomCollectionSummaryV1:
        statuses = tuple(item.status for item in self.by_status)
        if statuses != tuple(sorted(set(statuses))):
            raise ValueError("control-room status counts must be sorted and unique")
        if sum(item.count for item in self.by_status) != self.total:
            raise ValueError("control-room status counts must equal the collection total")
        return self


class ControlRoomArtifactSummaryV1(FrozenContractModel):
    """Artifact counts without URI, metadata, or restricted evaluation content."""

    schema_version: Literal["control_room_artifact_summary.v1"] = (
        "control_room_artifact_summary.v1"
    )
    total: int = Field(ge=0)
    sealed: int = Field(ge=0)
    valid: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_subtotals(self) -> ControlRoomArtifactSummaryV1:
        if self.sealed > self.total or self.valid > self.total:
            raise ValueError("artifact subtotals cannot exceed the total")
        return self


class CampaignControlRoomStateV1(FrozenContractModel):
    """One-transaction durable state used to compose a control-room snapshot."""

    schema_version: Literal["control_room_state.v1"] = "control_room_state.v1"
    campaign: ControlRoomCampaignV1
    latest_event_cursor: int = Field(ge=0)
    proposals: ControlRoomCollectionSummaryV1
    studies: ControlRoomCollectionSummaryV1
    actions: ControlRoomCollectionSummaryV1
    attempts: ControlRoomCollectionSummaryV1
    artifacts: ControlRoomArtifactSummaryV1
    state_observed_at: datetime


class ControlRoomControllerObservationV1(FrozenContractModel):
    """Separately timestamped controller observation outside the SQLite snapshot."""

    schema_version: Literal["control_room_controller_observation.v1"] = (
        "control_room_controller_observation.v1"
    )
    online: bool
    controller_observation_version: int = Field(default=0, ge=0)
    state: Literal["online", "stale", "offline"]
    code: Identifier
    observed_at: datetime
    heartbeat_age_seconds: float | None = Field(default=None, ge=0)
    guidance: str | None = Field(default=None, max_length=2000)
    owner_id: Identifier | None = None
    generation: int | None = Field(default=None, ge=1)
    heartbeat_at: datetime | None = None
    expires_at: datetime | None = None


class CampaignControlRoomSnapshotV1(FrozenContractModel):
    """Authenticated control-room response with explicit observation boundaries."""

    schema_version: Literal["control_room_snapshot.v1"] = "control_room_snapshot.v1"
    authorization_revision: int = Field(ge=1)
    durable_state: CampaignControlRoomStateV1
    controller_observation: ControlRoomControllerObservationV1


class Study(ContractModel):
    """Accepted proposal with one immutable stage plan and execution cursor."""

    schema_version: Literal["campaign_study.v1"] = "campaign_study.v1"
    study_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    proposal_id: Identifier
    status: StudyStatus = StudyStatus.VALIDATED
    stage_plan: StagePlan
    current_stage_index: int = Field(default=0, ge=0)
    candidate_digest: HexDigest
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ActionAttempt(ContractModel):
    """Durable attempt for one logical stage action."""

    schema_version: Literal["campaign_action_attempt.v1"] = "campaign_action_attempt.v1"
    attempt_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_number: int = Field(ge=1)
    claim_generation: int = Field(default=0, ge=0)
    status: AttemptStatus = AttemptStatus.SCHEDULED
    input_digest: HexDigest
    candidate_digest: HexDigest
    manifest_revision: int = Field(ge=1)
    stage: StageKind
    lease_owner: Identifier | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    executor: dict[str, Any] = Field(default_factory=dict)
    sealed_result_uri: str | None = Field(default=None, max_length=4096)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CampaignEvent(FrozenContractModel):
    schema_version: Literal["campaign_event.v1"] = "campaign_event.v1"
    event_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    sequence: int = Field(ge=1)
    aggregate_version: int = Field(ge=1)
    event_type: Identifier
    payload: dict[str, Any] = Field(default_factory=dict)
    actor_id: Identifier
    credential_kind: CredentialKind
    correlation_id: Identifier
    idempotency_key: Identifier
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("payload")
    @classmethod
    def reject_secret_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        secret_name = re.compile(r"(?:token|password|secret|api[_-]?key)", re.IGNORECASE)

        def walk(item: Any) -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    if secret_name.search(str(key)):
                        raise ValueError("event payload contains a secret-like field")
                    walk(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    walk(nested)

        walk(value)
        return value


class ActorPrincipal(FrozenContractModel):
    schema_version: Literal["campaign_actor_principal.v1"] = "campaign_actor_principal.v1"
    actor_id: Identifier
    autonomy_profile: AutonomyProfile
    credential_id: Identifier
    credential_kind: CredentialKind
    workspace_ids: tuple[Identifier, ...]
    capabilities: frozenset[Capability]
    authorization_revision: int = Field(default=1, ge=1)
    expires_at: datetime

    def require(self, workspace_id: str, capability: Capability) -> None:
        """Fail closed unless this principal owns the workspace and capability."""

        has_local_desktop_scope = (
            self.autonomy_profile == AutonomyProfile.DESKTOP_USER
            and self.workspace_ids == (DESKTOP_LOCAL_SCOPE,)
        )
        if not has_local_desktop_scope and workspace_id not in self.workspace_ids:
            raise PermissionError("campaign_workspace_forbidden")
        if capability not in self.capabilities:
            raise PermissionError(f"campaign_capability_required:{capability.value}")


class IssuedCredential(FrozenContractModel):
    """One-time credential issuance response; never persist or log this model."""

    schema_version: Literal["campaign_issued_credential.v1"] = "campaign_issued_credential.v1"
    credential_id: Identifier
    raw_token: str = Field(min_length=32, max_length=1000, repr=False)
    kind: CredentialKind
    expires_at: datetime


class ArtifactOutput(FrozenContractModel):
    """One immutable file covered by an action-result seal."""

    schema_version: Literal["campaign_artifact_output.v1"] = "campaign_artifact_output.v1"
    path: str = Field(min_length=1, max_length=4096)
    sha256: HexDigest
    size_bytes: int = Field(ge=0)
    schema_name: str = Field(min_length=1, max_length=240)
    contract_valid: bool = True
    validation_errors: tuple[str, ...] = ()


class ResourceUsage(FrozenContractModel):
    """Measured or conservative resource use for budget settlement."""

    schema_version: Literal["campaign_resource_usage.v1"] = "campaign_resource_usage.v1"
    unit: Identifier
    amount: float = Field(ge=0)
    source: str = Field(min_length=1, max_length=240)
    confidence: Literal["measured", "estimated", "ceiling"]


class SealedActionResult(FrozenContractModel):
    """Replay-complete manifest signed before an atomic directory rename."""

    schema_version: Literal["sealed_action_result.v1"] = "sealed_action_result.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    manifest_revision: int = Field(ge=1)
    candidate_digest: HexDigest
    input_digest: HexDigest
    claim_generation: int = Field(ge=1)
    executor_id: Identifier
    executor_version: str = Field(min_length=1, max_length=240)
    compute_profile_id: Identifier
    remote_process_identity: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    ended_at: datetime
    outcome: Literal["completed", "failed", "cancelled", "force_stopped"]
    exit_code: int | None = None
    exit_reason: str = Field(min_length=1, max_length=2000)
    resource_usage: tuple[ResourceUsage, ...] = ()
    log_reference: str | None = Field(default=None, max_length=4096)
    outputs: tuple[ArtifactOutput, ...]

    @field_validator("outputs")
    @classmethod
    def validate_outputs(cls, value: tuple[ArtifactOutput, ...]) -> tuple[ArtifactOutput, ...]:
        if not value:
            raise ValueError("sealed action result requires at least one output")
        paths = tuple(item.path for item in value)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("sealed output paths must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_timing_and_outcome(self) -> SealedActionResult:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.outcome == "completed" and self.exit_code not in {0, None}:
            raise ValueError("completed result cannot have a failing exit code")
        return self


class ProtectedEvaluationResult(FrozenContractModel):
    """Candidate-locked, bounded result for a one-use protected evaluation epoch."""

    schema_version: Literal["campaign_protected_evaluation.v1"] = (
        "campaign_protected_evaluation.v1"
    )
    protected_epoch_id: Identifier
    candidate_digest: HexDigest
    passed: bool
    metrics: dict[Identifier, float]
    artifact_sha256: HexDigest

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or len(value) > 100:
            raise ValueError("protected evaluation metrics must contain 1 to 100 values")
        if any(not math.isfinite(metric) for metric in value.values()):
            raise ValueError("protected evaluation metrics must be finite")
        return dict(sorted(value.items()))

    @property
    def result_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class BudgetLedgerEntry(FrozenContractModel):
    """Append-only deltas for one campaign budget unit."""

    schema_version: Literal["campaign_budget_entry.v1"] = "campaign_budget_entry.v1"
    entry_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    unit: Identifier
    kind: BudgetEntryKind
    reserved_delta: float = 0
    actual_delta: float = 0
    limit_delta: float = 0
    action_id: Identifier | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    actor_id: Identifier
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_deltas(self) -> BudgetLedgerEntry:
        values = (self.reserved_delta, self.actual_delta, self.limit_delta)
        if self.kind == BudgetEntryKind.RESERVE and not (
            self.reserved_delta > 0 and self.actual_delta == 0 and self.limit_delta == 0
        ):
            raise ValueError("reserve entries require only a positive reserved_delta")
        if self.kind == BudgetEntryKind.RELEASE and not (
            self.reserved_delta < 0 and self.actual_delta == 0 and self.limit_delta == 0
        ):
            raise ValueError("release entries require only a negative reserved_delta")
        if self.kind == BudgetEntryKind.SETTLE and not (
            self.reserved_delta <= 0
            and self.actual_delta >= 0
            and self.limit_delta == 0
            and any(values)
        ):
            raise ValueError(
                "settle entries release reservation and record non-negative actual use"
            )
        if self.kind == BudgetEntryKind.AMEND and not (
            self.reserved_delta == 0 and self.actual_delta == 0 and self.limit_delta != 0
        ):
            raise ValueError("amend entries require only a non-zero limit_delta")
        if self.kind == BudgetEntryKind.CORRECTION and not any(values):
            raise ValueError("correction entry requires at least one non-zero delta")
        return self


__all__ = [
    "ActionStatus",
    "ActionAttempt",
    "ActorPrincipal",
    "ArtifactOutput",
    "AttemptStatus",
    "AutonomyProfile",
    "BudgetEntryKind",
    "BudgetLedgerEntry",
    "CODEX_CAPABILITIES",
    "DESKTOP_LOCAL_SCOPE",
    "Campaign",
    "CampaignArtifactReference",
    "CampaignControlRoomSnapshotV1",
    "CampaignControlRoomStateV1",
    "CampaignEvidenceSnapshot",
    "CampaignEvent",
    "CampaignKind",
    "CampaignManifest",
    "CampaignStatus",
    "CampaignTrigger",
    "Capability",
    "CodeLineageRecord",
    "CodeLineageState",
    "CodeMutationKind",
    "CredentialKind",
    "HERMES_CAPABILITIES",
    "IssuedCredential",
    "ManifestRevision",
    "NemoGymEvidenceReference",
    "CompletedHypothesisSummary",
    "ControlRoomArtifactSummaryV1",
    "ControlRoomCampaignV1",
    "ControlRoomCollectionSummaryV1",
    "ControlRoomControllerObservationV1",
    "ControlRoomStatusCountV1",
    "ProposalRecord",
    "ProposalStatus",
    "ProposalValidation",
    "ProtectedEvaluationResult",
    "ResourceUsage",
    "SealedActionResult",
    "StageDisposition",
    "StageKind",
    "StagePlan",
    "StagePlanItem",
    "StudyStatus",
    "Study",
    "StudyProposal",
    "StudyProposalSubmission",
    "TERMINAL_CAMPAIGN_STATES",
    "TargetModelContract",
    "canonical_hash",
    "utc_now",
]
