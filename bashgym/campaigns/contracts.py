"""Stable, secret-free contracts for durable experiment campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
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

from bashgym._compat import UTC


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


CANONICAL_CAMPAIGN_EVENT_TYPES = frozenset(
    {
        "campaign:created",
        "campaign:validation-started",
        "campaign:validation-failed",
        "campaign:ready",
        "campaign:started",
        "campaign:paused",
        "campaign:resumed",
        "campaign:authority-required",
        "campaign:authority-satisfied",
        "campaign:cancelling",
        "campaign:cancelled",
        "campaign:completed",
        "campaign:failed",
        "campaign:exhausted",
        "campaign:proposal-submitted",
        "campaign:proposal-rejected",
        "campaign:proposal-withdrawn",
        "campaign:proposal-accepted",
        "campaign:advance-requested",
        "campaign:manifest-revised",
        "campaign:source-approved",
        "campaign:study-abandoned",
        "campaign:action-blocked",
        "campaign:stages-skipped",
        "campaign:action-retry-scheduled",
        "campaign:force-stop-requested",
        "campaign:training-metrics-appended",
        "campaign:remote-run-registered",
        "campaign:remote-run-adopted",
        "campaign:remote-capacity-blocked",
        "campaign:action-scheduled",
        "campaign:action-claimed",
        "campaign:action-unknown",
        "campaign:action-completed",
        "campaign:action-failed",
        "campaign:action-cancelled",
        "campaign:action-force-stopped",
        "campaign:budget-recorded",
        "campaign:budget-overrun",
        "campaign:protected-lease-acquired",
        "campaign:protected-evaluation-completed",
        "campaign:promotion-committed",
        "campaign:export-completed",
        "campaign:human-work-claimed",
        "campaign:human-work-enqueued",
        "campaign:human-work-submitted",
        "campaign:human-promotion-held",
        "campaign:human-promotion-approved",
    }
)

PUBLIC_CAMPAIGN_BLOCKER_CODES = frozenset(
    {
        "campaign_controller_action_blocked",
        "campaign_not_found",
        "campaign_stage_cursor_exhausted",
        "campaign_code_lineage_not_captured",
        "campaign_code_lineage_not_registered",
        "campaign_code_lineage_mutation_kind_mismatch",
        "campaign_code_lineage_execution_binding_required",
        "campaign_code_lineage_execution_binding_mismatch",
        "campaign_recipe_runtime_invalid",
        "campaign_remote_stage_not_allowed",
        "campaign_remote_profile_unavailable",
        "campaign_remote_target_model_mismatch",
        "campaign_remote_profile_material_invalid",
        "campaign_executor_kind_not_registered",
        "campaign_budget_unit_not_approved",
    }
)

PUBLIC_CAMPAIGN_ARTIFACT_SCHEMA_NAMES = frozenset(
    {
        "campaign_development_comparison.v1",
        "campaign_fake_summary.v1",
        "campaign_remote_exit_code.v1",
        "campaign_remote_launch_manifest.v2",
        "campaign_remote_output.v1",
        "campaign_retrieval_evaluation.v1",
        "campaign_scored_development_rows.v1",
        "campaign_training_log.v1",
        "campaign_unlaunched_cancellation.v1",
        "campaign_validated_dev_dataset.v1",
        "embedding_training_manifest.v1",
        "huggingface_model_file.v1",
        "memexai_query_format_ablation_manifest.v1",
        "nemo_gym_campaign_evidence.v1",
        "query_format_ablation_manifest.v2",
        "training_manifest.v1",
        "training_metrics_jsonl.v1",
        "unclassified_artifact.v1",
    }
)


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
    HANDOFF_EXTERNAL_PREPARE = "handoff.external_prepare"
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
CODEX_CAPABILITIES = frozenset(
    capability
    for capability in Capability
    if capability
    not in {
        Capability.PROMOTION_OVERRIDE,
        Capability.HANDOFF_MEMEXAI_PREPARE,
    }
)
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
    allow_external_handoff: bool = False
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

    schema_version: Literal["control_room_artifact_summary.v1"] = "control_room_artifact_summary.v1"
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


class CampaignSummaryV1(FrozenContractModel):
    schema_version: Literal["campaign_summary.v1"] = "campaign_summary.v1"
    campaign_id: Identifier
    title: str
    objective: str
    kind: CampaignKind
    status: CampaignStatus
    aggregate_version: int = Field(ge=1)
    manifest_revision: int = Field(ge=1)
    active_study_id: Identifier | None
    active_action_id: Identifier | None
    champion_ref: str | None
    stop_reason: str | None


class ControllerObservationV1(FrozenContractModel):
    schema_version: Literal["controller_observation.v1"] = "controller_observation.v1"
    controller_observation_version: int = Field(ge=0)
    state: Literal["online", "stale", "offline"]
    observed_at: datetime
    heartbeat_age_seconds: float | None = Field(ge=0)
    lease_expires_at: datetime | None
    controller_instance_id: Identifier | None
    safe_guidance: str | None


class ReadinessSummaryV1(FrozenContractModel):
    schema_version: Literal["readiness_summary.v1"] = "readiness_summary.v1"
    materializable: bool
    launch_ready: bool
    checked_at: datetime
    activation_receipt_digest: HexDigest | None
    doctor_receipt_digest: HexDigest | None
    blocking_codes: tuple[Identifier, ...] = Field(max_length=64)

    @field_validator("blocking_codes")
    @classmethod
    def validate_blocking_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("readiness blocking codes must be unique")
        return value


class SafeBindingIdentityV1(FrozenContractModel):
    schema_version: Literal["safe_binding_identity.v1"] = "safe_binding_identity.v1"
    binding_id: Identifier
    immutable_digest: HexDigest | None
    display_label: str


class BindingSummaryV1(FrozenContractModel):
    schema_version: Literal["binding_summary.v1"] = "binding_summary.v1"
    model: SafeBindingIdentityV1 | None
    data: SafeBindingIdentityV1 | None
    evaluator: SafeBindingIdentityV1 | None
    source: SafeBindingIdentityV1 | None
    compute: SafeBindingIdentityV1 | None


class DecisionBlockerV1(FrozenContractModel):
    schema_version: Literal["decision_blocker.v1"] = "decision_blocker.v1"
    code: Identifier
    summary: str
    evidence_ids: tuple[Identifier, ...] = Field(max_length=64)
    secondary_codes: tuple[Identifier, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_unique_codes_and_evidence(self) -> DecisionBlockerV1:
        if self.code in self.secondary_codes or len(self.secondary_codes) != len(
            set(self.secondary_codes)
        ):
            raise ValueError("decision blocker codes must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("decision blocker evidence IDs must be unique")
        return self


class JourneyPhaseSummaryV1(FrozenContractModel):
    schema_version: Literal["journey_phase_summary.v1"] = "journey_phase_summary.v1"
    phase_id: Literal["setup", "baseline", "experiments", "human_review", "decision"]
    state: Literal["not_started", "ready", "active", "blocked", "complete", "failed", "skipped"]
    execution_owner: Literal["bashgym", "human", "none"]
    attention_owner: Literal["bashgym", "agent", "human", "none"]
    primary_blocker: DecisionBlockerV1 | None
    evidence_count: int = Field(ge=0)
    next_action_ids: tuple[Identifier, ...] = Field(max_length=64)

    @field_validator("next_action_ids")
    @classmethod
    def validate_next_action_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("journey action IDs must be unique")
        return value


class OpaqueProcessIdentityV1(FrozenContractModel):
    schema_version: Literal["opaque_process_identity.v1"] = "opaque_process_identity.v1"
    run_id: Identifier
    compute_profile_id: Identifier
    state: Literal["launching", "running", "completed", "failed", "cancelled", "unknown"]


class ActiveWorkSummaryV1(FrozenContractModel):
    schema_version: Literal["active_work_summary.v1"] = "active_work_summary.v1"
    study_id: Identifier | None
    proposal_id: Identifier | None
    action_id: Identifier | None
    attempt_id: Identifier | None
    stage: StageKind | None
    hypothesis_summary: str | None
    primary_variable_summary: str | None
    controlled_variable_summary: tuple[str, ...] = Field(max_length=64)
    progress_fraction: float | None = Field(ge=0, le=1)
    eta_seconds: float | None = Field(ge=0)
    executor_type: Literal["fake", "ssh_remote", "development_evaluation"] | None
    process_identity: OpaqueProcessIdentityV1 | None


class CandidateSummaryV1(FrozenContractModel):
    schema_version: Literal["candidate_summary.v1"] = "candidate_summary.v1"
    candidate_ref: str
    source_attempt_ids: tuple[Identifier, ...] = Field(max_length=100)
    source_artifact_ids: tuple[Identifier, ...] = Field(max_length=100)
    latest_comparable_evaluation_id: Identifier | None
    comparison_verdict: Literal["passed", "failed", "insufficient_evidence"] | None
    gate_state: Literal["not_evaluated", "blocked", "passed", "failed", "promoted"]


class MetricDescriptorV1(FrozenContractModel):
    schema_version: Literal["metric_descriptor.v1"] = "metric_descriptor.v1"
    metric_id: Identifier
    display_name: str
    unit: str | None
    direction: Literal["maximize", "minimize", "target"]
    target: float | None
    tolerance: float | None
    evaluator_revision: str | None
    sample_count: int | None = Field(ge=0)
    uncertainty_method: str | None
    comparability_key: HexDigest


class BudgetResourceSummaryV1(FrozenContractModel):
    schema_version: Literal["budget_resource_summary.v1"] = "budget_resource_summary.v1"
    unit: Identifier
    limit: float
    reserved: float
    settled: float
    remaining: float
    blocked: bool
    blocker_code: Identifier | None


class BudgetSummaryV1(FrozenContractModel):
    schema_version: Literal["budget_summary.v1"] = "budget_summary.v1"
    resources: tuple[BudgetResourceSummaryV1, ...] = Field(max_length=64)
    blocked: bool


class HumanWorkItemSummaryV1(FrozenContractModel):
    schema_version: Literal["human_work_item_summary.v1"] = "human_work_item_summary.v1"
    work_item_id: Identifier
    kind: Literal["blinded_sample_evaluation", "promotion_decision"]
    status: Literal[
        "open",
        "claimed",
        "accepted",
        "rejected",
        "revision_requested",
        "abstained",
        "cancelled",
        "expired",
    ]
    blocking_scope: Identifier
    assigned_actor_id: Identifier | None
    required_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    due_at: datetime | None


class HumanWorkSummaryV1(FrozenContractModel):
    schema_version: Literal["human_work_summary.v1"] = "human_work_summary.v1"
    blocking_count: int = Field(ge=0)
    open_count: int = Field(ge=0)
    newest: tuple[HumanWorkItemSummaryV1, ...] = Field(max_length=10)


class AttachedAgentSummaryV1(FrozenContractModel):
    schema_version: Literal["attached_agent_summary.v1"] = "attached_agent_summary.v1"
    session_id: Identifier
    actor_id: Identifier
    origin_id: Identifier
    bundle_id: Identifier
    capability_revision: int = Field(ge=1)
    expires_at: datetime
    liveness: Literal["active", "disconnected", "expired", "revoked"]
    last_cursor: int = Field(ge=0)
    last_request_id: Identifier | None


class CollectionCursorV1(FrozenContractModel):
    schema_version: Literal["collection_cursor.v1"] = "collection_cursor.v1"
    count: int = Field(ge=0)
    next_cursor: str | None
    has_more: bool

    @model_validator(mode="after")
    def validate_cursor(self) -> CollectionCursorV1:
        if self.has_more != (self.next_cursor is not None):
            raise ValueError("collection has_more must match next_cursor presence")
        return self


class CollectionSummaryV1(FrozenContractModel):
    schema_version: Literal["collection_summary.v1"] = "collection_summary.v1"
    events: CollectionCursorV1
    proposals: CollectionCursorV1
    studies: CollectionCursorV1
    attempts: CollectionCursorV1
    artifacts: CollectionCursorV1
    comparisons: CollectionCursorV1
    human_work: CollectionCursorV1


class DecisionActionV1(FrozenContractModel):
    schema_version: Literal["decision_action.v1"] = "decision_action.v1"
    action: Identifier
    capability: Capability
    freshness_class: Literal["read", "lifecycle", "privileged"]
    requires_human_work: bool


class DecisionSurfaceV1(FrozenContractModel):
    schema_version: Literal["decision_surface.v1"] = "decision_surface.v1"
    execution_owner: Literal["bashgym", "human", "none"]
    attention_owner: Literal["bashgym", "agent", "human", "none"]
    blocker: DecisionBlockerV1 | None
    next_actions: tuple[DecisionActionV1, ...] = Field(max_length=64)
    recovery_actions: tuple[Identifier, ...] = Field(max_length=16)
    promotion_eligible: bool

    @model_validator(mode="after")
    def validate_unique_actions(self) -> DecisionSurfaceV1:
        actions = tuple(item.action for item in self.next_actions)
        if len(actions) != len(set(actions)):
            raise ValueError("decision actions must be unique")
        if len(self.recovery_actions) != len(set(self.recovery_actions)):
            raise ValueError("recovery actions must be unique")
        return self


class CampaignControlRoomSnapshotV1(FrozenContractModel):
    """Complete principal-filtered public campaign projection."""

    schema_version: Literal["control_room_snapshot.v1"] = "control_room_snapshot.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    aggregate_version: int = Field(ge=1)
    manifest_revision: int = Field(ge=1)
    authorization_revision: int = Field(ge=1)
    snapshot_at: datetime
    latest_event_cursor: int = Field(ge=0)
    campaign: CampaignSummaryV1
    controller: ControllerObservationV1
    readiness: ReadinessSummaryV1
    bindings: BindingSummaryV1
    journey: tuple[JourneyPhaseSummaryV1, ...] = Field(min_length=5, max_length=5)
    active_work: ActiveWorkSummaryV1 | None
    champion: CandidateSummaryV1 | None
    candidate: CandidateSummaryV1 | None
    metrics: tuple[MetricDescriptorV1, ...] = Field(max_length=64)
    budget: BudgetSummaryV1
    human_work: HumanWorkSummaryV1
    agents: tuple[AttachedAgentSummaryV1, ...] = Field(max_length=32)
    collections: CollectionSummaryV1
    decision_surface: DecisionSurfaceV1

    @model_validator(mode="after")
    def validate_identity_and_journey(self) -> CampaignControlRoomSnapshotV1:
        if self.campaign_id != self.campaign.campaign_id:
            raise ValueError("snapshot and campaign IDs must match")
        if self.aggregate_version != self.campaign.aggregate_version:
            raise ValueError("snapshot and campaign aggregate versions must match")
        if self.manifest_revision != self.campaign.manifest_revision:
            raise ValueError("snapshot and campaign manifest revisions must match")
        if tuple(phase.phase_id for phase in self.journey) != (
            "setup",
            "baseline",
            "experiments",
            "human_review",
            "decision",
        ):
            raise ValueError("control-room journey must use the exact ordered phases")
        return self


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


class PublicCampaignEventSummaryV1(FrozenContractModel):
    """Bounded workspace-safe fields classified for public campaign timelines."""

    schema_version: Literal["public_campaign_event_summary.v1"] = "public_campaign_event_summary.v1"
    action_id: Identifier | None = None
    attempt_id: Identifier | None = None
    study_id: Identifier | None = None
    proposal_id: Identifier | None = None
    entry_id: Identifier | None = None
    stage: Identifier | None = None
    code: Identifier | None = None
    manifest_revision: int | None = Field(default=None, ge=1)
    stage_index: int | None = Field(default=None, ge=0)
    next_stage_index: int | None = Field(default=None, ge=0)
    claim_generation: int | None = Field(default=None, ge=0)
    cursor_end: int | None = Field(default=None, ge=0)
    alert_count: int | None = Field(default=None, ge=0)
    study_completed: bool | None = None


class PublicCampaignEventV1(FrozenContractModel):
    """Fail-closed campaign event projection for readers and presentation surfaces."""

    schema_version: Literal["public_campaign_event.v1"] = "public_campaign_event.v1"
    event_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    sequence: int = Field(ge=1)
    aggregate_version: int = Field(ge=1)
    event_type: Identifier
    summary: PublicCampaignEventSummaryV1 | None = None
    actor_id: Identifier
    credential_kind: CredentialKind
    created_at: datetime


class PublicCampaignArtifactV1(FrozenContractModel):
    """Opaque artifact identity and seal metadata safe for campaign readers."""

    schema_version: Literal["public_campaign_artifact.v1"] = "public_campaign_artifact.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    artifact_id: Identifier
    producer_action_id: Identifier | None = None
    sha256: HexDigest
    size_bytes: int = Field(ge=0)
    schema_name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=240,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
        ),
    ]
    sealed: bool
    valid: bool
    created_at: datetime


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

    schema_version: Literal["campaign_protected_evaluation.v1"] = "campaign_protected_evaluation.v1"
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
    "ActiveWorkSummaryV1",
    "ActorPrincipal",
    "AttachedAgentSummaryV1",
    "ArtifactOutput",
    "AttemptStatus",
    "AutonomyProfile",
    "BudgetEntryKind",
    "BudgetLedgerEntry",
    "BudgetResourceSummaryV1",
    "BudgetSummaryV1",
    "BindingSummaryV1",
    "CODEX_CAPABILITIES",
    "DESKTOP_LOCAL_SCOPE",
    "Campaign",
    "CANONICAL_CAMPAIGN_EVENT_TYPES",
    "CampaignArtifactReference",
    "CampaignControlRoomSnapshotV1",
    "CampaignControlRoomStateV1",
    "CampaignSummaryV1",
    "CampaignEvidenceSnapshot",
    "CampaignEvent",
    "PublicCampaignEventSummaryV1",
    "PublicCampaignEventV1",
    "PublicCampaignArtifactV1",
    "PUBLIC_CAMPAIGN_ARTIFACT_SCHEMA_NAMES",
    "PUBLIC_CAMPAIGN_BLOCKER_CODES",
    "CampaignKind",
    "CampaignManifest",
    "CampaignStatus",
    "CampaignTrigger",
    "CandidateSummaryV1",
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
    "CollectionCursorV1",
    "CollectionSummaryV1",
    "ControllerObservationV1",
    "ControlRoomArtifactSummaryV1",
    "ControlRoomCampaignV1",
    "ControlRoomCollectionSummaryV1",
    "ControlRoomControllerObservationV1",
    "ControlRoomStatusCountV1",
    "DecisionActionV1",
    "DecisionBlockerV1",
    "DecisionSurfaceV1",
    "HumanWorkItemSummaryV1",
    "HumanWorkSummaryV1",
    "JourneyPhaseSummaryV1",
    "MetricDescriptorV1",
    "OpaqueProcessIdentityV1",
    "ProposalRecord",
    "ProposalStatus",
    "ProposalValidation",
    "ProtectedEvaluationResult",
    "ReadinessSummaryV1",
    "ResourceUsage",
    "SealedActionResult",
    "StageDisposition",
    "StageKind",
    "StagePlan",
    "StagePlanItem",
    "SafeBindingIdentityV1",
    "StudyStatus",
    "Study",
    "StudyProposal",
    "StudyProposalSubmission",
    "TERMINAL_CAMPAIGN_STATES",
    "TargetModelContract",
    "canonical_hash",
    "utc_now",
]
