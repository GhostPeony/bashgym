"""Durable, baseline-first AutoResearch control over campaign execution.

This module deliberately does not launch training.  It turns the existing campaign
repository into a scientific control loop: prepare an approved campaign, submit one
declared intervention at a time, record an executor-backed result, decide whether it
beat the incumbent, and expose the next safe action.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from bashgym._compat import UTC
from bashgym.campaigns.contracts import (
    TERMINAL_CAMPAIGN_STATES,
    ActionAttempt,
    ActorPrincipal,
    Campaign,
    CampaignEvent,
    CampaignKind,
    CampaignManifest,
    CampaignStatus,
    CampaignTrigger,
    Capability,
    CodeLineageRecord,
    CodeMutationKind,
    CredentialKind,
    FailureClass,
    FrozenContractModel,
    Identifier,
    ProposalStatus,
    StageDisposition,
    StageKind,
    StudyProposalSubmission,
    StudyStatus,
    TargetModelContract,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.diagnostic_actions import AutoResearchDiagnosticRecipe
from bashgym.campaigns.failure_classification import NON_SCIENTIFIC_FAILURE_CLASSES
from bashgym.campaigns.failure_observations import build_research_failure_packet
from bashgym.campaigns.lineage import code_mutation_kind_for_variable
from bashgym.campaigns.method_policy import AutoResearchMethodThresholds
from bashgym.campaigns.persistence import (
    CampaignPersistenceError,
    MigrationChecksumError,
    ProposalMutation,
    RecordNotFoundError,
    RevisionConflictError,
)
from bashgym.campaigns.research_diagnostics import (
    AutoResearchDiagnostics,
    build_autoresearch_diagnostics,
)
from bashgym.campaigns.runtime import CampaignArtifactRecord, CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.training_seed import training_seed, training_stages_required
from bashgym.ledger.contracts import DecisionSpec, LedgerEventSpec, stable_ledger_id
from bashgym.ledger.persistence import ExperimentLedgerRepository


class AutoResearchError(CampaignPersistenceError):
    """Stable base error for AutoResearch policy and persistence failures."""


class AutoResearchInvariantError(AutoResearchError):
    code = "autoresearch_invariant_failed"


class AutoResearchConflictError(AutoResearchError):
    code = "autoresearch_conflict"


class AutoResearchBudgetError(AutoResearchError):
    code = "autoresearch_budget_exceeded"


_SCIENTIFIC_PROPOSAL_FIELDS = (
    "study_family",
    "dataset_recipe",
    "training_recipe",
    "evaluation_recipe",
)
_MISSING_SCIENTIFIC_VALUE = object()


def _normalized_scientific_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _normalized_scientific_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_scientific_value(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    return value


def _scientific_leaves(proposal: Any) -> dict[tuple[str, ...], Any]:
    leaves: dict[tuple[str, ...], Any] = {}

    def visit(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit((*path, str(key)), item)
            return
        leaves[path] = _normalized_scientific_value(value)

    for field in _SCIENTIFIC_PROPOSAL_FIELDS:
        visit((field,), getattr(proposal, field))
    return leaves


def _validate_controlled_candidate_change(
    parent: Any,
    candidate: StudyProposalSubmission,
    *,
    declared_variable: str | None = None,
    declared_variables: tuple[str, ...] | None = None,
    intervention_mode: InterventionMode | str = "controlled",
    code_mutation_kind: CodeMutationKind | None,
) -> None:
    mode = InterventionMode(intervention_mode)
    declarations = declared_variables or ((declared_variable,) if declared_variable else ())
    if mode == InterventionMode.CONTROLLED:
        if len(declarations) != 1:
            raise AutoResearchInvariantError(
                "autoresearch_controlled_intervention_requires_one_variable"
            )
    elif not 2 <= len(declarations) <= 16:
        raise AutoResearchInvariantError(
            "autoresearch_exploratory_intervention_variable_limit_invalid"
        )
    if mode == InterventionMode.EXPLORATORY and code_mutation_kind is not None:
        raise AutoResearchInvariantError("autoresearch_exploratory_code_bundle_not_supported")

    parent_leaves = _scientific_leaves(parent)
    candidate_leaves = _scientific_leaves(candidate)
    all_paths = set(parent_leaves) | set(candidate_leaves)
    changed_paths = {
        path
        for path in all_paths
        if parent_leaves.get(path, _MISSING_SCIENTIFIC_VALUE)
        != candidate_leaves.get(path, _MISSING_SCIENTIFIC_VALUE)
    }

    if code_mutation_kind is not None:
        if changed_paths:
            raise AutoResearchInvariantError("autoresearch_candidate_changed_undeclared_variable")
        return
    if not changed_paths:
        raise AutoResearchInvariantError("autoresearch_candidate_declared_variable_unchanged")

    covered: set[tuple[str, ...]] = set()
    for declaration in declarations:
        declared = declaration.strip()
        if "." in declared:
            declared_path = tuple(declared.split("."))
            matches = {path for path in all_paths if path[: len(declared_path)] == declared_path}
        else:
            matches = {path for path in all_paths if path[-1] == declared}
        changed_matches = matches & changed_paths
        if not matches or not changed_matches:
            raise AutoResearchInvariantError("autoresearch_candidate_declared_variable_unchanged")
        if covered & changed_matches:
            raise AutoResearchInvariantError("autoresearch_candidate_changed_undeclared_variable")
        covered.update(changed_matches)
    if covered != changed_paths:
        raise AutoResearchInvariantError("autoresearch_candidate_changed_undeclared_variable")


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ExperimentRole(str, Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    DIAGNOSTIC = "diagnostic"


class InterventionMode(str, Enum):
    CONTROLLED = "controlled"
    EXPLORATORY = "exploratory"


class HypothesisFamilyDisposition(str, Enum):
    SUPPORTED = "supported"
    EXHAUSTED = "exhausted"
    INCONCLUSIVE = "inconclusive"


class ExperimentProvenance(str, Enum):
    REAL = "real"
    SIMULATED = "simulated"


class ExperimentOutcome(str, Enum):
    COMPLETED = "completed"
    CRASHED = "crashed"


class ResultDecision(str, Enum):
    BASELINE = "baseline"
    KEEP = "keep"
    DISCARD = "discard"
    CRASH = "crash"
    INELIGIBLE = "ineligible"


class AutoResearchNextAction(str, Enum):
    PREPARE_CAMPAIGN = "prepare_campaign"
    START_CAMPAIGN = "start_campaign"
    SUBMIT_BASELINE = "submit_baseline"
    WAIT_FOR_RESULT = "wait_for_result"
    PROPOSE_CANDIDATE = "propose_candidate"
    STOP = "stop"
    BLOCKED = "blocked"


class ProtectedMetricGate(FrozenContractModel):
    """Maximum allowed regression for one metric measured by the fixed suite."""

    metric_name: Identifier
    direction: MetricDirection
    max_regression: float = Field(default=0.0, ge=0)

    @field_validator("max_regression")
    @classmethod
    def finite_regression(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("protected metric regression must be finite")
        return value


class AutoResearchStopRules(FrozenContractModel):
    schema_version: Literal["autoresearch_stop_rules.v1"] = "autoresearch_stop_rules.v1"
    max_attempts: int = Field(ge=1, le=100)
    budget_unit: Identifier
    max_total_cost: float = Field(gt=0)
    target_metric: float | None = None
    minimum_improvement: float = Field(ge=0)
    protected_metrics: tuple[ProtectedMetricGate, ...] = ()
    deadline: datetime | None = None

    @model_validator(mode="after")
    def validate_finite_values(self) -> AutoResearchStopRules:
        values = (self.max_total_cost, self.minimum_improvement)
        if self.target_metric is not None:
            values += (self.target_metric,)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("AutoResearch stop-rule numeric values must be finite")
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("AutoResearch deadline must be timezone-aware")
        names = tuple(gate.metric_name for gate in self.protected_metrics)
        if len(names) != len(set(names)):
            raise ValueError("protected metric names must be unique")
        return self


class AutoResearchCampaignSpec(FrozenContractModel):
    schema_version: Literal["autoresearch_campaign_spec.v1"] = "autoresearch_campaign_spec.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    primary_metric: Identifier
    metric_direction: MetricDirection
    stop_rules: AutoResearchStopRules
    method_thresholds: AutoResearchMethodThresholds = Field(
        default_factory=AutoResearchMethodThresholds
    )
    ledger_project_id: Identifier | None = None
    evaluation_suite_id: Identifier | None = None
    require_sealed_artifact: bool = True
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evaluation_binding(self) -> AutoResearchCampaignSpec:
        if (self.ledger_project_id is None) != (self.evaluation_suite_id is None):
            raise ValueError(
                "ledger_project_id and evaluation_suite_id must be configured together"
            )
        return self

    @property
    def spec_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"created_at"}))


class AutoResearchProposalControl(FrozenContractModel):
    schema_version: Literal["autoresearch_proposal_control.v1"] = "autoresearch_proposal_control.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    proposal_id: Identifier
    role: ExperimentRole
    parent_proposal_id: Identifier | None = None
    changed_variables: tuple[str, ...] = ()
    intervention_mode: InterventionMode = InterventionMode.CONTROLLED
    hypothesis_family_id: Identifier | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("changed_variables")
    @classmethod
    def validate_changed_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned) or len(set(cleaned)) != len(cleaned):
            raise ValueError("changed_variables must be non-empty strings and unique")
        return cleaned

    @model_validator(mode="after")
    def validate_lineage(self) -> AutoResearchProposalControl:
        if self.role == ExperimentRole.BASELINE:
            if (
                self.parent_proposal_id is not None
                or self.changed_variables
                or self.intervention_mode != InterventionMode.CONTROLLED
                or self.hypothesis_family_id is not None
            ):
                raise ValueError("baseline cannot have a parent or changed variables")
        elif self.role == ExperimentRole.DIAGNOSTIC:
            if (
                self.parent_proposal_id is None
                or self.changed_variables
                or self.intervention_mode != InterventionMode.CONTROLLED
                or self.hypothesis_family_id is not None
            ):
                raise ValueError("diagnostic requires one parent and cannot change the model")
        elif self.parent_proposal_id is None:
            raise ValueError("candidate requires one parent")
        elif self.intervention_mode == InterventionMode.CONTROLLED:
            if len(self.changed_variables) != 1:
                raise ValueError("controlled candidate requires exactly one changed variable")
        elif not 2 <= len(self.changed_variables) <= 16:
            raise ValueError("exploratory candidate requires 2 to 16 changed variables")
        elif self.hypothesis_family_id is None:
            raise ValueError("exploratory candidate requires a hypothesis family")
        return self

    @property
    def control_digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "workspace_id": self.workspace_id,
            "campaign_id": self.campaign_id,
            "proposal_id": self.proposal_id,
            "role": self.role.value,
            "parent_proposal_id": self.parent_proposal_id,
            "changed_variables": list(self.changed_variables),
        }
        if self.intervention_mode != InterventionMode.CONTROLLED:
            payload["intervention_mode"] = self.intervention_mode.value
        if self.hypothesis_family_id is not None:
            payload["hypothesis_family_id"] = self.hypothesis_family_id
        return canonical_hash(payload)


class AutoResearchHypothesisFamilyConclusion(FrozenContractModel):
    """One immutable, evidence-bound conclusion for a hypothesis family."""

    schema_version: Literal["autoresearch_hypothesis_family_conclusion.v1"] = (
        "autoresearch_hypothesis_family_conclusion.v1"
    )
    workspace_id: Identifier
    campaign_id: Identifier
    hypothesis_family_id: Identifier
    disposition: HypothesisFamilyDisposition
    summary: str = Field(min_length=1, max_length=2000)
    proposal_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=32)
    result_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=32)
    follow_up_family_id: Identifier | None = None
    follow_up_hypothesis: str | None = Field(default=None, min_length=1, max_length=2000)
    aggregate_version: int = Field(ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    replayed: bool = False

    @model_validator(mode="after")
    def validate_conclusion(self) -> AutoResearchHypothesisFamilyConclusion:
        if len(self.proposal_ids) != len(self.result_ids):
            raise ValueError("hypothesis family proposals and results must have equal length")
        if len(set(self.proposal_ids)) != len(self.proposal_ids):
            raise ValueError("hypothesis family proposal IDs must be unique")
        if len(set(self.result_ids)) != len(self.result_ids):
            raise ValueError("hypothesis family result IDs must be unique")
        if (self.follow_up_family_id is None) != (self.follow_up_hypothesis is None):
            raise ValueError("follow-up family ID and hypothesis must be supplied together")
        if self.follow_up_family_id == self.hypothesis_family_id:
            raise ValueError("follow-up family must differ from the concluded family")
        return self

    @property
    def conclusion_digest(self) -> str:
        return canonical_hash(
            self.model_dump(
                mode="json",
                exclude={"created_at", "replayed"},
            )
        )


class AutoResearchDiagnosticResult(FrozenContractModel):
    """Durable aggregate result from a non-quality research diagnostic."""

    schema_version: Literal["autoresearch_diagnostic_result.v1"] = (
        "autoresearch_diagnostic_result.v1"
    )
    workspace_id: Identifier
    campaign_id: Identifier
    proposal_id: Identifier
    study_id: Identifier
    attempt_id: Identifier
    status: Literal["completed", "unsupported"]
    projection: dict[str, Any]
    actual_cost: float = Field(ge=0)
    recorded_at: datetime = Field(default_factory=utc_now)
    replayed: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> AutoResearchDiagnosticResult:
        if not math.isfinite(self.actual_cost):
            raise ValueError("diagnostic actual_cost must be finite")
        if self.projection.get("schema_version") != "bashgym.research_diagnostic_result.v1":
            raise ValueError("diagnostic projection schema is invalid")
        return self

    @property
    def result_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"replayed"}))


class AutoResearchResult(FrozenContractModel):
    schema_version: Literal["autoresearch_result.v1"] = "autoresearch_result.v1"
    result_id: Identifier
    workspace_id: Identifier
    campaign_id: Identifier
    proposal_id: Identifier
    study_id: Identifier
    role: ExperimentRole
    provenance: ExperimentProvenance
    outcome: ExperimentOutcome
    metric_name: Identifier
    metric_value: float | None = None
    metrics: dict[Identifier, float] = Field(default_factory=dict)
    failure_class: FailureClass | None = None
    actual_cost: float = Field(ge=0)
    attempt_ids: tuple[Identifier, ...]
    evidence_references: tuple[Identifier, ...] = ()
    recorded_at: datetime = Field(default_factory=utc_now)

    @field_validator("attempt_ids", "evidence_references")
    @classmethod
    def validate_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("AutoResearch references must be unique")
        return value

    @model_validator(mode="after")
    def validate_outcome_metric(self) -> AutoResearchResult:
        if not self.attempt_ids:
            raise ValueError("AutoResearch result requires at least one durable attempt")
        if self.outcome == ExperimentOutcome.COMPLETED:
            if self.metric_value is None or not math.isfinite(self.metric_value):
                raise ValueError("completed AutoResearch result requires a finite metric")
            if self.failure_class is not None:
                raise ValueError("completed AutoResearch result cannot carry a failure class")
        elif self.metric_value is not None:
            raise ValueError("crashed AutoResearch result cannot claim a final metric")
        if not math.isfinite(self.actual_cost):
            raise ValueError("AutoResearch actual_cost must be finite")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("AutoResearch metrics must be finite")
        if self.metric_value is not None and self.metrics.get(
            self.metric_name, self.metric_value
        ) != (self.metric_value):
            raise ValueError("primary metric must match the metrics projection")
        return self

    @property
    def result_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"recorded_at"}))

    def digest_matches(self, digest: str) -> bool:
        """Accept this result's digest, or the one written before failure_class existed."""

        if digest == self.result_digest:
            return True
        if self.failure_class is not None:
            return False
        return digest == canonical_hash(
            self.model_dump(mode="json", exclude={"recorded_at", "failure_class"})
        )


def counts_as_experiment(result: AutoResearchResult) -> bool:
    """A completed result or an execution crash is scientific evidence; infra faults are not."""

    if result.outcome != ExperimentOutcome.CRASHED:
        return True
    return result.failure_class not in NON_SCIENTIFIC_FAILURE_CLASSES


class AutoResearchDecision(FrozenContractModel):
    schema_version: Literal["autoresearch_decision.v1"] = "autoresearch_decision.v1"
    proposal_id: Identifier
    decision: ResultDecision
    reason_code: Identifier
    eligible_for_best: bool
    previous_best_proposal_id: Identifier | None = None
    previous_best_metric: float | None = None
    improvement: float | None = None
    protected_metric_margins: dict[Identifier, float] = Field(default_factory=dict)
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    decided_at: datetime = Field(default_factory=utc_now)


class AutoResearchOutcomeRecord(FrozenContractModel):
    schema_version: Literal["autoresearch_outcome_record.v1"] = "autoresearch_outcome_record.v1"
    result: AutoResearchResult
    decision: AutoResearchDecision
    replayed: bool = False


class AutoResearchLedgerCommitContext(FrozenContractModel):
    """Authoritative ledger lineage required for an atomic outcome commit."""

    project_id: Identifier
    experiment_id: Identifier
    run_id: Identifier
    attempt_id: Identifier
    correlation_id: Identifier
    actor_id: Identifier = "autoresearch-controller"


class AutoResearchState(FrozenContractModel):
    schema_version: Literal["autoresearch_state.v1"] = "autoresearch_state.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    campaign_status: CampaignStatus
    next_action: AutoResearchNextAction
    ready_for_next_proposal: bool
    reason_code: Identifier
    baseline_verified: bool
    pending_proposal_id: Identifier | None = None
    best_proposal_id: Identifier | None = None
    best_study_id: Identifier | None = None
    best_metric: float | None = None
    attempts_used: int = Field(ge=0)
    proposals_used: int = Field(ge=0)
    budget_used: float = Field(ge=0)
    budget_remaining: float
    latest_decision: ResultDecision | None = None


class AutoResearchTemplatePolicy(FrozenContractModel):
    """Portable AutoResearch policy paired with a scientific campaign template."""

    schema_version: Literal["autoresearch_template_policy.v1"] = "autoresearch_template_policy.v1"
    template_revision: Identifier
    primary_metric: Identifier
    metric_direction: MetricDirection
    stop_rules: AutoResearchStopRules
    method_thresholds: AutoResearchMethodThresholds = Field(
        default_factory=AutoResearchMethodThresholds
    )
    ledger_project_id: Identifier
    evaluation_suite_id: Identifier
    require_sealed_artifact: bool = True
    quality_claim_eligible: bool = False


class AutoResearchTemplateDefinition(FrozenContractModel):
    """Source-managed input for an API-compatible campaign template registry."""

    schema_version: Literal["autoresearch_template_definition.v1"] = (
        "autoresearch_template_definition.v1"
    )
    template_id: Identifier
    kind: CampaignKind = CampaignKind.GENERAL
    objective: str = Field(min_length=1, max_length=4000)
    target_model: TargetModelContract
    manifest: CampaignManifest
    policy: AutoResearchTemplatePolicy | None = None

    @model_validator(mode="after")
    def validate_policy_matches_manifest(self) -> AutoResearchTemplateDefinition:
        if self.policy is None:
            return self
        evaluation = self.manifest.evaluation_plan
        promotion = self.manifest.promotion_gates
        checks = (
            (evaluation.get("primary_metric"), self.policy.primary_metric),
            (evaluation.get("metric_direction"), self.policy.metric_direction.value),
            (evaluation.get("ledger_project_id"), self.policy.ledger_project_id),
            (evaluation.get("evaluation_suite_id"), self.policy.evaluation_suite_id),
        )
        if any(actual != expected for actual, expected in checks):
            raise ValueError("AutoResearch policy does not match the manifest evaluation plan")
        if self.policy.stop_rules.max_attempts > self.manifest.max_proposal_rounds:
            raise ValueError("AutoResearch attempts exceed the manifest proposal limit")
        manifest_limit = self.manifest.budget_limits.get(self.policy.stop_rules.budget_unit)
        if manifest_limit is None or self.policy.stop_rules.max_total_cost > manifest_limit:
            raise ValueError("AutoResearch policy budget is not covered by the manifest")
        if bool(promotion.get("quality_claim_eligible", False)) != (
            self.policy.quality_claim_eligible
        ):
            raise ValueError("AutoResearch quality eligibility must match promotion gates")
        return self

    @property
    def definition_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))

    def validate_campaign_stop_rules(
        self, stop_rules: AutoResearchStopRules
    ) -> AutoResearchStopRules:
        """Validate one campaign's explicit limits against the approved envelope."""

        if self.policy is None:
            raise ValueError("campaign_stop_rules_not_supported")
        if stop_rules.max_attempts > self.manifest.max_proposal_rounds:
            raise ValueError("autoresearch_max_attempts_exceeds_template")
        manifest_limit = self.manifest.budget_limits.get(stop_rules.budget_unit)
        if manifest_limit is None:
            raise ValueError("autoresearch_budget_unit_not_approved")
        if stop_rules.max_total_cost > manifest_limit:
            raise ValueError("autoresearch_budget_exceeds_template")
        if stop_rules.protected_metrics != self.policy.stop_rules.protected_metrics:
            raise ValueError("autoresearch_protected_metrics_must_match_template")
        return stop_rules

    def materialize_spec(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        stop_rules: AutoResearchStopRules,
    ) -> AutoResearchCampaignSpec | None:
        if self.policy is None:
            return None
        selected_stop_rules = self.validate_campaign_stop_rules(stop_rules)
        return AutoResearchCampaignSpec(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            primary_metric=self.policy.primary_metric,
            metric_direction=self.policy.metric_direction,
            stop_rules=selected_stop_rules,
            method_thresholds=self.policy.method_thresholds,
            ledger_project_id=self.policy.ledger_project_id,
            evaluation_suite_id=self.policy.evaluation_suite_id,
            require_sealed_artifact=self.policy.require_sealed_artifact,
        )

    def campaign_template_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "objective": self.objective,
            "target_model": self.target_model.model_dump(mode="json"),
            "manifest": self.manifest.model_dump(mode="json"),
        }


def build_autoresearch_template_registry(
    definitions: Iterable[AutoResearchTemplateDefinition],
) -> dict[str, dict[str, Any]]:
    """Build deterministic plain payloads accepted by the REST template boundary."""

    registry: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        if definition.template_id in registry:
            raise ValueError(f"duplicate AutoResearch template: {definition.template_id}")
        registry[definition.template_id] = definition.campaign_template_payload()
    return dict(sorted(registry.items()))


_MAX_SOURCE_TEMPLATE_BYTES = 64 * 1024


def load_autoresearch_template_definitions(
    directory: Path | None = None,
) -> tuple[AutoResearchTemplateDefinition, ...]:
    """Load bounded, source-managed JSON definitions without installation authority."""

    if directory is None:
        root = resources.files("bashgym.campaigns.templates")
        candidates = sorted(
            (item for item in root.iterdir() if item.name.endswith(".json")),
            key=lambda item: item.name,
        )
        payloads = []
        for item in candidates:
            raw = item.read_bytes()
            if len(raw) > _MAX_SOURCE_TEMPLATE_BYTES:
                raise ValueError(f"AutoResearch template is too large: {item.name}")
            payloads.append((item.name, raw))
    else:
        root_path = directory.resolve()
        payloads = []
        for path in sorted(root_path.glob("*.json")):
            resolved = path.resolve()
            if resolved.parent != root_path or path.is_symlink() or not path.is_file():
                raise ValueError(f"unsafe AutoResearch template path: {path.name}")
            raw = path.read_bytes()
            if len(raw) > _MAX_SOURCE_TEMPLATE_BYTES:
                raise ValueError(f"AutoResearch template is too large: {path.name}")
            payloads.append((path.name, raw))
    definitions: list[AutoResearchTemplateDefinition] = []
    seen: set[str] = set()
    for name, raw in payloads:
        try:
            definition = AutoResearchTemplateDefinition.model_validate_json(raw)
        except Exception as exc:
            raise ValueError(f"invalid AutoResearch template: {name}") from exc
        if definition.template_id in seen:
            raise ValueError(f"duplicate AutoResearch template: {definition.template_id}")
        seen.add(definition.template_id)
        definitions.append(definition)
    return tuple(definitions)


AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID = "autoresearch-control-smoke-v1"


def builtin_autoresearch_template_definitions() -> tuple[AutoResearchTemplateDefinition, ...]:
    """Return portable, source-managed templates with no machine-local material.

    The first built-in is deliberately a control-plane smoke template.  Its fake
    executor may prove orchestration and restart safety, but its evaluation and
    promotion contracts explicitly prohibit model-quality claims.
    """

    control = AutoResearchTemplateDefinition(
        template_id=AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID,
        objective=(
            "Exercise BashGym's durable baseline-first AutoResearch loop before "
            "binding an approved dataset, model, evaluator, and compute profile."
        ),
        target_model=TargetModelContract(
            target_contract_key="autoresearch-control-smoke-v1",
            base_model_ref="unconfigured://pin-a-real-base-model-before-quality-work",
            task="terminal-agent-autoresearch-control-smoke",
            representation_contract={"quality_claim_eligible": False},
        ),
        manifest=CampaignManifest(
            approved_data_scopes=("autoresearch-control-smoke",),
            compute_profile_id="autoresearch-control-smoke",
            budget_limits={"gpu_hours": 0.25, "study_count": 3.0},
            evaluation_plan={
                "schema_version": "autoresearch_evaluation_plan.v1",
                "primary_metric": "control_path_score",
                "metric_direction": "maximize",
                "ledger_project_id": "autoresearch-control-smoke-v1",
                "evaluation_suite_id": "autoresearch-control-smoke-v1",
                "baseline_required": True,
                "quality_claim": False,
            },
            promotion_gates={
                "requires_real_baseline": True,
                "quality_claim_eligible": False,
            },
            max_proposal_rounds=3,
        ),
        policy=AutoResearchTemplatePolicy(
            template_revision="1",
            primary_metric="control_path_score",
            metric_direction=MetricDirection.MAXIMIZE,
            stop_rules=AutoResearchStopRules(
                max_attempts=3,
                budget_unit="gpu_hours",
                max_total_cost=0.25,
                minimum_improvement=0.0,
            ),
            ledger_project_id="autoresearch-control-smoke-v1",
            evaluation_suite_id="autoresearch-control-smoke-v1",
            quality_claim_eligible=False,
        ),
    )
    return (
        control,
        *load_autoresearch_template_definitions(),
    )


def builtin_autoresearch_template_registry() -> dict[str, dict[str, Any]]:
    return build_autoresearch_template_registry(builtin_autoresearch_template_definitions())


def autoresearch_spec_for_template(
    template_id: str,
    *,
    workspace_id: str,
    campaign_id: str,
    stop_rules: AutoResearchStopRules,
    definitions: Iterable[AutoResearchTemplateDefinition] | None = None,
) -> AutoResearchCampaignSpec | None:
    """Materialize the durable policy paired with any registered definition."""

    values = tuple(definitions or builtin_autoresearch_template_definitions())
    for definition in values:
        if definition.template_id == template_id:
            return definition.materialize_spec(
                workspace_id,
                campaign_id,
                stop_rules=stop_rules,
            )
    return None


_AUTORESEARCH_MIGRATIONS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (
        1,
        "durable_autoresearch_control_loop",
        (
            """
            CREATE TABLE autoresearch_campaign_specs (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                spec_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, campaign_id),
                FOREIGN KEY(workspace_id, campaign_id)
                    REFERENCES campaigns(workspace_id, campaign_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE autoresearch_proposal_controls (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                role TEXT NOT NULL,
                parent_proposal_id TEXT,
                changed_variables_json TEXT NOT NULL,
                control_json TEXT NOT NULL,
                control_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, proposal_id),
                FOREIGN KEY(workspace_id, campaign_id)
                    REFERENCES autoresearch_campaign_specs(workspace_id, campaign_id)
                    ON DELETE RESTRICT,
                FOREIGN KEY(workspace_id, proposal_id)
                    REFERENCES campaign_proposals(workspace_id, proposal_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE autoresearch_results (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                result_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, result_id),
                UNIQUE(workspace_id, campaign_id, proposal_id),
                FOREIGN KEY(workspace_id, proposal_id)
                    REFERENCES autoresearch_proposal_controls(workspace_id, proposal_id)
                    ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_autoresearch_results_campaign ON autoresearch_results(workspace_id, campaign_id, created_at, result_id)",
        ),
    ),
    (
        2,
        "durable_autoresearch_diagnostics",
        (
            """
            CREATE TABLE autoresearch_diagnostic_results (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, campaign_id, proposal_id),
                FOREIGN KEY(workspace_id, proposal_id)
                    REFERENCES autoresearch_proposal_controls(workspace_id, proposal_id)
                    ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_autoresearch_diagnostics_campaign ON autoresearch_diagnostic_results(workspace_id, campaign_id, created_at, proposal_id)",
        ),
    ),
    (
        3,
        "durable_autoresearch_hypothesis_family_conclusions",
        (
            """
            CREATE TABLE autoresearch_hypothesis_family_conclusions (
                workspace_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                hypothesis_family_id TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                conclusion_json TEXT NOT NULL,
                conclusion_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(workspace_id, campaign_id, hypothesis_family_id),
                FOREIGN KEY(workspace_id, campaign_id)
                    REFERENCES autoresearch_campaign_specs(workspace_id, campaign_id)
                    ON DELETE RESTRICT
            )
            """,
            "CREATE INDEX idx_autoresearch_family_conclusions_campaign ON autoresearch_hypothesis_family_conclusions(workspace_id, campaign_id, created_at, hypothesis_family_id)",
        ),
    ),
)


class AutoResearchRepository(CampaignRuntimeRepository):
    """Campaign runtime plus a small immutable AutoResearch evidence projection."""

    def initialize(self) -> None:
        super().initialize()
        with self._connection(immediate=True) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS autoresearch_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """)
            for version, name, statements in _AUTORESEARCH_MIGRATIONS:
                checksum = canonical_hash(list(statements))
                row = connection.execute(
                    "SELECT name, checksum FROM autoresearch_schema_migrations WHERE version = ?",
                    (version,),
                ).fetchone()
                if row is not None:
                    if row["name"] != name or row["checksum"] != checksum:
                        raise MigrationChecksumError(
                            f"AutoResearch migration {version} checksum mismatch"
                        )
                    continue
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO autoresearch_schema_migrations(version, name, checksum, applied_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (version, name, checksum, utc_now().isoformat()),
                )

    def create_autoresearch_spec(self, spec: AutoResearchCampaignSpec) -> AutoResearchCampaignSpec:
        campaign = self.get_campaign(spec.workspace_id, spec.campaign_id)
        manifest = self.get_manifest_revision(
            spec.workspace_id, spec.campaign_id, campaign.manifest_revision
        ).manifest
        rules = spec.stop_rules
        if rules.max_attempts > manifest.max_proposal_rounds:
            raise AutoResearchInvariantError(
                "autoresearch_max_attempts_exceeds_campaign_proposal_rounds"
            )
        manifest_limit = manifest.budget_limits.get(rules.budget_unit)
        if manifest_limit is None:
            raise AutoResearchInvariantError("autoresearch_budget_unit_not_in_manifest")
        if rules.max_total_cost > manifest_limit:
            raise AutoResearchInvariantError("autoresearch_budget_exceeds_manifest")
        with self._connection(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT spec_json, spec_digest FROM autoresearch_campaign_specs
                WHERE workspace_id = ? AND campaign_id = ?
                """,
                (spec.workspace_id, spec.campaign_id),
            ).fetchone()
            if existing is not None:
                if existing["spec_digest"] != spec.spec_digest:
                    raise AutoResearchConflictError("autoresearch_spec_already_exists")
                return AutoResearchCampaignSpec.model_validate_json(existing["spec_json"])
            connection.execute(
                """
                INSERT INTO autoresearch_campaign_specs(
                    workspace_id, campaign_id, spec_json, spec_digest, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    spec.workspace_id,
                    spec.campaign_id,
                    spec.model_dump_json(),
                    spec.spec_digest,
                    spec.created_at.isoformat(),
                ),
            )
        return spec

    def get_autoresearch_spec(
        self, workspace_id: str, campaign_id: str
    ) -> AutoResearchCampaignSpec:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT spec_json FROM autoresearch_campaign_specs
                WHERE workspace_id = ? AND campaign_id = ?
                """,
                (workspace_id, campaign_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("AutoResearch campaign spec not found")
        return AutoResearchCampaignSpec.model_validate_json(row["spec_json"])

    def study_budget_usage(
        self,
        workspace_id: str,
        campaign_id: str,
        study_id: str,
        unit: str,
    ) -> dict[str, float]:
        """Derive one study's settled spend from the append-only campaign ledger."""

        self.get_study(workspace_id, campaign_id, study_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(b.reserved_delta), 0) AS reserved,
                       COALESCE(SUM(b.actual_delta), 0) AS actual
                FROM campaign_budget_ledger b
                JOIN campaign_actions a
                  ON a.workspace_id = b.workspace_id AND a.action_id = b.action_id
                WHERE a.workspace_id = ? AND a.campaign_id = ? AND a.study_id = ?
                  AND b.unit = ?
                """,
                (workspace_id, campaign_id, study_id, unit),
            ).fetchone()
        return {"reserved": float(row["reserved"]), "actual": float(row["actual"])}

    def list_study_attempts(
        self, workspace_id: str, campaign_id: str, study_id: str
    ) -> tuple[ActionAttempt, ...]:
        """Return only attempts durably owned by one exact campaign study."""

        self.get_study(workspace_id, campaign_id, study_id)
        with self._connection() as connection:
            rows = connection.execute(
                self._attempt_select() + """
                  WHERE a.workspace_id = ? AND a.campaign_id = ? AND a.study_id = ?
                  ORDER BY a.stage_index, a.action_id, t.attempt_number
                  """,
                (workspace_id, campaign_id, study_id),
            ).fetchall()
        return tuple(self._attempt_from_row(row) for row in rows)

    def list_action_artifacts(
        self, workspace_id: str, campaign_id: str, action_id: str
    ) -> tuple[CampaignArtifactRecord, ...]:
        """Return artifacts produced by one exact campaign action."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM campaign_artifacts
                WHERE workspace_id = ? AND campaign_id = ? AND producer_action_id = ?
                ORDER BY created_at, artifact_id
                """,
                (workspace_id, campaign_id, action_id),
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

    def register_autoresearch_proposal(
        self, control: AutoResearchProposalControl
    ) -> AutoResearchProposalControl:
        self.get_autoresearch_spec(control.workspace_id, control.campaign_id)
        with self._connection(immediate=True) as connection:
            proposal = connection.execute(
                """
                SELECT campaign_id FROM campaign_proposals
                WHERE workspace_id = ? AND proposal_id = ?
                """,
                (control.workspace_id, control.proposal_id),
            ).fetchone()
            if proposal is None or proposal["campaign_id"] != control.campaign_id:
                raise RecordNotFoundError("AutoResearch proposal not found")
            existing = connection.execute(
                """
                SELECT control_json, control_digest FROM autoresearch_proposal_controls
                WHERE workspace_id = ? AND proposal_id = ?
                """,
                (control.workspace_id, control.proposal_id),
            ).fetchone()
            if existing is not None:
                if existing["control_digest"] != control.control_digest:
                    raise AutoResearchConflictError("autoresearch_proposal_control_conflict")
                return AutoResearchProposalControl.model_validate_json(existing["control_json"])
            connection.execute(
                """
                INSERT INTO autoresearch_proposal_controls(
                    workspace_id, campaign_id, proposal_id, role, parent_proposal_id,
                    changed_variables_json, control_json, control_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    control.workspace_id,
                    control.campaign_id,
                    control.proposal_id,
                    control.role.value,
                    control.parent_proposal_id,
                    json.dumps(control.changed_variables, separators=(",", ":")),
                    control.model_dump_json(),
                    control.control_digest,
                    control.created_at.isoformat(),
                ),
            )
        return control

    def get_autoresearch_proposal(
        self, workspace_id: str, campaign_id: str, proposal_id: str
    ) -> AutoResearchProposalControl:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT control_json FROM autoresearch_proposal_controls
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (workspace_id, campaign_id, proposal_id),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("AutoResearch proposal control not found")
        return AutoResearchProposalControl.model_validate_json(row["control_json"])

    def list_autoresearch_proposals(
        self, workspace_id: str, campaign_id: str
    ) -> tuple[AutoResearchProposalControl, ...]:
        self.get_autoresearch_spec(workspace_id, campaign_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT control_json FROM autoresearch_proposal_controls
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, proposal_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(
            AutoResearchProposalControl.model_validate_json(row["control_json"]) for row in rows
        )

    def list_autoresearch_outcomes(
        self, workspace_id: str, campaign_id: str
    ) -> tuple[AutoResearchOutcomeRecord, ...]:
        self.get_autoresearch_spec(workspace_id, campaign_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT result_json, decision_json FROM autoresearch_results
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, result_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(
            AutoResearchOutcomeRecord(
                result=AutoResearchResult.model_validate_json(row["result_json"]),
                decision=AutoResearchDecision.model_validate_json(row["decision_json"]),
            )
            for row in rows
        )

    def get_hypothesis_family_conclusion(
        self,
        workspace_id: str,
        campaign_id: str,
        hypothesis_family_id: str,
    ) -> AutoResearchHypothesisFamilyConclusion | None:
        self.get_autoresearch_spec(workspace_id, campaign_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT conclusion_json
                FROM autoresearch_hypothesis_family_conclusions
                WHERE workspace_id = ? AND campaign_id = ? AND hypothesis_family_id = ?
                """,
                (workspace_id, campaign_id, hypothesis_family_id),
            ).fetchone()
        return (
            AutoResearchHypothesisFamilyConclusion.model_validate_json(row["conclusion_json"])
            if row is not None
            else None
        )

    def list_hypothesis_family_conclusions(
        self, workspace_id: str, campaign_id: str
    ) -> tuple[AutoResearchHypothesisFamilyConclusion, ...]:
        self.get_autoresearch_spec(workspace_id, campaign_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT conclusion_json
                FROM autoresearch_hypothesis_family_conclusions
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, hypothesis_family_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(
            AutoResearchHypothesisFamilyConclusion.model_validate_json(row["conclusion_json"])
            for row in rows
        )

    def conclude_hypothesis_family(
        self,
        workspace_id: str,
        campaign_id: str,
        hypothesis_family_id: str,
        *,
        disposition: HypothesisFamilyDisposition,
        summary: str,
        follow_up_family_id: str | None,
        follow_up_hypothesis: str | None,
        expected_version: int,
        actor_id: str,
        credential_kind: CredentialKind,
        correlation_id: str,
        idempotency_key: str,
    ) -> AutoResearchHypothesisFamilyConclusion:
        """Close one completed family and wake observers without choosing new work."""

        request_digest = canonical_hash(
            {
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "hypothesis_family_id": hypothesis_family_id,
                "disposition": disposition.value,
                "summary": summary,
                "follow_up_family_id": follow_up_family_id,
                "follow_up_hypothesis": follow_up_hypothesis,
            }
        )
        concluded_at = utc_now()
        with self._connection(immediate=True) as connection:
            existing = connection.execute(
                """
                SELECT request_digest, conclusion_json
                FROM autoresearch_hypothesis_family_conclusions
                WHERE workspace_id = ? AND campaign_id = ? AND hypothesis_family_id = ?
                """,
                (workspace_id, campaign_id, hypothesis_family_id),
            ).fetchone()
            if existing is not None:
                if existing["request_digest"] != request_digest:
                    raise AutoResearchConflictError(
                        "autoresearch_hypothesis_family_conclusion_conflict"
                    )
                stored = AutoResearchHypothesisFamilyConclusion.model_validate_json(
                    existing["conclusion_json"]
                )
                return stored.model_copy(update={"replayed": True})

            campaign_row = connection.execute(
                """
                SELECT * FROM campaigns
                WHERE workspace_id = ? AND campaign_id = ?
                """,
                (workspace_id, campaign_id),
            ).fetchone()
            if campaign_row is None:
                raise RecordNotFoundError("campaign not found")
            campaign = self._campaign_from_row(campaign_row)
            if campaign.version != expected_version:
                raise RevisionConflictError(expected_version, campaign.version)

            rows = connection.execute(
                """
                SELECT control_json
                FROM autoresearch_proposal_controls
                WHERE workspace_id = ? AND campaign_id = ? AND role = ?
                ORDER BY created_at, proposal_id
                """,
                (workspace_id, campaign_id, ExperimentRole.CANDIDATE.value),
            ).fetchall()
            controls = tuple(
                control
                for control in (
                    AutoResearchProposalControl.model_validate_json(row["control_json"])
                    for row in rows
                )
                if control.hypothesis_family_id == hypothesis_family_id
            )
            if not controls:
                raise AutoResearchInvariantError("autoresearch_hypothesis_family_not_found")
            if len(controls) > 32:
                raise AutoResearchInvariantError("autoresearch_hypothesis_family_too_large")

            proposal_ids: list[str] = []
            result_ids: list[str] = []
            for control in controls:
                result_row = connection.execute(
                    """
                    SELECT result_json FROM autoresearch_results
                    WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                    """,
                    (workspace_id, campaign_id, control.proposal_id),
                ).fetchone()
                if result_row is None:
                    raise AutoResearchInvariantError(
                        "autoresearch_hypothesis_family_has_pending_results"
                    )
                result = AutoResearchResult.model_validate_json(result_row["result_json"])
                proposal_ids.append(control.proposal_id)
                result_ids.append(result.result_id)

            aggregate_version = campaign.version + 1
            conclusion = AutoResearchHypothesisFamilyConclusion(
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                hypothesis_family_id=hypothesis_family_id,
                disposition=disposition,
                summary=summary,
                proposal_ids=tuple(proposal_ids),
                result_ids=tuple(result_ids),
                follow_up_family_id=follow_up_family_id,
                follow_up_hypothesis=follow_up_hypothesis,
                aggregate_version=aggregate_version,
                created_at=concluded_at,
            )
            connection.execute(
                """
                INSERT INTO autoresearch_hypothesis_family_conclusions(
                    workspace_id, campaign_id, hypothesis_family_id, request_digest,
                    conclusion_json, conclusion_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    campaign_id,
                    hypothesis_family_id,
                    request_digest,
                    conclusion.model_dump_json(),
                    conclusion.conclusion_digest,
                    conclusion.created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE campaigns SET version = ?, updated_at = ?
                WHERE workspace_id = ? AND campaign_id = ? AND version = ?
                """,
                (
                    aggregate_version,
                    conclusion.created_at.isoformat(),
                    workspace_id,
                    campaign_id,
                    expected_version,
                ),
            )
            event = CampaignEvent(
                event_id=f"evt-{request_digest[:24]}",
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                sequence=self._next_event_sequence(connection, workspace_id, campaign_id),
                aggregate_version=aggregate_version,
                event_type="campaign:autoresearch-family-concluded",
                payload={
                    "hypothesis_family_id": hypothesis_family_id,
                    "disposition": disposition.value,
                    "conclusion_digest": conclusion.conclusion_digest,
                    "follow_up_family_id": follow_up_family_id,
                },
                actor_id=actor_id,
                credential_kind=credential_kind,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                created_at=conclusion.created_at,
            )
            self._insert_event(connection, event)
        return conclusion

    def list_autoresearch_diagnostic_results(
        self, workspace_id: str, campaign_id: str
    ) -> tuple[AutoResearchDiagnosticResult, ...]:
        self.get_autoresearch_spec(workspace_id, campaign_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT result_json FROM autoresearch_diagnostic_results
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, proposal_id
                """,
                (workspace_id, campaign_id),
            ).fetchall()
        return tuple(
            AutoResearchDiagnosticResult.model_validate_json(row["result_json"]) for row in rows
        )

    def record_autoresearch_diagnostic_result(
        self, result: AutoResearchDiagnosticResult
    ) -> AutoResearchDiagnosticResult:
        """Persist one authenticated aggregate diagnostic projection idempotently."""

        self.get_autoresearch_spec(result.workspace_id, result.campaign_id)
        canonical = result.model_copy(update={"replayed": False})
        with self._connection(immediate=True) as connection:
            control_row = connection.execute(
                """
                SELECT control_json FROM autoresearch_proposal_controls
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (result.workspace_id, result.campaign_id, result.proposal_id),
            ).fetchone()
            if control_row is None:
                raise RecordNotFoundError("AutoResearch proposal control not found")
            control = AutoResearchProposalControl.model_validate_json(control_row["control_json"])
            if control.role != ExperimentRole.DIAGNOSTIC:
                raise AutoResearchInvariantError("autoresearch_diagnostic_role_mismatch")
            existing = connection.execute(
                """
                SELECT result_json, result_digest FROM autoresearch_diagnostic_results
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (result.workspace_id, result.campaign_id, result.proposal_id),
            ).fetchone()
            if existing is not None:
                stored = AutoResearchDiagnosticResult.model_validate_json(existing["result_json"])
                if (
                    existing["result_digest"] != canonical.result_digest
                    or stored.result_digest != canonical.result_digest
                    or stored != canonical
                ):
                    raise AutoResearchConflictError("autoresearch_diagnostic_result_conflict")
                return stored.model_copy(update={"replayed": True})
            connection.execute(
                """
                INSERT INTO autoresearch_diagnostic_results(
                    workspace_id, campaign_id, proposal_id, result_json,
                    result_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical.workspace_id,
                    canonical.campaign_id,
                    canonical.proposal_id,
                    canonical.model_dump_json(),
                    canonical.result_digest,
                    canonical.recorded_at.isoformat(),
                ),
            )
        return canonical

    @staticmethod
    def _improvement(direction: MetricDirection, incumbent: float, candidate: float) -> float:
        return (
            candidate - incumbent
            if direction == MetricDirection.MAXIMIZE
            else incumbent - candidate
        )

    @classmethod
    def _protected_metric_margins(
        cls,
        gates: tuple[ProtectedMetricGate, ...],
        incumbent: AutoResearchResult,
        candidate: AutoResearchResult,
    ) -> dict[str, float]:
        """Headroom per protected metric; negative means the gate is breached."""

        margins: dict[str, float] = {}
        for gate in gates:
            previous = incumbent.metrics.get(gate.metric_name)
            current = candidate.metrics.get(gate.metric_name)
            if previous is None or current is None:
                continue
            regression = -cls._improvement(gate.direction, previous, current)
            margins[gate.metric_name] = gate.max_regression - regression
        return margins

    @classmethod
    def _protected_metric_failure(
        cls,
        gates: tuple[ProtectedMetricGate, ...],
        incumbent: AutoResearchResult,
        candidate: AutoResearchResult,
    ) -> str | None:
        margins = cls._protected_metric_margins(gates, incumbent, candidate)
        for gate in gates:
            margin = margins.get(gate.metric_name)
            if margin is None or margin < 0:
                return gate.metric_name
        return None

    def _record_outcome_ledger_in_connection(
        self,
        connection: Any,
        spec: AutoResearchCampaignSpec,
        outcome: AutoResearchOutcomeRecord,
        context: AutoResearchLedgerCommitContext,
        *,
        result_digest: str,
    ) -> None:
        """Mirror one outcome into the ledger under the results row's digest of record."""

        result = outcome.result
        decision = outcome.decision
        if spec.ledger_project_id != context.project_id:
            raise AutoResearchInvariantError("autoresearch_ledger_project_mismatch")
        run = connection.execute(
            """SELECT experiment_id, campaign_id FROM ledger_runs
               WHERE workspace_id = ? AND project_id = ? AND run_id = ?""",
            (result.workspace_id, context.project_id, context.run_id),
        ).fetchone()
        if run is None:
            raise RecordNotFoundError("ledger run not found")
        if run["experiment_id"] != context.experiment_id:
            raise AutoResearchInvariantError("autoresearch_ledger_experiment_mismatch")
        if run["campaign_id"] != result.campaign_id:
            raise AutoResearchInvariantError("autoresearch_run_campaign_lineage_mismatch")

        evidence_refs = tuple(dict.fromkeys((result.result_id, *result.evidence_references)))
        ledger_decision = DecisionSpec(
            workspace_id=result.workspace_id,
            project_id=context.project_id,
            decision_id=stable_ledger_id(
                "autoresearch-decision",
                result.workspace_id,
                result.campaign_id,
                result.result_id,
            ),
            experiment_id=context.experiment_id,
            run_id=context.run_id,
            decision_type="autoresearch_outcome",
            outcome=decision.decision.value,
            rationale=(
                f"{decision.reason_code}; primary_metric={result.metric_name}; "
                f"metric_value={result.metric_value}; improvement={decision.improvement}; "
                f"actual_cost={result.actual_cost}"
            ),
            evidence_refs=evidence_refs,
            actor_id=context.actor_id,
            created_at=decision.decided_at,
        )
        ledger_event = LedgerEventSpec(
            workspace_id=result.workspace_id,
            project_id=context.project_id,
            event_type="autoresearch_outcome_recorded",
            source_system="bashgym",
            source_event_id=stable_ledger_id(
                "autoresearch-outcome",
                result.workspace_id,
                result.campaign_id,
                result.result_id,
            ),
            correlation_id=context.correlation_id,
            experiment_id=context.experiment_id,
            run_id=context.run_id,
            attempt_id=context.attempt_id,
            payload={
                "campaign_id": result.campaign_id,
                "proposal_id": result.proposal_id,
                "study_id": result.study_id,
                "result_id": result.result_id,
                "result_digest": result_digest,
                "decision": decision.decision.value,
                "reason_code": decision.reason_code,
                "eligible_for_best": decision.eligible_for_best,
                "metric_name": result.metric_name,
                "metric_value": result.metric_value,
                "improvement": decision.improvement,
                "actual_cost": result.actual_cost,
            },
            created_at=decision.decided_at,
        )
        self._record_decision_in_connection(connection, ledger_decision)
        self._append_event_in_connection(connection, ledger_event)

    def _record_decision_in_connection(
        self, connection: Any, spec: DecisionSpec
    ) -> tuple[dict[str, Any], bool]:
        ledger = ExperimentLedgerRepository(self.db_path)
        return ledger._record_decision_in_connection(connection, spec)

    def _append_event_in_connection(
        self, connection: Any, spec: LedgerEventSpec
    ) -> tuple[dict[str, Any], bool]:
        ledger = ExperimentLedgerRepository(self.db_path)
        return ledger._append_event_in_connection(connection, spec)

    def record_autoresearch_result(
        self,
        result: AutoResearchResult,
        *,
        ledger_context: AutoResearchLedgerCommitContext | None = None,
    ) -> AutoResearchOutcomeRecord:
        """Persist non-authoritative simulated/crashed outcomes from public callers."""

        if (
            result.provenance == ExperimentProvenance.REAL
            and result.outcome == ExperimentOutcome.COMPLETED
        ):
            raise AutoResearchInvariantError("autoresearch_real_result_requires_sealed_projection")
        return self._record_autoresearch_result(result, ledger_context=ledger_context)

    def _record_autoresearch_result(
        self,
        result: AutoResearchResult,
        *,
        ledger_context: AutoResearchLedgerCommitContext | None = None,
    ) -> AutoResearchOutcomeRecord:
        """Internal commit path; completed REAL requires authoritative ledger lineage."""

        if (
            result.provenance == ExperimentProvenance.REAL
            and result.outcome == ExperimentOutcome.COMPLETED
            and ledger_context is None
        ):
            raise AutoResearchInvariantError("autoresearch_real_result_requires_sealed_projection")
        if len(result.attempt_ids) > 100 or len(result.evidence_references) > 100:
            raise AutoResearchInvariantError("autoresearch_result_reference_limit_exceeded")
        spec = self.get_autoresearch_spec(result.workspace_id, result.campaign_id)
        with self._connection(immediate=True) as connection:
            by_proposal = connection.execute(
                """
                SELECT result_json, result_digest, decision_json FROM autoresearch_results
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (result.workspace_id, result.campaign_id, result.proposal_id),
            ).fetchone()
            by_id = connection.execute(
                """
                SELECT proposal_id, result_digest FROM autoresearch_results
                WHERE workspace_id = ? AND result_id = ?
                """,
                (result.workspace_id, result.result_id),
            ).fetchone()
            stored_result: AutoResearchResult | None = None
            stored_decision: AutoResearchDecision | None = None
            digest_of_record = result.result_digest
            if by_proposal is not None:
                try:
                    stored_result = AutoResearchResult.model_validate_json(
                        by_proposal["result_json"]
                    )
                    stored_decision = AutoResearchDecision.model_validate_json(
                        by_proposal["decision_json"]
                    )
                except (TypeError, ValueError) as exc:
                    raise AutoResearchConflictError("autoresearch_result_conflict") from exc
                stored_digest = by_proposal["result_digest"]
                if (
                    not result.digest_matches(stored_digest)
                    or not stored_result.digest_matches(stored_digest)
                    or stored_result != result
                ):
                    raise AutoResearchConflictError("autoresearch_result_conflict")
                digest_of_record = stored_digest
            elif by_id is not None:
                raise AutoResearchConflictError("autoresearch_result_id_conflict")

            control_row = connection.execute(
                """
                SELECT control_json FROM autoresearch_proposal_controls
                WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                """,
                (result.workspace_id, result.campaign_id, result.proposal_id),
            ).fetchone()
            if control_row is None:
                raise RecordNotFoundError("AutoResearch proposal control not found")
            control = AutoResearchProposalControl.model_validate_json(control_row["control_json"])
            if result.role != control.role:
                raise AutoResearchInvariantError("autoresearch_result_role_mismatch")

            if control.role == ExperimentRole.CANDIDATE:
                parent_row = connection.execute(
                    """
                    SELECT result_json, decision_json FROM autoresearch_results
                    WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
                    """,
                    (
                        result.workspace_id,
                        result.campaign_id,
                        control.parent_proposal_id,
                    ),
                ).fetchone()
                if parent_row is None:
                    raise AutoResearchInvariantError("autoresearch_exact_parent_outcome_required")
                parent = AutoResearchOutcomeRecord(
                    result=AutoResearchResult.model_validate_json(parent_row["result_json"]),
                    decision=AutoResearchDecision.model_validate_json(parent_row["decision_json"]),
                )
                if (
                    parent.result.provenance != ExperimentProvenance.REAL
                    or parent.result.outcome != ExperimentOutcome.COMPLETED
                    or parent.result.metric_value is None
                    or parent.decision.decision
                    not in {ResultDecision.BASELINE, ResultDecision.KEEP, ResultDecision.DISCARD}
                ):
                    raise AutoResearchInvariantError("autoresearch_exact_parent_outcome_required")
                reference_row = connection.execute(
                    """
                    SELECT result_json, decision_json FROM autoresearch_results
                    WHERE workspace_id = ? AND campaign_id = ? AND proposal_id != ?
                      AND json_extract(decision_json, '$.eligible_for_best') = 1
                    ORDER BY created_at DESC, result_id DESC
                    LIMIT 1
                    """,
                    (
                        result.workspace_id,
                        result.campaign_id,
                        result.proposal_id,
                    ),
                ).fetchone()
                if reference_row is None:
                    raise AutoResearchInvariantError("autoresearch_real_baseline_required")
                incumbent = AutoResearchOutcomeRecord(
                    result=AutoResearchResult.model_validate_json(reference_row["result_json"]),
                    decision=AutoResearchDecision.model_validate_json(
                        reference_row["decision_json"]
                    ),
                )
            else:
                parent = None
                incumbent = None
            previous_id = incumbent.result.proposal_id if incumbent else None
            previous_metric = incumbent.result.metric_value if incumbent else None

            improvement: float | None = None
            protected_margins: dict[str, float] = {}
            if result.outcome == ExperimentOutcome.CRASHED:
                choice = ResultDecision.CRASH
                reason = "experiment_crashed"
                is_eligible = False
            elif result.provenance == ExperimentProvenance.SIMULATED:
                choice = ResultDecision.INELIGIBLE
                reason = "simulated_result_not_quality_evidence"
                is_eligible = False
            elif result.role == ExperimentRole.BASELINE:
                baseline_exists = connection.execute(
                    """
                    SELECT 1 FROM autoresearch_results
                    WHERE workspace_id = ? AND campaign_id = ?
                      AND proposal_id != ?
                      AND json_extract(decision_json, '$.decision') = ?
                    LIMIT 1
                    """,
                    (
                        result.workspace_id,
                        result.campaign_id,
                        result.proposal_id,
                        ResultDecision.BASELINE.value,
                    ),
                ).fetchone()
                if baseline_exists is not None:
                    raise AutoResearchInvariantError("autoresearch_baseline_already_verified")
                choice = ResultDecision.BASELINE
                reason = "real_baseline_verified"
                is_eligible = True
            else:
                if incumbent is None or previous_metric is None or result.metric_value is None:
                    raise AutoResearchInvariantError("autoresearch_real_baseline_required")
                improvement = self._improvement(
                    spec.metric_direction, previous_metric, result.metric_value
                )
                threshold = spec.stop_rules.minimum_improvement
                clears_primary = improvement > 0 and improvement >= threshold
                protected_failure = self._protected_metric_failure(
                    spec.stop_rules.protected_metrics,
                    incumbent.result,
                    result,
                )
                protected_margins = self._protected_metric_margins(
                    spec.stop_rules.protected_metrics,
                    incumbent.result,
                    result,
                )
                improved = clears_primary and protected_failure is None
                choice = ResultDecision.KEEP if improved else ResultDecision.DISCARD
                if protected_failure is not None:
                    reason = "candidate_failed_protected_metric_gate"
                elif improved:
                    reason = "candidate_improved_primary_metric"
                else:
                    reason = "candidate_did_not_clear_improvement_gate"
                is_eligible = improved

            decision = AutoResearchDecision(
                proposal_id=result.proposal_id,
                decision=choice,
                reason_code=reason,
                eligible_for_best=is_eligible,
                previous_best_proposal_id=previous_id,
                previous_best_metric=previous_metric,
                improvement=improvement,
                protected_metric_margins=protected_margins,
                result_digest=digest_of_record,
                decided_at=result.recorded_at,
            )
            if stored_result is not None and stored_decision is not None:
                if stored_decision.model_dump(
                    mode="json", exclude={"protected_metric_margins"}
                ) != decision.model_dump(mode="json", exclude={"protected_metric_margins"}):
                    raise AutoResearchConflictError("autoresearch_result_conflict")
                outcome = AutoResearchOutcomeRecord(
                    result=stored_result,
                    decision=stored_decision,
                    replayed=True,
                )
                if ledger_context is not None:
                    self._record_outcome_ledger_in_connection(
                        connection,
                        spec,
                        outcome,
                        ledger_context,
                        result_digest=digest_of_record,
                    )
                return outcome
            connection.execute(
                """
                INSERT INTO autoresearch_results(
                    workspace_id, campaign_id, result_id, proposal_id, result_json,
                    result_digest, decision_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.workspace_id,
                    result.campaign_id,
                    result.result_id,
                    result.proposal_id,
                    result.model_dump_json(),
                    result.result_digest,
                    decision.model_dump_json(),
                    result.recorded_at.isoformat(),
                ),
            )
            outcome = AutoResearchOutcomeRecord(result=result, decision=decision)
            if ledger_context is not None:
                self._record_outcome_ledger_in_connection(
                    connection,
                    spec,
                    outcome,
                    ledger_context,
                    result_digest=digest_of_record,
                )
        return outcome


class AutoResearchCampaignCore:
    """Controller-facing orchestration policy over durable campaign primitives."""

    _SUCCESS_STUDY_STATES = frozenset(
        {
            StudyStatus.COMPLETED,
            StudyStatus.DEVELOPMENT_PASSED,
            StudyStatus.REJECTED,
            StudyStatus.RECIPE_LOCKED,
            StudyStatus.PROMOTED,
            StudyStatus.FINAL_REJECTED,
        }
    )
    _FAILED_STUDY_STATES = frozenset(
        {
            StudyStatus.EXECUTION_FAILED,
            StudyStatus.ABANDONED,
            StudyStatus.CANCELLED,
        }
    )

    def __init__(
        self,
        repository: AutoResearchRepository,
        *,
        evaluation_reader: Any | None = None,
    ):
        self.repository = repository
        self.service = CampaignService(repository)
        self.ledger = ExperimentLedgerRepository(repository.db_path)
        self.ledger.initialize()
        self.evaluation_reader = evaluation_reader

    def register(self, spec: AutoResearchCampaignSpec) -> AutoResearchCampaignSpec:
        return self.repository.create_autoresearch_spec(spec)

    def prepare(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        controller_id: str,
        correlation_id: str,
        idempotency_prefix: str,
    ) -> Campaign:
        """Controller-owned deterministic validation ending at the actor START gate."""

        spec = self.repository.get_autoresearch_spec(workspace_id, campaign_id)
        self.repository.create_autoresearch_spec(spec)  # Re-run manifest compatibility checks.
        while True:
            campaign = self.repository.get_campaign(workspace_id, campaign_id)
            if campaign.status in {CampaignStatus.READY, CampaignStatus.ACTIVE}:
                return campaign
            if campaign.status == CampaignStatus.DRAFT:
                trigger, suffix = CampaignTrigger.VALIDATE, "validate"
            elif campaign.status == CampaignStatus.VALIDATING:
                trigger, suffix = CampaignTrigger.VALIDATION_PASSED, "validated"
            else:
                raise AutoResearchInvariantError(
                    f"autoresearch_campaign_not_preparable:{campaign.status.value}"
                )
            self.repository.transition_campaign(
                workspace_id,
                campaign_id,
                trigger,
                expected_version=campaign.version,
                actor_id=controller_id,
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=correlation_id,
                idempotency_key=f"{idempotency_prefix}-{suffix}",
                payload={"control_plane": "autoresearch.v1"},
            )

    @staticmethod
    def _target_reached(spec: AutoResearchCampaignSpec, metric: float | None) -> bool:
        target = spec.stop_rules.target_metric
        if target is None or metric is None:
            return False
        if spec.metric_direction == MetricDirection.MAXIMIZE:
            return metric >= target
        return metric <= target

    def state(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        now: datetime | None = None,
    ) -> AutoResearchState:
        spec = self.repository.get_autoresearch_spec(workspace_id, campaign_id)
        campaign = self.repository.get_campaign(workspace_id, campaign_id)
        manifest = self.repository.get_manifest_revision(
            workspace_id, campaign_id, campaign.manifest_revision
        ).manifest
        controls = self.repository.list_autoresearch_proposals(workspace_id, campaign_id)
        outcomes = self.repository.list_autoresearch_outcomes(workspace_id, campaign_id)
        diagnostics = self.repository.list_autoresearch_diagnostic_results(
            workspace_id, campaign_id
        )
        outcome_by_proposal = {item.result.proposal_id: item for item in outcomes}
        diagnostic_by_proposal = {item.proposal_id: item for item in diagnostics}
        pending = [
            item
            for item in controls
            if item.proposal_id not in outcome_by_proposal
            and item.proposal_id not in diagnostic_by_proposal
        ]
        controlled_ids = {item.proposal_id for item in controls}
        untracked = [
            record.proposal.proposal_id
            for record in self.repository.list_proposals(workspace_id, campaign_id)
            if record.proposal.status in {ProposalStatus.SUBMITTED, ProposalStatus.ACCEPTED}
            and record.proposal.proposal_id not in controlled_ids
        ]
        eligible = [item for item in outcomes if item.decision.eligible_for_best]
        best = eligible[-1] if eligible else None
        baseline_verified = any(
            item.decision.decision == ResultDecision.BASELINE for item in outcomes
        )
        budget_used = sum(item.result.actual_cost for item in outcomes) + sum(
            item.actual_cost for item in diagnostics
        )
        scientific_attempts = sum(1 for item in outcomes if counts_as_experiment(item.result))
        manifest_remaining = self.repository.build_evidence_snapshot(
            workspace_id, campaign_id
        ).budget_remaining[spec.stop_rules.budget_unit]
        budget_remaining = min(
            spec.stop_rules.max_total_cost - budget_used,
            manifest_remaining,
        )
        current_time = now or datetime.now(UTC)
        next_action = AutoResearchNextAction.BLOCKED
        reason = "autoresearch_blocked"

        if campaign.status in {CampaignStatus.DRAFT, CampaignStatus.VALIDATING}:
            next_action, reason = (
                AutoResearchNextAction.PREPARE_CAMPAIGN,
                "campaign_requires_controller_preparation",
            )
        elif campaign.status == CampaignStatus.READY:
            next_action, reason = (
                AutoResearchNextAction.START_CAMPAIGN,
                "campaign_requires_authorized_start",
            )
        elif campaign.status in TERMINAL_CAMPAIGN_STATES:
            next_action, reason = (
                AutoResearchNextAction.STOP,
                (campaign.stop_reason or f"campaign_{campaign.status.value}"),
            )
        elif campaign.status != CampaignStatus.ACTIVE:
            next_action, reason = (
                AutoResearchNextAction.BLOCKED,
                f"campaign_{campaign.status.value}",
            )
        elif untracked:
            next_action, reason = (
                AutoResearchNextAction.BLOCKED,
                "untracked_campaign_proposal_requires_reconciliation",
            )
        elif pending or campaign.active_study_id or campaign.active_action_id:
            next_action, reason = (
                AutoResearchNextAction.WAIT_FOR_RESULT,
                "experiment_result_pending",
            )
        elif spec.stop_rules.deadline is not None and current_time >= spec.stop_rules.deadline:
            next_action, reason = AutoResearchNextAction.STOP, "deadline_reached"
        elif scientific_attempts >= spec.stop_rules.max_attempts:
            next_action, reason = AutoResearchNextAction.STOP, "attempt_limit_reached"
        elif budget_remaining <= 0:
            next_action, reason = AutoResearchNextAction.STOP, "budget_exhausted"
        elif self._target_reached(spec, best.result.metric_value if best else None):
            next_action, reason = AutoResearchNextAction.STOP, "target_metric_reached"
        elif len(self.repository.list_proposals(workspace_id, campaign_id)) >= (
            manifest.max_proposal_rounds
        ):
            next_action, reason = AutoResearchNextAction.STOP, "proposal_round_limit_reached"
        elif not baseline_verified:
            next_action, reason = (
                AutoResearchNextAction.SUBMIT_BASELINE,
                "real_baseline_required",
            )
        else:
            next_action, reason = (
                AutoResearchNextAction.PROPOSE_CANDIDATE,
                (
                    "diagnostic_evidence_ready"
                    if diagnostics
                    and diagnostics[-1].recorded_at >= outcomes[-1].result.recorded_at
                    else "ready_for_controlled_hypothesis"
                ),
            )

        latest = outcomes[-1].decision.decision if outcomes else None
        best_study_id = best.result.study_id if best else None
        return AutoResearchState(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            campaign_status=campaign.status,
            next_action=next_action,
            ready_for_next_proposal=next_action
            in {
                AutoResearchNextAction.SUBMIT_BASELINE,
                AutoResearchNextAction.PROPOSE_CANDIDATE,
            },
            reason_code=reason,
            baseline_verified=baseline_verified,
            pending_proposal_id=pending[0].proposal_id if pending else None,
            best_proposal_id=best.result.proposal_id if best else None,
            best_study_id=best_study_id,
            best_metric=best.result.metric_value if best else None,
            attempts_used=scientific_attempts,
            proposals_used=len(controls),
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            latest_decision=latest,
        )

    def failures(self, workspace_id: str, campaign_id: str) -> dict[str, Any]:
        """Compare evaluator-authored failure categories for the latest decision."""

        spec = self.repository.get_autoresearch_spec(workspace_id, campaign_id)
        outcomes = self.repository.list_autoresearch_outcomes(workspace_id, campaign_id)
        reference = None
        candidate = None
        if outcomes:
            latest = outcomes[-1]
            if latest.result.role == ExperimentRole.BASELINE:
                reference = latest
            else:
                candidate = latest
                previous_id = latest.decision.previous_best_proposal_id
                reference = next(
                    (
                        item
                        for item in reversed(outcomes[:-1])
                        if item.result.proposal_id == previous_id
                    ),
                    None,
                )
        evaluations: list[dict[str, Any]] = []
        if spec.ledger_project_id is not None:
            try:
                evaluations = self.ledger.list_evaluation_results(
                    workspace_id,
                    spec.ledger_project_id,
                    limit=1000,
                )
            except RecordNotFoundError:
                pass
        return build_research_failure_packet(
            campaign_id=campaign_id,
            reference_outcome=(reference.model_dump(mode="json") if reference else None),
            candidate_outcome=(candidate.model_dump(mode="json") if candidate else None),
            evaluations=evaluations,
        )

    def diagnostics(
        self,
        workspace_id: str,
        campaign_id: str,
    ) -> AutoResearchDiagnostics:
        """Derive advisory diagnostics from immutable campaign and ledger evidence."""

        spec = self.repository.get_autoresearch_spec(workspace_id, campaign_id)
        outcomes = self.repository.list_autoresearch_outcomes(workspace_id, campaign_id)
        evaluations: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        dataset_versions: list[dict[str, Any]] = []
        training_metrics: list[dict[str, Any]] = []
        if spec.ledger_project_id is not None:
            try:
                evaluations = self.ledger.list_evaluation_results(
                    workspace_id,
                    spec.ledger_project_id,
                    limit=1000,
                )
                runs = [
                    run
                    for run in self.ledger.list_runs(
                        workspace_id,
                        spec.ledger_project_id,
                        limit=1000,
                    )
                    if run.get("campaign_id") == campaign_id
                ]
                dataset_versions = self.ledger.list_dataset_versions(
                    workspace_id,
                    spec.ledger_project_id,
                )
                attempt_ids = {
                    attempt_id for outcome in outcomes for attempt_id in outcome.result.attempt_ids
                }
                for attempt_id in sorted(attempt_ids):
                    for metric_name in ("train_loss", "validation_loss"):
                        training_metrics.extend(
                            {
                                "attempt_id": attempt_id,
                                "metric_name": metric_name,
                                "step": point.step,
                                "value": point.value,
                            }
                            for point in self.repository.get_metric_series(
                                workspace_id,
                                attempt_id,
                                metric_name,
                            )
                        )
            except RecordNotFoundError:
                # A registered campaign can precede materialization of its logical
                # ledger binding; diagnostics remain an empty advisory projection.
                pass
        return build_autoresearch_diagnostics(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            primary_metric=spec.primary_metric,
            metric_direction=spec.metric_direction.value,
            evaluation_suite_id=spec.evaluation_suite_id,
            outcomes=[item.model_dump(mode="json") for item in outcomes],
            evaluations=evaluations,
            runs=runs,
            dataset_versions=dataset_versions,
            training_metrics=training_metrics,
        )

    def _submit(
        self,
        submission: StudyProposalSubmission,
        control: AutoResearchProposalControl,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        if (
            submission.workspace_id != control.workspace_id
            or submission.campaign_id != control.campaign_id
            or submission.proposal_id != control.proposal_id
        ):
            raise AutoResearchInvariantError("autoresearch_proposal_identity_mismatch")
        state = self.state(submission.workspace_id, submission.campaign_id)
        expected_action = (
            AutoResearchNextAction.SUBMIT_BASELINE
            if control.role == ExperimentRole.BASELINE
            else AutoResearchNextAction.PROPOSE_CANDIDATE
        )
        if state.next_action != expected_action:
            raise AutoResearchInvariantError(f"autoresearch_proposal_not_ready:{state.reason_code}")
        if submission.estimated_cost > state.budget_remaining:
            raise AutoResearchBudgetError("autoresearch_estimated_cost_exceeds_remaining_budget")
        lineage_kind: CodeMutationKind | None = None
        source_repository_profile_id: str | None = None
        if control.role == ExperimentRole.BASELINE:
            if submission.prerequisite_study_ids:
                raise AutoResearchInvariantError("autoresearch_baseline_cannot_have_prerequisite")
        elif control.role == ExperimentRole.DIAGNOSTIC:
            parent_outcome = next(
                (
                    item
                    for item in self.repository.list_autoresearch_outcomes(
                        submission.workspace_id,
                        submission.campaign_id,
                    )
                    if item.result.proposal_id == control.parent_proposal_id
                ),
                None,
            )
            if (
                parent_outcome is None
                or parent_outcome.result.provenance != ExperimentProvenance.REAL
                or parent_outcome.result.outcome != ExperimentOutcome.COMPLETED
                or parent_outcome.decision.decision
                not in {ResultDecision.BASELINE, ResultDecision.KEEP, ResultDecision.DISCARD}
            ):
                raise AutoResearchInvariantError(
                    "autoresearch_diagnostic_parent_not_research_eligible"
                )
            if parent_outcome.result.study_id not in submission.prerequisite_study_ids:
                raise AutoResearchInvariantError(
                    "autoresearch_diagnostic_must_depend_on_parent_study"
                )
            required_sequence = tuple(
                item.stage
                for item in submission.stage_plan.items
                if item.disposition == StageDisposition.REQUIRED
            )
            if required_sequence != (StageKind.CONTRACT_EVALUATION,):
                raise AutoResearchInvariantError("autoresearch_diagnostic_stage_plan_invalid")
            try:
                AutoResearchDiagnosticRecipe.model_validate(
                    {
                        key: value
                        for key, value in submission.evaluation_recipe.items()
                        if key != "runtime"
                    }
                )
            except ValueError as exc:
                raise AutoResearchInvariantError("autoresearch_diagnostic_recipe_invalid") from exc
        else:
            if (
                control.hypothesis_family_id is not None
                and self.repository.get_hypothesis_family_conclusion(
                    submission.workspace_id,
                    submission.campaign_id,
                    control.hypothesis_family_id,
                )
                is not None
            ):
                raise AutoResearchInvariantError("autoresearch_hypothesis_family_concluded")
            if control.changed_variables[0] != submission.primary_variable.strip():
                raise AutoResearchInvariantError(
                    "autoresearch_candidate_primary_variable_must_lead_intervention"
                )
            if not submission.controlled_variables:
                raise AutoResearchInvariantError(
                    "autoresearch_candidate_requires_controlled_variables"
                )
            if training_stages_required(submission.stage_plan) and (
                training_seed(submission.training_recipe) is None
            ):
                raise AutoResearchInvariantError("autoresearch_candidate_requires_training_seed")
            parent_outcome = next(
                (
                    item
                    for item in self.repository.list_autoresearch_outcomes(
                        submission.workspace_id,
                        submission.campaign_id,
                    )
                    if item.result.proposal_id == control.parent_proposal_id
                ),
                None,
            )
            if (
                parent_outcome is None
                or parent_outcome.result.provenance != ExperimentProvenance.REAL
                or parent_outcome.result.outcome != ExperimentOutcome.COMPLETED
                or parent_outcome.result.metric_value is None
                or parent_outcome.decision.decision
                not in {ResultDecision.BASELINE, ResultDecision.KEEP, ResultDecision.DISCARD}
            ):
                raise AutoResearchInvariantError(
                    "autoresearch_candidate_parent_not_research_eligible"
                )
            if parent_outcome.result.study_id not in submission.prerequisite_study_ids:
                raise AutoResearchInvariantError(
                    "autoresearch_candidate_must_depend_on_parent_study"
                )
            lineage_kinds = {
                kind
                for variable in control.changed_variables
                if (kind := code_mutation_kind_for_variable(variable)) is not None
            }
            if len(lineage_kinds) > 1:
                raise AutoResearchInvariantError(
                    "autoresearch_exploratory_code_bundle_not_supported"
                )
            lineage_kind = next(iter(lineage_kinds), None)
            parent = self.repository.get_proposal(
                submission.workspace_id,
                submission.campaign_id,
                control.parent_proposal_id,
            ).proposal
            _validate_controlled_candidate_change(
                parent,
                submission,
                declared_variables=control.changed_variables,
                intervention_mode=control.intervention_mode,
                code_mutation_kind=lineage_kind,
            )
            if lineage_kind is not None:
                principal.require(submission.workspace_id, Capability.EXPERIMENT_CODE_MUTATE)
                campaign = self.repository.get_campaign(
                    submission.workspace_id, submission.campaign_id
                )
                manifest = self.repository.get_manifest_revision(
                    submission.workspace_id,
                    submission.campaign_id,
                    campaign.manifest_revision,
                ).manifest
                binding = manifest.evaluation_plan.get("source_repository_binding_id")
                if (
                    not isinstance(binding, str)
                    or not binding
                    or len(binding) > 160
                    or not binding[0].isalnum()
                    or any(
                        character
                        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
                        for character in binding
                    )
                ):
                    raise AutoResearchInvariantError(
                        "autoresearch_source_repository_binding_required"
                    )
                source_repository_profile_id = binding

        mutation = self.service.submit_proposal(
            submission,
            expected_version=expected_version,
            principal=principal,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )
        if not mutation.record.validation.valid:
            raise AutoResearchInvariantError(
                "autoresearch_proposal_rejected:"
                + ",".join(mutation.record.validation.reason_codes)
            )
        registered_control = self.repository.register_autoresearch_proposal(control)
        if lineage_kind is not None:
            assert source_repository_profile_id is not None
            lineage_id = (
                "lineage-"
                + canonical_hash(
                    [
                        registered_control.control_digest,
                        lineage_kind.value,
                        source_repository_profile_id,
                    ]
                )[:32]
            )
            self.repository.register_code_lineage_requirement(
                CodeLineageRecord(
                    lineage_id=lineage_id,
                    workspace_id=submission.workspace_id,
                    campaign_id=submission.campaign_id,
                    proposal_id=submission.proposal_id,
                    mutation_kind=lineage_kind,
                    source_repository_profile_id=source_repository_profile_id,
                    state="required",
                    created_at=registered_control.created_at,
                    updated_at=registered_control.created_at,
                )
            )
        return mutation

    def submit_baseline(
        self,
        submission: StudyProposalSubmission,
        *,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        return self._submit(
            submission,
            AutoResearchProposalControl(
                workspace_id=submission.workspace_id,
                campaign_id=submission.campaign_id,
                proposal_id=submission.proposal_id,
                role=ExperimentRole.BASELINE,
            ),
            expected_version=expected_version,
            principal=principal,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def submit_controlled_candidate(
        self,
        submission: StudyProposalSubmission,
        *,
        parent_proposal_id: str,
        changed_variable: str,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        return self.submit_candidate(
            submission,
            parent_proposal_id=parent_proposal_id,
            changed_variables=(changed_variable,),
            intervention_mode=InterventionMode.CONTROLLED,
            hypothesis_family_id=None,
            expected_version=expected_version,
            principal=principal,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def submit_diagnostic(
        self,
        submission: StudyProposalSubmission,
        *,
        parent_proposal_id: str,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        """Submit an agent-designed probe without changing or ranking a model."""

        return self._submit(
            submission,
            AutoResearchProposalControl(
                workspace_id=submission.workspace_id,
                campaign_id=submission.campaign_id,
                proposal_id=submission.proposal_id,
                role=ExperimentRole.DIAGNOSTIC,
                parent_proposal_id=parent_proposal_id,
            ),
            expected_version=expected_version,
            principal=principal,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def submit_candidate(
        self,
        submission: StudyProposalSubmission,
        *,
        parent_proposal_id: str,
        changed_variables: tuple[str, ...],
        intervention_mode: InterventionMode = InterventionMode.CONTROLLED,
        hypothesis_family_id: str | None = None,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
    ) -> ProposalMutation:
        return self._submit(
            submission,
            AutoResearchProposalControl(
                workspace_id=submission.workspace_id,
                campaign_id=submission.campaign_id,
                proposal_id=submission.proposal_id,
                role=ExperimentRole.CANDIDATE,
                parent_proposal_id=parent_proposal_id,
                changed_variables=changed_variables,
                intervention_mode=intervention_mode,
                hypothesis_family_id=hypothesis_family_id,
            ),
            expected_version=expected_version,
            principal=principal,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    def conclude_hypothesis_family(
        self,
        workspace_id: str,
        campaign_id: str,
        hypothesis_family_id: str,
        *,
        disposition: HypothesisFamilyDisposition,
        summary: str,
        expected_version: int,
        principal: ActorPrincipal,
        correlation_id: str,
        idempotency_key: str,
        follow_up_family_id: str | None = None,
        follow_up_hypothesis: str | None = None,
    ) -> AutoResearchHypothesisFamilyConclusion:
        """Record the agent's evidence-bound conclusion without proposing new work."""

        principal.require(workspace_id, Capability.STUDY_PROPOSE)
        return self.repository.conclude_hypothesis_family(
            workspace_id,
            campaign_id,
            hypothesis_family_id,
            disposition=disposition,
            summary=summary.strip(),
            follow_up_family_id=follow_up_family_id,
            follow_up_hypothesis=(
                follow_up_hypothesis.strip() if follow_up_hypothesis is not None else None
            ),
            expected_version=expected_version,
            actor_id=principal.actor_id,
            credential_kind=principal.credential_kind,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _proposal_is_simulated(submission) -> bool:
        runtime = submission.training_recipe.get("runtime")
        if isinstance(runtime, dict) and runtime.get("executor_kind") == "fake":
            return True
        return any(
            item.disposition == StageDisposition.REQUIRED
            and item.input_contract.get("quality_claim") is False
            for item in submission.stage_plan.items
        )

    def record_result(
        self,
        result: AutoResearchResult,
        *,
        ledger_context: AutoResearchLedgerCommitContext | None = None,
    ) -> AutoResearchOutcomeRecord:
        if (
            result.provenance == ExperimentProvenance.REAL
            and result.outcome == ExperimentOutcome.COMPLETED
        ):
            raise AutoResearchInvariantError("autoresearch_real_result_requires_sealed_projection")
        return self._record_result(result, ledger_context=ledger_context)

    def _record_result(
        self,
        result: AutoResearchResult,
        *,
        ledger_context: AutoResearchLedgerCommitContext | None = None,
    ) -> AutoResearchOutcomeRecord:
        if len(result.attempt_ids) > 100 or len(result.evidence_references) > 100:
            raise AutoResearchInvariantError("autoresearch_result_reference_limit_exceeded")
        spec = self.repository.get_autoresearch_spec(result.workspace_id, result.campaign_id)
        if result.metric_name != spec.primary_metric:
            raise AutoResearchInvariantError("autoresearch_primary_metric_mismatch")
        control = self.repository.get_autoresearch_proposal(
            result.workspace_id, result.campaign_id, result.proposal_id
        )
        if result.role != control.role:
            raise AutoResearchInvariantError("autoresearch_result_role_mismatch")
        proposal = self.repository.get_proposal(
            result.workspace_id, result.campaign_id, result.proposal_id
        )
        if proposal.study_id != result.study_id:
            raise AutoResearchInvariantError("autoresearch_result_study_mismatch")
        study = self.repository.get_study(result.workspace_id, result.campaign_id, result.study_id)
        if result.outcome == ExperimentOutcome.COMPLETED:
            if study.status not in self._SUCCESS_STUDY_STATES:
                raise AutoResearchInvariantError("autoresearch_study_not_successfully_terminal")
        elif study.status not in self._FAILED_STUDY_STATES:
            raise AutoResearchInvariantError("autoresearch_study_not_failed_terminal")

        attempts = {
            attempt.attempt_id: attempt
            for attempt in self.repository.list_study_attempts(
                result.workspace_id, result.campaign_id, result.study_id
            )
        }
        if any(
            attempt_id not in attempts or attempts[attempt_id].study_id != result.study_id
            for attempt_id in result.attempt_ids
        ):
            raise AutoResearchInvariantError("autoresearch_result_attempt_mismatch")
        if result.provenance == ExperimentProvenance.REAL and self._proposal_is_simulated(
            proposal.proposal
        ):
            raise AutoResearchInvariantError("autoresearch_fake_executor_cannot_claim_real_result")
        return self.repository._record_autoresearch_result(result, ledger_context=ledger_context)

    def ingest_evaluation_result(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        project_id: str,
        evaluation_result_id: str,
    ) -> AutoResearchOutcomeRecord:
        """Derive a real outcome only from the shared sealed projection authority."""

        spec = self.repository.get_autoresearch_spec(workspace_id, campaign_id)
        if spec.ledger_project_id is None or spec.evaluation_suite_id is None:
            raise AutoResearchInvariantError("autoresearch_evaluation_binding_required")
        if not spec.require_sealed_artifact:
            raise AutoResearchInvariantError("autoresearch_sealed_projection_required")
        if project_id != spec.ledger_project_id:
            raise AutoResearchInvariantError("autoresearch_ledger_project_mismatch")
        if self.evaluation_reader is None:
            raise AutoResearchInvariantError("autoresearch_sealed_evaluation_reader_required")

        evaluation = self.ledger.get_evaluation_result(
            workspace_id, project_id, evaluation_result_id
        )
        run = self.ledger.get_run(workspace_id, project_id, evaluation["run_id"])
        if (
            run.get("source_system") != "bashgym"
            or run.get("campaign_id") != campaign_id
            or not run.get("study_id")
        ):
            raise AutoResearchInvariantError("autoresearch_run_campaign_lineage_mismatch")
        study = self.repository.get_study(workspace_id, campaign_id, run["study_id"])

        from bashgym.campaigns.autoresearch_evidence import CampaignEvaluationProjector

        projector = CampaignEvaluationProjector(
            self.repository,
            self.ledger,
            self.evaluation_reader,
        )
        return projector.project_and_ingest(
            workspace_id,
            campaign_id,
            study.proposal_id,
            expected_evaluation_result_id=evaluation_result_id,
        )

    def enforce_stop(
        self,
        workspace_id: str,
        campaign_id: str,
        *,
        controller_id: str,
        correlation_id: str,
        idempotency_key: str,
        now: datetime | None = None,
    ) -> Campaign:
        state = self.state(workspace_id, campaign_id, now=now)
        if state.next_action != AutoResearchNextAction.STOP:
            raise AutoResearchInvariantError("autoresearch_stop_rule_not_met")
        campaign = self.repository.get_campaign(workspace_id, campaign_id)
        if campaign.status in TERMINAL_CAMPAIGN_STATES:
            return campaign
        if campaign.active_study_id or campaign.active_action_id:
            raise AutoResearchInvariantError("autoresearch_cannot_stop_with_active_work")
        mutation = self.repository.transition_campaign(
            workspace_id,
            campaign_id,
            CampaignTrigger.STOPPING_RULE_MET,
            expected_version=campaign.version,
            actor_id=controller_id,
            credential_kind=CredentialKind.CONTROLLER,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            payload={"control_plane": "autoresearch.v1"},
            stop_reason=state.reason_code,
        )
        return mutation.campaign


__all__ = [
    "AutoResearchBudgetError",
    "AutoResearchCampaignCore",
    "AutoResearchCampaignSpec",
    "AutoResearchConflictError",
    "AutoResearchDecision",
    "AutoResearchDiagnostics",
    "AutoResearchError",
    "AutoResearchInvariantError",
    "AutoResearchHypothesisFamilyConclusion",
    "AutoResearchNextAction",
    "AutoResearchOutcomeRecord",
    "AutoResearchProposalControl",
    "AutoResearchRepository",
    "AutoResearchResult",
    "AutoResearchState",
    "AutoResearchStopRules",
    "AutoResearchTemplateDefinition",
    "AutoResearchTemplatePolicy",
    "AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID",
    "ExperimentOutcome",
    "ExperimentProvenance",
    "ExperimentRole",
    "InterventionMode",
    "HypothesisFamilyDisposition",
    "MetricDirection",
    "ResultDecision",
    "autoresearch_spec_for_template",
    "build_autoresearch_template_registry",
    "builtin_autoresearch_template_definitions",
    "builtin_autoresearch_template_registry",
    "load_autoresearch_template_definitions",
]
