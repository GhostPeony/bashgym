"""Durable, baseline-first AutoResearch control over campaign execution.

This module deliberately does not launch training.  It turns the existing campaign
repository into a scientific control loop: prepare an approved campaign, submit one
controlled study at a time, record an executor-backed result, decide whether it beat
the incumbent, and expose the next safe action.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import (
    TERMINAL_CAMPAIGN_STATES,
    ActorPrincipal,
    AttemptStatus,
    Campaign,
    CampaignKind,
    CampaignManifest,
    CampaignStatus,
    CampaignTrigger,
    Capability,
    CodeLineageRecord,
    CodeMutationKind,
    CredentialKind,
    FrozenContractModel,
    Identifier,
    ProposalStatus,
    StageDisposition,
    StudyProposalSubmission,
    StudyStatus,
    TargetModelContract,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.lineage import code_mutation_kind_for_variable
from bashgym.campaigns.persistence import (
    CampaignPersistenceError,
    MigrationChecksumError,
    ProposalMutation,
    RecordNotFoundError,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from bashgym.ledger.contracts import ContextStatus, RunStatus
from bashgym.ledger.persistence import ExperimentLedgerRepository


class AutoResearchError(CampaignPersistenceError):
    """Stable base error for AutoResearch policy and persistence failures."""


class AutoResearchInvariantError(AutoResearchError):
    code = "autoresearch_invariant_failed"


class AutoResearchConflictError(AutoResearchError):
    code = "autoresearch_conflict"


class AutoResearchBudgetError(AutoResearchError):
    code = "autoresearch_budget_exceeded"


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ExperimentRole(str, Enum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


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


class AutoResearchStopRules(FrozenContractModel):
    schema_version: Literal["autoresearch_stop_rules.v1"] = "autoresearch_stop_rules.v1"
    max_attempts: int = Field(ge=1, le=100)
    budget_unit: Identifier
    max_total_cost: float = Field(gt=0)
    target_metric: float | None = None
    minimum_improvement: float = Field(default=0.0, ge=0)
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
        return self


class AutoResearchCampaignSpec(FrozenContractModel):
    schema_version: Literal["autoresearch_campaign_spec.v1"] = "autoresearch_campaign_spec.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    primary_metric: Identifier
    metric_direction: MetricDirection
    stop_rules: AutoResearchStopRules
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
            if self.parent_proposal_id is not None or self.changed_variables:
                raise ValueError("baseline cannot have a parent or changed variables")
        elif self.parent_proposal_id is None or len(self.changed_variables) != 1:
            raise ValueError("candidate requires one parent and exactly one changed variable")
        return self

    @property
    def control_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"created_at"}))


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
        elif self.metric_value is not None:
            raise ValueError("crashed AutoResearch result cannot claim a final metric")
        if not math.isfinite(self.actual_cost):
            raise ValueError("AutoResearch actual_cost must be finite")
        return self

    @property
    def result_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json", exclude={"recorded_at"}))


class AutoResearchDecision(FrozenContractModel):
    schema_version: Literal["autoresearch_decision.v1"] = "autoresearch_decision.v1"
    proposal_id: Identifier
    decision: ResultDecision
    reason_code: Identifier
    eligible_for_best: bool
    previous_best_proposal_id: Identifier | None = None
    previous_best_metric: float | None = None
    improvement: float | None = None
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    decided_at: datetime = Field(default_factory=utc_now)


class AutoResearchOutcomeRecord(FrozenContractModel):
    schema_version: Literal["autoresearch_outcome_record.v1"] = "autoresearch_outcome_record.v1"
    result: AutoResearchResult
    decision: AutoResearchDecision
    replayed: bool = False


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

    schema_version: Literal["autoresearch_template_policy.v1"] = (
        "autoresearch_template_policy.v1"
    )
    template_revision: Identifier
    primary_metric: Identifier
    metric_direction: MetricDirection
    stop_rules: AutoResearchStopRules
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

    def materialize_spec(self, workspace_id: str, campaign_id: str) -> AutoResearchCampaignSpec | None:
        if self.policy is None:
            return None
        return AutoResearchCampaignSpec(
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            primary_metric=self.policy.primary_metric,
            metric_direction=self.policy.metric_direction,
            stop_rules=self.policy.stop_rules,
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
    definitions: Iterable[AutoResearchTemplateDefinition] | None = None,
) -> AutoResearchCampaignSpec | None:
    """Materialize the durable policy paired with any registered definition."""

    values = tuple(definitions or builtin_autoresearch_template_definitions())
    for definition in values:
        if definition.template_id == template_id:
            return definition.materialize_spec(workspace_id, campaign_id)
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
)


class AutoResearchRepository(CampaignRuntimeRepository):
    """Campaign runtime plus a small immutable AutoResearch evidence projection."""

    def initialize(self) -> None:
        super().initialize()
        with self._connection(immediate=True) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS autoresearch_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
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

    @staticmethod
    def _improvement(direction: MetricDirection, incumbent: float, candidate: float) -> float:
        return (
            candidate - incumbent
            if direction == MetricDirection.MAXIMIZE
            else incumbent - candidate
        )

    def record_autoresearch_result(self, result: AutoResearchResult) -> AutoResearchOutcomeRecord:
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
            if by_proposal is not None:
                if by_proposal["result_digest"] != result.result_digest:
                    raise AutoResearchConflictError("autoresearch_result_conflict")
                return AutoResearchOutcomeRecord(
                    result=AutoResearchResult.model_validate_json(by_proposal["result_json"]),
                    decision=AutoResearchDecision.model_validate_json(by_proposal["decision_json"]),
                    replayed=True,
                )
            if by_id is not None:
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

            rows = connection.execute(
                """
                SELECT result_json, decision_json FROM autoresearch_results
                WHERE workspace_id = ? AND campaign_id = ?
                ORDER BY created_at, result_id
                """,
                (result.workspace_id, result.campaign_id),
            ).fetchall()
            prior = tuple(
                AutoResearchOutcomeRecord(
                    result=AutoResearchResult.model_validate_json(row["result_json"]),
                    decision=AutoResearchDecision.model_validate_json(row["decision_json"]),
                )
                for row in rows
            )
            eligible = [
                item
                for item in prior
                if item.decision.eligible_for_best and item.result.metric_value is not None
            ]
            incumbent = eligible[-1] if eligible else None
            previous_id = incumbent.result.proposal_id if incumbent else None
            previous_metric = incumbent.result.metric_value if incumbent else None

            improvement: float | None = None
            if result.outcome == ExperimentOutcome.CRASHED:
                choice = ResultDecision.CRASH
                reason = "experiment_crashed"
                is_eligible = False
            elif result.provenance == ExperimentProvenance.SIMULATED:
                choice = ResultDecision.INELIGIBLE
                reason = "simulated_result_not_quality_evidence"
                is_eligible = False
            elif result.role == ExperimentRole.BASELINE:
                if any(item.decision.decision == ResultDecision.BASELINE for item in prior):
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
                improved = improvement > 0 if threshold == 0 else improvement >= threshold
                choice = ResultDecision.KEEP if improved else ResultDecision.DISCARD
                reason = (
                    "candidate_improved_primary_metric"
                    if improved
                    else "candidate_did_not_clear_improvement_gate"
                )
                is_eligible = improved

            decision = AutoResearchDecision(
                proposal_id=result.proposal_id,
                decision=choice,
                reason_code=reason,
                eligible_for_best=is_eligible,
                previous_best_proposal_id=previous_id,
                previous_best_metric=previous_metric,
                improvement=improvement,
                result_digest=result.result_digest,
            )
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
        return AutoResearchOutcomeRecord(result=result, decision=decision)


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

    def __init__(self, repository: AutoResearchRepository):
        self.repository = repository
        self.service = CampaignService(repository)
        self.ledger = ExperimentLedgerRepository(repository.db_path)
        self.ledger.initialize()

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
        outcome_by_proposal = {item.result.proposal_id: item for item in outcomes}
        pending = [item for item in controls if item.proposal_id not in outcome_by_proposal]
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
        budget_used = sum(item.result.actual_cost for item in outcomes)
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
        elif len(controls) >= spec.stop_rules.max_attempts:
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
                "ready_for_controlled_hypothesis",
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
            attempts_used=len(outcomes),
            proposals_used=len(controls),
            budget_used=budget_used,
            budget_remaining=budget_remaining,
            latest_decision=latest,
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
        else:
            if control.changed_variables != (submission.primary_variable.strip(),):
                raise AutoResearchInvariantError(
                    "autoresearch_candidate_must_change_declared_primary_variable_only"
                )
            if not submission.controlled_variables:
                raise AutoResearchInvariantError(
                    "autoresearch_candidate_requires_controlled_variables"
                )
            if control.parent_proposal_id != state.best_proposal_id:
                raise AutoResearchInvariantError("autoresearch_candidate_parent_is_not_incumbent")
            if state.best_study_id not in submission.prerequisite_study_ids:
                raise AutoResearchInvariantError(
                    "autoresearch_candidate_must_depend_on_incumbent_study"
                )
            lineage_kind = code_mutation_kind_for_variable(control.changed_variables[0])
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
            lineage_id = "lineage-" + canonical_hash(
                [
                    registered_control.control_digest,
                    lineage_kind.value,
                    source_repository_profile_id,
                ]
            )[:32]
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
        return self._submit(
            submission,
            AutoResearchProposalControl(
                workspace_id=submission.workspace_id,
                campaign_id=submission.campaign_id,
                proposal_id=submission.proposal_id,
                role=ExperimentRole.CANDIDATE,
                parent_proposal_id=parent_proposal_id,
                changed_variables=(changed_variable,),
            ),
            expected_version=expected_version,
            principal=principal,
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

    def record_result(self, result: AutoResearchResult) -> AutoResearchOutcomeRecord:
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
            for attempt in self.repository.list_attempts(result.workspace_id, result.campaign_id)
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
        return self.repository.record_autoresearch_result(result)

    def ingest_evaluation_result(
        self,
        *,
        workspace_id: str,
        campaign_id: str,
        project_id: str,
        evaluation_result_id: str,
    ) -> AutoResearchOutcomeRecord:
        """Derive a real AutoResearch outcome from ledger and sealed campaign evidence."""

        spec = self.repository.get_autoresearch_spec(workspace_id, campaign_id)
        if spec.ledger_project_id is None or spec.evaluation_suite_id is None:
            raise AutoResearchInvariantError("autoresearch_evaluation_binding_required")
        if project_id != spec.ledger_project_id:
            raise AutoResearchInvariantError("autoresearch_ledger_project_mismatch")

        evaluation = self.ledger.get_evaluation_result(
            workspace_id, project_id, evaluation_result_id
        )
        if evaluation["evaluation_suite_id"] != spec.evaluation_suite_id:
            raise AutoResearchInvariantError("autoresearch_evaluation_suite_mismatch")
        if evaluation["status"] != RunStatus.COMPLETED.value:
            raise AutoResearchInvariantError("autoresearch_evaluation_not_completed")
        metric_value = evaluation["metrics"].get(spec.primary_metric)
        if metric_value is None or not math.isfinite(float(metric_value)):
            raise AutoResearchInvariantError("autoresearch_primary_metric_missing")

        run = self.ledger.get_run(workspace_id, project_id, evaluation["run_id"])
        if run["source_system"] != "bashgym":
            raise AutoResearchInvariantError("autoresearch_run_source_mismatch")
        if run["campaign_id"] != campaign_id or not run["study_id"] or not run["action_id"]:
            raise AutoResearchInvariantError("autoresearch_run_campaign_lineage_mismatch")
        if run["status"] != RunStatus.COMPLETED.value:
            raise AutoResearchInvariantError("autoresearch_run_not_completed")
        if run["context_status"] != ContextStatus.VERIFIED.value:
            raise AutoResearchInvariantError("autoresearch_run_context_not_verified")

        study = self.repository.get_study(workspace_id, campaign_id, run["study_id"])
        if study.status not in self._SUCCESS_STUDY_STATES:
            raise AutoResearchInvariantError("autoresearch_study_not_successfully_terminal")
        control = self.repository.get_autoresearch_proposal(
            workspace_id, campaign_id, study.proposal_id
        )
        proposal = self.repository.get_proposal(workspace_id, campaign_id, study.proposal_id)

        campaign_attempts = tuple(
            attempt
            for attempt in self.repository.list_attempts(workspace_id, campaign_id)
            if attempt.study_id == study.study_id
        )
        terminal_attempts = {
            AttemptStatus.COMPLETED,
            AttemptStatus.FAILED,
            AttemptStatus.FORCE_STOPPED,
            AttemptStatus.CANCELLED,
        }
        if not campaign_attempts or any(
            attempt.status not in terminal_attempts for attempt in campaign_attempts
        ):
            raise AutoResearchInvariantError("autoresearch_study_attempts_not_terminal")
        if run["action_id"] not in {attempt.action_id for attempt in campaign_attempts}:
            raise AutoResearchInvariantError("autoresearch_run_action_mismatch")

        ledger_attempt_id = evaluation.get("attempt_id")
        if not ledger_attempt_id:
            raise AutoResearchInvariantError("autoresearch_evaluation_attempt_required")
        ledger_attempt = self.ledger.get_attempt_record(
            workspace_id, project_id, ledger_attempt_id
        )
        if ledger_attempt["run_id"] != run["run_id"]:
            raise AutoResearchInvariantError("autoresearch_evaluation_attempt_run_mismatch")
        if ledger_attempt["status"] != RunStatus.COMPLETED.value:
            raise AutoResearchInvariantError("autoresearch_evaluation_attempt_not_completed")
        if evaluation.get("model_version_id") != run.get("model_version_id"):
            raise AutoResearchInvariantError("autoresearch_evaluation_model_mismatch")
        source_attempt_id = ledger_attempt.get("source_attempt_id")
        mapped_attempt = next(
            (
                attempt
                for attempt in campaign_attempts
                if attempt.attempt_id == source_attempt_id
            ),
            None,
        )
        if mapped_attempt is None or mapped_attempt.action_id != run["action_id"]:
            raise AutoResearchInvariantError("autoresearch_campaign_attempt_mapping_mismatch")
        if mapped_attempt.status != AttemptStatus.COMPLETED:
            raise AutoResearchInvariantError("autoresearch_mapped_attempt_not_completed")

        usage = self.repository.study_budget_usage(
            workspace_id,
            campaign_id,
            study.study_id,
            spec.stop_rules.budget_unit,
        )
        if abs(usage["reserved"]) > 1e-9:
            raise AutoResearchInvariantError("autoresearch_study_budget_not_settled")

        campaign_artifacts = tuple(
            artifact
            for artifact in self.repository.list_artifacts(workspace_id, campaign_id)
            if artifact.producer_action_id in {attempt.action_id for attempt in campaign_attempts}
            and artifact.sealed
            and artifact.valid
        )
        evidence_references: list[str] = [evaluation_result_id, run["run_id"]]
        evaluation_artifact_id = evaluation.get("artifact_id")
        if spec.require_sealed_artifact:
            if not evaluation_artifact_id:
                raise AutoResearchInvariantError("autoresearch_evaluation_artifact_required")
            ledger_artifact = next(
                (
                    artifact
                    for artifact in self.ledger.list_artifacts(
                        workspace_id, project_id, run_id=run["run_id"]
                    )
                    if artifact["artifact_id"] == evaluation_artifact_id
                ),
                None,
            )
            if ledger_artifact is None:
                raise AutoResearchInvariantError("autoresearch_evaluation_artifact_missing")
            if ledger_artifact.get("attempt_id") not in {None, ledger_attempt_id}:
                raise AutoResearchInvariantError("autoresearch_evaluation_artifact_attempt_mismatch")
            campaign_artifact = next(
                (
                    artifact
                    for artifact in campaign_artifacts
                    if artifact.sha256 == ledger_artifact["sha256"]
                ),
                None,
            )
            if campaign_artifact is None:
                raise AutoResearchInvariantError("autoresearch_sealed_artifact_hash_mismatch")
            evidence_references.extend(
                [ledger_artifact["artifact_id"], campaign_artifact.artifact_id]
            )

        simulated = bool(run["is_simulation"]) or self._proposal_is_simulated(
            proposal.proposal
        )
        if not simulated and any(
            run.get(field) is None
            for field in ("model_version_id", "dataset_version_id", "environment_id")
        ):
            raise AutoResearchInvariantError("autoresearch_real_run_context_pins_required")
        provenance = (
            ExperimentProvenance.SIMULATED if simulated else ExperimentProvenance.REAL
        )
        result = AutoResearchResult(
            result_id=f"autoresearch-result-{canonical_hash([project_id, evaluation_result_id])[:32]}",
            workspace_id=workspace_id,
            campaign_id=campaign_id,
            proposal_id=study.proposal_id,
            study_id=study.study_id,
            role=control.role,
            provenance=provenance,
            outcome=ExperimentOutcome.COMPLETED,
            metric_name=spec.primary_metric,
            metric_value=float(metric_value),
            actual_cost=usage["actual"],
            attempt_ids=tuple(attempt.attempt_id for attempt in campaign_attempts),
            evidence_references=tuple(dict.fromkeys(evidence_references)),
            recorded_at=utc_now(),
        )
        return self.record_result(result)

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
    "AutoResearchError",
    "AutoResearchInvariantError",
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
    "MetricDirection",
    "ResultDecision",
    "autoresearch_spec_for_template",
    "build_autoresearch_template_registry",
    "builtin_autoresearch_template_definitions",
    "builtin_autoresearch_template_registry",
    "load_autoresearch_template_definitions",
]
