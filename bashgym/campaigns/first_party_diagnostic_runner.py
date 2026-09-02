"""First-party aggregate diagnostic runner for durable AutoResearch campaigns."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import FrozenContractModel, HexDigest, Identifier
from bashgym.campaigns.diagnostic_actions import (
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
    AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME,
    MAX_AUTORESEARCH_DIAGNOSTIC_BYTES,
    AutoResearchDiagnosticEvidence,
    AutoResearchDiagnosticRequest,
    DiagnosticMeasurementResult,
)
from bashgym.campaigns.remote import DiagnosticCapability, DiagnosticStageContract
from bashgym.campaigns.reward_integrity import AutoResearchRewardIntegrityEvidence

FIRST_PARTY_DIAGNOSTIC_RUNNER_ID = "bashgym-scientific-diagnostics"
FIRST_PARTY_DIAGNOSTIC_RUNNER_VERSION = "1"
FIRST_PARTY_DIAGNOSTIC_SOURCE_FILENAME = "autoresearch_diagnostic_sources.json"
FIRST_PARTY_DIAGNOSTIC_SOURCE_SCHEMA = "bashgym.autoresearch_first_party_diagnostic_sources.v1"


class PlasticityProbeSummary(FrozenContractModel):
    """Aggregate receipt from installed fixed-budget parent and child probes."""

    source_kind: Literal["plasticity_probe_summary"] = "plasticity_probe_summary"
    data_scope_id: Identifier
    metric_direction: Literal["maximize", "minimize"]
    fixed_step_budget: int = Field(ge=1)
    seed: int = Field(ge=0, le=2**63 - 1)
    sample_count: int = Field(ge=1, le=10_000)
    initial_probe_metric: float
    final_probe_metric: float
    retention_delta: float
    cumulative_training_steps: int = Field(ge=1)
    cumulative_training_tokens: int = Field(ge=1)
    dataset_revision_count: int = Field(ge=1)
    parent_model_digest: HexDigest
    candidate_model_digest: HexDigest

    @field_validator("initial_probe_metric", "final_probe_metric", "retention_delta")
    @classmethod
    def finite_metrics(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("plasticity metrics must be finite")
        return value

    @model_validator(mode="after")
    def distinct_model_receipts(self) -> PlasticityProbeSummary:
        if self.parent_model_digest == self.candidate_model_digest:
            raise ValueError("plasticity probe requires distinct parent and candidate models")
        return self


class RewardIntegritySummary(FrozenContractModel):
    """Existing first-party reward-integrity aggregate plus canary identity."""

    source_kind: Literal["reward_integrity_summary"] = "reward_integrity_summary"
    data_scope_id: Identifier
    canary_suite_id: Identifier
    evidence: AutoResearchRewardIntegrityEvidence


class PreferenceIntegritySummary(FrozenContractModel):
    """Counts sufficient to derive preference-data integrity rates."""

    source_kind: Literal["preference_integrity_summary"] = "preference_integrity_summary"
    data_scope_id: Identifier
    preference_dataset_digest: HexDigest
    labeling_contract_digest: HexDigest
    preference_pairs: int = Field(ge=1, le=10_000)
    agreement_cases: int = Field(ge=1, le=10_000)
    agreement_successes: int = Field(ge=0, le=10_000)
    ambiguous_pairs: int = Field(ge=0, le=10_000)
    position_swap_cases: int = Field(ge=1, le=10_000)
    position_swap_disagreements: int = Field(ge=0, le=10_000)
    label_conflicts: int = Field(ge=0, le=10_000)
    heldout_overlaps: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def consistent_counts(self) -> PreferenceIntegritySummary:
        if (
            self.agreement_successes > self.agreement_cases
            or self.ambiguous_pairs > self.preference_pairs
            or self.position_swap_disagreements > self.position_swap_cases
            or self.label_conflicts > self.preference_pairs
            or self.heldout_overlaps > self.preference_pairs
        ):
            raise ValueError("preference integrity counts are inconsistent")
        return self


class TeacherGapProbeSummary(FrozenContractModel):
    """Fixed-suite teacher/student comparison plus validated-output counts."""

    source_kind: Literal["teacher_gap_probe_summary"] = "teacher_gap_probe_summary"
    data_scope_id: Identifier
    evaluation_suite_id: Identifier
    metric_direction: Literal["maximize", "minimize"]
    teacher_model_digest: HexDigest
    student_model_digest: HexDigest
    output_validation_contract_digest: HexDigest
    sample_count: int = Field(ge=1, le=10_000)
    teacher_metric: float
    student_metric: float
    evaluated_outputs: int = Field(ge=1, le=10_000)
    accepted_outputs: int = Field(ge=0, le=10_000)

    @field_validator("teacher_metric", "student_metric")
    @classmethod
    def finite_metrics(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("teacher gap metrics must be finite")
        return value

    @model_validator(mode="after")
    def valid_comparison(self) -> TeacherGapProbeSummary:
        if self.teacher_model_digest == self.student_model_digest:
            raise ValueError("teacher gap probe requires distinct models")
        if self.accepted_outputs > self.evaluated_outputs:
            raise ValueError("teacher output counts are inconsistent")
        return self


class SessionRecoveryProbeSummary(FrozenContractModel):
    """Paired no-hint and hinted recovery outcomes for the same cases."""

    source_kind: Literal["session_recovery_probe_summary"] = "session_recovery_probe_summary"
    data_scope_id: Identifier
    recovery_dataset_digest: HexDigest
    reader_contract_digest: HexDigest
    confidence_level: Literal[0.95] = 0.95
    accepted_recovery_traces: int = Field(ge=1, le=10_000)
    both_failed: int = Field(ge=0, le=10_000)
    baseline_only_success: int = Field(ge=0, le=10_000)
    hinted_only_success: int = Field(ge=0, le=10_000)
    both_succeeded: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def consistent_counts(self) -> SessionRecoveryProbeSummary:
        paired_cases = (
            self.both_failed
            + self.baseline_only_success
            + self.hinted_only_success
            + self.both_succeeded
        )
        if paired_cases < 1 or self.accepted_recovery_traces > paired_cases:
            raise ValueError("session recovery counts are inconsistent")
        return self


FirstPartyDiagnosticSource = Annotated[
    PlasticityProbeSummary
    | PreferenceIntegritySummary
    | RewardIntegritySummary
    | SessionRecoveryProbeSummary
    | TeacherGapProbeSummary,
    Field(discriminator="source_kind"),
]


class FirstPartyDiagnosticSourceBundle(FrozenContractModel):
    """Canonical pinned aggregates available to the first-party runner."""

    schema_version: Literal["bashgym.autoresearch_first_party_diagnostic_sources.v1"] = (
        FIRST_PARTY_DIAGNOSTIC_SOURCE_SCHEMA
    )
    sources: tuple[FirstPartyDiagnosticSource, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def canonical_unique_sources(self) -> FirstPartyDiagnosticSourceBundle:
        keys = tuple((item.data_scope_id, item.source_kind) for item in self.sources)
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("diagnostic sources must be sorted and unique")
        return self

    @classmethod
    def from_file(cls, path: Path) -> FirstPartyDiagnosticSourceBundle:
        candidate = path.expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("diagnostic source bundle must be a regular file")
        if candidate.stat().st_size > MAX_AUTORESEARCH_DIAGNOSTIC_BYTES:
            raise ValueError("diagnostic source bundle exceeds the 1 MiB limit")
        return cls.model_validate_json(candidate.read_text(encoding="utf-8"))


def first_party_diagnostic_contract() -> DiagnosticStageContract:
    """Describe only the aggregate measurements this runner can execute."""

    return DiagnosticStageContract(
        runner_id=FIRST_PARTY_DIAGNOSTIC_RUNNER_ID,
        runner_version=FIRST_PARTY_DIAGNOSTIC_RUNNER_VERSION,
        max_sample_limit=10_000,
        max_measurements=16,
        capabilities=(
            DiagnosticCapability(
                capability_id="plasticity_probe",
                description=(
                    "Project a pinned receipt from installed fixed-budget parent and child probes."
                ),
                measurements=(
                    "cumulative_training_steps",
                    "cumulative_training_tokens",
                    "dataset_revision_count",
                    "final_probe_metric",
                    "initial_probe_metric",
                    "retention_delta",
                ),
                evidence_sources=("fixed_budget_probe_receipt",),
            ),
            DiagnosticCapability(
                capability_id="preference_integrity_probe",
                description=(
                    "Derive agreement, ambiguity, position-bias, conflict, and overlap rates "
                    "from pinned aggregate counts."
                ),
                measurements=(
                    "ambiguous_pair_rate",
                    "preference_agreement_lower_bound",
                    "preference_contamination_rate",
                    "preference_label_conflict_rate",
                    "preference_pairs",
                    "preference_position_bias_rate",
                ),
                evidence_sources=("preference_integrity_counts",),
            ),
            DiagnosticCapability(
                capability_id="recovery_trace_probe",
                description=(
                    "Derive paired no-hint versus hinted recovery lift from pinned counts."
                ),
                measurements=("recovery_lift_lower_bound", "recovery_traces"),
                evidence_sources=("paired_session_recovery_counts",),
            ),
            DiagnosticCapability(
                capability_id="reward_integrity_probe",
                description=(
                    "Project verified reward-component, constraint, and exploit-canary aggregates."
                ),
                measurements=(
                    "hard_constraint_violation_rate",
                    "reward_canary_cases",
                    "reward_canary_failure_rate",
                ),
                evidence_sources=("reward_integrity_evidence",),
            ),
            DiagnosticCapability(
                capability_id="teacher_gap_probe",
                description=(
                    "Compare a teacher and student on one pinned suite and validate outputs."
                ),
                measurements=("teacher_metric_gap", "teacher_output_acceptance_rate"),
                evidence_sources=("teacher_student_evaluation_summary",),
            ),
        ),
    )


def _source_for(
    request: AutoResearchDiagnosticRequest,
    bundle: FirstPartyDiagnosticSourceBundle,
) -> FirstPartyDiagnosticSource | None:
    if len(request.recipe.data_scope_ids) != 1:
        return None
    scope_id = request.recipe.data_scope_ids[0]
    expected_kind = {
        "plasticity_probe": "plasticity_probe_summary",
        "reward_integrity_probe": "reward_integrity_summary",
        "preference_integrity_probe": "preference_integrity_summary",
        "recovery_trace_probe": "session_recovery_probe_summary",
        "teacher_gap_probe": "teacher_gap_probe_summary",
    }.get(request.recipe.probe_family)
    if expected_kind is None:
        return None
    return next(
        (
            item
            for item in bundle.sources
            if item.data_scope_id == scope_id and item.source_kind == expected_kind
        ),
        None,
    )


def _wilson_lower_bound(successes: int, total: int) -> float:
    z = 1.959963984540054
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = proportion + z_squared / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z_squared / (4 * total)) / total)
    return (center - margin) / denominator


def _plasticity_values(
    request: AutoResearchDiagnosticRequest,
    source: PlasticityProbeSummary,
) -> dict[str, tuple[float, int]]:
    parameters = request.recipe.parameters
    if (
        source.metric_direction != parameters["metric_direction"]
        or source.fixed_step_budget != parameters["fixed_step_budget"]
        or source.seed != request.recipe.seed
        or source.sample_count > request.recipe.sample_limit
    ):
        raise ValueError("plasticity source does not match the diagnostic recipe")
    return {
        "initial_probe_metric": (source.initial_probe_metric, source.sample_count),
        "final_probe_metric": (source.final_probe_metric, source.sample_count),
        "retention_delta": (source.retention_delta, source.sample_count),
        "cumulative_training_steps": (float(source.cumulative_training_steps), 1),
        "cumulative_training_tokens": (float(source.cumulative_training_tokens), 1),
        "dataset_revision_count": (float(source.dataset_revision_count), 1),
    }


def _reward_values(
    request: AutoResearchDiagnosticRequest,
    source: RewardIntegritySummary,
) -> dict[str, tuple[float, int]]:
    evidence = source.evidence
    if (
        evidence.reward_spec.reward_spec_digest != request.recipe.parameters["reward_spec_digest"]
        or source.canary_suite_id != request.recipe.parameters["canary_suite_id"]
        or evidence.rollout_count > request.recipe.sample_limit
        or not 1 <= evidence.canaries.total <= request.recipe.sample_limit
    ):
        raise ValueError("reward integrity source does not match the diagnostic recipe")
    values = evidence.method_evidence()
    return {
        "reward_canary_cases": (float(values["reward_canary_cases"]), evidence.canaries.total),
        "reward_canary_failure_rate": (
            float(values["reward_canary_failure_rate"]),
            evidence.canaries.total,
        ),
        "hard_constraint_violation_rate": (
            float(values["hard_constraint_violation_rate"]),
            evidence.rollout_count,
        ),
    }


def _preference_values(
    request: AutoResearchDiagnosticRequest,
    source: PreferenceIntegritySummary,
) -> dict[str, tuple[float, int]]:
    if (
        source.preference_dataset_digest != request.recipe.parameters["preference_dataset_digest"]
        or source.labeling_contract_digest != request.recipe.parameters["labeling_contract_digest"]
        or source.preference_pairs > request.recipe.sample_limit
        or source.agreement_cases > request.recipe.sample_limit
        or source.position_swap_cases > request.recipe.sample_limit
    ):
        raise ValueError("preference integrity source does not match the diagnostic recipe")
    pairs = source.preference_pairs
    return {
        "preference_pairs": (float(pairs), pairs),
        "preference_agreement_lower_bound": (
            _wilson_lower_bound(source.agreement_successes, source.agreement_cases),
            source.agreement_cases,
        ),
        "ambiguous_pair_rate": (source.ambiguous_pairs / pairs, pairs),
        "preference_position_bias_rate": (
            source.position_swap_disagreements / source.position_swap_cases,
            source.position_swap_cases,
        ),
        "preference_label_conflict_rate": (source.label_conflicts / pairs, pairs),
        "preference_contamination_rate": (source.heldout_overlaps / pairs, pairs),
    }


def _teacher_gap_values(
    request: AutoResearchDiagnosticRequest,
    source: TeacherGapProbeSummary,
) -> dict[str, tuple[float, int]]:
    parameters = request.recipe.parameters
    if (
        source.evaluation_suite_id != parameters["evaluation_suite_id"]
        or source.metric_direction != parameters["metric_direction"]
        or source.teacher_model_digest != parameters["teacher_model_digest"]
        or source.student_model_digest != parameters["student_model_digest"]
        or source.output_validation_contract_digest
        != parameters["output_validation_contract_digest"]
        or source.sample_count > request.recipe.sample_limit
        or source.evaluated_outputs > request.recipe.sample_limit
    ):
        raise ValueError("teacher gap source does not match the diagnostic recipe")
    gap = (
        source.teacher_metric - source.student_metric
        if source.metric_direction == "maximize"
        else source.student_metric - source.teacher_metric
    )
    return {
        "teacher_metric_gap": (gap, source.sample_count),
        "teacher_output_acceptance_rate": (
            source.accepted_outputs / source.evaluated_outputs,
            source.evaluated_outputs,
        ),
    }


def _session_recovery_values(
    request: AutoResearchDiagnosticRequest,
    source: SessionRecoveryProbeSummary,
) -> dict[str, tuple[float, int]]:
    parameters = request.recipe.parameters
    paired_cases = (
        source.both_failed
        + source.baseline_only_success
        + source.hinted_only_success
        + source.both_succeeded
    )
    if (
        source.recovery_dataset_digest != parameters["recovery_dataset_digest"]
        or source.reader_contract_digest != parameters["reader_contract_digest"]
        or source.confidence_level != parameters["confidence_level"]
        or paired_cases > request.recipe.sample_limit
        or source.accepted_recovery_traces > request.recipe.sample_limit
    ):
        raise ValueError("session recovery source does not match the diagnostic recipe")
    lift = (source.hinted_only_success - source.baseline_only_success) / paired_cases
    discordant_rate = (source.hinted_only_success + source.baseline_only_success) / paired_cases
    variance = max(0.0, discordant_rate - lift * lift)
    lower_bound = max(
        -1.0,
        min(1.0, lift - 1.959963984540054 * math.sqrt(variance / paired_cases)),
    )
    return {
        "recovery_traces": (
            float(source.accepted_recovery_traces),
            source.accepted_recovery_traces,
        ),
        "recovery_lift_lower_bound": (lower_bound, paired_cases),
    }


def _completed_evidence(
    request: AutoResearchDiagnosticRequest,
    source: FirstPartyDiagnosticSource,
) -> AutoResearchDiagnosticEvidence:
    if isinstance(source, PlasticityProbeSummary):
        values = _plasticity_values(request, source)
    elif isinstance(source, RewardIntegritySummary):
        values = _reward_values(request, source)
    elif isinstance(source, PreferenceIntegritySummary):
        values = _preference_values(request, source)
    elif isinstance(source, TeacherGapProbeSummary):
        values = _teacher_gap_values(request, source)
    else:
        values = _session_recovery_values(request, source)
    requested_names = tuple(item.name for item in request.recipe.measurements)
    if set(requested_names) != set(values):
        raise ValueError("diagnostic source measurements do not match the request")
    return AutoResearchDiagnosticEvidence(
        workspace_id=request.workspace_id,
        campaign_id=request.campaign_id,
        proposal_id=request.proposal_id,
        study_id=request.study_id,
        action_id=request.action_id,
        attempt_id=request.attempt_id,
        recipe_digest=request.recipe_digest,
        runner_id=request.runner_id,
        runner_version=request.runner_version,
        status="completed",
        measurements=tuple(
            DiagnosticMeasurementResult(
                name=item.name,
                value=values[item.name][0],
                sample_count=values[item.name][1],
                unit=item.unit,
            )
            for item in request.recipe.measurements
        ),
    )


def _unsupported_evidence(
    request: AutoResearchDiagnosticRequest,
) -> AutoResearchDiagnosticEvidence:
    return AutoResearchDiagnosticEvidence(
        workspace_id=request.workspace_id,
        campaign_id=request.campaign_id,
        proposal_id=request.proposal_id,
        study_id=request.study_id,
        action_id=request.action_id,
        attempt_id=request.attempt_id,
        recipe_digest=request.recipe_digest,
        runner_id=request.runner_id,
        runner_version=request.runner_version,
        status="unsupported",
        unsupported_reason="diagnostic_source_unavailable",
    )


def _bounded_request(path: Path) -> AutoResearchDiagnosticRequest:
    if path.is_symlink() or not path.is_file():
        raise ValueError("diagnostic request must be a regular file")
    if path.stat().st_size > MAX_AUTORESEARCH_DIAGNOSTIC_BYTES:
        raise ValueError("diagnostic request exceeds the 1 MiB limit")
    request = AutoResearchDiagnosticRequest.model_validate_json(path.read_text(encoding="utf-8"))
    if (
        request.runner_id != FIRST_PARTY_DIAGNOSTIC_RUNNER_ID
        or request.runner_version != FIRST_PARTY_DIAGNOSTIC_RUNNER_VERSION
    ):
        raise ValueError("diagnostic request runner identity mismatch")
    return request


def _atomic_write(path: Path, evidence: AutoResearchDiagnosticEvidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = evidence.model_dump_json() + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_first_party_diagnostic(
    request_path: Path,
    output_path: Path,
    *,
    source_path: Path,
) -> AutoResearchDiagnosticEvidence:
    """Execute one aggregate diagnostic and emit the existing evidence schema."""

    request = _bounded_request(request_path)
    bundle = FirstPartyDiagnosticSourceBundle.from_file(source_path)
    source = _source_for(request, bundle)
    evidence = (
        _completed_evidence(request, source)
        if source is not None
        else _unsupported_evidence(request)
    )
    _atomic_write(output_path, evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BashGym aggregate scientific diagnostic")
    parser.add_argument("--request", default=AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME)
    parser.add_argument("--output", default=AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_first_party_diagnostic(
        Path(args.request),
        Path(args.output),
        source_path=Path(FIRST_PARTY_DIAGNOSTIC_SOURCE_FILENAME),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIRST_PARTY_DIAGNOSTIC_RUNNER_ID",
    "FIRST_PARTY_DIAGNOSTIC_RUNNER_VERSION",
    "FIRST_PARTY_DIAGNOSTIC_SOURCE_FILENAME",
    "FirstPartyDiagnosticSourceBundle",
    "PlasticityProbeSummary",
    "PreferenceIntegritySummary",
    "RewardIntegritySummary",
    "SessionRecoveryProbeSummary",
    "TeacherGapProbeSummary",
    "first_party_diagnostic_contract",
    "run_first_party_diagnostic",
]
