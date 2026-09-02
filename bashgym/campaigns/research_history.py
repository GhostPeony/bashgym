"""Bounded experiment history projected from durable AutoResearch facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import sqrt
from statistics import mean, stdev
from typing import Any

from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignSpec,
    AutoResearchHypothesisFamilyConclusion,
    AutoResearchOutcomeRecord,
    AutoResearchProposalControl,
    ExperimentOutcome,
    ExperimentProvenance,
    ExperimentRole,
    MetricDirection,
    ResultDecision,
)
from bashgym.campaigns.contracts import StudyProposal, canonical_hash
from bashgym.campaigns.experiment_power import build_experiment_power_projection
from bashgym.campaigns.failure_observations import build_research_failure_packet
from bashgym.campaigns.outcome_assessment import build_outcome_assessment

_MAX_HISTORY = 100
_MAX_METRICS = 16
_MAX_REFERENCES = 20
_MAX_ATTEMPTS = 20
_MAX_HYPOTHESIS_FAMILIES = 20
_QUALITY_FIELDS = (
    "generated_rows",
    "accepted_rows",
    "rejected_rows",
    "acceptance_rate",
    "deterministic_verified_rows",
    "verification_failed_rows",
    "verification_pass_rate",
    "duplicate_rows_removed",
    "contamination_rows_removed",
    "verifier_digest",
)
_REPLICATION_DIMENSIONS = frozenset({"training_recipe.seed"})


def _signed_change(direction: MetricDirection, reference: float, candidate: float) -> float:
    return candidate - reference if direction == MetricDirection.MAXIMIZE else reference - candidate


def _bounded_values(values: Sequence[str], limit: int) -> tuple[list[str], int]:
    unique = list(dict.fromkeys(item for item in values if item))
    return unique[:limit], max(0, len(unique) - limit)


def _without_declared_dimensions(
    recipe: Mapping[str, Any],
    *,
    recipe_name: str,
    dimensions: frozenset[str],
) -> dict[str, Any]:
    shared = deepcopy(dict(recipe))
    prefix = f"{recipe_name}."
    for dimension in dimensions:
        if not dimension.startswith(prefix):
            continue
        path = dimension.removeprefix(prefix).split(".")
        current: Any = shared
        for key in path[:-1]:
            if not isinstance(current, dict):
                break
            current = current.get(key)
        else:
            if isinstance(current, dict):
                current.pop(path[-1], None)
    return shared


def _replication_comparison_digest(
    *,
    spec: AutoResearchCampaignSpec,
    proposal: StudyProposal,
    control: AutoResearchProposalControl,
    outcome: AutoResearchOutcomeRecord,
) -> str | None:
    """Bind a replicate to its shared scientific factors and reference."""

    dimensions = frozenset(control.changed_variables)
    reference_id = outcome.decision.previous_best_proposal_id
    if dimensions != _REPLICATION_DIMENSIONS or reference_id is None:
        return None
    return canonical_hash(
        {
            "schema_version": "bashgym.autoresearch_replication_comparison.v1",
            "evaluation_suite_id": spec.evaluation_suite_id,
            "primary_metric": spec.primary_metric,
            "metric_direction": spec.metric_direction.value,
            "study_family": proposal.study_family,
            "controlled_variables": list(proposal.controlled_variables),
            "replicate_dimensions": sorted(dimensions),
            "parent_proposal_id": control.parent_proposal_id,
            "reference_proposal_id": reference_id,
            "dataset_recipe": _without_declared_dimensions(
                proposal.dataset_recipe,
                recipe_name="dataset_recipe",
                dimensions=dimensions,
            ),
            "training_recipe": _without_declared_dimensions(
                proposal.training_recipe,
                recipe_name="training_recipe",
                dimensions=dimensions,
            ),
            "evaluation_recipe": _without_declared_dimensions(
                proposal.evaluation_recipe,
                recipe_name="evaluation_recipe",
                dimensions=dimensions,
            ),
            "stage_plan": proposal.stage_plan.model_dump(mode="json"),
        }
    )


def _dataset_for_outcome(
    versions: Sequence[Mapping[str, Any]], outcome: AutoResearchOutcomeRecord
) -> dict[str, Any] | None:
    attempts = frozenset(outcome.result.attempt_ids)
    for version in versions:
        metadata = version.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        quality = metadata.get("data_quality")
        if metadata.get("producer_attempt_id") not in attempts or not isinstance(quality, Mapping):
            continue
        projected_quality = {
            key: deepcopy(quality[key]) for key in _QUALITY_FIELDS if key in quality
        }
        return {
            "dataset_version_id": version.get("dataset_version_id"),
            "content_digest": version.get("content_digest"),
            "quality": projected_quality,
        }
    return None


def _learning(outcome: AutoResearchOutcomeRecord) -> dict[str, str]:
    result = outcome.result
    decision = outcome.decision
    if result.outcome != ExperimentOutcome.COMPLETED:
        return {
            "status": "inconclusive",
            "summary": "Execution did not produce a completed quality result.",
        }
    if result.provenance == ExperimentProvenance.SIMULATED or decision.decision == (
        ResultDecision.INELIGIBLE
    ):
        return {
            "status": "ineligible",
            "summary": "The result was not eligible to change the reference.",
        }
    if decision.decision == ResultDecision.BASELINE:
        return {
            "status": "baseline_recorded",
            "summary": "Starting performance was recorded on the fixed evaluation suite.",
        }
    if decision.decision == ResultDecision.KEEP:
        return {
            "status": "retained",
            "summary": (
                "The candidate cleared the configured primary and protected metric gates and "
                "became the reference."
            ),
        }
    if decision.reason_code == "candidate_failed_protected_metric_gate":
        return {
            "status": "not_retained",
            "summary": (
                "The candidate exceeded at least one configured protected-metric regression "
                "limit and was not retained."
            ),
        }
    return {
        "status": "not_retained",
        "summary": (
            "The candidate did not clear the configured primary metric gate and was not "
            "retained."
        ),
    }


def _entry(
    *,
    spec: AutoResearchCampaignSpec,
    proposal: StudyProposal,
    control: AutoResearchProposalControl,
    outcome: AutoResearchOutcomeRecord,
    outcomes_by_proposal: Mapping[str, AutoResearchOutcomeRecord],
    dataset_versions: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    evidence_strength: str,
    hypothesis_family: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = outcome.result
    decision = outcome.decision
    completed = result.outcome == ExperimentOutcome.COMPLETED
    primary_passed = None
    is_candidate = control.role == ExperimentRole.CANDIDATE
    parent_outcome = (
        outcomes_by_proposal.get(control.parent_proposal_id)
        if control.parent_proposal_id is not None
        else None
    )
    parent_value = parent_outcome.result.metric_value if parent_outcome is not None else None
    parent_improvement = None
    if completed and result.metric_value is not None and parent_value is not None:
        parent_improvement = _signed_change(
            spec.metric_direction,
            parent_value,
            result.metric_value,
        )
    if completed and is_candidate and decision.improvement is not None:
        primary_passed = (
            decision.improvement > 0 and decision.improvement >= spec.stop_rules.minimum_improvement
        )

    protected = []
    for gate in spec.stop_rules.protected_metrics:
        reference_value = None
        candidate_value = result.metrics.get(gate.metric_name)
        reference_outcome = (
            outcomes_by_proposal.get(decision.previous_best_proposal_id)
            if decision.previous_best_proposal_id is not None
            else None
        )
        if reference_outcome is not None:
            reference_value = reference_outcome.result.metrics.get(gate.metric_name)
        signed_change = None
        regression = None
        passed = None
        if completed and is_candidate:
            if reference_value is None or candidate_value is None:
                passed = False
            else:
                signed_change = _signed_change(gate.direction, reference_value, candidate_value)
                regression = max(0.0, -signed_change)
                passed = regression <= gate.max_regression
        protected.append(
            {
                "metric_name": gate.metric_name,
                "direction": gate.direction.value,
                "reference_value": reference_value,
                "candidate_value": candidate_value,
                "signed_change": signed_change,
                "observed_regression": regression,
                "maximum_regression": gate.max_regression,
                "passed": passed,
            }
        )

    metric_items = sorted(result.metrics.items())
    attempts, attempts_omitted = _bounded_values(result.attempt_ids, _MAX_ATTEMPTS)
    references, references_omitted = _bounded_values(
        (*proposal.evidence_references, *result.evidence_references), _MAX_REFERENCES
    )
    reference_outcome = (
        outcomes_by_proposal.get(decision.previous_best_proposal_id)
        if decision.previous_best_proposal_id is not None
        else outcome if control.role == ExperimentRole.BASELINE else None
    )
    failure_analysis = build_research_failure_packet(
        campaign_id=spec.campaign_id,
        reference_outcome=(
            reference_outcome.model_dump(mode="json") if reference_outcome is not None else None
        ),
        candidate_outcome=(
            outcome.model_dump(mode="json") if control.role == ExperimentRole.CANDIDATE else None
        ),
        evaluations=evaluations,
    )
    return {
        "proposal_id": proposal.proposal_id,
        "study_id": result.study_id,
        "result_id": result.result_id,
        "role": control.role.value,
        "parent_proposal_id": control.parent_proposal_id,
        "intervention": {
            "mode": control.intervention_mode.value,
            "changed_variables": list(control.changed_variables),
            "hypothesis_family_id": control.hypothesis_family_id,
        },
        "proposal": {
            "hypothesis": proposal.hypothesis,
            "changed_variable": proposal.primary_variable,
            "expected_outcome": proposal.expected_outcome,
            "falsification_criterion": proposal.falsification_criterion,
        },
        "performance": {
            "evaluation_suite_id": spec.evaluation_suite_id,
            "parent": {
                "proposal_id": control.parent_proposal_id,
                "value": parent_value,
                "improvement": parent_improvement,
            },
            "primary": {
                "metric_name": spec.primary_metric,
                "direction": spec.metric_direction.value,
                "reference_proposal_id": decision.previous_best_proposal_id,
                "reference_value": decision.previous_best_metric,
                "candidate_value": result.metric_value,
                "improvement": decision.improvement,
                "minimum_improvement": spec.stop_rules.minimum_improvement,
                "passed": primary_passed,
            },
            "protected_metrics": protected,
            "metrics": dict(metric_items[:_MAX_METRICS]),
            "metrics_omitted": max(0, len(metric_items) - _MAX_METRICS),
        },
        "result": {
            "outcome": result.outcome.value,
            "provenance": result.provenance.value,
            "actual_cost": result.actual_cost,
            "recorded_at": result.recorded_at.isoformat(),
        },
        "decision": {
            "decision": decision.decision.value,
            "reason_code": decision.reason_code,
            "eligible_for_best": decision.eligible_for_best,
        },
        "learning": _learning(outcome),
        "failure_analysis": failure_analysis,
        "outcome_assessment": build_outcome_assessment(
            outcome=result.outcome.value,
            provenance=result.provenance.value,
            decision=decision.decision.value,
            reason_code=decision.reason_code,
            failure_analysis=failure_analysis,
            evidence_strength=evidence_strength,
        ),
        "experiment_power": build_experiment_power_projection(
            outcome=outcome.model_dump(mode="json"),
            evaluations=evaluations,
            hypothesis_family=hypothesis_family,
        ),
        "data": _dataset_for_outcome(dataset_versions, outcome),
        "attempt_ids": attempts,
        "attempt_ids_omitted": attempts_omitted,
        "evidence_references": references,
        "evidence_references_omitted": references_omitted,
    }


def _hypothesis_families(
    *,
    spec: AutoResearchCampaignSpec,
    proposals: Sequence[StudyProposal],
    controls: Sequence[AutoResearchProposalControl],
    outcomes_by_proposal: Mapping[str, AutoResearchOutcomeRecord],
    conclusions: Sequence[AutoResearchHypothesisFamilyConclusion],
) -> list[dict[str, Any]]:
    proposal_by_id = {item.proposal_id: item for item in proposals}
    grouped: dict[str, list[AutoResearchProposalControl]] = {}
    for control in controls:
        if control.hypothesis_family_id is not None:
            grouped.setdefault(control.hypothesis_family_id, []).append(control)
    conclusion_by_family = {item.hypothesis_family_id: item for item in conclusions}

    ordered_families = sorted(
        grouped.items(),
        key=lambda item: min(
            (
                proposal_by_id[control.proposal_id].creation_sequence
                for control in item[1]
                if control.proposal_id in proposal_by_id
            ),
            default=0,
        ),
    )[-_MAX_HYPOTHESIS_FAMILIES:]
    summaries: list[dict[str, Any]] = []
    for family_id, family_controls in ordered_families:
        ordered_controls = sorted(
            family_controls,
            key=lambda control: (
                (
                    proposal_by_id[control.proposal_id].creation_sequence
                    if control.proposal_id in proposal_by_id
                    else 0
                ),
                control.proposal_id,
            ),
        )
        proposal_ids = [control.proposal_id for control in ordered_controls]
        seeds: list[int] = []
        completed_seeds: list[int] = []
        comparison_digests: list[str] = []
        metrics: list[float] = []
        decisions: dict[str, int] = {}
        active = False
        for control in ordered_controls:
            proposal = proposal_by_id.get(control.proposal_id)
            if proposal is None:
                continue
            seed = proposal.training_recipe.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool) and seed not in seeds:
                seeds.append(seed)
            outcome = outcomes_by_proposal.get(control.proposal_id)
            if outcome is None:
                active = True
                continue
            decision = outcome.decision.decision.value
            decisions[decision] = decisions.get(decision, 0) + 1
            result = outcome.result
            if (
                result.outcome == ExperimentOutcome.COMPLETED
                and result.provenance == ExperimentProvenance.REAL
                and result.metric_value is not None
            ):
                metrics.append(result.metric_value)
                if isinstance(seed, int) and not isinstance(seed, bool):
                    completed_seeds.append(seed)
                comparison_digest = _replication_comparison_digest(
                    spec=spec,
                    proposal=proposal,
                    control=control,
                    outcome=outcome,
                )
                if comparison_digest is not None:
                    comparison_digests.append(comparison_digest)

        replicated = (
            len(metrics) >= 2
            and len(comparison_digests) == len(metrics)
            and len(set(comparison_digests)) == 1
            and len(set(completed_seeds)) >= 2
        )
        deviation = stdev(metrics) if len(metrics) >= 2 else None
        conclusion = conclusion_by_family.get(family_id)
        summaries.append(
            {
                "hypothesis_family_id": family_id,
                "status": (
                    "active" if active else ("replicated" if replicated else "single_observation")
                ),
                "proposal_ids": proposal_ids,
                "training_seeds": seeds,
                "completed_real_results": len(metrics),
                "decisions": dict(sorted(decisions.items())),
                "primary_metric_summary": {
                    "metric_name": spec.primary_metric,
                    "count": len(metrics),
                    "mean": mean(metrics) if metrics else None,
                    "sample_standard_deviation": deviation,
                    "standard_error": (
                        deviation / sqrt(len(metrics)) if deviation is not None else None
                    ),
                    "minimum": min(metrics) if metrics else None,
                    "maximum": max(metrics) if metrics else None,
                    "uncertainty_method": (
                        "between_run_sample_standard_deviation" if deviation is not None else None
                    ),
                },
                "lifecycle": {
                    "status": conclusion.disposition.value if conclusion is not None else "open",
                    "conclusion": (
                        {
                            "summary": conclusion.summary,
                            "proposal_ids": list(conclusion.proposal_ids),
                            "result_ids": list(conclusion.result_ids),
                            "aggregate_version": conclusion.aggregate_version,
                        }
                        if conclusion is not None
                        else None
                    ),
                    "follow_up": (
                        {
                            "hypothesis_family_id": conclusion.follow_up_family_id,
                            "hypothesis": conclusion.follow_up_hypothesis,
                        }
                        if conclusion is not None and conclusion.follow_up_family_id is not None
                        else None
                    ),
                },
            }
        )
    return summaries


def build_autoresearch_history(
    *,
    objective: str,
    spec: AutoResearchCampaignSpec,
    proposals: Sequence[StudyProposal],
    controls: Sequence[AutoResearchProposalControl],
    outcomes: Sequence[AutoResearchOutcomeRecord],
    dataset_versions: Sequence[Mapping[str, Any]] = (),
    evaluations: Sequence[Mapping[str, Any]] = (),
    hypothesis_family_conclusions: Sequence[AutoResearchHypothesisFamilyConclusion] = (),
    limit: int = _MAX_HISTORY,
) -> dict[str, Any]:
    """Join completed experiment records and project bounded scientific facts."""

    if not 1 <= limit <= _MAX_HISTORY:
        raise ValueError("autoresearch_history_limit_invalid")
    proposal_by_id = {item.proposal_id: item for item in proposals}
    control_by_id = {item.proposal_id: item for item in controls}
    ordered = sorted(outcomes, key=lambda item: (item.result.recorded_at, item.result.result_id))
    outcomes_by_proposal = {item.result.proposal_id: item for item in ordered}
    hypothesis_families = _hypothesis_families(
        spec=spec,
        proposals=proposals,
        controls=controls,
        outcomes_by_proposal=outcomes_by_proposal,
        conclusions=hypothesis_family_conclusions,
    )
    replicated_families = {
        item["hypothesis_family_id"]
        for item in hypothesis_families
        if item["status"] == "replicated"
    }
    hypothesis_family_by_id = {item["hypothesis_family_id"]: item for item in hypothesis_families}
    selected = ordered[-limit:]
    experiments = []
    for outcome in selected:
        proposal_id = outcome.result.proposal_id
        proposal = proposal_by_id.get(proposal_id)
        control = control_by_id.get(proposal_id)
        if proposal is None or control is None:
            raise ValueError("autoresearch_history_record_missing")
        experiments.append(
            _entry(
                spec=spec,
                proposal=proposal,
                control=control,
                outcome=outcome,
                outcomes_by_proposal=outcomes_by_proposal,
                dataset_versions=dataset_versions,
                evaluations=evaluations,
                evidence_strength=(
                    "replicated"
                    if control.hypothesis_family_id in replicated_families
                    else "single_observation"
                ),
                hypothesis_family=(
                    hypothesis_family_by_id.get(control.hypothesis_family_id)
                    if control.hypothesis_family_id is not None
                    else None
                ),
            )
        )
    return {
        "schema_version": "bashgym.autoresearch_history.v1",
        "workspace_id": spec.workspace_id,
        "campaign_id": spec.campaign_id,
        "ledger_project_id": spec.ledger_project_id,
        "objective": objective,
        "evaluation_suite_id": spec.evaluation_suite_id,
        "primary_metric": spec.primary_metric,
        "metric_direction": spec.metric_direction.value,
        "method_thresholds": spec.method_thresholds.model_dump(mode="json", exclude_none=True),
        "total_experiments": len(ordered),
        "returned_experiments": len(experiments),
        "omitted_experiments": len(ordered) - len(experiments),
        "experiments": experiments,
        "hypothesis_families": hypothesis_families,
    }


__all__ = ["build_autoresearch_history"]
