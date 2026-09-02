"""Compact scientific context for the next AutoResearch decision.

The packet is a read-only projection of existing campaign records. It carries
enough evidence-linked context for a host agent to decide what to inspect or
change next without duplicating BashGym's campaign state or becoming a planner.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignSpec,
    AutoResearchNextAction,
    AutoResearchOutcomeRecord,
    AutoResearchProposalControl,
    AutoResearchState,
)
from bashgym.campaigns.contracts import StudyProposal
from bashgym.campaigns.method_selection import build_method_selection_packet
from bashgym.campaigns.outcome_assessment import build_outcome_assessment
from bashgym.campaigns.research_diagnostics import AutoResearchDiagnostics
from bashgym.campaigns.tmax_recipe import TMAX_COMPOSITE_TRAINING_RECIPE_SCHEMA

_MAX_SIGNALS = 5
_MAX_CHECKPOINTS = 5
_MAX_ERROR_SLICES = 5
_MAX_HYPOTHESES = 3
_MAX_KNOWLEDGE_ENTRIES = 5
_MAX_HYPOTHESIS_FAMILIES = 5
_AGENT_ACTIONS = frozenset(
    {
        AutoResearchNextAction.PREPARE_CAMPAIGN,
        AutoResearchNextAction.START_CAMPAIGN,
        AutoResearchNextAction.SUBMIT_BASELINE,
        AutoResearchNextAction.PROPOSE_CANDIDATE,
        AutoResearchNextAction.BLOCKED,
    }
)


def latest_data_quality_for_outcome(
    dataset_versions: Sequence[Mapping[str, Any]],
    outcome: AutoResearchOutcomeRecord | None,
) -> dict[str, Any] | None:
    """Select quality metadata bound to one of the outcome's exact attempts."""

    if outcome is None:
        return None
    attempt_ids = frozenset(outcome.result.attempt_ids)
    for version in reversed(dataset_versions):
        metadata = version.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        quality = metadata.get("data_quality")
        if metadata.get("producer_attempt_id") in attempt_ids and isinstance(quality, Mapping):
            return deepcopy(dict(quality))
    return None


def _append_references(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _project_last_experiment(
    proposal: StudyProposal | None,
    control: AutoResearchProposalControl | None,
) -> dict[str, Any] | None:
    if proposal is None:
        return None
    return {
        "proposal_id": proposal.proposal_id,
        "role": control.role.value if control is not None else None,
        "parent_proposal_id": control.parent_proposal_id if control is not None else None,
        "intervention": (
            {
                "mode": control.intervention_mode.value,
                "changed_variables": list(control.changed_variables),
                "hypothesis_family_id": control.hypothesis_family_id,
            }
            if control is not None
            else None
        ),
        "hypothesis": proposal.hypothesis,
        "changed_variable": proposal.primary_variable,
        "expected_outcome": proposal.expected_outcome,
        "falsification_criterion": proposal.falsification_criterion,
        "stages": [item.stage.value for item in proposal.stage_plan.items],
    }


def _project_result(outcome: AutoResearchOutcomeRecord | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    result = outcome.result
    decision = outcome.decision
    return {
        "proposal_id": result.proposal_id,
        "outcome": result.outcome.value,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "metrics": dict(sorted(result.metrics.items())),
        "actual_cost": result.actual_cost,
        "decision": decision.decision.value,
        "reason_code": decision.reason_code,
        "improvement": decision.improvement,
    }


def _bounded_diagnostics(
    diagnostics: AutoResearchDiagnostics,
) -> tuple[dict[str, Any], list[str]]:
    signals = diagnostics.signals[:_MAX_SIGNALS]
    checkpoints = diagnostics.checkpoint_comparisons[-_MAX_CHECKPOINTS:]
    error_slices = diagnostics.error_slices[:_MAX_ERROR_SLICES]
    hypotheses = diagnostics.ranked_hypotheses[:_MAX_HYPOTHESES]
    references: list[str] = []
    for item in (*signals, *error_slices, *hypotheses):
        _append_references(references, item.evidence_references)
    for checkpoint in checkpoints:
        _append_references(references, (checkpoint.evaluation_result_id,))
    return (
        {
            "low_signal": diagnostics.low_signal,
            "signals": [item.model_dump(mode="json") for item in signals],
            "checkpoint_comparisons": [item.model_dump(mode="json") for item in checkpoints],
            "error_slices": [item.model_dump(mode="json") for item in error_slices],
            "ranked_hypotheses": [item.model_dump(mode="json") for item in hypotheses],
        },
        references,
    )


def _bounded_campaign_knowledge(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "schema_version": "bashgym.autoresearch_history.v1",
            "total_experiments": 0,
            "returned_experiments": 0,
            "omitted_experiments": 0,
            "experiments": [],
        }
    payload = deepcopy(dict(value))
    experiments = value.get("experiments")
    entries = list(experiments) if isinstance(experiments, (list, tuple)) else []
    selected = entries[-_MAX_KNOWLEDGE_ENTRIES:]
    total = value.get("total_experiments")
    total_count = total if isinstance(total, int) and total >= len(entries) else len(entries)
    payload["total_experiments"] = total_count
    payload["returned_experiments"] = len(selected)
    payload["omitted_experiments"] = max(0, total_count - len(selected))
    payload["experiments"] = deepcopy(selected)
    families = value.get("hypothesis_families")
    if isinstance(families, (list, tuple)):
        payload["hypothesis_families"] = deepcopy(list(families)[-_MAX_HYPOTHESIS_FAMILIES:])
    return payload


def _typed_runner_methods(proposal: StudyProposal | None) -> tuple[str, ...]:
    """Return methods proven by a validated, installation-owned recipe ABI."""

    if (
        proposal is not None
        and proposal.training_recipe.get("schema_version") == TMAX_COMPOSITE_TRAINING_RECIPE_SCHEMA
    ):
        return ("grpo",)
    return ()


def method_evidence_from_diagnostic_results(
    diagnostic_results: Sequence[Mapping[str, Any] | Any],
) -> dict[str, bool | float]:
    """Return latest complete contract-bound measurements for method readiness."""

    probe_contracts = {
        "reward_integrity_probe": {
            "measurements": {
                "reward_canary_cases",
                "reward_canary_failure_rate",
                "hard_constraint_violation_rate",
            },
            "verified_key": "reward_spec_verified",
        },
        "preference_integrity_probe": {
            "measurements": {
                "preference_pairs",
                "preference_agreement_lower_bound",
                "ambiguous_pair_rate",
                "preference_position_bias_rate",
                "preference_label_conflict_rate",
                "preference_contamination_rate",
            },
            "verified_key": "preference_contract_verified",
        },
        "teacher_gap_probe": {
            "measurements": {
                "teacher_metric_gap",
                "teacher_output_acceptance_rate",
            },
            "verified_key": None,
        },
        "recovery_trace_probe": {
            "measurements": {"recovery_traces", "recovery_lift_lower_bound"},
            "verified_key": None,
        },
    }

    def valid_contract(probe_family: str, contract: Mapping[str, Any]) -> bool:
        def digest(key: str) -> bool:
            return re.fullmatch(r"[0-9a-f]{64}", str(contract.get(key) or "")) is not None

        if probe_family == "reward_integrity_probe":
            return digest("reward_spec_digest") and isinstance(contract.get("canary_suite_id"), str)
        if probe_family == "preference_integrity_probe":
            return digest("preference_dataset_digest") and digest("labeling_contract_digest")
        if probe_family == "teacher_gap_probe":
            return (
                isinstance(contract.get("evaluation_suite_id"), str)
                and contract.get("metric_direction") in {"maximize", "minimize"}
                and digest("teacher_model_digest")
                and digest("student_model_digest")
                and digest("output_validation_contract_digest")
                and contract.get("teacher_model_digest") != contract.get("student_model_digest")
            )
        return (
            digest("recovery_dataset_digest")
            and digest("reader_contract_digest")
            and contract.get("confidence_level") == 0.95
        )

    latest_by_probe: dict[str, dict[str, bool | float]] = {}
    for result in diagnostic_results:
        value = getattr(result, "projection", result)
        if not isinstance(value, Mapping):
            continue
        probe_family = value.get("probe_family")
        probe_contract = probe_contracts.get(str(probe_family))
        if probe_contract is None or value.get("status") != "completed":
            continue
        contract = value.get("comparison_contract")
        if not isinstance(contract, Mapping) or not valid_contract(str(probe_family), contract):
            continue
        required = probe_contract["measurements"]
        measurements = value.get("measurements")
        if not isinstance(measurements, Sequence) or isinstance(
            measurements, (str, bytes, bytearray)
        ):
            continue
        observed: dict[str, float] = {}
        for item in measurements:
            if not isinstance(item, Mapping) or item.get("name") not in required:
                continue
            metric = item.get("value")
            if isinstance(metric, bool) or not isinstance(metric, (int, float)):
                continue
            numeric = float(metric)
            if math.isfinite(numeric):
                observed[str(item["name"])] = numeric
        if set(observed) == required:
            verified_key = probe_contract["verified_key"]
            latest_by_probe[str(probe_family)] = (
                {str(verified_key): True, **observed} if verified_key else observed
            )
    combined: dict[str, bool | float] = {}
    for probe_family in probe_contracts:
        combined.update(latest_by_probe.get(probe_family, {}))
    return combined


def build_decision_packet(
    *,
    objective: str,
    spec: AutoResearchCampaignSpec,
    state: AutoResearchState,
    diagnostics: AutoResearchDiagnostics,
    latest_proposal: StudyProposal | None = None,
    latest_control: AutoResearchProposalControl | None = None,
    latest_outcome: AutoResearchOutcomeRecord | None = None,
    latest_data_quality: Mapping[str, Any] | None = None,
    current_work: Mapping[str, Any] | None = None,
    campaign_knowledge: Mapping[str, Any] | None = None,
    supported_methods: Sequence[str] = (),
    method_evidence: Mapping[str, Any] | None = None,
    method_thresholds: Mapping[str, Any] | None = None,
    failure_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project bounded decision context from authoritative campaign records."""

    diagnostic_payload, diagnostic_references = _bounded_diagnostics(diagnostics)
    evidence_references: list[str] = []
    if latest_proposal is not None:
        _append_references(evidence_references, latest_proposal.evidence_references)
    if latest_outcome is not None:
        _append_references(
            evidence_references,
            latest_outcome.result.evidence_references,
        )
    _append_references(evidence_references, diagnostic_references)

    rules = spec.stop_rules
    selection_evidence: dict[str, Any] = {}
    if latest_data_quality is not None:
        selection_evidence.update(latest_data_quality)
    if latest_outcome is not None:
        selection_evidence.update(latest_outcome.result.metrics)
    if method_evidence is not None:
        selection_evidence.update(method_evidence)
    packet = {
        "schema_version": "bashgym.autoresearch_decision_packet.v1",
        "campaign": {
            "objective": objective,
            "primary_metric": spec.primary_metric,
            "metric_direction": spec.metric_direction.value,
            "baseline_verified": state.baseline_verified,
            "current_reference": {
                "proposal_id": state.best_proposal_id,
                "metric": state.best_metric,
            },
            "stop_rules": {
                "max_attempts": rules.max_attempts,
                "budget_unit": rules.budget_unit,
                "max_total_cost": rules.max_total_cost,
                "target_metric": rules.target_metric,
                "minimum_improvement": rules.minimum_improvement,
                "protected_metrics": [
                    gate.model_dump(mode="json") for gate in rules.protected_metrics
                ],
                "deadline": rules.deadline.isoformat() if rules.deadline else None,
            },
        },
        "current_work": deepcopy(dict(current_work)) if current_work is not None else None,
        "last_experiment": _project_last_experiment(latest_proposal, latest_control),
        "result": _project_result(latest_outcome),
        "diagnostics": diagnostic_payload,
        "resources": {
            "attempts_used": state.attempts_used,
            "proposals_used": state.proposals_used,
            "budget_unit": rules.budget_unit,
            "budget_used": state.budget_used,
            "budget_remaining": state.budget_remaining,
        },
        "decision_required": {
            "action": state.next_action.value,
            "reason_code": state.reason_code,
            "agent_action_required": state.next_action in _AGENT_ACTIONS,
        },
        "method_selection": build_method_selection_packet(
            diagnostics=diagnostics,
            supported_methods=supported_methods or _typed_runner_methods(latest_proposal),
            evidence=selection_evidence,
            thresholds=method_thresholds or {},
        ),
        "failure_analysis": (
            deepcopy(dict(failure_analysis)) if failure_analysis is not None else None
        ),
        "outcome_assessment": (
            build_outcome_assessment(
                outcome=latest_outcome.result.outcome.value,
                provenance=latest_outcome.result.provenance.value,
                decision=latest_outcome.decision.decision.value,
                reason_code=latest_outcome.decision.reason_code,
                failure_analysis=failure_analysis,
            )
            if latest_outcome is not None
            else None
        ),
        "campaign_knowledge": _bounded_campaign_knowledge(campaign_knowledge),
        "evidence_references": evidence_references,
    }
    if latest_data_quality is not None:
        packet["data_quality"] = deepcopy(dict(latest_data_quality))
    return packet


__all__ = [
    "build_decision_packet",
    "latest_data_quality_for_outcome",
    "method_evidence_from_diagnostic_results",
]
