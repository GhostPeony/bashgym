"""Scientific outcome semantics projected from durable AutoResearch evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_MAX_CATEGORIES = 12
_EVIDENCE_STRENGTHS = frozenset({"single_observation", "replicated", "not_applicable"})


def _comparison_categories(
    failure_analysis: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    if not isinstance(failure_analysis, Mapping):
        return [], []
    comparison = failure_analysis.get("comparison")
    if isinstance(comparison, (str, bytes, bytearray)) or not isinstance(comparison, Sequence):
        return [], []
    tradeoffs: list[str] = []
    improvements: list[str] = []
    for item in comparison[:_MAX_CATEGORIES]:
        if not isinstance(item, Mapping):
            continue
        category = item.get("category")
        status = item.get("status")
        if not isinstance(category, str) or not category:
            continue
        target = (
            tradeoffs if status == "regressed" else improvements if status == "improved" else None
        )
        if target is not None and category not in target:
            target.append(category)
    return tradeoffs, improvements


def build_outcome_assessment(
    *,
    outcome: str,
    provenance: str,
    decision: str,
    reason_code: str,
    failure_analysis: Mapping[str, Any] | None,
    evidence_strength: str = "single_observation",
) -> dict[str, Any]:
    """Explain one persisted decision without changing reference-selection policy."""

    if evidence_strength not in _EVIDENCE_STRENGTHS:
        raise ValueError("autoresearch_outcome_evidence_strength_invalid")
    tradeoffs, improvements = _comparison_categories(failure_analysis)

    classification = "inconclusive"
    is_failure = False
    failure_kind = None
    assessment_reason = "completed_evidence_inconclusive"

    if outcome == "crashed":
        classification = "invalid_execution"
        is_failure = True
        failure_kind = "execution"
        assessment_reason = "execution_did_not_produce_quality_evidence"
    elif outcome != "completed":
        assessment_reason = "quality_evidence_incomplete"
    elif provenance != "real" or decision == "ineligible":
        classification = "ineligible"
        assessment_reason = "result_not_eligible_quality_evidence"
        evidence_strength = "not_applicable"
    elif decision == "baseline":
        classification = "baseline"
        assessment_reason = "baseline_recorded"
        evidence_strength = "not_applicable"
    elif reason_code == "candidate_failed_protected_metric_gate":
        classification = "unacceptable_regression"
        is_failure = True
        failure_kind = "scientific_guardrail"
        assessment_reason = "predeclared_protected_metric_limit_exceeded"
    elif decision == "keep":
        if tradeoffs:
            classification = "acceptable_tradeoff"
            assessment_reason = "primary_gain_with_nonblocking_tradeoff"
        else:
            classification = "clear_improvement"
            assessment_reason = "primary_and_protected_gates_cleared"
    elif decision == "discard":
        if improvements:
            classification = "mixed_evidence"
            assessment_reason = "mixed_evidence_not_retained"
        else:
            classification = "no_demonstrated_gain"
            assessment_reason = "primary_gate_not_cleared"

    return {
        "schema_version": "bashgym.autoresearch_outcome_assessment.v1",
        "classification": classification,
        "is_failure": is_failure,
        "failure_kind": failure_kind,
        "decision": decision,
        "reason_code": assessment_reason,
        "observed_tradeoffs": tradeoffs,
        "observed_improvements": improvements,
        "evidence_strength": evidence_strength,
    }


__all__ = ["build_outcome_assessment"]
