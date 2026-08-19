"""Evidence-linked training-method readiness for an AutoResearch agent.

This module is a read-only projection.  It does not choose a method, mutate a
campaign, or authorize compute.  A method is eligible only when the installed
runner declares support and the campaign supplies both the required evidence
and its own explicit thresholds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from bashgym.campaigns.research_diagnostics import AutoResearchDiagnostics

TrainingMethod = Literal[
    "sft",
    "dpo",
    "grpo",
    "rlvr",
    "teacher_distillation",
    "session_distillation",
]

_METHODS: tuple[TrainingMethod, ...] = (
    "sft",
    "dpo",
    "grpo",
    "rlvr",
    "teacher_distillation",
    "session_distillation",
)

_REQUIREMENTS: dict[TrainingMethod, tuple[tuple[str, str, str], ...]] = {
    "sft": (
        ("demonstration_examples", "min_demonstration_examples", "min"),
        ("target_slice_coverage", "min_target_slice_coverage", "min"),
        ("contamination_rate", "max_contamination_rate", "max"),
    ),
    "dpo": (
        ("preference_pairs", "min_preference_pairs", "min"),
        (
            "preference_agreement_lower_bound",
            "min_preference_agreement_lower_bound",
            "min",
        ),
        ("ambiguous_pair_rate", "max_ambiguous_pair_rate", "max"),
        (
            "preference_position_bias_rate",
            "max_preference_position_bias_rate",
            "max",
        ),
        (
            "preference_label_conflict_rate",
            "max_preference_label_conflict_rate",
            "max",
        ),
        (
            "preference_contamination_rate",
            "max_preference_contamination_rate",
            "max",
        ),
    ),
    "grpo": (
        ("rollout_groups", "min_rollout_groups", "min"),
        ("rollout_success_rate", "min_rollout_success_rate", "min"),
        ("rollout_success_rate", "max_rollout_success_rate", "max"),
        ("zero_std_group_fraction", "max_zero_std_group_fraction", "max"),
        ("verifier_error_rate", "max_verifier_error_rate", "max"),
        ("reward_canary_cases", "min_reward_canary_cases", "min"),
        (
            "reward_canary_failure_rate",
            "max_reward_canary_failure_rate",
            "max",
        ),
        (
            "hard_constraint_violation_rate",
            "max_hard_constraint_violation_rate",
            "max",
        ),
    ),
    "rlvr": (
        ("rollout_groups", "min_rollout_groups", "min"),
        ("rollout_success_rate", "min_rollout_success_rate", "min"),
        ("rollout_success_rate", "max_rollout_success_rate", "max"),
        ("zero_std_group_fraction", "max_zero_std_group_fraction", "max"),
        ("verifier_error_rate", "max_verifier_error_rate", "max"),
        ("reward_canary_cases", "min_reward_canary_cases", "min"),
        (
            "reward_canary_failure_rate",
            "max_reward_canary_failure_rate",
            "max",
        ),
        (
            "hard_constraint_violation_rate",
            "max_hard_constraint_violation_rate",
            "max",
        ),
    ),
    "teacher_distillation": (
        ("teacher_metric_gap", "min_teacher_metric_gap", "min"),
        (
            "teacher_output_acceptance_rate",
            "min_teacher_output_acceptance_rate",
            "min",
        ),
    ),
    "session_distillation": (
        ("recovery_traces", "min_recovery_traces", "min"),
        ("recovery_lift_lower_bound", "min_recovery_lift_lower_bound", "min"),
    ),
}

_PROBES: dict[TrainingMethod, dict[str, Any]] = {
    "sft": {
        "kind": "dataset_readiness",
        "measure": [
            "demonstration_examples",
            "target_slice_coverage",
            "contamination_rate",
        ],
    },
    "dpo": {
        "kind": "preference_integrity_probe",
        "measure": [
            "preference_dataset_digest",
            "labeling_contract_digest",
            "preference_pairs",
            "preference_agreement_lower_bound",
            "ambiguous_pair_rate",
            "preference_contract_verified",
            "preference_position_bias_rate",
            "preference_label_conflict_rate",
            "preference_contamination_rate",
        ],
    },
    "grpo": {
        "kind": "verifier_rollout_probe",
        "measure": [
            "rollout_groups",
            "rollout_success_rate",
            "zero_std_group_fraction",
            "verifier_error_rate",
            "reward_spec_verified",
            "reward_canary_cases",
            "reward_canary_failure_rate",
            "hard_constraint_violation_rate",
        ],
    },
    "rlvr": {
        "kind": "verifier_rollout_probe",
        "measure": [
            "rollout_groups",
            "rollout_success_rate",
            "zero_std_group_fraction",
            "verifier_error_rate",
            "reward_spec_verified",
            "reward_canary_cases",
            "reward_canary_failure_rate",
            "hard_constraint_violation_rate",
        ],
    },
    "teacher_distillation": {
        "kind": "teacher_gap_probe",
        "measure": ["teacher_metric_gap", "teacher_output_acceptance_rate"],
    },
    "session_distillation": {
        "kind": "recovery_trace_probe",
        "measure": ["recovery_traces", "recovery_lift_lower_bound"],
    },
}

_INTERVENTION_PROBES: dict[str, dict[str, Any]] = {
    "prompt_or_context": {
        "kind": "prompt_or_context_control",
        "measure": [
            "fixed_prompt_baseline",
            "candidate_prompt_or_context_result",
            "heldout_slice_delta",
        ],
    },
    "retrieval_or_tool": {
        "kind": "retrieval_or_tool_control",
        "measure": [
            "fixed_query_baseline",
            "retrieval_or_tool_assisted_result",
            "heldout_slice_delta",
        ],
    },
    "weight_update": {
        "kind": "smallest_eligible_training_control",
        "measure": [
            "fixed_suite_delta",
            "protected_metric_deltas",
            "actual_compute_cost",
        ],
    },
    "serving_optimization": {
        "kind": "served_artifact_parity",
        "measure": [
            "training_representation_metric",
            "served_representation_metric",
            "conversion_delta",
        ],
    },
}


def _intervention_families(
    *,
    evidence: Mapping[str, Any],
    eligible_methods: Sequence[str],
    critical_codes: Sequence[str],
) -> list[dict[str, Any]]:
    """Project cheap controls beside weight-changing methods."""

    failure_mode = evidence.get("failure_mode")
    recognized_mode = (
        failure_mode
        if failure_mode
        in {
            "dynamic_knowledge",
            "missing_external_context",
            "stable_format_or_instruction",
            "serving_representation",
        }
        else None
    )
    families: list[dict[str, Any]] = []
    for family in (
        "prompt_or_context",
        "retrieval_or_tool",
        "weight_update",
        "serving_optimization",
    ):
        item: dict[str, Any] = {
            "family": family,
            "status": "diagnostic_needed",
            "evidence_for": [],
            "recommended_probe": dict(_INTERVENTION_PROBES[family]),
        }
        if critical_codes:
            item["status"] = "blocked"
            item["evidence_for"] = [f"critical_pipeline_signal:{code}" for code in critical_codes]
        elif family == "prompt_or_context" and recognized_mode == "stable_format_or_instruction":
            item["status"] = "probe_recommended"
            item["evidence_for"] = [f"failure_mode:{recognized_mode}"]
        elif family == "retrieval_or_tool" and recognized_mode in {
            "dynamic_knowledge",
            "missing_external_context",
        }:
            item["status"] = "probe_recommended"
            item["evidence_for"] = [f"failure_mode:{recognized_mode}"]
        elif family == "weight_update" and eligible_methods:
            item["status"] = "eligible"
            item["evidence_for"] = [
                f"eligible_training_method:{method}" for method in eligible_methods
            ]
        elif family == "serving_optimization" and recognized_mode == "serving_representation":
            item["status"] = "probe_recommended"
            item["evidence_for"] = [f"failure_mode:{recognized_mode}"]
        families.append(item)
    return families


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _threshold_result(
    method: TrainingMethod,
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    missing_evidence: set[str] = set()
    missing_thresholds: set[str] = set()
    failed: list[str] = []
    for evidence_key, threshold_key, direction in _REQUIREMENTS[method]:
        observed = _number(evidence.get(evidence_key))
        threshold = _number(thresholds.get(threshold_key))
        if observed is None:
            missing_evidence.add(evidence_key)
        if threshold is None:
            missing_thresholds.add(threshold_key)
        if observed is None or threshold is None:
            continue
        reason_key = "rollout_success" if evidence_key == "rollout_success_rate" else evidence_key
        if direction == "min" and observed < threshold:
            failed.append(f"{reason_key}_below_threshold")
        elif direction == "max" and observed > threshold:
            failed.append(f"{reason_key}_above_threshold")
    if method in {"grpo", "rlvr"}:
        verified = evidence.get("reward_spec_verified")
        if verified is None:
            missing_evidence.add("reward_spec_verified")
        elif verified is not True:
            failed.append("reward_spec_not_verified")
    if method == "dpo":
        verified = evidence.get("preference_contract_verified")
        if verified is None:
            missing_evidence.add("preference_contract_verified")
        elif verified is not True:
            failed.append("preference_contract_not_verified")
    return sorted(missing_evidence), sorted(missing_thresholds), failed


def _evidence_for(method: TrainingMethod) -> list[str]:
    if method == "sft":
        return [
            "demonstration_count_meets_threshold",
            "target_slices_are_covered",
            "contamination_rate_within_threshold",
        ]
    if method == "dpo":
        return [
            "preference_pair_count_meets_threshold",
            "preference_labels_separate_above_chance",
            "ambiguous_pair_rate_within_threshold",
            "preference_contract_is_verified",
            "preference_position_bias_within_threshold",
            "preference_label_conflicts_within_threshold",
            "preference_contamination_within_threshold",
        ]
    if method in {"grpo", "rlvr"}:
        return [
            "rollout_groups_meet_threshold",
            "rollout_success_has_learning_contrast",
            "reward_groups_have_variance",
            "verifier_error_rate_within_threshold",
            "reward_spec_is_verified",
            "reward_canaries_within_threshold",
            "hard_reward_constraints_within_threshold",
        ]
    if method == "teacher_distillation":
        return [
            "teacher_outperforms_student",
            "teacher_outputs_pass_validation",
        ]
    return ["recovery_traces_meet_threshold", "recovery_lift_is_positive"]


def _probe_after_failure(method: TrainingMethod, failed: Sequence[str]) -> dict[str, Any]:
    if method == "dpo" and any(
        reason
        in {
            "preference_contract_not_verified",
            "preference_position_bias_rate_above_threshold",
            "preference_label_conflict_rate_above_threshold",
            "preference_contamination_rate_above_threshold",
        }
        for reason in failed
    ):
        return {
            "kind": "preference_integrity_probe",
            "measure": [
                "preference_dataset_digest",
                "labeling_contract_digest",
                "preference_agreement_lower_bound",
                "ambiguous_pair_rate",
                "preference_position_bias_rate",
                "preference_label_conflict_rate",
                "preference_contamination_rate",
            ],
        }
    if method in {"grpo", "rlvr"} and any(
        reason
        in {
            "reward_spec_not_verified",
            "reward_canary_failure_rate_above_threshold",
            "hard_constraint_violation_rate_above_threshold",
        }
        for reason in failed
    ):
        return {
            "kind": "reward_integrity_probe",
            "measure": [
                "reward_spec_digest",
                "reward_component_distributions",
                "hard_constraint_violation_rate",
                "reward_canary_cases",
                "reward_canary_failure_rate",
            ],
        }
    if method in {"grpo", "rlvr"} and (
        "rollout_success_below_threshold" in failed
        or "zero_std_group_fraction_above_threshold" in failed
    ):
        return {
            "kind": "curriculum_or_supervised_warm_start",
            "measure": [
                "demonstration_coverage",
                "easier_task_success_rate",
                "post_warm_start_rollout_success_rate",
            ],
        }
    return dict(_PROBES[method])


def build_method_selection_packet(
    *,
    diagnostics: AutoResearchDiagnostics,
    supported_methods: Sequence[str],
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Explain method readiness without selecting or launching a method."""

    supported = frozenset(value for value in supported_methods if value in _METHODS)
    critical_codes = sorted(
        signal.code for signal in diagnostics.signals if signal.severity == "critical"
    )
    methods: list[dict[str, Any]] = []
    eligible: list[str] = []
    for method in _METHODS:
        item: dict[str, Any] = {
            "method": method,
            "status": "diagnostic_needed",
            "evidence_for": [],
            "blocking_reasons": [],
            "missing_evidence": [],
            "missing_thresholds": [],
            "recommended_probe": dict(_PROBES[method]),
        }
        if method not in supported:
            item["status"] = "unsupported_by_runner"
            item["blocking_reasons"] = ["installed_runner_did_not_declare_method"]
            methods.append(item)
            continue
        if critical_codes:
            item["status"] = "blocked"
            item["blocking_reasons"] = [
                f"critical_pipeline_signal:{code}" for code in critical_codes
            ]
            methods.append(item)
            continue
        missing_evidence, missing_thresholds, failed = _threshold_result(
            method, evidence, thresholds
        )
        item["missing_evidence"] = missing_evidence
        item["missing_thresholds"] = missing_thresholds
        if failed:
            item["status"] = "blocked"
            item["blocking_reasons"] = failed
            item["recommended_probe"] = _probe_after_failure(method, failed)
        elif not missing_evidence and not missing_thresholds:
            item["status"] = "eligible"
            item["evidence_for"] = _evidence_for(method)
            eligible.append(method)
        methods.append(item)
    intervention_families = _intervention_families(
        evidence=evidence,
        eligible_methods=eligible,
        critical_codes=critical_codes,
    )
    retrieval_probe_required = any(
        item["family"] == "retrieval_or_tool" and item["status"] == "probe_recommended"
        for item in intervention_families
    )
    return {
        "schema_version": "bashgym.autoresearch_method_selection.v1",
        "recommended_action": (
            "diagnose_before_training"
            if critical_codes
            else (
                "probe_non_weight_intervention"
                if retrieval_probe_required
                else "agent_select_or_probe"
            )
        ),
        "selection_authority": "host_agent",
        "thresholds": dict(sorted(thresholds.items())),
        "eligible_methods": eligible,
        "methods": methods,
        "recommended_intervention_families": intervention_families,
    }


__all__ = ["build_method_selection_packet"]
