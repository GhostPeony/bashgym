import pytest
from pydantic import ValidationError

from bashgym.campaigns.method_policy import AutoResearchMethodThresholds
from bashgym.campaigns.method_selection import build_method_selection_packet
from bashgym.campaigns.research_diagnostics import (
    AutoResearchDiagnostics,
    AutoResearchDiagnosticSignal,
)


def _diagnostics(*signals: AutoResearchDiagnosticSignal) -> AutoResearchDiagnostics:
    return AutoResearchDiagnostics(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        primary_metric="task_success",
        metric_direction="maximize",
        low_signal=not signals,
        signals=signals,
    )


def test_method_selection_blocks_training_when_pipeline_integrity_is_critical():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(
            AutoResearchDiagnosticSignal(
                code="degenerate_constant_output",
                severity="critical",
                summary="Every prediction is identical.",
                evidence_references=("evaluation-candidate",),
            )
        ),
        supported_methods=("sft", "grpo"),
        evidence={},
        thresholds={},
    )

    assert packet["recommended_action"] == "diagnose_before_training"
    assert {item["method"]: item["status"] for item in packet["methods"]} == {
        "sft": "blocked",
        "dpo": "unsupported_by_runner",
        "grpo": "blocked",
        "rlvr": "unsupported_by_runner",
        "teacher_distillation": "unsupported_by_runner",
        "session_distillation": "unsupported_by_runner",
    }
    assert packet["methods"][0]["blocking_reasons"] == [
        "critical_pipeline_signal:degenerate_constant_output"
    ]


def test_method_selection_requires_explicit_thresholds_before_sft_is_eligible():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=("sft",),
        evidence={
            "demonstration_examples": 96,
            "target_slice_coverage": 0.8,
            "contamination_rate": 0.0,
        },
        thresholds={},
    )

    sft = packet["methods"][0]
    assert sft["method"] == "sft"
    assert sft["status"] == "diagnostic_needed"
    assert sft["missing_thresholds"] == [
        "max_contamination_rate",
        "min_demonstration_examples",
        "min_target_slice_coverage",
    ]
    assert sft["recommended_probe"]["kind"] == "dataset_readiness"


def test_method_selection_distinguishes_dpo_and_verifier_rl_readiness():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=("dpo", "grpo", "rlvr"),
        evidence={
            "preference_contract_verified": True,
            "preference_pairs": 240,
            "preference_agreement_lower_bound": 0.72,
            "ambiguous_pair_rate": 0.04,
            "preference_position_bias_rate": 0.02,
            "preference_label_conflict_rate": 0.0,
            "preference_contamination_rate": 0.0,
            "rollout_groups": 64,
            "rollout_success_rate": 0.35,
            "zero_std_group_fraction": 0.2,
            "verifier_error_rate": 0.0,
            "reward_spec_verified": True,
            "reward_canary_cases": 4,
            "reward_canary_failure_rate": 0.0,
            "hard_constraint_violation_rate": 0.0,
        },
        thresholds={
            "min_preference_pairs": 128,
            "min_preference_agreement_lower_bound": 0.6,
            "max_ambiguous_pair_rate": 0.1,
            "max_preference_position_bias_rate": 0.05,
            "max_preference_label_conflict_rate": 0.0,
            "max_preference_contamination_rate": 0.0,
            "min_rollout_groups": 32,
            "min_rollout_success_rate": 0.05,
            "max_rollout_success_rate": 0.95,
            "max_zero_std_group_fraction": 0.5,
            "max_verifier_error_rate": 0.0,
            "min_reward_canary_cases": 4,
            "max_reward_canary_failure_rate": 0.0,
            "max_hard_constraint_violation_rate": 0.0,
        },
    )

    methods = {item["method"]: item for item in packet["methods"]}
    assert methods["dpo"]["status"] == "eligible"
    assert methods["grpo"]["status"] == "eligible"
    assert methods["rlvr"]["status"] == "eligible"
    assert methods["grpo"]["evidence_for"] == [
        "rollout_groups_meet_threshold",
        "rollout_success_has_learning_contrast",
        "reward_groups_have_variance",
        "verifier_error_rate_within_threshold",
        "reward_spec_is_verified",
        "reward_canaries_within_threshold",
        "hard_reward_constraints_within_threshold",
    ]
    assert packet["eligible_methods"] == ["dpo", "grpo", "rlvr"]


def test_dpo_requires_verified_preference_integrity_and_blocks_biased_labels():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=("dpo",),
        evidence={
            "preference_contract_verified": False,
            "preference_pairs": 240,
            "preference_agreement_lower_bound": 0.72,
            "ambiguous_pair_rate": 0.04,
            "preference_position_bias_rate": 0.15,
            "preference_label_conflict_rate": 0.04,
            "preference_contamination_rate": 0.02,
        },
        thresholds={
            "min_preference_pairs": 128,
            "min_preference_agreement_lower_bound": 0.6,
            "max_ambiguous_pair_rate": 0.1,
            "max_preference_position_bias_rate": 0.05,
            "max_preference_label_conflict_rate": 0.0,
            "max_preference_contamination_rate": 0.0,
        },
    )

    dpo = {item["method"]: item for item in packet["methods"]}["dpo"]
    assert dpo["status"] == "blocked"
    assert dpo["blocking_reasons"] == [
        "preference_position_bias_rate_above_threshold",
        "preference_label_conflict_rate_above_threshold",
        "preference_contamination_rate_above_threshold",
        "preference_contract_not_verified",
    ]
    assert dpo["recommended_probe"]["kind"] == "preference_integrity_probe"


def test_method_selection_recommends_sft_or_curriculum_when_rl_has_no_successes():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=("sft", "grpo"),
        evidence={
            "rollout_groups": 64,
            "rollout_success_rate": 0.0,
            "zero_std_group_fraction": 1.0,
            "verifier_error_rate": 0.0,
            "reward_spec_verified": True,
            "reward_canary_cases": 4,
            "reward_canary_failure_rate": 0.0,
            "hard_constraint_violation_rate": 0.0,
        },
        thresholds={
            "min_rollout_groups": 32,
            "min_rollout_success_rate": 0.05,
            "max_rollout_success_rate": 0.95,
            "max_zero_std_group_fraction": 0.5,
            "max_verifier_error_rate": 0.0,
            "min_reward_canary_cases": 4,
            "max_reward_canary_failure_rate": 0.0,
            "max_hard_constraint_violation_rate": 0.0,
        },
    )

    grpo = {item["method"]: item for item in packet["methods"]}["grpo"]
    assert grpo["status"] == "blocked"
    assert grpo["blocking_reasons"] == [
        "rollout_success_below_threshold",
        "zero_std_group_fraction_above_threshold",
    ]
    assert grpo["recommended_probe"] == {
        "kind": "curriculum_or_supervised_warm_start",
        "measure": [
            "demonstration_coverage",
            "easier_task_success_rate",
            "post_warm_start_rollout_success_rate",
        ],
    }


def test_verifier_rl_is_blocked_by_failed_hard_constraint_or_exploit_canary():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=("grpo", "rlvr"),
        evidence={
            "rollout_groups": 64,
            "rollout_success_rate": 0.35,
            "zero_std_group_fraction": 0.2,
            "verifier_error_rate": 0.0,
            "reward_spec_verified": True,
            "reward_canary_cases": 4,
            "reward_canary_failure_rate": 0.25,
            "hard_constraint_violation_rate": 0.1,
        },
        thresholds={
            "min_rollout_groups": 32,
            "min_rollout_success_rate": 0.05,
            "max_rollout_success_rate": 0.95,
            "max_zero_std_group_fraction": 0.5,
            "max_verifier_error_rate": 0.0,
            "min_reward_canary_cases": 4,
            "max_reward_canary_failure_rate": 0.0,
            "max_hard_constraint_violation_rate": 0.0,
        },
    )

    methods = {item["method"]: item for item in packet["methods"]}
    assert methods["grpo"]["status"] == "blocked"
    assert methods["grpo"]["blocking_reasons"] == [
        "reward_canary_failure_rate_above_threshold",
        "hard_constraint_violation_rate_above_threshold",
    ]
    assert methods["rlvr"]["status"] == "blocked"
    assert methods["grpo"]["recommended_probe"] == {
        "kind": "reward_integrity_probe",
        "measure": [
            "reward_spec_digest",
            "reward_component_distributions",
            "hard_constraint_violation_rate",
            "reward_canary_cases",
            "reward_canary_failure_rate",
        ],
    }


def test_verifier_rl_requires_a_verified_reward_spec_even_with_reward_variance():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=("grpo",),
        evidence={
            "rollout_groups": 64,
            "rollout_success_rate": 0.35,
            "zero_std_group_fraction": 0.2,
            "verifier_error_rate": 0.0,
            "reward_spec_verified": False,
            "reward_canary_cases": 4,
            "reward_canary_failure_rate": 0.0,
            "hard_constraint_violation_rate": 0.0,
        },
        thresholds={
            "min_rollout_groups": 32,
            "min_rollout_success_rate": 0.05,
            "max_rollout_success_rate": 0.95,
            "max_zero_std_group_fraction": 0.5,
            "max_verifier_error_rate": 0.0,
            "min_reward_canary_cases": 4,
            "max_reward_canary_failure_rate": 0.0,
            "max_hard_constraint_violation_rate": 0.0,
        },
    )

    grpo = {item["method"]: item for item in packet["methods"]}["grpo"]
    assert grpo["status"] == "blocked"
    assert grpo["blocking_reasons"] == ["reward_spec_not_verified"]
    assert grpo["recommended_probe"]["kind"] == "reward_integrity_probe"


def test_dynamic_knowledge_failure_recommends_retrieval_probe_before_weight_update():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=(),
        evidence={"failure_mode": "dynamic_knowledge"},
        thresholds={},
    )

    families = {item["family"]: item for item in packet["recommended_intervention_families"]}
    assert packet["recommended_action"] == "probe_non_weight_intervention"
    assert families["retrieval_or_tool"] == {
        "family": "retrieval_or_tool",
        "status": "probe_recommended",
        "evidence_for": ["failure_mode:dynamic_knowledge"],
        "recommended_probe": {
            "kind": "retrieval_or_tool_control",
            "measure": [
                "fixed_query_baseline",
                "retrieval_or_tool_assisted_result",
                "heldout_slice_delta",
            ],
        },
    }
    assert families["weight_update"]["status"] == "diagnostic_needed"
    assert packet["selection_authority"] == "host_agent"


def test_stable_format_failure_exposes_prompt_probe_and_eligible_sft():
    packet = build_method_selection_packet(
        diagnostics=_diagnostics(),
        supported_methods=("sft",),
        evidence={
            "failure_mode": "stable_format_or_instruction",
            "demonstration_examples": 96,
            "target_slice_coverage": 0.8,
            "contamination_rate": 0.0,
        },
        thresholds={
            "min_demonstration_examples": 64,
            "min_target_slice_coverage": 0.75,
            "max_contamination_rate": 0.01,
        },
    )

    families = {item["family"]: item for item in packet["recommended_intervention_families"]}
    assert packet["recommended_action"] == "agent_select_or_probe"
    assert families["prompt_or_context"]["status"] == "probe_recommended"
    assert families["prompt_or_context"]["evidence_for"] == [
        "failure_mode:stable_format_or_instruction"
    ]
    assert families["weight_update"] == {
        "family": "weight_update",
        "status": "eligible",
        "evidence_for": ["eligible_training_method:sft"],
        "recommended_probe": {
            "kind": "smallest_eligible_training_control",
            "measure": [
                "fixed_suite_delta",
                "protected_metric_deltas",
                "actual_compute_cost",
            ],
        },
    }
    assert packet["eligible_methods"] == ["sft"]


def test_persisted_method_thresholds_validate_bounds_and_canonical_shape():
    thresholds = AutoResearchMethodThresholds(
        min_demonstration_examples=64,
        min_target_slice_coverage=0.8,
        max_contamination_rate=0.01,
        max_preference_position_bias_rate=0.05,
        max_preference_label_conflict_rate=0.0,
        max_preference_contamination_rate=0.0,
        min_rollout_groups=32,
        min_rollout_success_rate=0.05,
        max_rollout_success_rate=0.95,
        min_reward_canary_cases=4,
        max_reward_canary_failure_rate=0.0,
        max_hard_constraint_violation_rate=0.0,
    )

    assert thresholds.model_dump(mode="json", exclude_none=True) == {
        "schema_version": "autoresearch_method_thresholds.v1",
        "min_demonstration_examples": 64,
        "min_target_slice_coverage": 0.8,
        "max_contamination_rate": 0.01,
        "max_preference_position_bias_rate": 0.05,
        "max_preference_label_conflict_rate": 0.0,
        "max_preference_contamination_rate": 0.0,
        "min_rollout_groups": 32,
        "min_rollout_success_rate": 0.05,
        "max_rollout_success_rate": 0.95,
        "min_reward_canary_cases": 4,
        "max_reward_canary_failure_rate": 0.0,
        "max_hard_constraint_violation_rate": 0.0,
    }
    assert AutoResearchMethodThresholds.model_validate({}).model_dump(exclude_none=True) == {
        "schema_version": "autoresearch_method_thresholds.v1"
    }


@pytest.mark.parametrize(
    "payload",
    (
        {"min_demonstration_examples": 0},
        {"max_contamination_rate": 1.1},
        {"max_preference_position_bias_rate": 1.1},
        {"max_preference_label_conflict_rate": -0.1},
        {"max_preference_contamination_rate": 1.1},
        {"min_rollout_success_rate": 0.8, "max_rollout_success_rate": 0.2},
        {"min_reward_canary_cases": 0},
        {"max_reward_canary_failure_rate": 1.1},
        {"max_hard_constraint_violation_rate": -0.1},
        {"allowed_methods": ["sft"]},
    ),
)
def test_persisted_method_thresholds_reject_invalid_or_authorizing_values(payload):
    with pytest.raises(ValidationError):
        AutoResearchMethodThresholds.model_validate(payload)
