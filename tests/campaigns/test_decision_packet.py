from datetime import UTC, datetime

from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignSpec,
    AutoResearchDecision,
    AutoResearchNextAction,
    AutoResearchOutcomeRecord,
    AutoResearchProposalControl,
    AutoResearchResult,
    AutoResearchState,
    AutoResearchStopRules,
    ExperimentOutcome,
    ExperimentProvenance,
    ExperimentRole,
    InterventionMode,
    MetricDirection,
    ResultDecision,
)
from bashgym.campaigns.contracts import (
    CampaignStatus,
    ProposalStatus,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyProposal,
)
from bashgym.campaigns.decision_packet import (
    build_decision_packet,
    latest_data_quality_for_outcome,
    method_evidence_from_diagnostic_results,
)
from bashgym.campaigns.research_diagnostics import (
    AutoResearchDiagnostics,
    AutoResearchDiagnosticSignal,
    AutoResearchRankedHypothesis,
)

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)


def test_reward_integrity_diagnostic_becomes_method_evidence_without_raw_details() -> None:
    evidence = method_evidence_from_diagnostic_results(
        (
            {
                "probe_family": "reward_integrity_probe",
                "status": "completed",
                "comparison_contract": {
                    "reward_spec_digest": "a" * 64,
                    "canary_suite_id": "reward-hacking-v1",
                },
                "measurements": [
                    {"name": "reward_canary_cases", "value": 4.0},
                    {"name": "reward_canary_failure_rate", "value": 0.0},
                    {"name": "hard_constraint_violation_rate", "value": 0.0},
                ],
            },
        )
    )

    assert evidence == {
        "reward_spec_verified": True,
        "reward_canary_cases": 4.0,
        "reward_canary_failure_rate": 0.0,
        "hard_constraint_violation_rate": 0.0,
    }


def test_incomplete_or_unsupported_reward_diagnostic_does_not_claim_verification() -> None:
    assert (
        method_evidence_from_diagnostic_results(
            (
                {
                    "probe_family": "reward_integrity_probe",
                    "status": "completed",
                    "comparison_contract": {"reward_spec_digest": "a" * 64},
                    "measurements": [
                        {"name": "reward_canary_cases", "value": 4.0},
                    ],
                },
                {
                    "probe_family": "reward_integrity_probe",
                    "status": "unsupported",
                    "measurements": [],
                },
            )
        )
        == {}
    )


def test_preference_integrity_diagnostic_becomes_dpo_evidence_without_raw_pairs() -> None:
    evidence = method_evidence_from_diagnostic_results(
        (
            {
                "probe_family": "preference_integrity_probe",
                "status": "completed",
                "comparison_contract": {
                    "preference_dataset_digest": "b" * 64,
                    "labeling_contract_digest": "c" * 64,
                },
                "measurements": [
                    {"name": "preference_pairs", "value": 240.0},
                    {"name": "preference_agreement_lower_bound", "value": 0.72},
                    {"name": "ambiguous_pair_rate", "value": 0.04},
                    {"name": "preference_position_bias_rate", "value": 0.02},
                    {"name": "preference_label_conflict_rate", "value": 0.0},
                    {"name": "preference_contamination_rate", "value": 0.0},
                ],
            },
        )
    )

    assert evidence == {
        "preference_contract_verified": True,
        "preference_pairs": 240.0,
        "preference_agreement_lower_bound": 0.72,
        "ambiguous_pair_rate": 0.04,
        "preference_position_bias_rate": 0.02,
        "preference_label_conflict_rate": 0.0,
        "preference_contamination_rate": 0.0,
    }


def test_incomplete_preference_integrity_diagnostic_does_not_claim_verification() -> None:
    assert (
        method_evidence_from_diagnostic_results(
            (
                {
                    "probe_family": "preference_integrity_probe",
                    "status": "completed",
                    "comparison_contract": {
                        "preference_dataset_digest": "b" * 64,
                        "labeling_contract_digest": "c" * 64,
                    },
                    "measurements": [{"name": "preference_pairs", "value": 240.0}],
                },
            )
        )
        == {}
    )


def test_distillation_diagnostics_become_method_evidence_from_exact_contracts() -> None:
    evidence = method_evidence_from_diagnostic_results(
        (
            {
                "probe_family": "teacher_gap_probe",
                "status": "completed",
                "comparison_contract": {
                    "evaluation_suite_id": "heldout-v1",
                    "metric_direction": "maximize",
                    "teacher_model_digest": "d" * 64,
                    "student_model_digest": "e" * 64,
                    "output_validation_contract_digest": "f" * 64,
                },
                "measurements": [
                    {"name": "teacher_metric_gap", "value": 0.25},
                    {"name": "teacher_output_acceptance_rate", "value": 0.9},
                ],
            },
            {
                "probe_family": "recovery_trace_probe",
                "status": "completed",
                "comparison_contract": {
                    "recovery_dataset_digest": "a" * 64,
                    "reader_contract_digest": "b" * 64,
                    "confidence_level": 0.95,
                },
                "measurements": [
                    {"name": "recovery_traces", "value": 80.0},
                    {"name": "recovery_lift_lower_bound", "value": 0.1},
                ],
            },
        )
    )

    assert evidence == {
        "teacher_metric_gap": 0.25,
        "teacher_output_acceptance_rate": 0.9,
        "recovery_traces": 80.0,
        "recovery_lift_lower_bound": 0.1,
    }


def test_distillation_diagnostic_with_incomplete_identity_is_not_method_evidence() -> None:
    assert (
        method_evidence_from_diagnostic_results(
            (
                {
                    "probe_family": "teacher_gap_probe",
                    "status": "completed",
                    "comparison_contract": {
                        "evaluation_suite_id": "heldout-v1",
                        "metric_direction": "maximize",
                        "teacher_model_digest": "d" * 64,
                        "student_model_digest": "e" * 64,
                    },
                    "measurements": [
                        {"name": "teacher_metric_gap", "value": 0.25},
                        {"name": "teacher_output_acceptance_rate", "value": 0.9},
                    ],
                },
            )
        )
        == {}
    )


def _spec() -> AutoResearchCampaignSpec:
    return AutoResearchCampaignSpec(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        primary_metric="task_success",
        metric_direction=MetricDirection.MAXIMIZE,
        stop_rules=AutoResearchStopRules(
            max_attempts=4,
            budget_unit="gpu_hours",
            max_total_cost=8.0,
            target_metric=0.8,
            minimum_improvement=0.02,
        ),
    )


def _proposal() -> StudyProposal:
    return StudyProposal(
        proposal_id="candidate-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        hypothesis="Filtering unverifiable trajectories improves task completion.",
        evidence_references=("evaluation-baseline", "slice-tool-recovery"),
        study_family="data_quality",
        primary_variable="dataset_recipe.verifier_filter",
        controlled_variables=("training_recipe", "evaluation_recipe"),
        expected_outcome="Task success rises while verifier errors do not regress.",
        falsification_criterion="Task success fails to improve by 0.02.",
        estimated_cost=1.5,
        dataset_recipe={"verifier_filter": True},
        training_recipe={"learning_rate": 0.0001},
        evaluation_recipe={"suite_id": "heldout-v1"},
        stage_plan=StagePlan(
            items=(
                StagePlanItem(
                    stage=StageKind.FULL_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="Train the controlled candidate.",
                ),
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Compare on the fixed suite.",
                ),
            )
        ),
        planner_actor_id="codex-agent",
        rationale="Baseline failures cluster around invalid recovery steps.",
        status=ProposalStatus.ACCEPTED,
        creation_sequence=2,
        created_at=NOW,
    )


def _diagnostics() -> AutoResearchDiagnostics:
    signals = tuple(
        AutoResearchDiagnosticSignal(
            code=f"signal_{index}",
            severity="warning",
            summary=f"Diagnostic signal {index}",
            evidence_references=(f"signal-evidence-{index}",),
        )
        for index in range(6)
    )
    hypotheses = tuple(
        AutoResearchRankedHypothesis(
            hypothesis_id=f"hypothesis-{index}",
            rank=index + 1,
            action_kind="candidate",
            changed_variable=f"dataset_recipe.field_{index}",
            hypothesis=f"Hypothesis {index}",
            rationale=f"Rationale {index}",
            expected_outcome=f"Expected outcome {index}",
            falsification_criterion=f"Falsifier {index}",
            evidence_references=(f"hypothesis-evidence-{index}",),
            eligible_for_submission=True,
        )
        for index in range(4)
    )
    return AutoResearchDiagnostics(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        primary_metric="task_success",
        metric_direction="maximize",
        low_signal=False,
        signals=signals,
        ranked_hypotheses=hypotheses,
    )


def test_builds_bounded_scientific_decision_packet_from_existing_records():
    state = AutoResearchState(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        campaign_status=CampaignStatus.ACTIVE,
        next_action=AutoResearchNextAction.PROPOSE_CANDIDATE,
        ready_for_next_proposal=True,
        reason_code="ready_for_controlled_hypothesis",
        baseline_verified=True,
        best_proposal_id="candidate-1",
        best_study_id="study-1",
        best_metric=0.66,
        attempts_used=2,
        proposals_used=2,
        budget_used=2.5,
        budget_remaining=5.5,
        latest_decision=ResultDecision.KEEP,
    )
    control = AutoResearchProposalControl(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id="candidate-1",
        role=ExperimentRole.CANDIDATE,
        parent_proposal_id="baseline-1",
        changed_variables=(
            "dataset_recipe.verifier_filter",
            "training_recipe.seed",
        ),
        intervention_mode=InterventionMode.EXPLORATORY,
        hypothesis_family_id="family-verified-data",
        created_at=NOW,
    )
    outcome = AutoResearchOutcomeRecord(
        result=AutoResearchResult(
            result_id="result-1",
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            proposal_id="candidate-1",
            study_id="study-1",
            role=ExperimentRole.CANDIDATE,
            provenance=ExperimentProvenance.REAL,
            outcome=ExperimentOutcome.COMPLETED,
            metric_name="task_success",
            metric_value=0.66,
            metrics={"task_success": 0.66, "valid_tool_calls": 0.91},
            actual_cost=1.5,
            attempt_ids=("attempt-train", "attempt-eval"),
            evidence_references=("evaluation-candidate",),
            recorded_at=NOW,
        ),
        decision=AutoResearchDecision(
            proposal_id="candidate-1",
            decision=ResultDecision.KEEP,
            reason_code="candidate_improved",
            eligible_for_best=True,
            previous_best_proposal_id="baseline-1",
            previous_best_metric=0.42,
            improvement=0.24,
            result_digest="a" * 64,
            decided_at=NOW,
        ),
    )

    packet = build_decision_packet(
        objective="Improve reliable terminal-task completion.",
        spec=_spec(),
        state=state,
        diagnostics=_diagnostics(),
        latest_proposal=_proposal(),
        latest_control=control,
        latest_outcome=outcome,
        latest_data_quality={
            "generated_rows": 96,
            "accepted_rows": 64,
            "rejected_rows": 32,
            "acceptance_rate": 2 / 3,
            "deterministic_verified_rows": 72,
            "verification_failed_rows": 24,
            "verification_pass_rate": 0.75,
            "duplicate_rows_removed": 4,
            "contamination_rows_removed": 3,
            "verifier_digest": "d" * 64,
        },
        current_work={
            "phase": "experiments",
            "state": "idle",
            "stage": None,
            "summary": "Candidate evaluation completed.",
            "progress_fraction": None,
            "eta_seconds": None,
        },
        campaign_knowledge={
            "schema_version": "bashgym.autoresearch_history.v1",
            "total_experiments": 6,
            "returned_experiments": 6,
            "omitted_experiments": 0,
            "experiments": [
                {"proposal_id": f"proposal-{index}", "decision": {"decision": "keep"}}
                for index in range(6)
            ],
            "hypothesis_families": [
                {"hypothesis_family_id": f"family-{index}", "status": "replicated"}
                for index in range(7)
            ],
        },
        supported_methods=("grpo",),
        method_evidence={
            "rollout_groups": 64,
            "rollout_success_rate": 0.35,
            "zero_std_group_fraction": 0.2,
            "verifier_error_rate": 0.0,
            "reward_spec_verified": True,
            "reward_canary_cases": 4,
            "reward_canary_failure_rate": 0.0,
            "hard_constraint_violation_rate": 0.0,
        },
        method_thresholds={
            "min_rollout_groups": 32,
            "min_rollout_success_rate": 0.05,
            "max_rollout_success_rate": 0.95,
            "max_zero_std_group_fraction": 0.5,
            "max_verifier_error_rate": 0.0,
            "min_reward_canary_cases": 4,
            "max_reward_canary_failure_rate": 0.0,
            "max_hard_constraint_violation_rate": 0.0,
        },
        failure_analysis={
            "schema_version": "bashgym.research_failures.v1",
            "campaign_id": "campaign-1",
            "reference": None,
            "candidate": None,
            "comparison": [],
            "truncated": False,
        },
    )

    assert packet["schema_version"] == "bashgym.autoresearch_decision_packet.v1"
    assert packet["campaign"] == {
        "objective": "Improve reliable terminal-task completion.",
        "primary_metric": "task_success",
        "metric_direction": "maximize",
        "baseline_verified": True,
        "current_reference": {"proposal_id": "candidate-1", "metric": 0.66},
        "stop_rules": {
            "max_attempts": 4,
            "budget_unit": "gpu_hours",
            "max_total_cost": 8.0,
            "target_metric": 0.8,
            "minimum_improvement": 0.02,
            "protected_metrics": [],
            "deadline": None,
        },
    }
    assert packet["last_experiment"] == {
        "proposal_id": "candidate-1",
        "role": "candidate",
        "parent_proposal_id": "baseline-1",
        "intervention": {
            "mode": "exploratory",
            "changed_variables": [
                "dataset_recipe.verifier_filter",
                "training_recipe.seed",
            ],
            "hypothesis_family_id": "family-verified-data",
        },
        "hypothesis": "Filtering unverifiable trajectories improves task completion.",
        "changed_variable": "dataset_recipe.verifier_filter",
        "controlled_variables": ["training_recipe", "evaluation_recipe"],
        "training_seed": None,
        "expected_outcome": "Task success rises while verifier errors do not regress.",
        "falsification_criterion": "Task success fails to improve by 0.02.",
        "stages": ["full_training", "development_evaluation"],
    }
    assert packet["result"] == {
        "proposal_id": "candidate-1",
        "outcome": "completed",
        "metric_name": "task_success",
        "metric_value": 0.66,
        "metrics": {"task_success": 0.66, "valid_tool_calls": 0.91},
        "actual_cost": 1.5,
        "decision": "keep",
        "reason_code": "candidate_improved",
        "improvement": 0.24,
    }
    assert len(packet["diagnostics"]["signals"]) == 5
    assert [item["rank"] for item in packet["diagnostics"]["ranked_hypotheses"]] == [1, 2, 3]
    assert packet["resources"] == {
        "attempts_used": 2,
        "proposals_used": 2,
        "budget_unit": "gpu_hours",
        "budget_used": 2.5,
        "budget_remaining": 5.5,
    }
    assert packet["data_quality"] == {
        "generated_rows": 96,
        "accepted_rows": 64,
        "rejected_rows": 32,
        "acceptance_rate": 2 / 3,
        "deterministic_verified_rows": 72,
        "verification_failed_rows": 24,
        "verification_pass_rate": 0.75,
        "duplicate_rows_removed": 4,
        "contamination_rows_removed": 3,
        "verifier_digest": "d" * 64,
    }
    assert packet["decision_required"] == {
        "action": "propose_candidate",
        "reason_code": "ready_for_controlled_hypothesis",
        "agent_action_required": True,
    }
    assert packet["method_selection"]["eligible_methods"] == ["grpo"]
    assert packet["method_selection"]["selection_authority"] == "host_agent"
    assert packet["failure_analysis"] == {
        "schema_version": "bashgym.research_failures.v1",
        "campaign_id": "campaign-1",
        "reference": None,
        "candidate": None,
        "comparison": [],
        "truncated": False,
    }
    assert packet["outcome_assessment"] == {
        "schema_version": "bashgym.autoresearch_outcome_assessment.v1",
        "classification": "clear_improvement",
        "is_failure": False,
        "failure_kind": None,
        "decision": "keep",
        "reason_code": "primary_and_protected_gates_cleared",
        "observed_tradeoffs": [],
        "observed_improvements": [],
        "evidence_strength": "single_observation",
    }
    assert packet["campaign_knowledge"] == {
        "schema_version": "bashgym.autoresearch_history.v1",
        "total_experiments": 6,
        "returned_experiments": 5,
        "omitted_experiments": 1,
        "experiments": [
            {"proposal_id": f"proposal-{index}", "decision": {"decision": "keep"}}
            for index in range(1, 6)
        ],
        "hypothesis_families": [
            {"hypothesis_family_id": f"family-{index}", "status": "replicated"}
            for index in range(2, 7)
        ],
    }
    assert packet["evidence_references"] == [
        "evaluation-baseline",
        "slice-tool-recovery",
        "evaluation-candidate",
        "signal-evidence-0",
        "signal-evidence-1",
        "signal-evidence-2",
        "signal-evidence-3",
        "signal-evidence-4",
        "hypothesis-evidence-0",
        "hypothesis-evidence-1",
        "hypothesis-evidence-2",
    ]


def test_last_experiment_projects_controlled_variables_and_training_seed() -> None:
    proposal = _proposal().model_copy(
        update={"training_recipe": {"learning_rate": 0.0001, "seed": 23}}
    )
    state = AutoResearchState(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        campaign_status=CampaignStatus.ACTIVE,
        next_action=AutoResearchNextAction.PROPOSE_CANDIDATE,
        ready_for_next_proposal=True,
        reason_code="ready_for_controlled_hypothesis",
        baseline_verified=True,
        best_proposal_id="candidate-1",
        best_study_id="study-1",
        best_metric=0.66,
        attempts_used=2,
        proposals_used=2,
        budget_used=2.5,
        budget_remaining=5.5,
        latest_decision=ResultDecision.KEEP,
    )
    packet = build_decision_packet(
        objective="Improve task success.",
        spec=_spec(),
        state=state,
        diagnostics=_diagnostics(),
        latest_proposal=proposal,
    )

    assert packet["last_experiment"]["controlled_variables"] == [
        "training_recipe",
        "evaluation_recipe",
    ]
    assert packet["last_experiment"]["training_seed"] == 23


def test_sparse_packet_reports_only_known_state_without_inventing_findings():
    state = AutoResearchState(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        campaign_status=CampaignStatus.READY,
        next_action=AutoResearchNextAction.START_CAMPAIGN,
        ready_for_next_proposal=False,
        reason_code="campaign_requires_authorized_start",
        baseline_verified=False,
        attempts_used=0,
        proposals_used=0,
        budget_used=0,
        budget_remaining=8.0,
    )
    diagnostics = AutoResearchDiagnostics(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        primary_metric="task_success",
        metric_direction="maximize",
        low_signal=True,
    )

    packet = build_decision_packet(
        objective="Establish the fixed baseline.",
        spec=_spec(),
        state=state,
        diagnostics=diagnostics,
    )

    assert packet["last_experiment"] is None
    assert packet["result"] is None
    assert packet["current_work"] is None
    assert packet["diagnostics"] == {
        "low_signal": True,
        "signals": [],
        "checkpoint_comparisons": [],
        "error_slices": [],
        "ranked_hypotheses": [],
    }
    assert packet["decision_required"] == {
        "action": "start_campaign",
        "reason_code": "campaign_requires_authorized_start",
        "agent_action_required": True,
    }
    assert packet["method_selection"]["eligible_methods"] == []
    assert all(
        item["status"] == "unsupported_by_runner" for item in packet["method_selection"]["methods"]
    )
    assert packet["evidence_references"] == []
    assert packet["campaign_knowledge"] == {
        "schema_version": "bashgym.autoresearch_history.v1",
        "total_experiments": 0,
        "returned_experiments": 0,
        "omitted_experiments": 0,
        "experiments": [],
    }
    assert "data_quality" not in packet
    assert packet["failure_analysis"] is None
    assert "reproducibility" not in packet


def test_latest_data_quality_uses_only_the_outcomes_data_build_attempt():
    outcome = AutoResearchOutcomeRecord(
        result=AutoResearchResult(
            result_id="result-1",
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            proposal_id="candidate-1",
            study_id="study-1",
            role=ExperimentRole.CANDIDATE,
            provenance=ExperimentProvenance.REAL,
            outcome=ExperimentOutcome.COMPLETED,
            metric_name="task_success",
            metric_value=0.66,
            actual_cost=1.5,
            attempt_ids=("attempt-data", "attempt-train", "attempt-eval"),
            recorded_at=NOW,
        ),
        decision=AutoResearchDecision(
            proposal_id="candidate-1",
            decision=ResultDecision.KEEP,
            reason_code="candidate_improved",
            eligible_for_best=True,
            improvement=0.24,
            result_digest="a" * 64,
            decided_at=NOW,
        ),
    )
    expected = {
        "generated_rows": 96,
        "accepted_rows": 64,
        "rejected_rows": 32,
        "acceptance_rate": 2 / 3,
        "deterministic_verified_rows": 72,
        "verification_failed_rows": 24,
        "verification_pass_rate": 0.75,
        "duplicate_rows_removed": 4,
        "contamination_rows_removed": 3,
        "verifier_digest": "d" * 64,
    }

    selected = latest_data_quality_for_outcome(
        (
            {"metadata": {"producer_attempt_id": "unrelated", "data_quality": {}}},
            {
                "metadata": {
                    "producer_attempt_id": "attempt-data",
                    "data_quality": expected,
                }
            },
        ),
        outcome,
    )

    assert selected == expected


def test_decision_packet_infers_only_the_typed_runner_method():
    proposal = _proposal().model_copy(
        update={
            "training_recipe": {
                "schema_version": "bashgym.tmax_composite_training_recipe.v1",
                "runtime": {"executor_kind": "registered_training"},
                "algorithm": "grpo",
                "sft_enabled": False,
                "learning_rate": 0.00002,
                "max_steps": 100,
                "group_size": 8,
                "temperature": 0.8,
                "seed": 42,
            }
        }
    )
    packet = build_decision_packet(
        objective="Improve reliable terminal-task completion.",
        spec=_spec(),
        state=AutoResearchState(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            campaign_status=CampaignStatus.ACTIVE,
            next_action=AutoResearchNextAction.PROPOSE_CANDIDATE,
            ready_for_next_proposal=True,
            reason_code="ready_for_controlled_hypothesis",
            baseline_verified=True,
            attempts_used=1,
            proposals_used=1,
            budget_used=1.0,
            budget_remaining=7.0,
        ),
        diagnostics=AutoResearchDiagnostics(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            primary_metric="task_success",
            metric_direction="maximize",
            low_signal=False,
        ),
        latest_proposal=proposal,
        latest_data_quality={
            "rollout_groups": 64,
            "rollout_success_rate": 0.35,
            "zero_std_group_fraction": 0.2,
            "verifier_error_rate": 0.0,
            "reward_spec_verified": True,
            "reward_canary_cases": 4,
            "reward_canary_failure_rate": 0.0,
            "hard_constraint_violation_rate": 0.0,
        },
        method_thresholds={
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

    methods = {item["method"]: item for item in packet["method_selection"]["methods"]}
    assert methods["grpo"]["status"] == "eligible"
    assert methods["sft"]["status"] == "unsupported_by_runner"
