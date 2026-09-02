from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignSpec,
    AutoResearchDecision,
    AutoResearchHypothesisFamilyConclusion,
    AutoResearchOutcomeRecord,
    AutoResearchProposalControl,
    AutoResearchResult,
    AutoResearchStopRules,
    ExperimentOutcome,
    ExperimentProvenance,
    ExperimentRole,
    HypothesisFamilyDisposition,
    InterventionMode,
    MetricDirection,
    ProtectedMetricGate,
    ResultDecision,
)
from bashgym.campaigns.contracts import (
    ProposalStatus,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyProposal,
)
from bashgym.campaigns.method_policy import AutoResearchMethodThresholds
from bashgym.campaigns.research_history import build_autoresearch_history

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def _spec() -> AutoResearchCampaignSpec:
    return AutoResearchCampaignSpec(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        ledger_project_id="project-a",
        evaluation_suite_id="suite-heldout",
        primary_metric="task_success",
        metric_direction=MetricDirection.MAXIMIZE,
        stop_rules=AutoResearchStopRules(
            max_attempts=8,
            budget_unit="gpu_hours",
            max_total_cost=10,
            minimum_improvement=0.02,
            protected_metrics=(
                ProtectedMetricGate(
                    metric_name="invalid_tool_calls",
                    direction=MetricDirection.MINIMIZE,
                    max_regression=0.01,
                ),
                ProtectedMetricGate(
                    metric_name="recovery_rate",
                    direction=MetricDirection.MAXIMIZE,
                    max_regression=0.03,
                ),
            ),
        ),
        created_at=NOW,
    )


def _proposal(
    proposal_id: str,
    *,
    sequence: int,
    variable: str,
    hypothesis: str,
    seed: int | None = None,
    learning_rate: float = 0.0001,
) -> StudyProposal:
    training_recipe = {"learning_rate": learning_rate}
    if seed is not None:
        training_recipe["seed"] = seed
    return StudyProposal(
        proposal_id=proposal_id,
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        hypothesis=hypothesis,
        evidence_references=(f"proposal-evidence-{proposal_id}",),
        study_family="terminal_tasks",
        primary_variable=variable,
        controlled_variables=("evaluation_recipe",),
        expected_outcome="Task success improves without protected regressions.",
        falsification_criterion="The primary gate or a protected gate fails.",
        estimated_cost=1.0,
        dataset_recipe={"source": "verified"},
        training_recipe=training_recipe,
        evaluation_recipe={"suite_id": "suite-heldout"},
        stage_plan=StagePlan(
            items=(
                StagePlanItem(
                    stage=StageKind.DEVELOPMENT_EVALUATION,
                    disposition=StageDisposition.REQUIRED,
                    reason="Measure the fixed suite.",
                ),
            )
        ),
        planner_actor_id="codex-agent",
        rationale="One bounded experiment.",
        status=ProposalStatus.ACCEPTED,
        creation_sequence=sequence,
        created_at=NOW + timedelta(minutes=sequence),
    )


def _control(
    proposal_id: str,
    *,
    role: ExperimentRole,
    parent: str | None = None,
    variable: str | None = None,
    variables: tuple[str, ...] | None = None,
    mode: InterventionMode = InterventionMode.CONTROLLED,
    family: str | None = None,
) -> AutoResearchProposalControl:
    return AutoResearchProposalControl(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        proposal_id=proposal_id,
        role=role,
        parent_proposal_id=parent,
        changed_variables=(
            variables if variables is not None else ((variable,) if variable is not None else ())
        ),
        intervention_mode=mode,
        hypothesis_family_id=family,
        created_at=NOW,
    )


def _outcome(
    proposal_id: str,
    *,
    role: ExperimentRole,
    decision: ResultDecision,
    metric: float | None,
    metrics: dict[str, float],
    reference_id: str | None = None,
    reference_metric: float | None = None,
    improvement: float | None = None,
    reason: str,
    minute: int,
    outcome: ExperimentOutcome = ExperimentOutcome.COMPLETED,
    provenance: ExperimentProvenance = ExperimentProvenance.REAL,
) -> AutoResearchOutcomeRecord:
    result = AutoResearchResult(
        result_id=f"result-{proposal_id}",
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        proposal_id=proposal_id,
        study_id=f"study-{proposal_id}",
        role=role,
        provenance=provenance,
        outcome=outcome,
        metric_name="task_success",
        metric_value=metric,
        metrics=metrics,
        actual_cost=1.0,
        attempt_ids=(f"attempt-data-{proposal_id}", f"attempt-eval-{proposal_id}"),
        evidence_references=(f"evaluation-{proposal_id}",),
        recorded_at=NOW + timedelta(minutes=minute),
    )
    return AutoResearchOutcomeRecord(
        result=result,
        decision=AutoResearchDecision(
            proposal_id=proposal_id,
            decision=decision,
            reason_code=reason,
            eligible_for_best=decision in {ResultDecision.BASELINE, ResultDecision.KEEP},
            previous_best_proposal_id=reference_id,
            previous_best_metric=reference_metric,
            improvement=improvement,
            result_digest=result.result_digest,
            decided_at=result.recorded_at,
        ),
    )


def test_projects_exact_fixed_suite_performance_and_factual_learning():
    proposals = (
        _proposal(
            "baseline",
            sequence=1,
            variable="baseline",
            hypothesis="Record starting performance.",
        ),
        _proposal(
            "candidate-kept",
            sequence=2,
            variable="training_recipe.learning_rate",
            hypothesis="A lower learning rate improves task completion.",
        ),
        _proposal(
            "candidate-discarded",
            sequence=3,
            variable="dataset_recipe.filter",
            hypothesis="Stricter filtering improves task completion.",
        ),
    )
    controls = (
        _control("baseline", role=ExperimentRole.BASELINE),
        _control(
            "candidate-kept",
            role=ExperimentRole.CANDIDATE,
            parent="baseline",
            variable="training_recipe.learning_rate",
        ),
        _control(
            "candidate-discarded",
            role=ExperimentRole.CANDIDATE,
            parent="candidate-kept",
            variable="dataset_recipe.filter",
        ),
    )
    outcomes = (
        _outcome(
            "baseline",
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            metric=0.50,
            metrics={
                "task_success": 0.50,
                "invalid_tool_calls": 0.04,
                "recovery_rate": 0.70,
            },
            reason="real_baseline_verified",
            minute=1,
        ),
        _outcome(
            "candidate-kept",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.KEEP,
            metric=0.62,
            metrics={
                "task_success": 0.62,
                "invalid_tool_calls": 0.03,
                "recovery_rate": 0.72,
            },
            reference_id="baseline",
            reference_metric=0.50,
            improvement=0.12,
            reason="candidate_improved_primary_metric",
            minute=2,
        ),
        _outcome(
            "candidate-discarded",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.DISCARD,
            metric=0.66,
            metrics={
                "task_success": 0.66,
                "invalid_tool_calls": 0.08,
            },
            reference_id="candidate-kept",
            reference_metric=0.62,
            improvement=0.04,
            reason="candidate_failed_protected_metric_gate",
            minute=3,
        ),
    )
    dataset_versions = (
        {
            "dataset_version_id": "dataset-candidate-kept",
            "content_digest": "a" * 64,
            "metadata": {
                "producer_attempt_id": "attempt-data-candidate-kept",
                "data_quality": {
                    "generated_rows": 90,
                    "accepted_rows": 60,
                    "acceptance_rate": 2 / 3,
                    "verification_pass_rate": 0.8,
                },
            },
        },
    )

    history = build_autoresearch_history(
        objective="Improve reliable terminal-task completion.",
        spec=_spec(),
        proposals=proposals,
        controls=controls,
        outcomes=outcomes,
        dataset_versions=dataset_versions,
    )

    assert history["schema_version"] == "bashgym.autoresearch_history.v1"
    assert history["total_experiments"] == 3
    assert history["omitted_experiments"] == 0
    assert [item["proposal_id"] for item in history["experiments"]] == [
        "baseline",
        "candidate-kept",
        "candidate-discarded",
    ]

    baseline, kept, discarded = history["experiments"]
    assert baseline["performance"]["primary"] == {
        "metric_name": "task_success",
        "direction": "maximize",
        "reference_proposal_id": None,
        "reference_value": None,
        "candidate_value": 0.5,
        "improvement": None,
        "minimum_improvement": 0.02,
        "passed": None,
    }
    assert baseline["learning"] == {
        "status": "baseline_recorded",
        "summary": "Starting performance was recorded on the fixed evaluation suite.",
    }
    assert [
        (item["candidate_value"], item["reference_value"], item["passed"])
        for item in baseline["performance"]["protected_metrics"]
    ] == [(0.04, None, None), (0.70, None, None)]

    assert kept["proposal"] == {
        "hypothesis": "A lower learning rate improves task completion.",
        "changed_variable": "training_recipe.learning_rate",
        "expected_outcome": "Task success improves without protected regressions.",
        "falsification_criterion": "The primary gate or a protected gate fails.",
    }
    assert kept["performance"]["evaluation_suite_id"] == "suite-heldout"
    assert kept["performance"]["primary"]["reference_proposal_id"] == "baseline"
    assert kept["performance"]["primary"]["improvement"] == 0.12
    assert kept["performance"]["primary"]["passed"] is True
    assert kept["performance"]["protected_metrics"] == [
        {
            "metric_name": "invalid_tool_calls",
            "direction": "minimize",
            "reference_value": 0.04,
            "candidate_value": 0.03,
            "signed_change": 0.010000000000000002,
            "observed_regression": 0.0,
            "maximum_regression": 0.01,
            "passed": True,
        },
        {
            "metric_name": "recovery_rate",
            "direction": "maximize",
            "reference_value": 0.7,
            "candidate_value": 0.72,
            "signed_change": 0.020000000000000018,
            "observed_regression": 0.0,
            "maximum_regression": 0.03,
            "passed": True,
        },
    ]
    assert kept["data"] == {
        "dataset_version_id": "dataset-candidate-kept",
        "content_digest": "a" * 64,
        "quality": {
            "generated_rows": 90,
            "accepted_rows": 60,
            "acceptance_rate": 2 / 3,
            "verification_pass_rate": 0.8,
        },
    }
    assert kept["learning"]["status"] == "retained"
    assert kept["learning"]["summary"] == (
        "The candidate cleared the configured primary and protected metric gates and became "
        "the reference."
    )

    protected = discarded["performance"]["protected_metrics"]
    assert protected[0]["observed_regression"] == 0.05
    assert protected[0]["passed"] is False
    assert protected[1]["candidate_value"] is None
    assert protected[1]["passed"] is False
    assert discarded["decision"]["reason_code"] == "candidate_failed_protected_metric_gate"
    assert discarded["learning"] == {
        "status": "not_retained",
        "summary": (
            "The candidate exceeded at least one configured protected-metric regression limit "
            "and was not retained."
        ),
    }
    assert discarded["evidence_references"] == [
        "proposal-evidence-candidate-discarded",
        "evaluation-candidate-discarded",
    ]


def test_export_history_distinguishes_behavioral_tradeoffs_from_failed_candidates():
    baseline = _outcome(
        "baseline",
        role=ExperimentRole.BASELINE,
        decision=ResultDecision.BASELINE,
        metric=0.50,
        metrics={"task_success": 0.50},
        reason="real_baseline_verified",
        minute=1,
    )
    candidate = _outcome(
        "candidate",
        role=ExperimentRole.CANDIDATE,
        decision=ResultDecision.KEEP,
        metric=0.65,
        metrics={"task_success": 0.65},
        reference_id="baseline",
        reference_metric=0.50,
        improvement=0.15,
        reason="candidate_improved_primary_metric",
        minute=2,
    )

    def observations(count: int, *, format_count: int) -> list[dict[str, object]]:
        return [
            {
                "schema_version": "autoresearch_failure_observation.v1",
                "observation_id": "task-failure",
                "category": "task_failure",
                "summary": "The fixed evaluator marked the task response incorrect.",
                "slice_path": "behavior.task_failure",
                "checkpoint_step": None,
                "count": count,
            },
            {
                "schema_version": "autoresearch_failure_observation.v1",
                "observation_id": "format-failure",
                "category": "format_failure",
                "summary": "The fixed evaluator marked the response format invalid.",
                "slice_path": "behavior.format_failure",
                "checkpoint_step": None,
                "count": format_count,
            },
        ]

    spec = _spec().model_copy(
        update={
            "method_thresholds": AutoResearchMethodThresholds(
                min_demonstration_examples=64,
                min_target_slice_coverage=0.8,
                max_contamination_rate=0.01,
            )
        }
    )

    history = build_autoresearch_history(
        objective="Improve reliable terminal-task completion.",
        spec=spec,
        proposals=(
            _proposal("baseline", sequence=1, variable="baseline", hypothesis="Baseline."),
            _proposal(
                "candidate",
                sequence=2,
                variable="training_recipe.learning_rate",
                hypothesis="Improve the target behavior.",
            ),
        ),
        controls=(
            _control("baseline", role=ExperimentRole.BASELINE),
            _control(
                "candidate",
                role=ExperimentRole.CANDIDATE,
                parent="baseline",
                variable="training_recipe.learning_rate",
            ),
        ),
        outcomes=(baseline, candidate),
        evaluations=(
            {
                "evaluation_result_id": "evaluation-baseline",
                "slice_metrics": {
                    "autoresearch_failure_observations": observations(20, format_count=1)
                },
            },
            {
                "evaluation_result_id": "evaluation-candidate",
                "slice_metrics": {
                    "autoresearch_failure_observations": observations(7, format_count=2)
                },
            },
        ),
    )

    assert history["method_thresholds"] == {
        "schema_version": "autoresearch_method_thresholds.v1",
        "min_demonstration_examples": 64,
        "min_target_slice_coverage": 0.8,
        "max_contamination_rate": 0.01,
    }
    packet = history["experiments"][-1]["failure_analysis"]
    assert packet["comparison"] == [
        {
            "category": "format_failure",
            "reference_count": 1,
            "candidate_count": 2,
            "delta": 1,
            "status": "regressed",
        },
        {
            "category": "task_failure",
            "reference_count": 20,
            "candidate_count": 7,
            "delta": -13,
            "status": "improved",
        },
    ]
    assert "prediction" not in str(packet).lower()
    assert history["experiments"][-1]["outcome_assessment"] == {
        "schema_version": "bashgym.autoresearch_outcome_assessment.v1",
        "classification": "acceptable_tradeoff",
        "is_failure": False,
        "failure_kind": None,
        "decision": "keep",
        "reason_code": "primary_gain_with_nonblocking_tradeoff",
        "observed_tradeoffs": ["format_failure"],
        "observed_improvements": ["task_failure"],
        "evidence_strength": "single_observation",
    }


def test_uses_recorded_reference_and_keeps_incomplete_work_inconclusive():
    proposal = _proposal(
        "candidate-crashed",
        sequence=4,
        variable="training_recipe.max_steps",
        hypothesis="More steps may improve completion.",
    )
    control = _control(
        "candidate-crashed",
        role=ExperimentRole.CANDIDATE,
        parent="declared-parent",
        variable="training_recipe.max_steps",
    )
    outcome = _outcome(
        "candidate-crashed",
        role=ExperimentRole.CANDIDATE,
        decision=ResultDecision.CRASH,
        metric=None,
        metrics={},
        reference_id="actual-recorded-reference",
        reference_metric=0.61,
        reason="experiment_crashed",
        minute=4,
        outcome=ExperimentOutcome.CRASHED,
    )

    history = build_autoresearch_history(
        objective="Improve reliable terminal-task completion.",
        spec=_spec(),
        proposals=(proposal,),
        controls=(control,),
        outcomes=(outcome,),
    )

    experiment = history["experiments"][0]
    assert experiment["parent_proposal_id"] == "declared-parent"
    assert experiment["performance"]["primary"]["reference_proposal_id"] == (
        "actual-recorded-reference"
    )
    assert experiment["performance"]["primary"]["passed"] is None
    assert all(
        metric["passed"] is None for metric in experiment["performance"]["protected_metrics"]
    )
    assert experiment["learning"] == {
        "status": "inconclusive",
        "summary": "Execution did not produce a completed quality result.",
    }


def test_branch_history_separates_exact_parent_from_current_reference():
    proposals = tuple(
        _proposal(
            proposal_id,
            sequence=index,
            variable=variable,
            hypothesis=f"Hypothesis for {proposal_id}",
        )
        for index, (proposal_id, variable) in enumerate(
            (
                ("baseline", "baseline"),
                ("candidate-best", "training_recipe.learning_rate"),
                ("candidate-near-miss", "training_recipe.seed"),
                ("candidate-branch", "training_recipe.learning_rate"),
            ),
            start=1,
        )
    )
    controls = (
        _control("baseline", role=ExperimentRole.BASELINE),
        _control(
            "candidate-best",
            role=ExperimentRole.CANDIDATE,
            parent="baseline",
            variable="training_recipe.learning_rate",
        ),
        _control(
            "candidate-near-miss",
            role=ExperimentRole.CANDIDATE,
            parent="candidate-best",
            variable="training_recipe.seed",
        ),
        _control(
            "candidate-branch",
            role=ExperimentRole.CANDIDATE,
            parent="candidate-near-miss",
            variables=("training_recipe.learning_rate", "training_recipe.seed"),
            mode=InterventionMode.EXPLORATORY,
            family="family-optimizer-schedule",
        ),
    )
    outcomes = (
        _outcome(
            "baseline",
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            metric=0.50,
            metrics={"task_success": 0.50},
            reason="real_baseline_verified",
            minute=1,
        ),
        _outcome(
            "candidate-best",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.KEEP,
            metric=0.70,
            metrics={"task_success": 0.70},
            reference_id="baseline",
            reference_metric=0.50,
            improvement=0.20,
            reason="candidate_improved_primary_metric",
            minute=2,
        ),
        _outcome(
            "candidate-near-miss",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.DISCARD,
            metric=0.65,
            metrics={"task_success": 0.65},
            reference_id="candidate-best",
            reference_metric=0.70,
            improvement=-0.05,
            reason="candidate_did_not_clear_improvement_gate",
            minute=3,
        ),
        _outcome(
            "candidate-branch",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.KEEP,
            metric=0.72,
            metrics={"task_success": 0.72},
            reference_id="candidate-best",
            reference_metric=0.70,
            improvement=0.02,
            reason="candidate_improved_primary_metric",
            minute=4,
        ),
    )

    conclusion = AutoResearchHypothesisFamilyConclusion(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        hypothesis_family_id="family-optimizer-schedule",
        disposition=HypothesisFamilyDisposition.EXHAUSTED,
        summary="The optimizer-schedule bundle did not justify another arm.",
        proposal_ids=("candidate-branch",),
        result_ids=("result-candidate-branch",),
        follow_up_family_id="family-data-coverage",
        follow_up_hypothesis="Target the remaining error categories with better data coverage.",
        aggregate_version=9,
        created_at=NOW,
    )
    history = build_autoresearch_history(
        objective="Improve task success.",
        spec=_spec(),
        proposals=proposals,
        controls=controls,
        outcomes=outcomes,
        hypothesis_family_conclusions=(conclusion,),
    )

    branch = history["experiments"][-1]
    assert branch["intervention"] == {
        "mode": "exploratory",
        "changed_variables": [
            "training_recipe.learning_rate",
            "training_recipe.seed",
        ],
        "hypothesis_family_id": "family-optimizer-schedule",
    }
    assert branch["performance"]["parent"] == {
        "proposal_id": "candidate-near-miss",
        "value": 0.65,
        "improvement": pytest.approx(0.07),
    }
    assert branch["performance"]["primary"]["reference_proposal_id"] == ("candidate-best")
    assert branch["performance"]["primary"]["improvement"] == 0.02
    assert history["hypothesis_families"][0]["status"] == "single_observation"
    assert history["hypothesis_families"][0]["lifecycle"] == {
        "status": "exhausted",
        "conclusion": {
            "summary": "The optimizer-schedule bundle did not justify another arm.",
            "proposal_ids": ["candidate-branch"],
            "result_ids": ["result-candidate-branch"],
            "aggregate_version": 9,
        },
        "follow_up": {
            "hypothesis_family_id": "family-data-coverage",
            "hypothesis": "Target the remaining error categories with better data coverage.",
        },
    }


def test_history_summarizes_completed_seed_replications_without_inventing_confidence():
    family = "family-learning-rate-replication"
    proposals = tuple(
        _proposal(
            f"candidate-seed-{seed}",
            sequence=index,
            variable="training_recipe.seed",
            hypothesis="The retained recipe improves reliably across seeds.",
            seed=seed,
        )
        for index, seed in enumerate((11, 22, 33), start=1)
    )
    controls = tuple(
        _control(
            proposal.proposal_id,
            role=ExperimentRole.CANDIDATE,
            parent="baseline-1",
            variable="training_recipe.seed",
            family=family,
        )
        for proposal in proposals
    )
    outcomes = (
        _outcome(
            "candidate-seed-11",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.KEEP,
            metric=0.60,
            metrics={"task_success": 0.60},
            reference_id="baseline-1",
            reference_metric=0.50,
            improvement=0.10,
            reason="candidate_improved_primary_metric",
            minute=1,
        ),
        _outcome(
            "candidate-seed-22",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.DISCARD,
            metric=0.66,
            metrics={"task_success": 0.66},
            reference_id="baseline-1",
            reference_metric=0.50,
            improvement=0.16,
            reason="candidate_did_not_clear_improvement_gate",
            minute=2,
        ),
        _outcome(
            "candidate-seed-33",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.DISCARD,
            metric=0.63,
            metrics={"task_success": 0.63},
            reference_id="baseline-1",
            reference_metric=0.50,
            improvement=0.13,
            reason="candidate_did_not_clear_improvement_gate",
            minute=3,
        ),
    )

    history = build_autoresearch_history(
        objective="Improve task success reliably.",
        spec=_spec(),
        proposals=proposals,
        controls=controls,
        outcomes=outcomes,
    )

    assert history["hypothesis_families"] == [
        {
            "hypothesis_family_id": family,
            "status": "replicated",
            "proposal_ids": [
                "candidate-seed-11",
                "candidate-seed-22",
                "candidate-seed-33",
            ],
            "training_seeds": [11, 22, 33],
            "completed_real_results": 3,
            "decisions": {"discard": 2, "keep": 1},
            "primary_metric_summary": {
                "metric_name": "task_success",
                "count": 3,
                "mean": pytest.approx(0.63),
                "sample_standard_deviation": pytest.approx(0.03),
                "standard_error": pytest.approx(0.017320508),
                "minimum": 0.60,
                "maximum": 0.66,
                "uncertainty_method": "between_run_sample_standard_deviation",
            },
            "lifecycle": {
                "status": "open",
                "conclusion": None,
                "follow_up": None,
            },
        }
    ]
    assert {item["outcome_assessment"]["evidence_strength"] for item in history["experiments"]} == {
        "replicated"
    }
    assert {
        item["experiment_power"]["seed_uncertainty"]["status"] for item in history["experiments"]
    } == {"replicated"}


def test_history_attaches_exact_evaluation_power_without_inventing_sufficiency() -> None:
    proposal = _proposal(
        "baseline",
        sequence=1,
        variable="baseline",
        hypothesis="Record starting performance.",
    )
    outcome = _outcome(
        "baseline",
        role=ExperimentRole.BASELINE,
        decision=ResultDecision.BASELINE,
        metric=0.5,
        metrics={"task_success": 0.5},
        reason="real_baseline_verified",
        minute=1,
    )

    history = build_autoresearch_history(
        objective="Establish a fixed-suite baseline.",
        spec=_spec(),
        proposals=(proposal,),
        controls=(_control("baseline", role=ExperimentRole.BASELINE),),
        outcomes=(outcome,),
        evaluations=(
            {
                "evaluation_result_id": "evaluation-baseline",
                "slice_metrics": {"example_count": 64},
            },
        ),
    )

    power = history["experiments"][0]["experiment_power"]
    assert power["evaluation"]["sample_count"] == 64
    assert power["evaluation"]["sufficiency"]["status"] == "not_assessed"
    assert power["sequential_stopping"]["status"] == "not_predeclared"


@pytest.mark.parametrize("mismatch", ("shared_factor", "reference"))
def test_history_does_not_claim_replication_across_incomparable_experiments(
    mismatch: str,
) -> None:
    family = f"family-incomparable-{mismatch}"
    proposals = (
        _proposal(
            "candidate-seed-11",
            sequence=1,
            variable="training_recipe.seed",
            hypothesis="The retained recipe improves reliably across seeds.",
            seed=11,
        ),
        _proposal(
            "candidate-seed-22",
            sequence=2,
            variable="training_recipe.seed",
            hypothesis="The retained recipe improves reliably across seeds.",
            seed=22,
            learning_rate=0.0002 if mismatch == "shared_factor" else 0.0001,
        ),
    )
    controls = tuple(
        _control(
            proposal.proposal_id,
            role=ExperimentRole.CANDIDATE,
            parent="baseline-1",
            variable="training_recipe.seed",
            family=family,
        )
        for proposal in proposals
    )
    second_reference = "baseline-2" if mismatch == "reference" else "baseline-1"
    outcomes = (
        _outcome(
            "candidate-seed-11",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.KEEP,
            metric=0.60,
            metrics={"task_success": 0.60},
            reference_id="baseline-1",
            reference_metric=0.50,
            improvement=0.10,
            reason="candidate_improved_primary_metric",
            minute=1,
        ),
        _outcome(
            "candidate-seed-22",
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.DISCARD,
            metric=0.66,
            metrics={"task_success": 0.66},
            reference_id=second_reference,
            reference_metric=0.50,
            improvement=0.16,
            reason="candidate_did_not_clear_improvement_gate",
            minute=2,
        ),
    )

    history = build_autoresearch_history(
        objective="Improve task success reliably.",
        spec=_spec(),
        proposals=proposals,
        controls=controls,
        outcomes=outcomes,
    )

    assert history["hypothesis_families"][0]["status"] == "single_observation"
    assert {item["outcome_assessment"]["evidence_strength"] for item in history["experiments"]} == {
        "single_observation"
    }


def test_history_is_deterministically_bounded_without_inventing_sparse_findings():
    empty = build_autoresearch_history(
        objective="Establish a baseline.",
        spec=_spec(),
        proposals=(),
        controls=(),
        outcomes=(),
        limit=2,
    )
    assert empty["experiments"] == []
    assert empty["total_experiments"] == 0
    assert empty["omitted_experiments"] == 0

    pending = _proposal(
        "candidate-pending",
        sequence=1,
        variable="training_recipe.seed",
        hypothesis="The candidate needs a second seed.",
        seed=44,
    )
    active = build_autoresearch_history(
        objective="Test a hypothesis across seeds.",
        spec=_spec(),
        proposals=(pending,),
        controls=(
            _control(
                pending.proposal_id,
                role=ExperimentRole.CANDIDATE,
                parent="baseline-1",
                variable="training_recipe.seed",
                family="family-pending",
            ),
        ),
        outcomes=(),
        limit=2,
    )
    assert active["hypothesis_families"][0]["status"] == "active"

    proposals = tuple(
        _proposal(
            f"baseline-{index}",
            sequence=index,
            variable="baseline",
            hypothesis=f"Baseline {index}",
        )
        for index in (1, 2, 3)
    )
    controls = tuple(
        _control(f"baseline-{index}", role=ExperimentRole.BASELINE) for index in (1, 2, 3)
    )
    outcomes = tuple(
        _outcome(
            f"baseline-{index}",
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            metric=0.4 + index / 100,
            metrics={"task_success": 0.4 + index / 100},
            reason="real_baseline_verified",
            minute=index,
        )
        for index in (3, 1, 2)
    )

    bounded = build_autoresearch_history(
        objective="Establish a baseline.",
        spec=_spec(),
        proposals=proposals,
        controls=controls,
        outcomes=outcomes,
        limit=2,
    )

    assert bounded["total_experiments"] == 3
    assert bounded["returned_experiments"] == 2
    assert bounded["omitted_experiments"] == 1
    assert [item["proposal_id"] for item in bounded["experiments"]] == [
        "baseline-2",
        "baseline-3",
    ]
