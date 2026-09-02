"""Baseline-first AutoResearch campaign control-loop tests."""

import json
from datetime import datetime

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.autoresearch import (
    AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID,
    AutoResearchCampaignCore,
    AutoResearchCampaignSpec,
    AutoResearchDecision,
    AutoResearchDiagnosticResult,
    AutoResearchHypothesisFamilyConclusion,
    AutoResearchInvariantError,
    AutoResearchLedgerCommitContext,
    AutoResearchNextAction,
    AutoResearchOutcomeRecord,
    AutoResearchProposalControl,
    AutoResearchRepository,
    AutoResearchResult,
    AutoResearchStopRules,
    AutoResearchTemplateDefinition,
    ExperimentOutcome,
    ExperimentProvenance,
    ExperimentRole,
    HypothesisFamilyDisposition,
    MetricDirection,
    ProtectedMetricGate,
    ResultDecision,
    _validate_controlled_candidate_change,
    build_autoresearch_template_registry,
    builtin_autoresearch_template_definitions,
    builtin_autoresearch_template_registry,
)
from bashgym.campaigns.contracts import (
    CampaignStatus,
    CampaignTrigger,
    CodeMutationKind,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyStatus,
    canonical_hash,
)
from bashgym.campaigns.method_policy import AutoResearchMethodThresholds
from bashgym.campaigns.service import CampaignControllerService, CampaignService
from bashgym.ledger.contracts import (
    ArtifactSpec,
    AttemptSpec,
    ContextStatus,
    DatasetSpec,
    DatasetVersionSpec,
    EnvironmentSpec,
    EvaluationResultSpec,
    EvaluationSuiteSpec,
    ExperimentSpec,
    ModelSpec,
    ModelVersionSpec,
    ProjectSpec,
    RunSpec,
    RunStatus,
)
from tests.campaigns.test_persistence import campaign, create, manifest
from tests.campaigns.test_proposals import diagnostic_proposal, principal, proposal

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def make_spec(
    *,
    max_attempts: int = 3,
    target: float | None = 0.95,
    evaluation_binding: bool = False,
    protected_metrics: tuple[ProtectedMetricGate, ...] = (),
):
    return AutoResearchCampaignSpec(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        primary_metric="mrr_at_10",
        metric_direction=MetricDirection.MAXIMIZE,
        stop_rules=AutoResearchStopRules(
            max_attempts=max_attempts,
            budget_unit="gpu_hours",
            max_total_cost=3.0,
            target_metric=target,
            minimum_improvement=0.01,
            protected_metrics=protected_metrics,
            deadline=datetime(2099, 7, 14, 12, 0, tzinfo=UTC),
        ),
        ledger_project_id="project-a" if evaluation_binding else None,
        evaluation_suite_id="suite-a" if evaluation_binding else None,
        created_at=NOW,
    )


def test_template_method_thresholds_materialize_into_campaign_spec():
    definition = builtin_autoresearch_template_definitions()[0]
    assert definition.policy is not None
    thresholds = AutoResearchMethodThresholds(
        min_rollout_groups=32,
        min_rollout_success_rate=0.05,
        max_rollout_success_rate=0.95,
        max_zero_std_group_fraction=0.5,
        max_verifier_error_rate=0.0,
    )
    definition = definition.model_copy(
        update={"policy": definition.policy.model_copy(update={"method_thresholds": thresholds})}
    )

    spec = definition.materialize_spec(
        "workspace-a",
        "campaign-a",
        stop_rules=definition.policy.stop_rules,
    )

    assert spec is not None
    assert spec.method_thresholds == thresholds
    legacy = make_spec().model_dump(mode="json")
    legacy.pop("method_thresholds", None)
    assert AutoResearchCampaignSpec.model_validate(legacy).method_thresholds == (
        AutoResearchMethodThresholds()
    )


def fresh_core(
    tmp_path,
    *,
    max_attempts=3,
    target=0.95,
    evaluation_binding=False,
    protected_metrics: tuple[ProtectedMetricGate, ...] = (),
):
    path = tmp_path / "campaigns.sqlite3"
    repository = AutoResearchRepository(path)
    repository.initialize()
    create(repository)
    core = AutoResearchCampaignCore(repository)
    core.register(
        make_spec(
            max_attempts=max_attempts,
            target=target,
            evaluation_binding=evaluation_binding,
            protected_metrics=protected_metrics,
        )
    )
    return path, repository, core


def activate(core):
    prepared = core.prepare(
        "workspace-a",
        "campaign-1",
        controller_id="autoresearch-controller",
        correlation_id="autoresearch-prepare",
        idempotency_prefix="autoresearch-prepare",
    )
    actor = principal(core.repository)
    return (
        CampaignService(core.repository)
        .transition(
            "workspace-a",
            "campaign-1",
            CampaignTrigger.START,
            expected_version=prepared.version,
            principal=actor,
            correlation_id="autoresearch-start",
            idempotency_key="autoresearch-start",
        )
        .campaign
    )


def select_and_finish(repository, proposal_id: str, *, failed: bool = False):
    """Create terminal executor evidence without coupling this policy test to a worker."""

    campaign_value = repository.get_campaign("workspace-a", "campaign-1")
    selected = CampaignControllerService(
        repository, controller_id="autoresearch-controller"
    ).select_next_proposal(
        "workspace-a",
        "campaign-1",
        expected_version=campaign_value.version,
        correlation_id=f"select-{proposal_id}",
        idempotency_key=f"select-{proposal_id}",
    )
    assert selected is not None
    assert selected.record.proposal.proposal_id == proposal_id
    action_id = f"action-{proposal_id}"
    attempt_id = f"attempt-{proposal_id}"
    ended_status = StudyStatus.EXECUTION_FAILED if failed else StudyStatus.COMPLETED
    attempt_status = "failed" if failed else "completed"
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_actions(
                workspace_id, campaign_id, study_id, action_id, stage_index,
                stage_kind, input_digest, status, version, created_at, updated_at,
                candidate_digest, manifest_revision, reservation_json
            ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, 1, ?, ?, ?, 1, '{}')
            """,
            (
                "workspace-a",
                "campaign-1",
                selected.study.study_id,
                action_id,
                StageKind.DEVELOPMENT_EVALUATION.value,
                "b" * 64,
                attempt_status,
                NOW.isoformat(),
                NOW.isoformat(),
                selected.study.candidate_digest,
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_attempts(
                workspace_id, action_id, attempt_id, attempt_number,
                claim_generation, status, executor_json, result_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, 1, 1, ?, ?, '{}', ?, ?)
            """,
            (
                "workspace-a",
                action_id,
                attempt_id,
                attempt_status,
                json.dumps({"executor_kind": "local_process"}),
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        connection.execute(
            """
            UPDATE campaign_studies SET status = ?, current_stage_index = 1,
                version = version + 1, updated_at = ?
            WHERE workspace_id = ? AND study_id = ?
            """,
            (
                ended_status.value,
                NOW.isoformat(),
                "workspace-a",
                selected.study.study_id,
            ),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = NULL, active_action_id = NULL,
                version = version + 1, updated_at = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            (NOW.isoformat(), "workspace-a", "campaign-1"),
        )
    return selected.study.study_id, attempt_id


def result(proposal_id, study_id, attempt_id, metric, *, role, provenance="real"):
    return AutoResearchResult(
        result_id=f"result-{proposal_id}",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id=proposal_id,
        study_id=study_id,
        role=role,
        provenance=ExperimentProvenance(provenance),
        outcome=ExperimentOutcome.COMPLETED,
        metric_name="mrr_at_10",
        metric_value=metric,
        metrics={"mrr_at_10": metric},
        actual_cost=0.5,
        attempt_ids=(attempt_id,),
        evidence_references=(f"eval-{proposal_id}",),
        recorded_at=NOW,
    )


def _recipe_proposal(
    proposal_id: str,
    *,
    learning_rate: float,
    seed: int,
    extra: dict[str, object] | None = None,
):
    training_recipe = {
        "schema_version": "recipe.v1",
        "learning_rate": learning_rate,
        "seed": seed,
        **(extra or {}),
    }
    return proposal(proposal_id, estimated_cost=0.1).model_copy(
        update={"training_recipe": training_recipe}
    )


def _insert_authoritative_outcome(
    repository: AutoResearchRepository,
    outcome: AutoResearchOutcomeRecord,
) -> None:
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO autoresearch_results(
                workspace_id, campaign_id, result_id, proposal_id, result_json,
                result_digest, decision_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.result.workspace_id,
                outcome.result.campaign_id,
                outcome.result.result_id,
                outcome.result.proposal_id,
                outcome.result.model_dump_json(),
                outcome.result.result_digest,
                outcome.decision.model_dump_json(),
                outcome.result.recorded_at.isoformat(),
            ),
        )


def _authoritative_outcome(
    proposal_id: str,
    study_id: str,
    attempt_id: str,
    metric: float,
    *,
    role: ExperimentRole,
    decision: ResultDecision,
    eligible_for_best: bool,
    previous_best_proposal_id: str | None = None,
    previous_best_metric: float | None = None,
) -> AutoResearchOutcomeRecord:
    value = result(proposal_id, study_id, attempt_id, metric, role=role)
    improvement = metric - previous_best_metric if previous_best_metric is not None else None
    return AutoResearchOutcomeRecord(
        result=value,
        decision=AutoResearchDecision(
            proposal_id=proposal_id,
            decision=decision,
            reason_code=(
                "real_baseline_verified"
                if decision == ResultDecision.BASELINE
                else "candidate_improved_primary_metric"
            ),
            eligible_for_best=eligible_for_best,
            previous_best_proposal_id=previous_best_proposal_id,
            previous_best_metric=previous_best_metric,
            improvement=improvement,
            result_digest=value.result_digest,
            decided_at=NOW,
        ),
    )


def test_exploratory_intervention_accepts_exact_declared_change_bundle():
    parent = _recipe_proposal("parent", learning_rate=0.001, seed=17)
    candidate = _recipe_proposal("candidate", learning_rate=0.002, seed=23)
    control = AutoResearchProposalControl(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id="candidate",
        role=ExperimentRole.CANDIDATE,
        parent_proposal_id="parent",
        intervention_mode="exploratory",
        changed_variables=("training_recipe.learning_rate", "training_recipe.seed"),
        hypothesis_family_id="family-optimizer-schedule",
    )

    _validate_controlled_candidate_change(
        parent,
        candidate,
        declared_variables=control.changed_variables,
        intervention_mode=control.intervention_mode,
        code_mutation_kind=None,
    )

    assert control.intervention_mode.value == "exploratory"
    assert control.hypothesis_family_id == "family-optimizer-schedule"


@pytest.mark.parametrize(
    ("changed_variables", "candidate", "reason"),
    (
        (
            ("training_recipe.learning_rate", "training_recipe.seed"),
            _recipe_proposal(
                "candidate-extra",
                learning_rate=0.002,
                seed=23,
                extra={"temperature": 0.7},
            ),
            "autoresearch_candidate_changed_undeclared_variable",
        ),
        (
            ("training_recipe.learning_rate", "training_recipe.temperature"),
            _recipe_proposal("candidate-unchanged", learning_rate=0.002, seed=17),
            "autoresearch_candidate_declared_variable_unchanged",
        ),
    ),
)
def test_exploratory_intervention_rejects_inaccurate_change_declarations(
    changed_variables,
    candidate,
    reason,
):
    parent = _recipe_proposal("parent", learning_rate=0.001, seed=17)
    with pytest.raises(AutoResearchInvariantError, match=reason):
        _validate_controlled_candidate_change(
            parent,
            candidate,
            declared_variables=changed_variables,
            intervention_mode="exploratory",
            code_mutation_kind=None,
        )


def test_exploratory_intervention_rejects_code_mutation_bundles():
    parent = _recipe_proposal("parent", learning_rate=0.001, seed=17)
    candidate = parent.model_copy(update={"proposal_id": "candidate-code"})
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_exploratory_code_bundle_not_supported",
    ):
        _validate_controlled_candidate_change(
            parent,
            candidate,
            declared_variables=("trainer.optimizer", "trainer.scheduler"),
            intervention_mode="exploratory",
            code_mutation_kind=CodeMutationKind.TRAINER,
        )


def test_legacy_control_digest_is_stable_when_new_fields_use_defaults():
    control = AutoResearchProposalControl(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id="candidate",
        role=ExperimentRole.CANDIDATE,
        parent_proposal_id="parent",
        changed_variables=("learning_rate",),
        created_at=NOW,
    )
    expected = canonical_hash(
        {
            "schema_version": "autoresearch_proposal_control.v1",
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "proposal_id": "candidate",
            "role": "candidate",
            "parent_proposal_id": "parent",
            "changed_variables": ["learning_rate"],
        }
    )

    assert control.control_digest == expected


def _completed_hypothesis_family(tmp_path):
    _path, repository, core = fresh_core(tmp_path, max_attempts=4, target=None)
    activate(core)
    baseline = _recipe_proposal("baseline-family", learning_rate=0.001, seed=17)
    core.submit_baseline(
        baseline,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=principal(repository),
        correlation_id="submit-baseline-family",
        idempotency_key="submit-baseline-family",
    )
    baseline_study, baseline_attempt = select_and_finish(repository, "baseline-family")
    _insert_authoritative_outcome(
        repository,
        _authoritative_outcome(
            "baseline-family",
            baseline_study,
            baseline_attempt,
            0.5,
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            eligible_for_best=True,
        ),
    )
    candidate = _recipe_proposal("candidate-family", learning_rate=0.002, seed=17).model_copy(
        update={"prerequisite_study_ids": (baseline_study,)}
    )
    core.submit_candidate(
        candidate,
        parent_proposal_id="baseline-family",
        changed_variables=("learning_rate",),
        hypothesis_family_id="family-learning-rate",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=principal(repository),
        correlation_id="submit-candidate-family",
        idempotency_key="submit-candidate-family",
    )
    candidate_study, candidate_attempt = select_and_finish(repository, "candidate-family")
    _insert_authoritative_outcome(
        repository,
        _authoritative_outcome(
            "candidate-family",
            candidate_study,
            candidate_attempt,
            0.48,
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.DISCARD,
            eligible_for_best=False,
            previous_best_proposal_id="baseline-family",
            previous_best_metric=0.5,
        ),
    )
    return repository, core, baseline_study


def test_agent_concludes_completed_family_and_records_open_follow_up(tmp_path) -> None:
    repository, core, _baseline_study = _completed_hypothesis_family(tmp_path)
    campaign = repository.get_campaign("workspace-a", "campaign-1")

    conclusion = core.conclude_hypothesis_family(
        "workspace-a",
        "campaign-1",
        "family-learning-rate",
        disposition=HypothesisFamilyDisposition.EXHAUSTED,
        summary="The tested learning-rate direction did not improve the fixed suite.",
        follow_up_family_id="family-data-coverage",
        follow_up_hypothesis="Targeted data coverage may address the remaining failures.",
        expected_version=campaign.version,
        principal=principal(repository),
        correlation_id="conclude-learning-rate",
        idempotency_key="conclude-learning-rate",
    )

    assert conclusion == AutoResearchHypothesisFamilyConclusion(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        hypothesis_family_id="family-learning-rate",
        disposition=HypothesisFamilyDisposition.EXHAUSTED,
        summary="The tested learning-rate direction did not improve the fixed suite.",
        proposal_ids=("candidate-family",),
        result_ids=("result-candidate-family",),
        follow_up_family_id="family-data-coverage",
        follow_up_hypothesis="Targeted data coverage may address the remaining failures.",
        aggregate_version=campaign.version + 1,
        created_at=conclusion.created_at,
    )
    assert repository.list_hypothesis_family_conclusions("workspace-a", "campaign-1") == (
        conclusion,
    )
    event_types = [
        event.event_type for _cursor, event in repository.list_events("workspace-a", "campaign-1")
    ]
    assert event_types[-1] == "campaign:autoresearch-family-concluded"

    replay = core.conclude_hypothesis_family(
        "workspace-a",
        "campaign-1",
        "family-learning-rate",
        disposition=HypothesisFamilyDisposition.EXHAUSTED,
        summary="The tested learning-rate direction did not improve the fixed suite.",
        follow_up_family_id="family-data-coverage",
        follow_up_hypothesis="Targeted data coverage may address the remaining failures.",
        expected_version=campaign.version,
        principal=principal(repository),
        correlation_id="conclude-learning-rate-replay",
        idempotency_key="conclude-learning-rate-replay",
    )
    assert replay.replayed is True


def test_family_conclusion_requires_complete_results_and_closes_only_that_family(tmp_path) -> None:
    repository, core, baseline_study = _completed_hypothesis_family(tmp_path)
    campaign = repository.get_campaign("workspace-a", "campaign-1")
    conclusion = core.conclude_hypothesis_family(
        "workspace-a",
        "campaign-1",
        "family-learning-rate",
        disposition=HypothesisFamilyDisposition.INCONCLUSIVE,
        summary="The completed arm does not justify another learning-rate run.",
        expected_version=campaign.version,
        principal=principal(repository),
        correlation_id="conclude-family",
        idempotency_key="conclude-family",
    )
    closed_candidate = _recipe_proposal(
        "candidate-closed-family", learning_rate=0.003, seed=17
    ).model_copy(update={"prerequisite_study_ids": (baseline_study,)})
    with pytest.raises(
        AutoResearchInvariantError, match="autoresearch_hypothesis_family_concluded"
    ):
        core.submit_candidate(
            closed_candidate,
            parent_proposal_id="baseline-family",
            changed_variables=("learning_rate",),
            hypothesis_family_id="family-learning-rate",
            expected_version=campaign.version,
            principal=principal(repository),
            correlation_id="reuse-closed-family",
            idempotency_key="reuse-closed-family",
        )

    follow_up = closed_candidate.model_copy(update={"proposal_id": "candidate-follow-up"})
    mutation = core.submit_candidate(
        follow_up,
        parent_proposal_id="baseline-family",
        changed_variables=("learning_rate",),
        hypothesis_family_id="family-new-agent-idea",
        expected_version=conclusion.aggregate_version,
        principal=principal(repository),
        correlation_id="submit-open-family",
        idempotency_key="submit-open-family",
    )
    assert mutation.record.proposal.proposal_id == "candidate-follow-up"


def test_family_conclusion_rejects_a_family_with_pending_results(tmp_path) -> None:
    repository, core, baseline_study = _completed_hypothesis_family(tmp_path)
    pending = _recipe_proposal("candidate-pending-family", learning_rate=0.004, seed=17).model_copy(
        update={"prerequisite_study_ids": (baseline_study,)}
    )
    campaign = repository.get_campaign("workspace-a", "campaign-1")
    core.submit_candidate(
        pending,
        parent_proposal_id="baseline-family",
        changed_variables=("learning_rate",),
        hypothesis_family_id="family-pending",
        expected_version=campaign.version,
        principal=principal(repository),
        correlation_id="submit-pending-family",
        idempotency_key="submit-pending-family",
    )

    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_hypothesis_family_has_pending_results",
    ):
        core.conclude_hypothesis_family(
            "workspace-a",
            "campaign-1",
            "family-pending",
            disposition=HypothesisFamilyDisposition.EXHAUSTED,
            summary="This must wait for the pending result.",
            expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
            principal=principal(repository),
            correlation_id="conclude-pending-family",
            idempotency_key="conclude-pending-family",
        )


def test_diagnostic_result_satisfies_pending_without_consuming_candidate_attempt(tmp_path):
    _path, repository, core = fresh_core(tmp_path, max_attempts=2, target=None)
    activate(core)
    baseline_submission = proposal("baseline", estimated_cost=0.1)
    core.submit_baseline(
        baseline_submission,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=principal(repository),
        correlation_id="submit-baseline-diagnostic-test",
        idempotency_key="submit-baseline-diagnostic-test",
    )
    baseline_study, baseline_attempt = select_and_finish(repository, "baseline")
    _insert_authoritative_outcome(
        repository,
        _authoritative_outcome(
            "baseline",
            baseline_study,
            baseline_attempt,
            0.5,
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            eligible_for_best=True,
        ),
    )
    diagnostic_control = AutoResearchProposalControl(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        proposal_id="diagnostic-one",
        role=ExperimentRole.DIAGNOSTIC,
        parent_proposal_id="baseline",
        created_at=NOW,
    )
    diagnostic_submission = proposal("diagnostic-one", estimated_cost=0.05).model_copy(
        update={"prerequisite_study_ids": (baseline_study,)}
    )
    CampaignService(repository).submit_proposal(
        diagnostic_submission,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=principal(repository),
        correlation_id="submit-diagnostic-one",
        idempotency_key="submit-diagnostic-one",
    )
    repository.register_autoresearch_proposal(diagnostic_control)

    waiting = core.state("workspace-a", "campaign-1", now=NOW)
    assert waiting.next_action == AutoResearchNextAction.WAIT_FOR_RESULT
    assert waiting.attempts_used == 1

    stored = repository.record_autoresearch_diagnostic_result(
        AutoResearchDiagnosticResult(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            proposal_id="diagnostic-one",
            study_id="study-diagnostic-one",
            attempt_id="attempt-diagnostic-one",
            status="completed",
            projection={
                "schema_version": "bashgym.research_diagnostic_result.v1",
                "probe_family": "loss_landscape",
            },
            actual_cost=0.05,
            recorded_at=NOW,
        )
    )

    state = core.state("workspace-a", "campaign-1", now=NOW)
    assert stored.replayed is False
    assert state.next_action == AutoResearchNextAction.PROPOSE_CANDIDATE
    assert state.reason_code == "diagnostic_evidence_ready"
    assert state.attempts_used == 1
    assert state.proposals_used == 2
    assert state.budget_used == pytest.approx(0.55)


def test_agent_can_submit_open_diagnostic_against_verified_reference(tmp_path):
    _path, repository, core = fresh_core(tmp_path, max_attempts=3, target=None)
    activate(core)
    actor = principal(repository)
    core.submit_baseline(
        proposal("baseline", estimated_cost=0.1),
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-baseline-before-diagnostic",
        idempotency_key="submit-baseline-before-diagnostic",
    )
    baseline_study, baseline_attempt = select_and_finish(repository, "baseline")
    _insert_authoritative_outcome(
        repository,
        _authoritative_outcome(
            "baseline",
            baseline_study,
            baseline_attempt,
            0.5,
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            eligible_for_best=True,
        ),
    )
    submission = diagnostic_proposal("diagnostic-open-probe").model_copy(
        update={"prerequisite_study_ids": (baseline_study,)}
    )

    mutation = core.submit_diagnostic(
        submission,
        parent_proposal_id="baseline",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-diagnostic-open-probe",
        idempotency_key="submit-diagnostic-open-probe",
    )

    assert mutation.record.validation.valid is True
    control = repository.get_autoresearch_proposal(
        "workspace-a", "campaign-1", "diagnostic-open-probe"
    )
    assert control.role == ExperimentRole.DIAGNOSTIC
    assert control.changed_variables == ()
    assert control.parent_proposal_id == "baseline"


def test_candidate_can_branch_from_verified_ancestor_instead_of_current_reference(tmp_path):
    _path, repository, core = fresh_core(tmp_path, max_attempts=4, target=None)
    activate(core)
    actor = principal(repository)

    baseline = _recipe_proposal("baseline", learning_rate=0.001, seed=17)
    core.submit_baseline(
        baseline,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-baseline",
        idempotency_key="submit-baseline",
    )
    baseline_study, baseline_attempt = select_and_finish(repository, "baseline")
    _insert_authoritative_outcome(
        repository,
        _authoritative_outcome(
            "baseline",
            baseline_study,
            baseline_attempt,
            0.50,
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            eligible_for_best=True,
        ),
    )

    incumbent = _recipe_proposal("candidate-best", learning_rate=0.002, seed=17).model_copy(
        update={
            "primary_variable": "training_recipe.learning_rate",
            "prerequisite_study_ids": (baseline_study,),
        }
    )
    core.submit_controlled_candidate(
        incumbent,
        parent_proposal_id="baseline",
        changed_variable="training_recipe.learning_rate",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-incumbent",
        idempotency_key="submit-incumbent",
    )
    incumbent_study, incumbent_attempt = select_and_finish(repository, "candidate-best")
    _insert_authoritative_outcome(
        repository,
        _authoritative_outcome(
            "candidate-best",
            incumbent_study,
            incumbent_attempt,
            0.70,
            role=ExperimentRole.CANDIDATE,
            decision=ResultDecision.KEEP,
            eligible_for_best=True,
            previous_best_proposal_id="baseline",
            previous_best_metric=0.50,
        ),
    )
    assert core.state("workspace-a", "campaign-1", now=NOW).best_proposal_id == ("candidate-best")

    branch = _recipe_proposal("candidate-branch", learning_rate=0.001, seed=23).model_copy(
        update={
            "primary_variable": "training_recipe.seed",
            "prerequisite_study_ids": (baseline_study,),
        }
    )
    mutation = core.submit_controlled_candidate(
        branch,
        parent_proposal_id="baseline",
        changed_variable="training_recipe.seed",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-branch",
        idempotency_key="submit-branch",
    )

    assert mutation.record.proposal.proposal_id == "candidate-branch"
    control = repository.get_autoresearch_proposal("workspace-a", "campaign-1", "candidate-branch")
    assert control.parent_proposal_id == "baseline"


def test_protected_metric_gate_blocks_a_primary_gain_with_a_regression() -> None:
    incumbent = result(
        "baseline",
        "study-baseline",
        "attempt-baseline",
        0.50,
        role=ExperimentRole.BASELINE,
    ).model_copy(update={"metrics": {"mrr_at_10": 0.50, "valid_tool_calls": 0.98}})
    candidate = result(
        "candidate",
        "study-candidate",
        "attempt-candidate",
        0.55,
        role=ExperimentRole.CANDIDATE,
    ).model_copy(update={"metrics": {"mrr_at_10": 0.55, "valid_tool_calls": 0.70}})
    gates = (
        ProtectedMetricGate(
            metric_name="valid_tool_calls",
            direction=MetricDirection.MAXIMIZE,
            max_regression=0.02,
        ),
    )

    assert AutoResearchRepository._protected_metric_failure(gates, incumbent, candidate) == (
        "valid_tool_calls"
    )
    acceptable = candidate.model_copy(
        update={"metrics": {"mrr_at_10": 0.55, "valid_tool_calls": 0.97}}
    )
    assert AutoResearchRepository._protected_metric_failure(gates, incumbent, acceptable) is None


def test_protected_metric_margins_report_headroom_and_breach() -> None:
    incumbent = result(
        "baseline", "study-baseline", "attempt-baseline", 0.50, role=ExperimentRole.BASELINE
    ).model_copy(update={"metrics": {"mrr_at_10": 0.50, "valid_tool_calls": 0.98}})
    candidate = result(
        "candidate", "study-candidate", "attempt-candidate", 0.55, role=ExperimentRole.CANDIDATE
    ).model_copy(update={"metrics": {"mrr_at_10": 0.55, "valid_tool_calls": 0.97}})
    gates = (
        ProtectedMetricGate(
            metric_name="valid_tool_calls",
            direction=MetricDirection.MAXIMIZE,
            max_regression=0.02,
        ),
    )

    margins = AutoResearchRepository._protected_metric_margins(gates, incumbent, candidate)

    assert margins == {"valid_tool_calls": pytest.approx(0.01)}
    breached = candidate.model_copy(
        update={"metrics": {"mrr_at_10": 0.55, "valid_tool_calls": 0.90}}
    )
    assert AutoResearchRepository._protected_metric_margins(gates, incumbent, breached) == {
        "valid_tool_calls": pytest.approx(-0.06)
    }
    assert AutoResearchRepository._protected_metric_failure(gates, incumbent, breached) == (
        "valid_tool_calls"
    )


def test_legacy_decision_without_margins_replays_without_conflict(tmp_path):
    """A decision_json row written before protected_metric_margins existed must still replay."""

    _path, repository, core = fresh_core(
        tmp_path,
        max_attempts=4,
        target=None,
        evaluation_binding=True,
        protected_metrics=(
            ProtectedMetricGate(
                metric_name="valid_tool_calls",
                direction=MetricDirection.MAXIMIZE,
                max_regression=0.02,
            ),
        ),
    )
    activate(core)
    actor = principal(repository)

    baseline = _recipe_proposal("baseline", learning_rate=0.001, seed=17)
    core.submit_baseline(
        baseline,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-baseline-legacy-margins",
        idempotency_key="submit-baseline-legacy-margins",
    )
    baseline_study, baseline_attempt = select_and_finish(repository, "baseline")
    baseline_outcome = _authoritative_outcome(
        "baseline",
        baseline_study,
        baseline_attempt,
        0.50,
        role=ExperimentRole.BASELINE,
        decision=ResultDecision.BASELINE,
        eligible_for_best=True,
    )
    baseline_outcome = baseline_outcome.model_copy(
        update={
            "result": baseline_outcome.result.model_copy(
                update={"metrics": {"mrr_at_10": 0.50, "valid_tool_calls": 0.98}}
            )
        }
    )
    _insert_authoritative_outcome(repository, baseline_outcome)

    candidate_submission = _recipe_proposal(
        "candidate-legacy-margins", learning_rate=0.002, seed=17
    ).model_copy(
        update={
            "primary_variable": "training_recipe.learning_rate",
            "prerequisite_study_ids": (baseline_study,),
        }
    )
    core.submit_controlled_candidate(
        candidate_submission,
        parent_proposal_id="baseline",
        changed_variable="training_recipe.learning_rate",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-candidate-legacy-margins",
        idempotency_key="submit-candidate-legacy-margins",
    )

    candidate_result = result(
        "candidate-legacy-margins",
        "study-candidate-legacy-margins",
        "attempt-candidate-legacy-margins",
        0.55,
        role=ExperimentRole.CANDIDATE,
    ).model_copy(update={"metrics": {"mrr_at_10": 0.55, "valid_tool_calls": 0.97}})

    ledger = core.ledger
    ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            display_name="AutoResearch",
            owner_actor_id="codex-agent",
        )
    )
    ledger.register_experiment(
        ExperimentSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            experiment_id="experiment-candidate-legacy-margins",
            name="Candidate",
            objective="Evaluate the candidate recipe.",
        )
    )
    ledger.register_run(
        RunSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            experiment_id="experiment-candidate-legacy-margins",
            run_id="run-candidate-legacy-margins",
            source_system="bashgym",
            source_run_id="run-candidate-legacy-margins",
            campaign_id="campaign-1",
            run_kind="training",
            task_type="retrieval",
            training_method="lora",
            status=RunStatus.COMPLETED,
            context_status=ContextStatus.VERIFIED,
            recipe_digest="d" * 64,
            correlation_id="run-candidate-legacy-margins",
        )
    )
    ledger_context = AutoResearchLedgerCommitContext(
        project_id="project-a",
        experiment_id="experiment-candidate-legacy-margins",
        run_id="run-candidate-legacy-margins",
        attempt_id="attempt-candidate-legacy-margins",
        correlation_id="record-candidate-legacy-margins",
    )

    recorded = repository._record_autoresearch_result(
        candidate_result, ledger_context=ledger_context
    )
    assert recorded.replayed is False
    assert recorded.decision.protected_metric_margins == {"valid_tool_calls": pytest.approx(0.01)}

    with repository._connection(immediate=True) as connection:
        row = connection.execute(
            """
            SELECT decision_json FROM autoresearch_results
            WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
            """,
            ("workspace-a", "campaign-1", "candidate-legacy-margins"),
        ).fetchone()
        legacy_decision = json.loads(row["decision_json"])
        del legacy_decision["protected_metric_margins"]
        connection.execute(
            """
            UPDATE autoresearch_results SET decision_json = ?
            WHERE workspace_id = ? AND campaign_id = ? AND proposal_id = ?
            """,
            (
                json.dumps(legacy_decision),
                "workspace-a",
                "campaign-1",
                "candidate-legacy-margins",
            ),
        )

    replayed = repository._record_autoresearch_result(
        candidate_result, ledger_context=ledger_context
    )
    assert replayed.replayed is True
    assert replayed.decision.protected_metric_margins == {}


def test_result_write_rejects_unbounded_candidate_references_before_lineage_lookup(
    tmp_path,
):
    _path, _repository, core = fresh_core(tmp_path)
    oversized = result(
        "candidate-oversized",
        "study-missing",
        "attempt-000",
        0.5,
        role=ExperimentRole.CANDIDATE,
        provenance="simulated",
    ).model_copy(
        update={
            "attempt_ids": tuple(f"attempt-{index:03d}" for index in range(101)),
            "evidence_references": tuple(f"artifact-{index:03d}" for index in range(101)),
        }
    )

    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_result_reference_limit_exceeded",
    ):
        core.record_result(oversized)


def test_public_record_result_rejects_caller_built_completed_real_result(tmp_path):
    _path, repository, core = fresh_core(tmp_path)
    activate(core)
    core.submit_baseline(
        proposal("baseline-direct-real", estimated_cost=0.5),
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=principal(repository),
        correlation_id="baseline-direct-real-submit",
        idempotency_key="baseline-direct-real-submit",
    )
    study_id, attempt_id = select_and_finish(repository, "baseline-direct-real")

    caller_result = result(
        "baseline-direct-real",
        study_id,
        attempt_id,
        0.99,
        role=ExperimentRole.BASELINE,
    )
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_real_result_requires_sealed_projection",
    ):
        core.record_result(caller_result)
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_real_result_requires_sealed_projection",
    ):
        repository.record_autoresearch_result(caller_result)
    assert repository.list_autoresearch_outcomes("workspace-a", "campaign-1") == ()


def test_fresh_draft_campaign_has_controller_owned_preparation_and_source_template(tmp_path):
    _path, repository, core = fresh_core(tmp_path)

    before = core.state("workspace-a", "campaign-1", now=NOW)
    assert before.next_action == AutoResearchNextAction.PREPARE_CAMPAIGN

    prepared = core.prepare(
        "workspace-a",
        "campaign-1",
        controller_id="autoresearch-controller",
        correlation_id="autoresearch-prepare",
        idempotency_prefix="autoresearch-prepare",
    )
    assert prepared.status == CampaignStatus.READY
    assert prepared.version == 3
    assert core.state("workspace-a", "campaign-1", now=NOW).next_action == (
        AutoResearchNextAction.START_CAMPAIGN
    )
    started = (
        CampaignService(repository)
        .transition(
            "workspace-a",
            "campaign-1",
            CampaignTrigger.START,
            expected_version=prepared.version,
            principal=principal(repository),
            correlation_id="autoresearch-start",
            idempotency_key="autoresearch-start",
        )
        .campaign
    )
    assert started.status == CampaignStatus.ACTIVE
    assert started.version == 4
    assert core.state("workspace-a", "campaign-1", now=NOW).next_action == (
        AutoResearchNextAction.SUBMIT_BASELINE
    )
    event_types = [
        event.event_type for _cursor, event in repository.list_events("workspace-a", "campaign-1")
    ]
    assert event_types[-3:] == [
        "campaign:validation-started",
        "campaign:ready",
        "campaign:started",
    ]
    assert all(
        event.actor_id == "autoresearch-controller"
        for _cursor, event in (repository.list_events("workspace-a", "campaign-1")[-3:-1])
    )
    assert repository.list_events("workspace-a", "campaign-1")[-1][1].actor_id == "codex-agent"

    definition = AutoResearchTemplateDefinition(
        template_id="autoresearch-local-v1",
        objective=campaign().objective,
        target_model=campaign().target_model,
        manifest=manifest(),
    )
    registry = build_autoresearch_template_registry((definition,))
    assert list(registry) == ["autoresearch-local-v1"]
    assert registry["autoresearch-local-v1"]["manifest"]["max_proposal_rounds"] == 5
    assert "template_id" not in registry["autoresearch-local-v1"]
    builtins = builtin_autoresearch_template_registry()
    assert list(builtins) == [AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID]
    assert (
        builtins[AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID]["manifest"]["promotion_gates"][
            "quality_claim_eligible"
        ]
        is False
    )
    assert all(
        not payload["manifest"]["promotion_gates"].get("quality_claim_eligible", False)
        for payload in builtins.values()
    )


def test_direct_ledger_evaluation_cannot_bypass_sealed_projection(tmp_path):
    _path, repository, core = fresh_core(tmp_path, evaluation_binding=True)
    activate(core)
    actor = principal(repository)
    core.submit_baseline(
        proposal("baseline-ledger", estimated_cost=0.5),
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="baseline-ledger-submit",
        idempotency_key="baseline-ledger-submit",
    )
    study_id, campaign_attempt_id = select_and_finish(repository, "baseline-ledger")
    attempt = repository.get_attempt("workspace-a", campaign_attempt_id)
    artifact_sha = "f" * 64
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_budget_ledger(
                workspace_id, campaign_id, entry_id, unit, entry_kind,
                reserved_delta, actual_delta, limit_delta, action_id,
                evidence_json, actor_id, created_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, 0, ?, '{}', ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "settled-baseline-ledger",
                "gpu_hours",
                "settle",
                0.5,
                attempt.action_id,
                "autoresearch-controller",
                NOW.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id,
                uri, sha256, size_bytes, schema_name, sealed, valid,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 12, ?, 1, 1, '{}', ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "campaign-eval-artifact",
                attempt.action_id,
                "artifact://campaign/eval.json",
                artifact_sha,
                "evaluation-result.v1",
                NOW.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id,
                uri, sha256, size_bytes, schema_name, sealed, valid,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 24, ?, 1, 1, '{}', ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "campaign-nemo-gym-evidence",
                attempt.action_id,
                "artifact://campaign/nemo-gym-evidence.json",
                "9" * 64,
                "nemo_gym_campaign_evidence.v1",
                NOW.isoformat(),
            ),
        )

    ledger = core.ledger
    ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            display_name="AutoResearch",
            owner_actor_id="codex-agent",
        )
    )
    ledger.register_experiment(
        ExperimentSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            experiment_id="experiment-baseline-ledger",
            name="Baseline",
            objective="Establish the real baseline.",
        )
    )
    ledger.register_model(
        ModelSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            model_id="model-a",
            display_name="Pinned model",
            task_type="retrieval",
            architecture="encoder",
        )
    )
    ledger.register_model_version(
        ModelVersionSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            model_id="model-a",
            model_version_id="model-version-a",
            source_uri="hf://example/model",
            source_revision="abc123",
            config_digest="a" * 64,
        )
    )
    ledger.register_dataset(
        DatasetSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            dataset_id="dataset-a",
            display_name="Pinned data",
            task_type="retrieval",
        )
    )
    ledger.register_dataset_version(
        DatasetVersionSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            dataset_id="dataset-a",
            dataset_version_id="dataset-version-a",
            source_uri="artifact://dataset/manifest.json",
            content_digest="b" * 64,
        )
    )
    ledger.register_environment(
        EnvironmentSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            environment_id="environment-a",
            compute_target="registered-gpu",
            runtime_digest="c" * 64,
        )
    )
    ledger.register_run(
        RunSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            experiment_id="experiment-baseline-ledger",
            run_id="run-baseline-ledger",
            source_system="bashgym",
            source_run_id="run-baseline-ledger",
            campaign_id="campaign-1",
            study_id=study_id,
            action_id=attempt.action_id,
            run_kind="training",
            task_type="retrieval",
            training_method="lora",
            status=RunStatus.COMPLETED,
            context_status=ContextStatus.VERIFIED,
            model_version_id="model-version-a",
            dataset_version_id="dataset-version-a",
            environment_id="environment-a",
            recipe_digest="d" * 64,
            correlation_id="baseline-ledger",
        )
    )
    ledger.register_attempt(
        AttemptSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            run_id="run-baseline-ledger",
            attempt_id="ledger-attempt-baseline",
            attempt_number=1,
            source_attempt_id=campaign_attempt_id,
            status=RunStatus.COMPLETED,
        )
    )
    ledger.register_evaluation_suite(
        EvaluationSuiteSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_suite_id="suite-a",
            name="Held-out suite",
            task_type="retrieval",
            metric_contract={"primary_metric": "mrr_at_10"},
            code_digest="e" * 64,
        )
    )
    ledger.record_artifact(
        ArtifactSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            artifact_id="ledger-eval-artifact",
            run_id="run-baseline-ledger",
            attempt_id="ledger-attempt-baseline",
            kind="evaluation",
            uri="artifact://ledger/eval.json",
            sha256=artifact_sha,
            size_bytes=12,
            media_type="application/json",
        )
    )
    ledger.record_evaluation_result(
        EvaluationResultSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_result_id="evaluation-baseline-ledger",
            evaluation_suite_id="suite-a",
            run_id="run-baseline-ledger",
            attempt_id="ledger-attempt-baseline",
            model_version_id="model-version-a",
            status=RunStatus.COMPLETED,
            metrics={"mrr_at_10": 0.61},
            artifact_id="ledger-eval-artifact",
            completed_at=NOW,
        )
    )

    # Task 2 closes the legacy direct-ledger ingestion path: only deterministic
    # sealed campaign projection can create an authoritative evaluation.
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_sealed_evaluation_reader_required",
    ):
        core.ingest_evaluation_result(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            project_id="project-a",
            evaluation_result_id="evaluation-baseline-ledger",
        )
    assert repository.list_autoresearch_outcomes("workspace-a", "campaign-1") == ()


def test_caller_built_real_parent_cannot_unlock_code_candidate_or_lineage(tmp_path):
    _path, repository, core = fresh_core(tmp_path)
    activate(core)
    actor = principal(repository)
    core.submit_baseline(
        proposal("baseline-lineage", estimated_cost=0.5),
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="baseline-lineage-submit",
        idempotency_key="baseline-lineage-submit",
    )
    baseline_study, baseline_attempt = select_and_finish(repository, "baseline-lineage")
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_real_result_requires_sealed_projection",
    ):
        core.record_result(
            result(
                "baseline-lineage",
                baseline_study,
                baseline_attempt,
                0.50,
                role=ExperimentRole.BASELINE,
            )
        )
    candidate = proposal("candidate-code", estimated_cost=0.5).model_copy(
        update={
            "primary_variable": "trainer.optimizer",
            "prerequisite_study_ids": (baseline_study,),
        }
    )

    with pytest.raises(AutoResearchInvariantError, match="proposal_not_ready"):
        core.submit_controlled_candidate(
            candidate,
            parent_proposal_id="baseline-lineage",
            changed_variable="trainer.optimizer",
            expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
            principal=actor,
            correlation_id="candidate-code-submit",
            idempotency_key="candidate-code-submit",
        )

    assert repository.list_autoresearch_outcomes("workspace-a", "campaign-1") == ()
    assert repository.list_code_lineages("workspace-a", "campaign-1") == ()


def test_crashed_real_baseline_records_without_unlocking_quality_search(tmp_path):
    _path, repository, core = fresh_core(tmp_path)
    activate(core)
    actor = principal(repository)

    core.submit_baseline(
        proposal("baseline-crashed", estimated_cost=0.5),
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="baseline-crashed-submit",
        idempotency_key="baseline-crashed-submit",
    )
    baseline_study, baseline_attempt = select_and_finish(
        repository, "baseline-crashed", failed=True
    )
    crashed_result = result(
        "baseline-crashed",
        baseline_study,
        baseline_attempt,
        0.50,
        role=ExperimentRole.BASELINE,
    ).model_copy(update={"outcome": ExperimentOutcome.CRASHED, "metric_value": None})

    recorded = core.record_result(crashed_result)

    assert recorded.decision.decision == ResultDecision.CRASH
    assert recorded.decision.eligible_for_best is False
    assert recorded.result == crashed_result
    state = core.state("workspace-a", "campaign-1", now=NOW)
    assert state.baseline_verified is False
    assert state.best_proposal_id is None
    assert state.next_action == AutoResearchNextAction.SUBMIT_BASELINE


def test_simulated_baseline_is_explicit_and_never_unlocks_quality_search(tmp_path):
    _path, repository, core = fresh_core(tmp_path, max_attempts=2, target=None)
    activate(core)
    actor = principal(repository)
    fake = proposal("baseline-fake", estimated_cost=0.1).model_copy(
        update={
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {
                    "executor_kind": "fake",
                    "budget_unit": "gpu_hours",
                    "budget_reservation": 0.1,
                    "fake_steps": 3,
                },
            },
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Only exercise the dry control path.",
                        input_contract={"quality_claim": False},
                    ),
                )
            ),
        }
    )
    core.submit_baseline(
        fake,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="fake-submit",
        idempotency_key="fake-submit",
    )
    study_id, attempt_id = select_and_finish(repository, "baseline-fake")
    claimed_real = result(
        "baseline-fake",
        study_id,
        attempt_id,
        0.99,
        role=ExperimentRole.BASELINE,
    )
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_real_result_requires_sealed_projection",
    ):
        core.record_result(claimed_real)

    simulated = claimed_real.model_copy(update={"provenance": ExperimentProvenance.SIMULATED})
    recorded = core.record_result(simulated)
    assert recorded.decision.decision == ResultDecision.INELIGIBLE
    next_state = core.state("workspace-a", "campaign-1", now=NOW)
    assert next_state.baseline_verified is False
    assert next_state.best_proposal_id is None
    assert next_state.next_action == AutoResearchNextAction.SUBMIT_BASELINE


def test_candidate_cannot_run_before_a_real_baseline(tmp_path):
    _path, repository, core = fresh_core(tmp_path)
    activate(core)
    actor = principal(repository)
    value = proposal("candidate-too-early", estimated_cost=0.1)

    with pytest.raises(AutoResearchInvariantError, match="proposal_not_ready"):
        core.submit_controlled_candidate(
            value,
            parent_proposal_id="missing-baseline",
            changed_variable="learning_rate",
            expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
            principal=actor,
            correlation_id="too-early",
            idempotency_key="too-early",
        )


def _unseeded_verified_baseline(tmp_path):
    _path, repository, core = fresh_core(tmp_path, max_attempts=4, target=None)
    activate(core)
    baseline = _recipe_proposal("baseline-seed", learning_rate=0.001, seed=17).model_copy(
        update={"training_recipe": {"schema_version": "recipe.v1", "learning_rate": 0.001}}
    )
    core.submit_baseline(
        baseline,
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=principal(repository),
        correlation_id="submit-baseline-seed",
        idempotency_key="submit-baseline-seed",
    )
    baseline_study, baseline_attempt = select_and_finish(repository, "baseline-seed")
    _insert_authoritative_outcome(
        repository,
        _authoritative_outcome(
            "baseline-seed",
            baseline_study,
            baseline_attempt,
            0.5,
            role=ExperimentRole.BASELINE,
            decision=ResultDecision.BASELINE,
            eligible_for_best=True,
        ),
    )
    return repository, core, baseline_study


def test_candidate_with_training_stage_requires_a_declared_seed(tmp_path) -> None:
    repository, core, baseline_study = _unseeded_verified_baseline(tmp_path)
    unseeded = _recipe_proposal("candidate-unseeded", learning_rate=0.002, seed=17).model_copy(
        update={
            "training_recipe": {"schema_version": "recipe.v1", "learning_rate": 0.002},
            "prerequisite_study_ids": (baseline_study,),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Train the candidate.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Compare on the fixed suite.",
                    ),
                )
            ),
        }
    )

    with pytest.raises(AutoResearchInvariantError) as excinfo:
        core.submit_candidate(
            unseeded,
            parent_proposal_id="baseline-seed",
            changed_variables=("learning_rate",),
            expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
            principal=principal(repository),
            correlation_id="candidate-unseeded",
            idempotency_key="candidate-unseeded",
        )

    assert "autoresearch_candidate_requires_training_seed" in str(excinfo.value)
