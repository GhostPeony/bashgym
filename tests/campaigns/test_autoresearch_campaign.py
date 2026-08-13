"""Baseline-first AutoResearch campaign control-loop tests."""

import json
from datetime import datetime

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.autoresearch import (
    AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID,
    AutoResearchCampaignCore,
    AutoResearchCampaignSpec,
    AutoResearchInvariantError,
    AutoResearchNextAction,
    AutoResearchRepository,
    AutoResearchResult,
    AutoResearchStopRules,
    AutoResearchTemplateDefinition,
    ExperimentOutcome,
    ExperimentProvenance,
    ExperimentRole,
    MetricDirection,
    ProtectedMetricGate,
    ResultDecision,
    build_autoresearch_template_registry,
    builtin_autoresearch_template_registry,
)
from bashgym.campaigns.contracts import (
    CampaignStatus,
    CampaignTrigger,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyStatus,
)
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
from tests.campaigns.test_proposals import principal, proposal

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def make_spec(
    *,
    max_attempts: int = 3,
    target: float | None = 0.95,
    evaluation_binding: bool = False,
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
            deadline=datetime(2099, 7, 14, 12, 0, tzinfo=UTC),
        ),
        ledger_project_id="project-a" if evaluation_binding else None,
        evaluation_suite_id="suite-a" if evaluation_binding else None,
        created_at=NOW,
    )


def fresh_core(tmp_path, *, max_attempts=3, target=0.95, evaluation_binding=False):
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
