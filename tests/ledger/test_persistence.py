"""Experiment-ledger identity, isolation, replay, and lineage tests."""

import sqlite3
from datetime import datetime, timedelta

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.contracts import canonical_hash
from bashgym.campaigns.persistence import RecordNotFoundError
from bashgym.ledger.contracts import (
    ArtifactSpec,
    AttemptSpec,
    ContextStatus,
    DatasetSpec,
    DatasetVersionSpec,
    DecisionSpec,
    EnvironmentSpec,
    EvaluationResultSpec,
    EvaluationSuiteSpec,
    ExperimentSpec,
    LedgerEventSpec,
    MetricPointSpec,
    ModelSpec,
    ModelVersionSpec,
    ProjectSpec,
    RunSpec,
    RunStatus,
)
from bashgym.ledger.persistence import (
    ExperimentLedgerRepository,
    LedgerConflictError,
    LedgerPersistenceError,
    LedgerTransitionError,
)
from bashgym.ledger.synthesis import build_project_context, compare_runs, metric_trend


@pytest.fixture
def repository(tmp_path) -> ExperimentLedgerRepository:
    value = ExperimentLedgerRepository(tmp_path / "campaigns.sqlite3")
    value.initialize()
    return value


def test_open_existing_ledger_is_read_only(repository):
    seed_project(repository)

    reader = ExperimentLedgerRepository.open_existing(repository.db_path)

    assert reader.get_project("workspace-a", "project-a")["project_id"] == "project-a"
    with pytest.raises(LedgerPersistenceError, match="ledger_read_only"):
        reader.register_project(
            ProjectSpec(
                workspace_id="workspace-a",
                project_id="project-read-only",
                display_name="Must not be written",
                owner_actor_id="codex",
            )
        )


def test_open_existing_ledger_rejects_migration_checksum_mismatch(repository):
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE campaign_schema_migrations SET checksum = ? WHERE version = 1",
            ("0" * 64,),
        )

    with pytest.raises(LedgerPersistenceError, match="ledger_schema_unavailable"):
        ExperimentLedgerRepository.open_existing(repository.db_path)


def seed_project(
    repository: ExperimentLedgerRepository,
    *,
    workspace_id: str = "workspace-a",
    project_id: str = "project-a",
) -> None:
    repository.register_project(
        ProjectSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            display_name=f"Project {project_id}",
            owner_actor_id="codex",
            tags=("retrieval",),
        )
    )
    repository.register_experiment(
        ExperimentSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            experiment_id="experiment-1",
            name="Retriever quality",
            objective="Improve retrieval quality without regressing latency.",
        )
    )
    repository.register_model(
        ModelSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            model_id="model-1",
            display_name="Embedding model",
            task_type="retrieval",
            architecture="encoder",
        )
    )
    repository.register_model_version(
        ModelVersionSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            model_id="model-1",
            model_version_id="model-version-1",
            source_uri="hf://example/model",
            source_revision="abc123",
            config_digest="a" * 64,
        )
    )
    repository.register_dataset(
        DatasetSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            dataset_id="dataset-1",
            display_name="Training pairs",
            task_type="retrieval",
        )
    )
    repository.register_dataset_version(
        DatasetVersionSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            dataset_id="dataset-1",
            dataset_version_id="dataset-version-1",
            source_uri="file://artifacts/dataset-manifest.json",
            content_digest="b" * 64,
            split_manifest={"train": "train.jsonl", "dev": "dev.jsonl"},
            row_counts={"train": 100, "dev": 20},
        )
    )
    repository.register_environment(
        EnvironmentSpec(
            workspace_id=workspace_id,
            project_id=project_id,
            environment_id="environment-1",
            compute_target="local-gpu",
            runtime_digest="c" * 64,
            hardware={"accelerator_family": "blackwell", "accelerator_count": 1},
        )
    )


def run_spec(
    *, workspace_id: str = "workspace-a", project_id: str = "project-a", run_id: str = "run-1"
) -> RunSpec:
    return RunSpec(
        workspace_id=workspace_id,
        project_id=project_id,
        experiment_id="experiment-1",
        run_id=run_id,
        source_system="bashgym",
        source_run_id=run_id,
        run_kind="training",
        task_type="retrieval",
        training_method="embedding",
        context_status=ContextStatus.VERIFIED,
        model_version_id="model-version-1",
        dataset_version_id="dataset-version-1",
        environment_id="environment-1",
        recipe_digest="d" * 64,
        config={"epochs": 3, "learning_rate": 2e-5},
        correlation_id="correlation-1",
    )


def test_project_and_run_replays_are_exact_and_conflicts_fail(repository):
    seed_project(repository)
    first, first_replay = repository.register_run(run_spec())
    replay, replayed = repository.register_run(run_spec())

    assert first_replay is False
    assert replayed is True
    assert first == replay

    changed = run_spec().model_copy(update={"recipe_digest": "e" * 64})
    with pytest.raises(LedgerConflictError, match="different identity"):
        repository.register_run(changed)


def test_workspace_and_project_boundaries_do_not_leak(repository):
    seed_project(repository)
    repository.register_run(run_spec())
    seed_project(repository, project_id="project-b")

    assert repository.list_runs("workspace-a", "project-b") == []
    with pytest.raises(RecordNotFoundError, match="ledger run not found"):
        repository.get_run("workspace-a", "project-b", "run-1")
    with pytest.raises(RecordNotFoundError, match="ledger project not found"):
        repository.get_project("workspace-b", "project-a")


def test_attempt_metric_event_and_transition_ingestion_are_idempotent(repository):
    seed_project(repository)
    repository.register_run(run_spec())
    repository.register_attempt(
        AttemptSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            run_id="run-1",
            attempt_id="attempt-1",
            attempt_number=1,
        )
    )

    running = repository.transition_run(
        "workspace-a", "project-a", "run-1", RunStatus.RUNNING
    )
    repository.transition_attempt(
        "workspace-a", "project-a", "attempt-1", RunStatus.RUNNING
    )
    assert running["started_at"] is not None

    point = MetricPointSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        run_id="run-1",
        attempt_id="attempt-1",
        source="trainer",
        step=1,
        metric_name="train.loss",
        metric_value=1.25,
        raw_sha256=canonical_hash({"loss": 1.25, "step": 1}),
    )
    assert repository.append_metric(point) is True
    assert repository.append_metric(point) is False
    with pytest.raises(LedgerConflictError, match="different data"):
        repository.append_metric(point.model_copy(update={"metric_value": 1.5}))

    event = LedgerEventSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        experiment_id="experiment-1",
        run_id="run-1",
        attempt_id="attempt-1",
        event_type="training-progress",
        source_system="bashgym",
        source_event_id="run-1-step-1",
        correlation_id="correlation-1",
        payload={"step": 1, "metric_names": ["train.loss"]},
    )
    _, replayed = repository.append_event(event)
    _, replayed_again = repository.append_event(event)
    assert replayed is False
    assert replayed_again is True

    repository.transition_attempt(
        "workspace-a", "project-a", "attempt-1", RunStatus.COMPLETED
    )
    repository.transition_run(
        "workspace-a", "project-a", "run-1", RunStatus.COMPLETED
    )
    with pytest.raises(LedgerTransitionError):
        repository.transition_run("workspace-a", "project-a", "run-1", RunStatus.RUNNING)


def test_evaluation_artifact_decision_and_context_pack_are_linked(repository):
    seed_project(repository)
    repository.register_run(run_spec())
    repository.register_attempt(
        AttemptSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            run_id="run-1",
            attempt_id="attempt-1",
            attempt_number=1,
            status=RunStatus.RUNNING,
        )
    )
    repository.append_metric(
        MetricPointSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            run_id="run-1",
            attempt_id="attempt-1",
            source="trainer",
            step=1,
            metric_name="train.loss",
            metric_value=1.5,
            raw_sha256="1" * 64,
        )
    )
    repository.append_metric(
        MetricPointSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            run_id="run-1",
            attempt_id="attempt-1",
            source="trainer",
            step=2,
            metric_name="train.loss",
            metric_value=1.0,
            raw_sha256="2" * 64,
        )
    )
    repository.transition_run("workspace-a", "project-a", "run-1", RunStatus.RUNNING)
    repository.transition_run("workspace-a", "project-a", "run-1", RunStatus.COMPLETED)
    repository.record_artifact(
        ArtifactSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            artifact_id="artifact-1",
            run_id="run-1",
            attempt_id="attempt-1",
            kind="checkpoint",
            uri="file://artifacts/run-1/checkpoint",
            sha256="3" * 64,
            size_bytes=1024,
            media_type="application/x-directory-manifest",
        )
    )
    repository.register_evaluation_suite(
        EvaluationSuiteSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_suite_id="eval-suite-1",
            name="Frozen retrieval development set",
            task_type="retrieval",
            dataset_version_id="dataset-version-1",
            metric_contract={"mrr_at_10": {"direction": "maximize"}},
            code_digest="4" * 64,
        )
    )
    repository.record_evaluation_result(
        EvaluationResultSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_result_id="eval-result-1",
            evaluation_suite_id="eval-suite-1",
            run_id="run-1",
            attempt_id="attempt-1",
            model_version_id="model-version-1",
            status=RunStatus.COMPLETED,
            metrics={"mrr_at_10": 0.72},
            artifact_id="artifact-1",
            completed_at=datetime.now(UTC),
        )
    )
    repository.record_decision(
        DecisionSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            decision_id="decision-1",
            experiment_id="experiment-1",
            run_id="run-1",
            decision_type="retain_candidate",
            outcome="Retain for another development comparison.",
            rationale="Quality improved and the protected test remains unopened.",
            evidence_refs=("eval-result-1", "artifact-1"),
            actor_id="codex",
        )
    )

    trend = metric_trend(
        repository.metric_series(
            "workspace-a", "project-a", metric_name="train.loss", run_id="run-1"
        )
    )
    context = build_project_context(repository, "workspace-a", "project-a")

    assert trend["delta"] == pytest.approx(-0.5)
    assert context["health"]["status"] == "healthy"
    assert context["inventory"]["evaluation_count"] == 1
    assert context["recent_runs"][0]["model_version_id"] == "model-version-1"
    assert context["evidence"]["decision_ids"] == ["decision-1"]


def test_context_pack_flags_stale_or_incomplete_runs(repository):
    seed_project(repository)
    unverified = run_spec().model_copy(
        update={
            "run_id": "run-unassigned",
            "source_run_id": "run-unassigned",
            "context_status": ContextStatus.UNASSIGNED,
            "model_version_id": None,
            "dataset_version_id": None,
            "environment_id": None,
            "queued_at": datetime.now(UTC) - timedelta(days=1),
        }
    )
    repository.register_run(unverified)

    context = build_project_context(
        repository,
        "workspace-a",
        "project-a",
        now=datetime.now(UTC),
        stale_after=timedelta(hours=1),
    )

    assert context["health"]["status"] == "attention"
    assert context["health"]["stale_run_count"] == 1
    assert context["health"]["missing_context_count"] == 1


def test_run_comparison_requires_the_same_evaluation_suite(repository):
    seed_project(repository)
    repository.register_run(run_spec(run_id="run-1"))
    second = run_spec(run_id="run-2").model_copy(
        update={"correlation_id": "correlation-2"}
    )
    repository.register_run(second)
    repository.register_evaluation_suite(
        EvaluationSuiteSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_suite_id="eval-suite-1",
            name="Frozen retrieval development set",
            task_type="retrieval",
            dataset_version_id="dataset-version-1",
            metric_contract={"mrr_at_10": {"direction": "maximize"}},
            code_digest="4" * 64,
        )
    )
    for run_id, score in (("run-1", 0.70), ("run-2", 0.75)):
        repository.record_evaluation_result(
            EvaluationResultSpec(
                workspace_id="workspace-a",
                project_id="project-a",
                evaluation_result_id=f"result-{run_id}",
                evaluation_suite_id="eval-suite-1",
                run_id=run_id,
                model_version_id="model-version-1",
                status=RunStatus.COMPLETED,
                metrics={"mrr_at_10": score},
                completed_at=datetime.now(UTC),
            )
        )

    comparison = compare_runs(
        repository, "workspace-a", "project-a", ["run-1", "run-2"]
    )

    metric = comparison["comparisons"][0]["metrics"][0]
    assert metric["direction"] == "maximize"
    assert metric["values"]["run-2"]["delta_from_baseline"] == pytest.approx(0.05)
    assert comparison["not_comparable_run_ids"] == []


def test_secret_shaped_metadata_is_rejected():
    with pytest.raises(ValueError, match="forbidden key"):
        ExperimentSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            experiment_id="unsafe-experiment",
            name="Unsafe",
            objective="No secrets should be stored.",
            metadata={"api_key": "not-allowed"},
        )
