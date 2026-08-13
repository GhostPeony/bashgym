"""Authenticated experiment-ledger REST projections."""

from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym._compat import UTC
from bashgym.api.campaign_routes import campaign_auth_router
from bashgym.api.ledger_routes import router as ledger_router
from bashgym.api.routes import create_app
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.autoresearch import AutoResearchInvariantError
from bashgym.campaigns.contracts import AutonomyProfile, canonical_hash
from bashgym.ledger.contracts import (
    ArtifactSpec,
    AttemptSpec,
    DecisionSpec,
    EvaluationResultSpec,
    EvaluationSuiteSpec,
    LedgerEventSpec,
    MetricPointSpec,
    RunStatus,
)
from bashgym.ledger.persistence import ExperimentLedgerRepository
from tests.api.test_campaign_routes import bearer, exchange
from tests.ledger.test_persistence import run_spec, seed_project


def ledger_client(tmp_path):
    repository = ExperimentLedgerRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="hermes-agent",
        autonomy_profile=AutonomyProfile.HERMES_BOUNDED,
        workspace_ids=("workspace-a",),
    )
    app = FastAPI()
    app.state.campaign_repository = repository
    app.state.experiment_ledger_repository = repository
    app.state.campaign_auth_service = auth
    app.include_router(campaign_auth_router)
    app.include_router(ledger_router)
    return TestClient(app), repository, refresh


def test_create_app_registers_ledger_routes():
    paths = set(create_app().openapi()["paths"])
    assert {
        "/api/ledger/health",
        "/api/ledger/projects",
        "/api/ledger/projects/{project_id}",
        "/api/ledger/projects/{project_id}/context",
        "/api/ledger/projects/{project_id}/runs",
        "/api/ledger/projects/{project_id}/runs/{run_id}",
        "/api/ledger/projects/{project_id}/compare",
        "/api/ledger/projects/{project_id}/metrics/{metric_name}/trend",
        "/api/ledger/projects/{project_id}/evaluations",
        "/api/ledger/projects/{project_id}/evaluation-suites",
        "/api/ledger/projects/{project_id}/evaluation-results",
        "/api/ledger/projects/{project_id}/artifacts",
        "/api/ledger/projects/{project_id}/decisions",
        "/api/ledger/projects/{project_id}/events",
    } <= paths


def test_ledger_requires_campaign_read_capability(tmp_path):
    http, _repository, _refresh = ledger_client(tmp_path)
    response = http.get("/api/ledger/projects", params={"workspace_id": "workspace-a"})
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "campaign_auth_required"


def test_project_context_trend_health_and_incremental_events(tmp_path):
    http, repository, refresh = ledger_client(tmp_path)
    token = exchange(http, refresh.raw_token)
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
    for step, loss in ((1, 1.5), (2, 1.0)):
        repository.append_metric(
            MetricPointSpec(
                workspace_id="workspace-a",
                project_id="project-a",
                run_id="run-1",
                attempt_id="attempt-1",
                source="trainer",
                step=step,
                metric_name="train.loss",
                metric_value=loss,
                raw_sha256=canonical_hash({"step": step, "loss": loss}),
            )
        )
        repository.append_event(
            LedgerEventSpec(
                workspace_id="workspace-a",
                project_id="project-a",
                experiment_id="experiment-1",
                run_id="run-1",
                attempt_id="attempt-1",
                event_type="training-progress",
                source_system="bashgym",
                source_event_id=f"run-1-step-{step}",
                correlation_id="correlation-1",
                payload={"step": step, "metric_names": ["train.loss"]},
            )
        )

    headers = bearer(token)
    projects = http.get(
        "/api/ledger/projects", params={"workspace_id": "workspace-a"}, headers=headers
    )
    context = http.get(
        "/api/ledger/projects/project-a/context",
        params={"workspace_id": "workspace-a"},
        headers=headers,
    )
    trend = http.get(
        "/api/ledger/projects/project-a/metrics/train.loss/trend",
        params={"workspace_id": "workspace-a", "run_id": "run-1"},
        headers=headers,
    )
    first_events = http.get(
        "/api/ledger/projects/project-a/events",
        params={"workspace_id": "workspace-a", "after_cursor": 0, "limit": 1},
        headers=headers,
    )
    health = http.get("/api/ledger/health", params={"workspace_id": "workspace-a"}, headers=headers)

    assert projects.status_code == 200
    assert projects.json()["projects"][0]["project_id"] == "project-a"
    assert context.status_code == 200
    assert context.json()["inventory"]["run_count"] == 1
    assert trend.status_code == 200
    assert trend.json()["trend"]["delta"] == -0.5
    assert first_events.status_code == 200
    assert first_events.json()["next_cursor"] == 1
    assert first_events.json()["has_more"] is True
    assert health.status_code == 200
    assert health.json()["quick_check"] == "ok"
    assert health.json()["counts"]["metric_points"] == 2


def test_project_boundary_returns_not_found_without_leaking_other_project(tmp_path):
    http, repository, refresh = ledger_client(tmp_path)
    token = exchange(http, refresh.raw_token)
    seed_project(repository)
    repository.register_run(run_spec())

    response = http.get(
        "/api/ledger/projects/missing/runs/run-1",
        params={"workspace_id": "workspace-a"},
        headers=bearer(token),
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "ledger_not_found"


def test_authenticated_ingestion_is_idempotent_and_project_scoped(tmp_path):
    http, repository, refresh = ledger_client(tmp_path)
    token = exchange(http, refresh.raw_token)
    headers = bearer(token)
    seed_project(repository)
    repository.register_run(run_spec())

    suite = EvaluationSuiteSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_suite_id="eval-suite-1",
        name="Frozen retrieval development set",
        task_type="retrieval",
        dataset_version_id="dataset-version-1",
        metric_contract={"mrr_at_10": {"direction": "maximize"}},
        code_digest="4" * 64,
    )
    artifact = ArtifactSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        artifact_id="artifact-1",
        run_id="run-1",
        kind="evaluation-report",
        uri="file://artifacts/run-1/evaluation.json",
        sha256="5" * 64,
        size_bytes=512,
        media_type="application/json",
    )
    result = EvaluationResultSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_result_id="eval-result-1",
        evaluation_suite_id="eval-suite-1",
        run_id="run-1",
        model_version_id="model-version-1",
        status=RunStatus.COMPLETED,
        metrics={"mrr_at_10": 0.72},
        artifact_id="artifact-1",
        completed_at=datetime.now(UTC),
    )
    decision = DecisionSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        decision_id="decision-1",
        experiment_id="experiment-1",
        run_id="run-1",
        decision_type="retain-candidate",
        outcome="Retain for another development comparison.",
        rationale="Quality improved on the frozen development suite.",
        evidence_refs=("eval-result-1", "artifact-1"),
        actor_id="hermes-agent",
    )
    event = LedgerEventSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        experiment_id="experiment-1",
        run_id="run-1",
        event_type="evaluation-completed",
        source_system="bashgym-eval",
        source_event_id="eval-result-1-completed",
        correlation_id="correlation-1",
        payload={"evaluation_result_id": "eval-result-1"},
    )

    calls = (
        ("evaluation-suites", suite),
        ("artifacts", artifact),
        ("evaluation-results", result),
        ("decisions", decision),
        ("events", event),
    )
    for route, record in calls:
        first = http.post(
            f"/api/ledger/projects/project-a/{route}",
            params={"workspace_id": "workspace-a"},
            headers=headers,
            json=record.model_dump(mode="json"),
        )
        replay = http.post(
            f"/api/ledger/projects/project-a/{route}",
            params={"workspace_id": "workspace-a"},
            headers=headers,
            json=record.model_dump(mode="json"),
        )
        assert first.status_code == 200
        assert first.json()["replayed"] is False
        assert replay.status_code == 200
        assert replay.json()["replayed"] is True

    mismatch = http.post(
        "/api/ledger/projects/other-project/artifacts",
        params={"workspace_id": "workspace-a"},
        headers=headers,
        json=artifact.model_dump(mode="json"),
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["code"] == "ledger_scope_mismatch"

    context = http.get(
        "/api/ledger/projects/project-a/context",
        params={"workspace_id": "workspace-a"},
        headers=headers,
    )
    assert context.status_code == 200
    assert context.json()["inventory"]["evaluation_count"] == 1
    assert context.json()["evidence"]["decision_ids"] == ["decision-1"]


def test_completed_campaign_evaluation_triggers_authoritative_autoresearch_ingestion(
    tmp_path, monkeypatch
):
    http, repository, refresh = ledger_client(tmp_path)
    token = exchange(http, refresh.raw_token)
    seed_project(repository)
    repository.register_run(
        run_spec().model_copy(
            update={
                "campaign_id": "campaign-1",
                "study_id": "study-1",
                "action_id": "action-1",
                "status": RunStatus.COMPLETED,
            }
        )
    )
    repository.register_evaluation_suite(
        EvaluationSuiteSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_suite_id="eval-suite-1",
            name="Pinned held-out suite",
            task_type="retrieval",
            dataset_version_id="dataset-version-1",
            metric_contract={"primary_metric": "mrr_at_10"},
            code_digest="4" * 64,
        )
    )
    calls = []

    class Outcome:
        replayed = False

        def model_dump(self, *, mode):
            assert mode == "json"
            return {"schema_version": "autoresearch_outcome_record.v1"}

    class Core:
        def ingest_evaluation_result(self, **kwargs):
            calls.append(kwargs)
            return Outcome()

    monkeypatch.setattr("bashgym.api.ledger_routes._autoresearch_core", lambda _repo: Core())
    result = EvaluationResultSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_result_id="eval-result-campaign-1",
        evaluation_suite_id="eval-suite-1",
        run_id="run-1",
        model_version_id="model-version-1",
        status=RunStatus.COMPLETED,
        metrics={"mrr_at_10": 0.72},
        completed_at=datetime.now(UTC),
    )

    response = http.post(
        "/api/ledger/projects/project-a/evaluation-results",
        params={"workspace_id": "workspace-a"},
        headers=bearer(token),
        json=result.model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json()["autoresearch_ingestion"] == {
        "schema_version": "autoresearch_evaluation_ingestion.v1",
        "status": "ingested",
        "code": "autoresearch_evaluation_ingested",
        "campaign_id": "campaign-1",
        "outcome": {"schema_version": "autoresearch_outcome_record.v1"},
    }
    assert calls == [
        {
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "project_id": "project-a",
            "evaluation_result_id": "eval-result-campaign-1",
        }
    ]

    checkpoint = result.model_copy(
        update={
            "evaluation_result_id": "eval-result-checkpoint-1",
            "slice_metrics": {
                "autoresearch_role": "checkpoint",
                "checkpoint_step": 80,
            },
        }
    )
    checkpoint_response = http.post(
        "/api/ledger/projects/project-a/evaluation-results",
        params={"workspace_id": "workspace-a"},
        headers=bearer(token),
        json=checkpoint.model_dump(mode="json"),
    )

    assert checkpoint_response.status_code == 200
    assert checkpoint_response.json()["autoresearch_ingestion"] == {
        "schema_version": "autoresearch_evaluation_ingestion.v1",
        "status": "not_applicable",
        "code": "autoresearch_checkpoint_evaluation",
        "campaign_id": "campaign-1",
        "outcome": None,
    }
    assert len(calls) == 1

    class DeferredCore:
        def ingest_evaluation_result(self, **_kwargs):
            raise AutoResearchInvariantError("autoresearch_study_budget_not_settled")

    monkeypatch.setattr(
        "bashgym.api.ledger_routes._autoresearch_core", lambda _repo: DeferredCore()
    )
    deferred = result.model_copy(update={"evaluation_result_id": "eval-result-deferred"})
    deferred_response = http.post(
        "/api/ledger/projects/project-a/evaluation-results",
        params={"workspace_id": "workspace-a"},
        headers=bearer(token),
        json=deferred.model_dump(mode="json"),
    )

    assert deferred_response.status_code == 200
    assert deferred_response.json()["autoresearch_ingestion"] == {
        "schema_version": "autoresearch_evaluation_ingestion.v1",
        "status": "deferred",
        "code": "autoresearch_study_budget_not_settled",
        "campaign_id": "campaign-1",
        "outcome": None,
    }
    assert (
        repository.get_evaluation_result("workspace-a", "project-a", "eval-result-deferred")[
            "status"
        ]
        == "completed"
    )
