"""E2E tests for orchestrator API routes.

Uses FastAPI TestClient against the actual route handlers
with mocked LLM and CLI backends.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import create_app
from bashgym.orchestrator.models import (
    LLMConfig,
    LLMProvider,
    OrchestratorSpec,
    TaskNode,
    TaskStatus,
    WorkerResult,
)
from bashgym.orchestrator.task_dag import TaskDAG

from tests.orchestrator.conftest import THREE_TASK_DECOMPOSITION, make_task, make_result


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Create a fresh FastAPI app for each test."""
    return create_app()


@pytest.fixture
def client(app):
    """TestClient wrapping the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_jobs():
    """Clear the in-memory job store before each test."""
    from bashgym.api import orchestrator_routes
    orchestrator_routes._jobs.clear()
    yield
    orchestrator_routes._jobs.clear()


def valid_spec_payload():
    """Return a valid SpecRequest JSON payload."""
    return {
        "title": "Add auth system",
        "description": "Implement JWT auth with login and register",
        "constraints": ["Use bcrypt"],
        "acceptance_criteria": ["Login returns JWT"],
        "max_budget_usd": 5.0,
        "max_workers": 3,
    }


# =============================================================================
# Submit Endpoint
# =============================================================================


class TestSubmitEndpoint:
    """Tests for POST /api/orchestrate/submit."""

    def test_submit_returns_job_id(self, client):
        """Valid spec → 200 with job_id and status=decomposing."""
        # Mock the background decomposition so it doesn't actually call LLM
        with patch(
            "bashgym.api.orchestrator_routes._decompose_spec",
            new_callable=AsyncMock,
        ):
            resp = client.post("/api/orchestrate/submit", json=valid_spec_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "decomposing"
        assert data["provider"] == "anthropic"

    def test_submit_missing_title_returns_422(self, client):
        """Missing required field title → 422."""
        payload = valid_spec_payload()
        del payload["title"]
        resp = client.post("/api/orchestrate/submit", json=payload)
        assert resp.status_code == 422

    def test_submit_missing_description_returns_422(self, client):
        """Missing required field description → 422."""
        payload = valid_spec_payload()
        del payload["description"]
        resp = client.post("/api/orchestrate/submit", json=payload)
        assert resp.status_code == 422

    def test_submit_with_custom_provider(self, client):
        """Submit with openai provider config."""
        payload = valid_spec_payload()
        payload["llm_config"] = {"provider": "openai", "model": "gpt-4o"}

        with patch(
            "bashgym.api.orchestrator_routes._decompose_spec",
            new_callable=AsyncMock,
        ):
            resp = client.post("/api/orchestrate/submit", json=payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "openai"

    def test_submit_invalid_provider_returns_400(self, client):
        """Invalid provider → 400."""
        payload = valid_spec_payload()
        payload["llm_config"] = {"provider": "invalid_provider"}

        resp = client.post("/api/orchestrate/submit", json=payload)
        assert resp.status_code == 400


# =============================================================================
# Approve Endpoint
# =============================================================================


class TestApproveEndpoint:
    """Tests for POST /api/orchestrate/{job_id}/approve."""

    def test_approve_starts_execution(self, client):
        """Submit → manually set to awaiting_approval → approve → executing."""
        from bashgym.api import orchestrator_routes
        from bashgym.orchestrator.agent import OrchestrationAgent

        # Create a job in awaiting_approval state
        dag = TaskDAG()
        dag.add_task(make_task("t1"))

        spec = OrchestratorSpec(
            title="Test", description="Test desc",
            max_budget_usd=5.0, max_workers=3,
        )

        agent = MagicMock(spec=OrchestrationAgent)
        agent.pool = MagicMock()

        job_id = "test-001"
        orchestrator_routes._jobs[job_id] = {
            "id": job_id,
            "status": "awaiting_approval",
            "spec": spec,
            "llm_config": LLMConfig(),
            "agent": agent,
            "dag": dag,
            "results": [],
            "error": None,
        }

        with patch(
            "bashgym.api.orchestrator_routes._execute_dag",
            new_callable=AsyncMock,
        ):
            resp = client.post(
                f"/api/orchestrate/{job_id}/approve",
                json={"base_branch": "main"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "executing"
        assert data["task_count"] == 1

    def test_approve_nonexistent_job_returns_404(self, client):
        """Approve unknown job_id → 404."""
        resp = client.post(
            "/api/orchestrate/nonexistent/approve",
            json={"base_branch": "main"},
        )
        assert resp.status_code == 404

    def test_approve_wrong_status_returns_400(self, client):
        """Approve a job not in awaiting_approval → 400."""
        from bashgym.api import orchestrator_routes

        orchestrator_routes._jobs["job-x"] = {
            "id": "job-x",
            "status": "decomposing",
            "spec": None,
            "llm_config": None,
            "agent": None,
            "dag": None,
            "results": [],
            "error": None,
        }

        resp = client.post(
            "/api/orchestrate/job-x/approve",
            json={"base_branch": "main"},
        )
        assert resp.status_code == 400


# =============================================================================
# Status Endpoint
# =============================================================================


class TestStatusEndpoint:
    """Tests for GET /api/orchestrate/{job_id}/status."""

    def test_status_returns_dag_and_stats(self, client):
        """Get status for a job with a DAG."""
        from bashgym.api import orchestrator_routes

        dag = TaskDAG()
        dag.add_task(make_task("t1"))
        dag.add_task(make_task("t2", deps=["t1"]))

        agent = MagicMock()
        agent.budget_status = {
            "spent_usd": 0.5,
            "limit_usd": 10.0,
            "remaining_usd": 9.5,
            "exceeded": False,
        }

        orchestrator_routes._jobs["job-s"] = {
            "id": "job-s",
            "status": "awaiting_approval",
            "spec": None,
            "llm_config": None,
            "agent": agent,
            "dag": dag,
            "results": [],
            "error": None,
        }

        resp = client.get("/api/orchestrate/job-s/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-s"
        assert data["status"] == "awaiting_approval"
        assert "dag" in data
        assert "stats" in data
        assert "budget" in data
        assert data["budget"]["spent_usd"] == 0.5

    def test_status_nonexistent_job_returns_404(self, client):
        """GET status for unknown job → 404."""
        resp = client.get("/api/orchestrate/nonexistent/status")
        assert resp.status_code == 404

    def test_status_includes_results(self, client):
        """Status includes cost/time when results exist."""
        from bashgym.api import orchestrator_routes

        dag = TaskDAG()
        dag.add_task(make_task("t1"))

        results = [make_result("t1", cost=1.5, duration=30.0)]

        orchestrator_routes._jobs["job-r"] = {
            "id": "job-r",
            "status": "completed",
            "spec": None,
            "llm_config": None,
            "agent": None,
            "dag": dag,
            "results": results,
            "error": None,
        }

        resp = client.get("/api/orchestrate/job-r/status")
        data = resp.json()
        assert data["total_cost"] == pytest.approx(1.5)
        assert data["total_time"] == pytest.approx(30.0)
        assert data["completed"] == 1


# =============================================================================
# Retry Endpoint
# =============================================================================


class TestRetryEndpoint:
    """Tests for POST /api/orchestrate/{job_id}/task/{task_id}/retry."""

    def test_retry_failed_task(self, client):
        """Retry a failed task → status resets to retrying."""
        from bashgym.api import orchestrator_routes

        dag = TaskDAG()
        task = make_task("t1")
        task.status = TaskStatus.FAILED
        dag.add_task(task)

        spec = OrchestratorSpec(
            title="Test", description="desc",
            base_branch="main",
        )

        orchestrator_routes._jobs["job-rt"] = {
            "id": "job-rt",
            "status": "completed",
            "spec": spec,
            "llm_config": None,
            "agent": MagicMock(),
            "dag": dag,
            "results": [],
            "error": None,
        }

        with patch(
            "bashgym.api.orchestrator_routes._execute_dag",
            new_callable=AsyncMock,
        ):
            resp = client.post(
                "/api/orchestrate/job-rt/task/t1/retry",
                json={},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "retrying"
        assert data["task_id"] == "t1"
        assert data["retry_count"] == 1
        # Task should be reset to PENDING
        assert dag.nodes["t1"].status == TaskStatus.PENDING

    def test_retry_with_modified_prompt(self, client):
        """Retry with custom prompt → prompt stored on task."""
        from bashgym.api import orchestrator_routes

        dag = TaskDAG()
        task = make_task("t1")
        task.status = TaskStatus.FAILED
        dag.add_task(task)

        spec = OrchestratorSpec(
            title="Test", description="desc", base_branch="main",
        )

        orchestrator_routes._jobs["job-rtp"] = {
            "id": "job-rtp",
            "status": "completed",
            "spec": spec,
            "llm_config": None,
            "agent": MagicMock(),
            "dag": dag,
            "results": [],
            "error": None,
        }

        with patch(
            "bashgym.api.orchestrator_routes._execute_dag",
            new_callable=AsyncMock,
        ):
            resp = client.post(
                "/api/orchestrate/job-rtp/task/t1/retry",
                json={"modified_prompt": "Try a different approach"},
            )

        assert resp.status_code == 200
        assert dag.nodes["t1"].worker_prompt == "Try a different approach"

    def test_retry_nonexistent_task_returns_404(self, client):
        """Retry unknown task → 404."""
        from bashgym.api import orchestrator_routes

        dag = TaskDAG()
        dag.add_task(make_task("t1"))

        orchestrator_routes._jobs["job-rne"] = {
            "id": "job-rne",
            "status": "completed",
            "spec": None,
            "llm_config": None,
            "agent": None,
            "dag": dag,
            "results": [],
            "error": None,
        }

        resp = client.post(
            "/api/orchestrate/job-rne/task/nonexistent/retry",
            json={},
        )
        assert resp.status_code == 404

    def test_retry_non_failed_task_returns_400(self, client):
        """Retry a PENDING task → 400."""
        from bashgym.api import orchestrator_routes

        dag = TaskDAG()
        dag.add_task(make_task("t1"))  # Status is PENDING

        orchestrator_routes._jobs["job-rnf"] = {
            "id": "job-rnf",
            "status": "completed",
            "spec": None,
            "llm_config": None,
            "agent": None,
            "dag": dag,
            "results": [],
            "error": None,
        }

        resp = client.post(
            "/api/orchestrate/job-rnf/task/t1/retry",
            json={},
        )
        assert resp.status_code == 400


# =============================================================================
# Cancel Endpoint
# =============================================================================


class TestCancelEndpoint:
    """Tests for DELETE /api/orchestrate/{job_id}."""

    def test_cancel_running_job(self, client):
        """Cancel a job → status becomes cancelled."""
        from bashgym.api import orchestrator_routes

        mock_pool = MagicMock()
        mock_pool.cancel_all = AsyncMock()
        mock_worktrees = MagicMock()
        mock_worktrees.cleanup_all = AsyncMock()

        agent = MagicMock()
        agent.pool = mock_pool
        agent.worktrees = mock_worktrees

        orchestrator_routes._jobs["job-c"] = {
            "id": "job-c",
            "status": "executing",
            "spec": None,
            "llm_config": None,
            "agent": agent,
            "dag": None,
            "results": [],
            "error": None,
        }

        resp = client.delete("/api/orchestrate/job-c")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        assert data["job_id"] == "job-c"

    def test_cancel_nonexistent_job_returns_404(self, client):
        """Cancel unknown job → 404."""
        resp = client.delete("/api/orchestrate/nonexistent")
        assert resp.status_code == 404


# =============================================================================
# Providers Endpoint
# =============================================================================


class TestProvidersEndpoint:
    """Tests for GET /api/orchestrate/providers."""

    def test_list_providers(self, client):
        """Returns list of available LLM providers."""
        resp = client.get("/api/orchestrate/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        providers = data["providers"]
        assert len(providers) >= 1
        # At minimum, anthropic should be present
        provider_names = [p["provider"] for p in providers]
        assert "anthropic" in provider_names

        # Each provider should have required fields
        for p in providers:
            assert "default_model" in p
            assert "env_key" in p

    def test_providers_include_all_supported(self, client):
        """All 4 providers should be listed."""
        resp = client.get("/api/orchestrate/providers")
        providers = resp.json()["providers"]
        names = {p["provider"] for p in providers}
        assert "anthropic" in names
        assert "openai" in names
        assert "gemini" in names
        assert "ollama" in names


# =============================================================================
# Jobs Endpoint
# =============================================================================


class TestJobsEndpoint:
    """Tests for GET /api/orchestrate/jobs."""

    def test_list_jobs_empty(self, client):
        """No jobs submitted → empty list."""
        resp = client.get("/api/orchestrate/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []

    def test_list_jobs_after_submit(self, client):
        """Submit 2 specs → GET /jobs returns 2 entries."""
        from bashgym.api import orchestrator_routes

        spec1 = OrchestratorSpec(title="Job A", description="Desc A")
        spec2 = OrchestratorSpec(title="Job B", description="Desc B")

        orchestrator_routes._jobs["j1"] = {
            "id": "j1",
            "status": "decomposing",
            "spec": spec1,
            "llm_config": None,
            "agent": None,
            "dag": None,
            "results": [],
            "error": None,
        }
        orchestrator_routes._jobs["j2"] = {
            "id": "j2",
            "status": "awaiting_approval",
            "spec": spec2,
            "llm_config": None,
            "agent": None,
            "dag": TaskDAG(),
            "results": [],
            "error": None,
        }

        resp = client.get("/api/orchestrate/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 2
        titles = {j["title"] for j in data["jobs"]}
        assert "Job A" in titles
        assert "Job B" in titles

    def test_list_jobs_includes_stats_when_dag_exists(self, client):
        """Jobs with a DAG include task_count and stats."""
        from bashgym.api import orchestrator_routes

        dag = TaskDAG()
        dag.add_task(make_task("t1"))
        dag.add_task(make_task("t2"))

        orchestrator_routes._jobs["j1"] = {
            "id": "j1",
            "status": "awaiting_approval",
            "spec": OrchestratorSpec(title="With DAG", description="desc"),
            "llm_config": None,
            "agent": None,
            "dag": dag,
            "results": [],
            "error": None,
        }

        resp = client.get("/api/orchestrate/jobs")
        data = resp.json()
        job = data["jobs"][0]
        assert job["task_count"] == 2
        assert "stats" in job
