"""Tests for AutoResearch API routes -- schema research endpoints."""

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_schema_researcher():
    """Clear schema_researcher from app state before each test."""
    if hasattr(app.state, "schema_researcher"):
        # Stop any running researcher first
        researcher = app.state.schema_researcher
        if hasattr(researcher, "stop"):
            researcher.stop()
        delattr(app.state, "schema_researcher")
    yield
    # Cleanup after test too
    if hasattr(app.state, "schema_researcher"):
        researcher = app.state.schema_researcher
        if hasattr(researcher, "stop"):
            researcher.stop()
        delattr(app.state, "schema_researcher")


class TestSchemaResearchStart:
    def test_start_schema_research(self):
        response = client.post(
            "/api/autoresearch/schema-research/start",
            json={
                "base_template": "coding_agent_sft",
                "max_experiments": 5,
                "mode": "simulate",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["template"] == "coding_agent_sft"
        assert data["max_experiments"] == 5

    def test_start_with_defaults(self):
        response = client.post(
            "/api/autoresearch/schema-research/start",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["template"] == "coding_agent_sft"

    def test_start_with_custom_template(self):
        response = client.post(
            "/api/autoresearch/schema-research/start",
            json={
                "base_template": "tool_use_sft",
                "max_experiments": 3,
                "mode": "simulate",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["template"] == "tool_use_sft"

    def test_start_with_custom_mutation_params(self):
        response = client.post(
            "/api/autoresearch/schema-research/start",
            json={
                "mutation_rate": 0.5,
                "mutation_scale": 0.3,
                "stage1_examples": 50,
                "stage2_train_steps": 100,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"


class TestSchemaResearchStatus:
    def test_status_when_idle(self):
        response = client.get("/api/autoresearch/schema-research/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
        assert data["total_experiments"] == 0


class TestSchemaResearchQuality:
    def test_quality_no_experiments(self):
        response = client.get("/api/autoresearch/schema-research/quality")
        assert response.status_code == 200
        data = response.json()
        assert data["experiments_count"] == 0
        assert data["improvements"] == 0
        assert data["best_metric"] is None

    def test_quality_has_expected_fields(self):
        response = client.get("/api/autoresearch/schema-research/quality")
        assert response.status_code == 200
        data = response.json()
        assert "experiments_count" in data
        assert "improvements" in data
        assert "best_metric" in data
        assert "score_distribution" in data
        assert "template" in data


class TestSchemaResearchControls:
    def test_stop_no_session(self):
        response = client.post("/api/autoresearch/schema-research/stop")
        assert response.status_code == 404

    def test_pause_no_session(self):
        response = client.post("/api/autoresearch/schema-research/pause")
        assert response.status_code == 404

    def test_resume_no_session(self):
        response = client.post("/api/autoresearch/schema-research/resume")
        assert response.status_code == 404


class TestHyperparamResearchStatus:
    def test_status_when_idle(self):
        # Clear any existing session
        if hasattr(app.state, "autoresearcher"):
            delattr(app.state, "autoresearcher")

        response = client.get("/api/autoresearch/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
        assert data["total_experiments"] == 0
        assert data["completed_experiments"] == 0

    def test_stop_no_session(self):
        if hasattr(app.state, "autoresearcher"):
            delattr(app.state, "autoresearcher")

        response = client.post("/api/autoresearch/stop")
        assert response.status_code == 404

    def test_pause_no_session(self):
        if hasattr(app.state, "autoresearcher"):
            delattr(app.state, "autoresearcher")

        response = client.post("/api/autoresearch/pause")
        assert response.status_code == 404
