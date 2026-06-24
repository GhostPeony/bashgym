"""Tests for AutoResearch API routes -- schema research endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app

client = TestClient(app)


def _environment_payload(
    env_id: str,
    *,
    domain: str,
    skills: list[str],
    verifier_kind: str,
    fixture_kind: str,
    pass_at_1: float,
) -> dict:
    return {
        "id": env_id,
        "instruction": f"Complete {env_id}",
        "source": "test",
        "domain": domain,
        "skills": skills,
        "axes": [
            {"name": "task_complexity", "value": "moderate"},
            {"name": "command_complexity", "value": "single"},
            {"name": "language", "value": "python"},
            {"name": "fixture_kind", "value": fixture_kind},
        ],
        "fixtures": [{"path": f"{env_id}.txt", "kind": fixture_kind}],
        "verifier": {"kind": verifier_kind, "command": "./verify.sh", "path": "verify.sh"},
        "metadata": {"pass@1": pass_at_1},
    }


def _environment_payloads() -> list[dict]:
    return [
        _environment_payload(
            "env_file_py",
            domain="file_ops",
            skills=["edit"],
            verifier_kind="pytest",
            fixture_kind="file",
            pass_at_1=0.25,
        ),
        _environment_payload(
            "env_shell_logs",
            domain="bash",
            skills=["search"],
            verifier_kind="script",
            fixture_kind="files",
            pass_at_1=0.4,
        ),
        _environment_payload(
            "env_repo_fix",
            domain="repo",
            skills=["debug"],
            verifier_kind="pytest",
            fixture_kind="repo",
            pass_at_1=0.3,
        ),
        _environment_payload(
            "env_service",
            domain="service",
            skills=["http"],
            verifier_kind="unit",
            fixture_kind="service",
            pass_at_1=0.5,
        ),
    ]


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


class TestEnvironmentRecipeProposal:
    def test_propose_from_inline_environments(self):
        response = client.post(
            "/api/autoresearch/environment-recipe/propose",
            json={
                "environments": _environment_payloads(),
                "max_experiments": 3,
                "sample_size": 3,
                "pass_at_1_target": 0.35,
                "seed": 7,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["source_count"] == 4
        assert data["best_metric"] is not None
        assert len(data["experiments"]) == 3
        assert data["proposal"]["schema_version"] == "bashgym.environment_recipe_proposal.v1"
        assert data["proposal"]["selected_count"] == 3
        assert len(data["proposal"]["selected_environment_ids"]) == 3
        assert "domain" in data["proposal"]["mix_report"]["axis_balance"]

    def test_propose_writes_export_file(self, tmp_path):
        output_path = tmp_path / "recipe-proposal.json"
        response = client.post(
            "/api/autoresearch/environment-recipe/propose",
            json={
                "environments": _environment_payloads(),
                "max_experiments": 2,
                "sample_size": 2,
                "output_path": str(output_path),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["output_path"] == str(output_path)
        exported = json.loads(output_path.read_text(encoding="utf-8"))
        assert exported["schema_version"] == "bashgym.environment_recipe_proposal.v1"
        assert exported["selected_environment_ids"] == data["proposal"]["selected_environment_ids"]

    def test_propose_requires_environment_source(self):
        response = client.post(
            "/api/autoresearch/environment-recipe/propose",
            json={"environments": [], "max_experiments": 2},
        )

        assert response.status_code == 400
