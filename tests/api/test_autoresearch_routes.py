"""Tests for AutoResearch API routes -- schema research endpoints."""

import json
from types import SimpleNamespace

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
    """Clear AutoResearch route state before each test."""
    for attr in (
        "autoresearcher",
        "schema_researcher",
        "data_recipe_researcher",
        "data_recipe_proposal",
        "data_recipe_output_path",
        "data_recipe_metadata",
    ):
        if hasattr(app.state, attr):
            value = getattr(app.state, attr)
            if hasattr(value, "stop"):
                value.stop()
            delattr(app.state, attr)
    yield
    # Cleanup after test too
    for attr in (
        "autoresearcher",
        "schema_researcher",
        "data_recipe_researcher",
        "data_recipe_proposal",
        "data_recipe_output_path",
        "data_recipe_metadata",
    ):
        if hasattr(app.state, attr):
            value = getattr(app.state, attr)
            if hasattr(value, "stop"):
                value.stop()
            delattr(app.state, attr)


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
        assert data["execution_path"] == "prototype_compatibility"
        assert data["durable"] is False

    def test_real_mode_fails_closed_when_training_data_is_missing(self, monkeypatch, tmp_path):
        settings = SimpleNamespace(
            data=SimpleNamespace(
                gold_traces_dir=str(tmp_path / "missing-gold-traces"),
                training_batches_dir=str(tmp_path / "training-batches"),
            )
        )
        monkeypatch.setattr("bashgym.config.get_settings", lambda: settings)

        response = client.post(
            "/api/autoresearch/start",
            json={"mode": "real", "max_experiments": 1, "train_steps": 10},
        )

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == (
            "autoresearch_real_prerequisites_missing"
        )
        assert not hasattr(app.state, "autoresearcher")

    def test_invalid_mode_is_rejected(self):
        response = client.post(
            "/api/autoresearch/start",
            json={"mode": "maybe-real", "max_experiments": 1, "train_steps": 10},
        )

        assert response.status_code == 422

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


class TestDataRecipeProposal:
    def test_status_when_idle(self):
        response = client.get("/api/autoresearch/data-recipe/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
        assert data["total_experiments"] == 0
        assert data["completed_experiments"] == 0
        assert data["proposal"] is None

    def test_propose_from_training_sources(self):
        response = client.post(
            "/api/autoresearch/data-recipe/propose",
            json={
                "goal": "dpo",
                "source_ids": ["ultrafeedback_binarized", "helpsteer2"],
                "max_experiments": 3,
                "sample_size": 128,
                "mutation_rate": 0.0,
                "seed": 7,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["goal"] == "dpo"
        assert data["source_count"] == 2
        assert data["excluded_sources"] == []
        assert data["best_metric"] is not None
        assert len(data["experiments"]) == 3
        assert data["proposal"]["schema_version"] == "bashgym.data_recipe_proposal.v1"
        assert data["proposal"]["goal"] == "dpo"
        assert data["proposal"]["data_designer"]["pipeline"] == "from_source"
        assert data["proposal"]["data_designer"]["sample_size"] == 128
        assert {source["id"] for source in data["proposal"]["sources"]} == {
            "ultrafeedback_binarized",
            "helpsteer2",
        }

    def test_status_tracks_latest_proposal(self):
        propose_response = client.post(
            "/api/autoresearch/data-recipe/propose",
            json={
                "goal": "dpo",
                "source_ids": ["ultrafeedback_binarized", "helpsteer2"],
                "max_experiments": 2,
                "mutation_rate": 0.0,
                "seed": 11,
            },
        )
        assert propose_response.status_code == 200

        response = client.get("/api/autoresearch/data-recipe/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["goal"] == "dpo"
        assert data["source_count"] == 2
        assert data["completed_experiments"] == 2
        assert data["proposal"]["schema_version"] == "bashgym.data_recipe_proposal.v1"
        assert data["best_config"]["source_weights"]

    def test_propose_blocks_eval_only_training_source_by_default(self):
        response = client.post(
            "/api/autoresearch/data-recipe/propose",
            json={
                "goal": "sft",
                "source_ids": ["harbor_terminal_bench"],
                "max_experiments": 2,
            },
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["message"] == "No source cards remain after data-recipe filters"
        assert detail["excluded"] == [
            {"id": "harbor_terminal_bench", "reason": "eval_only_excluded"}
        ]

    def test_propose_writes_data_recipe_export_file(self, tmp_path):
        output_path = tmp_path / "data-recipe-proposal.json"
        response = client.post(
            "/api/autoresearch/data-recipe/propose",
            json={
                "goal": "reward_model",
                "source_ids": ["ultrafeedback_binarized", "helpsteer2"],
                "max_experiments": 2,
                "mutation_rate": 0.0,
                "output_path": str(output_path),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["output_path"] == str(output_path)
        exported = json.loads(output_path.read_text(encoding="utf-8"))
        assert exported["schema_version"] == "bashgym.data_recipe_proposal.v1"
        assert exported["sources"] == data["proposal"]["sources"]

    def test_export_latest_data_recipe_proposal(self, tmp_path):
        propose_response = client.post(
            "/api/autoresearch/data-recipe/propose",
            json={
                "goal": "dpo",
                "source_ids": ["ultrafeedback_binarized", "helpsteer2"],
                "max_experiments": 2,
                "mutation_rate": 0.0,
            },
        )
        assert propose_response.status_code == 200

        output_path = tmp_path / "latest-data-recipe.json"
        response = client.post(
            "/api/autoresearch/data-recipe/export",
            json={"output_path": str(output_path)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "exported"
        assert data["output_path"] == str(output_path)
        exported = json.loads(output_path.read_text(encoding="utf-8"))
        assert exported == data["proposal"]

    def test_export_without_proposal_returns_404(self):
        response = client.post(
            "/api/autoresearch/data-recipe/export",
            json={"output_path": "unused.json"},
        )

        assert response.status_code == 404

    def test_stop_without_session_returns_404(self):
        response = client.post("/api/autoresearch/data-recipe/stop")

        assert response.status_code == 404

    def test_stop_completed_session_returns_conflict(self):
        propose_response = client.post(
            "/api/autoresearch/data-recipe/propose",
            json={
                "goal": "dpo",
                "source_ids": ["ultrafeedback_binarized", "helpsteer2"],
                "max_experiments": 1,
                "mutation_rate": 0.0,
            },
        )
        assert propose_response.status_code == 200

        response = client.post("/api/autoresearch/data-recipe/stop")

        assert response.status_code == 409
        assert "not running" in response.json()["detail"]

    def test_propose_rejects_unknown_goal(self):
        response = client.post(
            "/api/autoresearch/data-recipe/propose",
            json={"goal": "latest_magic", "max_experiments": 2},
        )

        assert response.status_code == 400
        assert "unknown goal" in response.json()["detail"]


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
