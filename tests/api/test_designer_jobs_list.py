# tests/api/test_designer_jobs_list.py
"""Tests for GET /api/factory/designer/jobs (list) endpoint."""

import pytest
from fastapi.testclient import TestClient

from bashgym.api.factory_routes import designer_jobs
from bashgym.api.routes import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_jobs():
    designer_jobs.clear()
    yield
    designer_jobs.clear()


def test_list_empty():
    response = client.get("/api/factory/designer/jobs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_returns_newest_first_with_limit():
    for i in range(3):
        designer_jobs[f"dd_test{i}"] = {
            "status": "completed",
            "pipeline": "coding_agent_sft",
            "num_records": 10 + i,
            "progress": {"current": 10 + i, "total": 10 + i},
        }

    response = client.get("/api/factory/designer/jobs?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert [j["job_id"] for j in data] == ["dd_test2", "dd_test1"]
    assert data[0]["num_records"] == 12
    assert data[0]["status"] == "completed"


def test_list_includes_runtime_identity_and_compute_metadata():
    designer_jobs["dd_metadata"] = {
        "status": "running",
        "pipeline": "coding_agent_sft",
        "num_records": 40,
        "progress": {"current": 0, "total": 40, "phase": "generating"},
        "started_at": "2026-07-09T20:00:00+00:00",
        "config": {
            "seed_source": "data/gold.jsonl",
            "seed_type": "file",
            "provider": "vllm",
            "provider_endpoint": "http://192.0.2.10:8889/v1",
            "text_model": "hermes-qwen3.6-27b-dense",
            "origin": {"kind": "terminal", "terminal_id": "term-42", "agent": "codex"},
            "correlation_id": "designer-intent-42",
        },
    }

    response = client.get("/api/factory/designer/jobs")

    assert response.status_code == 200
    job = response.json()[0]
    assert job["job_name"] == "coding_agent_sft / dd_metadata"
    assert job["dataset"] == "data/gold.jsonl"
    assert job["model"] == "hermes-qwen3.6-27b-dense"
    assert job["provider"] == "vllm"
    assert job["execution"] == "private"
    assert job["origin"] == {"kind": "terminal", "terminal_id": "term-42", "agent": "codex"}
    assert job["correlation_id"] == "designer-intent-42"
    assert job["progress"] == {"current": 0, "total": 40, "phase": "generating"}
