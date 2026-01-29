# tests/api/test_factory_routes.py
"""Tests for synthetic data generation API endpoints."""

import pytest
from fastapi.testclient import TestClient
from bashgym.api.routes import app

client = TestClient(app)


class TestSyntheticGenerateEndpoint:
    """Tests for POST /api/factory/synthetic/generate endpoint."""

    def test_post_synthetic_generate(self):
        """Should accept synthetic generation request and return job ID."""
        response = client.post("/api/factory/synthetic/generate", json={
            "strategy": "trace_seeded",
            "repo_filter": "single",
            "selected_repos": ["ghostwork"],
            "preset": "quick_test",
            "provider": "nim"
        })

        assert response.status_code in [200, 202]
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "queued"
        assert data["job_id"].startswith("gen_")

    def test_post_synthetic_generate_default_values(self):
        """Should use default values when not provided."""
        response = client.post("/api/factory/synthetic/generate", json={})

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_post_synthetic_generate_augmented_strategy(self):
        """Should accept augmented strategy."""
        response = client.post("/api/factory/synthetic/generate", json={
            "strategy": "augmented",
            "preset": "balanced"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

    def test_post_synthetic_generate_schema_driven_strategy(self):
        """Should accept schema_driven strategy."""
        response = client.post("/api/factory/synthetic/generate", json={
            "strategy": "schema_driven",
            "preset": "quick_test"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"


class TestJobStatusEndpoint:
    """Tests for GET /api/factory/synthetic/jobs/{job_id} endpoint."""

    def test_get_job_status(self):
        """Should return job status for existing job."""
        # First create a job
        create_response = client.post("/api/factory/synthetic/generate", json={
            "preset": "quick_test"
        })
        job_id = create_response.json()["job_id"]

        # Then check status
        response = client.get(f"/api/factory/synthetic/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress" in data

    def test_get_job_status_not_found(self):
        """Should return 404 for non-existent job."""
        response = client.get("/api/factory/synthetic/jobs/gen_nonexistent")

        assert response.status_code == 404


class TestListJobsEndpoint:
    """Tests for GET /api/factory/synthetic/jobs endpoint."""

    def test_list_jobs(self):
        """Should list all jobs."""
        # Create a job first
        client.post("/api/factory/synthetic/generate", json={
            "preset": "quick_test"
        })

        response = client.get("/api/factory/synthetic/jobs")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1


class TestPresetsEndpoint:
    """Tests for GET /api/factory/synthetic/presets endpoint."""

    def test_get_presets(self):
        """Should return available presets."""
        response = client.get("/api/factory/synthetic/presets")

        assert response.status_code == 200
        data = response.json()

        # Check expected presets exist
        assert "quick_test" in data
        assert "balanced" in data
        assert "production" in data
        assert "custom" in data

        # Check preset structure
        assert data["quick_test"]["target_examples"] == 100
        assert data["balanced"]["target_examples"] == 500
        assert data["production"]["target_examples"] == 2000
