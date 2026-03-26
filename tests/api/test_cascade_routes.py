"""Tests for Cascade RL API routes."""

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_cascade_state():
    """Clean cascade state between tests."""
    app.state.cascade_scheduler = None
    app.state.cascade_result = None
    app.state.mopd_result = None
    yield
    app.state.cascade_scheduler = None
    app.state.cascade_result = None
    app.state.mopd_result = None


class TestCascadeStart:
    def test_start_cascade(self):
        response = client.post(
            "/api/cascade/start",
            json={
                "domains": ["file_operations", "bash_commands"],
                "mode": "simulate",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["stages"] == 2

    def test_start_with_defaults(self):
        response = client.post("/api/cascade/start", json={"mode": "simulate"})
        assert response.status_code == 200
        data = response.json()
        assert data["stages"] == 4  # All 4 domains


class TestCascadeStatus:
    def test_status_idle(self):
        response = client.get("/api/cascade/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"

    def test_status_after_start(self):
        client.post("/api/cascade/start", json={"mode": "simulate"})
        response = client.get("/api/cascade/status")
        assert response.status_code == 200
        data = response.json()
        assert "stages" in data


class TestCascadeStop:
    def test_stop_no_session(self):
        response = client.post("/api/cascade/stop")
        assert response.status_code == 404

    def test_stop_not_running(self):
        # Start then it may auto-complete in simulate mode
        client.post("/api/cascade/start", json={"mode": "simulate"})
        response = client.post("/api/cascade/stop")
        # Either 200 (was running) or 409 (already completed)
        assert response.status_code in [200, 409]


class TestCascadeDistill:
    def test_distill_no_cascade(self):
        response = client.post("/api/cascade/distill", json={})
        assert response.status_code == 404

    def test_distill_status_idle(self):
        response = client.get("/api/cascade/distill/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
