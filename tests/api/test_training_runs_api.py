"""Run-history and dataset-inspection endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app


@pytest.fixture
def client():
    return TestClient(app)


class TestTrainingRuns:
    def test_list_runs_returns_runs_key(self, client):
        resp = client.get("/api/training/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_runs_not_captured_by_run_id_route(self, client):
        """'/training/runs' must hit the literal route, not /training/{run_id}."""
        resp = client.get("/api/training/runs")
        assert resp.status_code == 200
        # The {run_id} route returns TrainingResponse (run_id key) or 404 — neither applies
        assert "runs" in resp.json()

    def test_metrics_404_for_unknown_run(self, client):
        resp = client.get("/api/training/runs/definitely-not-a-run/metrics")
        assert resp.status_code == 404

    def test_metrics_rejects_traversal(self, client):
        resp = client.get("/api/training/runs/..%2Fescape/metrics")
        assert resp.status_code in (400, 404)


class TestDatasetInspect:
    def test_inspect_rejects_path_outside_allowed_roots(self, client):
        resp = client.get(
            "/api/training/dataset/inspect",
            params={"path": "C:/Windows/System32/drivers/etc/hosts"},
        )
        assert resp.status_code == 400

    def test_inspect_missing_dataset_404(self, client, tmp_path):
        # default path may not exist in a clean checkout
        resp = client.get("/api/training/dataset/inspect")
        assert resp.status_code in (200, 404)

    def test_inspect_valid_dataset(self, client, tmp_path, monkeypatch):
        dataset = tmp_path / "train.jsonl"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]
        dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        # Point the configured data dir at tmp_path so the path is allowed
        from bashgym import config as bashgym_config

        settings = bashgym_config.get_settings()
        monkeypatch.setattr(settings.data, "data_dir", str(tmp_path))

        resp = client.get("/api/training/dataset/inspect", params={"path": str(dataset)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["examples"][0]["warnings"] == []
