import json
import types

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app


@pytest.fixture
def client():
    return TestClient(app)


def _point_models_dir_at(monkeypatch, tmp_path):
    stub = types.SimpleNamespace(config=types.SimpleNamespace(output_dir=str(tmp_path)))
    monkeypatch.setattr(app.state, "trainer", stub, raising=False)


def test_analysis_404_for_unknown_run(client):
    resp = client.get("/api/training/runs/no-such-run/analysis")
    assert resp.status_code == 404


def test_analysis_returns_verdict_and_findings(client, tmp_path, monkeypatch):
    _point_models_dir_at(monkeypatch, tmp_path)
    run_dir = tmp_path / "run-analyze"
    run_dir.mkdir()
    # An increasing-loss series should surface a health finding.
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for step, loss in enumerate([0.5, 0.9, 1.4], start=1):
            f.write(json.dumps({"step": step, "loss": loss, "grad_norm": 1.0}) + "\n")

    resp = client.get("/api/training/runs/run-analyze/analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert "verdict" in data and "level" in data["verdict"]
    assert isinstance(data["findings"], list)
    assert data["training_metrics"]["points"] == 3
