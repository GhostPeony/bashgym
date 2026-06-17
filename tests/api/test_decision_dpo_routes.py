"""Tests for the decision-DPO mining routes (data-quality toggles).

Hermetic: ``DataFactory.process_trace_directory`` and ``save_dpo_batch`` are
stubbed, so the FastAPI ``BackgroundTasks`` worker (run synchronously by the
TestClient) never hits the LLM/network or writes a batch file.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bashgym.api.routes import app
from bashgym.factory.data_factory import DataFactory, DPOExample, TrainingExample

client = TestClient(app)


def _training(n: int) -> list[TrainingExample]:
    return [
        TrainingExample(
            example_id=f"t{i}", system_prompt="s", user_prompt="u", assistant_response="a"
        )
        for i in range(n)
    ]


def _dpo(n: int) -> list[DPOExample]:
    return [
        DPOExample(example_id=f"d{i}", prompt="p", chosen="good", rejected="bad") for i in range(n)
    ]


def test_data_quality_defaults():
    r = client.get("/api/factory/data-quality/defaults")
    assert r.status_code == 200
    data = r.json()
    assert data["generate_decision_dpo"] is True
    assert data["require_successful_verification"] is True
    assert data["min_trace_steps"] >= 1
    assert data["max_trace_steps"] > data["min_trace_steps"]


def test_generate_bad_dir(tmp_path):
    r = client.post(
        "/api/factory/decision-dpo/generate",
        json={"gold_dir": str(tmp_path / "nope")},
    )
    assert r.status_code == 400


def test_generate_bad_step_bounds(tmp_path):
    r = client.post(
        "/api/factory/decision-dpo/generate",
        json={"gold_dir": str(tmp_path), "min_trace_steps": 10, "max_trace_steps": 5},
    )
    assert r.status_code == 400


def test_generate_runs_and_counts(tmp_path, monkeypatch):
    captured = {}

    async def fake_process(self, gold_dir, failed_dir=None):
        captured["gold_dir"] = str(gold_dir)
        captured["min"] = self.config.min_trace_steps
        captured["verif"] = self.config.require_successful_verification
        return _training(4), _dpo(3)

    monkeypatch.setattr(DataFactory, "process_trace_directory", fake_process)
    monkeypatch.setattr(
        DataFactory, "save_dpo_batch", lambda self, dpo, name="dpo": Path(f"out/{name}.jsonl")
    )

    r = client.post(
        "/api/factory/decision-dpo/generate",
        json={
            "gold_dir": str(tmp_path),
            "min_trace_steps": 3,
            "require_successful_verification": False,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    got = client.get(f"/api/factory/decision-dpo/jobs/{job_id}")
    data = got.json()
    assert data["status"] == "completed"
    assert data["n_training_examples"] == 4
    assert data["n_dpo_pairs"] == 3
    assert data["dpo_output_path"] == str(Path("out/decision_dpo.jsonl"))
    # quality toggles threaded into the factory config
    assert captured["min"] == 3
    assert captured["verif"] is False


def test_generate_no_pairs_skips_export(tmp_path, monkeypatch):
    async def fake_process(self, gold_dir, failed_dir=None):
        return _training(2), []

    def boom_save(self, dpo, name="dpo"):
        raise AssertionError("save_dpo_batch should not run when there are no pairs")

    monkeypatch.setattr(DataFactory, "process_trace_directory", fake_process)
    monkeypatch.setattr(DataFactory, "save_dpo_batch", boom_save)

    r = client.post("/api/factory/decision-dpo/generate", json={"gold_dir": str(tmp_path)})
    job_id = r.json()["job_id"]
    data = client.get(f"/api/factory/decision-dpo/jobs/{job_id}").json()
    assert data["status"] == "completed"
    assert data["n_dpo_pairs"] == 0
    assert data["dpo_output_path"] is None


def test_generate_failure_captured(tmp_path, monkeypatch):
    async def boom(self, gold_dir, failed_dir=None):
        raise RuntimeError("extractor blew up")

    monkeypatch.setattr(DataFactory, "process_trace_directory", boom)
    r = client.post("/api/factory/decision-dpo/generate", json={"gold_dir": str(tmp_path)})
    job_id = r.json()["job_id"]
    data = client.get(f"/api/factory/decision-dpo/jobs/{job_id}").json()
    assert data["status"] == "failed"
    assert "extractor blew up" in data["error"]


def test_status_missing_job():
    assert client.get("/api/factory/decision-dpo/jobs/nope").status_code == 404
