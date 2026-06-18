"""Tests for the advanced eval routes (held-out gate + benchmark ingest).

Hermetic: ``service.run_heldout`` and the model registry are stubbed, and the job
store is reset per-test so the FastAPI ``BackgroundTasks`` worker (which the
TestClient runs synchronously) never touches the network or disk.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bashgym.api import eval_routes
from bashgym.api.routes import app

client = TestClient(app)


class _FakeReport:
    def __init__(self, ship=True, delta=0.42):
        self._ship = ship
        self._delta = delta

    def to_dict(self) -> dict:
        return {
            "ship": self._ship,
            "trace_delta": self._delta,
            "base_pass_rate": 0.3,
            "candidate_pass_rate": 0.3 + self._delta,
            "metric": "exact_match",
            "reasons": [] if self._ship else ["trace delta too small"],
        }


class _FakeProfile:
    display_name = "candidate-sft"
    heldout_evals = [{"ship": True, "trace_delta": 0.42}]

    @property
    def latest_heldout_eval(self):
        return self.heldout_evals[-1]


class _FakeRegistry:
    def __init__(self, profile=None):
        self._profile = profile
        self.recorded: list[tuple] = []
        self.benchmarks: list[tuple] = []

    def get(self, model_id):
        return self._profile

    def record_heldout_eval(self, model_id, report):
        self.recorded.append((model_id, report))
        return self._profile

    def add_benchmark_result(self, model_id, name, score, passed, total, metrics):
        self.benchmarks.append((model_id, name, score))
        return object()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Fresh job store, no disk writes, per test."""
    app.state.heldout_jobs = {}
    monkeypatch.setattr(eval_routes, "_save_jobs", lambda jobs: None)
    yield


def _dataset(tmp_path):
    p = tmp_path / "holdout.jsonl"
    p.write_text(
        '{"messages": [{"role": "user", "content": "hi"}, '
        '{"role": "assistant", "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}]}], '
        '"metadata": {"session_id": "s1"}}\n',
        encoding="utf-8",
    )
    return str(p)


def _body(tmp_path, **over):
    body = {
        "model_id": "candidate-sft",
        "dataset_path": _dataset(tmp_path),
        "candidate": {"base_url": "http://h/v1", "model": "candidate"},
        "base": {"base_url": "http://h/v1", "model": "base"},
        "metric": "exact_match",
    }
    body.update(over)
    return body


# ── validation ───────────────────────────────────────────────────────────────


def test_heldout_bad_metric(tmp_path):
    r = client.post("/api/eval/heldout", json=_body(tmp_path, metric="bogus"))
    assert r.status_code == 400


def test_heldout_missing_dataset(tmp_path):
    body = _body(tmp_path)
    body["dataset_path"] = str(tmp_path / "nope.jsonl")
    r = client.post("/api/eval/heldout", json=body)
    assert r.status_code == 400


def test_heldout_unresolvable_endpoint(tmp_path):
    body = _body(tmp_path)
    body["base"] = {"model": "base"}  # no provider, no base_url
    r = client.post("/api/eval/heldout", json=body)
    assert r.status_code == 400
    assert "base endpoint" in r.json()["detail"]


# ── run + poll ───────────────────────────────────────────────────────────────


def test_heldout_runs_records_and_polls(tmp_path, monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    monkeypatch.setattr(eval_routes.service, "run_heldout", lambda *a, **k: _FakeReport(ship=True))

    r = client.post("/api/eval/heldout", json=_body(tmp_path))
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # background task already ran (TestClient runs it synchronously)
    got = client.get(f"/api/eval/heldout/{job_id}")
    assert got.status_code == 200
    data = got.json()
    assert data["status"] == "completed"
    assert data["report"]["ship"] is True
    assert data["report"]["trace_delta"] == pytest.approx(0.42)
    # verdict recorded to the registry
    assert fake_reg.recorded and fake_reg.recorded[0][0] == "candidate-sft"


def test_heldout_job_failure_is_captured(tmp_path, monkeypatch):
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: _FakeRegistry())

    def boom(*a, **k):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(eval_routes.service, "run_heldout", boom)
    r = client.post("/api/eval/heldout", json=_body(tmp_path))
    job_id = r.json()["job_id"]
    data = client.get(f"/api/eval/heldout/{job_id}").json()
    assert data["status"] == "failed"
    assert "endpoint down" in data["error"]


def test_heldout_list(tmp_path, monkeypatch):
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: _FakeRegistry())
    monkeypatch.setattr(eval_routes.service, "run_heldout", lambda *a, **k: _FakeReport())
    client.post("/api/eval/heldout", json=_body(tmp_path))
    client.post("/api/eval/heldout", json=_body(tmp_path))
    r = client.get("/api/eval/heldout?limit=5")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_heldout_get_missing():
    assert client.get("/api/eval/heldout/nope").status_code == 404


# ── verdict ──────────────────────────────────────────────────────────────────


def test_verdict_returns_latest(monkeypatch):
    monkeypatch.setattr(
        "bashgym.models.get_registry", lambda *a, **k: _FakeRegistry(_FakeProfile())
    )
    r = client.get("/api/eval/verdict/candidate-sft")
    assert r.status_code == 200
    data = r.json()
    assert data["latest_heldout_eval"]["ship"] is True
    assert data["n_heldout_evals"] == 1


def test_verdict_not_found(monkeypatch):
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: _FakeRegistry(None))
    assert client.get("/api/eval/verdict/ghost").status_code == 404


# ── benchmark commands + ingest ──────────────────────────────────────────────


def test_benchmark_commands():
    r = client.get(
        "/api/eval/benchmark-commands",
        params={"base_url": "http://h/v1", "model": "m", "include": "forgetting,bfcl"},
    )
    assert r.status_code == 200
    cmds = r.json()["commands"]
    assert set(cmds) == {"forgetting", "bfcl"}
    assert "http://h/v1" in " ".join(cmds["forgetting"])


def test_ingest_benchmarks_records(monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    r = client.post(
        "/api/eval/benchmarks/ingest",
        json={
            "model_id": "candidate-sft",
            "base_results": {"results": {"mmlu": {"acc,none": 0.70}}},
            "candidate_results": {"results": {"mmlu": {"acc,none": 0.62}}},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["forgetting"]["drops"]["mmlu"] == pytest.approx(0.08)
    assert data["forgetting_ok"] is False  # 0.08 > 0.05 default
    assert data["recorded"] == ["mmlu"]
    assert fake_reg.benchmarks == [("candidate-sft", "mmlu", pytest.approx(0.62))]
