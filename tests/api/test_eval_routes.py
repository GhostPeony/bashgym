"""Tests for the advanced eval routes (held-out gate + benchmark ingest).

Hermetic: ``service.run_heldout`` and the model registry are stubbed, and the job
store is reset per-test so the FastAPI ``BackgroundTasks`` worker (which the
TestClient runs synchronously) never touches the network or disk.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from bashgym.api import eval_routes
from bashgym.api.routes import app

client = TestClient(app)


def _py(code: str) -> str:
    return subprocess.list2cmdline([sys.executable, "-c", code])


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
    environment_holdout_evals = [{"gate": {"ship": True}, "contamination": []}]

    @property
    def latest_heldout_eval(self):
        return self.heldout_evals[-1]

    @property
    def latest_environment_holdout_eval(self):
        return self.environment_holdout_evals[-1]


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

    def record_environment_holdout_eval(self, model_id, result):
        self.recorded.append((model_id, result))
        return self._profile or object()

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


def test_heldout_environment_evidence_blocks_and_records_combined_verdict(
    tmp_path, monkeypatch
):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    monkeypatch.setattr(eval_routes.service, "run_heldout", lambda *a, **k: _FakeReport(ship=True))

    r = client.post(
        "/api/eval/heldout",
        json=_body(
            tmp_path,
            environment_evidence={
                "required": True,
                "holdout_gate": {
                    "result": {
                        "gate": {
                            "ship": False,
                            "reasons": ["pass@1 0.100 < required 0.500"],
                        }
                    }
                },
            },
        ),
    )

    assert r.status_code == 200
    job_id = r.json()["job_id"]
    data = client.get(f"/api/eval/heldout/{job_id}").json()
    report = data["report"]
    assert data["status"] == "completed"
    assert report["ship"] is False
    assert report["release_gate"]["trace_ship"] is True
    assert report["release_gate"]["environment_ship"] is False
    assert report["release_gate"]["environment_sections"] == ["holdout_gate"]
    assert report["reasons"] == ["environment holdout: pass@1 0.100 < required 0.500"]
    assert fake_reg.recorded == [("candidate-sft", report)]


def test_heldout_required_environment_evidence_blocks_when_missing(tmp_path, monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    monkeypatch.setattr(eval_routes.service, "run_heldout", lambda *a, **k: _FakeReport(ship=True))

    r = client.post(
        "/api/eval/heldout",
        json=_body(tmp_path, environment_evidence={"required": True}),
    )

    assert r.status_code == 200
    report = client.get(f"/api/eval/heldout/{r.json()['job_id']}").json()["report"]
    assert report["ship"] is False
    assert report["release_gate"]["environment_required"] is True
    assert report["release_gate"]["environment_sections"] == []
    assert report["reasons"] == ["environment gate evidence required but missing"]
    assert fake_reg.recorded == [("candidate-sft", report)]


def test_heldout_external_benchmark_evidence_blocks_and_records(tmp_path, monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    monkeypatch.setattr(eval_routes.service, "run_heldout", lambda *a, **k: _FakeReport(ship=True))

    r = client.post(
        "/api/eval/heldout",
        json=_body(
            tmp_path,
            environment_evidence={
                "external_benchmarks": {
                    "report": {
                        "scores": {"harbor_terminal_bench": 0.42},
                        "failures": [],
                        "results": [],
                    },
                    "manifest": {"dataset": "terminal-bench@2.0"},
                },
                "external_benchmark_min_scores": {"harbor_terminal_bench": 0.5},
            },
        ),
    )

    assert r.status_code == 200
    report = client.get(f"/api/eval/heldout/{r.json()['job_id']}").json()["report"]
    assert report["ship"] is False
    assert report["release_gate"]["trace_ship"] is True
    assert report["release_gate"]["environment_ship"] is True
    assert report["release_gate"]["external_benchmark_ship"] is False
    assert report["release_gate"]["external_benchmark_sections"] == ["external_benchmarks"]
    assert report["reasons"] == [
        "external benchmark harbor_terminal_bench: score 0.420 < required 0.500"
    ]
    assert report["environment_evidence"]["external_benchmarks"]["manifest"] == {
        "dataset": "terminal-bench@2.0"
    }
    assert fake_reg.recorded == [("candidate-sft", report)]


def test_heldout_world_model_quality_evidence_records_diagnostic(tmp_path, monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    monkeypatch.setattr(eval_routes.service, "run_heldout", lambda *a, **k: _FakeReport(ship=True))

    r = client.post(
        "/api/eval/heldout",
        json=_body(
            tmp_path,
            environment_evidence={
                "world_model_quality": {
                    "training_metrics": {
                        "echo_loss": {"first": 1.1, "last": 0.7},
                        "rwml_pass_rate": {"last": 0.64},
                    },
                    "coverage": {"world_model_records": 8},
                }
            },
        ),
    )

    assert r.status_code == 200
    report = client.get(f"/api/eval/heldout/{r.json()['job_id']}").json()["report"]
    quality = report["release_gate"]["world_model_quality"]
    assert report["ship"] is True
    assert report["release_gate"]["world_model_quality_present"] is True
    assert report["release_gate"]["world_model_quality_sections"] == ["world_model_quality"]
    assert quality["signal"] == "improving"
    assert quality["metrics"]["rwml_pass_rate"] == pytest.approx(0.64)
    assert report["environment_evidence"]["world_model_quality"]["coverage"] == {
        "world_model_records": 8
    }
    assert fake_reg.recorded == [("candidate-sft", report)]


def test_heldout_learned_reward_evidence_records_diagnostic(tmp_path, monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    monkeypatch.setattr(eval_routes.service, "run_heldout", lambda *a, **k: _FakeReport(ship=True))

    r = client.post(
        "/api/eval/heldout",
        json=_body(
            tmp_path,
            environment_evidence={
                "learned_reward_evidence": {
                    "schema_version": "bashgym.reward_model_eval.v1",
                    "ok": True,
                    "metrics": {
                        "heldout_pair_accuracy": 0.81,
                        "calibration_error": 0.07,
                        "reward_variance": 0.04,
                        "eval_only_leakage_count": 0,
                    },
                    "findings": [],
                }
            },
        ),
    )

    assert r.status_code == 200
    report = client.get(f"/api/eval/heldout/{r.json()['job_id']}").json()["report"]
    reward = report["release_gate"]["learned_reward_evidence"]
    assert report["ship"] is True
    assert report["release_gate"]["learned_reward_evidence_present"] is True
    assert report["release_gate"]["learned_reward_evidence_sections"] == [
        "learned_reward_evidence"
    ]
    assert reward["signal"] == "healthy"
    assert reward["metrics"]["heldout_pair_accuracy"] == pytest.approx(0.81)
    assert report["environment_evidence"]["learned_reward_evidence"]["ok"] is True
    assert fake_reg.recorded == [("candidate-sft", report)]


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
    assert data["latest_environment_holdout_eval"]["gate"]["ship"] is True
    assert data["n_environment_holdout_evals"] == 1


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


def test_benchmark_commands_default_includes_harbor_terminal_bench():
    r = client.get(
        "/api/eval/benchmark-commands",
        params={"base_url": "http://h/v1", "model": "m"},
    )

    assert r.status_code == 200
    cmds = r.json()["commands"]
    assert "terminal_bench" in cmds
    assert "harbor_terminal_bench" in cmds
    assert cmds["harbor_terminal_bench"][:2] == ["harbor", "run"]


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


def test_ingest_external_benchmarks_records(monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    r = client.post(
        "/api/eval/benchmarks/external-ingest",
        json={
            "model_id": "candidate-sft",
            "benchmark_name": "harbor_terminal_bench",
            "results": {
                "trials": [
                    {"result": {"reward": 1.0}},
                    {"result": {"reward": 0.0}},
                    {"result": {"reward": 1.0}},
                ]
            },
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert data["report"]["scores"]["harbor_terminal_bench"] == pytest.approx(2 / 3)
    assert data["recorded"] == ["harbor_terminal_bench"]
    assert fake_reg.benchmarks == [
        ("candidate-sft", "harbor_terminal_bench", pytest.approx(2 / 3))
    ]


def test_ingest_external_rewardbench_records(monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    r = client.post(
        "/api/eval/benchmarks/external-ingest",
        json={
            "model_id": "candidate-rm",
            "benchmark_name": "rewardbench",
            "results": {
                "subsets": {
                    "Chat": {"accuracy": 0.75, "num_correct": 3, "total": 4},
                    "Safety": {"accuracy": 0.5, "num_correct": 1, "total": 2},
                }
            },
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert data["report"]["scores"]["rewardbench"] == pytest.approx(0.625)
    assert data["recorded"] == ["rewardbench"]
    assert fake_reg.benchmarks == [("candidate-rm", "rewardbench", pytest.approx(0.625))]


# ── environment pass@k ───────────────────────────────────────────────────────


def _env_passk_body(**over):
    body = {
        "model_id": "candidate-sft",
        "environments": [
            {"id": "env_a", "instruction": "Task A"},
            {"id": "env_b", "instruction": "Task B"},
        ],
        "k_values": [1, 2],
        "attempts": [
            {"environment_id": "env_a", "attempt_index": 0, "passed": True},
            {"environment_id": "env_a", "attempt_index": 1, "passed": False},
            {"environment_id": "env_b", "attempt_index": 0, "passed": False},
            {"environment_id": "env_b", "attempt_index": 1, "passed": False},
        ],
    }
    body.update(over)
    return body


def test_environment_passk_records(monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)

    r = client.post("/api/eval/environments/passk", json=_env_passk_body())

    assert r.status_code == 200
    data = r.json()
    assert data["report"]["pass_at_k"]["pass@1"] == pytest.approx(0.25)
    assert data["report"]["pass_at_k"]["pass@2"] == pytest.approx(0.5)
    assert data["recorded"] == ["bashgym_env_pass@1", "bashgym_env_pass@2"]
    assert [row[1] for row in fake_reg.benchmarks] == [
        "bashgym_env_pass@1",
        "bashgym_env_pass@2",
    ]


def test_environment_passk_rejects_bad_k():
    r = client.post("/api/eval/environments/passk", json=_env_passk_body(k_values=[0]))

    assert r.status_code == 400
    assert "positive" in r.json()["detail"]


def test_environment_passk_rejects_unknown_attempt_env():
    body = _env_passk_body(record_to_registry=False)
    body["attempts"][0]["environment_id"] = "missing"

    r = client.post("/api/eval/environments/passk", json=body)

    assert r.status_code == 400
    assert "unknown environments" in r.json()["detail"]


def test_environment_holdout_gate_reports_split_records_and_verdict(monkeypatch):
    fake_reg = _FakeRegistry()
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)
    body = {
        "model_id": "candidate-sft",
        "environments": [
            {"id": "env_a", "instruction": "Task A", "metadata": {"task_family": "alpha"}},
            {"id": "env_b", "instruction": "Task B", "metadata": {"task_family": "beta"}},
            {"id": "env_c", "instruction": "Task C", "metadata": {"task_family": "gamma"}},
            {"id": "env_d", "instruction": "Task D", "metadata": {"task_family": "delta"}},
        ],
        "attempts": [
            {"environment_id": env_id, "attempt_index": 0, "passed": True, "verifier_status": "passed"}
            for env_id in ["env_a", "env_b", "env_c", "env_d"]
        ],
        "split_by": "task_family",
        "holdout_fraction": 0.5,
        "seed": 3,
        "k_values": [1],
        "min_pass_at_1": 0.5,
        "record_to_registry": True,
    }

    r = client.post("/api/eval/environments/holdout-gate", json=body)

    assert r.status_code == 200
    result = r.json()["result"]
    assert result["schema_version"] == "bashgym.environment_holdout.v1"
    assert result["split"]["split_by"] == "task_family"
    assert result["report"]["n_environments"] == 2
    assert result["gate"]["ship"] is True
    assert r.json()["recorded"] == ["bashgym_env_holdout_gate", "bashgym_env_holdout_pass@1"]
    assert fake_reg.recorded[0][0] == "candidate-sft"
    assert fake_reg.recorded[0][1]["gate"]["ship"] is True
    assert fake_reg.benchmarks == [
        ("candidate-sft", "bashgym_env_holdout_pass@1", pytest.approx(1.0))
    ]


def test_environment_holdout_gate_rejects_bad_split_key():
    body = _env_passk_body(record_to_registry=False)
    body.update({"split_by": "bad", "holdout_fraction": 0.5, "k_values": [1]})

    r = client.post("/api/eval/environments/holdout-gate", json=body)

    assert r.status_code == 400
    assert "split_by must be one of" in r.json()["detail"]


def test_environment_holdout_comparison_reports_bootstrap_gate():
    body = {
        "environments": [
            {"id": "env_a", "instruction": "Task A", "metadata": {"task_family": "alpha"}},
            {"id": "env_b", "instruction": "Task B", "metadata": {"task_family": "beta"}},
            {"id": "env_c", "instruction": "Task C", "metadata": {"task_family": "gamma"}},
            {"id": "env_d", "instruction": "Task D", "metadata": {"task_family": "delta"}},
        ],
        "base_attempts": [
            {"environment_id": env_id, "attempt_index": 0, "passed": False, "verifier_status": "failed"}
            for env_id in ["env_a", "env_b", "env_c", "env_d"]
        ],
        "candidate_attempts": [
            {"environment_id": env_id, "attempt_index": 0, "passed": True, "verifier_status": "passed"}
            for env_id in ["env_a", "env_b", "env_c", "env_d"]
        ],
        "split_by": "task_family",
        "cluster_by": "task_family",
        "holdout_fraction": 0.5,
        "seed": 3,
        "k_values": [1],
        "compare_k": 1,
        "min_delta": 0.5,
        "n_resamples": 50,
    }

    r = client.post("/api/eval/environments/holdout-comparison", json=body)

    assert r.status_code == 200
    result = r.json()["result"]
    assert result["schema_version"] == "bashgym.environment_holdout_comparison.v1"
    assert result["compare_metric"] == "pass@1"
    assert result["bootstrap"]["mean"] == pytest.approx(1.0)
    assert result["bootstrap"]["better"] is True
    assert result["gate"]["ship"] is True


def test_environment_holdout_comparison_rejects_bad_k():
    body = {
        "environments": [{"id": "env_a", "instruction": "Task A"}],
        "base_attempts": [{"environment_id": "env_a", "attempt_index": 0, "passed": False}],
        "candidate_attempts": [{"environment_id": "env_a", "attempt_index": 0, "passed": True}],
        "holdout_fraction": 0.5,
        "k_values": [0],
    }

    r = client.post("/api/eval/environments/holdout-comparison", json=body)

    assert r.status_code == 400
    assert "positive" in r.json()["detail"]


def test_environment_spurious_reward_control_reports_negative_control_gate():
    body = {
        "environments": [
            {"id": "env_a", "instruction": "Task A", "metadata": {"task_family": "alpha"}},
            {"id": "env_b", "instruction": "Task B", "metadata": {"task_family": "beta"}},
            {"id": "env_c", "instruction": "Task C", "metadata": {"task_family": "gamma"}},
            {"id": "env_d", "instruction": "Task D", "metadata": {"task_family": "delta"}},
        ],
        "attempts": [
            {"environment_id": env_id, "attempt_index": 0, "passed": True, "verifier_status": "passed"}
            for env_id in ["env_a", "env_b", "env_c", "env_d"]
        ],
        "split_by": "task_family",
        "holdout_fraction": 0.5,
        "seed": 3,
        "k_values": [1],
        "n_trials": 8,
        "random_pass_probability": 0.0,
        "max_control_pass_at_1": 0.0,
        "min_lift_over_control": 0.5,
    }

    r = client.post("/api/eval/environments/spurious-reward-control", json=body)

    assert r.status_code == 200
    result = r.json()["result"]
    assert result["schema_version"] == "bashgym.environment_spurious_reward_control.v1"
    assert result["control"]["mode"] == "simulated_random_labels"
    assert result["observed_report"]["pass_at_k"]["pass@1"] == pytest.approx(1.0)
    assert result["control"]["pass_at_k_summary"]["pass@1"]["p95"] == pytest.approx(0.0)
    assert result["gate"]["ship"] is True


def test_environment_spurious_reward_control_rejects_bad_k():
    body = _env_passk_body(record_to_registry=False)
    body.update({"holdout_fraction": 0.5, "k_values": [0]})

    r = client.post("/api/eval/environments/spurious-reward-control", json=body)

    assert r.status_code == 400
    assert "positive" in r.json()["detail"]


# ── local rollout pass@k ─────────────────────────────────────────────────────


def _local_rollout_body(tmp_path, **over):
    body = {
        "model_id": "candidate-sft",
        "workspace_root": str(tmp_path / "rollouts"),
        "environments": [
            {
                "id": "env_rollout_ok",
                "instruction": "Create answer.txt containing ok.",
                "verifier": {
                    "command": _py(
                        "from pathlib import Path; "
                        "raise SystemExit(0 if Path('answer.txt').read_text() == 'ok' else 1)"
                    )
                },
            }
        ],
        "k_values": [1],
        "command_attempts": [
            {
                "environment_id": "env_rollout_ok",
                "attempt_index": 0,
                "commands": [_py("from pathlib import Path; Path('answer.txt').write_text('ok')")],
            }
        ],
        "record_to_registry": False,
    }
    body.update(over)
    return body


def test_environment_local_rollout_passk_executes_and_reports(tmp_path):
    r = client.post(
        "/api/eval/environments/local-rollout-passk",
        json=_local_rollout_body(tmp_path),
    )

    assert r.status_code == 200
    data = r.json()
    assert data["report"]["pass_at_k"]["pass@1"] == pytest.approx(1.0)
    assert data["attempts"][0]["passed"] is True
    assert data["rollouts"][0]["verifier_observation"]["exit_code"] == 0


def test_environment_local_rollout_passk_rejects_unknown_env(tmp_path):
    body = _local_rollout_body(tmp_path)
    body["command_attempts"][0]["environment_id"] = "missing"

    r = client.post("/api/eval/environments/local-rollout-passk", json=body)

    assert r.status_code == 400
    assert "unknown environment_id" in r.json()["detail"]


# ── model rollout pass@k ─────────────────────────────────────────────────────


class _FakeEnvPassKReport:
    pass_at_k = {"pass@1": 1.0}
    n_environments = 1

    def to_dict(self):
        return {
            "k_values": [1],
            "n_environments": 1,
            "n_attempts": 1,
            "pass_at_k": self.pass_at_k,
            "mean_success_rate": 1.0,
            "per_environment": {"env_model_ok": {"pass@1": 1.0}},
            "attempt_summary": {
                "timeout_rate": 0.0,
                "verifier_status_distribution": {"passed": 1},
            },
            "warnings": [],
        }


class _FakeAttempt:
    environment_id = "env_model_ok"
    attempt_index = 0
    passed = True
    reward = 1.0
    verifier_status = "passed"
    metadata = {"behavior_logprob_tokens": 2}

    def to_dict(self):
        return {
            "environment_id": self.environment_id,
            "attempt_index": self.attempt_index,
            "passed": self.passed,
            "reward": self.reward,
            "verifier_status": self.verifier_status,
            "metadata": self.metadata,
        }


class _FakeRollout:
    attempt = _FakeAttempt()
    observations = []
    verifier_observation = None

    def to_dict(self):
        return {
            "attempt": self.attempt.to_dict(),
            "workspace": "tmp/env_model_ok",
            "observations": [],
            "verifier_observation": {"exit_code": 0},
        }


def test_environment_model_rollout_passk_resolves_endpoint_and_records(monkeypatch, tmp_path):
    captured = {}
    fake_reg = _FakeRegistry()
    replay_path = tmp_path / "dppo_replay.jsonl"
    monkeypatch.setattr("bashgym.models.get_registry", lambda *a, **k: fake_reg)

    def fake_run(environments, endpoint, **kwargs):
        captured["endpoint"] = endpoint
        captured["kwargs"] = kwargs
        return _FakeEnvPassKReport(), [_FakeRollout()], {
            "sampling_enabled": True,
            "selected_environment_ids": ["env_model_ok"],
        }

    monkeypatch.setattr(eval_routes.service, "run_model_environment_rollout_passk", fake_run)

    r = client.post(
        "/api/eval/environments/model-rollout-passk",
        json={
            "model_id": "candidate-sft",
            "endpoint": {"base_url": "http://h/v1", "model": "candidate"},
            "workspace_root": str(tmp_path / "rollouts"),
            "environments": [{"id": "env_model_ok", "instruction": "Do it."}],
            "attempts_per_environment": 2,
            "k_values": [1],
            "filter_zero_std_groups": True,
            "active_sampling": True,
            "target_prompt_groups": 1,
            "capture_logprobs": True,
            "top_logprobs": 5,
            "max_observation_chars": 1234,
            "dppo_replay_output_path": str(replay_path),
            "include_world_model_replay": True,
            "rwml_history_window": 2,
            "record_to_registry": True,
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert data["report"]["pass_at_k"]["pass@1"] == pytest.approx(1.0)
    assert data["recorded"] == ["bashgym_env_pass@1"]
    assert data["sampling_report"]["sampling_enabled"] is True
    assert data["dppo_report"]["attempts"] == 1
    assert data["dppo_report"]["rollout_logprobs_ready"] is True
    assert data["dppo_replay"]["records"] == 1
    assert data["dppo_replay"]["world_model_records"] == 1
    assert data["dppo_replay"]["path"] == str(replay_path)
    assert replay_path.exists()
    replay_record = json.loads(replay_path.read_text(encoding="utf-8").strip())
    assert "world_model" in replay_record
    assert captured["endpoint"].model == "candidate"
    assert captured["kwargs"]["attempts_per_environment"] == 2
    assert captured["kwargs"]["filter_zero_std_groups"] is True
    assert captured["kwargs"]["active_sampling"] is True
    assert captured["kwargs"]["target_prompt_groups"] == 1
    assert captured["kwargs"]["capture_logprobs"] is True
    assert captured["kwargs"]["top_logprobs"] == 5
    assert captured["kwargs"]["max_observation_chars"] == 1234


def test_environment_model_rollout_passk_requires_endpoint():
    r = client.post(
        "/api/eval/environments/model-rollout-passk",
        json={
            "endpoint": {"model": "candidate"},
            "environments": [{"id": "env_model_ok", "instruction": "Do it."}],
        },
    )

    assert r.status_code == 400
    assert "model endpoint" in r.json()["detail"]


def test_environment_reward_hacking_canaries_runs_filtered(tmp_path):
    r = client.post(
        "/api/eval/environments/reward-hacking-canaries",
        json={
            "categories": ["verifier_tamper"],
            "workspace_root": str(tmp_path / "canaries"),
            "keep_workspace": False,
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert data["summary"]["total"] == 1
    assert data["summary"]["guarded"] == 1
    assert data["summary"]["failed"] == 0
    assert data["summary"]["results"][0]["category"] == "verifier_tamper"
    assert data["summary"]["results"][0]["verifier_status"] == "tampered"
    assert data["canaries"][0]["expected_status"] == "tampered"
    assert data["rollouts"][0]["attempt"]["passed"] is False


def test_environment_reward_hacking_canaries_rejects_unknown_category(tmp_path):
    r = client.post(
        "/api/eval/environments/reward-hacking-canaries",
        json={
            "categories": ["not_real"],
            "workspace_root": str(tmp_path / "canaries"),
        },
    )

    assert r.status_code == 400
    assert "unknown canary categories" in r.json()["detail"]


def test_environment_dppo_replay_enrich_attaches_train_logprobs(tmp_path):
    input_path = tmp_path / "dppo_replay.jsonl"
    output_path = tmp_path / "dppo_replay_scored.jsonl"
    record = {
        "schema_version": "bashgym.dppo_replay.v1",
        "batch_id": "batch-1",
        "environment_id": "env_model_ok",
        "attempt_index": 0,
        "prompt": "Do it.",
        "environment": {"id": "env_model_ok", "instruction": "Do it."},
        "reward": 1.0,
        "passed": True,
        "verifier_status": "passed",
        "active_sampling_selected": True,
        "reward_group_std": 0.5,
        "trajectory": {"commands": ["echo ok"], "observations": [], "verifier_observation": None},
        "behavior_policy": {
            "response_logprobs": [
                {"tokens": ["echo", " ok"], "token_logprobs": [-0.2, -0.3]}
            ],
            "behavior_logprob_tokens": 2,
        },
        "optimizer": {
            "behavior_logprobs_ready": True,
            "train_logprobs_ready": False,
            "train_logprob_replay_required": True,
            "train_logprob_tokens": 0,
        },
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    r = client.post(
        "/api/eval/environments/dppo-replay/enrich",
        json={
            "input_path": str(input_path),
            "output_path": str(output_path),
            "train_logprobs": [
                {
                    "environment_id": "env_model_ok",
                    "attempt_index": 0,
                    "token_logprobs": [-0.1, -0.15],
                    "model": "train-policy",
                }
            ],
        },
    )

    assert r.status_code == 200
    summary = r.json()["dppo_replay"]
    assert summary["records"] == 1
    assert summary["train_logprobs_ready_records"] == 1
    assert summary["dppo"]["n_tokens"] == 2
    scored = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert scored["train_policy"]["model"] == "train-policy"
    assert scored["optimizer"]["train_logprobs_ready"] is True


def test_environment_dppo_smoke_plan_returns_launcher_contract(monkeypatch, tmp_path):
    replay_path = tmp_path / "scored.jsonl"
    replay_path.write_text("{}\n", encoding="utf-8")
    captured = {}

    class _Plan:
        def to_dict(self):
            return {
                "backend": "verl",
                "runnable": True,
                "command": ["python", "-m", "verl.trainer.main_ppo"],
                "script_path": str(tmp_path / "launch_dppo_smoke.sh"),
            }

    def fake_prepare(config):
        captured["config"] = config
        return _Plan()

    monkeypatch.setattr(eval_routes, "prepare_dppo_smoke_launch", fake_prepare)

    r = client.post(
        "/api/eval/environments/dppo-replay/smoke-plan",
        json={
            "replay_path": str(replay_path),
            "output_dir": str(tmp_path / "run"),
            "base_model": "Qwen/Qwen3.5-4B",
            "backend": "verl",
            "max_steps": 1,
            "echo_enabled": True,
            "echo_aux_lambda": 0.05,
            "rwml_enabled": True,
            "rwml_distance_threshold": 0.15,
            "rwml_easy_pass_rate_threshold": 0.75,
            "rwml_easy_keep_probability": 0.2,
            "rwml_history_window": 6,
            "rwml_embedding_model": "qwen3-embedding",
            "rwml_kl_beta": 0.03,
        },
    )

    assert r.status_code == 200
    assert r.json()["plan"]["backend"] == "verl"
    assert r.json()["plan"]["runnable"] is True
    assert captured["config"].replay_path == replay_path
    assert captured["config"].base_model == "Qwen/Qwen3.5-4B"
    assert captured["config"].echo_enabled is True
    assert captured["config"].rwml_enabled is True
    assert captured["config"].rwml_distance_threshold == 0.15
    assert captured["config"].rwml_easy_pass_rate_threshold == 0.75
    assert captured["config"].rwml_easy_keep_probability == 0.2
    assert captured["config"].rwml_history_window == 6
    assert captured["config"].rwml_embedding_model == "qwen3-embedding"
    assert captured["config"].rwml_kl_beta == 0.03
