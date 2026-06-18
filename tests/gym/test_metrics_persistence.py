"""Training metrics are appended to metrics.jsonl in the run directory."""

import json


def test_record_metric_appends_jsonl(tmp_path):
    from bashgym.gym.run_metrics import record_run_metric

    run_dir = tmp_path / "run-001"
    run_dir.mkdir()

    record_run_metric(run_dir, {"step": 1, "loss": 2.5, "learning_rate": 1e-4})
    record_run_metric(run_dir, {"step": 2, "loss": 2.1, "learning_rate": 9e-5})

    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["loss"] == 2.5
    assert second["step"] == 2
    assert "ts" in first


def test_record_metric_never_raises_on_bad_dir(tmp_path):
    from bashgym.gym.run_metrics import record_run_metric

    missing = tmp_path / "does" / "not" / "exist"
    # Must be best-effort: no exception even when the directory is absent
    record_run_metric(missing, {"step": 1, "loss": 1.0})


def test_list_runs_orders_newest_first(tmp_path):
    from bashgym.gym.run_metrics import list_runs, record_run_metric

    old = tmp_path / "run-old"
    new = tmp_path / "run-new"
    old.mkdir()
    new.mkdir()
    record_run_metric(new, {"step": 1, "loss": 1.0})

    import os
    import time

    past = time.time() - 3600
    os.utime(old, (past, past))

    runs = list_runs(models_dir=tmp_path)
    assert [r["run_id"] for r in runs] == ["run-new", "run-old"]
    assert runs[0]["has_metrics"] is True
    assert runs[1]["has_metrics"] is False


def test_read_run_metrics(tmp_path):
    from bashgym.gym.run_metrics import read_run_metrics, record_run_metric

    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    record_run_metric(run_dir, {"step": 1, "loss": 2.5})

    points = read_run_metrics("run-001", models_dir=tmp_path)
    assert points is not None
    assert points[0]["step"] == 1

    assert read_run_metrics("run-missing", models_dir=tmp_path) is None


def test_read_run_metrics_rejects_traversal(tmp_path):
    import pytest

    from bashgym.gym.run_metrics import read_run_metrics

    with pytest.raises(ValueError):
        read_run_metrics("../escape", models_dir=tmp_path)
