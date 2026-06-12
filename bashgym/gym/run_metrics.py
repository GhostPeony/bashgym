"""Persistent per-run training metrics.

Each training run appends metric points to ``<run_dir>/metrics.jsonl`` next to
its checkpoints, so loss curves survive the session and back the run-comparison
API (``GET /api/training/runs`` and ``GET /api/training/runs/{run_id}/metrics``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.jsonl"


def default_models_dir() -> Path:
    """Directory where training runs store their artifacts (TrainerConfig.output_dir default)."""
    return Path("data/models")


def record_run_metric(run_dir: Path | str, metric: dict) -> None:
    """Append one metric point to ``<run_dir>/metrics.jsonl`` (best-effort, never raises)."""
    try:
        point = {**metric, "ts": time.time()}
        with open(Path(run_dir) / METRICS_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(point) + "\n")
    except OSError:
        logger.warning("Failed to write %s in %s", METRICS_FILENAME, run_dir)


def list_runs(models_dir: Path | None = None) -> list[dict]:
    """List training runs found under the models directory, newest first."""
    base = models_dir or default_models_dir()
    runs: list[dict] = []
    if not base.exists():
        return runs
    for run_dir in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        runs.append(
            {
                "run_id": run_dir.name,
                "modified": run_dir.stat().st_mtime,
                "has_metrics": (run_dir / METRICS_FILENAME).exists(),
                "has_final": (run_dir / "final").exists(),
            }
        )
    return runs


def read_run_metrics(run_id: str, models_dir: Path | None = None) -> list[dict] | None:
    """Read the persisted metric history for one run.

    Returns ``None`` if the run has no metrics file. ``run_id`` values containing
    path separators are rejected to prevent traversal.
    """
    if Path(run_id).name != run_id:
        raise ValueError(f"Invalid run_id: {run_id!r}")
    base = models_dir or default_models_dir()
    metrics_file = base / run_id / METRICS_FILENAME
    if not metrics_file.exists():
        return None
    points: list[dict] = []
    with open(metrics_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return points
