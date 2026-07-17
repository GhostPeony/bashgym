"""Deterministic health, trend, and agent-context projections for the ledger."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from bashgym._compat import UTC
from bashgym.ledger.persistence import ExperimentLedgerRepository


def metric_trend(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a chronological metric series without inventing comparability."""

    if not points:
        return {"point_count": 0, "first": None, "latest": None, "delta": None, "slope": None}
    values = [float(point["metric_value"]) for point in points]
    count = len(values)
    if count == 1:
        slope = 0.0
    else:
        x_mean = (count - 1) / 2
        y_mean = sum(values) / count
        numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
        denominator = sum((index - x_mean) ** 2 for index in range(count))
        slope = numerator / denominator if denominator else 0.0
    return {
        "point_count": count,
        "first": values[0],
        "latest": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "delta": values[-1] - values[0],
        "slope": slope,
        "first_observed_at": points[0]["observed_at"],
        "latest_observed_at": points[-1]["observed_at"],
    }


def build_project_context(
    repository: ExperimentLedgerRepository,
    workspace_id: str,
    project_id: str,
    *,
    recent_limit: int = 20,
    stale_after: timedelta = timedelta(hours=6),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a bounded, evidence-linked context pack for an agent or canvas view."""

    project = repository.get_project(workspace_id, project_id)
    experiments = repository.list_experiments(workspace_id, project_id)
    model_versions = repository.list_model_versions(workspace_id, project_id)
    dataset_versions = repository.list_dataset_versions(workspace_id, project_id)
    environments = repository.list_environments(workspace_id, project_id)
    evaluation_suites = repository.list_evaluation_suites(workspace_id, project_id)
    runs = repository.list_runs(workspace_id, project_id, limit=max(recent_limit, 200))
    recent_runs = runs[: max(1, min(recent_limit, 100))]
    evaluations = repository.list_evaluation_results(
        workspace_id, project_id, limit=max(recent_limit * 5, 100)
    )
    artifacts = repository.list_artifacts(
        workspace_id, project_id, limit=max(recent_limit * 5, 100)
    )
    decisions = repository.list_decisions(workspace_id, project_id, limit=recent_limit)
    events = repository.list_events(workspace_id, project_id, limit=recent_limit)

    status_counts = Counter(str(run["status"]) for run in runs)
    method_counts = Counter(str(run["training_method"]) for run in runs)
    task_counts = Counter(str(run["task_type"]) for run in runs)
    evaluation_run_ids = {str(result["run_id"]) for result in evaluations}
    artifact_run_ids = {str(artifact["run_id"]) for artifact in artifacts}
    moment = now or datetime.now(UTC)
    stale_cutoff = moment - stale_after

    stale_runs: list[str] = []
    for run in runs:
        if run["status"] not in {"queued", "preparing", "running", "paused", "unknown"}:
            continue
        updated = datetime.fromisoformat(str(run["updated_at"]))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        if updated < stale_cutoff:
            stale_runs.append(str(run["run_id"]))

    missing_context = [
        str(run["run_id"])
        for run in runs
        if run["context_status"] != "verified"
        or not run.get("model_version_id")
        or not run.get("dataset_version_id")
        or not run.get("environment_id")
    ]
    completed_without_eval = [
        str(run["run_id"])
        for run in runs
        if run["status"] == "completed" and run["run_id"] not in evaluation_run_ids
    ]
    terminal_without_artifact = [
        str(run["run_id"])
        for run in runs
        if run["status"] in {"completed", "failed", "cancelled"}
        and run["run_id"] not in artifact_run_ids
    ]

    signals: list[dict[str, Any]] = []
    if stale_runs:
        signals.append({"severity": "warning", "code": "stale_active_runs", "run_ids": stale_runs})
    if missing_context:
        signals.append(
            {"severity": "warning", "code": "incomplete_lineage", "run_ids": missing_context}
        )
    if completed_without_eval:
        signals.append(
            {
                "severity": "warning",
                "code": "completed_without_eval",
                "run_ids": completed_without_eval,
            }
        )
    if terminal_without_artifact:
        signals.append(
            {
                "severity": "info",
                "code": "terminal_without_artifact",
                "run_ids": terminal_without_artifact,
            }
        )
    if status_counts.get("failed", 0):
        signals.append(
            {
                "severity": "warning",
                "code": "failed_runs_present",
                "count": status_counts["failed"],
            }
        )

    health = (
        "attention" if any(signal["severity"] == "warning" for signal in signals) else "healthy"
    )
    return {
        "schema_version": "experiment_project_context.v1",
        "workspace_id": workspace_id,
        "project_id": project_id,
        "generated_at": moment.isoformat(),
        "project": project,
        "health": {
            "status": health,
            "signals": signals,
            "active_run_count": sum(
                status_counts.get(status, 0)
                for status in ("queued", "preparing", "running", "paused", "unknown")
            ),
            "stale_run_count": len(stale_runs),
            "missing_context_count": len(missing_context),
            "completed_without_eval_count": len(completed_without_eval),
        },
        "inventory": {
            "experiment_count": len(experiments),
            "model_version_count": len(model_versions),
            "dataset_version_count": len(dataset_versions),
            "environment_count": len(environments),
            "evaluation_suite_count": len(evaluation_suites),
            "run_count": len(runs),
            "evaluation_count": len(evaluations),
            "artifact_count": len(artifacts),
            "decision_count": len(decisions),
            "status_counts": dict(sorted(status_counts.items())),
            "training_method_counts": dict(sorted(method_counts.items())),
            "task_type_counts": dict(sorted(task_counts.items())),
        },
        "experiments": experiments[:recent_limit],
        "lineage": {
            "model_versions": model_versions[:recent_limit],
            "dataset_versions": dataset_versions[:recent_limit],
            "environments": environments[:recent_limit],
            "evaluation_suites": evaluation_suites[:recent_limit],
        },
        "recent_runs": recent_runs,
        "recent_evaluations": evaluations[:recent_limit],
        "recent_decisions": decisions,
        "recent_events": events,
        "evidence": {
            "run_ids": [run["run_id"] for run in recent_runs],
            "evaluation_result_ids": [
                result["evaluation_result_id"] for result in evaluations[:recent_limit]
            ],
            "artifact_ids": [artifact["artifact_id"] for artifact in artifacts[:recent_limit]],
            "decision_ids": [decision["decision_id"] for decision in decisions],
            "event_cursors": [event["cursor"] for event in events],
        },
    }


def compare_runs(
    repository: ExperimentLedgerRepository,
    workspace_id: str,
    project_id: str,
    run_ids: list[str],
) -> dict[str, Any]:
    """Compare only results that share the same evaluation-suite contract."""

    ordered_run_ids = list(dict.fromkeys(run_ids))
    if len(ordered_run_ids) < 2 or len(ordered_run_ids) > 20:
        raise ValueError("compare_runs requires 2 to 20 unique run IDs")
    for run_id in ordered_run_ids:
        repository.get_run(workspace_id, project_id, run_id)
    suites = {
        suite["evaluation_suite_id"]: suite
        for suite in repository.list_evaluation_suites(workspace_id, project_id)
    }
    results = [
        result
        for result in repository.list_evaluation_results(workspace_id, project_id, limit=1000)
        if result["run_id"] in ordered_run_ids and result["status"] == "completed"
    ]
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["evaluation_suite_id"], {})[result["run_id"]] = result

    comparisons = []
    for suite_id, by_run in sorted(grouped.items()):
        if len(by_run) < 2:
            continue
        suite = suites.get(suite_id)
        if suite is None:
            continue
        baseline_id = next(run_id for run_id in ordered_run_ids if run_id in by_run)
        metric_contract = suite["metric_contract"]
        metric_names = sorted(
            set.intersection(*(set(result["metrics"]) for result in by_run.values()))
        )
        metrics = []
        for metric_name in metric_names:
            baseline = float(by_run[baseline_id]["metrics"][metric_name])
            contract = metric_contract.get(metric_name, {})
            direction = (
                contract.get("direction", "unspecified")
                if isinstance(contract, dict)
                else "unspecified"
            )
            values = {
                run_id: {
                    "value": float(by_run[run_id]["metrics"][metric_name]),
                    "delta_from_baseline": float(by_run[run_id]["metrics"][metric_name]) - baseline,
                }
                for run_id in ordered_run_ids
                if run_id in by_run
            }
            metrics.append({"metric_name": metric_name, "direction": direction, "values": values})
        comparisons.append(
            {
                "evaluation_suite_id": suite_id,
                "evaluation_suite_name": suite["name"],
                "baseline_run_id": baseline_id,
                "run_ids": [run_id for run_id in ordered_run_ids if run_id in by_run],
                "metrics": metrics,
                "result_ids": {
                    run_id: result["evaluation_result_id"] for run_id, result in by_run.items()
                },
            }
        )
    missing = [
        run_id
        for run_id in ordered_run_ids
        if not any(run_id in comparison["run_ids"] for comparison in comparisons)
    ]
    return {
        "schema_version": "experiment_run_comparison.v1",
        "workspace_id": workspace_id,
        "project_id": project_id,
        "requested_run_ids": ordered_run_ids,
        "comparisons": comparisons,
        "not_comparable_run_ids": missing,
        "comparability_rule": (
            "Metrics are compared only when completed results share the exact evaluation suite ID."
        ),
    }


def build_sync_envelope(
    repository: ExperimentLedgerRepository,
    workspace_id: str,
    project_id: str,
    *,
    after_cursor: int = 0,
    limit: int = 200,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the stable incremental boundary for GBrain or optional cloud sinks."""

    bounded = max(1, min(limit, 1000))
    events = repository.list_events(
        workspace_id, project_id, after=max(0, after_cursor), limit=bounded
    )
    next_cursor = int(events[-1]["cursor"]) if events else max(0, after_cursor)
    return {
        "schema_version": "experiment_ledger_sync.v1",
        "workspace_id": workspace_id,
        "project_id": project_id,
        "generated_at": (now or datetime.now(UTC)).isoformat(),
        "after_cursor": max(0, after_cursor),
        "next_cursor": next_cursor,
        "has_more": len(events) == bounded,
        "events": events,
        "curation_policy": {
            "store_in_sink": [
                "goals",
                "configuration decisions",
                "lineage references",
                "milestones",
                "anomalies",
                "evaluation summaries",
                "conclusions",
                "report references",
                "follow-up work",
            ],
            "keep_in_bashgym": [
                "raw logs",
                "full metric series",
                "datasets",
                "checkpoints",
                "secrets",
            ],
        },
    }
