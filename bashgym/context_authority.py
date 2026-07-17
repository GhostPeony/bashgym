"""Shared authority, freshness, and conflict metadata for agent context packs."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

from bashgym._compat import UTC

SOURCE_PRECEDENCE = (
    {
        "source_id": "live_runtime",
        "rank": 1,
        "authority": "Current processes, jobs, progress, and runtime health",
    },
    {
        "source_id": "durable_ledger",
        "rank": 2,
        "authority": "Run identities, lineage, metrics, evaluations, artifacts, and decisions",
    },
    {
        "source_id": "workspace_snapshot",
        "rank": 3,
        "authority": "Current canvas layout, connected tools, and operator-visible selections",
    },
    {
        "source_id": "curated_gbrain",
        "rank": 4,
        "authority": "Project history, goals, prior decisions, and concise milestone summaries",
    },
    {
        "source_id": "conversation_memory",
        "rank": 5,
        "authority": "Unverified conversational continuity only",
    },
)

_ACTIVE_STATUSES = {"queued", "preparing", "running", "active", "paused", "cancelling"}


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _latest_timestamp(values: Iterable[Any]) -> str | None:
    parsed = [item for item in (_parse_timestamp(value) for value in values) if item is not None]
    return max(parsed).isoformat() if parsed else None


def _freshness(
    observed_at: str | None,
    *,
    generated_at: datetime,
    stale_after: timedelta,
    missing_state: str,
) -> tuple[str, float | None]:
    observed = _parse_timestamp(observed_at)
    if observed is None:
        return missing_state, None
    age = max(0.0, (generated_at - observed).total_seconds())
    return ("stale" if age > stale_after.total_seconds() else "fresh"), age


def _source(
    source_id: str,
    *,
    observed_at: str | None,
    data_latest_at: str | None,
    generated_at: datetime,
    stale_after: timedelta,
    available: bool,
    empty: bool = False,
    source_ref: str,
) -> dict[str, Any]:
    freshness, age_seconds = _freshness(
        observed_at,
        generated_at=generated_at,
        stale_after=stale_after,
        missing_state="unavailable" if not available else "unknown",
    )
    return {
        "source_id": source_id,
        "rank": next(item["rank"] for item in SOURCE_PRECEDENCE if item["source_id"] == source_id),
        "available": available,
        "empty": bool(empty),
        "freshness": freshness,
        "observed_at": observed_at,
        "age_seconds": age_seconds,
        "data_latest_at": data_latest_at,
        "source_ref": source_ref,
    }


def build_context_authority(
    *,
    generated_at: str,
    canvas_updated_at: str | None,
    training_runs: list[dict[str, Any]],
    runtime_jobs: list[dict[str, Any]],
    campaigns: list[dict[str, Any]],
    ledger_loaded: bool = True,
    gbrain_observed_at: str | None = None,
) -> dict[str, Any]:
    """Describe which evidence wins and expose concrete stale/conflicting observations."""

    moment = _parse_timestamp(generated_at) or datetime.now(UTC)
    runtime_latest = _latest_timestamp(
        value
        for job in runtime_jobs
        for value in (
            job.get("observed_at"),
            job.get("updated_at"),
            job.get("started_at"),
            job.get("completed_at"),
        )
    )
    durable_latest = _latest_timestamp(
        value
        for item in [*training_runs, *campaigns]
        for value in (
            item.get("updated_at"),
            item.get("completed_at"),
            item.get("started_at"),
            item.get("created_at"),
        )
    )
    # Querying an observer or ledger is itself a fresh observation even when the result is empty.
    runtime_observed = generated_at
    durable_observed = generated_at if ledger_loaded else None
    sources = [
        _source(
            "live_runtime",
            observed_at=runtime_observed,
            data_latest_at=runtime_latest,
            generated_at=moment,
            stale_after=timedelta(minutes=2),
            available=True,
            empty=not runtime_jobs,
            source_ref="runtime observer",
        ),
        _source(
            "durable_ledger",
            observed_at=durable_observed,
            data_latest_at=durable_latest,
            generated_at=moment,
            stale_after=timedelta(minutes=2),
            available=ledger_loaded,
            empty=not training_runs and not campaigns,
            source_ref="BashGym campaign and experiment ledger",
        ),
        _source(
            "workspace_snapshot",
            observed_at=canvas_updated_at,
            data_latest_at=canvas_updated_at,
            generated_at=moment,
            stale_after=timedelta(minutes=15),
            available=canvas_updated_at is not None,
            source_ref="renderer-owned canvas snapshot",
        ),
        _source(
            "curated_gbrain",
            observed_at=gbrain_observed_at,
            data_latest_at=gbrain_observed_at,
            generated_at=moment,
            stale_after=timedelta(hours=24),
            available=gbrain_observed_at is not None,
            source_ref="authoritative remote GBrain source",
        ),
        {
            "source_id": "conversation_memory",
            "rank": 5,
            "available": True,
            "empty": False,
            "freshness": "unverified",
            "observed_at": None,
            "age_seconds": None,
            "data_latest_at": None,
            "source_ref": "current endpoint conversation",
        },
    ]

    conflicts: list[dict[str, Any]] = []
    runtime_by_id = {
        str(job.get("run_id") or job.get("job_id")): job
        for job in runtime_jobs
        if job.get("run_id") or job.get("job_id")
    }
    for run in training_runs:
        run_id = str(run.get("run_id") or "")
        runtime = runtime_by_id.get(run_id)
        if not runtime:
            continue
        durable_status = str(run.get("status") or "unknown").casefold()
        runtime_status = str(runtime.get("status") or "unknown").casefold()
        if durable_status != runtime_status and (
            durable_status in _ACTIVE_STATUSES or runtime_status in _ACTIVE_STATUSES
        ):
            conflicts.append(
                {
                    "code": "run_status_mismatch",
                    "entity_id": run_id,
                    "higher_authority": "live_runtime",
                    "higher_value": runtime_status,
                    "lower_authority": "durable_ledger",
                    "lower_value": durable_status,
                    "decision_required": False,
                    "resolution": "Report runtime status and reconcile the durable run record.",
                }
            )
    canvas_source = next(item for item in sources if item["source_id"] == "workspace_snapshot")
    if canvas_source["freshness"] == "stale":
        conflicts.append(
            {
                "code": "stale_workspace_snapshot",
                "entity_id": "workspace",
                "higher_authority": "durable_ledger",
                "lower_authority": "workspace_snapshot",
                "decision_required": False,
                "resolution": "Refresh the canvas before relying on visible selections.",
            }
        )

    return {
        "schema_version": "bashgym.context-authority.v1",
        "generated_at": moment.isoformat(),
        "source_precedence": list(SOURCE_PRECEDENCE),
        "sources": sources,
        "conflicts": conflicts,
        "decision_required": any(bool(item.get("decision_required")) for item in conflicts),
        "agent_rules": [
            "Use the highest-ranked available source for each claim.",
            "Treat conversation memory as unverified when it disagrees with live or durable evidence.",
            "State source IDs, evidence IDs, and observation times for current-state claims.",
            "Expose unresolved conflicts or missing project identity instead of blending contexts.",
        ],
    }
