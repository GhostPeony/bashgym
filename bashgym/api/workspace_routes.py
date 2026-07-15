"""Workspace canvas context and intent routes.

These desktop-only endpoints give local agents a sanctioned way to read the
current canvas and emit semantic intents. The renderer owns layout; the backend
keeps a live sanitized snapshot and broadcasts intents to connected canvases.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from bashgym.api.websocket import (
    broadcast_workspace_canvas_intent,
    broadcast_workspace_context_updated,
)
from bashgym.context_authority import build_context_authority

router = APIRouter(prefix="/api/workspace", tags=["workspace"])

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)
HF_TOKEN_VALUE = re.compile(r"^hf_[A-Za-z0-9]{16,}$")


def _now() -> str:
    return datetime.utcnow().isoformat()


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SECRET_KEY_PARTS)


def redact_workspace_value(value: Any) -> Any:
    """Return a recursively redacted JSON-compatible value."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(str(key)):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = redact_workspace_value(item)
        return redacted
    if isinstance(value, list):
        return [redact_workspace_value(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if value.startswith(("sk-", "ghp_", "xoxb-", "xoxp-")) or HF_TOKEN_VALUE.fullmatch(
            value
        ):
            return "[redacted]"
        if "bearer " in lowered:
            return "[redacted]"
    return value


class WorkspacePanel(BaseModel):
    panel_id: str
    type: str
    title: str
    terminal_id: str | None = None
    adapter_config: dict[str, Any] | None = None
    visible: bool = True


class WorkspaceEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str | None = None
    data: dict[str, Any] | None = None


class WorkspaceTerminal(BaseModel):
    terminal_id: str
    panel_id: str | None = None
    title: str | None = None
    cwd: str | None = None
    agent_kind: str | None = None
    status: str | None = None
    current_tool: str | None = None


class WorkspaceCanvasSnapshot(BaseModel):
    schema_version: str = "bashgym.workspace.canvas.v1"
    updated_at: str = Field(default_factory=_now)
    # Named canvas workspaces: snapshots are keyed by workspace_id; defaulted
    # fields keep single-workspace clients fully back-compatible
    workspace_id: str = "default"
    workspace_name: str | None = None
    panels: list[WorkspacePanel] = Field(default_factory=list)
    edges: list[WorkspaceEdge] = Field(default_factory=list)
    terminals: list[WorkspaceTerminal] = Field(default_factory=list)
    data_summaries: dict[str, Any] = Field(default_factory=dict)
    allowed_actions: list[str] = Field(
        default_factory=lambda: [
            "workspace.context.read",
            "workspace.canvas.intent.emit",
            "training.start",
            "data_designer.start",
            "skill_lab.inspect",
            "skill_lab.prepare",
            "skill_lab.run",
            "hf_context.search",
            "hf_context.inspect",
            "hf_context.pin",
            "hf_context.activate",
            "hf_context.deactivate",
            "hf_context.prepare_eval",
        ]
    )


class WorkspaceEventSource(BaseModel):
    kind: str = "agent"
    terminal_id: str | None = None
    panel_id: str | None = None
    agent: str | None = None


class WorkspaceSuggestedNode(BaseModel):
    recipe: str
    title: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class WorkspaceRelationship(BaseModel):
    source: str
    target: str
    type: str


class WorkspaceEvent(BaseModel):
    type: str
    workspace_id: str | None = None
    source: WorkspaceEventSource = Field(default_factory=WorkspaceEventSource)
    correlation_id: str | None = None
    title: str | None = None
    summary: str | None = None
    entity: dict[str, Any] = Field(default_factory=dict)
    suggested_nodes: list[WorkspaceSuggestedNode] = Field(default_factory=list)
    relationships: list[WorkspaceRelationship] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


def _snapshot_store(request: Request) -> dict[str, WorkspaceCanvasSnapshot]:
    store = getattr(request.app.state, "workspace_canvas_snapshots", None)
    if not isinstance(store, dict):
        store = {}
        request.app.state.workspace_canvas_snapshots = store
    return store


def _state_snapshot(request: Request, workspace_id: str | None = None) -> WorkspaceCanvasSnapshot:
    """The requested workspace's snapshot; without an id, the freshest one."""
    store = _snapshot_store(request)
    if workspace_id is not None:
        snapshot = store.get(workspace_id)
        return (
            snapshot
            if isinstance(snapshot, WorkspaceCanvasSnapshot)
            else WorkspaceCanvasSnapshot(workspace_id=workspace_id)
        )
    if store:
        return max(store.values(), key=lambda snap: snap.updated_at)
    return WorkspaceCanvasSnapshot()


def _state_events(request: Request, workspace_id: str) -> list[dict[str, Any]]:
    store = getattr(request.app.state, "workspace_events", None)
    if not isinstance(store, dict):
        store = {}
        request.app.state.workspace_events = store
    events = store.get(workspace_id)
    if not isinstance(events, list):
        events = []
        store[workspace_id] = events
    return events


def _run_workspace_id(run: dict[str, Any]) -> str | None:
    origin = run.get("origin")
    if isinstance(origin, dict) and origin.get("workspace_id"):
        return str(origin["workspace_id"])
    config = run.get("config")
    if isinstance(config, dict):
        config_origin = config.get("origin")
        if isinstance(config_origin, dict) and config_origin.get("workspace_id"):
            return str(config_origin["workspace_id"])
    return None


def _training_runs(request: Request, workspace_id: str) -> list[dict[str, Any]]:
    runs = getattr(request.app.state, "training_runs", {}) or {}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in runs.values():
        if not isinstance(run, dict):
            continue
        scoped_workspace_id = _run_workspace_id(run)
        if scoped_workspace_id and scoped_workspace_id != workspace_id:
            continue
        run_id = str(run.get("run_id") or "")
        if run_id:
            seen.add(run_id)
        result.append(
            redact_workspace_value(
                jsonable_encoder(
                    {
                        "run_id": run_id or None,
                        "workspace_id": scoped_workspace_id,
                        "status": run.get("status"),
                        "strategy": run.get("strategy"),
                        "started_at": run.get("started_at"),
                        "completed_at": run.get("completed_at"),
                        "origin": run.get("origin"),
                        "correlation_id": run.get("correlation_id"),
                        "compute_target": run.get("compute_target"),
                        "config": run.get("config", {}),
                        "metrics": run.get("metrics"),
                        "error": run.get("error"),
                    }
                )
            )
        )

    try:
        from bashgym.api.training_state import list_run_states

        trainer = getattr(request.app.state, "trainer", None)
        output_dir = getattr(getattr(trainer, "config", None), "output_dir", None)
        for state in list_run_states(Path(output_dir) if output_dir else None):
            if state.run_id in seen:
                continue
            persisted = {
                "run_id": state.run_id,
                "status": state.status,
                "strategy": state.config.get("strategy", "sft"),
                "started_at": state.started_at,
                "completed_at": state.completed_at,
                "origin": state.config.get("origin"),
                "correlation_id": state.config.get("correlation_id"),
                "compute_target": state.config.get("compute_target"),
                "config": state.config,
                "metrics": state.last_metrics,
            }
            scoped_workspace_id = _run_workspace_id(persisted)
            if scoped_workspace_id and scoped_workspace_id != workspace_id:
                continue
            persisted["workspace_id"] = scoped_workspace_id
            result.append(redact_workspace_value(jsonable_encoder(persisted)))
    except Exception:
        pass

    result.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return result[:50]


def _runtime_jobs(request: Request) -> list[dict[str, Any]]:
    """Return a bounded path-free projection of locally observed jobs."""

    try:
        from bashgym.api.runtime_observer import RuntimeObserver

        observer = getattr(request.app.state, "runtime_observer", None)
        if not isinstance(observer, RuntimeObserver):
            observer = RuntimeObserver(Path.cwd())
            request.app.state.runtime_observer = observer
        jobs = []
        for raw in observer.list_jobs():
            if not isinstance(raw, dict):
                continue
            artifacts = []
            for artifact in raw.get("artifacts") or []:
                if not isinstance(artifact, dict):
                    continue
                artifacts.append(
                    {
                        "name": Path(str(artifact.get("name") or "artifact")).name,
                        "size": artifact.get("size"),
                        "modified_at": artifact.get("modified_at"),
                    }
                )
            jobs.append(
                redact_workspace_value(
                    {
                        "job_id": raw.get("job_id"),
                        "kind": raw.get("kind"),
                        "status": raw.get("status"),
                        "title": raw.get("title"),
                        "script": Path(str(raw.get("script") or "")).name,
                        "started_at": raw.get("started_at"),
                        "completed_at": raw.get("completed_at"),
                        "pipeline": raw.get("pipeline"),
                        "job_name": raw.get("job_name"),
                        "model": raw.get("model"),
                        "provider": raw.get("provider"),
                        "execution": raw.get("execution"),
                        "strategy": raw.get("strategy"),
                        "progress": raw.get("progress"),
                        "artifacts": artifacts,
                        "source": raw.get("source"),
                    }
                )
            )
        jobs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        return jobs[:30]
    except Exception:
        return []


def _status_counts(values: list[Any]) -> dict[str, int]:
    counts = Counter(str(getattr(value, "status", "unknown")).split(".")[-1].lower() for value in values)
    return dict(sorted(counts.items()))


def _campaign_repository(request: Request):
    try:
        from bashgym.campaigns.runtime import CampaignRuntimeRepository
        from bashgym.config import get_bashgym_dir, get_settings

        if not get_settings().campaigns_enabled:
            return None
        repository = getattr(request.app.state, "campaign_repository", None)
        if isinstance(repository, CampaignRuntimeRepository):
            return repository
        database = get_bashgym_dir() / "campaigns" / "campaigns.sqlite3"
        if not database.is_file():
            return None
        repository = CampaignRuntimeRepository(database)
        repository.initialize()
        request.app.state.campaign_repository = repository
        return repository
    except Exception:
        return None


def _campaign_context(
    request: Request, workspace_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Project durable campaign evidence into bounded agent-readable context."""

    repository = _campaign_repository(request)
    if repository is None:
        return [], [], []
    try:
        campaigns = repository.list_campaigns(workspace_id)
    except Exception:
        return [], [], []

    projected: list[dict[str, Any]] = []
    activity: list[dict[str, Any]] = []
    report_refs: list[dict[str, Any]] = []
    for campaign in sorted(campaigns, key=lambda item: item.updated_at, reverse=True)[:20]:
        try:
            evidence = repository.build_evidence_snapshot(workspace_id, campaign.campaign_id)
            studies = list(repository.list_studies(workspace_id, campaign.campaign_id))
            attempts = list(repository.list_attempts(workspace_id, campaign.campaign_id))
            comparisons = list(
                repository.list_development_comparisons(workspace_id, campaign.campaign_id)
            )
            events = repository.list_recent_events(
                workspace_id, campaign.campaign_id, limit=30
            )
            exports = repository.list_exports(workspace_id, campaign.campaign_id, limit=5)
        except Exception:
            continue

        reports = []
        for export in exports:
            manifest = export.get("manifest") if isinstance(export, dict) else {}
            manifest = manifest if isinstance(manifest, dict) else {}
            report = {
                "campaign_id": campaign.campaign_id,
                "export_id": export.get("export_id"),
                "formats": export.get("formats", []),
                "source_digest": manifest.get("source_digest"),
                "quality_findings_available": manifest.get("quality_findings_available"),
                "files": [
                    {
                        "name": Path(str(item.get("name") or "artifact")).name,
                        "sha256": item.get("sha256"),
                        "size_bytes": item.get("size_bytes"),
                    }
                    for item in (manifest.get("files") or [])
                    if isinstance(item, dict)
                ],
                "created_at": export.get("created_at"),
            }
            reports.append(report)
            report_refs.append(report)

        latest_comparison = comparisons[-1].model_dump(mode="json") if comparisons else None
        if isinstance(latest_comparison, dict):
            latest_comparison.pop("slice_metrics", None)
            latest_comparison.pop("gate_contract", None)

        projected.append(
            redact_workspace_value(
                {
                    "campaign_id": campaign.campaign_id,
                    "title": campaign.title,
                    "kind": campaign.kind.value,
                    "objective": campaign.objective,
                    "status": campaign.status.value,
                    "version": campaign.version,
                    "updated_at": campaign.updated_at,
                    "target_model": campaign.target_model.model_dump(mode="json"),
                    "manifest_revision": campaign.manifest_revision,
                    "active_study_id": campaign.active_study_id,
                    "active_action_id": campaign.active_action_id,
                    "champion_ref": campaign.champion_ref,
                    "best_development_candidate_ref": campaign.best_development_candidate_ref,
                    "stop_reason": campaign.stop_reason,
                    "budget_remaining": evidence.budget_remaining,
                    "study_status_counts": _status_counts(studies),
                    "attempt_status_counts": _status_counts(attempts),
                    "latest_comparison": latest_comparison,
                    "evidence_digest": evidence.snapshot_digest,
                    "latest_event_cursor": events[-1][0] if events else 0,
                    "reports": reports,
                }
            )
        )
        for cursor, event in events[-10:]:
            activity.append(
                redact_workspace_value(
                    {
                        "event_id": event.event_id,
                        "cursor": cursor,
                        "campaign_id": event.campaign_id,
                        "event_type": event.event_type,
                        "aggregate_version": event.aggregate_version,
                        "payload": event.payload,
                        "actor_id": event.actor_id,
                        "correlation_id": event.correlation_id,
                        "created_at": event.created_at,
                    }
                )
            )

    projected.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    activity.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    report_refs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return projected, activity[:50], report_refs[:25]


def _experiment_project_context(request: Request, workspace_id: str) -> list[dict[str, Any]]:
    """Return bounded project-isolated ledger summaries for agent reconciliation."""

    repository = _campaign_repository(request)
    if repository is None:
        return []
    try:
        from bashgym.ledger.persistence import ExperimentLedgerRepository
        from bashgym.ledger.synthesis import build_project_context

        ledger = ExperimentLedgerRepository(repository.db_path)
        ledger.initialize()
        return [
            build_project_context(
                ledger,
                workspace_id,
                str(project["project_id"]),
                recent_limit=12,
            )
            for project in ledger.list_projects(workspace_id)[:25]
        ]
    except Exception:
        return []


def _settings_readiness() -> list[dict[str, Any]]:
    """Expose provider readiness only; never include masked or raw credentials."""

    try:
        from bashgym.api import settings_routes
        from bashgym.secrets import get_secret_with_source

        env_file_values = settings_routes._read_env_file()
        readiness = []
        for key in sorted(settings_routes.ALLOWED_ENV_KEYS):
            if key == "HF_TOKEN":
                effective, source = get_secret_with_source(key)
                configured = bool(effective)
                safe_source = "environment" if source == "env" else "stored" if configured else ""
            else:
                file_value = env_file_values.get(key, "")
                env_value = os.environ.get(key, "")
                if file_value and not settings_routes._is_placeholder(file_value):
                    configured, safe_source = True, "env_file"
                elif env_value and not settings_routes._is_placeholder(env_value):
                    configured, safe_source = True, "environment"
                else:
                    configured, safe_source = False, ""
            readiness.append(
                {
                    "provider": settings_routes.PROVIDER_META[key]["display_name"],
                    "configured": configured,
                    "source": safe_source,
                }
            )
        return readiness
    except Exception:
        return []


def _skill_lab_runs(workspace_id: str) -> list[dict[str, Any]]:
    try:
        from bashgym.api.skill_lab_routes import _load_run, _storage_root

        runs_dir = _storage_root() / "runs"
        runs = []
        if runs_dir.exists():
            for path in runs_dir.rglob("*.json"):
                run = _load_run(path.stem)
                if run is None or run.workspace_id != workspace_id:
                    continue
                runs.append(
                    redact_workspace_value(
                        {
                            "run_id": run.run_id,
                            "skill_id": run.skill_id,
                            "skill_name": run.skill_name,
                            "skill_revision": run.skill_revision,
                            "endpoint_id": run.endpoint_id,
                            "status": run.status,
                            "created_at": run.created_at,
                            "completed_at": run.completed_at,
                            "progress": run.progress,
                            "kpis": run.kpis,
                            "error": run.error,
                        }
                    )
                )
        runs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return runs[:20]
    except Exception:
        return []


def _hf_context_summary(request: Request, workspace_id: str) -> dict[str, Any] | None:
    """Return only the active bundle's visibility-safe materialized summary."""

    try:
        from bashgym.integrations.huggingface.context_service import HFContextService

        service = getattr(request.app.state, "hf_context_service", None)
        if not isinstance(service, HFContextService):
            return None
        return service.active_summary(workspace_id)
    except Exception:
        return None


def _context_payload(request: Request, workspace_id: str | None = None) -> dict[str, Any]:
    snapshot = _state_snapshot(request, workspace_id)
    events = _state_events(request, snapshot.workspace_id)
    training_runs = _training_runs(request, snapshot.workspace_id)
    runtime_jobs = _runtime_jobs(request)
    campaigns, campaign_activity, report_refs = _campaign_context(
        request, snapshot.workspace_id
    )
    experiment_projects = _experiment_project_context(request, snapshot.workspace_id)
    allowed_actions = list(
        dict.fromkeys(
            [
                *snapshot.allowed_actions,
                "runtime.inspect",
                "campaign.read",
                "campaign.evidence.read",
                "campaign.events.read",
                "campaign.export",
            ]
        )
    )
    generated_at = _now()
    authority = build_context_authority(
        generated_at=generated_at,
        canvas_updated_at=snapshot.updated_at,
        training_runs=training_runs,
        runtime_jobs=runtime_jobs,
        campaigns=campaigns,
    )
    return {
        "schema_version": "bashgym.workspace.context.v2",
        "operator_projection_version": "bashgym.operator.context.v1",
        "generated_at": generated_at,
        "workspace_id": snapshot.workspace_id,
        "workspace_name": snapshot.workspace_name,
        "authority": authority,
        "canvas": redact_workspace_value(snapshot.dict()),
        "training_runs": training_runs,
        "runtime_jobs": runtime_jobs,
        "campaigns": campaigns,
        "experiment_projects": experiment_projects,
        "campaign_activity": campaign_activity,
        "report_refs": report_refs,
        "settings_readiness": _settings_readiness(),
        "skill_lab_runs": _skill_lab_runs(snapshot.workspace_id),
        "huggingface_context": _hf_context_summary(request, snapshot.workspace_id),
        "recent_events": events[-20:],
        "allowed_actions": allowed_actions,
    }


def _as_markdown(payload: dict[str, Any]) -> str:
    authority = payload.get("authority", {})
    canvas = payload.get("canvas", {})
    panels = canvas.get("panels", [])
    edges = canvas.get("edges", [])
    terminals = canvas.get("terminals", [])
    runs = payload.get("training_runs", [])
    runtime_jobs = payload.get("runtime_jobs", [])
    campaigns = payload.get("campaigns", [])
    experiment_projects = payload.get("experiment_projects", [])
    campaign_activity = payload.get("campaign_activity", [])
    reports = payload.get("report_refs", [])
    settings_readiness = payload.get("settings_readiness", [])
    skill_runs = payload.get("skill_lab_runs", [])
    hf_context = payload.get("huggingface_context")
    events = payload.get("recent_events", [])

    lines = [
        "# BashGym Workspace Context",
        "",
        f"Generated: {payload.get('generated_at')}",
        "",
        "## Evidence Authority",
        "- Precedence: live runtime > durable ledger > workspace snapshot > curated GBrain > conversation memory",
        "- Current-state claims must cite the source, evidence ID, and observation time.",
        "- Conversation history is unverified when it disagrees with live or durable evidence.",
    ]
    for source in authority.get("sources", []):
        observed = source.get("observed_at") or "not loaded"
        empty = " (empty observation)" if source.get("empty") else ""
        lines.append(
            f"- source {source.get('rank')} · {source.get('source_id')}: "
            f"{source.get('freshness')} at {observed}{empty}"
        )
    conflicts = authority.get("conflicts", [])
    if conflicts:
        lines.extend(["", "### Evidence Conflicts"])
        for conflict in conflicts:
            lines.append(
                f"- {conflict.get('code')} ({conflict.get('entity_id')}): "
                f"{conflict.get('resolution')}"
            )
    lines.extend([
        "",
        "## Canvas",
        f"- panels: {len(panels)}",
        f"- edges: {len(edges)}",
        f"- terminals: {len(terminals)}",
    ])
    if panels:
        lines.extend(["", "### Panels"])
        for panel in panels:
            bits = [panel.get("type", "unknown"), panel.get("panel_id", "?")]
            if panel.get("terminal_id"):
                bits.append(f"terminal {panel['terminal_id']}")
            lines.append(f"- {panel.get('title') or panel.get('panel_id')}: {' | '.join(bits)}")
    if runs:
        lines.extend(["", "## Training Runs"])
        for run in runs:
            bits = [
                str(run.get("status") or "unknown"),
                str(run.get("strategy") or "unknown"),
            ]
            if run.get("compute_target"):
                bits.append(str(run["compute_target"]))
            lines.append(f"- {run.get('run_id')}: {' | '.join(bits)}")
    if runtime_jobs:
        lines.extend(["", "## Observed Runtime Jobs"])
        for job in runtime_jobs:
            bits = [str(job.get("status") or "unknown"), str(job.get("kind") or "job")]
            if job.get("progress"):
                progress = job["progress"]
                bits.append(
                    f"{progress.get('current', '?')}/{progress.get('total') or '?'} {progress.get('unit') or ''}".strip()
                )
            lines.append(f"- {job.get('title') or job.get('job_id')}: {' | '.join(bits)}")
    if campaigns:
        lines.extend(["", "## Training Sessions / Campaigns"])
        for campaign in campaigns:
            bits = [
                str(campaign.get("status") or "unknown"),
                str(campaign.get("kind") or "general"),
                f"version {campaign.get('version')}",
            ]
            lines.append(
                f"- {campaign.get('title') or campaign.get('campaign_id')}: {' | '.join(bits)}"
            )
            lines.append(f"  goal: {campaign.get('objective')}")
            target = campaign.get("target_model") or {}
            if target:
                lines.append(
                    f"  model/task: {target.get('base_model_ref')} | {target.get('task')}"
                )
            lines.append(f"  budget remaining: {campaign.get('budget_remaining')}")
            lines.append(
                f"  studies: {campaign.get('study_status_counts')} | attempts: {campaign.get('attempt_status_counts')}"
            )
            comparison = campaign.get("latest_comparison")
            if comparison:
                lines.append(
                    f"  latest comparison: {comparison.get('verdict')} | {comparison.get('metrics')}"
                )
    if experiment_projects:
        lines.extend(["", "## Experiment Projects"])
        for project_context in experiment_projects:
            project = project_context.get("project") or {}
            inventory = project_context.get("inventory") or {}
            health = project_context.get("health") or {}
            lines.append(
                f"- {project.get('display_name') or project_context.get('project_id')}: "
                f"project {project_context.get('project_id')} | health {health.get('status')} | "
                f"{inventory.get('run_count', 0)} runs | {inventory.get('evaluation_count', 0)} evals | "
                f"{inventory.get('decision_count', 0)} decisions"
            )
            for run in (project_context.get("recent_runs") or [])[:5]:
                lines.append(
                    f"  run {run.get('run_id')}: {run.get('training_method')} | "
                    f"{run.get('status')} | experiment {run.get('experiment_id')}"
                )
            for evaluation in (project_context.get("recent_evaluations") or [])[:5]:
                lines.append(
                    f"  eval {evaluation.get('evaluation_result_id')}: suite "
                    f"{evaluation.get('evaluation_suite_id')} | run {evaluation.get('run_id')} | "
                    f"{evaluation.get('status')}"
                )
            for decision in (project_context.get("recent_decisions") or [])[:5]:
                lines.append(
                    f"  decision {decision.get('decision_id')}: {decision.get('outcome')}"
                )
    if reports:
        lines.extend(["", "## Reports"])
        for report in reports[:10]:
            names = [item.get("name") for item in report.get("files", []) if item.get("name")]
            lines.append(
                f"- {report.get('export_id')} ({report.get('campaign_id')}): {', '.join(names) or report.get('formats')}"
            )
    if campaign_activity:
        lines.extend(["", "## Durable Campaign Activity"])
        for event in campaign_activity[:12]:
            lines.append(
                f"- [{event.get('cursor')}] {event.get('event_type')} ({event.get('campaign_id')})"
            )
    if settings_readiness:
        lines.extend(["", "## Provider Readiness"])
        for provider in settings_readiness:
            state = "configured" if provider.get("configured") else "not configured"
            source = f" via {provider.get('source')}" if provider.get("source") else ""
            lines.append(f"- {provider.get('provider')}: {state}{source}")
    if skill_runs:
        lines.extend(["", "## Skill Lab Runs"])
        for run in skill_runs:
            verdict = (run.get("kpis") or {}).get("verdict")
            bits = [str(run.get("status") or "unknown")]
            if verdict:
                bits.append(str(verdict))
            lines.append(
                f"- {run.get('skill_name') or run.get('skill_id')}: {' | '.join(bits)}"
            )
    if hf_context:
        lines.extend(
            [
                "",
                "## Hugging Face Context",
                f"- bundle: {hf_context.get('bundle_id')} v{hf_context.get('version')}",
                f"- intent: {hf_context.get('intent')}",
                f"- state: {hf_context.get('lifecycle')} / {hf_context.get('freshness')}",
                f"- evidence: {hf_context.get('evidence_counts')}",
            ]
        )
    if events:
        lines.extend(["", "## Recent Workspace Events"])
        for event in events[-8:]:
            lines.append(
                f"- {event.get('type')}: {event.get('title') or event.get('summary') or event.get('event_id')}"
            )
    lines.extend(["", "## Allowed Actions"])
    for action in payload.get("allowed_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines)


@router.get("/context")
async def get_workspace_context(
    request: Request,
    format: str = Query("json", pattern="^(json|markdown)$"),
    workspace_id: str | None = Query(None),
):
    """Return the sanitized workspace context for local agents.

    Without workspace_id, the most recently updated workspace is returned so
    single-workspace clients keep working unchanged.
    """
    payload = _context_payload(request, workspace_id)
    if format == "markdown":
        return PlainTextResponse(_as_markdown(payload), media_type="text/markdown")
    return payload


@router.put("/canvas/snapshot")
async def put_workspace_canvas_snapshot(request: Request, snapshot: WorkspaceCanvasSnapshot):
    """Store the renderer-owned sanitized canvas snapshot (keyed by workspace)."""
    safe_snapshot = WorkspaceCanvasSnapshot(**redact_workspace_value(snapshot.dict()))
    safe_snapshot.updated_at = _now()
    _snapshot_store(request)[safe_snapshot.workspace_id] = safe_snapshot
    await broadcast_workspace_context_updated(
        {
            "updated_at": safe_snapshot.updated_at,
            "workspace_id": safe_snapshot.workspace_id,
            "workspace_name": safe_snapshot.workspace_name,
            "panels": len(safe_snapshot.panels),
            "edges": len(safe_snapshot.edges),
            "terminals": len(safe_snapshot.terminals),
        }
    )
    return {"ok": True, "updated_at": safe_snapshot.updated_at}


@router.post("/events")
async def post_workspace_event(request: Request, event: WorkspaceEvent):
    """Accept a semantic workspace/canvas intent from an agent or local tool."""
    payload = redact_workspace_value(event.dict())
    payload["event_id"] = f"workspace_evt_{uuid4().hex[:12]}"
    payload["received_at"] = _now()
    if not payload.get("correlation_id"):
        payload["correlation_id"] = f"intent_{uuid4().hex[:12]}"

    events = _state_events(request, event.workspace_id or "default")
    events.append(payload)
    del events[:-100]

    await broadcast_workspace_canvas_intent(payload)
    return {"ok": True, "event": payload}
