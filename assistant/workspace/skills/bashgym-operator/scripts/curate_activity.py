#!/usr/bin/env python3
"""Create idempotent, bounded GBrain receipts from BashGym evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "bashgym.activity.v1"
HANDOFF_SCHEMA = "bashgym.session-handoff.v1"
SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|bearer|password|secret|token)", re.I)
HIGH_VOLUME_KEY = re.compile(
    r"(?:raw[_-]?logs?|log[_-]?lines?|transcripts?|rows|samples|tensor|checkpoint[_-]?content)",
    re.I,
)
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
UNC_ABSOLUTE = re.compile(r"^(?:\\\\|//)[^\\/]+[\\/][^\\/]+")
EMBEDDED_ABSOLUTE = re.compile(
    r"(?:"
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>]+"
    r"|(?<![A-Za-z0-9_\\])\\\\[^\\/\s\"'<>]+[\\/][^\s\"'<>]+"
    r"|(?<![A-Za-z0-9_/:])//[^\\/\s\"'<>]+[\\/][^\s\"'<>]+"
    r"|(?<![A-Za-z0-9_/])/(?!/)(?:[^/\s\"'<>]+/)*[^/\s\"'<>]+"
    r")"
)
PATH_TRAILING_PUNCTUATION = ".,;:!?)]}"
SLUG_PART = re.compile(r"[^a-z0-9._-]+")
MAX_STRING = 2_000
MAX_ITEMS = 50


def _slug(value: Any, fallback: str) -> str:
    text = SLUG_PART.sub("-", str(value or "").strip().casefold()).strip("-.")
    return (text or fallback)[:120]


def _looks_local_path(value: str) -> bool:
    if value.startswith("file://") or WINDOWS_ABSOLUTE.match(value) or UNC_ABSOLUTE.match(value):
        return True
    if not value.startswith("/"):
        return False
    route_prefixes = (
        "/api/",
        "/factory/",
        "/models/",
        "/system/",
        "/traces/",
        "/training/",
    )
    return value not in {"/health", "/stats"} and not value.startswith(route_prefixes)


def _local_artifact_name(value: str) -> str:
    path = value.removeprefix("file://")
    if WINDOWS_ABSOLUTE.match(path) or UNC_ABSOLUTE.match(path):
        return PureWindowsPath(path).name or "reference"
    return PurePosixPath(path).name or "reference"


def _redact_embedded_paths(value: str) -> str:
    def replace_path(match: re.Match[str]) -> str:
        candidate = match.group(0)
        path = candidate.rstrip(PATH_TRAILING_PUNCTUATION)
        suffix = candidate[len(path) :]
        if not _looks_local_path(path):
            return candidate
        return f"[local artifact: {_local_artifact_name(path)}]{suffix}"

    return EMBEDDED_ABSOLUTE.sub(replace_path, value)


def _safe(value: Any, *, key: str = "") -> Any:
    if SECRET_KEY.search(key):
        return "[redacted]"
    if HIGH_VOLUME_KEY.search(key):
        return "[retained in BashGym]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        cleaned = "".join(char for char in value if char in "\n\t" or ord(char) >= 32)
        if _looks_local_path(cleaned):
            return f"[local artifact: {_local_artifact_name(cleaned)}]"
        return _redact_embedded_paths(cleaned[:MAX_STRING])
    if isinstance(value, dict):
        return {
            str(item_key)[:120]: _safe(item, key=str(item_key))
            for item_key, item in list(value.items())[:MAX_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item, key=key) for item in list(value)[:MAX_ITEMS]]
    return _safe(str(value), key=key)


def _load(path: str) -> dict[str, Any]:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("input must be a JSON object")
    return value


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(_safe(value), indent=2, sort_keys=True) + "\n```"


def _kind_directory(kind: str) -> str:
    return {
        "training-session": "training-sessions",
        "run-inventory": "run-inventories",
        "evaluation": "evaluations",
        "decision": "decisions",
        "dataset": "datasets",
        "model": "models",
        "run": "runs",
        "session-handoff": "session-handoffs",
    }.get(kind, f"{kind}s")


def _list_section(title: str, values: Any) -> list[str]:
    if not values:
        return []
    items = values if isinstance(values, list) else [values]
    return [f"## {title}", "", *[f"- {_safe(item)}" for item in items], ""]


def render_receipt(receipt: dict[str, Any]) -> tuple[str, str, str, str]:
    if receipt.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    workspace_id = str(receipt.get("workspace_id") or "").strip()
    entity_id = str(receipt.get("entity_id") or "").strip()
    kind = str(receipt.get("kind") or "").strip()
    project_id = str(receipt.get("project_id") or "").strip()
    if not workspace_id or not entity_id or not kind:
        raise ValueError("kind, workspace_id, and entity_id are required")

    safe = _safe(receipt)
    occurred_at = str(safe.get("occurred_at") or "")
    status = str(safe.get("status") or "unknown")
    title = str(safe.get("title") or safe.get("objective") or entity_id)
    lines = [
        "---",
        f"type: {json.dumps('bashgym-' + _slug(kind, 'activity'))}",
        f"schema_version: {json.dumps(SCHEMA)}",
        f"workspace_id: {json.dumps(workspace_id)}",
        f"project_id: {json.dumps(project_id)}",
        f"entity_id: {json.dumps(entity_id)}",
        f"status: {json.dumps(status)}",
        f"occurred_at: {json.dumps(occurred_at)}",
        'privacy: "full-tier-only"',
        'source: "bashgym"',
        "---",
        "",
        f"# {title}",
        "",
    ]
    if safe.get("objective"):
        lines.extend(["## Objective", "", str(safe["objective"]), ""])
    if safe.get("summary"):
        lines.extend(["## Summary", "", str(safe["summary"]), ""])
    for heading, key in (
        ("Configuration and lineage", "configuration"),
        ("KPI snapshot", "metrics"),
        ("Artifact references", "artifact_references"),
        ("Relationships", "relationships"),
    ):
        if safe.get(key):
            lines.extend([f"## {heading}", "", _json_block(safe[key]), ""])
    if safe.get("decision"):
        lines.extend(["## Decision", "", str(safe["decision"]), ""])
    lines.extend(_list_section("Limitations", safe.get("limitations")))
    lines.extend(_list_section("Follow-up", safe.get("follow_up")))
    if safe.get("source_event_ids"):
        lines.extend(["## Provenance", "", _json_block(safe["source_event_ids"]), ""])
    content = "\n".join(lines).rstrip() + "\n"
    return _slug(kind, "activity"), _slug(workspace_id, "default"), _slug(entity_id, "entity"), content


def _write_receipt(root: Path, receipt: dict[str, Any]) -> tuple[str, bool]:
    kind, workspace, entity, content = render_receipt(receipt)
    relative = Path(_kind_directory(kind)) / workspace / f"{entity}.md"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_text(encoding="utf-8") == content:
        return relative.as_posix(), False
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)
    return relative.as_posix(), True


def _campaign_receipt(context: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
    workspace_id = str(context.get("workspace_id") or "").strip()
    if not workspace_id:
        raise ValueError("context workspace_id is required")
    campaign_id = str(campaign.get("campaign_id") or "campaign")
    events = [
        event
        for event in context.get("campaign_activity") or []
        if isinstance(event, dict) and event.get("campaign_id") == campaign_id
    ]
    comparison = campaign.get("latest_comparison") or {}
    reports = campaign.get("reports") or []
    status = str(campaign.get("status") or "unknown")
    if status in {"completed", "exhausted", "failed", "cancelled"}:
        follow_up = ["Consolidate the final decision and verify the report package."]
    elif status == "active":
        follow_up = ["Monitor the active action and evaluate the next durable milestone."]
    elif status == "paused":
        follow_up = ["Resolve the pause reason before resuming compute."]
    else:
        follow_up = ["Complete preflight and perform only the next authorized campaign action."]
    decision = None
    if isinstance(comparison, dict) and comparison.get("verdict"):
        decision = f"Latest development comparison verdict: {comparison['verdict']}."
    return {
        "schema_version": SCHEMA,
        "kind": "training-session",
        "workspace_id": workspace_id,
        "entity_id": campaign_id,
        "title": campaign.get("title") or campaign_id,
        "status": status,
        "occurred_at": campaign.get("updated_at") or context.get("generated_at"),
        "objective": campaign.get("objective"),
        "summary": (
            f"{campaign.get('kind') or 'general'} session at version {campaign.get('version')}; "
            f"durable cursor {campaign.get('latest_event_cursor') or 0}."
        ),
        "configuration": {
            "profile": campaign.get("kind"),
            "target_model": campaign.get("target_model"),
            "manifest_revision": campaign.get("manifest_revision"),
            "champion_ref": campaign.get("champion_ref"),
            "best_development_candidate_ref": campaign.get(
                "best_development_candidate_ref"
            ),
        },
        "metrics": {
            "budget_remaining": campaign.get("budget_remaining"),
            "study_status_counts": campaign.get("study_status_counts"),
            "attempt_status_counts": campaign.get("attempt_status_counts"),
            "latest_comparison": comparison,
        },
        "artifact_references": reports,
        "relationships": [
            {"relation": "active-study", "target_id": campaign.get("active_study_id")},
            {"relation": "active-action", "target_id": campaign.get("active_action_id")},
        ],
        "decision": decision,
        "limitations": [campaign.get("stop_reason")] if campaign.get("stop_reason") else [],
        "follow_up": follow_up,
        "source_event_ids": [event.get("event_id") for event in events if event.get("event_id")],
    }


def _run_receipt(context: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    workspace_id = str(context.get("workspace_id") or "").strip()
    if not workspace_id:
        raise ValueError("context workspace_id is required")
    return {
        "schema_version": SCHEMA,
        "kind": "run",
        "workspace_id": workspace_id,
        "entity_id": run.get("run_id") or "run",
        "status": run.get("status") or "unknown",
        "occurred_at": run.get("completed_at") or run.get("started_at") or context.get("generated_at"),
        "objective": "Execute a BashGym training run under the current workspace goal.",
        "summary": f"{run.get('strategy') or 'training'} run on {run.get('compute_target') or 'unspecified compute'}.",
        "configuration": run.get("config") or {},
        "metrics": run.get("metrics") or {},
        "relationships": [
            {"relation": "correlation", "target_id": run.get("correlation_id")}
        ],
        "follow_up": ["Evaluate and compare the run before any promotion decision."],
    }


def _handoff_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != HANDOFF_SCHEMA:
        raise ValueError(f"schema_version must be {HANDOFF_SCHEMA}")
    required = ("workspace_id", "project_id", "session_id", "updated_at", "summary")
    missing = [key for key in required if not str(payload.get(key) or "").strip()]
    if missing:
        raise ValueError(f"handoff is missing required fields: {', '.join(missing)}")
    return {
        "schema_version": SCHEMA,
        "kind": "session-handoff",
        "workspace_id": payload["workspace_id"],
        "project_id": payload["project_id"],
        "entity_id": payload["session_id"],
        "title": payload.get("title") or f"Session handoff — {payload['project_id']}",
        "status": payload.get("status") or "active",
        "occurred_at": payload["updated_at"],
        "objective": payload.get("objective"),
        "summary": payload["summary"],
        "configuration": {
            "channel": payload.get("channel") or "coding-agent",
            "branch": payload.get("branch"),
            "workspace_revision": payload.get("workspace_revision"),
        },
        "metrics": payload.get("kpis") or {},
        "artifact_references": payload.get("evidence_refs") or [],
        "decision": "; ".join(str(item) for item in payload.get("decisions") or []),
        "limitations": payload.get("limitations") or [],
        "follow_up": payload.get("next_actions") or [],
        "source_event_ids": payload.get("source_event_ids") or [],
    }


def _run(args: argparse.Namespace) -> int:
    root = Path(args.output_root).expanduser().resolve()
    payload = _load(args.input)
    receipts: list[dict[str, Any]]
    if args.command == "receipt":
        receipts = [payload]
    elif args.command == "handoff":
        receipts = [_handoff_receipt(payload)]
    else:
        if not str(payload.get("schema_version") or "").startswith("bashgym.workspace.context."):
            raise ValueError("context input must be a BashGym workspace context projection")
        if not str(payload.get("workspace_id") or "").strip():
            raise ValueError("context workspace_id is required")
        receipts = [
            _campaign_receipt(payload, item)
            for item in payload.get("campaigns") or []
            if isinstance(item, dict)
        ]
        terminal = {"completed", "failed", "cancelled"}
        receipts.extend(
            _run_receipt(payload, item)
            for item in payload.get("training_runs") or []
            if isinstance(item, dict)
            and (args.include_active_runs or str(item.get("status") or "") in terminal)
        )

    written: list[str] = []
    unchanged: list[str] = []
    for receipt in receipts:
        relative, changed = _write_receipt(root, receipt)
        (written if changed else unchanged).append(relative)
    print(
        json.dumps(
            {
                "schema_version": "bashgym.activity.curator-result.v1",
                "written": written,
                "unchanged": unchanged,
                "receipt_count": len(receipts),
            },
            sort_keys=True,
        )
    )
    return 0


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("receipt", "context", "handoff"):
        child = subparsers.add_parser(command)
        child.add_argument("input", help="JSON file path or - for stdin")
        child.add_argument("--output-root", required=True)
        if command == "context":
            child.add_argument("--include-active-runs", action="store_true")
        else:
            child.set_defaults(include_active_runs=False)
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc), "schema_version": SCHEMA}), file=sys.stderr)
        return 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
