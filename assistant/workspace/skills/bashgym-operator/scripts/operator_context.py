#!/usr/bin/env python3
"""Discover BashGym operator abilities and emit compact, safe local context."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bashgym.api_base import normalize_api_base, open_api_url


@dataclass(frozen=True)
class OperatorPaths:
    bashgym_root: Path
    training_root: Path
    observer: Path
    gbrain: Path
    activity_root: Path
    api_base_url: str
    ledger_db: Path | None = None

    @classmethod
    def from_environment(cls) -> OperatorPaths:
        home = Path.home()
        return cls(
            bashgym_root=Path(os.environ.get("BASHGYM_ROOT", home / "bashgym")).expanduser(),
            training_root=Path(
                os.environ.get("BASHGYM_TRAINING_ROOT", home / "bashgym-training")
            ).expanduser(),
            observer=Path(
                os.environ.get("BASHGYM_RUN_OBSERVER", home / ".hermes/training_runs.py")
            ).expanduser(),
            gbrain=Path(os.environ.get("GBRAIN_BIN", home / "gbrain/bin/gbrain")).expanduser(),
            activity_root=Path(
                os.environ.get("BASHGYM_ACTIVITY_ROOT", home / "brain-sources/bashgym-activity")
            ).expanduser(),
            # BASHGYM_API_BASE_URL is a deprecated compatibility fallback.
            api_base_url=normalize_api_base(
                os.environ.get("BASHGYM_API_BASE")
                or os.environ.get("BASHGYM_API_BASE_URL")
                or "http://127.0.0.1:8003/api"
            ),
            ledger_db=Path(
                os.environ.get(
                    "BASHGYM_LEDGER_DB",
                    home / ".bashgym/campaigns/campaigns.sqlite3",
                )
            ).expanduser(),
        )

    @property
    def experiment_ledger_db(self) -> Path:
        return self.ledger_db or Path.home() / ".bashgym/campaigns/campaigns.sqlite3"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "name",
        "source",
        "status",
        "strategy",
        "model",
        "compute_target",
        "started_at",
        "updated_at",
        "progress",
        "metrics",
        "latest_checkpoint",
    )
    compact = {key: run[key] for key in fields if key in run and run[key] not in (None, {}, [])}
    tracking = run.get("tracking_ids")
    if isinstance(tracking, dict):
        compact["tracking_ids"] = {
            key: tracking[key]
            for key in (
                "workspace_id",
                "project_id",
                "experiment_id",
                "run_id",
                "attempt_id",
                "context_status",
            )
            if key in tracking and tracking[key] not in (None, "")
        }
    return compact


def _runtime_summary(runs: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [str(run.get("status") or "").casefold() for run in runs]
    return {
        "count": len(runs),
        "active": sum(status in {"queued", "running", "paused"} for status in statuses),
        "running": statuses.count("running"),
        "failed": statuses.count("failed"),
    }


def _observe_runs(
    observer: Path,
    *,
    workspace_id: str,
    project_id: str | None,
    limit: int,
) -> tuple[dict[str, Any], str | None]:
    scope = {
        "workspace_id": workspace_id,
        "project_id": project_id,
        "excluded_other_scope": 0,
        "excluded_unscoped": 0,
    }
    if not observer.is_file():
        return {"summary": _runtime_summary([]), "runs": [], "scope": scope}, (
            f"observer not found: {observer}"
        )
    try:
        spec = importlib.util.spec_from_file_location("hermes_training_runs", observer)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        snapshot = module.discover_training_runs()
    except Exception as exc:  # pragma: no cover - defensive host integration boundary
        return {"summary": _runtime_summary([]), "runs": [], "scope": scope}, (
            f"observer failed: {type(exc).__name__}: {exc}"
        )
    runs = snapshot.get("runs", []) if isinstance(snapshot, dict) else []
    selected: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        tracking = run.get("tracking_ids")
        if not isinstance(tracking, dict):
            scope["excluded_unscoped"] += 1
            continue
        run_workspace = str(tracking.get("workspace_id") or "").strip()
        run_project = str(tracking.get("project_id") or "").strip()
        if not run_workspace or not run_project:
            scope["excluded_unscoped"] += 1
            continue
        if project_id is None or run_workspace != workspace_id or run_project != project_id:
            scope["excluded_other_scope"] += 1
            continue
        selected.append(run)
    return {
        "schema_version": snapshot.get("schema_version"),
        "observed_at": snapshot.get("observed_at"),
        "summary": _runtime_summary(selected),
        "runs": [_compact_run(run) for run in selected[:limit]],
        "scope": scope,
    }, None


def _require_nonblank(value: str | None, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    return normalized


def _project_context(
    paths: OperatorPaths,
    *,
    workspace_id: str,
    project_id: str | None,
    limit: int,
) -> dict[str, Any]:
    """Combine live runtime state with project-isolated durable ledger evidence."""

    workspace_id = str(_require_nonblank(workspace_id, "workspace_id"))
    project_id = _require_nonblank(project_id, "project_id", optional=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    runtime, runtime_error = _observe_runs(
        paths.observer,
        workspace_id=workspace_id,
        project_id=project_id,
        limit=limit,
    )
    ledger: dict[str, Any] | None = None
    ledger_error: str | None = None
    try:
        ledger = _ledger_context(
            paths,
            workspace_id=workspace_id,
            project_id=project_id,
            limit=limit,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"

    available_projects = []
    if isinstance(ledger, dict):
        if project_id:
            available_projects = [ledger.get("project") or {"project_id": project_id}]
        else:
            available_projects = list(ledger.get("projects") or [])
    conflicts: list[dict[str, Any]] = []
    if project_id is None and len(available_projects) > 1:
        conflicts.append(
            {
                "code": "project_selection_required",
                "project_ids": [item.get("project_id") for item in available_projects],
                "decision_required": True,
                "resolution": "Select one project ID before loading project-local evidence.",
            }
        )
    return {
        "schema_version": "bashgym.operator.project-context.v2",
        "generated_at": generated_at,
        "workspace_id": workspace_id,
        "project_id": project_id,
        "authority": {
            "schema_version": "bashgym.context-authority.v1",
            "source_precedence": [
                "live_runtime",
                "durable_ledger",
                "project_local_evidence",
                "curated_gbrain",
                "conversation_memory",
            ],
            "sources": [
                {
                    "source_id": "live_runtime",
                    "freshness": "fresh" if runtime_error is None else "unavailable",
                    "observed_at": runtime.get("observed_at") or generated_at,
                    "error": runtime_error,
                },
                {
                    "source_id": "durable_ledger",
                    "freshness": "fresh" if ledger is not None else "unavailable",
                    "observed_at": generated_at if ledger is not None else None,
                    "error": ledger_error,
                },
                {
                    "source_id": "project_local_evidence",
                    "freshness": "not_loaded",
                    "observed_at": None,
                    "instruction": "Load project-specific evidence through a local profile.",
                },
                {
                    "source_id": "curated_gbrain",
                    "freshness": "not_loaded",
                    "observed_at": None,
                    "instruction": "Query the authoritative GBrain source explicitly.",
                },
                {
                    "source_id": "conversation_memory",
                    "freshness": "unverified",
                    "observed_at": None,
                },
            ],
            "conflicts": conflicts,
            "decision_required": any(item["decision_required"] for item in conflicts),
        },
        "runtime": runtime,
        "ledger": ledger,
        "task_profile": None,
        "selection": {
            "available_projects": available_projects,
            "requires_project": project_id is None,
        },
        "rules": [
            "Use runtime for current process state and the ledger for durable identities and decisions.",
            "Load task-specific evidence only through an explicitly selected local profile.",
            "Do not infer project identity from a conversation or task-specific artifact directory.",
            "Report source timestamps and conflicts with every current-state synthesis.",
        ],
    }


def _campaign_cli(paths: OperatorPaths) -> str | None:
    executable = shutil.which("bashgym")
    if executable:
        command = [executable]
        if _supports_campaign(command, paths.bashgym_root):
            return shlex.join(command)
    cli = paths.bashgym_root / "bashgym/cli.py"
    if cli.is_file():
        python = paths.bashgym_root / "venv/bin/python"
        command = [str(python if python.is_file() else sys.executable), str(cli)]
        if _supports_campaign(command, paths.bashgym_root):
            return shlex.join(command)
    return None


def _training_cli(paths: OperatorPaths) -> str | None:
    candidates: list[list[str]] = []
    executable = shutil.which("bashgym")
    if executable:
        candidates.append([executable])
    cli = paths.bashgym_root / "bashgym/cli.py"
    if cli.is_file():
        python = paths.bashgym_root / "venv/bin/python"
        candidates.append([str(python if python.is_file() else sys.executable), str(cli)])
    for command in candidates:
        if _supports_training_start(command, paths.bashgym_root):
            return shlex.join(command)
    return None


def _supports_campaign(command: list[str], working_directory: Path) -> bool:
    try:
        result = subprocess.run(
            [*command, "campaign", "--help"],
            cwd=working_directory if working_directory.is_dir() else None,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "campaign" in (result.stdout + result.stderr).casefold()


def _supports_training_start(command: list[str], working_directory: Path) -> bool:
    try:
        result = subprocess.run(
            [*command, "training", "start", "--help"],
            cwd=working_directory if working_directory.is_dir() else None,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = (result.stdout + result.stderr).casefold()
    return result.returncode == 0 and "artifact-retention" in output


def _hf_jobs_cli() -> str | None:
    candidates = [shutil.which("hf")]
    candidates.extend(
        [
            str(Path.home() / ".local/bin/hf"),
            str(Path(sys.executable).with_name("hf")),
            str(Path(sys.executable).with_name("hf.exe")),
        ]
    )
    executable = next(
        (candidate for candidate in candidates if candidate and Path(candidate).is_file()),
        None,
    )
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "jobs", "--help"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout + result.stderr).casefold()
    if result.returncode != 0 or "job" not in output:
        return None
    return shlex.join([executable, "jobs"])


def _hf_token_configured() -> bool:
    if os.environ.get("HF_TOKEN", "").strip():
        return True
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser()
    return (hf_home / "token").is_file()


def _hf_cli_check(command: str | None, *arguments: str) -> bool:
    if not command:
        return False
    executable = shlex.split(command)[0]
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _api_health(base_url: str) -> bool:
    health_path = "/health" if base_url.endswith("/api") else "/api/health"
    try:
        with open_api_url(f"{base_url}{health_path}", timeout=0.75) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _bundle_integrity(root: Path | None = None) -> dict[str, Any]:
    bundle_root = root or Path(__file__).resolve().parents[1]
    lock_path = bundle_root / "bundle.lock.json"
    lock = _read_json(lock_path)
    schema_version = lock.get("schema_version") if lock else None
    if schema_version not in {
        "bashgym.operator-bundle-lock.v1",
        "bashgym.operator-bundle-lock.v2",
    }:
        return {
            "verified": False,
            "lock_available": False,
            "mismatches": ["bundle.lock.json"],
        }
    mismatches: list[str] = []
    for relative, expected in (lock.get("files") or {}).items():
        candidate = (bundle_root / str(relative)).resolve()
        try:
            candidate.relative_to(bundle_root.parent.resolve())
        except ValueError:
            mismatches.append(str(relative))
            continue
        if not candidate.is_file():
            mismatches.append(str(relative))
            continue
        if schema_version == "bashgym.operator-bundle-lock.v1":
            payload = candidate.read_bytes()
        else:
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                mismatches.append(str(relative))
                continue
            payload = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            mismatches.append(str(relative))
    return {
        "verified": not mismatches,
        "lock_available": True,
        "mismatches": sorted(mismatches),
    }


def _doctor(paths: OperatorPaths) -> dict[str, Any]:
    api_reachable = _api_health(paths.api_base_url)
    campaign_cli = _campaign_cli(paths)
    training_cli = _training_cli(paths)
    hf_jobs_cli = _hf_jobs_cli()
    hf_token_configured = _hf_token_configured()
    hf_identity_verified = _hf_cli_check(hf_jobs_cli, "auth", "whoami")
    hf_jobs_access = hf_identity_verified and _hf_cli_check(hf_jobs_cli, "jobs", "hardware")
    bundle_integrity = _bundle_integrity()
    return {
        "schema_version": "bashgym.operator.doctor.v1",
        "host": socket.gethostname(),
        "sources": {
            "run_observer": {"path": str(paths.observer), "available": paths.observer.is_file()},
            "bashgym_checkout": {
                "path": str(paths.bashgym_root),
                "available": paths.bashgym_root.is_dir(),
            },
            "training_artifacts": {
                "path": str(paths.training_root),
                "available": paths.training_root.is_dir(),
            },
            "gbrain": {"path": str(paths.gbrain), "available": paths.gbrain.is_file()},
            "gbrain_activity": {
                "path": str(paths.activity_root),
                "available": paths.activity_root.is_dir(),
                "source_id": "bashgym-activity",
                "requires_explicit_source": True,
            },
            "desktop_api": {"base_url": paths.api_base_url, "reachable": api_reachable},
            "experiment_ledger": {
                "available": paths.experiment_ledger_db.is_file(),
                "source": "local BashGym campaign database",
            },
            "critical_skill_integrity": bundle_integrity,
            "campaign_cli": {"command": campaign_cli, "available": campaign_cli is not None},
            "training_cli": {"command": training_cli, "available": training_cli is not None},
            "hf_jobs_cli": {"command": hf_jobs_cli, "available": hf_jobs_cli is not None},
            "hf_credentials": {
                "configured": hf_token_configured,
                "verified": hf_identity_verified,
                "jobs_access": hf_jobs_access,
                "note": "Identity and Jobs access checks expose no token or account value.",
            },
        },
        "abilities": {
            "inspect_local_runs": paths.observer.is_file(),
            "search_curated_activity": paths.gbrain.is_file() and paths.activity_root.is_dir(),
            "mutate_desktop_campaign": campaign_cli is not None or api_reachable,
            "launch_general_training": training_cli is not None or api_reachable,
            "read_experiment_ledger": paths.experiment_ledger_db.is_file(),
            "launch_hf_jobs": hf_jobs_cli is not None
            and hf_token_configured
            and hf_identity_verified
            and hf_jobs_access,
        },
        "activation_lanes": {
            "same_device": {
                "supported": False,
                "ready": False,
                "surface": "registered SSH device",
                "requires": "register localhost SSH when the training hardware is this machine",
            },
            "private_ssh": {
                "controller_ready": training_cli is not None or api_reachable,
                "surface": "BashGym device preflight plus training CLI/API",
                "requires": "registered device and successful per-device preflight",
            },
            "huggingface_jobs": {
                "ready": hf_jobs_cli is not None
                and hf_token_configured
                and hf_identity_verified
                and hf_jobs_access,
                "surface": "Hugging Face Jobs CLI",
                "requires": "paid Jobs access, verified write token, remote dataset/script, Hub persistence",
            },
            "managed_provider": {
                "controller_ready": api_reachable,
                "surface": "/api/training/managed/submit",
                "requires": "connected provider credentials and provider-compatible dataset/model",
            },
        },
        "notes": [
            "Stop before mutation or compute launch when critical_skill_integrity.verified is false.",
            "Canvas workspace context is prompt-injected and cannot be probed by this local helper.",
            "Launch availability means a BashGym control surface is reachable; preflight and user authority still apply.",
            "General training launch requires the current BashGym training CLI or a reachable desktop API.",
            "A SkyPilot/dstack compute launch plan is a dry run, not an active training job.",
            "Hugging Face Jobs must push durable artifacts to the Hub before its ephemeral job exits.",
        ],
    }


def _ledger_context(
    paths: OperatorPaths,
    *,
    workspace_id: str,
    project_id: str | None,
    limit: int,
) -> dict[str, Any]:
    workspace_id = str(_require_nonblank(workspace_id, "workspace_id"))
    project_id = _require_nonblank(project_id, "project_id", optional=True)
    if not paths.experiment_ledger_db.is_file():
        raise FileNotFoundError("BashGym experiment ledger is not available on this device")
    root_text = str(paths.bashgym_root)
    if paths.bashgym_root.is_dir() and root_text not in sys.path:
        sys.path.insert(0, root_text)
    from bashgym.ledger.persistence import ExperimentLedgerRepository
    from bashgym.ledger.synthesis import build_project_context

    repository = ExperimentLedgerRepository(paths.experiment_ledger_db)
    repository.initialize()
    if project_id is None:
        return {
            "schema_version": "experiment_projects.v1",
            "workspace_id": workspace_id,
            "projects": repository.list_projects(workspace_id),
            "database_health": repository.database_health(workspace_id),
        }
    return build_project_context(
        repository,
        workspace_id,
        project_id,
        recent_limit=max(1, min(limit, 100)),
    )


def _workspace_context(paths: OperatorPaths, workspace_id: str, format_name: str) -> Any:
    workspace_id = str(_require_nonblank(workspace_id, "workspace_id"))
    query = urllib.parse.urlencode({"workspace_id": workspace_id, "format": format_name})
    api_path = "/workspace/context" if paths.api_base_url.endswith("/api") else "/api/workspace/context"
    url = f"{paths.api_base_url}{api_path}?{query}"
    with open_api_url(url, timeout=3) as response:
        body = response.read().decode("utf-8")
    if format_name == "json":
        return json.loads(body)
    return body


def _nonblank_argument(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("identifier cannot be blank")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Verify exact local ability sources without mutation.")
    context = subparsers.add_parser("context", help="Emit compact local project evidence.")
    context.add_argument("--project", type=_nonblank_argument)
    context.add_argument("--workspace-id", required=True, type=_nonblank_argument)
    context.add_argument("--limit", type=int, default=8)
    workspace = subparsers.add_parser("workspace", help="Read the desktop workspace projection.")
    workspace.add_argument("--workspace-id", required=True, type=_nonblank_argument)
    workspace.add_argument("--format", choices=["json", "markdown"], default="json")
    workspace.add_argument("--api-base")
    ledger = subparsers.add_parser(
        "ledger", help="Read project-isolated local experiment context without an MCP server."
    )
    ledger.add_argument("--workspace-id", required=True, type=_nonblank_argument)
    ledger.add_argument("--project-id", type=_nonblank_argument)
    ledger.add_argument("--limit", type=int, default=20)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = OperatorPaths.from_environment()
    if getattr(args, "api_base", None):
        paths = replace(paths, api_base_url=normalize_api_base(args.api_base))
    try:
        if args.command == "doctor":
            result: Any = _doctor(paths)
        elif args.command == "context":
            result = _project_context(
                paths,
                workspace_id=args.workspace_id,
                project_id=args.project,
                limit=max(1, min(args.limit, 20)),
            )
        elif args.command == "workspace":
            result = _workspace_context(paths, args.workspace_id, args.format)
        else:
            result = _ledger_context(
                paths,
                workspace_id=args.workspace_id,
                project_id=args.project_id,
                limit=args.limit,
            )
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
