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
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "bashgym.operator.local-context.v1"
MEMEXAI_DIRECTORY = "memexai-positive-aware-v1-20260712"
KEY_METRICS = (
    "exact_chunk_mrr",
    "exact_chunk_ndcg_at_10",
    "exact_chunk_recall_at_10",
    "local_window_mrr",
    "local_window_ndcg_at_10",
    "wrong_top_video_rate",
    "hard_negative_win_rate",
)


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
            api_base_url=os.environ.get("BASHGYM_API_BASE_URL", "http://127.0.0.1:8003").rstrip(
                "/"
            ),
            ledger_db=Path(
                os.environ.get(
                    "BASHGYM_LEDGER_DB",
                    home / ".bashgym/campaigns/campaigns.sqlite3",
                )
            ).expanduser(),
        )

    @property
    def memexai_root(self) -> Path:
        override = os.environ.get("MEMEXAI_TRAINING_ROOT")
        return Path(override).expanduser() if override else self.training_root / MEMEXAI_DIRECTORY

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
    return {key: run[key] for key in fields if key in run and run[key] not in (None, {}, [])}


def _observe_runs(observer: Path, *, limit: int) -> tuple[dict[str, Any], str | None]:
    if not observer.is_file():
        return {"summary": {"count": 0, "active": 0, "running": 0, "failed": 0}, "runs": []}, (
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
        return {"summary": {"count": 0, "active": 0, "running": 0, "failed": 0}, "runs": []}, (
            f"observer failed: {type(exc).__name__}: {exc}"
        )
    runs = snapshot.get("runs", []) if isinstance(snapshot, dict) else []
    return {
        "schema_version": snapshot.get("schema_version"),
        "observed_at": snapshot.get("observed_at"),
        "summary": snapshot.get("summary", {}),
        "runs": [_compact_run(run) for run in runs[:limit] if isinstance(run, dict)],
    }, None


def _safe_model_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).name


def _summarize_training_manifest(path: Path) -> dict[str, Any] | None:
    data = _read_json(path)
    if not data:
        return None
    training = data.get("training") if isinstance(data.get("training"), dict) else {}
    result = training.get("result_metrics")
    result_metrics = result if isinstance(result, dict) else {}
    summary = {
        "run_id": path.parent.name,
        "status": data.get("status"),
        "run_kind": data.get("run_kind"),
        "created_at": data.get("created_at"),
        "completed_at": data.get("completed_at"),
        "base_model": _safe_model_name(data.get("base_model_path")),
        "train_pairs": data.get("train_pairs"),
        "train_splits": data.get("train_splits"),
        "source_hashes": data.get("source_hashes"),
        "training": {
            key: training.get(key)
            for key in (
                "loss",
                "batch_size",
                "batch_size_requested",
                "batch_size_realized_mean",
                "mini_batch_size",
                "epochs",
                "optimizer_steps",
                "learning_rate",
                "precision",
                "temperature",
                "truncate_dim_for_eval",
            )
            if training.get(key) is not None
        },
        "result_metrics": {
            key: value
            for key, value in result_metrics.items()
            if isinstance(value, (int, float, str, bool)) and len(str(value)) < 200
        },
        "artifact_refs": {
            "manifest": str(path),
            "metrics": str(path.parent / "training_metrics.jsonl")
            if (path.parent / "training_metrics.jsonl").is_file()
            else None,
            "final_model": str(path.parent / "final") if (path.parent / "final").is_dir() else None,
        },
    }
    return summary


def _summarize_dataset_manifest(path: Path) -> dict[str, Any] | None:
    data = _read_json(path)
    if not data:
        return None
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    sources = data.get("sources") if isinstance(data.get("sources"), dict) else {}
    return {
        "schema_version": data.get("schema_version"),
        "passage_representation": data.get("passage_representation"),
        "policy": data.get("policy"),
        "statistics": data.get("statistics"),
        "source_digests": {
            key: value.get("sha256")
            for key, value in sources.items()
            if isinstance(value, dict) and value.get("sha256")
        },
        "output_digest": output.get("sha256") or output.get("canonical_records_sha256"),
        "artifact_ref": str(path),
    }


def _summarize_ablation(path: Path) -> dict[str, Any] | None:
    data = _read_json(path)
    if not data:
        return None
    protocol = data.get("protocol") if isinstance(data.get("protocol"), dict) else {}
    dense_protocol = protocol.get("dense") if isinstance(protocol.get("dense"), dict) else {}
    systems = data.get("systems") if isinstance(data.get("systems"), dict) else {}
    metrics: dict[str, Any] = {}
    for lane in ("dense", "bm25", "rrf"):
        lane_payload = systems.get(lane) if isinstance(systems.get(lane), dict) else {}
        lane_metrics = (
            lane_payload.get("metrics") if isinstance(lane_payload.get("metrics"), dict) else {}
        )
        metrics[lane] = {key: lane_metrics.get(key) for key in KEY_METRICS if key in lane_metrics}
    return {
        "candidate": path.parent.name,
        "created_at": data.get("created_at"),
        "query_count": (data.get("counts") or {}).get("queries")
        if isinstance(data.get("counts"), dict)
        else None,
        "model_id": dense_protocol.get("model_id"),
        "model_revision": dense_protocol.get("model_revision"),
        "selected_splits": protocol.get("selected_splits"),
        "protocol_sha256": data.get("protocol_sha256"),
        "systems": metrics,
        "reranker": data.get("reranker"),
        "artifact_ref": str(path),
    }


def _memexai_context(paths: OperatorPaths, *, limit: int) -> dict[str, Any]:
    project = paths.memexai_root
    runs, observer_error = _observe_runs(paths.observer, limit=limit)
    manifests = (
        sorted(
            (project / "runs").glob("*/training_manifest.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if (project / "runs").is_dir()
        else []
    )
    training_runs = [
        summary
        for summary in (_summarize_training_manifest(path) for path in manifests[:limit])
        if summary is not None
    ]
    ablation_paths = sorted(
        (project / "system-ablations" / "protected-dev-v1").glob(
            "*/retrieval_system_ablation_manifest.json"
        )
    )
    evaluations = [
        summary
        for summary in (_summarize_ablation(path) for path in ablation_paths)
        if summary is not None
    ]
    scripts = project / "scripts"
    python = project / ".venv/bin/python"
    command = lambda script: f"{python} {scripts / script} --help"  # noqa: E731
    return {
        "schema_version": SCHEMA_VERSION,
        "host": socket.gethostname(),
        "project": "memexai",
        "authority": {
            "live_jobs": str(paths.observer),
            "training_evidence": str(project / "runs"),
            "dataset_evidence": str(project / "inputs"),
            "development_evaluations": str(project / "system-ablations/protected-dev-v1"),
            "curated_activity": str(paths.activity_root),
        },
        "runtime": runs,
        "runtime_error": observer_error,
        "dataset": _summarize_dataset_manifest(project / "inputs/positive-aware-manifest.json"),
        "recent_training_runs": training_runs,
        "development_comparisons": evaluations,
        "protected_test": {
            "artifact_exists": (project / "inputs/heldout-dev-test.jsonl").is_file(),
            "policy": "Do not read or evaluate until the declared development gate authorizes it.",
        },
        "ability_commands": {
            "dataset_builder_help": command("build_positive_aware_dataset.py"),
            "training_help": command("train_embedding_retriever.py"),
            "training_preflight": command("train_embedding_retriever.py").removesuffix(" --help")
            + " <verified arguments> --dry-run",
            "corpus_embedding_help": command("embed_corpus_with_model.py"),
            "development_eval_help": command("evaluate_retrieval_system_ablation.py"),
            "product_fixture_eval_help": command("evaluate_product_retrieval_fixture.py"),
            "gbrain_activity_search": (
                f"{paths.gbrain} search '<query>' --source bashgym-activity --limit 10"
            ),
            "gbrain_project_search": f"{paths.gbrain} search '<query>' --source default --limit 10",
        },
        "campaign_control": {
            "available_here": _campaign_cli(paths) is not None or _api_health(paths.api_base_url),
            "rule": (
                "Canvas prompts may include the desktop workspace/campaign projection. "
                "Discord-local evidence does not imply desktop campaign mutation access."
            ),
        },
    }


def _project_context(
    paths: OperatorPaths,
    *,
    workspace_id: str,
    project_id: str | None,
    limit: int,
) -> dict[str, Any]:
    """Combine task-general ledger evidence with optional project-local evidence."""

    generated_at = datetime.now(timezone.utc).isoformat()
    runtime, runtime_error = _observe_runs(paths.observer, limit=limit)
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
    task_profile: dict[str, Any] | None = None
    if project_id and project_id.casefold() == "memexai" and paths.memexai_root.is_dir():
        task_profile = _memexai_context(paths, limit=limit)

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
    if project_id and ledger is None and task_profile is not None:
        conflicts.append(
            {
                "code": "durable_ledger_unavailable",
                "project_id": project_id,
                "decision_required": False,
                "resolution": (
                    "Use local runtime/manifests only for observation; do not claim a durable "
                    "campaign transition or project decision."
                ),
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
                    "freshness": "fresh" if task_profile is not None else "not_loaded",
                    "observed_at": generated_at if task_profile is not None else None,
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
        "task_profile": task_profile,
        "selection": {
            "available_projects": available_projects,
            "requires_project": project_id is None and len(available_projects) != 1,
        },
        "rules": [
            "Use runtime for current process state and the ledger for durable identities and decisions.",
            "Treat task_profile as supplemental evidence for the selected project only.",
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
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=0.75) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _bundle_integrity(root: Path | None = None) -> dict[str, Any]:
    bundle_root = root or Path(__file__).resolve().parents[1]
    lock_path = bundle_root / "bundle.lock.json"
    lock = _read_json(lock_path)
    if not lock or lock.get("schema_version") != "bashgym.operator-bundle-lock.v1":
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
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(str(relative))
    return {
        "verified": not mismatches,
        "lock_available": True,
        "mismatches": sorted(mismatches),
    }


def _doctor(paths: OperatorPaths) -> dict[str, Any]:
    project = paths.memexai_root
    api_reachable = _api_health(paths.api_base_url)
    campaign_cli = _campaign_cli(paths)
    training_cli = _training_cli(paths)
    hf_jobs_cli = _hf_jobs_cli()
    hf_token_configured = _hf_token_configured()
    hf_identity_verified = _hf_cli_check(hf_jobs_cli, "auth", "whoami")
    hf_jobs_access = hf_identity_verified and _hf_cli_check(hf_jobs_cli, "jobs", "hardware")
    venv_python = project / ".venv/bin/python"
    scripts = project / "scripts"
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
            "memexai_project": {"path": str(project), "available": project.is_dir()},
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
            "read_memexai_evidence": project.is_dir(),
            "build_memexai_dataset": (scripts / "build_positive_aware_dataset.py").is_file(),
            "dry_run_memexai_training": venv_python.is_file()
            and (scripts / "train_embedding_retriever.py").is_file(),
            "launch_memexai_training": venv_python.is_file()
            and (scripts / "train_embedding_retriever.py").is_file(),
            "evaluate_memexai_development": (
                scripts / "evaluate_retrieval_system_ablation.py"
            ).is_file(),
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
                "ready": training_cli is not None or api_reachable,
                "surface": "BashGym training CLI/API",
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
            "Launch availability means the guarded script exists; preflight and user authority still apply.",
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
    query = urllib.parse.urlencode({"workspace_id": workspace_id, "format": format_name})
    url = f"{paths.api_base_url}/api/workspace/context?{query}"
    with urllib.request.urlopen(url, timeout=3) as response:
        body = response.read().decode("utf-8")
    if format_name == "json":
        return json.loads(body)
    return body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Verify exact local ability sources without mutation.")
    context = subparsers.add_parser("context", help="Emit compact local project evidence.")
    context.add_argument("--project")
    context.add_argument("--workspace-id", default="desktop-local")
    context.add_argument("--limit", type=int, default=8)
    workspace = subparsers.add_parser("workspace", help="Read the desktop workspace projection.")
    workspace.add_argument("--workspace-id", required=True)
    workspace.add_argument("--format", choices=["json", "markdown"], default="json")
    ledger = subparsers.add_parser(
        "ledger", help="Read project-isolated local experiment context without an MCP server."
    )
    ledger.add_argument("--workspace-id", default="desktop-local")
    ledger.add_argument("--project-id")
    ledger.add_argument("--limit", type=int, default=20)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = OperatorPaths.from_environment()
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


if __name__ == "__main__":
    raise SystemExit(main())
