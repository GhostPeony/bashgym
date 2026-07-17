"""Discover BashGym work that was launched outside the HTTP job registries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil

from bashgym._compat import UTC

_IGNORED_MARKERS = (
    "uvicorn",
    "run_backend.py",
    "pytest",
    "runtime_observer",
)
_DESIGNER_MARKERS = (
    "data-designer",
    "data_designer",
    "generate_dd",
    "designer/create",
    "factory/designer",
)
_TRAINING_MARKERS = (
    "torchrun",
    "accelerate launch",
    "train_sft",
    "train_dpo",
    "train_grpo",
    "train_model",
    "training run",
    "bashgym training",
)
_CANDIDATE_PROCESS_NAMES = (
    "python",
    "python.exe",
    "pythonw.exe",
    "torchrun",
    "torchrun.exe",
    "accelerate",
    "accelerate.exe",
    "bashgym",
    "bashgym.exe",
    "data-designer",
    "data-designer.exe",
)


def _command_parts(command: Iterable[str] | str) -> list[str]:
    if not isinstance(command, str):
        return [str(part) for part in command if part]
    try:
        return [part.strip('"') for part in shlex.split(command, posix=False)]
    except ValueError:
        return command.split()


def _within_workspace(cwd: str | None, workspace_root: Path) -> bool:
    if not cwd:
        return False
    try:
        Path(cwd).resolve().relative_to(workspace_root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _command_references_workspace(command: Iterable[str] | str, workspace_root: Path) -> bool:
    for part in _command_parts(command):
        if not part.lower().endswith(".py"):
            continue
        candidate = Path(part)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        try:
            candidate.resolve().relative_to(workspace_root.resolve())
        except (OSError, ValueError):
            continue
        if candidate.exists():
            return True
    return False


def _runtime_kind_for_command(command: Iterable[str] | str) -> str | None:
    normalized = " ".join(_command_parts(command)).lower().replace("\\", "/")
    if not normalized or any(marker in normalized for marker in _IGNORED_MARKERS):
        return None
    if any(marker in normalized for marker in _DESIGNER_MARKERS):
        return "designer"
    if any(marker in normalized for marker in _TRAINING_MARKERS):
        return "training"
    return None


def classify_runtime_command(
    command: Iterable[str] | str,
    cwd: str | None,
    workspace_root: Path,
) -> str | None:
    """Classify workspace-owned processes without treating the API itself as work."""
    kind = _runtime_kind_for_command(command)
    if kind is None:
        return None
    if not _within_workspace(cwd, workspace_root) and not _command_references_workspace(
        command, workspace_root
    ):
        return None
    return kind


def parse_cli_options(command: Iterable[str] | str) -> dict[str, str]:
    parts = _command_parts(command)
    options: dict[str, str] = {}
    index = 0
    while index < len(parts):
        part = parts[index]
        if not part.startswith("--"):
            index += 1
            continue
        key_value = part[2:].split("=", 1)
        if len(key_value) == 2:
            options[key_value[0].replace("-", "_")] = key_value[1]
            index += 1
            continue
        key = key_value[0].replace("-", "_")
        if index + 1 < len(parts) and not parts[index + 1].startswith("--"):
            options[key] = parts[index + 1]
            index += 2
        else:
            options[key] = "true"
            index += 1
    return options


def _integer_option(options: dict[str, str], *names: str) -> int | None:
    for name in names:
        try:
            value = int(options.get(name, ""))
        except ValueError:
            continue
        if value >= 0:
            return value
    return None


def _count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _tail_text(path: Path, max_bytes: int = 128 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def inspect_runtime_log(
    workspace_root: Path,
    script: str,
    started_at: float,
) -> dict[str, str]:
    """Recover model/provider metadata from a nearby runner log when available."""
    script_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", Path(script).stem.lower())
        if len(token) > 1 and token not in {"generate", "script"}
    }
    candidates: list[tuple[int, float, Path]] = []
    for directory in (workspace_root / ".tmp", workspace_root):
        if not directory.exists():
            continue
        for path in directory.glob("*.log"):
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            if modified < started_at - 60:
                continue
            name_tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
            score = len(script_tokens & name_tokens)
            if score >= 2:
                candidates.append((score, modified, path))
    if not candidates:
        return {}

    _, _, log_path = max(candidates, key=lambda item: (item[0], item[1]))
    tail = _tail_text(log_path)
    model_matches = re.findall(r"\bmodel:\s*['\"]?([^'\"\s]+)", tail, flags=re.IGNORECASE)
    provider_matches = re.findall(
        r"\bmodel provider:\s*['\"]?([^'\"\s]+)", tail, flags=re.IGNORECASE
    )
    return {
        **({"model": model_matches[-1]} if model_matches else {}),
        **({"provider": provider_matches[-1]} if provider_matches else {}),
        "log_path": str(log_path),
    }


def runtime_execution_label(options: dict[str, str], provider: str | None) -> str:
    endpoint = options.get("llm_endpoint") or options.get("provider_endpoint") or ""
    host = (urlparse(endpoint).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return "local"
    if host.startswith(("10.", "192.168.")):
        return "private"
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return "private"
        except (IndexError, ValueError):
            pass
    if provider and provider.lower() in {"local", "ollama", "vllm"}:
        return "local"
    if endpoint or provider:
        return "cloud"
    return "unknown"


def inspect_runtime_artifacts(
    kind: str,
    options: dict[str, str],
    cwd: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    raw_output = options.get("output_dir") or options.get("output")
    if not raw_output:
        return None, [], None
    output_dir = Path(raw_output)
    if not output_dir.is_absolute():
        output_dir = Path(cwd) / output_dir
    try:
        output_dir = output_dir.resolve()
    except OSError:
        pass

    artifacts: list[dict[str, Any]] = []
    try:
        files = sorted(
            (path for path in output_dir.iterdir() if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files[:8]:
            stat = path.stat()
            artifacts.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                }
            )
    except OSError:
        files = []

    progress: dict[str, Any] | None = None
    if kind == "designer":
        total = _integer_option(options, "num_seeds", "num_records")
        batch_files = sorted(output_dir.glob("*batch*.jsonl")) if output_dir.exists() else []
        if total is not None:
            batch_size = _count_jsonl_rows(batch_files[0]) if batch_files else 0
            completed_batches = max(0, len(batch_files) - 1)
            current = min(total, completed_batches * batch_size)
            progress = {"current": current, "total": total, "unit": "seeds"}
    elif kind == "training":
        metrics = output_dir / "metrics.jsonl"
        if metrics.exists():
            progress = {"current": _count_jsonl_rows(metrics), "total": None, "unit": "steps"}

    return progress, artifacts, str(output_dir)


def completed_runtime_jobs_from_manifests(
    workspace_root: Path,
    *,
    max_age_seconds: int = 24 * 60 * 60,
) -> list[dict[str, Any]]:
    """Recover recently finalized Data Designer jobs after their process exits."""
    now = time.time()
    manifests: list[Path] = []
    for root in (workspace_root / ".tmp", workspace_root / "data"):
        if not root.exists():
            continue
        try:
            manifests.extend(root.rglob("dd_train_pairs_manifest.json"))
        except OSError:
            continue

    jobs: list[dict[str, Any]] = []
    for manifest_path in manifests:
        try:
            stat = manifest_path.stat()
            if now - stat.st_mtime > max_age_seconds:
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("status") != "completed":
            continue

        arm = str(manifest.get("arm") or "")
        # Ignore smoke/dry-run siblings such as real_chunks_smoke4.
        if arm and manifest_path.parent.name != arm:
            continue
        output_dir = manifest_path.parent.resolve()
        data_designer = manifest.get("data_designer")
        designer_meta = data_designer if isinstance(data_designer, dict) else {}
        outputs = manifest.get("outputs")
        output_meta = outputs if isinstance(outputs, dict) else {}
        try:
            total = int(manifest.get("seed_rows") or manifest.get("raw_rows") or 0)
        except (TypeError, ValueError):
            continue
        endpoint = str(designer_meta.get("endpoint") or "")
        provider = "local" if endpoint else None
        _, artifacts, _ = inspect_runtime_artifacts(
            "designer",
            {"output_dir": str(output_dir)},
            str(workspace_root),
        )
        path_key = hashlib.sha256(str(output_dir).encode("utf-8")).hexdigest()[:12]
        started_at = (
            str(manifest.get("created_at") or "")
            or datetime.fromtimestamp(stat.st_ctime, UTC).isoformat()
        )
        jobs.append(
            {
                "job_id": f"runtime_designer_manifest_{path_key}",
                "kind": "designer",
                "status": "completed",
                "pid": 0,
                "title": "Data Designer Train Pairs",
                "script": "generate_dd_train_pairs.py",
                "cwd": str(workspace_root),
                "started_at": started_at,
                "completed_at": manifest.get("completed_at"),
                "pipeline": manifest.get("run_kind") or "generate_dd_train_pairs",
                "job_name": f"{output_dir.parent.name} / {arm or output_dir.name}",
                "dataset": manifest.get("corpus_jsonl"),
                "model": designer_meta.get("model"),
                "provider": provider,
                "execution": runtime_execution_label({"llm_endpoint": endpoint}, provider),
                "log_path": None,
                "strategy": None,
                "output_dir": str(output_dir),
                "progress": {"current": total, "total": total, "unit": "seeds"},
                "artifacts": artifacts,
                "options": {
                    "arm": arm,
                    **(
                        {"train_queries_jsonl": str(output_meta["train_queries_jsonl"])}
                        if output_meta.get("train_queries_jsonl")
                        else {}
                    ),
                },
                "source": "process_observer",
            }
        )
    return sorted(jobs, key=lambda job: job["started_at"], reverse=True)


def runtime_job_from_process_info(
    info: dict[str, Any],
    workspace_root: Path,
) -> dict[str, Any] | None:
    command = _command_parts(info.get("cmdline") or [])
    cwd = info.get("cwd") or str(workspace_root)
    kind = classify_runtime_command(command, cwd, workspace_root)
    if not kind or not cwd:
        return None

    options = parse_cli_options(command)
    progress, artifacts, output_dir = inspect_runtime_artifacts(kind, options, cwd)
    pid = int(info["pid"])
    create_time = float(info.get("create_time") or 0)
    script = next(
        (Path(str(part)).name for part in command if str(part).lower().endswith(".py")),
        kind,
    )
    log_metadata = inspect_runtime_log(workspace_root, script, create_time)
    strategy = options.get("strategy")
    if kind == "training" and not strategy:
        lowered_script = script.lower()
        strategy = next(
            (candidate for candidate in ("sft", "dpo", "grpo") if candidate in lowered_script),
            "training",
        )

    entity_id = f"runtime_{kind}_{pid}_{int(create_time * 1000)}"
    output_path = Path(output_dir) if output_dir else None
    arm = options.get("arm")
    job_name = (
        f"{output_path.parent.name} / {arm}"
        if output_path and arm
        else output_path.name if output_path else script.removesuffix(".py")
    )
    dataset = (
        options.get("dataset_path")
        or options.get("corpus_jsonl")
        or options.get("seed_source")
        or options.get("template_queries_jsonl")
    )
    provider = log_metadata.get("provider")
    return {
        "job_id": entity_id,
        "kind": kind,
        "status": "running",
        "pid": pid,
        "title": script.removesuffix(".py").replace("_", " ").strip().title(),
        "script": script,
        "cwd": cwd,
        "started_at": datetime.fromtimestamp(create_time, UTC).isoformat(),
        "pipeline": options.get("pipeline")
        or (script.removesuffix(".py") if kind == "designer" else None),
        "job_name": job_name,
        "dataset": dataset,
        "model": log_metadata.get("model"),
        "provider": provider,
        "execution": runtime_execution_label(options, provider),
        "log_path": log_metadata.get("log_path"),
        "strategy": strategy,
        "output_dir": output_dir,
        "progress": progress,
        "artifacts": artifacts,
        "options": {
            key: value
            for key, value in options.items()
            if key not in {"api_key", "token", "authorization", "password", "secret"}
        },
        "source": "process_observer",
    }


class RuntimeObserver:
    """Read-only scanner for active BashGym training and Data Designer processes."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()
        self._cached_jobs: list[dict[str, Any]] = []
        self._last_scan = 0.0
        self._scan_lock = threading.Lock()

    def _windows_process_infos(self) -> list[dict[str, Any]]:
        command = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR "
            "Name='pythonw.exe' OR Name='torchrun.exe' OR Name='accelerate.exe'\" | "
            "Select-Object @{Name='pid';Expression={$_.ProcessId}},"
            "@{Name='create_time';Expression={([DateTimeOffset]$_.CreationDate).ToUnixTimeMilliseconds()/1000}},"
            "@{Name='cmdline';Expression={$_.CommandLine}} | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            payload = json.loads(result.stdout)
            rows = payload if isinstance(payload, list) else [payload]
            return [
                {
                    "pid": row.get("pid"),
                    "create_time": row.get("create_time"),
                    "cmdline": row.get("cmdline") or "",
                    "cwd": str(self.workspace_root),
                }
                for row in rows
                if isinstance(row, dict)
            ]
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return []

    def _process_infos(self) -> list[dict[str, Any]]:
        if os.name == "nt":
            return self._windows_process_infos()
        infos: list[dict[str, Any]] = []
        for process in psutil.process_iter(["pid", "create_time", "name"]):
            try:
                name = str(process.info.get("name") or "").lower()
                if name not in _CANDIDATE_PROCESS_NAMES:
                    continue
                infos.append(
                    {
                        **process.info,
                        "cmdline": process.cmdline(),
                        "cwd": process.cwd(),
                    }
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
        return infos

    def list_jobs(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        if now - self._last_scan < 5:
            return [dict(job) for job in self._cached_jobs]
        if not self._scan_lock.acquire(blocking=False):
            return [dict(job) for job in self._cached_jobs]
        jobs: list[dict[str, Any]] = []
        try:
            for info in self._process_infos():
                try:
                    if _runtime_kind_for_command(info.get("cmdline") or []) is None:
                        continue
                    job = runtime_job_from_process_info(info, self.workspace_root)
                except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, ValueError):
                    continue
                if job:
                    jobs.append(job)
            active_output_dirs = {
                str(Path(job["output_dir"]).resolve()).lower()
                for job in jobs
                if job.get("output_dir")
            }
            jobs.extend(
                job
                for job in completed_runtime_jobs_from_manifests(self.workspace_root)
                if str(Path(job["output_dir"]).resolve()).lower() not in active_output_dirs
            )
            jobs = sorted(jobs, key=lambda job: job["started_at"], reverse=True)
            self._cached_jobs = jobs
            self._last_scan = time.monotonic()
            return [dict(job) for job in jobs]
        finally:
            self._scan_lock.release()
