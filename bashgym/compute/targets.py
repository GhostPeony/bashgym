"""Dry-run compute target planning for BashGym training jobs."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ComputeLauncher(str, Enum):
    LOCAL = "local"
    SSH = "ssh"
    SKYPILOT = "skypilot"
    DSTACK = "dstack"


@dataclass(frozen=True)
class ComputeTarget:
    id: str
    provider: str
    launcher: ComputeLauncher
    gpu_type: str
    gpu_count: int = 1
    region: str | None = None
    image: str | None = None
    python_version: str = "3.12"
    cuda_version: str | None = None
    disk_gb: int = 100
    max_budget_usd: float | None = None
    env_vars: tuple[str, ...] = ()
    secret_refs: tuple[str, ...] = ()
    dataset_mount: str | None = None
    output_sync: str | None = None
    preflight_command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["launcher"] = self.launcher.value
        return payload

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.id:
            errors.append("id is required")
        if self.gpu_count <= 0:
            errors.append("gpu_count must be positive")
        if self.disk_gb <= 0:
            errors.append("disk_gb must be positive")
        if self.launcher == ComputeLauncher.SSH and "host_env" not in self.metadata:
            errors.append("ssh targets require metadata.host_env")
        return errors


DEFAULT_TARGETS: tuple[ComputeTarget, ...] = (
    ComputeTarget(
        id="local_cpu_or_gpu",
        provider="local",
        launcher=ComputeLauncher.LOCAL,
        gpu_type="auto",
        gpu_count=1,
        disk_gb=50,
        env_vars=("CUDA_VISIBLE_DEVICES",),
        preflight_command="python -m bashgym.cli manifest --json",
        metadata={"description": "Current workstation using the active Python environment."},
    ),
    ComputeTarget(
        id="gx10_ssh",
        provider="ssh",
        launcher=ComputeLauncher.SSH,
        gpu_type="NVIDIA GB10/GX10",
        gpu_count=1,
        disk_gb=200,
        env_vars=("BASHGYM_GX10_HOST", "BASHGYM_GX10_WORKDIR"),
        preflight_command="python -m bashgym.cli training smoke-bundle --help",
        metadata={
            "description": "User-managed GX10 or remote SSH training host.",
            "host_env": "BASHGYM_GX10_HOST",
            "workdir_env": "BASHGYM_GX10_WORKDIR",
        },
    ),
    ComputeTarget(
        id="skypilot_a10g",
        provider="cloud",
        launcher=ComputeLauncher.SKYPILOT,
        gpu_type="A10G",
        gpu_count=1,
        disk_gb=200,
        max_budget_usd=20.0,
        secret_refs=("HF_TOKEN", "WANDB_API_KEY"),
        output_sync="rsync",
        metadata={"description": "Portable SkyPilot dry-run target for small LoRA jobs."},
    ),
    ComputeTarget(
        id="dstack_a10g",
        provider="cloud",
        launcher=ComputeLauncher.DSTACK,
        gpu_type="A10G",
        gpu_count=1,
        disk_gb=200,
        max_budget_usd=20.0,
        secret_refs=("HF_TOKEN", "WANDB_API_KEY"),
        output_sync="artifact sync",
        metadata={"description": "Portable dstack dry-run target for small LoRA jobs."},
    ),
)


def list_compute_targets() -> list[ComputeTarget]:
    return list(DEFAULT_TARGETS)


def get_compute_target(target_id: str) -> ComputeTarget:
    for target in DEFAULT_TARGETS:
        if target.id == target_id:
            return target
    raise KeyError(target_id)


def preflight_compute_target(target: ComputeTarget) -> dict[str, Any]:
    """Return non-invasive readiness checks for a compute target."""

    checks: list[dict[str, Any]] = []
    for error in target.validation_errors():
        checks.append({"code": "target_schema", "status": "fail", "message": error})

    if target.launcher == ComputeLauncher.LOCAL:
        checks.append(
            {
                "code": "python_available",
                "status": "pass" if shutil.which("python") else "fail",
                "message": "python is available on PATH" if shutil.which("python") else "python not found",
            }
        )
    elif target.launcher == ComputeLauncher.SSH:
        host_env = target.metadata.get("host_env", "")
        host = os.environ.get(host_env, "")
        checks.append(
            {
                "code": "ssh_host_configured",
                "status": "pass" if host else "needs_config",
                "message": f"{host_env} is set" if host else f"Set {host_env} before SSH preflight.",
            }
        )
    elif target.launcher == ComputeLauncher.SKYPILOT:
        checks.append(
            {
                "code": "skypilot_cli",
                "status": "pass" if shutil.which("sky") else "needs_install",
                "message": "sky CLI found" if shutil.which("sky") else "Install SkyPilot CLI for launch.",
            }
        )
    elif target.launcher == ComputeLauncher.DSTACK:
        checks.append(
            {
                "code": "dstack_cli",
                "status": "pass" if shutil.which("dstack") else "needs_install",
                "message": "dstack CLI found" if shutil.which("dstack") else "Install dstack CLI for launch.",
            }
        )

    level = "ready"
    if any(check["status"] == "fail" for check in checks):
        level = "blocked"
    elif any(check["status"] in {"needs_config", "needs_install"} for check in checks):
        level = "needs_setup"
    return {
        "schema_version": "bashgym.compute_preflight.v1",
        "target": target.to_dict(),
        "level": level,
        "checks": checks,
    }


def _training_command(plan_path: str | Path | None) -> str:
    if plan_path:
        return f"python scripts/train_model.py --config {Path(plan_path).as_posix()}"
    return "python scripts/train_model.py --help"


def launch_plan(target: ComputeTarget, *, plan_path: str | Path | None = None) -> dict[str, Any]:
    """Generate a dry-run launch plan without executing anything."""

    command = _training_command(plan_path)
    if target.launcher == ComputeLauncher.LOCAL:
        provider_config = {"command": command}
    elif target.launcher == ComputeLauncher.SSH:
        host_env = target.metadata.get("host_env", "BASHGYM_REMOTE_HOST")
        workdir_env = target.metadata.get("workdir_env", "BASHGYM_REMOTE_WORKDIR")
        provider_config = {
            "host_env": host_env,
            "workdir_env": workdir_env,
            "command": f"ssh ${host_env} 'cd ${{{workdir_env}:-~/ghostwork}} && {command}'",
        }
    elif target.launcher == ComputeLauncher.SKYPILOT:
        provider_config = {
            "filename": "sky.yaml",
            "content": "\n".join(
                [
                    "resources:",
                    f"  accelerators: {target.gpu_type}:{target.gpu_count}",
                    f"  disk_size: {target.disk_gb}",
                    "setup: |",
                    "  pip install -r requirements-training.txt",
                    "run: |",
                    f"  {command}",
                    "",
                ]
            ),
        }
    elif target.launcher == ComputeLauncher.DSTACK:
        provider_config = {
            "filename": ".dstack.yml",
            "content": "\n".join(
                [
                    "type: task",
                    "resources:",
                    f"  gpu: {target.gpu_type}:{target.gpu_count}",
                    "commands:",
                    "  - pip install -r requirements-training.txt",
                    f"  - {command}",
                    "",
                ]
            ),
        }
    else:
        provider_config = {"command": command}

    return {
        "schema_version": "bashgym.compute_launch_plan.v1",
        "dry_run": True,
        "target": target.to_dict(),
        "training_plan": str(plan_path) if plan_path else None,
        "provider_config": provider_config,
        "approval_required": target.launcher != ComputeLauncher.LOCAL,
        "redaction": "secret_refs are names only; secret values are never serialized",
    }
