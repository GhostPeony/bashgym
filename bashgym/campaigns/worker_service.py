"""Safe resident-worker launch configuration and per-user service definitions.

This module deliberately keeps service installation separate from the campaign
scheduler.  It only launches the existing foreground worker with a restricted
configuration file; supervisors remain responsible for restarting a crashed
process.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import plistlib
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.contracts import utc_now
from bashgym.campaigns.persistence import CampaignRepository
from bashgym.campaigns.remote import ApprovedRemoteExecutorProfile, RemoteTrainingAdapter
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.worker import CampaignWorker, scheduler_lease_key
from bashgym.gym.remote_trainer import SSHConfig
from bashgym.mcp.policy import resolve_secret_reference, validate_secret_ref_name
from bashgym.secrets import get_secret

MAX_CONFIG_BYTES = 64 * 1024
TASK_NAME = r"\BashGym\Campaign Worker"
SYSTEMD_UNIT_NAME = "bashgym-campaign-worker.service"
LAUNCHD_LABEL = "com.ghostpeony.bashgym.campaign-worker"
CONTROLLER_OFFLINE_GUIDANCE = (
    "Install or restart the per-user campaign worker. Durable campaigns remain paused "
    "and remote training runs remain untouched."
)
CONTROLLER_STALE_GUIDANCE = (
    "The campaign worker stopped renewing its scheduler lease. Restart it and inspect "
    "the bounded lifecycle status before advancing durable campaigns."
)


class WorkerServiceError(RuntimeError):
    """Safe worker-service failure with a stable machine-readable code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class WorkerPlatform(str, Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    DARWIN = "darwin"

    @classmethod
    def current(cls) -> WorkerPlatform:
        system = platform.system().casefold()
        if system == "windows":
            return cls.WINDOWS
        if system == "linux":
            return cls.LINUX
        if system == "darwin":
            return cls.DARWIN
        raise WorkerServiceError("campaign_worker_platform_unsupported")


class WorkerRunConfig(BaseModel):
    """Secret-free, canonical configuration consumed by a resident worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "campaign_worker_config.v1"
    data_directory: Path
    database_path: Path
    artifact_root: Path
    seal_key_ref: str = "BASHGYM_CAMPAIGN_SEAL_KEY"
    seal_key_version: str = Field(default="v1", min_length=1, max_length=64)
    approved_remote_profiles: tuple[ApprovedRemoteExecutorProfile, ...] = ()
    # Legacy discovery IDs remain parseable so existing service configuration can
    # be inspected or replaced. They never authorize campaign remote execution.
    compute_profile_ids: tuple[str, ...] = ()
    leader_ttl_seconds: float = Field(default=15.0, gt=0, le=300)
    action_ttl_seconds: float = Field(default=15.0, gt=0, le=300)
    heartbeat_seconds: float = Field(default=5.0, gt=0, le=60)
    ready_poll_seconds: float = Field(default=2.0, gt=0, le=60)
    idle_poll_seconds: float = Field(default=30.0, gt=0, le=3600)

    @field_validator("data_directory", "database_path", "artifact_root")
    @classmethod
    def canonical_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("seal_key_ref")
    @classmethod
    def opaque_secret_ref(cls, value: str) -> str:
        return validate_secret_ref_name(value)

    @field_validator("compute_profile_ids")
    @classmethod
    def canonical_profile_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("compute_profile_ids must be sorted and unique")
        for profile_id in value:
            if not profile_id or len(profile_id) > 128 or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                for character in profile_id
            ):
                raise ValueError("compute profile IDs must be simple identifiers")
        return value

    @field_validator("approved_remote_profiles")
    @classmethod
    def canonical_remote_profiles(
        cls, value: tuple[ApprovedRemoteExecutorProfile, ...]
    ) -> tuple[ApprovedRemoteExecutorProfile, ...]:
        ordering = tuple(
            (
                profile.compute_profile_id,
                profile.target_contract_key,
                profile.profile_id,
                profile.profile_revision,
            )
            for profile in value
        )
        if tuple(sorted(ordering)) != ordering:
            raise ValueError("approved remote profiles must be sorted")
        resolution_keys = tuple(
            (profile.compute_profile_id, profile.target_contract_key) for profile in value
        )
        if len(set(resolution_keys)) != len(resolution_keys):
            raise ValueError(
                "approved remote profiles must be unique per compute and target contract"
            )
        return value

    @model_validator(mode="after")
    def confine_runtime_paths(self) -> WorkerRunConfig:
        for path in (self.database_path, self.artifact_root):
            try:
                path.relative_to(self.data_directory)
            except ValueError as exc:
                raise ValueError("worker runtime paths must remain inside data_directory") from exc
        if self.heartbeat_seconds >= self.leader_ttl_seconds:
            raise ValueError("heartbeat_seconds must be shorter than leader_ttl_seconds")
        return self

    @classmethod
    def for_data_directory(
        cls,
        data_directory: Path,
        *,
        approved_remote_profiles: tuple[ApprovedRemoteExecutorProfile, ...] = (),
        compute_profile_ids: tuple[str, ...] = (),
    ) -> WorkerRunConfig:
        root = data_directory.expanduser().resolve()
        return cls(
            data_directory=root,
            database_path=root / "campaigns" / "campaigns.sqlite3",
            artifact_root=root / "campaigns" / "artifacts",
            approved_remote_profiles=approved_remote_profiles,
            compute_profile_ids=compute_profile_ids,
        )

    @property
    def status_path(self) -> Path:
        return self.data_directory / "campaigns" / "worker-status.v1.json"


class WorkerLifecycleStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "campaign_worker_lifecycle.v1"
    state: str
    worker_id: str | None = None
    restart_count: int = Field(default=0, ge=0)
    observed_at: datetime
    last_error_code: str | None = None


class ControllerStatusProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "campaign_controller_status.v1"
    online: bool
    state: Literal["online", "stale", "offline"]
    code: str
    observed_at: datetime
    heartbeat_age_seconds: float | None = Field(default=None, ge=0)
    guidance: str | None = None
    owner_id: str | None = None
    generation: int | None = None
    heartbeat_at: datetime | None = None
    expires_at: datetime | None = None


def _read_restricted_json(path: Path, *, maximum_bytes: int = MAX_CONFIG_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise WorkerServiceError("campaign_worker_config_not_regular")
    if path.stat().st_size > maximum_bytes:
        raise WorkerServiceError("campaign_worker_config_too_large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerServiceError("campaign_worker_config_invalid") from exc
    if not isinstance(payload, dict):
        raise WorkerServiceError("campaign_worker_config_invalid")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise WorkerServiceError("campaign_worker_file_not_regular")
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def read_worker_config(path: Path) -> WorkerRunConfig:
    return WorkerRunConfig.model_validate(_read_restricted_json(path.expanduser()))


def write_worker_config(path: Path, config: WorkerRunConfig) -> None:
    payload = config.model_dump_json(indent=2).encode("utf-8") + b"\n"
    _atomic_write(path.expanduser(), payload)


def _read_lifecycle_status(path: Path) -> WorkerLifecycleStatus | None:
    if not path.exists():
        return None
    try:
        return WorkerLifecycleStatus.model_validate(_read_restricted_json(path))
    except (WorkerServiceError, ValueError):
        return None


def _write_lifecycle_status(path: Path, status: WorkerLifecycleStatus) -> None:
    _atomic_write(path, status.model_dump_json(indent=2).encode("utf-8") + b"\n")


def project_controller_status(
    repository: CampaignRepository | None,
    data_directory: Path,
    *,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(seconds=15),
) -> ControllerStatusProjection:
    """Project a lease into a safe online/offline controller state."""

    observed_at = now or utc_now()
    lease = repository.get_lease(scheduler_lease_key(data_directory)) if repository else None
    if lease is None:
        return ControllerStatusProjection(
            online=False,
            state="offline",
            code="controller_offline",
            observed_at=observed_at,
            guidance=CONTROLLER_OFFLINE_GUIDANCE,
        )
    heartbeat_age_seconds = max(
        0.0, (observed_at - lease.heartbeat_at).total_seconds()
    )
    if lease.expires_at <= observed_at or heartbeat_age_seconds > stale_after.total_seconds():
        return ControllerStatusProjection(
            online=False,
            state="stale",
            code="controller_stale",
            observed_at=observed_at,
            heartbeat_age_seconds=heartbeat_age_seconds,
            guidance=CONTROLLER_STALE_GUIDANCE,
            owner_id=lease.owner_id,
            generation=lease.generation,
            heartbeat_at=lease.heartbeat_at,
            expires_at=lease.expires_at,
        )
    return ControllerStatusProjection(
        online=True,
        state="online",
        code="controller_online",
        observed_at=observed_at,
        heartbeat_age_seconds=heartbeat_age_seconds,
        owner_id=lease.owner_id,
        generation=lease.generation,
        heartbeat_at=lease.heartbeat_at,
        expires_at=lease.expires_at,
    )


def load_approved_remote_profiles(
    config: WorkerRunConfig,
) -> dict[tuple[str, str], ApprovedRemoteExecutorProfile]:
    """Load and re-verify the protected registry used by the resident controller."""

    registry: dict[tuple[str, str], ApprovedRemoteExecutorProfile] = {}
    try:
        for profile in config.approved_remote_profiles:
            profile.verify_materials()
            key = (profile.compute_profile_id, profile.target_contract_key)
            if key in registry:
                raise ValueError("duplicate approved remote profile resolution key")
            registry[key] = profile
    except (OSError, ValueError) as exc:
        raise WorkerServiceError("campaign_worker_remote_profile_material_invalid") from exc
    return registry


def _load_remote_adapters(config: WorkerRunConfig) -> dict[str, RemoteTrainingAdapter]:
    profiles = load_approved_remote_profiles(config)
    adapters: dict[str, RemoteTrainingAdapter] = {}
    ssh_authority: dict[str, tuple[str, str, int, str, str]] = {}
    for profile in profiles.values():
        authority = (
            profile.host,
            profile.username,
            profile.port,
            profile.key_path,
            profile.remote_work_dir,
        )
        existing = ssh_authority.get(profile.compute_profile_id)
        if existing is not None and existing != authority:
            raise WorkerServiceError("campaign_worker_compute_profile_authority_conflict")
        ssh_authority[profile.compute_profile_id] = authority
        if profile.compute_profile_id not in adapters:
            adapters[profile.compute_profile_id] = RemoteTrainingAdapter(
                SSHConfig(
                    host=profile.host,
                    username=profile.username,
                    port=profile.port,
                    key_path=profile.key_path,
                    remote_work_dir=profile.remote_work_dir,
                ),
                compute_profile_id=profile.compute_profile_id,
            )
    return adapters


def build_worker(
    config: WorkerRunConfig,
    *,
    secret_resolver: Callable[[str], str | None] = get_secret,
    adapter_loader: Callable[[WorkerRunConfig], dict[str, RemoteTrainingAdapter]] = (
        _load_remote_adapters
    ),
) -> CampaignWorker:
    """Build the existing worker without importing cloud or environment fallbacks."""

    try:
        key = resolve_secret_reference(config.seal_key_ref, secret_resolver)
    except ValueError as exc:
        raise WorkerServiceError("campaign_worker_seal_key_unavailable") from exc
    repository = CampaignRuntimeRepository(config.database_path)
    repository.initialize()
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    remote_executor_profiles = load_approved_remote_profiles(config)
    return CampaignWorker(
        repository,
        config.artifact_root,
        ArtifactSealer(key.encode("utf-8"), key_version=config.seal_key_version),
        data_directory=config.data_directory,
        leader_ttl=timedelta(seconds=config.leader_ttl_seconds),
        action_ttl=timedelta(seconds=config.action_ttl_seconds),
        remote_adapters=adapter_loader(config),
        remote_executor_profiles=remote_executor_profiles,
    )


class ForegroundWorker(Protocol):
    worker_id: str

    def request_stop(self) -> None: ...

    def run_forever(
        self,
        *,
        heartbeat_seconds: float,
        ready_poll_seconds: float,
        idle_poll_seconds: float,
    ) -> None: ...


def run_foreground(
    config: WorkerRunConfig,
    *,
    worker_factory: Callable[[WorkerRunConfig], ForegroundWorker] = build_worker,
    install_signal_handlers: bool = True,
) -> None:
    """Run one resident worker; failures remain non-zero for supervisor restart."""

    previous = _read_lifecycle_status(config.status_path)
    restart_count = (previous.restart_count + 1) if previous and previous.state == "crashed" else 0
    starting = WorkerLifecycleStatus(
        state="starting", restart_count=restart_count, observed_at=utc_now()
    )
    _write_lifecycle_status(config.status_path, starting)
    worker: ForegroundWorker | None = None
    prior_handlers: dict[signal.Signals, Any] = {}
    try:
        worker = worker_factory(config)
        _write_lifecycle_status(
            config.status_path,
            WorkerLifecycleStatus(
                state="running",
                worker_id=worker.worker_id,
                restart_count=restart_count,
                observed_at=utc_now(),
            ),
        )
        if install_signal_handlers:
            for signum in (signal.SIGINT, signal.SIGTERM):
                prior_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda _signum, _frame: worker.request_stop())
        worker.run_forever(
            heartbeat_seconds=config.heartbeat_seconds,
            ready_poll_seconds=config.ready_poll_seconds,
            idle_poll_seconds=config.idle_poll_seconds,
        )
    except BaseException as exc:
        _write_lifecycle_status(
            config.status_path,
            WorkerLifecycleStatus(
                state="crashed",
                worker_id=worker.worker_id if worker else None,
                restart_count=restart_count,
                observed_at=utc_now(),
                last_error_code=exc.__class__.__name__,
            ),
        )
        raise
    else:
        _write_lifecycle_status(
            config.status_path,
            WorkerLifecycleStatus(
                state="stopped",
                worker_id=worker.worker_id if worker else None,
                restart_count=restart_count,
                observed_at=utc_now(),
            ),
        )
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)


@dataclass(frozen=True)
class WorkerServiceDefinition:
    platform: WorkerPlatform
    config_path: Path
    definition_path: Path
    launch_argv: tuple[str, ...]
    definition_payload: bytes
    install_argvs: tuple[tuple[str, ...], ...]
    status_argv: tuple[str, ...]
    uninstall_argvs: tuple[tuple[str, ...], ...]
    post_uninstall_argvs: tuple[tuple[str, ...], ...] = ()


def _reject_controls(value: str) -> str:
    if any(ord(character) < 32 for character in value):
        raise WorkerServiceError("campaign_worker_service_argument_invalid")
    return value


def _systemd_quote(value: str) -> str:
    value = _reject_controls(value)
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
    )
    return f'"{escaped}"'


def _default_executable(target: WorkerPlatform) -> Path:
    executable = Path(sys.executable).resolve()
    if target is WorkerPlatform.WINDOWS and executable.name.casefold() == "python.exe":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            return pythonw
    return executable


def _windows_xml(launch_argv: tuple[str, ...], username: str) -> bytes:
    namespace = "http://schemas.microsoft.com/windows/2004/02/mit/task"
    ET.register_namespace("", namespace)
    task = ET.Element(f"{{{namespace}}}Task", {"version": "1.4"})
    registration = ET.SubElement(task, f"{{{namespace}}}RegistrationInfo")
    ET.SubElement(registration, f"{{{namespace}}}Description").text = (
        "BashGym resident campaign worker"
    )
    triggers = ET.SubElement(task, f"{{{namespace}}}Triggers")
    logon = ET.SubElement(triggers, f"{{{namespace}}}LogonTrigger")
    ET.SubElement(logon, f"{{{namespace}}}Enabled").text = "true"
    principals = ET.SubElement(task, f"{{{namespace}}}Principals")
    principal = ET.SubElement(principals, f"{{{namespace}}}Principal", {"id": "Author"})
    ET.SubElement(principal, f"{{{namespace}}}UserId").text = username
    ET.SubElement(principal, f"{{{namespace}}}LogonType").text = "InteractiveToken"
    ET.SubElement(principal, f"{{{namespace}}}RunLevel").text = "LeastPrivilege"
    settings = ET.SubElement(task, f"{{{namespace}}}Settings")
    ET.SubElement(settings, f"{{{namespace}}}MultipleInstancesPolicy").text = "IgnoreNew"
    ET.SubElement(settings, f"{{{namespace}}}Hidden").text = "true"
    restart = ET.SubElement(settings, f"{{{namespace}}}RestartOnFailure")
    ET.SubElement(restart, f"{{{namespace}}}Interval").text = "PT1M"
    ET.SubElement(restart, f"{{{namespace}}}Count").text = "3"
    ET.SubElement(settings, f"{{{namespace}}}ExecutionTimeLimit").text = "PT0S"
    actions = ET.SubElement(task, f"{{{namespace}}}Actions", {"Context": "Author"})
    execute = ET.SubElement(actions, f"{{{namespace}}}Exec")
    ET.SubElement(execute, f"{{{namespace}}}Command").text = launch_argv[0]
    ET.SubElement(execute, f"{{{namespace}}}Arguments").text = subprocess.list2cmdline(
        list(launch_argv[1:])
    )
    return ET.tostring(task, encoding="utf-8", xml_declaration=True)


def build_service_definition(
    config_path: Path,
    *,
    target: WorkerPlatform | None = None,
    home: Path | None = None,
    executable: Path | None = None,
    username: str | None = None,
    uid: int | None = None,
) -> WorkerServiceDefinition:
    """Build a user-scoped service using typed argv and no command shell."""

    target = target or WorkerPlatform.current()
    home = (home or Path.home()).expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    executable = (executable or _default_executable(target)).expanduser().resolve()
    launch_argv = (
        str(executable),
        "-m",
        "bashgym.campaigns.worker_service",
        "run",
        "--config",
        str(config_path),
    )
    for argument in launch_argv:
        _reject_controls(argument)

    if target is WorkerPlatform.WINDOWS:
        definition_path = home / "AppData" / "Local" / "BashGym" / "campaign-worker.xml"
        return WorkerServiceDefinition(
            platform=target,
            config_path=config_path,
            definition_path=definition_path,
            launch_argv=launch_argv,
            definition_payload=_windows_xml(launch_argv, username or getpass.getuser()),
            install_argvs=((
                "schtasks.exe",
                "/Create",
                "/TN",
                TASK_NAME,
                "/XML",
                str(definition_path),
                "/F",
            ),),
            status_argv=("schtasks.exe", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"),
            uninstall_argvs=(("schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"),),
        )

    if target is WorkerPlatform.LINUX:
        definition_path = home / ".config" / "systemd" / "user" / SYSTEMD_UNIT_NAME
        exec_start = " ".join(_systemd_quote(argument) for argument in launch_argv)
        payload = (
            "[Unit]\nDescription=BashGym resident campaign worker\n\n"
            "[Service]\nType=simple\n"
            f"ExecStart={exec_start}\n"
            "Restart=on-failure\nRestartSec=5\n\n"
            "[Install]\nWantedBy=default.target\n"
        ).encode()
        return WorkerServiceDefinition(
            platform=target,
            config_path=config_path,
            definition_path=definition_path,
            launch_argv=launch_argv,
            definition_payload=payload,
            install_argvs=(
                ("systemctl", "--user", "daemon-reload"),
                ("systemctl", "--user", "enable", "--now", SYSTEMD_UNIT_NAME),
            ),
            status_argv=(
                "systemctl",
                "--user",
                "show",
                SYSTEMD_UNIT_NAME,
                "--no-pager",
                "--property=ActiveState,SubState,MainPID,NRestarts,ExecMainStatus",
            ),
            uninstall_argvs=(
                ("systemctl", "--user", "disable", "--now", SYSTEMD_UNIT_NAME),
            ),
            post_uninstall_argvs=(("systemctl", "--user", "daemon-reload"),),
        )

    definition_path = home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    user_uid = uid if uid is not None else getattr(os, "getuid", lambda: 0)()
    domain = f"gui/{user_uid}"
    payload = plistlib.dumps(
        {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": list(launch_argv),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 5,
            "ProcessType": "Background",
            "StandardOutPath": str(home / "Library" / "Logs" / "bashgym-campaign-worker.log"),
            "StandardErrorPath": str(
                home / "Library" / "Logs" / "bashgym-campaign-worker.error.log"
            ),
        },
        fmt=plistlib.FMT_XML,
        sort_keys=True,
    )
    return WorkerServiceDefinition(
        platform=target,
        config_path=config_path,
        definition_path=definition_path,
        launch_argv=launch_argv,
        definition_payload=payload,
        install_argvs=(
            ("launchctl", "bootstrap", domain, str(definition_path)),
            ("launchctl", "kickstart", "-k", f"{domain}/{LAUNCHD_LABEL}"),
        ),
        status_argv=("launchctl", "print", f"{domain}/{LAUNCHD_LABEL}"),
        uninstall_argvs=(("launchctl", "bootout", domain, str(definition_path)),),
    )


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> CommandResult: ...


def run_command(argv: Sequence[str]) -> CommandResult:
    completed = subprocess.run(  # noqa: S603 - fixed executable + typed argv, never a shell
        list(argv),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class WorkerServiceManager:
    """Install and inspect one generated per-user service definition."""

    def __init__(self, runner: CommandRunner = run_command):
        self._runner = runner

    def install(
        self, definition: WorkerServiceDefinition, config: WorkerRunConfig
    ) -> tuple[CommandResult, ...]:
        write_worker_config(definition.config_path, config)
        _atomic_write(definition.definition_path, definition.definition_payload)
        results = tuple(self._runner(argv) for argv in definition.install_argvs)
        if any(result.returncode != 0 for result in results):
            raise WorkerServiceError("campaign_worker_service_install_failed")
        return results

    def status(
        self,
        definition: WorkerServiceDefinition,
        controller: ControllerStatusProjection,
        lifecycle: WorkerLifecycleStatus | None = None,
    ) -> dict[str, Any]:
        result = self._runner(definition.status_argv)
        return {
            "schema_version": "campaign_worker_service_status.v1",
            "platform": definition.platform.value,
            "installed": definition.definition_path.is_file(),
            "supervisor_returncode": result.returncode,
            # Supervisor output commonly includes local usernames, executable
            # paths, and command arguments.  Project only the bounded state.
            "supervisor_state": "available" if result.returncode == 0 else "unavailable",
            "lifecycle": lifecycle.model_dump(mode="json") if lifecycle else None,
            "controller": controller.model_dump(mode="json"),
        }

    def uninstall(self, definition: WorkerServiceDefinition) -> tuple[CommandResult, ...]:
        results = [self._runner(argv) for argv in definition.uninstall_argvs]
        definition.definition_path.unlink(missing_ok=True)
        results.extend(self._runner(argv) for argv in definition.post_uninstall_argvs)
        if any(result.returncode != 0 for result in results):
            raise WorkerServiceError("campaign_worker_service_uninstall_failed")
        return tuple(results)


def _controller_from_config(config: WorkerRunConfig) -> ControllerStatusProjection:
    if not config.database_path.is_file():
        return project_controller_status(None, config.data_directory)
    repository = CampaignRepository(config.database_path)
    repository.initialize()
    return project_controller_status(repository, config.data_directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bashgym.campaigns.worker_service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "install", "status", "uninstall"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_worker_config(args.config)
        if args.command == "run":
            run_foreground(config)
            return 0
        definition = build_service_definition(args.config)
        manager = WorkerServiceManager()
        if args.command == "install":
            manager.install(definition, config)
            result: dict[str, Any] = {"ok": True, "operation": "install"}
        elif args.command == "status":
            result = manager.status(
                definition,
                _controller_from_config(config),
                _read_lifecycle_status(config.status_path),
            )
        else:
            manager.uninstall(definition)
            result = {"ok": True, "operation": "uninstall"}
        print(json.dumps(result, sort_keys=True))
        return 0
    except WorkerServiceError as exc:
        print(json.dumps({"ok": False, "code": exc.code}, sort_keys=True), file=sys.stderr)
        return 1
    except ValueError:
        print(
            json.dumps({"ok": False, "code": "campaign_worker_config_invalid"}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    except Exception:
        # The foreground lifecycle file already carries a safe error class. Do
        # not echo arbitrary exception messages from remote tooling or secrets.
        print(
            json.dumps({"ok": False, "code": "campaign_worker_process_failed"}, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess/module help
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_OFFLINE_GUIDANCE",
    "CONTROLLER_STALE_GUIDANCE",
    "CommandResult",
    "ControllerStatusProjection",
    "WorkerLifecycleStatus",
    "WorkerPlatform",
    "WorkerRunConfig",
    "WorkerServiceDefinition",
    "WorkerServiceError",
    "WorkerServiceManager",
    "build_service_definition",
    "build_worker",
    "load_approved_remote_profiles",
    "main",
    "project_controller_status",
    "read_worker_config",
    "run_foreground",
    "write_worker_config",
]
