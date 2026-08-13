"""Safe resident-worker launch configuration and per-user service definitions.

This module deliberately keeps service installation separate from the campaign
scheduler.  It only launches the existing foreground worker with a restricted
configuration file; supervisors remain responsible for restarting a crashed
process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from ipaddress import ip_address
from pathlib import Path, PurePosixPath
from secrets import token_urlsafe
from time import monotonic, sleep
from typing import Any, Literal, Protocol
from uuid import uuid4

import psutil
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bashgym.api_base import open_api_url
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.autoresearch import AutoResearchCampaignCore, AutoResearchRepository
from bashgym.campaigns.autoresearch_evidence import (
    CampaignEvaluationProjector,
    SealedEvaluationReader,
)
from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator
from bashgym.campaigns.contracts import StageKind, canonical_hash, utc_now
from bashgym.campaigns.lineage import (
    ApprovedSourceRepositoryProfile,
    GitHypothesisLineageManager,
    GitLineageError,
)
from bashgym.campaigns.persistence import CampaignRepository, LeaseRecord
from bashgym.campaigns.remote import ApprovedRemoteExecutorProfile, RemoteTrainingAdapter
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.worker import CampaignWorker, scheduler_lease_key
from bashgym.config import state_root_digest
from bashgym.gym.remote_trainer import SSHConfig
from bashgym.mcp.policy import resolve_secret_reference, validate_secret_ref_name
from bashgym.secrets import get_secret, set_secret

MAX_CONFIG_BYTES = 64 * 1024
MAX_API_HEALTH_RESPONSE_BYTES = 16 * 1024
MAX_API_HEALTH_TIMEOUT_SECONDS = 5.0
TASK_NAME = r"\BashGym\Campaign Worker"
SYSTEMD_UNIT_NAME = "bashgym-campaign-worker.service"
LAUNCHD_LABEL = "com.ghostpeony.bashgym.campaign-worker"
API_TASK_NAME = r"\BashGym\API"
API_SYSTEMD_UNIT_NAME = "bashgym-api.service"
API_LAUNCHD_LABEL = "com.ghostpeony.bashgym.api"
WINDOWS_RUN_KEY = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
CONTROLLER_OFFLINE_GUIDANCE = (
    "Install or restart the per-user campaign worker. Durable campaigns remain paused "
    "and remote training runs remain untouched."
)
CONTROLLER_STALE_GUIDANCE = (
    "The campaign worker stopped renewing its scheduler lease. Restart it and inspect "
    "the bounded lifecycle status before advancing durable campaigns."
)
WORKER_STARTING_GUIDANCE = (
    "The desktop-managed campaign worker is starting. Durable campaigns remain paused "
    "until its scheduler lease is online."
)
WORKER_RESTARTING_GUIDANCE = (
    "The desktop-managed campaign worker is restarting. Durable campaigns remain paused "
    "until readiness is restored."
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

    schema_version: Literal["campaign_worker_config.v1"] = "campaign_worker_config.v1"
    controller_owner_id: str | None = Field(default=None, min_length=1, max_length=160)
    data_directory: Path
    database_path: Path
    artifact_root: Path
    seal_key_ref: str = "BASHGYM_CAMPAIGN_SEAL_KEY"
    seal_key_version: str = Field(default="v1", min_length=1, max_length=64)
    approved_remote_profiles: tuple[ApprovedRemoteExecutorProfile, ...] = ()
    approved_source_profiles: tuple[ApprovedSourceRepositoryProfile, ...] = ()
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

    @field_validator("controller_owner_id")
    @classmethod
    def bounded_controller_owner(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not all(character.isalnum() or character in "_.:-" for character in value):
            raise ValueError("controller owner ID must be a bounded identifier")
        return value

    @field_validator("compute_profile_ids")
    @classmethod
    def canonical_profile_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("compute_profile_ids must be sorted and unique")
        for profile_id in value:
            if (
                not profile_id
                or len(profile_id) > 128
                or any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                    for character in profile_id
                )
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

    @field_validator("approved_source_profiles")
    @classmethod
    def canonical_source_profiles(
        cls, value: tuple[ApprovedSourceRepositoryProfile, ...]
    ) -> tuple[ApprovedSourceRepositoryProfile, ...]:
        ordering = tuple(profile.profile_id for profile in value)
        if tuple(sorted(ordering)) != ordering or len(set(ordering)) != len(ordering):
            raise ValueError("approved source profiles must be sorted and unique")
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
        approved_source_profiles: tuple[ApprovedSourceRepositoryProfile, ...] = (),
        compute_profile_ids: tuple[str, ...] = (),
    ) -> WorkerRunConfig:
        root = data_directory.expanduser().resolve()
        return cls(
            data_directory=root,
            database_path=root / "campaigns" / "campaigns.sqlite3",
            artifact_root=root / "campaigns" / "artifacts",
            approved_remote_profiles=approved_remote_profiles,
            approved_source_profiles=approved_source_profiles,
            compute_profile_ids=compute_profile_ids,
        )

    @property
    def status_path(self) -> Path:
        return self.data_directory / "campaigns" / "worker-status.v1.json"


@dataclass(frozen=True)
class WorkerBootstrapResult:
    """Secret-free result of preparing one installation-owned worker config."""

    config: WorkerRunConfig
    config_path: Path
    config_created: bool
    seal_key_created: bool


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
    controller_observation_version: int = Field(default=0, ge=0)
    state: Literal["online", "stale", "offline"]
    code: str
    observed_at: datetime
    heartbeat_age_seconds: float | None = Field(default=None, ge=0)
    guidance: str | None = None
    owner_id: str | None = None
    generation: int | None = None
    heartbeat_at: datetime | None = None
    expires_at: datetime | None = None


class DesktopWorkerStatusProjection(BaseModel):
    """Bounded runtime health safe for authenticated renderer projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "campaign_desktop_worker_status.v1"
    managed: bool
    online: bool
    state: Literal[
        "online",
        "starting",
        "restarting",
        "stale",
        "offline",
        "stopped",
        "failed",
    ]
    code: str
    observed_at: datetime
    thread_alive: bool
    restart_count: int = Field(default=0, ge=0)
    controller: ControllerStatusProjection
    guidance: str | None = None


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


def _migrate_legacy_evaluator_output_args(payload: dict[str, Any]) -> bool:
    """Remove only the v1 evaluator output flag now owned by the launch ABI."""

    if payload.get("schema_version", "campaign_worker_config.v1") != "campaign_worker_config.v1":
        return False
    profiles = payload.get("approved_remote_profiles")
    if not isinstance(profiles, list):
        return False
    changed = False
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        profile_changed = False
        stages = profile.get("stages")
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if (
                not isinstance(stage, dict)
                or stage.get("stage") != StageKind.DEVELOPMENT_EVALUATION.value
                or not isinstance(stage.get("script_args"), list)
            ):
                continue
            arguments = stage["script_args"]
            migrated: list[Any] = []
            index = 0
            stage_changed = False
            while index < len(arguments):
                argument = arguments[index]
                normalized = argument.casefold() if isinstance(argument, str) else ""
                if normalized == "--output":
                    index += 1
                    if index < len(arguments):
                        following = arguments[index]
                        if isinstance(following, str) and not following.startswith("--"):
                            index += 1
                    stage_changed = True
                    continue
                if normalized.startswith("--output="):
                    index += 1
                    stage_changed = True
                    continue
                migrated.append(argument)
                index += 1
            if stage_changed:
                stage["script_args"] = migrated
                profile_changed = True
        if profile_changed:
            profile.pop("profile_digest", None)
            profile["profile_digest"] = ApprovedRemoteExecutorProfile.model_validate(
                profile
            ).profile_digest
            changed = True
    return changed


def read_worker_config(path: Path) -> WorkerRunConfig:
    resolved = path.expanduser()
    payload = _read_restricted_json(resolved)
    if _migrate_legacy_evaluator_output_args(payload):
        original = resolved.read_bytes()
        config = WorkerRunConfig.model_validate(payload)
        backup = resolved.with_name(f"{resolved.name}.pre-migration.v1.json")
        receipt = resolved.with_name(f"{resolved.name}.migration.v1.json")
        if backup.exists():
            if backup.is_symlink() or not backup.is_file() or backup.read_bytes() != original:
                raise WorkerServiceError("campaign_worker_config_migration_conflict")
        else:
            _atomic_write(backup, original)
        write_worker_config(resolved, config)
        migrated = resolved.read_bytes()
        _atomic_write(
            receipt,
            json.dumps(
                {
                    "schema_version": "campaign_worker_config_migration.v1",
                    "migration": "remove_legacy_evaluator_output_args",
                    "before_sha256": hashlib.sha256(original).hexdigest(),
                    "after_sha256": hashlib.sha256(migrated).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n",
        )
        return config
    return WorkerRunConfig.model_validate(payload)


def write_worker_config(path: Path, config: WorkerRunConfig) -> None:
    payload = config.model_dump_json(indent=2).encode("utf-8") + b"\n"
    _atomic_write(path.expanduser(), payload)


def ensure_worker_bootstrap(
    data_directory: Path,
    *,
    config_path: Path | None = None,
    secret_resolver: Callable[[str], str | None] = get_secret,
    secret_writer: Callable[[str, str], None] = set_secret,
    key_factory: Callable[[], str] = token_urlsafe,
) -> WorkerBootstrapResult:
    """Prepare an idempotent worker config and seal key for one installation.

    Existing installation-owned configuration is preserved verbatim. A config
    bound to another data directory is rejected instead of being silently
    rewritten, and no secret value is ever returned or written into the config.
    """

    root = data_directory.expanduser().resolve()
    resolved_path = (
        config_path.expanduser().resolve()
        if config_path is not None
        else root / "campaigns" / "worker-config.v1.json"
    )
    if resolved_path.is_symlink():
        raise WorkerServiceError("campaign_worker_config_not_regular")
    if resolved_path.exists():
        config = read_worker_config(resolved_path)
        config_created = False
    else:
        config = WorkerRunConfig.for_data_directory(root)
        write_worker_config(resolved_path, config)
        config_created = True
    if config.data_directory != root:
        raise WorkerServiceError("campaign_worker_config_data_directory_mismatch")

    try:
        existing_key = secret_resolver(config.seal_key_ref)
    except Exception as exc:
        raise WorkerServiceError("campaign_worker_seal_key_unavailable") from exc
    seal_key_created = False
    if existing_key and len(existing_key.encode("utf-8")) < 32:
        raise WorkerServiceError("campaign_worker_seal_key_invalid")
    if not existing_key or not existing_key.strip():
        material = key_factory()
        if (
            not isinstance(material, str)
            or not material.strip()
            or len(material.encode("utf-8")) < 32
        ):
            raise WorkerServiceError("campaign_worker_seal_key_generation_failed")
        try:
            secret_writer(config.seal_key_ref, material)
            persisted_key = secret_resolver(config.seal_key_ref)
        except Exception as exc:
            raise WorkerServiceError("campaign_worker_seal_key_persistence_failed") from exc
        if (
            not persisted_key
            or not persisted_key.strip()
            or len(persisted_key.encode("utf-8")) < 32
        ):
            raise WorkerServiceError("campaign_worker_seal_key_persistence_failed")
        seal_key_created = True
    return WorkerBootstrapResult(
        config=config,
        config_path=resolved_path,
        config_created=config_created,
        seal_key_created=seal_key_created,
    )


def _read_lifecycle_status(path: Path) -> WorkerLifecycleStatus | None:
    if not path.exists():
        return None
    try:
        return WorkerLifecycleStatus.model_validate(_read_restricted_json(path))
    except (WorkerServiceError, ValueError):
        return None


def _write_lifecycle_status(path: Path, status: WorkerLifecycleStatus) -> None:
    _atomic_write(path, status.model_dump_json(indent=2).encode("utf-8") + b"\n")


class _LeaseReader(Protocol):
    def get_lease(self, lease_key: str) -> LeaseRecord | None: ...


class _LeaseSnapshotReader:
    def __init__(self, lease: LeaseRecord | None) -> None:
        self._lease = lease

    def get_lease(self, lease_key: str) -> LeaseRecord | None:
        if self._lease is None or self._lease.lease_key != lease_key:
            return None
        return self._lease


def project_controller_status(
    repository: _LeaseReader | None,
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
            controller_observation_version=0,
            state="offline",
            code="controller_offline",
            observed_at=observed_at,
            guidance=CONTROLLER_OFFLINE_GUIDANCE,
        )
    heartbeat_age_seconds = max(0.0, (observed_at - lease.heartbeat_at).total_seconds())
    if lease.expires_at <= observed_at or heartbeat_age_seconds > stale_after.total_seconds():
        return ControllerStatusProjection(
            online=False,
            controller_observation_version=lease.controller_observation_version,
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
        controller_observation_version=lease.controller_observation_version,
        state="online",
        code="controller_online",
        observed_at=observed_at,
        heartbeat_age_seconds=heartbeat_age_seconds,
        owner_id=lease.owner_id,
        generation=lease.generation,
        heartbeat_at=lease.heartbeat_at,
        expires_at=lease.expires_at,
    )


def project_controller_status_from_database(
    database_path: Path,
    data_directory: Path,
    *,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(seconds=15),
) -> ControllerStatusProjection:
    """Read one controller lease without creating or migrating campaign state."""

    path = database_path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        return project_controller_status(None, data_directory, now=now, stale_after=stale_after)
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                """
                SELECT lease_key, owner_id, generation, controller_observation_version,
                       expires_at, heartbeat_at
                FROM campaign_scheduler_leases
                WHERE lease_key = ?
                """,
                (scheduler_lease_key(data_directory),),
            ).fetchone()
        finally:
            connection.close()
        lease = None
        if row is not None:
            expires_at = datetime.fromisoformat(str(row["expires_at"]))
            heartbeat_at = datetime.fromisoformat(str(row["heartbeat_at"]))
            if expires_at.utcoffset() is None or heartbeat_at.utcoffset() is None:
                raise ValueError("campaign_scheduler_lease_invalid")
            lease = LeaseRecord(
                lease_key=str(row["lease_key"]),
                owner_id=str(row["owner_id"]),
                generation=int(row["generation"]),
                controller_observation_version=int(row["controller_observation_version"]),
                expires_at=expires_at,
                heartbeat_at=heartbeat_at,
            )
    except (OSError, TypeError, ValueError, sqlite3.Error):
        lease = None
    return project_controller_status(
        _LeaseSnapshotReader(lease),
        data_directory,
        now=now,
        stale_after=stale_after,
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


def load_approved_source_profiles(
    config: WorkerRunConfig,
) -> dict[str, ApprovedSourceRepositoryProfile]:
    """Load and verify installation-owned source repositories and path scopes."""

    registry: dict[str, ApprovedSourceRepositoryProfile] = {}
    manager = GitHypothesisLineageManager(config.data_directory / "campaigns" / "source-worktrees")
    try:
        for profile in config.approved_source_profiles:
            manager.verify_profile(profile)
            if profile.profile_id in registry:
                raise ValueError("duplicate approved source profile resolution key")
            registry[profile.profile_id] = profile
    except (GitLineageError, OSError, ValueError) as exc:
        raise WorkerServiceError("campaign_worker_source_profile_material_invalid") from exc
    return registry


def validate_code_lineage_execution_bindings(
    remote_profiles: Mapping[tuple[str, str], ApprovedRemoteExecutorProfile],
    source_profiles: Mapping[str, ApprovedSourceRepositoryProfile],
) -> None:
    """Cross-check every optional code entrypoint against its logical source profile."""

    try:
        for profile in remote_profiles.values():
            for stage in profile.stages:
                binding = stage.code_lineage_binding
                if binding is None:
                    continue
                source = source_profiles[binding.source_repository_profile_id]
                repository = source.repository_path.expanduser().resolve()
                entrypoint = repository.joinpath(*PurePosixPath(binding.entrypoint_path).parts)
                resolved = entrypoint.resolve(strict=True)
                resolved.relative_to(repository)
                if entrypoint.is_symlink() or not resolved.is_file():
                    raise ValueError("code lineage entrypoint is not a regular file")
    except (KeyError, OSError, ValueError) as exc:
        raise WorkerServiceError("campaign_worker_code_lineage_execution_binding_invalid") from exc


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
    repository = AutoResearchRepository(config.database_path)
    repository.initialize()
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    sealer = ArtifactSealer(key.encode("utf-8"), key_version=config.seal_key_version)
    evaluation_reader = SealedEvaluationReader(sealer)
    autoresearch_core = AutoResearchCampaignCore(repository, evaluation_reader=evaluation_reader)
    evaluation_projector = CampaignEvaluationProjector(
        repository, autoresearch_core.ledger, evaluation_reader
    )
    autoresearch_loop = AutoResearchLoopCoordinator(
        repository, evaluation_projector, autoresearch_core
    )
    remote_executor_profiles = load_approved_remote_profiles(config)
    source_repository_profiles = load_approved_source_profiles(config)
    validate_code_lineage_execution_bindings(remote_executor_profiles, source_repository_profiles)
    return CampaignWorker(
        repository,
        config.artifact_root,
        sealer,
        data_directory=config.data_directory,
        worker_id=config.controller_owner_id,
        leader_ttl=timedelta(seconds=config.leader_ttl_seconds),
        action_ttl=timedelta(seconds=config.action_ttl_seconds),
        remote_adapters=adapter_loader(config),
        remote_executor_profiles=remote_executor_profiles,
        source_repository_profiles=source_repository_profiles,
        lineage_manager=GitHypothesisLineageManager(
            config.data_directory / "campaigns" / "source-worktrees"
        ),
        autoresearch_loop=autoresearch_loop,
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


class DesktopWorkerSupervisor:
    """Own exactly one restartable campaign-worker thread for a desktop backend."""

    def __init__(
        self,
        config: WorkerRunConfig,
        *,
        worker_factory: Callable[[WorkerRunConfig], ForegroundWorker] = build_worker,
        foreground_runner: Callable[..., None] = run_foreground,
        restart_delay_seconds: float = 1.0,
    ) -> None:
        if restart_delay_seconds <= 0:
            raise ValueError("restart_delay_seconds must be positive")
        self.config = config
        self._worker_factory = worker_factory
        self._foreground_runner = foreground_runner
        self._restart_delay_seconds = restart_delay_seconds
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker: ForegroundWorker | None = None
        self._started = False
        self._stopped = False
        self._restart_count = 0
        self._last_failure_code: str | None = None

    @property
    def is_alive(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    @property
    def restart_count(self) -> int:
        with self._lock:
            return self._restart_count

    def _capture_worker(self, config: WorkerRunConfig) -> ForegroundWorker:
        worker = self._worker_factory(config)
        with self._lock:
            self._worker = worker
            stop_requested = self._stop_event.is_set()
        if stop_requested:
            worker.request_stop()
        return worker

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    self._foreground_runner(
                        self.config,
                        worker_factory=self._capture_worker,
                        install_signal_handlers=False,
                    )
                    if self._stop_event.is_set():
                        break
                    failure_code = "campaign_worker_exited_unexpectedly"
                except Exception:
                    if self._stop_event.is_set():
                        break
                    failure_code = "campaign_worker_crashed"
                finally:
                    with self._lock:
                        self._worker = None
                with self._lock:
                    self._restart_count += 1
                    self._last_failure_code = failure_code
                    restart_count = self._restart_count
                restart_delay = min(
                    self._restart_delay_seconds * (2 ** min(restart_count - 1, 5)),
                    30.0,
                )
                if self._stop_event.wait(restart_delay):
                    break
        finally:
            with self._lock:
                self._worker = None
                self._stopped = True

    def start(self) -> bool:
        """Start the owned thread once; return False for a duplicate live start."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._started = True
            self._stopped = False
            self._last_failure_code = None
            self._thread = threading.Thread(
                target=self._run,
                name="bashgym-campaign-worker",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self, *, timeout_seconds: float = 10.0) -> bool:
        """Request graceful shutdown and report whether the owned thread exited."""

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        with self._lock:
            thread = self._thread
            worker = self._worker
            self._stop_event.set()
        if worker is not None:
            worker.request_stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_seconds)
        stopped = thread is None or not thread.is_alive()
        if stopped:
            with self._lock:
                self._stopped = True
        return stopped

    def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 3.0,
        poll_seconds: float = 0.05,
    ) -> bool:
        """Wait for the real scheduler lease, never merely for a live thread."""

        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("readiness timing must be positive")
        repository = CampaignRuntimeRepository(self.config.database_path)
        repository.initialize()
        deadline = monotonic() + timeout_seconds
        while True:
            if project_controller_status(repository, self.config.data_directory).online:
                return True
            remaining = deadline - monotonic()
            with self._lock:
                stopped = self._stopped
            if remaining <= 0 or (stopped and not self.is_alive):
                return False
            self._stop_event.wait(min(poll_seconds, remaining))

    def status(
        self,
        *,
        repository: CampaignRepository | None = None,
    ) -> DesktopWorkerStatusProjection:
        """Project thread and lease state without paths, secrets, or exception text."""

        if repository is None:
            runtime_repository = CampaignRuntimeRepository(self.config.database_path)
            runtime_repository.initialize()
            repository = runtime_repository
        observed_at = utc_now()
        controller = project_controller_status(
            repository,
            self.config.data_directory,
            now=observed_at,
        )
        with self._lock:
            thread_alive = bool(self._thread and self._thread.is_alive())
            restart_count = self._restart_count
            last_failure_code = self._last_failure_code
            started = self._started
            stopped = self._stopped
        if controller.online:
            state = "online"
            code = "worker_online"
            guidance = None
        elif started and stopped and not thread_alive:
            state = "stopped"
            code = "worker_stopped"
            guidance = CONTROLLER_OFFLINE_GUIDANCE
        elif controller.state == "stale":
            state = "stale"
            code = "worker_controller_stale"
            guidance = controller.guidance
        elif thread_alive and last_failure_code:
            state = "restarting"
            code = "worker_restarting"
            guidance = WORKER_RESTARTING_GUIDANCE
        elif thread_alive:
            state = "starting"
            code = "worker_starting"
            guidance = WORKER_STARTING_GUIDANCE
        elif last_failure_code:
            state = "failed"
            code = "worker_failed"
            guidance = CONTROLLER_OFFLINE_GUIDANCE
        else:
            state = "offline"
            code = "worker_offline"
            guidance = CONTROLLER_OFFLINE_GUIDANCE
        return DesktopWorkerStatusProjection(
            managed=True,
            online=controller.online,
            state=state,
            code=code,
            observed_at=observed_at,
            thread_alive=thread_alive,
            restart_count=restart_count,
            controller=controller,
            guidance=guidance,
        )


@dataclass(frozen=True)
class WorkerServiceDefinition:
    platform: WorkerPlatform
    config_path: Path
    definition_path: Path
    launch_argv: tuple[str, ...]
    definition_payload: bytes
    install_argvs: tuple[tuple[str, ...], ...]
    start_argvs: tuple[tuple[str, ...], ...]
    stop_argvs: tuple[tuple[str, ...], ...]
    install_starts_service: bool
    status_argv: tuple[str, ...]
    uninstall_argvs: tuple[tuple[str, ...], ...]
    post_uninstall_argvs: tuple[tuple[str, ...], ...] = ()


def _reject_controls(value: str) -> str:
    if any(ord(character) < 32 for character in value):
        raise WorkerServiceError("campaign_worker_service_argument_invalid")
    return value


def _systemd_quote(value: str) -> str:
    value = _reject_controls(value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$")
    return f'"{escaped}"'


def _default_executable(target: WorkerPlatform) -> Path:
    executable = Path(sys.executable).resolve()
    if target is WorkerPlatform.WINDOWS and executable.name.casefold() == "python.exe":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            return pythonw
    return executable


@dataclass(frozen=True)
class ApiServiceDefinition:
    platform: WorkerPlatform
    definition_path: Path
    launch_argv: tuple[str, ...]
    definition_payload: bytes
    install_argvs: tuple[tuple[str, ...], ...]
    start_argvs: tuple[tuple[str, ...], ...]
    stop_argvs: tuple[tuple[str, ...], ...]
    install_starts_service: bool
    status_argv: tuple[str, ...]
    uninstall_argvs: tuple[tuple[str, ...], ...]
    post_uninstall_argvs: tuple[tuple[str, ...], ...] = ()


class BackgroundServiceLaunch(BaseModel):
    """One no-elevation Windows process definition stored under the user profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bashgym_background_service.v1"] = "bashgym_background_service.v1"
    launch_argv: tuple[str, ...] = Field(min_length=1, max_length=128)
    receipt_path: Path
    stdout_path: Path
    stderr_path: Path
    restart_delay_seconds: float = Field(default=5.0, ge=0.1, le=60.0)

    @field_validator("launch_argv")
    @classmethod
    def bounded_launch_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument or len(argument) > 4096 for argument in value):
            raise ValueError("background service arguments must be bounded")
        for argument in value:
            _reject_controls(argument)
        return value

    @field_validator("receipt_path", "stdout_path", "stderr_path")
    @classmethod
    def absolute_service_path(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.is_absolute():
            raise ValueError("background service paths must be absolute")
        return resolved


class BackgroundProcessReceipt(BaseModel):
    """Exact process identity used by start, status, and stop helpers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bashgym_background_process.v1"] = "bashgym_background_process.v1"
    supervisor_pid: int = Field(gt=0)
    supervisor_create_time: float = Field(gt=0)
    launch_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_token: str = Field(pattern=r"^[a-f0-9]{32}$")
    child_pid: int | None = Field(default=None, gt=0)
    child_create_time: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def complete_child_identity(self) -> BackgroundProcessReceipt:
        if (self.child_pid is None) != (self.child_create_time is None):
            raise ValueError("background child identity must be complete")
        return self


def _background_helper_argv(
    executable: str, operation: str, definition_path: Path
) -> tuple[str, ...]:
    return (
        executable,
        "-m",
        "bashgym.campaigns.worker_service",
        f"{operation}-background",
        "--definition",
        str(definition_path),
    )


def _windows_value_name(task_name: str) -> str:
    return task_name.strip("\\").replace("\\", " ")


def _resident_service_fields(
    *,
    target: WorkerPlatform,
    home: Path,
    launch_argv: tuple[str, ...],
    task_name: str,
    systemd_unit_name: str,
    launchd_label: str,
    description: str,
    definition_stem: str,
    log_stem: str,
    username: str | None,
    uid: int | None,
) -> dict[str, Any]:
    """Build the shared per-user supervisor contract for one typed command."""

    for argument in launch_argv:
        _reject_controls(argument)
    if target is WorkerPlatform.WINDOWS:
        local_root = home / "AppData" / "Local" / "BashGym"
        definition_path = local_root / f"{definition_stem}.json"
        launch = BackgroundServiceLaunch(
            launch_argv=launch_argv,
            receipt_path=local_root / f"{definition_stem}.process.json",
            stdout_path=local_root / "logs" / f"{log_stem}.log",
            stderr_path=local_root / "logs" / f"{log_stem}.error.log",
        )
        start_argv = _background_helper_argv(launch_argv[0], "start", definition_path)
        value_name = _windows_value_name(task_name)
        return {
            "platform": target,
            "definition_path": definition_path,
            "launch_argv": launch_argv,
            "definition_payload": launch.model_dump_json(indent=2).encode("utf-8") + b"\n",
            "install_argvs": (
                (
                    "reg.exe",
                    "ADD",
                    WINDOWS_RUN_KEY,
                    "/V",
                    value_name,
                    "/T",
                    "REG_SZ",
                    "/D",
                    subprocess.list2cmdline(list(start_argv)),
                    "/F",
                ),
            ),
            "start_argvs": (start_argv,),
            "stop_argvs": (_background_helper_argv(launch_argv[0], "stop", definition_path),),
            "install_starts_service": False,
            "status_argv": _background_helper_argv(launch_argv[0], "status", definition_path),
            "uninstall_argvs": (
                _background_helper_argv(launch_argv[0], "stop", definition_path),
                ("reg.exe", "DELETE", WINDOWS_RUN_KEY, "/V", value_name, "/F"),
            ),
        }
    if target is WorkerPlatform.LINUX:
        definition_path = home / ".config" / "systemd" / "user" / systemd_unit_name
        exec_start = " ".join(_systemd_quote(argument) for argument in launch_argv)
        payload = (
            f"[Unit]\nDescription={description}\n\n"
            "[Service]\nType=simple\n"
            f"ExecStart={exec_start}\n"
            "Restart=on-failure\nRestartSec=5\n\n"
            "[Install]\nWantedBy=default.target\n"
        ).encode()
        return {
            "platform": target,
            "definition_path": definition_path,
            "launch_argv": launch_argv,
            "definition_payload": payload,
            "install_argvs": (
                ("systemctl", "--user", "daemon-reload"),
                ("systemctl", "--user", "enable", "--now", systemd_unit_name),
            ),
            "start_argvs": (("systemctl", "--user", "start", systemd_unit_name),),
            "stop_argvs": (("systemctl", "--user", "stop", systemd_unit_name),),
            "install_starts_service": True,
            "status_argv": (
                "systemctl",
                "--user",
                "show",
                systemd_unit_name,
                "--no-pager",
                "--property=ActiveState,SubState,MainPID,NRestarts,ExecMainStatus",
            ),
            "uninstall_argvs": (("systemctl", "--user", "disable", "--now", systemd_unit_name),),
            "post_uninstall_argvs": (("systemctl", "--user", "daemon-reload"),),
        }
    definition_path = home / "Library" / "LaunchAgents" / f"{launchd_label}.plist"
    user_uid = uid if uid is not None else getattr(os, "getuid", lambda: 0)()
    domain = f"gui/{user_uid}"
    payload = plistlib.dumps(
        {
            "Label": launchd_label,
            "ProgramArguments": list(launch_argv),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 5,
            "ProcessType": "Background",
            "StandardOutPath": str(home / "Library" / "Logs" / f"{log_stem}.log"),
            "StandardErrorPath": str(home / "Library" / "Logs" / f"{log_stem}.error.log"),
        },
        fmt=plistlib.FMT_XML,
        sort_keys=True,
    )
    return {
        "platform": target,
        "definition_path": definition_path,
        "launch_argv": launch_argv,
        "definition_payload": payload,
        "install_argvs": (
            ("launchctl", "bootstrap", domain, str(definition_path)),
            ("launchctl", "kickstart", "-k", f"{domain}/{launchd_label}"),
        ),
        "start_argvs": (("launchctl", "kickstart", "-k", f"{domain}/{launchd_label}"),),
        "stop_argvs": (("launchctl", "kill", "SIGTERM", f"{domain}/{launchd_label}"),),
        "install_starts_service": True,
        "status_argv": ("launchctl", "print", f"{domain}/{launchd_label}"),
        "uninstall_argvs": (("launchctl", "bootout", domain, str(definition_path)),),
    }


def build_service_definition(
    config_path: Path,
    *,
    target: WorkerPlatform | None = None,
    home: Path | None = None,
    executable: Path | None = None,
    username: str | None = None,
    uid: int | None = None,
) -> WorkerServiceDefinition:
    """Build a user-scoped worker service using typed argv and no shell."""

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
    fields = _resident_service_fields(
        target=target,
        home=home,
        launch_argv=launch_argv,
        task_name=TASK_NAME,
        systemd_unit_name=SYSTEMD_UNIT_NAME,
        launchd_label=LAUNCHD_LABEL,
        description="BashGym resident campaign worker",
        definition_stem="campaign-worker",
        log_stem="bashgym-campaign-worker",
        username=username,
        uid=uid,
    )
    return WorkerServiceDefinition(config_path=config_path, **fields)


def build_api_service_definition(
    *,
    target: WorkerPlatform | None = None,
    home: Path | None = None,
    executable: Path | None = None,
    username: str | None = None,
    uid: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8003,
    data_directory: Path | None = None,
) -> ApiServiceDefinition:
    """Build the headless API's per-user resident-service definition."""

    if not 1 <= port <= 65535:
        raise WorkerServiceError("bashgym_api_service_argument_invalid")
    target = target or WorkerPlatform.current()
    home = (home or Path.home()).expanduser().resolve()
    executable = (executable or _default_executable(target)).expanduser().resolve()
    launch_argv = (
        str(executable),
        "-m",
        "bashgym.campaigns.worker_service",
        "run-api",
        "--host",
        _reject_controls(host),
        "--port",
        str(port),
        *(
            ("--data-dir", str(data_directory.expanduser().resolve()))
            if data_directory is not None
            else ()
        ),
    )
    fields = _resident_service_fields(
        target=target,
        home=home,
        launch_argv=launch_argv,
        task_name=API_TASK_NAME,
        systemd_unit_name=API_SYSTEMD_UNIT_NAME,
        launchd_label=API_LAUNCHD_LABEL,
        description="BashGym headless API",
        definition_stem="api",
        log_stem="bashgym-api",
        username=username,
        uid=uid,
    )
    return ApiServiceDefinition(**fields)


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


def _supervisor_state(
    target: WorkerPlatform, result: CommandResult
) -> Literal["available", "unavailable"]:
    if result.returncode != 0:
        return "unavailable"
    if target is not WorkerPlatform.LINUX:
        return "available"
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return (
        "available"
        if properties.get("ActiveState") == "active" and properties.get("SubState") == "running"
        else "unavailable"
    )


def _run_checked(
    runner: CommandRunner,
    commands: Sequence[Sequence[str]],
    *,
    failure_code: str,
) -> tuple[CommandResult, ...]:
    results: list[CommandResult] = []
    for argv in commands:
        result = runner(argv)
        results.append(result)
        if result.returncode != 0:
            raise WorkerServiceError(failure_code)
    return tuple(results)


def _read_background_launch(definition_path: Path) -> BackgroundServiceLaunch:
    try:
        return BackgroundServiceLaunch.model_validate(_read_restricted_json(definition_path))
    except WorkerServiceError:
        raise
    except Exception as exc:
        raise WorkerServiceError("bashgym_background_service_invalid") from exc


def _read_background_receipt(
    launch: BackgroundServiceLaunch,
) -> BackgroundProcessReceipt | None:
    if not launch.receipt_path.exists():
        return None
    try:
        return BackgroundProcessReceipt.model_validate(_read_restricted_json(launch.receipt_path))
    except Exception as exc:
        raise WorkerServiceError("bashgym_background_process_receipt_invalid") from exc


def _matching_process(pid: int, create_time: float) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - create_time) > 0.01:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, ValueError):
        return None


@contextmanager
def _background_service_lock(launch: BackgroundServiceLaunch):
    """Serialize current-user start/stop helpers without administrator rights."""

    if os.name == "nt":
        import msvcrt

        def lock(handle) -> None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

        def unlock(handle) -> None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    else:
        import fcntl

        def lock(handle) -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def unlock(handle) -> None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    lock_path = launch.receipt_path.with_suffix(launch.receipt_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = monotonic() + 10
        while True:
            try:
                lock(handle)
                break
            except OSError as exc:
                if monotonic() >= deadline:
                    raise WorkerServiceError("bashgym_background_service_busy") from exc
                sleep(0.05)
        try:
            yield
        finally:
            unlock(handle)


def _terminate_process(process: psutil.Process | None) -> None:
    if process is None:
        return
    try:
        process.terminate()
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        pass


def _stop_receipt_process(
    launch: BackgroundServiceLaunch, receipt: BackgroundProcessReceipt
) -> None:
    if receipt.child_pid is not None and receipt.child_create_time is not None:
        _terminate_process(_matching_process(receipt.child_pid, receipt.child_create_time))
    _terminate_process(_matching_process(receipt.supervisor_pid, receipt.supervisor_create_time))
    launch.receipt_path.unlink(missing_ok=True)


def start_background_service(definition_path: Path) -> None:
    """Start one restart supervisor and persist its exact identity."""

    launch = _read_background_launch(definition_path)
    launch_digest = canonical_hash(list(launch.launch_argv))
    with _background_service_lock(launch):
        receipt = _read_background_receipt(launch)
        if receipt is not None:
            existing = _matching_process(receipt.supervisor_pid, receipt.supervisor_create_time)
            if existing is not None and receipt.launch_digest == launch_digest:
                return
            _stop_receipt_process(launch, receipt)

        launch.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        launch.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        creation_flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
        start_token = uuid4().hex
        supervisor_argv = _background_helper_argv(
            launch.launch_argv[0], "supervise", definition_path.resolve()
        ) + ("--start-token", start_token)
        try:
            with (
                launch.stdout_path.open("ab", buffering=0) as stdout_handle,
                launch.stderr_path.open("ab", buffering=0) as stderr_handle,
            ):
                subprocess.Popen(  # noqa: S603 - typed argv, never a shell
                    supervisor_argv,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    close_fds=True,
                    creationflags=creation_flags,
                )
        except (OSError, psutil.Error) as exc:
            raise WorkerServiceError("bashgym_background_process_start_failed") from exc
        deadline = monotonic() + 10
        while monotonic() < deadline:
            receipt = _read_background_receipt(launch)
            if (
                receipt is not None
                and receipt.start_token == start_token
                and _matching_process(receipt.supervisor_pid, receipt.supervisor_create_time)
                is not None
            ):
                return
            sleep(0.05)
        raise WorkerServiceError("bashgym_background_process_start_failed")


def supervise_background_service(definition_path: Path, *, start_token: str) -> None:
    """Restart the typed service command after an unexpected exit."""

    launch = _read_background_launch(definition_path)
    launch_digest = canonical_hash(list(launch.launch_argv))
    if re.fullmatch(r"[a-f0-9]{32}", start_token) is None:
        raise WorkerServiceError("bashgym_background_supervisor_token_invalid")
    supervisor = psutil.Process(os.getpid())
    _atomic_write(
        launch.receipt_path,
        BackgroundProcessReceipt(
            supervisor_pid=os.getpid(),
            supervisor_create_time=supervisor.create_time(),
            launch_digest=launch_digest,
            start_token=start_token,
        )
        .model_dump_json(indent=2)
        .encode("utf-8")
        + b"\n",
    )

    while True:
        try:
            with (
                launch.stdout_path.open("ab", buffering=0) as stdout_handle,
                launch.stderr_path.open("ab", buffering=0) as stderr_handle,
            ):
                child = subprocess.Popen(  # noqa: S603 - validated typed argv, never a shell
                    launch.launch_argv,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    close_fds=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            child_create_time = psutil.Process(child.pid).create_time()
            with _background_service_lock(launch):
                current = _read_background_receipt(launch)
                if current is None or current.supervisor_pid != os.getpid():
                    _terminate_process(_matching_process(child.pid, child_create_time))
                    return
                _atomic_write(
                    launch.receipt_path,
                    current.model_copy(
                        update={
                            "child_pid": child.pid,
                            "child_create_time": child_create_time,
                        }
                    )
                    .model_dump_json(indent=2)
                    .encode("utf-8")
                    + b"\n",
                )
            child.wait()
            with _background_service_lock(launch):
                current = _read_background_receipt(launch)
                if current is None or current.supervisor_pid != os.getpid():
                    return
                _atomic_write(
                    launch.receipt_path,
                    current.model_copy(update={"child_pid": None, "child_create_time": None})
                    .model_dump_json(indent=2)
                    .encode("utf-8")
                    + b"\n",
                )
        except (OSError, psutil.Error):
            pass
        sleep(launch.restart_delay_seconds)


def stop_background_service(definition_path: Path) -> None:
    """Stop only the process identified by the service's private receipt."""

    launch = _read_background_launch(definition_path)
    with _background_service_lock(launch):
        receipt = _read_background_receipt(launch)
        if receipt is not None:
            _stop_receipt_process(launch, receipt)


def background_service_running(definition_path: Path) -> bool:
    """Return whether the exact launched process remains alive."""

    launch = _read_background_launch(definition_path)
    receipt = _read_background_receipt(launch)
    if receipt is None or receipt.launch_digest != canonical_hash(list(launch.launch_argv)):
        return False
    return _matching_process(receipt.supervisor_pid, receipt.supervisor_create_time) is not None


class WorkerServiceManager:
    """Install and inspect one generated per-user service definition."""

    def __init__(self, runner: CommandRunner = run_command):
        self._runner = runner

    def install(
        self, definition: WorkerServiceDefinition, config: WorkerRunConfig
    ) -> tuple[CommandResult, ...]:
        write_worker_config(definition.config_path, config)
        _atomic_write(definition.definition_path, definition.definition_payload)
        results = list(
            _run_checked(
                self._runner,
                definition.install_argvs,
                failure_code="campaign_worker_service_install_failed",
            )
        )
        if not definition.install_starts_service:
            results.extend(
                _run_checked(
                    self._runner,
                    definition.start_argvs,
                    failure_code="campaign_worker_service_install_failed",
                )
            )
        return tuple(results)

    def start(self, definition: WorkerServiceDefinition) -> tuple[CommandResult, ...]:
        return _run_checked(
            self._runner,
            definition.start_argvs,
            failure_code="campaign_worker_service_start_failed",
        )

    def stop(self, definition: WorkerServiceDefinition) -> tuple[CommandResult, ...]:
        return _run_checked(
            self._runner,
            definition.stop_argvs,
            failure_code="campaign_worker_service_stop_failed",
        )

    def replace(
        self, definition: WorkerServiceDefinition, config: WorkerRunConfig
    ) -> tuple[CommandResult, ...]:
        """Converge a changed service definition, unloading a live old job first."""

        results: list[CommandResult] = []
        status = self._runner(definition.status_argv)
        results.append(status)
        if status.returncode == 0:
            results.extend(
                _run_checked(
                    self._runner,
                    definition.uninstall_argvs,
                    failure_code="campaign_worker_service_uninstall_failed",
                )
            )
            results.extend(
                _run_checked(
                    self._runner,
                    definition.post_uninstall_argvs,
                    failure_code="campaign_worker_service_uninstall_failed",
                )
            )
        definition.definition_path.unlink(missing_ok=True)
        results.extend(self.install(definition, config))
        return tuple(results)

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
            "supervisor_state": _supervisor_state(definition.platform, result),
            "lifecycle": lifecycle.model_dump(mode="json") if lifecycle else None,
            "controller": controller.model_dump(mode="json"),
        }

    def uninstall(self, definition: WorkerServiceDefinition) -> tuple[CommandResult, ...]:
        results = list(
            _run_checked(
                self._runner,
                definition.uninstall_argvs,
                failure_code="campaign_worker_service_uninstall_failed",
            )
        )
        definition.definition_path.unlink(missing_ok=True)
        results.extend(
            _run_checked(
                self._runner,
                definition.post_uninstall_argvs,
                failure_code="campaign_worker_service_uninstall_failed",
            )
        )
        return tuple(results)


def probe_api_health(
    *,
    expected_state_root: Path,
    host: str = "127.0.0.1",
    port: int = 8003,
    timeout_seconds: float = 1.0,
    opener: Callable[..., Any] = open_api_url,
) -> dict[str, Any]:
    """Project a bounded, path-free readiness check for the resident API."""

    normalized_host = host.casefold().rstrip(".")
    try:
        is_loopback = normalized_host == "localhost" or ip_address(host).is_loopback
    except ValueError:
        is_loopback = normalized_host == "localhost"
    if (
        not is_loopback
        or not 1 <= port <= 65535
        or not 0 < timeout_seconds <= MAX_API_HEALTH_TIMEOUT_SECONDS
    ):
        raise WorkerServiceError("bashgym_api_health_probe_argument_invalid")

    url_host = f"[{host}]" if ":" in host else host
    request = urllib.request.Request(
        f"http://{url_host}:{port}/api/health",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise ValueError("api health status unavailable")
            raw_payload = response.read(MAX_API_HEALTH_RESPONSE_BYTES + 1)
        if len(raw_payload) > MAX_API_HEALTH_RESPONSE_BYTES:
            raise ValueError("api health response too large")
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict) or payload.get("status") != "healthy":
            raise ValueError("api health response invalid")
    except (OSError, TypeError, UnicodeError, ValueError):
        return {
            "schema_version": "bashgym_api_health.v1",
            "healthy": False,
            "state_root_match": False,
            "code": "api_http_unavailable",
        }

    state_root_match = payload.get("state_root_digest") == state_root_digest(expected_state_root)
    return {
        "schema_version": "bashgym_api_health.v1",
        "healthy": True,
        "state_root_match": state_root_match,
        "code": "api_http_healthy" if state_root_match else "api_state_root_mismatch",
    }


class ApiServiceManager:
    """Install and operate the generated per-user headless API service."""

    def __init__(self, runner: CommandRunner = run_command):
        self._runner = runner

    def install(self, definition: ApiServiceDefinition) -> tuple[CommandResult, ...]:
        _atomic_write(definition.definition_path, definition.definition_payload)
        results = list(
            _run_checked(
                self._runner,
                definition.install_argvs,
                failure_code="bashgym_api_service_install_failed",
            )
        )
        if not definition.install_starts_service:
            results.extend(
                _run_checked(
                    self._runner,
                    definition.start_argvs,
                    failure_code="bashgym_api_service_install_failed",
                )
            )
        return tuple(results)

    def start(self, definition: ApiServiceDefinition) -> tuple[CommandResult, ...]:
        return _run_checked(
            self._runner,
            definition.start_argvs,
            failure_code="bashgym_api_service_start_failed",
        )

    def stop(self, definition: ApiServiceDefinition) -> tuple[CommandResult, ...]:
        return _run_checked(
            self._runner,
            definition.stop_argvs,
            failure_code="bashgym_api_service_stop_failed",
        )

    def replace(self, definition: ApiServiceDefinition) -> tuple[CommandResult, ...]:
        """Converge a changed service definition, unloading a live old job first."""

        results: list[CommandResult] = []
        status = self._runner(definition.status_argv)
        results.append(status)
        if status.returncode == 0:
            results.extend(
                _run_checked(
                    self._runner,
                    definition.uninstall_argvs,
                    failure_code="bashgym_api_service_uninstall_failed",
                )
            )
            results.extend(
                _run_checked(
                    self._runner,
                    definition.post_uninstall_argvs,
                    failure_code="bashgym_api_service_uninstall_failed",
                )
            )
        definition.definition_path.unlink(missing_ok=True)
        results.extend(self.install(definition))
        return tuple(results)

    def status(self, definition: ApiServiceDefinition) -> dict[str, Any]:
        result = self._runner(definition.status_argv)
        return {
            "schema_version": "bashgym_api_service_status.v1",
            "installed": definition.definition_path.is_file(),
            "platform": definition.platform.value,
            "supervisor_returncode": result.returncode,
            "supervisor_state": _supervisor_state(definition.platform, result),
        }

    def uninstall(self, definition: ApiServiceDefinition) -> tuple[CommandResult, ...]:
        results = list(
            _run_checked(
                self._runner,
                definition.uninstall_argvs,
                failure_code="bashgym_api_service_uninstall_failed",
            )
        )
        definition.definition_path.unlink(missing_ok=True)
        results.extend(
            _run_checked(
                self._runner,
                definition.post_uninstall_argvs,
                failure_code="bashgym_api_service_uninstall_failed",
            )
        )
        return tuple(results)


def _controller_from_config(config: WorkerRunConfig) -> ControllerStatusProjection:
    if not config.database_path.is_file():
        return project_controller_status(None, config.data_directory)
    repository = CampaignRepository(config.database_path)
    repository.initialize()
    return project_controller_status(repository, config.data_directory)


def run_headless_api(
    *,
    host: str = "127.0.0.1",
    port: int = 8003,
    data_directory: Path | None = None,
    server_runner: Callable[..., Any] | None = None,
) -> None:
    """Run one headless API process without attaching desktop-owned services."""

    if not 1 <= port <= 65535:
        raise WorkerServiceError("bashgym_api_service_argument_invalid")
    _reject_controls(host)
    os.environ["BASHGYM_MODE"] = "headless"
    if data_directory is not None:
        os.environ["BASHGYM_DIR"] = str(data_directory.expanduser().resolve())
    if server_runner is None:
        import uvicorn

        server_runner = uvicorn.run
    server_runner(
        "bashgym.api.routes:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bashgym.campaigns.worker_service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "install", "status", "uninstall"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)
    for command in ("start-background", "status-background", "stop-background"):
        background_parser = subparsers.add_parser(command)
        background_parser.add_argument("--definition", type=Path, required=True)
    supervisor_parser = subparsers.add_parser("supervise-background")
    supervisor_parser.add_argument("--definition", type=Path, required=True)
    supervisor_parser.add_argument("--start-token", required=True)
    api_parser = subparsers.add_parser("run-api")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8003)
    api_parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "start-background":
            start_background_service(args.definition)
            return 0
        if args.command == "status-background":
            return 0 if background_service_running(args.definition) else 1
        if args.command == "stop-background":
            stop_background_service(args.definition)
            return 0
        if args.command == "supervise-background":
            supervise_background_service(
                args.definition,
                start_token=args.start_token,
            )
            return 0
        if args.command == "run-api":
            run_headless_api(
                host=args.host,
                port=args.port,
                data_directory=args.data_dir,
            )
            return 0
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
    "API_LAUNCHD_LABEL",
    "API_SYSTEMD_UNIT_NAME",
    "API_TASK_NAME",
    "ApiServiceDefinition",
    "ApiServiceManager",
    "CONTROLLER_OFFLINE_GUIDANCE",
    "CONTROLLER_STALE_GUIDANCE",
    "CommandResult",
    "ControllerStatusProjection",
    "DesktopWorkerStatusProjection",
    "DesktopWorkerSupervisor",
    "WorkerBootstrapResult",
    "WorkerLifecycleStatus",
    "WorkerPlatform",
    "WorkerRunConfig",
    "WorkerServiceDefinition",
    "WorkerServiceError",
    "WorkerServiceManager",
    "build_service_definition",
    "build_api_service_definition",
    "build_worker",
    "background_service_running",
    "ensure_worker_bootstrap",
    "load_approved_remote_profiles",
    "load_approved_source_profiles",
    "validate_code_lineage_execution_bindings",
    "main",
    "probe_api_health",
    "project_controller_status",
    "project_controller_status_from_database",
    "read_worker_config",
    "run_foreground",
    "run_headless_api",
    "start_background_service",
    "stop_background_service",
    "supervise_background_service",
    "write_worker_config",
]
