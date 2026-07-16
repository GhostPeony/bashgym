"""Typed SSH lifecycle for restart-safe private campaign training."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError, field_validator, model_validator

from bashgym.campaigns.contracts import (
    CodeLineageRecord,
    CodeLineageState,
    ContractModel,
    FrozenContractModel,
    GitObjectId,
    HexDigest,
    Identifier,
    StageKind,
    canonical_hash,
    utc_now,
)
from bashgym.campaigns.nemo_rl import ApprovedNemoRLProfile, NemoRLRuntimeReceipt
from bashgym.gym.remote_trainer import HAS_ASYNCSSH, SSHConfig


class RemoteRunState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RemoteControl(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    TERMINATE = "terminate"
    FORCE_STOP = "force_stop"


CONTROL_SIGNALS = {
    RemoteControl.PAUSE: "STOP",
    RemoteControl.RESUME: "CONT",
    RemoteControl.TERMINATE: "TERM",
    RemoteControl.FORCE_STOP: "KILL",
}


def _safe_python_entrypoint(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.as_posix() != value
        or path.suffix.casefold() != ".py"
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("code lineage entrypoint must be a safe repository-relative Python file")
    return value


class RemoteCommandResult(FrozenContractModel):
    stdout: str = ""
    stderr: str = ""
    exit_status: int


class RemoteRunIdentity(FrozenContractModel):
    """Server-neutral process identity persisted immediately after launch."""

    schema_version: str = "campaign_remote_run_identity.v2"
    compute_profile_id: str
    run_id: str
    remote_run_directory: str
    remote_pid: int = Field(ge=1)
    process_group_id: int = Field(ge=1)
    process_start_ticks: int = Field(ge=1)
    boot_id: str = Field(min_length=1)
    command_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    launch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    launched_at: datetime


class RemoteSupervisorState(FrozenContractModel):
    """Atomic on-host identity record used for launch discovery."""

    schema_version: str = "campaign_remote_supervisor_state.v1"
    compute_profile_id: str
    run_id: str
    remote_run_directory: str
    remote_pid: int = Field(ge=1)
    process_group_id: int = Field(ge=1)
    process_start_ticks: int = Field(ge=1)
    boot_id: str = Field(min_length=1)
    command_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    launch_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    launched_at: datetime

    def identity(self) -> RemoteRunIdentity:
        return RemoteRunIdentity(**self.model_dump(exclude={"schema_version"}))


class ApprovedCodeLineageExecutionBinding(FrozenContractModel):
    """Installation-owned mapping from captured source to one training entrypoint."""

    schema_version: Literal["campaign_code_lineage_execution_binding.v1"] = (
        "campaign_code_lineage_execution_binding.v1"
    )
    binding_id: Identifier
    binding_revision: int = Field(ge=1)
    binding_digest: HexDigest = ""
    source_repository_profile_id: Identifier
    entrypoint_path: str
    working_directory: Literal["run", "source"] = "run"
    max_archive_bytes: int = Field(default=512 * 1024 * 1024, ge=1024, le=4 * 1024**3)

    @field_validator("entrypoint_path")
    @classmethod
    def safe_entrypoint(cls, value: str) -> str:
        return _safe_python_entrypoint(value)

    @model_validator(mode="after")
    def verify_binding_digest(self) -> ApprovedCodeLineageExecutionBinding:
        expected = canonical_hash(self.model_dump(mode="json", exclude={"binding_digest"}))
        if self.binding_digest and self.binding_digest != expected:
            raise ValueError("code lineage execution binding digest mismatch")
        if not self.binding_digest:
            object.__setattr__(self, "binding_digest", expected)
        return self


class CodeLineageLaunchSnapshot(FrozenContractModel):
    """Transient verified archive consumed by one private-compute launch."""

    schema_version: Literal["campaign_code_lineage_launch_snapshot.v1"] = (
        "campaign_code_lineage_launch_snapshot.v1"
    )
    binding_id: Identifier
    binding_revision: int = Field(ge=1)
    binding_digest: HexDigest
    source_repository_profile_id: Identifier
    lineage_id: Identifier
    record_digest: HexDigest
    commit_sha: GitObjectId
    patch_sha256: HexDigest
    entrypoint_path: str
    working_directory: Literal["run", "source"]
    archive_path: Path
    archive_sha256: HexDigest
    archive_size_bytes: int = Field(ge=1)

    @field_validator("entrypoint_path")
    @classmethod
    def safe_entrypoint(cls, value: str) -> str:
        return _safe_python_entrypoint(value)

    @field_validator("archive_path")
    @classmethod
    def verified_archive_path(cls, value: Path) -> Path:
        candidate = value.expanduser()
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.suffix.casefold() != ".tar"
        ):
            raise ValueError("code lineage snapshot must be an absolute regular tar file")
        return candidate.resolve()

    @model_validator(mode="after")
    def verify_archive_material(self) -> CodeLineageLaunchSnapshot:
        if self.archive_path.stat().st_size != self.archive_size_bytes:
            raise ValueError("code lineage snapshot size mismatch")
        if _sha256_file(self.archive_path) != self.archive_sha256:
            raise ValueError("code lineage snapshot digest mismatch")
        return self


class RemoteLaunchRequest(ContractModel):
    """Typed launch inputs; an approved recipe builds these arguments."""

    schema_version: str = "campaign_remote_launch_request.v3"
    compute_profile_id: str
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    script_path: Path
    input_files: tuple[Path, ...]
    script_args: tuple[str, ...]
    python_executable: str = Field(default="python3", min_length=1, max_length=512)
    recipe_digest: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    output_paths: tuple[str, ...] = (
        "final",
        "training_manifest.json",
        "training_metrics.jsonl",
    )
    source_snapshot: CodeLineageLaunchSnapshot | None = None

    @field_validator("script_path")
    @classmethod
    def validate_script(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError("training script must be an existing local file")
        if value.suffix != ".py":
            raise ValueError("training script must be a Python file")
        return value.resolve()

    @field_validator("input_files")
    @classmethod
    def validate_inputs(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if not value or any(not path.is_file() for path in value):
            raise ValueError("every remote training input must be an existing file")
        resolved = tuple(path.resolve() for path in value)
        names = [path.name for path in resolved]
        if len(set(names)) != len(names):
            raise ValueError("remote training input basenames must be unique")
        return resolved

    @field_validator("script_args")
    @classmethod
    def reject_secret_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        lowered = tuple(item.casefold() for item in value)
        forbidden = ("--token", "--api-key", "--password", "--secret")
        if any(item.startswith(forbidden) for item in lowered):
            raise ValueError("remote credentials must use configured references, not arguments")
        return value

    @field_validator("python_executable")
    @classmethod
    def exact_python_executable(cls, value: str) -> str:
        if any(character.isspace() or character in "\x00;&|`$<>" for character in value):
            raise ValueError("remote Python executable must be one exact executable path")
        path = PurePosixPath(value)
        if "/" in value and (not path.is_absolute() or ".." in path.parts):
            raise ValueError("remote Python executable path must be absolute")
        return value

    @field_validator("output_paths")
    @classmethod
    def validate_output_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("output_paths must be non-empty, sorted, and unique")
        for item in value:
            path = PurePosixPath(item)
            if path.is_absolute() or ".." in path.parts or not path.parts or item in {".", ""}:
                raise ValueError("output paths must stay inside the remote run directory")
        return value

    @model_validator(mode="after")
    def reject_script_input_collision(self) -> RemoteLaunchRequest:
        input_names = {path.name for path in self.input_files}
        if self.source_snapshot is None and self.script_path.name in input_names:
            raise ValueError("training script and input files must have distinct basenames")
        if (
            self.source_snapshot is not None
            and self.source_snapshot.archive_path.name in input_names
        ):
            raise ValueError("code snapshot and input files must have distinct basenames")
        return self


class RemoteObservation(FrozenContractModel):
    schema_version: str = "campaign_remote_observation.v2"
    identity: RemoteRunIdentity
    state: RemoteRunState
    observed_at: datetime
    exit_code: int | None = None
    safe_reason: str


class RemoteCapacityPolicy(FrozenContractModel):
    schema_version: str = "campaign_remote_capacity_policy.v1"
    minimum_available_memory_gib: float = Field(default=48.0, ge=0)
    minimum_available_disk_gib: float = Field(default=50.0, ge=0)
    maximum_external_gpu_processes: int = Field(default=0, ge=0)


class RemoteCapacitySnapshot(FrozenContractModel):
    schema_version: str = "campaign_remote_capacity_snapshot.v1"
    compute_profile_id: str
    available_memory_gib: float = Field(ge=0)
    available_disk_gib: float = Field(ge=0)
    external_gpu_processes: tuple[str, ...]
    admitted: bool
    blocking_reasons: tuple[str, ...]
    observed_at: datetime


class RemoteStreamCursor(FrozenContractModel):
    schema_version: str = "campaign_remote_stream_cursor.v1"
    byte_offset: int = Field(default=0, ge=0)
    partial_line: str = ""


class RemoteStreamChunk(FrozenContractModel):
    schema_version: str = "campaign_remote_stream_chunk.v1"
    source: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    complete_lines: tuple[str, ...]
    next_cursor: RemoteStreamCursor


class RemoteSession(Protocol):
    async def run(self, command: str, *, timeout: float | None = None) -> RemoteCommandResult: ...

    async def upload(self, local_path: Path, remote_path: str) -> None: ...

    async def download(self, remote_path: str, local_path: Path) -> bool: ...


SessionFactory = Callable[[], Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class PinnedRemoteStageProfile(FrozenContractModel):
    """Exact server-owned launch material for one approved private-compute stage."""

    schema_version: Literal["campaign_pinned_remote_stage_profile.v1"] = (
        "campaign_pinned_remote_stage_profile.v1"
    )
    stage: StageKind
    script_path: Path
    script_sha256: HexDigest
    input_files: tuple[Path, ...]
    input_sha256: dict[str, HexDigest]
    script_args: tuple[str, ...] = ()
    output_paths: tuple[str, ...] = (
        "final",
        "training_manifest.json",
        "training_metrics.jsonl",
    )
    capacity_policy: RemoteCapacityPolicy = Field(default_factory=RemoteCapacityPolicy)
    budget_unit: Identifier = "gpu_hours"
    budget_reservation: float = Field(gt=0)
    python_executable: str = Field(default="python3", min_length=1, max_length=512)
    code_lineage_binding: ApprovedCodeLineageExecutionBinding | None = None

    @field_validator("stage")
    @classmethod
    def approved_compute_stage_only(cls, value: StageKind) -> StageKind:
        if value not in {
            StageKind.SMOKE_TRAINING,
            StageKind.FULL_TRAINING,
            StageKind.DEVELOPMENT_EVALUATION,
        }:
            raise ValueError("remote executor profiles are restricted to approved compute stages")
        return value

    @field_validator("script_path")
    @classmethod
    def pinned_script(cls, value: Path) -> Path:
        candidate = value.expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("approved compute script must be a regular non-symlink file")
        if candidate.suffix.casefold() != ".py":
            raise ValueError("approved compute script must be a Python file")
        return candidate.resolve()

    @field_validator("input_files")
    @classmethod
    def pinned_inputs(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if not value:
            raise ValueError("approved remote profile requires at least one input file")
        resolved: list[Path] = []
        for raw_path in value:
            candidate = raw_path.expanduser()
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("approved input files must be regular non-symlink files")
            resolved.append(candidate.resolve())
        names = [path.name for path in resolved]
        if len(set(names)) != len(names):
            raise ValueError("approved remote input basenames must be unique")
        return tuple(resolved)

    @field_validator("script_args")
    @classmethod
    def exact_secret_free_args(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 256 or any(len(argument) > 4096 for argument in value):
            raise ValueError("approved remote arguments exceed bounded limits")
        if any("\x00" in argument or "\n" in argument or "\r" in argument for argument in value):
            raise ValueError("approved remote arguments cannot contain control characters")
        forbidden = ("--token", "--api-key", "--password", "--secret")
        if any(argument.casefold().startswith(forbidden) for argument in value):
            raise ValueError("remote credentials must use configured references, not arguments")
        return value

    @field_validator("output_paths")
    @classmethod
    def confined_outputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("approved output paths must be non-empty, sorted, and unique")
        for item in value:
            path = PurePosixPath(item)
            if path.is_absolute() or ".." in path.parts or not path.parts or item in {".", ""}:
                raise ValueError("approved output paths must stay inside the remote run directory")
        return value

    @field_validator("python_executable")
    @classmethod
    def pinned_python(cls, value: str) -> str:
        if any(character.isspace() or character in "\x00;&|`$<>" for character in value):
            raise ValueError("approved Python executable must be one exact executable path")
        path = PurePosixPath(value)
        if "/" in value and (not path.is_absolute() or ".." in path.parts):
            raise ValueError("approved Python executable path must be absolute")
        return value

    @model_validator(mode="after")
    def verify_contract_and_materials(self) -> PinnedRemoteStageProfile:
        if self.script_path.name in {path.name for path in self.input_files}:
            raise ValueError("approved script and input files must have distinct basenames")
        expected_inputs = {path.name for path in self.input_files}
        if set(self.input_sha256) != expected_inputs:
            raise ValueError("approved input hashes must exactly match input basenames")
        self.verify_materials()
        return self

    def verify_materials(self) -> None:
        """Fail closed if approved local launch material changes after configuration."""

        if self.script_path.is_symlink() or not self.script_path.is_file():
            raise ValueError("approved training script is missing or not a regular file")
        if _sha256_file(self.script_path) != self.script_sha256:
            raise ValueError("approved training script hash mismatch")
        for path in self.input_files:
            if path.is_symlink() or not path.is_file():
                raise ValueError("approved input file is missing or not a regular file")
            if _sha256_file(path) != self.input_sha256[path.name]:
                raise ValueError("approved input file hash mismatch")


class ApprovedRemoteExecutorProfile(FrozenContractModel):
    """Protected worker profile that owns SSH authority and pinned launch material."""

    schema_version: Literal["campaign_approved_remote_executor_profile.v2"] = (
        "campaign_approved_remote_executor_profile.v2"
    )
    profile_id: Identifier
    profile_revision: int = Field(ge=1)
    profile_digest: HexDigest = ""
    compute_profile_id: Identifier
    target_contract_key: Identifier
    target_model_digest: HexDigest
    host: str = Field(min_length=1, max_length=512)
    username: str = Field(min_length=1, max_length=256)
    port: int = Field(default=22, ge=1, le=65535)
    key_path: str = Field(min_length=1, max_length=4096)
    remote_work_dir: str = Field(default="~/bashgym-training", min_length=1, max_length=4096)
    stages: tuple[PinnedRemoteStageProfile, ...]
    nemo_rl: ApprovedNemoRLProfile | None = None

    @field_validator("host", "username")
    @classmethod
    def exact_ssh_identity(cls, value: str) -> str:
        if any(character.isspace() or character in "\x00/@;&|`$<>" for character in value):
            raise ValueError("approved SSH identity fields must be exact non-shell values")
        return value

    @field_validator("key_path")
    @classmethod
    def protected_key_path(cls, value: str) -> str:
        candidate = Path(value).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("approved SSH key path must be a regular non-symlink file")
        return str(candidate.resolve())

    @field_validator("remote_work_dir")
    @classmethod
    def confined_remote_root(cls, value: str) -> str:
        if any(character in "\x00\n\r" for character in value):
            raise ValueError("approved remote work directory contains control characters")
        normalized = value[1:] if value == "~" or value.startswith("~/") else value
        path = PurePosixPath(normalized or "/")
        if not (value == "~" or value.startswith("~/") or path.is_absolute()):
            raise ValueError("approved remote work directory must be absolute or home-relative")
        if ".." in path.parts:
            raise ValueError("approved remote work directory cannot traverse parents")
        return value.rstrip("/") or "/"

    @field_validator("stages")
    @classmethod
    def canonical_stages(
        cls, value: tuple[PinnedRemoteStageProfile, ...]
    ) -> tuple[PinnedRemoteStageProfile, ...]:
        if not value:
            raise ValueError("approved remote executor profile requires a compute stage")
        keys = tuple(stage.stage.value for stage in value)
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("approved remote stages must be sorted and unique")
        return value

    @model_validator(mode="after")
    def verify_profile_digest(self) -> ApprovedRemoteExecutorProfile:
        if self.nemo_rl is not None:
            required_stages = {
                StageKind.SMOKE_TRAINING,
                StageKind.FULL_TRAINING,
            }
            configured_stages = {stage.stage for stage in self.stages}
            if (
                self.nemo_rl.compute_profile_id != self.compute_profile_id
                or self.nemo_rl.target_contract_key != self.target_contract_key
                or self.nemo_rl.target_model_digest != self.target_model_digest
                or not required_stages.issubset(configured_stages)
            ):
                raise ValueError("NeMo RL profile does not match its remote executor")
        excluded = {"profile_digest"}
        if self.nemo_rl is None:
            # Preserve the v2 digest of profiles written before the optional
            # NeMo RL extension existed.
            excluded.add("nemo_rl")
        payload = self.model_dump(mode="json", exclude=excluded)
        expected = canonical_hash(payload)
        if self.profile_digest and self.profile_digest != expected:
            raise ValueError("approved remote executor profile digest mismatch")
        if not self.profile_digest:
            object.__setattr__(self, "profile_digest", expected)
        self.verify_materials()
        return self

    def verify_materials(self) -> None:
        for stage in self.stages:
            stage.verify_materials()

    def stage_profile(self, stage: StageKind) -> PinnedRemoteStageProfile:
        for configured in self.stages:
            if configured.stage == stage:
                return configured
        raise KeyError(stage.value)


def remote_executor_config(
    profile: ApprovedRemoteExecutorProfile,
    stage: StageKind,
    *,
    recipe_digest: HexDigest,
    code_lineage: CodeLineageRecord | None = None,
) -> dict[str, Any]:
    """Project one protected profile stage into the persisted executor contract."""

    profile.verify_materials()
    configured = profile.stage_profile(stage)
    result: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "profile_revision": profile.profile_revision,
        "profile_digest": profile.profile_digest,
        "compute_profile_id": profile.compute_profile_id,
        "target_contract_key": profile.target_contract_key,
        "target_model_digest": profile.target_model_digest,
        "stage": stage.value,
        "script_path": str(configured.script_path),
        "expected_script_sha256": configured.script_sha256,
        "input_files": [str(path) for path in configured.input_files],
        "expected_input_sha256": dict(sorted(configured.input_sha256.items())),
        "script_args": list(configured.script_args),
        "python_executable": configured.python_executable,
        "output_paths": list(configured.output_paths),
        "capacity_policy": configured.capacity_policy.model_dump(mode="json"),
        "budget_unit": configured.budget_unit,
        "budget_reservation": configured.budget_reservation,
        "recipe_digest": recipe_digest,
    }
    if profile.nemo_rl is not None and stage in {
        StageKind.SMOKE_TRAINING,
        StageKind.FULL_TRAINING,
    }:
        result["nemo_rl"] = {
            "profile_id": profile.nemo_rl.profile_id,
            "profile_revision": profile.nemo_rl.profile_revision,
            "profile_digest": profile.nemo_rl.profile_digest,
            "release": profile.nemo_rl.release,
            "source_revision": profile.nemo_rl.source_revision,
            "image_digest": profile.nemo_rl.image_digest,
            "model_support_level": profile.nemo_rl.model_support_level.value,
            "recipe_sha256": profile.nemo_rl.recipe_sha256,
            "dataset_sha256": profile.nemo_rl.dataset_sha256,
            "verifier_digest": profile.nemo_rl.verifier_digest,
        }
    if code_lineage is not None:
        if code_lineage.state != CodeLineageState.CAPTURED:
            raise ValueError("code lineage must be captured before remote execution")
        binding = configured.code_lineage_binding
        if binding is None:
            raise ValueError("remote stage has no code lineage execution binding")
        if binding.source_repository_profile_id != code_lineage.source_repository_profile_id:
            raise ValueError("code lineage execution binding source profile mismatch")
        assert code_lineage.commit_sha is not None and code_lineage.patch_sha256 is not None
        result["code_lineage_execution"] = {
            "binding_id": binding.binding_id,
            "binding_revision": binding.binding_revision,
            "binding_digest": binding.binding_digest,
            "source_repository_profile_id": binding.source_repository_profile_id,
            "entrypoint_path": binding.entrypoint_path,
            "working_directory": binding.working_directory,
            "max_archive_bytes": binding.max_archive_bytes,
            "lineage_id": code_lineage.lineage_id,
            "record_digest": code_lineage.record_digest,
            "commit_sha": code_lineage.commit_sha,
            "patch_sha256": code_lineage.patch_sha256,
        }
    return result


class AsyncSSHSession:
    """Small asyncssh projection kept behind a mockable campaign protocol."""

    def __init__(self, config: SSHConfig):
        self.config = config
        self.connection = None
        self.sftp = None

    async def __aenter__(self) -> AsyncSSHSession:
        if not HAS_ASYNCSSH:
            raise RuntimeError("asyncssh is required for remote campaign training")
        from bashgym.gym import remote_trainer

        key_path = Path(self.config.key_path).expanduser()
        self.connection = await remote_trainer.asyncssh.connect(
            self.config.host,
            port=self.config.port,
            username=self.config.username,
            client_keys=[str(key_path)],
            known_hosts=None,
            connect_timeout=10,
        )
        self.sftp = await self.connection.start_sftp_client()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.connection is not None:
            self.connection.close()
            await self.connection.wait_closed()

    async def run(self, command: str, *, timeout: float | None = None) -> RemoteCommandResult:
        result = await self.connection.run(command, check=False, timeout=timeout)
        return RemoteCommandResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            exit_status=result.exit_status,
        )

    async def upload(self, local_path: Path, remote_path: str) -> None:
        await self.sftp.put(str(local_path), remote_path)

    async def download(self, remote_path: str, local_path: Path) -> bool:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self.sftp.get(remote_path, str(local_path), recurse=True)
        except Exception:
            return False
        return True


class RemoteTrainingAdapter:
    """Launch and reconcile a remote run without ever using a local PID."""

    def __init__(
        self,
        config: SSHConfig,
        *,
        compute_profile_id: str,
        session_factory: SessionFactory | None = None,
    ):
        self.config = config
        self.compute_profile_id = compute_profile_id
        self._session_factory = session_factory or (lambda: AsyncSSHSession(config))

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[RemoteSession]:
        async with self._session_factory() as session:
            yield session

    async def _resolve_remote_root(self, session: RemoteSession) -> str:
        root = self.config.remote_work_dir
        if root == "~" or root.startswith("~/"):
            result = await session.run('printf %s "$HOME"', timeout=5)
            if result.exit_status != 0 or not result.stdout.strip().startswith("/"):
                raise RuntimeError("campaign_remote_home_unavailable")
            root = result.stdout.strip() + root[1:]
        if not root.startswith("/"):
            raise RuntimeError("campaign_remote_root_must_be_absolute")
        return root.rstrip("/")

    async def nemo_rl_preflight(
        self,
        profile: ApprovedNemoRLProfile,
        *,
        pull_image: bool = False,
    ) -> NemoRLRuntimeReceipt:
        """Probe an approved optional NeMo RL runtime without exposing host identity."""

        if profile.compute_profile_id != self.compute_profile_id:
            raise ValueError("NeMo RL profile does not match remote adapter")

        async with self._session() as session:
            if pull_image:
                pulled = await session.run(
                    f"docker pull {shlex.quote(profile.image_reference)}",
                    timeout=3600,
                )
                if pulled.exit_status != 0:
                    raise RuntimeError("nemo_rl_image_pull_failed")

            async def probe(command: str, code: str, *, timeout: float = 30) -> str:
                result = await session.run(command, timeout=timeout)
                if result.exit_status != 0:
                    raise RuntimeError(code)
                return result.stdout.strip()

            docker_version = await probe(
                "docker version --format '{{.Server.Version}}'",
                "nemo_rl_docker_unavailable",
            )
            runtimes = await probe(
                "docker info --format '{{json .Runtimes}}'",
                "nemo_rl_docker_info_unavailable",
            )
            architecture = (
                await probe("uname -m", "nemo_rl_platform_unavailable")
            ).casefold()
            platform = {
                "amd64": "linux/amd64",
                "x86_64": "linux/amd64",
                "aarch64": "linux/arm64",
                "arm64": "linux/arm64",
            }.get(architecture)
            if platform is None:
                raise RuntimeError("nemo_rl_platform_unsupported")

            gpu_lines = await probe(
                "nvidia-smi -L",
                "nemo_rl_gpu_unavailable",
            )
            disk_kib = float(
                await probe(
                    "df -Pk . | awk 'NR==2 {print $4}'",
                    "nemo_rl_disk_probe_failed",
                )
            )
            shared_memory_kib = float(
                await probe(
                    "df -Pk /dev/shm | awk 'NR==2 {print $4}'",
                    "nemo_rl_shared_memory_probe_failed",
                )
            )
            repo_digests = await probe(
                "docker image inspect "
                f"{shlex.quote(profile.image_reference)} "
                "--format '{{join .RepoDigests \"\\n\"}}'",
                "nemo_rl_image_not_ready",
            )
            source_revision = await probe(
                "docker run --rm --network=none --entrypoint git "
                f"{shlex.quote(profile.image_reference)} "
                "-C /opt/nemo-rl rev-parse HEAD",
                "nemo_rl_source_probe_failed",
                timeout=60,
            )
            recipe_sha256 = (
                await probe(
                    "docker run --rm --network=none "
                    "--entrypoint sha256sum "
                    f"{shlex.quote(profile.image_reference)} "
                    f"{shlex.quote(profile.recipe_path)}",
                    "nemo_rl_recipe_probe_failed",
                    timeout=60,
                )
            ).split(maxsplit=1)[0]

            remote_model_path = profile.remote_model_path
            if remote_model_path == "~" or remote_model_path.startswith("~/"):
                home = await probe('printf %s "$HOME"', "nemo_rl_remote_home_unavailable")
                remote_model_path = home + remote_model_path[1:]
            quoted_model = shlex.quote(remote_model_path)
            quoted_revision = shlex.quote(profile.model_revision)
            model_check = await session.run(
                f"test -f {quoted_model}/config.json && "
                f"(test \"$(basename {quoted_model})\" = {quoted_revision} || "
                f"test \"$(cat {quoted_model}/.bashgym-model-revision 2>/dev/null)\" = {quoted_revision})",
                timeout=10,
            )

        return NemoRLRuntimeReceipt(
            compute_profile_id=self.compute_profile_id,
            platform=platform,
            docker_version=docker_version,
            docker_ready=True,
            nvidia_runtime_ready="nvidia" in runtimes.casefold(),
            gpu_count=sum(1 for line in gpu_lines.splitlines() if line.strip().startswith("GPU ")),
            available_disk_gib=disk_kib / 1024 / 1024,
            shared_memory_gib=shared_memory_kib / 1024 / 1024,
            image_digest=profile.image_digest,
            image_ready=profile.image_reference in repo_digests,
            source_revision=profile.source_revision,
            source_ready=source_revision == profile.source_revision,
            recipe_sha256=profile.recipe_sha256,
            recipe_ready=recipe_sha256 == profile.recipe_sha256,
            model_revision=profile.model_revision,
            model_ready=model_check.exit_status == 0,
        )

    @staticmethod
    def _argv(request: RemoteLaunchRequest, remote_directory: str) -> tuple[str, ...]:
        if request.source_snapshot is not None:
            entrypoint = f"{remote_directory}/source/{request.source_snapshot.entrypoint_path}"
        else:
            entrypoint = f"{remote_directory}/{request.script_path.name}"
        return (
            request.python_executable,
            entrypoint,
            *request.script_args,
        )

    @staticmethod
    def _launch_files(request: RemoteLaunchRequest) -> tuple[Path, ...]:
        entrypoint_material = (
            (request.source_snapshot.archive_path,)
            if request.source_snapshot is not None
            else (request.script_path,)
        )
        return (*entrypoint_material, *request.input_files)

    @staticmethod
    def _execution_context(
        request: RemoteLaunchRequest, remote_directory: str
    ) -> dict[str, str | None]:
        if request.source_snapshot is None:
            return {
                "entrypoint_kind": "pinned_script",
                "working_directory": remote_directory,
                "python_path": None,
            }
        working_directory = (
            f"{remote_directory}/source"
            if request.source_snapshot.working_directory == "source"
            else remote_directory
        )
        return {
            "entrypoint_kind": "captured_source_snapshot",
            "working_directory": working_directory,
            "python_path": f"{remote_directory}/source",
        }

    @staticmethod
    def _launch_manifest(request: RemoteLaunchRequest, remote_directory: str) -> dict[str, Any]:
        files = RemoteTrainingAdapter._launch_files(request)
        if request.source_snapshot is not None and (
            request.source_snapshot.archive_path.stat().st_size
            != request.source_snapshot.archive_size_bytes
            or _sha256_file(request.source_snapshot.archive_path)
            != request.source_snapshot.archive_sha256
        ):
            raise ValueError("code lineage snapshot changed before launch")
        argv = RemoteTrainingAdapter._argv(request, remote_directory)
        execution_context = RemoteTrainingAdapter._execution_context(request, remote_directory)
        command_contract = {"argv": list(argv), **execution_context}
        manifest: dict[str, Any] = {
            "schema_version": "campaign_remote_launch_manifest.v2",
            "compute_profile_id": request.compute_profile_id,
            "run_id": request.run_id,
            "recipe_digest": request.recipe_digest,
            "argv": list(argv),
            "execution_context": execution_context,
            "command_hash": canonical_hash(command_contract),
            "files": [
                {"name": path.name, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
                for path in files
            ],
            "output_paths": list(request.output_paths),
        }
        if request.source_snapshot is not None:
            manifest["code_lineage"] = request.source_snapshot.model_dump(
                mode="json", exclude={"archive_path"}
            )
        return manifest

    async def launch(self, request: RemoteLaunchRequest) -> RemoteRunIdentity:
        if request.compute_profile_id != self.compute_profile_id:
            raise ValueError("campaign compute profile does not match remote adapter")
        async with self._session() as session:
            root = await self._resolve_remote_root(session)
            remote_directory = f"{root}/{request.run_id}"
            quoted_root = shlex.quote(root)
            quoted_directory = shlex.quote(remote_directory)
            created = await session.run(
                f"umask 077 && mkdir -p {quoted_root} && mkdir {quoted_directory}", timeout=10
            )
            if created.exit_status != 0:
                raise RuntimeError("campaign_remote_run_already_exists")
            files = self._launch_files(request)
            manifest = self._launch_manifest(request, remote_directory)
            manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            manifest_sha256 = hashlib.sha256(manifest_json.encode()).hexdigest()
            for path in files:
                await session.upload(path, f"{remote_directory}/{path.name}")
            checks = " && ".join(
                f"printf '%s  %s\\n' {item['sha256']} {shlex.quote(item['name'])} | sha256sum -c -"
                for item in manifest["files"]
            )
            source_preparation = ""
            if request.source_snapshot is not None:
                archive_name = shlex.quote(request.source_snapshot.archive_path.name)
                entrypoint = shlex.quote(f"source/{request.source_snapshot.entrypoint_path}")
                source_preparation = (
                    " && test ! -e source"
                    f" && tar --extract --file {archive_name} --no-same-owner"
                    f" && test -f {entrypoint} && test ! -L {entrypoint}"
                )
            prepared = await session.run(
                f"cd {quoted_directory} && {checks}{source_preparation} && "
                f"printf %s {shlex.quote(manifest_json)} > launch_manifest.json && "
                f"printf '%s  launch_manifest.json\\n' {manifest_sha256} | sha256sum -c -",
                timeout=30,
            )
            if prepared.exit_status != 0:
                raise RuntimeError("campaign_remote_upload_verification_failed")

            argv = self._argv(request, remote_directory)
            command_hash = manifest["command_hash"]
            command = " ".join(shlex.quote(item) for item in argv)
            execution_context = self._execution_context(request, remote_directory)
            working_directory = shlex.quote(str(execution_context["working_directory"]))
            python_path = execution_context["python_path"]
            python_environment = (
                f"PYTHONPATH={shlex.quote(str(python_path))} " if python_path is not None else ""
            )
            inner = (
                f"cd {working_directory} || exit 125; "
                f"source {shlex.quote(root + '/venv/bin/activate')} 2>/dev/null || true; "
                f"{python_environment}PYTHONUNBUFFERED=1 {command}; code=$?; "
                "printf '%s\\n' \"$code\" > exit_code.tmp && mv exit_code.tmp exit_code; "
                'exit "$code"'
            )
            state_writer = (
                "import json,sys;"
                "print(json.dumps({"
                "'schema_version':'campaign_remote_supervisor_state.v1',"
                "'compute_profile_id':sys.argv[1],"
                "'run_id':sys.argv[2],"
                "'remote_run_directory':sys.argv[3],"
                "'remote_pid':int(sys.argv[4]),"
                "'process_group_id':int(sys.argv[5]),"
                "'process_start_ticks':int(sys.argv[6]),"
                "'boot_id':sys.argv[7],"
                "'command_hash':sys.argv[8],"
                "'launch_manifest_sha256':sys.argv[9],"
                "'launched_at':sys.argv[10]},"
                "sort_keys=True,separators=(',',':')))"
            )
            state_command = " ".join(
                (
                    "python3",
                    "-c",
                    shlex.quote(state_writer),
                    shlex.quote(request.compute_profile_id),
                    shlex.quote(request.run_id),
                    shlex.quote(remote_directory),
                    '"$pid"',
                    '"$pgid"',
                    '"$start"',
                    '"$boot"',
                    shlex.quote(command_hash),
                    shlex.quote(manifest_sha256),
                    '"$launched"',
                )
            )
            launch_command = (
                f"cd {quoted_directory} || exit 1; "
                "boot=$(cat /proc/sys/kernel/random/boot_id) || exit 2; "
                f"nohup setsid bash -lc {shlex.quote(inner)} > training.log 2>&1 < /dev/null & "
                "pid=$!; sleep 0.05; "
                "start=$(awk '{print $22}' /proc/$pid/stat 2>/dev/null); "
                "pgid=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' '); "
                "launched=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
                'test -n "$start" -a -n "$pgid" || exit 3; '
                f"{state_command} > remote_run_state.v1.json.tmp && "
                "mv remote_run_state.v1.json.tmp remote_run_state.v1.json && "
                "cat remote_run_state.v1.json"
            )
            launched = await session.run(launch_command, timeout=15)
            if launched.exit_status != 0:
                raise RuntimeError("campaign_remote_launch_failed")
        try:
            state = RemoteSupervisorState.model_validate_json(launched.stdout)
        except ValidationError as exc:
            raise RuntimeError("campaign_remote_identity_unavailable") from exc
        identity = state.identity()
        self._validate_adapter_identity(identity)
        if identity.launch_manifest_sha256 != manifest_sha256:
            raise RuntimeError("campaign_remote_launch_manifest_mismatch")
        return identity

    async def discover(self, request: RemoteLaunchRequest) -> RemoteRunIdentity | None:
        """Recover identity from the deterministic run directory before any launch."""

        if request.compute_profile_id != self.compute_profile_id:
            raise ValueError("campaign compute profile does not match remote adapter")
        async with self._session() as session:
            root = await self._resolve_remote_root(session)
            remote_directory = f"{root}/{request.run_id}"
            result = await session.run(
                f"cat {shlex.quote(remote_directory + '/remote_run_state.v1.json')}", timeout=10
            )
        if result.exit_status == 1:
            return None
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_identity_unavailable")
        try:
            identity = RemoteSupervisorState.model_validate_json(result.stdout).identity()
        except ValidationError as exc:
            raise RuntimeError("campaign_remote_identity_unavailable") from exc
        expected_manifest = self._launch_manifest(request, remote_directory)
        expected_manifest_json = json.dumps(
            expected_manifest, sort_keys=True, separators=(",", ":")
        )
        expected_manifest_sha = hashlib.sha256(expected_manifest_json.encode()).hexdigest()
        if (
            identity.remote_run_directory != remote_directory
            or identity.command_hash != expected_manifest["command_hash"]
            or identity.launch_manifest_sha256 != expected_manifest_sha
        ):
            raise RuntimeError("campaign_remote_command_identity_mismatch")
        self._validate_adapter_identity(identity)
        return identity

    async def observe(self, identity: RemoteRunIdentity) -> RemoteObservation:
        self._validate_adapter_identity(identity)
        directory = shlex.quote(identity.remote_run_directory)
        command = (
            f"dir={directory}; pid={identity.remote_pid}; "
            "boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null); "
            "start=$(awk '{print $22}' /proc/$pid/stat 2>/dev/null); "
            "pgid=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' '); "
            "stat=$(ps -o stat= -p $pid 2>/dev/null | tr -d ' '); "
            "manifest=$(sha256sum \"$dir/launch_manifest.json\" 2>/dev/null | awk '{print $1}'); "
            'exit_code=$(cat "$dir/exit_code" 2>/dev/null); '
            'printf \'%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n\' "$boot" "$start" "$pgid" '
            '"$stat" "$manifest" "$exit_code"'
        )
        async with self._session() as session:
            result = await session.run(command, timeout=10)
        if result.exit_status != 0:
            return self._unknown(identity, "remote_observation_failed")
        fields = result.stdout.rstrip("\n").split("\t")
        if len(fields) != 6:
            return self._unknown(identity, "remote_observation_malformed")
        boot_id, start_text, pgid_text, process_state, manifest_sha, exit_text = fields
        try:
            identity_matches = (
                boot_id == identity.boot_id
                and manifest_sha == identity.launch_manifest_sha256
                and (not start_text or int(start_text) == identity.process_start_ticks)
                and (not pgid_text or int(pgid_text) == identity.process_group_id)
            )
            exit_code = int(exit_text) if exit_text else None
        except ValueError:
            return self._unknown(identity, "remote_observation_malformed")
        if not identity_matches:
            return self._unknown(identity, "remote_process_identity_mismatch")
        if exit_code is not None:
            return RemoteObservation(
                identity=identity,
                state=RemoteRunState.COMPLETED if exit_code == 0 else RemoteRunState.FAILED,
                observed_at=utc_now(),
                exit_code=exit_code,
                safe_reason="remote_exit_code_recorded",
            )
        if process_state.startswith("T") and start_text and pgid_text:
            return RemoteObservation(
                identity=identity,
                state=RemoteRunState.PAUSED,
                observed_at=utc_now(),
                safe_reason="remote_process_paused",
            )
        if process_state and not process_state.startswith("Z") and start_text and pgid_text:
            return RemoteObservation(
                identity=identity,
                state=RemoteRunState.RUNNING,
                observed_at=utc_now(),
                safe_reason="remote_process_alive",
            )
        return self._unknown(identity, "remote_exit_unproven")

    async def capacity_preflight(
        self, policy: RemoteCapacityPolicy | None = None
    ) -> RemoteCapacitySnapshot:
        """Fail closed when private compute is already occupied or undersized."""

        contract = policy or RemoteCapacityPolicy()
        async with self._session() as session:
            root = await self._resolve_remote_root(session)
            command = (
                f"probe={shlex.quote(root)}; "
                'while [ ! -e "$probe" ] && [ "$probe" != "/" ]; '
                'do probe=$(dirname "$probe"); done; '
                'test -d "$probe" || exit 4; '
                "mem=$(awk '/MemAvailable:/ {printf \"%.3f\", $2/1048576}' /proc/meminfo); "
                'disk=$(df -BG --output=avail "$probe" 2>/dev/null '
                "| tail -1 | tr -dc '0-9'); "
                "gpu=$(nvidia-smi --query-compute-apps=pid,process_name "
                "--format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | tr '\\n' ';'); "
                'printf \'%s\\t%s\\t%s\\n\' "$mem" "$disk" "$gpu"'
            )
            result = await session.run(command, timeout=15)
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_capacity_unavailable")
        fields = result.stdout.rstrip("\n").split("\t")
        if len(fields) != 3:
            raise RuntimeError("campaign_remote_capacity_malformed")
        try:
            memory_gib = float(fields[0])
            disk_gib = float(fields[1])
        except ValueError as exc:
            raise RuntimeError("campaign_remote_capacity_malformed") from exc
        processes = tuple(sorted(item.strip() for item in fields[2].split(";") if item.strip()))
        reasons: list[str] = []
        if memory_gib < contract.minimum_available_memory_gib:
            reasons.append("available_memory_below_minimum")
        if disk_gib < contract.minimum_available_disk_gib:
            reasons.append("available_disk_below_minimum")
        if len(processes) > contract.maximum_external_gpu_processes:
            reasons.append("external_gpu_process_limit_exceeded")
        return RemoteCapacitySnapshot(
            compute_profile_id=self.compute_profile_id,
            available_memory_gib=memory_gib,
            available_disk_gib=disk_gib,
            external_gpu_processes=processes,
            admitted=not reasons,
            blocking_reasons=tuple(reasons),
            observed_at=utc_now(),
        )

    async def read_stream(
        self,
        identity: RemoteRunIdentity,
        source: str,
        cursor: RemoteStreamCursor | None = None,
        *,
        max_bytes: int = 65_536,
    ) -> RemoteStreamChunk:
        """Read an append-only log/metric stream with a durable byte cursor."""

        self._validate_adapter_identity(identity)
        if source not in {"training.log", "training_metrics.jsonl"}:
            raise ValueError("campaign_remote_stream_source_invalid")
        if max_bytes < 1 or max_bytes > 1_048_576:
            raise ValueError("campaign_remote_stream_limit_invalid")
        current = cursor or RemoteStreamCursor()
        remote_path = f"{identity.remote_run_directory}/{source}"
        python = (
            "import base64,json,sys;"
            "p=sys.argv[1];o=int(sys.argv[2]);m=int(sys.argv[3]);"
            "f=open(p,'rb');f.seek(o);b=f.read(m);"
            "print(json.dumps({'end_offset':o+len(b),'data':base64.b64encode(b).decode()}))"
        )
        command = (
            f"test $(sha256sum {shlex.quote(identity.remote_run_directory + '/launch_manifest.json')} "
            f"| awk '{{print $1}}') = {identity.launch_manifest_sha256} && "
            f"python3 -c {shlex.quote(python)} {shlex.quote(remote_path)} "
            f"{current.byte_offset} {max_bytes}"
        )
        async with self._session() as session:
            result = await session.run(command, timeout=15)
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_stream_unavailable")
        try:
            payload = json.loads(result.stdout)
            end_offset = int(payload["end_offset"])
            decoded = base64.b64decode(payload["data"], validate=True).decode("utf-8")
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("campaign_remote_stream_malformed") from exc
        if end_offset < current.byte_offset or end_offset - current.byte_offset > max_bytes:
            raise RuntimeError("campaign_remote_stream_cursor_invalid")
        pieces = (current.partial_line + decoded).splitlines(keepends=True)
        partial_line = ""
        if pieces and not pieces[-1].endswith(("\n", "\r")):
            partial_line = pieces.pop()
        lines = tuple(piece.rstrip("\r\n") for piece in pieces)
        next_cursor = RemoteStreamCursor(byte_offset=end_offset, partial_line=partial_line)
        return RemoteStreamChunk(
            source=source,
            start_offset=current.byte_offset,
            end_offset=end_offset,
            complete_lines=lines,
            next_cursor=next_cursor,
        )

    async def collect_outputs(
        self,
        identity: RemoteRunIdentity,
        request: RemoteLaunchRequest,
        local_directory: Path,
        *,
        observation: RemoteObservation | None = None,
    ) -> tuple[Path, ...]:
        proven = observation or await self.observe(identity)
        if proven.identity != identity or proven.state != RemoteRunState.COMPLETED:
            raise RuntimeError("campaign_remote_outputs_not_ready")
        all_paths = (*request.output_paths, "training.log", "exit_code", "launch_manifest.json")
        predicates = []
        for relative in all_paths:
            remote = f"{identity.remote_run_directory}/{relative}"
            quoted = shlex.quote(remote)
            predicates.append(
                f"test -e {quoted} && test ! -L {quoted} && "
                f'test -z "$(find {quoted} -type l -print -quit 2>/dev/null)"'
            )
        async with self._session() as session:
            checked = await session.run(" && ".join(predicates), timeout=30)
            if checked.exit_status != 0:
                raise RuntimeError("campaign_remote_outputs_invalid")
            downloaded: list[Path] = []
            for relative in all_paths:
                local_path = local_directory / PurePosixPath(relative)
                remote_path = f"{identity.remote_run_directory}/{relative}"
                if not await session.download(remote_path, local_path):
                    raise RuntimeError("campaign_remote_outputs_incomplete")
                downloaded.append(local_path)
        for path in downloaded:
            if not path.exists() or path.is_symlink():
                raise RuntimeError("campaign_remote_outputs_invalid")
            if path.is_dir() and any(child.is_symlink() for child in path.rglob("*")):
                raise RuntimeError("campaign_remote_outputs_invalid")
        return tuple(downloaded)

    async def collect_terminal_evidence(
        self,
        identity: RemoteRunIdentity,
        local_directory: Path,
        *,
        observation: RemoteObservation | None = None,
    ) -> tuple[Path, ...]:
        """Download the closed supervisor evidence even when training failed."""

        proven = observation or await self.observe(identity)
        if proven.identity != identity or proven.state != RemoteRunState.FAILED:
            raise RuntimeError("campaign_remote_terminal_evidence_not_ready")
        required = ("training.log", "exit_code", "launch_manifest.json")
        optional = "training_metrics.jsonl"
        predicates = []
        for relative in required:
            remote = f"{identity.remote_run_directory}/{relative}"
            quoted = shlex.quote(remote)
            predicates.append(f"test -f {quoted} && test ! -L {quoted}")
        optional_remote = f"{identity.remote_run_directory}/{optional}"
        optional_quoted = shlex.quote(optional_remote)
        check_command = (
            " && ".join(predicates)
            + f" && if test -f {optional_quoted} -a ! -L {optional_quoted}; "
            "then printf present; else printf absent; fi"
        )
        async with self._session() as session:
            checked = await session.run(check_command, timeout=30)
            if checked.exit_status != 0 or checked.stdout not in {"present", "absent"}:
                raise RuntimeError("campaign_remote_terminal_evidence_invalid")
            paths = (*required, optional) if checked.stdout == "present" else required
            downloaded: list[Path] = []
            for relative in paths:
                local_path = local_directory / PurePosixPath(relative)
                remote_path = f"{identity.remote_run_directory}/{relative}"
                if not await session.download(remote_path, local_path):
                    raise RuntimeError("campaign_remote_terminal_evidence_incomplete")
                downloaded.append(local_path)
        if any(not path.is_file() or path.is_symlink() for path in downloaded):
            raise RuntimeError("campaign_remote_terminal_evidence_invalid")
        return tuple(downloaded)

    async def control(self, identity: RemoteRunIdentity, action: RemoteControl) -> bool:
        """Validate and signal the remote process group in one SSH command."""

        self._validate_adapter_identity(identity)
        signal = CONTROL_SIGNALS[action]
        directory = shlex.quote(identity.remote_run_directory)
        command = (
            f"dir={directory}; pid={identity.remote_pid}; expected_start={identity.process_start_ticks}; "
            f"expected_pgid={identity.process_group_id}; expected_boot={shlex.quote(identity.boot_id)}; "
            f"expected_manifest={identity.launch_manifest_sha256}; "
            "boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null); "
            "start=$(awk '{print $22}' /proc/$pid/stat 2>/dev/null); "
            "pgid=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' '); "
            "manifest=$(sha256sum \"$dir/launch_manifest.json\" 2>/dev/null | awk '{print $1}'); "
            'test "$boot" = "$expected_boot" -a "$start" = "$expected_start" '
            '-a "$pgid" = "$expected_pgid" -a "$manifest" = "$expected_manifest" || exit 42; '
            f'kill -{signal} -- "-$pgid"'
        )
        async with self._session() as session:
            result = await session.run(command, timeout=10)
        if result.exit_status == 42:
            return False
        return result.exit_status == 0

    async def pause(self, identity: RemoteRunIdentity) -> bool:
        return await self.control(identity, RemoteControl.PAUSE)

    async def resume(self, identity: RemoteRunIdentity) -> bool:
        return await self.control(identity, RemoteControl.RESUME)

    async def terminate(self, identity: RemoteRunIdentity) -> bool:
        return await self.control(identity, RemoteControl.TERMINATE)

    async def force_stop(self, identity: RemoteRunIdentity) -> bool:
        return await self.control(identity, RemoteControl.FORCE_STOP)

    @staticmethod
    def _unknown(identity: RemoteRunIdentity, reason: str) -> RemoteObservation:
        return RemoteObservation(
            identity=identity,
            state=RemoteRunState.UNKNOWN,
            observed_at=utc_now(),
            safe_reason=reason,
        )

    def _validate_adapter_identity(self, identity: RemoteRunIdentity) -> None:
        if identity.compute_profile_id != self.compute_profile_id:
            raise ValueError("campaign remote identity belongs to another compute profile")


def remote_command_fingerprint(request: RemoteLaunchRequest, remote_directory: str) -> str:
    return canonical_hash(
        {
            "argv": list(RemoteTrainingAdapter._argv(request, remote_directory)),
            **RemoteTrainingAdapter._execution_context(request, remote_directory),
        }
    )


__all__ = [
    "ApprovedCodeLineageExecutionBinding",
    "ApprovedRemoteExecutorProfile",
    "AsyncSSHSession",
    "CodeLineageLaunchSnapshot",
    "PinnedRemoteStageProfile",
    "RemoteCommandResult",
    "RemoteCapacityPolicy",
    "RemoteCapacitySnapshot",
    "RemoteControl",
    "RemoteLaunchRequest",
    "RemoteObservation",
    "RemoteRunIdentity",
    "RemoteRunState",
    "RemoteStreamChunk",
    "RemoteStreamCursor",
    "RemoteSupervisorState",
    "RemoteTrainingAdapter",
    "remote_executor_config",
    "remote_command_fingerprint",
]
