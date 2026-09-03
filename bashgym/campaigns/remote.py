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
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import Field, ValidationError, field_validator, model_validator

from bashgym.campaigns.autoresearch_evidence import AUTORESEARCH_EVALUATION_FILENAME
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
from bashgym.campaigns.diagnostic_actions import (
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
    AutoResearchDiagnosticRecipe,
    diagnostic_recipe_digest,
    validate_diagnostic_envelope,
)
from bashgym.campaigns.lineage import canonical_model_manifest_digest
from bashgym.campaigns.nemo_rl import ApprovedNemoRLProfile, NemoRLRuntimeReceipt
from bashgym.gym.remote_trainer import HAS_ASYNCSSH, SSHConfig

if TYPE_CHECKING:
    from bashgym.ledger.contracts import DatasetVersionSpec, EvaluationSuiteSpec


REMOTE_OUTPUT_SEAL_EXECUTOR_ID = "campaign-ssh-remote-executor"
REMOTE_OUTPUT_SEAL_EXECUTOR_VERSION = "1"
_REMOTE_CONTROLLER_READABLE_OUTPUTS = frozenset(
    {
        "autoresearch_dataset_receipt.json",
        AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
        AUTORESEARCH_EVALUATION_FILENAME,
    }
)


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


class SealedStageArtifactInput(FrozenContractModel):
    """One verified campaign artifact uploaded at an explicit remote path."""

    schema_version: Literal["campaign_sealed_stage_artifact_input.v1"] = (
        "campaign_sealed_stage_artifact_input.v1"
    )
    campaign_artifact_id: Identifier
    sha256: HexDigest
    size_bytes: int = Field(ge=0)
    schema_name: Identifier
    local_sealed_path: Path
    remote_relative_path: str = Field(min_length=1, max_length=4096)

    @field_validator("local_sealed_path")
    @classmethod
    def verified_local_material(cls, value: Path) -> Path:
        candidate = value.expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("sealed stage artifact must be a regular non-symlink file")
        return candidate.resolve()

    @field_validator("remote_relative_path")
    @classmethod
    def safe_remote_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("sealed stage artifact remote relative path is unsafe")
        return value

    def verify_material(self) -> None:
        if self.local_sealed_path.is_symlink() or not self.local_sealed_path.is_file():
            raise ValueError("sealed stage artifact material changed before launch")
        if (
            self.local_sealed_path.stat().st_size != self.size_bytes
            or _sha256_file(self.local_sealed_path) != self.sha256
        ):
            raise ValueError("sealed stage artifact material changed before launch")


class RemoteModelArtifactReceipt(FrozenContractModel):
    """Bounded physical identity computed beside an immutable model artifact."""

    schema_version: Literal["campaign_remote_model_artifact_receipt.v1"] = (
        "campaign_remote_model_artifact_receipt.v1"
    )
    model_id: str = Field(min_length=3, max_length=512)
    revision: GitObjectId
    artifact_manifest_sha256: HexDigest
    weight_file_count: int = Field(ge=1)
    total_size_bytes: int = Field(ge=1)

    @field_validator("model_id")
    @classmethod
    def exact_hugging_face_model_id(cls, value: str) -> str:
        parts = value.split("/")
        if (
            len(parts) != 2
            or any(not part or part in {".", ".."} for part in parts)
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                for character in value.replace("/", "")
            )
        ):
            raise ValueError("model ID must be one exact Hugging Face repository ID")
        return value


class RemoteModelRegistrationRequest(FrozenContractModel):
    """Explicit target-side register or acquire request for one pinned model."""

    schema_version: Literal["campaign_remote_model_registration_request.v1"] = (
        "campaign_remote_model_registration_request.v1"
    )
    operation: Literal["register", "acquire"]
    source_id: Identifier
    compute_profile_id: Identifier
    target_contract_key: Identifier
    target_model_digest: HexDigest
    model_id: str = Field(min_length=3, max_length=512)
    revision: GitObjectId
    remote_model_path: str = Field(min_length=2, max_length=4096)
    target_auth_env: str | None = Field(default=None, min_length=1, max_length=128)
    timeout_seconds: int = Field(default=21_600, ge=60, le=86_400)

    @field_validator("model_id")
    @classmethod
    def exact_hugging_face_model_id(cls, value: str) -> str:
        return RemoteModelArtifactReceipt.exact_hugging_face_model_id(value)

    @field_validator("revision", mode="before")
    @classmethod
    def immutable_revision(cls, value: Any) -> Any:
        if not (
            isinstance(value, str)
            and len(value) in {40, 64}
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise ValueError("model revision must be an immutable commit digest")
        return value

    @field_validator("remote_model_path")
    @classmethod
    def absolute_remote_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or path.as_posix() != value
            or value == "/"
            or "\\" in value
            or ".." in path.parts
            or any(character in "\x00\n\r" for character in value)
        ):
            raise ValueError("registered base model path must be one absolute remote directory")
        return value.rstrip("/")

    @field_validator("target_auth_env")
    @classmethod
    def target_environment_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not (
            value[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
            and all(character in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in value)
        ):
            raise ValueError("target auth must name a target environment variable")
        return value

    @model_validator(mode="after")
    def auth_is_acquisition_only(self) -> RemoteModelRegistrationRequest:
        if self.operation == "register" and self.target_auth_env is not None:
            raise ValueError("target auth is only valid for acquire")
        return self

    @property
    def request_digest(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class RegisteredRemoteModelSource(FrozenContractModel):
    """Server-registered base model already present on one compute profile."""

    schema_version: Literal[
        "campaign_registered_remote_model_source.v1",
        "campaign_registered_remote_model_source.v2",
    ] = "campaign_registered_remote_model_source.v1"
    source_id: Identifier
    compute_profile_id: Identifier
    target_contract_key: Identifier
    model_digest: HexDigest
    remote_model_path: str = Field(min_length=2, max_length=4096)
    artifact_receipt: RemoteModelArtifactReceipt | None = None

    @field_validator("remote_model_path")
    @classmethod
    def absolute_remote_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or path.as_posix() != value
            or value == "/"
            or "\\" in value
            or ".." in path.parts
            or any(character in "\x00\n\r" for character in value)
        ):
            raise ValueError("registered base model path must be one absolute remote directory")
        return value.rstrip("/")

    @model_validator(mode="after")
    def versioned_physical_identity(self) -> RegisteredRemoteModelSource:
        if self.schema_version == "campaign_registered_remote_model_source.v2":
            if self.artifact_receipt is None:
                raise ValueError("registered model v2 requires a physical model receipt")
        elif self.artifact_receipt is not None:
            raise ValueError("legacy registered model cannot carry a physical model receipt")
        return self

    @property
    def physical_model_digest(self) -> str:
        """Digest of evaluated bytes when attested, with v1 compatibility."""

        if self.artifact_receipt is not None:
            return self.artifact_receipt.artifact_manifest_sha256
        return self.model_digest


class RegisteredRemoteEvaluationDatasetSource(FrozenContractModel):
    """One immutable held-out dataset file registered on private compute."""

    schema_version: Literal["campaign_registered_remote_evaluation_dataset_source.v1"] = (
        "campaign_registered_remote_evaluation_dataset_source.v1"
    )
    source_id: Identifier
    compute_profile_id: Identifier
    dataset_version_id: Identifier
    content_digest: HexDigest
    remote_dataset_path: str = Field(min_length=1, max_length=4096)

    @field_validator("remote_dataset_path")
    @classmethod
    def absolute_remote_file(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or path.as_posix() != value
            or value == "/"
            or value.endswith("/")
            or "\\" in value
            or ".." in path.parts
            or any(character in "\x00\n\r" for character in value)
        ):
            raise ValueError("registered evaluation dataset path must be one absolute remote file")
        return value

    @property
    def source_uri(self) -> str:
        """Return the opaque public ledger reference without exposing the remote path."""

        return f"bashgym-remote-dataset://{self.source_id}"


class RemoteResidentModelFile(FrozenContractModel):
    """One immutable checkpoint file that remains on private compute."""

    schema_version: Literal["campaign_remote_resident_model_file.v1"] = (
        "campaign_remote_resident_model_file.v1"
    )
    remote_relative_path: str = Field(min_length=7, max_length=4096)
    sha256: HexDigest
    size_bytes: int = Field(ge=0)

    @field_validator("remote_relative_path")
    @classmethod
    def safe_model_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or not value.startswith("model/")
            or len(path.parts) < 2
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("remote-resident model path is unsafe")
        return value


class RemoteResidentModelSource(FrozenContractModel):
    """Exact training checkpoint consumed in place on the same compute profile."""

    schema_version: Literal["campaign_remote_resident_model_source.v1"] = (
        "campaign_remote_resident_model_source.v1"
    )
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    stage_index: int = Field(ge=0)
    compute_profile_id: Identifier
    remote_model_path: str = Field(min_length=2, max_length=4096)
    files: tuple[RemoteResidentModelFile, ...] = Field(min_length=1, max_length=1000)
    model_digest: HexDigest = ""

    @field_validator("remote_model_path")
    @classmethod
    def absolute_remote_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or path.as_posix() != value
            or value == "/"
            or "\\" in value
            or ".." in path.parts
            or any(character in "\x00\n\r" for character in value)
        ):
            raise ValueError("remote-resident model path must be one absolute remote directory")
        return value.rstrip("/")

    @model_validator(mode="after")
    def canonical_file_manifest(self) -> RemoteResidentModelSource:
        paths = tuple(item.remote_relative_path for item in self.files)
        if not paths or tuple(sorted(set(paths))) != paths:
            raise ValueError("remote-resident model files must be non-empty, sorted, and unique")
        expected = canonical_model_manifest_digest(self.files)
        if self.model_digest and self.model_digest != expected:
            raise ValueError("remote-resident model digest mismatch")
        if not self.model_digest:
            object.__setattr__(self, "model_digest", expected)
        return self


class RemoteResidentDatasetFile(FrozenContractModel):
    """One immutable dataset shard consumed in place on private compute."""

    schema_version: Literal["campaign_remote_resident_dataset_file.v1"] = (
        "campaign_remote_resident_dataset_file.v1"
    )
    remote_relative_path: str = Field(min_length=1, max_length=4096)
    sha256: HexDigest
    size_bytes: int = Field(ge=0)

    @field_validator("remote_relative_path")
    @classmethod
    def safe_dataset_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("remote-resident dataset path is unsafe")
        return value


class RemoteResidentDatasetSource(FrozenContractModel):
    """Exact generated dataset consumed without copying its rows to the controller."""

    schema_version: Literal["campaign_remote_resident_dataset_source.v1"] = (
        "campaign_remote_resident_dataset_source.v1"
    )
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    stage_index: int = Field(
        ge=0,
        description=(
            "Data-build stage index inside the consuming study's plan; every other"
            " identifier on this model names the producing attempt."
        ),
    )
    compute_profile_id: Identifier
    remote_dataset_path: str = Field(min_length=2, max_length=4096)
    dataset_id: Identifier
    dataset_version_id: Identifier
    content_digest: HexDigest
    files: tuple[RemoteResidentDatasetFile, ...] = Field(min_length=1, max_length=1000)

    @field_validator("remote_dataset_path")
    @classmethod
    def absolute_remote_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or path.as_posix() != value
            or value == "/"
            or "\\" in value
            or ".." in path.parts
            or any(character in "\x00\n\r" for character in value)
        ):
            raise ValueError("remote-resident dataset path must be one absolute remote directory")
        return value.rstrip("/")

    @model_validator(mode="after")
    def canonical_file_manifest(self) -> RemoteResidentDatasetSource:
        paths = tuple(item.remote_relative_path for item in self.files)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("remote-resident dataset files must be sorted and unique")
        return self


class SealedStageArtifactSource(FrozenContractModel):
    """Exact preceding training attempt that produced evaluator model inputs."""

    schema_version: Literal["campaign_sealed_stage_artifact_source.v1"] = (
        "campaign_sealed_stage_artifact_source.v1"
    )
    campaign_id: Identifier
    study_id: Identifier
    action_id: Identifier
    attempt_id: Identifier
    stage_index: int = Field(ge=0)


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
    sealed_stage_artifact_inputs: tuple[SealedStageArtifactInput, ...] = ()
    source_training: SealedStageArtifactSource | None = None
    registered_base_model: RegisteredRemoteModelSource | None = None
    registered_evaluation_dataset: RegisteredRemoteEvaluationDatasetSource | None = None
    remote_resident_model: RemoteResidentModelSource | None = None
    remote_resident_dataset: RemoteResidentDatasetSource | None = None
    evaluation_context_sha256: HexDigest | None = None

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
        remote_names = [
            *(path.name for path in self.input_files),
            *(item.remote_relative_path for item in self.sealed_stage_artifact_inputs),
        ]
        if len(set(remote_names)) != len(remote_names):
            raise ValueError("remote launch inputs must have unique destination paths")
        for item in self.sealed_stage_artifact_inputs:
            item.verify_material()
        if bool(self.sealed_stage_artifact_inputs) != (self.source_training is not None):
            raise ValueError("sealed stage artifacts require one explicit source identity")
        model_sources = sum(
            source is not None
            for source in (
                self.registered_base_model,
                self.source_training,
                self.remote_resident_model,
            )
        )
        if model_sources > 1:
            raise ValueError("evaluation requires exactly one evaluated model source")
        if (
            self.registered_base_model is not None
            and self.registered_base_model.compute_profile_id != self.compute_profile_id
        ):
            raise ValueError("registered base model compute profile mismatch")
        if (
            self.registered_evaluation_dataset is not None
            and self.registered_evaluation_dataset.compute_profile_id != self.compute_profile_id
        ):
            raise ValueError("registered evaluation dataset compute profile mismatch")
        if (
            self.remote_resident_model is not None
            and self.remote_resident_model.compute_profile_id != self.compute_profile_id
        ):
            raise ValueError("remote-resident model compute profile mismatch")
        if (
            self.remote_resident_dataset is not None
            and self.remote_resident_dataset.compute_profile_id != self.compute_profile_id
        ):
            raise ValueError("remote-resident dataset compute profile mismatch")
        return self

    @property
    def request_digest(self) -> str:
        """Bind launch identity to all local material and remote destinations."""

        entrypoint = (
            {
                "kind": "source_snapshot",
                "sha256": self.source_snapshot.archive_sha256,
                "size_bytes": self.source_snapshot.archive_size_bytes,
            }
            if self.source_snapshot is not None
            else {
                "kind": "pinned_script",
                "sha256": _sha256_file(self.script_path),
                "size_bytes": self.script_path.stat().st_size,
            }
        )
        return canonical_hash(
            {
                "compute_profile_id": self.compute_profile_id,
                "run_id": self.run_id,
                "entrypoint": entrypoint,
                "input_files": [
                    {
                        "name": path.name,
                        "sha256": _sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in self.input_files
                ],
                "sealed_stage_artifact_inputs": [
                    item.model_dump(mode="json", exclude={"local_sealed_path"})
                    for item in self.sealed_stage_artifact_inputs
                ],
                "source_training": (
                    self.source_training.model_dump(mode="json")
                    if self.source_training is not None
                    else None
                ),
                "registered_base_model": (
                    self.registered_base_model.model_dump(mode="json")
                    if self.registered_base_model is not None
                    else None
                ),
                "registered_evaluation_dataset": (
                    self.registered_evaluation_dataset.model_dump(mode="json")
                    if self.registered_evaluation_dataset is not None
                    else None
                ),
                "remote_resident_model": (
                    self.remote_resident_model.model_dump(mode="json")
                    if self.remote_resident_model is not None
                    else None
                ),
                "remote_resident_dataset": (
                    self.remote_resident_dataset.model_dump(mode="json")
                    if self.remote_resident_dataset is not None
                    else None
                ),
                "script_args": list(self.script_args),
                "python_executable": self.python_executable,
                "recipe_digest": self.recipe_digest,
                "output_paths": list(self.output_paths),
                "evaluation_context_sha256": self.evaluation_context_sha256,
            }
        )


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


class RemoteOutputFile(FrozenContractModel):
    """Hash-and-size identity for one file inventoried on private compute."""

    schema_version: Literal["campaign_remote_output_file.v1"] = "campaign_remote_output_file.v1"
    path: str = Field(min_length=1, max_length=4096)
    sha256: HexDigest
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or value in {"", "."}
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("remote output path is unsafe")
        return value


class RemoteOutputInventory(FrozenContractModel):
    """Ordered file inventory computed without moving remote output bytes."""

    schema_version: Literal["campaign_remote_output_inventory.v1"] = (
        "campaign_remote_output_inventory.v1"
    )
    compute_profile_id: Identifier
    run_id: Identifier
    files: tuple[RemoteOutputFile, ...] = Field(min_length=1, max_length=10_000)
    aggregate_digest: HexDigest = ""

    @model_validator(mode="after")
    def canonical_inventory(self) -> RemoteOutputInventory:
        paths = tuple(item.path for item in self.files)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("remote output inventory must be sorted and unique")
        expected = canonical_hash(
            [[item.path, item.sha256, item.size_bytes] for item in self.files]
        )
        if self.aggregate_digest and self.aggregate_digest != expected:
            raise ValueError("remote output inventory digest mismatch")
        if not self.aggregate_digest:
            object.__setattr__(self, "aggregate_digest", expected)
        return self


class RemoteSession(Protocol):
    async def run(self, command: str, *, timeout: float | None = None) -> RemoteCommandResult: ...

    async def upload(self, local_path: Path, remote_path: str) -> None: ...

    async def upload_bytes(self, payload: bytes, remote_path: str) -> None: ...

    async def download(self, remote_path: str, local_path: Path) -> bool: ...


SessionFactory = Callable[[], Any]
_SOURCE_ENTRYPOINT_BOOTSTRAP = (
    "import runpy,sys;"
    "source=sys.argv.pop(1);entrypoint=sys.argv.pop(1);"
    "sys.path.insert(0,source);"
    "runpy.run_path(entrypoint,run_name='__main__')"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class DiagnosticCapability(FrozenContractModel):
    """One installation-owned measurement capability exposed to the host agent."""

    schema_version: Literal["campaign_diagnostic_capability.v1"] = (
        "campaign_diagnostic_capability.v1"
    )
    capability_id: Identifier
    description: str = Field(min_length=1, max_length=1000)
    measurements: tuple[Identifier, ...] = Field(min_length=1, max_length=32)
    evidence_sources: tuple[Identifier, ...] = Field(default=(), max_length=16)

    @field_validator("description")
    @classmethod
    def plain_description(cls, value: str) -> str:
        normalized = value.strip()
        if any(character in normalized for character in "\x00\r\n"):
            raise ValueError("diagnostic capability description must be one line")
        return normalized

    @field_validator("measurements")
    @classmethod
    def canonical_measurements(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("diagnostic capability measurements must be sorted and unique")
        return value

    @field_validator("evidence_sources")
    @classmethod
    def canonical_evidence_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("diagnostic capability evidence sources must be sorted and unique")
        return value


class DiagnosticStageContract(FrozenContractModel):
    """Installation-owned limits for the open diagnostic request ABI."""

    schema_version: Literal["campaign_diagnostic_stage_contract.v1"] = (
        "campaign_diagnostic_stage_contract.v1"
    )
    runner_id: Identifier
    runner_version: str = Field(min_length=1, max_length=240)
    max_sample_limit: int = Field(ge=1, le=10_000)
    max_measurements: int = Field(ge=1, le=16)
    capabilities: tuple[DiagnosticCapability, ...] = Field(default=(), max_length=32)

    @field_validator("capabilities")
    @classmethod
    def canonical_capabilities(
        cls, value: tuple[DiagnosticCapability, ...]
    ) -> tuple[DiagnosticCapability, ...]:
        identifiers = tuple(item.capability_id for item in value)
        if tuple(sorted(set(identifiers))) != identifiers:
            raise ValueError("diagnostic capability IDs must be sorted and unique")
        return value


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
    diagnostic_contract: DiagnosticStageContract | None = None

    @field_validator("stage")
    @classmethod
    def approved_compute_stage_only(cls, value: StageKind) -> StageKind:
        if value not in {
            StageKind.DATA_BUILD,
            StageKind.CONTRACT_EVALUATION,
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
        if (
            self.stage
            not in {
                StageKind.CONTRACT_EVALUATION,
                StageKind.DEVELOPMENT_EVALUATION,
            }
            and not self.input_files
        ):
            raise ValueError("training and data-build profiles require at least one input file")
        if self.script_path.name in {path.name for path in self.input_files}:
            raise ValueError("approved script and input files must have distinct basenames")
        expected_inputs = {path.name for path in self.input_files}
        if set(self.input_sha256) != expected_inputs:
            raise ValueError("approved input hashes must exactly match input basenames")
        if self.stage == StageKind.DEVELOPMENT_EVALUATION:
            reserved = ("--context", "--model-dir", "--dataset", "--output")
            if any(
                argument.casefold() == flag or argument.casefold().startswith(f"{flag}=")
                for argument in self.script_args
                for flag in reserved
            ):
                raise ValueError("reserved evaluator argument must be supplied by the ABI")
        if self.stage == StageKind.CONTRACT_EVALUATION:
            if self.diagnostic_contract is None or self.output_paths != (
                AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
            ):
                raise ValueError("diagnostic stage requires its fixed ABI")
            reserved = ("--request", "--output")
            if any(
                argument.casefold() == flag or argument.casefold().startswith(f"{flag}=")
                for argument in self.script_args
                for flag in reserved
            ):
                raise ValueError("reserved diagnostic argument must be supplied by the ABI")
        elif self.diagnostic_contract is not None:
            raise ValueError("diagnostic contract is stage-specific")
        if self.stage == StageKind.FULL_TRAINING and any(
            argument.casefold() in {"--dataset-dir", "--model-dir"}
            or argument.casefold().startswith(("--dataset-dir=", "--model-dir="))
            for argument in self.script_args
        ):
            raise ValueError("reserved training input argument must be supplied by the ABI")
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
    registered_base_model: RegisteredRemoteModelSource | None = None
    registered_evaluation_dataset: RegisteredRemoteEvaluationDatasetSource | None = None

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
        if self.registered_base_model is not None:
            registered = self.registered_base_model
            if (
                registered.compute_profile_id != self.compute_profile_id
                or registered.target_contract_key != self.target_contract_key
                or registered.model_digest != self.target_model_digest
                or not {
                    StageKind.DEVELOPMENT_EVALUATION,
                    StageKind.FULL_TRAINING,
                }.intersection(stage.stage for stage in self.stages)
            ):
                raise ValueError("registered base model does not match its remote executor")
        if self.registered_evaluation_dataset is not None:
            registered_dataset = self.registered_evaluation_dataset
            if (
                registered_dataset.compute_profile_id != self.compute_profile_id
                or StageKind.DEVELOPMENT_EVALUATION not in {stage.stage for stage in self.stages}
            ):
                raise ValueError("registered evaluation dataset does not match its remote executor")
        excluded = {"profile_digest"}
        if self.nemo_rl is None:
            # Preserve the v2 digest of profiles written before the optional
            # NeMo RL extension existed.
            excluded.add("nemo_rl")
        if self.registered_base_model is None:
            excluded.add("registered_base_model")
        if self.registered_evaluation_dataset is None:
            excluded.add("registered_evaluation_dataset")
        payload = self.model_dump(mode="json", exclude=excluded)
        expected = canonical_hash(payload)
        if self.profile_digest and self.profile_digest != expected:
            legacy_payload = json.loads(json.dumps(payload))
            legacy_stages = legacy_payload.get("stages", [])
            for stage in legacy_stages:
                stage.pop("diagnostic_contract", None)
            legacy_digest_matches = all(
                stage.diagnostic_contract is None for stage in self.stages
            ) and self.profile_digest == canonical_hash(legacy_payload)
            if not legacy_digest_matches:
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
    recipe_script_args: tuple[str, ...] = (),
    code_lineage: CodeLineageRecord | None = None,
    sealed_stage_artifact_inputs: tuple[SealedStageArtifactInput, ...] = (),
    evaluation_suite: EvaluationSuiteSpec | None = None,
    dataset_version: DatasetVersionSpec | None = None,
    source_training: SealedStageArtifactSource | dict[str, Any] | None = None,
    remote_resident_model: RemoteResidentModelSource | dict[str, Any] | None = None,
    remote_resident_dataset: RemoteResidentDatasetSource | dict[str, Any] | None = None,
    bind_registered_training_base: bool = False,
    evaluate_registered_base_model: bool = False,
    diagnostic_recipe: AutoResearchDiagnosticRecipe | dict[str, Any] | None = None,
    approved_data_scopes: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Project one protected profile stage into the persisted executor contract."""

    profile.verify_materials()
    configured = profile.stage_profile(stage)
    recipe_script_args = PinnedRemoteStageProfile.exact_secret_free_args(recipe_script_args)
    if recipe_script_args and stage not in {
        StageKind.DATA_BUILD,
        StageKind.SMOKE_TRAINING,
        StageKind.FULL_TRAINING,
    }:
        raise ValueError("recipe arguments are supported only for data-build and training stages")
    result: dict[str, Any] = {
        "seal_executor_id": REMOTE_OUTPUT_SEAL_EXECUTOR_ID,
        "seal_executor_version": REMOTE_OUTPUT_SEAL_EXECUTOR_VERSION,
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
        "script_args": [*configured.script_args, *recipe_script_args],
        "python_executable": configured.python_executable,
        "output_paths": list(configured.output_paths),
        "capacity_policy": configured.capacity_policy.model_dump(mode="json"),
        "budget_unit": configured.budget_unit,
        "budget_reservation": configured.budget_reservation,
        "recipe_digest": recipe_digest,
    }
    if stage == StageKind.CONTRACT_EVALUATION:
        if diagnostic_recipe is None or configured.diagnostic_contract is None:
            raise ValueError("diagnostic stage requires one typed recipe")
        recipe = AutoResearchDiagnosticRecipe.model_validate(diagnostic_recipe)
        if diagnostic_recipe_digest(recipe) != recipe_digest:
            raise ValueError("diagnostic recipe digest mismatch")
        contract = configured.diagnostic_contract
        validate_diagnostic_envelope(
            recipe,
            approved_data_scopes=approved_data_scopes,
            max_sample_limit=contract.max_sample_limit,
            max_measurements=contract.max_measurements,
        )
        result["diagnostic_recipe"] = recipe.model_dump(mode="json")
        result["diagnostic_contract"] = contract.model_dump(mode="json")
    elif diagnostic_recipe is not None:
        raise ValueError("diagnostic recipe is restricted to contract evaluation")
    if recipe_script_args:
        result["recipe_script_args"] = list(recipe_script_args)
    if remote_resident_dataset is not None:
        if stage != StageKind.FULL_TRAINING:
            raise ValueError("remote-resident datasets are consumed only by full training")
        resident_dataset = RemoteResidentDatasetSource.model_validate(remote_resident_dataset)
        if resident_dataset.compute_profile_id != profile.compute_profile_id:
            raise ValueError("remote-resident dataset compute profile mismatch")
        result["remote_resident_dataset"] = resident_dataset.model_dump(mode="json")
    resident_model = None
    if remote_resident_model is not None:
        resident_model = RemoteResidentModelSource.model_validate(remote_resident_model)
        if resident_model.compute_profile_id != profile.compute_profile_id:
            raise ValueError("remote-resident model compute profile mismatch")
        if stage == StageKind.FULL_TRAINING:
            result["remote_resident_model"] = resident_model.model_dump(mode="json")
        elif stage != StageKind.DEVELOPMENT_EVALUATION:
            raise ValueError("remote-resident models are consumed only by training or evaluation")
    if bind_registered_training_base and stage != StageKind.FULL_TRAINING:
        raise ValueError("registered training bases are consumed only by full training")
    if bind_registered_training_base and resident_model is not None:
        raise ValueError("full training requires exactly one model source")
    if stage == StageKind.FULL_TRAINING and resident_model is None:
        if profile.registered_base_model is None:
            raise ValueError("full training requires one registered base model")
        result["training_base_model"] = profile.registered_base_model.model_dump(mode="json")
    if stage == StageKind.DEVELOPMENT_EVALUATION:
        if evaluation_suite is None or dataset_version is None:
            raise ValueError("development evaluation requires registered evaluator inputs")
        if (
            evaluation_suite.workspace_id != dataset_version.workspace_id
            or evaluation_suite.project_id != dataset_version.project_id
            or evaluation_suite.dataset_version_id != dataset_version.dataset_version_id
            or configured.script_sha256 != evaluation_suite.code_digest
            or AUTORESEARCH_EVALUATION_FILENAME not in configured.output_paths
        ):
            raise ValueError("development evaluation profile does not match registrations")
        registered_dataset = profile.registered_evaluation_dataset
        if registered_dataset is None:
            raise ValueError("development evaluation requires one registered remote dataset")
        if (
            registered_dataset.compute_profile_id != profile.compute_profile_id
            or registered_dataset.dataset_version_id != dataset_version.dataset_version_id
            or registered_dataset.content_digest != dataset_version.content_digest
        ):
            raise ValueError("development evaluation dataset registration mismatch")
        result["registered_evaluation_dataset"] = registered_dataset.model_dump(mode="json")
        result["evaluation_binding"] = {
            "ledger_project_id": evaluation_suite.project_id,
            "evaluation_suite_id": evaluation_suite.evaluation_suite_id,
            "evaluation_code_digest": evaluation_suite.code_digest,
            "dataset_version_id": dataset_version.dataset_version_id,
            "dataset_content_digest": dataset_version.content_digest,
            "dataset_remote_path": registered_dataset.remote_dataset_path,
            # Compatibility for workers written against the original local-input key.
            "dataset_remote_name": registered_dataset.remote_dataset_path,
        }
        if evaluate_registered_base_model:
            if (
                source_training is not None
                or sealed_stage_artifact_inputs
                or remote_resident_model is not None
            ):
                raise ValueError("evaluation requires exactly one evaluated model source")
            registered = profile.registered_base_model
            if registered is None:
                raise ValueError("development baseline requires one registered base model")
            result["registered_base_model"] = registered.model_dump(mode="json")
            result["evaluated_model_digest"] = registered.physical_model_digest
        else:
            if resident_model is not None:
                if source_training is not None or sealed_stage_artifact_inputs:
                    raise ValueError("evaluation requires exactly one evaluated model source")
                source = resident_model
                result["remote_resident_model"] = source.model_dump(mode="json")
                result["evaluated_model_digest"] = source.model_digest
            else:
                if source_training is None or not sealed_stage_artifact_inputs:
                    raise ValueError(
                        "development candidate requires one full-training model source"
                    )
                source = SealedStageArtifactSource.model_validate(source_training)
                for item in sealed_stage_artifact_inputs:
                    item.verify_material()
                    if (
                        item.schema_name != "huggingface_model_file.v1"
                        or not item.remote_relative_path.startswith("model/")
                    ):
                        raise ValueError("development evaluation model input is invalid")
                result["sealed_stage_artifact_inputs"] = [
                    item.model_dump(mode="json") for item in sealed_stage_artifact_inputs
                ]
                result["source_training"] = source.model_dump(
                    mode="json", exclude={"schema_version"}
                )
                result["evaluated_model_digest"] = canonical_model_manifest_digest(
                    sealed_stage_artifact_inputs
                )
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

    async def upload_bytes(self, payload: bytes, remote_path: str) -> None:
        temporary_path = f"{remote_path}.tmp-{hashlib.sha256(payload).hexdigest()[:16]}"
        async with self.sftp.open(temporary_path, "wb") as handle:
            await handle.write(payload)
        await self.sftp.chmod(temporary_path, 0o600)
        await self.sftp.rename(temporary_path, remote_path)

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

    async def register_remote_model(
        self,
        request: RemoteModelRegistrationRequest,
    ) -> RegisteredRemoteModelSource:
        """Inspect or acquire one model entirely on its selected compute target."""

        if request.compute_profile_id != self.compute_profile_id:
            raise ValueError("remote model registration compute profile mismatch")
        command = _remote_model_registration_command(request)
        async with self._session() as session:
            result = await session.run(command, timeout=request.timeout_seconds)
        if result.exit_status != 0:
            reason = (
                "campaign_remote_model_acquisition_failed"
                if request.operation == "acquire"
                else "campaign_remote_model_registration_failed"
            )
            raise RuntimeError(reason)
        receipt = _parse_remote_model_artifact_receipt(result.stdout, request)
        return RegisteredRemoteModelSource(
            schema_version="campaign_registered_remote_model_source.v2",
            source_id=request.source_id,
            compute_profile_id=request.compute_profile_id,
            target_contract_key=request.target_contract_key,
            model_digest=request.target_model_digest,
            remote_model_path=request.remote_model_path,
            artifact_receipt=receipt,
        )

    async def verify_registered_base_model(self, source: RegisteredRemoteModelSource) -> None:
        """Confirm an operator-selected base already exists on this compute target."""

        if source.compute_profile_id != self.compute_profile_id:
            raise ValueError("registered base model compute profile mismatch")
        async with self._session() as session:
            await self._verify_registered_base_model(session, source)

    async def _verify_registered_base_model(
        self,
        session: RemoteSession,
        source: RegisteredRemoteModelSource,
    ) -> None:
        if source.compute_profile_id != self.compute_profile_id:
            raise ValueError("registered base model compute profile mismatch")
        if source.artifact_receipt is not None:
            request = RemoteModelRegistrationRequest(
                operation="register",
                source_id=source.source_id,
                compute_profile_id=source.compute_profile_id,
                target_contract_key=source.target_contract_key,
                target_model_digest=source.model_digest,
                model_id=source.artifact_receipt.model_id,
                revision=source.artifact_receipt.revision,
                remote_model_path=source.remote_model_path,
            )
            result = await session.run(_remote_model_registration_command(request), timeout=3600)
            if result.exit_status != 0:
                raise RuntimeError("campaign_registered_base_model_not_ready")
            observed = _parse_remote_model_artifact_receipt(result.stdout, request)
            if observed != source.artifact_receipt:
                raise RuntimeError("campaign_registered_base_model_changed")
            return
        quoted = shlex.quote(source.remote_model_path)
        result = await session.run(
            f"test -d {quoted} && test ! -L {quoted} && "
            f"test -f {quoted}/config.json && test ! -L {quoted}/config.json && "
            f"find {quoted} -maxdepth 1 -type f "
            "\\( -name '*.safetensors' -o -name 'pytorch_model*.bin' \\) "
            "-print -quit | grep -q .",
            timeout=10,
        )
        if result.exit_status != 0:
            raise RuntimeError("campaign_registered_base_model_not_ready")

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
            architecture = (await probe("uname -m", "nemo_rl_platform_unavailable")).casefold()
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
            nemo_gym_source_revision = None
            if profile.nemo_gym is not None:
                nemo_gym_source_revision = await probe(
                    "docker run --rm --network=none --entrypoint git "
                    f"{shlex.quote(profile.image_reference)} "
                    "-C /opt/nemo-rl/3rdparty/Gym-workspace/Gym rev-parse HEAD",
                    "nemo_gym_source_probe_failed",
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
                f'(test "$(basename {quoted_model})" = {quoted_revision} || '
                f'test "$(cat {quoted_model}/.bashgym-model-revision 2>/dev/null)" = {quoted_revision})',
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
            nemo_gym_source_revision=nemo_gym_source_revision,
            nemo_gym_source_ready=(
                nemo_gym_source_revision == profile.nemo_gym.nemo_gym_source_revision
                if profile.nemo_gym is not None
                else None
            ),
        )

    @staticmethod
    def _argv(request: RemoteLaunchRequest, remote_directory: str) -> tuple[str, ...]:
        if request.source_snapshot is not None:
            entrypoint = f"{remote_directory}/source/{request.source_snapshot.entrypoint_path}"
            return (
                request.python_executable,
                "-c",
                _SOURCE_ENTRYPOINT_BOOTSTRAP,
                f"{remote_directory}/source",
                entrypoint,
                *request.script_args,
            )
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
        return (
            *entrypoint_material,
            *request.input_files,
            *(item.local_sealed_path for item in request.sealed_stage_artifact_inputs),
        )

    @staticmethod
    def _launch_items(request: RemoteLaunchRequest) -> tuple[tuple[Path, str], ...]:
        entrypoint_material = (
            ((request.source_snapshot.archive_path, request.source_snapshot.archive_path.name),)
            if request.source_snapshot is not None
            else ((request.script_path, request.script_path.name),)
        )
        return (
            *entrypoint_material,
            *((path, path.name) for path in request.input_files),
            *(
                (item.local_sealed_path, item.remote_relative_path)
                for item in request.sealed_stage_artifact_inputs
            ),
        )

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

    async def _verify_remote_model_source(
        self, session: RemoteSession, source: RemoteResidentModelSource
    ) -> None:
        if source.compute_profile_id != self.compute_profile_id:
            raise ValueError("remote-resident model compute profile mismatch")
        predicates = []
        for item in source.files:
            relative = item.remote_relative_path.removeprefix("model/")
            remote_path = f"{source.remote_model_path}/{relative}"
            quoted = shlex.quote(remote_path)
            predicates.append(
                f"test -f {quoted} && test ! -L {quoted} && "
                f'test "$(stat -c %s {quoted})" = {item.size_bytes} && '
                f"test \"$(sha256sum {quoted} | awk '{{print $1}}')\" = {item.sha256}"
            )
        result = await session.run(" && ".join(predicates), timeout=3600)
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_resident_model_invalid")

    async def verify_remote_model_source(self, source: RemoteResidentModelSource) -> None:
        """Verify an in-place candidate checkpoint without moving any model bytes."""

        async with self._session() as session:
            await self._verify_remote_model_source(session, source)

    async def _verify_remote_dataset_source(
        self, session: RemoteSession, source: RemoteResidentDatasetSource
    ) -> None:
        if source.compute_profile_id != self.compute_profile_id:
            raise ValueError("remote-resident dataset compute profile mismatch")
        predicates = []
        for item in source.files:
            remote_path = f"{source.remote_dataset_path}/{item.remote_relative_path}"
            quoted = shlex.quote(remote_path)
            predicates.append(
                f"test -f {quoted} && test ! -L {quoted} && "
                f'test "$(stat -c %s {quoted})" = {item.size_bytes} && '
                f"test \"$(sha256sum {quoted} | awk '{{print $1}}')\" = {item.sha256}"
            )
        result = await session.run(" && ".join(predicates), timeout=3600)
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_resident_dataset_invalid")

    async def verify_remote_dataset_source(self, source: RemoteResidentDatasetSource) -> None:
        """Verify generated rows on their source compute target without moving them."""

        async with self._session() as session:
            await self._verify_remote_dataset_source(session, source)

    async def _verify_registered_evaluation_dataset(
        self,
        session: RemoteSession,
        source: RegisteredRemoteEvaluationDatasetSource,
    ) -> None:
        if source.compute_profile_id != self.compute_profile_id:
            raise ValueError("registered evaluation dataset compute profile mismatch")
        quoted = shlex.quote(source.remote_dataset_path)
        command = (
            f"test -f {quoted} && test ! -L {quoted} && "
            f"test \"$(sha256sum {quoted} | awk '{{print $1}}')\" = {source.content_digest}"
        )
        result = await session.run(command, timeout=3600)
        if result.exit_status != 0:
            raise RuntimeError("campaign_registered_evaluation_dataset_invalid")

    async def verify_registered_evaluation_dataset(
        self, source: RegisteredRemoteEvaluationDatasetSource
    ) -> None:
        """Verify a held-out file in place without uploading or downloading its rows."""

        async with self._session() as session:
            await self._verify_registered_evaluation_dataset(session, source)

    @staticmethod
    def _launch_manifest(request: RemoteLaunchRequest, remote_directory: str) -> dict[str, Any]:
        launch_items = RemoteTrainingAdapter._launch_items(request)
        if request.source_snapshot is not None and (
            request.source_snapshot.archive_path.stat().st_size
            != request.source_snapshot.archive_size_bytes
            or _sha256_file(request.source_snapshot.archive_path)
            != request.source_snapshot.archive_sha256
        ):
            raise ValueError("code lineage snapshot changed before launch")
        for item in request.sealed_stage_artifact_inputs:
            item.verify_material()
        argv = RemoteTrainingAdapter._argv(request, remote_directory)
        execution_context = RemoteTrainingAdapter._execution_context(request, remote_directory)
        command_contract = {"argv": list(argv), **execution_context}
        manifest: dict[str, Any] = {
            "schema_version": "campaign_remote_launch_manifest.v2",
            "compute_profile_id": request.compute_profile_id,
            "run_id": request.run_id,
            "request_digest": request.request_digest,
            "evaluation_context_sha256": request.evaluation_context_sha256,
            "recipe_digest": request.recipe_digest,
            "argv": list(argv),
            "execution_context": execution_context,
            "command_hash": canonical_hash(command_contract),
            "files": [
                {
                    "name": remote_path,
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path, remote_path in launch_items
            ],
            "output_paths": list(request.output_paths),
        }
        if request.source_snapshot is not None:
            manifest["code_lineage"] = request.source_snapshot.model_dump(
                mode="json", exclude={"archive_path"}
            )
        if request.source_training is not None:
            manifest["source_training"] = request.source_training.model_dump(mode="json")
        if request.registered_base_model is not None:
            manifest["registered_base_model"] = request.registered_base_model.model_dump(
                mode="json"
            )
        if request.registered_evaluation_dataset is not None:
            manifest["registered_evaluation_dataset"] = (
                request.registered_evaluation_dataset.model_dump(mode="json")
            )
        if request.remote_resident_model is not None:
            manifest["remote_resident_model"] = request.remote_resident_model.model_dump(
                mode="json"
            )
        if request.remote_resident_dataset is not None:
            manifest["remote_resident_dataset"] = request.remote_resident_dataset.model_dump(
                mode="json"
            )
        return manifest

    async def launch(self, request: RemoteLaunchRequest) -> RemoteRunIdentity:
        if request.compute_profile_id != self.compute_profile_id:
            raise ValueError("campaign compute profile does not match remote adapter")
        async with self._session() as session:
            root = await self._resolve_remote_root(session)
            if request.registered_base_model is not None:
                await self._verify_registered_base_model(session, request.registered_base_model)
            if request.registered_evaluation_dataset is not None:
                await self._verify_registered_evaluation_dataset(
                    session, request.registered_evaluation_dataset
                )
            if request.remote_resident_model is not None:
                await self._verify_remote_model_source(session, request.remote_resident_model)
            if request.remote_resident_dataset is not None:
                await self._verify_remote_dataset_source(session, request.remote_resident_dataset)
            remote_directory = f"{root}/{request.run_id}"
            quoted_root = shlex.quote(root)
            quoted_directory = shlex.quote(remote_directory)
            input_directories = sorted(
                {
                    str(PurePosixPath(item.remote_relative_path).parent)
                    for item in request.sealed_stage_artifact_inputs
                    if str(PurePosixPath(item.remote_relative_path).parent) != "."
                }
            )
            directory_setup = "".join(
                f" && mkdir -p {shlex.quote(remote_directory + '/' + relative)}"
                for relative in input_directories
            )
            created = await session.run(
                f"umask 077 && mkdir -p {quoted_root} && mkdir {quoted_directory}{directory_setup}",
                timeout=10,
            )
            if created.exit_status != 0:
                raise RuntimeError("campaign_remote_run_already_exists")
            launch_items = self._launch_items(request)
            manifest = self._launch_manifest(request, remote_directory)
            manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            manifest_sha256 = hashlib.sha256(manifest_json.encode()).hexdigest()
            for path, remote_path in launch_items:
                await session.upload(path, f"{remote_directory}/{remote_path}")
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
            gated_inner = f"kill -STOP $$; {inner}"
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
                f"nohup setsid bash -lc {shlex.quote(gated_inner)} "
                "> training.log 2>&1 < /dev/null & "
                "pid=$!; state=; start=; pgid=; i=0; "
                'while [ "$i" -lt 100 ]; do '
                "state=$(ps -o stat= -p $pid 2>/dev/null | tr -d ' '); "
                "start=$(awk '{print $22}' /proc/$pid/stat 2>/dev/null); "
                "pgid=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' '); "
                'case "$state" in T*) break ;; esac; '
                "i=$((i + 1)); sleep 0.01; done; "
                "launched=$(date -u +%Y-%m-%dT%H:%M:%SZ); "
                'test -n "$start" -a -n "$pgid" || '
                '{ kill -KILL "$pid" 2>/dev/null; exit 3; }; '
                f"{state_command} > remote_run_state.v1.json.tmp && "
                "mv remote_run_state.v1.json.tmp remote_run_state.v1.json || "
                '{ kill -KILL "$pid" 2>/dev/null; exit 4; }; '
                'kill -CONT "$pid" && '
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

    async def inventory_outputs(
        self,
        identity: RemoteRunIdentity,
        request: RemoteLaunchRequest,
        *,
        observation: RemoteObservation | None = None,
    ) -> RemoteOutputInventory:
        """Hash the complete declared result set on compute without downloading it."""

        proven = observation or await self.observe(identity)
        if proven.identity != identity or proven.state != RemoteRunState.COMPLETED:
            raise RuntimeError("campaign_remote_outputs_not_ready")
        requested = (*request.output_paths, "training.log", "exit_code", "launch_manifest.json")
        return await self._inventory_paths(identity, requested)

    async def inventory_terminal_evidence(
        self,
        identity: RemoteRunIdentity,
        *,
        observation: RemoteObservation | None = None,
    ) -> RemoteOutputInventory:
        """Hash closed failure evidence on compute without downloading it."""

        proven = observation or await self.observe(identity)
        if proven.identity != identity or proven.state != RemoteRunState.FAILED:
            raise RuntimeError("campaign_remote_terminal_evidence_not_ready")
        return await self._inventory_paths(
            identity, ("training.log", "exit_code", "launch_manifest.json")
        )

    async def _inventory_paths(
        self, identity: RemoteRunIdentity, requested: tuple[str, ...]
    ) -> RemoteOutputInventory:
        script = """
import hashlib, json, os, sys
root = sys.argv[1]
requested = json.loads(sys.argv[2])
found = {}
for relative in requested:
    target = os.path.join(root, *relative.split('/'))
    if not os.path.lexists(target) or os.path.islink(target):
        raise SystemExit(41)
    candidates = []
    if os.path.isdir(target):
        for base, directories, filenames in os.walk(target, followlinks=False):
            if any(os.path.islink(os.path.join(base, name)) for name in directories + filenames):
                raise SystemExit(42)
            candidates.extend(os.path.join(base, name) for name in filenames)
    elif os.path.isfile(target):
        candidates.append(target)
    else:
        raise SystemExit(43)
    for path in candidates:
        relative_path = os.path.relpath(path, root).replace(os.sep, '/')
        digest = hashlib.sha256()
        size = 0
        with open(path, 'rb') as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
        found[relative_path] = {
            'path': relative_path,
            'sha256': digest.hexdigest(),
            'size_bytes': size,
        }
if not found or len(found) > 10000:
    raise SystemExit(44)
print(json.dumps([found[path] for path in sorted(found)], separators=(',', ':')))
""".strip()
        command = (
            f"python3 -c {shlex.quote(script)} "
            f"{shlex.quote(identity.remote_run_directory)} "
            f"{shlex.quote(json.dumps(requested, separators=(',', ':')))}"
        )
        async with self._session() as session:
            result = await session.run(command, timeout=3600)
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_outputs_invalid")
        try:
            files = tuple(
                RemoteOutputFile.model_validate(item) for item in json.loads(result.stdout)
            )
            return RemoteOutputInventory(
                compute_profile_id=identity.compute_profile_id,
                run_id=identity.run_id,
                files=files,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("campaign_remote_output_inventory_malformed") from exc

    async def persist_action_seal(self, identity: RemoteRunIdentity, envelope: bytes) -> str:
        """Persist an HMAC envelope beside its outputs without a controller file."""

        self._validate_adapter_identity(identity)
        if not envelope or len(envelope) > 16 * 1024 * 1024:
            raise ValueError("campaign_remote_action_seal_size_invalid")
        remote_path = f"{identity.remote_run_directory}/sealed_action_result.v1.json"
        async with self._session() as session:
            await session.upload_bytes(envelope, remote_path)
        return remote_path

    async def read_action_seal(self, identity: RemoteRunIdentity) -> bytes | None:
        """Read the bounded remote seal into memory for crash-safe reconciliation."""

        self._validate_adapter_identity(identity)
        remote_path = f"{identity.remote_run_directory}/sealed_action_result.v1.json"
        script = (
            "import base64,json,os,sys;"
            "p=sys.argv[1];m=int(sys.argv[2]);"
            "sys.exit(3) if not os.path.exists(p) else None;"
            "assert os.path.isfile(p) and not os.path.islink(p);"
            "f=open(p,'rb');b=f.read(m+1);f.close();assert len(b)<=m;"
            "print(json.dumps({'data':base64.b64encode(b).decode()},separators=(',',':')))"
        )
        command = f"python3 -c {shlex.quote(script)} {shlex.quote(remote_path)} {16 * 1024 * 1024}"
        async with self._session() as session:
            result = await session.run(command, timeout=30)
        if result.exit_status == 3:
            return None
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_action_seal_unavailable")
        try:
            payload = base64.b64decode(json.loads(result.stdout)["data"], validate=True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("campaign_remote_action_seal_malformed") from exc
        if not payload or len(payload) > 16 * 1024 * 1024:
            raise RuntimeError("campaign_remote_action_seal_malformed")
        return payload

    async def read_output_bytes(
        self,
        identity: RemoteRunIdentity,
        relative_path: str,
        *,
        expected_sha256: HexDigest,
        expected_size_bytes: int,
        max_bytes: int,
    ) -> bytes:
        """Read one bounded remote output into memory and verify its inventory identity."""

        self._validate_adapter_identity(identity)
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or path.as_posix() != relative_path
            or relative_path in {"", "."}
            or ".." in path.parts
            or "\\" in relative_path
            or relative_path not in _REMOTE_CONTROLLER_READABLE_OUTPUTS
        ):
            raise ValueError("campaign_remote_output_path_invalid")
        if max_bytes < 1 or max_bytes > 16 * 1024 * 1024:
            raise ValueError("campaign_remote_output_read_limit_invalid")
        if expected_size_bytes < 0 or expected_size_bytes > max_bytes:
            raise ValueError("campaign_remote_output_size_invalid")
        remote_path = f"{identity.remote_run_directory}/{relative_path}"
        script = (
            "import base64,hashlib,json,os,sys;"
            "p=sys.argv[1];m=int(sys.argv[2]);"
            "assert os.path.isfile(p) and not os.path.islink(p);"
            "f=open(p,'rb');b=f.read(m+1);f.close();"
            "assert len(b)<=m;"
            "print(json.dumps({'data':base64.b64encode(b).decode(),"
            "'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)},"
            "separators=(',',':')))"
        )
        command = f"python3 -c {shlex.quote(script)} {shlex.quote(remote_path)} {max_bytes}"
        async with self._session() as session:
            result = await session.run(command, timeout=30)
        if result.exit_status != 0:
            raise RuntimeError("campaign_remote_output_unavailable")
        try:
            payload = json.loads(result.stdout)
            decoded = base64.b64decode(payload["data"], validate=True)
            digest = hashlib.sha256(decoded).hexdigest()
            if (
                payload["sha256"] != digest
                or int(payload["size_bytes"]) != len(decoded)
                or digest != expected_sha256
                or len(decoded) != expected_size_bytes
            ):
                raise ValueError("identity mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("campaign_remote_output_identity_mismatch") from exc
        return decoded

    async def collect_outputs(
        self,
        identity: RemoteRunIdentity,
        request: RemoteLaunchRequest,
        local_directory: Path,
        *,
        observation: RemoteObservation | None = None,
    ) -> tuple[Path, ...]:
        del identity, request, local_directory, observation
        raise RuntimeError("campaign_controller_output_download_disabled")

    async def collect_terminal_evidence(
        self,
        identity: RemoteRunIdentity,
        local_directory: Path,
        *,
        observation: RemoteObservation | None = None,
    ) -> tuple[Path, ...]:
        """Retain failed-run logs and evidence on private compute."""

        del identity, local_directory, observation
        raise RuntimeError("campaign_controller_output_download_disabled")

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


_REMOTE_MODEL_INSPECTION_SCRIPT = """\
import hashlib
import json
from pathlib import Path
import sys


def inspect_model(path, model_id, revision):
    root = Path(path)
    resolved = root.resolve(strict=True)
    if str(resolved) != path or root.is_symlink() or not root.is_dir():
        raise RuntimeError("model directory is not exact")
    config = root / "config.json"
    if config.is_symlink() or not config.is_file():
        raise RuntimeError("model config is missing")
    files = []
    weight_count = 0
    total_size = 0
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative_path = candidate.relative_to(root)
        if (
            relative_path.parts
            and relative_path.parts[0] == ".cache"
            or relative_path.as_posix() == ".bashgym-acquisition.json"
        ):
            continue
        if candidate.is_symlink():
            raise RuntimeError("model artifact contains a symlink")
        if not candidate.is_file():
            continue
        relative = relative_path.as_posix()
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size = candidate.stat().st_size
        files.append([relative, digest.hexdigest(), size])
        total_size += size
        name = candidate.name
        if name.endswith(".safetensors") or (
            name.startswith("pytorch_model") and name.endswith(".bin")
        ):
            weight_count += 1
    if weight_count < 1:
        raise RuntimeError("model weights are missing")
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "campaign_remote_model_artifact_receipt.v1",
        "model_id": model_id,
        "revision": revision,
        "artifact_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "weight_file_count": weight_count,
        "total_size_bytes": total_size,
    }
"""


_REMOTE_MODEL_REGISTER_SCRIPT = _REMOTE_MODEL_INSPECTION_SCRIPT + """\
path, model_id, revision = sys.argv[1:]
receipt = inspect_model(path, model_id, revision)
print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
"""


_REMOTE_MODEL_ACQUIRE_SCRIPT = _REMOTE_MODEL_INSPECTION_SCRIPT + """\
import os

path, model_id, revision, auth_env, request_digest = sys.argv[1:]
destination = Path(path)
parent = destination.parent.resolve(strict=True)
if str(parent) != str(destination.parent) or destination.is_symlink():
    raise RuntimeError("model destination is not exact")
partial = parent / ("." + destination.name + ".partial-" + request_digest[:16])
owner = {
    "schema_version": "bashgym_remote_model_acquisition.v1",
    "request_digest": request_digest,
    "model_id": model_id,
    "revision": revision,
}
owner_payload = json.dumps(owner, sort_keys=True, separators=(",", ":"))


def owned(directory):
    if directory.is_symlink() or not directory.is_dir():
        return False
    marker = directory / ".bashgym-acquisition.json"
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 4096:
        return False
    return marker.read_text(encoding="utf-8") == owner_payload


if destination.exists():
    if not owned(destination):
        raise RuntimeError("model destination is occupied by another acquisition")
    receipt = inspect_model(str(destination), model_id, revision)
else:
    if partial.exists():
        if not owned(partial):
            raise RuntimeError("model partial destination belongs to another acquisition")
    else:
        partial.mkdir(mode=0o700)
        marker = partial / ".bashgym-acquisition.json"
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(owner_payload)
    token = os.environ.get(auth_env) if auth_env else None
    if auth_env and not token:
        raise RuntimeError("target auth is unavailable")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(partial),
        token=token,
    )
    receipt = inspect_model(str(partial), model_id, revision)
    os.rename(partial, destination)
print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
"""


def _remote_model_registration_command(request: RemoteModelRegistrationRequest) -> str:
    if request.operation == "acquire":
        arguments = (
            "python3",
            "-c",
            _REMOTE_MODEL_ACQUIRE_SCRIPT,
            request.remote_model_path,
            request.model_id,
            request.revision,
            request.target_auth_env or "",
            request.request_digest,
        )
    else:
        arguments = (
            "python3",
            "-c",
            _REMOTE_MODEL_REGISTER_SCRIPT,
            request.remote_model_path,
            request.model_id,
            request.revision,
        )
    return " ".join(shlex.quote(argument) for argument in arguments)


def _parse_remote_model_artifact_receipt(
    payload: str,
    request: RemoteModelRegistrationRequest,
) -> RemoteModelArtifactReceipt:
    try:
        receipt = RemoteModelArtifactReceipt.model_validate_json(payload.strip())
    except (ValidationError, ValueError) as exc:
        raise RuntimeError("campaign_remote_model_receipt_invalid") from exc
    if receipt.model_id != request.model_id or receipt.revision != request.revision:
        raise RuntimeError("campaign_remote_model_receipt_invalid")
    return receipt


__all__ = [
    "ApprovedCodeLineageExecutionBinding",
    "ApprovedRemoteExecutorProfile",
    "AsyncSSHSession",
    "CodeLineageLaunchSnapshot",
    "DiagnosticCapability",
    "DiagnosticStageContract",
    "PinnedRemoteStageProfile",
    "RegisteredRemoteEvaluationDatasetSource",
    "RegisteredRemoteModelSource",
    "RemoteModelArtifactReceipt",
    "RemoteModelRegistrationRequest",
    "RemoteResidentDatasetFile",
    "RemoteResidentDatasetSource",
    "RemoteResidentModelFile",
    "RemoteResidentModelSource",
    "SealedStageArtifactInput",
    "SealedStageArtifactSource",
    "RemoteCommandResult",
    "RemoteCapacityPolicy",
    "RemoteCapacitySnapshot",
    "RemoteControl",
    "RemoteLaunchRequest",
    "RemoteObservation",
    "RemoteOutputFile",
    "RemoteOutputInventory",
    "RemoteRunIdentity",
    "RemoteRunState",
    "RemoteStreamChunk",
    "RemoteStreamCursor",
    "RemoteSupervisorState",
    "RemoteTrainingAdapter",
    "remote_executor_config",
    "remote_command_fingerprint",
]
