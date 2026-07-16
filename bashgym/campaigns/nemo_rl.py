"""Typed, installation-owned contracts for optional NeMo RL execution."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import (
    FrozenContractModel,
    GitObjectId,
    HexDigest,
    Identifier,
    StageKind,
    canonical_hash,
    utc_now,
)

_IMAGE_REFERENCE = re.compile(
    r"^[a-z0-9][a-z0-9._:/-]*@sha256:(?P<digest>[0-9a-f]{64})$"
)
_OVERRIDE_KEY = re.compile(r"^\+?[A-Za-z][A-Za-z0-9_.]*$")
_CONTROLLER_OWNED_OVERRIDES = frozenset(
    {
        "checkpointing.checkpoint_dir",
        "grpo.max_num_steps",
        "logger.log_dir",
        "policy.model_name",
        "policy.optimizer.kwargs.lr",
        "policy.tokenizer.name",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_container_path(value: str, *, suffixes: tuple[str, ...]) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not path.name
        or not value.endswith(suffixes)
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("NeMo RL container path is invalid")
    return value


def _safe_remote_path(value: str) -> str:
    normalized = value[1:] if value == "~" or value.startswith("~/") else value
    path = PurePosixPath(normalized or "/")
    if (
        not (value == "~" or value.startswith("~/") or path.is_absolute())
        or ".." in path.parts
        or any(character in "\x00\n\r" for character in value)
    ):
        raise ValueError("NeMo RL remote model path must be absolute or home-relative")
    return value.rstrip("/") or "/"


def _validate_overrides(value: tuple[str, ...]) -> tuple[str, ...]:
    keys: list[str] = []
    for override in value:
        if (
            len(override) > 1000
            or "=" not in override
            or any(character in "\x00\n\r" for character in override)
        ):
            raise ValueError("NeMo RL override is invalid")
        key, _raw = override.split("=", 1)
        normalized_key = key.removeprefix("+")
        if (
            _OVERRIDE_KEY.fullmatch(key) is None
            or normalized_key in _CONTROLLER_OWNED_OVERRIDES
        ):
            raise ValueError("NeMo RL override key is invalid or controller-owned")
        keys.append(normalized_key)
    if len(set(keys)) != len(keys) or tuple(sorted(value)) != value:
        raise ValueError("NeMo RL overrides must be sorted and unique by key")
    return value


def _validate_model_id(value: str) -> str:
    if "@" in value or any(character in "\x00\n\r" for character in value):
        raise ValueError("NeMo RL model ID and immutable revision must be separate")
    return value


class NemoRLModelSupportLevel(str, Enum):
    BROAD_API_COMPATIBLE = "broad_api_compatible"
    RECIPE_REPRODUCED = "recipe_reproduced"
    OPTIMIZED = "optimized"


class NemoRLExecutionMode(str, Enum):
    NO_UPDATE = "no_update"
    GRPO = "grpo"


class NemoRLStageBinding(FrozenContractModel):
    schema_version: Literal["nemo_rl_stage_binding.v1"] = "nemo_rl_stage_binding.v1"
    stage: StageKind
    mode: NemoRLExecutionMode
    max_steps: int = Field(ge=1, le=10)
    learning_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def bounded_mode(self) -> NemoRLStageBinding:
        expected = {
            StageKind.SMOKE_TRAINING: NemoRLExecutionMode.NO_UPDATE,
            StageKind.FULL_TRAINING: NemoRLExecutionMode.GRPO,
        }
        if expected.get(self.stage) != self.mode:
            raise ValueError("NeMo RL stage mode must be smoke=no_update or full=grpo")
        if self.mode is NemoRLExecutionMode.NO_UPDATE and (
            self.max_steps != 1 or self.learning_rate != 0
        ):
            raise ValueError("no-update smoke requires one rollout step and zero learning rate")
        if self.mode is NemoRLExecutionMode.GRPO and self.learning_rate <= 0:
            raise ValueError("GRPO smoke requires a positive learning rate")
        return self


class NemoRLRuntimeReceipt(FrozenContractModel):
    """Bounded remote facts captured by setup; no host or credential is exposed."""

    schema_version: Literal["nemo_rl_runtime_receipt.v1"] = "nemo_rl_runtime_receipt.v1"
    compute_profile_id: Identifier
    platform: Literal["linux/amd64", "linux/arm64"]
    docker_version: str = Field(min_length=1, max_length=100)
    docker_ready: bool
    nvidia_runtime_ready: bool
    gpu_count: int = Field(ge=0, le=1024)
    available_disk_gib: float = Field(ge=0)
    shared_memory_gib: float = Field(ge=0)
    image_digest: HexDigest
    image_ready: bool
    source_revision: GitObjectId
    source_ready: bool
    recipe_sha256: HexDigest
    recipe_ready: bool
    model_revision: GitObjectId
    model_ready: bool
    observed_at: datetime = Field(default_factory=utc_now)
    receipt_digest: HexDigest = ""

    @model_validator(mode="after")
    def verify_digest(self) -> NemoRLRuntimeReceipt:
        expected = canonical_hash(self.model_dump(mode="json", exclude={"receipt_digest"}))
        if self.receipt_digest and self.receipt_digest != expected:
            raise ValueError("NeMo RL runtime receipt digest mismatch")
        if not self.receipt_digest:
            object.__setattr__(self, "receipt_digest", expected)
        return self


class NemoRLContainerContract(FrozenContractModel):
    """Secret-free contract accepted by the remote host container wrapper."""

    schema_version: Literal["nemo_rl_container_contract.v1"] = (
        "nemo_rl_container_contract.v1"
    )
    release: Identifier
    source_revision: GitObjectId
    image_reference: str = Field(min_length=1, max_length=1000)
    image_digest: HexDigest
    platform: Literal["linux/amd64", "linux/arm64"]
    model_id: str = Field(min_length=1, max_length=500)
    model_revision: GitObjectId
    remote_model_path: str = Field(min_length=1, max_length=4096)
    model_support_level: NemoRLModelSupportLevel
    entrypoint_path: str = "examples/run_grpo.py"
    recipe_path: str
    recipe_sha256: HexDigest
    dataset_file: str = Field(min_length=1, max_length=240)
    dataset_sha256: HexDigest
    verifier_id: Identifier
    verifier_digest: HexDigest
    stage: StageKind
    mode: NemoRLExecutionMode
    max_steps: int = Field(ge=1, le=10)
    learning_rate: float = Field(ge=0, le=1)
    gpu_count: int = Field(default=1, ge=1, le=8)
    shared_memory_gib: int = Field(default=16, ge=1, le=256)
    overrides: tuple[str, ...] = ()

    @field_validator("image_reference")
    @classmethod
    def immutable_image(cls, value: str) -> str:
        if _IMAGE_REFERENCE.fullmatch(value) is None:
            raise ValueError("NeMo RL image must be pinned by sha256 digest")
        return value

    @field_validator("remote_model_path")
    @classmethod
    def remote_model_directory(cls, value: str) -> str:
        return _safe_remote_path(value)

    @field_validator("model_id")
    @classmethod
    def safe_model_id(cls, value: str) -> str:
        return _validate_model_id(value)

    @field_validator("entrypoint_path")
    @classmethod
    def safe_entrypoint(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".py"
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("NeMo RL entrypoint must be a repository-relative Python file")
        return value

    @field_validator("recipe_path")
    @classmethod
    def safe_recipe(cls, value: str) -> str:
        return _safe_container_path(value, suffixes=(".yaml", ".yml"))

    @field_validator("dataset_file")
    @classmethod
    def safe_dataset(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.name != value or path.suffix not in {".json", ".jsonl"}:
            raise ValueError("NeMo RL dataset must be one JSON/JSONL basename")
        return value

    @field_validator("overrides")
    @classmethod
    def safe_overrides(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_overrides(value)

    @model_validator(mode="after")
    def exact_identity_and_mode(self) -> NemoRLContainerContract:
        match = _IMAGE_REFERENCE.fullmatch(self.image_reference)
        assert match is not None
        if match.group("digest") != self.image_digest:
            raise ValueError("NeMo RL image reference and digest disagree")
        NemoRLStageBinding(
            stage=self.stage,
            mode=self.mode,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
        )
        return self


class ApprovedNemoRLProfile(FrozenContractModel):
    """Installation-owned NeMo RL binding nested in a private-compute profile."""

    schema_version: Literal["approved_nemo_rl_profile.v1"] = "approved_nemo_rl_profile.v1"
    profile_id: Identifier
    profile_revision: int = Field(ge=1)
    profile_digest: HexDigest = ""
    compute_profile_id: Identifier
    target_contract_key: Identifier
    target_model_digest: HexDigest
    release: Identifier
    source_revision: GitObjectId
    image_reference: str = Field(min_length=1, max_length=1000)
    image_digest: HexDigest
    platform: Literal["linux/amd64", "linux/arm64"]
    model_id: str = Field(min_length=1, max_length=500)
    model_revision: GitObjectId
    remote_model_path: str = Field(min_length=1, max_length=4096)
    model_support_level: NemoRLModelSupportLevel
    entrypoint_path: str = "examples/run_grpo.py"
    recipe_path: str
    recipe_sha256: HexDigest
    dataset_path: Path
    dataset_sha256: HexDigest
    verifier_id: Identifier
    verifier_digest: HexDigest
    stage_bindings: tuple[NemoRLStageBinding, ...]
    gpu_count: int = Field(default=1, ge=1, le=8)
    shared_memory_gib: int = Field(default=16, ge=1, le=256)
    minimum_available_disk_gib: float = Field(default=80.0, ge=1)
    overrides: tuple[str, ...] = ()
    runtime_receipt: NemoRLRuntimeReceipt | None = None

    @field_validator("image_reference")
    @classmethod
    def immutable_image(cls, value: str) -> str:
        if _IMAGE_REFERENCE.fullmatch(value) is None:
            raise ValueError("NeMo RL image must be pinned by sha256 digest")
        return value

    @field_validator("remote_model_path")
    @classmethod
    def remote_model_directory(cls, value: str) -> str:
        return _safe_remote_path(value)

    @field_validator("model_id")
    @classmethod
    def safe_model_id(cls, value: str) -> str:
        return _validate_model_id(value)

    @field_validator("entrypoint_path")
    @classmethod
    def safe_entrypoint(cls, value: str) -> str:
        return NemoRLContainerContract.safe_entrypoint(value)

    @field_validator("recipe_path")
    @classmethod
    def safe_recipe(cls, value: str) -> str:
        return _safe_container_path(value, suffixes=(".yaml", ".yml"))

    @field_validator("dataset_path")
    @classmethod
    def pinned_dataset(cls, value: Path) -> Path:
        candidate = value.expanduser()
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.suffix.casefold() not in {".json", ".jsonl"}
        ):
            raise ValueError("NeMo RL dataset must be a regular JSON/JSONL file")
        return candidate.resolve()

    @field_validator("stage_bindings")
    @classmethod
    def canonical_bindings(
        cls, value: tuple[NemoRLStageBinding, ...]
    ) -> tuple[NemoRLStageBinding, ...]:
        stages = tuple(binding.stage.value for binding in value)
        expected = (StageKind.FULL_TRAINING.value, StageKind.SMOKE_TRAINING.value)
        if stages != expected:
            raise ValueError("NeMo RL profile requires sorted full and smoke bindings")
        return value

    @field_validator("overrides")
    @classmethod
    def safe_overrides(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_overrides(value)

    @model_validator(mode="after")
    def verify_material_and_digest(self) -> ApprovedNemoRLProfile:
        match = _IMAGE_REFERENCE.fullmatch(self.image_reference)
        assert match is not None
        if match.group("digest") != self.image_digest:
            raise ValueError("NeMo RL image reference and digest disagree")
        if sha256_file(self.dataset_path) != self.dataset_sha256:
            raise ValueError("NeMo RL dataset digest mismatch")
        if self.runtime_receipt is not None:
            receipt = self.runtime_receipt
            if (
                receipt.compute_profile_id != self.compute_profile_id
                or receipt.platform != self.platform
                or receipt.image_digest != self.image_digest
                or receipt.source_revision != self.source_revision
                or receipt.recipe_sha256 != self.recipe_sha256
                or receipt.model_revision != self.model_revision
            ):
                raise ValueError("NeMo RL runtime receipt does not match the profile")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"profile_digest"}))
        if self.profile_digest and self.profile_digest != expected:
            raise ValueError("NeMo RL profile digest mismatch")
        if not self.profile_digest:
            object.__setattr__(self, "profile_digest", expected)
        return self

    def stage_binding(self, stage: StageKind) -> NemoRLStageBinding:
        for binding in self.stage_bindings:
            if binding.stage == stage:
                return binding
        raise KeyError(stage.value)

    def container_contract(self, stage: StageKind) -> NemoRLContainerContract:
        binding = self.stage_binding(stage)
        return NemoRLContainerContract(
            release=self.release,
            source_revision=self.source_revision,
            image_reference=self.image_reference,
            image_digest=self.image_digest,
            platform=self.platform,
            model_id=self.model_id,
            model_revision=self.model_revision,
            remote_model_path=self.remote_model_path,
            model_support_level=self.model_support_level,
            entrypoint_path=self.entrypoint_path,
            recipe_path=self.recipe_path,
            recipe_sha256=self.recipe_sha256,
            dataset_file=self.dataset_path.name,
            dataset_sha256=self.dataset_sha256,
            verifier_id=self.verifier_id,
            verifier_digest=self.verifier_digest,
            stage=binding.stage,
            mode=binding.mode,
            max_steps=binding.max_steps,
            learning_rate=binding.learning_rate,
            gpu_count=self.gpu_count,
            shared_memory_gib=self.shared_memory_gib,
            overrides=self.overrides,
        )


class NemoRLInstallationReceipt(FrozenContractModel):
    schema_version: Literal["nemo_rl_installation_receipt.v1"] = (
        "nemo_rl_installation_receipt.v1"
    )
    worker_config_path: str
    executor_profile_id: Identifier
    nemo_profile_id: Identifier
    nemo_profile_digest: HexDigest
    runtime_receipt: NemoRLRuntimeReceipt
    replaced: bool


__all__ = [
    "ApprovedNemoRLProfile",
    "NemoRLContainerContract",
    "NemoRLExecutionMode",
    "NemoRLInstallationReceipt",
    "NemoRLModelSupportLevel",
    "NemoRLRuntimeReceipt",
    "NemoRLStageBinding",
    "sha256_file",
]
