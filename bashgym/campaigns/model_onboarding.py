"""Fail-closed inspection plans for installation-owned trainable model artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from bashgym.campaigns.contracts import FrozenContractModel, HexDigest, Identifier, canonical_hash
from bashgym.models.artifact_capabilities import classify_model_artifact

_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_FILES = 10_000
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_INFERENCE_QUANTIZERS = frozenset(
    {"awq", "gptq", "fp8", "nvfp4", "compressed-tensors", "compressed_tensors"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class ModelBackendCandidate(FrozenContractModel):
    backend_id: Identifier
    compatibility: Literal["candidate", "requires_runtime_doctor", "incompatible"]
    reason: str = Field(min_length=1, max_length=500)


class ModelOnboardingPlan(FrozenContractModel):
    """Secret-free result of inspecting one operator-selected local artifact."""

    schema_version: Literal["model_onboarding_plan.v1"] = "model_onboarding_plan.v1"
    model_id: str = Field(min_length=3, max_length=500)
    model_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    model_ref: str = Field(min_length=1, max_length=1000)
    suggested_target_contract_key: Identifier
    task: Identifier
    architecture: str = Field(min_length=1, max_length=300)
    model_type: str = Field(min_length=1, max_length=200)
    artifact_role: Literal["trainable_base", "inference_quant", "adapter", "unsupported"]
    trainable: bool
    quantization: str | None = Field(default=None, max_length=100)
    config_sha256: HexDigest
    artifact_manifest_sha256: HexDigest
    weight_file_count: int = Field(ge=0, le=_MAX_FILES)
    backend_candidates: tuple[ModelBackendCandidate, ...]
    blockers: tuple[str, ...] = ()
    ready_for_binding: bool
    plan_digest: HexDigest = ""

    @model_validator(mode="after")
    def exact_readiness_and_digest(self) -> ModelOnboardingPlan:
        if self.ready_for_binding != (self.trainable and not self.blockers):
            raise ValueError("model onboarding readiness does not match its blockers")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"plan_digest"}))
        if self.plan_digest and self.plan_digest != expected:
            raise ValueError("model onboarding plan digest mismatch")
        if not self.plan_digest:
            object.__setattr__(self, "plan_digest", expected)
        return self


def _task_family(config: dict[str, Any]) -> tuple[str, str, str | None]:
    architectures = tuple(str(item) for item in config.get("architectures") or ())
    architecture = architectures[0] if architectures else "unknown"
    model_type = str(config.get("model_type") or "unknown")
    markers = " ".join((architecture, *architectures, model_type)).casefold()
    if any(marker in markers for marker in ("image", "vision", "vlm", "conditionalgeneration")):
        return architecture, model_type, "vision_language"
    if any(marker in markers for marker in ("embedding", "sentence", "reranker")):
        return architecture, model_type, "embedding"
    if "causallm" in markers or "forcausallm" in markers:
        return architecture, model_type, "causal_lm"
    return architecture, model_type, None


def _quantization(config: dict[str, Any]) -> tuple[str | None, bool | None]:
    raw = config.get("quantization_config")
    if not isinstance(raw, dict):
        return None, None
    method = str(raw.get("quant_method") or raw.get("quantization_method") or "").casefold()
    if raw.get("load_in_4bit") is True or raw.get("load_in_8bit") is True:
        return "bitsandbytes", True
    if method in {"bitsandbytes", "bnb", "bnb-4bit", "bnb-8bit"}:
        return "bitsandbytes", True
    if method in _INFERENCE_QUANTIZERS:
        return method, False
    if method:
        return method, False
    return "unknown", False


def _artifact_manifest(root: Path, files: tuple[Path, ...]) -> str:
    manifest = []
    for path in files:
        manifest.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return canonical_hash(manifest)


def inspect_model_artifact(
    artifact_directory: str | Path,
    *,
    model_id: str,
    model_revision: str,
) -> ModelOnboardingPlan:
    """Inspect one explicit local snapshot without scanning unrelated model caches."""

    normalized_model_id = str(model_id).strip()
    revision = str(model_revision).strip().casefold()
    if _MODEL_ID.fullmatch(normalized_model_id) is None:
        raise ValueError("model_id must be an exact organization/repository identifier")
    if _IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise ValueError("model_revision must be an immutable 40/64-character revision")
    candidate = Path(artifact_directory).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("model artifact must be one regular operator-selected directory")
    root = candidate.resolve()
    config_path = root / "config.json"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or config_path.stat().st_size > _MAX_CONFIG_BYTES
    ):
        raise ValueError("model artifact requires one bounded regular config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("model artifact config.json is invalid") from exc
    if not isinstance(config, dict):
        raise ValueError("model artifact config.json must contain an object")

    files = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file() and not path.is_symlink()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files or len(files) > _MAX_FILES:
        raise ValueError("model artifact file inventory is empty or too large")
    weight_files = tuple(path for path in files if path.suffix.casefold() in _WEIGHT_SUFFIXES)
    has_adapter = any(path.name == "adapter_config.json" for path in files)
    architecture, model_type, task = _task_family(config)
    id_capabilities = classify_model_artifact(normalized_model_id)
    quantization, quant_trainable = _quantization(config)

    blockers: list[str] = []
    artifact_role: Literal["trainable_base", "inference_quant", "adapter", "unsupported"]
    trainable = False
    if has_adapter:
        artifact_role = "adapter"
        blockers.append("artifact_is_adapter_not_trainable_base")
    elif not id_capabilities.trainable or quant_trainable is False:
        artifact_role = "inference_quant"
        blockers.append("artifact_is_inference_quant_not_trainable_base")
    elif not weight_files:
        artifact_role = "unsupported"
        blockers.append("model_weight_files_missing")
    elif task is None:
        artifact_role = "unsupported"
        blockers.append("model_architecture_task_unsupported")
    else:
        artifact_role = "trainable_base"
        trainable = True
    if task is None:
        task = "unsupported"

    backend_candidates = (
        ModelBackendCandidate(
            backend_id="bashgym_transformers",
            compatibility="candidate" if trainable else "incompatible",
            reason=(
                "standard registered-training loader must pass a bounded model-load smoke"
                if trainable
                else "artifact did not qualify as a trainable base"
            ),
        ),
        ModelBackendCandidate(
            backend_id="bashgym_unsloth",
            compatibility=(
                "candidate"
                if trainable and task in {"causal_lm", "vision_language"}
                else "incompatible"
            ),
            reason=(
                "eligible architecture; installation runtime doctor remains authoritative"
                if trainable and task in {"causal_lm", "vision_language"}
                else "backend requires a supported causal or vision-language trainable base"
            ),
        ),
        ModelBackendCandidate(
            backend_id="nemo_rl",
            compatibility="requires_runtime_doctor" if trainable else "incompatible",
            reason=(
                "optional backend requires its own pinned image/source/model compatibility smoke"
                if trainable
                else "artifact did not qualify as a trainable base"
            ),
        ),
    )
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_model_id.casefold()).strip("-")
    target_key = f"{slug[:96]}-{revision[:12]}"
    return ModelOnboardingPlan(
        model_id=normalized_model_id,
        model_revision=revision,
        model_ref=f"hf://{normalized_model_id}@{revision}",
        suggested_target_contract_key=target_key,
        task=task,
        architecture=architecture,
        model_type=model_type,
        artifact_role=artifact_role,
        trainable=trainable,
        quantization=quantization or id_capabilities.quantization,
        config_sha256=_sha256_file(config_path),
        artifact_manifest_sha256=_artifact_manifest(root, files),
        weight_file_count=len(weight_files),
        backend_candidates=backend_candidates,
        blockers=tuple(blockers),
        ready_for_binding=trainable and not blockers,
    )


__all__ = [
    "ModelBackendCandidate",
    "ModelOnboardingPlan",
    "inspect_model_artifact",
]
