"""Classify model artifacts before offering them to training or serving flows.

The model family alone is not enough to decide how an artifact can be used. A
post-training inference quant such as NVFP4 belongs to the same Gemma family as
its BF16 training base, but it requires a different runtime and must not be
passed to the SFT loader as a base model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

GEMMA4_12B_TRAINING_MODEL = "unsloth/gemma-4-12b-it"
GEMMA4_12B_NVFP4_MODEL = "unsloth/gemma-4-12b-it-NVFP4"
GEMMA4_12B_NVFP4_REVISION = "4c77b7a6786b83271b3cd3285c03d7191a7ca442"


@dataclass(frozen=True)
class ArtifactCapabilities:
    """The operational role and compatible runtime for a model artifact."""

    artifact_role: str
    trainable: bool
    quantization: str | None
    runtime: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def classify_model_artifact(
    repo_id: str,
    *,
    tags: Iterable[str] = (),
) -> ArtifactCapabilities:
    """Classify common Hub artifact formats conservatively.

    Unknown artifacts remain eligible as trainable bases because the training
    loader performs the authoritative compatibility check. Known deployment
    quants fail closed so they cannot enter a training run by accident.
    """

    normalized_id = repo_id.casefold()
    normalized_tags = {str(tag).casefold() for tag in tags}

    if "nvfp4" in normalized_id or "nvfp4" in normalized_tags:
        return ArtifactCapabilities("inference_quant", False, "nvfp4", "vllm")
    if "gguf" in normalized_id or "gguf" in normalized_tags:
        return ArtifactCapabilities("inference_quant", False, "gguf", "llama.cpp")
    if any(marker in normalized_id for marker in ("-awq", "-gptq", "-fp8")):
        quantization = next(
            marker.removeprefix("-")
            for marker in ("-awq", "-gptq", "-fp8")
            if marker in normalized_id
        )
        return ArtifactCapabilities("inference_quant", False, quantization, "vllm")
    if "compressed-tensors" in normalized_tags:
        return ArtifactCapabilities("inference_quant", False, "compressed-tensors", "vllm")
    if "bnb-4bit" in normalized_id or "bitsandbytes" in normalized_tags:
        return ArtifactCapabilities("trainable_base", True, "bnb-4bit", "unsloth")
    return ArtifactCapabilities("trainable_base", True, None, "unsloth")


def require_trainable_base(repo_id: str, *, tags: Iterable[str] = ()) -> None:
    """Raise a user-facing error when an inference artifact is used for SFT."""

    capabilities = classify_model_artifact(repo_id, tags=tags)
    if capabilities.trainable:
        return
    replacement = (
        GEMMA4_12B_TRAINING_MODEL
        if repo_id.casefold() == GEMMA4_12B_NVFP4_MODEL.casefold()
        else "the matching BF16 or supported bitsandbytes training base"
    )
    raise ValueError(
        f"{repo_id} is a {capabilities.quantization} inference artifact for "
        f"{capabilities.runtime}, not a fine-tuning base. Use {replacement} for "
        "training, then deploy the resulting adapter with the inference artifact."
    )
