"""HuggingFace-backed open-model discovery for the fine-tuning catalog.

The base-model directory should not be a hardcoded list that goes stale (e.g.
listing Qwen3.5 but not Qwen3.6). This module live-discovers current open models
from the Hub and reads exact parameter counts from safetensors metadata without
downloading weights.

Network calls (``discover_training_models``, ``fetch_model_size``) are thin
wrappers over ``huggingface_hub``; the parsing/normalization is pure and unit
tested by injecting fakes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bashgym.models.artifact_capabilities import classify_model_artifact
from bashgym.models.hardware_estimator import guess_params_billions_from_id

# Orgs that publish open, trainable, code/reasoning models. Examples, not a
# whitelist — `discover_training_models` accepts any authors.
DEFAULT_TRAINING_AUTHORS = (
    "Qwen",
    "google",
    "deepseek-ai",
    "meta-llama",
    "microsoft",
    "mistralai",
    "unsloth",
)

HF_MODEL_URL = "https://huggingface.co/{repo_id}"


def total_and_dominant_dtype(param_count_by_dtype: dict[str, int]) -> tuple[int, str]:
    """Total parameter count and the dtype holding the most parameters.

    ``param_count_by_dtype`` is the ``parameter_count`` map from
    ``huggingface_hub.get_safetensors_metadata`` (e.g. ``{"BF16": 7_000_000_000}``).
    """

    if not param_count_by_dtype:
        return 0, ""
    total = sum(int(count) for count in param_count_by_dtype.values())
    dominant = max(param_count_by_dtype.items(), key=lambda item: item[1])[0]
    return total, dominant


def params_billions(total_params: int) -> float:
    """Convert a raw parameter count into billions."""

    return total_params / 1_000_000_000


def normalize_training_models(
    raw_models: Iterable[Any],
    *,
    limit: int | None = None,
    trainable_only: bool = False,
) -> list[dict[str, Any]]:
    """Shape HuggingFace model listings into catalog dicts, sorted by downloads.

    ``raw_models`` is an iterable of objects with ``id``/``downloads``/``likes``/
    ``tags``/``pipeline_tag`` attributes (huggingface_hub ``ModelInfo``). Sizes are
    guessed from the id so the directory can show "fits your GPU" without a fetch.
    """

    entries: list[dict[str, Any]] = []
    for model in raw_models:
        repo_id = str(getattr(model, "id", "") or "")
        if not repo_id:
            continue
        tags = list(getattr(model, "tags", []) or [])
        capabilities = classify_model_artifact(repo_id, tags=tags)
        if trainable_only and not capabilities.trainable:
            continue
        entries.append(
            {
                "id": repo_id,
                "downloads": int(getattr(model, "downloads", 0) or 0),
                "likes": int(getattr(model, "likes", 0) or 0),
                "tags": tags,
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "params_billions": guess_params_billions_from_id(repo_id),
                "hf_url": HF_MODEL_URL.format(repo_id=repo_id),
                **capabilities.to_dict(),
            }
        )

    entries.sort(key=lambda entry: entry["downloads"], reverse=True)
    if limit is not None:
        entries = entries[:limit]
    return entries


def discover_training_models(
    *,
    authors: Iterable[str] = DEFAULT_TRAINING_AUTHORS,
    pipeline_tag: str = "text-generation",
    sort: str = "downloads",
    per_author_limit: int = 10,
    api: Any | None = None,
) -> list[dict[str, Any]]:
    """Live-discover trainable open models from the Hub (one query per author).

    Defensive by design: returns whatever could be fetched, swallowing per-author
    errors so a single failing org never blocks the catalog.
    """

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    discovered: list[Any] = []
    for author in authors:
        try:
            discovered.extend(
                api.list_models(
                    author=author,
                    pipeline_tag=pipeline_tag,
                    sort=sort,
                    direction=-1,
                    limit=per_author_limit,
                )
            )
        except Exception:  # noqa: BLE001 - one bad org must not break discovery
            continue
    return normalize_training_models(discovered, trainable_only=True)


def fetch_model_size(repo_id: str, *, api: Any | None = None) -> dict[str, Any]:
    """Read exact parameter count + dtype from safetensors metadata (no download)."""

    if api is None:
        from huggingface_hub import get_safetensors_metadata

        metadata = get_safetensors_metadata(repo_id)
    else:
        metadata = api.get_safetensors_metadata(repo_id)

    param_count_by_dtype = dict(getattr(metadata, "parameter_count", {}) or {})
    total, dominant = total_and_dominant_dtype(param_count_by_dtype)
    return {
        "id": repo_id,
        "total_params": total,
        "params_billions": params_billions(total),
        "dominant_dtype": dominant,
    }
