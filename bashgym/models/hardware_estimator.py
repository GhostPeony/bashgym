"""Hardware feasibility estimator: "what size model can this budget run?"

Dependency-free VRAM math so the discovery/recommendation layer is hardware
agnostic and offline-testable. The estimates follow the same accounting as
HuggingFace ``accelerate estimate-memory`` (weights + grads + Adam states +
activation/serving overhead) but are deliberately conservative *planning*
numbers; for an exact figure on a specific model, defer to accelerate.

Regimes:
- ``inference``      : serve weights in ``dtype`` + KV-cache/activation overhead
- ``qlora``          : 4-bit frozen base + LoRA grads/optimizer/activations
- ``lora``           : bf16 frozen base + LoRA grads/optimizer/activations
- ``full_finetune``  : fp16 weights + grads + Adam states + fp32 master (~16 B/param)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Bytes per parameter for a stored/served dtype.
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4.0,
    "fp32": 4.0,
    "f32": 4.0,
    "float16": 2.0,
    "fp16": 2.0,
    "f16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "8bit": 1.0,
    "int4": 0.5,
    "4bit": 0.5,
}

# Serving overhead multiplier on raw weights (KV cache + activations); HF docs
# suggest adding <=20% for inference.
INFERENCE_OVERHEAD = 1.2
# LoRA/QLoRA per-parameter training overhead in GB-per-billion (adapter grads,
# optimizer state for the small adapter, and activations).
LORA_TRAIN_OVERHEAD_GB_PER_B = 1.0
QLORA_TRAIN_OVERHEAD_GB_PER_B = 0.5
# Full fine-tune with Adam in mixed precision: 2 (fp16 weights) + 2 (grads)
# + 4 + 4 (Adam m, v) + 4 (fp32 master) = 16 bytes/param.
FULL_FINETUNE_BYTES_PER_PARAM = 16.0

REGIMES = ("inference", "qlora", "lora", "full_finetune")

_SIZE_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*([BM])(?![A-Za-z0-9])")


def dtype_bytes(dtype: str) -> float:
    """Bytes per parameter for a dtype name (case/format tolerant)."""

    key = dtype.strip().lower().replace("-", "").replace("_", "")
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(set(_DTYPE_BYTES))}")
    return _DTYPE_BYTES[key]


def estimate_vram_gb(
    params_billions: float,
    *,
    regime: str,
    dtype: str = "float16",
) -> float:
    """Estimate VRAM (GB) needed to run ``params_billions`` under ``regime``."""

    params = float(params_billions)
    if regime == "inference":
        return params * dtype_bytes(dtype) * INFERENCE_OVERHEAD
    if regime == "qlora":
        return params * (dtype_bytes("4bit") + QLORA_TRAIN_OVERHEAD_GB_PER_B)
    if regime == "lora":
        return params * (dtype_bytes("bf16") + LORA_TRAIN_OVERHEAD_GB_PER_B)
    if regime == "full_finetune":
        return params * FULL_FINETUNE_BYTES_PER_PARAM
    raise ValueError(f"unknown regime {regime!r}; known: {REGIMES}")


def max_params_billions(
    vram_gb: float,
    *,
    regime: str,
    dtype: str = "float16",
) -> float:
    """Largest model (in billions of params) that fits ``vram_gb`` under regime."""

    if regime == "inference":
        per_b = dtype_bytes(dtype) * INFERENCE_OVERHEAD
    elif regime == "qlora":
        per_b = dtype_bytes("4bit") + QLORA_TRAIN_OVERHEAD_GB_PER_B
    elif regime == "lora":
        per_b = dtype_bytes("bf16") + LORA_TRAIN_OVERHEAD_GB_PER_B
    elif regime == "full_finetune":
        per_b = FULL_FINETUNE_BYTES_PER_PARAM
    else:
        raise ValueError(f"unknown regime {regime!r}; known: {REGIMES}")
    return vram_gb / per_b


def model_fits(
    params_billions: float,
    vram_gb: float,
    *,
    regime: str,
    dtype: str = "float16",
) -> bool:
    """Whether a model fits the budget under the given regime."""

    return estimate_vram_gb(params_billions, regime=regime, dtype=dtype) <= vram_gb


def guess_params_billions_from_id(model_id: str) -> float | None:
    """Best-effort parameter count (billions) from a HuggingFace model id.

    Picks the size token (e.g. ``4B``, ``1.5B``, ``700M``) while ignoring version
    numbers that are not immediately followed by a B/M unit. Returns ``None`` when
    no size token is present (caller can fall back to safetensors metadata).
    """

    best: float | None = None
    for value, unit in _SIZE_TOKEN.findall(model_id):
        billions = float(value) / 1000.0 if unit == "M" else float(value)
        if best is None or billions > best:
            best = billions
    return best


def recommend_for_budget(
    vram_gb: float,
    *,
    candidates: Iterable[dict[str, Any]] | None = None,
    dtype: str = "float16",
) -> dict[str, Any]:
    """Recommend what a VRAM budget can run.

    Returns the largest model (billions of params) that fits each regime, plus a
    per-candidate fit annotation. ``candidates`` are dicts with an ``id`` and an
    optional ``params_billions`` (guessed from the id when absent). Candidates
    whose size is unknown carry ``None`` for every ``can_*`` flag so the caller can
    surface them for manual sizing rather than making a false claim.
    """

    regime_capacities = {
        regime: round(max_params_billions(vram_gb, regime=regime, dtype=dtype), 1)
        for regime in REGIMES
    }

    runnable: list[dict[str, Any]] = []
    for candidate in candidates or []:
        model_id = str(candidate.get("id", ""))
        params = candidate.get("params_billions")
        if params is None:
            params = guess_params_billions_from_id(model_id)

        if params is None:
            fits = {regime: None for regime in REGIMES}
        else:
            fits = {
                regime: model_fits(params, vram_gb, regime=regime, dtype=dtype)
                for regime in REGIMES
            }

        runnable.append(
            {
                "id": model_id,
                "params_billions": params,
                "can_infer": fits["inference"],
                "can_qlora": fits["qlora"],
                "can_lora": fits["lora"],
                "can_full": fits["full_finetune"],
            }
        )

    return {
        "vram_gb": vram_gb,
        "regime_capacities": regime_capacities,
        "runnable": runnable,
    }
