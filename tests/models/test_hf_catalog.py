"""Tests for HuggingFace-backed model catalog discovery + size parsing.

The pure parsing/normalization is tested here; the thin network adapters
(`discover_training_models`, `fetch_model_size`) call huggingface_hub and are
exercised via injected fakes so no network is required.
"""

from types import SimpleNamespace

import pytest

from bashgym.models.hf_catalog import (
    normalize_training_models,
    params_billions,
    total_and_dominant_dtype,
)


def test_total_and_dominant_dtype():
    total, dominant = total_and_dominant_dtype({"BF16": 7_000_000_000, "F32": 1_000_000})

    assert total == 7_001_000_000
    assert dominant == "BF16"


def test_total_and_dominant_dtype_empty():
    assert total_and_dominant_dtype({}) == (0, "")


def test_params_billions():
    assert params_billions(7_000_000_000) == pytest.approx(7.0)
    assert params_billions(1_500_000_000) == pytest.approx(1.5)


def test_normalize_training_models_shapes_each_entry():
    raw = [
        SimpleNamespace(
            id="Qwen/Qwen3.6-4B-Instruct",
            downloads=1000,
            likes=50,
            tags=["text-generation", "code"],
            pipeline_tag="text-generation",
        ),
    ]

    out = normalize_training_models(raw)

    assert out[0]["id"] == "Qwen/Qwen3.6-4B-Instruct"
    assert out[0]["downloads"] == 1000
    assert out[0]["likes"] == 50
    assert out[0]["hf_url"] == "https://huggingface.co/Qwen/Qwen3.6-4B-Instruct"
    # params guessed from the id so the directory can size it without a download
    assert out[0]["params_billions"] == pytest.approx(4.0)


def test_normalize_training_models_sorts_by_downloads_desc():
    raw = [
        SimpleNamespace(id="a/x", downloads=5, likes=0, tags=[], pipeline_tag="text-generation"),
        SimpleNamespace(id="b/y", downloads=99, likes=0, tags=[], pipeline_tag="text-generation"),
    ]

    out = normalize_training_models(raw)

    assert [m["id"] for m in out] == ["b/y", "a/x"]


def test_normalize_training_models_respects_limit():
    raw = [
        SimpleNamespace(id=f"o/m{i}", downloads=i, likes=0, tags=[], pipeline_tag="text-generation")
        for i in range(5)
    ]

    out = normalize_training_models(raw, limit=2)

    assert len(out) == 2
    assert out[0]["id"] == "o/m4"
