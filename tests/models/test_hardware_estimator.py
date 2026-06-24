"""Tests for the hardware feasibility estimator.

Pure VRAM-budget math: given a model's parameter count and a training/inference
regime, estimate the memory needed and invert it to "largest model that fits a
given budget". Mirrors what HuggingFace `accelerate estimate-memory` computes,
but dependency-free so it stays hardware-agnostic and offline-testable. Numbers
are conservative planning estimates; exact values come from accelerate.
"""

import pytest

from bashgym.models.hardware_estimator import (
    dtype_bytes,
    estimate_vram_gb,
    guess_params_billions_from_id,
    max_params_billions,
    model_fits,
    recommend_for_budget,
)


def test_dtype_bytes_known_and_unknown():
    assert dtype_bytes("float16") == 2
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("float32") == 4
    assert dtype_bytes("int4") == 0.5
    with pytest.raises(ValueError, match="dtype"):
        dtype_bytes("fp6")


def test_inference_vram_includes_overhead():
    # 7B fp16 weights = 14 GB, * 1.2 serving overhead = 16.8 GB
    assert estimate_vram_gb(7, regime="inference", dtype="float16") == pytest.approx(16.8)


def test_full_finetune_uses_adam_16_bytes_per_param():
    # weights + grads + Adam states + fp32 master ~= 16 bytes/param
    assert estimate_vram_gb(7, regime="full_finetune") == pytest.approx(112.0)


def test_qlora_fits_a_small_budget():
    assert estimate_vram_gb(7, regime="qlora") == pytest.approx(7.0)
    assert model_fits(7, vram_gb=12, regime="qlora") is True


def test_lora_estimate_between_qlora_and_full():
    lora = estimate_vram_gb(7, regime="lora")
    assert lora == pytest.approx(21.0)
    assert estimate_vram_gb(7, regime="qlora") < lora < estimate_vram_gb(7, regime="full_finetune")


def test_large_full_finetune_does_not_fit_consumer_card():
    assert model_fits(70, vram_gb=24, regime="full_finetune") is False


def test_max_params_inverts_each_regime():
    assert max_params_billions(112, regime="full_finetune") == pytest.approx(7.0)
    assert max_params_billions(24, regime="qlora") == pytest.approx(24.0)
    assert max_params_billions(16.8, regime="inference", dtype="float16") == pytest.approx(7.0)


def test_invalid_regime_raises():
    with pytest.raises(ValueError, match="regime"):
        estimate_vram_gb(7, regime="teleport")


def test_guess_params_billions_from_id():
    assert guess_params_billions_from_id("Qwen/Qwen3.5-4B") == pytest.approx(4.0)
    assert guess_params_billions_from_id("google/gemma-4-E2B-it") == pytest.approx(2.0)
    assert guess_params_billions_from_id("Qwen/Qwen3.5-35B-A3B") == pytest.approx(35.0)
    assert guess_params_billions_from_id("meta-llama/Llama-3.1-8B-Instruct") == pytest.approx(8.0)
    assert guess_params_billions_from_id("org/model-1.5B-Instruct") == pytest.approx(1.5)
    assert guess_params_billions_from_id("org/model-700M") == pytest.approx(0.7)
    # no parseable size token (the "4" in phi-4 is a version, not a size)
    assert guess_params_billions_from_id("microsoft/phi-4") is None


def test_recommend_for_budget_reports_per_regime_capacity():
    report = recommend_for_budget(128.0)

    caps = report["regime_capacities"]
    assert caps["full_finetune"] == pytest.approx(8.0)  # 128 / 16
    assert caps["qlora"] == pytest.approx(128.0)  # 128 / 1
    assert report["vram_gb"] == 128.0


def test_recommend_for_budget_annotates_candidate_fit():
    report = recommend_for_budget(
        12.0,
        candidates=[
            {"id": "Qwen/Qwen3.5-4B"},  # params guessed from id
            {"id": "big/Model-70B", "params_billions": 70.0},
        ],
    )

    by_id = {entry["id"]: entry for entry in report["runnable"]}
    small = by_id["Qwen/Qwen3.5-4B"]
    assert small["params_billions"] == pytest.approx(4.0)
    assert small["can_qlora"] is True
    assert small["can_full"] is False

    big = by_id["big/Model-70B"]
    assert big["can_qlora"] is False  # 70B needs ~70 GB for QLoRA, budget is 12
    assert big["can_infer"] is False


def test_recommend_for_budget_skips_candidates_without_a_known_size():
    report = recommend_for_budget(24.0, candidates=[{"id": "microsoft/phi-4"}])

    assert report["runnable"][0]["params_billions"] is None
    # unknown size -> no fit claims, just surfaced for manual sizing
    assert report["runnable"][0]["can_qlora"] is None
