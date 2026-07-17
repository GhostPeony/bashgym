"""Installation-owned model onboarding inspection tests."""

from __future__ import annotations

import json

import pytest

from bashgym.campaigns.model_onboarding import inspect_model_artifact

REVISION = "a" * 40


def _artifact(tmp_path, *, config, adapter=False):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"test-weights")
    if adapter:
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    return root


def test_onboarding_emits_secret_free_trainable_binding_plan(tmp_path):
    root = _artifact(
        tmp_path,
        config={"architectures": ["ModernForCausalLM"], "model_type": "modern"},
    )

    plan = inspect_model_artifact(
        root,
        model_id="example/modern-open-model",
        model_revision=REVISION,
    )

    assert plan.ready_for_binding is True
    assert plan.artifact_role == "trainable_base"
    assert plan.task == "causal_lm"
    assert plan.model_ref == f"hf://example/modern-open-model@{REVISION}"
    assert {item.backend_id for item in plan.backend_candidates} == {
        "bashgym_transformers",
        "bashgym_unsloth",
        "nemo_rl",
    }
    assert str(tmp_path) not in plan.model_dump_json()
    assert len(plan.artifact_manifest_sha256) == 64
    assert len(plan.plan_digest) == 64


def test_onboarding_rejects_inference_quant_and_adapter(tmp_path):
    quant = _artifact(
        tmp_path,
        config={
            "architectures": ["ModernForCausalLM"],
            "model_type": "modern",
            "quantization_config": {"quant_method": "awq"},
        },
    )
    quant_plan = inspect_model_artifact(
        quant,
        model_id="example/modern-open-model-awq",
        model_revision=REVISION,
    )
    assert quant_plan.ready_for_binding is False
    assert quant_plan.artifact_role == "inference_quant"
    assert quant_plan.blockers == ("artifact_is_inference_quant_not_trainable_base",)

    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    (adapter_root / "config.json").write_text(
        json.dumps({"architectures": ["ModernForCausalLM"], "model_type": "modern"}),
        encoding="utf-8",
    )
    (adapter_root / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_root / "adapter_model.safetensors").write_bytes(b"adapter")
    adapter_plan = inspect_model_artifact(
        adapter_root,
        model_id="example/modern-open-model-adapter",
        model_revision=REVISION,
    )
    assert adapter_plan.artifact_role == "adapter"
    assert adapter_plan.ready_for_binding is False


def test_onboarding_requires_explicit_revision_and_supported_task(tmp_path):
    root = _artifact(
        tmp_path,
        config={"architectures": ["MysteryModel"], "model_type": "mystery"},
    )
    with pytest.raises(ValueError, match="immutable"):
        inspect_model_artifact(root, model_id="example/mystery", model_revision="main")

    plan = inspect_model_artifact(
        root,
        model_id="example/mystery",
        model_revision=REVISION,
    )
    assert plan.artifact_role == "unsupported"
    assert plan.blockers == ("model_architecture_task_unsupported",)


def test_onboarding_manifest_changes_when_weight_bytes_change(tmp_path):
    root = _artifact(
        tmp_path,
        config={"architectures": ["ModernForCausalLM"], "model_type": "modern"},
    )
    first = inspect_model_artifact(
        root,
        model_id="example/modern-open-model",
        model_revision=REVISION,
    )
    (root / "model.safetensors").write_bytes(b"different-weights")
    second = inspect_model_artifact(
        root,
        model_id="example/modern-open-model",
        model_revision=REVISION,
    )
    assert first.artifact_manifest_sha256 != second.artifact_manifest_sha256
