"""Storage-policy coverage for generated SFT, DPO, and GRPO scripts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bashgym.api.schemas import TrainingRequest
from bashgym.gym.trainer import GRPOTrainer, Trainer, TrainerConfig, TrainingRun, TrainingStrategy


def _run(strategy: TrainingStrategy) -> TrainingRun:
    return TrainingRun(
        run_id=f"test-{strategy.value}",
        strategy=strategy,
        base_model="google/gemma-4-12b-it",
        dataset_path=Path("train.jsonl"),
        output_path=Path("out"),
    )


@pytest.mark.parametrize("policy", ["adapter_only", "adapter_checkpoint", "deployable", "full_run"])
@pytest.mark.parametrize("backend", ["plain", "unsloth"])
def test_sft_dpo_and_grpo_scripts_apply_storage_policy(policy: str, backend: str) -> None:
    config = TrainerConfig(
        base_model="google/gemma-4-12b-it",
        artifact_retention=policy,
        checkpoint_limit=2,
        auto_export_gguf=True,
        sft_backend=backend,
        dpo_backend=backend,
        grpo_backend=backend,
    )

    scripts = [
        Trainer(config)._generate_sft_script(_run(TrainingStrategy.SFT)),
        Trainer(config)._generate_dpo_script(_run(TrainingStrategy.DPO)),
        GRPOTrainer(config)._generate_grpo_script(_run(TrainingStrategy.GRPO)),
    ]

    for script in scripts:
        ast.parse(script)
        assert "save_total_limit=2" in script
        assert "checkpoint-*" in script
        if policy in {"deployable", "full_run"}:
            assert "if True:" in script
        else:
            assert "if False:" in script


def test_adapter_only_is_safe_default() -> None:
    config = TrainerConfig(base_model="google/gemma-4-12b-it")

    assert config.artifact_policy_settings() == {
        "policy": "adapter_only",
        "checkpoint_limit": 1,
        "keep_checkpoints_after_success": False,
        "save_merged": False,
        "save_gguf": False,
    }


def test_training_request_exposes_storage_and_hf_artifact_controls() -> None:
    request = TrainingRequest()

    assert request.artifact_retention.value == "adapter_only"
    assert request.checkpoint_limit == 1
    assert request.hf_upload_artifact.value == "auto"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_retention", "unknown"),
        ("checkpoint_limit", 0),
        ("checkpoint_limit", 21),
        ("hf_upload_artifact", "full_run"),
    ],
)
def test_invalid_storage_policy_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        TrainerConfig(base_model="google/gemma-4-12b-it", **{field: value})
