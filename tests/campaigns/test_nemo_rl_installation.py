"""Focused installation behavior for the optional NeMo RL backend."""

from __future__ import annotations

import hashlib
from pathlib import Path

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.nemo_rl import (
    ApprovedNemoRLProfile,
    NemoRLExecutionMode,
    NemoRLModelSupportLevel,
    NemoRLStageBinding,
    sha256_file,
)
from bashgym.campaigns.nemo_rl_installation import bind_nemo_rl_profile
from bashgym.campaigns.remote import ApprovedRemoteExecutorProfile, PinnedRemoteStageProfile


def _stage(tmp_path: Path, stage: StageKind, input_file: Path) -> PinnedRemoteStageProfile:
    script = tmp_path / f"{stage.value}.py"
    script.write_text("print('configured')\n", encoding="utf-8")
    return PinnedRemoteStageProfile(
        stage=stage,
        script_path=script,
        script_sha256=sha256_file(script),
        input_files=(input_file,),
        input_sha256={input_file.name: sha256_file(input_file)},
        budget_reservation=0.25,
    )


def nemo_profile(
    dataset: Path,
    *,
    compute_profile_id: str = "private-compute-v1",
    target_contract_key: str = "modern-open-model-v1",
    target_model_digest: str = "f" * 64,
) -> ApprovedNemoRLProfile:
    return ApprovedNemoRLProfile(
        profile_id="nemo-test-v1",
        profile_revision=1,
        compute_profile_id=compute_profile_id,
        target_contract_key=target_contract_key,
        target_model_digest=target_model_digest,
        release="v0.6.0",
        source_revision="a" * 40,
        image_reference=f"registry.example/nemo-rl@sha256:{'c' * 64}",
        image_digest="c" * 64,
        platform="linux/arm64",
        model_id="example/modern-open-model",
        model_revision="b" * 40,
        remote_model_path="/srv/bashgym/models/modern-open-model-v1",
        model_support_level=NemoRLModelSupportLevel.BROAD_API_COMPATIBLE,
        recipe_path="/opt/nemo-rl/examples/configs/grpo.yaml",
        recipe_sha256="d" * 64,
        dataset_path=dataset,
        dataset_sha256=sha256_file(dataset),
        verifier_id="exact-answer-v1",
        verifier_digest="e" * 64,
        stage_bindings=(
            NemoRLStageBinding(
                stage=StageKind.FULL_TRAINING,
                mode=NemoRLExecutionMode.GRPO,
                max_steps=10,
                learning_rate=1e-6,
            ),
            NemoRLStageBinding(
                stage=StageKind.SMOKE_TRAINING,
                mode=NemoRLExecutionMode.NO_UPDATE,
                max_steps=1,
                learning_rate=0,
            ),
        ),
    )


def test_data_build_executor_keeps_small_stage_inputs_and_does_not_upload_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "controller-training-rows.jsonl"
    dataset.write_text('{"prompt":"private row"}\n', encoding="utf-8")
    installation_input = tmp_path / "installation.json"
    installation_input.write_text('{"runner":"nemo"}\n', encoding="utf-8")
    key = tmp_path / "id_ed25519"
    key.write_text("test-only", encoding="utf-8")
    stages = tuple(
        _stage(tmp_path, stage, installation_input)
        for stage in (
            StageKind.DATA_BUILD,
            StageKind.FULL_TRAINING,
            StageKind.SMOKE_TRAINING,
        )
    )
    executor = ApprovedRemoteExecutorProfile(
        profile_id="private-executor-v1",
        profile_revision=1,
        compute_profile_id="private-compute-v1",
        target_contract_key="modern-open-model-v1",
        target_model_digest="f" * 64,
        host="private-compute.invalid",
        username="operator",
        key_path=str(key),
        stages=stages,
    )

    revised = bind_nemo_rl_profile(
        executor,
        nemo_profile(dataset),
        replace=False,
        allow_training_stage_replacement=True,
    )

    for stage in (StageKind.FULL_TRAINING, StageKind.SMOKE_TRAINING):
        configured = revised.stage_profile(stage)
        assert configured.input_files == (installation_input.resolve(),)
        assert configured.input_sha256 == {
            installation_input.name: hashlib.sha256(installation_input.read_bytes()).hexdigest()
        }
        assert dataset.resolve() not in configured.input_files
