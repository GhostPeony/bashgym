from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.nemo_rl import (
    ApprovedNemoRLProfile,
    NemoRLContainerContract,
    NemoRLExecutionMode,
    NemoRLModelSupportLevel,
    NemoRLStageBinding,
    sha256_file,
)
from bashgym.campaigns.nemo_rl_installation import bind_nemo_rl_profile
from bashgym.campaigns.nemo_rl_runner import ModelMount, docker_argv
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
    RemoteCommandResult,
    RemoteTrainingAdapter,
)
from bashgym.gym.remote_trainer import SSHConfig

SOURCE_REVISION = "a" * 40
MODEL_REVISION = "b" * 40
IMAGE_DIGEST = "c" * 64
RECIPE_DIGEST = "d" * 64
VERIFIER_DIGEST = "e" * 64
TARGET_DIGEST = "f" * 64


def _nemo_profile(dataset: Path) -> ApprovedNemoRLProfile:
    return ApprovedNemoRLProfile(
        profile_id="nemo-test-v1",
        profile_revision=1,
        compute_profile_id="private-compute-v1",
        target_contract_key="modern-open-model-v1",
        target_model_digest=TARGET_DIGEST,
        release="v0.6.0",
        source_revision=SOURCE_REVISION,
        image_reference=f"registry.example/nemo-rl@sha256:{IMAGE_DIGEST}",
        image_digest=IMAGE_DIGEST,
        platform="linux/arm64",
        model_id="example/modern-open-model",
        model_revision=MODEL_REVISION,
        remote_model_path=f"~/models/snapshots/{MODEL_REVISION}",
        model_support_level=NemoRLModelSupportLevel.BROAD_API_COMPATIBLE,
        entrypoint_path="examples/run_grpo.py",
        recipe_path="/opt/nemo-rl/examples/configs/grpo.yaml",
        recipe_sha256=RECIPE_DIGEST,
        dataset_path=dataset,
        dataset_sha256=sha256_file(dataset),
        verifier_id="exact-answer-v1",
        verifier_digest=VERIFIER_DIGEST,
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
        overrides=("cluster.gpus_per_node=1",),
    )


def _executor(tmp_path: Path, dataset: Path) -> ApprovedRemoteExecutorProfile:
    script = tmp_path / "old_runner.py"
    script.write_text("print('old')\n", encoding="utf-8")
    key = tmp_path / "id_ed25519"
    key.write_text("test-only", encoding="utf-8")
    binding = ApprovedCodeLineageExecutionBinding(
        binding_id="old-binding-v1",
        binding_revision=1,
        source_repository_profile_id="source-v1",
        entrypoint_path="old_runner.py",
    )
    stages = tuple(
        PinnedRemoteStageProfile(
            stage=stage,
            script_path=script,
            script_sha256=sha256_file(script),
            input_files=(dataset,),
            input_sha256={dataset.name: sha256_file(dataset)},
            budget_reservation=0.25,
            code_lineage_binding=binding,
        )
        for stage in (StageKind.FULL_TRAINING, StageKind.SMOKE_TRAINING)
    )
    return ApprovedRemoteExecutorProfile(
        profile_id="private-executor-v1",
        profile_revision=1,
        compute_profile_id="private-compute-v1",
        target_contract_key="modern-open-model-v1",
        target_model_digest=TARGET_DIGEST,
        host="private-compute.invalid",
        username="operator",
        key_path=str(key),
        stages=stages,
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "deterministic.jsonl"
    path.write_text('{"prompt":"1+1","answer":"2"}\n', encoding="utf-8")
    return path


def test_contract_enforces_no_update_and_ten_step_ceiling(dataset: Path):
    profile = _nemo_profile(dataset)
    smoke = profile.container_contract(StageKind.SMOKE_TRAINING)
    assert smoke.mode is NemoRLExecutionMode.NO_UPDATE
    assert smoke.max_steps == 1
    assert smoke.learning_rate == 0

    payload = profile.container_contract(StageKind.FULL_TRAINING).model_dump()
    payload["max_steps"] = 11
    with pytest.raises(ValidationError):
        NemoRLContainerContract.model_validate(payload)


def test_contract_rejects_mutable_image_and_controller_override(dataset: Path):
    payload = _nemo_profile(dataset).container_contract(
        StageKind.FULL_TRAINING
    ).model_dump()
    payload["image_reference"] = "registry.example/nemo-rl:latest"
    with pytest.raises(ValidationError):
        NemoRLContainerContract.model_validate(payload)

    payload = _nemo_profile(dataset).container_contract(
        StageKind.FULL_TRAINING
    ).model_dump()
    payload["overrides"] = ("grpo.max_num_steps=100",)
    with pytest.raises(ValidationError):
        NemoRLContainerContract.model_validate(payload)


def test_docker_wrapper_uses_typed_bounded_argv(dataset: Path, tmp_path: Path):
    contract = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING)
    argv = docker_argv(
        contract,
        run_directory=tmp_path,
        model_mount=ModelMount(
            host_directory=tmp_path / "model",
            container_path="/bashgym/model-repo/snapshots/revision",
        ),
        container_name="bashgym-nemo-test",
    )
    assert argv[:3] == ("docker", "run", "--rm")
    assert "--network=none" in argv
    assert f"grpo.max_num_steps={contract.max_steps}" in argv
    assert f"policy.optimizer.kwargs.lr={contract.learning_rate}" in argv
    assert contract.image_reference in argv
    assert all(";" not in value for value in argv)


def test_binding_reuses_registered_executor_lifecycle(dataset: Path, tmp_path: Path):
    executor = _executor(tmp_path, dataset)
    nemo = _nemo_profile(dataset)
    revised = bind_nemo_rl_profile(executor, nemo, replace=False)

    assert revised.profile_revision == 2
    assert revised.nemo_rl == nemo
    for stage in revised.stages:
        assert stage.script_path.name == "nemo_rl_runner.py"
        assert stage.output_paths[0] == "effective_config.json"
        assert stage.budget_reservation == 0.25
        assert stage.code_lineage_binding is not None
        assert stage.code_lineage_binding.entrypoint_path == "bashgym/campaigns/nemo_rl_runner.py"
        parsed = NemoRLContainerContract.model_validate_json(stage.script_args[1])
        assert parsed.stage == stage.stage


@pytest.mark.asyncio
async def test_remote_preflight_returns_secret_free_runtime_receipt(dataset: Path):
    profile = _nemo_profile(dataset)

    class Session:
        async def run(self, command: str, *, timeout: float | None = None):
            del timeout
            if "docker version" in command:
                output = "28.0.0"
            elif "docker info" in command:
                output = '{"nvidia":{"path":"nvidia-container-runtime"}}'
            elif command == "uname -m":
                output = "aarch64"
            elif command == "nvidia-smi -L":
                output = "GPU 0: registered private GPU"
            elif "df -Pk ." in command:
                output = str(200 * 1024 * 1024)
            elif "df -Pk /dev/shm" in command:
                output = str(32 * 1024 * 1024)
            elif "docker image inspect" in command:
                output = profile.image_reference
            elif "rev-parse HEAD" in command:
                output = SOURCE_REVISION
            elif "sha256sum" in command:
                output = f"{RECIPE_DIGEST}  recipe.yaml"
            elif 'printf %s "$HOME"' in command:
                output = "/home/operator"
            elif "config.json" in command:
                output = ""
            else:
                raise AssertionError(command)
            return RemoteCommandResult(stdout=output, exit_status=0)

        async def upload(self, local_path: Path, remote_path: str) -> None:
            raise AssertionError((local_path, remote_path))

        async def download(self, remote_path: str, local_path: Path) -> bool:
            raise AssertionError((remote_path, local_path))

    @asynccontextmanager
    async def factory():
        yield Session()

    adapter = RemoteTrainingAdapter(
        SSHConfig(host="private.invalid", username="operator"),
        compute_profile_id=profile.compute_profile_id,
        session_factory=factory,
    )
    receipt = await adapter.nemo_rl_preflight(profile)
    assert receipt.image_ready
    assert receipt.recipe_ready
    assert receipt.model_ready
    assert receipt.gpu_count == 1
    assert "private.invalid" not in receipt.model_dump_json()
