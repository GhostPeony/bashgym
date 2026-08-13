from __future__ import annotations

from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns import nemo_rl_runner
from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.metrics import parse_metric_lines
from bashgym.campaigns.nemo_rl import (
    ApprovedNemoGymBinding,
    ApprovedNemoRLProfile,
    NemoRLContainerContract,
    NemoRLExecutionMode,
    NemoRLModelSupportLevel,
    NemoRLStageBinding,
    sha256_file,
)
from bashgym.campaigns.nemo_rl_installation import bind_nemo_rl_profile
from bashgym.campaigns.nemo_rl_runner import (
    DatasetMount,
    ModelMount,
    docker_argv,
    resolve_dataset_mount,
)
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
    RegisteredRemoteEvaluationDatasetSource,
    RegisteredRemoteModelSource,
    RemoteCommandResult,
    RemoteTrainingAdapter,
)
from bashgym.campaigns.tmax_recipe import TMaxCompositeTrainingRecipe
from bashgym.environments.nemo_gym import (
    create_nemo_gym_bundle_archive,
    export_star_count_nemo_gym_bundle,
)
from bashgym.environments.star_count import generate_star_count_dataset
from bashgym.gym.remote_trainer import SSHConfig

SOURCE_REVISION = "a" * 40
MODEL_REVISION = "b" * 40
IMAGE_DIGEST = "c" * 64
RECIPE_DIGEST = "d" * 64
VERIFIER_DIGEST = "e" * 64
TARGET_DIGEST = "f" * 64


def _nemo_profile(
    dataset: Path, *, nemo_gym: ApprovedNemoGymBinding | None = None
) -> ApprovedNemoRLProfile:
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
        entrypoint_path=(
            "examples/nemo_gym/run_grpo_nemo_gym.py"
            if nemo_gym is not None
            else "examples/run_grpo.py"
        ),
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
        nemo_gym=nemo_gym,
    )


def _nemo_gym_binding(tmp_path: Path) -> ApprovedNemoGymBinding:
    dataset = tmp_path / "star-count"
    generate_star_count_dataset(
        dataset,
        train_size=2,
        validation_size=1,
        heldout_size=1,
        seed=17,
    )
    bundle = tmp_path / "nemo-gym-bundle"
    manifest = export_star_count_nemo_gym_bundle(
        dataset,
        bundle,
        nemo_gym_revision="9" * 40,
        bashgym_revision="8" * 40,
        dataset_license="MIT",
    )
    archive = tmp_path / "nemo-gym-bundle.zip"
    receipt = create_nemo_gym_bundle_archive(bundle, archive)
    return ApprovedNemoGymBinding(
        bundle_archive_path=archive,
        bundle_archive_sha256=receipt["archive_sha256"],
        bundle_digest=manifest["bundle_digest"],
        nemo_gym_source_revision=manifest["nemo_gym_source_revision"],
        environment_id=manifest["environment_id"],
        environment_digest=manifest["environment_digest"],
        resources_server_id=manifest["resources_server_id"],
        resources_config_path=(
            "resources_servers/bashgym_star_count/configs/bashgym_star_count.yaml"
        ),
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


def test_profile_can_record_a_proven_unsupported_model(dataset: Path):
    payload = _nemo_profile(dataset).model_dump(exclude={"profile_digest"})
    payload["model_support_level"] = "unsupported"
    profile = ApprovedNemoRLProfile.model_validate(payload)
    assert profile.model_support_level is NemoRLModelSupportLevel.UNSUPPORTED


def test_contract_rejects_mutable_image_and_controller_override(dataset: Path):
    payload = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING).model_dump()
    payload["image_reference"] = "registry.example/nemo-rl:latest"
    with pytest.raises(ValidationError):
        NemoRLContainerContract.model_validate(payload)

    payload = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING).model_dump()
    payload["overrides"] = ("+data.train.data_path=/bashgym/run/data.jsonl",)
    assert NemoRLContainerContract.model_validate(payload).overrides == payload["overrides"]

    payload = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING).model_dump()
    payload["overrides"] = ("grpo.max_num_steps=100",)
    with pytest.raises(ValidationError):
        NemoRLContainerContract.model_validate(payload)

    for key in (
        "grpo.num_generations_per_prompt",
        "policy.generation.temperature",
        "grpo.seed",
    ):
        payload = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING).model_dump()
        payload["overrides"] = (f"{key}=1",)
        with pytest.raises(ValidationError, match="controller-owned"):
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
    assert argv[argv.index("uv") : argv.index("uv") + 3] == ("uv", "run", "--no-sync")
    assert all(";" not in value for value in argv)


def test_tmax_recipe_drives_nemo_grpo_argv(dataset: Path, tmp_path: Path):
    contract = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING)
    recipe = TMaxCompositeTrainingRecipe(
        algorithm="grpo",
        sft_enabled=False,
        learning_rate=3e-6,
        max_steps=17,
        group_size=6,
        temperature=0.4,
        seed=9,
    )

    argv = docker_argv(
        contract,
        run_directory=tmp_path,
        model_mount=ModelMount(
            host_directory=tmp_path / "model",
            container_path="/bashgym/model-repo",
        ),
        container_name="bashgym-nemo-tmax-recipe",
        experiment_recipe=recipe,
    )

    assert "policy.optimizer.kwargs.lr=3e-06" in argv
    assert "grpo.max_num_steps=17" in argv
    assert "grpo.num_generations_per_prompt=6" in argv
    assert "policy.generation.temperature=0.4" in argv
    assert "grpo.seed=9" in argv


def test_nemo_runner_rejects_unimplemented_sft_composition(dataset: Path, tmp_path: Path):
    del dataset, tmp_path
    with pytest.raises(ValidationError, match="False"):
        TMaxCompositeTrainingRecipe(sft_enabled=True)


def test_nemo_metrics_use_campaign_wide_stream_shape():
    stream = StringIO()
    nemo_rl_runner._append_metric(stream, name="loss", value=1.25, step=3)
    nemo_rl_runner._append_metric(stream, name="reward", value=0.75, step=3)

    points = parse_metric_lines(tuple(stream.getvalue().splitlines()))

    assert [point.step for point in points] == [3, 3]
    assert [point.values for point in points] == [{"loss": 1.25}, {"reward": 0.75}]


def test_worker_owned_resident_model_and_dataset_paths_reach_runner(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING)
    model_dir = tmp_path / "registered-model"
    dataset_dir = tmp_path / "generated-dataset"
    observed = {}

    def fake_run_contract(
        received: NemoRLContainerContract,
        run_directory: Path,
        *,
        model_directory: Path | None = None,
        dataset_directory: Path | None = None,
        experiment_recipe: TMaxCompositeTrainingRecipe | None = None,
    ) -> int:
        observed.update(
            contract=received,
            run_directory=run_directory,
            model_directory=model_directory,
            dataset_directory=dataset_directory,
            experiment_recipe=experiment_recipe,
        )
        return 17

    monkeypatch.setattr(nemo_rl_runner, "run_contract", fake_run_contract)

    result = nemo_rl_runner.main(
        (
            "--model-dir",
            str(model_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--contract-json",
            contract.model_dump_json(),
        )
    )

    assert result == 17
    assert observed == {
        "contract": contract,
        "run_directory": Path.cwd(),
        "model_directory": model_dir,
        "dataset_directory": dataset_dir,
        "experiment_recipe": None,
    }


def test_worker_tmax_recipe_arguments_reach_nemo_runner(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING)
    observed = {}

    def fake_run_contract(
        received: NemoRLContainerContract,
        run_directory: Path,
        *,
        model_directory: Path | None = None,
        dataset_directory: Path | None = None,
        experiment_recipe: TMaxCompositeTrainingRecipe | None = None,
    ) -> int:
        observed.update(
            contract=received,
            run_directory=run_directory,
            experiment_recipe=experiment_recipe,
        )
        return 19

    monkeypatch.setattr(nemo_rl_runner, "run_contract", fake_run_contract)
    recipe = TMaxCompositeTrainingRecipe(
        algorithm="grpo",
        learning_rate=4e-6,
        max_steps=23,
        group_size=4,
        temperature=0.6,
        seed=11,
    )

    result = nemo_rl_runner.main(
        ("--contract-json", contract.model_dump_json(), *recipe.script_args())
    )

    assert result == 19
    assert observed["contract"] == contract
    assert observed["run_directory"] == Path.cwd()
    assert observed["experiment_recipe"] == recipe


def test_resident_dataset_mount_uses_generated_train_rows_in_place(dataset: Path, tmp_path: Path):
    contract = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING)
    resident = tmp_path / "generated-dataset"
    resident.mkdir()
    train = resident / "train.jsonl"
    validation = resident / "validation.jsonl"
    train.write_text('{"prompt":"generated"}\n', encoding="utf-8")
    validation.write_text('{"prompt":"validation"}\n', encoding="utf-8")

    mount = resolve_dataset_mount(
        contract,
        run_directory=tmp_path / "attempt",
        dataset_directory=resident,
    )

    assert mount.host_directory == resident.resolve()
    assert mount.container_train_path == "/bashgym/dataset/train.jsonl"
    assert mount.container_validation_path == "/bashgym/dataset/validation.jsonl"
    assert mount.sha256 == sha256_file(train)
    assert not (tmp_path / "attempt" / "train.jsonl").exists()


def test_docker_wrapper_mounts_resident_dataset_read_only(dataset: Path, tmp_path: Path):
    contract = _nemo_profile(dataset).container_contract(StageKind.FULL_TRAINING)
    resident = (tmp_path / "generated-dataset").resolve()
    mount = DatasetMount(
        host_directory=resident,
        container_train_path="/bashgym/dataset/train.jsonl",
        container_validation_path="/bashgym/dataset/validation.jsonl",
        sha256="1" * 64,
    )

    argv = docker_argv(
        contract,
        run_directory=tmp_path,
        model_mount=ModelMount(
            host_directory=tmp_path / "model",
            container_path="/bashgym/model-repo",
        ),
        dataset_mount=mount,
        container_name="bashgym-nemo-resident-dataset",
    )

    assert f"type=bind,src={resident},dst=/bashgym/dataset,readonly" in argv
    assert "data.train.data_path=/bashgym/dataset/train.jsonl" in argv
    assert "data.validation.data_path=/bashgym/dataset/validation.jsonl" in argv


def test_resident_training_paths_are_rejected_for_nemo_gym(dataset: Path, tmp_path: Path):
    contract = _nemo_profile(dataset, nemo_gym=_nemo_gym_binding(tmp_path)).container_contract(
        StageKind.FULL_TRAINING
    )
    resident = tmp_path / "generated"
    resident.mkdir()

    with pytest.raises(RuntimeError, match="nemo_gym_resident_dataset_override_unsupported"):
        resolve_dataset_mount(
            contract,
            run_directory=tmp_path / "attempt",
            dataset_directory=resident,
        )


def test_nemo_gym_profile_uses_exact_entrypoint_bundle_and_rollout_settings(
    dataset: Path, tmp_path: Path
):
    gym = _nemo_gym_binding(tmp_path)
    profile = _nemo_profile(dataset, nemo_gym=gym)
    contract = profile.container_contract(StageKind.SMOKE_TRAINING)
    argv = docker_argv(
        contract,
        run_directory=tmp_path,
        model_mount=ModelMount(
            host_directory=tmp_path / "model",
            container_path="/bashgym/model-repo/snapshots/revision",
        ),
        container_name="bashgym-nemo-gym-test",
    )

    assert contract.entrypoint_path == "examples/nemo_gym/run_grpo_nemo_gym.py"
    assert "env.should_use_nemo_gym=true" in argv
    assert "env.nemo_gym.is_trajectory_collection=true" in argv
    assert "policy.generation.vllm_cfg.async_engine=true" in argv
    assert "policy.generation.vllm_cfg.expose_http_server=true" in argv
    assert any("vllm_model_for_training.yaml" in item for item in argv)
    assert any(
        "resources_servers/bashgym_star_count" in item and item.endswith(",readonly")
        for item in argv
    )

    revised = bind_nemo_rl_profile(
        _executor(tmp_path, dataset),
        profile,
        replace=False,
        allow_training_stage_replacement=True,
    )
    for stage in revised.stages:
        assert gym.bundle_archive_path in stage.input_files
        assert "logs" in stage.output_paths
        assert "nemo_gym_bundle_manifest.json" in stage.output_paths
        assert "nemo_gym_environment_contract.json" in stage.output_paths


def test_nemo_gym_binding_rejects_generic_grpo_entrypoint(dataset: Path, tmp_path: Path):
    payload = _nemo_profile(dataset, nemo_gym=_nemo_gym_binding(tmp_path)).model_dump(
        exclude={"profile_digest"}
    )
    payload["entrypoint_path"] = "examples/run_grpo.py"

    with pytest.raises(ValidationError, match="exact Gym GRPO entrypoint"):
        ApprovedNemoRLProfile.model_validate(payload)


def test_binding_reuses_registered_executor_lifecycle(dataset: Path, tmp_path: Path):
    executor = _executor(tmp_path, dataset)
    nemo = _nemo_profile(dataset)
    revised = bind_nemo_rl_profile(
        executor,
        nemo,
        replace=False,
        allow_training_stage_replacement=True,
    )

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


def test_binding_preserves_registered_base_model(dataset: Path, tmp_path: Path):
    executor = _executor(tmp_path, dataset)
    evaluator_payload = executor.stages[0].model_dump(mode="python")
    evaluator_payload["stage"] = StageKind.DEVELOPMENT_EVALUATION
    evaluator = PinnedRemoteStageProfile.model_validate(evaluator_payload)
    base_model = RegisteredRemoteModelSource(
        source_id="modern-open-model-v1",
        compute_profile_id=executor.compute_profile_id,
        target_contract_key=executor.target_contract_key,
        model_digest=executor.target_model_digest,
        remote_model_path="/srv/bashgym/models/modern-open-model-v1",
    )
    heldout = RegisteredRemoteEvaluationDatasetSource(
        source_id="terminal-heldout-v1",
        compute_profile_id=executor.compute_profile_id,
        dataset_version_id="terminal-heldout-v1",
        content_digest="1" * 64,
        remote_dataset_path="/srv/bashgym/datasets/terminal-heldout-v1.jsonl",
    )
    executor_payload = executor.model_dump(mode="python", exclude={"profile_digest"})
    executor_payload["stages"] = (evaluator, *executor.stages)
    executor_payload["registered_base_model"] = base_model
    executor_payload["registered_evaluation_dataset"] = heldout
    executor = ApprovedRemoteExecutorProfile.model_validate(executor_payload)

    revised = bind_nemo_rl_profile(
        executor,
        _nemo_profile(dataset),
        replace=False,
        allow_training_stage_replacement=True,
    )

    assert revised.registered_base_model == base_model
    assert revised.registered_evaluation_dataset == heldout
    assert revised.stage_profile(StageKind.DEVELOPMENT_EVALUATION) == evaluator


def test_initial_binding_refuses_to_replace_existing_training_stages(
    dataset: Path,
    tmp_path: Path,
):
    executor = _executor(tmp_path, dataset)

    with pytest.raises(ValueError, match="dedicated executor profile"):
        bind_nemo_rl_profile(executor, _nemo_profile(dataset), replace=False)


@pytest.mark.asyncio
async def test_remote_preflight_returns_secret_free_runtime_receipt(dataset: Path, tmp_path: Path):
    profile = _nemo_profile(dataset, nemo_gym=_nemo_gym_binding(tmp_path))

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
            elif "Gym-workspace/Gym" in command:
                output = "9" * 40
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
    assert receipt.nemo_gym_source_ready is True
    assert receipt.nemo_gym_source_revision == "9" * 40
    assert receipt.gpu_count == 1
    assert "private.invalid" not in receipt.model_dump_json()
