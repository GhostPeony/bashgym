"""Installation helpers for the optional registered-compute NeMo RL backend."""

from __future__ import annotations

from pathlib import Path

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.nemo_rl import ApprovedNemoRLProfile, sha256_file
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
)

_RUNNER_PATH = Path(__file__).with_name("nemo_rl_runner.py").resolve()
_OUTPUT_PATHS = (
    "effective_config.json",
    "final",
    "logs",
    "training_manifest.json",
    "training_metrics.jsonl",
)
_GYM_OUTPUT_PATHS = tuple(
    sorted(
        (
            *_OUTPUT_PATHS,
            "nemo_gym_bundle_manifest.json",
            "nemo_gym_environment_contract.json",
        )
    )
)


def _runner_binding(
    configured: PinnedRemoteStageProfile,
    *,
    nemo_profile: ApprovedNemoRLProfile,
) -> ApprovedCodeLineageExecutionBinding | None:
    previous = configured.code_lineage_binding
    if previous is None:
        return None
    return ApprovedCodeLineageExecutionBinding(
        binding_id=f"{nemo_profile.profile_id}-{configured.stage.value}-runner",
        binding_revision=nemo_profile.profile_revision,
        source_repository_profile_id=previous.source_repository_profile_id,
        entrypoint_path="bashgym/campaigns/nemo_rl_runner.py",
        working_directory="run",
        max_archive_bytes=previous.max_archive_bytes,
    )


def bind_nemo_rl_profile(
    executor: ApprovedRemoteExecutorProfile,
    nemo_profile: ApprovedNemoRLProfile,
    *,
    replace: bool,
    allow_training_stage_replacement: bool = False,
) -> ApprovedRemoteExecutorProfile:
    """Return a revised executor whose training stages invoke the typed wrapper."""

    if executor.nemo_rl is not None and not replace:
        raise ValueError("NeMo RL is already configured; pass replace explicitly")
    if executor.nemo_rl is None and not allow_training_stage_replacement:
        raise ValueError(
            "initial NeMo RL setup replaces smoke/full training stages; "
            "use a dedicated executor profile and pass "
            "allow_training_stage_replacement explicitly"
        )
    if (
        nemo_profile.compute_profile_id != executor.compute_profile_id
        or nemo_profile.target_contract_key != executor.target_contract_key
        or nemo_profile.target_model_digest != executor.target_model_digest
    ):
        raise ValueError("NeMo RL profile does not match the selected executor")

    stages: list[PinnedRemoteStageProfile] = []
    for configured in executor.stages:
        if configured.stage not in {
            StageKind.SMOKE_TRAINING,
            StageKind.FULL_TRAINING,
        }:
            stages.append(configured)
            continue
        contract = nemo_profile.container_contract(configured.stage)
        input_files = [nemo_profile.dataset_path]
        input_sha256 = {
            nemo_profile.dataset_path.name: nemo_profile.dataset_sha256,
        }
        if nemo_profile.nemo_gym is not None:
            input_files.append(nemo_profile.nemo_gym.bundle_archive_path)
            input_sha256[nemo_profile.nemo_gym.bundle_archive_path.name] = (
                nemo_profile.nemo_gym.bundle_archive_sha256
            )
        stages.append(
            PinnedRemoteStageProfile(
                stage=configured.stage,
                script_path=_RUNNER_PATH,
                script_sha256=sha256_file(_RUNNER_PATH),
                input_files=tuple(input_files),
                input_sha256=input_sha256,
                script_args=("--contract-json", contract.model_dump_json()),
                output_paths=(
                    _GYM_OUTPUT_PATHS if nemo_profile.nemo_gym is not None else _OUTPUT_PATHS
                ),
                capacity_policy=configured.capacity_policy,
                budget_unit=configured.budget_unit,
                budget_reservation=configured.budget_reservation,
                python_executable=configured.python_executable,
                code_lineage_binding=_runner_binding(
                    configured,
                    nemo_profile=nemo_profile,
                ),
            )
        )

    return ApprovedRemoteExecutorProfile(
        profile_id=executor.profile_id,
        profile_revision=executor.profile_revision + 1,
        compute_profile_id=executor.compute_profile_id,
        target_contract_key=executor.target_contract_key,
        target_model_digest=executor.target_model_digest,
        host=executor.host,
        username=executor.username,
        port=executor.port,
        key_path=executor.key_path,
        remote_work_dir=executor.remote_work_dir,
        stages=tuple(stages),
        nemo_rl=nemo_profile,
    )


__all__ = ["bind_nemo_rl_profile"]
