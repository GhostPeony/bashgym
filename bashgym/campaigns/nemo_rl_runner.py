"""Bounded host wrapper for an installation-approved NeMo RL container run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from bashgym._compat import UTC
from bashgym.campaigns.contracts import canonical_hash
from bashgym.campaigns.nemo_gym_ingestion import (
    NEMO_GYM_BUNDLE_MANIFEST_FILENAME,
    NEMO_GYM_ENVIRONMENT_CONTRACT_FILENAME,
)
from bashgym.campaigns.nemo_rl import NemoRLContainerContract, sha256_file
from bashgym.campaigns.tmax_recipe import TMaxCompositeTrainingRecipe
from bashgym.environments.nemo_gym import (
    extract_nemo_gym_bundle_archive,
    inspect_nemo_gym_bundle_archive,
)

_METRIC = re.compile(
    r"(?P<name>reward|loss|kl|entropy|grad_norm)[\s/:=]+(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_GYM_WORKSPACE = "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
_VLLM_TRAINING_CONFIG = (
    f"{_GYM_WORKSPACE}/responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
)


@dataclass(frozen=True)
class ModelMount:
    host_directory: Path
    container_path: str


@dataclass(frozen=True)
class DatasetMount:
    """One compute-resident training dataset mounted read-only in the container."""

    host_directory: Path
    container_train_path: str
    container_validation_path: str | None
    sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_checked(argv: Sequence[str], *, timeout: float = 30) -> str:
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError("nemo_rl_runtime_identity_check_failed")
    return completed.stdout.strip()


def resolve_dataset_mount(
    contract: NemoRLContainerContract,
    *,
    run_directory: Path,
    dataset_directory: Path | None = None,
) -> DatasetMount:
    """Resolve static or worker-bound training rows without copying them off compute."""

    if dataset_directory is None:
        root = run_directory.resolve()
        dataset = root / contract.dataset_file
        container_root = "/bashgym/run"
        if (
            dataset.is_symlink()
            or not dataset.is_file()
            or sha256_file(dataset) != contract.dataset_sha256
        ):
            raise RuntimeError("nemo_rl_dataset_identity_mismatch")
    else:
        if contract.nemo_gym is not None:
            raise RuntimeError("nemo_gym_resident_dataset_override_unsupported")
        if not dataset_directory.is_absolute() or dataset_directory.is_symlink():
            raise RuntimeError("nemo_rl_resident_dataset_not_ready")
        root = dataset_directory.resolve()
        if not root.is_dir():
            raise RuntimeError("nemo_rl_resident_dataset_not_ready")
        dataset = root / contract.dataset_file
        if not dataset.is_file():
            dataset = root / "train.jsonl"
        if dataset.is_symlink() or not dataset.is_file():
            raise RuntimeError("nemo_rl_resident_dataset_not_ready")
        container_root = "/bashgym/dataset"

    validation = root / "validation.jsonl"
    if validation.is_symlink():
        raise RuntimeError("nemo_rl_resident_dataset_not_ready")
    validation_path = f"{container_root}/validation.jsonl" if validation.is_file() else None
    return DatasetMount(
        host_directory=root,
        container_train_path=f"{container_root}/{dataset.name}",
        container_validation_path=validation_path,
        sha256=sha256_file(dataset),
    )


def validate_runtime_identity(
    contract: NemoRLContainerContract,
    run_directory: Path,
    *,
    model_directory: Path | None = None,
) -> ModelMount:
    model_source = model_directory or Path(contract.remote_model_path)
    if model_directory is not None and not model_directory.is_absolute():
        raise RuntimeError("nemo_rl_model_not_ready")
    model = model_source.expanduser().resolve()
    if model.is_symlink() or not model.is_dir() or not (model / "config.json").is_file():
        raise RuntimeError("nemo_rl_model_not_ready")

    image_id = _run_checked(
        ("docker", "image", "inspect", contract.image_reference, "--format", "{{.Id}}")
    )
    if not image_id.startswith("sha256:"):
        raise RuntimeError("nemo_rl_image_identity_mismatch")

    source = _run_checked(
        (
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--entrypoint",
            "git",
            contract.image_reference,
            "-C",
            "/opt/nemo-rl",
            "rev-parse",
            "HEAD",
        ),
        timeout=60,
    )
    if source != contract.source_revision:
        raise RuntimeError("nemo_rl_source_identity_mismatch")

    recipe = _run_checked(
        (
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--entrypoint",
            "sha256sum",
            contract.image_reference,
            contract.recipe_path,
        ),
        timeout=60,
    ).split(maxsplit=1)[0]
    if recipe != contract.recipe_sha256:
        raise RuntimeError("nemo_rl_recipe_identity_mismatch")
    if contract.nemo_gym is not None:
        gym = contract.nemo_gym
        archive = run_directory / gym.bundle_archive_file
        if (
            archive.is_symlink()
            or not archive.is_file()
            or sha256_file(archive) != gym.bundle_archive_sha256
        ):
            raise RuntimeError("nemo_gym_bundle_archive_identity_mismatch")
        manifest = inspect_nemo_gym_bundle_archive(archive)
        expected = {
            "bundle_digest": gym.bundle_digest,
            "nemo_gym_source_revision": gym.nemo_gym_source_revision,
            "environment_id": gym.environment_id,
            "environment_digest": gym.environment_digest,
            "resources_server_id": gym.resources_server_id,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError("nemo_gym_bundle_manifest_identity_mismatch")
        gym_source = _run_checked(
            (
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--entrypoint",
                "git",
                contract.image_reference,
                "-C",
                _GYM_WORKSPACE,
                "rev-parse",
                "HEAD",
            ),
            timeout=60,
        )
        if gym_source != gym.nemo_gym_source_revision:
            raise RuntimeError("nemo_gym_source_identity_mismatch")
    if model.parent.name == "snapshots" and model.name == contract.model_revision:
        return ModelMount(
            host_directory=model.parent.parent,
            container_path=f"/bashgym/model-repo/snapshots/{contract.model_revision}",
        )
    return ModelMount(host_directory=model, container_path="/bashgym/model-repo")


def docker_argv(
    contract: NemoRLContainerContract,
    *,
    run_directory: Path,
    model_mount: ModelMount,
    dataset_mount: DatasetMount | None = None,
    container_name: str,
    experiment_recipe: TMaxCompositeTrainingRecipe | None = None,
) -> tuple[str, ...]:
    """Return a typed argv; callers never invoke a shell."""

    if experiment_recipe is not None and experiment_recipe.sft_enabled:
        raise RuntimeError("nemo_rl_sft_composition_unsupported")
    effective_max_steps = experiment_recipe.max_steps if experiment_recipe else contract.max_steps
    effective_learning_rate = (
        experiment_recipe.learning_rate if experiment_recipe else contract.learning_rate
    )
    controller_overrides: tuple[str, ...] = (
        "checkpointing.checkpoint_dir=/bashgym/run/final",
        f"grpo.max_num_steps={effective_max_steps}",
        "logger.log_dir=/bashgym/run/logs",
        f"policy.model_name={model_mount.container_path}",
        f"policy.optimizer.kwargs.lr={effective_learning_rate}",
        f"policy.tokenizer.name={model_mount.container_path}",
    )
    if experiment_recipe is not None:
        controller_overrides += (
            f"grpo.num_generations_per_prompt={experiment_recipe.group_size}",
            f"policy.generation.temperature={experiment_recipe.temperature}",
            f"grpo.seed={experiment_recipe.seed}",
        )
    mounts = [
        "--mount",
        f"type=bind,src={run_directory},dst=/bashgym/run",
        "--mount",
        f"type=bind,src={model_mount.host_directory},dst=/bashgym/model-repo,readonly",
    ]
    if contract.nemo_gym is None and dataset_mount is not None:
        controller_overrides += (f"data.train.data_path={dataset_mount.container_train_path}",)
        if dataset_mount.container_validation_path is not None:
            controller_overrides += (
                f"data.validation.data_path={dataset_mount.container_validation_path}",
            )
        if dataset_mount.host_directory != run_directory.resolve():
            mounts.extend(
                (
                    "--mount",
                    "type=bind,"
                    f"src={dataset_mount.host_directory},"
                    "dst=/bashgym/dataset,readonly",
                )
            )
    if contract.nemo_gym is not None:
        gym = contract.nemo_gym
        resource_source = (
            run_directory / "nemo_gym_bundle" / "resources_servers" / gym.resources_server_id
        )
        resource_destination = f"{_GYM_WORKSPACE}/resources_servers/{gym.resources_server_id}"
        mounts.extend(
            (
                "--mount",
                f"type=bind,src={resource_source},dst={resource_destination},readonly",
            )
        )
        resources_config = f"{_GYM_WORKSPACE}/{gym.resources_config_path}"
        controller_overrides += (
            "data.default.dataset_name=NemoGymDataset",
            "data.default.env_name=nemo_gym",
            f"data.train.data_path={resource_destination}/data/train.jsonl",
            f"data.validation.data_path={resource_destination}/data/validation.jsonl",
            "env.should_log_nemo_gym_responses=true",
            "env.should_use_nemo_gym=true",
            "env.nemo_gym.config_paths=" + f"[{_VLLM_TRAINING_CONFIG},{resources_config}]",
            "env.nemo_gym.is_trajectory_collection="
            + ("true" if contract.mode.value == "no_update" else "false"),
            "policy.generation.backend=vllm",
            "policy.generation.vllm_cfg.async_engine=true",
            "policy.generation.vllm_cfg.expose_http_server=true",
        )
    return (
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network=none",
        "--gpus",
        f"device={','.join(str(index) for index in range(contract.gpu_count))}",
        "--shm-size",
        f"{contract.shared_memory_gib}g",
        *mounts,
        "--workdir",
        "/opt/nemo-rl",
        contract.image_reference,
        "uv",
        "run",
        "--no-sync",
        contract.entrypoint_path,
        "--config",
        contract.recipe_path,
        *controller_overrides,
        *contract.overrides,
    )


def _append_metric(handle: IO[str], *, name: str, value: float, step: int) -> None:
    record = {
        "schema_version": "nemo_rl_training_metric.v1",
        "observed_at": _utc_now(),
        "step": step,
        name.casefold(): value,
    }
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()


def run_contract(
    contract: NemoRLContainerContract,
    run_directory: Path,
    *,
    model_directory: Path | None = None,
    dataset_directory: Path | None = None,
    experiment_recipe: TMaxCompositeTrainingRecipe | None = None,
) -> int:
    if experiment_recipe is not None and experiment_recipe.sft_enabled:
        raise RuntimeError("nemo_rl_sft_composition_unsupported")
    effective_max_steps = experiment_recipe.max_steps if experiment_recipe else contract.max_steps
    effective_learning_rate = (
        experiment_recipe.learning_rate if experiment_recipe else contract.learning_rate
    )
    run_directory = run_directory.resolve()
    final_directory = run_directory / "final"
    final_directory.mkdir(exist_ok=True)
    (run_directory / "logs").mkdir(exist_ok=True)
    dataset_mount = resolve_dataset_mount(
        contract,
        run_directory=run_directory,
        dataset_directory=dataset_directory,
    )
    model_mount = validate_runtime_identity(
        contract,
        run_directory,
        model_directory=model_directory,
    )
    if contract.nemo_gym is not None:
        bundle_root = run_directory / "nemo_gym_bundle"
        extract_nemo_gym_bundle_archive(
            run_directory / contract.nemo_gym.bundle_archive_file,
            bundle_root,
        )
        for source_name, output_name in (
            ("bundle_manifest.json", NEMO_GYM_BUNDLE_MANIFEST_FILENAME),
            ("environment_contract.json", NEMO_GYM_ENVIRONMENT_CONTRACT_FILENAME),
        ):
            source = bundle_root / source_name
            destination = run_directory / output_name
            if destination.exists() or destination.is_symlink():
                raise RuntimeError("nemo_gym_evidence_companion_already_exists")
            destination.write_bytes(source.read_bytes())
    experiment_payload = (
        experiment_recipe.model_dump(mode="json") if experiment_recipe is not None else None
    )
    identity = canonical_hash(
        {"contract": contract.model_dump(mode="json"), "experiment_recipe": experiment_payload}
    )[:20]
    container_name = f"bashgym-nemo-{identity}-{os.getpid()}"
    argv = docker_argv(
        contract,
        run_directory=run_directory,
        model_mount=model_mount,
        dataset_mount=dataset_mount,
        container_name=container_name,
        experiment_recipe=experiment_recipe,
    )
    effective_config = {
        "schema_version": "nemo_rl_effective_config.v1",
        "contract": contract.model_dump(mode="json"),
        "contract_digest": canonical_hash(contract.model_dump(mode="json")),
        "container_name": container_name,
        "dataset_sha256": dataset_mount.sha256,
        "experiment_recipe": experiment_payload,
        "experiment_recipe_digest": (
            canonical_hash(experiment_payload) if experiment_payload is not None else None
        ),
        "argv_sha256": hashlib.sha256("\0".join(argv).encode()).hexdigest(),
        "started_at": _utc_now(),
    }
    _write_json(run_directory / "effective_config.json", effective_config)

    process: subprocess.Popen[str] | None = None

    def stop_container(_signum: int, _frame: object) -> None:
        subprocess.run(
            ("docker", "stop", "--time", "10", container_name),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if process is not None and process.poll() is None:
            process.terminate()

    previous = {sig: signal.signal(sig, stop_container) for sig in (signal.SIGINT, signal.SIGTERM)}
    exit_code = 125
    metrics_path = run_directory / "training_metrics.jsonl"
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics:
            _append_metric(metrics, name="run_started", value=1.0, step=0)
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            observed_step: int | None = None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                step_match = re.search(r"(?:step|iteration)[\s/:=]+(\d+)", line, re.I)
                if step_match:
                    observed_step = int(step_match.group(1))
                for match in _METRIC.finditer(line):
                    _append_metric(
                        metrics,
                        name=match.group("name"),
                        value=float(match.group("value")),
                        step=observed_step if observed_step is not None else 0,
                    )
            exit_code = process.wait()
            _append_metric(
                metrics,
                name="run_completed" if exit_code == 0 else "run_failed",
                value=1.0,
                step=effective_max_steps,
            )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    checkpoints = sorted(
        str(path.relative_to(run_directory))
        for path in final_directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    _write_json(
        run_directory / "training_manifest.json",
        {
            "schema_version": "nemo_rl_training_manifest.v1",
            "contract_digest": effective_config["contract_digest"],
            "release": contract.release,
            "source_revision": contract.source_revision,
            "image_digest": contract.image_digest,
            "model_id": contract.model_id,
            "model_revision": contract.model_revision,
            "model_support_level": contract.model_support_level,
            "recipe_sha256": contract.recipe_sha256,
            "dataset_sha256": dataset_mount.sha256,
            "verifier_id": contract.verifier_id,
            "verifier_digest": contract.verifier_digest,
            "nemo_gym": (
                {
                    "bundle_digest": contract.nemo_gym.bundle_digest,
                    "environment_id": contract.nemo_gym.environment_id,
                    "environment_digest": contract.nemo_gym.environment_digest,
                    "nemo_gym_source_revision": contract.nemo_gym.nemo_gym_source_revision,
                    "trajectory_collection": contract.mode.value == "no_update",
                }
                if contract.nemo_gym is not None
                else None
            ),
            "mode": contract.mode,
            "algorithm": experiment_recipe.algorithm if experiment_recipe else "grpo",
            "max_steps": effective_max_steps,
            "learning_rate": effective_learning_rate,
            "experiment_recipe": experiment_payload,
            "checkpoints": checkpoints,
            "exit_code": exit_code,
            "completed_at": _utc_now(),
        },
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--algorithm", choices=("grpo",))
    parser.add_argument("--sft-enabled", choices=("true", "false"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    contract = NemoRLContainerContract.model_validate_json(args.contract_json)
    experiment_values = (
        args.algorithm,
        args.sft_enabled,
        args.learning_rate,
        args.max_steps,
        args.group_size,
        args.temperature,
        args.seed,
    )
    experiment_recipe = None
    if any(value is not None for value in experiment_values):
        if any(value is None for value in experiment_values):
            parser.error("the complete typed experiment recipe is required")
        try:
            experiment_recipe = TMaxCompositeTrainingRecipe(
                algorithm=args.algorithm,
                sft_enabled=args.sft_enabled == "true",
                learning_rate=args.learning_rate,
                max_steps=args.max_steps,
                group_size=args.group_size,
                temperature=args.temperature,
                seed=args.seed,
            )
        except ValueError as exc:
            parser.error(f"invalid typed experiment recipe: {exc}")
    return run_contract(
        contract,
        Path.cwd(),
        model_directory=args.model_dir,
        dataset_directory=args.dataset_dir,
        experiment_recipe=experiment_recipe,
    )


if __name__ == "__main__":
    raise SystemExit(main())
