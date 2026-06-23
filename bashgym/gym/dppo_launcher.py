"""DPPO smoke-launch planning for external trainer backends."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bashgym.gym.dppo_backend import (
    VALID_DPPO_BACKENDS,
    DPPOBackendCapability,
    DPPOBackendSelection,
    select_dppo_backend,
)
from bashgym.gym.echo import ECHO_DEFAULT_LAMBDA
from bashgym.gym.rwml import (
    RWML_DEFAULT_DISTANCE_THRESHOLD,
    RWML_DEFAULT_EASY_KEEP_PROBABILITY,
    RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD,
    RWML_DEFAULT_HISTORY_WINDOW,
)

COMMAND_TEMPLATE_ENV = {
    "verl": "BASHGYM_DPPO_VERL_COMMAND_TEMPLATE",
    "skyrl": "BASHGYM_DPPO_SKYRL_COMMAND_TEMPLATE",
    "tmax_open_instruct": "BASHGYM_DPPO_TMAX_OPEN_INSTRUCT_COMMAND_TEMPLATE",
}


@dataclass(frozen=True)
class DPPOSmokeLaunchConfig:
    replay_path: Path
    output_dir: Path
    base_model: str
    backend: str = "auto"
    max_steps: int = 1
    n_gpus_per_node: int = 1
    write_script: bool = True
    command_template: str | None = None
    # World-model objectives passed through to the backend's training entrypoint
    # (consumed via the BASHGYM_DPPO_ECHO_*/RWML_* env vars below). See
    # bashgym.gym.echo_trainer (ECHO loss) and bashgym.gym.rwml (RWML reward).
    echo_enabled: bool = False
    echo_aux_lambda: float = ECHO_DEFAULT_LAMBDA
    rwml_enabled: bool = False
    rwml_distance_threshold: float = RWML_DEFAULT_DISTANCE_THRESHOLD
    rwml_easy_pass_rate_threshold: float = RWML_DEFAULT_EASY_PASS_RATE_THRESHOLD
    rwml_easy_keep_probability: float = RWML_DEFAULT_EASY_KEEP_PROBABILITY
    rwml_history_window: int = RWML_DEFAULT_HISTORY_WINDOW
    rwml_embedding_model: str = ""
    rwml_kl_beta: float = 0.0

    def __post_init__(self) -> None:
        normalized_backend = (self.backend or "auto").strip().lower()
        if normalized_backend not in VALID_DPPO_BACKENDS:
            raise ValueError(f"backend={self.backend!r} must be one of {list(VALID_DPPO_BACKENDS)}")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.n_gpus_per_node <= 0:
            raise ValueError("n_gpus_per_node must be positive")
        if not self.base_model.strip():
            raise ValueError("base_model is required")
        if self.echo_aux_lambda < 0:
            raise ValueError("echo_aux_lambda must be non-negative")
        if not 0.0 < self.rwml_distance_threshold <= 2.0:
            raise ValueError("rwml_distance_threshold must be in (0, 2] (cosine distance)")
        if not 0.0 <= self.rwml_easy_pass_rate_threshold <= 1.0:
            raise ValueError("rwml_easy_pass_rate_threshold must be a probability in [0, 1]")
        if not 0.0 <= self.rwml_easy_keep_probability <= 1.0:
            raise ValueError("rwml_easy_keep_probability must be a probability in [0, 1]")
        if self.rwml_history_window < 0:
            raise ValueError("rwml_history_window must be non-negative")
        if self.rwml_kl_beta < 0:
            raise ValueError("rwml_kl_beta must be non-negative")
        object.__setattr__(self, "backend", normalized_backend)
        object.__setattr__(self, "replay_path", Path(self.replay_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(frozen=True)
class DPPOSmokeLaunchPlan:
    config: DPPOSmokeLaunchConfig
    selection: DPPOBackendSelection
    command: list[str]
    cwd: str | None
    env: dict[str, str]
    runnable: bool
    reason: str
    warnings: list[str] = field(default_factory=list)
    script_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.selection.selected,
            "requested_backend": self.selection.requested,
            "available": self.selection.available,
            "fallback_to_grpo": self.selection.fallback_to_grpo,
            "runnable": self.runnable,
            "reason": self.reason,
            "command": self.command,
            "cwd": self.cwd,
            "env": self.env,
            "warnings": self.warnings,
            "script_path": self.script_path,
            "replay_path": str(self.config.replay_path),
            "output_dir": str(self.config.output_dir),
            "base_model": self.config.base_model,
            "max_steps": self.config.max_steps,
            "n_gpus_per_node": self.config.n_gpus_per_node,
            "selection": self.selection.to_dict(),
            "world_model": {
                "echo_enabled": self.config.echo_enabled,
                "echo_aux_lambda": self.config.echo_aux_lambda,
                "rwml_enabled": self.config.rwml_enabled,
                "rwml_distance_threshold": self.config.rwml_distance_threshold,
                "rwml_easy_pass_rate_threshold": self.config.rwml_easy_pass_rate_threshold,
                "rwml_easy_keep_probability": self.config.rwml_easy_keep_probability,
                "rwml_history_window": self.config.rwml_history_window,
                "rwml_embedding_model": self.config.rwml_embedding_model,
                "rwml_kl_beta": self.config.rwml_kl_beta,
            },
        }


def build_dppo_smoke_launch_plan(
    config: DPPOSmokeLaunchConfig,
    *,
    capabilities: dict[str, DPPOBackendCapability] | None = None,
    env: Mapping[str, str] | None = None,
) -> DPPOSmokeLaunchPlan:
    """Build a backend-specific command plan for a tiny DPPO smoke run."""

    env_map = env or os.environ
    selection = select_dppo_backend(config.backend, capabilities=capabilities)
    output_dir = config.output_dir
    replay_path = config.replay_path
    warnings: list[str] = []

    if not replay_path.exists():
        return DPPOSmokeLaunchPlan(
            config=config,
            selection=selection,
            command=[],
            cwd=None,
            env={},
            runnable=False,
            reason=f"replay_path does not exist: {replay_path}",
            warnings=warnings,
        )

    if selection.fallback_to_grpo:
        return DPPOSmokeLaunchPlan(
            config=config,
            selection=selection,
            command=[],
            cwd=None,
            env={},
            runnable=False,
            reason=selection.reason,
            warnings=[
                "DPPO backend unavailable; run GRPO fallback or configure a backend checkout."
            ],
        )

    values = {
        "replay_path": str(replay_path),
        "output_dir": str(output_dir),
        "base_model": config.base_model,
        "max_steps": str(config.max_steps),
        "n_gpus_per_node": str(config.n_gpus_per_node),
    }
    template = _resolve_command_template(selection.selected, config, env_map)
    if template:
        command = _split_command_template(template, values)
        reason = "using configured DPPO command template"
    else:
        command, warnings = _default_backend_command(selection, values, config)
        reason = f"using BashGym default {selection.selected} smoke command"

    cwd = _backend_cwd(selection)
    launch_env = {
        "BASHGYM_DPPO_REPLAY_PATH": str(replay_path),
        "BASHGYM_DPPO_OUTPUT_DIR": str(output_dir),
        "BASHGYM_DPPO_BASE_MODEL": config.base_model,
        "BASHGYM_DPPO_MAX_STEPS": str(config.max_steps),
        # World-model objectives the backend's training entrypoint reads.
        "BASHGYM_DPPO_ECHO_ENABLED": "1" if config.echo_enabled else "0",
        "BASHGYM_DPPO_ECHO_LAMBDA": str(config.echo_aux_lambda),
        "BASHGYM_DPPO_RWML_ENABLED": "1" if config.rwml_enabled else "0",
        "BASHGYM_DPPO_RWML_DISTANCE_THRESHOLD": str(config.rwml_distance_threshold),
        "BASHGYM_DPPO_RWML_EASY_PASS_RATE_THRESHOLD": str(
            config.rwml_easy_pass_rate_threshold
        ),
        "BASHGYM_DPPO_RWML_EASY_KEEP_PROBABILITY": str(config.rwml_easy_keep_probability),
        "BASHGYM_DPPO_RWML_HISTORY_WINDOW": str(config.rwml_history_window),
        "BASHGYM_DPPO_RWML_EMBEDDING_MODEL": config.rwml_embedding_model,
        "BASHGYM_DPPO_RWML_KL_BETA": str(config.rwml_kl_beta),
        # Stable import targets for backend entrypoints/subclasses.
        "BASHGYM_DPPO_WORLD_MODEL_ADAPTER": (
            "bashgym.gym.world_model_backend:WorldModelTrainerAdapter"
        ),
        "BASHGYM_DPPO_ECHO_LOSS_HOOK": (
            "bashgym.gym.world_model_backend:WorldModelTrainerAdapter.apply_echo_loss"
        ),
        "BASHGYM_DPPO_TRL_RWML_REWARD_FACTORY": (
            "bashgym.gym.world_model_backend:build_trl_rwml_reward_func"
        ),
        "BASHGYM_DPPO_VERL_RWML_REWARD_FACTORY": (
            "bashgym.gym.world_model_backend:build_verl_rwml_reward_fn"
        ),
    }
    return DPPOSmokeLaunchPlan(
        config=config,
        selection=selection,
        command=command,
        cwd=cwd,
        env=launch_env,
        runnable=bool(command),
        reason=reason,
        warnings=warnings,
    )


def materialize_dppo_smoke_launcher(plan: DPPOSmokeLaunchPlan) -> DPPOSmokeLaunchPlan:
    """Write a small shell script for the launch plan and return an updated plan."""

    if not plan.command:
        return plan
    plan.config.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = plan.config.output_dir / "launch_dppo_smoke.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    # Export every launch env var (replay/output/model/steps + world-model config)
    # so the backend entrypoint sees the full contract.
    for key in sorted(plan.env):
        lines.append(f"export {key}={shlex.quote(plan.env[key])}")
    if plan.cwd:
        lines.append(f"cd {shlex.quote(plan.cwd)}")
    lines.append(" ".join(shlex.quote(part) for part in plan.command))
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        script_path.chmod(0o755)
    except OSError:
        pass
    return DPPOSmokeLaunchPlan(
        config=plan.config,
        selection=plan.selection,
        command=plan.command,
        cwd=plan.cwd,
        env=plan.env,
        runnable=plan.runnable,
        reason=plan.reason,
        warnings=plan.warnings,
        script_path=str(script_path),
    )


def prepare_dppo_smoke_launch(
    config: DPPOSmokeLaunchConfig,
    *,
    capabilities: dict[str, DPPOBackendCapability] | None = None,
    env: Mapping[str, str] | None = None,
) -> DPPOSmokeLaunchPlan:
    """Build a launch plan and optionally materialize its shell script."""

    plan = build_dppo_smoke_launch_plan(config, capabilities=capabilities, env=env)
    if config.write_script and plan.runnable:
        return materialize_dppo_smoke_launcher(plan)
    return plan


def run_dppo_smoke_launch(
    plan: DPPOSmokeLaunchPlan,
    *,
    timeout_sec: int = 300,
    run: Any = subprocess.run,
) -> subprocess.CompletedProcess:
    """Execute a launch plan. Intended for operator-triggered smoke runs only."""

    if not plan.runnable or not plan.command:
        raise ValueError(f"DPPO smoke launch is not runnable: {plan.reason}")
    merged_env = {**os.environ, **plan.env}
    return run(
        plan.command,
        cwd=plan.cwd,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def _resolve_command_template(
    backend: str,
    config: DPPOSmokeLaunchConfig,
    env: Mapping[str, str],
) -> str | None:
    if config.command_template:
        return config.command_template
    env_key = COMMAND_TEMPLATE_ENV.get(backend)
    return env.get(env_key, "") if env_key else None


def _split_command_template(template: str, values: dict[str, str]) -> list[str]:
    formatted = template.format(**values)
    return shlex.split(formatted, posix=os.name != "nt")


def _backend_cwd(selection: DPPOBackendSelection) -> str | None:
    capability = selection.capabilities.get(selection.selected)
    if capability and capability.path:
        return capability.path
    return None


def _default_backend_command(
    selection: DPPOBackendSelection,
    values: dict[str, str],
    config: DPPOSmokeLaunchConfig,
) -> tuple[list[str], list[str]]:
    backend = selection.selected
    if backend == "verl":
        return _verl_command(values, config), [
            "verl expects its data reader schema to match the replay artifact; "
            "override with BASHGYM_DPPO_VERL_COMMAND_TEMPLATE for a project-specific adapter."
        ]
    if backend == "skyrl":
        return _skyrl_command(values, config), [
            "SkyRL entrypoints/config keys vary by release; override with "
            "BASHGYM_DPPO_SKYRL_COMMAND_TEMPLATE if your checkout differs."
        ]
    if backend == "tmax_open_instruct":
        return _tmax_open_instruct_command(values, config), [
            "open-instruct RLVR uses grpo_fast.py and Ray/vLLM placement; this smoke command "
            "uses conservative local settings and may need a cluster wrapper."
        ]
    return [], [f"no command template for backend {backend}"]


def _verl_command(values: dict[str, str], config: DPPOSmokeLaunchConfig) -> list[str]:
    return [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={values['replay_path']}",
        f"data.val_files={values['replay_path']}",
        "data.train_batch_size=1",
        "data.val_batch_size=1",
        "data.max_prompt_length=2048",
        "data.max_response_length=512",
        f"actor_rollout_ref.model.path={values['base_model']}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "trainer.val_before_train=False",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        "trainer.nnodes=1",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "trainer.total_epochs=1",
        f"trainer.default_local_dir={values['output_dir']}",
    ]


def _skyrl_command(values: dict[str, str], config: DPPOSmokeLaunchConfig) -> list[str]:
    quoted_replay = f"['{values['replay_path']}']"
    return [
        "uv",
        "run",
        "--isolated",
        "-m",
        "skyrl.train.entrypoints.main_base",
        f"data.train_data={quoted_replay}",
        f"data.val_data={quoted_replay}",
        "trainer.algorithm.advantage_estimator=grpo",
        f"trainer.policy.model.path={values['base_model']}",
        f"trainer.num_train_steps={config.max_steps}",
        f"trainer.ckpt_path={values['output_dir']}",
        f"trainer.placement.policy_num_gpus_per_node={config.n_gpus_per_node}",
    ]


def _tmax_open_instruct_command(
    values: dict[str, str],
    config: DPPOSmokeLaunchConfig,
) -> list[str]:
    return [
        "python",
        "open_instruct/grpo_fast.py",
        "--exp_name",
        "bashgym_dppo_smoke",
        "--model_name_or_path",
        values["base_model"],
        "--output_dir",
        values["output_dir"],
        "--num_unique_prompts_rollout",
        "1",
        "--num_samples_per_prompt_rollout",
        "2",
        "--num_epochs",
        "1",
        "--total_episodes",
        str(config.max_steps),
        "--learning_rate",
        "1e-6",
        "--per_device_train_batch_size",
        "1",
        "--save_traces",
    ]
