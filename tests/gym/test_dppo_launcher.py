import subprocess
from pathlib import Path

from bashgym.gym.dppo_backend import DPPOBackendCapability
from bashgym.gym.dppo_launcher import (
    DPPOSmokeLaunchConfig,
    build_dppo_smoke_launch_plan,
    prepare_dppo_smoke_launch,
    run_dppo_smoke_launch,
)


def _capabilities(**available: bool) -> dict[str, DPPOBackendCapability]:
    return {
        name: DPPOBackendCapability(
            name=name,
            available=available.get(name, False),
            reason=f"{name} {'available' if available.get(name, False) else 'missing'}",
            path=(f"/opt/{name}" if available.get(name, False) else None),
        )
        for name in ("verl", "skyrl", "tmax_open_instruct")
    }


def test_prepare_verl_smoke_launch_writes_script(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text('{"schema_version":"bashgym.dppo_replay.v1"}\n', encoding="utf-8")
    config = DPPOSmokeLaunchConfig(
        replay_path=replay_path,
        output_dir=tmp_path / "run",
        base_model="Qwen/Qwen3.5-4B",
        backend="verl",
    )

    plan = prepare_dppo_smoke_launch(config, capabilities=_capabilities(verl=True))

    assert plan.runnable is True
    assert plan.selection.selected == "verl"
    assert plan.command[:3] == ["python", "-m", "verl.trainer.main_ppo"]
    assert f"data.train_files={replay_path}" in plan.command
    assert plan.cwd == "/opt/verl"
    assert plan.script_path is not None
    script = Path(plan.script_path).read_text(encoding="utf-8")
    assert "BASHGYM_DPPO_REPLAY_PATH" in script
    assert "verl.trainer.main_ppo" in script


def test_auto_falls_back_when_no_dppo_backend_available(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text("{}\n", encoding="utf-8")
    config = DPPOSmokeLaunchConfig(
        replay_path=replay_path,
        output_dir=tmp_path / "run",
        base_model="model",
        backend="auto",
    )

    plan = build_dppo_smoke_launch_plan(config, capabilities=_capabilities())

    assert plan.runnable is False
    assert plan.selection.selected == "grpo_fallback"
    assert "No DPPO-capable backend detected" in plan.reason


def test_missing_replay_path_is_not_runnable(tmp_path):
    config = DPPOSmokeLaunchConfig(
        replay_path=tmp_path / "missing.jsonl",
        output_dir=tmp_path / "run",
        base_model="model",
        backend="verl",
    )

    plan = build_dppo_smoke_launch_plan(config, capabilities=_capabilities(verl=True))

    assert plan.runnable is False
    assert "replay_path does not exist" in plan.reason


def test_command_template_overrides_backend_default(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text("{}\n", encoding="utf-8")
    config = DPPOSmokeLaunchConfig(
        replay_path=replay_path,
        output_dir=tmp_path / "run",
        base_model="model",
        backend="skyrl",
    )

    plan = build_dppo_smoke_launch_plan(
        config,
        capabilities=_capabilities(skyrl=True),
        env={"BASHGYM_DPPO_SKYRL_COMMAND_TEMPLATE": "python train.py --data {replay_path}"},
    )

    assert plan.command == ["python", "train.py", "--data", str(replay_path)]
    assert plan.reason == "using configured DPPO command template"


def test_explicit_command_template_can_run_when_backend_probe_misses(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text("{}\n", encoding="utf-8")
    config = DPPOSmokeLaunchConfig(
        replay_path=replay_path,
        output_dir=tmp_path / "run",
        base_model="model",
        backend="verl",
        command_template="python train.py --data {replay_path} --model {base_model}",
    )

    plan = build_dppo_smoke_launch_plan(config, capabilities=_capabilities())

    assert plan.runnable is True
    assert plan.selection.selected == "verl"
    assert plan.command == ["python", "train.py", "--data", str(replay_path), "--model", "model"]
    assert "explicit DPPO command template" in plan.reason
    assert plan.env["BASHGYM_DPPO_REPLAY_PATH"] == str(replay_path)


def test_launch_plan_threads_world_model_settings_to_backend(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text("{}\n", encoding="utf-8")
    config = DPPOSmokeLaunchConfig(
        replay_path=replay_path,
        output_dir=tmp_path / "run",
        base_model="Qwen/Qwen3.5-4B",
        backend="verl",
        echo_enabled=True,
        echo_aux_lambda=0.05,
        rwml_enabled=True,
        rwml_distance_threshold=0.15,
        rwml_easy_pass_rate_threshold=0.75,
        rwml_easy_keep_probability=0.2,
        rwml_history_window=6,
        rwml_embedding_model="qwen3-embedding",
        rwml_kl_beta=0.03,
    )

    plan = prepare_dppo_smoke_launch(config, capabilities=_capabilities(verl=True))

    # The external backend reads ECHO/RWML config from the launch env
    assert plan.env["BASHGYM_DPPO_ECHO_ENABLED"] == "1"
    assert plan.env["BASHGYM_DPPO_ECHO_LAMBDA"] == "0.05"
    assert plan.env["BASHGYM_DPPO_RWML_ENABLED"] == "1"
    assert plan.env["BASHGYM_DPPO_RWML_DISTANCE_THRESHOLD"] == "0.15"
    assert plan.env["BASHGYM_DPPO_RWML_EASY_PASS_RATE_THRESHOLD"] == "0.75"
    assert plan.env["BASHGYM_DPPO_RWML_EASY_KEEP_PROBABILITY"] == "0.2"
    assert plan.env["BASHGYM_DPPO_RWML_HISTORY_WINDOW"] == "6"
    assert plan.env["BASHGYM_DPPO_RWML_EMBEDDING_MODEL"] == "qwen3-embedding"
    assert plan.env["BASHGYM_DPPO_RWML_KL_BETA"] == "0.03"
    assert (
        plan.env["BASHGYM_DPPO_WORLD_MODEL_ADAPTER"]
        == "bashgym.gym.world_model_backend:WorldModelTrainerAdapter"
    )
    assert plan.env["BASHGYM_DPPO_ECHO_LOSS_HOOK"].endswith(
        "WorldModelTrainerAdapter.apply_echo_loss"
    )
    assert plan.env["BASHGYM_DPPO_TRL_RWML_REWARD_FACTORY"].endswith("build_trl_rwml_reward_func")
    assert plan.env["BASHGYM_DPPO_VERL_RWML_REWARD_FACTORY"].endswith("build_verl_rwml_reward_fn")

    world_model = plan.to_dict()["world_model"]
    assert world_model["echo_enabled"] is True
    assert world_model["rwml_distance_threshold"] == 0.15
    assert world_model["rwml_easy_pass_rate_threshold"] == 0.75
    assert world_model["rwml_easy_keep_probability"] == 0.2
    assert world_model["rwml_history_window"] == 6
    assert world_model["rwml_kl_beta"] == 0.03

    # And the materialized script exports them
    script = Path(plan.script_path).read_text(encoding="utf-8")
    assert "BASHGYM_DPPO_ECHO_LAMBDA" in script
    assert "BASHGYM_DPPO_WORLD_MODEL_ADAPTER" in script
    assert "qwen3-embedding" in script


def test_run_dppo_smoke_launch_invokes_runner(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text("{}\n", encoding="utf-8")
    config = DPPOSmokeLaunchConfig(
        replay_path=replay_path,
        output_dir=tmp_path / "run",
        base_model="model",
        backend="verl",
        write_script=False,
    )
    plan = build_dppo_smoke_launch_plan(config, capabilities=_capabilities(verl=True))
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = run_dppo_smoke_launch(plan, run=fake_run)

    assert result.returncode == 0
    assert captured["command"] == plan.command
    assert captured["kwargs"]["env"]["BASHGYM_DPPO_REPLAY_PATH"] == str(replay_path)
