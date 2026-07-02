from pathlib import Path

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.rollout import (
    CommandObservation,
    EnvironmentRolloutResult,
    RolloutAttempt,
)
from bashgym.eval.dppo_replay import build_dppo_replay_records, write_dppo_records_jsonl
from bashgym.gym.backend_smoke_bundle import (
    BackendSmokeBundleConfig,
    prepare_backend_smoke_bundle,
)
from bashgym.gym.dppo_backend import DPPOBackendCapability


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


def _write_replay(
    tmp_path: Path,
    *,
    include_world_model: bool,
    include_behavior_logprobs: bool,
    include_train_logprobs: bool,
) -> Path:
    environment = EnvironmentSpec.from_dict(
        {
            "id": "env_smoke",
            "instruction": "List files and find answer.txt",
            "verifier": {"command": "test -f answer.txt"},
        }
    )
    metadata = {}
    if include_behavior_logprobs:
        metadata.update(
            {
                "model": "teacher",
                "response_logprobs": [{"tokens": ["ls"], "token_logprobs": [-0.2]}],
                "behavior_logprob_tokens": 1,
                "behavior_logprob_sum": -0.2,
                "behavior_mean_logprob": -0.2,
            }
        )
    if include_train_logprobs:
        metadata["train_logprob_tokens"] = 1

    rollout = EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_smoke",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata=metadata,
        ),
        workspace=tmp_path,
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="answer.txt\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=None,
    )
    records = build_dppo_replay_records(
        [environment],
        [rollout],
        batch_id="batch-smoke",
        include_world_model=include_world_model,
    )
    replay_path = tmp_path / "replay.jsonl"
    write_dppo_records_jsonl(replay_path, records)
    return replay_path


def test_backend_smoke_bundle_materializes_ready_contract(tmp_path):
    replay_path = _write_replay(
        tmp_path,
        include_world_model=True,
        include_behavior_logprobs=True,
        include_train_logprobs=True,
    )

    report = prepare_backend_smoke_bundle(
        BackendSmokeBundleConfig(
            replay_path=replay_path,
            output_dir=tmp_path / "bundle",
            base_model="tiny-local-model",
            backend="verl",
            rwml_embedding_model="qwen3-embedding",
        ),
        capabilities=_capabilities(verl=True),
    )

    assert report["ok"] is True
    assert report["contract_ready"] is True
    assert report["optimizer_ready"] is True
    assert report["backend_launch_ready"] is True
    assert report["verdict"]["level"] == "ready"
    assert report["launch_env"]["BASHGYM_DPPO_ECHO_ENABLED"] == "1"
    assert report["launch_env"]["BASHGYM_DPPO_RWML_ENABLED"] == "1"
    assert report["world_model_probe"]["batch"]["echo_observation_tokens"] == len("answer.txt\n")
    assert report["world_model_probe"]["batch"]["rwml_transitions"] == 1
    assert Path(report["artifacts"]["readiness"]).exists()
    assert Path(report["artifacts"]["replay_summary"]).exists()
    assert Path(report["artifacts"]["world_model_probe"]).exists()
    assert Path(report["artifacts"]["launch_env"]).exists()
    assert report["artifacts"]["launch_script"] is not None


def test_backend_smoke_bundle_reports_missing_contract_inputs(tmp_path):
    replay_path = _write_replay(
        tmp_path,
        include_world_model=False,
        include_behavior_logprobs=False,
        include_train_logprobs=False,
    )

    report = prepare_backend_smoke_bundle(
        BackendSmokeBundleConfig(
            replay_path=replay_path,
            output_dir=tmp_path / "bundle",
            base_model="model",
            backend="auto",
        ),
        capabilities=_capabilities(),
    )

    check_status = {check["code"]: check["status"] for check in report["checks"]}
    assert report["ok"] is False
    assert report["contract_ready"] is False
    assert report["verdict"]["level"] == "blocked"
    assert check_status["missing_behavior_logprobs"] == "fail"
    assert check_status["missing_world_model_payloads"] == "fail"
    assert check_status["missing_rwml_transitions"] == "fail"
    assert check_status["missing_echo_observations"] == "fail"
    assert check_status["backend_launch_plan"] == "warn"
    assert any("include_world_model_replay=true" in action for action in report["next_actions"])
