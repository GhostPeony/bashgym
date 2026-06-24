import json
from pathlib import Path

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.rollout import (
    CommandObservation,
    EnvironmentRolloutResult,
    RolloutAttempt,
)
from bashgym.eval.dppo_replay import (
    DPPO_REPLAY_SCHEMA_VERSION,
    build_dppo_replay_records,
    enrich_dppo_replay_jsonl,
    enrich_dppo_replay_records,
    read_dppo_replay_jsonl,
    write_dppo_replay_jsonl,
)


def _environment() -> EnvironmentSpec:
    return EnvironmentSpec.from_dict(
        {
            "id": "env_dppo",
            "instruction": "Write ok to answer.txt",
            "verifier": {"command": 'test "$(cat answer.txt)" = ok'},
        }
    )


def _rollout() -> EnvironmentRolloutResult:
    return EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_dppo",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={
                "model": "candidate",
                "base_url": "http://h/v1",
                "active_sampling_selected": True,
                "reward_group_std": 0.5,
                "response_logprobs": [
                    {
                        "n_tokens": 2,
                        "tokens": ["echo", " ok"],
                        "token_logprobs": [-0.1, -0.2],
                    }
                ],
                "behavior_logprob_tokens": 2,
                "behavior_logprob_sum": -0.3,
                "behavior_mean_logprob": -0.15,
            },
        ),
        workspace=Path("unused"),
        observations=[
            CommandObservation(
                command="echo ok > answer.txt",
                cwd=".",
                exit_code=0,
                stdout="",
                stderr="",
                duration_sec=0.01,
            )
        ],
        verifier_observation=CommandObservation(
            command='test "$(cat answer.txt)" = ok',
            cwd=".",
            exit_code=0,
            stdout="",
            stderr="",
            duration_sec=0.01,
        ),
    )


def _multi_command_rollout() -> EnvironmentRolloutResult:
    return EnvironmentRolloutResult(
        attempt=RolloutAttempt(
            environment_id="env_dppo",
            attempt_index=0,
            passed=True,
            reward=1.0,
            verifier_status="passed",
            metadata={"model": "candidate"},
        ),
        workspace=Path("unused"),
        observations=[
            CommandObservation(
                command="ls",
                cwd=".",
                exit_code=0,
                stdout="fileA fileB\n",
                stderr="",
                duration_sec=0.01,
            ),
            CommandObservation(
                command="cat fileA",
                cwd=".",
                exit_code=0,
                stdout="hello",
                stderr="",
                duration_sec=0.01,
            ),
            CommandObservation(
                command="rm missing",
                cwd=".",
                exit_code=1,
                stdout="",
                stderr="rm: missing: No such file\n",
                duration_sec=0.01,
            ),
        ],
        verifier_observation=None,
    )


def test_build_dppo_replay_records_preserves_prompt_reward_and_logprobs():
    records = build_dppo_replay_records([_environment()], [_rollout()], batch_id="batch-1")

    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == DPPO_REPLAY_SCHEMA_VERSION
    assert record["batch_id"] == "batch-1"
    assert record["environment_id"] == "env_dppo"
    assert record["prompt"] == "Write ok to answer.txt"
    assert record["reward"] == 1.0
    assert record["active_sampling_selected"] is True
    assert record["behavior_policy"]["behavior_logprob_tokens"] == 2
    assert record["optimizer"]["behavior_logprobs_ready"] is True
    assert record["optimizer"]["train_logprob_replay_required"] is True
    assert record["trajectory"]["commands"] == ["echo ok > answer.txt"]


def test_build_dppo_replay_records_omits_world_model_by_default():
    records = build_dppo_replay_records([_environment()], [_rollout()], batch_id="batch-1")

    assert "world_model" not in records[0]


def test_build_dppo_replay_records_adds_rwml_transitions_when_requested():
    records = build_dppo_replay_records(
        [_environment()],
        [_multi_command_rollout()],
        batch_id="batch-1",
        include_world_model=True,
        history_window=1,
    )

    world_model = records[0]["world_model"]
    transitions = world_model["rwml_transitions"]
    assert [t["action"] for t in transitions] == ["ls", "cat fileA", "rm missing"]
    # state maps to stdout/stderr output for each command
    assert transitions[0]["next_state"] == "fileA fileB\n"
    assert transitions[2]["next_state"] == "rm: missing: No such file\n"
    # instruction is the environment prompt
    assert transitions[0]["instruction"] == "Write ok to answer.txt"
    # history-windowed prior carries up to history_window prior (action, state) pairs
    assert transitions[0]["prior"] == []
    assert transitions[1]["prior"] == [["ls", "fileA fileB\n"]]
    assert transitions[2]["prior"] == [["cat fileA", "hello"]]


def test_build_dppo_replay_records_adds_echo_text_segments_and_counts():
    records = build_dppo_replay_records(
        [_environment()],
        [_multi_command_rollout()],
        batch_id="batch-1",
        include_world_model=True,
    )

    echo = records[0]["world_model"]["echo"]
    # role-tagged text spans alternate action -> observation per command
    assert [seg["role"] for seg in echo["segments"]] == [
        "action",
        "observation",
        "action",
        "observation",
        "action",
        "observation",
    ]
    assert echo["segments"][0] == {"role": "action", "text": "ls"}
    assert echo["segments"][1] == {"role": "observation", "text": "fileA fileB\n"}
    # char counts let a tokenizer-equipped backend size masks downstream
    expected_action_chars = len("ls") + len("cat fileA") + len("rm missing")
    expected_obs_chars = len("fileA fileB\n") + len("hello") + len("rm: missing: No such file\n")
    assert echo["n_action_chars"] == expected_action_chars
    assert echo["n_observation_chars"] == expected_obs_chars
    # no fabricated token ids/masks; that is documented as built downstream
    assert "token_ids" not in echo
    assert "input_ids" not in echo
    assert "note" in echo


def test_build_dppo_replay_records_world_model_keeps_schema_version_v1():
    records = build_dppo_replay_records(
        [_environment()],
        [_multi_command_rollout()],
        batch_id="batch-1",
        include_world_model=True,
    )

    assert records[0]["schema_version"] == DPPO_REPLAY_SCHEMA_VERSION


def test_write_dppo_replay_jsonl_returns_artifact_summary(tmp_path):
    output_path = tmp_path / "dppo" / "batch.jsonl"

    summary = write_dppo_replay_jsonl(
        output_path,
        [_environment()],
        [_rollout()],
        batch_id="batch-1",
    )

    assert summary["path"] == str(output_path)
    assert summary["batch_id"] == "batch-1"
    assert summary["records"] == 1
    assert summary["behavior_logprobs_ready_records"] == 1
    assert summary["world_model_records"] == 0
    assert summary["world_model"]["rwml_transitions"] == 0
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert written["schema_version"] == DPPO_REPLAY_SCHEMA_VERSION
    assert "world_model" not in written


def test_write_dppo_replay_jsonl_can_include_world_model_payload(tmp_path):
    output_path = tmp_path / "dppo" / "world-model.jsonl"

    summary = write_dppo_replay_jsonl(
        output_path,
        [_environment()],
        [_multi_command_rollout()],
        batch_id="batch-1",
        include_world_model=True,
        history_window=1,
    )

    written = json.loads(output_path.read_text(encoding="utf-8").strip())
    expected_action_chars = len("ls") + len("cat fileA") + len("rm missing")
    expected_obs_chars = len("fileA fileB\n") + len("hello") + len("rm: missing: No such file\n")
    assert summary["world_model_records"] == 1
    assert summary["world_model"] == {
        "records": 1,
        "records_missing_world_model": 0,
        "rwml_transitions": 3,
        "rwml_mean_transitions_per_record": 3.0,
        "rwml_mean_prior_pairs": 2 / 3,
        "rwml_max_prior_pairs": 1,
        "echo_segments": 6,
        "echo_action_chars": expected_action_chars,
        "echo_observation_chars": expected_obs_chars,
        "echo_observation_char_fraction": expected_obs_chars
        / (expected_action_chars + expected_obs_chars),
    }
    assert written["world_model"]["rwml_transitions"][1]["prior"] == [["ls", "fileA fileB\n"]]
    assert written["world_model"]["echo"]["n_observation_chars"] > 0


def test_enrich_dppo_replay_records_adds_train_logprobs_and_mask_telemetry():
    records = build_dppo_replay_records([_environment()], [_rollout()], batch_id="batch-1")

    enriched, summary = enrich_dppo_replay_records(
        records,
        lambda _record: {
            "model": "train-policy",
            "base_url": "http://train/v1",
            "token_logprobs": [-0.04, -0.03],
        },
    )

    assert summary["records"] == 1
    assert summary["train_logprobs_ready_records"] == 1
    assert summary["train_logprob_replay_required_records"] == 0
    assert summary["dppo"]["n_tokens"] == 2
    assert enriched[0]["train_policy"]["model"] == "train-policy"
    assert enriched[0]["train_policy"]["train_logprob_tokens"] == 2
    assert enriched[0]["optimizer"]["train_logprobs_ready"] is True
    assert enriched[0]["optimizer"]["train_logprob_replay_required"] is False


def test_enrich_dppo_replay_records_rejects_train_logprob_length_mismatch():
    records = build_dppo_replay_records([_environment()], [_rollout()], batch_id="batch-1")

    try:
        enrich_dppo_replay_records(records, lambda _record: {"token_logprobs": [-0.04]})
    except ValueError as exc:
        assert "1 logprobs for 2 behavior tokens" in str(exc)
    else:
        raise AssertionError("expected mismatch to raise")


def test_enrich_dppo_replay_jsonl_round_trips_enriched_records(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_dppo_replay_jsonl(
        input_path,
        [_environment()],
        [_rollout()],
        batch_id="batch-1",
    )

    summary = enrich_dppo_replay_jsonl(
        input_path,
        output_path,
        lambda _record: {"token_logprobs": [-0.04, -0.03]},
    )
    enriched = read_dppo_replay_jsonl(output_path)

    assert summary["input_path"] == str(input_path)
    assert summary["path"] == str(output_path)
    assert enriched[0]["optimizer"]["train_logprobs_ready"] is True
