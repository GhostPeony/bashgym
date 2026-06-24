import pytest

from bashgym.gym.world_model_backend import (
    build_world_model_backend_batch,
    echo_masks_from_replay_record,
    echo_segments_from_replay_record,
    rwml_rewards_from_predictions,
    rwml_transitions_from_replay_record,
)


def _tokenizer(text: str):
    return [ord(char) for char in text]


def _record():
    return {
        "schema_version": "bashgym.dppo_replay.v1",
        "world_model": {
            "echo": {
                "segments": [
                    {"role": "action", "text": "ls"},
                    {"role": "observation", "text": "ok!"},
                    {"role": "action", "text": "cat file"},
                    {"role": "observation", "text": "hello"},
                ]
            },
            "rwml_transitions": [
                {
                    "instruction": "inspect files",
                    "prior": [],
                    "action": "ls",
                    "next_state": "ok!",
                },
                {
                    "instruction": "inspect files",
                    "prior": [["ls", "ok!"]],
                    "action": "cat file",
                    "next_state": "hello",
                },
            ],
        },
    }


def test_echo_segments_from_replay_record_tokenizes_text_spans():
    segments = echo_segments_from_replay_record(_record(), _tokenizer)

    assert [segment.role for segment in segments] == [
        "action",
        "observation",
        "action",
        "observation",
    ]
    assert segments[0].token_ids == [ord("l"), ord("s")]
    assert segments[1].token_ids == [ord("o"), ord("k"), ord("!")]


def test_echo_masks_from_replay_record_builds_backend_masks():
    masks = echo_masks_from_replay_record(
        _record(),
        _tokenizer,
        exclude_token_ids=(ord("!"),),
    )

    assert len(masks.input_ids) == len("lsok!cat filehello")
    assert sum(masks.action_mask) == len("ls") + len("cat file")
    assert sum(masks.observation_mask) == len("okhello")
    assert masks.total_observation_tokens == len("ok!") + len("hello")


def test_rwml_transitions_from_replay_record_restores_typed_triplets():
    transitions = rwml_transitions_from_replay_record(_record())

    assert len(transitions) == 2
    assert transitions[0].instruction == "inspect files"
    assert transitions[0].prior == ()
    assert transitions[1].prior == (("ls", "ok!"),)
    assert transitions[1].action == "cat file"
    assert transitions[1].next_state == "hello"


def test_build_world_model_backend_batch_ignores_plain_records():
    batch = build_world_model_backend_batch([{}, _record()], _tokenizer)

    assert batch.records_total == 2
    assert batch.records_with_world_model == 1
    assert len(batch.echo_masks) == 1
    assert len(batch.rwml_transitions) == 2
    assert batch.actual_next_states == ("ok!", "hello")
    assert batch.to_dict()["echo_observation_tokens"] == len("ok!") + len("hello")


def test_rwml_rewards_from_predictions_scores_targets():
    transitions = rwml_transitions_from_replay_record(_record())
    vectors = {
        "ok!": [1.0, 0.0],
        "hello": [0.0, 1.0],
        "miss": [1.0, 0.0],
    }

    def embed(text):
        return vectors[text]

    rewards = rwml_rewards_from_predictions(
        ["ok!", "miss"],
        transitions,
        embed,
        distance_threshold=0.2,
    )

    assert rewards == [1.0, 0.0]


def test_echo_masks_reject_unknown_segment_role():
    record = _record()
    record["world_model"]["echo"]["segments"][0]["role"] = "tool"

    with pytest.raises(ValueError, match="role"):
        echo_masks_from_replay_record(record, _tokenizer)
