import json
import math

import pytest

from bashgym.gym.world_model_backend import (
    CachedEmbeddingProvider,
    WorldModelTrainerAdapter,
    WorldModelTrainerSettings,
    build_verl_rwml_reward_fn,
    read_replay_jsonl,
    score_rwml_prediction_pairs,
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


def _vectors(texts):
    values = {
        "ok!": [1.0, 0.0],
        "same as ok": [1.0, 0.0],
        "hello": [0.0, 1.0],
        "miss": [1.0, 0.0],
    }
    return [values[text] for text in texts]


def test_settings_read_dppo_world_model_env_contract():
    settings = WorldModelTrainerSettings.from_env(
        {
            "BASHGYM_DPPO_ECHO_ENABLED": "1",
            "BASHGYM_DPPO_ECHO_LAMBDA": "0.07",
            "BASHGYM_DPPO_RWML_ENABLED": "true",
            "BASHGYM_DPPO_RWML_DISTANCE_THRESHOLD": "0.15",
            "BASHGYM_DPPO_RWML_EASY_PASS_RATE_THRESHOLD": "0.75",
            "BASHGYM_DPPO_RWML_EASY_KEEP_PROBABILITY": "0.25",
            "BASHGYM_DPPO_RWML_HISTORY_WINDOW": "6",
            "BASHGYM_DPPO_RWML_EMBEDDING_MODEL": "qwen3-embedding",
            "BASHGYM_DPPO_RWML_KL_BETA": "0.03",
        }
    )

    assert settings.echo_enabled is True
    assert settings.echo_aux_lambda == 0.07
    assert settings.rwml_enabled is True
    assert settings.rwml_distance_threshold == 0.15
    assert settings.rwml_easy_pass_rate_threshold == 0.75
    assert settings.rwml_easy_keep_probability == 0.25
    assert settings.rwml_history_window == 6
    assert settings.rwml_embedding_model == "qwen3-embedding"
    assert settings.rwml_kl_beta == 0.03


def test_read_replay_jsonl_and_from_env_build_adapter(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps(_record()) + "\n\n", encoding="utf-8")

    records = read_replay_jsonl(replay_path)
    adapter = WorldModelTrainerAdapter.from_env(
        _tokenizer,
        env={
            "BASHGYM_DPPO_REPLAY_PATH": str(replay_path),
            "BASHGYM_DPPO_ECHO_ENABLED": "1",
            "BASHGYM_DPPO_RWML_ENABLED": "1",
        },
        batch_embed_fn=_vectors,
    )

    assert len(records) == 1
    assert adapter.settings.echo_enabled is True
    assert adapter.settings.rwml_enabled is True
    assert adapter.batch.records_with_world_model == 1
    assert adapter.batch.actual_next_states == ("ok!", "hello")


def test_apply_echo_loss_adds_auxiliary_loss_from_replay_masks():
    torch = pytest.importorskip("torch")
    adapter = WorldModelTrainerAdapter.from_records(
        [_record()],
        _tokenizer,
        settings=WorldModelTrainerSettings(echo_enabled=True, echo_aux_lambda=0.05),
    )
    masks = adapter.batch.echo_masks[0]
    vocab = 128
    logits = torch.zeros((len(masks.input_ids), vocab))
    base_loss = torch.tensor(1.0)

    loss = adapter.apply_echo_loss(base_loss, logits)

    assert float(loss) == pytest.approx(1.0 + 0.05 * math.log(vocab), abs=1e-5)


def test_apply_echo_loss_noops_when_echo_disabled():
    torch = pytest.importorskip("torch")
    adapter = WorldModelTrainerAdapter.from_records([_record()], _tokenizer)
    masks = adapter.batch.echo_masks[0]
    logits = torch.zeros((len(masks.input_ids), 128))
    base_loss = torch.tensor(1.0)

    assert adapter.apply_echo_loss(base_loss, logits) is base_loss


def test_score_rwml_predictions_uses_cached_batch_embeddings():
    calls = []

    def batch_embed(texts):
        calls.append(list(texts))
        return _vectors(texts)

    adapter = WorldModelTrainerAdapter.from_records(
        [_record()],
        _tokenizer,
        settings=WorldModelTrainerSettings(rwml_enabled=True, rwml_distance_threshold=0.2),
        batch_embed_fn=batch_embed,
    )

    result = adapter.score_rwml_predictions(["ok!", "miss"])
    result_again = adapter.score_rwml_predictions(["ok!", "miss"])

    assert result.rewards == (1.0, 0.0)
    assert result.pass_rate == 0.5
    assert result.embedding_distance_mean == pytest.approx(0.5)
    assert result.embedding_distance_p95 == pytest.approx(1.0)
    assert result_again.rewards == (1.0, 0.0)
    assert calls == [["ok!", "miss", "hello"]]


def test_trl_reward_function_scores_completion_predictions():
    adapter = WorldModelTrainerAdapter.from_records(
        [_record()],
        _tokenizer,
        settings=WorldModelTrainerSettings(rwml_enabled=True, rwml_distance_threshold=0.2),
        batch_embed_fn=_vectors,
    )
    reward_func = adapter.build_trl_rwml_reward_func()

    rewards = reward_func(
        completions=["same as ok", "miss"],
        actual_next_state=["ok!", "hello"],
    )

    assert rewards == [1.0, 0.0]
    assert reward_func.last_rwml_result["rwml_pass_rate"] == 0.5


def test_verl_reward_function_matches_custom_reward_signature():
    reward_fn = build_verl_rwml_reward_fn(
        batch_embed_fn=_vectors,
        settings=WorldModelTrainerSettings(rwml_enabled=True, rwml_distance_threshold=0.2),
    )

    assert (
        reward_fn(
            "bashgym",
            "same as ok",
            "ignored",
            extra_info={"actual_next_state": "ok!"},
        )
        == 1.0
    )
    assert reward_fn.last_rwml_result["rwml_pass_rate"] == 1.0


def test_score_rwml_prediction_pairs_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        score_rwml_prediction_pairs(
            ["ok!"],
            ["ok!", "hello"],
            CachedEmbeddingProvider(batch_embed_fn=_vectors),
            distance_threshold=0.2,
        )
