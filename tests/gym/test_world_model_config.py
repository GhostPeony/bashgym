"""TrainerConfig plumbing for the ECHO + RWML world-model objectives.

These knobs are exposed as a dedicated ``world_model_settings()`` contract so the
existing ``terminal_rl_settings()`` exact-dict contract stays untouched, mirroring
how DPPO backend selection lives in its own method.
"""

import pytest

from bashgym.gym.echo import ECHO_DEFAULT_LAMBDA
from bashgym.gym.rwml import RWML_DEFAULT_DISTANCE_THRESHOLD
from bashgym.gym.trainer import TrainerConfig


def test_world_model_settings_defaults_are_disabled():
    config = TrainerConfig(base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct")

    assert config.echo_enabled is False
    assert config.echo_aux_lambda == ECHO_DEFAULT_LAMBDA
    assert config.rwml_enabled is False

    settings = config.world_model_settings()
    assert settings == {
        "echo_enabled": False,
        "echo_aux_lambda": ECHO_DEFAULT_LAMBDA,
        "rwml_enabled": False,
        "rwml_distance_threshold": RWML_DEFAULT_DISTANCE_THRESHOLD,
        "rwml_easy_pass_rate_threshold": 0.8,
        "rwml_easy_keep_probability": 0.1,
        "rwml_history_window": 4,
        "rwml_embedding_model": "",
        "rwml_kl_beta": 0.0,
    }


def test_world_model_settings_reflects_overrides():
    config = TrainerConfig(
        base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        echo_enabled=True,
        echo_aux_lambda=0.1,
        rwml_enabled=True,
        rwml_distance_threshold=0.15,
        rwml_easy_pass_rate_threshold=0.7,
        rwml_embedding_model="qwen3-embedding",
        rwml_kl_beta=0.04,
    )

    settings = config.world_model_settings()
    assert settings["echo_enabled"] is True
    assert settings["echo_aux_lambda"] == 0.1
    assert settings["rwml_enabled"] is True
    assert settings["rwml_distance_threshold"] == 0.15
    assert settings["rwml_easy_pass_rate_threshold"] == 0.7
    assert settings["rwml_embedding_model"] == "qwen3-embedding"
    assert settings["rwml_kl_beta"] == 0.04


def test_negative_echo_lambda_rejected():
    with pytest.raises(ValueError, match="echo_aux_lambda"):
        TrainerConfig(base_model="x", echo_aux_lambda=-0.1)


def test_rwml_distance_threshold_must_be_in_cosine_range():
    with pytest.raises(ValueError, match="rwml_distance_threshold"):
        TrainerConfig(base_model="x", rwml_distance_threshold=0.0)
    with pytest.raises(ValueError, match="rwml_distance_threshold"):
        TrainerConfig(base_model="x", rwml_distance_threshold=2.5)


def test_rwml_easy_keep_probability_must_be_a_probability():
    with pytest.raises(ValueError, match="rwml_easy_keep_probability"):
        TrainerConfig(base_model="x", rwml_easy_keep_probability=1.5)


def test_rwml_easy_pass_rate_threshold_must_be_a_probability():
    with pytest.raises(ValueError, match="rwml_easy_pass_rate_threshold"):
        TrainerConfig(base_model="x", rwml_easy_pass_rate_threshold=-0.1)
