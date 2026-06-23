import pytest

from bashgym.gym.terminal_rl import (
    TERMINAL_RL_TMAX_LIKE_PROFILE,
    RewardGroup,
    active_sample_groups,
    normalize_training_profile,
)
from bashgym.gym.trainer import TrainerConfig


def test_tmax_like_profile_applies_terminal_rl_defaults():
    config = TrainerConfig(training_profile=TERMINAL_RL_TMAX_LIKE_PROFILE)

    assert config.training_profile == TERMINAL_RL_TMAX_LIKE_PROFILE
    assert config.grpo_num_generations == 32
    assert config.effective_grpo_group_size() == 32
    assert config.grpo_loss_type == "dapo"
    assert config.terminal_rl_settings() == {
        "training_profile": TERMINAL_RL_TMAX_LIKE_PROFILE,
        "grpo_group_size": 32,
        "prompts_per_rollout_batch": 8,
        "max_tool_calls_per_episode": 64,
        "token_level_loss": True,
        "filter_zero_std_groups": True,
        "active_sampling": True,
        "lm_head_fp32": True,
        "interleaved_thinking": True,
        "sft_warm_start_policy": "weak_models_only",
        "dppo_backend": "auto",
        "dppo_divergence": "binary_tv",
        "dppo_binary_tv_threshold": 0.15,
        "dppo_binary_kl_threshold": 0.05,
    }
    assert config.terminal_rl_warnings() == []


def test_dppo_backend_selection_reports_fallback_status():
    config = TrainerConfig(dppo_backend="grpo_fallback")

    selection = config.dppo_backend_selection()

    assert selection["requested"] == "grpo_fallback"
    assert selection["selected"] == "grpo_fallback"
    assert selection["fallback_to_grpo"] is True


def test_tmax_like_profile_preserves_direct_overrides():
    config = TrainerConfig(
        training_profile=TERMINAL_RL_TMAX_LIKE_PROFILE,
        grpo_group_size=12,
        grpo_loss_type="gspo",
        active_sampling=False,
    )

    assert config.grpo_num_generations == 12
    assert config.grpo_loss_type == "gspo"
    assert "grpo_group_size >= 16" in " ".join(config.terminal_rl_warnings())
    assert "active sampling is disabled" in " ".join(config.terminal_rl_warnings())


def test_normalize_training_profile_accepts_aliases():
    assert normalize_training_profile("default") == "default"
    assert normalize_training_profile("tmax") == TERMINAL_RL_TMAX_LIKE_PROFILE
    assert normalize_training_profile("terminal_rl") == TERMINAL_RL_TMAX_LIKE_PROFILE
    with pytest.raises(ValueError, match="training_profile"):
        normalize_training_profile("surprise")


def test_active_sampling_refills_zero_std_groups_to_maintain_batch():
    result = active_sample_groups(
        [
            RewardGroup("all-fail", (0.0, 0.0, 0.0, 0.0)),
            RewardGroup("mixed-a", (0.0, 1.0, 0.0, 1.0)),
            RewardGroup("all-pass", (1.0, 1.0, 1.0, 1.0)),
            RewardGroup("mixed-b", (0.25, 0.5, 0.75, 1.0)),
            RewardGroup("mixed-c", (0.0, 0.0, 1.0, 1.0)),
        ],
        target_groups=3,
    )

    assert [group.prompt_id for group in result.selected] == ["mixed-a", "mixed-b", "mixed-c"]
    assert [group.prompt_id for group in result.dropped] == ["all-fail", "all-pass"]
    assert result.maintained_batch is True
    assert result.active_sampling_refills == 2
    assert result.zero_std_groups_dropped == 2
    assert result.all_zero_groups_dropped == 1
    assert result.all_one_groups_dropped == 1
    assert result.telemetry()["effective_prompt_groups"] == 3


def test_active_sampling_reports_when_candidates_cannot_maintain_batch():
    result = active_sample_groups(
        [
            RewardGroup("all-fail", (0.0, 0.0)),
            RewardGroup("mixed", (0.0, 1.0)),
        ],
        target_groups=2,
    )

    assert result.maintained_batch is False
    assert result.effective_groups == 1
    assert result.zero_std_groups_dropped == 1
