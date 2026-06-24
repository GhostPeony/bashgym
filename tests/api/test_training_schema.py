from bashgym.api.schemas import TrainingRequest


def test_training_request_accepts_tmax_like_terminal_rl_profile():
    request = TrainingRequest(
        strategy="grpo",
        training_profile="terminal_rl_tmax_like",
        grpo_num_generations=32,
        grpo_group_size=32,
        prompts_per_rollout_batch=8,
        max_tool_calls_per_episode=64,
        token_level_loss=True,
        filter_zero_std_groups=True,
        active_sampling=True,
        lm_head_fp32=True,
        interleaved_thinking=True,
        sft_warm_start_policy="weak_models_only",
        dppo_backend="grpo_fallback",
        dppo_divergence="binary_kl",
        dppo_binary_tv_threshold=0.12,
        dppo_binary_kl_threshold=0.04,
    )

    assert request.grpo_num_generations == 32
    assert request.grpo_group_size == 32
    assert request.training_profile == "terminal_rl_tmax_like"
    assert request.dppo_backend == "grpo_fallback"
    assert request.dppo_divergence == "binary_kl"
    assert request.dppo_binary_tv_threshold == 0.12
    assert request.dppo_binary_kl_threshold == 0.04


def test_training_request_accepts_world_model_objectives():
    request = TrainingRequest(
        strategy="grpo",
        echo_enabled=True,
        echo_aux_lambda=0.08,
        rwml_enabled=True,
        rwml_distance_threshold=0.18,
        rwml_easy_pass_rate_threshold=0.75,
        rwml_easy_keep_probability=0.2,
        rwml_history_window=6,
        rwml_embedding_model="qwen3-embedding",
        rwml_kl_beta=0.03,
    )

    assert request.echo_enabled is True
    assert request.echo_aux_lambda == 0.08
    assert request.rwml_enabled is True
    assert request.rwml_distance_threshold == 0.18
    assert request.rwml_easy_pass_rate_threshold == 0.75
    assert request.rwml_easy_keep_probability == 0.2
    assert request.rwml_history_window == 6
    assert request.rwml_embedding_model == "qwen3-embedding"
    assert request.rwml_kl_beta == 0.03
