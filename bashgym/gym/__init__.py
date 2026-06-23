"""Gym - Training Layer"""

from bashgym.gym.dppo import (
    DPPO_BINARY_KL_THRESHOLD,
    DPPO_BINARY_TV_THRESHOLD,
    DPPOMaskDecision,
    DPPOTelemetry,
    DPPOToken,
    analyze_dppo_batch,
    binary_kl_divergence,
    binary_tv_divergence,
    dppo_mask_decision,
    policy_ratio,
    probability_from_logprob,
)
from bashgym.gym.dppo_backend import (
    DPPOBackendCapability,
    DPPOBackendSelection,
    probe_dppo_backends,
    select_dppo_backend,
)
from bashgym.gym.dppo_launcher import (
    DPPOSmokeLaunchConfig,
    DPPOSmokeLaunchPlan,
    build_dppo_smoke_launch_plan,
    prepare_dppo_smoke_launch,
    run_dppo_smoke_launch,
)
from bashgym.gym.echo import (
    ACTION_ROLE,
    ECHO_DEFAULT_LAMBDA,
    OBSERVATION_ROLE,
    EchoConfig,
    EchoSegment,
    build_echo_masks,
    combine_echo_loss,
    environment_prediction_loss,
)
from bashgym.gym.echo_trainer import (
    echo_augmented_loss as torch_echo_augmented_loss,
)
from bashgym.gym.echo_trainer import (
    environment_prediction_loss_from_logits,
)
from bashgym.gym.environment import (
    Action,
    ActionType,
    BashGymEnv,
    BatchGymEnv,
    GymEnvConfig,
    Observation,
)
from bashgym.gym.environment_recipe_search_space import EnvironmentRecipeSearchSpace
from bashgym.gym.router import ModelConfig, ModelRouter, ModelType, RouterConfig, RoutingStrategy
from bashgym.gym.rwml import (
    RWMLConfig,
    WorldModelTransition,
    build_world_model_reward_fn,
    extract_transitions,
    group_relative_advantages,
    keep_easy_sample,
    world_model_reward,
)
from bashgym.gym.trainer import GRPOTrainer, Trainer, TrainerConfig, TrainingRun, TrainingStrategy
from bashgym.gym.world_model_backend import (
    CachedEmbeddingProvider,
    WorldModelBackendBatch,
    WorldModelTrainerAdapter,
    WorldModelTrainerSettings,
    build_trl_rwml_reward_func,
    build_verl_rwml_reward_fn,
    build_world_model_backend_batch,
    echo_masks_from_replay_record,
    echo_segments_from_replay_record,
    read_replay_jsonl,
    rwml_rewards_from_predictions,
    rwml_transitions_from_replay_record,
    score_rwml_prediction_pairs,
)

__all__ = [
    # DPPO
    "DPPO_BINARY_KL_THRESHOLD",
    "DPPO_BINARY_TV_THRESHOLD",
    "DPPOMaskDecision",
    "DPPOTelemetry",
    "DPPOToken",
    "analyze_dppo_batch",
    "binary_kl_divergence",
    "binary_tv_divergence",
    "dppo_mask_decision",
    "policy_ratio",
    "probability_from_logprob",
    "DPPOBackendCapability",
    "DPPOBackendSelection",
    "probe_dppo_backends",
    "select_dppo_backend",
    "DPPOSmokeLaunchConfig",
    "DPPOSmokeLaunchPlan",
    "build_dppo_smoke_launch_plan",
    "prepare_dppo_smoke_launch",
    "run_dppo_smoke_launch",
    # ECHO/RWML world-model objectives
    "ACTION_ROLE",
    "OBSERVATION_ROLE",
    "ECHO_DEFAULT_LAMBDA",
    "EchoConfig",
    "EchoSegment",
    "build_echo_masks",
    "combine_echo_loss",
    "environment_prediction_loss",
    "torch_echo_augmented_loss",
    "environment_prediction_loss_from_logits",
    "RWMLConfig",
    "WorldModelTransition",
    "build_world_model_reward_fn",
    "extract_transitions",
    "group_relative_advantages",
    "keep_easy_sample",
    "world_model_reward",
    "CachedEmbeddingProvider",
    "WorldModelBackendBatch",
    "WorldModelTrainerAdapter",
    "WorldModelTrainerSettings",
    "build_trl_rwml_reward_func",
    "build_verl_rwml_reward_fn",
    "build_world_model_backend_batch",
    "echo_masks_from_replay_record",
    "echo_segments_from_replay_record",
    "read_replay_jsonl",
    "rwml_rewards_from_predictions",
    "rwml_transitions_from_replay_record",
    "score_rwml_prediction_pairs",
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainingRun",
    "TrainingStrategy",
    "GRPOTrainer",
    # Environment
    "BashGymEnv",
    "GymEnvConfig",
    "Action",
    "ActionType",
    "Observation",
    "BatchGymEnv",
    "EnvironmentRecipeSearchSpace",
    # Router
    "ModelRouter",
    "RouterConfig",
    "RoutingStrategy",
    "ModelConfig",
    "ModelType",
]
