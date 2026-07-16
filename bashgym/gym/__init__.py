"""Training-layer public API with lazy compatibility exports."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "bashgym.gym.dppo": (
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
    ),
    "bashgym.gym.dppo_backend": (
        "DPPOBackendCapability",
        "DPPOBackendSelection",
        "probe_dppo_backends",
        "select_dppo_backend",
    ),
    "bashgym.gym.dppo_launcher": (
        "DPPOSmokeLaunchConfig",
        "DPPOSmokeLaunchPlan",
        "build_dppo_smoke_launch_plan",
        "prepare_dppo_smoke_launch",
        "run_dppo_smoke_launch",
    ),
    "bashgym.gym.echo": (
        "ACTION_ROLE",
        "OBSERVATION_ROLE",
        "ECHO_DEFAULT_LAMBDA",
        "EchoConfig",
        "EchoSegment",
        "build_echo_masks",
        "combine_echo_loss",
        "environment_prediction_loss",
    ),
    "bashgym.gym.echo_trainer": ("environment_prediction_loss_from_logits",),
    "bashgym.gym.rwml": (
        "RWMLConfig",
        "WorldModelTransition",
        "build_world_model_reward_fn",
        "extract_transitions",
        "group_relative_advantages",
        "keep_easy_sample",
        "world_model_reward",
    ),
    "bashgym.gym.world_model_backend": (
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
    ),
    "bashgym.gym.trainer": (
        "Trainer",
        "TrainerConfig",
        "TrainingRun",
        "TrainingStrategy",
        "GRPOTrainer",
    ),
    "bashgym.gym.environment": (
        "BashGymEnv",
        "GymEnvConfig",
        "Action",
        "ActionType",
        "Observation",
        "BatchGymEnv",
    ),
    "bashgym.gym.data_recipe_search_space": ("DataRecipeSearchSpace",),
    "bashgym.gym.environment_recipe_search_space": ("EnvironmentRecipeSearchSpace",),
    "bashgym.gym.gdpo_adapter": (
        "GDPOBindingReceipt",
        "GDPOComponent",
        "NamedRewardGDPOAdapter",
        "NemoGDPOBatch",
        "NemoGDPOConfig",
    ),
    "bashgym.gym.policy_optimization": (
        "ClippedPolicyObjective",
        "GDPOAdvantageResult",
        "RewardShapingResult",
        "clipped_policy_objective",
        "dapo_overlong_reward",
        "gdpo_advantages",
        "global_normalized_microbatch_sum",
    ),
    "bashgym.gym.router": (
        "ModelRouter",
        "RouterConfig",
        "RoutingStrategy",
        "ModelConfig",
        "ModelType",
    ),
}

_EXPORTS = {name: module_name for module_name, names in _MODULE_EXPORTS.items() for name in names}
_EXPORTS["torch_echo_augmented_loss"] = "bashgym.gym.echo_trainer"


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attribute = "echo_augmented_loss" if name == "torch_echo_augmented_loss" else name
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
