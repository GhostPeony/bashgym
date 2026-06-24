"""Executable terminal-environment artifacts for BashGym.

This package is the bridge between trace-shaped training data and TMax-style
RL environments: task instructions, files, fixtures, build context, verifier,
rollout limits, and metrics all share one serializable contract.
"""

from bashgym.environments.builder import EnvironmentBuild, materialize_environment
from bashgym.environments.canaries import (
    RewardHackingCanary,
    reward_hacking_canaries,
    run_reward_hacking_canaries,
    summarize_reward_hacking_canaries,
)
from bashgym.environments.contracts import (
    BuildSpec,
    EnvironmentAxis,
    EnvironmentSpec,
    FixtureSpec,
    RolloutSpec,
    VerifierSpec,
)
from bashgym.environments.decontaminate import (
    environment_text,
    filter_contaminated_environments,
)
from bashgym.environments.loader import (
    environment_from_record,
    load_environment,
    load_environments,
    save_environment,
)
from bashgym.environments.metrics import (
    EnvironmentMixReport,
    axis_distribution,
    balance_score,
    summarize_environment_mix,
)
from bashgym.environments.rollout import (
    CommandObservation,
    EnvironmentRolloutResult,
    LocalPersistentShell,
    ModelCompleter,
    ModelRolloutPlan,
    RolloutAttempt,
    RolloutCommandPlan,
    build_environment_rollout_messages,
    parse_shell_command_response,
    run_local_environment_attempt,
    run_local_environment_rollouts,
    run_local_model_environment_attempt,
    run_local_model_environment_rollouts,
)
from bashgym.environments.tmax_importer import TMAX_HF_DATASETS, TMaxImporter

__all__ = [
    "BuildSpec",
    "CommandObservation",
    "EnvironmentAxis",
    "EnvironmentBuild",
    "EnvironmentMixReport",
    "EnvironmentRolloutResult",
    "EnvironmentSpec",
    "FixtureSpec",
    "LocalPersistentShell",
    "ModelCompleter",
    "ModelRolloutPlan",
    "RewardHackingCanary",
    "RolloutAttempt",
    "RolloutSpec",
    "RolloutCommandPlan",
    "TMAX_HF_DATASETS",
    "TMaxImporter",
    "VerifierSpec",
    "axis_distribution",
    "balance_score",
    "build_environment_rollout_messages",
    "environment_from_record",
    "environment_text",
    "filter_contaminated_environments",
    "load_environment",
    "load_environments",
    "materialize_environment",
    "parse_shell_command_response",
    "reward_hacking_canaries",
    "run_local_environment_attempt",
    "run_local_environment_rollouts",
    "run_local_model_environment_attempt",
    "run_local_model_environment_rollouts",
    "run_reward_hacking_canaries",
    "save_environment",
    "summarize_environment_mix",
    "summarize_reward_hacking_canaries",
]
