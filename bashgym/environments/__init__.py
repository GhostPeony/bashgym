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
    RewardComponentSpec,
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
from bashgym.environments.nemo_gym import (
    NemoGymMessageTokenEvidence,
    NemoGymRefitReceipt,
    NemoGymRolloutEvidence,
    assert_message_token_evidence_preserved,
    export_star_count_nemo_gym_bundle,
    score_star_count_nemo_response,
    validate_nemo_gym_rollout_batch,
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
    extract_verifier_rewards,
    parse_shell_command_response,
    run_local_environment_attempt,
    run_local_environment_rollouts,
    run_local_model_environment_attempt,
    run_local_model_environment_rollouts,
)
from bashgym.environments.star_count import (
    STAR_COUNT_COLORS,
    STAR_COUNT_PROMPT,
    StarCountScore,
    canonical_star_count_answer,
    create_star_count_archive,
    generate_star_count_dataset,
    parse_star_count_prediction,
    score_star_count_prediction,
    star_count_environment_spec,
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
    "NemoGymMessageTokenEvidence",
    "NemoGymRefitReceipt",
    "NemoGymRolloutEvidence",
    "RewardHackingCanary",
    "RewardComponentSpec",
    "RolloutAttempt",
    "RolloutSpec",
    "RolloutCommandPlan",
    "STAR_COUNT_COLORS",
    "STAR_COUNT_PROMPT",
    "StarCountScore",
    "TMAX_HF_DATASETS",
    "TMaxImporter",
    "VerifierSpec",
    "axis_distribution",
    "assert_message_token_evidence_preserved",
    "balance_score",
    "build_environment_rollout_messages",
    "canonical_star_count_answer",
    "create_star_count_archive",
    "environment_from_record",
    "environment_text",
    "extract_verifier_rewards",
    "export_star_count_nemo_gym_bundle",
    "filter_contaminated_environments",
    "generate_star_count_dataset",
    "load_environment",
    "load_environments",
    "materialize_environment",
    "parse_shell_command_response",
    "parse_star_count_prediction",
    "reward_hacking_canaries",
    "run_local_environment_attempt",
    "run_local_environment_rollouts",
    "run_local_model_environment_attempt",
    "run_local_model_environment_rollouts",
    "run_reward_hacking_canaries",
    "save_environment",
    "score_star_count_prediction",
    "score_star_count_nemo_response",
    "star_count_environment_spec",
    "summarize_environment_mix",
    "summarize_reward_hacking_canaries",
    "validate_nemo_gym_rollout_batch",
]
