"""Executable environment APIs with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "bashgym.environments.contracts": (
        "BuildSpec",
        "EnvironmentAxis",
        "EnvironmentSpec",
        "FixtureSpec",
        "RewardComponentSpec",
        "RolloutSpec",
        "VerifierSpec",
    ),
    "bashgym.environments.builder": ("EnvironmentBuild", "materialize_environment"),
    "bashgym.environments.canaries": (
        "RewardHackingCanary",
        "reward_hacking_canaries",
        "run_reward_hacking_canaries",
        "summarize_reward_hacking_canaries",
    ),
    "bashgym.environments.decontaminate": (
        "environment_text",
        "filter_contaminated_environments",
    ),
    "bashgym.environments.loader": (
        "environment_from_record",
        "load_environment",
        "load_environments",
        "save_environment",
    ),
    "bashgym.environments.metrics": (
        "EnvironmentMixReport",
        "axis_distribution",
        "balance_score",
        "summarize_environment_mix",
    ),
    "bashgym.environments.nemo_gym": (
        "NemoGymMessageTokenEvidence",
        "NemoGymRefitReceipt",
        "NemoGymRolloutEvidence",
        "assert_message_token_evidence_preserved",
        "create_nemo_gym_bundle_archive",
        "export_star_count_nemo_gym_bundle",
        "extract_nemo_gym_bundle_archive",
        "inspect_nemo_gym_bundle_archive",
        "score_star_count_nemo_response",
        "validate_nemo_gym_rollout_batch",
    ),
    "bashgym.environments.rollout": (
        "CommandObservation",
        "EnvironmentRolloutResult",
        "LocalPersistentShell",
        "ModelCompleter",
        "ModelRolloutPlan",
        "RolloutAttempt",
        "RolloutCommandPlan",
        "build_environment_rollout_messages",
        "extract_verifier_rewards",
        "parse_shell_command_response",
        "run_local_environment_attempt",
        "run_local_environment_rollouts",
        "run_local_model_environment_attempt",
        "run_local_model_environment_rollouts",
    ),
    "bashgym.environments.star_count": (
        "STAR_COUNT_COLORS",
        "STAR_COUNT_PROMPT",
        "StarCountScore",
        "canonical_star_count_answer",
        "create_star_count_archive",
        "generate_star_count_dataset",
        "parse_star_count_prediction",
        "score_star_count_prediction",
        "star_count_environment_spec",
    ),
    "bashgym.environments.tmax_importer": ("TMAX_HF_DATASETS", "TMaxImporter"),
}
_EXPORTS = {name: module_name for module_name, names in _MODULE_EXPORTS.items() for name in names}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
