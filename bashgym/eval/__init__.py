"""Held-out and executable evaluation APIs with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "bashgym.eval.metrics": ("tool_name_match", "tool_arg_f1", "score_tool_call"),
    "bashgym.eval.stats": ("paired_bootstrap", "BootstrapResult"),
    "bashgym.eval.split": ("make_holdout_split", "HoldoutSplit", "contamination", "example_hash"),
    "bashgym.eval.gate": ("evaluate_gate", "GateThresholds", "GateVerdict"),
    "bashgym.eval.heldout": (
        "ExampleEval",
        "HeldoutReport",
        "score_predictions",
        "run_heldout_eval",
        "evaluate_candidate",
        "first_gold_tool_call",
    ),
    "bashgym.eval.forgetting": (
        "compute_forgetting",
        "ForgettingReport",
        "parse_lm_eval_results",
        "lm_eval_command",
        "DEFAULT_FORGETTING_TASKS",
    ),
    "bashgym.eval.soft": ("soft_call_score", "soft_trajectory_score"),
    "bashgym.eval.passk": (
        "pass_at_k",
        "compute_pass_at_k",
        "evaluate_pass_at_k",
        "EpisodeResult",
        "PassKReport",
    ),
    "bashgym.eval.environment_passk": (
        "EnvironmentAttempt",
        "EnvironmentPassKReport",
        "evaluate_environment_attempts",
        "evaluate_environment_pass_at_k",
    ),
    "bashgym.eval.environment_holdout_comparison": (
        "ENVIRONMENT_HOLDOUT_COMPARISON_SCHEMA_VERSION",
        "evaluate_environment_holdout_comparison_gate",
    ),
    "bashgym.eval.environment_spurious_reward": (
        "ENVIRONMENT_SPURIOUS_REWARD_SCHEMA_VERSION",
        "evaluate_environment_spurious_reward_control",
    ),
    "bashgym.eval.dppo_replay": (
        "DPPO_REPLAY_SCHEMA_VERSION",
        "build_dppo_replay_records",
        "enrich_dppo_replay_jsonl",
        "enrich_dppo_replay_records",
        "read_dppo_replay_jsonl",
        "summarize_dppo_replay_records",
        "summarize_world_model_payloads",
        "write_dppo_records_jsonl",
        "write_dppo_replay_jsonl",
    ),
    "bashgym.eval.predictors": (
        "endpoint_predictor",
        "openai_complete",
        "parse_tool_call",
        "build_prompt_messages",
    ),
    "bashgym.eval.release_gate": (
        "RELEASE_GATE_SCHEMA_VERSION",
        "combine_release_gate_evidence",
    ),
    "bashgym.eval.benchmarks_ext": (
        "BenchmarkResult",
        "BenchmarkReport",
        "BenchmarkSpec",
        "run_benchmarks",
        "record_benchmarks",
        "forgetting_suite_spec",
        "terminal_bench_command",
        "harbor_terminal_bench_command",
        "bfcl_command",
        "swebench_command",
        "parse_bfcl_results",
        "parse_swebench_results",
        "parse_rewardbench_results",
        "parse_cua_rewardbench_results",
        "normalize_external_benchmark_results",
    ),
    "bashgym.eval.service": (
        "EndpointConfig",
        "resolve_endpoint",
        "load_jsonl_examples",
        "run_heldout",
        "thresholds_from",
        "benchmark_commands",
        "ingest_forgetting",
        "ingest_external_benchmarks",
        "record_forgetting",
        "record_external_benchmarks",
        "run_environment_passk",
        "run_environment_holdout_comparison_gate",
        "run_environment_spurious_reward_control",
        "run_local_environment_rollout_passk",
        "run_model_environment_rollout_passk",
        "record_environment_passk",
    ),
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
