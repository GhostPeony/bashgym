"""Held-out trace evaluation: structured tool-call metrics + session-clustered
paired-bootstrap statistics for answering "is this fine-tune actually better?".
"""

from .benchmarks_ext import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSpec,
    bfcl_command,
    forgetting_suite_spec,
    record_benchmarks,
    run_benchmarks,
    swebench_command,
    terminal_bench_command,
)
from .forgetting import (
    DEFAULT_FORGETTING_TASKS,
    ForgettingReport,
    compute_forgetting,
    lm_eval_command,
    parse_lm_eval_results,
)
from .gate import GateThresholds, GateVerdict, evaluate_gate
from .heldout import (
    ExampleEval,
    HeldoutReport,
    evaluate_candidate,
    first_gold_tool_call,
    run_heldout_eval,
    score_predictions,
)
from .metrics import score_tool_call, tool_arg_f1, tool_name_match
from .passk import (
    EpisodeResult,
    PassKReport,
    compute_pass_at_k,
    evaluate_pass_at_k,
    pass_at_k,
)
from .predictors import build_prompt_messages, endpoint_predictor, openai_complete, parse_tool_call
from .service import (
    EndpointConfig,
    benchmark_commands,
    ingest_forgetting,
    load_jsonl_examples,
    record_forgetting,
    resolve_endpoint,
    run_heldout,
    thresholds_from,
)
from .soft import soft_call_score, soft_trajectory_score
from .split import HoldoutSplit, contamination, example_hash, make_holdout_split
from .stats import BootstrapResult, paired_bootstrap

__all__ = [
    "tool_name_match",
    "tool_arg_f1",
    "score_tool_call",
    "paired_bootstrap",
    "BootstrapResult",
    "make_holdout_split",
    "HoldoutSplit",
    "contamination",
    "example_hash",
    "evaluate_gate",
    "GateThresholds",
    "GateVerdict",
    "ExampleEval",
    "HeldoutReport",
    "score_predictions",
    "run_heldout_eval",
    "evaluate_candidate",
    "first_gold_tool_call",
    "compute_forgetting",
    "ForgettingReport",
    "parse_lm_eval_results",
    "lm_eval_command",
    "DEFAULT_FORGETTING_TASKS",
    "soft_call_score",
    "soft_trajectory_score",
    "pass_at_k",
    "compute_pass_at_k",
    "evaluate_pass_at_k",
    "EpisodeResult",
    "PassKReport",
    "endpoint_predictor",
    "openai_complete",
    "parse_tool_call",
    "build_prompt_messages",
    "BenchmarkResult",
    "BenchmarkReport",
    "BenchmarkSpec",
    "run_benchmarks",
    "record_benchmarks",
    "forgetting_suite_spec",
    "terminal_bench_command",
    "bfcl_command",
    "swebench_command",
    "EndpointConfig",
    "resolve_endpoint",
    "load_jsonl_examples",
    "run_heldout",
    "thresholds_from",
    "benchmark_commands",
    "ingest_forgetting",
    "record_forgetting",
]
