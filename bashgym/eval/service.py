"""Orchestration seam between the eval modules and the API.

The eval modules (``heldout``, ``forgetting``, ``passk``, ``benchmarks_ext``) are
deliberately model-agnostic and hermetic — they take injected predictors and
run-command callables and never touch the network themselves. This module is the
thin layer that turns an API request into those injected seams: it resolves a
served endpoint (a connected OpenAI-compatible provider or an explicit
``base_url``/``model``), builds predictors from it, loads the frozen held-out
set, and runs the held-out gate. The network/predictor factory is itself
injectable so the whole module stays unit-testable with stubs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bashgym.environments.canaries import (
    RewardHackingCanary,
    run_reward_hacking_canaries,
)
from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.rollout import (
    EnvironmentRolloutResult,
    ModelRolloutPlan,
    RolloutCommandPlan,
    run_local_environment_rollouts,
    run_local_model_environment_rollouts,
)
from bashgym.gym.terminal_rl import RewardGroup, active_sample_groups

from .benchmarks_ext import (
    BenchmarkReport,
    bfcl_command,
    harbor_terminal_bench_command,
    normalize_external_benchmark_results,
    swebench_command,
    terminal_bench_command,
)
from .environment_holdout import evaluate_environment_holdout_gate
from .environment_holdout_comparison import evaluate_environment_holdout_comparison_gate
from .environment_passk import (
    EnvironmentAttempt,
    EnvironmentPassKReport,
    evaluate_environment_attempts,
)
from .environment_spurious_reward import evaluate_environment_spurious_reward_control
from .forgetting import (
    DEFAULT_FORGETTING_TASKS,
    ForgettingReport,
    compute_forgetting,
    lm_eval_command,
)
from .gate import GateThresholds
from .heldout import HeldoutReport, evaluate_candidate, first_gold_tool_call
from .predictors import Predictor, endpoint_predictor, openai_complete


@dataclass
class EndpointConfig:
    """A served model the eval can call: an OpenAI-compatible base URL + model."""

    base_url: str
    model: str
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("endpoint requires a base_url")
        if not self.model:
            raise ValueError("endpoint requires a model name")


def resolve_endpoint(
    *,
    provider_registry: Any = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> EndpointConfig:
    """Build an :class:`EndpointConfig` from a connected provider or explicit fields.

    When ``provider`` names a connected provider (registered via
    ``/api/providers/connect``), its ``base_url``/``api_key``/``default_model`` are
    reused so the eval hits the same endpoint the router serves — an explicit
    ``base_url``/``model``/``api_key`` still overrides per field. Raises
    ``ValueError`` if neither path yields a usable endpoint.
    """
    if provider:
        prov = provider_registry.get_provider(provider) if provider_registry else None
        if prov is None:
            raise ValueError(f"provider {provider!r} is not connected")
        base_url = base_url or getattr(prov, "base_url", None)
        api_key = api_key if api_key is not None else getattr(prov, "api_key", None)
        model = model or getattr(prov, "default_model", "") or ""
    return EndpointConfig(base_url=base_url or "", model=model or "", api_key=api_key)


def load_jsonl_examples(path: str | Path, *, limit: int | None = None) -> list[dict]:
    """Read a ``.jsonl`` of NeMo-format examples (the frozen held-out set).

    Blank and non-JSON lines are skipped rather than aborting the load. ``limit``
    caps how many examples are read (useful for a fast smoke eval).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"held-out dataset not found: {p}")
    out: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
                if limit and len(out) >= limit:
                    break
    return out


def _predictor_for(cfg: EndpointConfig) -> Predictor:
    """Default predictor factory: a real OpenAI-compatible endpoint call."""
    return endpoint_predictor(openai_complete(cfg.base_url, cfg.model, api_key=cfg.api_key))


ENVIRONMENT_COMMAND_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run one shell command in the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to execute, or submit when done.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def _environment_completer_for(
    cfg: EndpointConfig,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 60.0,
    use_tool_calling: bool = True,
    capture_logprobs: bool = False,
    top_logprobs: int | None = None,
) -> Callable[[list[dict[str, str]]], Any]:
    """Default served-model command completer for environment rollouts."""
    return openai_complete(
        cfg.base_url,
        cfg.model,
        api_key=cfg.api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        tools=ENVIRONMENT_COMMAND_TOOLS if use_tool_calling else None,
        logprobs=capture_logprobs,
        top_logprobs=top_logprobs,
    )


def thresholds_from(
    *,
    min_trace_delta: float | None = None,
    max_forgetting_drop: float | None = None,
    require_ci_excludes_zero: bool | None = None,
) -> GateThresholds:
    """A :class:`GateThresholds` with any unset field defaulted to the pre-registered value."""
    base = GateThresholds()
    return GateThresholds(
        min_trace_delta=base.min_trace_delta if min_trace_delta is None else min_trace_delta,
        require_ci_excludes_zero=(
            base.require_ci_excludes_zero
            if require_ci_excludes_zero is None
            else require_ci_excludes_zero
        ),
        max_forgetting_drop=(
            base.max_forgetting_drop if max_forgetting_drop is None else max_forgetting_drop
        ),
    )


def run_heldout(
    examples: list[dict],
    base_cfg: EndpointConfig,
    candidate_cfg: EndpointConfig,
    *,
    metric: str = "exact_match",
    thresholds: GateThresholds | None = None,
    forgetting_drops: dict | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    predictor_factory: Callable[[EndpointConfig], Predictor] = _predictor_for,
) -> HeldoutReport:
    """Run the held-out base-vs-candidate gate against two served endpoints.

    ``predictor_factory`` is the seam tests stub to avoid the network — it maps an
    :class:`EndpointConfig` to a sync predictor. Defaults to a real endpoint call.
    """
    if not examples:
        raise ValueError("no held-out examples to evaluate")
    base_predictor = predictor_factory(base_cfg)
    candidate_predictor = predictor_factory(candidate_cfg)
    return evaluate_candidate(
        examples,
        base_predictor,
        candidate_predictor,
        gold_of=first_gold_tool_call,
        metric=metric,
        thresholds=thresholds,
        forgetting_drops=forgetting_drops,
        n_resamples=n_resamples,
        seed=seed,
    )


def benchmark_commands(
    base_url: str,
    model: str,
    *,
    forgetting_tasks: tuple[str, ...] | list[str] | None = None,
    include: tuple[str, ...] = ("forgetting", "terminal_bench", "bfcl", "swebench"),
) -> dict[str, list[str]]:
    """Build the argv for each external benchmark harness against a served endpoint.

    The heavy harnesses (lm-eval, Terminal-Bench, BFCL, SWE-bench) run in the
    serving venv on the host that serves the model — this returns the exact
    commands to run there, so the dashboard can surface them for copy/run.
    """
    cmds: dict[str, list[str]] = {}
    if "forgetting" in include:
        cmds["forgetting"] = lm_eval_command(
            base_url, model_name=model, tasks=forgetting_tasks or DEFAULT_FORGETTING_TASKS
        )
    if "terminal_bench" in include:
        cmds["terminal_bench"] = terminal_bench_command(model)
    if "harbor_terminal_bench" in include:
        cmds["harbor_terminal_bench"] = harbor_terminal_bench_command(model)
    if "bfcl" in include:
        cmds["bfcl"] = bfcl_command(model)
    if "swebench" in include:
        cmds["swebench"] = swebench_command(model)
    return cmds


def ingest_forgetting(base_results: dict, candidate_results: dict) -> ForgettingReport:
    """Diff two lm-eval / NeMo-Evaluator result dicts into a forgetting report."""
    return compute_forgetting(base_results, candidate_results)


def record_forgetting(registry: Any, model_id: str, report: ForgettingReport) -> list[str]:
    """Write the candidate's per-task benchmark scores onto a model profile.

    Returns the task names recorded. The forgetting *drops* feed the deploy gate;
    these per-task scores are what the dashboard renders alongside the verdict.
    """
    recorded: list[str] = []
    for task, score in report.candidate.items():
        if registry.add_benchmark_result(model_id, task, score, 0, 0, {}) is not None:
            recorded.append(task)
    return recorded


def ingest_external_benchmarks(
    results: Any,
    *,
    benchmark_name: str | None = None,
    source: str | None = None,
) -> BenchmarkReport:
    """Normalize external harness output into registry-ready benchmark results."""

    return normalize_external_benchmark_results(
        results,
        benchmark_name=benchmark_name,
        source=source,
    )


def record_external_benchmarks(registry: Any, model_id: str, report: BenchmarkReport) -> list[str]:
    """Write normalized external benchmark scores onto a model profile."""

    recorded: list[str] = []
    for result in report.results:
        if result.error is not None:
            continue
        if (
            registry.add_benchmark_result(
                model_id,
                result.name,
                result.score,
                result.passed,
                result.total,
                result.metrics,
            )
            is not None
        ):
            recorded.append(result.name)
    return recorded


def run_environment_passk(
    environments: list[dict],
    attempts: list[dict],
    *,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
) -> EnvironmentPassKReport:
    """Compute executable-environment pass@k from collected rollout attempts."""
    return evaluate_environment_attempts(
        environments,
        [EnvironmentAttempt.from_dict(attempt) for attempt in attempts],
        k_values=k_values,
    )


def run_environment_holdout_gate(
    environments: list[dict],
    attempts: list[dict],
    *,
    split_by: str = "task_family",
    holdout_fraction: float = 0.2,
    seed: int = 0,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    min_pass_at_1: float = 0.0,
    max_timeout_rate: float = 0.25,
    max_tamper_rate: float = 0.0,
    require_no_contamination: bool = True,
) -> dict[str, Any]:
    """Evaluate pass@k on a deterministic environment holdout split."""

    return evaluate_environment_holdout_gate(
        environments,
        attempts,
        split_by=split_by,
        holdout_fraction=holdout_fraction,
        seed=seed,
        k_values=k_values,
        min_pass_at_1=min_pass_at_1,
        max_timeout_rate=max_timeout_rate,
        max_tamper_rate=max_tamper_rate,
        require_no_contamination=require_no_contamination,
    )


def run_environment_spurious_reward_control(
    environments: list[dict],
    attempts: list[dict],
    *,
    control_attempts: list[dict] | None = None,
    split_by: str = "task_family",
    holdout_fraction: float = 0.2,
    seed: int = 0,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    n_trials: int = 200,
    random_pass_probability: float = 0.05,
    min_observed_pass_at_1: float = 0.0,
    max_control_pass_at_1: float = 0.25,
    min_lift_over_control: float = 0.0,
    require_no_contamination: bool = True,
) -> dict[str, Any]:
    """Run an Olmo-style spurious-reward negative control on the environment holdout."""

    return evaluate_environment_spurious_reward_control(
        environments,
        attempts,
        control_attempts=control_attempts,
        split_by=split_by,
        holdout_fraction=holdout_fraction,
        seed=seed,
        k_values=k_values,
        n_trials=n_trials,
        random_pass_probability=random_pass_probability,
        min_observed_pass_at_1=min_observed_pass_at_1,
        max_control_pass_at_1=max_control_pass_at_1,
        min_lift_over_control=min_lift_over_control,
        require_no_contamination=require_no_contamination,
    )


def run_environment_holdout_comparison_gate(
    environments: list[dict],
    base_attempts: list[dict],
    candidate_attempts: list[dict],
    *,
    split_by: str = "task_family",
    cluster_by: str = "task_family",
    holdout_fraction: float = 0.2,
    seed: int = 0,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    compare_k: int = 1,
    min_delta: float = 0.0,
    min_candidate_pass_at_1: float = 0.0,
    require_ci_excludes_zero: bool = True,
    max_candidate_timeout_rate: float = 0.25,
    max_candidate_tamper_rate: float = 0.0,
    require_no_contamination: bool = True,
    n_resamples: int = 1000,
) -> dict[str, Any]:
    """Compare base and candidate attempts on a deterministic environment holdout."""

    return evaluate_environment_holdout_comparison_gate(
        environments,
        base_attempts,
        candidate_attempts,
        split_by=split_by,
        cluster_by=cluster_by,
        holdout_fraction=holdout_fraction,
        seed=seed,
        k_values=k_values,
        compare_k=compare_k,
        min_delta=min_delta,
        min_candidate_pass_at_1=min_candidate_pass_at_1,
        require_ci_excludes_zero=require_ci_excludes_zero,
        max_candidate_timeout_rate=max_candidate_timeout_rate,
        max_candidate_tamper_rate=max_candidate_tamper_rate,
        require_no_contamination=require_no_contamination,
        n_resamples=n_resamples,
    )


def run_local_environment_rollout_passk(
    environments: list[dict],
    command_attempts: list[dict],
    *,
    workspace_root: str | Path,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    keep_workspace: bool = True,
    allow_dangerous_commands: bool = False,
    stop_on_error: bool = True,
) -> tuple[EnvironmentPassKReport, list[EnvironmentRolloutResult]]:
    """Run local command-script attempts and compute environment pass@k."""
    specs = [EnvironmentSpec.from_dict(environment) for environment in environments]
    by_id = {spec.id: spec for spec in specs}
    plans: list[RolloutCommandPlan] = []
    for raw in command_attempts:
        env_id = str(raw.get("environment_id", ""))
        if env_id not in by_id:
            raise ValueError(f"unknown environment_id: {env_id}")
        commands = raw.get("commands") or []
        if not isinstance(commands, list) or not all(
            isinstance(command, str) for command in commands
        ):
            raise ValueError(f"commands for {env_id} must be a list of strings")
        plans.append(
            RolloutCommandPlan(
                environment=by_id[env_id],
                commands=commands,
                attempt_index=int(raw.get("attempt_index", 0)),
                metadata=dict(raw.get("metadata") or {}),
            )
        )
    rollouts = run_local_environment_rollouts(
        plans,
        workspace_root,
        keep_workspace=keep_workspace,
        allow_dangerous_commands=allow_dangerous_commands,
        stop_on_error=stop_on_error,
    )
    report = evaluate_environment_attempts(
        specs,
        [rollout.attempt.to_dict() for rollout in rollouts],
        k_values=k_values,
    )
    return report, rollouts


def run_reward_hacking_canary_suite(
    *,
    workspace_root: str | Path,
    categories: list[str] | tuple[str, ...] | None = None,
    keep_workspace: bool = True,
) -> tuple[list[RewardHackingCanary], list[EnvironmentRolloutResult], dict[str, Any]]:
    """Run built-in reward-hacking canaries through the environment harness."""

    return run_reward_hacking_canaries(
        workspace_root,
        categories=categories,
        keep_workspace=keep_workspace,
    )


def run_model_environment_rollout_passk(
    environments: list[dict],
    endpoint: EndpointConfig,
    *,
    workspace_root: str | Path,
    attempts_per_environment: int = 1,
    k_values: list[int] | tuple[int, ...] = (1, 4, 8),
    keep_workspace: bool = True,
    allow_dangerous_commands: bool = False,
    stop_on_error: bool = False,
    max_tool_calls: int | None = None,
    max_observation_chars: int = 6000,
    temperature: float = 0.0,
    max_tokens: int = 512,
    request_timeout: float = 60.0,
    use_tool_calling: bool = True,
    capture_logprobs: bool = False,
    top_logprobs: int | None = None,
    filter_zero_std_groups: bool = False,
    active_sampling: bool = False,
    target_prompt_groups: int | None = None,
    complete_factory: (
        Callable[[EndpointConfig], Callable[[list[dict[str, str]]], Any]] | None
    ) = None,
) -> tuple[EnvironmentPassKReport, list[EnvironmentRolloutResult], dict[str, Any] | None]:
    """Run served-model environment attempts and compute verifier-backed pass@k."""
    if attempts_per_environment <= 0:
        raise ValueError("attempts_per_environment must be positive")
    if max_tool_calls is not None and max_tool_calls <= 0:
        raise ValueError("max_tool_calls must be positive")
    if top_logprobs is not None and top_logprobs < 0:
        raise ValueError("top_logprobs must be non-negative")
    if target_prompt_groups is not None and target_prompt_groups <= 0:
        raise ValueError("target_prompt_groups must be positive")
    specs = [EnvironmentSpec.from_dict(environment) for environment in environments]
    complete = (
        complete_factory(endpoint)
        if complete_factory
        else _environment_completer_for(
            endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
            use_tool_calling=use_tool_calling,
            capture_logprobs=capture_logprobs,
            top_logprobs=top_logprobs,
        )
    )
    plans = [
        ModelRolloutPlan(
            environment=spec,
            attempt_index=attempt_index,
            max_tool_calls=max_tool_calls,
            metadata={
                "model": endpoint.model,
                "base_url": endpoint.base_url,
                "rollout_source": "served_model",
                "max_observation_chars": max_observation_chars,
            },
        )
        for spec in specs
        for attempt_index in range(attempts_per_environment)
    ]
    rollouts = run_local_model_environment_rollouts(
        plans,
        workspace_root,
        complete,
        keep_workspace=keep_workspace,
        allow_dangerous_commands=allow_dangerous_commands,
        stop_on_error=stop_on_error,
        max_observation_chars=max_observation_chars,
    )
    sampling_report: dict[str, Any] | None = None
    report_specs = specs
    report_rollouts = rollouts
    if filter_zero_std_groups or active_sampling:
        report_specs, report_rollouts, sampling_report = _apply_active_sampling_to_rollouts(
            specs,
            rollouts,
            target_prompt_groups=target_prompt_groups or len(specs),
        )
    report = evaluate_environment_attempts(
        report_specs,
        [rollout.attempt.to_dict() for rollout in report_rollouts],
        k_values=k_values,
    )
    return report, report_rollouts, sampling_report


def summarize_dppo_readiness(rollouts: list[EnvironmentRolloutResult]) -> dict[str, Any]:
    """Summarize whether rollout records carry the logprobs DPPO needs."""

    attempts = len(rollouts)
    attempts_with_behavior_logprobs = 0
    behavior_logprob_tokens = 0
    attempts_with_train_logprobs = 0
    train_logprob_tokens = 0
    for rollout in rollouts:
        metadata = getattr(rollout.attempt, "metadata", {}) or {}
        behavior_tokens = int(metadata.get("behavior_logprob_tokens") or 0)
        train_tokens = int(metadata.get("train_logprob_tokens") or 0)
        if behavior_tokens > 0:
            attempts_with_behavior_logprobs += 1
            behavior_logprob_tokens += behavior_tokens
        if train_tokens > 0:
            attempts_with_train_logprobs += 1
            train_logprob_tokens += train_tokens

    missing_behavior_attempts = attempts - attempts_with_behavior_logprobs
    missing_train_attempts = attempts - attempts_with_train_logprobs
    return {
        "attempts": attempts,
        "attempts_with_behavior_logprobs": attempts_with_behavior_logprobs,
        "behavior_logprob_tokens": behavior_logprob_tokens,
        "missing_behavior_logprob_attempts": missing_behavior_attempts,
        "attempts_with_train_logprobs": attempts_with_train_logprobs,
        "train_logprob_tokens": train_logprob_tokens,
        "missing_train_logprob_attempts": missing_train_attempts,
        "rollout_logprobs_ready": attempts > 0 and missing_behavior_attempts == 0,
        "optimizer_logprobs_ready": attempts > 0
        and missing_behavior_attempts == 0
        and missing_train_attempts == 0,
        "needs_train_logprob_replay": attempts_with_behavior_logprobs > 0
        and missing_train_attempts > 0,
    }


def _rollout_reward(rollout: EnvironmentRolloutResult) -> float:
    if rollout.attempt.reward is not None:
        return float(rollout.attempt.reward)
    return 1.0 if rollout.attempt.passed else 0.0


def _apply_active_sampling_to_rollouts(
    specs: list[EnvironmentSpec],
    rollouts: list[EnvironmentRolloutResult],
    *,
    target_prompt_groups: int,
) -> tuple[list[EnvironmentSpec], list[EnvironmentRolloutResult], dict[str, Any]]:
    """Filter rollout prompt groups with no reward variance.

    For terminal RL, one environment is the prompt group and repeated attempts are
    the completion rewards. Zero-std groups produce no GRPO advantage, so they are
    excluded from the training/eval batch when filtering is requested.
    """

    by_env: dict[str, list[EnvironmentRolloutResult]] = {spec.id: [] for spec in specs}
    for rollout in rollouts:
        by_env.setdefault(rollout.attempt.environment_id, []).append(rollout)

    groups = [
        RewardGroup(
            prompt_id=spec.id,
            rewards=tuple(_rollout_reward(rollout) for rollout in by_env.get(spec.id, [])),
        )
        for spec in specs
        if by_env.get(spec.id)
    ]
    result = active_sample_groups(groups, target_groups=target_prompt_groups)
    selected_ids = {group.prompt_id for group in result.selected}
    dropped_ids = [group.prompt_id for group in result.dropped]
    if not selected_ids:
        raise ValueError(
            "active sampling dropped every prompt group; reward signal has zero variance"
        )

    group_std = {group.prompt_id: group.std for group in result.selected}
    selected_specs = [spec for spec in specs if spec.id in selected_ids]
    selected_rollouts = [
        rollout for rollout in rollouts if rollout.attempt.environment_id in selected_ids
    ]
    for rollout in selected_rollouts:
        rollout.attempt.metadata["active_sampling_selected"] = True
        rollout.attempt.metadata["reward_group_std"] = group_std.get(
            rollout.attempt.environment_id, 0.0
        )

    telemetry = result.telemetry()
    telemetry.update(
        {
            "sampling_enabled": True,
            "selected_environment_ids": [group.prompt_id for group in result.selected],
            "dropped_environment_ids": dropped_ids,
        }
    )
    return selected_specs, selected_rollouts, telemetry


def record_environment_passk(
    registry: Any, model_id: str, report: EnvironmentPassKReport
) -> list[str]:
    """Record environment pass@k means as model benchmark results."""
    recorded: list[str] = []
    report_dict = report.to_dict()
    for metric_name, score in report.pass_at_k.items():
        name = f"bashgym_env_{metric_name}"
        if (
            registry.add_benchmark_result(
                model_id,
                name,
                score,
                int(round(score * report.n_environments)),
                report.n_environments,
                report_dict,
            )
            is not None
        ):
            recorded.append(name)
    return recorded


def record_environment_holdout_gate(
    registry: Any, model_id: str, result: dict[str, Any]
) -> list[str]:
    """Record an environment holdout gate plus headline holdout pass@k metrics."""

    recorded: list[str] = []
    if getattr(registry, "record_environment_holdout_eval")(model_id, result) is not None:
        recorded.append("bashgym_env_holdout_gate")

    report = result.get("report") or {}
    pass_at_k = report.get("pass_at_k") or {}
    n_environments = int(report.get("n_environments") or 0)
    for metric_name, score in pass_at_k.items():
        score_float = float(score)
        name = f"bashgym_env_holdout_{metric_name}"
        if (
            registry.add_benchmark_result(
                model_id,
                name,
                score_float,
                int(round(score_float * n_environments)),
                n_environments,
                result,
            )
            is not None
        ):
            recorded.append(name)
    return recorded
