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

from .benchmarks_ext import bfcl_command, swebench_command, terminal_bench_command
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
