"""Held-out trace eval runner: wire metrics + clustered bootstrap + deploy gate
into one model-agnostic "is the candidate actually better than the base?" answer.

The runner never calls a model itself. Callers inject two ``predictor`` callables
(base and candidate) that map a held-out example to a predicted tool call — so the
same runner works for any provider (Anthropic, NIM, Ollama, a local checkpoint) and
stays trivially testable with stub predictors. Scoring is per-example against the
gold tool call; per-example deltas are bootstrapped clustered by session (segments
from one session are correlated), and the result is run through the pre-registered
deploy gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .gate import GateThresholds, GateVerdict, evaluate_gate
from .metrics import score_tool_call
from .split import _group_key, example_hash
from .stats import BootstrapResult, paired_bootstrap

# A predictor maps a held-out example to its predicted tool call (OpenAI-style or flat).
Predictor = Callable[[dict], dict]

_METRICS = ("exact_match", "name_match", "arg_f1", "soft")


def first_gold_tool_call(example: dict) -> dict:
    """Default gold extractor: the first assistant tool call in a NeMo-format example.

    Returns ``{}`` when the example has no assistant tool call (those examples score
    0/0 and contribute a zero delta rather than crashing the run).
    """
    for msg in example.get("messages", []) or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls")
        if isinstance(calls, list) and calls:
            first = calls[0]
            if isinstance(first, dict):
                return first
    return {}


def _metric_score(predicted: dict, gold: dict, metric: str) -> float:
    """Scalar in [0, 1] for one predicted/gold pair under the chosen metric."""
    if metric == "soft":
        from .soft import soft_call_score  # SERA-style graded partial credit

        return soft_call_score(predicted, gold)
    scores = score_tool_call(predicted, gold)
    return float(scores[metric])


@dataclass
class ExampleEval:
    """Per-example base-vs-candidate scores, tagged with its bootstrap cluster."""

    example_id: str
    session_id: str
    base_score: float
    candidate_score: float

    @property
    def delta(self) -> float:
        return self.candidate_score - self.base_score


@dataclass
class HeldoutReport:
    n: int
    n_clusters: int
    base_pass_rate: float
    candidate_pass_rate: float
    trace_delta: float
    metric: str
    bootstrap: BootstrapResult
    verdict: GateVerdict
    forgetting_drops: dict = field(default_factory=dict)

    @property
    def ship(self) -> bool:
        return self.verdict.ship

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_clusters": self.n_clusters,
            "metric": self.metric,
            "base_pass_rate": self.base_pass_rate,
            "candidate_pass_rate": self.candidate_pass_rate,
            "trace_delta": self.trace_delta,
            "bootstrap": {
                "mean": self.bootstrap.mean,
                "ci_low": self.bootstrap.ci_low,
                "ci_high": self.bootstrap.ci_high,
                "significant": self.bootstrap.significant,
                "better": self.bootstrap.better,
                "n_clusters": self.bootstrap.n_clusters,
            },
            "forgetting_drops": self.forgetting_drops,
            "ship": self.verdict.ship,
            "reasons": self.verdict.reasons,
        }


def score_predictions(
    examples: list[dict],
    base_predictions: list[dict],
    candidate_predictions: list[dict],
    *,
    gold_of: Callable[[dict], dict] = first_gold_tool_call,
    metric: str = "exact_match",
) -> list[ExampleEval]:
    """Score aligned base/candidate predictions against each example's gold call.

    ``examples[i]``, ``base_predictions[i]`` and ``candidate_predictions[i]`` must be
    the same example. Returns one :class:`ExampleEval` per example.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    if not (len(examples) == len(base_predictions) == len(candidate_predictions)):
        raise ValueError("examples, base_predictions and candidate_predictions must align")

    evals: list[ExampleEval] = []
    for ex, base_pred, cand_pred in zip(examples, base_predictions, candidate_predictions):
        gold = gold_of(ex) or {}
        evals.append(
            ExampleEval(
                example_id=example_hash(ex),
                session_id=_group_key(ex, "session"),
                base_score=_metric_score(base_pred or {}, gold, metric),
                candidate_score=_metric_score(cand_pred or {}, gold, metric),
            )
        )
    return evals


def run_heldout_eval(
    evals: list[ExampleEval],
    *,
    metric: str = "exact_match",
    thresholds: GateThresholds | None = None,
    forgetting_drops: dict | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> HeldoutReport:
    """Aggregate per-example scores into pass rates, a clustered CI, and a gate verdict."""
    if not evals:
        raise ValueError("no evaluated examples")

    deltas = [e.delta for e in evals]
    clusters = [e.session_id for e in evals]
    base_rate = sum(e.base_score for e in evals) / len(evals)
    cand_rate = sum(e.candidate_score for e in evals) / len(evals)
    trace_delta = cand_rate - base_rate

    boot = paired_bootstrap(deltas, clusters, n_resamples=n_resamples, seed=seed)
    verdict = evaluate_gate(
        trace_delta=trace_delta,
        ci_low=boot.ci_low,
        ci_excludes_zero=boot.significant,
        forgetting_drops=forgetting_drops,
        thresholds=thresholds,
    )
    return HeldoutReport(
        n=len(evals),
        n_clusters=boot.n_clusters,
        base_pass_rate=base_rate,
        candidate_pass_rate=cand_rate,
        trace_delta=trace_delta,
        metric=metric,
        bootstrap=boot,
        verdict=verdict,
        forgetting_drops=forgetting_drops or {},
    )


def evaluate_candidate(
    examples: list[dict],
    base_predictor: Predictor,
    candidate_predictor: Predictor,
    *,
    gold_of: Callable[[dict], dict] = first_gold_tool_call,
    metric: str = "exact_match",
    thresholds: GateThresholds | None = None,
    forgetting_drops: dict | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> HeldoutReport:
    """End-to-end convenience: collect predictions from injected predictors, score, gate.

    Keeps the module free of any network/provider code — the predictors are the seam
    where a caller plugs in Ollama, NIM, the Anthropic teacher, or a stub in tests.
    """
    base_predictions = [base_predictor(ex) for ex in examples]
    candidate_predictions = [candidate_predictor(ex) for ex in examples]
    evals = score_predictions(
        examples,
        base_predictions,
        candidate_predictions,
        gold_of=gold_of,
        metric=metric,
    )
    return run_heldout_eval(
        evals,
        metric=metric,
        thresholds=thresholds,
        forgetting_drops=forgetting_drops,
        n_resamples=n_resamples,
        seed=seed,
    )
