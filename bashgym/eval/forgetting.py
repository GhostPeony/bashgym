"""Forgetting / regression eval: did fine-tuning degrade general capability?

A fine-tune that improves on our traces but regresses on broad benchmarks
(MMLU, GSM8K, IFEval, HumanEval) has overfit — it gained narrow skill and lost
general capability. This module turns the JSON output of NVIDIA NeMo Evaluator /
EleutherAI lm-evaluation-harness (base vs candidate) into the ``forgetting_drops``
the deploy gate already consumes, so a capability regression blocks deployment.

Parsing follows lm-eval's results schema (also what NeMo Evaluator emits):
    {"results": {"<task>": {"<metric>,<filter>": <float>, "<metric>_stderr,<filter>": ...}}}

The actual benchmark *run* happens in the serving venv on the Spark (lm-eval +
the vLLM ``local-completions`` endpoint); this module stays import-light and
hermetic — it builds the command and parses/diffs the results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The "forgetting suite": broad capabilities a coding fine-tune must not regress.
DEFAULT_FORGETTING_TASKS: tuple[str, ...] = ("mmlu", "gsm8k", "ifeval", "hellaswag")

# Preferred headline metric per task, in priority order (lm-eval reports several).
_METRIC_PRIORITY = ("exact_match", "pass@1", "acc_norm", "acc", "f1", "score", "mc2")


def _primary_metric(task_scores: dict) -> float | None:
    """Pick the headline metric from lm-eval's {'metric,filter': value} dict for one task."""
    cleaned: dict[str, float] = {}
    for key, val in task_scores.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        name = str(key).split(",")[0]  # strip lm-eval's ",<filter>" suffix
        if name.endswith("_stderr") or name == "alias":
            continue
        cleaned.setdefault(name, float(val))
    for pref in _METRIC_PRIORITY:
        if pref in cleaned:
            return cleaned[pref]
    return next(iter(cleaned.values()), None)  # fall back to first numeric metric


def parse_lm_eval_results(results_json: dict) -> dict[str, float]:
    """Extract {task: headline_score} from a NeMo Evaluator / lm-eval results dict."""
    results = results_json.get("results", results_json)
    if not isinstance(results, dict):
        return {}
    scores: dict[str, float] = {}
    for task, task_scores in results.items():
        if isinstance(task_scores, dict):
            m = _primary_metric(task_scores)
            if m is not None:
                scores[str(task)] = m
    return scores


def _as_scores(obj: dict) -> dict[str, float]:
    """Normalize raw lm-eval JSON OR an already-parsed {task: score} dict to {task: score}."""
    if "results" in obj and isinstance(obj["results"], dict):
        return parse_lm_eval_results(obj)
    if obj and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj.values()):
        return {str(k): float(v) for k, v in obj.items()}
    return parse_lm_eval_results({"results": obj})  # bare {task: {metric: val}} map


@dataclass
class ForgettingReport:
    drops: dict[str, float]  # task -> (base - candidate); positive = candidate regressed
    base: dict[str, float] = field(default_factory=dict)
    candidate: dict[str, float] = field(default_factory=dict)

    @property
    def worst(self) -> tuple[str, float] | None:
        """(task, drop) with the largest regression, or None if no shared tasks."""
        return max(self.drops.items(), key=lambda kv: kv[1]) if self.drops else None

    @property
    def regressed(self) -> dict[str, float]:
        """Only the tasks where the candidate got worse (drop > 0)."""
        return {k: v for k, v in self.drops.items() if v > 0}

    def to_dict(self) -> dict:
        return {
            "drops": self.drops,
            "base": self.base,
            "candidate": self.candidate,
            "worst": list(self.worst) if self.worst else None,
            "regressed": self.regressed,
        }


def compute_forgetting(base_results: dict, candidate_results: dict) -> ForgettingReport:
    """Per-task forgetting drops from two lm-eval / NeMo-Evaluator result dicts.

    Each argument may be raw lm-eval JSON (top-level ``results`` key), a bare
    ``{task: {metric: value}}`` map, or an already-parsed ``{task: score}`` dict.
    ``drop = base - candidate`` (positive = the candidate regressed). Only tasks
    present in BOTH are compared. The ``.drops`` dict is what the deploy gate's
    ``forgetting_drops`` parameter expects.
    """
    base = _as_scores(base_results)
    cand = _as_scores(candidate_results)
    drops = {t: round(base[t] - cand[t], 6) for t in base if t in cand}
    return ForgettingReport(drops=drops, base=base, candidate=cand)


def lm_eval_command(
    base_url: str,
    *,
    model_name: str = "candidate",
    tasks: tuple[str, ...] | list[str] = DEFAULT_FORGETTING_TASKS,
    num_fewshot: int | None = None,
    limit: int | None = None,
    num_concurrent: int = 8,
) -> list[str]:
    """Build the lm-eval CLI argv to evaluate a vLLM-served model via ``local-completions``.

    Returned as an argv list (not executed here — run it in ``~/bashgym-serve`` on
    the Spark, where lm-eval and the vLLM endpoint live). ``local-completions`` is
    the OpenAI-compatible adapter NeMo Evaluator/lm-eval use to hit a served model.
    """
    args = [
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        f"base_url={base_url},model={model_name},num_concurrent={num_concurrent}",
        "--tasks",
        ",".join(tasks),
        "--output_path",
        f"forgetting_{model_name}.json",
    ]
    if num_fewshot is not None:
        args += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        args += ["--limit", str(limit)]
    return args
