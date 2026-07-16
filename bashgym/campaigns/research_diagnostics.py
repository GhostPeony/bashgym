"""Deterministic diagnostics derived from sealed AutoResearch evidence.

The projection in this module is deliberately advisory.  It does not mutate a
campaign, submit a proposal, or promote a model.  Re-reading the same immutable
outcomes and ledger records produces the same diagnostics and ranked hypotheses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

from bashgym.campaigns.contracts import (
    FrozenContractModel,
    Identifier,
    canonical_hash,
)

DiagnosticSeverity = Literal["info", "warning", "critical"]
DiagnosticDirection = Literal["maximize", "minimize", "unknown"]
SliceStatus = Literal["observed", "improved", "regressed", "unchanged"]
HypothesisAction = Literal["diagnostic", "evaluation", "candidate"]


class AutoResearchDiagnosticSignal(FrozenContractModel):
    code: Identifier
    severity: DiagnosticSeverity
    summary: str = Field(min_length=1, max_length=1000)
    evidence_references: tuple[Identifier, ...] = ()


class AutoResearchCheckpointComparison(FrozenContractModel):
    evaluation_result_id: Identifier
    run_id: Identifier
    role: Literal["checkpoint", "final"]
    step: int | None = Field(default=None, ge=0)
    metric_name: Identifier
    metric_value: float
    improvement_from_previous: float | None = None
    improvement_from_baseline: float | None = None


class AutoResearchErrorSlice(FrozenContractModel):
    slice_path: str = Field(min_length=1, max_length=1000)
    direction: DiagnosticDirection
    candidate_value: float
    baseline_value: float | None = None
    improvement: float | None = None
    status: SliceStatus
    evidence_references: tuple[Identifier, ...] = ()


class AutoResearchRankedHypothesis(FrozenContractModel):
    hypothesis_id: Identifier
    rank: int = Field(ge=1)
    action_kind: HypothesisAction
    changed_variable: str = Field(min_length=1, max_length=240)
    hypothesis: str = Field(min_length=1, max_length=2000)
    rationale: str = Field(min_length=1, max_length=4000)
    expected_outcome: str = Field(min_length=1, max_length=2000)
    falsification_criterion: str = Field(min_length=1, max_length=2000)
    evidence_references: tuple[Identifier, ...] = ()
    eligible_for_submission: bool = False


class AutoResearchDiagnostics(FrozenContractModel):
    schema_version: Literal["autoresearch_diagnostics.v1"] = "autoresearch_diagnostics.v1"
    workspace_id: Identifier
    campaign_id: Identifier
    primary_metric: Identifier
    metric_direction: Literal["maximize", "minimize"]
    low_signal: bool
    signals: tuple[AutoResearchDiagnosticSignal, ...] = ()
    checkpoint_comparisons: tuple[AutoResearchCheckpointComparison, ...] = ()
    error_slices: tuple[AutoResearchErrorSlice, ...] = ()
    ranked_hypotheses: tuple[AutoResearchRankedHypothesis, ...] = ()


_SLICE_CONTROL_KEYS = frozenset(
    {
        "_metric_directions",
        "autoresearch_role",
        "baseline_delta",
        "checkpoint",
        "checkpoint_step",
        "degenerate_constant_output",
        "example_count",
        "modal_prediction_fraction",
        "unique_prediction_count",
    }
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _signed_improvement(candidate: float, baseline: float, direction: str) -> float:
    return candidate - baseline if direction == "maximize" else baseline - candidate


def _evaluation_id(evaluation: Mapping[str, Any]) -> str:
    return str(evaluation.get("evaluation_result_id") or "")


def _outcome_evaluation(
    outcome: Mapping[str, Any] | None,
    evaluations_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if outcome is None:
        return None
    references = outcome.get("result", {}).get("evidence_references", ())
    for reference in references:
        evaluation = evaluations_by_id.get(str(reference))
        if evaluation is not None:
            return evaluation
    return None


def _checkpoint_step(evaluation: Mapping[str, Any], run: Mapping[str, Any]) -> int | None:
    slices = evaluation.get("slice_metrics") or {}
    candidates = [slices.get("checkpoint_step")]
    checkpoint = slices.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        candidates.append(checkpoint.get("step"))
    config = run.get("config") or {}
    if isinstance(config, Mapping):
        candidates.append(config.get("checkpoint_step"))
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _flatten_numeric_slices(
    value: Mapping[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key in sorted(value):
        if key in _SLICE_CONTROL_KEYS:
            continue
        item = value[key]
        path = (*prefix, str(key))
        if isinstance(item, Mapping):
            flattened.update(_flatten_numeric_slices(item, prefix=path))
            continue
        numeric = _finite_number(item)
        if numeric is not None:
            flattened[".".join(path)] = numeric
    return flattened


def _slice_direction(path: str, explicit: Mapping[str, Any]) -> DiagnosticDirection:
    configured = explicit.get(path)
    if configured in {"maximize", "minimize"}:
        return configured
    leaf = path.rsplit(".", 1)[-1].lower()
    if any(token in leaf for token in ("error", "loss", "latency", "mae")):
        return "minimize"
    if any(
        token in leaf
        for token in ("accuracy", "reward", "recall", "precision", "f1", "mrr")
    ):
        return "maximize"
    return "unknown"


def _signal(
    code: str,
    severity: DiagnosticSeverity,
    summary: str,
    *references: str,
) -> AutoResearchDiagnosticSignal:
    return AutoResearchDiagnosticSignal(
        code=code,
        severity=severity,
        summary=summary,
        evidence_references=tuple(dict.fromkeys(item for item in references if item)),
    )


def _ranked_hypothesis(
    *,
    rank: int,
    action_kind: HypothesisAction,
    changed_variable: str,
    hypothesis: str,
    rationale: str,
    expected_outcome: str,
    falsification_criterion: str,
    evidence_references: Sequence[str],
    eligible_for_submission: bool,
) -> AutoResearchRankedHypothesis:
    identity = canonical_hash(
        {
            "action_kind": action_kind,
            "changed_variable": changed_variable,
            "hypothesis": hypothesis,
            "evidence_references": list(evidence_references),
        }
    )[:24]
    return AutoResearchRankedHypothesis(
        hypothesis_id=f"hypothesis-{identity}",
        rank=rank,
        action_kind=action_kind,
        changed_variable=changed_variable,
        hypothesis=hypothesis,
        rationale=rationale,
        expected_outcome=expected_outcome,
        falsification_criterion=falsification_criterion,
        evidence_references=tuple(dict.fromkeys(evidence_references)),
        eligible_for_submission=eligible_for_submission,
    )


def build_autoresearch_diagnostics(
    *,
    workspace_id: str,
    campaign_id: str,
    primary_metric: str,
    metric_direction: Literal["maximize", "minimize"],
    evaluation_suite_id: str | None,
    outcomes: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
) -> AutoResearchDiagnostics:
    """Build a replay-safe diagnostic projection from authoritative evidence."""

    relevant_evaluations = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation_suite_id is None
        or evaluation.get("evaluation_suite_id") == evaluation_suite_id
    )
    evaluations_by_id = {
        _evaluation_id(evaluation): evaluation
        for evaluation in relevant_evaluations
        if _evaluation_id(evaluation)
    }
    runs_by_id = {str(run.get("run_id")): run for run in runs if run.get("run_id")}
    baseline_outcome = next(
        (
            outcome
            for outcome in outcomes
            if outcome.get("decision", {}).get("decision") == "baseline"
        ),
        None,
    )
    candidate_outcomes = [
        outcome
        for outcome in outcomes
        if outcome.get("result", {}).get("role") == "candidate"
    ]
    candidate_outcome = candidate_outcomes[-1] if candidate_outcomes else None
    baseline_evaluation = _outcome_evaluation(baseline_outcome, evaluations_by_id)
    candidate_evaluation = _outcome_evaluation(candidate_outcome, evaluations_by_id)
    baseline_evaluation_id = (
        _evaluation_id(baseline_evaluation) if baseline_evaluation is not None else ""
    )
    candidate_evaluation_id = (
        _evaluation_id(candidate_evaluation) if candidate_evaluation is not None else ""
    )

    signals: list[AutoResearchDiagnosticSignal] = []
    error_slices: list[AutoResearchErrorSlice] = []
    comparisons: list[AutoResearchCheckpointComparison] = []

    baseline_metric = None
    if baseline_evaluation is not None:
        baseline_metric = _finite_number(
            (baseline_evaluation.get("metrics") or {}).get(primary_metric)
        )
    candidate_metric = None
    if candidate_evaluation is not None:
        candidate_metric = _finite_number(
            (candidate_evaluation.get("metrics") or {}).get(primary_metric)
        )

    if candidate_metric is not None and baseline_metric is not None:
        primary_improvement = _signed_improvement(
            candidate_metric, baseline_metric, metric_direction
        )
        if primary_improvement <= 0:
            signals.append(
                _signal(
                    "primary_metric_no_improvement",
                    "warning",
                    "The latest candidate did not improve the exact primary metric over the verified baseline.",
                    candidate_evaluation_id,
                    baseline_evaluation_id,
                )
            )

    candidate_slices: Mapping[str, Any] = (
        candidate_evaluation.get("slice_metrics") or {}
        if candidate_evaluation is not None
        else {}
    )
    baseline_slices: Mapping[str, Any] = (
        baseline_evaluation.get("slice_metrics") or {}
        if baseline_evaluation is not None
        else {}
    )
    example_count = _finite_number(candidate_slices.get("example_count"))
    unique_count = _finite_number(candidate_slices.get("unique_prediction_count"))
    modal_fraction = _finite_number(candidate_slices.get("modal_prediction_fraction"))
    if candidate_slices.get("degenerate_constant_output") is True:
        signals.append(
            _signal(
                "degenerate_constant_output",
                "critical",
                "The candidate produced a degenerate constant output across the evaluation suite.",
                candidate_evaluation_id,
            )
        )
    if example_count is not None and example_count > 1 and unique_count is not None and unique_count <= 1:
        signals.append(
            _signal(
                "prediction_diversity_collapsed",
                "critical",
                "Prediction diversity collapsed to one unique output.",
                candidate_evaluation_id,
            )
        )
    if example_count is not None and example_count > 1 and modal_fraction is not None and modal_fraction >= 0.95:
        signals.append(
            _signal(
                "modal_prediction_dominates",
                "warning",
                "At least 95% of examples share the same prediction.",
                candidate_evaluation_id,
            )
        )

    explicit_directions = candidate_slices.get("_metric_directions") or {}
    if not isinstance(explicit_directions, Mapping):
        explicit_directions = {}
    candidate_flat = _flatten_numeric_slices(candidate_slices)
    baseline_flat = _flatten_numeric_slices(baseline_slices)
    for path, candidate_value in sorted(candidate_flat.items()):
        direction = _slice_direction(path, explicit_directions)
        baseline_value = baseline_flat.get(path)
        improvement = None
        status: SliceStatus = "observed"
        if baseline_value is not None and direction != "unknown":
            improvement = _signed_improvement(candidate_value, baseline_value, direction)
            if abs(improvement) <= 1e-12:
                status = "unchanged"
            elif improvement > 0:
                status = "improved"
            else:
                status = "regressed"
        error_slices.append(
            AutoResearchErrorSlice(
                slice_path=path,
                direction=direction,
                candidate_value=candidate_value,
                baseline_value=baseline_value,
                improvement=improvement,
                status=status,
                evidence_references=tuple(
                    item
                    for item in (candidate_evaluation_id, baseline_evaluation_id)
                    if item
                ),
            )
        )

    known_slices = [item for item in error_slices if item.improvement is not None]
    regressed_slices = [item for item in known_slices if item.status == "regressed"]
    improved_slices = [item for item in known_slices if item.status == "improved"]
    if regressed_slices and len(regressed_slices) > len(improved_slices):
        signals.append(
            _signal(
                "secondary_metrics_regressed",
                "warning",
                "More comparable error-slice metrics regressed than improved.",
                candidate_evaluation_id,
                baseline_evaluation_id,
            )
        )

    format_paths = [path for path in candidate_flat if "format" in path.lower()]
    format_gain = any(
        path in baseline_flat and candidate_flat[path] > baseline_flat[path]
        for path in format_paths
    )
    if format_gain and candidate_metric is not None and baseline_metric is not None:
        if _signed_improvement(candidate_metric, baseline_metric, metric_direction) <= 0:
            signals.append(
                _signal(
                    "format_only_gain",
                    "warning",
                    "Output formatting improved while the exact primary metric did not.",
                    candidate_evaluation_id,
                    baseline_evaluation_id,
                )
            )

    candidate_run_id = (
        str(candidate_evaluation.get("run_id")) if candidate_evaluation is not None else ""
    )
    run_evaluations = [
        evaluation
        for evaluation in relevant_evaluations
        if candidate_run_id and str(evaluation.get("run_id")) == candidate_run_id
    ]
    sortable: list[tuple[int, str, Mapping[str, Any], int | None]] = []
    for evaluation in run_evaluations:
        run = runs_by_id.get(str(evaluation.get("run_id")), {})
        step = _checkpoint_step(evaluation, run)
        sortable.append(
            (
                step if step is not None else 2**63 - 1,
                _evaluation_id(evaluation),
                evaluation,
                step,
            )
        )
    previous_metric: float | None = None
    for _sort_step, _evaluation_key, evaluation, step in sorted(sortable):
        metric = _finite_number((evaluation.get("metrics") or {}).get(primary_metric))
        if metric is None:
            continue
        comparison = AutoResearchCheckpointComparison(
            evaluation_result_id=_evaluation_id(evaluation),
            run_id=str(evaluation.get("run_id")),
            role=(
                "checkpoint"
                if (evaluation.get("slice_metrics") or {}).get("autoresearch_role")
                == "checkpoint"
                or step is not None
                else "final"
            ),
            step=step,
            metric_name=primary_metric,
            metric_value=metric,
            improvement_from_previous=(
                _signed_improvement(metric, previous_metric, metric_direction)
                if previous_metric is not None
                else None
            ),
            improvement_from_baseline=(
                _signed_improvement(metric, baseline_metric, metric_direction)
                if baseline_metric is not None
                else None
            ),
        )
        comparisons.append(comparison)
        previous_metric = metric
    checkpoint_count = sum(item.role == "checkpoint" for item in comparisons)
    if candidate_evaluation is not None and checkpoint_count == 0:
        signals.append(
            _signal(
                "checkpoint_evidence_missing",
                "warning",
                "Only the terminal candidate is evaluated; retained checkpoints cannot yet be compared.",
                candidate_evaluation_id,
                candidate_run_id,
            )
        )

    deduplicated_signals = tuple(
        {signal.code: signal for signal in signals}.values()
    )
    has_critical_signal = any(
        signal.severity == "critical" for signal in deduplicated_signals
    )
    signal_codes = {signal.code for signal in deduplicated_signals}
    hypothesis_specs: list[dict[str, Any]] = []
    if {"degenerate_constant_output", "prediction_diversity_collapsed"} & signal_codes:
        hypothesis_specs.append(
            {
                "action_kind": "diagnostic",
                "changed_variable": "training.signal_path",
                "hypothesis": "The candidate collapsed because the trainable signal path or multimodal input binding is invalid or too weak.",
                "rationale": "A constant prediction is a pipeline-integrity warning, not evidence that more optimization steps will help.",
                "expected_outcome": "A bounded smoke evaluation produces diverse predictions before another full candidate is launched.",
                "falsification_criterion": "The signal-path checks pass and a smoke checkpoint still collapses to one prediction.",
                "evidence_references": [candidate_evaluation_id],
            }
        )
    if "checkpoint_evidence_missing" in signal_codes:
        hypothesis_specs.append(
            {
                "action_kind": "evaluation",
                "changed_variable": "evaluation.checkpoint_selection",
                "hypothesis": "An earlier retained checkpoint may outperform the terminal checkpoint on the same fixed suite.",
                "rationale": "The current evidence contains only the terminal candidate, so training trajectory quality is unmeasured.",
                "expected_outcome": "Checkpoint comparisons identify a best observed step or confirm that the run never gained signal.",
                "falsification_criterion": "All retained checkpoints match or underperform the terminal candidate and baseline.",
                "evidence_references": [candidate_evaluation_id, candidate_run_id],
            }
        )
    if regressed_slices:
        worst = sorted(
            regressed_slices,
            key=lambda item: (item.improvement or 0.0, item.slice_path),
        )[0]
        hypothesis_specs.append(
            {
                "action_kind": "candidate",
                "changed_variable": "training.intervention",
                "hypothesis": f"A single bounded training intervention targeting {worst.slice_path} can recover the largest measured slice regression.",
                "rationale": f"{worst.slice_path} is the worst comparable slice by signed improvement.",
                "expected_outcome": "The targeted slice improves without regressing the exact primary metric.",
                "falsification_criterion": "The targeted slice or exact primary metric fails to improve on the fixed suite.",
                "evidence_references": list(worst.evidence_references),
            }
        )
    if "format_only_gain" in signal_codes:
        hypothesis_specs.append(
            {
                "action_kind": "candidate",
                "changed_variable": "training.objective",
                "hypothesis": "The objective currently rewards output shape more strongly than task correctness.",
                "rationale": "Format compliance improved while the exact primary metric did not.",
                "expected_outcome": "A correctness-aligned objective improves the primary metric while retaining valid output format.",
                "falsification_criterion": "The revised objective improves formatting only or reduces primary-metric performance.",
                "evidence_references": [candidate_evaluation_id, baseline_evaluation_id],
            }
        )
    if not hypothesis_specs and candidate_evaluation is not None:
        hypothesis_specs.append(
            {
                "action_kind": "evaluation",
                "changed_variable": "evaluation.robustness",
                "hypothesis": "The observed candidate result should be reproduced before expanding the search space.",
                "rationale": "The evidence does not expose a dominant failure mode, so replication is the smallest next test.",
                "expected_outcome": "A repeat evaluation agrees within the suite's declared tolerance.",
                "falsification_criterion": "The repeat evaluation falls outside the declared tolerance.",
                "evidence_references": [candidate_evaluation_id],
            }
        )

    ranked_hypotheses = tuple(
        _ranked_hypothesis(
            rank=index,
            eligible_for_submission=(
                spec["action_kind"] == "candidate" and not has_critical_signal
            ),
            **spec,
        )
        for index, spec in enumerate(hypothesis_specs[:5], start=1)
    )
    return AutoResearchDiagnostics(
        workspace_id=workspace_id,
        campaign_id=campaign_id,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        low_signal=any(
            signal.severity in {"warning", "critical"}
            for signal in deduplicated_signals
        ),
        signals=deduplicated_signals,
        checkpoint_comparisons=tuple(comparisons),
        error_slices=tuple(error_slices),
        ranked_hypotheses=ranked_hypotheses,
    )


__all__ = [
    "AutoResearchCheckpointComparison",
    "AutoResearchDiagnosticSignal",
    "AutoResearchDiagnostics",
    "AutoResearchErrorSlice",
    "AutoResearchRankedHypothesis",
    "build_autoresearch_diagnostics",
]
