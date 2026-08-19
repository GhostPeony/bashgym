"""Longitudinal adaptation-efficiency diagnostics for AutoResearch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from bashgym.campaigns.autoresearch import (
    AutoResearchDiagnosticResult,
    AutoResearchProposalControl,
    ExperimentRole,
)


def build_plasticity_comparison(
    *,
    diagnostic_results: Sequence[AutoResearchDiagnosticResult],
    controls: Sequence[AutoResearchProposalControl],
) -> dict[str, Any]:
    """Compare completed fixed-budget probes across an exact model lineage."""

    empty = {
        "schema_version": "bashgym.autoresearch_plasticity_comparison.v1",
        "status": "insufficient_comparable_probes",
        "reason_code": "matching_fixed_probe_required",
        "classification": None,
        "observations": [],
        "comparison": None,
    }
    probes = [
        item
        for item in diagnostic_results
        if item.status == "completed"
        and item.projection.get("status") == "completed"
        and item.projection.get("probe_family") == "plasticity_probe"
    ]
    if not probes:
        return empty
    probes.sort(key=lambda item: (item.recorded_at, item.proposal_id))
    latest_digest = _recipe_digest(probes[-1].projection)
    if latest_digest is None:
        return empty
    matching = [item for item in probes if _recipe_digest(item.projection) == latest_digest]
    if len(matching) < 2:
        return empty

    controls_by_id = {item.proposal_id: item for item in controls}
    contract = matching[-1].projection.get("comparison_contract")
    if not isinstance(contract, dict) or any(
        item.projection.get("comparison_contract") != contract for item in matching
    ):
        return empty
    parsed_contract = _comparison_contract(contract)
    if parsed_contract is None:
        return empty

    observations: list[dict[str, Any]] = []
    for result in matching:
        control = controls_by_id.get(result.proposal_id)
        if control is None or control.role != ExperimentRole.DIAGNOSTIC:
            return empty
        parent_id = control.parent_proposal_id
        depth = _lineage_depth(parent_id, controls_by_id)
        observation = _observation(
            result=result,
            parent_proposal_id=parent_id,
            lineage_depth=depth,
            contract=parsed_contract,
        )
        if observation is None:
            return empty
        observations.append(observation)
    observations.sort(
        key=lambda item: (
            item["lineage_depth"],
            item["recorded_at"],
            item["diagnostic_proposal_id"],
        )
    )
    if len({item["probe_sample_count"] for item in observations}) != 1:
        return empty
    reference = observations[0]
    latest = observations[-1]
    if reference["parent_proposal_id"] == latest["parent_proposal_id"]:
        return empty
    reference_efficiency = reference["adaptation_gain_per_step"]
    latest_efficiency = latest["adaptation_gain_per_step"]
    if reference_efficiency <= 0:
        return {
            **empty,
            "reason_code": "positive_reference_adaptation_required",
            "observations": observations,
        }
    efficiency_ratio = _rounded(latest_efficiency / reference_efficiency)
    retention_concern = latest["retention_delta"] < -parsed_contract["maximum_retention_drop"]
    plasticity_concern = efficiency_ratio < parsed_contract["minimum_efficiency_ratio"]
    if retention_concern and plasticity_concern:
        classification = "retention_and_plasticity_concerns"
    elif retention_concern:
        classification = "retention_regression_observed"
    elif plasticity_concern:
        classification = "plasticity_loss_suspected"
    else:
        classification = "no_material_decline_observed"
    return {
        "schema_version": "bashgym.autoresearch_plasticity_comparison.v1",
        "status": "comparable",
        "reason_code": "fixed_probe_comparison_ready",
        "classification": classification,
        "observations": observations,
        "comparison": {
            "reference_parent_proposal_id": reference["parent_proposal_id"],
            "latest_parent_proposal_id": latest["parent_proposal_id"],
            "reference_lineage_depth": reference["lineage_depth"],
            "latest_lineage_depth": latest["lineage_depth"],
            "adaptation_efficiency_ratio": efficiency_ratio,
            "retention_delta_change": _rounded(
                latest["retention_delta"] - reference["retention_delta"]
            ),
            "minimum_efficiency_ratio": parsed_contract["minimum_efficiency_ratio"],
            "maximum_retention_drop": parsed_contract["maximum_retention_drop"],
        },
    }


def _recipe_digest(projection: dict[str, Any]) -> str | None:
    reference = projection.get("evidence_reference")
    if not isinstance(reference, dict):
        return None
    value = reference.get("recipe_digest")
    return value if isinstance(value, str) and len(value) == 64 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _comparison_contract(value: dict[str, Any]) -> dict[str, Any] | None:
    direction = value.get("metric_direction")
    step_budget = value.get("fixed_step_budget")
    ratio = _number(value.get("minimum_efficiency_ratio"))
    retention = _number(value.get("maximum_retention_drop"))
    if (
        direction not in {"maximize", "minimize"}
        or isinstance(step_budget, bool)
        or not isinstance(step_budget, int)
        or step_budget < 1
        or ratio is None
        or not 0 <= ratio <= 1
        or retention is None
        or retention < 0
    ):
        return None
    return {
        "metric_direction": direction,
        "fixed_step_budget": step_budget,
        "minimum_efficiency_ratio": ratio,
        "maximum_retention_drop": retention,
    }


def _lineage_depth(
    proposal_id: str | None,
    controls: dict[str, AutoResearchProposalControl],
) -> int | None:
    depth = 0
    current = proposal_id
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            return None
        seen.add(current)
        control = controls.get(current)
        if control is None or control.role == ExperimentRole.DIAGNOSTIC:
            return None
        if control.role == ExperimentRole.BASELINE:
            return depth
        current = control.parent_proposal_id
        depth += 1
    return None


def _observation(
    *,
    result: AutoResearchDiagnosticResult,
    parent_proposal_id: str | None,
    lineage_depth: int | None,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    if parent_proposal_id is None or lineage_depth is None:
        return None
    measurements = result.projection.get("measurements")
    if not isinstance(measurements, list):
        return None
    by_name = {
        item.get("name"): item
        for item in measurements
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    required = (
        "initial_probe_metric",
        "final_probe_metric",
        "retention_delta",
        "cumulative_training_steps",
        "cumulative_training_tokens",
        "dataset_revision_count",
    )
    values = {name: _number((by_name.get(name) or {}).get("value")) for name in required}
    if any(value is None for value in values.values()):
        return None
    sample_counts = {
        (by_name.get(name) or {}).get("sample_count")
        for name in ("initial_probe_metric", "final_probe_metric", "retention_delta")
    }
    if len(sample_counts) != 1:
        return None
    probe_sample_count = sample_counts.pop()
    if (
        isinstance(probe_sample_count, bool)
        or not isinstance(probe_sample_count, int)
        or probe_sample_count < 1
    ):
        return None
    initial = values["initial_probe_metric"]
    final = values["final_probe_metric"]
    if initial is None or final is None:
        return None
    gain = final - initial if contract["metric_direction"] == "maximize" else initial - final
    return {
        "diagnostic_proposal_id": result.proposal_id,
        "parent_proposal_id": parent_proposal_id,
        "lineage_depth": lineage_depth,
        "probe_sample_count": probe_sample_count,
        "recorded_at": result.recorded_at.isoformat(),
        "initial_probe_metric": initial,
        "final_probe_metric": final,
        "adaptation_gain": _rounded(gain),
        "adaptation_gain_per_step": _rounded(gain / contract["fixed_step_budget"]),
        "retention_delta": values["retention_delta"],
        "cumulative_training_steps": values["cumulative_training_steps"],
        "cumulative_training_tokens": values["cumulative_training_tokens"],
        "dataset_revision_count": values["dataset_revision_count"],
        "evidence_reference": result.projection.get("evidence_reference"),
    }


def _rounded(value: float) -> float:
    return round(value, 12)


__all__ = ["build_plasticity_comparison"]
