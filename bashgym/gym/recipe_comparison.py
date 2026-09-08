"""Offline comparison of two explicitly supplied measured recipe runs.

JSON record schema ``bashgym.recipe_run_record.v1``:
  run_id, role (baseline/candidate), provenance (measured),
  contract: model_digest, data_digest, evaluation_digest, target_digest,
            software_digest, training_digest, measurement_digest (SHA-256 strings),
  optimization: backend (plain/unsloth), liger_kernel (boolean),
  performance: tokens_per_second, gpu_memory_peak_gb, elapsed_seconds, cost, cost_unit,
  quality_contract: primary_metric and protected_metrics (an explicit list),
  quality: {metric_name: measured_value}, evaluation_sample_count.
Each metric guard has name, direction (maximize/minimize), and max_regression
(absolute metric units). An empty protected_metrics list explicitly declares no
additional guards. Model digest identifies the starting checkpoint. Training
digest pins the data order, seed, budget, precision, batch/sequence sizes and all
other training settings except the two explicit optimization switches. Software
digest pins the common installed stack, including optional optimization versions.
Measurement digest pins timing, memory, throughput aggregation and cost accounting.

Unlike run_analysis's exploratory JSONL summaries, every measurement is required;
missing values are not inferred from loss logs or filled with zero. The supplied
records remain the provenance trust boundary. This module hashes them for review
but does not authenticate their measurements, run benchmarks, select a winner, or
claim statistical confirmation from two runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
MAX_RECORD_BYTES = 1024 * 1024


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class RecipeInputContract(_Contract):
    model_digest: Digest
    data_digest: Digest
    evaluation_digest: Digest
    target_digest: Digest
    software_digest: Digest
    training_digest: Digest
    measurement_digest: Digest


class OptimizationSettings(_Contract):
    backend: Literal["plain", "unsloth"]
    liger_kernel: bool = Field(strict=True)

    @model_validator(mode="after")
    def supported_combination(self) -> OptimizationSettings:
        if self.backend == "unsloth" and self.liger_kernel:
            raise ValueError("Unsloth and Liger cannot be combined in this comparison contract")
        return self


class MeasuredPerformance(_Contract):
    tokens_per_second: Number = Field(gt=0)
    gpu_memory_peak_gb: Number = Field(gt=0)
    elapsed_seconds: Number = Field(gt=0)
    cost: Number = Field(ge=0)
    cost_unit: Name


class MetricGuard(_Contract):
    name: Name
    direction: Literal["maximize", "minimize"]
    max_regression: Number = Field(ge=0)


class QualityContract(_Contract):
    primary_metric: MetricGuard
    protected_metrics: list[MetricGuard] = Field(max_length=32)

    @model_validator(mode="after")
    def unique_metrics(self) -> QualityContract:
        names = [guard.name for guard in (self.primary_metric, *self.protected_metrics)]
        if len(names) != len(set(names)):
            raise ValueError("quality metric names must be unique")
        return self


class RecipeRunRecord(_Contract):
    schema_version: Literal["bashgym.recipe_run_record.v1"]
    run_id: Name
    role: Literal["baseline", "candidate"]
    provenance: Literal["measured"]
    contract: RecipeInputContract
    optimization: OptimizationSettings
    performance: MeasuredPerformance
    quality_contract: QualityContract
    quality: dict[Name, Number] = Field(min_length=1, max_length=33)
    evaluation_sample_count: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def complete_quality(self) -> RecipeRunRecord:
        guards = (self.quality_contract.primary_metric, *self.quality_contract.protected_metrics)
        if set(self.quality) != {guard.name for guard in guards}:
            raise ValueError(
                "quality values must match every declared primary and protected metric"
            )
        return self


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("comparison arithmetic produced a nonfinite value")
    return value


def _delta(baseline: float, candidate: float) -> dict[str, float | None]:
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": _finite(candidate - baseline),
        "ratio": _finite(candidate / baseline) if baseline != 0 else None,
    }


def _record_digest(record: RecipeRunRecord) -> str:
    encoded = json.dumps(
        record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_recipe_runs(records: Sequence[Mapping[str, Any] | RecipeRunRecord]) -> dict[str, Any]:
    """Compare exactly one baseline and candidate; invalid/incomparable input raises ValueError."""
    if isinstance(records, (str, bytes, Mapping)) or len(records) != 2:
        raise ValueError("exactly two records with explicit baseline/candidate roles are required")
    parsed = [RecipeRunRecord.model_validate(record) for record in records]
    by_role = {record.role: record for record in parsed}
    if set(by_role) != {"baseline", "candidate"}:
        raise ValueError("exactly one baseline role and one candidate role are required")
    baseline, candidate = by_role["baseline"], by_role["candidate"]
    if baseline.run_id == candidate.run_id:
        raise ValueError("distinct run identities are required")
    mismatches = [
        name
        for name in RecipeInputContract.model_fields
        if getattr(baseline.contract, name) != getattr(candidate.contract, name)
    ]
    if baseline.quality_contract != candidate.quality_contract:
        mismatches.append("quality_contract")
    if baseline.evaluation_sample_count != candidate.evaluation_sample_count:
        mismatches.append("evaluation_sample_count")
    if baseline.performance.cost_unit != candidate.performance.cost_unit:
        mismatches.append("cost_unit")
    if mismatches:
        raise ValueError("incomparable run contracts: " + ", ".join(mismatches))
    performance = {
        name: _delta(getattr(baseline.performance, name), getattr(candidate.performance, name))
        for name in ("tokens_per_second", "gpu_memory_peak_gb", "elapsed_seconds", "cost")
    }
    quality = []
    for index, guard in enumerate(
        (baseline.quality_contract.primary_metric, *baseline.quality_contract.protected_metrics)
    ):
        before, after = baseline.quality[guard.name], candidate.quality[guard.name]
        delta = _finite(after - before)
        improvement = delta if guard.direction == "maximize" else -delta
        boundary = _finite(
            before - guard.max_regression
            if guard.direction == "maximize"
            else before + guard.max_regression
        )
        breached = after < boundary if guard.direction == "maximize" else after > boundary
        quality.append(
            {
                "name": guard.name,
                "role": "primary" if index == 0 else "protected",
                "direction": guard.direction,
                "baseline": before,
                "candidate": after,
                "delta": delta,
                "improvement": improvement,
                "max_regression": guard.max_regression,
                "guard_breached": breached,
            }
        )
    return {
        "schema_version": "bashgym.recipe_comparison.v1",
        "status": "comparable",
        "evidence_strength": "exploratory",
        "evidence_scope": "supplied_run_records",
        "selection_authority": "host_agent",
        "runs": {
            role: {
                "run_id": record.run_id,
                "record_digest": _record_digest(record),
                "optimization": record.optimization.model_dump(mode="json"),
            }
            for role, record in by_role.items()
        },
        "contract": baseline.contract.model_dump(mode="json"),
        "evaluation_sample_count": baseline.evaluation_sample_count,
        "performance": performance,
        "cost_unit": baseline.performance.cost_unit,
        "quality": quality,
        "quality_guard_breached": any(row["guard_breached"] for row in quality),
        "ratio_convention": "candidate_over_baseline; undefined for zero baseline",
    }


def compare_recipe_run_files(
    baseline_path: Path | str, candidate_path: Path | str
) -> dict[str, Any]:
    """Load two bounded JSON records with explicit roles and retain their file hashes."""
    records = []
    digests = {}
    for role, supplied in (("baseline", baseline_path), ("candidate", candidate_path)):
        path = Path(supplied)
        if not path.is_file() or path.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError("recipe run record must be a file no larger than 1 MiB")
        with path.open("rb") as stream:
            payload = stream.read(MAX_RECORD_BYTES + 1)
        if len(payload) > MAX_RECORD_BYTES:
            raise ValueError("recipe run record exceeds 1 MiB")
        record = RecipeRunRecord.model_validate(json.loads(payload, object_pairs_hook=_unique_keys))
        if record.role != role:
            raise ValueError(f"{role} file has the wrong explicit role")
        records.append(record)
        digests[role] = hashlib.sha256(payload).hexdigest()
    report = compare_recipe_runs(records)
    report["source_file_digests"] = digests
    return report


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


__all__ = ["RecipeRunRecord", "compare_recipe_runs", "compare_recipe_run_files"]
