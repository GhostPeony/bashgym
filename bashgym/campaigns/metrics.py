"""Provider-neutral training metric stream and diagnostic contracts."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum

from pydantic import Field, ValidationError, field_validator

from bashgym.campaigns.contracts import FrozenContractModel


class MetricStreamError(ValueError):
    code = "campaign_metric_stream_invalid"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class TrainingMetricPoint(FrozenContractModel):
    schema_version: str = "campaign_training_metric_point.v1"
    step: int = Field(ge=0)
    values: dict[str, float]
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("metric point requires at least one numeric value")
        if any(not name or not math.isfinite(number) for name, number in value.items()):
            raise ValueError("metric names and values must be finite")
        return dict(sorted(value.items()))


class TrainingAlert(FrozenContractModel):
    schema_version: str = "campaign_training_alert.v1"
    code: str
    step: int = Field(ge=0)
    metric_name: str
    severity: AlertSeverity
    metric_value: float
    message: str


class MetricSeriesValue(FrozenContractModel):
    schema_version: str = "campaign_metric_series_value.v1"
    step: int = Field(ge=0)
    source: str
    value: float
    observed_at: str


def parse_metric_lines(lines: tuple[str, ...]) -> tuple[TrainingMetricPoint, ...]:
    points = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("metric row must be an object")
            raw_step = payload.get("step", payload.get("global_step"))
            if isinstance(raw_step, bool) or int(raw_step) != raw_step or int(raw_step) < 0:
                raise ValueError("metric step must be a non-negative integer")
            values = {
                str(key): float(value)
                for key, value in payload.items()
                if key not in {"step", "global_step"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            points.append(
                TrainingMetricPoint(
                    step=int(raw_step),
                    values=values,
                    raw_sha256=hashlib.sha256(line.encode()).hexdigest(),
                )
            )
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise MetricStreamError(f"{MetricStreamError.code}: line {line_number}") from exc
    return tuple(points)


def detect_training_alerts(
    points: tuple[TrainingMetricPoint, ...],
) -> tuple[TrainingAlert, ...]:
    """Create deterministic autonomous diagnostics without owning source metrics."""

    alerts: list[TrainingAlert] = []
    prior_loss: float | None = None
    for point in sorted(points, key=lambda item: item.step):
        loss_name = next(
            (name for name in ("loss", "train_loss") if name in point.values), None
        )
        if loss_name is None:
            continue
        loss = point.values[loss_name]
        if point.step > 0 and abs(loss) < 1e-8:
            alerts.append(
                TrainingAlert(
                    code="loss_near_zero",
                    step=point.step,
                    metric_name=loss_name,
                    severity=AlertSeverity.WARN,
                    metric_value=loss,
                    message="Loss is near zero; check for collapse or leakage.",
                )
            )
        if prior_loss is not None and loss > max(0.5, prior_loss * 2.0):
            alerts.append(
                TrainingAlert(
                    code="loss_spike",
                    step=point.step,
                    metric_name=loss_name,
                    severity=AlertSeverity.ERROR,
                    metric_value=loss,
                    message="Loss increased by more than 2x between reported steps.",
                )
            )
        prior_loss = loss
    return tuple(alerts)


def trackio_projection(
    *, project: str, run: str, metric_name: str, points: tuple[TrainingMetricPoint, ...]
) -> dict[str, object]:
    """Optional JSON-compatible projection matching Trackio CLI retrieval shape."""

    return {
        "project": project,
        "run": run,
        "metric": metric_name,
        "values": [
            {"step": point.step, "value": point.values[metric_name]}
            for point in sorted(points, key=lambda item: item.step)
            if metric_name in point.values
        ],
    }


__all__ = [
    "AlertSeverity",
    "MetricStreamError",
    "MetricSeriesValue",
    "TrainingAlert",
    "TrainingMetricPoint",
    "detect_training_alerts",
    "parse_metric_lines",
    "trackio_projection",
]
