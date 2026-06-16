"""Deploy gate: turn eval results into a ship / no-ship verdict against
pre-registered thresholds, so a regression blocks deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateThresholds:
    min_trace_delta: float = 0.05  # candidate must beat base by >= this on held-out pass rate
    require_ci_excludes_zero: bool = True  # ...and the improvement must be statistically reliable
    max_forgetting_drop: float = 0.05  # no general benchmark may drop more than this


@dataclass
class GateVerdict:
    ship: bool
    reasons: list[str] = field(default_factory=list)


def evaluate_gate(
    *,
    trace_delta: float,
    ci_low: float,
    ci_excludes_zero: bool,
    forgetting_drops: dict | None = None,
    thresholds: GateThresholds | None = None,
) -> GateVerdict:
    """Decide ship/no-ship from held-out + forgetting results.

    trace_delta: observed held-out pass-rate delta (candidate - base).
    ci_low / ci_excludes_zero: from the paired bootstrap.
    forgetting_drops: {benchmark: drop} where drop = base - candidate (positive = regressed).
    """
    t = thresholds or GateThresholds()
    reasons: list[str] = []

    if trace_delta < t.min_trace_delta:
        reasons.append(f"trace delta {trace_delta:.3f} < required {t.min_trace_delta:.3f}")
    if t.require_ci_excludes_zero and not (ci_excludes_zero and ci_low > 0):
        reasons.append("held-out improvement not statistically reliable (95% CI must exclude 0)")
    for bench, drop in (forgetting_drops or {}).items():
        if drop > t.max_forgetting_drop:
            reasons.append(f"forgetting: {bench} dropped {drop:.3f} > {t.max_forgetting_drop:.3f}")

    return GateVerdict(ship=not reasons, reasons=reasons)
