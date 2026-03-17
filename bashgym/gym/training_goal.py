"""
Training Goal -- Structured multi-criteria outcome aggregation.

Replaces the single-scalar val_loss optimization with weighted goals,
hard/soft constraints, and actionable recommendations. Inspired by
the OutcomeAggregator pattern.

Backward compatible: when no goal is provided, AutoResearcher and
TraceResearcher behave exactly as before (single metric optimization).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid comparator strings for SuccessCriterion
VALID_COMPARATORS = {"lt", "gt", "eq", "lte", "gte"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SuccessCriterion:
    """A single weighted success criterion.

    Example: SuccessCriterion(
        description="eval_loss < 1.5",
        metric_key="eval_loss",
        target=1.5,
        comparator="lt",
        weight=0.8,
    )
    """

    description: str
    metric_key: str
    target: float
    comparator: str  # "lt", "gt", "eq", "lte", "gte"
    weight: float  # 0.0 - 1.0


@dataclass
class GoalConstraint:
    """A hard or soft constraint on a metric.

    Hard constraints cause immediate termination when violated.
    Soft constraints produce warnings when within 10% of the limit.
    """

    description: str
    metric_key: str
    limit: float
    hard: bool = True  # hard = stop immediately; soft = warn


@dataclass
class TrainingGoal:
    """A structured training objective with weighted criteria and constraints."""

    criteria: List[SuccessCriterion] = field(default_factory=list)
    constraints: List[GoalConstraint] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate this goal definition.

        Returns a list of error strings. Empty list means valid.
        """
        errors: List[str] = []

        if not self.criteria:
            errors.append("At least one success criterion is required")

        # Check weights sum to approximately 1.0
        total_weight = sum(c.weight for c in self.criteria)
        if self.criteria and not (0.95 <= total_weight <= 1.05):
            errors.append(
                f"Criterion weights must sum to ~1.0 (got {total_weight:.3f})"
            )

        # Check individual criteria
        for i, criterion in enumerate(self.criteria):
            if criterion.comparator not in VALID_COMPARATORS:
                errors.append(
                    f"Criterion[{i}] has invalid comparator '{criterion.comparator}' "
                    f"(valid: {VALID_COMPARATORS})"
                )
            if not (0.0 <= criterion.weight <= 1.0):
                errors.append(
                    f"Criterion[{i}] weight {criterion.weight} not in [0, 1]"
                )
            if not criterion.metric_key:
                errors.append(f"Criterion[{i}] has empty metric_key")

        # Check constraints
        for i, constraint in enumerate(self.constraints):
            if not constraint.metric_key:
                errors.append(f"Constraint[{i}] has empty metric_key")

        return errors

    @classmethod
    def default_loss_goal(cls, target_loss: float = 1.5) -> TrainingGoal:
        """Create a simple goal: minimize eval_loss below target.

        This is backward compatible with the old single-metric approach.
        """
        return cls(
            criteria=[
                SuccessCriterion(
                    description=f"eval_loss < {target_loss}",
                    metric_key="eval_loss",
                    target=target_loss,
                    comparator="lt",
                    weight=1.0,
                )
            ],
        )


@dataclass
class GoalProgress:
    """Snapshot of goal progress after an experiment."""

    criteria_scores: Dict[str, float]  # metric_key -> normalized score (0-1)
    weighted_score: float  # combined weighted score (0-1, higher = better)
    constraints_status: Dict[str, str]  # metric_key -> "ok" | "warning" | "violated"
    recommendation: str  # "continue" | "adjust" | "complete"
    reasoning: str  # human-readable explanation


# =============================================================================
# OutcomeAggregator
# =============================================================================


class OutcomeAggregator:
    """Track goal progress across experiments.

    Provides:
    - Weighted multi-criteria scoring
    - Hard/soft constraint enforcement
    - Recommendations: continue, adjust, complete
    - History tracking for stall detection
    """

    # Number of recent experiments to examine for stalling
    STALL_WINDOW = 5
    # Minimum improvement in weighted score to not count as stalling
    STALL_THRESHOLD = 0.01

    def __init__(self, goal: TrainingGoal):
        errors = goal.validate()
        if errors:
            raise ValueError(
                f"Invalid TrainingGoal: {'; '.join(errors)}"
            )

        self.goal = goal
        self.history: List[tuple[Dict[str, Any], GoalProgress]] = []

    def record(self, metrics: Dict[str, Any]) -> GoalProgress:
        """Record experiment metrics and return a progress assessment."""
        criteria_scores: Dict[str, float] = {}
        for criterion in self.goal.criteria:
            value = metrics.get(criterion.metric_key)
            if value is not None:
                score = self._check_criterion(criterion, float(value))
            else:
                score = 0.0  # missing metric = not met
            criteria_scores[criterion.metric_key] = score

        w_score = self.weighted_score(metrics)

        constraints_status: Dict[str, str] = {}
        for constraint in self.goal.constraints:
            value = metrics.get(constraint.metric_key)
            if value is not None:
                status = self._check_constraint(constraint, float(value))
            else:
                status = "ok"  # missing constraint metric = assume ok
            constraints_status[constraint.metric_key] = status

        recommendation = self._compute_recommendation(
            criteria_scores, w_score, constraints_status
        )

        reasoning = self._build_reasoning(
            criteria_scores, w_score, constraints_status, recommendation, metrics
        )

        progress = GoalProgress(
            criteria_scores=criteria_scores,
            weighted_score=w_score,
            constraints_status=constraints_status,
            recommendation=recommendation,
            reasoning=reasoning,
        )

        self.history.append((dict(metrics), progress))
        return progress

    def recommend(self) -> str:
        """Return the current recommendation based on accumulated history.

        Returns 'continue', 'adjust', or 'complete'.
        """
        if not self.history:
            return "continue"
        return self.history[-1][1].recommendation

    def weighted_score(self, metrics: Dict[str, Any]) -> float:
        """Compute multi-criteria weighted score (0.0 to 1.0).

        Higher is better: 1.0 means all criteria fully met.
        """
        total = 0.0
        for criterion in self.goal.criteria:
            value = metrics.get(criterion.metric_key)
            if value is not None:
                score = self._check_criterion(criterion, float(value))
            else:
                score = 0.0
            total += score * criterion.weight
        return round(total, 4)

    def _check_criterion(self, criterion: SuccessCriterion, value: float) -> float:
        """Score a single criterion on [0.0, 1.0].

        0.0 = not met at all, 1.0 = fully met or exceeded.

        Scoring logic varies by comparator:
        - "lt"/"lte": value at or below target = 1.0; higher values decay toward 0
        - "gt"/"gte": value at or above target = 1.0; lower values decay toward 0
        - "eq": closeness to target, with 10% tolerance for full score
        """
        target = criterion.target
        comp = criterion.comparator

        if comp in ("lt", "lte"):
            # Lower is better. Fully met if value <= target.
            if comp == "lt" and value < target:
                return 1.0
            if comp == "lte" and value <= target:
                return 1.0
            if comp == "lt" and value == target:
                # At the boundary: almost met
                return 0.95
            # value > target: decay based on how far over
            if target == 0:
                return 0.0
            overshoot = (value - target) / abs(target)
            return max(0.0, 1.0 - overshoot)

        if comp in ("gt", "gte"):
            # Higher is better. Fully met if value >= target.
            if comp == "gt" and value > target:
                return 1.0
            if comp == "gte" and value >= target:
                return 1.0
            if comp == "gt" and value == target:
                return 0.95
            # value < target: decay based on how far under
            if target == 0:
                return 0.0
            undershoot = (target - value) / abs(target)
            return max(0.0, 1.0 - undershoot)

        if comp == "eq":
            # Equality: full score within 10% tolerance
            if target == 0:
                return 1.0 if value == 0 else max(0.0, 1.0 - abs(value))
            relative_error = abs(value - target) / abs(target)
            if relative_error <= 0.1:
                return 1.0
            return max(0.0, 1.0 - relative_error)

        # Unknown comparator (should have been caught by validate)
        logger.warning(f"Unknown comparator '{comp}', returning 0.0")
        return 0.0

    def _check_constraint(self, constraint: GoalConstraint, value: float) -> str:
        """Check a constraint against a value.

        Returns:
            "ok": well within limit
            "warning": within 10% of the limit (soft constraints only surface this)
            "violated": limit exceeded
        """
        limit = constraint.limit

        # For constraints, we assume the value should stay below the limit.
        # Example: budget_usd < 50, training_time_minutes < 30
        if value > limit:
            return "violated"

        # Check if within 10% of the limit
        if limit > 0:
            ratio = value / limit
            if ratio >= 0.9:
                return "warning"

        return "ok"

    def _compute_recommendation(
        self,
        criteria_scores: Dict[str, float],
        w_score: float,
        constraints_status: Dict[str, str],
    ) -> str:
        """Determine recommendation based on current state and history."""
        # Hard constraint violated -> stop
        for constraint in self.goal.constraints:
            if constraint.hard and constraints_status.get(constraint.metric_key) == "violated":
                return "complete"

        # All criteria fully met (score >= 0.95 for each) -> done
        all_met = all(s >= 0.95 for s in criteria_scores.values()) if criteria_scores else False
        if all_met:
            return "complete"

        # Check for stalling: no meaningful improvement in last N experiments
        if len(self.history) >= self.STALL_WINDOW:
            recent_scores = [
                h[1].weighted_score for h in self.history[-self.STALL_WINDOW:]
            ]
            best_recent = max(recent_scores)
            worst_recent = min(recent_scores)
            if (best_recent - worst_recent) < self.STALL_THRESHOLD:
                return "adjust"

        return "continue"

    def _build_reasoning(
        self,
        criteria_scores: Dict[str, float],
        w_score: float,
        constraints_status: Dict[str, str],
        recommendation: str,
        metrics: Dict[str, Any],
    ) -> str:
        """Build a human-readable explanation of the progress assessment."""
        parts: List[str] = []

        parts.append(f"Weighted score: {w_score:.3f}")

        for criterion in self.goal.criteria:
            score = criteria_scores.get(criterion.metric_key, 0.0)
            value = metrics.get(criterion.metric_key, "N/A")
            status = "MET" if score >= 0.95 else f"{score:.0%}"
            parts.append(
                f"  {criterion.description}: {value} ({status}, weight={criterion.weight})"
            )

        for constraint in self.goal.constraints:
            status = constraints_status.get(constraint.metric_key, "ok")
            value = metrics.get(constraint.metric_key, "N/A")
            kind = "HARD" if constraint.hard else "soft"
            parts.append(f"  Constraint [{kind}] {constraint.description}: {value} -> {status}")

        if recommendation == "complete":
            # Determine why
            hard_violated = any(
                c.hard and constraints_status.get(c.metric_key) == "violated"
                for c in self.goal.constraints
            )
            if hard_violated:
                parts.append("Recommendation: COMPLETE (hard constraint violated)")
            else:
                parts.append("Recommendation: COMPLETE (all criteria met)")
        elif recommendation == "adjust":
            parts.append(
                f"Recommendation: ADJUST (stalled over last {self.STALL_WINDOW} experiments)"
            )
        else:
            parts.append("Recommendation: CONTINUE")

        return "\n".join(parts)
