"""One rule for reading a declared training seed from a recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bashgym.campaigns.contracts import StageDisposition, StageKind, StagePlan

TRAINING_SEED_MAX = 2_147_483_647
_TRAINING_STAGES = frozenset({StageKind.SMOKE_TRAINING, StageKind.FULL_TRAINING})


def training_seed(recipe: Mapping[str, Any]) -> int | None:
    """Return the declared integer seed, or None when the recipe does not pin one."""

    value = recipe.get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > TRAINING_SEED_MAX:
        return None
    return value


def training_stages_required(stage_plan: StagePlan) -> bool:
    """True when the plan will run a training stage."""

    return any(
        item.stage in _TRAINING_STAGES and item.disposition == StageDisposition.REQUIRED
        for item in stage_plan.items
    )


__all__ = ["TRAINING_SEED_MAX", "training_seed", "training_stages_required"]
