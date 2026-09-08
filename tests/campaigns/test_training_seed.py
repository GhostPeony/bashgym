"""Training seed accessor shared by validation, replication grouping, and the packet."""

from bashgym.campaigns.contracts import StageDisposition, StageKind, StagePlan, StagePlanItem
from bashgym.campaigns.training_seed import (
    TRAINING_SEED_MAX,
    training_seed,
    training_stages_required,
)


def test_training_seed_accepts_only_non_bool_integers_in_range() -> None:
    assert training_seed({"schema_version": "recipe.v1", "seed": 17}) == 17
    assert training_seed({"schema_version": "recipe.v1", "seed": 0}) == 0
    assert training_seed({"schema_version": "recipe.v1", "seed": TRAINING_SEED_MAX}) == (
        TRAINING_SEED_MAX
    )
    assert training_seed({"schema_version": "recipe.v1"}) is None
    assert training_seed({"schema_version": "recipe.v1", "seed": True}) is None
    assert training_seed({"schema_version": "recipe.v1", "seed": "17"}) is None
    assert training_seed({"schema_version": "recipe.v1", "seed": -1}) is None
    assert training_seed({"schema_version": "recipe.v1", "seed": TRAINING_SEED_MAX + 1}) is None


def test_schema_default_seed_is_not_a_declared_seed() -> None:
    recipe = {
        "schema_version": "bashgym.tmax_composite_training_recipe.v1",
        "algorithm": "grpo",
    }

    assert training_seed(recipe) is None
    assert training_seed({**recipe, "seed": 42}) == 42


def _plan(*stages: StageKind, disposition=StageDisposition.REQUIRED) -> StagePlan:
    return StagePlan(
        items=tuple(
            StagePlanItem(stage=stage, disposition=disposition, reason="test plan")
            for stage in stages
        )
    )


def test_training_stages_required_only_for_required_training_items() -> None:
    assert training_stages_required(
        _plan(StageKind.FULL_TRAINING, StageKind.DEVELOPMENT_EVALUATION)
    )
    assert training_stages_required(_plan(StageKind.SMOKE_TRAINING))
    assert not training_stages_required(_plan(StageKind.DEVELOPMENT_EVALUATION))
    assert not training_stages_required(
        _plan(StageKind.FULL_TRAINING, disposition=StageDisposition.NOT_APPLICABLE)
    )
