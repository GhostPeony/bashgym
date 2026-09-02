"""Explicit data edges between stage plan items."""

import pytest

from bashgym.campaigns.contracts import StageDisposition, StageKind, StagePlan, StagePlanItem


def _item(stage: StageKind, *, consumes=()) -> StagePlanItem:
    return StagePlanItem(
        stage=stage,
        disposition=StageDisposition.REQUIRED,
        reason="edge test",
        consumes=tuple(consumes),
    )


def test_declared_edges_are_returned_verbatim() -> None:
    plan = StagePlan(
        items=(
            _item(StageKind.DATA_BUILD),
            _item(StageKind.FULL_TRAINING, consumes=(StageKind.DATA_BUILD,)),
            _item(StageKind.DEVELOPMENT_EVALUATION, consumes=(StageKind.FULL_TRAINING,)),
        )
    )

    assert plan.consumed_stages(0) == ()
    assert plan.consumed_stages(1) == (StageKind.DATA_BUILD,)
    assert plan.consumed_stages(2) == (StageKind.FULL_TRAINING,)


def test_legacy_plans_without_edges_use_the_positional_rule() -> None:
    plan = StagePlan(
        items=(
            _item(StageKind.DATA_BUILD),
            _item(StageKind.FULL_TRAINING),
            _item(StageKind.DEVELOPMENT_EVALUATION),
        )
    )

    assert plan.consumed_stages(1) == (StageKind.DATA_BUILD,)
    assert plan.consumed_stages(2) == (StageKind.FULL_TRAINING,)
    assert StagePlan(items=(_item(StageKind.DEVELOPMENT_EVALUATION),)).consumed_stages(0) == ()


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (
            (_item(StageKind.FULL_TRAINING, consumes=(StageKind.DATA_BUILD,)),),
            "unknown stage",
        ),
        (
            (
                _item(StageKind.FULL_TRAINING, consumes=(StageKind.DEVELOPMENT_EVALUATION,)),
                _item(StageKind.DEVELOPMENT_EVALUATION),
            ),
            "later stage",
        ),
        (
            (_item(StageKind.DATA_BUILD, consumes=(StageKind.DATA_BUILD,)),),
            "consume itself",
        ),
    ],
)
def test_invalid_edges_are_rejected(items, message) -> None:
    with pytest.raises(ValueError, match=message):
        StagePlan(items=items)


def test_persisted_v1_items_without_consumes_still_validate() -> None:
    item = StagePlanItem.model_validate(
        {
            "schema_version": "campaign_stage_plan_item.v1",
            "stage": "full_training",
            "disposition": "required",
            "reason": "legacy row",
        }
    )

    assert item.consumes == ()
