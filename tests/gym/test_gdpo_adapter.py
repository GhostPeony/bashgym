"""Tests for the named-reward NeMo RL GDPO adapter."""

from __future__ import annotations

import math

import pytest

from bashgym.environments.contracts import RewardComponentSpec
from bashgym.gym.gdpo_adapter import NamedRewardGDPOAdapter, NemoGDPOConfig
from bashgym.gym.policy_optimization import gdpo_advantages


def _adapter() -> NamedRewardGDPOAdapter:
    return NamedRewardGDPOAdapter.from_reward_specs(
        [
            RewardComponentSpec(name="correctness", weight=1.0),
            RewardComponentSpec(name="format", weight=0.25),
        ]
    )


def test_adapter_builds_stable_nemo_columns_and_reference_advantages():
    adapter = _adapter()
    group_ids = ["prompt-a", "prompt-a", "prompt-b", "prompt-b"]
    rows = [
        {"correctness": 0.0, "format": 1.0},
        {"format": 0.0, "correctness": 1.0},
        {"correctness": 1.0, "format": 0.5},
        {"correctness": 0.0, "format": 1.0},
    ]

    binding = adapter.bind(group_ids=group_ids, reward_rows=rows)

    assert binding.component_order == ("correctness", "format")
    assert binding.component_to_nemo_key == {
        "correctness": "reward1",
        "format": "reward2",
    }
    assert binding.to_nemo_batch() == {
        "group_ids": group_ids,
        "reward1": [0.0, 1.0, 1.0, 0.0],
        "reward2": [1.0, 0.0, 0.5, 1.0],
        "total_reward": [0.25, 1.0, 1.125, 0.25],
    }

    direct = gdpo_advantages(
        group_ids=group_ids,
        reward_components={
            "correctness": [0.0, 1.0, 1.0, 0.0],
            "format": [1.0, 0.0, 0.5, 1.0],
        },
        weights={"correctness": 1.0, "format": 0.25},
    )
    assert binding.reference_advantages.advantages == direct.advantages
    assert binding.reference_advantages.component_advantages == direct.component_advantages


def test_adapter_exposes_exact_nemo_gdpo_overrides():
    config = NemoGDPOConfig()

    assert config.to_nested_dict() == {
        "grpo": {
            "adv_estimator": {
                "name": "gdpo",
                "normalize_rewards": True,
                "use_leave_one_out_baseline": False,
            }
        }
    }
    assert config.to_flat_dict() == {
        "grpo.adv_estimator.name": "gdpo",
        "grpo.adv_estimator.normalize_rewards": True,
        "grpo.adv_estimator.use_leave_one_out_baseline": False,
    }


def test_binding_receipt_is_deterministic_and_changes_with_batch():
    adapter = _adapter()
    kwargs = {
        "group_ids": [1, 1],
        "reward_rows": [
            {"correctness": 0.0, "format": 1.0},
            {"correctness": 1.0, "format": 0.0},
        ],
    }

    first = adapter.bind(**kwargs)
    second = adapter.bind(**kwargs)
    changed = adapter.bind(
        group_ids=[1, 1],
        reward_rows=[
            {"correctness": 0.0, "format": 1.0},
            {"correctness": 0.9, "format": 0.0},
        ],
    )

    assert first.receipt == second.receipt
    assert first.receipt.receipt_digest == second.receipt.receipt_digest
    assert first.receipt.batch_digest != changed.receipt.batch_digest
    assert first.receipt.receipt_digest != changed.receipt.receipt_digest
    assert len(first.receipt.receipt_digest) == 64


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"correctness": 1.0}, {"correctness": 0.0}], "component mismatch"),
        (
            [
                {"correctness": 1.0, "format": 0.0, "extra": 1.0},
                {"correctness": 0.0, "format": 1.0, "extra": 0.0},
            ],
            "component mismatch",
        ),
        (
            [
                {"correctness": math.nan, "format": 0.0},
                {"correctness": 0.0, "format": 1.0},
            ],
            "must be finite",
        ),
    ],
)
def test_adapter_rejects_missing_extra_and_non_finite_rewards(rows, message):
    with pytest.raises(ValueError, match=message):
        _adapter().bind(group_ids=["p", "p"], reward_rows=rows)


def test_adapter_rejects_component_order_and_weight_drift():
    adapter = _adapter()
    rows = [
        {"correctness": 1.0, "format": 0.0},
        {"correctness": 0.0, "format": 1.0},
    ]

    with pytest.raises(ValueError, match="order mismatch"):
        adapter.bind(
            group_ids=["p", "p"],
            reward_rows=rows,
            component_order=["format", "correctness"],
            weights={"correctness": 1.0, "format": 0.25},
        )
    with pytest.raises(ValueError, match="weight mismatch"):
        adapter.bind(
            group_ids=["p", "p"],
            reward_rows=rows,
            component_order=["correctness", "format"],
            weights={"correctness": 1.0, "format": 1.0},
        )


@pytest.mark.parametrize(
    ("group_ids", "message"),
    [
        (["prompt-a", "prompt-b"], "at least two samples"),
        (["", ""], "must be non-empty"),
        ([["unhashable"], ["unhashable"]], "string or integer"),
    ],
)
def test_adapter_rejects_malformed_groups(group_ids, message):
    rows = [
        {"correctness": 1.0, "format": 0.0},
        {"correctness": 0.0, "format": 1.0},
    ]

    with pytest.raises(ValueError, match=message):
        _adapter().bind(group_ids=group_ids, reward_rows=rows)


def test_adapter_rejects_invalid_component_contracts():
    with pytest.raises(ValueError, match="at least two"):
        NamedRewardGDPOAdapter.from_reward_specs([RewardComponentSpec(name="only")])
    with pytest.raises(ValueError, match="duplicate"):
        NamedRewardGDPOAdapter.from_reward_specs(
            [RewardComponentSpec(name="same"), RewardComponentSpec(name="same")]
        )
    with pytest.raises(ValueError, match="non-zero"):
        NamedRewardGDPOAdapter.from_reward_specs(
            [
                RewardComponentSpec(name="first", weight=0),
                RewardComponentSpec(name="second", weight=0),
            ]
        )
