from __future__ import annotations

import pytest

from scripts.memexai.build_positive_aware_campaign_report import (
    paired_bootstrap_mrr_deltas,
    rolling_mean,
    summarize_training_metrics,
)


def test_rolling_mean_uses_only_available_history() -> None:
    assert rolling_mean([1.0, 2.0, 3.0, 4.0], 3) == pytest.approx([1.0, 1.5, 2.0, 3.0])


def test_training_summary_requires_a_complete_step_series() -> None:
    manifest = {
        "training": {
            "optimizer_steps": 4,
            "collision_safe_batches_per_epoch": 2,
            "result_metrics": {"train_loss": 0.25},
        }
    }
    rows = [
        {"step": 1, "loss": 1.0, "learning_rate": 1e-6},
        {"step": 2, "loss": 0.5, "learning_rate": 8e-7},
        {"step": 3, "loss": 0.25, "learning_rate": 4e-7},
        {"step": 4, "loss": 0.125, "learning_rate": 0.0},
    ]
    summary = summarize_training_metrics(manifest, rows)
    assert summary["loss_steps"] == [1, 2, 3, 4]
    assert summary["loss_epoch_1"]["mean"] == pytest.approx(0.75)
    assert summary["loss_epoch_2"]["mean"] == pytest.approx(0.1875)

    with pytest.raises(ValueError, match="incomplete"):
        summarize_training_metrics(manifest, rows[:-1])


def test_paired_bootstrap_joins_rows_by_eval_id() -> None:
    base = [
        {
            "eval_id": "a",
            "positive_rank_exact": 1,
            "positive_rank_local_window": 1,
            "positive_rank_same_video": 1,
        },
        {
            "eval_id": "b",
            "positive_rank_exact": 2,
            "positive_rank_local_window": 2,
            "positive_rank_same_video": 2,
        },
    ]
    candidate = [
        {
            "eval_id": "b",
            "positive_rank_exact": 4,
            "positive_rank_local_window": 4,
            "positive_rank_same_video": 4,
        },
        {
            "eval_id": "a",
            "positive_rank_exact": 2,
            "positive_rank_local_window": 2,
            "positive_rank_same_video": 2,
        },
    ]
    result = paired_bootstrap_mrr_deltas(base, candidate, samples=200, seed=7)
    assert result["exact_mrr"]["delta"] == pytest.approx(-0.375)
    assert result["local_window_mrr"]["rows"] == 2
    assert result["same_video_mrr"]["bootstrap_samples"] == 200


def test_paired_bootstrap_rejects_mismatched_eval_sets() -> None:
    row = {
        "eval_id": "a",
        "positive_rank_exact": 1,
        "positive_rank_local_window": 1,
        "positive_rank_same_video": 1,
    }
    with pytest.raises(ValueError, match="identical eval IDs"):
        paired_bootstrap_mrr_deltas([row], [{**row, "eval_id": "b"}], samples=10)
