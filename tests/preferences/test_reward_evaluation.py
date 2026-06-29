import json

from bashgym.preferences import evaluate_reward_model_file, evaluate_reward_model_records


def _record(
    example_id: str, group: str, true_score: float, pred_score: float, *, family="test_fix"
):
    return {
        "id": example_id,
        "reward_type": "outcome_reward",
        "prompt": "Fix the failing test",
        "response": f"candidate {example_id}",
        "score": true_score,
        "predicted_reward": pred_score,
        "metadata": {
            "reward_example_id": example_id,
            "pair_id": group,
            "reward_scale": "0_to_1",
            "label_source": "trace_verifier",
            "source_id": "helpsteer2",
            "quality_score": 0.95,
            "domain": "terminal_agent",
            "task_family": family,
            "split": "eval",
            "decontamination_status": "checked",
        },
    }


def test_reward_model_eval_computes_heldout_metrics():
    result = evaluate_reward_model_records(
        [
            _record("reward-1a", "pair-1", 1.0, 0.9),
            _record("reward-1b", "pair-1", 0.0, 0.2),
            _record("reward-2a", "pair-2", 1.0, 0.8, family="tool_use"),
            _record("reward-2b", "pair-2", 0.0, 0.1, family="tool_use"),
        ],
        split="eval",
    )

    assert result["ok"] is True
    assert result["evaluated_records"] == 4
    assert result["prediction_records"] == 4
    assert result["metrics"]["pair_count"] == 2
    assert result["metrics"]["heldout_pair_accuracy"] == 1.0
    assert result["metrics"]["reward_margin"] > 0
    assert result["metrics"]["calibration_error"] is not None
    assert result["metrics"]["reward_variance"] is not None
    assert {row["task_family"] for row in result["task_family_breakdown"]} == {
        "test_fix",
        "tool_use",
    }


def test_reward_model_eval_fails_missing_predictions():
    record = _record("reward-1a", "pair-1", 1.0, 0.9)
    del record["predicted_reward"]

    result = evaluate_reward_model_records([record], split="eval")

    assert result["ok"] is False
    codes = {finding["code"] for finding in result["findings"]}
    assert "missing_reward_model_predictions" in codes


def test_reward_model_eval_blocks_eval_only_source_leakage():
    record = _record("reward-1a", "pair-1", 1.0, 0.9)
    record["metadata"]["source_id"] = "rewardbench"

    result = evaluate_reward_model_records([record], split="eval")

    assert result["ok"] is False
    assert result["metrics"]["eval_only_leakage_count"] == 1
    codes = {finding["code"] for finding in result["findings"]}
    assert "eval_only_leakage" in codes


def test_reward_model_eval_file_writes_path_metadata(tmp_path):
    path = tmp_path / "reward_predictions.jsonl"
    path.write_text(
        json.dumps(_record("reward-1a", "pair-1", 1.0, 0.9))
        + "\n"
        + json.dumps(_record("reward-1b", "pair-1", 0.0, 0.2))
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_reward_model_file(path)

    assert result["ok"] is True
    assert result["path"] == str(path)
