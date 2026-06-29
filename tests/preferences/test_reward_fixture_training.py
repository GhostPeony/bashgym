import json

from bashgym.preferences import train_reward_model_fixture_file


def _record(
    example_id: str,
    pair_id: str,
    split: str,
    score: float,
    response: str,
    *,
    source_id: str = "helpsteer2",
):
    return {
        "id": example_id,
        "reward_type": "outcome_reward",
        "prompt": "Fix the failing test",
        "response": response,
        "score": score,
        "metadata": {
            "reward_example_id": example_id,
            "pair_id": pair_id,
            "reward_scale": "0_to_1",
            "label_source": "trace_verifier",
            "source_id": source_id,
            "quality_score": 0.95 if score else 0.2,
            "domain": "terminal_agent",
            "task_family": "test_fix",
            "split": split,
            "decontamination_status": "checked",
        },
    }


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_reward_model_fixture_train_writes_evidence_artifacts(tmp_path):
    examples_path = tmp_path / "reward_examples.jsonl"
    output_dir = tmp_path / "smoke"
    _write_jsonl(
        examples_path,
        [
            _record("train-good", "train-pair", "train", 1.0, "patch fix test success"),
            _record("train-bad", "train-pair", "train", 0.0, "ignore failure broken"),
            _record("eval-good", "eval-pair", "eval", 1.0, "patch the failing test"),
            _record("eval-bad", "eval-pair", "eval", 0.0, "ignore the failing test"),
        ],
    )

    report = train_reward_model_fixture_file(
        examples_path,
        output_dir=output_dir,
        epochs=6,
        learning_rate=0.8,
    )

    assert report["ok"] is True
    assert report["train_records"] == 2
    assert report["eval_records"] == 2
    assert report["reward_eval"]["metrics"]["pair_count"] == 1
    assert output_dir.joinpath("reward_model_fixture.json").exists()
    assert output_dir.joinpath("metrics.jsonl").exists()
    assert output_dir.joinpath("reward_predictions.jsonl").exists()
    assert output_dir.joinpath("reward_eval.json").exists()
    predictions = [
        json.loads(line)
        for line in output_dir.joinpath("reward_predictions.jsonl").read_text().splitlines()
    ]
    assert all("predicted_reward" in record for record in predictions)


def test_reward_model_fixture_train_blocks_eval_only_training_sources(tmp_path):
    examples_path = tmp_path / "reward_examples.jsonl"
    output_dir = tmp_path / "smoke"
    _write_jsonl(
        examples_path,
        [
            _record(
                "train-leak",
                "train-pair",
                "train",
                1.0,
                "benchmark reward row",
                source_id="rewardbench",
            ),
            _record("eval-good", "eval-pair", "eval", 1.0, "patch the failing test"),
            _record("eval-bad", "eval-pair", "eval", 0.0, "ignore the failing test"),
        ],
    )

    report = train_reward_model_fixture_file(examples_path, output_dir=output_dir)

    assert report["ok"] is False
    codes = {finding["code"] for finding in report["findings"]}
    assert "eval_only_source_in_training_split" in codes
    assert output_dir.joinpath("reward_model_fixture_report.json").exists()
