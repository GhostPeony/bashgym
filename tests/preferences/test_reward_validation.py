import json

from bashgym.preferences import validate_reward_example_records, validate_reward_examples_file


def _strict_reward_record():
    return {
        "id": "reward-1",
        "reward_type": "outcome_reward",
        "prompt": "Fix the failing test",
        "response": "Run pytest, inspect the failure, patch the function.",
        "score": 0.9,
        "metadata": {
            "reward_example_id": "reward-1",
            "reward_scale": "0_to_1",
            "label_source": "trace_verifier",
            "source_id": "helpsteer2",
            "quality_score": 0.95,
            "domain": "terminal_agent",
            "task_family": "test_fix",
            "split": "train",
            "decontamination_status": "checked",
        },
    }


def test_strict_reward_validation_accepts_full_metadata():
    result = validate_reward_example_records([_strict_reward_record()], strict=True)

    assert result["ok"] is True
    assert result["fail_count"] == 0
    assert result["examples"][0]["example_id"] == "reward-1"


def test_lightweight_reward_validation_warns_about_serious_run_metadata():
    record = {
        "id": "reward-1",
        "prompt": "Fix it",
        "response": "Good fix",
        "score": 1.0,
        "metadata": {},
    }

    result = validate_reward_example_records([record], strict=False)

    assert result["ok"] is True
    codes = {finding["code"] for finding in result["findings"]}
    assert "missing_reward_type" in codes
    assert "missing_decontamination_metadata" in codes


def test_strict_reward_validation_fails_missing_process_steps():
    record = {
        "id": "reward-1",
        "reward_type": "process_reward",
        "prompt": "Fix it",
        "response": "Good fix",
        "score": 1.0,
        "metadata": {},
    }

    result = validate_reward_example_records([record], strict=True)

    assert result["ok"] is False
    codes = {finding["code"] for finding in result["findings"]}
    assert "missing_process_reward_steps" in codes
    assert "missing_reward_scale" in codes
    assert "missing_source_provenance" in codes


def test_reward_example_file_validation_reads_jsonl(tmp_path):
    path = tmp_path / "reward_examples.jsonl"
    path.write_text(json.dumps(_strict_reward_record()) + "\n", encoding="utf-8")

    result = validate_reward_examples_file(path, strict=True)

    assert result["ok"] is True
    assert result["path"] == str(path)
