import json

from bashgym.preferences import validate_preference_pair_records, validate_preference_pairs_file


def _strict_record():
    return {
        "id": "pair-1",
        "prompt": "Fix the failing test",
        "chosen_response": "Run pytest, inspect the failure, patch the function.",
        "rejected_response": "Ignore the failure and claim success.",
        "metadata": {
            "pair_id": "pair-1",
            "prompt_hash": "abc123",
            "chosen_trace_id": "gold-1",
            "rejected_trace_id": "failed-1",
            "pair_generation_method": "trace_pair",
            "label_strength": "verified_success_vs_failure",
            "label_source": "trace_verifier",
            "chosen_quality_score": 0.95,
            "rejected_quality_score": 0.25,
            "domain": "terminal_agent",
            "task_family": "test_fix",
            "split": "train",
            "decontamination_status": "checked",
        },
    }


def test_strict_preference_pair_validation_accepts_full_metadata():
    result = validate_preference_pair_records([_strict_record()], strict=True)

    assert result["ok"] is True
    assert result["fail_count"] == 0
    assert result["pairs"][0]["pair_id"] == "pair-1"


def test_lightweight_validation_warns_about_serious_run_metadata():
    record = {
        "id": "pair-1",
        "prompt": "Fix it",
        "chosen_response": "Good fix",
        "rejected_response": "Bad fix",
        "metadata": {},
    }

    result = validate_preference_pair_records([record], strict=False)

    assert result["ok"] is True
    codes = {finding["code"] for finding in result["findings"]}
    assert "missing_saved_prompt_hash" in codes
    assert "missing_decontamination_metadata" in codes


def test_strict_validation_fails_missing_provenance_and_identical_answers():
    record = {
        "id": "pair-1",
        "prompt": "Fix it",
        "chosen_response": "Same",
        "rejected_response": "Same",
        "metadata": {},
    }

    result = validate_preference_pair_records([record], strict=True)

    assert result["ok"] is False
    codes = {finding["code"] for finding in result["findings"]}
    assert "identical_chosen_rejected" in codes
    assert "missing_pair_generation_method" in codes
    assert "missing_chosen_trace_provenance" in codes
    assert "missing_rejected_trace_provenance" in codes


def test_preference_pair_file_validation_reads_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_strict_record()) + "\n", encoding="utf-8")

    result = validate_preference_pairs_file(path, strict=True)

    assert result["ok"] is True
    assert result["path"] == str(path)
