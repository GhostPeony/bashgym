import json

from bashgym.environments import EnvironmentSpec
from bashgym.preferences import validate_preference_pairs_file, validate_reward_examples_file
from bashgym.sources import get_source, prepare_source_artifacts


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _preference_record():
    return {
        "id": "uf-1",
        "prompt": "Explain how to debug a failing pytest case.",
        "chosen": [{"role": "assistant", "content": "Run pytest, read the failure, patch the cause."}],
        "rejected": [{"role": "assistant", "content": "Assume the tests are wrong and skip them."}],
        "split": "train",
        "metadata": {
            "quality_score": 0.9,
            "label_source": "fixture_preference",
            "decontamination_status": "checked",
        },
    }


def test_preference_source_adapter_writes_strict_dpo_pairs(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [_preference_record()])

    report = prepare_source_artifacts(
        get_source("ultrafeedback_binarized"),
        goal="dpo",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    artifact = report["artifacts"][0]
    assert report["ok"] is True
    assert artifact["artifact_type"] == "dpo_pairs"
    assert artifact["validation"]["ok"] is True
    assert artifact["record_count"] == 1
    assert validate_preference_pairs_file(artifact["path"], strict=True)["ok"] is True


def test_preference_source_adapter_writes_strict_reward_examples(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [_preference_record()])

    report = prepare_source_artifacts(
        get_source("helpsteer2"),
        goal="reward_model",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    artifact = report["artifacts"][0]
    assert report["ok"] is True
    assert artifact["artifact_type"] == "reward_examples"
    assert artifact["record_count"] == 2
    assert validate_reward_examples_file(artifact["path"], strict=True)["ok"] is True


def test_ultrafeedback_mapper_records_source_schema_metadata(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "id": "uf-1",
                "prompt": "Fix a flaky test.",
                "chosen": "Inspect the failure and remove nondeterminism.",
                "rejected": "Rerun until it passes.",
                "chosen_score": 4.0,
                "rejected_score": 1.0,
                "metadata": {"decontamination_status": "checked"},
            }
        ],
    )

    report = prepare_source_artifacts(
        get_source("ultrafeedback_binarized"),
        goal="dpo",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    pair = json.loads((tmp_path / "out" / "dpo_pairs.jsonl").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["source_schema_mapping"]["mapper"] == "ultrafeedback_binarized"
    assert pair["metadata"]["source_schema"] == "ultrafeedback_binarized"
    assert pair["metadata"]["chosen_quality_score"] == 1.0
    assert pair["metadata"]["rejected_quality_score"] == 0.25
    assert pair["metadata"]["score_delta"] == 3.0


def test_helpsteer2_preference_strength_maps_to_dpo_pairs(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "id": "hs-pref-1",
                "prompt": "Explain a safe migration plan.",
                "response_1": "Back up data, test in staging, then migrate.",
                "response_2": "Run it directly in production.",
                "preference_strength": -2,
                "metadata": {"decontamination_status": "checked"},
            },
            {
                "id": "hs-pref-2",
                "prompt": "Explain a rollback plan.",
                "response_1": "Hope it works.",
                "response_2": "Keep the previous build ready and document rollback steps.",
                "preference_strength": 3,
                "metadata": {"decontamination_status": "checked"},
            },
        ],
    )

    report = prepare_source_artifacts(
        get_source("helpsteer2"),
        goal="dpo",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    pairs = [
        json.loads(line)
        for line in (tmp_path / "out" / "dpo_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["ok"] is True
    assert report["source_schema_mapping"]["mapper"] == "helpsteer2_preference_pairs"
    assert report["converted_count"] == 2
    assert pairs[0]["chosen_response"].startswith("Back up data")
    assert pairs[0]["metadata"]["source_schema"] == "helpsteer2_preference"
    assert pairs[0]["metadata"]["preference_outcome"] == "response_1"
    assert pairs[1]["chosen_response"].startswith("Keep the previous build")
    assert pairs[1]["metadata"]["preference_outcome"] == "response_2"
    assert validate_preference_pairs_file(report["artifacts"][0]["path"], strict=True)["ok"] is True


def test_helpsteer2_scored_responses_build_dpo_pairs_by_prompt(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "id": "hs-score-1",
                "prompt": "How should I debug a failing CLI test?",
                "response": "Skip it.",
                "helpfulness": 1,
                "correctness": 1,
                "coherence": 2,
                "complexity": 1,
                "verbosity": 1,
                "metadata": {"decontamination_status": "checked"},
            },
            {
                "id": "hs-score-2",
                "prompt": "How should I debug a failing CLI test?",
                "response": "Reproduce the failure, inspect the traceback, patch the root cause.",
                "helpfulness": 4,
                "correctness": 4,
                "coherence": 4,
                "complexity": 3,
                "verbosity": 3,
                "metadata": {"decontamination_status": "checked"},
            },
        ],
    )

    report = prepare_source_artifacts(
        get_source("helpsteer2"),
        goal="dpo",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    pair = json.loads((tmp_path / "out" / "dpo_pairs.jsonl").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["source_schema_mapping"]["mapper"] == "helpsteer2_scored_response_pairs"
    assert report["source_schema_mapping"]["consumed_records"] == 2
    assert report["source_schema_mapping"]["dropped_records"] == 0
    assert "source_schema_mapping_dropped_records" not in report["warnings"]
    assert pair["chosen_response"].startswith("Reproduce the failure")
    assert pair["metadata"]["source_schema"] == "helpsteer2_scored_response_pair"
    assert pair["metadata"]["score_delta"] > 0
    assert pair["metadata"]["reward_scale"] == "likert_0_to_4"
    assert validate_preference_pairs_file(report["artifacts"][0]["path"], strict=True)["ok"] is True


def test_helpsteer2_reward_model_preserves_axis_scores(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "id": "hs-reward-1",
                "prompt": "Explain git bisect.",
                "response": "Use binary search over commits to isolate the regression.",
                "helpfulness": 4,
                "correctness": 4,
                "coherence": 3,
                "complexity": 2,
                "verbosity": 2,
                "metadata": {"decontamination_status": "checked"},
            }
        ],
    )

    report = prepare_source_artifacts(
        get_source("helpsteer2"),
        goal="reward_model",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    example = json.loads((tmp_path / "out" / "reward_examples.jsonl").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["source_schema_mapping"]["mapper"] == "helpsteer2_scored_response"
    assert example["score"] == 3.0
    assert example["metadata"]["source_schema"] == "helpsteer2_scored_response"
    assert example["metadata"]["score_axes"]["helpfulness"] == 4.0
    assert example["metadata"]["reward_scale"] == "likert_0_to_4"
    assert validate_reward_examples_file(report["artifacts"][0]["path"], strict=True)["ok"] is True


def test_eval_only_source_adapter_blocks_training_by_default(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [_preference_record()])

    report = prepare_source_artifacts(
        get_source("rewardbench"),
        goal="reward_model",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    assert report["ok"] is False
    assert "eval_only_source_for_training" in report["errors"]
    assert not report["artifacts"]


def test_eval_source_adapter_writes_eval_manifest(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [{"id": "bench-1", "prompt": "Call a tool", "expected": "ok"}])

    report = prepare_source_artifacts(
        get_source("bfcl"),
        goal="evaluation",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    artifact = report["artifacts"][0]
    manifest = json.loads((tmp_path / "out" / "eval_manifest.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert artifact["artifact_type"] == "eval_manifest"
    assert manifest["eval_only"] is True
    assert manifest["record_ids"] == ["bench-1"]


def test_environment_source_adapter_writes_environment_specs_for_eval(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "id": "term-1",
                "instruction": "Create a file named answer.txt with ok inside.",
                "domain": "terminal_agents",
                "skills": ["bash", "file_ops"],
                "files": {"verify.sh": "test -f answer.txt && grep -q ok answer.txt\n"},
                "verifier": {"kind": "script", "command": "bash verify.sh", "path": "verify.sh"},
                "rollout": {"max_tool_calls": 8, "timeout_sec": 300},
                "metadata": {"decontamination_status": "checked", "quality_score": 0.8},
            }
        ],
    )

    report = prepare_source_artifacts(
        get_source("harbor_terminal_bench"),
        goal="evaluation",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    artifacts = {artifact["artifact_type"]: artifact for artifact in report["artifacts"]}
    spec_line = (tmp_path / "out" / "environment_specs.jsonl").read_text(encoding="utf-8").strip()
    spec = EnvironmentSpec.from_dict(json.loads(spec_line))
    assert report["ok"] is True
    assert artifacts["environment_specs"]["validation"]["ok"] is True
    assert spec.id == "term-1"
    assert spec.verifier.command == "bash verify.sh"
    assert spec.metadata["source_id"] == "harbor_terminal_bench"


def test_environment_source_adapter_blocks_eval_only_terminal_training(tmp_path):
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "id": "term-1",
                "instruction": "Create answer.txt",
                "verifier": {"command": "test -f answer.txt"},
            }
        ],
    )

    report = prepare_source_artifacts(
        get_source("harbor_terminal_bench"),
        goal="terminal_rl",
        input_path=source_path,
        output_dir=tmp_path / "out",
    )

    assert report["ok"] is False
    assert "eval_only_source_for_training" in report["errors"]
    assert not report["artifacts"]
