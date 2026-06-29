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
