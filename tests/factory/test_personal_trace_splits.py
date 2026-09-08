from __future__ import annotations

import json

import pytest

from bashgym.factory.data_factory import TrainingExample
from bashgym.factory.example_generator import ExampleGenerator, ExampleGeneratorConfig


def _example(label, repo=None, **metadata):
    return TrainingExample(
        example_id=label,
        system_prompt="system",
        user_prompt="prompt " + label,
        assistant_response="answer " + label,
        metadata={**({"repo_id": repo} if repo else {}), **metadata},
    )


def _generator(tmp_path):
    return ExampleGenerator(ExampleGeneratorConfig(output_dir=str(tmp_path / "unused")))


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_repository_groups_and_session_links_never_cross(tmp_path):
    examples = [
        _example("a", "repo-a", repo_name="alias-a"),
        _example("b", repo_name="alias-a", session_id="shared-session"),
        _example("c", "repo-c", session_id="shared-session"),
        _example("d", "repo-d"),
        _example("e", "repo-d"),
    ]
    result = _generator(tmp_path).export_for_nemo(examples, tmp_path / "out", train_split=0.5)
    partitions = [_records(result[key]) for key in ("train", "validation")]
    groups = [{row["metadata"]["split_group_sha256"] for row in rows} for rows in partitions]
    assert groups[0].isdisjoint(groups[1])
    places = {
        row["metadata"]["source_example_id"]: partition
        for partition, rows in enumerate(partitions)
        for row in rows
    }
    assert places["a"] == places["b"] == places["c"]
    assert places["d"] == places["e"]
    assert places["a"] != places["d"]
    manifest = json.loads(result["manifest"].read_text())
    assert manifest["group_count"] == 2
    assert manifest["method"] == "deterministic_grouped_v1"
    assert manifest["files"]["train"]["name"] == result["train"].name
    assert "repo-a" not in result["manifest"].read_text()


def test_export_is_order_invariant_and_replays_exact_files(tmp_path):
    examples = [_example(str(index), "repo-" + str(index // 3)) for index in range(15)]
    generator = _generator(tmp_path)
    first = generator.export_for_nemo(examples, tmp_path / "one", split_seed=17)
    reversed_export = generator.export_for_nemo(
        list(reversed(examples)), tmp_path / "two", split_seed=17
    )
    assert first["export_id"] == reversed_export["export_id"]
    for key in ("train", "validation", "manifest"):
        assert first[key].read_bytes() == reversed_export[key].read_bytes()
    assert generator.export_for_nemo(examples, tmp_path / "one", split_seed=17) == first


@pytest.mark.parametrize("grouping", ["repository", "task"])
def test_missing_identity_fails_before_export_files(tmp_path, grouping):
    generator = _generator(tmp_path)
    with pytest.raises(ValueError, match="identity_required"):
        generator.export_for_nemo([_example("missing")], tmp_path / "out", split_group_by=grouping)
    assert not (tmp_path / "out").exists()


def test_single_repository_cannot_be_a_nonempty_holdout(tmp_path):
    with pytest.raises(ValueError, match="holdout_requires_multiple_groups"):
        _generator(tmp_path).export_for_nemo(
            [_example("a", "same"), _example("b", "same")], tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_exact_duplicates_merge_sources_without_cross_partition_leakage(tmp_path):
    one = _example("same", "repo-a", source_trace_sha256="a" * 64, generated_at="earlier")
    duplicate = _example("same", "repo-b", source_trace_sha256="b" * 64, generated_at="later")
    other = _example("different", "repo-c")
    result = _generator(tmp_path).export_for_nemo(
        [one, duplicate, one, other], tmp_path / "out", train_split=0.5
    )
    rows = _records(result["train"]) + _records(result["validation"])
    assert len(rows) == 2
    merged = next(row for row in rows if row["metadata"]["source_example_id"] == "same")
    assert {item["repo_id"] for item in merged["metadata"]["provenance"]} == {"repo-a", "repo-b"}
    manifest = json.loads(result["manifest"].read_text())
    assert manifest["duplicate_count"] == 2
    assert manifest["unique_count"] == 2
    assert "generated_at" not in json.dumps(rows)


def test_explicit_task_split_requires_task_ids_and_keeps_tasks_together(tmp_path):
    examples = [
        _example("a", task_id="one"),
        _example("b", task_id="one"),
        _example("c", task_id="two"),
    ]
    result = _generator(tmp_path).export_for_nemo(
        examples, tmp_path / "out", split_group_by="task", train_split=0.5
    )
    places = {
        row["metadata"]["source_example_id"]: key
        for key in ("train", "validation")
        for row in _records(result[key])
    }
    assert places["a"] == places["b"] != places["c"]


def test_training_only_allows_one_group_and_records_empty_validation(tmp_path):
    result = _generator(tmp_path).export_for_nemo(
        [_example("a", "only"), _example("b", "only")], tmp_path / "out", train_split=1
    )
    assert result["train_count"] == 2
    assert result["val_count"] == 0
    assert result["validation"].read_bytes() == b""
    manifest = json.loads(result["manifest"].read_text())
    assert manifest["group_hashes"]["validation"] == []


def test_generate_examples_preserves_explicit_provenance_and_dedups_copied_roots(tmp_path):
    payload = {
        "session_id": "session-one",
        "task_id": "task-one",
        "metadata": {"primary_repo": {"id": "repo-one", "name": "fixture"}},
        "trace": [
            {"tool": "Bash", "command": "echo fixture", "output": "fixture", "success": True}
        ],
    }
    first = tmp_path / "first.json"
    copy = tmp_path / "copy.json"
    first.write_text(json.dumps(payload))
    copy.write_bytes(first.read_bytes())
    generator = _generator(tmp_path)
    examples = generator.generate_examples(first) + generator.generate_examples(copy)
    assert len(examples) == 2
    assert examples[0].metadata["session_id"] == "session-one"
    assert examples[0].metadata["task_id"] == "task-one"
    assert examples[0].metadata["repo_id"] == "repo-one"
    assert (
        examples[0].metadata["source_trace_sha256"] == examples[1].metadata["source_trace_sha256"]
    )
    result = generator.export_for_nemo(examples, tmp_path / "out", train_split=1)
    assert result["train_count"] == 1
    assert json.loads(result["manifest"].read_text())["duplicate_count"] == 1


def test_interrupted_publication_does_not_publish_manifest_and_can_retry(tmp_path, monkeypatch):
    from bashgym.factory import example_generator

    generator = _generator(tmp_path)
    examples = [_example("a", "one"), _example("b", "two")]
    original = example_generator.os.link
    calls = []

    def interrupted(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("injected interruption")
        return original(source, destination)

    monkeypatch.setattr(example_generator.os, "link", interrupted)
    with pytest.raises(OSError, match="interruption"):
        generator.export_for_nemo(examples, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*_manifest.json"))
    result = generator.export_for_nemo(examples, tmp_path / "out")
    assert result["manifest"].is_file()


@pytest.mark.parametrize(
    "options",
    [
        {"train_split": 0},
        {"train_split": float("nan")},
        {"train_split": 1.1},
        {"split_group_by": "row"},
        {"split_seed": True},
    ],
)
def test_invalid_split_parameters_fail_without_output(tmp_path, options):
    with pytest.raises(ValueError, match="personal_trace_"):
        _generator(tmp_path).export_for_nemo(
            [_example("a", "one"), _example("b", "two")], tmp_path / "out", **options
        )
    assert not (tmp_path / "out").exists()
