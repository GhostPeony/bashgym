from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from bashgym.api.schemas import ExportExamplesRequest
from bashgym.factory.data_factory import TrainingExample
from bashgym.factory.example_generator import ExampleGenerator
from bashgym.factory.export_artifacts import resolve_training_export


def _examples():
    return [
        TrainingExample(
            str(index),
            "system",
            f"task {index}",
            f"answer {index}",
            metadata={"repo_name": f"repo-{index}", "task_id": f"task-{index}"},
        )
        for index in range(4)
    ]


@pytest.mark.asyncio
async def test_real_export_route_groups_and_downloads_exact_artifact(tmp_path, monkeypatch):
    from bashgym.api.routes import app

    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "bashgym.config.get_settings",
        lambda: SimpleNamespace(data=SimpleNamespace(data_dir=str(tmp_path))),
    )
    traces = tmp_path / "gold_traces"
    traces.mkdir()
    (traces / "fixture.json").write_text("{}")
    monkeypatch.setattr(ExampleGenerator, "generate_examples", lambda *args: _examples())
    endpoints = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
    result = await endpoints["/api/training/export"](
        ExportExamplesRequest(train_split=0.5, split_group_by="task", split_seed=37)
    )
    assert result.success, result.message
    # The configured and global directory are identical; repeated discovery must deduplicate.
    assert result.train_count + result.val_count == 4
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["export_id"] == result.export_id
    for split, expected in (("train", result.train_path), ("val", result.val_path)):
        response = await endpoints["/api/training/export/download"](split, result.export_id)
        assert str(response.path) == expected
    Path(result.train_path).write_text("changed")
    with pytest.raises(HTTPException) as error:
        await endpoints["/api/training/export/download"]("train", result.export_id)
    assert error.value.status_code == 409


def test_complete_manifest_selects_exact_files_and_detects_tampering(tmp_path):
    exported = ExampleGenerator().export_for_nemo(_examples(), tmp_path, train_split=0.5)
    assert resolve_training_export(tmp_path, "train") == exported["train"]
    assert resolve_training_export(tmp_path, "val", exported["export_id"]) == exported["validation"]
    manifest = json.loads(exported["manifest"].read_text())
    manifest["files"]["train"]["name"] = "../private.jsonl"
    exported["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        resolve_training_export(tmp_path, "train")


def test_unpublished_bundle_never_becomes_a_download(tmp_path):
    (tmp_path / ("personal_" + "a" * 64 + "_train.jsonl")).write_text("{}\n")
    with pytest.raises(FileNotFoundError):
        resolve_training_export(tmp_path, "train")
    with pytest.raises(FileNotFoundError):
        resolve_training_export(tmp_path, "train", "a" * 64)


def test_changing_data_and_manifest_cannot_preserve_export_identity(tmp_path):
    import hashlib

    exported = ExampleGenerator().export_for_nemo(_examples(), tmp_path, train_split=0.5)
    exported["train"].write_bytes(b"changed\n")
    manifest = json.loads(exported["manifest"].read_text())
    changed = hashlib.sha256(b"changed\n").hexdigest()
    manifest["files"]["train"]["sha256"] = changed
    manifest["file_hashes"]["train"] = changed
    exported["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        resolve_training_export(tmp_path, "train", exported["export_id"])


def test_legacy_download_and_invalid_identity(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("{}\n")
    assert resolve_training_export(tmp_path, "train") == path
    with pytest.raises(ValueError, match="identity"):
        resolve_training_export(tmp_path, "train", "../outside")


def test_export_request_rejects_row_splitting():
    with pytest.raises(ValueError):
        ExportExamplesRequest(split_group_by="row")
