from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from bashgym.api.routes import app

client = TestClient(app)


@pytest.fixture
def configured_hf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_client = SimpleNamespace(is_enabled=True, namespace="ghostpeony")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "bashgym.integrations.huggingface.get_hf_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "bashgym.config.get_settings",
        lambda: SimpleNamespace(data=SimpleNamespace(data_dir=tmp_path)),
    )
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path)

    def fake_upload_training_data(self, **kwargs):
        captured.update(kwargs)
        return "https://huggingface.co/datasets/ghostpeony/designer-run"

    monkeypatch.setattr(
        "bashgym.integrations.huggingface.datasets.HFDatasetManager.upload_training_data",
        fake_upload_training_data,
    )
    return captured


def test_uploads_generated_train_queries_as_canonical_train_split(
    configured_hf: dict[str, object],
    tmp_path: Path,
):
    output_dir = tmp_path / "designer-run"
    output_dir.mkdir()
    source = output_dir / "train_queries.jsonl"
    source.write_text('{"messages": []}\n{"messages": []}\n', encoding="utf-8")

    response = client.post(
        "/api/hf/datasets",
        json={
            "local_path": str(output_dir),
            "repo_name": "designer-run",
            "private": True,
            "metadata": {"job_name": "real_chunks"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "repo_id": "ghostpeony/designer-run",
        "url": "https://huggingface.co/datasets/ghostpeony/designer-run",
        "train_count": 2,
        "val_count": 0,
    }
    assert configured_hf["train_file"] == source
    assert configured_hf["val_file"] is None
    assert configured_hf["metadata"] == {
        "job_name": "real_chunks",
        "bashgym_source_file": "train_queries.jsonl",
    }


def test_rejects_completed_output_without_a_publishable_jsonl(
    configured_hf: dict[str, object],
    tmp_path: Path,
):
    output_dir = tmp_path / "empty-run"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")

    response = client.post(
        "/api/hf/datasets",
        json={"local_path": str(output_dir), "repo_name": "empty-run"},
    )

    assert response.status_code == 400
    assert "No train.jsonl" in response.json()["detail"]
    assert configured_hf == {}


def test_manager_publishes_custom_generated_file_as_train_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from bashgym.integrations.huggingface.datasets import HFDatasetManager

    generated_file = tmp_path / "train_queries.jsonl"
    generated_file.write_text('{"messages": []}\n', encoding="utf-8")
    api = MagicMock()
    fake_client = SimpleNamespace(namespace="ghostpeony", api=api)
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.datasets.HF_HUB_AVAILABLE",
        True,
    )

    manager = HFDatasetManager(client=fake_client)
    url = manager.upload_training_data(
        local_path=tmp_path,
        repo_name="designer-run",
        train_file=generated_file,
    )

    train_upload = next(
        call
        for call in api.upload_file.call_args_list
        if call.kwargs["path_in_repo"] == "train.jsonl"
    )
    assert train_upload.kwargs["path_or_fileobj"] == str(generated_file)
    assert url == "https://huggingface.co/datasets/ghostpeony/designer-run"
