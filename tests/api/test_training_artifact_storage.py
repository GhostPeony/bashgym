"""Checkpoint browser storage inventory and deletion coverage."""

from pathlib import Path

from fastapi.testclient import TestClient

from bashgym.api.routes import create_app


def test_training_artifact_inventory_includes_default_output_and_deletes_from_its_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "data" / "models" / "run-storage-test"
    for name in ("final", "merged", "checkpoint-100", "gguf"):
        artifact_dir = run_dir / name
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "artifact.bin").write_bytes(b"artifact")

    client = TestClient(create_app())
    response = client.get("/api/training/checkpoints")

    assert response.status_code == 200
    rows = {row["kind"]: row for row in response.json() if row["run_id"] == "run-storage-test"}
    assert set(rows) == {"final", "merged", "intermediate", "gguf"}

    checkpoint_id = rows["merged"]["id"].replace("\\", "/")
    delete_response = client.delete(f"/api/training/checkpoints/{checkpoint_id}")
    assert delete_response.status_code == 200
    assert not (run_dir / "merged").exists()
