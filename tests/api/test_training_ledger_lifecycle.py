"""No-compute API lifecycle proves canvas training events reach the durable ledger."""

from fastapi.testclient import TestClient

from bashgym.api.routes import create_app
from bashgym.ledger.persistence import ExperimentLedgerRepository


def test_simulated_training_populates_verified_ledger_run_and_loss_series(tmp_path):
    application = create_app()
    repository = ExperimentLedgerRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    application.state.experiment_ledger_repository = repository
    application.state.trainer = None
    http = TestClient(application)

    tracking = {
        "workspace_id": "workspace-a",
        "project_id": "project-a",
        "project_display_name": "Project A",
        "experiment_id": "experiment-a",
        "experiment_name": "Simulation lifecycle",
        "objective": "Verify ledger and canvas lifecycle wiring without compute.",
        "task_type": "terminal-agent",
        "model_id": "model-a",
        "model_version_id": "model-a-v1",
        "model_source_uri": "hf://example/model-a",
        "model_config_digest": "a" * 64,
        "dataset_id": "dataset-a",
        "dataset_version_id": "dataset-a-v1",
        "dataset_source_uri": "file://data/dataset-a.manifest.json",
        "dataset_content_digest": "b" * 64,
        "environment_id": "simulation-runtime-v1",
        "environment_runtime_digest": "c" * 64,
    }
    response = http.post(
        "/api/training/start",
        json={
            "strategy": "sft",
            "base_model": "example/model-a",
            "dataset_path": "data/train.jsonl",
            "num_epochs": 1,
            "tracking": tracking,
        },
    )

    assert response.status_code == 200
    queued = response.json()
    run_id = queued["run_id"]
    status = http.get(f"/api/training/{run_id}")
    details = repository.run_details("workspace-a", "project-a", run_id)
    points = repository.metric_series(
        "workspace-a", "project-a", metric_name="train.loss", run_id=run_id
    )

    assert queued["tracking_ids"]["context_status"] == "verified"
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["tracking_ids"]["run_id"] == run_id
    assert details["run"]["status"] == "completed"
    assert details["run"]["is_simulation"] is True
    assert details["attempts"][0]["status"] == "completed"
    assert len(points) == 100
