"""Direct training lifecycle ingestion into the experiment ledger."""

from bashgym.ledger.contracts import RunStatus
from bashgym.ledger.persistence import ExperimentLedgerRepository
from bashgym.ledger.training import (
    finalize_training_run,
    mark_training_running,
    prepare_training_run,
    record_training_progress,
)


def tracking_context() -> dict:
    return {
        "workspace_id": "workspace-a",
        "project_id": "project-a",
        "project_display_name": "Retrieval project",
        "project_description": "An isolated retrieval experiment namespace.",
        "experiment_id": "experiment-a",
        "experiment_name": "Longer contrastive run",
        "objective": "Improve held-out retrieval quality without latency regression.",
        "task_type": "retrieval",
        "model_id": "embedding-model",
        "model_version_id": "embedding-model-base-v1",
        "model_source_uri": "hf://example/embedding-model",
        "model_source_revision": "revision-abc",
        "model_config_digest": "a" * 64,
        "dataset_id": "retrieval-pairs",
        "dataset_version_id": "retrieval-pairs-v2",
        "dataset_source_uri": "file://data/retrieval-pairs-v2.manifest.json",
        "dataset_content_digest": "b" * 64,
        "dataset_split_manifest": {"train": "train.jsonl", "dev": "dev.jsonl"},
        "dataset_row_counts": {"train": 800, "dev": 80},
        "environment_id": "local-gpu-runtime-v1",
        "environment_runtime_digest": "c" * 64,
        "environment_hardware": {"accelerator_family": "blackwell"},
        "owner_actor_id": "hermes",
        "tags": ["embedding", "retrieval"],
    }


def repository(tmp_path) -> ExperimentLedgerRepository:
    value = ExperimentLedgerRepository(tmp_path / "campaigns.sqlite3")
    value.initialize()
    return value


def test_verified_training_lifecycle_records_ids_metrics_and_terminal_event(tmp_path):
    ledger = repository(tmp_path)
    handle = prepare_training_run(
        ledger,
        run_id="run-official-1",
        request_payload={
            "strategy": "sft",
            "base_model": "example/embedding-model",
            "dataset_path": "data/train.jsonl",
            "compute_target": "local-gpu",
            "correlation_id": "training-session-1",
            "num_epochs": 3,
            "tracking": tracking_context(),
        },
        is_simulation=False,
    )
    mark_training_running(ledger, handle)

    first_written = record_training_progress(
        ledger,
        handle,
        {
            "step": 1,
            "total_steps": 100,
            "epoch": 1,
            "loss": 1.5,
            "learning_rate": 2e-5,
            "gpu_memory_gb": 12.25,
        },
    )
    replay_written = record_training_progress(
        ledger,
        handle,
        {
            "step": 1,
            "total_steps": 100,
            "epoch": 1,
            "loss": 1.5,
            "learning_rate": 2e-5,
            "gpu_memory_gb": 12.25,
        },
    )
    finalize_training_run(
        ledger,
        handle,
        status=RunStatus.COMPLETED,
        metrics={"final_loss": 0.8, "simulation": False},
    )

    details = ledger.run_details("workspace-a", "project-a", "run-official-1")
    points = ledger.metric_series(
        "workspace-a", "project-a", metric_name="train.loss", run_id="run-official-1"
    )
    events = ledger.list_events("workspace-a", "project-a")

    assert handle.verified is True
    assert first_written == 3
    assert replay_written == 0
    assert details["run"]["status"] == "completed"
    assert details["run"]["model_version_id"] == "embedding-model-base-v1"
    assert details["run"]["dataset_version_id"] == "retrieval-pairs-v2"
    assert details["run"]["environment_id"] == "local-gpu-runtime-v1"
    assert details["attempts"][0]["status"] == "completed"
    assert points[0]["metric_value"] == 1.5
    assert [event["event_type"] for event in events] == [
        "training-queued",
        "training-started",
        "training-completed",
    ]


def test_legacy_training_is_retained_in_explicit_unassigned_context(tmp_path):
    ledger = repository(tmp_path)
    handle = prepare_training_run(
        ledger,
        run_id="run-legacy-1",
        request_payload={
            "strategy": "dpo",
            "base_model": "example/model",
            "compute_target": "local",
        },
        is_simulation=True,
    )

    run = ledger.get_run("desktop-local", "unassigned", "run-legacy-1")
    projects = ledger.list_projects("desktop-local")

    assert handle.verified is False
    assert run["context_status"] == "unassigned"
    assert run["model_version_id"] is None
    assert run["is_simulation"] is True
    assert projects[0]["project_id"] == "unassigned"
