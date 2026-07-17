import json
from pathlib import Path

from bashgym.api.routes import create_app
from bashgym.api.runtime_observer import (
    classify_runtime_command,
    completed_runtime_jobs_from_manifests,
    inspect_runtime_artifacts,
    runtime_job_from_process_info,
)


def test_runtime_route_is_registered_in_desktop_app():
    app = create_app()
    assert "/api/runtime/jobs" in app.openapi()["paths"]


def test_classifies_workspace_data_designer_and_training_processes(tmp_path: Path):
    assert (
        classify_runtime_command(
            ["python", "scripts/generate_dd_pairs.py", "--num-seeds", "20"],
            str(tmp_path),
            tmp_path,
        )
        == "designer"
    )
    assert (
        classify_runtime_command(
            ["python", "scripts/train_sft.py", "--output-dir", "runs/one"],
            str(tmp_path),
            tmp_path,
        )
        == "training"
    )
    assert (
        classify_runtime_command(
            ["python", "-m", "uvicorn", "bashgym.api.routes:create_app"],
            str(tmp_path),
            tmp_path,
        )
        is None
    )


def test_infers_designer_progress_from_completed_batch_artifacts(tmp_path: Path):
    output_dir = tmp_path / "designer-output"
    output_dir.mkdir()
    for index in (1, 2):
        (output_dir / f"dd_seed_batch_{index:03d}.jsonl").write_text(
            '{"seed": 1}\n{"seed": 2}\n', encoding="utf-8"
        )

    progress, artifacts, resolved_output = inspect_runtime_artifacts(
        "designer",
        {"output_dir": str(output_dir), "num_seeds": "10"},
        str(tmp_path),
    )

    assert progress == {"current": 2, "total": 10, "unit": "seeds"}
    assert len(artifacts) == 2
    assert resolved_output == str(output_dir.resolve())


def test_builds_stable_runtime_job_payload(tmp_path: Path):
    log_dir = tmp_path / ".tmp"
    log_dir.mkdir()
    (log_dir / "dd-pairs-run.log").write_text(
        "model: hermes-qwen3.6-27b-dense\nmodel provider: 'local'\n",
        encoding="utf-8",
    )
    job = runtime_job_from_process_info(
        {
            "pid": 42,
            "create_time": 123.5,
            "cwd": str(tmp_path),
            "cmdline": [
                "python",
                "scripts/generate_dd_pairs.py",
                "--output-dir",
                "out",
                "--num-seeds",
                "8",
                "--corpus-jsonl",
                "corpus.jsonl",
                "--llm-endpoint",
                "http://10.200.0.20:8000/v1",
            ],
        },
        tmp_path,
    )

    assert job is not None
    assert job["job_id"] == "runtime_designer_42_123500"
    assert job["kind"] == "designer"
    assert job["source"] == "process_observer"
    assert job["dataset"] == "corpus.jsonl"
    assert job["model"] == "hermes-qwen3.6-27b-dense"
    assert job["execution"] == "private"


def test_recovers_final_designer_progress_from_completed_manifest(tmp_path: Path):
    output_dir = tmp_path / ".tmp" / "dd-train-pairs" / "real_chunks"
    output_dir.mkdir(parents=True)
    (output_dir / "train_queries.jsonl").write_text('{"query": "one"}\n', encoding="utf-8")
    (output_dir / "dd_train_pairs_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "arm": "real_chunks",
                "run_kind": "generate_dd_train_pairs",
                "created_at": "2026-07-09T20:00:00-07:00",
                "completed_at": "2026-07-09T22:54:43-07:00",
                "seed_rows": 200,
                "raw_rows": 200,
                "corpus_jsonl": "data/corpus.jsonl",
                "data_designer": {
                    "model": "hermes-qwen3.6-27b-dense",
                    "endpoint": "http://10.200.0.20:8000/v1",
                },
                "outputs": {
                    "train_queries_jsonl": str(output_dir / "train_queries.jsonl"),
                },
            }
        ),
        encoding="utf-8",
    )

    jobs = completed_runtime_jobs_from_manifests(tmp_path)

    assert len(jobs) == 1
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["progress"] == {"current": 200, "total": 200, "unit": "seeds"}
    assert jobs[0]["model"] == "hermes-qwen3.6-27b-dense"
    assert jobs[0]["execution"] == "private"
    assert jobs[0]["output_dir"] == str(output_dir.resolve())


def test_ignores_smoke_manifest_siblings_and_malformed_totals(tmp_path: Path):
    smoke_dir = tmp_path / ".tmp" / "dd-train-pairs" / "real_chunks_smoke4"
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "dd_train_pairs_manifest.json").write_text(
        json.dumps({"status": "completed", "arm": "real_chunks", "seed_rows": 4}),
        encoding="utf-8",
    )
    malformed_dir = tmp_path / ".tmp" / "dd-train-pairs" / "fake_transcripts"
    malformed_dir.mkdir(parents=True)
    (malformed_dir / "dd_train_pairs_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "arm": "fake_transcripts",
                "seed_rows": {"not": "a count"},
            }
        ),
        encoding="utf-8",
    )

    assert completed_runtime_jobs_from_manifests(tmp_path) == []
