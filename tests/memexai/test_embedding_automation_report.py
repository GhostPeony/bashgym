from __future__ import annotations

import hashlib
import zipfile

import pytest

from scripts.memexai.build_embedding_automation_implementation_report import (
    milestone_rows,
    normalize_docx_zip,
    strict_dev_metrics,
    summarize_training_metrics,
    verify_export_manifest,
)


def test_training_summary_rejects_a_smoke_as_quality_evidence() -> None:
    manifest = {
        "training": {
            "optimizer_steps": 1,
            "batches_per_epoch": 1,
        }
    }
    rows = [
        {
            "step": 1,
            "epoch": 1.0,
            "loss": 1.25,
            "learning_rate": 0.0,
            "grad_norm": 4.0,
        }
    ]
    with pytest.raises(ValueError, match="full run, not a smoke"):
        summarize_training_metrics(manifest, rows)


def test_training_summary_keeps_epoch_progress_separate_from_epoch_count() -> None:
    manifest = {
        "training": {
            "optimizer_steps": 2,
            "batches_per_epoch": 1,
            "epochs": 2,
        }
    }
    rows = [
        {"step": 1, "epoch": 1.0, "loss": 0.8, "learning_rate": 1e-6, "grad_norm": 4.0},
        {"step": 2, "epoch": 2.0, "loss": 0.4, "learning_rate": 0.0, "grad_norm": 2.0},
    ]
    summary = summarize_training_metrics(manifest, rows)
    assert summary["epoch_progress"] == [1.0, 2.0]
    assert "epochs" not in summary


def test_strict_dev_metrics_requires_physical_dev_only_lineage() -> None:
    manifest = {
        "rows": 18,
        "splits": ["dev"],
        "queries_jsonl": "/inputs/heldout-dev.jsonl",
        "model_footprint_bytes": 100,
        "runs": {
            "memexai_youtube": {
                "median_query_latency_ms": 2.0,
                "metrics": {"overall": {"count": 18}},
            }
        },
    }
    assert strict_dev_metrics(manifest)["count"] == 18
    with pytest.raises(ValueError, match="18-row dev-only"):
        strict_dev_metrics({**manifest, "splits": ["test"]})
    with pytest.raises(ValueError, match="heldout-dev.jsonl"):
        strict_dev_metrics({**manifest, "queries_jsonl": "/inputs/combined.jsonl"})


def test_export_verifier_checks_declared_hash_and_size(tmp_path) -> None:
    export = tmp_path / "campaign_report.md"
    export.write_text("proof\n", encoding="utf-8")
    digest = hashlib.sha256(export.read_bytes()).hexdigest()
    manifest_path = tmp_path / "export_manifest.json"
    manifest = {
        "files": [
            {
                "name": export.name,
                "sha256": digest,
                "size_bytes": export.stat().st_size,
            }
        ]
    }
    assert verify_export_manifest(manifest_path, manifest)[0]["sha256"] == digest
    export.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verify_export_manifest(manifest_path, manifest)


def test_milestone_matrix_does_not_overclaim_missing_system_evidence() -> None:
    facts = {
        "training": {"optimizer_steps": 84},
        "campaign": {
            "attempt_id": "attempt-1",
            "remote_adoption_event_count": 7,
            "claim_generation": 8,
            "sealed_artifact_count": 19,
            "budget": {"actual": 0.02},
        },
        "campaign_export": {"verified_files": [{}] * 9},
    }
    rows = {row["milestone"]: row for row in milestone_rows(facts)}
    assert rows["Candidate B full cached-MNRL training"]["status"] == "complete"
    assert rows["BM25/dense RRF and independent reranker benchmark"]["status"] == "not evidenced"
    assert rows["Same-campaign API, CLI/MCP, and live canvas proof"]["status"] == "not evidenced"


def test_docx_zip_normalization_is_byte_stable(tmp_path) -> None:
    paths = [tmp_path / "one.docx", tmp_path / "two.docx"]
    for index, path in enumerate(paths):
        with zipfile.ZipFile(path, "w") as archive:
            info = zipfile.ZipInfo("word/document.xml", (2020 + index, 1, 1, 0, 0, 0))
            archive.writestr(info, b"<document/>")
            archive.writestr("[Content_Types].xml", b"<types/>")
        normalize_docx_zip(path)
    assert paths[0].read_bytes() == paths[1].read_bytes()
