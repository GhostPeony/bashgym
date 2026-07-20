"""Hash-reconciled campaign export and smoke/finding separation tests."""

from __future__ import annotations

import json

import pytest

from bashgym.campaigns.export import (
    CampaignExportError,
    CampaignExportSnapshot,
    export_campaign_evidence,
)


def snapshot(*, full: bool = False) -> CampaignExportSnapshot:
    stage = "full_training" if full else "smoke_training"
    return CampaignExportSnapshot(
        campaign={
            "campaign_id": "campaign-1",
            "objective": "Improve held-out retrieval",
            "status": "active",
            "champion_ref": "base-qwen",
        },
        attempts=(
            {
                "attempt_id": "attempt-1",
                "study_id": "study-1",
                "stage": stage,
                "status": "completed",
                "candidate_digest": "a" * 64,
                "created_at": "2026-07-13T00:00:00Z",
                "updated_at": "2026-07-13T00:01:00Z",
            },
        ),
        artifacts=(
            {
                "artifact_id": "artifact-1",
                "schema_name": "training_metrics_jsonl.v1",
                "sha256": "b" * 64,
                "size_bytes": 42,
                "sealed": True,
                "valid": True,
                "created_at": "2026-07-13T00:01:00Z",
            },
        ),
        comparisons=(
            {
                "comparison_digest": "c" * 64,
                "champion_digest": "d" * 64,
                "candidate_digest": "a" * 64,
                "sample_count": 300,
                "verdict": "pass",
                "blocking_reasons": [],
                "warnings": [],
                "created_at": "2026-07-13T00:02:00Z",
            },
        ),
        loss_by_attempt={
            "attempt-1": (
                {"step": 1, "source": "training_metrics.jsonl", "value": 1.0},
                {"step": 2, "source": "training_metrics.jsonl", "value": 0.5},
            )
        },
        flags=("protected evaluation not run",),
    )


def test_smoke_export_is_hash_reconciled_but_never_claims_quality_findings(tmp_path):
    manifest = export_campaign_evidence(snapshot(), tmp_path / "export")

    assert manifest["quality_findings_available"] is False
    report = (tmp_path / "export" / "campaign_report.md").read_text(encoding="utf-8")
    assert "No model-quality findings are claimed" in report
    assert "Smoke attempts: 1 (runtime/semantics/memory evidence only)" in report
    assert "protected evaluation not run" in report
    assert {item["name"] for item in manifest["files"]} == {
        "artifacts.csv",
        "attempts.csv",
        "campaign_evidence.json",
        "campaign_report.docx",
        "campaign_report.md",
        "campaign_report.pdf",
        "comparisons.csv",
        "training_loss.png",
        "training_loss.svg",
    }
    assert "Dashed = smoke engineering evidence" in (
        tmp_path / "export" / "training_loss.svg"
    ).read_text(encoding="utf-8")


def test_full_run_plus_comparison_enables_quality_findings_and_is_deterministic(tmp_path):
    first = export_campaign_evidence(snapshot(full=True), tmp_path / "first")
    second = export_campaign_evidence(snapshot(full=True), tmp_path / "second")

    assert first == second
    assert first["quality_findings_available"] is True
    assert (tmp_path / "first" / "campaign_report.md").read_bytes() == (
        tmp_path / "second" / "campaign_report.md"
    ).read_bytes()
    assert (tmp_path / "first" / "campaign_report.docx").read_bytes() == (
        tmp_path / "second" / "campaign_report.docx"
    ).read_bytes()
    assert (tmp_path / "first" / "campaign_report.pdf").read_bytes() == (
        tmp_path / "second" / "campaign_report.pdf"
    ).read_bytes()
    assert (tmp_path / "first" / "training_loss.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    persisted = json.loads((tmp_path / "first" / "export_manifest.json").read_text())
    assert persisted == first


def test_export_rejects_local_path_bearing_projections(tmp_path):
    unsafe = CampaignExportSnapshot(
        campaign={"campaign_id": "campaign-1"},
        attempts=({"attempt_id": "attempt-1", "sealed_result_uri": "C:/private/run"},),
    )
    with pytest.raises(CampaignExportError, match="campaign_export_contains_local_path"):
        export_campaign_evidence(unsafe, tmp_path / "export")
