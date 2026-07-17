"""Historical import provenance and protected-boundary tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer, ArtifactSealError
from bashgym.campaigns.history_import import (
    HistoricalImportAttestor,
    HistoricalImportError,
    HistoricalSource,
)

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def test_historical_attestation_is_path_free_read_only_and_not_a_live_seal(tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    metrics = approved / "candidate-a-training-metrics.jsonl"
    metrics.write_text('{"step":1,"loss":0.9}\n', encoding="utf-8")
    attestor = HistoricalImportAttestor(
        b"h" * 32,
        key_version="history-v1",
        allowed_roots=(approved,),
    )

    envelope = attestor.attest(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="candidate-a-history",
        sources=(
            HistoricalSource(
                logical_name="candidate-a-training-metrics",
                path=metrics,
                schema_name="training_metrics_jsonl.v1",
                provenance={"origin": "completed_2026_07_12_run"},
            ),
        ),
        imported_at=NOW,
        import_reason="Preserve the rejected Candidate A evidence as read-only history.",
    )
    persisted = attestor.write(envelope, tmp_path / "attestation.json")
    manifest = attestor.verify(envelope)

    assert manifest["read_only"] is True
    assert manifest["live_action_result"] is False
    assert "approved" not in persisted.read_text(encoding="utf-8")
    assert str(tmp_path) not in persisted.read_text(encoding="utf-8")
    with pytest.raises(ArtifactSealError):
        ArtifactSealer(b"h" * 32, key_version="history-v1").verify(tmp_path)


def test_protected_source_is_rejected_before_hashing(monkeypatch, tmp_path):
    approved = tmp_path / "approved"
    protected = approved / "protected"
    protected.mkdir(parents=True)
    frozen = protected / "heldout-test.jsonl"
    frozen.write_text("must-not-open", encoding="utf-8")
    attestor = HistoricalImportAttestor(
        b"h" * 32,
        key_version="history-v1",
        allowed_roots=(approved,),
        protected_roots=(protected,),
    )
    monkeypatch.setattr(
        "bashgym.campaigns.history_import._hash_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("protected file was opened")),
    )

    with pytest.raises(HistoricalImportError, match="protected source excluded"):
        attestor.attest(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="candidate-a-history",
            sources=(
                HistoricalSource(
                    logical_name="forbidden-protected-set",
                    path=frozen,
                    schema_name="protected_query_set.v1",
                    provenance={},
                ),
            ),
            imported_at=NOW,
            import_reason="forbidden",
        )


def test_tampering_invalidates_historical_attestation(tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    source = approved / "manifest.json"
    source.write_text("{}", encoding="utf-8")
    attestor = HistoricalImportAttestor(
        b"h" * 32,
        key_version="history-v1",
        allowed_roots=(approved,),
    )
    envelope = attestor.attest(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="candidate-a-history",
        sources=(
            HistoricalSource(
                logical_name="training-manifest",
                path=source,
                schema_name="training_manifest.v1",
                provenance={},
            ),
        ),
        imported_at=NOW,
        import_reason="history",
    )
    envelope["manifest"]["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(HistoricalImportError, match="invalid attestation"):
        attestor.verify(envelope)


def test_historical_source_symlink_is_rejected(tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    source = approved / "metrics.jsonl"
    source.write_text('{"loss":0.5}\n', encoding="utf-8")
    link = approved / "linked-metrics.jsonl"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation is unavailable for this Windows account")
    attestor = HistoricalImportAttestor(
        b"h" * 32,
        key_version="history-v1",
        allowed_roots=(approved,),
    )

    with pytest.raises(HistoricalImportError, match="regular file"):
        attestor.attest(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="candidate-a-history",
            sources=(
                HistoricalSource(
                    logical_name="linked-metrics",
                    path=link,
                    schema_name="training_metrics_jsonl.v1",
                    provenance={},
                ),
            ),
            imported_at=NOW,
            import_reason="history",
        )
