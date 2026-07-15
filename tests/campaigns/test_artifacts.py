"""Two-phase seal, deterministic fake executor, and tamper tests."""

import json

import pytest

from bashgym.campaigns.artifacts import SEAL_FILENAME, ArtifactSealer, ArtifactSealError
from bashgym.campaigns.executors import FakeExecutionRequest, FakeExecutor, fake_digest


def request() -> FakeExecutionRequest:
    return FakeExecutionRequest(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-1",
        attempt_id="attempt-1",
        manifest_revision=1,
        candidate_digest=fake_digest("candidate-1"),
        input_digest=fake_digest("input-1"),
        claim_generation=3,
        steps=8,
    )


def test_fake_executor_produces_verifiable_loss_artifacts(tmp_path):
    sealer = ArtifactSealer(b"a" * 32, key_version="test-key-v1")
    executor = FakeExecutor(tmp_path / "artifacts", sealer)
    sealed, manifest = executor.execute(request())

    verified = sealer.verify(
        sealed,
        expected_action_id="action-1",
        expected_input_digest=fake_digest("input-1"),
        expected_claim_generation=3,
    )
    metric_rows = [
        json.loads(line)
        for line in (sealed / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert verified == manifest
    assert len(metric_rows) == 8
    assert metric_rows[-1]["loss"] < metric_rows[0]["loss"]
    assert (sealed / SEAL_FILENAME).is_file()
    assert not list((tmp_path / "artifacts" / ".tmp").iterdir())


def test_sealed_file_tampering_and_wrong_fencing_identity_fail_closed(tmp_path):
    sealer = ArtifactSealer(b"b" * 32, key_version="test-key-v1")
    sealed, _manifest = FakeExecutor(tmp_path / "artifacts", sealer).execute(request())

    with pytest.raises(ArtifactSealError, match="claim generation mismatch"):
        sealer.verify(sealed, expected_claim_generation=2)

    (sealed / "summary.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactSealError, match="hash mismatch"):
        sealer.verify(sealed, expected_claim_generation=3)


def test_manifest_must_cover_exact_output_set_before_rename(tmp_path):
    sealer = ArtifactSealer(b"c" * 32, key_version="test-key-v1")
    executor = FakeExecutor(tmp_path / "artifacts", sealer)
    sealed, manifest = executor.execute(request())
    assert sealer.verify(sealed) == manifest

    envelope = json.loads((sealed / SEAL_FILENAME).read_text(encoding="utf-8"))
    envelope["manifest"]["claim_generation"] = 99
    (sealed / SEAL_FILENAME).write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ArtifactSealError, match="invalid seal envelope"):
        sealer.verify(sealed)
