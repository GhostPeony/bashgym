"""Sealed campaign ingestion for optional NeMo Gym rollout evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.contracts import ActionAttempt, AttemptStatus, StageKind, canonical_hash
from bashgym.campaigns.executors import RemoteOutputSealer
from bashgym.campaigns.nemo_gym_evidence import (
    NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
    NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA,
    build_nemo_gym_campaign_evidence,
    load_nemo_gym_campaign_evidence,
    write_nemo_gym_campaign_evidence,
)
from bashgym.campaigns.remote import RemoteObservation, RemoteRunIdentity, RemoteRunState
from bashgym.environments.nemo_gym import export_star_count_nemo_gym_bundle
from bashgym.environments.star_count import (
    generate_star_count_dataset,
    star_count_environment_spec,
)

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def _attempt(*, attempt_id: str = "attempt-1") -> ActionAttempt:
    return ActionAttempt(
        attempt_id=attempt_id,
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        study_id="study-a",
        action_id="action-a",
        attempt_number=1,
        claim_generation=1,
        status=AttemptStatus.RUNNING,
        input_digest="1" * 64,
        candidate_digest="2" * 64,
        manifest_revision=3,
        stage=StageKind.FULL_TRAINING,
        created_at=NOW,
        updated_at=NOW,
    )


def _message(item_id: str, offset: int) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "answer"}],
        "prompt_token_ids": [1 + offset, 2 + offset],
        "generation_token_ids": [3 + offset, 4 + offset],
        "generation_log_probs": [-0.1, -0.2],
    }


def _rollout(session_id: str, example_index: int) -> dict:
    environment = star_count_environment_spec()
    return {
        "session_id": session_id,
        "example_index": example_index,
        "environment_id": environment.id,
        "environment_digest": canonical_hash(environment.to_dict()),
        "response": {"output": [_message(f"message-{example_index}", example_index)]},
        "reward_components": {"count_accuracy": 1.0, "format_accuracy": 1.0},
        "total_reward": 1.0,
        "refit": {
            "refit_id": "refit-4",
            "training_step": 4,
            "source_checkpoint_sha256": "3" * 64,
            "policy_revision": 4,
            "generation_revision": 4,
            "synchronized": True,
        },
    }


def _bundle(tmp_path: Path) -> dict:
    dataset = tmp_path / "dataset"
    generate_star_count_dataset(
        dataset,
        train_size=1,
        validation_size=1,
        heldout_size=1,
        seed=7,
    )
    return export_star_count_nemo_gym_bundle(
        dataset,
        tmp_path / "bundle",
        nemo_gym_revision="a" * 40,
        bashgym_revision="b" * 40,
        dataset_license="MIT",
    )


def test_campaign_evidence_is_deterministic_and_preserves_exact_identities(tmp_path: Path):
    attempt = _attempt()
    bundle = _bundle(tmp_path)
    environment = star_count_environment_spec()
    payloads = [_rollout("session-b", 1), _rollout("session-a", 0)]

    first = build_nemo_gym_campaign_evidence(
        attempt,
        bundle_manifest=bundle,
        environment=environment,
        rollout_payloads=payloads,
    )
    second = build_nemo_gym_campaign_evidence(
        attempt,
        bundle_manifest=bundle,
        environment=environment,
        rollout_payloads=payloads,
    )

    assert first == second
    assert [item.example_index for item in first.rollouts] == [0, 1]
    assert first.bundle.bundle_digest == bundle["bundle_digest"]
    assert first.token_source == "nemo_gym_model_server_message_fields"
    assert first.rollouts[0].message_tokens[0].generation_token_ids == (3, 4)
    assert first.rollouts[0].refit.source_checkpoint_sha256 == "3" * 64
    assert first.rollout_batch_digest
    assert first.token_evidence_digest
    assert first.refit_receipt_digest
    assert first.evidence_digest


def test_loader_rejects_tampering_and_cross_attempt_replay(tmp_path: Path):
    evidence = build_nemo_gym_campaign_evidence(
        _attempt(),
        bundle_manifest=_bundle(tmp_path),
        environment=star_count_environment_spec(),
        rollout_payloads=[_rollout("session-a", 0)],
    )
    path = write_nemo_gym_campaign_evidence(
        tmp_path / NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
        evidence,
    )
    assert load_nemo_gym_campaign_evidence(path, expected_attempt=_attempt()) == evidence

    with pytest.raises(ValueError, match="does not match the action attempt"):
        load_nemo_gym_campaign_evidence(
            path,
            expected_attempt=_attempt(attempt_id="attempt-2"),
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rollouts"][0]["reward_components"]["count_accuracy"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="campaign evidence is invalid"):
        load_nemo_gym_campaign_evidence(path, expected_attempt=_attempt())


def test_remote_sealer_validates_contract_before_registering_schema(tmp_path: Path):
    attempt = _attempt()
    evidence = build_nemo_gym_campaign_evidence(
        attempt,
        bundle_manifest=_bundle(tmp_path),
        environment=star_count_environment_spec(),
        rollout_payloads=[_rollout("session-a", 0)],
    )
    artifact_root = tmp_path / "artifacts"
    temporary = artifact_root / ".tmp" / "download"
    temporary.mkdir(parents=True)
    write_nemo_gym_campaign_evidence(
        temporary / NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
        evidence,
    )
    identity = RemoteRunIdentity(
        compute_profile_id="private-compute-a",
        run_id="run-a",
        remote_run_directory="/private/run-a",
        remote_pid=100,
        process_group_id=100,
        process_start_ticks=10,
        boot_id="boot-a",
        command_hash="4" * 64,
        launch_manifest_sha256="5" * 64,
        launched_at=NOW,
    )
    observation = RemoteObservation(
        identity=identity,
        state=RemoteRunState.COMPLETED,
        observed_at=NOW + timedelta(seconds=5),
        exit_code=0,
        safe_reason="completed",
    )
    sealer = RemoteOutputSealer(
        artifact_root,
        ArtifactSealer(b"s" * 32, key_version="nemo-gym-evidence-test-v1"),
    )

    sealed, manifest = sealer.seal_completed(
        attempt,
        identity,
        observation,
        temporary,
    )

    assert sealed.is_dir()
    assert manifest.outputs[0].schema_name == NEMO_GYM_CAMPAIGN_EVIDENCE_SCHEMA
