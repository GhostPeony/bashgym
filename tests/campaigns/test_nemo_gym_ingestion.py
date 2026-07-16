"""Strict conversion of actual NeMo Gym trajectory and refit outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bashgym.campaigns.contracts import (
    ActionAttempt,
    AttemptStatus,
    StageKind,
    canonical_hash,
)
from bashgym.campaigns.nemo_gym_evidence import (
    NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
    load_nemo_gym_campaign_evidence,
)
from bashgym.campaigns.nemo_gym_ingestion import (
    build_nemo_gym_evidence_from_outputs,
    convert_nemo_gym_outputs,
    load_nemo_gym_trajectory_records,
)
from bashgym.environments.nemo_gym import export_star_count_nemo_gym_bundle
from bashgym.environments.star_count import (
    generate_star_count_dataset,
    star_count_environment_spec,
)

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def _attempt() -> ActionAttempt:
    return ActionAttempt(
        attempt_id="attempt-1",
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        study_id="study-a",
        action_id="action-a",
        attempt_number=1,
        claim_generation=2,
        status=AttemptStatus.RUNNING,
        input_digest="1" * 64,
        candidate_digest="2" * 64,
        manifest_revision=3,
        stage=StageKind.FULL_TRAINING,
        created_at=NOW,
        updated_at=NOW,
    )


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


def _refit(*, revision: int = 4) -> dict:
    return {
        "refit_id": f"refit-{revision}",
        "training_step": revision,
        "source_checkpoint_sha256": "3" * 64,
        "policy_revision": revision,
        "generation_revision": revision,
        "synchronized": True,
    }


def _rollout(*, session_id: str, example_index: int, refit: dict | None = None) -> dict:
    environment = star_count_environment_spec()
    payload = {
        "session_id": session_id,
        "example_index": example_index,
        "environment_id": environment.id,
        "environment_digest": canonical_hash(environment.to_dict()),
        "response": {
            "output": [
                {
                    "id": f"message-{example_index}",
                    "type": "message",
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ]
        },
        "reward_components": {"count_accuracy": 1.0, "format_accuracy": 1.0},
        "total_reward": 1.0,
    }
    if refit is not None:
        payload["refit"] = refit
    return payload


def _write_refit(tmp_path: Path, receipt: dict | None = None) -> Path:
    path = tmp_path / "nemo_gym_refit_receipt.json"
    path.write_text(json.dumps(receipt or _refit()), encoding="utf-8")
    return path


def test_converts_nested_actual_jsonl_and_writes_canonical_evidence(tmp_path: Path):
    trajectories = tmp_path / "nemo_gym_trajectories.jsonl"
    records = [
        {
            "full_result": json.dumps(
                _rollout(session_id="session-b", example_index=1, refit=_refit())
            )
        },
        {
            "trajectory": _rollout(session_id="session-a", example_index=0),
            "refit_receipt": _refit(),
        },
    ]
    trajectories.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    path = convert_nemo_gym_outputs(
        _attempt(),
        bundle_manifest=_bundle(tmp_path),
        environment_contract=star_count_environment_spec().to_dict(),
        trajectories=trajectories,
        refit_receipt_path=_write_refit(tmp_path),
        output_directory=tmp_path / "output",
    )

    assert path.name == NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME
    evidence = load_nemo_gym_campaign_evidence(path, expected_attempt=_attempt())
    assert [rollout.example_index for rollout in evidence.rollouts] == [0, 1]
    assert {rollout.refit.refit_id for rollout in evidence.rollouts} == {"refit-4"}


def test_requires_each_rollout_to_carry_the_exact_separate_refit_receipt(tmp_path: Path):
    missing = _rollout(session_id="session-a", example_index=0)
    with pytest.raises(ValueError, match="requires one explicit exact refit receipt"):
        build_nemo_gym_evidence_from_outputs(
            _attempt(),
            bundle_manifest=_bundle(tmp_path),
            environment_contract=star_count_environment_spec(),
            trajectories=[missing],
            refit_receipt_path=_write_refit(tmp_path),
        )


def test_rejects_rollout_bound_to_a_different_refit(tmp_path: Path):
    with pytest.raises(ValueError, match="not bound to the exact refit receipt"):
        build_nemo_gym_evidence_from_outputs(
            _attempt(),
            bundle_manifest=_bundle(tmp_path),
            environment_contract=star_count_environment_spec(),
            trajectories=[
                _rollout(session_id="session-a", example_index=0, refit=_refit(revision=5))
            ],
            refit_receipt_path=_write_refit(tmp_path),
        )


def test_rejects_unsynchronized_refit_without_using_process_status(tmp_path: Path):
    receipt = {**_refit(), "synchronized": False}
    with pytest.raises(ValueError, match="invalid or not synchronized"):
        build_nemo_gym_evidence_from_outputs(
            _attempt(),
            bundle_manifest=_bundle(tmp_path),
            environment_contract=star_count_environment_spec(),
            trajectories=[_rollout(session_id="session-a", example_index=0, refit=receipt)],
            refit_receipt_path=_write_refit(tmp_path, receipt),
        )


def test_loader_rejects_ambiguous_or_non_object_trajectory_records(tmp_path: Path):
    ambiguous = _rollout(session_id="session-a", example_index=0, refit=_refit())
    ambiguous["trajectory"] = dict(ambiguous)
    with pytest.raises(ValueError, match="ambiguous rollout payloads"):
        build_nemo_gym_evidence_from_outputs(
            _attempt(),
            bundle_manifest=_bundle(tmp_path),
            environment_contract=star_count_environment_spec(),
            trajectories=[ambiguous],
            refit_receipt_path=_write_refit(tmp_path),
        )

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_nemo_gym_trajectory_records(invalid)


def test_loader_rejects_oversized_in_memory_records():
    with pytest.raises(ValueError, match="exceeds the size limit"):
        load_nemo_gym_trajectory_records([{"payload": "x" * (4 * 1024 * 1024)}])
