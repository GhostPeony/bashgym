"""Compact contracts for compute-resident AutoResearch datasets."""

import hashlib
from datetime import datetime

import pytest
from pydantic import ValidationError

from bashgym._compat import UTC
from bashgym.campaigns.autoresearch_dataset import (
    AUTORESEARCH_DATASET_FILE_SCHEMA,
    AUTORESEARCH_DATASET_RECEIPT_FILENAME,
    AUTORESEARCH_DATASET_RECEIPT_SCHEMA,
    AutoResearchDatasetFile,
    AutoResearchDatasetReceipt,
    build_dataset_ledger_specs,
)
from bashgym.campaigns.contracts import ActionAttempt, AttemptStatus, StageKind, canonical_hash
from bashgym.campaigns.executors import RemoteOutputSealer
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
    RemoteLaunchRequest,
    RemoteResidentDatasetFile,
    RemoteResidentDatasetSource,
    RemoteTrainingAdapter,
    remote_executor_config,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def receipt() -> AutoResearchDatasetReceipt:
    return AutoResearchDatasetReceipt(
        files=(
            AutoResearchDatasetFile(
                path="dataset/train.jsonl",
                sha256="a" * 64,
                size_bytes=120,
                split="train",
                row_count=8,
            ),
            AutoResearchDatasetFile(
                path="dataset/validation.jsonl",
                sha256="b" * 64,
                size_bytes=40,
                split="validation",
                row_count=2,
            ),
        ),
        generator={"kind": "nvidia_data_designer", "pipeline": "terminal_env_generation"},
    )


def test_receipt_derives_one_stable_content_digest_and_split_summary():
    value = receipt()

    assert value.content_digest
    assert value.row_counts == {"train": 8, "validation": 2}
    assert value.split_manifest == {
        "train": ["dataset/train.jsonl"],
        "validation": ["dataset/validation.jsonl"],
    }
    assert AUTORESEARCH_DATASET_FILE_SCHEMA == "autoresearch_dataset_file.v1"
    assert (
        RemoteOutputSealer._schema_for_relative(AUTORESEARCH_DATASET_RECEIPT_FILENAME)
        == AUTORESEARCH_DATASET_RECEIPT_SCHEMA
    )
    assert (
        RemoteOutputSealer._schema_for_relative("dataset/train.jsonl")
        == AUTORESEARCH_DATASET_FILE_SCHEMA
    )


def test_receipt_rejects_unsafe_duplicate_or_non_dataset_files():
    with pytest.raises(ValidationError, match="dataset/"):
        AutoResearchDatasetFile(
            path="../train.jsonl",
            sha256="a" * 64,
            size_bytes=1,
            split="train",
            row_count=1,
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        AutoResearchDatasetReceipt(files=(receipt().files[1], receipt().files[0]))


def test_dataset_ledger_projection_is_opaque_and_attempt_bound():
    data_attempt = ActionAttempt(
        attempt_id="attempt-data-build-1",
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        study_id="study-a",
        action_id="action-data-build-1",
        attempt_number=1,
        claim_generation=1,
        status=AttemptStatus.COMPLETED,
        input_digest="c" * 64,
        candidate_digest=canonical_hash("candidate-a"),
        manifest_revision=1,
        stage=StageKind.DATA_BUILD,
        sealed_result_uri="bashgym-remote-seal://compute/attempt-data-build-1/sha256/" + "d" * 64,
        created_at=NOW,
        updated_at=NOW,
    )
    dataset, version = build_dataset_ledger_specs(
        data_attempt,
        receipt(),
        project_id="project-a",
        task_type="terminal-agent-sft",
        created_at=NOW,
    )

    assert version.dataset_id == dataset.dataset_id
    assert version.content_digest == receipt().content_digest
    assert version.source_uri == f"autoresearch-remote-dataset://sha256/{receipt().content_digest}"
    assert version.metadata["producer_attempt_id"] == data_attempt.attempt_id
    assert "/home/operator" not in version.model_dump_json()


def test_training_launch_references_remote_dataset_without_uploading_rows(tmp_path):
    script = tmp_path / "train.py"
    config = tmp_path / "runner-config.json"
    script.write_text("print('train')\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    source = RemoteResidentDatasetSource(
        campaign_id="campaign-a",
        study_id="study-a",
        action_id="action-data-build-1",
        attempt_id="attempt-data-build-1",
        stage_index=0,
        compute_profile_id="research-compute",
        remote_dataset_path="/home/operator/research-runs/attempt-data-build-1/dataset",
        dataset_id="autoresearch-generated-dataset-a",
        dataset_version_id="autoresearch-generated-version-a",
        content_digest=receipt().content_digest,
        files=tuple(
            RemoteResidentDatasetFile(
                remote_relative_path=item.path.removeprefix("dataset/"),
                sha256=item.sha256,
                size_bytes=item.size_bytes,
            )
            for item in receipt().files
        ),
    )
    request = RemoteLaunchRequest(
        compute_profile_id="research-compute",
        run_id="attempt-training-1",
        script_path=script,
        input_files=(config,),
        script_args=("--dataset-dir", source.remote_dataset_path),
        recipe_digest="e" * 64,
        output_paths=("final",),
        remote_resident_dataset=source,
    )

    manifest = RemoteTrainingAdapter._launch_manifest(request, "/remote/attempt-training-1")

    assert manifest["remote_resident_dataset"]["dataset_version_id"] == (source.dataset_version_id)
    assert [item["name"] for item in manifest["files"]] == ["train.py", "runner-config.json"]
    assert all("train.jsonl" not in item["name"] for item in manifest["files"])


def test_registered_data_build_stage_uses_typed_recipe_arguments(tmp_path):
    script = tmp_path / "build_data.py"
    config = tmp_path / "data-designer-config.json"
    key = tmp_path / "campaign-key"
    script.write_text("print('build data')\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    key.write_text("fixture-key\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.DATA_BUILD,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(config,),
        input_sha256={config.name: hashlib.sha256(config.read_bytes()).hexdigest()},
        output_paths=("autoresearch_dataset_receipt.json", "dataset"),
        budget_reservation=0.05,
    )
    profile = ApprovedRemoteExecutorProfile(
        profile_id="data-designer-v1",
        profile_revision=1,
        compute_profile_id="research-compute",
        target_contract_key="terminal-agent-v1",
        target_model_digest="c" * 64,
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(stage,),
    )

    executor = remote_executor_config(
        profile,
        StageKind.DATA_BUILD,
        recipe_digest="d" * 64,
        recipe_script_args=("--pipeline", "terminal_env_generation", "--rows", "64"),
    )

    assert executor["stage"] == StageKind.DATA_BUILD.value
    assert executor["script_args"][-4:] == [
        "--pipeline",
        "terminal_env_generation",
        "--rows",
        "64",
    ]
