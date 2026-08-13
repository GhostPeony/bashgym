"""Typed SSH launch, reconciliation, output, and signal tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns import remote as remote_contracts
from bashgym.campaigns.contracts import StageKind, utc_now
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
    CodeLineageLaunchSnapshot,
    PinnedRemoteStageProfile,
    RegisteredRemoteModelSource,
    RemoteCapacityPolicy,
    RemoteCommandResult,
    RemoteLaunchRequest,
    RemoteObservation,
    RemoteResidentDatasetFile,
    RemoteResidentDatasetSource,
    RemoteRunIdentity,
    RemoteRunState,
    RemoteStreamCursor,
    RemoteTrainingAdapter,
    SealedStageArtifactInput,
    SealedStageArtifactSource,
    remote_executor_config,
)
from bashgym.gym.remote_trainer import SSHConfig
from bashgym.ledger.contracts import DatasetVersionSpec, EvaluationSuiteSpec


class MockSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []
        self.uploads = []
        self.byte_uploads = []
        self.downloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def run(self, command, *, timeout=None):
        self.commands.append((command, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected remote command: {command}")
        return self.responses.pop(0)

    async def upload(self, local_path, remote_path):
        self.uploads.append((Path(local_path), remote_path))

    async def upload_bytes(self, payload, remote_path):
        self.byte_uploads.append((bytes(payload), remote_path))

    async def download(self, remote_path, local_path):
        local = Path(local_path)
        self.downloads.append((remote_path, local))
        if remote_path.endswith("/final"):
            local.mkdir(parents=True, exist_ok=True)
            (local / "config.json").write_text("{}", encoding="utf-8")
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text("fixture", encoding="utf-8")
        return True


def result(stdout="", *, status=0, stderr=""):
    return RemoteCommandResult(stdout=stdout, stderr=stderr, exit_status=status)


def config():
    return SSHConfig(
        host="192.0.2.10",
        username="trainer",
        key_path="~/.ssh/id_ed25519",
        remote_work_dir="~/bashgym-training",
    )


def registered_evaluation_dataset_source(*, compute_profile_id="ssh-gpu-lab"):
    source_type = getattr(remote_contracts, "RegisteredRemoteEvaluationDatasetSource", None)
    assert source_type is not None, "registered remote evaluation dataset contract is required"
    return source_type(
        source_id="terminal-heldout-v1",
        compute_profile_id=compute_profile_id,
        dataset_version_id="terminal-heldout-v1",
        content_digest="b" * 64,
        remote_dataset_path="/srv/bashgym/datasets/terminal-heldout-v1.jsonl",
    )


def remote_model_registration_request(*, operation="register", target_auth_env=None, **overrides):
    request_type = getattr(remote_contracts, "RemoteModelRegistrationRequest", None)
    assert request_type is not None, "typed remote model registration request is required"
    payload = dict(
        operation=operation,
        source_id="research-model-base",
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="research-model-v1",
        target_model_digest="d" * 64,
        model_id="example/research-model",
        revision="a" * 40,
        remote_model_path="/srv/bashgym/models/research-model",
        target_auth_env=target_auth_env,
    )
    payload.update(overrides)
    return request_type(**payload)


def remote_model_artifact_payload(**overrides):
    payload = {
        "schema_version": "campaign_remote_model_artifact_receipt.v1",
        "model_id": "example/research-model",
        "revision": "a" * 40,
        "artifact_manifest_sha256": "b" * 64,
        "weight_file_count": 16,
        "total_size_bytes": 61_000_000_000,
    }
    payload.update(overrides)
    return payload


def registered_v2_model_source(*, logical_digest="d" * 64):
    return RegisteredRemoteModelSource(
        schema_version="campaign_registered_remote_model_source.v2",
        source_id="research-model-base",
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="research-model-v1",
        model_digest=logical_digest,
        remote_model_path="/srv/bashgym/models/research-model",
        artifact_receipt=remote_model_artifact_payload(),
    )


@pytest.mark.asyncio
async def test_register_remote_model_returns_physical_receipt_without_copying_weights():
    request = remote_model_registration_request()
    session = MockSession([result(json.dumps(remote_model_artifact_payload()))])
    adapter = RemoteTrainingAdapter(
        config(),
        compute_profile_id="ssh-gpu-lab",
        session_factory=lambda: session,
    )

    source = await adapter.register_remote_model(request)

    assert source.schema_version == "campaign_registered_remote_model_source.v2"
    assert source.model_digest == "d" * 64
    assert source.artifact_receipt.model_id == request.model_id
    assert source.artifact_receipt.revision == request.revision
    assert source.artifact_receipt.artifact_manifest_sha256 == "b" * 64
    assert source.artifact_receipt.weight_file_count == 16
    assert source.artifact_receipt.total_size_bytes == 61_000_000_000
    assert source.physical_model_digest == "b" * 64
    command, timeout = session.commands[0]
    assert request.remote_model_path in command
    assert "sha256" in command
    assert "snapshot_download" not in command
    assert timeout == request.timeout_seconds
    assert session.uploads == []
    assert session.downloads == []


@pytest.mark.asyncio
async def test_acquire_remote_model_downloads_pinned_revision_on_target_then_renames_atomically(
    monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN", "hf_controller_secret_must_not_escape")
    request = remote_model_registration_request(
        operation="acquire",
        target_auth_env="HF_TOKEN",
    )
    session = MockSession([result(json.dumps(remote_model_artifact_payload()))])
    adapter = RemoteTrainingAdapter(
        config(),
        compute_profile_id="ssh-gpu-lab",
        session_factory=lambda: session,
    )

    source = await adapter.register_remote_model(request)

    command, timeout = session.commands[0]
    assert "snapshot_download" in command
    assert request.model_id in command
    assert request.revision in command
    assert ".partial-" in command
    assert "os.rename" in command
    assert request.target_auth_env in command
    assert "hf_controller_secret_must_not_escape" not in command
    assert "target_auth_env" not in source.model_dump_json()
    assert "hf_controller_secret_must_not_escape" not in source.model_dump_json()
    assert timeout == request.timeout_seconds
    assert session.uploads == []
    assert session.downloads == []


@pytest.mark.asyncio
async def test_remote_model_acquisition_uses_the_declared_bounded_timeout():
    request = remote_model_registration_request(operation="acquire", timeout_seconds=7_200)
    session = MockSession([result(json.dumps(remote_model_artifact_payload()))])
    adapter = RemoteTrainingAdapter(
        config(),
        compute_profile_id="ssh-gpu-lab",
        session_factory=lambda: session,
    )

    await adapter.register_remote_model(request)

    assert session.commands[0][1] == 7_200


def test_acquire_script_resumes_only_its_owned_partial_after_a_hard_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = (tmp_path / "models").resolve()
    parent.mkdir()
    destination = parent / "research-model"
    request_digest = "f" * 64
    suffix = request_digest[:16]
    partial = parent / f".{destination.name}.partial-{suffix}"
    partial.mkdir()
    owner = {
        "schema_version": "bashgym_remote_model_acquisition.v1",
        "request_digest": request_digest,
        "model_id": "example/research-model",
        "revision": "a" * 40,
    }
    (partial / ".bashgym-acquisition.json").write_text(
        json.dumps(owner, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (partial / ".cache" / "huggingface").mkdir(parents=True)
    (partial / ".cache" / "huggingface" / "interrupted").write_text("partial", encoding="utf-8")
    calls: list[Path] = []
    hub = types.ModuleType("huggingface_hub")

    def resume_snapshot(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        calls.append(local_dir)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "model.safetensors").write_bytes(b"weights")

    hub.snapshot_download = resume_snapshot
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    arguments = [
        "acquire-model",
        str(destination),
        owner["model_id"],
        owner["revision"],
        "",
        request_digest,
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    exec(remote_contracts._REMOTE_MODEL_ACQUIRE_SCRIPT, {"__name__": "__main__"})

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_id"] == owner["model_id"]
    assert calls == [partial]
    assert destination.is_dir()
    assert not partial.exists()

    def unexpected_download(**_kwargs):
        raise AssertionError("completed owned acquisition must replay without downloading")

    hub.snapshot_download = unexpected_download
    monkeypatch.setattr(sys, "argv", arguments)
    exec(remote_contracts._REMOTE_MODEL_ACQUIRE_SCRIPT, {"__name__": "__main__"})
    assert json.loads(capsys.readouterr().out) == receipt


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"revision": "main"}, "immutable"),
        ({"target_auth_env": "hf_raw_token_value"}, "environment variable"),
        ({"operation": "register", "target_auth_env": "HF_TOKEN"}, "only valid for acquire"),
    ),
)
def test_remote_model_registration_rejects_moving_revision_or_raw_auth(overrides, message):
    request_type = getattr(remote_contracts, "RemoteModelRegistrationRequest", None)
    assert request_type is not None, "typed remote model registration request is required"
    payload = remote_model_registration_request().model_dump(mode="python")
    payload.update(overrides)

    with pytest.raises(ValidationError, match=message):
        request_type.model_validate(payload)


@pytest.mark.asyncio
async def test_remote_model_registration_rejects_mismatched_target_metadata_without_fallback():
    request = remote_model_registration_request()
    session = MockSession(
        [result(json.dumps(remote_model_artifact_payload(model_id="other/model")))]
    )
    adapter = RemoteTrainingAdapter(
        config(),
        compute_profile_id="ssh-gpu-lab",
        session_factory=lambda: session,
    )

    with pytest.raises(RuntimeError, match="campaign_remote_model_receipt_invalid"):
        await adapter.register_remote_model(request)

    assert len(session.commands) == 1
    assert session.uploads == []
    assert session.downloads == []


def test_remote_model_manifest_ignores_hugging_face_download_bookkeeping(tmp_path):
    model = tmp_path / "model"
    cache = model / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    metadata = cache / "model.safetensors.metadata"
    metadata.write_text("first acquisition", encoding="utf-8")

    def inspect():
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                remote_contracts._REMOTE_MODEL_REGISTER_SCRIPT,
                str(model),
                "example/research-model",
                "a" * 40,
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        return json.loads(completed.stdout)

    before = inspect()
    metadata.write_text("different local cache state", encoding="utf-8")
    after = inspect()

    assert before == after


@pytest.mark.asyncio
async def test_registered_base_model_preflight_requires_resident_weights():
    session = MockSession([result()])
    source = RegisteredRemoteModelSource(
        source_id="registered-base-v1",
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="research-model-v1",
        model_digest="a" * 64,
        remote_model_path="/srv/bashgym/models/research-model",
    )
    adapter = RemoteTrainingAdapter(
        config(),
        compute_profile_id="ssh-gpu-lab",
        session_factory=lambda: session,
    )

    await adapter.verify_registered_base_model(source)

    command, timeout = session.commands[0]
    assert "test -f /srv/bashgym/models/research-model/config.json" in command
    assert "-name '*.safetensors'" in command
    assert "-name 'pytorch_model*.bin'" in command
    assert timeout == 10
    assert session.uploads == []
    assert session.downloads == []


@pytest.mark.asyncio
async def test_registered_base_model_preflight_fails_when_remote_model_is_incomplete():
    session = MockSession([result(status=1)])
    source = RegisteredRemoteModelSource(
        source_id="registered-base-v1",
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="research-model-v1",
        model_digest="a" * 64,
        remote_model_path="/srv/bashgym/models/research-model",
    )
    adapter = RemoteTrainingAdapter(
        config(),
        compute_profile_id="ssh-gpu-lab",
        session_factory=lambda: session,
    )

    with pytest.raises(RuntimeError, match="campaign_registered_base_model_not_ready"):
        await adapter.verify_registered_base_model(source)


@pytest.mark.asyncio
async def test_registered_v2_model_preflight_recomputes_and_matches_physical_identity():
    source = registered_v2_model_source()
    session = MockSession([result(json.dumps(remote_model_artifact_payload()))])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    await adapter.verify_registered_base_model(source)

    command, timeout = session.commands[0]
    assert source.remote_model_path in command
    assert source.artifact_receipt.model_id in command
    assert source.artifact_receipt.revision in command
    assert timeout == 3600
    assert session.uploads == []
    assert session.downloads == []


@pytest.mark.asyncio
async def test_registered_v2_model_preflight_rejects_changed_physical_identity():
    source = registered_v2_model_source()
    session = MockSession(
        [result(json.dumps(remote_model_artifact_payload(artifact_manifest_sha256="c" * 64)))]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    with pytest.raises(RuntimeError, match="campaign_registered_base_model_changed"):
        await adapter.verify_registered_base_model(source)


def launch_request(tmp_path):
    script = tmp_path / "train.py"
    dataset = tmp_path / "train.jsonl"
    script.write_text("print('fixture')\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    return RemoteLaunchRequest(
        compute_profile_id="ssh-gpu-lab",
        run_id="campaign-action-1",
        script_path=script,
        input_files=(dataset,),
        script_args=("--grouped-jsonl", "train.jsonl", "--output-dir", "."),
        recipe_digest="e" * 64,
        output_paths=("final", "training_manifest.json", "training_metrics.jsonl"),
    )


def lineage_launch_request(tmp_path):
    request = launch_request(tmp_path)
    archive = (tmp_path / ("d" * 64)).with_suffix(".tar")
    archive.write_bytes(b"deterministic source snapshot")
    snapshot = CodeLineageLaunchSnapshot(
        binding_id="bashgym-trainer-entrypoint-v1",
        binding_revision=1,
        binding_digest="a" * 64,
        source_repository_profile_id="bashgym-source-v1",
        lineage_id="lineage-candidate-1",
        record_digest="b" * 64,
        commit_sha="c" * 40,
        patch_sha256="d" * 64,
        entrypoint_path="bashgym/gym/trainer.py",
        working_directory="source",
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        archive_size_bytes=archive.stat().st_size,
    )
    return request.model_copy(update={"source_snapshot": snapshot})


def evaluation_launch_request(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    evaluator = tmp_path / "evaluate.py"
    context = tmp_path / "autoresearch_evaluation_context.json"
    model_root = tmp_path / "sealed-training" / "final"
    model_root.mkdir(parents=True)
    model_config = model_root / "config.json"
    model_weights = model_root / "weights.safetensors"
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    context.write_text('{"schema_version":"autoresearch_evaluation_context.v1"}', encoding="utf-8")
    model_config.write_text("{}", encoding="utf-8")
    model_weights.write_bytes(b"weights")
    sealed_inputs = tuple(
        SealedStageArtifactInput(
            campaign_artifact_id=f"artifact-{path.stem}",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            schema_name="huggingface_model_file.v1",
            local_sealed_path=path,
            remote_relative_path=f"model/{path.name}",
        )
        for path in (model_config, model_weights)
    )
    return RemoteLaunchRequest(
        compute_profile_id="ssh-gpu-lab",
        run_id="evaluation-attempt-1",
        script_path=evaluator,
        input_files=(context,),
        script_args=(
            "--context",
            context.name,
            "--model-dir",
            "model",
            "--dataset",
            "/srv/bashgym/datasets/terminal-heldout-v1.jsonl",
            "--output",
            "autoresearch_evaluation.json",
            "--batch-size",
            "8",
        ),
        recipe_digest="e" * 64,
        output_paths=("autoresearch_evaluation.json",),
        sealed_stage_artifact_inputs=sealed_inputs,
        source_training=SealedStageArtifactSource(
            campaign_id="campaign-a",
            study_id="study-a",
            action_id="action-training-a",
            attempt_id="attempt-training-a",
            stage_index=1,
        ),
        registered_evaluation_dataset=registered_evaluation_dataset_source(),
    )


def registered_base_evaluation_launch_request(tmp_path):
    evaluator = tmp_path / "evaluate-base.py"
    context = tmp_path / "autoresearch_evaluation_context_base.json"
    evaluator.write_text("print('evaluate base')\n", encoding="utf-8")
    context.write_text('{"schema_version":"autoresearch_evaluation_context.v1"}', encoding="utf-8")
    source = registered_v2_model_source()
    return RemoteLaunchRequest(
        compute_profile_id="ssh-gpu-lab",
        run_id="baseline-evaluation-attempt-1",
        script_path=evaluator,
        input_files=(context,),
        script_args=(
            "--context",
            context.name,
            "--model-dir",
            source.remote_model_path,
            "--dataset",
            "/srv/bashgym/datasets/terminal-heldout-v1.jsonl",
            "--output",
            "autoresearch_evaluation.json",
        ),
        recipe_digest="e" * 64,
        output_paths=("autoresearch_evaluation.json",),
        registered_base_model=source,
        registered_evaluation_dataset=registered_evaluation_dataset_source(),
    )


def test_baseline_evaluation_references_registered_remote_model_without_uploading_it(tmp_path):
    request = registered_base_evaluation_launch_request(tmp_path)

    manifest = RemoteTrainingAdapter._launch_manifest(request, "/remote/evaluation")

    assert manifest["registered_base_model"]["source_id"] == "research-model-base"
    assert manifest["registered_base_model"]["remote_model_path"] == (
        "/srv/bashgym/models/research-model"
    )
    assert [item["name"] for item in manifest["files"]] == [
        "evaluate-base.py",
        "autoresearch_evaluation_context_base.json",
    ]
    assert all(not item["name"].startswith("model/") for item in manifest["files"])
    assert manifest["registered_evaluation_dataset"] == {
        "schema_version": "campaign_registered_remote_evaluation_dataset_source.v1",
        "source_id": "terminal-heldout-v1",
        "compute_profile_id": "ssh-gpu-lab",
        "dataset_version_id": "terminal-heldout-v1",
        "content_digest": "b" * 64,
        "remote_dataset_path": "/srv/bashgym/datasets/terminal-heldout-v1.jsonl",
    }


def test_candidate_evaluation_references_remote_resident_checkpoint_without_uploading_it(
    tmp_path,
):
    source_type = getattr(remote_contracts, "RemoteResidentModelSource", None)
    assert source_type is not None, "remote-resident checkpoint contract is required"
    source = source_type(
        campaign_id="campaign-a",
        study_id="study-a",
        action_id="action-training-a",
        attempt_id="attempt-training-a",
        stage_index=1,
        compute_profile_id="ssh-gpu-lab",
        remote_model_path="/home/trainer/bashgym-training/attempt-training-a/final",
        files=(
            {
                "remote_relative_path": "model/config.json",
                "sha256": "a" * 64,
                "size_bytes": 2,
            },
            {
                "remote_relative_path": "model/weights.safetensors",
                "sha256": "b" * 64,
                "size_bytes": 7,
            },
        ),
    )
    baseline = registered_base_evaluation_launch_request(tmp_path)
    request = baseline.model_copy(
        update={
            "registered_base_model": None,
            "remote_resident_model": source,
            "script_args": tuple(
                (
                    source.remote_model_path
                    if value == baseline.registered_base_model.remote_model_path
                    else value
                )
                for value in baseline.script_args
            ),
        }
    )

    manifest = RemoteTrainingAdapter._launch_manifest(request, "/remote/evaluation")

    assert manifest["remote_resident_model"]["attempt_id"] == "attempt-training-a"
    assert [item["name"] for item in manifest["files"]] == [
        "evaluate-base.py",
        "autoresearch_evaluation_context_base.json",
    ]
    assert all(not item["name"].startswith("model/") for item in manifest["files"])


@pytest.mark.asyncio
async def test_candidate_checkpoint_files_are_verified_on_compute_before_launch(tmp_path):
    source_type = remote_contracts.RemoteResidentModelSource
    source = source_type(
        campaign_id="campaign-a",
        study_id="study-a",
        action_id="action-training-a",
        attempt_id="attempt-training-a",
        stage_index=1,
        compute_profile_id="ssh-gpu-lab",
        remote_model_path="/home/trainer/bashgym-training/attempt-training-a/final",
        files=(
            {
                "remote_relative_path": "model/config.json",
                "sha256": "a" * 64,
                "size_bytes": 2,
            },
        ),
    )
    session = MockSession([result()])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    await adapter.verify_remote_model_source(source)

    command = session.commands[0][0]
    assert f"{source.remote_model_path}/config.json" in command
    assert source.files[0].sha256 in command
    assert session.downloads == []


def test_evaluation_launch_rejects_mixed_registered_and_training_model_sources(tmp_path):
    baseline = registered_base_evaluation_launch_request(tmp_path)
    candidate = evaluation_launch_request(tmp_path / "candidate")

    with pytest.raises(ValidationError, match="exactly one evaluated model source"):
        RemoteLaunchRequest.model_validate(
            {
                **candidate.model_dump(mode="python"),
                "registered_base_model": baseline.registered_base_model,
            }
        )


def test_evaluation_launch_manifest_binds_context_and_checkpoint_destinations(tmp_path):
    request = evaluation_launch_request(tmp_path)

    manifest = RemoteTrainingAdapter._launch_manifest(request, "/remote/evaluation")

    assert manifest["request_digest"] == request.request_digest
    assert [item["name"] for item in manifest["files"]] == [
        "evaluate.py",
        "autoresearch_evaluation_context.json",
        "model/config.json",
        "model/weights.safetensors",
    ]
    assert manifest["argv"] == [
        "python3",
        "/remote/evaluation/evaluate.py",
        "--context",
        "autoresearch_evaluation_context.json",
        "--model-dir",
        "model",
        "--dataset",
        "/srv/bashgym/datasets/terminal-heldout-v1.jsonl",
        "--output",
        "autoresearch_evaluation.json",
        "--batch-size",
        "8",
    ]


@pytest.mark.parametrize(
    "remote_path",
    (
        "heldout.jsonl",
        "/",
        "/srv/bashgym/datasets/../heldout.jsonl",
        "/srv/bashgym/datasets/heldout.jsonl/",
        "C:\\heldout.jsonl",
    ),
)
def test_registered_evaluation_dataset_requires_one_absolute_file(remote_path):
    source_type = getattr(remote_contracts, "RegisteredRemoteEvaluationDatasetSource", None)
    assert source_type is not None, "registered remote evaluation dataset contract is required"

    with pytest.raises(ValidationError, match="evaluation dataset path"):
        source_type(
            source_id="terminal-heldout-v1",
            compute_profile_id="ssh-gpu-lab",
            dataset_version_id="terminal-heldout-v1",
            content_digest="b" * 64,
            remote_dataset_path=remote_path,
        )


def test_remote_launch_rejects_evaluation_dataset_from_another_compute_profile(tmp_path):
    request = registered_base_evaluation_launch_request(tmp_path)

    with pytest.raises(ValidationError, match="evaluation dataset compute profile mismatch"):
        RemoteLaunchRequest.model_validate(
            {
                **request.model_dump(mode="python", exclude={"registered_evaluation_dataset"}),
                "registered_evaluation_dataset": registered_evaluation_dataset_source(
                    compute_profile_id="other-private-compute"
                ),
            }
        )


@pytest.mark.asyncio
async def test_evaluation_dataset_hash_is_verified_before_launch_without_transfer(tmp_path):
    request = registered_base_evaluation_launch_request(tmp_path)
    session = MockSession(
        [
            result("/home/trainer"),
            result(json.dumps(remote_model_artifact_payload())),
            result(status=1),
        ]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    with pytest.raises(RuntimeError, match="campaign_registered_evaluation_dataset_invalid"):
        await adapter.launch(request)

    model_command, model_timeout = session.commands[1]
    assert request.registered_base_model.remote_model_path in model_command
    assert model_timeout == 3600
    verification_command, timeout = session.commands[2]
    assert request.registered_evaluation_dataset.remote_dataset_path in verification_command
    assert request.registered_evaluation_dataset.content_digest in verification_command
    assert timeout == 3600
    assert session.uploads == []
    assert session.downloads == []


def test_evaluation_profile_can_have_no_static_inputs_but_training_cannot(tmp_path):
    script = tmp_path / "runner.py"
    script.write_text("print('runner')\n", encoding="utf-8")
    script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()

    evaluation = PinnedRemoteStageProfile(
        stage=StageKind.DEVELOPMENT_EVALUATION,
        script_path=script,
        script_sha256=script_sha256,
        input_files=(),
        input_sha256={},
        output_paths=("autoresearch_evaluation.json",),
        budget_reservation=0.25,
    )
    assert evaluation.input_files == ()

    with pytest.raises(ValidationError, match="training.*input file"):
        PinnedRemoteStageProfile(
            stage=StageKind.FULL_TRAINING,
            script_path=script,
            script_sha256=script_sha256,
            input_files=(),
            input_sha256={},
            budget_reservation=0.25,
        )


def test_executor_config_binds_registered_evaluation_dataset_without_local_rows(tmp_path):
    key = tmp_path / "worker-key"
    evaluator = tmp_path / "evaluate.py"
    key.write_text("test-only-key\n", encoding="utf-8")
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    evaluator_digest = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    stage = PinnedRemoteStageProfile(
        stage=StageKind.DEVELOPMENT_EVALUATION,
        script_path=evaluator,
        script_sha256=evaluator_digest,
        input_files=(),
        input_sha256={},
        output_paths=("autoresearch_evaluation.json",),
        budget_reservation=0.25,
    )
    registered_dataset = registered_evaluation_dataset_source()
    profile = ApprovedRemoteExecutorProfile(
        profile_id="evaluation-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="research-model-v1",
        target_model_digest="d" * 64,
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(stage,),
        registered_base_model=RegisteredRemoteModelSource(
            source_id="research-model-base",
            compute_profile_id="ssh-gpu-lab",
            target_contract_key="research-model-v1",
            model_digest="d" * 64,
            remote_model_path="/srv/bashgym/models/research-model",
        ),
        registered_evaluation_dataset=registered_dataset,
    )
    dataset_version = DatasetVersionSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        dataset_id="terminal-heldout",
        dataset_version_id="terminal-heldout-v1",
        source_uri="bashgym-remote-dataset://terminal-heldout-v1",
        content_digest="b" * 64,
    )
    suite = EvaluationSuiteSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_suite_id="terminal-heldout-suite-v1",
        name="Terminal heldout",
        task_type="terminal-agent-sft",
        dataset_version_id="terminal-heldout-v1",
        metric_contract={"primary_metric": "task_success_rate"},
        code_digest=evaluator_digest,
    )

    executor = remote_executor_config(
        profile,
        StageKind.DEVELOPMENT_EVALUATION,
        recipe_digest="e" * 64,
        evaluation_suite=suite,
        dataset_version=dataset_version,
        evaluate_registered_base_model=True,
    )

    assert executor["input_files"] == []
    assert executor["expected_input_sha256"] == {}
    assert executor["registered_evaluation_dataset"] == registered_dataset.model_dump(mode="json")
    assert executor["evaluation_binding"]["dataset_remote_path"] == (
        "/srv/bashgym/datasets/terminal-heldout-v1.jsonl"
    )


def test_evaluation_launch_rejects_tampered_or_unsafe_sealed_stage_artifact(tmp_path):
    request = evaluation_launch_request(tmp_path)
    request.sealed_stage_artifact_inputs[0].local_sealed_path.write_text(
        '{"tampered":true}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="sealed stage artifact.*changed"):
        RemoteTrainingAdapter._launch_manifest(request, "/remote/evaluation")
    with pytest.raises(ValidationError, match="remote relative path"):
        SealedStageArtifactInput(
            campaign_artifact_id="artifact-a",
            sha256="a" * 64,
            size_bytes=1,
            schema_name="huggingface_model_file.v1",
            local_sealed_path=tmp_path / "missing",
            remote_relative_path="../escape",
        )


@pytest.mark.parametrize(
    "remote_path",
    ("model\\config.json", "model//config.json", "model/./config.json"),
)
def test_sealed_stage_artifact_rejects_noncanonical_remote_paths(tmp_path, remote_path):
    artifact = tmp_path / "config.json"
    artifact.write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError, match="remote relative path"):
        SealedStageArtifactInput(
            campaign_artifact_id="artifact-model-config",
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            size_bytes=artifact.stat().st_size,
            schema_name="huggingface_model_file.v1",
            local_sealed_path=artifact,
            remote_relative_path=remote_path,
        )


@pytest.mark.parametrize(
    "reserved_arg",
    (
        "--context",
        "--context=other.json",
        "--model-dir",
        "--model-dir=other",
        "--dataset",
        "--dataset=other.jsonl",
        "--output",
        "--output=other.json",
    ),
)
def test_evaluation_profile_rejects_reserved_abi_arguments(tmp_path, reserved_arg):
    evaluator = tmp_path / "evaluate.py"
    dataset = tmp_path / "development.jsonl"
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="reserved evaluator argument"):
        PinnedRemoteStageProfile(
            stage=StageKind.DEVELOPMENT_EVALUATION,
            script_path=evaluator,
            script_sha256=hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            input_files=(dataset,),
            input_sha256={dataset.name: hashlib.sha256(dataset.read_bytes()).hexdigest()},
            script_args=(reserved_arg,),
            output_paths=("autoresearch_evaluation.json",),
            budget_reservation=0.25,
        )


def identity():
    return RemoteRunIdentity(
        compute_profile_id="ssh-gpu-lab",
        run_id="campaign-action-1",
        remote_run_directory="/home/trainer/bashgym-training/campaign-action-1",
        remote_pid=4242,
        process_group_id=4242,
        process_start_ticks=9001,
        boot_id="boot-1",
        command_hash="a" * 64,
        launch_manifest_sha256="b" * 64,
        launched_at=utc_now(),
    )


def test_launch_request_pins_exact_python_executable(tmp_path):
    request = launch_request(tmp_path).model_copy(
        update={"python_executable": "/opt/memexai/.venv/bin/python"}
    )

    assert RemoteTrainingAdapter._argv(request, "/remote/run")[0] == (
        "/opt/memexai/.venv/bin/python"
    )
    with pytest.raises(ValidationError, match="exact executable path"):
        RemoteLaunchRequest(
            **launch_request(tmp_path).model_dump(exclude={"python_executable"}),
            python_executable="python3; touch /tmp/unsafe",
        )


@pytest.mark.asyncio
async def test_launch_executes_verified_captured_source_snapshot(tmp_path):
    request = lineage_launch_request(tmp_path)
    adapter = RemoteTrainingAdapter(config(), compute_profile_id="ssh-gpu-lab")
    remote_directory = "/home/trainer/bashgym-training/campaign-action-1"
    manifest = adapter._launch_manifest(request, remote_directory)
    launched_identity = identity().model_copy(
        update={
            "command_hash": manifest["command_hash"],
            "launch_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    session = MockSession(
        [result("/home/trainer"), result(), result(), result(supervisor_json(launched_identity))]
    )
    adapter._session_factory = lambda: session

    recovered = await adapter.launch(request)

    assert recovered == launched_identity
    assert manifest["code_lineage"]["commit_sha"] == "c" * 40
    assert manifest["execution_context"] == {
        "entrypoint_kind": "captured_source_snapshot",
        "working_directory": f"{remote_directory}/source",
        "python_path": f"{remote_directory}/source",
    }
    assert manifest["argv"][1] == "-c"
    assert "sys.path.insert(0,source)" in manifest["argv"][2]
    assert manifest["argv"][3] == f"{remote_directory}/source"
    assert manifest["argv"][4] == f"{remote_directory}/source/bashgym/gym/trainer.py"
    assert [local for local, _remote in session.uploads] == [
        request.source_snapshot.archive_path,
        request.input_files[0],
    ]
    assert "tar --extract" in session.commands[2][0]
    assert "test ! -L source/bashgym/gym/trainer.py" in session.commands[2][0]
    assert f"PYTHONPATH={remote_directory}/source" in session.commands[3][0]


def test_launch_rechecks_snapshot_after_request_construction(tmp_path):
    request = lineage_launch_request(tmp_path)
    request.source_snapshot.archive_path.write_bytes(b"changed after request validation")

    with pytest.raises(ValueError, match="snapshot changed before launch"):
        RemoteTrainingAdapter._launch_manifest(request, "/remote/run")


def supervisor_json(value: RemoteRunIdentity) -> str:
    payload = value.model_dump(mode="json", exclude={"schema_version"})
    payload["schema_version"] = "campaign_remote_supervisor_state.v1"
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_launch_exclusively_creates_verifies_and_returns_server_neutral_identity(tmp_path):
    request = launch_request(tmp_path)
    adapter = RemoteTrainingAdapter(config(), compute_profile_id="ssh-gpu-lab")
    remote_directory = "/home/trainer/bashgym-training/campaign-action-1"
    manifest = adapter._launch_manifest(request, remote_directory)
    launched_identity = identity().model_copy(
        update={
            "command_hash": manifest["command_hash"],
            "launch_manifest_sha256": __import__("hashlib")
            .sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
            .hexdigest(),
        }
    )
    session = MockSession(
        [result("/home/trainer"), result(), result(), result(supervisor_json(launched_identity))]
    )
    adapter._session_factory = lambda: session

    recovered = await adapter.launch(request)

    assert recovered == launched_identity
    assert "host" not in type(recovered).model_fields
    assert "username" not in type(recovered).model_fields
    assert [remote for _local, remote in session.uploads] == [
        f"{remote_directory}/train.py",
        f"{remote_directory}/train.jsonl",
    ]
    assert "mkdir /home/trainer/bashgym-training/campaign-action-1" in session.commands[1][0]
    assert "sha256sum -c" in session.commands[2][0]
    assert "setsid" in session.commands[3][0]
    assert "remote_run_state.v1.json.tmp" in session.commands[3][0]


@pytest.mark.asyncio
async def test_discover_recovers_exact_manifest_without_starting_a_second_process(tmp_path):
    request = launch_request(tmp_path)
    adapter = RemoteTrainingAdapter(config(), compute_profile_id="ssh-gpu-lab")
    directory = "/home/trainer/bashgym-training/campaign-action-1"
    manifest = adapter._launch_manifest(request, directory)
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    expected = identity().model_copy(
        update={
            "command_hash": manifest["command_hash"],
            "launch_manifest_sha256": __import__("hashlib")
            .sha256(manifest_json.encode())
            .hexdigest(),
        }
    )
    session = MockSession([result("/home/trainer"), result(supervisor_json(expected))])
    adapter._session_factory = lambda: session

    recovered = await adapter.discover(request)

    assert recovered == expected
    assert not session.uploads
    assert not any("nohup" in command for command, _timeout in session.commands)


@pytest.mark.asyncio
async def test_discover_returns_none_only_for_absent_state(tmp_path):
    session = MockSession([result("/home/trainer"), result(status=1)])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    assert await adapter.discover(launch_request(tmp_path)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "expected_state", "expected_reason"),
    [
        (f"boot-1\t9001\t4242\tS\t{'b' * 64}\t\n", RemoteRunState.RUNNING, "remote_process_alive"),
        (f"boot-1\t9001\t4242\tT\t{'b' * 64}\t\n", RemoteRunState.PAUSED, "remote_process_paused"),
        (f"boot-1\t\t\t\t{'b' * 64}\t0\n", RemoteRunState.COMPLETED, "remote_exit_code_recorded"),
        (f"boot-1\t\t\t\t{'b' * 64}\t7\n", RemoteRunState.FAILED, "remote_exit_code_recorded"),
        (
            f"boot-1\tbad\t4242\tS\t{'b' * 64}\t\n",
            RemoteRunState.UNKNOWN,
            "remote_observation_malformed",
        ),
        (
            f"boot-2\t9001\t4242\tS\t{'b' * 64}\t\n",
            RemoteRunState.UNKNOWN,
            "remote_process_identity_mismatch",
        ),
        (f"boot-1\t9001\t4242\tZ\t{'b' * 64}\t\n", RemoteRunState.UNKNOWN, "remote_exit_unproven"),
    ],
)
async def test_observe_handles_running_paused_exit_malformed_and_zombie_states(
    stdout, expected_state, expected_reason
):
    session = MockSession([result(stdout)])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    observation = await adapter.observe(identity())
    assert observation.state == expected_state
    assert observation.safe_reason == expected_reason


@pytest.mark.asyncio
async def test_controls_validate_identity_and_signal_process_group_in_one_command():
    session = MockSession([result(), result(status=42)])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    assert await adapter.terminate(identity()) is True
    assert await adapter.force_stop(identity()) is False
    first = session.commands[0][0]
    assert "expected_start=9001" in first
    assert "expected_pgid=4242" in first
    assert 'kill -TERM -- "-$pgid"' in first
    assert 'kill -KILL -- "-$pgid"' in session.commands[1][0]


@pytest.mark.asyncio
async def test_controller_output_download_is_disabled(tmp_path):
    request = launch_request(tmp_path)
    session = MockSession([])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    with pytest.raises(RuntimeError, match="campaign_controller_output_download_disabled"):
        await adapter.collect_outputs(identity(), request, tmp_path / "download")

    assert session.commands == []
    assert session.downloads == []
    assert not (tmp_path / "download").exists()


@pytest.mark.asyncio
async def test_remote_output_inventory_hashes_on_compute_without_downloading(tmp_path):
    request = launch_request(tmp_path)
    remote_files = [
        {"path": "exit_code", "sha256": "a" * 64, "size_bytes": 2},
        {"path": "final/config.json", "sha256": "b" * 64, "size_bytes": 2},
        {"path": "launch_manifest.json", "sha256": "c" * 64, "size_bytes": 20},
        {"path": "training.log", "sha256": "d" * 64, "size_bytes": 9},
        {"path": "training_manifest.json", "sha256": "e" * 64, "size_bytes": 20},
        {"path": "training_metrics.jsonl", "sha256": "f" * 64, "size_bytes": 24},
    ]
    session = MockSession([result(json.dumps(remote_files))])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    remote_identity = identity()
    observation = RemoteObservation(
        identity=remote_identity,
        state=RemoteRunState.COMPLETED,
        observed_at=utc_now(),
        exit_code=0,
        safe_reason="remote_exit_code_recorded",
    )

    inventory = await adapter.inventory_outputs(remote_identity, request, observation=observation)

    assert [item.path for item in inventory.files] == [item["path"] for item in remote_files]
    assert session.downloads == []
    assert not (tmp_path / "download").exists()


@pytest.mark.asyncio
async def test_remote_seal_and_bounded_evaluation_stay_in_memory(tmp_path):
    evaluation = b'{"schema_version":"autoresearch_evaluation.v1"}'
    evaluation_sha = hashlib.sha256(evaluation).hexdigest()
    session = MockSession(
        [
            result(
                json.dumps(
                    {
                        "data": base64.b64encode(evaluation).decode("ascii"),
                        "sha256": evaluation_sha,
                        "size_bytes": len(evaluation),
                    }
                )
            )
        ]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    remote_identity = identity()
    seal_payload = b'{"schema_version":"sealed_action_result_envelope.v1"}'

    await adapter.persist_action_seal(remote_identity, seal_payload)
    read_back = await adapter.read_output_bytes(
        remote_identity,
        "autoresearch_evaluation.json",
        expected_sha256=evaluation_sha,
        expected_size_bytes=len(evaluation),
        max_bytes=1024,
    )

    assert read_back == evaluation
    assert session.byte_uploads == [
        (
            seal_payload,
            f"{remote_identity.remote_run_directory}/sealed_action_result.v1.json",
        )
    ]
    assert session.downloads == []
    assert not (tmp_path / "download").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relative_path", "expected_size_bytes", "max_bytes", "response", "error"),
    (
        ("../private.json", 1, 1024, None, ValueError),
        ("evaluation.json", 2048, 1024, None, ValueError),
        (
            "autoresearch_evaluation.json",
            1,
            1024,
            "not-json",
            RuntimeError,
        ),
        (
            "autoresearch_evaluation.json",
            1,
            1024,
            json.dumps(
                {
                    "data": base64.b64encode(b"x").decode("ascii"),
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                }
            ),
            RuntimeError,
        ),
    ),
)
async def test_bounded_remote_output_read_rejects_unsafe_or_unverified_bytes(
    relative_path,
    expected_size_bytes,
    max_bytes,
    response,
    error,
):
    session = MockSession([] if response is None else [result(response)])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    with pytest.raises(error):
        await adapter.read_output_bytes(
            identity(),
            relative_path,
            expected_sha256=hashlib.sha256(b"x").hexdigest(),
            expected_size_bytes=expected_size_bytes,
            max_bytes=max_bytes,
        )


@pytest.mark.asyncio
async def test_remote_resident_dataset_verification_hashes_each_shard_in_place():
    source = RemoteResidentDatasetSource(
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-data-1",
        attempt_id="attempt-data-1",
        stage_index=0,
        compute_profile_id="ssh-gpu-lab",
        remote_dataset_path="/srv/bashgym/runs/attempt-data-1/dataset",
        dataset_id="dataset-generated",
        dataset_version_id="dataset-generated-v1",
        content_digest="a" * 64,
        files=(
            RemoteResidentDatasetFile(
                remote_relative_path="train.jsonl",
                sha256="b" * 64,
                size_bytes=17,
            ),
            RemoteResidentDatasetFile(
                remote_relative_path="validation.jsonl",
                sha256="c" * 64,
                size_bytes=19,
            ),
        ),
    )
    session = MockSession([result("")])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    await adapter.verify_remote_dataset_source(source)

    command = session.commands[0][0]
    assert source.remote_dataset_path in command
    assert "train.jsonl" in command and "validation.jsonl" in command
    assert "stat -c %s" in command
    assert "sha256sum" in command
    assert session.downloads == []


@pytest.mark.asyncio
async def test_controller_terminal_evidence_download_is_disabled(tmp_path):
    session = MockSession([])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    remote_identity = identity()
    observation = RemoteObservation(
        identity=remote_identity,
        state=RemoteRunState.FAILED,
        observed_at=utc_now(),
        exit_code=7,
        safe_reason="remote_exit_code_recorded",
    )

    with pytest.raises(RuntimeError, match="campaign_controller_output_download_disabled"):
        await adapter.collect_terminal_evidence(
            remote_identity, tmp_path / "terminal", observation=observation
        )

    assert session.commands == []
    assert session.downloads == []
    assert not (tmp_path / "terminal").exists()


@pytest.mark.asyncio
async def test_terminal_inventory_keeps_failure_evidence_on_compute():
    remote_files = [
        {"path": "exit_code", "sha256": "a" * 64, "size_bytes": 2},
        {"path": "launch_manifest.json", "sha256": "b" * 64, "size_bytes": 20},
        {"path": "training.log", "sha256": "c" * 64, "size_bytes": 9},
    ]
    session = MockSession([result(json.dumps(remote_files))])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    remote_identity = identity()
    observation = RemoteObservation(
        identity=remote_identity,
        state=RemoteRunState.FAILED,
        observed_at=utc_now(),
        exit_code=7,
        safe_reason="remote_exit_code_recorded",
    )

    inventory = await adapter.inventory_terminal_evidence(remote_identity, observation=observation)

    assert [item.path for item in inventory.files] == [item["path"] for item in remote_files]
    assert session.downloads == []


@pytest.mark.asyncio
async def test_capacity_preflight_blocks_hermes_occupancy_and_low_memory():
    session = MockSession(
        [result("/home/trainer"), result("42.125\t167\t111, llama-server;222, llama-server;\n")]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    snapshot = await adapter.capacity_preflight(RemoteCapacityPolicy())
    assert snapshot.admitted is False
    assert snapshot.available_memory_gib == 42.125
    assert snapshot.blocking_reasons == (
        "available_memory_below_minimum",
        "external_gpu_process_limit_exceeded",
    )
    capacity_command = session.commands[1][0]
    assert 'while [ ! -e "$probe" ]' in capacity_command
    assert 'df -BG --output=avail "$probe"' in capacity_command


def test_supervisor_state_writer_uses_typed_json_instead_of_printf_placeholders(tmp_path):
    request = launch_request(tmp_path)
    remote_directory = f"/home/trainer/bashgym-training/{request.run_id}"
    manifest = RemoteTrainingAdapter._launch_manifest(request, remote_directory)
    session = MockSession(
        [
            result("/home/trainer"),
            result(),
            result(),
            result(
                json.dumps(
                    {
                        "schema_version": "campaign_remote_supervisor_state.v1",
                        "compute_profile_id": request.compute_profile_id,
                        "run_id": request.run_id,
                        "remote_run_directory": remote_directory,
                        "remote_pid": 123,
                        "process_group_id": 123,
                        "process_start_ticks": 456,
                        "boot_id": "boot-proof",
                        "command_hash": manifest["command_hash"],
                        "launch_manifest_sha256": hashlib.sha256(
                            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "launched_at": "2026-07-13T21:00:00Z",
                    }
                )
            ),
        ]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    identity = asyncio.run(adapter.launch(request))

    assert identity.remote_pid == 123
    launch_command = session.commands[-1][0]
    assert "python3 -c" in launch_command
    assert "int(sys.argv[4])" in launch_command
    assert "remote_run_state.v1.json.tmp" in launch_command


@pytest.mark.asyncio
async def test_stream_cursor_preserves_partial_lines_across_reads():
    first_bytes = b'{"step":1}\n{"step"'
    second_bytes = b":2}\n"
    session = MockSession(
        [
            result(
                json.dumps(
                    {
                        "end_offset": len(first_bytes),
                        "data": base64.b64encode(first_bytes).decode(),
                    }
                )
            ),
            result(
                json.dumps(
                    {
                        "end_offset": len(first_bytes) + len(second_bytes),
                        "data": base64.b64encode(second_bytes).decode(),
                    }
                )
            ),
        ]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    first = await adapter.read_stream(identity(), "training_metrics.jsonl")
    assert first.complete_lines == ('{"step":1}',)
    assert first.next_cursor.partial_line == '{"step"'
    second = await adapter.read_stream(identity(), "training_metrics.jsonl", first.next_cursor)
    assert second.complete_lines == ('{"step":2}',)
    assert second.next_cursor == RemoteStreamCursor(
        byte_offset=len(first_bytes) + len(second_bytes), partial_line=""
    )


def test_launch_contract_rejects_secret_arguments_and_path_escape(tmp_path):
    request = launch_request(tmp_path)
    with pytest.raises(ValidationError, match="credentials"):
        RemoteLaunchRequest(
            **request.model_dump(exclude={"script_args"}),
            script_args=("--api-key=raw-secret",),
        )
    with pytest.raises(ValidationError, match="inside"):
        RemoteLaunchRequest(
            **request.model_dump(exclude={"output_paths"}),
            output_paths=("../escape",),
        )
