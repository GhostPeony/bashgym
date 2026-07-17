"""Portable, side-effect-bounded AutoResearch installation activation tests."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from bashgym.campaigns.activation import (
    AutoResearchActivationError,
    AutoResearchActivationRequest,
    activate_autoresearch,
)
from bashgym.campaigns.contracts import CodeMutationKind, StageKind, utc_now
from bashgym.campaigns.installation import (
    autoresearch_binding_plan,
    build_quality_autoresearch_definition,
)
from bashgym.campaigns.lineage import ApprovedSourceRepositoryProfile
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
)
from bashgym.campaigns.worker_service import (
    ControllerStatusProjection,
    WorkerPlatform,
    WorkerRunConfig,
    read_worker_config,
    write_worker_config,
)
from bashgym.ledger.contracts import (
    DatasetSpec,
    DatasetVersionSpec,
    EvaluationSuiteSpec,
    ProjectSpec,
)


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _source_repository(root: Path, *, value: str = "1e-4") -> Path:
    root.mkdir(parents=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Activation Test")
    _git(root, "config", "user.email", "activation@example.invalid")
    trainer = root / "bashgym" / "gym" / "trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(f"LEARNING_RATE = {value}\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "activation fixture")
    return root.resolve()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_profile(repository: Path, *, profile_id: str) -> ApprovedSourceRepositoryProfile:
    return ApprovedSourceRepositoryProfile(
        profile_id=profile_id,
        repository_path=repository,
        allowed_mutation_paths={
            CodeMutationKind.TRAINER: ("bashgym/gym/trainer.py",),
        },
    )


def _executor_profile(
    root: Path,
    *,
    profile_id: str,
    compute_profile_id: str,
    target_contract_key: str,
    target_model_digest: str,
    source_profile_id: str | None,
    required_stages: tuple[StageKind, ...],
) -> ApprovedRemoteExecutorProfile:
    root.mkdir(parents=True, exist_ok=True)
    key = root / f"{profile_id}.key"
    data = root / f"{profile_id}.jsonl"
    key.write_text("test-only-ssh-key\n", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")
    stages: list[PinnedRemoteStageProfile] = []
    for stage_kind in sorted(required_stages, key=lambda item: item.value):
        script = root / f"{profile_id}-{stage_kind.value}.py"
        script.write_text("print('bounded activation test')\n", encoding="utf-8")
        binding = (
            ApprovedCodeLineageExecutionBinding(
                binding_id=f"{profile_id}-{stage_kind.value}-source-v1",
                binding_revision=1,
                source_repository_profile_id=source_profile_id,
                entrypoint_path="bashgym/gym/trainer.py",
            )
            if source_profile_id is not None
            else None
        )
        stages.append(
            PinnedRemoteStageProfile(
                stage=stage_kind,
                script_path=script,
                script_sha256=_file_sha256(script),
                input_files=(data,),
                input_sha256={data.name: _file_sha256(data)},
                budget_unit="gpu_hours",
                budget_reservation=0.25,
                code_lineage_binding=binding,
            )
        )
    return ApprovedRemoteExecutorProfile(
        profile_id=profile_id,
        profile_revision=1,
        compute_profile_id=compute_profile_id,
        target_contract_key=target_contract_key,
        target_model_digest=target_model_digest,
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=tuple(stages),
    )


def _activation_fixture(tmp_path: Path):
    definition = build_quality_autoresearch_definition(
        template_id="portable-autoresearch-v1",
        template_revision="1",
        objective="Improve one held-out terminal-agent metric.",
        model_ref=f"hf://example/modern-open-model@{'a' * 40}",
        target_contract_key="modern-open-terminal-agent-v1",
        task="terminal-agent-sft",
        dataset_version_id="terminal-agent-data-v1",
        compute_profile_id="private-gpu-lab",
        source_repository_profile_id="bashgym-source-v1",
        ledger_project_id="autoresearch-project",
        evaluation_suite_id="terminal-agent-heldout-v1",
        primary_metric="exact_task_accuracy",
        metric_direction="maximize",
        budget_unit="gpu_hours",
        budget_limit=2.0,
        max_attempts=3,
        minimum_improvement=0.01,
    )
    binding = autoresearch_binding_plan(definition)
    source = _source_profile(
        _source_repository(tmp_path / "source"),
        profile_id=binding.source_repository_profile_id,
    )
    executor = _executor_profile(
        tmp_path / "launch-material",
        profile_id="private-terminal-agent-v1",
        compute_profile_id=binding.compute_profile_id,
        target_contract_key=binding.target_contract_key,
        target_model_digest=binding.target_model_digest,
        source_profile_id=source.profile_id,
        required_stages=binding.required_training_stages,
    )
    request = AutoResearchActivationRequest(
        workspace_id="workspace-a",
        project=ProjectSpec(
            workspace_id="workspace-a",
            project_id=binding.ledger_project_id,
            display_name="AutoResearch activation test",
            owner_actor_id="test-operator",
        ),
        dataset=DatasetSpec(
            workspace_id="workspace-a",
            project_id=binding.ledger_project_id,
            dataset_id="terminal-agent-data",
            display_name="Approved terminal-agent data",
            task_type="terminal-agent-sft",
        ),
        dataset_version=DatasetVersionSpec(
            workspace_id="workspace-a",
            project_id=binding.ledger_project_id,
            dataset_id="terminal-agent-data",
            dataset_version_id=binding.dataset_version_id,
            source_uri="artifact://fixture/terminal-agent-data-v1",
            content_digest="b" * 64,
        ),
        evaluation_suite=EvaluationSuiteSpec(
            workspace_id="workspace-a",
            project_id=binding.ledger_project_id,
            evaluation_suite_id=binding.evaluation_suite_id,
            name="Terminal agent heldout",
            task_type="terminal-agent-sft",
            dataset_version_id=binding.dataset_version_id,
            metric_contract={
                "primary_metric": binding.primary_metric,
                "metric_direction": binding.metric_direction.value,
            },
            code_digest="c" * 64,
        ),
        source_profile=source,
        executor_profile=executor,
    )
    return definition, request


def _memory_secrets():
    secrets: dict[str, str] = {}

    def resolve(name: str) -> str | None:
        return secrets.get(name)

    def write(name: str, value: str) -> None:
        secrets[name] = value

    return secrets, resolve, write


def test_plan_reports_exact_records_without_writing_installation_state(tmp_path: Path) -> None:
    definition, request = _activation_fixture(tmp_path)
    data_directory = tmp_path / "installation"

    def unexpected_secret_access(_name: str):
        raise AssertionError("planning must not access the installation secret store")

    receipt = activate_autoresearch(
        definition,
        request,
        data_directory=data_directory,
        service_target=WorkerPlatform.LINUX,
        service_home=tmp_path / "home",
        secret_resolver=unexpected_secret_access,
        secret_writer=lambda _name, _value: unexpected_secret_access(_name),
    )

    assert receipt.applied is False
    assert receipt.doctor is None
    assert receipt.service == "planned"
    assert {record.record_id for record in receipt.records} == {
        request.project.project_id,
        request.dataset.dataset_id,
        request.dataset_version.dataset_version_id,
        request.evaluation_suite.evaluation_suite_id,
    }
    assert {record.disposition for record in receipt.records} == {"planned"}
    assert not data_directory.exists()


def test_apply_creates_materializable_installation_with_offline_controller(
    tmp_path: Path,
) -> None:
    definition, request = _activation_fixture(tmp_path)
    data_directory = tmp_path / "installation"
    secrets, resolve, write = _memory_secrets()

    receipt = activate_autoresearch(
        definition,
        request,
        data_directory=data_directory,
        apply=True,
        service_target=WorkerPlatform.LINUX,
        service_home=tmp_path / "home",
        secret_resolver=resolve,
        secret_writer=write,
    )

    assert receipt.applied is True
    assert {record.disposition for record in receipt.records} == {"created"}
    assert receipt.worker_config == "created"
    assert receipt.service == "not_installed"
    assert receipt.doctor is not None
    assert receipt.doctor.materializable is True
    assert receipt.doctor.launch_ready is False
    assert receipt.doctor.blocking_codes == ("controller_offline",)
    assert len(secrets["BASHGYM_CAMPAIGN_SEAL_KEY"].encode("utf-8")) >= 32

    config = read_worker_config(data_directory / "campaigns" / "worker-config.v1.json")
    assert config.approved_source_profiles == (request.source_profile,)
    assert config.approved_remote_profiles == (request.executor_profile,)
    assert (data_directory / "campaigns" / "campaigns.sqlite3").is_file()


def test_apply_replays_every_record_and_preserves_the_existing_seal_key(
    tmp_path: Path,
) -> None:
    definition, request = _activation_fixture(tmp_path)
    data_directory = tmp_path / "installation"
    secrets, resolve, write = _memory_secrets()
    first = activate_autoresearch(
        definition,
        request,
        data_directory=data_directory,
        apply=True,
        service_target=WorkerPlatform.LINUX,
        service_home=tmp_path / "home",
        secret_resolver=resolve,
        secret_writer=write,
    )
    original_key = secrets["BASHGYM_CAMPAIGN_SEAL_KEY"]

    replay = activate_autoresearch(
        definition,
        request,
        data_directory=data_directory,
        apply=True,
        service_target=WorkerPlatform.LINUX,
        service_home=tmp_path / "home",
        secret_resolver=resolve,
        secret_writer=write,
    )

    assert {record.disposition for record in first.records} == {"created"}
    assert {record.disposition for record in replay.records} == {"replayed"}
    assert replay.worker_config == "replayed"
    assert replay.doctor is not None and replay.doctor.materializable is True
    assert secrets["BASHGYM_CAMPAIGN_SEAL_KEY"] == original_key


def test_conflicting_profile_fails_before_writing_and_preserves_worker_config(
    tmp_path: Path,
) -> None:
    definition, request = _activation_fixture(tmp_path)
    data_directory = tmp_path / "installation"
    config_path = data_directory / "campaigns" / "worker-config.v1.json"
    conflicting_source = _source_profile(
        _source_repository(tmp_path / "conflicting-source", value="2e-4"),
        profile_id=request.source_profile.profile_id,
    )
    initial = WorkerRunConfig.for_data_directory(
        data_directory,
        approved_source_profiles=(conflicting_source,),
    )
    write_worker_config(config_path, initial)
    original_config = config_path.read_bytes()

    with pytest.raises(
        AutoResearchActivationError,
        match="source profile ID already exists with a different identity",
    ):
        activate_autoresearch(
            definition,
            request,
            data_directory=data_directory,
            apply=True,
            service_target=WorkerPlatform.LINUX,
            service_home=tmp_path / "home",
            secret_resolver=lambda _name: None,
            secret_writer=lambda _name, _value: None,
        )

    assert config_path.read_bytes() == original_config
    assert not (data_directory / "campaigns" / "campaigns.sqlite3").exists()


def test_apply_preserves_unrelated_approved_source_and_compute_profiles(
    tmp_path: Path,
) -> None:
    definition, request = _activation_fixture(tmp_path)
    data_directory = tmp_path / "installation"
    config_path = data_directory / "campaigns" / "worker-config.v1.json"
    unrelated_source = _source_profile(
        _source_repository(tmp_path / "unrelated-source"),
        profile_id="unrelated-source-v1",
    )
    unrelated_executor = _executor_profile(
        tmp_path / "unrelated-launch-material",
        profile_id="unrelated-executor-v1",
        compute_profile_id="secondary-private-gpu",
        target_contract_key="other-model-v1",
        target_model_digest="d" * 64,
        source_profile_id=None,
        required_stages=(StageKind.FULL_TRAINING,),
    )
    initial = WorkerRunConfig.for_data_directory(
        data_directory,
        approved_remote_profiles=(unrelated_executor,),
        approved_source_profiles=(unrelated_source,),
    )
    write_worker_config(config_path, initial)
    _secrets, resolve, write = _memory_secrets()

    receipt = activate_autoresearch(
        definition,
        request,
        data_directory=data_directory,
        apply=True,
        service_target=WorkerPlatform.LINUX,
        service_home=tmp_path / "home",
        secret_resolver=resolve,
        secret_writer=write,
    )

    assert receipt.worker_config == "updated"
    merged = read_worker_config(config_path)
    assert {profile.profile_id for profile in merged.approved_source_profiles} == {
        unrelated_source.profile_id,
        request.source_profile.profile_id,
    }
    assert {profile.profile_id for profile in merged.approved_remote_profiles} == {
        unrelated_executor.profile_id,
        request.executor_profile.profile_id,
    }
    assert unrelated_source.profile_digest in {
        profile.profile_digest for profile in merged.approved_source_profiles
    }
    assert unrelated_executor.profile_digest in {
        profile.profile_digest for profile in merged.approved_remote_profiles
    }


def test_install_service_uses_the_merged_config_and_can_be_launch_ready(tmp_path: Path) -> None:
    definition, request = _activation_fixture(tmp_path)
    data_directory = tmp_path / "installation"
    _secrets, resolve, write = _memory_secrets()
    installed = []

    class RecordingServiceManager:
        def install(self, service_definition, config):
            installed.append((service_definition, config))
            return ()

    receipt = activate_autoresearch(
        definition,
        request,
        data_directory=data_directory,
        apply=True,
        install_service=True,
        service_target=WorkerPlatform.LINUX,
        service_home=tmp_path / "home",
        service_manager=RecordingServiceManager(),
        controller=ControllerStatusProjection(
            online=True,
            state="online",
            code="controller_online",
            observed_at=utc_now(),
        ),
        secret_resolver=resolve,
        secret_writer=write,
    )

    assert receipt.service == "installed"
    assert receipt.doctor is not None and receipt.doctor.launch_ready is True
    assert len(installed) == 1
    assert installed[0][1].approved_remote_profiles == (request.executor_profile,)
