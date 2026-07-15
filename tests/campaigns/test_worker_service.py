"""Resident-worker service definitions, safety, and restart lifecycle tests."""

from __future__ import annotations

import hashlib
import os
import plistlib
import threading
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.persistence import CampaignRepository
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
    RemoteCapacityPolicy,
)
from bashgym.campaigns.worker import scheduler_lease_key
from bashgym.campaigns.worker_service import (
    CONTROLLER_OFFLINE_GUIDANCE,
    CONTROLLER_STALE_GUIDANCE,
    CommandResult,
    DesktopWorkerSupervisor,
    WorkerLifecycleStatus,
    WorkerPlatform,
    WorkerRunConfig,
    WorkerServiceError,
    WorkerServiceManager,
    build_service_definition,
    build_worker,
    ensure_worker_bootstrap,
    load_approved_remote_profiles,
    project_controller_status,
    read_worker_config,
    run_foreground,
    write_worker_config,
)

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approved_profile(
    tmp_path: Path,
    *,
    profile_id: str = "retrieval-ssh-v1",
    compute_profile_id: str = "ssh-gpu-lab",
    target_contract_key: str = "memexai-embedding-v1",
    host: str = "192.0.2.10",
) -> ApprovedRemoteExecutorProfile:
    script = tmp_path / f"{profile_id}-train.py"
    dataset = tmp_path / f"{profile_id}-train.jsonl"
    key = tmp_path / "campaign-worker-key"
    script.write_text("print('approved training')\n", encoding="utf-8")
    dataset.write_text('{"query":"hello"}\n', encoding="utf-8")
    key.write_text("test-only-private-key\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.FULL_TRAINING,
        script_path=script,
        script_sha256=file_sha256(script),
        input_files=(dataset,),
        input_sha256={dataset.name: file_sha256(dataset)},
        script_args=("--grouped-jsonl", dataset.name, "--output-dir", "."),
        budget_unit="gpu_hours",
        budget_reservation=1.5,
        capacity_policy=RemoteCapacityPolicy(
            minimum_available_memory_gib=48,
            minimum_available_disk_gib=50,
            maximum_external_gpu_processes=0,
        ),
    )
    return ApprovedRemoteExecutorProfile(
        profile_id=profile_id,
        profile_revision=1,
        compute_profile_id=compute_profile_id,
        target_contract_key=target_contract_key,
        target_model_digest="a" * 64,
        host=host,
        username="trainer",
        port=22,
        key_path=str(key),
        remote_work_dir="~/bashgym-training",
        stages=(stage,),
    )


def config_for(tmp_path: Path, *, compute_profile_ids: tuple[str, ...] = ()) -> WorkerRunConfig:
    return WorkerRunConfig.for_data_directory(
        tmp_path / "data root", compute_profile_ids=compute_profile_ids
    )


@pytest.mark.parametrize(
    ("target", "restart_marker"),
    [
        (WorkerPlatform.WINDOWS, b"RestartOnFailure"),
        (WorkerPlatform.LINUX, b"Restart=on-failure"),
        (WorkerPlatform.DARWIN, b"KeepAlive"),
    ],
)
def test_service_definitions_are_user_scoped_typed_and_restartable(
    tmp_path: Path, target: WorkerPlatform, restart_marker: bytes
) -> None:
    config_path = tmp_path / "config with spaces;and&symbols.json"
    executable = tmp_path / "runtime with spaces" / "python"
    definition = build_service_definition(
        config_path,
        target=target,
        home=tmp_path / "home",
        executable=executable,
        username="test-user",
        uid=501,
    )

    assert definition.launch_argv == (
        str(executable.resolve()),
        "-m",
        "bashgym.campaigns.worker_service",
        "run",
        "--config",
        str(config_path.resolve()),
    )
    assert restart_marker in definition.definition_payload
    assert all(isinstance(argv, tuple) for argv in definition.install_argvs)
    assert all(argv[0] not in {"sh", "bash", "cmd.exe", "powershell.exe"} for argv in definition.install_argvs)

    if target is WorkerPlatform.WINDOWS:
        root = ET.fromstring(definition.definition_payload)
        namespace = {"task": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
        assert root.findtext(".//task:Hidden", namespaces=namespace) == "true"
        assert root.findtext(".//task:LogonType", namespaces=namespace) == "InteractiveToken"
        assert root.findtext(".//task:RunLevel", namespaces=namespace) == "LeastPrivilege"
    elif target is WorkerPlatform.LINUX:
        text = definition.definition_payload.decode("utf-8")
        assert "ExecStart=\"" in text
        assert ";and&symbols.json\"" in text
        assert "RestartSec=5" in text
    else:
        payload = plistlib.loads(definition.definition_payload)
        assert tuple(payload["ProgramArguments"]) == definition.launch_argv
        assert payload["RunAtLoad"] is True
        assert payload["KeepAlive"] == {"SuccessfulExit": False}
        assert payload["ProcessType"] == "Background"


def test_service_manager_uses_argv_and_preserves_config_on_uninstall(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config_path = tmp_path / "worker-config.json"
    definition = build_service_definition(
        config_path,
        target=WorkerPlatform.LINUX,
        home=tmp_path / "home",
        executable=tmp_path / "python",
    )
    invocations: list[tuple[str, ...]] = []

    def runner(argv) -> CommandResult:
        assert not isinstance(argv, str)
        invocations.append(tuple(argv))
        return CommandResult(0, stdout="active user=operator command=secret-bearing-path")

    manager = WorkerServiceManager(runner)
    manager.install(definition, config)
    assert read_worker_config(config_path) == config
    assert definition.definition_path.is_file()
    if os.name != "nt":
        assert config_path.stat().st_mode & 0o777 == 0o600

    status = manager.status(
        definition, project_controller_status(None, config.data_directory, now=NOW)
    )
    assert status["installed"] is True
    assert status["supervisor_state"] == "available"
    assert "supervisor_output" not in status
    assert "secret-bearing-path" not in str(status)
    assert status["controller"]["code"] == "controller_offline"
    manager.uninstall(definition)

    assert config_path.is_file(), "uninstall preserves operator config and evidence"
    assert not definition.definition_path.exists()
    assert invocations == [
        ("systemctl", "--user", "daemon-reload"),
        ("systemctl", "--user", "enable", "--now", "bashgym-campaign-worker.service"),
        (
            "systemctl",
            "--user",
            "show",
            "bashgym-campaign-worker.service",
            "--no-pager",
            "--property=ActiveState,SubState,MainPID,NRestarts,ExecMainStatus",
        ),
        ("systemctl", "--user", "disable", "--now", "bashgym-campaign-worker.service"),
        ("systemctl", "--user", "daemon-reload"),
    ]


def test_controller_projection_distinguishes_absent_current_and_stale_leases(
    tmp_path: Path,
) -> None:
    repository = CampaignRepository(tmp_path / "campaign.sqlite3")
    repository.initialize()
    data_directory = tmp_path / "data"

    absent = project_controller_status(repository, data_directory, now=NOW)
    assert absent.online is False
    assert absent.state == "offline"
    assert absent.code == "controller_offline"
    assert absent.observed_at == NOW
    assert absent.heartbeat_age_seconds is None
    assert absent.guidance == CONTROLLER_OFFLINE_GUIDANCE

    lease = repository.acquire_lease(
        scheduler_lease_key(data_directory),
        "worker-a",
        ttl=timedelta(seconds=15),
        now=NOW,
    )
    current = project_controller_status(repository, data_directory, now=NOW + timedelta(seconds=5))
    assert current.online is True
    assert current.state == "online"
    assert current.code == "controller_online"
    assert current.generation == lease.generation
    assert current.heartbeat_age_seconds == 5
    assert current.guidance is None

    stale = project_controller_status(
        repository, data_directory, now=NOW + timedelta(seconds=16)
    )
    assert stale.online is False
    assert stale.state == "stale"
    assert stale.code == "controller_stale"
    assert stale.heartbeat_age_seconds == 16
    assert stale.guidance == CONTROLLER_STALE_GUIDANCE


class FakeWorker:
    def __init__(self, *, crash: bool):
        self.worker_id = "resident-worker"
        self.crash = crash
        self.stop_requested = False
        self.intervals: tuple[float, float, float] | None = None

    def request_stop(self) -> None:
        self.stop_requested = True

    def run_forever(
        self,
        *,
        heartbeat_seconds: float,
        ready_poll_seconds: float,
        idle_poll_seconds: float,
    ) -> None:
        self.intervals = (heartbeat_seconds, ready_poll_seconds, idle_poll_seconds)
        if self.crash:
            raise RuntimeError("sensitive failure details must not be persisted")


def test_foreground_crash_is_restartable_and_next_run_records_recovery(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    crashed = FakeWorker(crash=True)
    with pytest.raises(RuntimeError, match="sensitive failure"):
        run_foreground(
            config,
            worker_factory=lambda _config: crashed,
            install_signal_handlers=False,
        )
    first_status = WorkerLifecycleStatus.model_validate_json(config.status_path.read_text())
    assert first_status.state == "crashed"
    assert first_status.restart_count == 0
    assert first_status.last_error_code == "RuntimeError"
    assert "sensitive failure" not in config.status_path.read_text()

    recovered = FakeWorker(crash=False)
    run_foreground(
        config,
        worker_factory=lambda _config: recovered,
        install_signal_handlers=False,
    )
    second_status = WorkerLifecycleStatus.model_validate_json(config.status_path.read_text())
    assert second_status.state == "stopped"
    assert second_status.restart_count == 1
    assert second_status.last_error_code is None
    assert recovered.intervals == (5.0, 2.0, 30.0)


def test_desktop_bootstrap_creates_idempotent_config_and_seal_material(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "managed data"
    stored_secrets: dict[str, str] = {}

    first = ensure_worker_bootstrap(
        data_directory,
        secret_resolver=stored_secrets.get,
        secret_writer=stored_secrets.__setitem__,
        key_factory=lambda: "generated-seal-material" * 2,
    )

    assert first.config_created is True
    assert first.seal_key_created is True
    assert first.config_path == data_directory.resolve() / "campaigns" / "worker-config.v1.json"
    assert read_worker_config(first.config_path) == first.config
    assert stored_secrets == {first.config.seal_key_ref: "generated-seal-material" * 2}
    assert "generated-seal-material" not in first.config_path.read_text(encoding="utf-8")
    original_config = first.config_path.read_bytes()

    second = ensure_worker_bootstrap(
        data_directory,
        secret_resolver=stored_secrets.get,
        secret_writer=stored_secrets.__setitem__,
        key_factory=lambda: "must-not-replace-existing-material",
    )

    assert second.config == first.config
    assert second.config_created is False
    assert second.seal_key_created is False
    assert second.config_path.read_bytes() == original_config
    assert stored_secrets == {first.config.seal_key_ref: "generated-seal-material" * 2}


def test_desktop_bootstrap_rejects_config_for_another_data_directory(tmp_path: Path) -> None:
    data_directory = tmp_path / "managed"
    config_path = data_directory / "campaigns" / "worker-config.v1.json"
    write_worker_config(
        config_path,
        WorkerRunConfig.for_data_directory(tmp_path / "different-installation"),
    )

    with pytest.raises(
        WorkerServiceError,
        match="campaign_worker_config_data_directory_mismatch",
    ):
        ensure_worker_bootstrap(
            data_directory,
            secret_resolver=lambda _reference: "existing-material",
            secret_writer=lambda _reference, _value: None,
        )


def test_desktop_supervisor_starts_once_becomes_ready_and_releases_lease(
    tmp_path: Path,
) -> None:
    config = WorkerRunConfig.model_validate(
        {
            **WorkerRunConfig.for_data_directory(tmp_path / "managed").model_dump(),
            "leader_ttl_seconds": 0.3,
            "action_ttl_seconds": 0.3,
            "heartbeat_seconds": 0.02,
            "ready_poll_seconds": 0.02,
            "idle_poll_seconds": 0.05,
        }
    )
    supervisor = DesktopWorkerSupervisor(
        config,
        worker_factory=lambda value: build_worker(
            value,
            secret_resolver=lambda _reference: "s" * 32,
        ),
        restart_delay_seconds=0.01,
    )
    try:
        assert supervisor.start() is True
        assert supervisor.start() is False
        assert supervisor.wait_until_ready(timeout_seconds=2, poll_seconds=0.01) is True
        status = supervisor.status()
        assert status.managed is True
        assert status.state == "online"
        assert status.code == "worker_online"
        assert status.thread_alive is True
    finally:
        assert supervisor.stop(timeout_seconds=2) is True

    repository = CampaignRepository(config.database_path)
    repository.initialize()
    released = repository.get_lease(scheduler_lease_key(config.data_directory))
    assert released is not None
    assert released.expires_at <= datetime.now(UTC)
    stopped = supervisor.status(repository=repository)
    assert stopped.state == "stopped"
    assert stopped.thread_alive is False


def test_desktop_supervisor_restarts_a_crashed_worker_and_stops_replacement(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    replacement_started = threading.Event()
    replacement_stopped = threading.Event()
    created: list[FakeWorker] = []

    class BlockingWorker(FakeWorker):
        def run_forever(
            self,
            *,
            heartbeat_seconds: float,
            ready_poll_seconds: float,
            idle_poll_seconds: float,
        ) -> None:
            self.intervals = (heartbeat_seconds, ready_poll_seconds, idle_poll_seconds)
            replacement_started.set()
            replacement_stopped.wait(timeout=3)

        def request_stop(self) -> None:
            super().request_stop()
            replacement_stopped.set()

    def factory(_config: WorkerRunConfig) -> FakeWorker:
        worker: FakeWorker = FakeWorker(crash=True) if not created else BlockingWorker(crash=False)
        created.append(worker)
        return worker

    supervisor = DesktopWorkerSupervisor(
        config,
        worker_factory=factory,
        restart_delay_seconds=0.01,
    )
    assert supervisor.start() is True
    assert replacement_started.wait(timeout=2), "replacement worker did not start"
    assert supervisor.restart_count == 1
    assert len(created) == 2
    assert supervisor.stop(timeout_seconds=2) is True
    assert created[1].stop_requested is True
    assert supervisor.is_alive is False


def test_desktop_supervisor_stop_during_construction_stops_late_worker(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    construction_started = threading.Event()
    allow_construction = threading.Event()
    worker = FakeWorker(crash=False)

    def delayed_factory(_config: WorkerRunConfig) -> FakeWorker:
        construction_started.set()
        allow_construction.wait(timeout=2)
        return worker

    supervisor = DesktopWorkerSupervisor(
        config,
        worker_factory=delayed_factory,
        restart_delay_seconds=0.01,
    )
    assert supervisor.start() is True
    assert construction_started.wait(timeout=2)
    stop_result: list[bool] = []
    stopper = threading.Thread(
        target=lambda: stop_result.append(supervisor.stop(timeout_seconds=2))
    )
    stopper.start()
    allow_construction.set()
    stopper.join(timeout=3)

    assert stop_result == [True]
    assert worker.stop_requested is True
    assert supervisor.is_alive is False


def test_config_rejects_unsafe_boundaries_and_secret_values(tmp_path: Path) -> None:
    root = (tmp_path / "data").resolve()
    with pytest.raises(ValidationError, match="inside data_directory"):
        WorkerRunConfig(
            data_directory=root,
            database_path=tmp_path / "outside.sqlite3",
            artifact_root=root / "artifacts",
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        config_for(tmp_path, compute_profile_ids=("z", "a", "a"))
    with pytest.raises(ValidationError, match="secret reference names"):
        WorkerRunConfig(
            data_directory=root,
            database_path=root / "campaign.sqlite3",
            artifact_root=root / "artifacts",
            seal_key_ref="raw-secret-value!",
        )


def test_protected_profile_round_trips_and_is_the_only_adapter_authority(tmp_path: Path) -> None:
    profile = approved_profile(tmp_path)
    config = WorkerRunConfig.for_data_directory(
        tmp_path / "data",
        approved_remote_profiles=(profile,),
        # A legacy ID remains loadable but does not create an adapter.
        compute_profile_ids=("actor-mutated-device",),
    )
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    loaded = read_worker_config(config_path)

    registry = load_approved_remote_profiles(loaded)
    assert registry[("ssh-gpu-lab", "memexai-embedding-v1")].profile_digest == (
        profile.profile_digest
    )
    with pytest.raises(ValidationError, match="profile digest mismatch"):
        ApprovedRemoteExecutorProfile(
            **profile.model_dump(exclude={"profile_digest"}),
            profile_digest="0" * 64,
        )
    worker = build_worker(loaded, secret_resolver=lambda _reference: "s" * 32)
    assert set(worker.remote_adapters) == {"ssh-gpu-lab"}
    assert worker.remote_executor_profiles == registry
    adapter = worker.remote_adapters["ssh-gpu-lab"]
    assert adapter.config.host == "192.0.2.10"
    assert adapter.config.username == "trainer"
    assert adapter.config.port == 22
    assert adapter.config.remote_work_dir == "~/bashgym-training"


def test_legacy_compute_ids_remain_parseable_but_cannot_authorize_remote_adapter(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path, compute_profile_ids=("legacy-device",))
    worker = build_worker(config, secret_resolver=lambda _reference: "s" * 32)
    assert worker.remote_adapters == {}


def test_profile_material_hash_mismatch_and_post_load_change_fail_closed(
    tmp_path: Path,
) -> None:
    profile = approved_profile(tmp_path)
    stage = profile.stages[0]
    with pytest.raises(ValidationError, match="script hash mismatch"):
        PinnedRemoteStageProfile(
            **stage.model_dump(exclude={"script_sha256"}),
            script_sha256="0" * 64,
        )

    config = WorkerRunConfig.for_data_directory(
        tmp_path / "data", approved_remote_profiles=(profile,)
    )
    stage.script_path.write_text("print('changed after approval')\n", encoding="utf-8")
    with pytest.raises(
        WorkerServiceError, match="campaign_worker_remote_profile_material_invalid"
    ):
        load_approved_remote_profiles(config)


def test_profile_rejects_missing_and_symlinked_launch_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.py"
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="regular non-symlink"):
        PinnedRemoteStageProfile(
            stage=StageKind.SMOKE_TRAINING,
            script_path=missing,
            script_sha256="0" * 64,
            input_files=(dataset,),
            input_sha256={dataset.name: file_sha256(dataset)},
            budget_reservation=0.1,
        )

    target = tmp_path / "real-train.py"
    target.write_text("print('real')\n", encoding="utf-8")
    link = tmp_path / "linked-train.py"
    try:
        link.symlink_to(target)
    except OSError:
        link.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda candidate: candidate == link or original_is_symlink(candidate),
        )
    with pytest.raises(ValidationError, match="regular non-symlink"):
        PinnedRemoteStageProfile(
            stage=StageKind.SMOKE_TRAINING,
            script_path=link,
            script_sha256=file_sha256(target),
            input_files=(dataset,),
            input_sha256={dataset.name: file_sha256(dataset)},
            budget_reservation=0.1,
        )


def test_profile_rejects_raw_secrets_unsafe_outputs_and_conflicting_ssh_authority(
    tmp_path: Path,
) -> None:
    profile = approved_profile(tmp_path, target_contract_key="a-target")
    stage = profile.stages[0]
    with pytest.raises(ValidationError, match="credentials"):
        PinnedRemoteStageProfile(
            **stage.model_dump(exclude={"script_args"}),
            script_args=("--api-key=raw-secret",),
        )
    with pytest.raises(ValidationError, match="inside the remote run directory"):
        PinnedRemoteStageProfile(
            **stage.model_dump(exclude={"output_paths"}),
            output_paths=("../escape",),
        )

    conflicting = approved_profile(
        tmp_path,
        profile_id="retrieval-ssh-v2",
        target_contract_key="b-target",
        host="redirected.invalid",
    )
    config = WorkerRunConfig.for_data_directory(
        tmp_path / "data", approved_remote_profiles=(profile, conflicting)
    )
    with pytest.raises(
        WorkerServiceError, match="campaign_worker_compute_profile_authority_conflict"
    ):
        build_worker(config, secret_resolver=lambda _reference: "s" * 32)


def test_config_reader_rejects_symlinks_and_oversize_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    regular = tmp_path / "regular.json"
    regular.write_text(config.model_dump_json(), encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(regular)
    except OSError:
        link.write_text(config.model_dump_json(), encoding="utf-8")
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda candidate: candidate == link or original_is_symlink(candidate),
        )
    with pytest.raises(WorkerServiceError, match="campaign_worker_config_not_regular"):
        read_worker_config(link)

    oversize = tmp_path / "oversize.json"
    oversize.write_bytes(b"{" + b" " * (64 * 1024) + b"}")
    with pytest.raises(WorkerServiceError, match="campaign_worker_config_too_large"):
        read_worker_config(oversize)
