"""Resident-worker service definitions, safety, and restart lifecycle tests."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import sys
import threading
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym._compat import UTC
from bashgym.campaigns import worker_service
from bashgym.campaigns.contracts import StageKind, canonical_hash
from bashgym.campaigns.persistence import CampaignRepository
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
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
    load_approved_source_profiles,
    project_controller_status,
    read_worker_config,
    run_foreground,
    write_worker_config,
)
from tests.campaigns.test_lineage import initialized_repository, source_profile

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
    code_lineage_binding: ApprovedCodeLineageExecutionBinding | None = None,
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
        code_lineage_binding=code_lineage_binding,
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
        (WorkerPlatform.WINDOWS, b"bashgym_background_service.v1"),
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
    if target is WorkerPlatform.WINDOWS:
        assert restart_marker in definition.definition_payload
    else:
        assert restart_marker in definition.definition_payload
    assert all(isinstance(argv, tuple) for argv in definition.install_argvs)
    assert all(
        argv[0] not in {"sh", "bash", "cmd.exe", "powershell.exe"}
        for argv in definition.install_argvs
    )

    if target is WorkerPlatform.WINDOWS:
        payload = json.loads(definition.definition_payload)
        assert payload["schema_version"] == "bashgym_background_service.v1"
        assert tuple(payload["launch_argv"]) == definition.launch_argv
        assert Path(payload["receipt_path"]).is_absolute()
        assert definition.install_argvs[0][:4] == (
            "reg.exe",
            "ADD",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
            "/V",
        )
        assert "start-background" in definition.start_argvs[0]
    elif target is WorkerPlatform.LINUX:
        text = definition.definition_payload.decode("utf-8")
        assert 'ExecStart="' in text
        assert ';and&symbols.json"' in text
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
        return CommandResult(
            0,
            stdout=(
                "ActiveState=active\nSubState=running\n" "user=operator command=secret-bearing-path"
            ),
        )

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


def test_linux_status_does_not_treat_an_inactive_unit_as_available(tmp_path: Path) -> None:
    result = CommandResult(0, stdout="ActiveState=inactive\nSubState=dead\n")

    def runner(_argv) -> CommandResult:
        return result

    worker_definition = build_service_definition(
        tmp_path / "worker-config.json",
        target=WorkerPlatform.LINUX,
        home=tmp_path / "home",
        executable=tmp_path / "python",
    )
    api_definition = worker_service.build_api_service_definition(
        target=WorkerPlatform.LINUX,
        home=tmp_path / "home",
        executable=tmp_path / "python",
    )

    worker_status = WorkerServiceManager(runner).status(
        worker_definition,
        project_controller_status(None, tmp_path, now=NOW),
    )
    api_status = worker_service.ApiServiceManager(runner).status(api_definition)

    assert worker_status["supervisor_state"] == "unavailable"
    assert api_status["supervisor_state"] == "unavailable"


def test_windows_worker_install_registers_then_starts_without_elevation(tmp_path: Path) -> None:
    """Windows residency must work for the current user without Task Scheduler rights."""

    config = config_for(tmp_path)
    definition = build_service_definition(
        tmp_path / "worker-config.json",
        target=WorkerPlatform.WINDOWS,
        home=tmp_path / "home",
        executable=tmp_path / "python.exe",
        username="test-user",
    )
    invocations: list[tuple[str, ...]] = []

    def runner(argv) -> CommandResult:
        invocations.append(tuple(argv))
        return CommandResult(0)

    WorkerServiceManager(runner).install(definition, config)

    assert invocations[0][:3] == (
        "reg.exe",
        "ADD",
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
    )
    assert "start-background" in invocations[1]


def test_windows_worker_install_fails_when_the_process_cannot_start(tmp_path: Path) -> None:
    """Ignoring background-start failure would report a worker that is not running."""

    config = config_for(tmp_path)
    definition = build_service_definition(
        tmp_path / "worker-config.json",
        target=WorkerPlatform.WINDOWS,
        home=tmp_path / "home",
        executable=tmp_path / "python.exe",
        username="test-user",
    )

    def runner(argv) -> CommandResult:
        return CommandResult(1 if "start-background" in argv else 0)

    with pytest.raises(WorkerServiceError, match="campaign_worker_service_install_failed"):
        WorkerServiceManager(runner).install(definition, config)


def test_headless_api_service_has_complete_per_user_lifecycle(tmp_path: Path) -> None:
    """Removing a lifecycle command would make API residency require manual process work."""

    definition = worker_service.build_api_service_definition(
        target=WorkerPlatform.WINDOWS,
        home=tmp_path / "home",
        executable=tmp_path / "python.exe",
        username="test-user",
        host="127.0.0.1",
        port=8123,
        data_directory=tmp_path / "data",
    )
    invocations: list[tuple[str, ...]] = []

    def runner(argv) -> CommandResult:
        invocations.append(tuple(argv))
        return CommandResult(0, stdout="user=operator executable=private-path")

    manager = worker_service.ApiServiceManager(runner)
    manager.install(definition)
    status = manager.status(definition)
    manager.stop(definition)
    manager.start(definition)
    manager.uninstall(definition)

    assert definition.launch_argv == (
        str((tmp_path / "python.exe").resolve()),
        "-m",
        "bashgym.campaigns.worker_service",
        "run-api",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--data-dir",
        str((tmp_path / "data").resolve()),
    )
    assert status == {
        "schema_version": "bashgym_api_service_status.v1",
        "installed": True,
        "platform": "windows",
        "supervisor_returncode": 0,
        "supervisor_state": "available",
    }
    assert "private-path" not in str(status)
    assert invocations[0][0:3] == (
        "reg.exe",
        "ADD",
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
    )
    assert "start-background" in invocations[1]
    assert "status-background" in invocations[2]
    assert "stop-background" in invocations[3]
    assert "start-background" in invocations[4]
    assert "stop-background" in invocations[5]
    assert invocations[6][0:3] == (
        "reg.exe",
        "DELETE",
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
    )
    assert not definition.definition_path.exists()


def test_api_health_probe_projects_http_unavailable_without_private_details(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, float]] = []

    def unavailable(request, *, timeout: float):
        calls.append((request.full_url, timeout))
        raise urllib.error.URLError("private network details")

    status = worker_service.probe_api_health(
        expected_state_root=tmp_path / "state",
        host="127.0.0.1",
        port=8123,
        timeout_seconds=0.2,
        opener=unavailable,
    )

    assert status == {
        "schema_version": "bashgym_api_health.v1",
        "healthy": False,
        "state_root_match": False,
        "code": "api_http_unavailable",
    }
    assert calls == [("http://127.0.0.1:8123/api/health", 0.2)]
    assert "private network details" not in str(status)
    assert str(tmp_path) not in str(status)


def test_api_health_probe_rejects_non_loopback_targets(tmp_path: Path) -> None:
    with pytest.raises(WorkerServiceError, match="bashgym_api_health_probe_argument_invalid"):
        worker_service.probe_api_health(
            expected_state_root=tmp_path / "state",
            host="api.example.test",
        )


def test_api_health_probe_requires_exact_state_root_digest(tmp_path: Path) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "status": "healthy",
                    "state_root_digest": "0" * 64,
                }
            ).encode("utf-8")

    status = worker_service.probe_api_health(
        expected_state_root=tmp_path / "state",
        opener=lambda _request, *, timeout: Response(),
    )

    assert status == {
        "schema_version": "bashgym_api_health.v1",
        "healthy": True,
        "state_root_match": False,
        "code": "api_state_root_mismatch",
    }


def test_api_health_probe_accepts_healthy_matching_state_root(tmp_path: Path) -> None:
    expected_state_root = tmp_path / "state"

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "status": "healthy",
                    "state_root_digest": worker_service.state_root_digest(expected_state_root),
                }
            ).encode("utf-8")

    status = worker_service.probe_api_health(
        expected_state_root=expected_state_root,
        opener=lambda _request, *, timeout: Response(),
    )

    assert status == {
        "schema_version": "bashgym_api_health.v1",
        "healthy": True,
        "state_root_match": True,
        "code": "api_http_healthy",
    }


def test_headless_api_install_fails_when_start_fails(tmp_path: Path) -> None:
    """Ignoring a failed start would make API install report a false success."""

    definition = worker_service.build_api_service_definition(
        target=WorkerPlatform.WINDOWS,
        home=tmp_path / "home",
        executable=tmp_path / "python.exe",
        username="test-user",
    )

    def runner(argv) -> CommandResult:
        return CommandResult(1 if "start-background" in argv else 0)

    with pytest.raises(WorkerServiceError, match="bashgym_api_service_install_failed"):
        worker_service.ApiServiceManager(runner).install(definition)


def test_headless_api_replace_unloads_launchd_job_before_bootstrap(tmp_path: Path) -> None:
    definition = worker_service.build_api_service_definition(
        target=WorkerPlatform.DARWIN,
        home=tmp_path / "home",
        executable=tmp_path / "python",
        uid=501,
    )
    definition.definition_path.parent.mkdir(parents=True)
    definition.definition_path.write_bytes(b"stale-definition")
    invocations: list[tuple[str, ...]] = []

    def runner(argv) -> CommandResult:
        invocations.append(tuple(argv))
        return CommandResult(0)

    worker_service.ApiServiceManager(runner).replace(definition)

    assert invocations[0][:2] == ("launchctl", "print")
    assert invocations[1][:2] == ("launchctl", "bootout")
    assert invocations[2][:2] == ("launchctl", "bootstrap")
    assert invocations[3][:2] == ("launchctl", "kickstart")
    assert definition.definition_path.read_bytes() == definition.definition_payload


def test_background_service_helper_starts_reports_and_stops_exact_process(
    tmp_path: Path,
) -> None:
    """The per-user Windows helper owns one verified process receipt."""

    definition_path = tmp_path / "service.json"
    receipt_path = tmp_path / "service.process.json"
    definition_path.write_text(
        json.dumps(
            {
                "schema_version": "bashgym_background_service.v1",
                "launch_argv": [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(30)",
                ],
                "receipt_path": str(receipt_path),
                "stdout_path": str(tmp_path / "service.log"),
                "stderr_path": str(tmp_path / "service.error.log"),
                "restart_delay_seconds": 0.1,
            }
        ),
        encoding="utf-8",
    )

    try:
        assert worker_service.main(["start-background", "--definition", str(definition_path)]) == 0
        assert receipt_path.is_file()
        assert worker_service.main(["status-background", "--definition", str(definition_path)]) == 0
        assert worker_service.main(["stop-background", "--definition", str(definition_path)]) == 0
        assert not receipt_path.exists()
        assert worker_service.main(["status-background", "--definition", str(definition_path)]) == 1
    finally:
        worker_service.main(["stop-background", "--definition", str(definition_path)])


def test_background_service_restarts_a_crashed_child_and_prevents_duplicates(
    tmp_path: Path,
) -> None:
    """One supervisor should replace failures without spawning duplicate supervisors."""

    import time

    definition_path = tmp_path / "restarting-service.json"
    receipt_path = tmp_path / "restarting-service.process.json"
    counter_path = tmp_path / "starts.txt"
    script = tmp_path / "short-child.py"
    script.write_text(
        "from pathlib import Path\n"
        f"path = Path({str(counter_path)!r})\n"
        "value = int(path.read_text()) + 1 if path.exists() else 1\n"
        "path.write_text(str(value))\n",
        encoding="utf-8",
    )
    definition_path.write_text(
        json.dumps(
            {
                "schema_version": "bashgym_background_service.v1",
                "launch_argv": [sys.executable, str(script)],
                "receipt_path": str(receipt_path),
                "stdout_path": str(tmp_path / "restart.log"),
                "stderr_path": str(tmp_path / "restart.error.log"),
                "restart_delay_seconds": 0.1,
            }
        ),
        encoding="utf-8",
    )

    try:
        worker_service.start_background_service(definition_path)
        first = json.loads(receipt_path.read_text(encoding="utf-8"))
        worker_service.start_background_service(definition_path)
        second = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert second["supervisor_pid"] == first["supervisor_pid"]
        deadline = time.monotonic() + 5
        while (
            not counter_path.exists() or int(counter_path.read_text()) < 2
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert int(counter_path.read_text()) >= 2
    finally:
        worker_service.stop_background_service(definition_path)


def test_run_headless_api_sets_mode_and_uses_one_server_worker(monkeypatch) -> None:
    """Removing headless mode would accidentally attach desktop-owned runtime work."""

    from bashgym.api import database

    monkeypatch.delenv("BASHGYM_MODE", raising=False)
    received: dict[str, object] = {}
    database_paths: list[Path] = []
    working_directories: list[Path] = []
    monkeypatch.setattr(database, "set_db_path", database_paths.append)
    monkeypatch.setattr(os, "chdir", lambda path: working_directories.append(Path(path)))

    def run_server(app: str, **kwargs) -> None:
        received["app"] = app
        received.update(kwargs)

    worker_service.run_headless_api(
        host="127.0.0.1",
        port=8123,
        data_directory=Path("test-data"),
        server_runner=run_server,
    )

    assert os.environ["BASHGYM_MODE"] == "headless"
    assert os.environ["BASHGYM_DIR"] == str(Path("test-data").resolve())
    assert database_paths == [Path("test-data").resolve() / "api" / "bashgym.db"]
    assert working_directories == [Path("test-data").resolve()]
    assert received == {
        "app": "bashgym.api.routes:app",
        "host": "127.0.0.1",
        "port": 8123,
        "workers": 1,
        "log_level": "info",
    }


def test_worker_config_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Loosening the schema discriminator would accept an unsupported config contract."""

    payload = config_for(tmp_path).model_dump(mode="python")
    payload["schema_version"] = "campaign_worker_config.v2"

    with pytest.raises(ValidationError, match="campaign_worker_config.v1"):
        WorkerRunConfig.model_validate(payload)


def test_worker_config_can_pin_the_resident_controller_identity(tmp_path: Path) -> None:
    config = config_for(tmp_path).model_copy(
        update={"controller_owner_id": "autoresearch-resident-worker"}
    )

    worker = build_worker(
        config,
        secret_resolver=lambda _name: "test-seal-key-that-is-long-enough-123",
        adapter_loader=lambda _config: {},
    )

    assert worker.worker_id == "autoresearch-resident-worker"


@pytest.mark.parametrize(
    ("legacy_args", "expected_args"),
    [
        (("--batch-size", "4", "--output", "legacy.json"), ("--batch-size", "4")),
        (("--output=legacy.json", "--temperature", "0.1"), ("--temperature", "0.1")),
    ],
)
def test_worker_config_atomically_migrates_only_legacy_evaluator_output_args(
    tmp_path: Path,
    legacy_args: tuple[str, ...],
    expected_args: tuple[str, ...],
) -> None:
    """Removing the migration would strand otherwise valid v1 worker configs."""

    script = tmp_path / "evaluate.py"
    script.write_text("print('evaluate')\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.DEVELOPMENT_EVALUATION,
        script_path=script,
        script_sha256=file_sha256(script),
        input_files=(),
        input_sha256={},
        output_paths=("autoresearch_evaluation.json",),
        budget_reservation=0.1,
    )
    key = tmp_path / "key"
    key.write_text("test-only-private-key\n", encoding="utf-8")
    profile = ApprovedRemoteExecutorProfile(
        profile_id="evaluation-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="terminal-v1",
        target_model_digest="a" * 64,
        host="192.0.2.10",
        username="trainer",
        port=22,
        key_path=str(key),
        remote_work_dir="~/bashgym-training",
        stages=(stage,),
    )
    config = WorkerRunConfig.for_data_directory(
        tmp_path / "data", approved_remote_profiles=(profile,)
    )
    payload = config.model_dump(mode="json")
    profile_payload = payload["approved_remote_profiles"][0]
    profile_payload["stages"][0]["script_args"] = list(legacy_args)
    profile_payload["profile_digest"] = canonical_hash(
        {
            key: value
            for key, value in profile_payload.items()
            if key
            not in {
                "profile_digest",
                "nemo_rl",
                "registered_base_model",
                "registered_evaluation_dataset",
            }
        }
    )
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    loaded = read_worker_config(config_path)

    assert loaded.approved_remote_profiles[0].stages[0].script_args == expected_args
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["approved_remote_profiles"][0]["stages"][0]["script_args"] == list(
        expected_args
    )
    backup = config_path.with_name(f"{config_path.name}.pre-migration.v1.json")
    receipt = config_path.with_name(f"{config_path.name}.migration.v1.json")
    assert json.loads(backup.read_text(encoding="utf-8"))["approved_remote_profiles"][0]["stages"][
        0
    ]["script_args"] == list(legacy_args)
    migration = json.loads(receipt.read_text(encoding="utf-8"))
    assert migration["schema_version"] == "campaign_worker_config_migration.v1"
    assert migration["migration"] == "remove_legacy_evaluator_output_args"
    assert migration["before_sha256"] != migration["after_sha256"]
    assert not list(config_path.parent.glob(f".{config_path.name}.*.tmp"))


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

    stale = project_controller_status(repository, data_directory, now=NOW + timedelta(seconds=16))
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
    source_root = tmp_path / "source-fixture"
    source_root.mkdir()
    source_repository, _base_commit = initialized_repository(source_root)
    source = source_profile(source_repository)
    config = WorkerRunConfig.for_data_directory(
        tmp_path / "data",
        approved_remote_profiles=(profile,),
        approved_source_profiles=(source,),
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
    source_registry = load_approved_source_profiles(loaded)
    assert source_registry["bashgym-source-v1"].profile_digest == source.profile_digest
    with pytest.raises(ValidationError, match="profile digest mismatch"):
        ApprovedRemoteExecutorProfile(
            **profile.model_dump(exclude={"profile_digest"}),
            profile_digest="0" * 64,
        )
    worker = build_worker(loaded, secret_resolver=lambda _reference: "s" * 32)
    assert set(worker.remote_adapters) == {"ssh-gpu-lab"}
    assert worker.remote_executor_profiles == registry
    assert worker.source_repository_profiles == source_registry
    assert worker.lineage_snapshot_root == (
        config.data_directory / "campaigns" / "source-snapshots"
    )
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


def test_worker_bootstrap_cross_checks_code_lineage_entrypoint_binding(
    tmp_path: Path,
) -> None:
    binding = ApprovedCodeLineageExecutionBinding(
        binding_id="bashgym-trainer-entrypoint-v1",
        binding_revision=1,
        source_repository_profile_id="bashgym-source-v1",
        entrypoint_path="bashgym/gym/trainer.py",
    )
    profile = approved_profile(tmp_path, code_lineage_binding=binding)
    missing_source = WorkerRunConfig.for_data_directory(
        tmp_path / "missing-source-data", approved_remote_profiles=(profile,)
    )

    with pytest.raises(
        WorkerServiceError,
        match="campaign_worker_code_lineage_execution_binding_invalid",
    ):
        build_worker(missing_source, secret_resolver=lambda _reference: "s" * 32)

    source_root = tmp_path / "binding-source-fixture"
    source_root.mkdir()
    source_repository, _base_commit = initialized_repository(source_root)
    source = source_profile(source_repository)
    configured = WorkerRunConfig.for_data_directory(
        tmp_path / "configured-data",
        approved_remote_profiles=(profile,),
        approved_source_profiles=(source,),
    )

    worker = build_worker(configured, secret_resolver=lambda _reference: "s" * 32)
    assert worker.source_repository_profiles == {source.profile_id: source}


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
    with pytest.raises(WorkerServiceError, match="campaign_worker_remote_profile_material_invalid"):
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

    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(candidate: Path) -> bytes:
        if candidate in {link, oversize}:
            raise AssertionError("invalid config must be rejected before raw byte reads")
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    with pytest.raises(WorkerServiceError, match="campaign_worker_config_too_large"):
        read_worker_config(oversize)
