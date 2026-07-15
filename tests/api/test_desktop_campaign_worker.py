"""Desktop-managed campaign worker lifecycle integration tests."""

from types import SimpleNamespace

from fastapi import FastAPI

from bashgym.api.routes import (
    _start_desktop_campaign_worker,
    _stop_desktop_campaign_worker,
)
from bashgym.campaigns.worker_service import WorkerRunConfig, WorkerServiceError


class FakeSupervisor:
    def __init__(self, config: WorkerRunConfig, *, ready: bool = True):
        self.config = config
        self.ready = ready
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> bool:
        self.start_calls += 1
        return self.start_calls == 1

    def wait_until_ready(self, *, timeout_seconds: float) -> bool:
        assert timeout_seconds > 0
        return self.ready

    def stop(self, *, timeout_seconds: float) -> bool:
        assert timeout_seconds > 0
        self.stop_calls += 1
        return True


def test_desktop_backend_owns_one_worker_and_stops_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BASHGYM_DESKTOP_BOOTSTRAP_SECRET", "desktop-launch-secret")
    app = FastAPI()
    app.state.campaign_worker_supervisor = None
    app.state.campaign_worker_bootstrap_failure_code = None
    app.state.campaign_worker_managed = False
    config = WorkerRunConfig.for_data_directory(tmp_path / "managed")
    bootstrap_result = SimpleNamespace(
        config=config,
        config_path=config.data_directory / "campaigns" / "worker-config.v1.json",
    )
    created: list[FakeSupervisor] = []

    def supervisor_factory(value: WorkerRunConfig) -> FakeSupervisor:
        supervisor = FakeSupervisor(value)
        created.append(supervisor)
        return supervisor

    settings = SimpleNamespace(mode="desktop", campaigns_enabled=True)
    assert (
        _start_desktop_campaign_worker(
            app,
            settings,
            data_directory=config.data_directory,
            bootstrapper=lambda _path: bootstrap_result,
            supervisor_factory=supervisor_factory,
        )
        is True
    )
    assert (
        _start_desktop_campaign_worker(
            app,
            settings,
            data_directory=config.data_directory,
            bootstrapper=lambda _path: bootstrap_result,
            supervisor_factory=supervisor_factory,
        )
        is True
    )
    assert len(created) == 1
    assert created[0].start_calls == 1
    assert app.state.campaign_worker_managed is True
    assert app.state.campaign_worker_config_path == bootstrap_result.config_path

    assert _stop_desktop_campaign_worker(app) is True
    assert created[0].stop_calls == 1


def test_worker_bootstrap_is_scoped_to_electron_desktop_and_fails_closed(
    tmp_path, monkeypatch
) -> None:
    config = WorkerRunConfig.for_data_directory(tmp_path / "managed")
    settings = SimpleNamespace(mode="desktop", campaigns_enabled=True)
    app = FastAPI()
    app.state.campaign_worker_supervisor = None
    app.state.campaign_worker_bootstrap_failure_code = None
    app.state.campaign_worker_managed = False
    calls: list[str] = []

    assert (
        _start_desktop_campaign_worker(
            app,
            settings,
            data_directory=config.data_directory,
            bootstrapper=lambda _path: calls.append("unexpected"),
        )
        is False
    )
    assert calls == []
    assert app.state.campaign_worker_managed is False

    monkeypatch.setenv("BASHGYM_DESKTOP_BOOTSTRAP_SECRET", "desktop-launch-secret")

    def fail_bootstrap(_path):
        raise WorkerServiceError("campaign_worker_config_invalid")

    assert (
        _start_desktop_campaign_worker(
            app,
            settings,
            data_directory=config.data_directory,
            bootstrapper=fail_bootstrap,
        )
        is False
    )
    assert app.state.campaign_worker_supervisor is None
    assert app.state.campaign_worker_managed is True
    assert app.state.campaign_worker_bootstrap_failure_code == "campaign_worker_config_invalid"


def test_readiness_exception_stops_started_worker_and_keeps_backend_fail_closed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BASHGYM_DESKTOP_BOOTSTRAP_SECRET", "desktop-launch-secret")
    app = FastAPI()
    app.state.campaign_worker_supervisor = None
    app.state.campaign_worker_bootstrap_failure_code = None
    app.state.campaign_worker_managed = False
    config = WorkerRunConfig.for_data_directory(tmp_path / "managed")
    bootstrap_result = SimpleNamespace(
        config=config,
        config_path=config.data_directory / "campaigns" / "worker-config.v1.json",
    )

    class BrokenReadinessSupervisor(FakeSupervisor):
        def wait_until_ready(self, *, timeout_seconds: float) -> bool:
            raise RuntimeError("installation details must not escape")

    supervisor = BrokenReadinessSupervisor(config)
    ready = _start_desktop_campaign_worker(
        app,
        SimpleNamespace(mode="desktop", campaigns_enabled=True),
        data_directory=config.data_directory,
        bootstrapper=lambda _path: bootstrap_result,
        supervisor_factory=lambda _config: supervisor,
    )

    assert ready is False
    assert supervisor.start_calls == 1
    assert supervisor.stop_calls == 1
    assert app.state.campaign_worker_supervisor is None
    assert app.state.campaign_worker_bootstrap_failure_code == "campaign_worker_bootstrap_failed"
