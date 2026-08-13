"""API startup work that must not delay or outlive the HTTP service."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from bashgym.config import state_root_digest


def test_configured_bashgym_directory_is_the_single_headless_state_root(
    tmp_path: Path, monkeypatch
) -> None:
    from bashgym.config import get_bashgym_dir

    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path / "state"))

    assert get_bashgym_dir() == (tmp_path / "state").resolve()


def _headless_app(tmp_path: Path, monkeypatch, build_index):
    from bashgym.api import routes
    from bashgym.api.trace_cache import TraceIndexCache
    from bashgym.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "mode", "headless")
    monkeypatch.setattr(settings, "campaigns_enabled", False)
    monkeypatch.setattr(settings.ollama, "enabled", False)
    monkeypatch.setattr(settings.data, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr("bashgym.config.get_bashgym_dir", lambda: tmp_path / "home")
    monkeypatch.setattr(routes, "start_pipeline_watcher", lambda _state: None)
    monkeypatch.setattr(routes, "stop_pipeline_watcher", lambda: None)
    monkeypatch.setattr(routes, "list_run_states", lambda: [])
    monkeypatch.setattr(TraceIndexCache, "build_index", build_index)
    return routes.create_app()


def test_health_is_available_while_trace_index_is_still_building(
    tmp_path: Path, monkeypatch
) -> None:
    """Removing background scheduling would make startup wait on the blocked scan."""

    index_started = threading.Event()
    release_index = threading.Event()
    health_returned = threading.Event()
    client_exited = threading.Event()
    result: dict[str, object] = {}

    def blocking_build_index(_cache, **_kwargs) -> None:
        index_started.set()
        release_index.wait(timeout=10)

    app = _headless_app(tmp_path, monkeypatch, blocking_build_index)

    def use_client() -> None:
        try:
            with TestClient(app) as client:
                response = client.get("/api/health")
                result["status_code"] = response.status_code
                result["body"] = response.json()
                health_returned.set()
        finally:
            client_exited.set()

    client_thread = threading.Thread(target=use_client, daemon=True)
    client_thread.start()
    try:
        assert index_started.wait(timeout=10), "trace indexing never started"
        assert health_returned.wait(timeout=1), "trace indexing blocked API health"
        assert result["status_code"] == 200
        assert result["body"]["status"] == "healthy"
        assert result["body"]["state_root_digest"] == state_root_digest(tmp_path / "home")
        assert str(tmp_path / "home") not in str(result["body"])
        assert client_exited.wait(timeout=1), "shutdown waited on the background scan"
    finally:
        release_index.set()
        client_thread.join(timeout=10)

    assert not client_thread.is_alive()


def test_trace_index_failure_is_consumed_and_logged(tmp_path: Path, monkeypatch, caplog) -> None:
    """Removing the task wrapper would leave an unhandled background exception."""

    failed = threading.Event()

    def failing_build_index(_cache, **_kwargs) -> None:
        failed.set()
        raise RuntimeError("trace fixture failed")

    app = _headless_app(tmp_path, monkeypatch, failing_build_index)
    caplog.set_level(logging.ERROR, logger="bashgym.api.routes")

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert failed.wait(timeout=5)
        deadline = time.monotonic() + 5
        task = app.state.trace_index_task
        while not task.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert task.done()
        assert task.exception() is None

    assert "Failed to build trace index" in caplog.text
