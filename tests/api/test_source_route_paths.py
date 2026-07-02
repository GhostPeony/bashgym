import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from bashgym.api.routes import create_app
from bashgym.api.source_routes import (
    _resolve_source_input_path,
    _resolve_source_output_dir,
)


def test_output_dir_outside_allowed_roots_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BASHGYM_SOURCE_ROOTS", str(tmp_path))
    with pytest.raises(HTTPException) as exc:
        _resolve_source_output_dir(str(tmp_path.parent / "escape"))
    assert exc.value.status_code == 400


def test_output_dir_within_allowed_root_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("BASHGYM_SOURCE_ROOTS", str(tmp_path))
    resolved = _resolve_source_output_dir(str(tmp_path / "sub" / "records"))
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_input_path_traversal_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("BASHGYM_SOURCE_ROOTS", str(tmp_path))
    with pytest.raises(HTTPException) as exc:
        _resolve_source_input_path(str(tmp_path.parent / "etc" / "passwd"))
    assert exc.value.status_code == 400


def test_fetch_route_rejects_output_dir_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("BASHGYM_SOURCE_ROOTS", str(tmp_path))
    client = TestClient(create_app())
    resp = client.post(
        "/api/sources/ultrafeedback_binarized/fetch",
        json={"output_dir": str(tmp_path.parent / "escape")},
    )
    assert resp.status_code == 400
    assert "allowed" in resp.text.lower()
