"""Regression coverage for the removed in-memory AutoResearch API."""

from fastapi.testclient import TestClient


def test_legacy_flag_cannot_register_a_second_autoresearch_product(monkeypatch):
    monkeypatch.setenv("BASHGYM_ENABLE_LEGACY_AUTORESEARCH", "1")

    from bashgym.api.routes import create_app

    client = TestClient(create_app())
    assert client.get("/api/autoresearch/status").status_code == 404
