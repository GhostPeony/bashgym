"""Read-only optional factory checks must not imply generation or recipe proof."""

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api.auth import AuthMiddleware
from bashgym.api.factory_routes import router
from bashgym.factory import data_designer, designer_pipelines


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("BASHGYM_MODE", "headless")
    monkeypatch.setenv("BASHGYM_API_KEY", "test-control-key")
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(AuthMiddleware)
    with TestClient(app) as client:
        yield client


def test_readiness_is_authenticated(client):
    assert client.get("/api/factory/designer/pipelines").status_code == 401


@pytest.mark.parametrize("missing", ["designer", "pandas", "builders"])
def test_imported_wrapper_does_not_imply_optional_dependencies(client, monkeypatch, missing):
    monkeypatch.setattr(data_designer, "DATA_DESIGNER_AVAILABLE", missing != "designer")
    monkeypatch.setattr(data_designer, "PANDAS_AVAILABLE", missing != "pandas")
    monkeypatch.setattr(designer_pipelines, "DATA_DESIGNER_AVAILABLE", missing != "builders")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    def unexpected_builder(_):
        pytest.fail("A missing dependency must stop pipeline construction")

    monkeypatch.setattr(designer_pipelines, "PIPELINES", {"example": unexpected_builder})
    response = client.get(
        "/api/factory/designer/pipelines", headers={"X-API-Key": "test-control-key"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["pipelines"] == []
    assert payload["readiness"]["credential_configured"] is False
    assert payload["readiness"]["scope"] == "backend_process_imports"
    assert datetime.fromisoformat(payload["readiness"]["checked_at"]).tzinfo is not None


def test_dependency_evidence_does_not_probe_provider_or_leak_configuration(client, monkeypatch):
    monkeypatch.setattr(data_designer, "DATA_DESIGNER_AVAILABLE", True)
    monkeypatch.setattr(data_designer, "PANDAS_AVAILABLE", True)
    monkeypatch.setattr(designer_pipelines, "DATA_DESIGNER_AVAILABLE", True)
    monkeypatch.setattr(designer_pipelines, "PIPELINES", {})
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-provider-canary")
    monkeypatch.setenv("NVIDIA_NIM_ENDPOINT", "https://private-provider.invalid")

    def unexpected_call(*args, **kwargs):
        pytest.fail("Dependency checks cannot probe or generate")

    monkeypatch.setattr(data_designer, "provider_model_ids", unexpected_call)
    monkeypatch.setattr(data_designer, "list_inference_models", unexpected_call)
    monkeypatch.setattr(data_designer, "DataDesignerPipeline", unexpected_call)
    response = client.get(
        "/api/factory/designer/pipelines", headers={"X-API-Key": "test-control-key"}
    )
    payload = response.json()
    assert payload["available"] is True
    assert payload["readiness"]["credential_configured"] is True
    assert payload["readiness"]["browser_provider"] == "nvidia"
    for key in ("provider_verified", "generation_verified", "recipe_verified"):
        assert payload["readiness"][key] is False
    assert "secret-provider-canary" not in response.text
    assert "private-provider.invalid" not in response.text
    assert "endpoint" not in payload["readiness"]
