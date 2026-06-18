"""Tests for the OpenAI-compatible provider connect route."""

from fastapi.testclient import TestClient

from bashgym.api.routes import app
from bashgym.providers.base import HealthStatus

client = TestClient(app)

_HEALTH_PATH = "bashgym.providers.openai_compatible.OpenAICompatibleProvider.health_check"


def test_presets_endpoint():
    r = client.get("/api/providers/openai-compatible/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    assert "together" in presets and presets["together"].startswith("https://")


def test_connect_requires_platform_or_base_url():
    assert client.post("/api/providers/connect", json={}).status_code == 400


def test_connect_unknown_platform_without_base_url():
    assert client.post("/api/providers/connect", json={"platform": "bogus"}).status_code == 400


def test_connect_known_platform_registers(monkeypatch):
    async def fake_health(self):
        return HealthStatus(available=True, models_loaded=["m1", "m2"])

    monkeypatch.setattr(_HEALTH_PATH, fake_health)
    r = client.post(
        "/api/providers/connect",
        json={"platform": "together", "api_key": "k", "default_model": "m1"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["provider_type"] == "together"
    assert data["available"] is True
    assert "m1" in data["models"]
    # registered into the live registry
    assert app.state.provider_registry.get_provider("together") is not None


def test_connect_custom_base_url(monkeypatch):
    async def fake_health(self):
        return HealthStatus(available=False, error="down")

    monkeypatch.setattr(_HEALTH_PATH, fake_health)
    r = client.post(
        "/api/providers/connect",
        json={"base_url": "https://host/v1", "name": "myhost", "api_key": "k"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True and data["provider_type"] == "myhost"
