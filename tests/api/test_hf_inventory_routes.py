from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bashgym.api import hf_routes
from bashgym.api.routes import app


class DummyHFClient:
    def __init__(
        self,
        enabled: bool = True,
        pro: bool = True,
        username: str = "cade",
        namespace: str = "cade",
        token: str = "hf_secret_token_123",
    ):
        self.is_enabled = enabled
        self.is_pro = pro
        self.username = username
        self.namespace = namespace
        self.token = token
        self._token = token


@pytest.fixture(autouse=True)
def clear_hf_inventory_cache():
    app.state.hf_inventory_cache = {}
    yield
    app.state.hf_inventory_cache = {}


@pytest.fixture
def client():
    return TestClient(app)


def install_inventory_fakes(
    monkeypatch, calls: dict[str, int], *, model_error: Exception | None = None
):
    dummy = DummyHFClient()
    monkeypatch.setenv("HF_TOKEN", dummy.token)
    monkeypatch.setattr("bashgym.integrations.huggingface.get_hf_client", lambda: dummy)
    monkeypatch.setattr(
        hf_routes,
        "_local_model_links",
        lambda: {
            "cade/bashgym-model-a": {
                "model_id": "run-123",
                "display_name": "BashGym Model A",
                "training_strategy": "sft",
                "base_model": "Qwen/Qwen3-Embedding-0.6B",
                "status": "ready",
            }
        },
    )

    class FakeModelManager:
        def __init__(self, hf_client):
            assert hf_client is dummy

        def list_my_models(self, limit=50):
            calls["models"] = calls.get("models", 0) + 1
            if model_error:
                raise model_error
            return [
                SimpleNamespace(
                    id="cade/bashgym-model-a",
                    url="",
                    downloads=12,
                    likes=3,
                    private=True,
                    last_modified="2026-07-09T00:00:00+00:00",
                    pipeline_tag="feature-extraction",
                    tags=["bashgym", "embedding", "extra"],
                )
            ]

    class FakeDatasetManager:
        def __init__(self, client=None):
            assert client is dummy

        def list_datasets(self, prefix="bashgym"):
            calls["datasets"] = calls.get("datasets", 0) + 1
            assert prefix == "bashgym"
            return ["cade/bashgym-youtube-retrieval"]

    class FakeTraceUploader:
        def __init__(self, token=None):
            assert token == dummy.token

        def list_trace_datasets(self, prefix="bashgym"):
            calls["trace_datasets"] = calls.get("trace_datasets", 0) + 1
            return [
                {
                    "id": "cade/bashgym-gold-traces",
                    "private": True,
                    "downloads": 4,
                    "last_modified": "2026-07-08T00:00:00+00:00",
                }
            ]

    class FakeBucketManager:
        def __init__(self, token=None):
            assert token == dummy.token

        def list_buckets(self, namespace=None):
            calls["buckets"] = calls.get("buckets", 0) + 1
            assert namespace == "cade"
            return [
                {
                    "id": "cade/bashgym-checkpoints",
                    "private": True,
                    "created_at": "2026-07-07T00:00:00+00:00",
                    "updated_at": "2026-07-09T00:00:00+00:00",
                }
            ]

    monkeypatch.setattr(
        "bashgym.integrations.huggingface.model_manager.HFModelManager", FakeModelManager
    )
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.datasets.HFDatasetManager", FakeDatasetManager
    )
    monkeypatch.setattr("bashgym.integrations.huggingface.traces.TraceUploader", FakeTraceUploader)
    monkeypatch.setattr("bashgym.integrations.huggingface.buckets.BucketManager", FakeBucketManager)


def test_hf_inventory_returns_normalized_snapshot_without_token(client, monkeypatch):
    calls: dict[str, int] = {}
    install_inventory_fakes(monkeypatch, calls)

    response = client.get("/api/hf/inventory", params={"limit": 2, "refresh": "true"})

    assert response.status_code == 200
    assert "hf_secret_token_123" not in response.text
    data = response.json()
    assert data["cached"] is False
    assert data["status"]["enabled"] is True
    assert data["status"]["token_source"] == "env"
    assert data["namespace"] == "cade"
    assert data["counts"] == {
        "models": 1,
        "datasets": 1,
        "trace_datasets": 1,
        "buckets": 1,
        "spaces": 0,
    }
    assert data["models"][0]["id"] == "cade/bashgym-model-a"
    assert data["models"][0]["local"]["model_id"] == "run-123"
    assert data["datasets"][0]["url"].endswith("/datasets/cade/bashgym-youtube-retrieval")
    assert data["trace_datasets"][0]["id"] == "cade/bashgym-gold-traces"
    assert data["buckets"][0]["id"] == "cade/bashgym-checkpoints"
    assert data["warnings"] == []


def test_hf_inventory_uses_server_side_cache(client, monkeypatch):
    calls: dict[str, int] = {}
    install_inventory_fakes(monkeypatch, calls)

    first = client.get("/api/hf/inventory", params={"limit": 2, "refresh": "true"})
    second = client.get("/api/hf/inventory", params={"limit": 2})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert calls == {
        "models": 1,
        "datasets": 1,
        "trace_datasets": 1,
        "buckets": 1,
    }


def test_hf_inventory_disabled_returns_empty_snapshot(client, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    dummy = DummyHFClient(enabled=False, pro=False, username="", namespace="", token="")
    monkeypatch.setattr("bashgym.integrations.huggingface.get_hf_client", lambda: dummy)
    monkeypatch.setattr("bashgym.secrets.get_secret", lambda _: None)

    response = client.get("/api/hf/inventory")

    assert response.status_code == 200
    data = response.json()
    assert data["status"]["enabled"] is False
    assert data["status"]["token_configured"] is False
    assert data["counts"] == {
        "models": 0,
        "datasets": 0,
        "trace_datasets": 0,
        "buckets": 0,
        "spaces": 0,
    }
    assert data["models"] == []
    assert data["warnings"] == []


def test_hf_inventory_section_errors_fail_soft_and_redact_tokens(client, monkeypatch):
    calls: dict[str, int] = {}
    install_inventory_fakes(
        monkeypatch,
        calls,
        model_error=RuntimeError("boom for hf_secret_token_123"),
    )

    response = client.get("/api/hf/inventory", params={"limit": 2, "refresh": "true"})

    assert response.status_code == 200
    assert "hf_secret_token_123" not in response.text
    data = response.json()
    assert data["counts"]["models"] == 0
    assert data["counts"]["datasets"] == 1
    assert data["warnings"][0]["section"] == "models"
    assert data["warnings"][0]["message"] == "boom for hf_***"
