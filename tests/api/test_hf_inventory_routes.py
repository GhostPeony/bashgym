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
        username: str = "example-org",
        namespace: str = "example-org",
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
    dummy.api = SimpleNamespace()
    monkeypatch.setenv("HF_TOKEN", dummy.token)
    monkeypatch.setattr("bashgym.integrations.huggingface.get_hf_client", lambda: dummy)
    monkeypatch.setattr(
        hf_routes,
        "_local_model_links",
        lambda: {
            "example-org/bashgym-model-a": {
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
                    id="example-org/bashgym-model-a",
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
            return [row["id"] for row in self.list_dataset_details(prefix)]

        def list_dataset_details(self, prefix="", limit=50):
            calls["datasets"] = calls.get("datasets", 0) + 1
            assert prefix == "bashgym"
            assert limit == 2
            return [
                {
                    "id": "example-org/bashgym-youtube-retrieval",
                    "url": "https://huggingface.co/datasets/example-org/bashgym-youtube-retrieval",
                    "private": True,
                    "downloads": 8,
                    "likes": 2,
                    "last_modified": "2026-07-09T00:00:00+00:00",
                    "created_at": "2026-07-01T00:00:00+00:00",
                    "used_storage": 2048,
                    "tags": ["bashgym", "retrieval"],
                }
            ]

    monkeypatch.setattr(
        "bashgym.integrations.huggingface.model_manager.HFModelManager", FakeModelManager
    )
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.datasets.HFDatasetManager", FakeDatasetManager
    )


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
    assert data["namespace"] == "example-org"
    assert data["counts"] == {
        "models": 1,
        "datasets": 1,
        "trace_datasets": 0,
        "buckets": 0,
        "spaces": 0,
    }
    assert data["models"][0]["id"] == "example-org/bashgym-model-a"
    assert data["models"][0]["local"]["model_id"] == "run-123"
    assert data["datasets"][0]["url"].endswith("/datasets/example-org/bashgym-youtube-retrieval")
    assert data["datasets"][0]["private"] is True
    assert data["datasets"][0]["used_storage"] == 2048
    assert data["trace_datasets"] == []
    assert data["buckets"] == []
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


@pytest.mark.asyncio
async def test_invalid_hf_replacement_does_not_overwrite_working_secret(monkeypatch):
    """Validation must happen before the submitted token reaches durable storage."""
    writes = []
    resets = []
    disabled_client = DummyHFClient(
        enabled=False,
        pro=False,
        username="",
        namespace="",
        token="",
    )
    monkeypatch.setattr(
        "bashgym.secrets.set_secret", lambda key, value: writes.append((key, value))
    )
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.get_hf_client",
        lambda **_: disabled_client,
    )
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.reset_hf_client",
        lambda: resets.append(True),
    )

    with pytest.raises(hf_routes.HTTPException) as exc_info:
        await hf_routes.configure_hf_token(
            hf_routes.HFConfigureRequest(token="hf_invalid_replacement")
        )

    assert exc_info.value.status_code == 401
    assert writes == []
    assert resets == [True]


@pytest.mark.asyncio
async def test_valid_hf_submission_persists_after_validation(monkeypatch):
    """Electron submissions validate first, then enter the shared durable store."""
    writes = []
    reloads = []
    resets = []
    enabled_client = DummyHFClient()
    monkeypatch.setattr(
        "bashgym.secrets.set_secret", lambda key, value: writes.append((key, value))
    )
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.get_hf_client",
        lambda **_: enabled_client,
    )
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.reset_hf_client",
        lambda: resets.append(True),
    )
    monkeypatch.setattr("bashgym.config.reload_settings", lambda: reloads.append(True))

    result = await hf_routes.configure_hf_token(
        hf_routes.HFConfigureRequest(token="hf_valid_replacement")
    )

    assert result["success"] is True
    assert writes == [("HF_TOKEN", "hf_valid_replacement")]
    assert reloads == [True]
    assert resets == [True]
