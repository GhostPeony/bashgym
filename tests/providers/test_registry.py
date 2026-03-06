"""Tests for ProviderRegistry — central provider management."""

import asyncio
import pytest
from typing import Any, Dict, List, Optional

from bashgym.providers.base import (
    InferenceProvider,
    ProviderResponse,
    HealthStatus,
    ProviderModel,
)
from bashgym.providers.registry import ProviderRegistry, get_registry, _registry


# ── Test double ────────────────────────────────────────────────────


class FakeProvider(InferenceProvider):
    """Minimal InferenceProvider for testing."""

    def __init__(self, ptype="fake", local=True, available=True):
        self._ptype = ptype
        self._local = local
        self._available = available
        self.closed = False

    @property
    def provider_type(self) -> str:
        return self._ptype

    @property
    def requires_api_key(self) -> bool:
        return False

    @property
    def is_local(self) -> bool:
        return self._local

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        return ProviderResponse(
            content="fake",
            model_name=model or "default",
            provider_type=self._ptype,
            latency_ms=10.0,
            tokens_used=5,
            success=True,
        )

    async def health_check(self) -> HealthStatus:
        return HealthStatus(available=self._available)

    async def list_models(self) -> List[ProviderModel]:
        return [
            ProviderModel(
                id=f"{self._ptype}/model-a",
                name="model-a",
                provider_type=self._ptype,
            )
        ]

    async def close(self) -> None:
        self.closed = True


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def registry():
    """Fresh ProviderRegistry per test."""
    return ProviderRegistry()


@pytest.fixture
def fake():
    return FakeProvider()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton between tests."""
    import bashgym.providers.registry as mod

    mod._registry = None
    yield
    mod._registry = None


# ── TestProviderRegistration ───────────────────────────────────────


class TestProviderRegistration:
    def test_register(self, registry, fake):
        registry.register(fake)
        assert registry.get_provider("fake") is fake

    def test_register_replaces_existing(self, registry):
        p1 = FakeProvider(ptype="dup")
        p2 = FakeProvider(ptype="dup")
        registry.register(p1)
        registry.register(p2)
        assert registry.get_provider("dup") is p2

    def test_unregister(self, registry, fake):
        registry.register(fake)
        # Map a model to this provider, then unregister
        registry.map_model("some-model", "fake")
        registry.unregister("fake")
        assert registry.get_provider("fake") is None
        # Model mapping should be gone too
        assert registry.get_provider_for_model("some-model") is None
        # Health cache should be gone
        assert "fake" not in registry.health_cache

    def test_unregister_nonexistent_is_noop(self, registry):
        # Should not raise
        registry.unregister("nonexistent")


# ── TestModelMapping ──────────────────────────────────────────────


class TestModelMapping:
    def test_map_model(self, registry, fake):
        registry.register(fake)
        registry.map_model("my-model", "fake")
        assert registry.get_provider_for_model("my-model") is fake

    def test_unknown_model_returns_none(self, registry):
        assert registry.get_provider_for_model("no-such-model") is None

    def test_unmap_model(self, registry, fake):
        registry.register(fake)
        registry.map_model("my-model", "fake")
        registry.unmap_model("my-model")
        assert registry.get_provider_for_model("my-model") is None

    def test_unmap_model_nonexistent_is_noop(self, registry):
        # Should not raise
        registry.unmap_model("ghost-model")

    def test_map_model_raises_for_unknown_provider(self, registry):
        with pytest.raises(ValueError, match="not registered"):
            registry.map_model("m", "nonexistent")

    def test_get_model_map_returns_copy(self, registry, fake):
        registry.register(fake)
        registry.map_model("m1", "fake")
        m = registry.get_model_map()
        m["m1"] = "tampered"
        assert registry.get_model_map()["m1"] == "fake"


# ── TestProviderResolution ────────────────────────────────────────


class TestProviderResolution:
    def test_generate_via_registry(self, registry, fake):
        registry.register(fake)
        registry.map_model("my-model", "fake")
        resp = asyncio.run(
            registry.generate("my-model", [{"role": "user", "content": "hi"}])
        )
        assert resp.success is True
        assert resp.content == "fake"
        assert resp.model_name == "my-model"
        assert resp.provider_type == "fake"

    def test_generate_unknown_model_raises(self, registry):
        with pytest.raises(ValueError, match="No provider"):
            asyncio.run(
                registry.generate("missing", [{"role": "user", "content": "hi"}])
            )


# ── TestHealthMonitoring ──────────────────────────────────────────


class TestHealthMonitoring:
    def test_check_all_health(self, registry):
        p1 = FakeProvider(ptype="a", available=True)
        p2 = FakeProvider(ptype="b", available=False)
        registry.register(p1)
        registry.register(p2)
        results = asyncio.run(registry.check_all_health())
        assert results["a"].available is True
        assert results["b"].available is False

    def test_health_caches_results(self, registry, fake):
        registry.register(fake)
        asyncio.run(registry.check_all_health())
        assert "fake" in registry.health_cache
        assert registry.health_cache["fake"].available is True

    def test_health_listener_called(self, registry, fake):
        registry.register(fake)
        notifications = []
        registry.on_health_change(lambda results: notifications.append(results))
        asyncio.run(registry.check_all_health())
        assert len(notifications) == 1
        assert "fake" in notifications[0]


# ── TestAutoDiscovery ─────────────────────────────────────────────


class TestAutoDiscovery:
    def test_discover_models_auto_maps(self, registry, fake):
        registry.register(fake)
        discovered = asyncio.run(registry.discover_models())
        assert "fake" in discovered
        assert len(discovered["fake"]) == 1
        assert discovered["fake"][0].id == "fake/model-a"
        # Model should be auto-mapped
        assert registry.get_provider_for_model("fake/model-a") is fake

    def test_discover_does_not_overwrite_existing_mapping(self, registry):
        p1 = FakeProvider(ptype="primary")
        p2 = FakeProvider(ptype="secondary")
        registry.register(p1)
        registry.register(p2)
        # Pre-map one of p2's models to p1
        # p2 will discover "secondary/model-a", but we pre-map it to primary
        registry.map_model("secondary/model-a", "primary")
        asyncio.run(registry.discover_models())
        # Should still point to primary, not overwritten
        assert registry.get_provider_for_model("secondary/model-a") is p1


# ── TestStatusSummary ─────────────────────────────────────────────


class TestStatusSummary:
    def test_get_status_summary(self, registry, fake):
        registry.register(fake)
        registry.map_model("m1", "fake")
        registry.map_model("m2", "fake")
        summary = registry.get_status_summary()
        assert summary["total_models"] == 2
        assert "fake" in summary["providers"]
        assert summary["model_map"] == {"m1": "fake", "m2": "fake"}


# ── TestClose ─────────────────────────────────────────────────────


class TestClose:
    def test_close_all_providers(self, registry):
        p1 = FakeProvider(ptype="a")
        p2 = FakeProvider(ptype="b")
        registry.register(p1)
        registry.register(p2)
        asyncio.run(registry.close())
        assert p1.closed is True
        assert p2.closed is True


# ── TestSingleton ─────────────────────────────────────────────────


class TestSingleton:
    def test_get_registry_returns_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_get_registry_creates_instance(self):
        r = get_registry()
        assert isinstance(r, ProviderRegistry)
