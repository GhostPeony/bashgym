"""Tests for InferenceProvider ABC and supporting data structures."""

import asyncio
import pytest
from abc import ABC

from bashgym.providers.base import (
    ProviderResponse,
    HealthStatus,
    ProviderModel,
    InferenceProvider,
)


# ── ProviderResponse tests ──────────────────────────────────────────


class TestProviderResponse:
    """Tests for ProviderResponse dataclass."""

    def test_success_creation(self):
        resp = ProviderResponse(
            content="Hello, world!",
            model_name="test-model",
            provider_type="ollama",
            latency_ms=42.5,
            tokens_used=10,
            success=True,
        )
        assert resp.content == "Hello, world!"
        assert resp.model_name == "test-model"
        assert resp.provider_type == "ollama"
        assert resp.latency_ms == 42.5
        assert resp.tokens_used == 10
        assert resp.success is True
        assert resp.error is None
        assert resp.metadata == {}

    def test_error_creation(self):
        resp = ProviderResponse(
            content="",
            model_name="bad-model",
            provider_type="anthropic",
            latency_ms=100.0,
            tokens_used=0,
            success=False,
            error="Model not found",
        )
        assert resp.success is False
        assert resp.error == "Model not found"
        assert resp.content == ""

    def test_to_dict(self):
        resp = ProviderResponse(
            content="output",
            model_name="m",
            provider_type="nim",
            latency_ms=1.0,
            tokens_used=5,
            success=True,
            metadata={"key": "value"},
        )
        d = resp.to_dict()
        assert isinstance(d, dict)
        assert d["content"] == "output"
        assert d["model_name"] == "m"
        assert d["provider_type"] == "nim"
        assert d["latency_ms"] == 1.0
        assert d["tokens_used"] == 5
        assert d["success"] is True
        assert d["error"] is None
        assert d["metadata"] == {"key": "value"}

    def test_metadata_default_is_independent(self):
        """Each instance should get its own metadata dict."""
        a = ProviderResponse(
            content="", model_name="", provider_type="", latency_ms=0,
            tokens_used=0, success=True,
        )
        b = ProviderResponse(
            content="", model_name="", provider_type="", latency_ms=0,
            tokens_used=0, success=True,
        )
        a.metadata["x"] = 1
        assert "x" not in b.metadata


# ── HealthStatus tests ──────────────────────────────────────────────


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_healthy(self):
        status = HealthStatus(available=True, latency_ms=15.0)
        assert status.available is True
        assert status.latency_ms == 15.0
        assert status.error is None
        assert status.models_loaded == []
        assert status.gpu_memory_used_mb is None
        assert status.gpu_memory_total_mb is None
        # last_checked should be an ISO timestamp string
        assert isinstance(status.last_checked, str)
        assert "T" in status.last_checked  # basic ISO format check

    def test_unhealthy(self):
        status = HealthStatus(available=False, error="Connection refused")
        assert status.available is False
        assert status.error == "Connection refused"
        assert status.latency_ms == 0.0

    def test_models_loaded(self):
        status = HealthStatus(
            available=True,
            latency_ms=5.0,
            models_loaded=["llama3", "codellama"],
            gpu_memory_used_mb=4096.0,
            gpu_memory_total_mb=8192.0,
        )
        assert status.models_loaded == ["llama3", "codellama"]
        assert status.gpu_memory_used_mb == 4096.0
        assert status.gpu_memory_total_mb == 8192.0

    def test_to_dict(self):
        status = HealthStatus(
            available=True,
            latency_ms=10.0,
            models_loaded=["m1"],
        )
        d = status.to_dict()
        assert isinstance(d, dict)
        assert d["available"] is True
        assert d["latency_ms"] == 10.0
        assert d["models_loaded"] == ["m1"]
        assert "last_checked" in d

    def test_models_loaded_default_is_independent(self):
        """Each instance should get its own models_loaded list."""
        a = HealthStatus(available=True)
        b = HealthStatus(available=True)
        a.models_loaded.append("x")
        assert "x" not in b.models_loaded


# ── ProviderModel tests ─────────────────────────────────────────────


class TestProviderModel:
    """Tests for ProviderModel dataclass."""

    def test_basic(self):
        model = ProviderModel(
            id="ollama/llama3",
            name="Llama 3",
            provider_type="ollama",
        )
        assert model.id == "ollama/llama3"
        assert model.name == "Llama 3"
        assert model.provider_type == "ollama"
        assert model.size_gb is None
        assert model.parameter_size is None
        assert model.is_code_model is False
        assert model.is_local is False
        assert model.context_length is None

    def test_local_code_model(self):
        model = ProviderModel(
            id="ollama/deepseek-coder",
            name="DeepSeek Coder",
            provider_type="ollama",
            size_gb=3.8,
            parameter_size="6.7B",
            is_code_model=True,
            is_local=True,
            context_length=16384,
        )
        assert model.is_code_model is True
        assert model.is_local is True
        assert model.size_gb == 3.8
        assert model.parameter_size == "6.7B"
        assert model.context_length == 16384

    def test_to_dict(self):
        model = ProviderModel(
            id="nim/qwen",
            name="Qwen",
            provider_type="nvidia_nim",
            is_code_model=True,
        )
        d = model.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "nim/qwen"
        assert d["name"] == "Qwen"
        assert d["provider_type"] == "nvidia_nim"
        assert d["is_code_model"] is True
        assert d["is_local"] is False
        assert d["size_gb"] is None


# ── InferenceProvider ABC tests ─────────────────────────────────────


class TestInferenceProvider:
    """Tests for InferenceProvider abstract base class."""

    def test_cannot_instantiate_directly(self):
        """ABC should not be instantiable."""
        with pytest.raises(TypeError):
            InferenceProvider()

    def test_incomplete_subclass_raises_type_error(self):
        """Subclass missing abstract methods should not be instantiable."""

        class IncompleteProvider(InferenceProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_partial_implementation_raises_type_error(self):
        """Subclass implementing only some abstract methods should fail."""

        class PartialProvider(InferenceProvider):
            @property
            def provider_type(self) -> str:
                return "test"

            @property
            def requires_api_key(self) -> bool:
                return False

            # Missing: is_local, generate, health_check, list_models

        with pytest.raises(TypeError):
            PartialProvider()

    def test_complete_subclass_can_instantiate(self):
        """Subclass implementing all abstract methods should work."""

        class FakeProvider(InferenceProvider):
            @property
            def provider_type(self) -> str:
                return "fake"

            @property
            def requires_api_key(self) -> bool:
                return False

            @property
            def is_local(self) -> bool:
                return True

            async def generate(self, messages, model=None, **kwargs):
                return ProviderResponse(
                    content="hi",
                    model_name=model or "fake-model",
                    provider_type="fake",
                    latency_ms=1.0,
                    tokens_used=1,
                    success=True,
                )

            async def health_check(self):
                return HealthStatus(available=True, latency_ms=1.0)

            async def list_models(self):
                return []

        provider = FakeProvider()
        assert provider.provider_type == "fake"
        assert provider.requires_api_key is False
        assert provider.is_local is True

    def test_default_warm_up_returns_true(self):
        """Default warm_up should return True."""

        class FakeProvider(InferenceProvider):
            @property
            def provider_type(self) -> str:
                return "fake"

            @property
            def requires_api_key(self) -> bool:
                return False

            @property
            def is_local(self) -> bool:
                return True

            async def generate(self, messages, model=None, **kwargs):
                return ProviderResponse(
                    content="", model_name="", provider_type="fake",
                    latency_ms=0, tokens_used=0, success=True,
                )

            async def health_check(self):
                return HealthStatus(available=True)

            async def list_models(self):
                return []

        provider = FakeProvider()
        result = asyncio.run(provider.warm_up("some-model"))
        assert result is True

    def test_default_close_is_noop(self):
        """Default close should complete without error."""

        class FakeProvider(InferenceProvider):
            @property
            def provider_type(self) -> str:
                return "fake"

            @property
            def requires_api_key(self) -> bool:
                return False

            @property
            def is_local(self) -> bool:
                return True

            async def generate(self, messages, model=None, **kwargs):
                return ProviderResponse(
                    content="", model_name="", provider_type="fake",
                    latency_ms=0, tokens_used=0, success=True,
                )

            async def health_check(self):
                return HealthStatus(available=True)

            async def list_models(self):
                return []

        provider = FakeProvider()
        # Should complete without raising
        asyncio.run(provider.close())

    def test_is_abstract_base_class(self):
        """InferenceProvider should be an ABC."""
        assert issubclass(InferenceProvider, ABC)
