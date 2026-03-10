"""Tests for OllamaProvider implementing InferenceProvider."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

from bashgym.providers.base import InferenceProvider, ProviderResponse, HealthStatus, ProviderModel
from bashgym.providers.ollama import OllamaProvider, OllamaModel


# ── Interface conformance ──────────────────────────────────────────


class TestOllamaProviderInterface:
    """OllamaProvider must implement InferenceProvider."""

    def test_inherits_from_inference_provider(self):
        assert issubclass(OllamaProvider, InferenceProvider)

    def test_can_instantiate(self):
        provider = OllamaProvider()
        assert provider is not None

    def test_provider_type(self):
        provider = OllamaProvider()
        assert provider.provider_type == "ollama"

    def test_requires_api_key_is_false(self):
        provider = OllamaProvider()
        assert provider.requires_api_key is False

    def test_is_local_default(self):
        """Default localhost URL should be local."""
        provider = OllamaProvider()
        assert provider.is_local is True

    def test_is_local_remote_url(self):
        """Remote URL should not be local."""
        provider = OllamaProvider(base_url="http://gpu-server.example.com:11434")
        assert provider.is_local is False


# ── Network security ──────────────────────────────────────────────


class TestOllamaNetworkSecurity:
    """Verify is_remote logic for network safety."""

    def test_default_is_localhost(self):
        provider = OllamaProvider()
        assert provider.is_remote is False

    def test_127_0_0_1_is_not_remote(self):
        provider = OllamaProvider(base_url="http://127.0.0.1:11434")
        assert provider.is_remote is False

    def test_ipv6_loopback_is_not_remote(self):
        provider = OllamaProvider(base_url="http://[::1]:11434")
        assert provider.is_remote is False

    def test_0_0_0_0_is_not_remote(self):
        provider = OllamaProvider(base_url="http://0.0.0.0:11434")
        assert provider.is_remote is False

    def test_external_host_is_remote(self):
        provider = OllamaProvider(base_url="http://gpu-box.local:11434")
        assert provider.is_remote is True

    def test_external_ip_is_remote(self):
        provider = OllamaProvider(base_url="http://192.168.1.100:11434")
        assert provider.is_remote is True


# ── generate (ABC method) ─────────────────────────────────────────


class TestOllamaGenerate:
    """Test the ABC generate() method that wraps chat()."""

    def test_generate_returns_provider_response_on_success(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "Hello there!"},
            "eval_count": 5,
        })

        messages = [{"role": "user", "content": "Hi"}]
        result = asyncio.run(provider.generate(messages, model="llama3"))

        assert isinstance(result, ProviderResponse)
        assert result.success is True
        assert result.content == "Hello there!"
        assert result.tokens_used == 5
        assert result.model_name == "llama3"
        assert result.provider_type == "ollama"
        assert result.latency_ms >= 0

    def test_generate_returns_error_on_chat_error(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(return_value={"error": "model not found"})

        messages = [{"role": "user", "content": "Hi"}]
        result = asyncio.run(provider.generate(messages, model="nonexistent"))

        assert isinstance(result, ProviderResponse)
        assert result.success is False
        assert result.error == "model not found"
        assert result.content == ""

    def test_generate_returns_error_on_exception(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(side_effect=ConnectionError("refused"))

        messages = [{"role": "user", "content": "Hi"}]
        result = asyncio.run(provider.generate(messages, model="llama3"))

        assert isinstance(result, ProviderResponse)
        assert result.success is False
        assert "refused" in result.error

    def test_generate_inserts_system_prompt(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "ok"},
            "eval_count": 1,
        })

        messages = [{"role": "user", "content": "Hi"}]
        asyncio.run(provider.generate(
            messages, model="llama3", system_prompt="You are helpful."
        ))

        call_args = provider.chat.call_args
        sent_messages = call_args[0][1]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "You are helpful."

    def test_generate_does_not_duplicate_system_prompt(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "ok"},
            "eval_count": 1,
        })

        messages = [
            {"role": "system", "content": "Existing system"},
            {"role": "user", "content": "Hi"},
        ]
        asyncio.run(provider.generate(
            messages, model="llama3", system_prompt="New system"
        ))

        call_args = provider.chat.call_args
        sent_messages = call_args[0][1]
        system_msgs = [m for m in sent_messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "Existing system"


# ── health_check ──────────────────────────────────────────────────


class TestOllamaHealthCheck:
    """Test health_check() implementation."""

    def test_healthy_with_loaded_models(self):
        provider = OllamaProvider()

        # Mock is_running to return True
        provider.is_running = AsyncMock(return_value=True)

        # Mock client.get for /api/ps
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "codellama:7b"},
            ]
        }
        original_client = provider.client
        provider._client = MagicMock()
        provider._client.get = AsyncMock(return_value=mock_response)

        result = asyncio.run(provider.health_check())

        assert isinstance(result, HealthStatus)
        assert result.available is True
        assert result.latency_ms >= 0
        assert "llama3:latest" in result.models_loaded
        assert "codellama:7b" in result.models_loaded

    def test_unhealthy_when_not_running(self):
        provider = OllamaProvider()
        provider.is_running = AsyncMock(return_value=False)

        result = asyncio.run(provider.health_check())

        assert isinstance(result, HealthStatus)
        assert result.available is False
        assert "not running" in result.error.lower()

    def test_unhealthy_on_exception(self):
        provider = OllamaProvider()
        provider.is_running = AsyncMock(side_effect=Exception("network down"))

        result = asyncio.run(provider.health_check())

        assert isinstance(result, HealthStatus)
        assert result.available is False
        assert "network down" in result.error


# ── warm_up ───────────────────────────────────────────────────────


class TestOllamaWarmUp:
    """Test warm_up() implementation."""

    def test_warm_up_calls_chat(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": "hi"},
        })

        result = asyncio.run(provider.warm_up("llama3"))

        assert result is True
        provider.chat.assert_called_once()
        call_args = provider.chat.call_args
        assert call_args[0][0] == "llama3"

    def test_warm_up_returns_false_on_error(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(return_value={"error": "not found"})

        result = asyncio.run(provider.warm_up("nonexistent"))

        assert result is False

    def test_warm_up_returns_false_on_exception(self):
        provider = OllamaProvider()
        provider.chat = AsyncMock(side_effect=Exception("boom"))

        result = asyncio.run(provider.warm_up("llama3"))

        assert result is False


# ── list_models (ABC) ─────────────────────────────────────────────


class TestOllamaListModels:
    """Test the ABC list_models() returning ProviderModel objects."""

    def test_list_models_returns_provider_models(self):
        provider = OllamaProvider()
        provider.list_ollama_models = AsyncMock(return_value=[
            OllamaModel(
                name="qwen2.5-coder:7b",
                size=int(4.5 * 1024**3),
                modified_at="2025-01-01",
                digest="abc123",
                details={"parameter_size": "7B", "family": "qwen"},
            ),
            OllamaModel(
                name="llama3:8b",
                size=int(5.0 * 1024**3),
                modified_at="2025-01-02",
                digest="def456",
                details={"parameter_size": "8B", "family": "llama"},
            ),
        ])

        result = asyncio.run(provider.list_models())

        assert len(result) == 2
        assert all(isinstance(m, ProviderModel) for m in result)

        qwen = result[0]
        assert qwen.id == "ollama/qwen2.5-coder:7b"
        assert qwen.name == "qwen2.5-coder:7b"
        assert qwen.provider_type == "ollama"
        assert qwen.is_local is True
        assert qwen.is_code_model is True
        assert qwen.parameter_size == "7B"
        assert qwen.size_gb == 4.5

        llama = result[1]
        assert llama.id == "ollama/llama3:8b"
        assert llama.name == "llama3:8b"
        assert llama.is_code_model is False

    def test_list_models_empty(self):
        provider = OllamaProvider()
        provider.list_ollama_models = AsyncMock(return_value=[])

        result = asyncio.run(provider.list_models())

        assert result == []


# ── Backward compatibility ────────────────────────────────────────


class TestOllamaBackwardCompat:
    """Ensure renamed methods still exist and work."""

    def test_complete_method_exists(self):
        """The old generate() should be renamed to complete()."""
        provider = OllamaProvider()
        assert hasattr(provider, "complete")
        assert callable(provider.complete)

    def test_list_ollama_models_method_exists(self):
        """The old list_models() should be renamed to list_ollama_models()."""
        provider = OllamaProvider()
        assert hasattr(provider, "list_ollama_models")
        assert callable(provider.list_ollama_models)

    def test_chat_still_exists(self):
        """chat() should remain unchanged."""
        provider = OllamaProvider()
        assert hasattr(provider, "chat")
        assert callable(provider.chat)

    def test_is_running_still_exists(self):
        """is_running() should remain unchanged."""
        provider = OllamaProvider()
        assert hasattr(provider, "is_running")
        assert callable(provider.is_running)
