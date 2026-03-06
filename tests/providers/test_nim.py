"""Tests for NIMProvider implementation."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from bashgym.providers.base import (
    InferenceProvider,
    ProviderResponse,
    HealthStatus,
    ProviderModel,
)
from bashgym.providers.nim import NIMProvider


# ── Property tests ─────────────────────────────────────────────────


class TestNIMProviderProperties:
    """Tests for NIMProvider static properties."""

    def test_provider_type(self):
        provider = NIMProvider(api_key="test-key")
        assert provider.provider_type == "nvidia_nim"

    def test_requires_api_key(self):
        provider = NIMProvider(api_key="test-key")
        assert provider.requires_api_key is True

    def test_is_local(self):
        provider = NIMProvider(api_key="test-key")
        assert provider.is_local is False

    def test_implements_inference_provider(self):
        provider = NIMProvider(api_key="test-key")
        assert isinstance(provider, InferenceProvider)

    def test_default_endpoint(self):
        provider = NIMProvider(api_key="test-key")
        assert provider._endpoint == "https://integrate.api.nvidia.com/v1"

    def test_custom_endpoint(self):
        provider = NIMProvider(api_key="test-key", endpoint="https://custom.api.com/v1")
        assert provider._endpoint == "https://custom.api.com/v1"

    def test_default_model(self):
        provider = NIMProvider(api_key="test-key")
        assert provider._default_model == "qwen/qwen2.5-coder-7b-instruct"

    def test_custom_default_model(self):
        provider = NIMProvider(api_key="test-key", default_model="meta/llama-3.1-8b-instruct")
        assert provider._default_model == "meta/llama-3.1-8b-instruct"


# ── Generate tests ─────────────────────────────────────────────────


class TestNIMProviderGenerate:
    """Tests for NIMProvider.generate()."""

    async def test_generate_success(self):
        """Successful generation returns ProviderResponse with content."""
        provider = NIMProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from NIM!",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
                model="qwen/qwen2.5-coder-7b-instruct",
            )

        assert isinstance(result, ProviderResponse)
        assert result.success is True
        assert result.content == "Hello from NIM!"
        assert result.model_name == "qwen/qwen2.5-coder-7b-instruct"
        assert result.provider_type == "nvidia_nim"
        assert result.tokens_used == 5
        assert result.error is None
        assert result.latency_ms >= 0

    async def test_generate_uses_default_model(self):
        """When model is None, uses default model."""
        provider = NIMProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}],
            "usage": {"completion_tokens": 3},
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await provider.generate(
                messages=[{"role": "user", "content": "test"}],
            )

        assert result.model_name == "qwen/qwen2.5-coder-7b-instruct"
        # Verify the payload used the default model
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "qwen/qwen2.5-coder-7b-instruct"

    async def test_generate_prepends_system_prompt(self):
        """System prompt is prepended when not already in messages."""
        provider = NIMProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1},
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                system_prompt="You are a coding assistant.",
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a coding assistant."
        assert payload["messages"][1]["role"] == "user"

    async def test_generate_skips_system_prompt_when_already_present(self):
        """System prompt is not prepended if messages already have a system message."""
        provider = NIMProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1},
        }

        messages = [
            {"role": "system", "content": "Existing system prompt"},
            {"role": "user", "content": "hi"},
        ]

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(
                messages=messages,
                system_prompt="This should be ignored.",
            )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        # Should only have the original messages, no extra system prepended
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["content"] == "Existing system prompt"

    async def test_generate_failure_returns_error_response(self):
        """Non-200 response returns ProviderResponse with success=False."""
        provider = NIMProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
                model="qwen/qwen2.5-coder-7b-instruct",
            )

        assert isinstance(result, ProviderResponse)
        assert result.success is False
        assert result.content == ""
        assert result.error is not None
        assert "500" in result.error

    async def test_generate_handles_exception(self):
        """Network exceptions return ProviderResponse with success=False."""
        provider = NIMProvider(api_key="test-key")

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Connection refused")

            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert isinstance(result, ProviderResponse)
        assert result.success is False
        assert "Connection refused" in result.error

    async def test_generate_sends_correct_headers(self):
        """Verify auth and content-type headers are sent."""
        provider = NIMProvider(api_key="my-secret-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1},
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(
                messages=[{"role": "user", "content": "test"}],
            )

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer my-secret-key"
        assert headers["Content-Type"] == "application/json"

    async def test_generate_calls_correct_endpoint(self):
        """Verify the chat completions endpoint is called."""
        provider = NIMProvider(api_key="test-key", endpoint="https://my-api.com/v1")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"completion_tokens": 1},
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await provider.generate(
                messages=[{"role": "user", "content": "test"}],
            )

        call_args = mock_post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
        assert url == "https://my-api.com/v1/chat/completions"


# ── Health check tests ─────────────────────────────────────────────


class TestNIMProviderHealthCheck:
    """Tests for NIMProvider.health_check()."""

    async def test_health_check_success(self):
        """Successful health check returns available=True."""
        provider = NIMProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen/qwen2.5-coder-7b-instruct"},
                {"id": "meta/llama-3.1-8b-instruct"},
            ]
        }

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await provider.health_check()

        assert isinstance(result, HealthStatus)
        assert result.available is True
        assert result.latency_ms >= 0
        assert result.error is None

    async def test_health_check_failure(self):
        """Failed health check returns available=False."""
        provider = NIMProvider(api_key="test-key")

        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = await provider.health_check()

        assert isinstance(result, HealthStatus)
        assert result.available is False
        assert result.error is not None


# ── List models tests ──────────────────────────────────────────────


class TestNIMProviderListModels:
    """Tests for NIMProvider.list_models()."""

    async def test_list_models_returns_models(self):
        """list_models returns a non-empty list of ProviderModel."""
        provider = NIMProvider(api_key="test-key")
        models = await provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, ProviderModel)
            assert model.provider_type == "nvidia_nim"

    async def test_list_models_includes_qwen_coder(self):
        """list_models includes qwen coder models."""
        provider = NIMProvider(api_key="test-key")
        models = await provider.list_models()
        model_ids = [m.id for m in models]

        assert "qwen/qwen2.5-coder-7b-instruct" in model_ids
        assert "qwen/qwen2.5-coder-32b-instruct" in model_ids

    async def test_list_models_includes_llama(self):
        """list_models includes llama model."""
        provider = NIMProvider(api_key="test-key")
        models = await provider.list_models()
        model_ids = [m.id for m in models]

        assert "meta/llama-3.1-8b-instruct" in model_ids


# ── Close tests ────────────────────────────────────────────────────


class TestNIMProviderClose:
    """Tests for NIMProvider.close()."""

    async def test_close_closes_client(self):
        """close() should close the httpx client."""
        provider = NIMProvider(api_key="test-key")

        with patch.object(provider._client, "aclose", new_callable=AsyncMock) as mock_aclose:
            await provider.close()
            mock_aclose.assert_called_once()
