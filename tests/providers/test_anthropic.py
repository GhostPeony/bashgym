"""Tests for AnthropicProvider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from bashgym.providers.anthropic import AnthropicProvider, ANTHROPIC_MODELS
from bashgym.providers.base import (
    InferenceProvider,
    ProviderResponse,
    HealthStatus,
    ProviderModel,
)


# ── Property tests ─────────────────────────────────────────────────


class TestAnthropicProviderProperties:
    """Tests for static properties."""

    def test_provider_type(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        assert provider.provider_type == "anthropic"

    def test_requires_api_key(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        assert provider.requires_api_key is True

    def test_is_local(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        assert provider.is_local is False

    def test_implements_inference_provider(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        assert isinstance(provider, InferenceProvider)


# ── Constructor tests ──────────────────────────────────────────────


class TestAnthropicProviderConstructor:
    """Tests for constructor validation."""

    def test_empty_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="api_key"):
            AnthropicProvider(api_key="")

    def test_none_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="api_key"):
            AnthropicProvider(api_key=None)

    def test_valid_api_key_succeeds(self):
        provider = AnthropicProvider(api_key="sk-ant-test123")
        assert provider is not None

    def test_default_model(self):
        provider = AnthropicProvider(api_key="sk-test")
        assert provider.default_model == "claude-sonnet-4-20250514"

    def test_custom_default_model(self):
        provider = AnthropicProvider(api_key="sk-test", default_model="claude-opus-4-6")
        assert provider.default_model == "claude-opus-4-6"

    def test_custom_timeout(self):
        provider = AnthropicProvider(api_key="sk-test", timeout=60.0)
        assert provider._timeout == 60.0


# ── Generate tests ─────────────────────────────────────────────────


class TestAnthropicProviderGenerate:
    """Tests for the generate method."""

    async def test_generate_success(self):
        """Successful API call returns proper ProviderResponse."""
        provider = AnthropicProvider(api_key="sk-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_01abc",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {
                "input_tokens": 25,
                "output_tokens": 10,
            },
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate(
                messages=[{"role": "user", "content": "Say hello"}],
            )

        assert isinstance(result, ProviderResponse)
        assert result.success is True
        assert result.content == "Hello from Claude!"
        assert result.model_name == "claude-sonnet-4-20250514"
        assert result.provider_type == "anthropic"
        assert result.tokens_used == 10
        assert result.error is None
        assert result.latency_ms > 0

    async def test_generate_with_system_prompt(self):
        """System prompt is passed correctly in the payload."""
        provider = AnthropicProvider(api_key="sk-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "I am a helpful bot."}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 30, "output_tokens": 8},
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await provider.generate(
                messages=[{"role": "user", "content": "Who are you?"}],
                system_prompt="You are a helpful bot.",
            )

            # Verify the system prompt was included in the payload
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["system"] == "You are a helpful bot."

    async def test_generate_extracts_system_from_messages(self):
        """System messages in the messages list are extracted to payload['system']."""
        provider = AnthropicProvider(api_key="sk-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Got it."}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 20, "output_tokens": 5},
        }

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await provider.generate(messages=messages)

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            # System messages should be extracted, not in the messages list
            assert payload["system"] == "Be concise."
            assert all(m["role"] != "system" for m in payload["messages"])

    async def test_generate_uses_custom_model(self):
        """Model parameter overrides default_model."""
        provider = AnthropicProvider(api_key="sk-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hi"}],
            "model": "claude-opus-4-6",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            result = await provider.generate(
                messages=[{"role": "user", "content": "Hi"}],
                model="claude-opus-4-6",
            )

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["model"] == "claude-opus-4-6"
            assert result.model_name == "claude-opus-4-6"

    async def test_generate_api_error_429(self):
        """Rate limit (429) returns error ProviderResponse, does not raise."""
        provider = AnthropicProvider(api_key="sk-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert isinstance(result, ProviderResponse)
        assert result.success is False
        assert result.error is not None
        assert "429" in result.error
        assert result.content == ""
        assert result.latency_ms > 0

    async def test_generate_api_error_500(self):
        """Server error (500) returns error ProviderResponse, does not raise."""
        provider = AnthropicProvider(api_key="sk-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch.object(provider._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert isinstance(result, ProviderResponse)
        assert result.success is False
        assert "500" in result.error

    async def test_generate_network_exception(self):
        """Network errors return error ProviderResponse, do not raise."""
        import httpx
        provider = AnthropicProvider(api_key="sk-test-key")

        with patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            result = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert isinstance(result, ProviderResponse)
        assert result.success is False
        assert result.error is not None


# ── Health check tests ─────────────────────────────────────────────


class TestAnthropicProviderHealthCheck:
    """Tests for health_check method."""

    async def test_health_check_available_when_key_set(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        status = await provider.health_check()

        assert isinstance(status, HealthStatus)
        assert status.available is True
        assert status.error is None


# ── List models tests ──────────────────────────────────────────────


class TestAnthropicProviderListModels:
    """Tests for list_models method."""

    async def test_list_models_returns_models(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        models = await provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, ProviderModel) for m in models)

    async def test_list_models_contains_sonnet(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        models = await provider.list_models()

        sonnet_models = [m for m in models if "sonnet" in m.id.lower()]
        assert len(sonnet_models) > 0

    async def test_list_models_all_anthropic_provider_type(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        models = await provider.list_models()

        assert all(m.provider_type == "anthropic" for m in models)

    async def test_list_models_all_not_local(self):
        provider = AnthropicProvider(api_key="sk-test-key")
        models = await provider.list_models()

        assert all(m.is_local is False for m in models)


# ── Module-level ANTHROPIC_MODELS tests ────────────────────────────


class TestAnthropicModels:
    """Tests for the module-level ANTHROPIC_MODELS list."""

    def test_anthropic_models_is_list(self):
        assert isinstance(ANTHROPIC_MODELS, list)

    def test_anthropic_models_not_empty(self):
        assert len(ANTHROPIC_MODELS) > 0

    def test_anthropic_models_are_provider_model_instances(self):
        assert all(isinstance(m, ProviderModel) for m in ANTHROPIC_MODELS)

    def test_anthropic_models_contain_sonnet(self):
        names = [m.id for m in ANTHROPIC_MODELS]
        assert any("sonnet" in n for n in names)


# ── Close tests ────────────────────────────────────────────────────


class TestAnthropicProviderClose:
    """Tests for close method."""

    async def test_close_closes_client(self):
        provider = AnthropicProvider(api_key="sk-test-key")

        with patch.object(provider._client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()
