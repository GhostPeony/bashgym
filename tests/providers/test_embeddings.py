"""Tests for the provider embedding capability and the rwml embed_fn adapter.

Covers:
- The non-abstract ``embed`` default on InferenceProvider (raises, does not
  break existing providers).
- Pure response parsing (``parse_embeddings_response``) for both the OpenAI
  ``/v1/embeddings`` shape and the Ollama embeddings shape.
- Concrete ``embed`` on OpenAICompatibleProvider and OllamaProvider via a
  mocked httpx transport (no network).
- ``make_embed_fn``: a pure, sync adapter that turns a provider's async
  ``embed`` into the ``Callable[[str], Sequence[float]]`` that rwml expects.
"""

import asyncio

import httpx
import pytest

from bashgym.providers.base import HealthStatus, InferenceProvider, ProviderResponse
from bashgym.providers.embeddings import (
    make_embed_fn,
    parse_embeddings_response,
    parse_ollama_embeddings_response,
)
from bashgym.providers.ollama import OllamaProvider
from bashgym.providers.openai_compatible import OpenAICompatibleProvider


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── Pure parsing: OpenAI /v1/embeddings shape ──────────────────────


class TestParseEmbeddingsResponse:
    """parse_embeddings_response: data[].embedding -> list[list[float]]."""

    def test_parses_data_embedding_list(self):
        payload = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "text-embedding-3-small",
        }
        vectors = parse_embeddings_response(payload)
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_preserves_data_order_by_index(self):
        payload = {
            "data": [
                {"embedding": [9.0], "index": 1},
                {"embedding": [1.0], "index": 0},
            ]
        }
        vectors = parse_embeddings_response(payload)
        assert vectors == [[1.0], [9.0]]

    def test_empty_data_is_empty_list(self):
        assert parse_embeddings_response({"data": []}) == []

    def test_missing_data_key_is_empty_list(self):
        assert parse_embeddings_response({}) == []

    def test_values_are_floats(self):
        payload = {"data": [{"embedding": [1, 2, 3], "index": 0}]}
        vectors = parse_embeddings_response(payload)
        assert vectors == [[1.0, 2.0, 3.0]]
        assert all(isinstance(x, float) for x in vectors[0])


# ── Pure parsing: Ollama embeddings shape ──────────────────────────


class TestParseOllamaEmbeddingsResponse:
    """Ollama /api/embed returns {"embeddings": [[...], ...]}."""

    def test_parses_embeddings_key(self):
        payload = {"embeddings": [[0.1, 0.2], [0.3, 0.4]], "model": "nomic"}
        assert parse_ollama_embeddings_response(payload) == [[0.1, 0.2], [0.3, 0.4]]

    def test_parses_legacy_single_embedding_key(self):
        # The older /api/embeddings endpoint returns a single "embedding".
        payload = {"embedding": [0.5, 0.6, 0.7]}
        assert parse_ollama_embeddings_response(payload) == [[0.5, 0.6, 0.7]]

    def test_empty_is_empty_list(self):
        assert parse_ollama_embeddings_response({}) == []

    def test_values_are_floats(self):
        payload = {"embeddings": [[1, 2]]}
        vectors = parse_ollama_embeddings_response(payload)
        assert vectors == [[1.0, 2.0]]
        assert all(isinstance(x, float) for x in vectors[0])


# ── Default embed on the ABC (must NOT break existing providers) ───


class _MinimalProvider(InferenceProvider):
    """A provider that implements only the required ABC surface."""

    @property
    def provider_type(self) -> str:
        return "minimal"

    @property
    def requires_api_key(self) -> bool:
        return False

    @property
    def is_local(self) -> bool:
        return True

    async def generate(self, messages, model=None, **kwargs):
        return ProviderResponse(
            content="",
            model_name="",
            provider_type="minimal",
            latency_ms=0,
            tokens_used=0,
            success=True,
        )

    async def health_check(self):
        return HealthStatus(available=True)

    async def list_models(self):
        return []


class TestDefaultEmbed:
    """embed is a non-abstract default so existing providers still instantiate."""

    def test_minimal_provider_instantiates(self):
        # If embed were abstract, this would raise TypeError.
        assert _MinimalProvider() is not None

    def test_supports_embeddings_defaults_false(self):
        assert _MinimalProvider().supports_embeddings is False

    def test_default_embed_raises_not_implemented(self):
        provider = _MinimalProvider()
        with pytest.raises(NotImplementedError):
            asyncio.run(provider.embed(["hello"]))


# ── OpenAI-compatible embed ────────────────────────────────────────


class TestOpenAICompatibleEmbed:
    def test_supports_embeddings_true(self):
        p = OpenAICompatibleProvider("together", "https://x/v1", api_key="k")
        assert p.supports_embeddings is True

    def test_embed_posts_to_embeddings_endpoint(self):
        captured = {}

        def handler(request):
            import json

            captured["path"] = request.url.path
            captured["json"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"data": [{"embedding": [0.1, 0.2], "index": 0}]},
            )

        p = OpenAICompatibleProvider(
            "together", "https://x/v1", api_key="k", client=_client(handler)
        )
        vectors = asyncio.run(p.embed(["hello"], model="text-embedding-3-small"))

        assert captured["path"].endswith("/embeddings")
        assert captured["json"]["model"] == "text-embedding-3-small"
        assert captured["json"]["input"] == ["hello"]
        assert vectors == [[0.1, 0.2]]

    def test_embed_uses_default_model_when_none(self):
        captured = {}

        def handler(request):
            import json

            captured["json"] = json.loads(request.content)
            return httpx.Response(200, json={"data": [{"embedding": [1.0], "index": 0}]})

        p = OpenAICompatibleProvider(
            "together",
            "https://x/v1",
            api_key="k",
            default_model="def-embed",
            client=_client(handler),
        )
        asyncio.run(p.embed(["hi"]))
        assert captured["json"]["model"] == "def-embed"

    def test_embed_error_status_raises(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        p = OpenAICompatibleProvider(
            "together", "https://x/v1", api_key="k", client=_client(handler)
        )
        with pytest.raises(RuntimeError):
            asyncio.run(p.embed(["hi"], model="m"))


# ── Ollama embed ───────────────────────────────────────────────────


class TestOllamaEmbed:
    def test_supports_embeddings_true(self):
        assert OllamaProvider().supports_embeddings is True

    def test_embed_posts_to_api_embed(self):
        captured = {}

        def handler(request):
            import json

            captured["path"] = request.url.path
            captured["json"] = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(
            base_url=provider.base_url, transport=httpx.MockTransport(handler)
        )

        vectors = asyncio.run(provider.embed(["hello"], model="nomic-embed-text"))

        assert captured["path"] == "/api/embed"
        assert captured["json"]["model"] == "nomic-embed-text"
        assert captured["json"]["input"] == ["hello"]
        assert vectors == [[0.1, 0.2, 0.3]]

    def test_embed_error_status_raises(self):
        def handler(request):
            return httpx.Response(404, text="model not found")

        provider = OllamaProvider()
        provider._client = httpx.AsyncClient(
            base_url=provider.base_url, transport=httpx.MockTransport(handler)
        )
        with pytest.raises(RuntimeError):
            asyncio.run(provider.embed(["hi"], model="nope"))


# ── make_embed_fn adapter for rwml ─────────────────────────────────


class _FakeEmbedProvider:
    """A minimal stand-in exposing async embed(texts, *, model)."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.calls = []

    async def embed(self, texts, *, model=None):
        self.calls.append((list(texts), model))
        return [self._mapping[t] for t in texts]


class TestMakeEmbedFn:
    """make_embed_fn(provider, model) -> Callable[[str], Sequence[float]]."""

    def test_returns_single_vector_for_single_string(self):
        provider = _FakeEmbedProvider({"hello": [0.1, 0.2, 0.3]})
        embed_fn = make_embed_fn(provider)
        assert embed_fn("hello") == [0.1, 0.2, 0.3]

    def test_is_synchronous_callable(self):
        provider = _FakeEmbedProvider({"x": [1.0]})
        embed_fn = make_embed_fn(provider)
        # Calling it directly (no await) must return the vector, not a coroutine.
        result = embed_fn("x")
        assert result == [1.0]

    def test_passes_model_through(self):
        provider = _FakeEmbedProvider({"q": [0.5]})
        embed_fn = make_embed_fn(provider, model="my-embed-model")
        embed_fn("q")
        assert provider.calls[-1] == (["q"], "my-embed-model")

    def test_usable_as_rwml_embed_fn(self):
        from bashgym.gym.rwml import world_model_reward

        provider = _FakeEmbedProvider({"same": [1.0, 0.0], "same2": [1.0, 0.0], "diff": [0.0, 1.0]})
        embed_fn = make_embed_fn(provider)

        # Identical embeddings -> distance 0 -> reward 1.0
        assert world_model_reward("same", "same2", embed_fn) == 1.0
        # Orthogonal embeddings -> distance 1.0 -> reward 0.0
        assert world_model_reward("same", "diff", embed_fn) == 0.0
