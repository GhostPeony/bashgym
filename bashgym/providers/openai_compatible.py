"""Generic OpenAI-compatible inference provider.

Talks to any OpenAI ``/chat/completions`` + ``/models`` endpoint via a
configurable ``base_url`` and API key. ``PRESETS`` maps known platforms
(Together, Fireworks, OpenRouter, Groq, DeepInfra, Hyperbolic, self-hosted
vLLM) to their base URLs.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import HealthStatus, InferenceProvider, ProviderModel, ProviderResponse
from .embeddings import parse_embeddings_response

# Known OpenAI-compatible platforms -> base URL. Pass any other base_url directly.
PRESETS: dict[str, str] = {
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "hyperbolic": "https://api.hyperbolic.xyz/v1",
    "vllm": "http://localhost:8000/v1",  # self-hosted; no key needed
}


class OpenAICompatibleProvider(InferenceProvider):
    """Talks to any OpenAI-compatible chat endpoint (cloud or self-hosted)."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        default_model: str = "",
        *,
        is_local: bool = False,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._is_local = is_local
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def for_platform(
        cls, platform: str, *, api_key: str | None = None, default_model: str = "", **kwargs: Any
    ) -> OpenAICompatibleProvider:
        """Build a provider for a known platform name (see ``PRESETS``)."""
        base = PRESETS.get(platform)
        if not base:
            raise ValueError(
                f"unknown platform {platform!r}; known: {sorted(PRESETS)} "
                f"(or construct with an explicit base_url)"
            )
        is_local = platform == "vllm"
        return cls(
            name=platform,
            base_url=base,
            api_key=api_key,
            default_model=default_model,
            is_local=is_local,
            **kwargs,
        )

    @property
    def provider_type(self) -> str:
        return self._name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def requires_api_key(self) -> bool:
        return not self._is_local

    @property
    def is_local(self) -> bool:
        return self._is_local

    @property
    def supports_embeddings(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        resolved_model = model or self._default_model
        start = time.perf_counter()
        try:
            request_messages = list(messages)
            if system_prompt and not any(m.get("role") == "system" for m in request_messages):
                request_messages.insert(0, {"role": "system", "content": system_prompt})

            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": resolved_model,
                    "messages": request_messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"].get("content") or ""
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                return ProviderResponse(
                    content=content,
                    model_name=resolved_model,
                    provider_type=self.provider_type,
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                    success=True,
                    metadata={"raw": data},
                )
            return ProviderResponse(
                content="",
                model_name=resolved_model,
                provider_type=self.provider_type,
                latency_ms=latency_ms,
                tokens_used=0,
                success=False,
                error=f"API error: {response.status_code} - {response.text}",
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure as an unsuccessful response
            latency_ms = (time.perf_counter() - start) * 1000
            return ProviderResponse(
                content="",
                model_name=resolved_model,
                provider_type=self.provider_type,
                latency_ms=latency_ms,
                tokens_used=0,
                success=False,
                error=str(exc),
            )

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Embed texts via the OpenAI-compatible ``POST /v1/embeddings`` endpoint.

        The embedding model is not hardcoded: it falls back to
        ``default_model`` when ``model`` is None, so the caller selects an
        embedding model from the live catalog.
        """
        resolved_model = model or self._default_model
        response = await self._client.post(
            f"{self._base_url}/embeddings",
            headers=self._headers(),
            json={"model": resolved_model, "input": list(texts)},
        )
        if response.status_code != 200:
            raise RuntimeError(f"embeddings API error: {response.status_code} - {response.text}")
        return parse_embeddings_response(response.json())

    async def health_check(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            response = await self._client.get(f"{self._base_url}/models", headers=self._headers())
            latency_ms = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                model_ids = [m.get("id", "") for m in response.json().get("data", [])]
                return HealthStatus(available=True, latency_ms=latency_ms, models_loaded=model_ids)
            return HealthStatus(
                available=False, latency_ms=latency_ms, error=f"API returned {response.status_code}"
            )
        except Exception as exc:  # noqa: BLE001 - health check never raises
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthStatus(available=False, latency_ms=latency_ms, error=str(exc))

    async def list_models(self) -> list[ProviderModel]:
        try:
            response = await self._client.get(f"{self._base_url}/models", headers=self._headers())
            if response.status_code != 200:
                return []
            return [
                ProviderModel(
                    id=m.get("id", ""),
                    name=m.get("id", ""),
                    provider_type=self.provider_type,
                    is_local=self._is_local,
                )
                for m in response.json().get("data", [])
            ]
        except Exception:  # noqa: BLE001 - return empty on any error
            return []

    async def close(self) -> None:
        await self._client.aclose()
