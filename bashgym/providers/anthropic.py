"""
Anthropic Provider for Bash Gym

Integrates with Anthropic's Claude API for inference via the
InferenceProvider interface.
"""

import time
from typing import Any

import httpx

from bashgym.providers.base import (
    HealthStatus,
    InferenceProvider,
    ProviderModel,
    ProviderResponse,
)

# ── Static model catalogue ─────────────────────────────────────────
# Fallback only. list_models() queries the live Models API first; this list is
# returned when that call fails (offline, bad key) so we never fall back to a
# retired model. Keep current with the latest GA Claude models.

ANTHROPIC_MODELS: list[ProviderModel] = [
    ProviderModel(
        id="claude-fable-5",
        name="Claude Fable 5",
        provider_type="anthropic",
        is_code_model=True,
        context_length=1_000_000,
    ),
    ProviderModel(
        id="claude-opus-4-8",
        name="Claude Opus 4.8",
        provider_type="anthropic",
        is_code_model=True,
        context_length=1_000_000,
    ),
    ProviderModel(
        id="claude-opus-4-7",
        name="Claude Opus 4.7",
        provider_type="anthropic",
        is_code_model=True,
        context_length=1_000_000,
    ),
    ProviderModel(
        id="claude-opus-4-6",
        name="Claude Opus 4.6",
        provider_type="anthropic",
        is_code_model=True,
        context_length=1_000_000,
    ),
    ProviderModel(
        id="claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider_type="anthropic",
        is_code_model=True,
        context_length=1_000_000,
    ),
    ProviderModel(
        id="claude-haiku-4-5",
        name="Claude Haiku 4.5",
        provider_type="anthropic",
        is_code_model=True,
        context_length=200_000,
    ),
]

API_URL = "https://api.anthropic.com/v1/messages"
MODELS_URL = "https://api.anthropic.com/v1/models"
API_VERSION = "2023-06-01"


def _model_from_api(entry: dict[str, Any]) -> ProviderModel:
    """Map a /v1/models entry to a ProviderModel.

    The list endpoint returns id/display_name only, so context length is
    derived from the family (Haiku is 200K; Opus/Sonnet/Fable are 1M).
    """
    model_id = entry["id"]
    context_length = 200_000 if "haiku" in model_id.lower() else 1_000_000
    return ProviderModel(
        id=model_id,
        name=entry.get("display_name") or model_id,
        provider_type="anthropic",
        is_code_model=True,
        context_length=context_length,
    )


class AnthropicProvider(InferenceProvider):
    """
    Anthropic Claude inference provider.

    Uses the Anthropic Messages API for completions.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided and non-empty")

        self._api_key = api_key
        self.default_model = default_model
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def provider_type(self) -> str:
        return "anthropic"

    @property
    def requires_api_key(self) -> bool:
        return True

    @property
    def is_local(self) -> bool:
        return False

    # ── Core methods ────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion via the Anthropic Messages API."""
        resolved_model = model or self.default_model
        start = time.perf_counter()

        try:
            # Separate system messages from the conversation
            system_parts: list[str] = []
            non_system_messages: list[dict[str, str]] = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_parts.append(msg["content"])
                else:
                    non_system_messages.append(msg)

            # Explicit system_prompt parameter takes priority
            if system_prompt:
                system_parts.insert(0, system_prompt)

            payload: dict[str, Any] = {
                "model": resolved_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": non_system_messages,
            }

            if system_parts:
                payload["system"] = "\n\n".join(system_parts)

            response = await self._client.post(
                API_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": API_VERSION,
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data["content"][0]["text"]
                tokens = data.get("usage", {}).get("output_tokens", 0)

                return ProviderResponse(
                    content=content,
                    model_name=resolved_model,
                    provider_type="anthropic",
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                    success=True,
                    metadata={
                        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                    },
                )
            else:
                return ProviderResponse(
                    content="",
                    model_name=resolved_model,
                    provider_type="anthropic",
                    latency_ms=latency_ms,
                    tokens_used=0,
                    success=False,
                    error=f"API error: {response.status_code} - {response.text}",
                )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ProviderResponse(
                content="",
                model_name=resolved_model,
                provider_type="anthropic",
                latency_ms=latency_ms,
                tokens_used=0,
                success=False,
                error=str(exc),
            )

    async def health_check(self) -> HealthStatus:
        """Check provider health (API key presence)."""
        if self._api_key:
            return HealthStatus(
                available=True,
                models_loaded=[m.id for m in ANTHROPIC_MODELS],
            )
        return HealthStatus(
            available=False,
            error="API key not configured",
        )

    async def list_models(self) -> list[ProviderModel]:
        """Return the live Claude model catalogue.

        Queries the Anthropic Models API (GET /v1/models) so the catalogue
        stays current without code changes. Falls back to ANTHROPIC_MODELS if
        the API is unreachable or returns nothing.
        """
        try:
            response = await self._client.get(
                MODELS_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": API_VERSION,
                },
            )
            if response.status_code == 200:
                data = response.json().get("data", [])
                models = [_model_from_api(m) for m in data if m.get("id")]
                if models:
                    return models
        except Exception:
            pass
        return list(ANTHROPIC_MODELS)

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
