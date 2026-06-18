"""
NVIDIA NIM Provider for Bash Gym

Integrates with NVIDIA NIM API (OpenAI-compatible) for cloud inference
using models like Qwen Coder, Llama, and others hosted on NVIDIA's platform.
"""

import re
import time
from typing import Any

import httpx

from bashgym.providers.base import (
    HealthStatus,
    InferenceProvider,
    ProviderModel,
    ProviderResponse,
)

# Fallback only. list_models() queries the live OpenAI-compatible /models
# endpoint first; this list is returned when that call fails (offline, no key).
# Entries are current served NIM coding models — keep in step with the endpoint.
_NIM_MODELS = [
    ProviderModel(
        id="deepseek-ai/deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        provider_type="nvidia_nim",
        parameter_size="284B-MoE",
        is_code_model=True,
        context_length=1_000_000,
    ),
    ProviderModel(
        id="qwen/qwen3-next-80b-a3b-instruct",
        name="Qwen3 Next 80B A3B Instruct",
        provider_type="nvidia_nim",
        parameter_size="80B-A3B",
        is_code_model=True,
    ),
    ProviderModel(
        id="nvidia/nemotron-3-super-120b-a12b",
        name="Nemotron 3 Super 120B A12B",
        provider_type="nvidia_nim",
        parameter_size="120B-A12B",
        is_code_model=True,
    ),
    ProviderModel(
        id="mistralai/codestral-22b-instruct-v0.1",
        name="Codestral 22B Instruct",
        provider_type="nvidia_nim",
        parameter_size="22B",
        is_code_model=True,
        context_length=32768,
    ),
    ProviderModel(
        id="bigcode/starcoder2-15b",
        name="StarCoder2 15B",
        provider_type="nvidia_nim",
        parameter_size="15B",
        is_code_model=True,
        context_length=16384,
    ),
]

# Heuristic for flagging code-capable models in the live catalog. Mirrors the
# detector's NIM heuristic, extended with current coder families.
_CODE_MODEL_PATTERN = re.compile(
    r"cod(e|er)|deepseek|starcoder|qwen|codestral|nemotron|granite", re.I
)


def _model_from_api(model_id: str) -> ProviderModel:
    """Map an OpenAI-compatible /models id to a ProviderModel."""
    return ProviderModel(
        id=model_id,
        name=model_id,
        provider_type="nvidia_nim",
        is_code_model=bool(_CODE_MODEL_PATTERN.search(model_id)),
    )


class NIMProvider(InferenceProvider):
    """
    NVIDIA NIM cloud inference provider.

    Uses the OpenAI-compatible chat completions API hosted on NVIDIA's
    infrastructure. Supports Qwen Coder, Llama, and other models.
    """

    DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: str,
        endpoint: str | None = None,
        default_model: str = "deepseek-ai/deepseek-v4-flash",
        timeout: float = 120.0,
    ):
        self._api_key = api_key
        self._endpoint = endpoint or self.DEFAULT_ENDPOINT
        self._default_model = default_model
        self._client = httpx.AsyncClient(timeout=timeout)

    # ── Properties ─────────────────────────────────────────────────

    @property
    def provider_type(self) -> str:
        return "nvidia_nim"

    @property
    def requires_api_key(self) -> bool:
        return True

    @property
    def is_local(self) -> bool:
        return False

    # ── Generate ───────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a chat completion via the NIM API.

        Uses the OpenAI-compatible /chat/completions endpoint. If a
        system_prompt is provided and the messages list does not already
        contain a system message, one is prepended automatically.
        """
        resolved_model = model or self._default_model
        start = time.perf_counter()

        try:
            # Build the messages list
            request_messages = list(messages)
            if system_prompt and not any(m.get("role") == "system" for m in request_messages):
                request_messages.insert(0, {"role": "system", "content": system_prompt})

            payload = {
                "model": resolved_model,
                "messages": request_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            response = await self._client.post(
                f"{self._endpoint}/chat/completions",
                headers=headers,
                json=payload,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("completion_tokens", 0)

                return ProviderResponse(
                    content=content,
                    model_name=resolved_model,
                    provider_type=self.provider_type,
                    latency_ms=latency_ms,
                    tokens_used=tokens,
                    success=True,
                )
            else:
                return ProviderResponse(
                    content="",
                    model_name=resolved_model,
                    provider_type=self.provider_type,
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
                provider_type=self.provider_type,
                latency_ms=latency_ms,
                tokens_used=0,
                success=False,
                error=str(exc),
            )

    # ── Health check ───────────────────────────────────────────────

    async def health_check(self) -> HealthStatus:
        """
        Check NIM API availability by listing models.

        Calls GET /models with the auth header. Returns available=True
        on a 200 response, available=False on any error.
        """
        start = time.perf_counter()

        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
            }

            response = await self._client.get(
                f"{self._endpoint}/models",
                headers=headers,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                model_ids = [m.get("id", "") for m in data.get("data", [])]
                return HealthStatus(
                    available=True,
                    latency_ms=latency_ms,
                    models_loaded=model_ids,
                )
            else:
                return HealthStatus(
                    available=False,
                    latency_ms=latency_ms,
                    error=f"API returned {response.status_code}",
                )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthStatus(
                available=False,
                latency_ms=latency_ms,
                error=str(exc),
            )

    # ── List models ────────────────────────────────────────────────

    async def list_models(self) -> list[ProviderModel]:
        """Return the live NIM model catalogue.

        Queries the OpenAI-compatible GET /models endpoint so the catalogue
        reflects what the configured endpoint actually serves. Falls back to
        _NIM_MODELS if the endpoint is unreachable or returns nothing.
        """
        try:
            response = await self._client.get(
                f"{self._endpoint}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if response.status_code == 200:
                data = response.json().get("data", [])
                models = [_model_from_api(m["id"]) for m in data if m.get("id")]
                if models:
                    return models
        except Exception:
            pass
        return list(_NIM_MODELS)

    # ── Cleanup ────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
