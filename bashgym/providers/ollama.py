"""
Ollama Provider for Bash Gym

Integrates with local Ollama installation for:
- Model discovery and listing
- Inference for routing/testing
- Base models for fine-tuning (via export)
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from bashgym.providers.base import HealthStatus, InferenceProvider, ProviderModel, ProviderResponse


@dataclass
class OllamaModel:
    """Information about an Ollama model."""

    name: str
    size: int  # bytes
    modified_at: str
    digest: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size / (1024**3)

    @property
    def family(self) -> str:
        """Model family (e.g., qwen, llama, deepseek)."""
        return self.details.get("family", "unknown")

    @property
    def parameter_size(self) -> str:
        """Parameter size (e.g., 7B, 14B)."""
        return self.details.get("parameter_size", "unknown")

    @property
    def quantization(self) -> str:
        """Quantization level (e.g., Q4_K_M, Q8_0)."""
        return self.details.get("quantization_level", "unknown")

    @property
    def is_code_model(self) -> bool:
        """Check if this is a coding-focused model."""
        code_indicators = ["coder", "code", "codellama", "starcoder", "deepseek-coder"]
        name_lower = self.name.lower()
        return any(ind in name_lower for ind in code_indicators)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "size": self.size,
            "size_gb": round(self.size_gb, 2),
            "modified_at": self.modified_at,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "is_code_model": self.is_code_model,
            "provider": "ollama",
        }


class OllamaProvider(InferenceProvider):
    """
    Ollama local model provider.

    Connects to local Ollama server for model management and inference.
    Implements InferenceProvider for unified provider API.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._client: httpx.AsyncClient | None = None

    # ── InferenceProvider abstract properties ──────────────────────

    @property
    def provider_type(self) -> str:
        return "ollama"

    @property
    def requires_api_key(self) -> bool:
        return False

    @property
    def is_local(self) -> bool:
        return not self.is_remote

    @property
    def is_remote(self) -> bool:
        parsed = urlparse(self.base_url)
        hostname = parsed.hostname or ""
        return hostname not in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def is_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = await self.client.get("/")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def list_ollama_models(self) -> list[OllamaModel]:
        """List all available Ollama models (returns OllamaModel objects)."""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code != 200:
                return []

            data = response.json()
            models = []

            for model_data in data.get("models", []):
                model = OllamaModel(
                    name=model_data.get("name", ""),
                    size=model_data.get("size", 0),
                    modified_at=model_data.get("modified_at", ""),
                    digest=model_data.get("digest", ""),
                    details=model_data.get("details", {}),
                )
                models.append(model)

            return models

        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return []

    async def get_model(self, name: str) -> OllamaModel | None:
        """Get info about a specific model."""
        try:
            response = await self.client.post("/api/show", json={"name": name})
            if response.status_code != 200:
                return None

            data = response.json()
            return OllamaModel(
                name=name,
                size=data.get("size", 0),
                modified_at=data.get("modified_at", ""),
                digest=data.get("digest", ""),
                details=data.get("details", {}),
            )
        except Exception:
            return None

    async def pull_model(self, name: str, on_progress: Callable | None = None) -> bool:
        """
        Pull (download) a model from Ollama registry.

        Args:
            name: Model name (e.g., "qwen2.5-coder:7b")
            on_progress: Optional callback for progress updates

        Returns:
            True if successful
        """
        try:
            async with self.client.stream(
                "POST",
                "/api/pull",
                json={"name": name, "stream": True},
                timeout=None,  # No timeout for large downloads
            ) as response:
                if response.status_code != 200:
                    return False

                async for line in response.aiter_lines():
                    if line and on_progress:
                        try:
                            data = __import__("json").loads(line)
                            on_progress(data)
                        except Exception:
                            pass

                return True

        except Exception as e:
            print(f"Error pulling model: {e}")
            return False

    async def delete_model(self, name: str) -> bool:
        """Delete a model."""
        try:
            response = await self.client.delete("/api/delete", json={"name": name})
            return response.status_code == 200
        except Exception:
            return False

    async def complete(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Generate completion from a model (completion-style API).

        This is the original generate() method, renamed to avoid collision
        with the InferenceProvider ABC's chat-style generate().

        Args:
            model: Model name
            prompt: User prompt
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            Generation result
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        if system:
            payload["system"] = system

        try:
            if stream:
                # Return async generator for streaming
                return self._stream_generate(payload)
            else:
                response = await self.client.post("/api/generate", json=payload, timeout=120.0)
                return response.json()

        except Exception as e:
            return {"error": str(e)}

    async def _stream_generate(self, payload: dict[str, Any]):
        """Stream generation responses."""
        async with self.client.stream(
            "POST", "/api/generate", json=payload, timeout=None
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    yield __import__("json").loads(line)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Chat completion with a model.

        Args:
            model: Model name
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Chat response
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            response = await self.client.post("/api/chat", json=payload, timeout=120.0)
            return response.json()

        except Exception as e:
            return {"error": str(e)}

    # ── InferenceProvider ABC implementations ──────────────────────

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response using the chat API (InferenceProvider interface)."""
        start = datetime.now()
        try:
            chat_messages = list(messages)
            if system_prompt and not any(m["role"] == "system" for m in chat_messages):
                chat_messages.insert(0, {"role": "system", "content": system_prompt})

            result = await self.chat(
                model,
                chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = (datetime.now() - start).total_seconds() * 1000

            if "error" in result:
                return ProviderResponse(
                    content="",
                    model_name=model,
                    provider_type="ollama",
                    latency_ms=latency,
                    tokens_used=0,
                    success=False,
                    error=result["error"],
                )

            content = result.get("message", {}).get("content", "")
            tokens = result.get("eval_count", 0)
            return ProviderResponse(
                content=content,
                model_name=model,
                provider_type="ollama",
                latency_ms=latency,
                tokens_used=tokens,
                success=True,
            )
        except Exception as e:
            latency = (datetime.now() - start).total_seconds() * 1000
            return ProviderResponse(
                content="",
                model_name=model,
                provider_type="ollama",
                latency_ms=latency,
                tokens_used=0,
                success=False,
                error=str(e),
            )

    async def health_check(self) -> HealthStatus:
        """Check Ollama server health and list loaded models."""
        try:
            start = datetime.now()
            running = await self.is_running()
            latency = (datetime.now() - start).total_seconds() * 1000

            if not running:
                return HealthStatus(
                    available=False,
                    latency_ms=latency,
                    error="Ollama not running",
                )

            # Check which models are loaded in VRAM via /api/ps
            models_loaded = []
            try:
                response = await self.client.get("/api/ps")
                if response.status_code == 200:
                    data = response.json()
                    models_loaded = [m.get("name", "") for m in data.get("models", [])]
            except Exception:
                pass

            return HealthStatus(
                available=True,
                latency_ms=latency,
                models_loaded=models_loaded,
            )
        except Exception as e:
            return HealthStatus(available=False, error=str(e))

    async def list_models(self) -> list[ProviderModel]:
        """List available models as ProviderModel objects (InferenceProvider interface)."""
        ollama_models = await self.list_ollama_models()
        return [
            ProviderModel(
                id=f"ollama/{m.name}",
                name=m.name,
                provider_type="ollama",
                size_gb=round(m.size_gb, 2),
                parameter_size=m.parameter_size,
                is_code_model=m.is_code_model,
                is_local=True,
            )
            for m in ollama_models
        ]

    async def warm_up(self, model: str | None = None) -> bool:
        """Warm up a model by sending a minimal chat request."""
        try:
            result = await self.chat(
                model,
                [{"role": "user", "content": "hi"}],
                temperature=0,
                max_tokens=1,
            )
            return "error" not in result
        except Exception:
            return False


# Singleton instance
_provider: OllamaProvider | None = None


def get_ollama_provider() -> OllamaProvider:
    """Get the singleton Ollama provider instance."""
    global _provider
    if _provider is None:
        _provider = OllamaProvider()
    return _provider


# Synchronous wrappers for non-async contexts
def is_ollama_running() -> bool:
    """Check if Ollama is running (sync wrapper)."""
    return asyncio.run(get_ollama_provider().is_running())


def list_ollama_models_sync() -> list[OllamaModel]:
    """List Ollama models (sync wrapper)."""
    return asyncio.run(get_ollama_provider().list_ollama_models())
