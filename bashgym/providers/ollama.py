"""
Ollama Provider for Bash Gym

Integrates with local Ollama installation for:
- Model discovery and listing
- Inference for routing/testing
- Base models for fine-tuning (via export)
"""

import httpx
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class OllamaModel:
    """Information about an Ollama model."""
    name: str
    size: int  # bytes
    modified_at: str
    digest: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        """Size in gigabytes."""
        return self.size / (1024 ** 3)

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
        code_indicators = ['coder', 'code', 'codellama', 'starcoder', 'deepseek-coder']
        name_lower = self.name.lower()
        return any(ind in name_lower for ind in code_indicators)

    def to_dict(self) -> Dict[str, Any]:
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
            "provider": "ollama"
        }


class OllamaProvider:
    """
    Ollama local model provider.

    Connects to local Ollama server for model management and inference.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._client: Optional[httpx.AsyncClient] = None

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

    async def list_models(self) -> List[OllamaModel]:
        """List all available models."""
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
                    details=model_data.get("details", {})
                )
                models.append(model)

            return models

        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return []

    async def get_model(self, name: str) -> Optional[OllamaModel]:
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
                details=data.get("details", {})
            )
        except Exception:
            return None

    async def pull_model(self, name: str, on_progress: Optional[callable] = None) -> bool:
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
                timeout=None  # No timeout for large downloads
            ) as response:
                if response.status_code != 200:
                    return False

                async for line in response.aiter_lines():
                    if line and on_progress:
                        try:
                            data = __import__('json').loads(line)
                            on_progress(data)
                        except:
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

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate completion from a model.

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
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            if stream:
                # Return async generator for streaming
                return self._stream_generate(payload)
            else:
                response = await self.client.post(
                    "/api/generate",
                    json=payload,
                    timeout=120.0
                )
                return response.json()

        except Exception as e:
            return {"error": str(e)}

    async def _stream_generate(self, payload: Dict[str, Any]):
        """Stream generation responses."""
        async with self.client.stream(
            "POST",
            "/api/generate",
            json=payload,
            timeout=None
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    yield __import__('json').loads(line)

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
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
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = await self.client.post(
                "/api/chat",
                json=payload,
                timeout=120.0
            )
            return response.json()

        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_provider: Optional[OllamaProvider] = None


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


def list_ollama_models() -> List[OllamaModel]:
    """List Ollama models (sync wrapper)."""
    return asyncio.run(get_ollama_provider().list_models())
