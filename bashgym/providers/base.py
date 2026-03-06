"""
InferenceProvider ABC and supporting data structures.

Foundation for the provider abstraction layer. All inference providers
(Anthropic, NIM, Ollama, etc.) implement the InferenceProvider ABC.

No imports from other bashgym modules to avoid circular dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ProviderResponse:
    """Response from an inference provider."""

    content: str
    model_name: str
    provider_type: str
    latency_ms: float
    tokens_used: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model_name": self.model_name,
            "provider_type": self.provider_type,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class HealthStatus:
    """Health check result for a provider."""

    available: bool
    latency_ms: float = 0.0
    error: Optional[str] = None
    models_loaded: List[str] = field(default_factory=list)
    last_checked: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "models_loaded": self.models_loaded,
            "last_checked": self.last_checked,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
        }


@dataclass
class ProviderModel:
    """Model available from an inference provider."""

    id: str
    name: str
    provider_type: str
    size_gb: Optional[float] = None
    parameter_size: Optional[str] = None
    is_code_model: bool = False
    is_local: bool = False
    context_length: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider_type": self.provider_type,
            "size_gb": self.size_gb,
            "parameter_size": self.parameter_size,
            "is_code_model": self.is_code_model,
            "is_local": self.is_local,
            "context_length": self.context_length,
        }


class InferenceProvider(ABC):
    """
    Abstract base class for inference providers.

    All model providers (Anthropic, NVIDIA NIM, Ollama, etc.) implement
    this interface for a unified inference API.
    """

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Identifier for this provider type (e.g. 'ollama', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether this provider requires an API key."""
        ...

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Whether this provider runs locally."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion from the model."""
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check provider health and availability."""
        ...

    @abstractmethod
    async def list_models(self) -> List[ProviderModel]:
        """List available models from this provider."""
        ...

    async def warm_up(self, model: Optional[str] = None) -> bool:
        """
        Warm up the provider, optionally loading a specific model.

        Returns True if warm-up succeeded.
        """
        return True

    async def close(self) -> None:
        """Clean up provider resources."""
        pass
