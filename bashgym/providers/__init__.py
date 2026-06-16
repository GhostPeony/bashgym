"""
Model Providers for Bash Gym

Supports both cloud and local model providers for training and inference.
"""

from .anthropic import AnthropicProvider
from .base import HealthStatus, InferenceProvider, ProviderModel, ProviderResponse
from .detector import ProviderType, detect_providers, get_available_models
from .nim import NIMProvider
from .ollama import OllamaModel, OllamaProvider
from .openai_compatible import PRESETS, OpenAICompatibleProvider
from .registry import ProviderRegistry, get_registry

__all__ = [
    "InferenceProvider",
    "ProviderResponse",
    "HealthStatus",
    "ProviderModel",
    "ProviderRegistry",
    "get_registry",
    "OllamaProvider",
    "OllamaModel",
    "AnthropicProvider",
    "NIMProvider",
    "OpenAICompatibleProvider",
    "PRESETS",
    "ProviderType",
    "detect_providers",
    "get_available_models",
]
