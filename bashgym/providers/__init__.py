"""
Model Providers for Bash Gym

Supports both cloud and local model providers for training and inference.
"""

from .base import InferenceProvider, ProviderResponse, HealthStatus, ProviderModel
from .registry import ProviderRegistry, get_registry
from .ollama import OllamaProvider, OllamaModel
from .anthropic import AnthropicProvider
from .nim import NIMProvider
from .detector import detect_providers, get_available_models, ProviderType

__all__ = [
    'InferenceProvider', 'ProviderResponse', 'HealthStatus', 'ProviderModel',
    'ProviderRegistry', 'get_registry',
    'OllamaProvider', 'OllamaModel',
    'AnthropicProvider', 'NIMProvider',
    'ProviderType', 'detect_providers', 'get_available_models',
]
