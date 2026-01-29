"""
Model Providers for Bash Gym

Supports both cloud and local model providers for training and inference.
"""

from .ollama import OllamaProvider, OllamaModel
from .detector import detect_providers, get_available_models

__all__ = [
    'OllamaProvider',
    'OllamaModel',
    'detect_providers',
    'get_available_models',
]
