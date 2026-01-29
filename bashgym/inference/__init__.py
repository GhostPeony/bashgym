"""
Inference module for loading and running trained models.

Provides model loading and generation capabilities for evaluation.
"""

from .model_loader import ModelLoader, InferenceConfig

__all__ = ["ModelLoader", "InferenceConfig"]
