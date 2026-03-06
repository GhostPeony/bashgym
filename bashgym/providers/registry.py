"""
ProviderRegistry — central registry for inference providers.

Manages provider registration, model-to-provider mapping, health
monitoring, and auto-discovery of available models.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from bashgym.providers.base import (
    HealthStatus,
    InferenceProvider,
    ProviderModel,
    ProviderResponse,
)

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Central registry that manages inference providers.

    Responsibilities:
    - Register/unregister providers by type
    - Map model names to providers
    - Health-check all providers and cache results
    - Auto-discover models from registered providers
    """

    def __init__(self) -> None:
        self.providers: Dict[str, InferenceProvider] = {}
        self._model_map: Dict[str, str] = {}
        self.health_cache: Dict[str, HealthStatus] = {}
        self._health_listeners: List[Callable] = []

    # ── Registration ───────────────────────────────────────────────

    def register(self, provider: InferenceProvider) -> None:
        """Register a provider, keyed by its provider_type."""
        ptype = provider.provider_type
        logger.info("Registering provider: %s", ptype)
        self.providers[ptype] = provider

    def unregister(self, provider_type: str) -> None:
        """Remove a provider, its health cache, and all model mappings pointing to it."""
        if provider_type not in self.providers:
            return
        del self.providers[provider_type]
        self.health_cache.pop(provider_type, None)
        # Remove all model mappings that point to this provider
        to_remove = [
            model for model, ptype in self._model_map.items()
            if ptype == provider_type
        ]
        for model in to_remove:
            del self._model_map[model]
        logger.info("Unregistered provider: %s", provider_type)

    # ── Model mapping ─────────────────────────────────────────────

    def map_model(self, model_name: str, provider_type: str) -> None:
        """Map a model name to a provider type. Raises ValueError if provider not registered."""
        if provider_type not in self.providers:
            raise ValueError(
                f"Provider '{provider_type}' not registered. "
                f"Available: {list(self.providers.keys())}"
            )
        self._model_map[model_name] = provider_type

    def unmap_model(self, model_name: str) -> None:
        """Remove a model mapping."""
        self._model_map.pop(model_name, None)

    def get_provider_for_model(self, model_name: str) -> Optional[InferenceProvider]:
        """Resolve a model name to its provider, or None if unmapped."""
        ptype = self._model_map.get(model_name)
        if ptype is None:
            return None
        return self.providers.get(ptype)

    def get_provider(self, provider_type: str) -> Optional[InferenceProvider]:
        """Get a provider by its type string, or None."""
        return self.providers.get(provider_type)

    def get_model_map(self) -> Dict[str, str]:
        """Return a copy of the model-to-provider map."""
        return dict(self._model_map)

    # ── Generation ────────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate via the provider mapped to the given model. Raises ValueError if unmapped."""
        provider = self.get_provider_for_model(model)
        if provider is None:
            raise ValueError(
                f"No provider mapped for model '{model}'. "
                f"Mapped models: {list(self._model_map.keys())}"
            )
        return await provider.generate(messages, model=model, **kwargs)

    # ── Health monitoring ─────────────────────────────────────────

    async def check_all_health(self) -> Dict[str, HealthStatus]:
        """Health-check all registered providers, update cache, notify listeners."""
        results: Dict[str, HealthStatus] = {}
        for ptype, provider in self.providers.items():
            try:
                status = await provider.health_check()
            except Exception as exc:
                logger.warning("Health check failed for %s: %s", ptype, exc)
                status = HealthStatus(available=False, error=str(exc))
            results[ptype] = status
            self.health_cache[ptype] = status

        # Notify listeners
        for callback in self._health_listeners:
            try:
                callback(results)
            except Exception:
                logger.exception("Health listener error")

        return results

    def on_health_change(self, callback: Callable) -> None:
        """Register a callback to be notified after health checks."""
        self._health_listeners.append(callback)

    # ── Auto-discovery ────────────────────────────────────────────

    async def discover_models(self) -> Dict[str, List[ProviderModel]]:
        """
        Ask all providers for their available models.

        Auto-maps discovered models unless a mapping already exists.
        """
        discovered: Dict[str, List[ProviderModel]] = {}
        for ptype, provider in self.providers.items():
            try:
                models = await provider.list_models()
                discovered[ptype] = models
                for model in models:
                    if model.id not in self._model_map:
                        self._model_map[model.id] = ptype
                        logger.debug("Auto-mapped %s -> %s", model.id, ptype)
            except Exception:
                logger.exception("Model discovery failed for %s", ptype)
                discovered[ptype] = []
        return discovered

    # ── Status ────────────────────────────────────────────────────

    def get_status_summary(self) -> Dict[str, Any]:
        """Summary of registered providers, model mappings, and totals."""
        providers_info: Dict[str, Dict[str, Any]] = {}
        for ptype, provider in self.providers.items():
            providers_info[ptype] = {
                "is_local": provider.is_local,
                "requires_api_key": provider.requires_api_key,
                "health": self.health_cache[ptype].to_dict()
                if ptype in self.health_cache
                else None,
            }
        return {
            "providers": providers_info,
            "model_map": dict(self._model_map),
            "total_models": len(self._model_map),
        }

    # ── Teardown ──────────────────────────────────────────────────

    async def close(self) -> None:
        """Close all registered providers."""
        for ptype, provider in self.providers.items():
            try:
                await provider.close()
            except Exception:
                logger.exception("Error closing provider %s", ptype)


# ── Module-level singleton ────────────────────────────────────────

_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """Return the global ProviderRegistry singleton, creating it if needed."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
