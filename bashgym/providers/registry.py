"""
ProviderRegistry — central registry for inference providers.

Manages provider registration, model-to-provider mapping, health
monitoring, and auto-discovery of available models.
"""

import logging
from collections.abc import Callable
from typing import Any

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
        self.providers: dict[str, InferenceProvider] = {}
        self._model_map: dict[str, str] = {}
        self._discovered_models: dict[str, ProviderModel] = {}
        self.health_cache: dict[str, HealthStatus] = {}
        self._health_listeners: list[Callable] = []

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
        to_remove = [model for model, ptype in self._model_map.items() if ptype == provider_type]
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

    def get_provider_for_model(self, model_name: str) -> InferenceProvider | None:
        """Resolve a model name to its provider, or None if unmapped."""
        ptype = self._model_map.get(model_name)
        if ptype is None:
            return None
        return self.providers.get(ptype)

    def get_provider(self, provider_type: str) -> InferenceProvider | None:
        """Get a provider by its type string, or None."""
        return self.providers.get(provider_type)

    def get_model_map(self) -> dict[str, str]:
        """Return a copy of the model-to-provider map."""
        return dict(self._model_map)

    # ── Generation ────────────────────────────────────────────────

    async def generate(
        self,
        model: str,
        messages: list[dict[str, str]],
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

    async def check_all_health(self) -> dict[str, HealthStatus]:
        """Health-check all registered providers, update cache, notify listeners."""
        results: dict[str, HealthStatus] = {}
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

    async def discover_models(self) -> dict[str, list[ProviderModel]]:
        """
        Ask all providers for their available models.

        Auto-maps discovered models unless a mapping already exists.
        """
        discovered: dict[str, list[ProviderModel]] = {}
        for ptype, provider in self.providers.items():
            try:
                models = await provider.list_models()
                discovered[ptype] = models
                for model in models:
                    self._discovered_models[model.id] = model
                    if model.id not in self._model_map:
                        self._model_map[model.id] = ptype
                        logger.debug("Auto-mapped %s -> %s", model.id, ptype)
            except Exception:
                logger.exception("Model discovery failed for %s", ptype)
                discovered[ptype] = []
        return discovered

    # ── Auto-selection ─────────────────────────────────────────────

    def select_best_model(
        self,
        provider_type: str,
        prefer_code: bool = True,
        default_model: str | None = None,
    ) -> ProviderModel | None:
        """
        Select the best model from a provider for use as Student.

        Priority:
        1. default_model (if set and exists in discovered models)
        2. Best code model (if prefer_code is True)
        3. Largest model by parameter size

        Returns the chosen ProviderModel, or None if no models found.
        """
        # Collect models mapped to this provider
        models: list[ProviderModel] = []
        for model_id, ptype in self._model_map.items():
            if ptype != provider_type:
                continue
            # We need the ProviderModel metadata; check _discovered cache
            if model_id in self._discovered_models:
                models.append(self._discovered_models[model_id])

        if not models:
            return None

        # Priority 1: explicit default
        if default_model:
            for m in models:
                if default_model in m.name or default_model in m.id:
                    return m

        # Priority 2: prefer code models (pick largest code model)
        if prefer_code:
            code_models = [m for m in models if m.is_code_model]
            if code_models:
                return max(code_models, key=lambda m: _parse_param_size(m.parameter_size))

        # Priority 3: largest model
        return max(models, key=lambda m: _parse_param_size(m.parameter_size))

    # ── Status ────────────────────────────────────────────────────

    def get_status_summary(self) -> dict[str, Any]:
        """Summary of registered providers, model mappings, and totals."""
        providers_info: dict[str, dict[str, Any]] = {}
        for ptype, provider in self.providers.items():
            providers_info[ptype] = {
                "is_local": provider.is_local,
                "requires_api_key": provider.requires_api_key,
                "health": (
                    self.health_cache[ptype].to_dict() if ptype in self.health_cache else None
                ),
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

_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the global ProviderRegistry singleton, creating it if needed."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def _parse_param_size(param_size: str | None) -> float:
    """Parse parameter size string like '35B', '7B', '1.5B' into a float for comparison."""
    if not param_size or param_size == "unknown":
        return 0.0
    s = param_size.strip().upper()
    try:
        if s.endswith("B"):
            return float(s[:-1])
        return float(s)
    except (ValueError, IndexError):
        return 0.0
