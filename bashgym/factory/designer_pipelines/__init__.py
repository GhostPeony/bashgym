"""
Pre-built DataDesigner Pipeline Builders

Each module exports a build_*_pipeline(config) function that returns
a DataDesignerConfigBuilder with the full column DAG configured.

Pipeline builders are registered in PIPELINES for lookup by name.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import data_designer.config as dd  # noqa: F401

    from bashgym.factory.data_designer import PipelineConfig, ProviderSpec  # noqa: F401

try:
    import data_designer.config as dd
    from data_designer.engine.secret_resolver import (
        CompositeResolver,
        EnvironmentResolver,
        PlaintextResolver,
    )

    DATA_DESIGNER_AVAILABLE = True
except ImportError:
    DATA_DESIGNER_AVAILABLE = False

# Registry of available pipeline builders
# Each entry maps a name to a function: (PipelineConfig) -> DataDesignerConfigBuilder
PIPELINES: dict[str, Callable] = {}


def _env_key_for_provider(provider: str) -> str:
    """Map provider name to environment variable key."""
    mapping = {
        "nvidia": "NVIDIA_API_KEY",
        "nvidia-nim": "NVIDIA_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "local": "LOCAL_API_KEY",
    }
    return mapping.get(provider, "NVIDIA_API_KEY")


def build_model_providers(config: "PipelineConfig") -> "list[dd.ModelProvider]":
    """Build the dd.ModelProvider list for the DataDesigner instance.

    As of Data Designer 0.6.x, providers are attached to the ``DataDesigner``
    object (``DataDesigner(model_providers=...)``), not to the config builder.
    Handles both single-provider (legacy) and multi-provider configurations.
    """
    if not DATA_DESIGNER_AVAILABLE:
        raise ImportError("data-designer>=0.6.1 is required")

    # api_key carries the env-var NAME (resolved by EnvironmentResolver in
    # build_secret_resolver) unless an explicit key is supplied, in which case the
    # PlaintextResolver fallback uses it verbatim. Data Designer 0.6.x does NOT
    # expand "${VAR}" placeholders, so the bare name is required.
    if config.providers:
        # Multi-provider: each ProviderSpec becomes a ModelProvider
        return [
            dd.ModelProvider(
                name=prov.name,
                endpoint=prov.endpoint,
                provider_type="openai",
                api_key=prov.api_key or _env_key_for_provider(prov.name),
            )
            for prov in config.providers
        ]
    # Single-provider (legacy/backward compat)
    return [
        dd.ModelProvider(
            name=config.provider,
            endpoint=config.provider_endpoint,
            provider_type="openai",
            api_key=config.provider_api_key or _env_key_for_provider(config.provider),
        ),
    ]


def build_secret_resolver():
    """Resolver for ModelProvider api_keys (Data Designer 0.6.x).

    Treats ``api_key`` as an environment-variable NAME first (EnvironmentResolver),
    then falls back to using it verbatim as a literal secret (PlaintextResolver).
    """
    if not DATA_DESIGNER_AVAILABLE:
        raise ImportError("data-designer>=0.6.1 is required")
    return CompositeResolver([EnvironmentResolver(), PlaintextResolver()])


def _provider_name_for(alias: str, config: "PipelineConfig") -> str:
    """Resolve which provider serves a given model alias.

    In multi-provider mode a ProviderSpec may declare the aliases it serves via
    ``models``; otherwise the first provider is used. In single-provider mode the
    one configured provider serves every alias.
    """
    if config.providers:
        for prov in config.providers:
            if alias in prov.models:
                return prov.name
        return config.providers[0].name
    return config.provider


def build_base_config(config: "PipelineConfig") -> "dd.DataDesignerConfigBuilder":
    """Build the shared ModelConfig base for all pipelines (Data Designer 0.6.x).

    Each ModelConfig binds to a provider by name; the providers themselves are
    created by ``build_model_providers`` and attached to the DataDesigner
    instance. Each pipeline builder calls this, then adds its columns on top.
    """
    if not DATA_DESIGNER_AVAILABLE:
        raise ImportError("data-designer>=0.6.1 is required")

    model_configs = [
        dd.ModelConfig(
            alias="text-model",
            model=config.text_model,
            provider=_provider_name_for("text-model", config),
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=config.temperature_text,
                top_p=0.99,
                max_tokens=2048,
            ),
        ),
        dd.ModelConfig(
            alias="code-model",
            model=config.code_model,
            provider=_provider_name_for("code-model", config),
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=config.temperature_code,
                top_p=0.95,
                max_tokens=4096,
            ),
        ),
        dd.ModelConfig(
            alias="judge-model",
            model=config.judge_model,
            provider=_provider_name_for("judge-model", config),
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=config.temperature_judge,
                max_tokens=1024,
            ),
        ),
    ]

    return dd.DataDesignerConfigBuilder(model_configs=model_configs)


def register_pipeline(name: str):
    """Decorator to register a pipeline builder function."""

    def decorator(fn):
        PIPELINES[name] = fn
        return fn

    return decorator


# Import pipeline modules to trigger registration
try:
    from bashgym.factory.designer_pipelines.coding_agent_distill import build_distill_pipeline
    from bashgym.factory.designer_pipelines.coding_agent_dpo import build_dpo_pipeline
    from bashgym.factory.designer_pipelines.coding_agent_sft import build_sft_pipeline
    from bashgym.factory.designer_pipelines.from_external import build_external_pipeline
    from bashgym.factory.designer_pipelines.from_unstructured import build_unstructured_pipeline
    from bashgym.factory.designer_pipelines.tool_use_sft import build_tool_use_pipeline

    PIPELINES["coding_agent_sft"] = build_sft_pipeline
    PIPELINES["coding_agent_dpo"] = build_dpo_pipeline
    PIPELINES["coding_agent_distill"] = build_distill_pipeline
    PIPELINES["tool_use_sft"] = build_tool_use_pipeline
    PIPELINES["from_external"] = build_external_pipeline
    PIPELINES["from_unstructured"] = build_unstructured_pipeline
except ImportError:
    # data-designer not installed - pipelines unavailable
    pass
