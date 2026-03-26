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


def build_base_config(config: "PipelineConfig") -> "dd.DataDesignerConfigBuilder":
    """Build the shared ModelConfig + ModelProvider base for all pipelines.

    Handles both single-provider (legacy) and multi-provider configurations.
    Each pipeline builder calls this, then adds its specific columns on top.
    """
    if not DATA_DESIGNER_AVAILABLE:
        raise ImportError("data-designer>=0.5.0 is required")

    # Build model configs (same for single or multi-provider)
    model_configs = [
        dd.ModelConfig(
            alias="text-model",
            model=config.text_model,
            inference_parameters=dd.InferenceParameters(
                temperature=config.temperature_text,
                top_p=0.99,
                max_tokens=2048,
            ),
        ),
        dd.ModelConfig(
            alias="code-model",
            model=config.code_model,
            inference_parameters=dd.InferenceParameters(
                temperature=config.temperature_code,
                top_p=0.95,
                max_tokens=4096,
            ),
        ),
        dd.ModelConfig(
            alias="judge-model",
            model=config.judge_model,
            inference_parameters=dd.InferenceParameters(
                temperature=config.temperature_judge,
                max_tokens=1024,
            ),
        ),
    ]

    # Build providers — multi-provider or single-provider
    if config.providers:
        # Multi-provider: each ProviderSpec becomes a ModelProvider
        model_providers = []
        for prov in config.providers:
            api_key = prov.api_key or f"${{{_env_key_for_provider(prov.name)}}}"
            model_providers.append(
                dd.ModelProvider(
                    name=prov.name,
                    endpoint=prov.endpoint,
                    provider_type="openai",
                    api_key=api_key,
                )
            )
    else:
        # Single-provider (legacy/backward compat)
        model_providers = [
            dd.ModelProvider(
                name=config.provider,
                endpoint=config.provider_endpoint,
                provider_type="openai",
                api_key=f"${{{_env_key_for_provider(config.provider)}}}",
            ),
        ]

    return dd.DataDesignerConfigBuilder(
        model_configs=model_configs,
        model_providers=model_providers,
    )


def register_pipeline(name: str):
    """Decorator to register a pipeline builder function."""

    def decorator(fn):
        PIPELINES[name] = fn
        return fn

    return decorator


# Import pipeline modules to trigger registration
try:
    from bashgym.factory.designer_pipelines.coding_agent_dpo import build_dpo_pipeline
    from bashgym.factory.designer_pipelines.coding_agent_sft import build_sft_pipeline
    from bashgym.factory.designer_pipelines.from_external import build_external_pipeline
    from bashgym.factory.designer_pipelines.from_unstructured import build_unstructured_pipeline
    from bashgym.factory.designer_pipelines.tool_use_sft import build_tool_use_pipeline

    PIPELINES["coding_agent_sft"] = build_sft_pipeline
    PIPELINES["coding_agent_dpo"] = build_dpo_pipeline
    PIPELINES["tool_use_sft"] = build_tool_use_pipeline
    PIPELINES["from_external"] = build_external_pipeline
    PIPELINES["from_unstructured"] = build_unstructured_pipeline
except ImportError:
    # data-designer not installed - pipelines unavailable
    pass
