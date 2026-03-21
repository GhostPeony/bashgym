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

    from bashgym.factory.data_designer import PipelineConfig  # noqa: F401

# Registry of available pipeline builders
# Each entry maps a name to a function: (PipelineConfig) -> DataDesignerConfigBuilder
PIPELINES: dict[str, Callable] = {}


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
