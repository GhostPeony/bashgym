"""
Pre-built DataDesigner Pipeline Builders

Each module exports a build_*_pipeline(config) function that returns
a DataDesignerConfigBuilder with the full column DAG configured.

Pipeline builders are registered in PIPELINES for lookup by name.
"""

from typing import Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import data_designer.config as dd
    from bashgym.factory.data_designer import PipelineConfig

# Registry of available pipeline builders
# Each entry maps a name to a function: (PipelineConfig) -> DataDesignerConfigBuilder
PIPELINES: Dict[str, Callable] = {}


def register_pipeline(name: str):
    """Decorator to register a pipeline builder function."""
    def decorator(fn):
        PIPELINES[name] = fn
        return fn
    return decorator


# Import pipeline modules to trigger registration
try:
    from bashgym.factory.designer_pipelines.coding_agent_sft import build_sft_pipeline
    from bashgym.factory.designer_pipelines.coding_agent_dpo import build_dpo_pipeline

    PIPELINES["coding_agent_sft"] = build_sft_pipeline
    PIPELINES["coding_agent_dpo"] = build_dpo_pipeline
except ImportError:
    # data-designer not installed - pipelines unavailable
    pass
