"""Data-synthesis APIs with lazy optional-integration exports."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "bashgym.factory.data_factory": (
        "DataFactory",
        "DataFactoryConfig",
        "TrainingExample",
        "DPOExample",
        "SynthesisStrategy",
        "TOOL_SCHEMAS",
        "TOOL_OUTPUT_MAX_CHARS",
        "build_tool_call_messages",
    ),
    "bashgym.factory.trace_processor": (
        "TraceProcessor",
        "ProcessedTrace",
        "TraceQualityMetrics",
    ),
    "bashgym.factory.prompt_optimizer": (
        "PromptOptimizer",
        "PromptOptConfig",
        "OptimizationResult",
    ),
    "bashgym.factory.safe_synthesizer": (
        "SafeSynthesizer",
        "SafeSynthesizerConfig",
        "PIIDetection",
        "PrivacyReport",
    ),
    "bashgym.factory.schema_builder": (
        "SchemaBuilder",
        "DataDesignerClient",
        "DataSchema",
        "ColumnType",
    ),
    "bashgym.factory.pattern_extractor": ("TracePatterns", "FileCluster", "ToolSequence"),
    "bashgym.factory.synthetic_generator": (
        "SyntheticTask",
        "GenerationStrategy",
        "GenerationPreset",
        "PRESETS",
        "SyntheticGenerator",
    ),
    "bashgym.factory.security_ingester": (
        "SecurityIngester",
        "IngestionConfig",
        "IngestionResult",
        "DatasetType",
        "ConversionMode",
        "SecurityDomain",
    ),
    "bashgym.factory.data_designer": ("DataDesignerPipeline", "PipelineConfig"),
    "bashgym.factory.dedup": ("EmbeddingDeduplicator", "DedupConfig", "DedupResult"),
    "bashgym.factory.session_distillation": (
        "SESSION_DISTILLATION_HINT_TAG",
        "SessionDistillationHint",
        "SessionDistillationRecord",
        "HeuristicSessionDistillationReader",
        "build_session_distillation_records",
        "build_session_distillation_records_from_traces",
        "inject_session_distillation_hint",
        "save_session_distillation_records",
        "validate_session_distillation_record",
        "validate_session_distillation_records",
    ),
}
_EXPORTS = {name: module_name for module_name, names in _MODULE_EXPORTS.items() for name in names}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
