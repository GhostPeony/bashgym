"""Factory - Data Synthesis Layer

Includes:
- DataFactory: Training data synthesis from traces
- TraceProcessor: Trace normalization and filtering
- PromptOptimizer: MIPROv2 prompt optimization
- SafeSynthesizer: Privacy-preserving data processing
- SchemaBuilder: Rich synthetic data generation
- PatternExtractor: Pattern extraction from traces for synthetic data generation
- SyntheticGenerator: Synthetic task generation from extracted patterns
- SecurityIngester: Public security dataset ingestion (EMBER, PhishTank, etc.)
- DataDesignerPipeline: NVIDIA NeMo DataDesigner v0.5.0 integration
- EmbeddingDeduplicator: Semantic deduplication via NIM embeddings
"""

from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig
from bashgym.factory.data_factory import (
    TOOL_OUTPUT_MAX_CHARS,
    TOOL_SCHEMAS,
    DataFactory,
    DataFactoryConfig,
    DPOExample,
    SynthesisStrategy,
    TrainingExample,
    build_tool_call_messages,
)
from bashgym.factory.dedup import DedupConfig, DedupResult, EmbeddingDeduplicator
from bashgym.factory.pattern_extractor import FileCluster, ToolSequence, TracePatterns
from bashgym.factory.prompt_optimizer import OptimizationResult, PromptOptConfig, PromptOptimizer
from bashgym.factory.safe_synthesizer import (
    PIIDetection,
    PrivacyReport,
    SafeSynthesizer,
    SafeSynthesizerConfig,
)
from bashgym.factory.schema_builder import ColumnType, DataDesignerClient, DataSchema, SchemaBuilder
from bashgym.factory.security_ingester import (
    ConversionMode,
    DatasetType,
    IngestionConfig,
    IngestionResult,
    SecurityDomain,
    SecurityIngester,
)
from bashgym.factory.synthetic_generator import (
    PRESETS,
    GenerationPreset,
    GenerationStrategy,
    SyntheticGenerator,
    SyntheticTask,
)
from bashgym.factory.trace_processor import ProcessedTrace, TraceProcessor, TraceQualityMetrics

__all__ = [
    # Data Factory
    "DataFactory",
    "DataFactoryConfig",
    "TrainingExample",
    "DPOExample",
    "SynthesisStrategy",
    "TOOL_SCHEMAS",
    "TOOL_OUTPUT_MAX_CHARS",
    "build_tool_call_messages",
    # Trace Processor
    "TraceProcessor",
    "ProcessedTrace",
    "TraceQualityMetrics",
    # Prompt Optimizer
    "PromptOptimizer",
    "PromptOptConfig",
    "OptimizationResult",
    # Safe Synthesizer
    "SafeSynthesizer",
    "SafeSynthesizerConfig",
    "PIIDetection",
    "PrivacyReport",
    # Schema Builder
    "SchemaBuilder",
    "DataDesignerClient",
    "DataSchema",
    "ColumnType",
    # Pattern Extractor
    "TracePatterns",
    "FileCluster",
    "ToolSequence",
    # Synthetic Generator
    "SyntheticTask",
    "GenerationStrategy",
    "GenerationPreset",
    "PRESETS",
    "SyntheticGenerator",
    # Security Dataset Ingester
    "SecurityIngester",
    "IngestionConfig",
    "IngestionResult",
    "DatasetType",
    "ConversionMode",
    "SecurityDomain",
    # DataDesigner Integration
    "DataDesignerPipeline",
    "PipelineConfig",
    # Embedding Deduplication
    "EmbeddingDeduplicator",
    "DedupConfig",
    "DedupResult",
]
