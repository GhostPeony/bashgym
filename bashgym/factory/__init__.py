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
"""

from bashgym.factory.data_factory import DataFactory, DataFactoryConfig, TrainingExample, DPOExample, SynthesisStrategy
from bashgym.factory.trace_processor import TraceProcessor, ProcessedTrace, TraceQualityMetrics
from bashgym.factory.prompt_optimizer import PromptOptimizer, PromptOptConfig, OptimizationResult
from bashgym.factory.safe_synthesizer import SafeSynthesizer, SafeSynthesizerConfig, PIIDetection, PrivacyReport
from bashgym.factory.schema_builder import SchemaBuilder, DataDesignerClient, DataSchema, ColumnType
from bashgym.factory.pattern_extractor import TracePatterns, FileCluster, ToolSequence
from bashgym.factory.synthetic_generator import SyntheticTask, GenerationStrategy, GenerationPreset, PRESETS, SyntheticGenerator
from bashgym.factory.security_ingester import SecurityIngester, IngestionConfig, IngestionResult, DatasetType, ConversionMode, SecurityDomain
from bashgym.factory.data_designer import DataDesignerPipeline, PipelineConfig

__all__ = [
    # Data Factory
    "DataFactory",
    "DataFactoryConfig",
    "TrainingExample",
    "DPOExample",
    "SynthesisStrategy",
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
]
