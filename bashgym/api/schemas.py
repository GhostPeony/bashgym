"""
Bash Gym API Schemas - Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingStatus(str, Enum):
    """Status of a training run."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingStrategy(str, Enum):
    """Available training strategies."""
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


class ExportFormat(str, Enum):
    """Model export formats."""
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    LORA = "lora"


class TraceStatus(str, Enum):
    """Status of a trace."""
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    FAILED = "failed"
    PENDING = "pending"


class TraceQualityTier(str, Enum):
    """Quality tier for trace classification based on NVIDIA NeMo recommendations.

    Tier thresholds (based on research):
    - GOLD: ≥90% success rate, ≥0.75 quality → SFT training (NVIDIA min_success_rate: 0.9)
    - SILVER: ≥75% success rate, ≥0.55 quality → DPO chosen, secondary SFT
    - BRONZE: ≥60% success rate, ≥0.40 quality → DPO rejected, review candidates
    - REJECTED: <60% success rate → Not suitable for training
    """
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    REJECTED = "rejected"


class RoutingStrategyEnum(str, Enum):
    """Routing strategies for model router."""
    TEACHER_ONLY = "teacher_only"
    STUDENT_ONLY = "student_only"
    CONFIDENCE_BASED = "confidence_based"
    TASK_COMPLEXITY = "task_complexity"
    PROGRESSIVE = "progressive"
    RANDOM_SAMPLE = "random_sample"


# =============================================================================
# Task Schemas
# =============================================================================

class TaskRequest(BaseModel):
    """Request to submit a new task."""
    prompt: str = Field(..., description="Task description for the agent")
    task_id: Optional[str] = Field(None, description="Optional custom task ID")
    repository_url: Optional[str] = Field(None, description="Git repository to clone")
    timeout: int = Field(1800, description="Timeout in seconds", ge=60, le=7200)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Write a Python function that calculates fibonacci numbers",
                "timeout": 300
            }
        }


class TaskResponse(BaseModel):
    """Response for task operations."""
    task_id: str
    status: TaskStatus
    message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    trace_path: Optional[str] = None


# =============================================================================
# Training Schemas
# =============================================================================

class TrainingRequest(BaseModel):
    """Request to start a training run."""
    strategy: TrainingStrategy = Field(TrainingStrategy.SFT, description="Training strategy")
    dataset_path: Optional[str] = Field(None, description="Path to training data")
    base_model: Optional[str] = Field(None, description="Override base model")
    num_epochs: int = Field(3, description="Number of training epochs", ge=1, le=100)
    batch_size: int = Field(1, description="Training batch size (use 1 for 12GB VRAM)", ge=1, le=64)
    learning_rate: float = Field(2e-5, description="Learning rate")
    use_lora: bool = Field(True, description="Use LoRA for efficient fine-tuning")
    lora_rank: Optional[int] = Field(16, description="LoRA rank")
    lora_alpha: Optional[int] = Field(32, description="LoRA alpha")
    warmup_steps: int = Field(100, description="Warmup steps")
    max_seq_length: int = Field(2048, description="Maximum sequence length")
    auto_export_gguf: bool = Field(True, description="Export to GGUF after training")
    gguf_quantization: str = Field("q4_k_m", description="GGUF quantization level")
    use_nemo_gym: bool = Field(False, description="Use NVIDIA NeMo cloud training instead of local")
    selected_repos: Optional[List[str]] = Field(None, description="Repos to include (None or empty = all repos)")

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "sft",
                "num_epochs": 3,
                "batch_size": 4,
                "auto_export_gguf": True
            }
        }


class TrainingResponse(BaseModel):
    """Response for training operations."""
    run_id: str
    status: TrainingStatus
    strategy: TrainingStrategy
    message: Optional[str] = None
    error: Optional[str] = None  # Error message if training failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    output_path: Optional[str] = None


# =============================================================================
# Model Schemas
# =============================================================================

class ModelInfo(BaseModel):
    """Information about a trained model."""
    model_id: str
    path: str
    created_at: str
    base_model: Optional[str] = None
    strategy: Optional[str] = None
    has_gguf: bool = False
    gguf_path: Optional[str] = None


class EvaluationRequest(BaseModel):
    """Request to run model evaluation."""
    model_id: str = Field(..., description="Model ID from /api/models")
    benchmarks: List[str] = Field(..., description="Benchmark IDs to run")
    num_samples: int = Field(5, description="Samples per benchmark", ge=1, le=100)


class ErrorAnalysisSchema(BaseModel):
    """Breakdown of error types in a benchmark run."""
    wrong_answer: int = Field(0, description="Tests failed due to incorrect output")
    syntax_error: int = Field(0, description="Code had syntax errors")
    runtime_error: int = Field(0, description="Code raised runtime exceptions")
    timeout: int = Field(0, description="Execution exceeded time limit")
    other: int = Field(0, description="Other/unknown errors")


class BenchmarkResultSchema(BaseModel):
    """Result of a single benchmark."""
    score: float = Field(..., description="Pass rate (0-1)")
    passed: int = Field(..., description="Number of passed samples")
    total: int = Field(..., description="Total samples")
    duration_seconds: float = Field(0, description="Time taken in seconds")
    errors: Optional[ErrorAnalysisSchema] = Field(None, description="Error breakdown")


class EvaluationResponse(BaseModel):
    """Response for evaluation operations."""
    job_id: str
    model_id: str
    benchmarks: List[str]
    status: str  # "pending", "running", "completed", "failed"
    results: Optional[Dict[str, BenchmarkResultSchema]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None


class ExportRequest(BaseModel):
    """Request to export a model."""
    format: ExportFormat = Field(ExportFormat.GGUF, description="Export format")
    quantization: str = Field("q4_k_m", description="GGUF quantization level")

    class Config:
        json_schema_extra = {
            "example": {
                "format": "gguf",
                "quantization": "q4_k_m"
            }
        }


class ExportResponse(BaseModel):
    """Response for model export."""
    model_id: str
    format: ExportFormat
    status: str
    message: Optional[str] = None
    output_path: Optional[str] = None


# =============================================================================
# Trace Schemas
# =============================================================================

class TraceQuality(BaseModel):
    """Quality metrics for a trace."""
    success_rate: float = Field(0.0, ge=0.0, le=1.0)
    verification_score: float = Field(0.0, ge=0.0, le=1.0)
    complexity_score: float = Field(0.0, ge=0.0, le=1.0)
    length_score: float = Field(0.0, ge=0.0, le=1.0)
    tool_diversity: float = Field(0.0, ge=0.0, le=1.0, description="Unique tools used score")
    efficiency_score: float = Field(0.0, ge=0.0, le=1.0, description="Error recovery and output quality")
    total_score: float = Field(0.0, ge=0.0, le=1.0)


class RepoInfo(BaseModel):
    """Repository information for trace filtering."""
    name: str = Field(..., description="Repository directory name")
    path: Optional[str] = Field(None, description="Full path to repository")
    git_remote: Optional[str] = Field(None, description="Git remote URL if available")
    git_branch: Optional[str] = Field(None, description="Git branch at time of trace")
    is_git_repo: bool = Field(False, description="Whether this is a git repository")


class TraceStep(BaseModel):
    """A single step in a trace."""
    index: int
    tool: str
    command: Optional[str] = None
    output: Optional[str] = None
    success: Optional[bool] = None
    timestamp: Optional[str] = None


class TraceInfo(BaseModel):
    """Information about a trace."""
    trace_id: str
    task_id: str
    task_description: str
    status: TraceStatus
    quality_tier: Optional[TraceQualityTier] = Field(
        None,
        description="Quality tier classification (gold/silver/bronze/rejected)"
    )
    steps_count: int = 0
    quality: TraceQuality
    repo: Optional[RepoInfo] = Field(None, description="Primary repository for this trace")
    repos_count: int = Field(0, description="Number of repositories touched in this trace")
    created_at: Optional[str] = None
    promoted_at: Optional[str] = None


class TraceDetail(TraceInfo):
    """Detailed trace information including steps."""
    steps: List[TraceStep] = []
    metadata: Dict[str, Any] = {}


# =============================================================================
# Training Example Schemas
# =============================================================================

class TrainingExampleResponse(BaseModel):
    """A single training example generated from a trace session."""
    example_id: str = Field(..., description="Unique example identifier")
    system_prompt: str = Field(..., description="System prompt for the model")
    user_prompt: str = Field(..., description="User task prompt")
    assistant_response: str = Field(..., description="Expected assistant response")
    step_count: int = Field(0, description="Number of steps in this example")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Success rate of steps")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in segmentation")
    source_trace_id: Optional[str] = Field(None, description="Source trace ID")


class GenerateExamplesRequest(BaseModel):
    """Request to generate training examples from a trace."""
    min_success_rate: float = Field(0.5, ge=0.0, le=1.0, description="Minimum success rate filter")
    max_steps_per_example: int = Field(50, ge=5, le=100, description="Maximum steps per example")


class GenerateExamplesResponse(BaseModel):
    """Response from example generation."""
    trace_id: str
    examples: List[TrainingExampleResponse]
    total_steps: int = 0
    examples_generated: int = 0


class ExportExamplesRequest(BaseModel):
    """Request to export training examples to JSONL."""
    trace_ids: Optional[List[str]] = Field(None, description="Specific trace IDs to export (None = all)")
    include_gold_only: bool = Field(True, description="Only include gold traces")
    train_split: float = Field(0.9, ge=0.5, le=1.0, description="Training set proportion")


class ExportExamplesResponse(BaseModel):
    """Response from export operation."""
    success: bool
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    train_count: int = 0
    val_count: int = 0
    message: Optional[str] = None


# =============================================================================
# Router Schemas
# =============================================================================

class RouterStats(BaseModel):
    """Router statistics."""
    total_requests: int = 0
    teacher_requests: int = 0
    student_requests: int = 0
    teacher_success_rate: float = 0.0
    student_success_rate: float = 0.0
    avg_teacher_latency: float = 0.0
    avg_student_latency: float = 0.0
    current_student_rate: float = 0.1


class RoutingDecisionInfo(BaseModel):
    """Information about a routing decision."""
    request_id: str
    model: str
    model_type: str
    confidence: float
    task_complexity: float
    timestamp: str


# =============================================================================
# System Schemas
# =============================================================================

class HealthCheck(BaseModel):
    """API health check response."""
    status: str
    timestamp: str
    version: str


class SystemStats(BaseModel):
    """System statistics."""
    gold_traces_count: int
    silver_traces_count: int = 0
    bronze_traces_count: int = 0
    failed_traces_count: int
    pending_traces_count: int = 0
    models_count: int
    base_model: str
    auto_export_gguf: bool
    active_tasks: int = 0
    active_training_runs: int = 0


# =============================================================================
# WebSocket Schemas
# =============================================================================

class WSMessage(BaseModel):
    """WebSocket message structure."""
    type: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None


class TrainingProgress(BaseModel):
    """Training progress update."""
    run_id: str
    epoch: int
    total_epochs: int
    step: int = 0
    total_steps: int = 0
    loss: float
    learning_rate: float
    grad_norm: float = 0.0
    eta: Optional[str] = None


# =============================================================================
# Hardware Info Schemas
# =============================================================================

class GpuInfo(BaseModel):
    """GPU information."""
    vendor: str = Field(..., description="GPU vendor (NVIDIA, AMD, Intel, Apple)")
    model: str = Field(..., description="GPU model name")
    vram: float = Field(0.0, description="VRAM in GB")
    vram_used: Optional[float] = Field(None, description="Used VRAM in GB")
    driver: Optional[str] = Field(None, description="Driver version")
    temperature: Optional[float] = Field(None, description="GPU temperature in Celsius")
    utilization: Optional[float] = Field(None, description="GPU utilization percentage")


class SystemInfoResponse(BaseModel):
    """Full system hardware information."""
    gpus: List[GpuInfo] = Field(default_factory=list, description="List of detected GPUs")
    total_ram: float = Field(0.0, description="Total system RAM in GB")
    available_ram: float = Field(0.0, description="Available system RAM in GB")
    platform: str = Field("", description="Operating system (win32, darwin, linux)")
    arch: str = Field("", description="System architecture (x64, arm64)")
    cuda_available: bool = Field(False, description="Whether CUDA is available")
    cuda_version: Optional[str] = Field(None, description="CUDA version if available")
    python_available: bool = Field(True, description="Whether Python is available")
    python_version: Optional[str] = Field(None, description="Python version")


class ModelRecommendations(BaseModel):
    """Model recommendations based on hardware."""
    max_vram_gb: float = Field(0.0, description="Maximum VRAM available in GB")
    cuda_available: bool = Field(False, description="Whether CUDA is available")
    recommended_models: List[str] = Field(default_factory=list, description="Recommended model IDs")
    recommended_quantization: str = Field("4bit", description="Recommended quantization")
    recommended_batch_size: int = Field(1, description="Recommended batch size")
    warning: Optional[str] = Field(None, description="Warning message if limitations exist")


# =============================================================================
# Factory Schemas (Data Designer, Privacy, Prompt Optimization)
# =============================================================================

class ColumnType(str, Enum):
    """Column types for Data Designer."""
    LLM = "llm"
    SAMPLER = "sampler"
    CATEGORY = "category"
    PERSON = "person"
    DATETIME = "datetime"
    EXPRESSION = "expression"
    UUID = "uuid"
    GAUSSIAN = "gaussian"
    VALIDATOR = "validator"


class RiskLevel(str, Enum):
    """Risk level for safety marking."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"


class SynthesisJobStatus(str, Enum):
    """Status of a synthesis job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SynthesisJobType(str, Enum):
    """Type of synthesis job."""
    PREVIEW = "preview"
    FULL = "full"


class SeedSource(str, Enum):
    """Source of seed example."""
    MANUAL = "manual"
    IMPORTED = "imported"
    GOLD_TRACE = "gold_trace"


class ReplacementStrategy(str, Enum):
    """PII replacement strategy."""
    SYNTHETIC = "synthetic"
    MASK = "mask"
    HASH = "hash"


class OutputFormat(str, Enum):
    """Output format for synthesis."""
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


# Model configuration for LLM columns
class ModelConfig(BaseModel):
    """Configuration for LLM model."""
    model_id: str = Field("meta/llama-3.1-8b-instruct", description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1024, ge=1, le=8192, description="Max output tokens")
    system_prompt: Optional[str] = Field(None, description="System prompt override")


# Column dependency rule
class ColumnDependency(BaseModel):
    """Dependency rule between columns."""
    depends_on: str = Field(..., description="Column ID this depends on")
    condition: str = Field("exists", description="Condition type: equals, not_equals, contains, exists")
    value: Optional[str] = Field(None, description="Value to compare against")
    required_when_true: bool = Field(True, description="Whether this column is required when condition is true")


# Validation constraint
class ColumnConstraint(BaseModel):
    """Validation constraint for column."""
    type: str = Field(..., description="Constraint type: enum, regex, json_schema, min_length, max_length")
    value: Any = Field(..., description="Constraint value")
    error_message: Optional[str] = Field(None, description="Custom error message")


class ColumnConfig(BaseModel):
    """Configuration for a Data Designer column."""
    id: str = Field(..., description="Unique column identifier")
    name: str = Field(..., description="Column name")
    type: ColumnType = Field(ColumnType.LLM, description="Column type")
    description: Optional[str] = Field(None, description="Column description")
    required: bool = Field(True, description="Whether column is required")
    risk_level: RiskLevel = Field(RiskLevel.NORMAL, description="Safety risk level")
    config: Dict[str, Any] = Field(default_factory=dict, description="Type-specific config")
    dependencies: List[ColumnDependency] = Field(default_factory=list, description="Column dependencies")
    constraints: List[ColumnConstraint] = Field(default_factory=list, description="Validation constraints")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "col_123",
                "name": "cli_command",
                "type": "llm",
                "description": "Generated CLI command",
                "required": True,
                "risk_level": "elevated",
                "config": {
                    "prompt": "Generate a CLI command for: {{user_request}}",
                    "model": {"model_id": "meta/llama-3.1-8b-instruct", "temperature": 0.3}
                },
                "constraints": [{"type": "regex", "value": "^[a-z]", "error_message": "Must start with lowercase"}]
            }
        }


class SeedExample(BaseModel):
    """Seed example for training data generation."""
    id: str = Field(..., description="Unique seed identifier")
    data: Dict[str, str] = Field(..., description="Column name to value mapping")
    source: SeedSource = Field(SeedSource.MANUAL, description="Source of seed")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    trace_id: Optional[str] = Field(None, description="Source trace ID if from gold trace")


class PrivacyConfig(BaseModel):
    """Privacy settings for Safe Synthesizer."""
    enabled: bool = Field(False, description="Enable privacy-preserving synthesis")
    epsilon: float = Field(8.0, ge=1.0, le=16.0, description="Differential privacy budget")
    pii_types: List[str] = Field(
        default_factory=lambda: ["person", "email", "ssn"],
        description="PII types to detect and replace"
    )
    replacement_strategy: ReplacementStrategy = Field(
        ReplacementStrategy.SYNTHETIC,
        description="How to replace detected PII"
    )


class PromptOptConfig(BaseModel):
    """Configuration for MIPROv2 prompt optimization."""
    enabled: bool = Field(False, description="Enable prompt optimization")
    intensity: str = Field("medium", description="Optimization intensity (light/medium/heavy)")
    max_demos: int = Field(4, ge=1, le=10, description="Max bootstrapped demos")
    metric_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Stop when metric reaches this")
    target_metric: str = Field("accuracy", description="Target metric: accuracy, f1, custom")


class OutputConfig(BaseModel):
    """Output configuration for synthesis."""
    row_count: int = Field(1000, ge=10, le=100000, description="Number of rows to generate")
    format: OutputFormat = Field(OutputFormat.JSONL, description="Output format")
    task_name: str = Field("default_task", description="Task name for GRPO integration")
    include_task_name: bool = Field(True, description="Include task_name in output")
    train_val_split: float = Field(0.9, ge=0.5, le=1.0, description="Training portion")
    include_negative_examples: bool = Field(False, description="Generate negative examples")
    negative_example_ratio: float = Field(0.1, ge=0.0, le=0.5, description="Ratio of negative examples")


class SafetyConfig(BaseModel):
    """Safety configuration for synthesis."""
    enabled: bool = Field(True, description="Enable safety checks")
    block_dangerous_commands: bool = Field(True, description="Block dangerous shell commands")
    require_confirmation_for_high_risk: bool = Field(True, description="Require confirmation for high risk")
    max_risk_level: RiskLevel = Field(RiskLevel.HIGH, description="Maximum allowed risk level")
    blocked_patterns: List[str] = Field(
        default_factory=lambda: ["rm -rf", "sudo", "chmod 777", ":(){:|:&};:"],
        description="Regex patterns to block"
    )


class FactoryConfig(BaseModel):
    """Full factory configuration."""
    columns: List[ColumnConfig] = Field(default_factory=list, description="Column schema")
    seeds: List[SeedExample] = Field(default_factory=list, description="Seed examples")
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig, description="Privacy settings")
    prompt_optimization: PromptOptConfig = Field(
        default_factory=PromptOptConfig,
        description="Prompt optimization settings"
    )
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    safety: SafetyConfig = Field(default_factory=SafetyConfig, description="Safety configuration")
    default_model: ModelConfig = Field(default_factory=ModelConfig, description="Default model config")


class PreviewRow(BaseModel):
    """Single row in preview result."""
    id: str
    data: Dict[str, str]
    validation_errors: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class PreviewResult(BaseModel):
    """Result of preview generation."""
    rows: List[PreviewRow]
    total_generated: int
    valid_count: int
    invalid_count: int
    validation_summary: Dict[str, int] = Field(default_factory=dict)
    column_coverage: Dict[str, float] = Field(default_factory=dict)


class SynthesisJob(BaseModel):
    """Synthesis job information."""
    id: str
    status: SynthesisJobStatus
    job_type: SynthesisJobType = Field(SynthesisJobType.FULL)
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    examples_created: Optional[int] = None
    valid_examples: Optional[int] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    config_snapshot: Optional[Dict[str, Any]] = None


class AvailableModel(BaseModel):
    """Available model for synthesis."""
    id: str
    name: str
    provider: str


# =============================================================================
# Hooks Schemas
# =============================================================================

class HooksInstallRequest(BaseModel):
    """Request to install trace capture hooks."""
    tools: Optional[List[str]] = Field(
        None,
        description="List of tools to install hooks for (e.g., ['claude_code', 'opencode']). If None, auto-detect."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tools": ["claude_code", "opencode"]
            }
        }


class HooksToolStatus(BaseModel):
    """Status of a single tool's hooks."""
    name: str
    installed: bool
    hooks_installed: bool
    adapter_type: str
    hooks_path: Optional[str] = None


class HooksStatusResponse(BaseModel):
    """Response for hooks status check."""
    tools: List[HooksToolStatus]
    summary: Dict[str, Any]


class HooksInstallResponse(BaseModel):
    """Response for hooks installation."""
    success: bool
    tools: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
