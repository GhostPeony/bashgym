"""
Bash Gym API Schemas - Pydantic models for request/response validation
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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
    RLVR = "rlvr"
    DISTILLATION = "distillation"


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
    task_id: str | None = Field(None, description="Optional custom task ID")
    repository_url: str | None = Field(None, description="Git repository to clone")
    timeout: int = Field(1800, description="Timeout in seconds", ge=60, le=7200)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Write a Python function that calculates fibonacci numbers",
                "timeout": 300,
            }
        }


class TaskResponse(BaseModel):
    """Response for task operations."""

    task_id: str
    status: TaskStatus
    message: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    result: dict[str, Any] | None = None
    trace_path: str | None = None


# =============================================================================
# Training Schemas
# =============================================================================


class DataSource(str, Enum):
    """Source of training data."""

    TRACES = "traces"
    DATASET_PATH = "dataset_path"
    SECURITY_DATASET = "security_dataset"


class TrainingRequest(BaseModel):
    """Request to start a training run."""

    strategy: TrainingStrategy = Field(TrainingStrategy.SFT, description="Training strategy")
    dataset_path: str | None = Field(None, description="Path to training data")
    base_model: str | None = Field(None, description="Override base model")
    model_type: str | None = Field(
        None,
        description="Model architecture type (qwen, llama, mistral, phi). Auto-detected if omitted.",
    )
    num_epochs: int = Field(3, description="Number of training epochs", ge=1, le=100)
    batch_size: int = Field(1, description="Training batch size (use 1 for 12GB VRAM)", ge=1, le=64)
    learning_rate: float = Field(2e-5, description="Learning rate")
    warmup_ratio: float = Field(
        0.1, description="Warmup ratio (fraction of total steps)", ge=0.0, le=1.0
    )
    gradient_accumulation_steps: int = Field(
        8,
        description="Gradient accumulation steps (effective batch = batch_size * this)",
        ge=1,
        le=128,
    )
    max_seq_length: int = Field(2048, description="Maximum sequence length")
    save_steps: int = Field(100, description="Save checkpoint every N steps", ge=10)
    # LoRA settings
    use_lora: bool = Field(True, description="Use LoRA for efficient fine-tuning")
    lora_rank: int | None = Field(16, description="LoRA rank")
    lora_alpha: int | None = Field(32, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout rate", ge=0.0, le=0.5)
    load_in_4bit: bool = Field(True, description="Load model in 4-bit quantization (QLoRA)")
    # Strategy-specific settings
    dpo_beta: float = Field(
        0.1, description="DPO beta parameter (controls divergence penalty)", ge=0.01, le=1.0
    )
    grpo_num_generations: int = Field(
        4, description="GRPO: number of generations per prompt", ge=2, le=16
    )
    grpo_temperature: float = Field(0.7, description="GRPO: sampling temperature", ge=0.1, le=2.0)
    grpo_reward_mode: str = Field(
        "syntax", description="GRPO reward mode: syntax, execution, or verification"
    )
    # Knowledge Distillation settings
    teacher_model: str | None = Field(None, description="Teacher model for distillation")
    teacher_temperature: float = Field(
        0.7, description="KD: softmax temperature for soft labels", ge=0.1, le=10.0
    )
    distillation_alpha: float = Field(
        0.5,
        description="KD: weight for soft labels vs hard labels (0=task only, 1=KD only)",
        ge=0.0,
        le=1.0,
    )
    # Export settings
    auto_export_gguf: bool = Field(True, description="Export to GGUF after training")
    gguf_quantization: str = Field("q4_k_m", description="GGUF quantization level")
    # Backend selection
    use_nemo_gym: bool = Field(False, description="Use NVIDIA NeMo cloud training instead of local")
    use_remote_ssh: bool = Field(False, description="Execute training on remote DGX Spark via SSH")
    device_id: str | None = Field(
        None, description="Target device ID for remote SSH training (uses default if omitted)"
    )
    selected_repos: list[str] | None = Field(
        None, description="Repos to include (None or empty = all repos)"
    )
    # Data source selection
    data_source: DataSource = Field(DataSource.TRACES, description="Source of training data")
    security_dataset_type: str | None = Field(
        None, description="Security dataset type (ember, phishtank, etc.)"
    )
    security_dataset_path: str | None = Field(None, description="Path to security dataset file")
    security_conversion_mode: str | None = Field(
        "direct", description="Security dataset conversion mode (direct/enriched)"
    )
    security_max_samples: int | None = Field(
        None, description="Max samples for security dataset ingestion"
    )
    security_balance_classes: bool = Field(True, description="Balance classes in security dataset")

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "sft",
                "num_epochs": 3,
                "batch_size": 4,
                "auto_export_gguf": True,
                "data_source": "traces",
            }
        }


class TrainingResponse(BaseModel):
    """Response for training operations."""

    run_id: str
    status: TrainingStatus
    strategy: TrainingStrategy
    message: str | None = None
    error: str | None = None  # Error message if training failed
    started_at: str | None = None
    completed_at: str | None = None
    metrics: dict[str, float] | None = None
    output_path: str | None = None


# =============================================================================
# Model Schemas
# =============================================================================


class ModelInfo(BaseModel):
    """Information about a trained model."""

    model_id: str
    path: str
    created_at: str
    base_model: str | None = None
    strategy: str | None = None
    has_gguf: bool = False
    gguf_path: str | None = None


class EvaluationRequest(BaseModel):
    """Request to run model evaluation."""

    model_id: str = Field(..., description="Model ID from /api/models")
    benchmarks: list[str] = Field(..., description="Benchmark IDs to run")
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
    errors: ErrorAnalysisSchema | None = Field(None, description="Error breakdown")


class EvaluationResponse(BaseModel):
    """Response for evaluation operations."""

    job_id: str
    model_id: str
    benchmarks: list[str]
    status: str  # "pending", "running", "completed", "failed"
    results: dict[str, BenchmarkResultSchema] | None = None
    error: str | None = None
    created_at: str | None = None


class ExportRequest(BaseModel):
    """Request to export a model."""

    format: ExportFormat = Field(ExportFormat.GGUF, description="Export format")
    quantization: str = Field("q4_k_m", description="GGUF quantization level")

    class Config:
        json_schema_extra = {"example": {"format": "gguf", "quantization": "q4_k_m"}}


class ExportResponse(BaseModel):
    """Response for model export."""

    model_id: str
    format: ExportFormat
    status: str
    message: str | None = None
    output_path: str | None = None


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
    efficiency_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Error recovery and output quality"
    )
    cognitive_quality: float = Field(
        0.0, ge=0.0, le=1.0, description="Thinking, planning, and reflection quality"
    )
    total_score: float = Field(0.0, ge=0.0, le=1.0)


class RepoInfo(BaseModel):
    """Repository information for trace filtering."""

    name: str = Field(..., description="Repository directory name")
    path: str | None = Field(None, description="Full path to repository")
    git_remote: str | None = Field(None, description="Git remote URL if available")
    git_branch: str | None = Field(None, description="Git branch at time of trace")
    is_git_repo: bool = Field(False, description="Whether this is a git repository")


class TraceStep(BaseModel):
    """A single step in a trace."""

    index: int
    tool: str
    command: str | None = None
    output: str | None = None
    success: bool | None = None
    timestamp: str | None = None


class TraceInfo(BaseModel):
    """Information about a trace."""

    trace_id: str
    task_id: str
    task_description: str
    status: TraceStatus
    quality_tier: TraceQualityTier | None = Field(
        None, description="Quality tier classification (gold/silver/bronze/rejected)"
    )
    steps_count: int = 0
    quality: TraceQuality
    repo: RepoInfo | None = Field(None, description="Primary repository for this trace")
    repos_count: int = Field(0, description="Number of repositories touched in this trace")
    source_tool: str = Field(
        "unknown",
        description="Source tool that generated this trace (claude_code, gemini_cli, copilot_cli, opencode, codex)",
    )
    tool_breakdown: dict[str, int] = Field(default_factory=dict, description="Per-tool call counts")
    created_at: str | None = None
    promoted_at: str | None = None


class TraceSummaryDetail(TraceInfo):
    """Enriched trace info with session metrics for promote/demote decisions."""

    duration_seconds: float | None = Field(
        None, description="Session duration from first to last step"
    )
    step_outcomes: list[bool | None] = Field(
        default_factory=list, description="Per-step pass/fail for spark chart"
    )
    cognitive_summary: dict[str, Any] | None = Field(
        None,
        description="Cognitive metrics: planning_phases, reflections, thinking_steps, cognitive_coverage",
    )
    raw_metrics: dict[str, Any] | None = Field(
        None,
        description="Aggregate step metrics: total_steps, successful_steps, failed_steps, unique_tools, unique_commands, cognitive_steps",
    )


class TraceDetail(TraceInfo):
    """Detailed trace information including steps."""

    steps: list[TraceStep] = []
    metadata: dict[str, Any] = {}


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
    source_trace_id: str | None = Field(None, description="Source trace ID")


class GenerateExamplesRequest(BaseModel):
    """Request to generate training examples from a trace."""

    min_success_rate: float = Field(0.5, ge=0.0, le=1.0, description="Minimum success rate filter")
    max_steps_per_example: int = Field(50, ge=5, le=100, description="Maximum steps per example")


class GenerateExamplesResponse(BaseModel):
    """Response from example generation."""

    trace_id: str
    examples: list[TrainingExampleResponse]
    total_steps: int = 0
    examples_generated: int = 0


class ExportExamplesRequest(BaseModel):
    """Request to export training examples to JSONL."""

    trace_ids: list[str] | None = Field(
        None, description="Specific trace IDs to export (None = all)"
    )
    include_gold_only: bool = Field(True, description="Only include gold traces")
    train_split: float = Field(0.9, ge=0.5, le=1.0, description="Training set proportion")


class ExportExamplesResponse(BaseModel):
    """Response from export operation."""

    success: bool
    train_path: str | None = None
    val_path: str | None = None
    train_count: int = 0
    val_count: int = 0
    message: str | None = None


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
    payload: dict[str, Any]
    timestamp: str | None = None


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
    eta: str | None = None


# =============================================================================
# Hardware Info Schemas
# =============================================================================


class GpuInfo(BaseModel):
    """GPU information."""

    vendor: str = Field(..., description="GPU vendor (NVIDIA, AMD, Intel, Apple)")
    model: str = Field(..., description="GPU model name")
    vram: float = Field(0.0, description="VRAM in GB")
    vram_used: float | None = Field(None, description="Used VRAM in GB")
    driver: str | None = Field(None, description="Driver version")
    temperature: float | None = Field(None, description="GPU temperature in Celsius")
    utilization: float | None = Field(None, description="GPU utilization percentage")


class SystemInfoResponse(BaseModel):
    """Full system hardware information."""

    gpus: list[GpuInfo] = Field(default_factory=list, description="List of detected GPUs")
    total_ram: float = Field(0.0, description="Total system RAM in GB")
    available_ram: float = Field(0.0, description="Available system RAM in GB")
    platform: str = Field("", description="Operating system (win32, darwin, linux)")
    arch: str = Field("", description="System architecture (x64, arm64)")
    cuda_available: bool = Field(False, description="Whether CUDA is available")
    cuda_version: str | None = Field(None, description="CUDA version if available")
    python_available: bool = Field(True, description="Whether Python is available")
    python_version: str | None = Field(None, description="Python version")


class ModelRecommendations(BaseModel):
    """Model recommendations based on hardware."""

    max_vram_gb: float = Field(0.0, description="Maximum VRAM available in GB")
    cuda_available: bool = Field(False, description="Whether CUDA is available")
    recommended_models: list[str] = Field(default_factory=list, description="Recommended model IDs")
    recommended_quantization: str = Field("4bit", description="Recommended quantization")
    recommended_batch_size: int = Field(1, description="Recommended batch size")
    warning: str | None = Field(None, description="Warning message if limitations exist")


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
    system_prompt: str | None = Field(None, description="System prompt override")


# Column dependency rule
class ColumnDependency(BaseModel):
    """Dependency rule between columns."""

    depends_on: str = Field(..., description="Column ID this depends on")
    condition: str = Field(
        "exists", description="Condition type: equals, not_equals, contains, exists"
    )
    value: str | None = Field(None, description="Value to compare against")
    required_when_true: bool = Field(
        True, description="Whether this column is required when condition is true"
    )


# Validation constraint
class ColumnConstraint(BaseModel):
    """Validation constraint for column."""

    type: str = Field(
        ..., description="Constraint type: enum, regex, json_schema, min_length, max_length"
    )
    value: Any = Field(..., description="Constraint value")
    error_message: str | None = Field(None, description="Custom error message")


class ColumnConfig(BaseModel):
    """Configuration for a Data Designer column."""

    id: str = Field(..., description="Unique column identifier")
    name: str = Field(..., description="Column name")
    type: ColumnType = Field(ColumnType.LLM, description="Column type")
    description: str | None = Field(None, description="Column description")
    required: bool = Field(True, description="Whether column is required")
    risk_level: RiskLevel = Field(RiskLevel.NORMAL, description="Safety risk level")
    config: dict[str, Any] = Field(default_factory=dict, description="Type-specific config")
    dependencies: list[ColumnDependency] = Field(
        default_factory=list, description="Column dependencies"
    )
    constraints: list[ColumnConstraint] = Field(
        default_factory=list, description="Validation constraints"
    )

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
                    "model": {"model_id": "meta/llama-3.1-8b-instruct", "temperature": 0.3},
                },
                "constraints": [
                    {
                        "type": "regex",
                        "value": "^[a-z]",
                        "error_message": "Must start with lowercase",
                    }
                ],
            }
        }


class SeedExample(BaseModel):
    """Seed example for training data generation."""

    id: str = Field(..., description="Unique seed identifier")
    data: dict[str, str] = Field(..., description="Column name to value mapping")
    source: SeedSource = Field(SeedSource.MANUAL, description="Source of seed")
    created_at: str | None = Field(None, description="Creation timestamp")
    trace_id: str | None = Field(None, description="Source trace ID if from gold trace")


class PrivacyConfig(BaseModel):
    """Privacy settings for Safe Synthesizer."""

    enabled: bool = Field(False, description="Enable privacy-preserving synthesis")
    epsilon: float = Field(8.0, ge=1.0, le=16.0, description="Differential privacy budget")
    pii_types: list[str] = Field(
        default_factory=lambda: ["person", "email", "ssn"],
        description="PII types to detect and replace",
    )
    replacement_strategy: ReplacementStrategy = Field(
        ReplacementStrategy.SYNTHETIC, description="How to replace detected PII"
    )


class PromptOptConfig(BaseModel):
    """Configuration for MIPROv2 prompt optimization."""

    enabled: bool = Field(False, description="Enable prompt optimization")
    intensity: str = Field("medium", description="Optimization intensity (light/medium/heavy)")
    max_demos: int = Field(4, ge=1, le=10, description="Max bootstrapped demos")
    metric_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Stop when metric reaches this"
    )
    target_metric: str = Field("accuracy", description="Target metric: accuracy, f1, custom")


class OutputConfig(BaseModel):
    """Output configuration for synthesis."""

    row_count: int = Field(1000, ge=10, le=100000, description="Number of rows to generate")
    format: OutputFormat = Field(OutputFormat.JSONL, description="Output format")
    task_name: str = Field("default_task", description="Task name for GRPO integration")
    include_task_name: bool = Field(True, description="Include task_name in output")
    train_val_split: float = Field(0.9, ge=0.5, le=1.0, description="Training portion")
    include_negative_examples: bool = Field(False, description="Generate negative examples")
    negative_example_ratio: float = Field(
        0.1, ge=0.0, le=0.5, description="Ratio of negative examples"
    )


class SafetyConfig(BaseModel):
    """Safety configuration for synthesis."""

    enabled: bool = Field(True, description="Enable safety checks")
    block_dangerous_commands: bool = Field(True, description="Block dangerous shell commands")
    require_confirmation_for_high_risk: bool = Field(
        True, description="Require confirmation for high risk"
    )
    max_risk_level: RiskLevel = Field(RiskLevel.HIGH, description="Maximum allowed risk level")
    blocked_patterns: list[str] = Field(
        default_factory=lambda: ["rm -rf", "sudo", "chmod 777", ":(){:|:&};:"],
        description="Regex patterns to block",
    )


class FactoryConfig(BaseModel):
    """Full factory configuration."""

    columns: list[ColumnConfig] = Field(default_factory=list, description="Column schema")
    seeds: list[SeedExample] = Field(default_factory=list, description="Seed examples")
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig, description="Privacy settings")
    prompt_optimization: PromptOptConfig = Field(
        default_factory=PromptOptConfig, description="Prompt optimization settings"
    )
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    safety: SafetyConfig = Field(default_factory=SafetyConfig, description="Safety configuration")
    default_model: ModelConfig = Field(
        default_factory=ModelConfig, description="Default model config"
    )


class PreviewRow(BaseModel):
    """Single row in preview result."""

    id: str
    data: dict[str, str]
    validation_errors: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class PreviewResult(BaseModel):
    """Result of preview generation."""

    rows: list[PreviewRow]
    total_generated: int
    valid_count: int
    invalid_count: int
    validation_summary: dict[str, int] = Field(default_factory=dict)
    column_coverage: dict[str, float] = Field(default_factory=dict)


class SynthesisJob(BaseModel):
    """Synthesis job information."""

    id: str
    status: SynthesisJobStatus
    job_type: SynthesisJobType = Field(SynthesisJobType.FULL)
    created_at: str | None = None
    completed_at: str | None = None
    examples_created: int | None = None
    valid_examples: int | None = None
    output_path: str | None = None
    error: str | None = None
    config_snapshot: dict[str, Any] | None = None


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

    tools: list[str] | None = Field(
        None,
        description="List of tools to install hooks for (e.g., ['claude_code', 'opencode']). If None, auto-detect.",
    )

    class Config:
        json_schema_extra = {"example": {"tools": ["claude_code", "opencode"]}}


class HooksToolStatus(BaseModel):
    """Status of a single tool's hooks."""

    name: str
    installed: bool
    hooks_installed: bool
    adapter_type: str
    hooks_path: str | None = None


class HooksStatusResponse(BaseModel):
    """Response for hooks status check."""

    tools: list[HooksToolStatus]
    summary: dict[str, Any]


class HooksInstallResponse(BaseModel):
    """Response for hooks installation."""

    success: bool
    tools: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace Import
# ---------------------------------------------------------------------------


class TraceImportRequest(BaseModel):
    """Request body for trace import endpoints."""

    days: int = 60
    limit: int = 100
    force: bool = False


class TraceImportResponse(BaseModel):
    """Per-source import result."""

    source: str
    imported: int
    skipped: int
    errors: int
    total: int
    new_trace_ids: list[str] = Field(default_factory=list)


class TraceImportAllResponse(BaseModel):
    """Aggregated result from importing all sources."""

    results: list[dict[str, Any]] = Field(default_factory=list)
    total_imported: int = 0


# =============================================================================
# AutoResearch Schemas
# =============================================================================


class AutoResearchRequest(BaseModel):
    """Request to start an autoresearch hyperparameter search."""

    search_params: list[str] = Field(
        default_factory=lambda: ["learning_rate", "lora_r", "lora_alpha", "warmup_ratio"],
        description="TrainerConfig fields to search over",
    )
    max_experiments: int = Field(50, ge=1, le=500, description="Maximum number of experiments")
    train_steps: int = Field(100, ge=10, le=10000, description="Training steps per experiment")
    dataset_subset_ratio: float = Field(
        0.1, ge=0.01, le=1.0, description="Fraction of training data per experiment"
    )
    eval_metric: str = Field(
        "val_loss", description="Metric to optimize (lower is better for loss)"
    )
    mutation_rate: float = Field(
        0.3, ge=0.05, le=1.0, description="Probability of mutating each param"
    )
    mutation_scale: float = Field(0.2, ge=0.01, le=1.0, description="Scale of mutations")
    base_config: dict[str, Any] | None = Field(
        None, description="Override base TrainerConfig values (e.g. learning_rate, lora_r)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "search_params": ["learning_rate", "lora_r", "lora_alpha", "warmup_ratio"],
                "max_experiments": 50,
                "train_steps": 100,
                "mutation_rate": 0.3,
                "mutation_scale": 0.2,
                "base_config": {"learning_rate": 2e-5, "lora_r": 16},
            }
        }


class ExperimentResultSchema(BaseModel):
    """Result of a single autoresearch experiment."""

    experiment_id: int = Field(..., description="Experiment number (1-indexed)")
    config_snapshot: dict[str, Any] = Field(..., description="Hyperparameter values used")
    metric_value: float = Field(..., description="Evaluation metric result")
    improved: bool = Field(..., description="Whether this beat the previous best")
    duration_seconds: float = Field(..., description="Wall-clock time for this experiment")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class AutoResearchStatusResponse(BaseModel):
    """Full autoresearch status response."""

    status: str = Field(
        ..., description="Current status: idle, running, paused, completed, failed, stopped"
    )
    total_experiments: int = Field(0, description="Total experiments planned")
    completed_experiments: int = Field(0, description="Experiments completed so far")
    best_metric: float | None = Field(None, description="Best metric value achieved")
    best_config: dict[str, Any] = Field(
        default_factory=dict, description="Current best hyperparameter config"
    )
    search_params: list[str] = Field(default_factory=list, description="Parameters being searched")
    experiments: list[ExperimentResultSchema] = Field(
        default_factory=list, description="All experiment results"
    )
    started_at: str | None = Field(None, description="When the search started")
    completed_at: str | None = Field(None, description="When the search finished")
    error: str | None = Field(None, description="Error message if failed")


# =============================================================================
# Trace Research Schemas (Data-centric AutoResearch)
# =============================================================================


class TraceResearchRequest(BaseModel):
    """Request to start trace data research."""

    search_params: list[str] = Field(
        default_factory=lambda: [
            "min_success_rate",
            "min_quality_score",
            "max_steps_per_example",
            "include_cognitive",
            "silver_inclusion_ratio",
            "time_gap_threshold_minutes",
        ],
        description="DataPipelineConfig fields to search over",
    )
    max_experiments: int = Field(30, ge=1, le=200)
    mutation_rate: float = Field(0.4, ge=0.05, le=1.0)
    mutation_scale: float = Field(0.25, ge=0.01, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "search_params": [
                    "min_success_rate",
                    "min_quality_score",
                    "include_cognitive",
                    "silver_inclusion_ratio",
                ],
                "max_experiments": 30,
                "mutation_rate": 0.4,
                "mutation_scale": 0.25,
            }
        }


class TraceExperimentResultSchema(BaseModel):
    """Result of a single trace research experiment."""

    experiment_id: int = Field(..., description="Experiment number (1-indexed)")
    config_snapshot: dict[str, Any] = Field(..., description="Data pipeline parameter values used")
    examples_generated: int = Field(..., description="Number of training examples produced")
    unique_repos: int = Field(..., description="Number of distinct repos in generated data")
    avg_example_length: float = Field(..., description="Average steps per example")
    metric_value: float = Field(..., description="Evaluation metric result (val_loss)")
    improved: bool = Field(..., description="Whether this beat the previous best")
    duration_seconds: float = Field(..., description="Wall-clock time for this experiment")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class TraceResearchStatusResponse(BaseModel):
    """Full trace research status."""

    status: str = Field(
        ..., description="Current status: idle, running, paused, completed, failed, stopped"
    )
    total_experiments: int = Field(0, description="Total experiments planned")
    completed_experiments: int = Field(0, description="Experiments completed so far")
    best_metric: float | None = Field(None, description="Best metric value achieved")
    best_pipeline: dict[str, Any] = Field(
        default_factory=dict, description="Current best data pipeline config"
    )
    best_data_stats: dict[str, Any] = Field(
        default_factory=dict, description="Data stats for best pipeline"
    )
    search_params: list[str] = Field(default_factory=list, description="Parameters being searched")
    experiments: list[TraceExperimentResultSchema] = Field(
        default_factory=list, description="All experiment results"
    )
    started_at: str | None = Field(None, description="When the search started")
    completed_at: str | None = Field(None, description="When the search finished")
    error: str | None = Field(None, description="Error message if failed")
