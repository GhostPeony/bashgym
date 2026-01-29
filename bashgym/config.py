"""
Application Settings for Bash Gym

Centralized configuration management using environment variables
and sensible defaults. Supports .env files for local development.

Config Module
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


def get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


def get_bashgym_dir() -> Path:
    """Get the global Bash Gym directory (~/.bashgym/)."""
    import platform
    if platform.system() == 'Windows':
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym"


def get_default_data_dir() -> str:
    """Get the default data directory (global ~/.bashgym/)."""
    return str(get_bashgym_dir())


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


def get_env_list(key: str, default: List[str] = None, separator: str = ",") -> List[str]:
    """Get list environment variable."""
    value = os.environ.get(key)
    if value:
        return [item.strip() for item in value.split(separator)]
    return default or []


@dataclass
class APISettings:
    """API key and endpoint settings."""

    # Anthropic (Claude)
    anthropic_api_key: str = field(default_factory=lambda: get_env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: get_env("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))

    # NVIDIA
    nvidia_api_key: str = field(default_factory=lambda: get_env("NVIDIA_API_KEY"))
    nemo_endpoint: str = field(default_factory=lambda: get_env("NEMO_ENDPOINT", "http://localhost:8000"))
    nim_endpoint: str = field(default_factory=lambda: get_env("NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1"))
    nim_model: str = field(default_factory=lambda: get_env("NIM_MODEL", "qwen/qwen2.5-coder-72b-instruct"))

    def validate(self) -> List[str]:
        """Validate required API keys are present."""
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        return errors


class InferenceProvider(Enum):
    """HuggingFace inference providers."""
    HUGGINGFACE = "huggingface"  # HF Inference API
    SERVERLESS = "serverless"   # HF Serverless Inference
    DEDICATED = "dedicated"     # HF Dedicated Inference Endpoints


class InferenceRouting(Enum):
    """Inference routing strategies."""
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    QUALITY = "quality"


class HFHardware(Enum):
    """HuggingFace Spaces/Jobs hardware tiers."""
    CPU_BASIC = "cpu-basic"
    CPU_UPGRADE = "cpu-upgrade"
    T4_SMALL = "t4-small"
    T4_MEDIUM = "t4-medium"
    A10G_SMALL = "a10g-small"
    A10G_LARGE = "a10g-large"
    A10G_LARGEX2 = "a10g-largex2"
    A10G_LARGEX4 = "a10g-largex4"
    A100_LARGE = "a100-large"
    H100 = "h100"
    H100X8 = "h100x8"


def get_hf_token() -> str:
    """Get HuggingFace token from env or stored secrets."""
    # Environment variable takes precedence
    env_token = get_env("HF_TOKEN")
    if env_token:
        return env_token

    # Check stored secrets
    try:
        from bashgym.secrets import get_secret
        stored = get_secret("HF_TOKEN")
        return stored or ""
    except ImportError:
        return ""


@dataclass
class HuggingFaceSettings:
    """HuggingFace Hub integration settings.

    Provides configuration for:
    - Authentication (token, username, org)
    - Storage repos (datasets, models)
    - Inference settings (provider, routing)
    - Hardware preferences for Jobs/Spaces
    """

    # Authentication
    token: str = field(default_factory=get_hf_token)
    username: str = field(default_factory=lambda: get_env("HF_USERNAME"))
    default_org: str = field(default_factory=lambda: get_env("HF_ORG"))

    # Pro status (auto-detected at runtime, can be overridden)
    pro_enabled: bool = field(default_factory=lambda: get_env_bool("HF_PRO_ENABLED", False))

    # Default repos for storage
    storage_repo: str = field(default_factory=lambda: get_env("HF_STORAGE_REPO"))
    models_repo: str = field(default_factory=lambda: get_env("HF_MODELS_REPO"))

    # Inference settings
    inference_provider: str = field(default_factory=lambda: get_env(
        "HF_INFERENCE_PROVIDER", InferenceProvider.SERVERLESS.value
    ))
    inference_routing: str = field(default_factory=lambda: get_env(
        "HF_INFERENCE_ROUTING", InferenceRouting.CHEAPEST.value
    ))

    # Hardware defaults for Jobs/Spaces
    default_hardware: str = field(default_factory=lambda: get_env(
        "HF_DEFAULT_HARDWARE", HFHardware.T4_SMALL.value
    ))
    job_timeout_minutes: int = field(default_factory=lambda: get_env_int(
        "HF_JOB_TIMEOUT_MINUTES", 60
    ))

    @property
    def enabled(self) -> bool:
        """Check if HuggingFace integration is enabled (token is set)."""
        return bool(self.token)

    @property
    def namespace(self) -> str:
        """Get the default namespace (org or username)."""
        return self.default_org or self.username or ""

    def validate(self) -> List[str]:
        """Validate HuggingFace settings."""
        errors = []
        if self.token and not self.username and not self.default_org:
            # Token is set but no username/org - will need to fetch from API
            pass  # This is OK, client will detect it
        return errors


@dataclass
class DockerSettings:
    """Docker and sandbox settings."""

    docker_host: str = field(default_factory=lambda: get_env("DOCKER_HOST", "unix:///var/run/docker.sock"))
    sandbox_image: str = field(default_factory=lambda: get_env("SANDBOX_IMAGE", "python:3.10-slim"))
    sandbox_memory_limit: str = field(default_factory=lambda: get_env("SANDBOX_MEMORY", "2g"))
    sandbox_cpu_limit: float = field(default_factory=lambda: get_env_float("SANDBOX_CPU", 2.0))
    sandbox_timeout: int = field(default_factory=lambda: get_env_int("SANDBOX_TIMEOUT", 3600))
    sandbox_network_mode: str = field(default_factory=lambda: get_env("SANDBOX_NETWORK", "none"))
    workspace_base: str = field(default_factory=lambda: get_env("WORKSPACE_BASE", "/tmp/bashgym_workspaces"))


@dataclass
class TrainingSettings:
    """Training and model settings."""

    # Base model
    base_model: str = field(default_factory=lambda: get_env("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"))
    model_type: str = field(default_factory=lambda: get_env("MODEL_TYPE", "qwen"))

    # Training hyperparameters
    learning_rate: float = field(default_factory=lambda: get_env_float("LEARNING_RATE", 2e-5))
    batch_size: int = field(default_factory=lambda: get_env_int("BATCH_SIZE", 4))
    gradient_accumulation_steps: int = field(default_factory=lambda: get_env_int("GRAD_ACCUM_STEPS", 4))
    num_epochs: int = field(default_factory=lambda: get_env_int("NUM_EPOCHS", 3))
    max_seq_length: int = field(default_factory=lambda: get_env_int("MAX_SEQ_LENGTH", 4096))
    warmup_ratio: float = field(default_factory=lambda: get_env_float("WARMUP_RATIO", 0.1))

    # LoRA settings
    use_lora: bool = field(default_factory=lambda: get_env_bool("USE_LORA", True))
    lora_r: int = field(default_factory=lambda: get_env_int("LORA_R", 16))
    lora_alpha: int = field(default_factory=lambda: get_env_int("LORA_ALPHA", 32))
    lora_dropout: float = field(default_factory=lambda: get_env_float("LORA_DROPOUT", 0.05))

    # Quantization
    load_in_4bit: bool = field(default_factory=lambda: get_env_bool("LOAD_IN_4BIT", True))

    # DPO settings
    dpo_beta: float = field(default_factory=lambda: get_env_float("DPO_BETA", 0.1))

    # GRPO settings
    grpo_num_generations: int = field(default_factory=lambda: get_env_int("GRPO_NUM_GENERATIONS", 4))
    grpo_temperature: float = field(default_factory=lambda: get_env_float("GRPO_TEMPERATURE", 0.7))

    # Hardware
    device_map: str = field(default_factory=lambda: get_env("DEVICE_MAP", "auto"))
    use_flash_attention: bool = field(default_factory=lambda: get_env_bool("USE_FLASH_ATTENTION", True))

    # Cloud training
    use_nemo_gym: bool = field(default_factory=lambda: get_env_bool("USE_NEMO_GYM", False))


@dataclass
class DataSettings:
    """Data processing and storage settings.

    By default, all traces are stored globally in ~/.bashgym/ with repo tagging.
    This enables both per-repo specialized training and cross-repo generalist training.
    """

    # Directories - default to global ~/.bashgym/ for hybrid storage approach
    data_dir: str = field(default_factory=lambda: get_env("DATA_DIR", get_default_data_dir()))
    gold_traces_dir: str = field(default_factory=lambda: get_env(
        "GOLD_TRACES_DIR", str(get_bashgym_dir() / "gold_traces")))
    failed_traces_dir: str = field(default_factory=lambda: get_env(
        "FAILED_TRACES_DIR", str(get_bashgym_dir() / "failed_traces")))
    training_batches_dir: str = field(default_factory=lambda: get_env(
        "TRAINING_BATCHES_DIR", str(get_bashgym_dir() / "training_batches")))
    models_dir: str = field(default_factory=lambda: get_env(
        "MODELS_DIR", str(get_bashgym_dir() / "models")))

    # Processing settings
    min_trace_steps: int = field(default_factory=lambda: get_env_int("MIN_TRACE_STEPS", 2))
    max_trace_steps: int = field(default_factory=lambda: get_env_int("MAX_TRACE_STEPS", 50))
    min_quality_score: float = field(default_factory=lambda: get_env_float("MIN_QUALITY_SCORE", 0.3))
    augmentation_factor: int = field(default_factory=lambda: get_env_int("AUGMENTATION_FACTOR", 3))

    # Batch settings
    batch_size: int = field(default_factory=lambda: get_env_int("DATA_BATCH_SIZE", 100))

    def ensure_directories(self) -> None:
        """Create all data directories."""
        for attr in ["data_dir", "gold_traces_dir", "failed_traces_dir",
                     "training_batches_dir", "models_dir"]:
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)


@dataclass
class VerificationSettings:
    """Verification and testing settings."""

    timeout: int = field(default_factory=lambda: get_env_int("VERIFY_TIMEOUT", 300))
    max_retries: int = field(default_factory=lambda: get_env_int("VERIFY_MAX_RETRIES", 1))
    test_patterns: List[str] = field(default_factory=lambda: get_env_list(
        "TEST_PATTERNS",
        ["test_*.py", "*_test.py", "tests/*.py", "verify.sh", "verify.py"]
    ))
    pytest_args: List[str] = field(default_factory=lambda: get_env_list(
        "PYTEST_ARGS",
        ["-v", "--tb=short", "-x"]
    ))


@dataclass
class RouterSettings:
    """Model routing settings."""

    strategy: str = field(default_factory=lambda: get_env("ROUTING_STRATEGY", "confidence_based"))
    confidence_threshold: float = field(default_factory=lambda: get_env_float("CONFIDENCE_THRESHOLD", 0.7))
    complexity_threshold: float = field(default_factory=lambda: get_env_float("COMPLEXITY_THRESHOLD", 0.5))
    student_sample_rate: float = field(default_factory=lambda: get_env_float("STUDENT_SAMPLE_RATE", 0.1))
    max_student_rate: float = field(default_factory=lambda: get_env_float("MAX_STUDENT_RATE", 0.9))
    fallback_to_teacher: bool = field(default_factory=lambda: get_env_bool("FALLBACK_TO_TEACHER", True))


@dataclass
class EvaluatorSettings:
    """NeMo Evaluator settings for comprehensive model evaluation."""

    enabled: bool = field(default_factory=lambda: get_env_bool("EVALUATOR_ENABLED", True))
    endpoint: str = field(default_factory=lambda: get_env("EVALUATOR_ENDPOINT", "http://localhost:8000"))
    benchmarks: List[str] = field(default_factory=lambda: get_env_list(
        "EVALUATOR_BENCHMARKS", ["humaneval", "mbpp", "bigcodebench"]
    ))
    judge_model: str = field(default_factory=lambda: get_env(
        "EVALUATOR_JUDGE_MODEL", "meta/llama-3.1-70b-instruct"
    ))
    timeout: int = field(default_factory=lambda: get_env_int("EVALUATOR_TIMEOUT", 3600))
    max_concurrent_jobs: int = field(default_factory=lambda: get_env_int("EVALUATOR_MAX_JOBS", 4))


@dataclass
class PrivacySettings:
    """Safe Synthesizer settings for privacy-preserving data generation."""

    enabled: bool = field(default_factory=lambda: get_env_bool("PRIVACY_ENABLED", False))
    epsilon: float = field(default_factory=lambda: get_env_float("PRIVACY_EPSILON", 8.0))
    pii_types: List[str] = field(default_factory=lambda: get_env_list(
        "PRIVACY_PII_TYPES", ["person", "email", "ssn", "phone", "address", "credit_card"]
    ))
    use_dp_sgd: bool = field(default_factory=lambda: get_env_bool("PRIVACY_USE_DP_SGD", False))
    safe_synthesizer_endpoint: str = field(default_factory=lambda: get_env(
        "SAFE_SYNTHESIZER_ENDPOINT", "http://localhost:8000"
    ))


@dataclass
class GuardrailsSettings:
    """NeMo Guardrails settings for safety checks."""

    enabled: bool = field(default_factory=lambda: get_env_bool("GUARDRAILS_ENABLED", True))

    # Check toggles
    injection_detection: bool = field(default_factory=lambda: get_env_bool("GUARDRAILS_INJECTION_DETECTION", True))
    code_safety: bool = field(default_factory=lambda: get_env_bool("GUARDRAILS_CODE_SAFETY", True))
    pii_filtering: bool = field(default_factory=lambda: get_env_bool("GUARDRAILS_PII_FILTERING", True))
    content_moderation: bool = field(default_factory=lambda: get_env_bool("GUARDRAILS_CONTENT_MODERATION", False))

    # Thresholds
    injection_threshold: float = field(default_factory=lambda: get_env_float("GUARDRAILS_INJECTION_THRESHOLD", 0.8))

    # Code safety - dangerous commands to block
    blocked_commands: List[str] = field(default_factory=lambda: get_env_list(
        "GUARDRAILS_BLOCKED_COMMANDS",
        ["rm -rf /", "rm -rf /*", ":(){:|:&};:", "dd if=/dev/zero", "mkfs.", "> /dev/sda"]
    ))

    # Topic control
    topic_control: List[str] = field(default_factory=lambda: get_env_list("GUARDRAILS_TOPICS", []))

    # NeMo endpoint (optional, for advanced checks)
    nemoguard_endpoint: str = field(default_factory=lambda: get_env(
        "NEMOGUARD_ENDPOINT", "http://localhost:8000"
    ))
    colang_config_path: str = field(default_factory=lambda: get_env("COLANG_CONFIG_PATH", ""))


@dataclass
class PromptOptSettings:
    """MIPROv2 prompt optimization settings."""

    enabled: bool = field(default_factory=lambda: get_env_bool("PROMPT_OPT_ENABLED", False))
    intensity: str = field(default_factory=lambda: get_env("PROMPT_OPT_INTENSITY", "medium"))
    max_bootstrapped_demos: int = field(default_factory=lambda: get_env_int("PROMPT_OPT_MAX_DEMOS", 4))
    metric_threshold: float = field(default_factory=lambda: get_env_float("PROMPT_OPT_THRESHOLD", 0.8))
    num_trials: int = field(default_factory=lambda: get_env_int("PROMPT_OPT_TRIALS", 20))
    optimizer_model: str = field(default_factory=lambda: get_env(
        "PROMPT_OPT_MODEL", "meta/llama-3.1-70b-instruct"
    ))


@dataclass
class ObservabilitySettings:
    """Profiler and observability settings (local storage)."""

    enabled: bool = field(default_factory=lambda: get_env_bool("PROFILER_ENABLED", True))

    # What to profile
    profile_tokens: bool = field(default_factory=lambda: get_env_bool("PROFILER_TOKENS", True))
    profile_latency: bool = field(default_factory=lambda: get_env_bool("PROFILER_LATENCY", True))
    profile_guardrails: bool = field(default_factory=lambda: get_env_bool("PROFILER_GUARDRAILS", True))

    # Storage
    output_dir: str = field(default_factory=lambda: get_env(
        "PROFILER_OUTPUT_DIR", str(get_bashgym_dir() / "profiler_traces")
    ))
    max_traces_in_memory: int = field(default_factory=lambda: get_env_int("PROFILER_MAX_TRACES", 1000))

    # Sampling
    trace_sampling_rate: float = field(default_factory=lambda: get_env_float("PROFILER_SAMPLING_RATE", 1.0))


@dataclass
class LoggingSettings:
    """Logging and monitoring settings."""

    log_level: str = field(default_factory=lambda: get_env("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: get_env(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    log_file: str = field(default_factory=lambda: get_env("LOG_FILE", ""))
    enable_metrics: bool = field(default_factory=lambda: get_env_bool("ENABLE_METRICS", True))
    metrics_port: int = field(default_factory=lambda: get_env_int("METRICS_PORT", 9090))


@dataclass
class Settings:
    """
    Main settings container for Bash Gym.

    Usage:
        settings = Settings()
        print(settings.api.anthropic_api_key)
        print(settings.training.base_model)
    """

    # Environment
    environment: Environment = field(default_factory=lambda: Environment(
        get_env("ENVIRONMENT", "development")
    ))
    debug: bool = field(default_factory=lambda: get_env_bool("DEBUG", False))

    # Component settings
    api: APISettings = field(default_factory=APISettings)
    docker: DockerSettings = field(default_factory=DockerSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    data: DataSettings = field(default_factory=DataSettings)
    verification: VerificationSettings = field(default_factory=VerificationSettings)
    router: RouterSettings = field(default_factory=RouterSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    # NeMo Microservices settings
    evaluator: EvaluatorSettings = field(default_factory=EvaluatorSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    guardrails: GuardrailsSettings = field(default_factory=GuardrailsSettings)
    prompt_opt: PromptOptSettings = field(default_factory=PromptOptSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)

    # HuggingFace integration
    huggingface: HuggingFaceSettings = field(default_factory=HuggingFaceSettings)

    def validate(self) -> List[str]:
        """Validate all settings."""
        errors = []
        errors.extend(self.api.validate())
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (hiding sensitive values)."""
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif hasattr(obj, '__dataclass_fields__'):
                result = {}
                for k in obj.__dataclass_fields__:
                    v = getattr(obj, k)
                    # Hide sensitive values
                    if any(s in k.lower() for s in ['key', 'token', 'secret', 'password']):
                        result[k] = "***" if v else ""
                    else:
                        result[k] = sanitize(v)
                return result
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            else:
                return obj

        return sanitize(self)

    def setup(self) -> None:
        """Perform initial setup (create directories, configure logging)."""
        self.data.ensure_directories()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        import logging

        level = getattr(logging, self.logging.log_level.upper(), logging.INFO)

        handlers = [logging.StreamHandler()]
        if self.logging.log_file:
            handlers.append(logging.FileHandler(self.logging.log_file))

        logging.basicConfig(
            level=level,
            format=self.logging.log_format,
            handlers=handlers
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings


# Environment variable template for .env.example
ENV_TEMPLATE = """# Bash Gym Configuration
# Copy this file to .env and fill in your values

# Environment
ENVIRONMENT=development
DEBUG=false

# API Keys (Required)
ANTHROPIC_API_KEY=your-anthropic-key-here
NVIDIA_API_KEY=your-nvidia-key-here

# Optional API Settings
ANTHROPIC_MODEL=claude-sonnet-4-20250514
NEMO_ENDPOINT=http://localhost:8000
NIM_ENDPOINT=https://integrate.api.nvidia.com/v1
NIM_MODEL=meta/llama-3.1-70b-instruct

# =============================================
# HuggingFace Integration
# =============================================
HF_TOKEN=your-huggingface-token
HF_USERNAME=your-hf-username
HF_ORG=your-hf-org
HF_PRO_ENABLED=false
HF_STORAGE_REPO=
HF_MODELS_REPO=
HF_INFERENCE_PROVIDER=serverless
HF_INFERENCE_ROUTING=cheapest
HF_DEFAULT_HARDWARE=t4-small
HF_JOB_TIMEOUT_MINUTES=60

# Docker/Sandbox Settings
DOCKER_HOST=unix:///var/run/docker.sock
SANDBOX_IMAGE=python:3.10-slim
SANDBOX_MEMORY=2g
SANDBOX_CPU=2.0
SANDBOX_TIMEOUT=3600
SANDBOX_NETWORK=none
WORKSPACE_BASE=/tmp/bashgym_workspaces

# Training Settings
BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
MODEL_TYPE=qwen
LEARNING_RATE=2e-5
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
NUM_EPOCHS=3
MAX_SEQ_LENGTH=4096
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LOAD_IN_4BIT=true
USE_NEMO_GYM=false

# Data Settings
DATA_DIR=data
MIN_TRACE_STEPS=2
MAX_TRACE_STEPS=50
MIN_QUALITY_SCORE=0.3
AUGMENTATION_FACTOR=3

# Verification Settings
VERIFY_TIMEOUT=300
VERIFY_MAX_RETRIES=1

# Router Settings
ROUTING_STRATEGY=confidence_based
CONFIDENCE_THRESHOLD=0.7
STUDENT_SAMPLE_RATE=0.1
FALLBACK_TO_TEACHER=true

# Logging
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090

# =============================================
# NeMo Microservices Settings
# =============================================

# NeMo Evaluator
EVALUATOR_ENABLED=true
EVALUATOR_ENDPOINT=http://localhost:8000
EVALUATOR_BENCHMARKS=humaneval,mbpp,bigcodebench
EVALUATOR_JUDGE_MODEL=meta/llama-3.1-70b-instruct
EVALUATOR_TIMEOUT=3600
EVALUATOR_MAX_JOBS=4

# Safe Synthesizer (Privacy)
PRIVACY_ENABLED=false
PRIVACY_EPSILON=8.0
PRIVACY_PII_TYPES=person,email,ssn,phone,address,credit_card
PRIVACY_USE_DP_SGD=false
SAFE_SYNTHESIZER_ENDPOINT=http://localhost:8000

# Guardrails
GUARDRAILS_ENABLED=true
GUARDRAILS_INJECTION_DETECTION=true
GUARDRAILS_CODE_SAFETY=true
GUARDRAILS_PII_FILTERING=true
GUARDRAILS_CONTENT_MODERATION=false
GUARDRAILS_INJECTION_THRESHOLD=0.8
NEMOGUARD_ENDPOINT=http://localhost:8000

# Prompt Optimization (MIPROv2)
PROMPT_OPT_ENABLED=false
PROMPT_OPT_INTENSITY=medium
PROMPT_OPT_MAX_DEMOS=4
PROMPT_OPT_THRESHOLD=0.8
PROMPT_OPT_TRIALS=20
PROMPT_OPT_MODEL=meta/llama-3.1-70b-instruct

# Profiler (Local)
PROFILER_ENABLED=true
PROFILER_TOKENS=true
PROFILER_LATENCY=true
PROFILER_GUARDRAILS=true
PROFILER_OUTPUT_DIR=~/.bashgym/profiler_traces
PROFILER_MAX_TRACES=1000
PROFILER_SAMPLING_RATE=1.0
"""


def generate_env_template(output_path: Path = Path(".env.example")) -> None:
    """Generate .env.example template file."""
    output_path.write_text(ENV_TEMPLATE)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    import json

    # Load and display settings
    settings = get_settings()
    print("Current Settings:")
    print(json.dumps(settings.to_dict(), indent=2))

    # Validate
    errors = settings.validate()
    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  - {error}")

    # Generate template
    generate_env_template()
