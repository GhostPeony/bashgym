"""
HuggingFace Hub Integration for Bash Gym

Provides:
- HuggingFaceClient: Core client for HF Hub API operations
- HFJobRunner: Cloud training job management
- HFInferenceClient: Serverless and dedicated inference
- Error classes for handling HF-specific exceptions
- Singleton access via get_hf_client()

Features:
- Pro subscription detection
- Graceful handling of missing huggingface_hub library
- Token validation and authentication
- Support for organizations and user namespaces
- Cloud training with GPU hardware (Pro required)
- Inference with routing (serverless, dedicated, :fastest, :cheapest)

Usage:
    from bashgym.integrations.huggingface import get_hf_client, HFError

    client = get_hf_client()
    if client.is_enabled:
        # Use HuggingFace features
        client.require_enabled()  # Raises if not configured
        client.require_pro()      # Raises if not Pro subscription

    # Cloud training (Pro required)
    from bashgym.integrations.huggingface import HFJobRunner, HFJobConfig

    runner = HFJobRunner(client)
    job = runner.submit_training_job("train.py", config=HFJobConfig(hardware="a10g-small"))

    # Inference
    from bashgym.integrations.huggingface import HFInferenceClient, HFInferenceConfig

    inference = HFInferenceClient(token="hf_...")
    response = inference.generate(model="meta-llama/Llama-3.1-8B-Instruct", prompt="Hello")
"""

from .client import (
    HF_HUB_AVAILABLE,
    HFAuthError,
    # Error classes
    HFError,
    HFJobFailedError,
    HFProRequiredError,
    HFQuotaExceededError,
    HFUserInfo,
    HuggingFaceClient,
    get_hf_client,
    reset_hf_client,
)
from .datasets import (
    DATASET_CARD_TEMPLATE,
    DatasetConfig,
    HFDatasetManager,
)
from .inference import (
    HF_INFERENCE_AVAILABLE,
    ClassificationResponse,
    EmbeddingResponse,
    GenerationResponse,
    HFInferenceClient,
    HFInferenceConfig,
    InferenceProvider,
    InferenceUsage,
    RoutingStrategy,
    get_inference_client,
    reset_inference_client,
)
from .jobs import (
    HARDWARE_SPECS,
    HFJobConfig,
    HFJobInfo,
    HFJobRunner,
    JobStatus,
    create_job_runner,
)
from .model_manager import (
    HFModelInfo,
    HFModelManager,
    get_model_manager,
)
from .script_adapter import (
    CloudScriptConfig,
    generate_cloud_script,
)
from .spaces import (
    GRADIO_APP_TEMPLATE,
    HFSpaceManager,
    SpaceConfig,
    SpaceStatus,
    SSHCredentials,
)

__all__ = [
    # Client
    "HuggingFaceClient",
    "HFUserInfo",
    "get_hf_client",
    "reset_hf_client",
    "HF_HUB_AVAILABLE",
    # Errors
    "HFError",
    "HFAuthError",
    "HFProRequiredError",
    "HFQuotaExceededError",
    "HFJobFailedError",
    # Jobs
    "HFJobRunner",
    "HFJobConfig",
    "HFJobInfo",
    "JobStatus",
    "HARDWARE_SPECS",
    "create_job_runner",
    # Script Adapter
    "CloudScriptConfig",
    "generate_cloud_script",
    # Inference
    "HFInferenceClient",
    "HFInferenceConfig",
    "InferenceUsage",
    "GenerationResponse",
    "EmbeddingResponse",
    "ClassificationResponse",
    "InferenceProvider",
    "RoutingStrategy",
    "HF_INFERENCE_AVAILABLE",
    "get_inference_client",
    "reset_inference_client",
    # Spaces
    "HFSpaceManager",
    "SpaceConfig",
    "SpaceStatus",
    "SSHCredentials",
    "GRADIO_APP_TEMPLATE",
    # Datasets
    "HFDatasetManager",
    "DatasetConfig",
    "DATASET_CARD_TEMPLATE",
    # Model Manager
    "HFModelManager",
    "HFModelInfo",
    "get_model_manager",
]
