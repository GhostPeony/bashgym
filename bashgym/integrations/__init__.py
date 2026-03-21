"""
External Service Integrations

Provides unified access to:
- NVIDIA NeMo Microservices (Customizer, Evaluator, Guardrails, NIM)
- HuggingFace Hub (datasets, models, inference, jobs)
- Bashbros Integration (security middleware + AI sidekick)
"""

from .bashbros import (
    BashbrosIntegration,
    CaptureMode,
    IntegrationSettings,
    IntegrationStatus,
    ModelManifest,
    ModelVersion,
    TrainingTrigger,
    get_integration,
    reset_integration,
)
from .huggingface import (
    DATASET_CARD_TEMPLATE,
    GRADIO_APP_TEMPLATE,
    HARDWARE_SPECS,
    HF_HUB_AVAILABLE,
    HF_INFERENCE_AVAILABLE,
    ClassificationResponse,
    DatasetConfig,
    EmbeddingResponse,
    GenerationResponse,
    HFAuthError,
    # Datasets
    HFDatasetManager,
    HFError,
    # Inference
    HFInferenceClient,
    HFInferenceConfig,
    HFJobConfig,
    HFJobFailedError,
    HFJobInfo,
    # Jobs
    HFJobRunner,
    HFProRequiredError,
    HFQuotaExceededError,
    # Spaces
    HFSpaceManager,
    HFUserInfo,
    HuggingFaceClient,
    InferenceProvider,
    InferenceUsage,
    JobStatus,
    RoutingStrategy,
    SpaceConfig,
    SpaceStatus,
    SSHCredentials,
    create_job_runner,
    get_hf_client,
    get_inference_client,
    reset_hf_client,
    reset_inference_client,
)
from .nemo_client import (
    NEMO_SDK_AVAILABLE,
    AsyncNeMoClient,
    NeMoClient,
    NeMoClientConfig,
)

__all__ = [
    # NeMo
    "NeMoClient",
    "AsyncNeMoClient",
    "NeMoClientConfig",
    "NEMO_SDK_AVAILABLE",
    # Bashbros Integration
    "BashbrosIntegration",
    "IntegrationSettings",
    "IntegrationStatus",
    "CaptureMode",
    "TrainingTrigger",
    "ModelManifest",
    "ModelVersion",
    "get_integration",
    "reset_integration",
    # HuggingFace Client
    "HuggingFaceClient",
    "HFUserInfo",
    "get_hf_client",
    "reset_hf_client",
    "HF_HUB_AVAILABLE",
    # HuggingFace Errors
    "HFError",
    "HFAuthError",
    "HFProRequiredError",
    "HFQuotaExceededError",
    "HFJobFailedError",
    # HuggingFace Jobs
    "HFJobRunner",
    "HFJobConfig",
    "HFJobInfo",
    "JobStatus",
    "HARDWARE_SPECS",
    "create_job_runner",
    # HuggingFace Inference
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
    # HuggingFace Spaces
    "HFSpaceManager",
    "SpaceConfig",
    "SpaceStatus",
    "SSHCredentials",
    "GRADIO_APP_TEMPLATE",
    # HuggingFace Datasets
    "HFDatasetManager",
    "DatasetConfig",
    "DATASET_CARD_TEMPLATE",
]
