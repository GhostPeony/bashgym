"""
External Service Integrations

Provides unified access to:
- NVIDIA NeMo Microservices (Customizer, Evaluator, Guardrails, NIM)
- HuggingFace Hub (datasets, models, inference, jobs)
- Bashbros Integration (security middleware + AI sidekick)
"""

from .nemo_client import (
    NeMoClient,
    AsyncNeMoClient,
    NeMoClientConfig,
    NEMO_SDK_AVAILABLE,
)

from .bashbros import (
    BashbrosIntegration,
    IntegrationSettings,
    IntegrationStatus,
    CaptureMode,
    TrainingTrigger,
    ModelManifest,
    ModelVersion,
    get_integration,
    reset_integration,
)

from .huggingface import (
    HuggingFaceClient,
    HFUserInfo,
    get_hf_client,
    reset_hf_client,
    HF_HUB_AVAILABLE,
    HFError,
    HFAuthError,
    HFProRequiredError,
    HFQuotaExceededError,
    HFJobFailedError,
    # Jobs
    HFJobRunner,
    HFJobConfig,
    HFJobInfo,
    JobStatus,
    HARDWARE_SPECS,
    create_job_runner,
    # Inference
    HFInferenceClient,
    HFInferenceConfig,
    InferenceUsage,
    GenerationResponse,
    EmbeddingResponse,
    ClassificationResponse,
    InferenceProvider,
    RoutingStrategy,
    HF_INFERENCE_AVAILABLE,
    get_inference_client,
    reset_inference_client,
    # Spaces
    HFSpaceManager,
    SpaceConfig,
    SpaceStatus,
    SSHCredentials,
    GRADIO_APP_TEMPLATE,
    # Datasets
    HFDatasetManager,
    DatasetConfig,
    DATASET_CARD_TEMPLATE,
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
