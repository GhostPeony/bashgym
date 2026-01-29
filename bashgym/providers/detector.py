"""
Model Provider Detector

Auto-detects available model providers (local and cloud) and aggregates models.
"""

import os
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class ProviderType(str, Enum):
    """Types of model providers."""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    HUGGINGFACE = "huggingface"
    NVIDIA_NIM = "nvidia_nim"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class ProviderStatus:
    """Status of a model provider."""
    type: ProviderType
    name: str
    available: bool
    endpoint: Optional[str] = None
    model_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "available": self.available,
            "endpoint": self.endpoint,
            "model_count": self.model_count,
            "error": self.error
        }


@dataclass
class UnifiedModel:
    """Unified model representation across providers."""
    id: str
    name: str
    provider: ProviderType
    size_gb: Optional[float] = None
    parameter_size: Optional[str] = None
    is_code_model: bool = False
    is_local: bool = False
    supports_training: bool = False
    supports_inference: bool = True
    context_length: Optional[int] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.value,
            "size_gb": self.size_gb,
            "parameter_size": self.parameter_size,
            "is_code_model": self.is_code_model,
            "is_local": self.is_local,
            "supports_training": self.supports_training,
            "supports_inference": self.supports_inference,
            "context_length": self.context_length,
            "description": self.description
        }


async def detect_ollama() -> ProviderStatus:
    """Detect Ollama installation and status."""
    from .ollama import get_ollama_provider

    provider = get_ollama_provider()

    try:
        is_running = await provider.is_running()
        if is_running:
            models = await provider.list_models()
            return ProviderStatus(
                type=ProviderType.OLLAMA,
                name="Ollama",
                available=True,
                endpoint=provider.base_url,
                model_count=len(models)
            )
        else:
            return ProviderStatus(
                type=ProviderType.OLLAMA,
                name="Ollama",
                available=False,
                error="Ollama server not running. Start with: ollama serve"
            )
    except Exception as e:
        return ProviderStatus(
            type=ProviderType.OLLAMA,
            name="Ollama",
            available=False,
            error=str(e)
        )


async def detect_lm_studio() -> ProviderStatus:
    """Detect LM Studio installation."""
    import httpx

    # LM Studio default endpoint
    endpoint = "http://localhost:1234"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{endpoint}/v1/models")
            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("data", []))
                return ProviderStatus(
                    type=ProviderType.LM_STUDIO,
                    name="LM Studio",
                    available=True,
                    endpoint=endpoint,
                    model_count=model_count
                )
    except:
        pass

    return ProviderStatus(
        type=ProviderType.LM_STUDIO,
        name="LM Studio",
        available=False,
        error="LM Studio not running"
    )


async def detect_nvidia_nim() -> ProviderStatus:
    """Detect NVIDIA NIM API availability."""
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")

    if api_key:
        return ProviderStatus(
            type=ProviderType.NVIDIA_NIM,
            name="NVIDIA NIM",
            available=True,
            endpoint="https://integrate.api.nvidia.com/v1"
        )

    return ProviderStatus(
        type=ProviderType.NVIDIA_NIM,
        name="NVIDIA NIM",
        available=False,
        error="NVIDIA_API_KEY not set"
    )


async def detect_huggingface() -> ProviderStatus:
    """Detect HuggingFace availability."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # HuggingFace is always available for downloading models
    return ProviderStatus(
        type=ProviderType.HUGGINGFACE,
        name="HuggingFace",
        available=True,
        endpoint="https://huggingface.co",
        error=None if hf_token else "HF_TOKEN not set (some models may be inaccessible)"
    )


async def detect_anthropic() -> ProviderStatus:
    """Detect Anthropic API availability."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        return ProviderStatus(
            type=ProviderType.ANTHROPIC,
            name="Anthropic",
            available=True,
            endpoint="https://api.anthropic.com"
        )

    return ProviderStatus(
        type=ProviderType.ANTHROPIC,
        name="Anthropic",
        available=False,
        error="ANTHROPIC_API_KEY not set"
    )


async def detect_openai() -> ProviderStatus:
    """Detect OpenAI API availability."""
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        return ProviderStatus(
            type=ProviderType.OPENAI,
            name="OpenAI",
            available=True,
            endpoint="https://api.openai.com/v1"
        )

    return ProviderStatus(
        type=ProviderType.OPENAI,
        name="OpenAI",
        available=False,
        error="OPENAI_API_KEY not set"
    )


async def detect_providers() -> List[ProviderStatus]:
    """Detect all available model providers."""
    detectors = [
        detect_ollama(),
        detect_lm_studio(),
        detect_nvidia_nim(),
        detect_huggingface(),
        detect_anthropic(),
        detect_openai(),
    ]

    results = await asyncio.gather(*detectors, return_exceptions=True)

    providers = []
    for result in results:
        if isinstance(result, ProviderStatus):
            providers.append(result)
        elif isinstance(result, Exception):
            print(f"Provider detection error: {result}")

    return providers


async def get_ollama_models() -> List[UnifiedModel]:
    """Get models from Ollama."""
    from .ollama import get_ollama_provider

    provider = get_ollama_provider()
    models = []

    try:
        ollama_models = await provider.list_models()
        for m in ollama_models:
            models.append(UnifiedModel(
                id=f"ollama/{m.name}",
                name=m.name,
                provider=ProviderType.OLLAMA,
                size_gb=round(m.size_gb, 2),
                parameter_size=m.parameter_size,
                is_code_model=m.is_code_model,
                is_local=True,
                supports_training=True,  # Can export GGUF for training
                supports_inference=True
            ))
    except:
        pass

    return models


async def get_lm_studio_models() -> List[UnifiedModel]:
    """Get models from LM Studio."""
    import httpx

    models = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:1234/v1/models")
            if response.status_code == 200:
                data = response.json()
                for m in data.get("data", []):
                    model_id = m.get("id", "")
                    models.append(UnifiedModel(
                        id=f"lm_studio/{model_id}",
                        name=model_id,
                        provider=ProviderType.LM_STUDIO,
                        is_local=True,
                        supports_inference=True
                    ))
    except:
        pass

    return models


# Curated list of recommended models for training
RECOMMENDED_TRAINING_MODELS = [
    UnifiedModel(
        id="hf/Qwen/Qwen2.5-Coder-1.5B-Instruct",
        name="Qwen2.5-Coder-1.5B-Instruct",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="1.5B",
        is_code_model=True,
        supports_training=True,
        context_length=32768,
        description="Fast, efficient coder. Great for fine-tuning on consumer GPUs."
    ),
    UnifiedModel(
        id="hf/Qwen/Qwen2.5-Coder-7B-Instruct",
        name="Qwen2.5-Coder-7B-Instruct",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="7B",
        is_code_model=True,
        supports_training=True,
        context_length=32768,
        description="Excellent balance of quality and efficiency."
    ),
    UnifiedModel(
        id="hf/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        name="DeepSeek-Coder-V2-Lite",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="16B",
        is_code_model=True,
        supports_training=True,
        context_length=128000,
        description="State-of-the-art code model with huge context."
    ),
    UnifiedModel(
        id="hf/codellama/CodeLlama-7b-Instruct-hf",
        name="CodeLlama-7B-Instruct",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="7B",
        is_code_model=True,
        supports_training=True,
        context_length=16384,
        description="Meta's code-specialized Llama model."
    ),
    UnifiedModel(
        id="hf/microsoft/phi-4",
        name="Phi-4",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="14B",
        is_code_model=True,
        supports_training=True,
        context_length=16384,
        description="Microsoft's efficient reasoning model."
    ),
]

# Curated list of teacher models for knowledge distillation
TEACHER_MODELS = [
    UnifiedModel(
        id="anthropic/claude-3-5-sonnet",
        name="Claude 3.5 Sonnet",
        provider=ProviderType.ANTHROPIC,
        is_code_model=True,
        supports_inference=True,
        description="Excellent for code generation and reasoning."
    ),
    UnifiedModel(
        id="openai/gpt-4-turbo",
        name="GPT-4 Turbo",
        provider=ProviderType.OPENAI,
        is_code_model=True,
        supports_inference=True,
        description="Strong general-purpose model."
    ),
    UnifiedModel(
        id="hf/Qwen/Qwen2.5-Coder-32B-Instruct",
        name="Qwen2.5-Coder-32B-Instruct",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="32B",
        is_code_model=True,
        supports_inference=True,
        description="Top-tier open-source code model."
    ),
]


async def get_available_models(
    include_local: bool = True,
    include_cloud: bool = True,
    code_only: bool = False
) -> Dict[str, List[UnifiedModel]]:
    """
    Get all available models organized by category.

    Returns:
        Dict with keys: 'local', 'training', 'teacher', 'inference'
    """
    result = {
        "local": [],
        "training": RECOMMENDED_TRAINING_MODELS.copy(),
        "teacher": TEACHER_MODELS.copy(),
        "inference": []
    }

    if include_local:
        # Get Ollama models
        ollama_models = await get_ollama_models()
        result["local"].extend(ollama_models)

        # Get LM Studio models
        lm_models = await get_lm_studio_models()
        result["local"].extend(lm_models)

        # Local models can also be used for inference
        result["inference"].extend(ollama_models)
        result["inference"].extend(lm_models)

    if code_only:
        for key in result:
            result[key] = [m for m in result[key] if m.is_code_model]

    return result


# Sync wrapper
def detect_providers_sync() -> List[ProviderStatus]:
    """Synchronous wrapper for detect_providers."""
    return asyncio.run(detect_providers())


def get_available_models_sync(**kwargs) -> Dict[str, List[UnifiedModel]]:
    """Synchronous wrapper for get_available_models."""
    return asyncio.run(get_available_models(**kwargs))
