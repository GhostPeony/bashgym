"""
Model Provider Detector

Auto-detects available model providers (local and cloud) and aggregates models.
"""

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


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
    endpoint: str | None = None
    model_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "available": self.available,
            "endpoint": self.endpoint,
            "model_count": self.model_count,
            "error": self.error,
        }


@dataclass
class UnifiedModel:
    """Unified model representation across providers."""

    id: str
    name: str
    provider: ProviderType
    size_gb: float | None = None
    parameter_size: str | None = None
    is_code_model: bool = False
    is_local: bool = False
    supports_training: bool = False
    supports_inference: bool = True
    context_length: int | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
            "description": self.description,
        }


async def detect_ollama() -> ProviderStatus:
    """Detect Ollama installation and status."""
    from .ollama import get_ollama_provider

    provider = get_ollama_provider()

    try:
        is_running = await provider.is_running()
        if is_running:
            models = await provider.list_ollama_models()
            return ProviderStatus(
                type=ProviderType.OLLAMA,
                name="Ollama",
                available=True,
                endpoint=provider.base_url,
                model_count=len(models),
            )
        else:
            return ProviderStatus(
                type=ProviderType.OLLAMA,
                name="Ollama",
                available=False,
                error="Ollama server not running. Start with: ollama serve",
            )
    except Exception as e:
        return ProviderStatus(
            type=ProviderType.OLLAMA, name="Ollama", available=False, error=str(e)
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
                    model_count=model_count,
                )
    except Exception:
        pass

    return ProviderStatus(
        type=ProviderType.LM_STUDIO,
        name="LM Studio",
        available=False,
        error="LM Studio not running",
    )


async def detect_nvidia_nim() -> ProviderStatus:
    """Detect NVIDIA NIM API availability."""
    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")

    if api_key:
        return ProviderStatus(
            type=ProviderType.NVIDIA_NIM,
            name="NVIDIA NIM",
            available=True,
            endpoint="https://integrate.api.nvidia.com/v1",
        )

    return ProviderStatus(
        type=ProviderType.NVIDIA_NIM,
        name="NVIDIA NIM",
        available=False,
        error="NVIDIA_API_KEY not set",
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
        error=None if hf_token else "HF_TOKEN not set (some models may be inaccessible)",
    )


async def detect_anthropic() -> ProviderStatus:
    """Detect Anthropic API availability."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        return ProviderStatus(
            type=ProviderType.ANTHROPIC,
            name="Anthropic",
            available=True,
            endpoint="https://api.anthropic.com",
        )

    return ProviderStatus(
        type=ProviderType.ANTHROPIC,
        name="Anthropic",
        available=False,
        error="ANTHROPIC_API_KEY not set",
    )


async def detect_openai() -> ProviderStatus:
    """Detect OpenAI API availability."""
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        return ProviderStatus(
            type=ProviderType.OPENAI,
            name="OpenAI",
            available=True,
            endpoint="https://api.openai.com/v1",
        )

    return ProviderStatus(
        type=ProviderType.OPENAI, name="OpenAI", available=False, error="OPENAI_API_KEY not set"
    )


async def detect_providers() -> list[ProviderStatus]:
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


async def get_ollama_models() -> list[UnifiedModel]:
    """Get models from Ollama."""
    from .ollama import get_ollama_provider

    provider = get_ollama_provider()
    models = []

    try:
        ollama_models = await provider.list_ollama_models()
        for m in ollama_models:
            models.append(
                UnifiedModel(
                    id=f"ollama/{m.name}",
                    name=m.name,
                    provider=ProviderType.OLLAMA,
                    size_gb=round(m.size_gb, 2),
                    parameter_size=m.parameter_size,
                    is_code_model=m.is_code_model,
                    is_local=True,
                    supports_training=True,  # Can export GGUF for training
                    supports_inference=True,
                )
            )
    except Exception:
        pass

    return models


async def get_lm_studio_models() -> list[UnifiedModel]:
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
                    models.append(
                        UnifiedModel(
                            id=f"lm_studio/{model_id}",
                            name=model_id,
                            provider=ProviderType.LM_STUDIO,
                            is_local=True,
                            supports_inference=True,
                        )
                    )
    except Exception:
        pass

    return models


async def get_nim_models() -> list[UnifiedModel]:
    """Get models served by the configured NVIDIA NIM endpoint (if a key is set)."""
    import re

    import httpx

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return []
    base = os.environ.get("NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1").rstrip("/")
    models = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base}/models", headers={"Authorization": f"Bearer {key}"}
            )
            if response.status_code == 200:
                for m in response.json().get("data", []):
                    model_id = m.get("id", "")
                    if not model_id:
                        continue
                    models.append(
                        UnifiedModel(
                            id=f"nim/{model_id}",
                            name=model_id,
                            provider=ProviderType.NVIDIA_NIM,
                            is_code_model=bool(
                                re.search(r"cod(e|er)|deepseek|starcoder", model_id, re.I)
                            ),
                            is_local=False,
                            supports_inference=True,
                        )
                    )
    except Exception:
        pass

    return models


async def get_anthropic_models() -> list[UnifiedModel]:
    """Get the live Claude catalogue from the Anthropic Models API (if a key is set)."""
    import httpx

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return []
    models = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            )
            if response.status_code == 200:
                for m in response.json().get("data", []):
                    model_id = m.get("id")
                    if not model_id:
                        continue
                    models.append(
                        UnifiedModel(
                            id=f"anthropic/{model_id}",
                            name=m.get("display_name") or model_id,
                            provider=ProviderType.ANTHROPIC,
                            is_code_model=True,  # Claude models are strong coders
                            is_local=False,
                            supports_inference=True,
                        )
                    )
    except Exception:
        pass

    return models


async def get_openai_models() -> list[UnifiedModel]:
    """Get chat-capable models served by OpenAI (if a key is set).

    OpenAI's /v1/models lists embeddings, audio, image, and moderation models too;
    filter to the chat/completion families so the catalogue stays usable.
    """
    import re

    import httpx

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return []
    chat_re = re.compile(r"^(gpt-|o[1-9]|chatgpt)", re.IGNORECASE)
    skip_re = re.compile(
        r"embedding|whisper|tts|dall-e|moderation|audio|realtime|image|transcribe|search",
        re.IGNORECASE,
    )
    models = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            if response.status_code == 200:
                for m in response.json().get("data", []):
                    model_id = m.get("id", "")
                    if not model_id or not chat_re.search(model_id) or skip_re.search(model_id):
                        continue
                    models.append(
                        UnifiedModel(
                            id=f"openai/{model_id}",
                            name=model_id,
                            provider=ProviderType.OPENAI,
                            is_code_model=True,
                            is_local=False,
                            supports_inference=True,
                        )
                    )
    except Exception:
        pass

    return models


# Curated list of recommended models for training
RECOMMENDED_TRAINING_MODELS = [
    UnifiedModel(
        id="hf/google/gemma-4-E2B-it",
        name="Gemma 4 E2B",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="2B",
        is_code_model=True,
        supports_training=True,
        context_length=128000,
        description="Smallest Gemma 4 — trains on a consumer GPU (~8 GB).",
    ),
    UnifiedModel(
        id="hf/google/gemma-4-E4B-it",
        name="Gemma 4 E4B",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="4B",
        is_code_model=True,
        supports_training=True,
        context_length=128000,
        description="Compact Gemma 4 for laptops and small GPUs.",
    ),
    UnifiedModel(
        id="hf/google/gemma-4-26B-A4B-it",
        name="Gemma 4 26B MoE",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="26B (4B active)",
        is_code_model=True,
        supports_training=True,
        context_length=256000,
        description="Mixture-of-experts — strong quality at low active cost on larger GPUs.",
    ),
    UnifiedModel(
        id="hf/Qwen/Qwen3.5-4B",
        name="Qwen3.5 4B",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="4B",
        is_code_model=True,
        supports_training=True,
        context_length=262144,
        description="Small dense Qwen3.5 with strong reasoning.",
    ),
    UnifiedModel(
        id="hf/Qwen/Qwen3.5-9B",
        name="Qwen3.5 9B",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="9B",
        is_code_model=True,
        supports_training=True,
        context_length=262144,
        description="Mid-size dense Qwen3.5.",
    ),
    UnifiedModel(
        id="hf/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        name="DeepSeek-Coder-V2-Lite",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="16B (2.4B active)",
        is_code_model=True,
        supports_training=True,
        context_length=128000,
        description="Efficient MoE code model with long context.",
    ),
    UnifiedModel(
        id="hf/microsoft/phi-4",
        name="Phi-4",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="14B",
        is_code_model=True,
        supports_training=True,
        context_length=16384,
        description="Microsoft's efficient reasoning model.",
    ),
]

# Curated list of teacher models for knowledge distillation
TEACHER_MODELS = [
    UnifiedModel(
        id="anthropic/claude-opus-4-8",
        name="Claude Opus 4.8",
        provider=ProviderType.ANTHROPIC,
        is_code_model=True,
        supports_inference=True,
        description="Most capable teacher for distillation.",
    ),
    UnifiedModel(
        id="anthropic/claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        provider=ProviderType.ANTHROPIC,
        is_code_model=True,
        supports_inference=True,
        description="Fast, capable teacher with a strong quality/cost balance.",
    ),
    UnifiedModel(
        id="openai/gpt-4o",
        name="GPT-4o",
        provider=ProviderType.OPENAI,
        is_code_model=True,
        supports_inference=True,
        description="Strong general-purpose teacher.",
    ),
    UnifiedModel(
        id="hf/Qwen/Qwen2.5-Coder-32B-Instruct",
        name="Qwen2.5-Coder-32B-Instruct",
        provider=ProviderType.HUGGINGFACE,
        parameter_size="32B",
        is_code_model=True,
        supports_inference=True,
        description="Top-tier open-source code model.",
    ),
]


async def get_available_models(
    include_local: bool = True, include_cloud: bool = True, code_only: bool = False
) -> dict[str, list[UnifiedModel]]:
    """
    Get all available models organized by category.

    Returns:
        Dict with keys: 'local', 'training', 'teacher', 'inference'
    """
    result = {
        "local": [],
        "training": RECOMMENDED_TRAINING_MODELS.copy(),
        "teacher": TEACHER_MODELS.copy(),
        "inference": [],
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

    if include_cloud:
        # Live cloud catalogs in parallel: NVIDIA NIM, Anthropic, OpenAI (each only
        # if its API key is set). HF/Unsloth weights appear once served (NIM, or
        # pulled into Ollama under "local").
        for batch in await asyncio.gather(
            get_nim_models(), get_anthropic_models(), get_openai_models()
        ):
            result["inference"].extend(batch)

    if code_only:
        for key in result:
            result[key] = [m for m in result[key] if m.is_code_model]

    return result


# Sync wrapper
def detect_providers_sync() -> list[ProviderStatus]:
    """Synchronous wrapper for detect_providers."""
    return asyncio.run(detect_providers())


def get_available_models_sync(**kwargs) -> dict[str, list[UnifiedModel]]:
    """Synchronous wrapper for get_available_models."""
    return asyncio.run(get_available_models(**kwargs))
