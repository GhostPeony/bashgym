"""
HuggingFace Inference Providers

Provides access to HuggingFace Inference API (serverless and dedicated endpoints).
Supports text generation, embeddings, and text classification.

Features:
- Serverless inference with automatic routing
- Support for :fastest and :cheapest routing suffixes
- Usage tracking (tokens, cost)
- Graceful handling of missing huggingface_hub library

Usage:
    from bashgym.integrations.huggingface import (
        HFInferenceClient,
        HFInferenceConfig,
    )

    client = HFInferenceClient(token="hf_...")

    # Text generation
    response = client.generate(
        model="meta-llama/Llama-3.1-8B-Instruct",
        prompt="Write a function that",
        max_tokens=256,
        temperature=0.7,
    )

    # Embeddings
    vectors = client.embed(
        model="sentence-transformers/all-MiniLM-L6-v2",
        texts=["Hello world", "How are you?"],
    )

    # Classification
    scores = client.classify(
        model="facebook/bart-large-mnli",
        text="This is a great product!",
    )

    # With routing suffix
    response = client.generate(
        model="meta-llama/Llama-3.1-8B-Instruct:fastest",
        prompt="Hello",
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from .client import (
    HuggingFaceClient,
    HFQuotaExceededError,
    HFAuthError,
    HFError,
    HF_HUB_AVAILABLE,
)

logger = logging.getLogger(__name__)


# Try to import the inference client from huggingface_hub
try:
    from huggingface_hub import InferenceClient as _HFInferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False
    _HFInferenceClient = None


# =============================================================================
# Enums
# =============================================================================

class InferenceProvider(Enum):
    """HuggingFace Inference API providers."""
    AUTO = "auto"
    SERVERLESS = "serverless"
    DEDICATED = "dedicated"
    FINETUNED = "finetuned"


class RoutingStrategy(Enum):
    """Routing strategies for model selection."""
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    QUALITY = "quality"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HFInferenceConfig:
    """Configuration for HuggingFace Inference Client."""

    provider: str = "auto"
    """Inference provider: 'auto', 'serverless', 'dedicated', or 'finetuned'."""

    routing: str = "fastest"
    """Routing strategy: 'fastest', 'cheapest', or 'quality'."""

    bill_to: Optional[str] = None
    """Organization ID to bill for inference usage (for Pro/Enterprise)."""

    timeout: float = 30.0
    """Request timeout in seconds."""

    def validate(self) -> List[str]:
        """Validate the configuration."""
        errors = []

        valid_providers = [p.value for p in InferenceProvider]
        if self.provider not in valid_providers:
            errors.append(f"Invalid provider '{self.provider}'. Valid: {valid_providers}")

        valid_routing = [r.value for r in RoutingStrategy]
        if self.routing not in valid_routing:
            errors.append(f"Invalid routing '{self.routing}'. Valid: {valid_routing}")

        if self.timeout <= 0:
            errors.append("timeout must be positive")

        return errors


@dataclass
class InferenceUsage:
    """Tracks inference usage for billing and monitoring."""

    provider: str
    """Provider used for inference (serverless, dedicated, etc.)."""

    model: str
    """Model ID used for inference."""

    input_tokens: int = 0
    """Number of input tokens processed."""

    output_tokens: int = 0
    """Number of output tokens generated."""

    cost_usd: float = 0.0
    """Estimated cost in USD (if available)."""

    latency_ms: Optional[float] = None
    """Request latency in milliseconds."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
        }

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class GenerationResponse:
    """Response from text generation."""

    text: str
    """Generated text."""

    usage: InferenceUsage
    """Usage information."""

    finish_reason: Optional[str] = None
    """Reason for generation stopping (length, eos, etc.)."""

    raw_response: Optional[Dict[str, Any]] = None
    """Raw API response for debugging."""


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    embeddings: List[List[float]]
    """List of embedding vectors."""

    usage: InferenceUsage
    """Usage information."""

    raw_response: Optional[Dict[str, Any]] = None
    """Raw API response for debugging."""


@dataclass
class ClassificationResponse:
    """Response from text classification."""

    labels: Dict[str, float]
    """Label scores as {label: score} dictionary."""

    usage: InferenceUsage
    """Usage information."""

    top_label: Optional[str] = None
    """Highest scoring label."""

    raw_response: Optional[Dict[str, Any]] = None
    """Raw API response for debugging."""


# =============================================================================
# Pricing (approximate, for cost estimation)
# =============================================================================

# Serverless inference pricing per 1M tokens (approximate as of 2025)
SERVERLESS_PRICING = {
    # Default pricing for unknown models
    "default": {"input": 0.10, "output": 0.20},
    # Common model families
    "llama": {"input": 0.10, "output": 0.15},
    "mistral": {"input": 0.10, "output": 0.15},
    "qwen": {"input": 0.08, "output": 0.12},
    "codellama": {"input": 0.10, "output": 0.15},
    "phi": {"input": 0.05, "output": 0.08},
    # Embedding models (typically much cheaper)
    "sentence-transformers": {"input": 0.01, "output": 0.0},
    "e5": {"input": 0.01, "output": 0.0},
    "bge": {"input": 0.01, "output": 0.0},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate inference cost based on model and tokens."""
    model_lower = model.lower()

    # Find matching pricing
    pricing = SERVERLESS_PRICING["default"]
    for key, price in SERVERLESS_PRICING.items():
        if key in model_lower:
            pricing = price
            break

    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token)."""
    return max(1, len(text) // 4)


# =============================================================================
# HF Inference Client
# =============================================================================

class HFInferenceClient:
    """
    Client for HuggingFace Inference API.

    Provides access to:
    - Text generation (serverless and dedicated endpoints)
    - Embedding generation
    - Text classification

    Supports routing suffixes:
    - :fastest - Route to fastest available provider
    - :cheapest - Route to cheapest available provider

    Example:
        client = HFInferenceClient(token="hf_...")

        # Generate text
        response = client.generate(
            model="meta-llama/Llama-3.1-8B-Instruct",
            prompt="Write a Python function",
            max_tokens=256,
        )
        print(response.text)

        # With routing
        response = client.generate(
            model="meta-llama/Llama-3.1-8B-Instruct:cheapest",
            prompt="Hello",
        )

        # Embeddings
        vectors = client.embed(
            model="sentence-transformers/all-MiniLM-L6-v2",
            texts=["Hello", "World"],
        )
    """

    def __init__(
        self,
        token: Optional[str] = None,
        config: Optional[HFInferenceConfig] = None,
        hf_client: Optional[HuggingFaceClient] = None,
    ):
        """
        Initialize the HuggingFace Inference Client.

        Args:
            token: HuggingFace API token. If None, uses cached token.
            config: Inference configuration.
            hf_client: Existing HuggingFaceClient instance for auth.
        """
        self._config = config or HFInferenceConfig()
        self._hf_client = hf_client
        self._token = token
        self._inference_client = None
        self._initialized = False
        self._init_error: Optional[str] = None

        # Validate config
        errors = self._config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Check library availability
        if not HF_INFERENCE_AVAILABLE:
            self._init_error = (
                "huggingface_hub not installed or InferenceClient not available. "
                "Install with: pip install huggingface_hub>=0.20.0"
            )

    def _ensure_initialized(self) -> None:
        """Lazily initialize the inference client."""
        if self._initialized:
            return

        self._initialized = True

        if not HF_INFERENCE_AVAILABLE:
            logger.warning(self._init_error)
            return

        # Resolve token
        token = self._token
        if not token and self._hf_client:
            token = self._hf_client.token

        if not token:
            # Try to get from settings or cached token
            try:
                from bashgym.config import get_settings
                token = get_settings().huggingface.token
            except Exception:
                pass

            if not token:
                try:
                    from huggingface_hub import HfFolder
                    token = HfFolder.get_token()
                except Exception:
                    pass

        if not token:
            self._init_error = "No HuggingFace token available"
            logger.warning(self._init_error)
            return

        try:
            self._inference_client = _HFInferenceClient(
                token=token,
                timeout=self._config.timeout,
            )
            logger.info("HF Inference client initialized")
        except Exception as e:
            self._init_error = f"Failed to initialize inference client: {e}"
            logger.error(self._init_error)

    @property
    def is_available(self) -> bool:
        """Check if the inference client is available."""
        self._ensure_initialized()
        return self._inference_client is not None

    def _parse_model_routing(self, model: str) -> tuple[str, Optional[str]]:
        """
        Parse model ID and routing suffix.

        Args:
            model: Model ID with optional :fastest or :cheapest suffix

        Returns:
            Tuple of (model_id, routing_strategy)
        """
        routing = None

        if model.endswith(":fastest"):
            model = model[:-8]
            routing = "fastest"
        elif model.endswith(":cheapest"):
            model = model[:-9]
            routing = "cheapest"
        elif model.endswith(":quality"):
            model = model[:-8]
            routing = "quality"

        return model, routing

    def _require_available(self) -> None:
        """Require that the client is available."""
        self._ensure_initialized()
        if not self.is_available:
            error_msg = self._init_error or "HF Inference client not available"
            raise HFAuthError(error_msg)

    def _check_quota_error(self, error: Exception) -> None:
        """Check if an error indicates quota exceeded."""
        error_str = str(error).lower()
        quota_indicators = [
            "quota",
            "rate limit",
            "too many requests",
            "429",
            "exceeded",
            "limit reached",
        ]
        if any(indicator in error_str for indicator in quota_indicators):
            raise HFQuotaExceededError(f"HuggingFace quota exceeded: {error}")

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using HuggingFace Inference API.

        Args:
            model: Model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
                   Can include :fastest or :cheapest suffix for routing.
            prompt: Input prompt for generation.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 to 2.0).
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            **kwargs: Additional parameters passed to the API.

        Returns:
            GenerationResponse with generated text and usage.

        Raises:
            HFAuthError: If client not available.
            HFQuotaExceededError: If quota exceeded.
            HFError: For other inference errors.
        """
        self._require_available()

        # Parse routing suffix
        model, routing = self._parse_model_routing(model)
        routing = routing or self._config.routing

        logger.debug(f"Generating with model={model}, routing={routing}")

        import time
        start_time = time.time()

        try:
            # Use text_generation endpoint
            response = self._inference_client.text_generation(
                prompt=prompt,
                model=model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop,
                details=True,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            if hasattr(response, "generated_text"):
                text = response.generated_text
                finish_reason = getattr(response, "finish_reason", None)
                # Extract token counts if available
                input_tokens = getattr(
                    getattr(response, "details", None),
                    "prefill_tokens",
                    _estimate_tokens(prompt),
                ) if hasattr(response, "details") else _estimate_tokens(prompt)
                output_tokens = getattr(
                    getattr(response, "details", None),
                    "generated_tokens",
                    _estimate_tokens(text),
                ) if hasattr(response, "details") else _estimate_tokens(text)
            else:
                # Simple string response
                text = str(response)
                finish_reason = None
                input_tokens = _estimate_tokens(prompt)
                output_tokens = _estimate_tokens(text)

            usage = InferenceUsage(
                provider=routing or "serverless",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=_estimate_cost(model, input_tokens, output_tokens),
                latency_ms=latency_ms,
            )

            return GenerationResponse(
                text=text,
                usage=usage,
                finish_reason=finish_reason,
                raw_response={"response": str(response)} if response else None,
            )

        except Exception as e:
            self._check_quota_error(e)
            logger.error(f"Generation failed: {e}")
            raise HFError(f"Generation failed: {e}")

    def embed(
        self,
        model: str,
        texts: Union[str, List[str]],
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Generate embeddings using HuggingFace Inference API.

        Args:
            model: Embedding model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            texts: Single text or list of texts to embed.
            **kwargs: Additional parameters passed to the API.

        Returns:
            EmbeddingResponse with embedding vectors.

        Raises:
            HFAuthError: If client not available.
            HFQuotaExceededError: If quota exceeded.
            HFError: For other inference errors.
        """
        self._require_available()

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]

        # Parse routing suffix
        model, routing = self._parse_model_routing(model)

        logger.debug(f"Embedding {len(texts)} texts with model={model}")

        import time
        start_time = time.time()

        try:
            # Use feature_extraction endpoint
            embeddings = self._inference_client.feature_extraction(
                text=texts,
                model=model,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Handle response format
            if isinstance(embeddings, list):
                if len(embeddings) > 0 and isinstance(embeddings[0], list):
                    # List of vectors
                    vectors = embeddings
                else:
                    # Single vector
                    vectors = [embeddings]
            else:
                # Numpy array or similar
                vectors = [list(embeddings)]

            # Estimate tokens (embeddings don't have output tokens)
            input_tokens = sum(_estimate_tokens(t) for t in texts)

            usage = InferenceUsage(
                provider=routing or "serverless",
                model=model,
                input_tokens=input_tokens,
                output_tokens=0,
                cost_usd=_estimate_cost(model, input_tokens, 0),
                latency_ms=latency_ms,
            )

            return EmbeddingResponse(
                embeddings=vectors,
                usage=usage,
                raw_response={"count": len(vectors)},
            )

        except Exception as e:
            self._check_quota_error(e)
            logger.error(f"Embedding failed: {e}")
            raise HFError(f"Embedding failed: {e}")

    def classify(
        self,
        model: str,
        text: str,
        candidate_labels: Optional[List[str]] = None,
        **kwargs,
    ) -> ClassificationResponse:
        """
        Classify text using HuggingFace Inference API.

        Args:
            model: Classification model ID (e.g., "facebook/bart-large-mnli")
            text: Text to classify.
            candidate_labels: Labels for zero-shot classification (if applicable).
            **kwargs: Additional parameters passed to the API.

        Returns:
            ClassificationResponse with label scores.

        Raises:
            HFAuthError: If client not available.
            HFQuotaExceededError: If quota exceeded.
            HFError: For other inference errors.
        """
        self._require_available()

        # Parse routing suffix
        model, routing = self._parse_model_routing(model)

        logger.debug(f"Classifying text with model={model}")

        import time
        start_time = time.time()

        try:
            if candidate_labels:
                # Zero-shot classification
                response = self._inference_client.zero_shot_classification(
                    text=text,
                    candidate_labels=candidate_labels,
                    model=model,
                    **kwargs,
                )
            else:
                # Standard text classification
                response = self._inference_client.text_classification(
                    text=text,
                    model=model,
                    **kwargs,
                )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response into label scores
            labels: Dict[str, float] = {}
            top_label = None
            max_score = -1

            if isinstance(response, list):
                # Standard classification returns list of {label, score}
                for item in response:
                    if isinstance(item, dict):
                        label = item.get("label", str(item))
                        score = item.get("score", 0.0)
                    else:
                        label = str(item)
                        score = 0.0
                    labels[label] = score
                    if score > max_score:
                        max_score = score
                        top_label = label
            elif isinstance(response, dict):
                # Zero-shot returns {labels: [], scores: []}
                resp_labels = response.get("labels", [])
                resp_scores = response.get("scores", [])
                for label, score in zip(resp_labels, resp_scores):
                    labels[label] = score
                    if score > max_score:
                        max_score = score
                        top_label = label

            input_tokens = _estimate_tokens(text)

            usage = InferenceUsage(
                provider=routing or "serverless",
                model=model,
                input_tokens=input_tokens,
                output_tokens=0,
                cost_usd=_estimate_cost(model, input_tokens, 0),
                latency_ms=latency_ms,
            )

            return ClassificationResponse(
                labels=labels,
                usage=usage,
                top_label=top_label,
                raw_response={"response": response} if response else None,
            )

        except Exception as e:
            self._check_quota_error(e)
            logger.error(f"Classification failed: {e}")
            raise HFError(f"Classification failed: {e}")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> GenerationResponse:
        """
        Chat completion using HuggingFace Inference API.

        Args:
            model: Model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional parameters.

        Returns:
            GenerationResponse with assistant message.
        """
        self._require_available()

        # Parse routing suffix
        model, routing = self._parse_model_routing(model)

        logger.debug(f"Chat completion with model={model}")

        import time
        start_time = time.time()

        try:
            # Use chat_completion endpoint
            response = self._inference_client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract text from response
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                text = choice.message.content
                finish_reason = getattr(choice, "finish_reason", None)
            else:
                text = str(response)
                finish_reason = None

            # Extract usage from response
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0)
                output_tokens = getattr(response.usage, "completion_tokens", 0)
            else:
                input_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)
                output_tokens = _estimate_tokens(text)

            usage = InferenceUsage(
                provider=routing or "serverless",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=_estimate_cost(model, input_tokens, output_tokens),
                latency_ms=latency_ms,
            )

            return GenerationResponse(
                text=text,
                usage=usage,
                finish_reason=finish_reason,
                raw_response={"response": str(response)} if response else None,
            )

        except Exception as e:
            self._check_quota_error(e)
            logger.error(f"Chat completion failed: {e}")
            raise HFError(f"Chat completion failed: {e}")

    def __repr__(self) -> str:
        """String representation."""
        if self.is_available:
            return f"<HFInferenceClient provider={self._config.provider} routing={self._config.routing}>"
        elif self._init_error:
            return f"<HFInferenceClient disabled: {self._init_error}>"
        else:
            return "<HFInferenceClient disabled>"


# =============================================================================
# Singleton Access
# =============================================================================

_inference_client: Optional[HFInferenceClient] = None


def get_inference_client(
    token: Optional[str] = None,
    config: Optional[HFInferenceConfig] = None,
    force_new: bool = False,
) -> HFInferenceClient:
    """
    Get or create the global HF Inference Client instance.

    Args:
        token: HuggingFace API token.
        config: Inference configuration.
        force_new: Force creation of new client instance.

    Returns:
        HFInferenceClient instance (singleton unless force_new=True)
    """
    global _inference_client

    if force_new or _inference_client is None or token is not None:
        _inference_client = HFInferenceClient(token=token, config=config)

    return _inference_client


def reset_inference_client() -> None:
    """Reset the global inference client instance (for testing)."""
    global _inference_client
    _inference_client = None
