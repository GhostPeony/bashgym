"""
Unified NeMo Microservices Client

Provides a professional, SDK-based interface to NVIDIA NeMo Microservices.
Falls back to direct HTTP calls when the SDK is not installed.

Usage:
    from bashgym.integrations import NeMoClient, NeMoClientConfig

    config = NeMoClientConfig(
        base_url="http://localhost:8000",
        inference_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-..."
    )

    client = NeMoClient(config)

    # Training/Customization
    job = client.customization.create_job(...)

    # Evaluation
    result = client.evaluation.run_benchmark(...)

    # Inference (NIM)
    response = client.inference.chat_completion(...)
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Iterator, AsyncIterator, Union
from datetime import datetime, timezone
from enum import Enum
import logging

# Try to import the official NeMo Microservices SDK
try:
    from nemo_microservices import NeMoMicroservices, AsyncNeMoMicroservices
    from nemo_microservices import (
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        RateLimitError,
        InternalServerError,
    )
    NEMO_SDK_AVAILABLE = True
except ImportError:
    NEMO_SDK_AVAILABLE = False
    # Define fallback exceptions
    class BadRequestError(Exception): pass
    class AuthenticationError(Exception): pass
    class PermissionDeniedError(Exception): pass
    class NotFoundError(Exception): pass
    class RateLimitError(Exception): pass
    class InternalServerError(Exception): pass

# HTTP client for fallback mode
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class CustomizationStrategy(Enum):
    """Training strategies supported by NeMo Customizer."""
    SFT = "sft"
    LORA = "lora"
    P_TUNING = "p_tuning"
    DPO = "dpo"


@dataclass
class NeMoClientConfig:
    """Configuration for the NeMo Microservices client."""

    # Endpoints
    base_url: str = "http://localhost:8000"
    inference_url: str = "https://integrate.api.nvidia.com/v1"

    # Authentication
    api_key: Optional[str] = None

    # Client settings
    timeout: float = 60.0
    max_retries: int = 2

    # Feature flags
    use_sdk: bool = True  # Use official SDK when available

    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("NVIDIA_API_KEY")


@dataclass
class CustomizationJob:
    """Represents a NeMo Customizer training job."""

    job_id: str
    status: str
    model: str
    strategy: str
    created_at: str
    updated_at: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    output_model: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "CustomizationJob":
        """Create from API response."""
        return cls(
            job_id=data.get("id", data.get("job_id", "")),
            status=data.get("status", "unknown"),
            model=data.get("model", ""),
            strategy=data.get("strategy", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
            metrics=data.get("metrics", {}),
            error=data.get("error"),
            output_model=data.get("output_model"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "model": self.model,
            "strategy": self.strategy,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metrics": self.metrics,
            "error": self.error,
            "output_model": self.output_model,
        }


@dataclass
class EvaluationJob:
    """Represents a NeMo Evaluator job."""

    job_id: str
    status: str
    benchmark: str
    created_at: str
    metrics: Dict[str, float] = field(default_factory=dict)
    samples_evaluated: int = 0
    samples_passed: int = 0
    error: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        if self.samples_evaluated == 0:
            return 0.0
        return self.samples_passed / self.samples_evaluated

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "EvaluationJob":
        return cls(
            job_id=data.get("id", data.get("job_id", "")),
            status=data.get("status", "unknown"),
            benchmark=data.get("benchmark", ""),
            created_at=data.get("created_at", ""),
            metrics=data.get("metrics", {}),
            samples_evaluated=data.get("samples_evaluated", 0),
            samples_passed=data.get("samples_passed", 0),
            error=data.get("error"),
        )


class CustomizationClient:
    """
    Client for NeMo Customizer (fine-tuning) operations.

    Wraps the official SDK or provides HTTP fallback.
    """

    def __init__(self, parent: "NeMoClient"):
        self._parent = parent
        self._sdk_client = parent._sdk_client
        self._http_client = parent._http_client
        self._config = parent._config

    def create_job(
        self,
        model: str,
        training_data: str,
        strategy: Union[CustomizationStrategy, str] = CustomizationStrategy.SFT,
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_model_name: Optional[str] = None,
    ) -> CustomizationJob:
        """
        Create a new customization (fine-tuning) job.

        Args:
            model: Base model to fine-tune (e.g., "meta/llama-3.1-8b-instruct")
            training_data: Path or URI to training dataset
            strategy: Training strategy (sft, lora, dpo, p_tuning)
            hyperparameters: Training hyperparameters
            output_model_name: Name for the output model

        Returns:
            CustomizationJob with job details
        """
        if isinstance(strategy, CustomizationStrategy):
            strategy = strategy.value

        hyperparameters = hyperparameters or {}

        if self._sdk_client and NEMO_SDK_AVAILABLE:
            return self._create_job_sdk(
                model, training_data, strategy, hyperparameters, output_model_name
            )
        else:
            return self._create_job_http(
                model, training_data, strategy, hyperparameters, output_model_name
            )

    def _create_job_sdk(
        self,
        model: str,
        training_data: str,
        strategy: str,
        hyperparameters: Dict[str, Any],
        output_model_name: Optional[str],
    ) -> CustomizationJob:
        """Create job using official SDK."""
        try:
            # Create customization config
            config = self._sdk_client.customization.configs.create(
                max_seq_length=hyperparameters.get("max_seq_length", 4096),
                training_options=[{
                    "training_type": strategy,
                    "epochs": hyperparameters.get("num_epochs", 3),
                    "learning_rate": hyperparameters.get("learning_rate", 2e-5),
                    "batch_size": hyperparameters.get("batch_size", 4),
                    "lora_r": hyperparameters.get("lora_r", 16),
                    "lora_alpha": hyperparameters.get("lora_alpha", 32),
                }],
            )

            # Create the job
            job = self._sdk_client.customization.jobs.create(
                config_id=config.id,
                model=model,
                training_data=training_data,
                output_model=output_model_name,
            )

            return CustomizationJob.from_api_response({
                "job_id": job.id,
                "status": job.status,
                "model": model,
                "strategy": strategy,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        except (BadRequestError, AuthenticationError, NotFoundError) as e:
            logger.error(f"SDK error creating customization job: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating customization job: {e}")
            raise

    def _create_job_http(
        self,
        model: str,
        training_data: str,
        strategy: str,
        hyperparameters: Dict[str, Any],
        output_model_name: Optional[str],
    ) -> CustomizationJob:
        """Create job using HTTP fallback."""
        if not self._http_client:
            raise RuntimeError("No HTTP client available. Install httpx or nemo-microservices.")

        payload = {
            "model": model,
            "training_data": training_data,
            "strategy": strategy,
            "hyperparameters": hyperparameters,
        }
        if output_model_name:
            payload["output_model"] = output_model_name

        response = self._http_client.post(
            f"{self._config.base_url}/v1/customization/jobs",
            json=payload,
        )

        self._check_http_response(response)
        return CustomizationJob.from_api_response(response.json())

    def get_job(self, job_id: str) -> CustomizationJob:
        """Get the status of a customization job."""
        if self._sdk_client and NEMO_SDK_AVAILABLE:
            job = self._sdk_client.customization.jobs.retrieve(job_id)
            return CustomizationJob.from_api_response({
                "job_id": job.id,
                "status": job.status,
                "model": getattr(job, "model", ""),
                "strategy": getattr(job, "strategy", ""),
                "created_at": getattr(job, "created_at", ""),
                "metrics": getattr(job, "metrics", {}),
                "error": getattr(job, "error", None),
                "output_model": getattr(job, "output_model", None),
            })
        else:
            response = self._http_client.get(
                f"{self._config.base_url}/v1/customization/jobs/{job_id}"
            )
            self._check_http_response(response)
            return CustomizationJob.from_api_response(response.json())

    def list_jobs(self, limit: int = 100) -> Iterator[CustomizationJob]:
        """List customization jobs with auto-pagination."""
        if self._sdk_client and NEMO_SDK_AVAILABLE:
            # SDK handles pagination automatically
            for job in self._sdk_client.customization.jobs.list():
                yield CustomizationJob.from_api_response({
                    "job_id": job.id,
                    "status": job.status,
                    "model": getattr(job, "model", ""),
                    "strategy": getattr(job, "strategy", ""),
                    "created_at": getattr(job, "created_at", ""),
                })
        else:
            # Manual pagination for HTTP fallback
            offset = 0
            while True:
                response = self._http_client.get(
                    f"{self._config.base_url}/v1/customization/jobs",
                    params={"limit": min(limit, 100), "offset": offset}
                )
                self._check_http_response(response)
                data = response.json()

                jobs = data.get("jobs", data.get("data", []))
                if not jobs:
                    break

                for job_data in jobs:
                    yield CustomizationJob.from_api_response(job_data)

                if len(jobs) < 100:
                    break
                offset += len(jobs)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running customization job."""
        if self._sdk_client and NEMO_SDK_AVAILABLE:
            self._sdk_client.customization.jobs.cancel(job_id)
            return True
        else:
            response = self._http_client.post(
                f"{self._config.base_url}/v1/customization/jobs/{job_id}/cancel"
            )
            self._check_http_response(response)
            return True

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 3600,
    ) -> CustomizationJob:
        """Wait for a job to complete."""
        import time
        start_time = time.time()

        while True:
            job = self.get_job(job_id)

            if job.status in ("completed", "succeeded"):
                return job
            elif job.status in ("failed", "cancelled", "error"):
                raise RuntimeError(f"Job {job_id} failed: {job.error}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            time.sleep(poll_interval)

    def _check_http_response(self, response) -> None:
        """Check HTTP response and raise appropriate exception."""
        if response.status_code >= 400:
            error_msg = response.text
            if response.status_code == 400:
                raise BadRequestError(error_msg)
            elif response.status_code == 401:
                raise AuthenticationError(error_msg)
            elif response.status_code == 403:
                raise PermissionDeniedError(error_msg)
            elif response.status_code == 404:
                raise NotFoundError(error_msg)
            elif response.status_code == 429:
                raise RateLimitError(error_msg)
            elif response.status_code >= 500:
                raise InternalServerError(error_msg)


class EvaluationClient:
    """
    Client for NeMo Evaluator operations.

    Supports academic benchmarks and LLM-as-Judge scoring.
    """

    def __init__(self, parent: "NeMoClient"):
        self._parent = parent
        self._sdk_client = parent._sdk_client
        self._http_client = parent._http_client
        self._config = parent._config

    def create_job(
        self,
        model: str,
        benchmark: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluationJob:
        """
        Create an evaluation job.

        Args:
            model: Model to evaluate
            benchmark: Benchmark name (humaneval, mbpp, bigcodebench, etc.)
            config: Evaluation configuration

        Returns:
            EvaluationJob with job details
        """
        config = config or {}

        if self._sdk_client and NEMO_SDK_AVAILABLE:
            try:
                job = self._sdk_client.evaluation.jobs.create(
                    model=model,
                    benchmark=benchmark,
                    config=config,
                )
                return EvaluationJob.from_api_response({
                    "job_id": job.id,
                    "status": job.status,
                    "benchmark": benchmark,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning(f"SDK evaluation failed, falling back to HTTP: {e}")

        # HTTP fallback
        if self._http_client:
            response = self._http_client.post(
                f"{self._config.base_url}/v1/evaluation/jobs",
                json={
                    "model": model,
                    "benchmark": benchmark,
                    "config": config,
                }
            )
            if response.status_code == 200:
                return EvaluationJob.from_api_response(response.json())

        # Local fallback
        return EvaluationJob(
            job_id=f"local_{benchmark}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            status="pending",
            benchmark=benchmark,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_job(self, job_id: str) -> EvaluationJob:
        """Get evaluation job status."""
        if self._sdk_client and NEMO_SDK_AVAILABLE:
            try:
                job = self._sdk_client.evaluation.jobs.retrieve(job_id)
                return EvaluationJob.from_api_response({
                    "job_id": job.id,
                    "status": job.status,
                    "benchmark": getattr(job, "benchmark", ""),
                    "created_at": getattr(job, "created_at", ""),
                    "metrics": getattr(job, "metrics", {}),
                    "samples_evaluated": getattr(job, "samples_evaluated", 0),
                    "samples_passed": getattr(job, "samples_passed", 0),
                })
            except Exception:
                pass

        if self._http_client:
            response = self._http_client.get(
                f"{self._config.base_url}/v1/evaluation/jobs/{job_id}"
            )
            if response.status_code == 200:
                return EvaluationJob.from_api_response(response.json())

        raise NotFoundError(f"Job {job_id} not found")


class InferenceClient:
    """
    Client for NIM inference operations.

    Provides OpenAI-compatible chat completions.
    """

    def __init__(self, parent: "NeMoClient"):
        self._parent = parent
        self._http_client = parent._http_client
        self._config = parent._config

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a chat completion using NIM.

        Args:
            model: Model ID (e.g., "meta/llama-3.1-70b-instruct")
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            OpenAI-compatible response dict
        """
        if not self._http_client:
            raise RuntimeError("HTTP client required for inference. Install httpx.")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = self._http_client.post(
            f"{self._config.inference_url}/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise InternalServerError(f"Inference failed: {response.text}")

        return response.json()

    def completion(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a text completion using NIM."""
        if not self._http_client:
            raise RuntimeError("HTTP client required for inference. Install httpx.")

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = self._http_client.post(
            f"{self._config.inference_url}/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise InternalServerError(f"Inference failed: {response.text}")

        return response.json()


class NeMoClient:
    """
    Unified client for NVIDIA NeMo Microservices.

    Provides access to:
    - NeMo Customizer (fine-tuning): client.customization
    - NeMo Evaluator (benchmarks): client.evaluation
    - NIM (inference): client.inference

    Uses the official nemo-microservices SDK when available,
    with automatic fallback to direct HTTP calls.

    Example:
        config = NeMoClientConfig(
            base_url="http://localhost:8000",
            inference_url="https://integrate.api.nvidia.com/v1",
            api_key="nvapi-..."
        )

        client = NeMoClient(config)

        # Fine-tuning
        job = client.customization.create_job(
            model="meta/llama-3.1-8b-instruct",
            training_data="s3://bucket/data.jsonl",
            strategy="sft"
        )

        # Wait for completion
        completed_job = client.customization.wait_for_completion(job.job_id)

        # Inference with fine-tuned model
        response = client.inference.chat_completion(
            model=completed_job.output_model,
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    def __init__(self, config: Optional[NeMoClientConfig] = None):
        """
        Initialize the NeMo client.

        Args:
            config: Client configuration. Uses defaults if not provided.
        """
        self._config = config or NeMoClientConfig()
        self._sdk_client = None
        self._http_client = None

        # Initialize SDK client if available and enabled
        if NEMO_SDK_AVAILABLE and self._config.use_sdk:
            try:
                self._sdk_client = NeMoMicroservices(
                    base_url=self._config.base_url,
                    inference_base_url=self._config.inference_url,
                    api_key=self._config.api_key,
                    timeout=self._config.timeout,
                    max_retries=self._config.max_retries,
                )
                logger.info("NeMo SDK client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SDK client: {e}")
                self._sdk_client = None

        # Initialize HTTP fallback client
        if HTTPX_AVAILABLE:
            headers = {"Content-Type": "application/json"}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            self._http_client = httpx.Client(
                timeout=self._config.timeout,
                headers=headers,
            )

        if not self._sdk_client and not self._http_client:
            raise RuntimeError(
                "No client available. Install nemo-microservices or httpx:\n"
                "  pip install nemo-microservices\n"
                "  # or\n"
                "  pip install httpx"
            )

        # Initialize sub-clients
        self._customization = CustomizationClient(self)
        self._evaluation = EvaluationClient(self)
        self._inference = InferenceClient(self)

    @property
    def customization(self) -> CustomizationClient:
        """Access NeMo Customizer (fine-tuning) operations."""
        return self._customization

    @property
    def evaluation(self) -> EvaluationClient:
        """Access NeMo Evaluator operations."""
        return self._evaluation

    @property
    def inference(self) -> InferenceClient:
        """Access NIM inference operations."""
        return self._inference

    @property
    def is_sdk_mode(self) -> bool:
        """Check if using the official SDK."""
        return self._sdk_client is not None

    def close(self) -> None:
        """Close all client connections."""
        if self._http_client:
            self._http_client.close()

    def __enter__(self) -> "NeMoClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AsyncNeMoClient:
    """
    Async version of the NeMo client.

    Example:
        async with AsyncNeMoClient(config) as client:
            job = await client.customization.create_job(...)
    """

    def __init__(self, config: Optional[NeMoClientConfig] = None):
        self._config = config or NeMoClientConfig()
        self._sdk_client = None
        self._http_client = None

        # Initialize async SDK client if available
        if NEMO_SDK_AVAILABLE and self._config.use_sdk:
            try:
                self._sdk_client = AsyncNeMoMicroservices(
                    base_url=self._config.base_url,
                    inference_base_url=self._config.inference_url,
                    api_key=self._config.api_key,
                    timeout=self._config.timeout,
                    max_retries=self._config.max_retries,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize async SDK client: {e}")

        # Initialize async HTTP fallback
        if HTTPX_AVAILABLE:
            headers = {"Content-Type": "application/json"}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            self._http_client = httpx.AsyncClient(
                timeout=self._config.timeout,
                headers=headers,
            )

    @property
    def is_sdk_mode(self) -> bool:
        return self._sdk_client is not None

    async def close(self) -> None:
        """Close all client connections."""
        if self._http_client:
            await self._http_client.aclose()

    async def __aenter__(self) -> "AsyncNeMoClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # Async inference helper
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        """Async chat completion."""
        if not self._http_client:
            raise RuntimeError("Async HTTP client required")

        response = await self._http_client.post(
            f"{self._config.inference_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )

        if response.status_code != 200:
            raise InternalServerError(f"Inference failed: {response.text}")

        return response.json()


# Convenience function for quick setup
def create_client(
    base_url: Optional[str] = None,
    inference_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> NeMoClient:
    """
    Create a NeMo client with common defaults.

    Args:
        base_url: NeMo Microservices endpoint
        inference_url: NIM inference endpoint
        api_key: NVIDIA API key (or set NVIDIA_API_KEY env var)

    Returns:
        Configured NeMoClient instance
    """
    config = NeMoClientConfig(
        base_url=base_url or os.environ.get("NEMO_ENDPOINT", "http://localhost:8000"),
        inference_url=inference_url or os.environ.get("NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
    )
    return NeMoClient(config)
