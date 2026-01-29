# HuggingFace Pro Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate HuggingFace Pro features (Jobs, Inference Providers, ZeroGPU Spaces, Dataset Storage) into Bash Gym.

**Architecture:** New `bashgym/integrations/huggingface/` module with separate files for each HF feature, all wrapped by a central client that handles Pro detection and graceful degradation.

**Tech Stack:** `huggingface_hub` library, FastAPI routes, WebSocket events, React components

---

## Task 1: Core HuggingFace Client & Settings

**Files:**
- Modify: `bashgym/config.py`
- Create: `bashgym/integrations/huggingface/__init__.py`
- Create: `bashgym/integrations/huggingface/client.py`
- Test: `tests/test_hf_integration.py`

### Step 1: Write the failing test

```python
# tests/test_hf_integration.py
"""Tests for HuggingFace integration."""

import pytest
from unittest.mock import patch, MagicMock

class TestHuggingFaceSettings:
    """Test HF settings in config."""

    def test_hf_settings_defaults(self):
        """HF settings should have sensible defaults."""
        from bashgym.config import HuggingFaceSettings

        settings = HuggingFaceSettings()
        assert settings.token == ""
        assert settings.username == ""
        assert settings.default_org is None
        assert settings.pro_enabled is False

    def test_hf_settings_from_env(self):
        """HF settings should load from environment."""
        import os
        os.environ["HF_TOKEN"] = "hf_test_token"
        os.environ["HF_USERNAME"] = "testuser"

        from bashgym.config import reload_settings
        settings = reload_settings()

        assert settings.huggingface.token == "hf_test_token"
        assert settings.huggingface.username == "testuser"

        # Cleanup
        del os.environ["HF_TOKEN"]
        del os.environ["HF_USERNAME"]


class TestHuggingFaceClient:
    """Test core HF client."""

    def test_client_disabled_without_token(self):
        """Client should be disabled without HF_TOKEN."""
        from bashgym.integrations.huggingface import HuggingFaceClient

        client = HuggingFaceClient(token="")
        assert client.enabled is False
        assert client.pro_enabled is False

    def test_client_enabled_with_token(self):
        """Client should be enabled with valid token."""
        from bashgym.integrations.huggingface import HuggingFaceClient

        with patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value.whoami.return_value = {"name": "testuser"}
            client = HuggingFaceClient(token="hf_test_token")
            assert client.enabled is True

    def test_pro_detection(self):
        """Client should detect Pro subscription."""
        from bashgym.integrations.huggingface import HuggingFaceClient

        with patch("huggingface_hub.HfApi") as mock_api:
            mock_api.return_value.whoami.return_value = {
                "name": "testuser",
                "isPro": True
            }
            client = HuggingFaceClient(token="hf_test_token")
            assert client.pro_enabled is True
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_hf_integration.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'bashgym.integrations.huggingface'"

### Step 3: Add HuggingFaceSettings to config.py

Add after line 300 in `bashgym/config.py`:

```python
@dataclass
class HuggingFaceSettings:
    """HuggingFace integration settings."""

    # Authentication
    token: str = field(default_factory=lambda: get_env("HF_TOKEN"))
    username: str = field(default_factory=lambda: get_env("HF_USERNAME"))
    default_org: Optional[str] = field(default_factory=lambda: get_env("HF_ORG") or None)

    # Feature flags (auto-detected from token)
    pro_enabled: bool = False

    # Default repos
    storage_repo: str = field(default_factory=lambda: get_env("HF_STORAGE_REPO"))
    models_repo: str = field(default_factory=lambda: get_env("HF_MODELS_REPO"))

    # Inference settings
    inference_provider: str = field(default_factory=lambda: get_env("HF_INFERENCE_PROVIDER", "auto"))
    inference_routing: str = field(default_factory=lambda: get_env("HF_INFERENCE_ROUTING", "fastest"))

    # Jobs settings
    default_hardware: str = field(default_factory=lambda: get_env("HF_DEFAULT_HARDWARE", "a10g-small"))
    job_timeout_minutes: int = field(default_factory=lambda: get_env_int("HF_JOB_TIMEOUT", 30))
```

Update the `Settings` dataclass to include `huggingface`:

```python
@dataclass
class Settings:
    # ... existing fields ...
    huggingface: HuggingFaceSettings = field(default_factory=HuggingFaceSettings)
```

### Step 4: Create the HuggingFace client module

```python
# bashgym/integrations/huggingface/__init__.py
"""
HuggingFace Integration for Bash Gym

Provides access to HuggingFace Pro features:
- Jobs (cloud training)
- Inference Providers
- ZeroGPU Spaces
- Dataset Storage
"""

from .client import (
    HuggingFaceClient,
    HFError,
    HFAuthError,
    HFProRequiredError,
    HFQuotaExceededError,
)

__all__ = [
    "HuggingFaceClient",
    "HFError",
    "HFAuthError",
    "HFProRequiredError",
    "HFQuotaExceededError",
]
```

```python
# bashgym/integrations/huggingface/client.py
"""
Core HuggingFace client for Bash Gym.

Handles authentication, Pro detection, and provides access to HF features.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Check if huggingface_hub is available
try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HfApi = None


class HFError(Exception):
    """Base exception for HuggingFace errors."""
    pass


class HFAuthError(HFError):
    """Authentication error (invalid/expired token)."""
    pass


class HFProRequiredError(HFError):
    """Feature requires Pro subscription."""
    pass


class HFQuotaExceededError(HFError):
    """Credits or storage quota exceeded."""
    pass


class HFJobFailedError(HFError):
    """Training job failed."""
    pass


@dataclass
class HFUserInfo:
    """HuggingFace user information."""
    username: str
    is_pro: bool
    orgs: list[str]


class HuggingFaceClient:
    """
    Central client for HuggingFace integration.

    Handles authentication, Pro detection, and provides access to HF features.
    All features gracefully degrade when HF is not available or Pro is not enabled.
    """

    def __init__(
        self,
        token: str = "",
        default_org: Optional[str] = None,
    ):
        self.token = token
        self.default_org = default_org
        self._api: Optional[HfApi] = None
        self._user_info: Optional[HFUserInfo] = None

        # Feature flags
        self.enabled = False
        self.pro_enabled = False

        if token and HF_HUB_AVAILABLE:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize the HF API and detect features."""
        try:
            self._api = HfApi(token=self.token)
            user_data = self._api.whoami()

            self._user_info = HFUserInfo(
                username=user_data.get("name", ""),
                is_pro=user_data.get("isPro", False),
                orgs=[org["name"] for org in user_data.get("orgs", [])],
            )

            self.enabled = True
            self.pro_enabled = self._user_info.is_pro

            logger.info(
                f"HuggingFace initialized: user={self._user_info.username}, "
                f"pro={self.pro_enabled}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace: {e}")
            self.enabled = False
            self.pro_enabled = False

    @property
    def api(self) -> HfApi:
        """Get the HF API instance."""
        if not self._api:
            raise HFAuthError("HuggingFace not initialized. Set HF_TOKEN.")
        return self._api

    @property
    def username(self) -> str:
        """Get the authenticated username."""
        if not self._user_info:
            return ""
        return self._user_info.username

    def require_enabled(self) -> None:
        """Raise if HF is not enabled."""
        if not self.enabled:
            raise HFAuthError(
                "HuggingFace integration not enabled. "
                "Set HF_TOKEN environment variable."
            )

    def require_pro(self, feature: str) -> None:
        """Raise if Pro is not enabled."""
        self.require_enabled()
        if not self.pro_enabled:
            raise HFProRequiredError(
                f"{feature} requires HuggingFace Pro subscription. "
                f"Subscribe at https://huggingface.co/subscribe/pro"
            )

    def get_bill_to(self) -> Optional[str]:
        """Get the org to bill to, if any."""
        return self.default_org


# Global client instance
_hf_client: Optional[HuggingFaceClient] = None


def get_hf_client() -> HuggingFaceClient:
    """Get or create the global HF client."""
    global _hf_client
    if _hf_client is None:
        from bashgym.config import get_settings
        settings = get_settings()
        _hf_client = HuggingFaceClient(
            token=settings.huggingface.token,
            default_org=settings.huggingface.default_org,
        )
    return _hf_client


def reset_hf_client() -> None:
    """Reset the global HF client (for testing)."""
    global _hf_client
    _hf_client = None
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/test_hf_integration.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add bashgym/config.py bashgym/integrations/huggingface/ tests/test_hf_integration.py
git commit -m "feat(hf): add core HuggingFace client and settings"
```

---

## Task 2: HuggingFace Jobs (Cloud Training)

**Files:**
- Create: `bashgym/integrations/huggingface/jobs.py`
- Modify: `bashgym/gym/trainer.py`
- Test: `tests/test_hf_integration.py`

### Step 1: Write the failing test

Add to `tests/test_hf_integration.py`:

```python
class TestHFJobs:
    """Test HuggingFace Jobs integration."""

    def test_job_config_defaults(self):
        """Job config should have sensible defaults."""
        from bashgym.integrations.huggingface.jobs import HFJobConfig

        config = HFJobConfig()
        assert config.hardware == "a10g-small"
        assert config.timeout_minutes == 30
        assert config.docker_image == "huggingface/transformers-pytorch-gpu"

    def test_submit_job_requires_pro(self):
        """Submitting jobs should require Pro."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner
        from bashgym.integrations.huggingface import HFProRequiredError

        runner = HFJobRunner(token="", pro_enabled=False)

        with pytest.raises(HFProRequiredError):
            runner.submit_training_job(
                script_content="print('hello')",
                dataset_repo="user/dataset",
                output_repo="user/model",
            )

    @patch("huggingface_hub.HfApi")
    def test_submit_job_success(self, mock_api):
        """Should submit job successfully with Pro."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobConfig

        mock_api.return_value.run_job.return_value = MagicMock(job_id="job-123")

        runner = HFJobRunner(token="hf_test", pro_enabled=True)
        runner._api = mock_api.return_value

        job_id = runner.submit_training_job(
            script_content="print('hello')",
            dataset_repo="user/dataset",
            output_repo="user/model",
            config=HFJobConfig(hardware="t4-small"),
        )

        assert job_id == "job-123"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_hf_integration.py::TestHFJobs -v`
Expected: FAIL with "ModuleNotFoundError"

### Step 3: Implement HFJobRunner

```python
# bashgym/integrations/huggingface/jobs.py
"""
HuggingFace Jobs integration for cloud training.

Submits training jobs to HuggingFace's serverless GPU infrastructure.
Requires HuggingFace Pro subscription.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HfApi = None

from .client import HFProRequiredError, HFJobFailedError


class JobStatus(Enum):
    """Status of a HuggingFace Job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class HFJobConfig:
    """Configuration for HuggingFace Jobs."""

    # Hardware selection
    hardware: str = "a10g-small"  # t4-small, a10g-small, a10g-large, a100-large

    # Timeout
    timeout_minutes: int = 30

    # Docker image
    docker_image: str = "huggingface/transformers-pytorch-gpu"

    # Environment variables (will include HF_TOKEN automatically)
    environment: Dict[str, str] = field(default_factory=dict)

    # Secrets (stored securely)
    secrets: Dict[str, str] = field(default_factory=dict)


@dataclass
class HFJobInfo:
    """Information about a submitted job."""
    job_id: str
    status: JobStatus
    hardware: str
    created_at: str
    logs_url: Optional[str] = None
    error_message: Optional[str] = None


class HFJobRunner:
    """
    Runs training jobs on HuggingFace's cloud infrastructure.

    Requires HuggingFace Pro subscription.
    """

    def __init__(
        self,
        token: str = "",
        pro_enabled: bool = False,
        bill_to: Optional[str] = None,
    ):
        self.token = token
        self.pro_enabled = pro_enabled
        self.bill_to = bill_to
        self._api: Optional[HfApi] = None

        if token and HF_HUB_AVAILABLE:
            self._api = HfApi(token=token)

    def _require_pro(self) -> None:
        """Raise if Pro is not enabled."""
        if not self.pro_enabled:
            raise HFProRequiredError(
                "HuggingFace Jobs requires Pro subscription. "
                "Subscribe at https://huggingface.co/subscribe/pro"
            )

    def submit_training_job(
        self,
        script_content: str,
        dataset_repo: str,
        output_repo: str,
        config: Optional[HFJobConfig] = None,
    ) -> str:
        """
        Submit a training job to HuggingFace.

        Args:
            script_content: Python script to run (will be saved as train.py)
            dataset_repo: HF repo containing training data
            output_repo: HF repo to push trained model
            config: Job configuration

        Returns:
            Job ID for tracking
        """
        self._require_pro()

        if config is None:
            config = HFJobConfig()

        # Build environment
        env = {
            "HF_TOKEN": self.token,
            "DATASET_REPO": dataset_repo,
            "OUTPUT_REPO": output_repo,
            **config.environment,
        }

        # Submit job via HF API
        try:
            job = self._api.run_job(
                command=["python", "train.py"],
                hardware=config.hardware,
                timeout=config.timeout_minutes * 60,
                image=config.docker_image,
                env=env,
                secrets=config.secrets,
                files={"train.py": script_content},
            )

            logger.info(f"Submitted HF job: {job.job_id}")
            return job.job_id

        except Exception as e:
            logger.error(f"Failed to submit HF job: {e}")
            raise HFJobFailedError(f"Failed to submit job: {e}")

    def get_job_status(self, job_id: str) -> HFJobInfo:
        """Get the status of a job."""
        self._require_pro()

        try:
            job = self._api.get_job(job_id)
            return HFJobInfo(
                job_id=job.job_id,
                status=JobStatus(job.status),
                hardware=job.hardware,
                created_at=job.created_at,
                logs_url=job.logs_url,
                error_message=job.error if hasattr(job, "error") else None,
            )
        except Exception as e:
            raise HFJobFailedError(f"Failed to get job status: {e}")

    def get_job_logs(self, job_id: str) -> str:
        """Get logs from a job."""
        self._require_pro()

        try:
            return self._api.get_job_logs(job_id)
        except Exception as e:
            raise HFJobFailedError(f"Failed to get job logs: {e}")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        self._require_pro()

        try:
            self._api.cancel_job(job_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
```

### Step 4: Update __init__.py

```python
# bashgym/integrations/huggingface/__init__.py
from .client import (
    HuggingFaceClient,
    HFError,
    HFAuthError,
    HFProRequiredError,
    HFQuotaExceededError,
    get_hf_client,
)
from .jobs import (
    HFJobRunner,
    HFJobConfig,
    HFJobInfo,
    JobStatus,
    HFJobFailedError,
)

__all__ = [
    "HuggingFaceClient",
    "HFError",
    "HFAuthError",
    "HFProRequiredError",
    "HFQuotaExceededError",
    "HFJobFailedError",
    "get_hf_client",
    "HFJobRunner",
    "HFJobConfig",
    "HFJobInfo",
    "JobStatus",
]
```

### Step 5: Run tests

Run: `pytest tests/test_hf_integration.py::TestHFJobs -v`
Expected: PASS

### Step 6: Commit

```bash
git add bashgym/integrations/huggingface/jobs.py bashgym/integrations/huggingface/__init__.py tests/test_hf_integration.py
git commit -m "feat(hf): add Jobs runner for cloud training"
```

---

## Task 3: HuggingFace Inference Providers

**Files:**
- Create: `bashgym/integrations/huggingface/inference.py`
- Modify: `bashgym/gym/router.py`
- Test: `tests/test_hf_integration.py`

### Step 1: Write the failing test

Add to `tests/test_hf_integration.py`:

```python
class TestHFInference:
    """Test HuggingFace Inference Providers integration."""

    def test_inference_config_defaults(self):
        """Inference config should have sensible defaults."""
        from bashgym.integrations.huggingface.inference import HFInferenceConfig

        config = HFInferenceConfig()
        assert config.provider == "auto"
        assert config.routing == "fastest"
        assert config.timeout == 30

    def test_generate_text(self):
        """Should generate text via inference API."""
        from bashgym.integrations.huggingface.inference import HFInferenceClient

        with patch("huggingface_hub.InferenceClient") as mock_client:
            mock_client.return_value.text_generation.return_value = "Hello world"

            client = HFInferenceClient(token="hf_test")
            result = client.generate(
                model="meta-llama/Llama-3.3-70B-Instruct",
                prompt="Say hello",
            )

            assert result == "Hello world"

    def test_embed_texts(self):
        """Should generate embeddings."""
        from bashgym.integrations.huggingface.inference import HFInferenceClient

        with patch("huggingface_hub.InferenceClient") as mock_client:
            mock_client.return_value.feature_extraction.return_value = [[0.1, 0.2, 0.3]]

            client = HFInferenceClient(token="hf_test")
            result = client.embed(
                model="BAAI/bge-large-en-v1.5",
                texts=["Hello world"],
            )

            assert len(result) == 1
            assert len(result[0]) == 3
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_hf_integration.py::TestHFInference -v`
Expected: FAIL

### Step 3: Implement HFInferenceClient

```python
# bashgym/integrations/huggingface/inference.py
"""
HuggingFace Inference Providers integration.

Provides unified access to multiple inference providers through HF's API.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False
    InferenceClient = None

from .client import HFError, HFQuotaExceededError


@dataclass
class HFInferenceConfig:
    """Configuration for HF Inference Providers."""

    # Provider selection
    provider: str = "auto"  # auto, together, replicate, sambanova, fal
    routing: str = "fastest"  # fastest, cheapest

    # Billing
    bill_to: Optional[str] = None

    # Request settings
    timeout: int = 30
    max_retries: int = 3


@dataclass
class InferenceUsage:
    """Track inference usage for cost monitoring."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class HFInferenceClient:
    """
    Client for HuggingFace Inference Providers.

    Provides a unified API for text generation, embeddings, and classification
    across multiple inference providers.
    """

    def __init__(
        self,
        token: str = "",
        config: Optional[HFInferenceConfig] = None,
    ):
        self.token = token
        self.config = config or HFInferenceConfig()
        self._client: Optional[InferenceClient] = None
        self._usage: List[InferenceUsage] = []

        if token and HF_INFERENCE_AVAILABLE:
            self._client = InferenceClient(
                token=token,
                timeout=self.config.timeout,
            )

    def _get_model_with_routing(self, model: str) -> str:
        """Apply routing strategy to model ID."""
        if self.config.routing == "fastest":
            return f"{model}:fastest"
        elif self.config.routing == "cheapest":
            return f"{model}:cheapest"
        return model

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate text using the specified model.

        Args:
            model: Model ID (e.g., "meta-llama/Llama-3.3-70B-Instruct")
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if not self._client:
            raise HFError("HF Inference not available. Set HF_TOKEN.")

        model_with_routing = self._get_model_with_routing(model)

        try:
            result = self._client.text_generation(
                prompt,
                model=model_with_routing,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return result

        except Exception as e:
            if "quota" in str(e).lower():
                raise HFQuotaExceededError(f"Inference quota exceeded: {e}")
            raise HFError(f"Inference failed: {e}")

    def embed(
        self,
        model: str,
        texts: List[str],
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            model: Embedding model ID (e.g., "BAAI/bge-large-en-v1.5")
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._client:
            raise HFError("HF Inference not available. Set HF_TOKEN.")

        try:
            result = self._client.feature_extraction(
                texts,
                model=model,
                **kwargs,
            )
            return result

        except Exception as e:
            if "quota" in str(e).lower():
                raise HFQuotaExceededError(f"Inference quota exceeded: {e}")
            raise HFError(f"Embedding failed: {e}")

    def classify(
        self,
        model: str,
        text: str,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Classify text using the specified model.

        Args:
            model: Classification model ID
            text: Text to classify

        Returns:
            Dict mapping labels to scores
        """
        if not self._client:
            raise HFError("HF Inference not available. Set HF_TOKEN.")

        try:
            result = self._client.text_classification(
                text,
                model=model,
                **kwargs,
            )
            return {item["label"]: item["score"] for item in result}

        except Exception as e:
            raise HFError(f"Classification failed: {e}")

    def get_usage(self) -> List[InferenceUsage]:
        """Get tracked usage for this session."""
        return self._usage.copy()
```

### Step 4: Update __init__.py

Add to `bashgym/integrations/huggingface/__init__.py`:

```python
from .inference import (
    HFInferenceClient,
    HFInferenceConfig,
    InferenceUsage,
)

# Add to __all__
__all__ = [
    # ... existing ...
    "HFInferenceClient",
    "HFInferenceConfig",
    "InferenceUsage",
]
```

### Step 5: Run tests

Run: `pytest tests/test_hf_integration.py::TestHFInference -v`
Expected: PASS

### Step 6: Commit

```bash
git add bashgym/integrations/huggingface/inference.py bashgym/integrations/huggingface/__init__.py tests/test_hf_integration.py
git commit -m "feat(hf): add Inference Providers client"
```

---

## Task 4: HuggingFace Spaces (ZeroGPU)

**Files:**
- Create: `bashgym/integrations/huggingface/spaces.py`
- Test: `tests/test_hf_integration.py`

### Step 1: Write the failing test

Add to `tests/test_hf_integration.py`:

```python
class TestHFSpaces:
    """Test HuggingFace Spaces integration."""

    def test_space_config_defaults(self):
        """Space config should have sensible defaults."""
        from bashgym.integrations.huggingface.spaces import SpaceConfig

        config = SpaceConfig(name="test-space")
        assert config.hardware == "zero-gpu"
        assert config.private is True
        assert config.sdk == "gradio"

    def test_create_space_requires_pro(self):
        """Creating ZeroGPU space should require Pro."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager
        from bashgym.integrations.huggingface import HFProRequiredError

        manager = HFSpaceManager(token="", pro_enabled=False)

        with pytest.raises(HFProRequiredError):
            manager.create_inference_space(
                model_repo="user/model",
                space_name="test-space",
            )

    @patch("huggingface_hub.HfApi")
    def test_create_space_success(self, mock_api):
        """Should create space successfully with Pro."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceConfig

        mock_api.return_value.create_repo.return_value = MagicMock(
            repo_id="user/test-space"
        )

        manager = HFSpaceManager(token="hf_test", pro_enabled=True, username="user")
        manager._api = mock_api.return_value

        url = manager.create_inference_space(
            model_repo="user/model",
            space_name="test-space",
        )

        assert "test-space" in url
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_hf_integration.py::TestHFSpaces -v`
Expected: FAIL

### Step 3: Implement HFSpaceManager

```python
# bashgym/integrations/huggingface/spaces.py
"""
HuggingFace Spaces integration for model deployment.

Deploys trained models to ZeroGPU Spaces for inference.
Requires HuggingFace Pro subscription for ZeroGPU.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, SpaceHardware
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HfApi = None
    SpaceHardware = None

from .client import HFProRequiredError, HFError


class SpaceStatus(Enum):
    """Status of a HuggingFace Space."""
    BUILDING = "building"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SpaceConfig:
    """Configuration for HuggingFace Spaces."""
    name: str
    hardware: str = "zero-gpu"  # zero-gpu, cpu-basic, cpu-upgrade, t4-small, etc.
    private: bool = True
    sdk: str = "gradio"
    python_version: str = "3.10"
    dev_mode: bool = False


@dataclass
class SSHCredentials:
    """SSH credentials for dev mode."""
    host: str
    port: int
    username: str
    key: str


# Gradio app template for inference
GRADIO_APP_TEMPLATE = '''"""
Auto-generated Gradio app for model inference.
Generated by Bash Gym.
"""

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import spaces

MODEL_ID = "{model_repo}"

print(f"Loading model: {{MODEL_ID}}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
print("Model loaded!")


@spaces.GPU(duration={gpu_duration})
def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=5),
        gr.Slider(64, 2048, value=512, label="Max Tokens"),
        gr.Slider(0, 1.5, value=0.7, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Bash Gym Model Inference",
    description="Generated by Bash Gym - Self-Improving Agentic Development Gym",
)

if __name__ == "__main__":
    demo.launch()
'''


class HFSpaceManager:
    """
    Manages HuggingFace Spaces for model deployment.

    Creates ZeroGPU Spaces with auto-generated Gradio apps for inference.
    Requires HuggingFace Pro subscription.
    """

    def __init__(
        self,
        token: str = "",
        pro_enabled: bool = False,
        username: str = "",
    ):
        self.token = token
        self.pro_enabled = pro_enabled
        self.username = username
        self._api: Optional[HfApi] = None

        if token and HF_HUB_AVAILABLE:
            self._api = HfApi(token=token)

    def _require_pro(self) -> None:
        """Raise if Pro is not enabled."""
        if not self.pro_enabled:
            raise HFProRequiredError(
                "ZeroGPU Spaces require HuggingFace Pro subscription. "
                "Subscribe at https://huggingface.co/subscribe/pro"
            )

    def create_inference_space(
        self,
        model_repo: str,
        space_name: str,
        config: Optional[SpaceConfig] = None,
        gpu_duration: int = 60,
    ) -> str:
        """
        Create a ZeroGPU Space for model inference.

        Args:
            model_repo: HF repo containing the model
            space_name: Name for the Space
            config: Space configuration
            gpu_duration: GPU allocation duration in seconds

        Returns:
            Space URL
        """
        self._require_pro()

        if config is None:
            config = SpaceConfig(name=space_name)

        repo_id = f"{self.username}/{space_name}"

        try:
            # Create the Space repo
            self._api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk=config.sdk,
                space_hardware=config.hardware,
                private=config.private,
            )

            # Generate and upload app.py
            app_code = GRADIO_APP_TEMPLATE.format(
                model_repo=model_repo,
                gpu_duration=gpu_duration,
            )

            self._api.upload_file(
                path_or_fileobj=app_code.encode(),
                path_in_repo="app.py",
                repo_id=repo_id,
                repo_type="space",
            )

            # Upload requirements.txt
            requirements = "transformers\ntorch\naccelerate\nspaces\n"
            self._api.upload_file(
                path_or_fileobj=requirements.encode(),
                path_in_repo="requirements.txt",
                repo_id=repo_id,
                repo_type="space",
            )

            url = f"https://huggingface.co/spaces/{repo_id}"
            logger.info(f"Created Space: {url}")
            return url

        except Exception as e:
            raise HFError(f"Failed to create Space: {e}")

    def get_space_status(self, space_name: str) -> SpaceStatus:
        """Get the status of a Space."""
        repo_id = f"{self.username}/{space_name}"

        try:
            info = self._api.space_info(repo_id)
            status_str = info.runtime.get("stage", "unknown")

            status_map = {
                "BUILDING": SpaceStatus.BUILDING,
                "RUNNING": SpaceStatus.RUNNING,
                "STOPPED": SpaceStatus.STOPPED,
                "ERROR": SpaceStatus.ERROR,
            }
            return status_map.get(status_str.upper(), SpaceStatus.ERROR)

        except Exception as e:
            raise HFError(f"Failed to get Space status: {e}")

    def delete_space(self, space_name: str) -> bool:
        """Delete a Space."""
        repo_id = f"{self.username}/{space_name}"

        try:
            self._api.delete_repo(repo_id=repo_id, repo_type="space")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Space: {e}")
            return False

    def update_space_model(self, space_name: str, new_model_repo: str) -> bool:
        """Update a Space to use a different model."""
        repo_id = f"{self.username}/{space_name}"

        try:
            app_code = GRADIO_APP_TEMPLATE.format(
                model_repo=new_model_repo,
                gpu_duration=60,
            )

            self._api.upload_file(
                path_or_fileobj=app_code.encode(),
                path_in_repo="app.py",
                repo_id=repo_id,
                repo_type="space",
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update Space: {e}")
            return False
```

### Step 4: Update __init__.py

Add to `bashgym/integrations/huggingface/__init__.py`:

```python
from .spaces import (
    HFSpaceManager,
    SpaceConfig,
    SpaceStatus,
    SSHCredentials,
)

# Add to __all__
```

### Step 5: Run tests

Run: `pytest tests/test_hf_integration.py::TestHFSpaces -v`
Expected: PASS

### Step 6: Commit

```bash
git add bashgym/integrations/huggingface/spaces.py bashgym/integrations/huggingface/__init__.py tests/test_hf_integration.py
git commit -m "feat(hf): add Spaces manager for ZeroGPU deployment"
```

---

## Task 5: HuggingFace Datasets

**Files:**
- Create: `bashgym/integrations/huggingface/datasets.py`
- Test: `tests/test_hf_integration.py`

### Step 1: Write the failing test

Add to `tests/test_hf_integration.py`:

```python
class TestHFDatasets:
    """Test HuggingFace Datasets integration."""

    def test_dataset_config_defaults(self):
        """Dataset config should have sensible defaults."""
        from bashgym.integrations.huggingface.datasets import DatasetConfig

        config = DatasetConfig(repo_name="user/dataset")
        assert config.private is True
        assert config.enable_viewer is True

    @patch("huggingface_hub.HfApi")
    def test_upload_training_data(self, mock_api):
        """Should upload training data to HF."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            train_file = Path(tmpdir) / "train.jsonl"
            train_file.write_text('{"messages": []}\n')

            manager = HFDatasetManager(token="hf_test", username="user")
            manager._api = mock_api.return_value

            url = manager.upload_training_data(
                local_path=Path(tmpdir),
                repo_name="test-dataset",
            )

            assert "test-dataset" in url
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_hf_integration.py::TestHFDatasets -v`
Expected: FAIL

### Step 3: Implement HFDatasetManager

```python
# bashgym/integrations/huggingface/datasets.py
"""
HuggingFace Datasets integration for training data storage.

Uploads and manages training datasets on HuggingFace Hub.
Pro users get 1TB storage and Data Studio access on private datasets.
"""

import logging
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HfApi = None

from .client import HFError


@dataclass
class DatasetConfig:
    """Configuration for HuggingFace Dataset upload."""
    repo_name: str
    private: bool = True
    enable_viewer: bool = True  # Data Studio (Pro feature on private)


DATASET_CARD_TEMPLATE = '''---
license: mit
task_categories:
  - text-generation
tags:
  - bashgym
  - code-generation
  - agent-training
---

# {title}

Generated by Bash Gym on {date}.

## Statistics

- Training examples: {train_count}
- Validation examples: {val_count}
- Source repos: {repos}

## Format

NeMo-compatible JSONL with messages array:

```json
{{"messages": [{{"role": "system", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```

---

Generated by [Bash Gym](https://github.com/yourusername/bashgym) - Self-Improving Agentic Development Gym
'''


class HFDatasetManager:
    """
    Manages training datasets on HuggingFace Hub.

    Uploads JSONL training data with auto-generated dataset cards.
    Pro users get 1TB storage and Data Studio access.
    """

    def __init__(
        self,
        token: str = "",
        username: str = "",
    ):
        self.token = token
        self.username = username
        self._api: Optional[HfApi] = None

        if token and HF_HUB_AVAILABLE:
            self._api = HfApi(token=token)

    def upload_training_data(
        self,
        local_path: Path,
        repo_name: str,
        config: Optional[DatasetConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload training data to HuggingFace Hub.

        Args:
            local_path: Directory containing train.jsonl and optionally val.jsonl
            repo_name: Name for the dataset repo
            config: Dataset configuration
            metadata: Additional metadata to include

        Returns:
            Dataset repo URL
        """
        if config is None:
            config = DatasetConfig(repo_name=repo_name)

        repo_id = f"{self.username}/{repo_name}"

        try:
            # Create the dataset repo
            self._api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=config.private,
                exist_ok=True,
            )

            # Count examples
            train_count = 0
            val_count = 0

            train_file = local_path / "train.jsonl"
            if train_file.exists():
                train_count = sum(1 for _ in train_file.open())
                self._api.upload_file(
                    path_or_fileobj=str(train_file),
                    path_in_repo="train.jsonl",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            val_file = local_path / "val.jsonl"
            if val_file.exists():
                val_count = sum(1 for _ in val_file.open())
                self._api.upload_file(
                    path_or_fileobj=str(val_file),
                    path_in_repo="val.jsonl",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            # Generate and upload README
            readme = DATASET_CARD_TEMPLATE.format(
                title=f"Bash Gym Training Dataset: {repo_name}",
                date=datetime.now().strftime("%Y-%m-%d"),
                train_count=train_count,
                val_count=val_count,
                repos=metadata.get("repos", "N/A") if metadata else "N/A",
                repo_id=repo_id,
            )

            self._api.upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )

            # Upload metadata if provided
            if metadata:
                self._api.upload_file(
                    path_or_fileobj=json.dumps(metadata, indent=2).encode(),
                    path_in_repo="metadata.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

            url = f"https://huggingface.co/datasets/{repo_id}"
            logger.info(f"Uploaded dataset: {url}")
            return url

        except Exception as e:
            raise HFError(f"Failed to upload dataset: {e}")

    def download_dataset(self, repo_id: str, local_path: Path) -> None:
        """Download a dataset from HuggingFace Hub."""
        try:
            self._api.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_path),
            )
        except Exception as e:
            raise HFError(f"Failed to download dataset: {e}")

    def list_datasets(self, prefix: str = "bashgym") -> List[str]:
        """List datasets with the given prefix."""
        try:
            datasets = self._api.list_datasets(
                author=self.username,
                search=prefix,
            )
            return [d.id for d in datasets]
        except Exception as e:
            raise HFError(f"Failed to list datasets: {e}")

    def delete_dataset(self, repo_name: str) -> bool:
        """Delete a dataset."""
        repo_id = f"{self.username}/{repo_name}"
        try:
            self._api.delete_repo(repo_id=repo_id, repo_type="dataset")
            return True
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            return False
```

### Step 4: Update __init__.py

Add to `bashgym/integrations/huggingface/__init__.py`:

```python
from .datasets import (
    HFDatasetManager,
    DatasetConfig,
)

# Add to __all__
```

### Step 5: Run tests

Run: `pytest tests/test_hf_integration.py::TestHFDatasets -v`
Expected: PASS

### Step 6: Commit

```bash
git add bashgym/integrations/huggingface/datasets.py bashgym/integrations/huggingface/__init__.py tests/test_hf_integration.py
git commit -m "feat(hf): add Dataset manager for training data storage"
```

---

## Task 6: API Routes for HuggingFace

**Files:**
- Create: `bashgym/api/hf_routes.py`
- Modify: `bashgym/api/routes.py`
- Test: `tests/test_hf_integration.py`

### Step 1: Write the failing test

Add to `tests/test_hf_integration.py`:

```python
from fastapi.testclient import TestClient

class TestHFAPIRoutes:
    """Test HuggingFace API routes."""

    def test_hf_status_endpoint(self):
        """Should return HF status."""
        from bashgym.api.routes import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/api/hf/status")
        assert response.status_code == 200

        data = response.json()
        assert "enabled" in data
        assert "pro_enabled" in data

    def test_hf_jobs_list(self):
        """Should list HF jobs."""
        from bashgym.api.routes import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/api/hf/jobs")
        assert response.status_code in [200, 403]  # 403 if Pro required
```

### Step 2: Implement HF routes

```python
# bashgym/api/hf_routes.py
"""
HuggingFace API routes for Bash Gym.

Provides REST endpoints for HF features:
- Status and Pro detection
- Cloud training jobs
- Inference API
- Spaces management
- Dataset management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/api/hf", tags=["huggingface"])


# === Schemas ===

class HFStatus(BaseModel):
    """HuggingFace integration status."""
    enabled: bool
    pro_enabled: bool
    username: str = ""
    credits_remaining: Optional[float] = None


class JobSubmitRequest(BaseModel):
    """Request to submit a training job."""
    dataset_repo: str
    output_repo: str
    hardware: str = "a10g-small"
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    num_epochs: int = 3
    learning_rate: float = 2e-5


class JobResponse(BaseModel):
    """Response for job operations."""
    job_id: str
    status: str
    hardware: str
    created_at: str
    logs_url: Optional[str] = None


class SpaceCreateRequest(BaseModel):
    """Request to create a Space."""
    model_repo: str
    space_name: str
    private: bool = True
    gpu_duration: int = 60


class SpaceResponse(BaseModel):
    """Response for Space operations."""
    url: str
    status: str


class DatasetUploadRequest(BaseModel):
    """Request to upload dataset."""
    repo_name: str
    private: bool = True


class InferenceRequest(BaseModel):
    """Request for inference."""
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7


# === Routes ===

@router.get("/status", response_model=HFStatus)
async def get_hf_status():
    """Get HuggingFace integration status."""
    from bashgym.integrations.huggingface import get_hf_client

    client = get_hf_client()
    return HFStatus(
        enabled=client.enabled,
        pro_enabled=client.pro_enabled,
        username=client.username,
    )


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs():
    """List all HF training jobs."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError

    client = get_hf_client()
    try:
        client.require_pro("Jobs")
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # TODO: Implement job listing via HF API
    return []


@router.post("/jobs", response_model=JobResponse)
async def submit_job(request: JobSubmitRequest, background_tasks: BackgroundTasks):
    """Submit a new training job."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError
    from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobConfig

    client = get_hf_client()
    try:
        client.require_pro("Jobs")
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    runner = HFJobRunner(
        token=client.token,
        pro_enabled=client.pro_enabled,
    )

    # Generate training script
    from bashgym.gym.trainer import Trainer, TrainerConfig
    trainer = Trainer(TrainerConfig(
        base_model=request.base_model,
        num_epochs=request.num_epochs,
        learning_rate=request.learning_rate,
    ))
    script = trainer._generate_sft_script(
        str(request.dataset_repo),
        str(request.output_repo),
    )

    job_id = runner.submit_training_job(
        script_content=script,
        dataset_repo=request.dataset_repo,
        output_repo=request.output_repo,
        config=HFJobConfig(hardware=request.hardware),
    )

    return JobResponse(
        job_id=job_id,
        status="pending",
        hardware=request.hardware,
        created_at=datetime.now().isoformat(),
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status."""
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.integrations.huggingface.jobs import HFJobRunner

    client = get_hf_client()
    runner = HFJobRunner(token=client.token, pro_enabled=client.pro_enabled)

    info = runner.get_job_status(job_id)
    return JobResponse(
        job_id=info.job_id,
        status=info.status.value,
        hardware=info.hardware,
        created_at=info.created_at,
        logs_url=info.logs_url,
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.integrations.huggingface.jobs import HFJobRunner

    client = get_hf_client()
    runner = HFJobRunner(token=client.token, pro_enabled=client.pro_enabled)

    success = runner.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel job")
    return {"status": "cancelled"}


@router.post("/inference/generate")
async def generate_text(request: InferenceRequest):
    """Generate text via HF Inference Providers."""
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.integrations.huggingface.inference import HFInferenceClient

    client = get_hf_client()
    inference = HFInferenceClient(token=client.token)

    result = inference.generate(
        model=request.model,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {"text": result}


@router.get("/spaces", response_model=List[SpaceResponse])
async def list_spaces():
    """List all Bash Gym Spaces."""
    # TODO: Implement
    return []


@router.post("/spaces", response_model=SpaceResponse)
async def create_space(request: SpaceCreateRequest):
    """Create a new inference Space."""
    from bashgym.integrations.huggingface import get_hf_client, HFProRequiredError
    from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceConfig

    client = get_hf_client()
    try:
        client.require_pro("ZeroGPU Spaces")
    except HFProRequiredError as e:
        raise HTTPException(status_code=403, detail=str(e))

    manager = HFSpaceManager(
        token=client.token,
        pro_enabled=client.pro_enabled,
        username=client.username,
    )

    url = manager.create_inference_space(
        model_repo=request.model_repo,
        space_name=request.space_name,
        config=SpaceConfig(
            name=request.space_name,
            private=request.private,
        ),
        gpu_duration=request.gpu_duration,
    )

    return SpaceResponse(url=url, status="building")


@router.delete("/spaces/{space_name}")
async def delete_space(space_name: str):
    """Delete a Space."""
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.integrations.huggingface.spaces import HFSpaceManager

    client = get_hf_client()
    manager = HFSpaceManager(
        token=client.token,
        pro_enabled=client.pro_enabled,
        username=client.username,
    )

    success = manager.delete_space(space_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete Space")
    return {"status": "deleted"}


@router.get("/datasets")
async def list_datasets():
    """List Bash Gym datasets."""
    from bashgym.integrations.huggingface import get_hf_client
    from bashgym.integrations.huggingface.datasets import HFDatasetManager

    client = get_hf_client()
    if not client.enabled:
        return []

    manager = HFDatasetManager(token=client.token, username=client.username)
    return manager.list_datasets()
```

### Step 3: Add router to main app

Modify `bashgym/api/routes.py` to include HF routes:

```python
# Add import at top
from bashgym.api.hf_routes import router as hf_router

# In create_app(), add:
app.include_router(hf_router)
```

### Step 4: Run tests

Run: `pytest tests/test_hf_integration.py::TestHFAPIRoutes -v`
Expected: PASS

### Step 5: Commit

```bash
git add bashgym/api/hf_routes.py bashgym/api/routes.py tests/test_hf_integration.py
git commit -m "feat(hf): add API routes for HuggingFace features"
```

---

## Task 7: WebSocket Events for HF

**Files:**
- Modify: `bashgym/api/websocket.py`

### Step 1: Add HF message types

Add to `bashgym/api/websocket.py`:

```python
# Add to MessageType enum
class MessageType(str, Enum):
    # ... existing types ...
    HF_JOB_STARTED = "hf:job:started"
    HF_JOB_LOG = "hf:job:log"
    HF_JOB_COMPLETED = "hf:job:completed"
    HF_JOB_FAILED = "hf:job:failed"
    HF_SPACE_READY = "hf:space:ready"
    HF_SPACE_ERROR = "hf:space:error"


# Add broadcast functions
async def broadcast_hf_job_started(job_id: str, hardware: str):
    """Broadcast when HF job starts."""
    await manager.broadcast({
        "type": MessageType.HF_JOB_STARTED.value,
        "data": {"job_id": job_id, "hardware": hardware},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def broadcast_hf_job_log(job_id: str, log_line: str):
    """Broadcast HF job log line."""
    await manager.broadcast({
        "type": MessageType.HF_JOB_LOG.value,
        "data": {"job_id": job_id, "log": log_line},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def broadcast_hf_job_completed(job_id: str, model_repo: str):
    """Broadcast when HF job completes."""
    await manager.broadcast({
        "type": MessageType.HF_JOB_COMPLETED.value,
        "data": {"job_id": job_id, "model_repo": model_repo},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


async def broadcast_hf_space_ready(space_name: str, url: str):
    """Broadcast when Space is ready."""
    await manager.broadcast({
        "type": MessageType.HF_SPACE_READY.value,
        "data": {"space_name": space_name, "url": url},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
```

### Step 2: Commit

```bash
git add bashgym/api/websocket.py
git commit -m "feat(hf): add WebSocket events for HF jobs and spaces"
```

---

## Task 8: Update .env.example

**Files:**
- Modify: `.env.example` (or create if doesn't exist)

### Step 1: Add HF environment variables

Add to `.env.example`:

```bash
# =============================================
# HuggingFace Integration (Optional - Pro features)
# =============================================

# Authentication
HF_TOKEN=                         # Your HuggingFace token
HF_USERNAME=                      # Your HuggingFace username
HF_ORG=                           # Organization to bill (optional)

# Default repos
HF_STORAGE_REPO=                  # Default dataset storage repo
HF_MODELS_REPO=                   # Default models repo

# Inference settings
HF_INFERENCE_PROVIDER=auto        # auto, together, replicate, sambanova
HF_INFERENCE_ROUTING=fastest      # fastest, cheapest

# Jobs settings
HF_DEFAULT_HARDWARE=a10g-small    # t4-small, a10g-small, a10g-large, a100-large
HF_JOB_TIMEOUT=30                 # Job timeout in minutes
```

### Step 2: Commit

```bash
git add .env.example
git commit -m "docs: add HuggingFace environment variables to .env.example"
```

---

## Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | config.py, client.py | Core settings and HF client |
| 2 | jobs.py | Cloud training via HF Jobs |
| 3 | inference.py | Inference Providers API |
| 4 | spaces.py | ZeroGPU Space deployment |
| 5 | datasets.py | Dataset storage and upload |
| 6 | hf_routes.py | REST API endpoints |
| 7 | websocket.py | Real-time event broadcasting |
| 8 | .env.example | Configuration documentation |

**Total estimated tasks:** 8 core tasks with ~40 individual steps

**Dependencies:** `huggingface_hub` library (add to requirements.txt)
