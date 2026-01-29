#!/usr/bin/env python3
"""
Tests for HuggingFace Integration

Tests:
- HuggingFaceSettings defaults and environment loading
- HuggingFaceClient enabled/disabled states
- Pro subscription detection
- Error handling

Run with: pytest tests/test_hf_integration.py -v
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "bashgym"))


# =============================================================================
# Test: HuggingFaceSettings
# =============================================================================

class TestHuggingFaceSettings:
    """Tests for HuggingFaceSettings dataclass in config.py."""

    def test_settings_defaults(self):
        """Test that HuggingFaceSettings has sensible defaults when no env vars set."""
        from bashgym.config import HuggingFaceSettings

        # Remove HF-related env vars entirely to test true defaults
        hf_env_keys = [
            "HF_TOKEN", "HF_USERNAME", "HF_ORG", "HF_PRO_ENABLED",
            "HF_STORAGE_REPO", "HF_MODELS_REPO", "HF_INFERENCE_PROVIDER",
            "HF_INFERENCE_ROUTING", "HF_DEFAULT_HARDWARE", "HF_JOB_TIMEOUT_MINUTES",
        ]

        # Save and remove env vars
        saved_vars = {}
        for key in hf_env_keys:
            if key in os.environ:
                saved_vars[key] = os.environ.pop(key)

        try:
            # Mock get_secret to return None (no stored secrets)
            with patch("bashgym.secrets.get_secret", return_value=None):
                settings = HuggingFaceSettings()

                # Default values when env vars are not set at all
                assert settings.token == ""  # Empty when not set
                assert settings.username == ""
                assert settings.default_org == ""
                assert settings.pro_enabled == False
                assert settings.storage_repo == ""
                assert settings.models_repo == ""
                assert settings.inference_provider == "serverless"
                assert settings.inference_routing == "cheapest"
                assert settings.default_hardware == "t4-small"
                assert settings.job_timeout_minutes == 60
        finally:
            # Restore env vars
            for key, value in saved_vars.items():
                os.environ[key] = value

    def test_settings_enabled_property(self):
        """Test the enabled property based on token presence."""
        from bashgym.config import HuggingFaceSettings

        # Clear token for this test (mock stored secrets too)
        with patch("bashgym.secrets.get_secret", return_value=None):
            with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
                # No token = disabled
                settings = HuggingFaceSettings()
                assert settings.enabled == False

        # With token = enabled
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False):
            settings_with_token = HuggingFaceSettings()
            assert settings_with_token.enabled == True

    def test_settings_namespace_property(self):
        """Test namespace falls back correctly."""
        from bashgym.config import HuggingFaceSettings

        # No org or username = empty namespace
        settings = HuggingFaceSettings()
        assert settings.namespace == ""

        # With username only
        settings.username = "testuser"
        assert settings.namespace == "testuser"

        # Org takes precedence
        settings.default_org = "testorg"
        assert settings.namespace == "testorg"

    def test_settings_env_loading(self):
        """Test that settings load from environment variables."""
        from bashgym.config import HuggingFaceSettings

        env_vars = {
            "HF_TOKEN": "hf_test_token_123",
            "HF_USERNAME": "testuser",
            "HF_ORG": "testorg",
            "HF_PRO_ENABLED": "true",
            "HF_STORAGE_REPO": "testorg/storage",
            "HF_MODELS_REPO": "testorg/models",
            "HF_INFERENCE_PROVIDER": "dedicated",
            "HF_INFERENCE_ROUTING": "fastest",
            "HF_DEFAULT_HARDWARE": "a10g-large",
            "HF_JOB_TIMEOUT_MINUTES": "120",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = HuggingFaceSettings()

            assert settings.token == "hf_test_token_123"
            assert settings.username == "testuser"
            assert settings.default_org == "testorg"
            assert settings.pro_enabled == True
            assert settings.storage_repo == "testorg/storage"
            assert settings.models_repo == "testorg/models"
            assert settings.inference_provider == "dedicated"
            assert settings.inference_routing == "fastest"
            assert settings.default_hardware == "a10g-large"
            assert settings.job_timeout_minutes == 120

    def test_settings_in_main_settings(self):
        """Test that HuggingFaceSettings is accessible from main Settings."""
        from bashgym.config import Settings

        settings = Settings()
        assert hasattr(settings, "huggingface")
        assert settings.huggingface is not None
        assert hasattr(settings.huggingface, "token")
        assert hasattr(settings.huggingface, "default_hardware")

    def test_settings_to_dict_hides_token(self):
        """Test that to_dict hides the HF token."""
        from bashgym.config import Settings

        settings = Settings()
        settings.huggingface.token = "hf_secret_token_123"

        result = settings.to_dict()
        # Token should be masked
        assert result["huggingface"]["token"] == "***"

    def test_settings_validation(self):
        """Test HuggingFaceSettings validation."""
        from bashgym.config import HuggingFaceSettings

        settings = HuggingFaceSettings()
        errors = settings.validate()
        # Should return empty list (no required fields)
        assert isinstance(errors, list)


# =============================================================================
# Test: HuggingFace Enums
# =============================================================================

class TestHuggingFaceEnums:
    """Tests for HuggingFace-related enums."""

    def test_inference_provider_enum(self):
        """Test InferenceProvider enum values."""
        from bashgym.config import InferenceProvider

        assert InferenceProvider.HUGGINGFACE.value == "huggingface"
        assert InferenceProvider.SERVERLESS.value == "serverless"
        assert InferenceProvider.DEDICATED.value == "dedicated"

    def test_inference_routing_enum(self):
        """Test InferenceRouting enum values."""
        from bashgym.config import InferenceRouting

        assert InferenceRouting.CHEAPEST.value == "cheapest"
        assert InferenceRouting.FASTEST.value == "fastest"
        assert InferenceRouting.QUALITY.value == "quality"

    def test_hf_hardware_enum(self):
        """Test HFHardware enum has expected tiers."""
        from bashgym.config import HFHardware

        # Check some key hardware tiers
        assert HFHardware.CPU_BASIC.value == "cpu-basic"
        assert HFHardware.T4_SMALL.value == "t4-small"
        assert HFHardware.A10G_LARGE.value == "a10g-large"
        assert HFHardware.A100_LARGE.value == "a100-large"
        assert HFHardware.H100.value == "h100"


# =============================================================================
# Test: HuggingFaceClient
# =============================================================================

class TestHuggingFaceClient:
    """Tests for HuggingFaceClient class."""

    def test_client_without_library(self):
        """Test client behavior when huggingface_hub is not installed."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            HF_HUB_AVAILABLE,
            HFAuthError,
            reset_hf_client,
        )

        reset_hf_client()

        # Mock the library as unavailable
        with patch.dict("bashgym.integrations.huggingface.client.__dict__",
                       {"HF_HUB_AVAILABLE": False}):
            # Re-import to get patched version
            import importlib
            from bashgym.integrations.huggingface import client as hf_client_module
            original_available = hf_client_module.HF_HUB_AVAILABLE

            try:
                hf_client_module.HF_HUB_AVAILABLE = False
                client = HuggingFaceClient(token="hf_test")
                client._initialized = False  # Force re-init with patched value

                # Should not be enabled
                assert not client.is_enabled
            finally:
                hf_client_module.HF_HUB_AVAILABLE = original_available

    def test_client_without_token(self):
        """Test client behavior with no token."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            HFAuthError,
            reset_hf_client,
        )

        reset_hf_client()

        # Patch to prevent using cached token
        with patch("bashgym.integrations.huggingface.client.HfApi", None):
            client = HuggingFaceClient(token=None)
            # Force initialization without trying to get cached token
            client._initialized = True
            client._init_error = "No HuggingFace token provided"

            assert not client.is_enabled
            assert client.username is None

            # require_enabled should raise
            with pytest.raises(HFAuthError):
                client.require_enabled()

    def test_client_with_mock_api(self):
        """Test client with mocked HfApi."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            HFUserInfo,
            reset_hf_client,
        )

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.whoami.return_value = {
            "name": "testuser",
            "fullname": "Test User",
            "email": "test@example.com",
            "orgs": [{"name": "testorg"}],
            "isPro": False,
            "canPay": False,
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test_token")

                assert client.is_enabled
                assert client.username == "testuser"
                assert client.is_pro == False
                assert "testorg" in client.organizations

    def test_client_pro_detection(self):
        """Test Pro subscription detection from various response formats."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            HFUserInfo,
            reset_hf_client,
        )

        reset_hf_client()

        # Test isPro flag
        mock_api = MagicMock()
        mock_api.whoami.return_value = {
            "name": "prouser",
            "isPro": True,
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test_token")
                assert client.is_pro == True

        reset_hf_client()

        # Test is_pro flag (snake_case variant)
        mock_api.whoami.return_value = {
            "name": "prouser",
            "is_pro": True,
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test_token")
                assert client.is_pro == True

        reset_hf_client()

        # Test plan field
        mock_api.whoami.return_value = {
            "name": "prouser",
            "plan": "pro",
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test_token")
                assert client.is_pro == True

    def test_require_pro_raises(self):
        """Test require_pro raises when not Pro."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            HFProRequiredError,
            reset_hf_client,
        )

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.whoami.return_value = {
            "name": "freeuser",
            "isPro": False,
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test_token")

                assert client.is_enabled
                assert not client.is_pro

                with pytest.raises(HFProRequiredError) as exc_info:
                    client.require_pro()

                assert "Pro subscription" in str(exc_info.value)
                assert "freeuser" in str(exc_info.value)

    def test_require_pro_passes_for_pro_user(self):
        """Test require_pro passes for Pro users."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            reset_hf_client,
        )

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.whoami.return_value = {
            "name": "prouser",
            "isPro": True,
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test_token")

                # Should not raise
                client.require_pro()

    def test_client_namespace_resolution(self):
        """Test namespace resolution (org vs username)."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            reset_hf_client,
        )

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.whoami.return_value = {
            "name": "testuser",
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                # Without default_org, namespace is username
                client = HuggingFaceClient(token="hf_test_token")
                assert client.namespace == "testuser"

        reset_hf_client()

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                # With default_org, namespace is org
                client = HuggingFaceClient(token="hf_test_token", default_org="myorg")
                assert client.namespace == "myorg"
                assert client.default_org == "myorg"

    def test_client_get_repo_id(self):
        """Test repo ID construction."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            reset_hf_client,
        )

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.whoami.return_value = {
            "name": "testuser",
        }

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test_token")

                # Name without namespace gets namespace added
                assert client.get_repo_id("mymodel") == "testuser/mymodel"

                # Name with namespace is returned as-is
                assert client.get_repo_id("otheruser/mymodel") == "otheruser/mymodel"

                # Custom namespace override
                assert client.get_repo_id("mymodel", namespace="customorg") == "customorg/mymodel"

    def test_client_repr(self):
        """Test string representation of client."""
        from bashgym.integrations.huggingface.client import (
            HuggingFaceClient,
            reset_hf_client,
        )

        reset_hf_client()

        # Disabled client
        client = HuggingFaceClient(token=None)
        client._initialized = True
        client._init_error = "No token"
        repr_str = repr(client)
        assert "disabled" in repr_str

        reset_hf_client()

        # Enabled non-Pro client
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser", "isPro": False}

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test")
                repr_str = repr(client)
                assert "testuser" in repr_str
                assert "[Pro]" not in repr_str

        reset_hf_client()

        # Enabled Pro client
        mock_api.whoami.return_value = {"name": "prouser", "isPro": True}

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test")
                repr_str = repr(client)
                assert "prouser" in repr_str
                assert "[Pro]" in repr_str


# =============================================================================
# Test: Singleton Access
# =============================================================================

class TestSingletonAccess:
    """Tests for get_hf_client singleton function."""

    def test_get_hf_client_returns_singleton(self):
        """Test that get_hf_client returns the same instance."""
        from bashgym.integrations.huggingface.client import (
            get_hf_client,
            reset_hf_client,
        )

        reset_hf_client()

        # Mock to avoid actual API calls
        with patch("bashgym.integrations.huggingface.client.HfApi"):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client1 = get_hf_client(token="hf_test")
                client2 = get_hf_client()

                assert client1 is client2

    def test_get_hf_client_force_new(self):
        """Test that force_new creates a new instance."""
        from bashgym.integrations.huggingface.client import (
            get_hf_client,
            reset_hf_client,
        )

        reset_hf_client()

        with patch("bashgym.integrations.huggingface.client.HfApi"):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client1 = get_hf_client(token="hf_test1")
                client2 = get_hf_client(token="hf_test2", force_new=True)

                # Different instances due to force_new
                assert client1 is not client2

    def test_get_hf_client_with_new_token(self):
        """Test that providing a new token creates a new instance."""
        from bashgym.integrations.huggingface.client import (
            get_hf_client,
            reset_hf_client,
        )

        reset_hf_client()

        with patch("bashgym.integrations.huggingface.client.HfApi"):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client1 = get_hf_client(token="hf_test1")
                client2 = get_hf_client(token="hf_test2")

                # Different instances due to different token
                assert client1 is not client2

    def test_reset_hf_client(self):
        """Test that reset_hf_client clears the singleton."""
        from bashgym.integrations.huggingface.client import (
            get_hf_client,
            reset_hf_client,
        )

        reset_hf_client()

        with patch("bashgym.integrations.huggingface.client.HfApi"):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client1 = get_hf_client(token="hf_test")
                reset_hf_client()
                client2 = get_hf_client(token="hf_test")

                # Different instances after reset
                assert client1 is not client2


# =============================================================================
# Test: Error Classes
# =============================================================================

class TestHFErrors:
    """Tests for HuggingFace error classes."""

    def test_hf_error_base(self):
        """Test base HFError."""
        from bashgym.integrations.huggingface import HFError

        error = HFError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_hf_auth_error(self):
        """Test HFAuthError."""
        from bashgym.integrations.huggingface import HFAuthError, HFError

        error = HFAuthError("Invalid token")
        assert str(error) == "Invalid token"
        assert isinstance(error, HFError)

    def test_hf_pro_required_error(self):
        """Test HFProRequiredError."""
        from bashgym.integrations.huggingface import HFProRequiredError, HFError

        error = HFProRequiredError("Need Pro subscription")
        assert str(error) == "Need Pro subscription"
        assert isinstance(error, HFError)

    def test_hf_quota_exceeded_error(self):
        """Test HFQuotaExceededError."""
        from bashgym.integrations.huggingface import HFQuotaExceededError, HFError

        error = HFQuotaExceededError("Quota limit reached")
        assert str(error) == "Quota limit reached"
        assert isinstance(error, HFError)

    def test_hf_job_failed_error(self):
        """Test HFJobFailedError with job details."""
        from bashgym.integrations.huggingface import HFJobFailedError, HFError

        error = HFJobFailedError(
            "Training failed",
            job_id="job_123",
            logs="Error: OOM"
        )
        assert str(error) == "Training failed"
        assert error.job_id == "job_123"
        assert error.logs == "Error: OOM"
        assert isinstance(error, HFError)


# =============================================================================
# Test: HFUserInfo
# =============================================================================

class TestHFUserInfo:
    """Tests for HFUserInfo dataclass."""

    def test_user_info_from_whoami(self):
        """Test creating HFUserInfo from whoami response."""
        from bashgym.integrations.huggingface.client import HFUserInfo

        data = {
            "name": "testuser",
            "fullname": "Test User",
            "email": "test@example.com",
            "orgs": [
                {"name": "org1"},
                {"name": "org2"},
            ],
            "isPro": True,
            "canPay": True,
            "avatarUrl": "https://example.com/avatar.png",
        }

        info = HFUserInfo.from_whoami(data)

        assert info.username == "testuser"
        assert info.fullname == "Test User"
        assert info.email == "test@example.com"
        assert info.orgs == ["org1", "org2"]
        assert info.is_pro == True
        assert info.can_pay == True
        assert info.avatar_url == "https://example.com/avatar.png"

    def test_user_info_minimal_response(self):
        """Test HFUserInfo with minimal whoami response."""
        from bashgym.integrations.huggingface.client import HFUserInfo

        data = {
            "name": "minuser",
        }

        info = HFUserInfo.from_whoami(data)

        assert info.username == "minuser"
        assert info.fullname is None
        assert info.email is None
        assert info.orgs == []
        assert info.is_pro == False
        assert info.can_pay == False


# =============================================================================
# Test: Integration with Main Package
# =============================================================================

class TestPackageIntegration:
    """Tests for integration with main bashgym.integrations package."""

    def test_imports_from_integrations(self):
        """Test that HF classes are importable from integrations."""
        from bashgym.integrations import (
            HuggingFaceClient,
            get_hf_client,
            reset_hf_client,
            HF_HUB_AVAILABLE,
            HFError,
            HFAuthError,
            HFProRequiredError,
            HFQuotaExceededError,
            HFJobFailedError,
        )

        # All should be importable
        assert HuggingFaceClient is not None
        assert get_hf_client is not None
        assert reset_hf_client is not None
        assert isinstance(HF_HUB_AVAILABLE, bool)

    def test_hf_hub_available_flag(self):
        """Test HF_HUB_AVAILABLE flag reflects library availability."""
        from bashgym.integrations.huggingface import HF_HUB_AVAILABLE

        # Check if huggingface_hub is installed
        try:
            import huggingface_hub
            expected = True
        except ImportError:
            expected = False

        assert HF_HUB_AVAILABLE == expected


# =============================================================================
# Test: HFJobConfig
# =============================================================================

class TestHFJobConfig:
    """Tests for HFJobConfig dataclass."""

    def test_job_config_defaults(self):
        """Test default configuration values."""
        from bashgym.integrations.huggingface.jobs import HFJobConfig

        config = HFJobConfig()
        assert config.hardware == "a10g-small"
        assert config.timeout_minutes == 30
        assert config.docker_image is None
        assert config.environment == {}
        assert config.secrets == {}

    def test_job_config_custom(self):
        """Test custom configuration values."""
        from bashgym.integrations.huggingface.jobs import HFJobConfig

        config = HFJobConfig(
            hardware="a100-large",
            timeout_minutes=120,
            docker_image="custom/image:latest",
            environment={"ENV_VAR": "value"},
            secrets={"API_KEY": "secret"},
            requirements="requirements.txt",
            dataset_repo="user/dataset",
            output_repo="user/model",
        )

        assert config.hardware == "a100-large"
        assert config.timeout_minutes == 120
        assert config.docker_image == "custom/image:latest"
        assert config.environment == {"ENV_VAR": "value"}
        assert config.secrets == {"API_KEY": "secret"}
        assert config.requirements == "requirements.txt"
        assert config.dataset_repo == "user/dataset"
        assert config.output_repo == "user/model"

    def test_job_config_validation_valid(self):
        """Test validation passes for valid config."""
        from bashgym.integrations.huggingface.jobs import HFJobConfig

        config = HFJobConfig()
        errors = config.validate()
        assert errors == []

    def test_job_config_validation_invalid_hardware(self):
        """Test validation fails for invalid hardware."""
        from bashgym.integrations.huggingface.jobs import HFJobConfig

        config = HFJobConfig(hardware="invalid-tier")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid hardware" in errors[0]

    def test_job_config_validation_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        from bashgym.integrations.huggingface.jobs import HFJobConfig

        # Too short
        config = HFJobConfig(timeout_minutes=0)
        errors = config.validate()
        assert any("at least 1" in e for e in errors)

        # Too long
        config = HFJobConfig(timeout_minutes=1000)
        errors = config.validate()
        assert any("cannot exceed 720" in e for e in errors)


# =============================================================================
# Test: HFJobInfo
# =============================================================================

class TestHFJobInfo:
    """Tests for HFJobInfo dataclass."""

    def test_job_info_creation(self):
        """Test creating HFJobInfo."""
        from bashgym.integrations.huggingface.jobs import HFJobInfo, JobStatus
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        info = HFJobInfo(
            job_id="job_123",
            status=JobStatus.RUNNING,
            hardware="a10g-small",
            created_at=now,
        )

        assert info.job_id == "job_123"
        assert info.status == JobStatus.RUNNING
        assert info.hardware == "a10g-small"
        assert info.created_at == now
        assert info.started_at is None
        assert info.completed_at is None
        assert info.is_terminal == False

    def test_job_info_terminal_states(self):
        """Test is_terminal property for different states."""
        from bashgym.integrations.huggingface.jobs import HFJobInfo, JobStatus
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Non-terminal states
        for status in [JobStatus.PENDING, JobStatus.RUNNING]:
            info = HFJobInfo(job_id="job", status=status, hardware="t4-small", created_at=now)
            assert info.is_terminal == False

        # Terminal states
        for status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            info = HFJobInfo(job_id="job", status=status, hardware="t4-small", created_at=now)
            assert info.is_terminal == True

    def test_job_info_to_dict(self):
        """Test to_dict serialization."""
        from bashgym.integrations.huggingface.jobs import HFJobInfo, JobStatus
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        info = HFJobInfo(
            job_id="job_123",
            status=JobStatus.COMPLETED,
            hardware="a10g-small",
            created_at=now,
            completed_at=now,
            metrics={"loss": 0.5},
        )

        d = info.to_dict()
        assert d["job_id"] == "job_123"
        assert d["status"] == "completed"
        assert d["hardware"] == "a10g-small"
        assert d["metrics"]["loss"] == 0.5

    def test_job_info_from_dict(self):
        """Test from_dict deserialization."""
        from bashgym.integrations.huggingface.jobs import HFJobInfo, JobStatus
        from datetime import datetime, timezone

        data = {
            "job_id": "job_456",
            "status": "running",
            "hardware": "h100",
            "created_at": "2025-01-15T10:00:00+00:00",
            "metrics": {"epoch": 1},
        }

        info = HFJobInfo.from_dict(data)
        assert info.job_id == "job_456"
        assert info.status == JobStatus.RUNNING
        assert info.hardware == "h100"
        assert info.metrics["epoch"] == 1

    def test_job_info_duration(self):
        """Test duration_seconds property."""
        from bashgym.integrations.huggingface.jobs import HFJobInfo, JobStatus
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        started = now - timedelta(minutes=30)

        # Running job - duration from start to now
        info = HFJobInfo(
            job_id="job",
            status=JobStatus.RUNNING,
            hardware="t4-small",
            created_at=now,
            started_at=started,
        )
        duration = info.duration_seconds
        assert duration is not None
        assert duration >= 1800  # At least 30 minutes

        # Completed job - duration from start to complete
        completed = started + timedelta(minutes=15)
        info.completed_at = completed
        info.status = JobStatus.COMPLETED
        assert info.duration_seconds == 900  # Exactly 15 minutes


# =============================================================================
# Test: JobStatus Enum
# =============================================================================

class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self):
        """Test JobStatus enum values."""
        from bashgym.integrations.huggingface.jobs import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_job_status_from_string(self):
        """Test creating JobStatus from string."""
        from bashgym.integrations.huggingface.jobs import JobStatus

        assert JobStatus("pending") == JobStatus.PENDING
        assert JobStatus("completed") == JobStatus.COMPLETED


# =============================================================================
# Test: HFJobRunner
# =============================================================================

class TestHFJobRunner:
    """Tests for HFJobRunner class."""

    def test_job_runner_without_pro(self):
        """Test job runner without Pro subscription."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner
        from bashgym.integrations.huggingface import HFProRequiredError, reset_hf_client

        reset_hf_client()

        # Create runner without Pro
        runner = HFJobRunner(token="hf_test", pro_enabled=False)

        assert not runner.is_pro

        # All operations should require Pro
        with pytest.raises(HFProRequiredError):
            runner.submit_training_job("train.py")

        with pytest.raises(HFProRequiredError):
            runner.get_job_status("job_123")

        with pytest.raises(HFProRequiredError):
            runner.get_job_logs("job_123")

        with pytest.raises(HFProRequiredError):
            runner.cancel_job("job_123")

        with pytest.raises(HFProRequiredError):
            runner.list_jobs()

    def test_job_runner_with_pro_override(self):
        """Test job runner with Pro override enabled."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        # Create runner with Pro override
        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        assert runner.is_pro

    def test_job_runner_with_client(self):
        """Test job runner with provided client."""
        from bashgym.integrations.huggingface import HuggingFaceClient, reset_hf_client
        from bashgym.integrations.huggingface.jobs import HFJobRunner

        reset_hf_client()

        # Create mock client
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "prouser", "isPro": True}

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test")
                runner = HFJobRunner(client=client)

                assert runner.is_pro
                assert runner.client is client

    def test_submit_job_validates_config(self):
        """Test that submit_training_job validates configuration."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobConfig
        from bashgym.integrations.huggingface import reset_hf_client
        import tempfile
        import os

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        # Create temp script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            script_path = f.name

        try:
            # Invalid hardware
            with pytest.raises(ValueError) as exc_info:
                runner.submit_training_job(
                    script_path,
                    config=HFJobConfig(hardware="invalid")
                )
            assert "Invalid hardware" in str(exc_info.value)

            # Invalid timeout
            with pytest.raises(ValueError) as exc_info:
                runner.submit_training_job(
                    script_path,
                    config=HFJobConfig(timeout_minutes=0)
                )
            assert "at least 1" in str(exc_info.value)

        finally:
            os.unlink(script_path)

    def test_submit_job_requires_script_exists(self):
        """Test that submit_training_job requires existing script."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        with pytest.raises(FileNotFoundError):
            runner.submit_training_job("/nonexistent/script.py")

    def test_submit_job_success(self):
        """Test successful job submission."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobConfig, JobStatus
        from bashgym.integrations.huggingface import reset_hf_client
        import tempfile
        import os

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        # Create temp script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('training')")
            script_path = f.name

        try:
            job = runner.submit_training_job(
                script_path,
                repo_id="testuser/training-job",
                config=HFJobConfig(hardware="a10g-small", timeout_minutes=60),
            )

            assert job.job_id is not None
            assert job.status == JobStatus.PENDING
            assert job.hardware == "a10g-small"
            assert job.logs_url is not None

            # Job should be tracked
            assert job.job_id in runner._jobs

        finally:
            os.unlink(script_path)

    def test_get_job_status(self):
        """Test getting job status."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobInfo, JobStatus
        from bashgym.integrations.huggingface import reset_hf_client
        from datetime import datetime, timezone

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        # Pre-populate a job
        job = HFJobInfo(
            job_id="test_job_001",
            status=JobStatus.RUNNING,
            hardware="a10g-small",
            created_at=datetime.now(timezone.utc),
        )
        runner._jobs["test_job_001"] = job

        # Get status
        result = runner.get_job_status("test_job_001")
        assert result.job_id == "test_job_001"
        assert result.status == JobStatus.RUNNING

        # Unknown job raises KeyError
        with pytest.raises(KeyError):
            runner.get_job_status("nonexistent_job")

    def test_get_job_logs(self):
        """Test getting job logs."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobInfo, JobStatus
        from bashgym.integrations.huggingface import reset_hf_client
        from datetime import datetime, timezone

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        # Pre-populate a job
        job = HFJobInfo(
            job_id="test_job_002",
            status=JobStatus.RUNNING,
            hardware="t4-small",
            created_at=datetime.now(timezone.utc),
        )
        runner._jobs["test_job_002"] = job

        logs = runner.get_job_logs("test_job_002")
        assert "test_job_002" in logs
        assert "running" in logs.lower()

    def test_cancel_job(self):
        """Test cancelling a job."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobInfo, JobStatus
        from bashgym.integrations.huggingface import HFJobFailedError, reset_hf_client
        from datetime import datetime, timezone

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        # Pre-populate a running job
        job = HFJobInfo(
            job_id="test_job_003",
            status=JobStatus.RUNNING,
            hardware="a10g-small",
            created_at=datetime.now(timezone.utc),
        )
        runner._jobs["test_job_003"] = job

        # Cancel job
        result = runner.cancel_job("test_job_003")
        assert result.status == JobStatus.CANCELLED
        assert result.completed_at is not None

        # Cancelling again should fail
        with pytest.raises(HFJobFailedError):
            runner.cancel_job("test_job_003")

    def test_cancel_job_already_completed(self):
        """Test that cancelling a completed job fails."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobInfo, JobStatus
        from bashgym.integrations.huggingface import HFJobFailedError, reset_hf_client
        from datetime import datetime, timezone

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        # Pre-populate a completed job
        job = HFJobInfo(
            job_id="test_job_004",
            status=JobStatus.COMPLETED,
            hardware="a10g-small",
            created_at=datetime.now(timezone.utc),
        )
        runner._jobs["test_job_004"] = job

        with pytest.raises(HFJobFailedError) as exc_info:
            runner.cancel_job("test_job_004")
        assert "terminal state" in str(exc_info.value)

    def test_list_jobs(self):
        """Test listing jobs."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner, HFJobInfo, JobStatus
        from bashgym.integrations.huggingface import reset_hf_client
        from datetime import datetime, timezone, timedelta

        reset_hf_client()

        runner = HFJobRunner(token="hf_test", pro_enabled=True)

        # Pre-populate multiple jobs
        now = datetime.now(timezone.utc)
        runner._jobs["job_1"] = HFJobInfo(
            job_id="job_1", status=JobStatus.COMPLETED,
            hardware="t4-small", created_at=now - timedelta(hours=2)
        )
        runner._jobs["job_2"] = HFJobInfo(
            job_id="job_2", status=JobStatus.RUNNING,
            hardware="a10g-small", created_at=now - timedelta(hours=1)
        )
        runner._jobs["job_3"] = HFJobInfo(
            job_id="job_3", status=JobStatus.PENDING,
            hardware="a100-large", created_at=now
        )

        # List all jobs (sorted by creation, newest first)
        jobs = runner.list_jobs()
        assert len(jobs) == 3
        assert jobs[0].job_id == "job_3"  # Newest first
        assert jobs[2].job_id == "job_1"  # Oldest last

        # Filter by status
        running = runner.list_jobs(status=JobStatus.RUNNING)
        assert len(running) == 1
        assert running[0].job_id == "job_2"

        completed = runner.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].job_id == "job_1"

        # Test limit
        limited = runner.list_jobs(limit=2)
        assert len(limited) == 2

    def test_job_runner_repr(self):
        """Test string representation of HFJobRunner."""
        from bashgym.integrations.huggingface.jobs import HFJobRunner
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        # Without Pro
        runner = HFJobRunner(token="hf_test", pro_enabled=False)
        repr_str = repr(runner)
        assert "HFJobRunner" in repr_str
        assert "jobs=0" in repr_str
        assert "[Pro]" not in repr_str

        # With Pro
        runner = HFJobRunner(token="hf_test", pro_enabled=True)
        repr_str = repr(runner)
        assert "[Pro]" in repr_str


# =============================================================================
# Test: HARDWARE_SPECS
# =============================================================================

class TestHardwareSpecs:
    """Tests for HARDWARE_SPECS constant."""

    def test_hardware_specs_exists(self):
        """Test HARDWARE_SPECS has expected entries."""
        from bashgym.integrations.huggingface.jobs import HARDWARE_SPECS

        assert "cpu-basic" in HARDWARE_SPECS
        assert "t4-small" in HARDWARE_SPECS
        assert "a10g-small" in HARDWARE_SPECS
        assert "a100-large" in HARDWARE_SPECS
        assert "h100" in HARDWARE_SPECS

    def test_hardware_specs_structure(self):
        """Test HARDWARE_SPECS entry structure."""
        from bashgym.integrations.huggingface.jobs import HARDWARE_SPECS

        for name, spec in HARDWARE_SPECS.items():
            assert "gpu" in spec
            assert "memory_gb" in spec
            assert "pro_required" in spec
            assert isinstance(spec["memory_gb"], (int, float))
            assert isinstance(spec["pro_required"], bool)

    def test_hardware_pro_requirements(self):
        """Test Pro requirements for hardware tiers."""
        from bashgym.integrations.huggingface.jobs import HARDWARE_SPECS

        # CPU tiers should not require Pro
        assert HARDWARE_SPECS["cpu-basic"]["pro_required"] == False
        assert HARDWARE_SPECS["cpu-upgrade"]["pro_required"] == False

        # GPU tiers should require Pro
        assert HARDWARE_SPECS["t4-small"]["pro_required"] == True
        assert HARDWARE_SPECS["a10g-small"]["pro_required"] == True
        assert HARDWARE_SPECS["a100-large"]["pro_required"] == True
        assert HARDWARE_SPECS["h100"]["pro_required"] == True


# =============================================================================
# Test: create_job_runner convenience function
# =============================================================================

class TestCreateJobRunner:
    """Tests for create_job_runner convenience function."""

    def test_create_job_runner(self):
        """Test creating job runner via convenience function."""
        from bashgym.integrations.huggingface.jobs import create_job_runner
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        runner = create_job_runner(token="hf_test", pro_enabled=True)
        assert runner.is_pro

    def test_create_job_runner_no_pro(self):
        """Test creating job runner without Pro."""
        from bashgym.integrations.huggingface.jobs import create_job_runner
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        runner = create_job_runner(token="hf_test", pro_enabled=False)
        assert not runner.is_pro


# =============================================================================
# Test: Integration with main integrations package
# =============================================================================

class TestJobsPackageIntegration:
    """Tests for job exports from main package."""

    def test_imports_from_huggingface_package(self):
        """Test job classes importable from huggingface package."""
        from bashgym.integrations.huggingface import (
            HFJobRunner,
            HFJobConfig,
            HFJobInfo,
            JobStatus,
            HARDWARE_SPECS,
            create_job_runner,
        )

        assert HFJobRunner is not None
        assert HFJobConfig is not None
        assert HFJobInfo is not None
        assert JobStatus is not None
        assert HARDWARE_SPECS is not None
        assert create_job_runner is not None

    def test_imports_from_integrations_package(self):
        """Test job classes importable from main integrations package."""
        from bashgym.integrations import (
            HFJobRunner,
            HFJobConfig,
            HFJobInfo,
            JobStatus,
            HARDWARE_SPECS,
            create_job_runner,
        )

        assert HFJobRunner is not None
        assert HFJobConfig is not None
        assert HFJobInfo is not None
        assert JobStatus is not None
        assert isinstance(HARDWARE_SPECS, dict)
        assert callable(create_job_runner)


# =============================================================================
# Test: HFInferenceConfig
# =============================================================================

class TestHFInferenceConfig:
    """Tests for HFInferenceConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from bashgym.integrations.huggingface.inference import HFInferenceConfig

        config = HFInferenceConfig()
        assert config.provider == "auto"
        assert config.routing == "fastest"
        assert config.bill_to is None
        assert config.timeout == 30.0

    def test_config_custom(self):
        """Test custom configuration values."""
        from bashgym.integrations.huggingface.inference import HFInferenceConfig

        config = HFInferenceConfig(
            provider="dedicated",
            routing="cheapest",
            bill_to="org-123",
            timeout=60.0,
        )

        assert config.provider == "dedicated"
        assert config.routing == "cheapest"
        assert config.bill_to == "org-123"
        assert config.timeout == 60.0

    def test_config_validation_valid(self):
        """Test validation passes for valid config."""
        from bashgym.integrations.huggingface.inference import HFInferenceConfig

        config = HFInferenceConfig()
        errors = config.validate()
        assert errors == []

    def test_config_validation_invalid_provider(self):
        """Test validation fails for invalid provider."""
        from bashgym.integrations.huggingface.inference import HFInferenceConfig

        config = HFInferenceConfig(provider="invalid")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid provider" in errors[0]

    def test_config_validation_invalid_routing(self):
        """Test validation fails for invalid routing."""
        from bashgym.integrations.huggingface.inference import HFInferenceConfig

        config = HFInferenceConfig(routing="invalid")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid routing" in errors[0]

    def test_config_validation_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        from bashgym.integrations.huggingface.inference import HFInferenceConfig

        config = HFInferenceConfig(timeout=-1)
        errors = config.validate()
        assert len(errors) == 1
        assert "timeout must be positive" in errors[0]


# =============================================================================
# Test: InferenceUsage
# =============================================================================

class TestInferenceUsage:
    """Tests for InferenceUsage dataclass."""

    def test_usage_creation(self):
        """Test creating InferenceUsage."""
        from bashgym.integrations.huggingface.inference import InferenceUsage

        usage = InferenceUsage(
            provider="serverless",
            model="meta-llama/Llama-3.1-8B-Instruct",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            latency_ms=500.0,
        )

        assert usage.provider == "serverless"
        assert usage.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cost_usd == 0.001
        assert usage.latency_ms == 500.0

    def test_usage_total_tokens(self):
        """Test total_tokens property."""
        from bashgym.integrations.huggingface.inference import InferenceUsage

        usage = InferenceUsage(
            provider="serverless",
            model="test",
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.total_tokens == 150

    def test_usage_to_dict(self):
        """Test to_dict serialization."""
        from bashgym.integrations.huggingface.inference import InferenceUsage

        usage = InferenceUsage(
            provider="dedicated",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.002,
            latency_ms=300.0,
        )

        d = usage.to_dict()
        assert d["provider"] == "dedicated"
        assert d["model"] == "test-model"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["cost_usd"] == 0.002
        assert d["latency_ms"] == 300.0


# =============================================================================
# Test: Response Classes
# =============================================================================

class TestResponseClasses:
    """Tests for response dataclasses."""

    def test_generation_response(self):
        """Test GenerationResponse creation."""
        from bashgym.integrations.huggingface.inference import (
            GenerationResponse,
            InferenceUsage,
        )

        usage = InferenceUsage(
            provider="serverless",
            model="test",
            input_tokens=10,
            output_tokens=20,
        )

        response = GenerationResponse(
            text="Hello world!",
            usage=usage,
            finish_reason="eos",
        )

        assert response.text == "Hello world!"
        assert response.usage.total_tokens == 30
        assert response.finish_reason == "eos"

    def test_embedding_response(self):
        """Test EmbeddingResponse creation."""
        from bashgym.integrations.huggingface.inference import (
            EmbeddingResponse,
            InferenceUsage,
        )

        usage = InferenceUsage(
            provider="serverless",
            model="test",
            input_tokens=5,
            output_tokens=0,
        )

        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            usage=usage,
        )

        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 3
        assert response.usage.output_tokens == 0

    def test_classification_response(self):
        """Test ClassificationResponse creation."""
        from bashgym.integrations.huggingface.inference import (
            ClassificationResponse,
            InferenceUsage,
        )

        usage = InferenceUsage(
            provider="serverless",
            model="test",
            input_tokens=10,
            output_tokens=0,
        )

        response = ClassificationResponse(
            labels={"positive": 0.9, "negative": 0.1},
            usage=usage,
            top_label="positive",
        )

        assert response.labels["positive"] == 0.9
        assert response.top_label == "positive"


# =============================================================================
# Test: Inference Provider Enums
# =============================================================================

class TestInferenceEnums:
    """Tests for inference-related enums."""

    def test_inference_provider_enum(self):
        """Test InferenceProvider enum values."""
        from bashgym.integrations.huggingface.inference import InferenceProvider

        assert InferenceProvider.AUTO.value == "auto"
        assert InferenceProvider.SERVERLESS.value == "serverless"
        assert InferenceProvider.DEDICATED.value == "dedicated"
        assert InferenceProvider.FINETUNED.value == "finetuned"

    def test_routing_strategy_enum(self):
        """Test RoutingStrategy enum values."""
        from bashgym.integrations.huggingface.inference import RoutingStrategy

        assert RoutingStrategy.FASTEST.value == "fastest"
        assert RoutingStrategy.CHEAPEST.value == "cheapest"
        assert RoutingStrategy.QUALITY.value == "quality"


# =============================================================================
# Test: HFInferenceClient
# =============================================================================

class TestHFInferenceClient:
    """Tests for HFInferenceClient class."""

    def test_client_without_library(self):
        """Test client behavior when huggingface_hub InferenceClient not available."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        # Mock the library as unavailable
        import bashgym.integrations.huggingface.inference as inf_module
        original_available = inf_module.HF_INFERENCE_AVAILABLE

        try:
            inf_module.HF_INFERENCE_AVAILABLE = False
            client = HFInferenceClient(token="hf_test")
            client._initialized = False  # Force re-init with patched value

            # Should not be available
            assert not client.is_available
        finally:
            inf_module.HF_INFERENCE_AVAILABLE = original_available

    def test_client_without_token(self):
        """Test client behavior with no token."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )
        from bashgym.integrations.huggingface import HFAuthError

        reset_inference_client()

        # Create client without token and with mocked unavailable inference
        import bashgym.integrations.huggingface.inference as inf_module
        original_available = inf_module.HF_INFERENCE_AVAILABLE

        try:
            # Simulate library available but no token
            inf_module.HF_INFERENCE_AVAILABLE = True
            inf_module._HFInferenceClient = None  # No actual client

            client = HFInferenceClient(token=None)
            client._initialized = True
            client._init_error = "No HuggingFace token available"
            client._inference_client = None

            assert not client.is_available

            # Operations should raise HFAuthError
            with pytest.raises(HFAuthError):
                client.generate(model="test", prompt="hello")

        finally:
            inf_module.HF_INFERENCE_AVAILABLE = original_available

    def test_client_config_validation(self):
        """Test that client validates config on creation."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            HFInferenceConfig,
        )

        with pytest.raises(ValueError) as exc_info:
            HFInferenceClient(
                token="hf_test",
                config=HFInferenceConfig(provider="invalid"),
            )
        assert "Invalid provider" in str(exc_info.value)

    def test_client_parse_routing(self):
        """Test model ID routing suffix parsing."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        client = HFInferenceClient(token="hf_test")

        # Without routing
        model, routing = client._parse_model_routing("meta-llama/Llama-3.1-8B")
        assert model == "meta-llama/Llama-3.1-8B"
        assert routing is None

        # With :fastest
        model, routing = client._parse_model_routing("meta-llama/Llama-3.1-8B:fastest")
        assert model == "meta-llama/Llama-3.1-8B"
        assert routing == "fastest"

        # With :cheapest
        model, routing = client._parse_model_routing("meta-llama/Llama-3.1-8B:cheapest")
        assert model == "meta-llama/Llama-3.1-8B"
        assert routing == "cheapest"

        # With :quality
        model, routing = client._parse_model_routing("meta-llama/Llama-3.1-8B:quality")
        assert model == "meta-llama/Llama-3.1-8B"
        assert routing == "quality"

    def test_client_generate_with_mock(self):
        """Test generate method with mocked inference client."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        # Create mock response with proper details attribute
        mock_details = MagicMock()
        mock_details.prefill_tokens = 5
        mock_details.generated_tokens = 10

        mock_response = MagicMock()
        mock_response.generated_text = "Hello, I am an AI."
        mock_response.finish_reason = "eos"
        mock_response.details = mock_details

        mock_inference = MagicMock()
        mock_inference.text_generation.return_value = mock_response

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._init_error = None
        client._inference_client = mock_inference

        response = client.generate(
            model="meta-llama/Llama-3.1-8B-Instruct",
            prompt="Hello!",
            max_tokens=50,
            temperature=0.7,
        )

        assert response.text == "Hello, I am an AI."
        assert response.finish_reason == "eos"
        assert response.usage.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert response.usage.input_tokens == 5
        assert response.usage.output_tokens == 10

        mock_inference.text_generation.assert_called_once()

    def test_client_generate_with_routing_suffix(self):
        """Test generate method with :fastest routing suffix."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        mock_response = MagicMock()
        mock_response.generated_text = "Fast response"
        mock_response.finish_reason = "length"

        mock_inference = MagicMock()
        mock_inference.text_generation.return_value = mock_response

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        response = client.generate(
            model="meta-llama/Llama-3.1-8B-Instruct:fastest",
            prompt="Hello!",
        )

        assert response.text == "Fast response"
        assert response.usage.provider == "fastest"

    def test_client_embed_with_mock(self):
        """Test embed method with mocked inference client."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        mock_inference = MagicMock()
        mock_inference.feature_extraction.return_value = mock_embeddings

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        response = client.embed(
            model="sentence-transformers/all-MiniLM-L6-v2",
            texts=["Hello", "World"],
        )

        assert len(response.embeddings) == 2
        assert response.embeddings[0] == [0.1, 0.2, 0.3]
        assert response.usage.output_tokens == 0  # Embeddings have no output tokens

    def test_client_embed_single_text(self):
        """Test embed method with single text string."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        mock_embedding = [0.1, 0.2, 0.3, 0.4]

        mock_inference = MagicMock()
        mock_inference.feature_extraction.return_value = [mock_embedding]

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        response = client.embed(
            model="sentence-transformers/all-MiniLM-L6-v2",
            texts="Hello world",  # Single string, not list
        )

        assert len(response.embeddings) == 1

    def test_client_classify_with_mock(self):
        """Test classify method with mocked inference client."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        mock_response = [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.1},
        ]

        mock_inference = MagicMock()
        mock_inference.text_classification.return_value = mock_response

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        response = client.classify(
            model="distilbert-base-uncased-finetuned-sst-2-english",
            text="I love this product!",
        )

        assert response.labels["positive"] == 0.9
        assert response.labels["negative"] == 0.1
        assert response.top_label == "positive"

    def test_client_classify_zero_shot(self):
        """Test classify method with zero-shot classification."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        mock_response = {
            "labels": ["technology", "sports", "politics"],
            "scores": [0.7, 0.2, 0.1],
        }

        mock_inference = MagicMock()
        mock_inference.zero_shot_classification.return_value = mock_response

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        response = client.classify(
            model="facebook/bart-large-mnli",
            text="The new iPhone has amazing features",
            candidate_labels=["technology", "sports", "politics"],
        )

        assert response.labels["technology"] == 0.7
        assert response.top_label == "technology"
        mock_inference.zero_shot_classification.assert_called_once()

    def test_client_chat_with_mock(self):
        """Test chat method with mocked inference client."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )

        reset_inference_client()

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello! How can I help you?"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 8

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_inference = MagicMock()
        mock_inference.chat_completion.return_value = mock_response

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        response = client.chat(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert response.text == "Hello! How can I help you?"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 8

    def test_client_quota_error_detection(self):
        """Test that quota errors are properly detected and raised."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )
        from bashgym.integrations.huggingface import HFQuotaExceededError

        reset_inference_client()

        mock_inference = MagicMock()
        mock_inference.text_generation.side_effect = Exception("Rate limit exceeded")

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        with pytest.raises(HFQuotaExceededError):
            client.generate(model="test", prompt="hello")

    def test_client_quota_error_429(self):
        """Test that 429 errors are detected as quota errors."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            reset_inference_client,
        )
        from bashgym.integrations.huggingface import HFQuotaExceededError

        reset_inference_client()

        mock_inference = MagicMock()
        mock_inference.text_generation.side_effect = Exception("HTTP 429 Too Many Requests")

        client = HFInferenceClient(token="hf_test")
        client._initialized = True
        client._inference_client = mock_inference

        with pytest.raises(HFQuotaExceededError):
            client.generate(model="test", prompt="hello")

    def test_client_repr(self):
        """Test string representation of HFInferenceClient."""
        from bashgym.integrations.huggingface.inference import (
            HFInferenceClient,
            HFInferenceConfig,
            reset_inference_client,
        )

        reset_inference_client()

        # Available client
        mock_inference = MagicMock()
        client = HFInferenceClient(token="hf_test", config=HFInferenceConfig(routing="cheapest"))
        client._initialized = True
        client._inference_client = mock_inference

        repr_str = repr(client)
        assert "HFInferenceClient" in repr_str
        assert "cheapest" in repr_str

        # Disabled client
        client._inference_client = None
        client._init_error = "No token"
        repr_str = repr(client)
        assert "disabled" in repr_str


# =============================================================================
# Test: Singleton Access
# =============================================================================

class TestInferenceSingletonAccess:
    """Tests for get_inference_client singleton function."""

    def test_get_inference_client_returns_singleton(self):
        """Test that get_inference_client returns the same instance."""
        from bashgym.integrations.huggingface.inference import (
            get_inference_client,
            reset_inference_client,
        )

        reset_inference_client()

        client1 = get_inference_client(token="hf_test")
        client2 = get_inference_client()

        assert client1 is client2

    def test_get_inference_client_force_new(self):
        """Test that force_new creates a new instance."""
        from bashgym.integrations.huggingface.inference import (
            get_inference_client,
            reset_inference_client,
        )

        reset_inference_client()

        client1 = get_inference_client(token="hf_test1")
        client2 = get_inference_client(token="hf_test2", force_new=True)

        # Different instances due to force_new
        assert client1 is not client2

    def test_reset_inference_client(self):
        """Test that reset_inference_client clears the singleton."""
        from bashgym.integrations.huggingface.inference import (
            get_inference_client,
            reset_inference_client,
        )

        reset_inference_client()

        client1 = get_inference_client(token="hf_test")
        reset_inference_client()
        client2 = get_inference_client(token="hf_test")

        # Different instances after reset
        assert client1 is not client2


# =============================================================================
# Test: Cost Estimation
# =============================================================================

class TestCostEstimation:
    """Tests for cost estimation functions."""

    def test_estimate_cost_default(self):
        """Test cost estimation for unknown models."""
        from bashgym.integrations.huggingface.inference import _estimate_cost

        cost = _estimate_cost("unknown/model", 1_000_000, 500_000)
        # default: input=0.10, output=0.20 per 1M
        expected = 0.10 + 0.10  # 1M input + 0.5M output
        assert cost == expected

    def test_estimate_cost_llama(self):
        """Test cost estimation for Llama models."""
        from bashgym.integrations.huggingface.inference import _estimate_cost

        cost = _estimate_cost("meta-llama/Llama-3.1-8B", 1_000_000, 1_000_000)
        # llama: input=0.10, output=0.15 per 1M
        expected = 0.10 + 0.15
        assert cost == expected

    def test_estimate_cost_embeddings(self):
        """Test cost estimation for embedding models."""
        from bashgym.integrations.huggingface.inference import _estimate_cost

        cost = _estimate_cost("sentence-transformers/all-MiniLM-L6-v2", 1_000_000, 0)
        # sentence-transformers: input=0.01, output=0.0 per 1M
        expected = 0.01
        assert cost == expected

    def test_estimate_tokens(self):
        """Test token estimation."""
        from bashgym.integrations.huggingface.inference import _estimate_tokens

        # ~4 chars per token
        assert _estimate_tokens("hello") == 1  # 5 chars / 4 = 1
        assert _estimate_tokens("hello world!") == 3  # 12 chars / 4 = 3


# =============================================================================
# Test: Inference Package Integration
# =============================================================================

class TestInferencePackageIntegration:
    """Tests for inference exports from packages."""

    def test_imports_from_huggingface_package(self):
        """Test inference classes importable from huggingface package."""
        from bashgym.integrations.huggingface import (
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
        )

        assert HFInferenceClient is not None
        assert HFInferenceConfig is not None
        assert InferenceUsage is not None
        assert GenerationResponse is not None
        assert EmbeddingResponse is not None
        assert ClassificationResponse is not None
        assert InferenceProvider is not None
        assert RoutingStrategy is not None
        assert isinstance(HF_INFERENCE_AVAILABLE, bool)
        assert callable(get_inference_client)
        assert callable(reset_inference_client)

    def test_imports_from_integrations_package(self):
        """Test inference classes importable from main integrations package."""
        from bashgym.integrations import (
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
        )

        assert HFInferenceClient is not None
        assert HFInferenceConfig is not None
        assert InferenceUsage is not None
        assert callable(get_inference_client)


# =============================================================================
# Test: HF_INFERENCE_AVAILABLE Flag
# =============================================================================

class TestHFInferenceAvailable:
    """Tests for HF_INFERENCE_AVAILABLE flag."""

    def test_hf_inference_available_flag(self):
        """Test HF_INFERENCE_AVAILABLE reflects library availability."""
        from bashgym.integrations.huggingface.inference import HF_INFERENCE_AVAILABLE

        # Check if InferenceClient is available
        try:
            from huggingface_hub import InferenceClient
            expected = True
        except ImportError:
            expected = False

        assert HF_INFERENCE_AVAILABLE == expected


# =============================================================================
# Test: SpaceConfig
# =============================================================================

class TestSpaceConfig:
    """Tests for SpaceConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from bashgym.integrations.huggingface.spaces import SpaceConfig

        config = SpaceConfig(name="my-space")
        assert config.name == "my-space"
        assert config.hardware == "zero-gpu"
        assert config.private == True
        assert config.sdk == "gradio"
        assert config.python_version == "3.10"
        assert config.dev_mode == False

    def test_config_custom(self):
        """Test custom configuration values."""
        from bashgym.integrations.huggingface.spaces import SpaceConfig

        config = SpaceConfig(
            name="custom-space",
            hardware="t4-small",
            private=False,
            sdk="streamlit",
            python_version="3.11",
            dev_mode=True,
        )

        assert config.name == "custom-space"
        assert config.hardware == "t4-small"
        assert config.private == False
        assert config.sdk == "streamlit"
        assert config.python_version == "3.11"
        assert config.dev_mode == True

    def test_config_validation_valid(self):
        """Test validation passes for valid config."""
        from bashgym.integrations.huggingface.spaces import SpaceConfig

        config = SpaceConfig(name="my-space")
        errors = config.validate()
        assert errors == []

    def test_config_validation_empty_name(self):
        """Test validation fails for empty name."""
        from bashgym.integrations.huggingface.spaces import SpaceConfig

        config = SpaceConfig(name="")
        errors = config.validate()
        assert len(errors) == 1
        assert "name is required" in errors[0]

    def test_config_validation_invalid_sdk(self):
        """Test validation fails for invalid SDK."""
        from bashgym.integrations.huggingface.spaces import SpaceConfig

        config = SpaceConfig(name="my-space", sdk="invalid")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid sdk" in errors[0]


# =============================================================================
# Test: SpaceStatus Enum
# =============================================================================

class TestSpaceStatus:
    """Tests for SpaceStatus enum."""

    def test_space_status_values(self):
        """Test SpaceStatus enum values."""
        from bashgym.integrations.huggingface.spaces import SpaceStatus

        assert SpaceStatus.BUILDING.value == "building"
        assert SpaceStatus.RUNNING.value == "running"
        assert SpaceStatus.STOPPED.value == "stopped"
        assert SpaceStatus.ERROR.value == "error"

    def test_space_status_from_string(self):
        """Test creating SpaceStatus from string."""
        from bashgym.integrations.huggingface.spaces import SpaceStatus

        assert SpaceStatus("building") == SpaceStatus.BUILDING
        assert SpaceStatus("running") == SpaceStatus.RUNNING
        assert SpaceStatus("stopped") == SpaceStatus.STOPPED
        assert SpaceStatus("error") == SpaceStatus.ERROR


# =============================================================================
# Test: SSHCredentials
# =============================================================================

class TestSSHCredentials:
    """Tests for SSHCredentials dataclass."""

    def test_ssh_credentials_creation(self):
        """Test SSHCredentials creation."""
        from bashgym.integrations.huggingface.spaces import SSHCredentials

        creds = SSHCredentials(
            host="ssh.spaces.huggingface.co",
            port=22,
            username="user123",
            key="-----BEGIN RSA PRIVATE KEY-----\n..."
        )

        assert creds.host == "ssh.spaces.huggingface.co"
        assert creds.port == 22
        assert creds.username == "user123"
        assert "RSA PRIVATE KEY" in creds.key


# =============================================================================
# Test: HFSpaceManager
# =============================================================================

class TestHFSpaceManager:
    """Tests for HFSpaceManager class."""

    def test_space_manager_without_pro(self):
        """Test space manager without Pro subscription."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceConfig
        from bashgym.integrations.huggingface import HFProRequiredError, reset_hf_client

        reset_hf_client()

        manager = HFSpaceManager(token="hf_test", pro_enabled=False)

        assert not manager.is_pro

        # Creating ZeroGPU Space requires Pro
        with pytest.raises(HFProRequiredError):
            manager.create_inference_space(
                model_repo="user/model",
                space_name="my-space",
                config=SpaceConfig(name="my-space"),
            )

    def test_space_manager_with_pro_override(self):
        """Test space manager with Pro override enabled."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        manager = HFSpaceManager(token="hf_test", pro_enabled=True)
        assert manager.is_pro

    def test_space_manager_with_client(self):
        """Test space manager with provided client."""
        from bashgym.integrations.huggingface import HuggingFaceClient, reset_hf_client
        from bashgym.integrations.huggingface.spaces import HFSpaceManager

        reset_hf_client()

        # Create mock client
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "prouser", "isPro": True}

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test")
                manager = HFSpaceManager(client=client)

                assert manager.is_pro
                assert manager.client is client

    def test_create_inference_space_success(self):
        """Test successful Space creation."""
        from bashgym.integrations.huggingface.spaces import (
            HFSpaceManager,
            SpaceConfig,
            SpaceStatus,
        )
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.create_repo.return_value = "https://huggingface.co/spaces/testuser/my-space"
        mock_api.upload_file.return_value = None
        mock_api.get_space_runtime.return_value = MagicMock(stage="RUNNING")

        manager = HFSpaceManager(token="hf_test", pro_enabled=True)
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None
        manager._client._user_info = MagicMock(username="testuser")

        url = manager.create_inference_space(
            model_repo="testuser/my-model",
            space_name="my-space",
            config=SpaceConfig(name="my-space"),
            gpu_duration=60,
        )

        assert "huggingface.co/spaces" in url or url.startswith("https://")
        mock_api.create_repo.assert_called_once()

    def test_create_inference_space_requires_pro(self):
        """Test that creating ZeroGPU Space requires Pro subscription."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceConfig
        from bashgym.integrations.huggingface import HFProRequiredError, reset_hf_client

        reset_hf_client()

        manager = HFSpaceManager(token="hf_test", pro_enabled=False)

        with pytest.raises(HFProRequiredError) as exc_info:
            manager.create_inference_space(
                model_repo="user/model",
                space_name="my-space",
                config=SpaceConfig(name="my-space"),
            )

        assert "Pro subscription" in str(exc_info.value)

    def test_get_space_status(self):
        """Test getting Space status."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceStatus
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.stage = "RUNNING"
        mock_api.get_space_runtime.return_value = mock_runtime

        manager = HFSpaceManager(token="hf_test", pro_enabled=True)
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None
        manager._client._user_info = MagicMock(username="testuser")

        status = manager.get_space_status("testuser/my-space")
        assert status == SpaceStatus.RUNNING

    def test_get_space_status_building(self):
        """Test getting Space status when building."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager, SpaceStatus
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.stage = "BUILDING"
        mock_api.get_space_runtime.return_value = mock_runtime

        manager = HFSpaceManager(token="hf_test", pro_enabled=True)
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None
        manager._client._user_info = MagicMock(username="testuser")

        status = manager.get_space_status("testuser/my-space")
        assert status == SpaceStatus.BUILDING

    def test_delete_space(self):
        """Test deleting a Space."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.delete_repo.return_value = None

        manager = HFSpaceManager(token="hf_test", pro_enabled=True)
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None
        manager._client._user_info = MagicMock(username="testuser")

        manager.delete_space("testuser/my-space")

        mock_api.delete_repo.assert_called_once_with(
            repo_id="testuser/my-space",
            repo_type="space",
        )

    def test_delete_space_requires_pro(self):
        """Test that deleting a Space requires Pro subscription."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager
        from bashgym.integrations.huggingface import HFProRequiredError, reset_hf_client

        reset_hf_client()

        manager = HFSpaceManager(token="hf_test", pro_enabled=False)

        with pytest.raises(HFProRequiredError):
            manager.delete_space("testuser/my-space")

    def test_update_space_model(self):
        """Test updating model in existing Space."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.upload_file.return_value = None
        mock_api.restart_space.return_value = None

        manager = HFSpaceManager(token="hf_test", pro_enabled=True)
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None
        manager._client._user_info = MagicMock(username="testuser")

        manager.update_space_model("testuser/my-space", "testuser/new-model")

        mock_api.upload_file.assert_called()
        mock_api.restart_space.assert_called_once_with(repo_id="testuser/my-space")

    def test_space_manager_repr(self):
        """Test string representation of HFSpaceManager."""
        from bashgym.integrations.huggingface.spaces import HFSpaceManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        # Without Pro
        manager = HFSpaceManager(token="hf_test", pro_enabled=False)
        repr_str = repr(manager)
        assert "HFSpaceManager" in repr_str
        assert "[Pro]" not in repr_str

        # With Pro
        manager = HFSpaceManager(token="hf_test", pro_enabled=True)
        repr_str = repr(manager)
        assert "[Pro]" in repr_str


# =============================================================================
# Test: GRADIO_APP_TEMPLATE
# =============================================================================

class TestGradioAppTemplate:
    """Tests for GRADIO_APP_TEMPLATE constant."""

    def test_template_exists(self):
        """Test GRADIO_APP_TEMPLATE exists."""
        from bashgym.integrations.huggingface.spaces import GRADIO_APP_TEMPLATE

        assert GRADIO_APP_TEMPLATE is not None
        assert isinstance(GRADIO_APP_TEMPLATE, str)

    def test_template_has_model_placeholder(self):
        """Test template has MODEL_ID placeholder."""
        from bashgym.integrations.huggingface.spaces import GRADIO_APP_TEMPLATE

        assert "{model_id}" in GRADIO_APP_TEMPLATE or "MODEL_ID" in GRADIO_APP_TEMPLATE

    def test_template_has_gpu_decorator(self):
        """Test template has @spaces.GPU decorator."""
        from bashgym.integrations.huggingface.spaces import GRADIO_APP_TEMPLATE

        assert "@spaces.GPU" in GRADIO_APP_TEMPLATE or "spaces.GPU" in GRADIO_APP_TEMPLATE

    def test_template_has_gradio_interface(self):
        """Test template creates Gradio interface."""
        from bashgym.integrations.huggingface.spaces import GRADIO_APP_TEMPLATE

        assert "gradio" in GRADIO_APP_TEMPLATE.lower() or "gr." in GRADIO_APP_TEMPLATE


# =============================================================================
# Test: Spaces Package Integration
# =============================================================================

class TestSpacesPackageIntegration:
    """Tests for Spaces exports from packages."""

    def test_imports_from_huggingface_package(self):
        """Test Spaces classes importable from huggingface package."""
        from bashgym.integrations.huggingface import (
            HFSpaceManager,
            SpaceConfig,
            SpaceStatus,
            SSHCredentials,
            GRADIO_APP_TEMPLATE,
        )

        assert HFSpaceManager is not None
        assert SpaceConfig is not None
        assert SpaceStatus is not None
        assert SSHCredentials is not None
        assert GRADIO_APP_TEMPLATE is not None

    def test_imports_from_integrations_package(self):
        """Test Spaces classes importable from main integrations package."""
        from bashgym.integrations import (
            HFSpaceManager,
            SpaceConfig,
            SpaceStatus,
            SSHCredentials,
            GRADIO_APP_TEMPLATE,
        )

        assert HFSpaceManager is not None
        assert SpaceConfig is not None
        assert SpaceStatus is not None
        assert SSHCredentials is not None
        assert isinstance(GRADIO_APP_TEMPLATE, str)


# =============================================================================
# Test: DatasetConfig
# =============================================================================

class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from bashgym.integrations.huggingface.datasets import DatasetConfig

        config = DatasetConfig(repo_name="my-dataset")
        assert config.repo_name == "my-dataset"
        assert config.private == True
        assert config.enable_viewer == True

    def test_config_custom(self):
        """Test custom configuration values."""
        from bashgym.integrations.huggingface.datasets import DatasetConfig

        config = DatasetConfig(
            repo_name="custom-dataset",
            private=False,
            enable_viewer=False,
        )

        assert config.repo_name == "custom-dataset"
        assert config.private == False
        assert config.enable_viewer == False

    def test_config_validation_valid(self):
        """Test validation passes for valid config."""
        from bashgym.integrations.huggingface.datasets import DatasetConfig

        config = DatasetConfig(repo_name="my-dataset")
        errors = config.validate()
        assert errors == []

    def test_config_validation_empty_name(self):
        """Test validation fails for empty name."""
        from bashgym.integrations.huggingface.datasets import DatasetConfig

        config = DatasetConfig(repo_name="")
        errors = config.validate()
        assert len(errors) == 1
        assert "repo_name is required" in errors[0]

    def test_config_validation_namespace_in_name(self):
        """Test validation fails if namespace included in name."""
        from bashgym.integrations.huggingface.datasets import DatasetConfig

        config = DatasetConfig(repo_name="user/dataset")
        errors = config.validate()
        assert len(errors) == 1
        assert "should not include namespace" in errors[0]


# =============================================================================
# Test: HFDatasetManager
# =============================================================================

class TestHFDatasetManager:
    """Tests for HFDatasetManager class."""

    def test_dataset_manager_creation(self):
        """Test dataset manager creation."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        manager = HFDatasetManager(token="hf_test", username="testuser")
        assert manager.username == "testuser"

    def test_dataset_manager_with_client(self):
        """Test dataset manager with provided client."""
        from bashgym.integrations.huggingface import HuggingFaceClient, reset_hf_client
        from bashgym.integrations.huggingface.datasets import HFDatasetManager

        reset_hf_client()

        # Create mock client
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "testuser", "isPro": True}

        with patch("bashgym.integrations.huggingface.client.HfApi", return_value=mock_api):
            with patch("bashgym.integrations.huggingface.client.HF_HUB_AVAILABLE", True):
                client = HuggingFaceClient(token="hf_test")
                manager = HFDatasetManager(client=client)

                assert manager.client is client

    def test_upload_training_data_requires_train_file(self):
        """Test upload requires train.jsonl."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client
        import tempfile

        reset_hf_client()

        manager = HFDatasetManager(token="hf_test", username="testuser")

        with tempfile.TemporaryDirectory() as tmpdir:
            # No train.jsonl exists
            with pytest.raises(FileNotFoundError):
                manager.upload_training_data(
                    local_path=Path(tmpdir),
                    repo_name="test-dataset",
                )

    def test_upload_training_data_success(self):
        """Test successful dataset upload."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager, DatasetConfig
        from bashgym.integrations.huggingface import reset_hf_client
        import tempfile

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.create_repo.return_value = None
        mock_api.upload_file.return_value = None

        manager = HFDatasetManager(token="hf_test", username="testuser")
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create train.jsonl
            train_file = tmppath / "train.jsonl"
            train_file.write_text('{"messages": []}\n{"messages": []}\n')

            # Create val.jsonl
            val_file = tmppath / "val.jsonl"
            val_file.write_text('{"messages": []}\n')

            url = manager.upload_training_data(
                local_path=tmppath,
                repo_name="test-dataset",
                config=DatasetConfig(repo_name="test-dataset"),
            )

            assert "huggingface.co/datasets" in url
            mock_api.create_repo.assert_called_once()
            # Should have uploaded train.jsonl, val.jsonl, and README.md
            assert mock_api.upload_file.call_count >= 3

    def test_upload_training_data_with_metadata(self):
        """Test dataset upload with metadata."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client
        import tempfile

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.create_repo.return_value = None
        mock_api.upload_file.return_value = None

        manager = HFDatasetManager(token="hf_test", username="testuser")
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create train.jsonl
            train_file = tmppath / "train.jsonl"
            train_file.write_text('{"messages": []}\n')

            url = manager.upload_training_data(
                local_path=tmppath,
                repo_name="test-dataset",
                metadata={"repos": "ghostwork, other-repo"},
            )

            assert "huggingface.co/datasets" in url
            # Should have uploaded train.jsonl, README.md, and metadata.json (3 files, no val.jsonl)
            assert mock_api.upload_file.call_count >= 3

    def test_download_dataset(self):
        """Test dataset download."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client
        import tempfile

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = None

        manager = HFDatasetManager(token="hf_test", username="testuser")
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None

        with tempfile.TemporaryDirectory() as tmpdir:
            manager.download_dataset(
                repo_id="testuser/test-dataset",
                local_path=Path(tmpdir),
            )

            mock_api.snapshot_download.assert_called_once()

    def test_list_datasets(self):
        """Test listing datasets."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_dataset1 = MagicMock()
        mock_dataset1.id = "testuser/dataset1"
        mock_dataset2 = MagicMock()
        mock_dataset2.id = "testuser/dataset2"

        mock_api = MagicMock()
        mock_api.list_datasets.return_value = [mock_dataset1, mock_dataset2]

        manager = HFDatasetManager(token="hf_test", username="testuser")
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None

        datasets = manager.list_datasets(prefix="bashgym")

        assert len(datasets) == 2
        assert "testuser/dataset1" in datasets
        assert "testuser/dataset2" in datasets

    def test_delete_dataset(self):
        """Test deleting a dataset."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.delete_repo.return_value = None

        manager = HFDatasetManager(token="hf_test", username="testuser")
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None

        result = manager.delete_dataset("test-dataset")

        assert result == True
        mock_api.delete_repo.assert_called_once()

    def test_delete_dataset_with_full_repo_id(self):
        """Test deleting a dataset with full repo ID."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.delete_repo.return_value = None

        manager = HFDatasetManager(token="hf_test", username="testuser")
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None

        result = manager.delete_dataset("otheruser/test-dataset")

        assert result == True
        # Should use the full repo ID as provided
        call_args = mock_api.delete_repo.call_args
        assert call_args[1]["repo_id"] == "otheruser/test-dataset"

    def test_upload_traces(self):
        """Test uploading traces as dataset."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        mock_api = MagicMock()
        mock_api.create_repo.return_value = None
        mock_api.upload_file.return_value = None

        manager = HFDatasetManager(token="hf_test", username="testuser")
        manager._client._api = mock_api
        manager._client._initialized = True
        manager._client._init_error = None

        traces = [
            {"trace_id": "1", "data": "test1"},
            {"trace_id": "2", "data": "test2"},
        ]

        url = manager.upload_traces(
            traces=traces,
            repo_name="traces-dataset",
        )

        assert "huggingface.co/datasets" in url
        mock_api.create_repo.assert_called_once()

    def test_count_lines(self):
        """Test line counting helper."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client
        import tempfile

        reset_hf_client()

        manager = HFDatasetManager(token="hf_test", username="testuser")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Test file with lines
            test_file = tmppath / "test.jsonl"
            test_file.write_text('line1\nline2\nline3\n')

            count = manager._count_lines(test_file)
            assert count == 3

            # Test nonexistent file
            count = manager._count_lines(tmppath / "nonexistent.jsonl")
            assert count == 0

    def test_get_size_category(self):
        """Test size category helper."""
        from bashgym.integrations.huggingface.datasets import HFDatasetManager
        from bashgym.integrations.huggingface import reset_hf_client

        reset_hf_client()

        manager = HFDatasetManager(token="hf_test", username="testuser")

        assert manager._get_size_category(500) == "n<1K"
        assert manager._get_size_category(5000) == "1K<n<10K"
        assert manager._get_size_category(50000) == "10K<n<100K"
        assert manager._get_size_category(500000) == "100K<n<1M"
        assert manager._get_size_category(5000000) == "n>1M"


# =============================================================================
# Test: DATASET_CARD_TEMPLATE
# =============================================================================

class TestDatasetCardTemplate:
    """Tests for DATASET_CARD_TEMPLATE."""

    def test_template_exists(self):
        """Test template is a non-empty string."""
        from bashgym.integrations.huggingface.datasets import DATASET_CARD_TEMPLATE

        assert isinstance(DATASET_CARD_TEMPLATE, str)
        assert len(DATASET_CARD_TEMPLATE) > 100

    def test_template_has_yaml_frontmatter(self):
        """Test template has YAML frontmatter."""
        from bashgym.integrations.huggingface.datasets import DATASET_CARD_TEMPLATE

        assert "---" in DATASET_CARD_TEMPLATE
        assert "license:" in DATASET_CARD_TEMPLATE
        assert "task_categories:" in DATASET_CARD_TEMPLATE

    def test_template_has_placeholders(self):
        """Test template has required placeholders."""
        from bashgym.integrations.huggingface.datasets import DATASET_CARD_TEMPLATE

        assert "{title}" in DATASET_CARD_TEMPLATE
        assert "{date}" in DATASET_CARD_TEMPLATE
        assert "{train_count" in DATASET_CARD_TEMPLATE
        assert "{val_count" in DATASET_CARD_TEMPLATE
        assert "{repo_id}" in DATASET_CARD_TEMPLATE


# =============================================================================
# Test: Dataset Package Integration
# =============================================================================

class TestDatasetPackageIntegration:
    """Tests for Dataset exports from packages."""

    def test_imports_from_huggingface_package(self):
        """Test Dataset classes importable from huggingface package."""
        from bashgym.integrations.huggingface import (
            HFDatasetManager,
            DatasetConfig,
            DATASET_CARD_TEMPLATE,
        )

        assert HFDatasetManager is not None
        assert DatasetConfig is not None
        assert DATASET_CARD_TEMPLATE is not None

    def test_imports_from_datasets_module(self):
        """Test Dataset classes importable from datasets module."""
        from bashgym.integrations.huggingface.datasets import (
            HFDatasetManager,
            DatasetConfig,
            DATASET_CARD_TEMPLATE,
        )

        assert HFDatasetManager is not None
        assert DatasetConfig is not None
        assert isinstance(DATASET_CARD_TEMPLATE, str)


# =============================================================================
# Test: Secrets Module
# =============================================================================

class TestSecretsModule:
    """Tests for bashgym.secrets module."""

    def test_mask_secret_normal(self):
        """Test masking a normal length secret."""
        from bashgym.secrets import mask_secret

        result = mask_secret("hf_abc123456789")
        assert result.endswith("6789")
        assert result.startswith("*")
        assert "hf_" not in result

    def test_mask_secret_short(self):
        """Test masking a short secret."""
        from bashgym.secrets import mask_secret

        result = mask_secret("abc")
        assert result == "***"

    def test_mask_secret_empty(self):
        """Test masking empty string."""
        from bashgym.secrets import mask_secret

        result = mask_secret("")
        assert result == ""

    def test_mask_secret_custom_visible(self):
        """Test masking with custom visible chars."""
        from bashgym.secrets import mask_secret

        result = mask_secret("hf_abc123456789", visible_chars=8)
        assert result.endswith("23456789")

    def test_get_secret_from_env(self):
        """Test that environment variables take precedence."""
        from bashgym.secrets import get_secret

        os.environ["TEST_SECRET_KEY"] = "env_value"
        try:
            result = get_secret("TEST_SECRET_KEY")
            assert result == "env_value"
        finally:
            del os.environ["TEST_SECRET_KEY"]

    def test_get_secret_missing(self):
        """Test getting a missing secret."""
        from bashgym.secrets import get_secret

        result = get_secret("DEFINITELY_NOT_A_REAL_SECRET_KEY_12345")
        assert result is None

    def test_has_secret(self):
        """Test checking if secret exists."""
        from bashgym.secrets import has_secret

        os.environ["TEST_HAS_SECRET"] = "value"
        try:
            assert has_secret("TEST_HAS_SECRET") is True
            assert has_secret("DEFINITELY_NOT_A_REAL_SECRET_KEY_12345") is False
        finally:
            del os.environ["TEST_HAS_SECRET"]

    def test_secrets_file_operations(self, tmp_path):
        """Test save/load secrets from file."""
        from bashgym.secrets import load_secrets, save_secrets, get_secrets_path
        from unittest.mock import patch

        # Mock the secrets path to use temp dir
        mock_path = tmp_path / "secrets.json"

        with patch("bashgym.secrets.get_secrets_path", return_value=mock_path):
            # Initially empty
            secrets = load_secrets()
            assert secrets == {}

            # Save some secrets
            save_secrets({"TEST_KEY": "test_value", "OTHER": "data"})

            # Load them back
            secrets = load_secrets()
            assert secrets["TEST_KEY"] == "test_value"
            assert secrets["OTHER"] == "data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
