"""
HuggingFace Hub Client

Core client for interacting with HuggingFace Hub API.
Gracefully handles missing huggingface_hub library and invalid tokens.

Features:
- Pro subscription auto-detection via whoami() API
- Token validation
- Singleton pattern for efficient reuse
- Error handling with specific exception types
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import (
        HfHubHTTPError,
        RepositoryNotFoundError,
        GatedRepoError,
    )
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HfApi = None
    HfHubHTTPError = Exception
    RepositoryNotFoundError = Exception
    GatedRepoError = Exception


# =============================================================================
# Error Classes
# =============================================================================

class HFError(Exception):
    """Base exception for HuggingFace integration errors."""
    pass


class HFAuthError(HFError):
    """Authentication failed - invalid or missing token."""
    pass


class HFProRequiredError(HFError):
    """Operation requires HuggingFace Pro subscription."""
    pass


class HFQuotaExceededError(HFError):
    """HuggingFace quota exceeded (compute, storage, etc.)."""
    pass


class HFJobFailedError(HFError):
    """HuggingFace job (training, inference) failed."""

    def __init__(self, message: str, job_id: Optional[str] = None, logs: Optional[str] = None):
        super().__init__(message)
        self.job_id = job_id
        self.logs = logs


# =============================================================================
# User Info Dataclass
# =============================================================================

@dataclass
class HFUserInfo:
    """Information about the authenticated HuggingFace user."""
    username: str
    fullname: Optional[str] = None
    email: Optional[str] = None
    orgs: Optional[list] = None
    is_pro: bool = False
    can_pay: bool = False
    avatar_url: Optional[str] = None

    @classmethod
    def from_whoami(cls, data: Dict[str, Any]) -> "HFUserInfo":
        """Create from whoami() API response."""
        # Extract organization names
        orgs = []
        if "orgs" in data:
            orgs = [org.get("name", "") for org in data.get("orgs", [])]

        # Detect Pro status from various indicators
        is_pro = (
            data.get("isPro", False) or
            data.get("is_pro", False) or
            data.get("plan", "") in ("pro", "enterprise") or
            data.get("canPay", False)
        )

        return cls(
            username=data.get("name", data.get("username", "")),
            fullname=data.get("fullname"),
            email=data.get("email"),
            orgs=orgs,
            is_pro=is_pro,
            can_pay=data.get("canPay", False),
            avatar_url=data.get("avatarUrl"),
        )


# =============================================================================
# HuggingFace Client
# =============================================================================

class HuggingFaceClient:
    """
    Client for HuggingFace Hub API operations.

    Provides:
    - Authentication and token validation
    - Pro subscription detection
    - Access to HfApi for hub operations
    - Graceful degradation when library not installed

    Usage:
        client = HuggingFaceClient(token="hf_...")

        # Check if enabled
        if client.is_enabled:
            print(f"Logged in as {client.username}")
            if client.is_pro:
                print("Pro features available")

        # Require features
        client.require_enabled()  # Raises HFAuthError if not configured
        client.require_pro()      # Raises HFProRequiredError if not Pro
    """

    def __init__(
        self,
        token: Optional[str] = None,
        default_org: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace client.

        Args:
            token: HF API token. If None, tries to use cached token or HF_TOKEN env var.
            default_org: Default organization namespace for operations.
        """
        self._token = token
        self._default_org = default_org
        self._api: Optional[Any] = None
        self._user_info: Optional[HFUserInfo] = None
        self._initialized = False
        self._init_error: Optional[str] = None

        # Lazy initialization - don't call API until needed
        if not HF_HUB_AVAILABLE:
            self._init_error = "huggingface_hub library not installed"

    def _ensure_initialized(self) -> None:
        """Lazily initialize the client and fetch user info."""
        if self._initialized:
            return

        self._initialized = True

        if not HF_HUB_AVAILABLE:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return

        if not self._token:
            # Try to use cached token from huggingface-cli login
            try:
                from huggingface_hub import HfFolder
                self._token = HfFolder.get_token()
            except Exception:
                pass

        if not self._token:
            self._init_error = "No HuggingFace token provided"
            return

        try:
            self._api = HfApi(token=self._token)
            # Validate token and get user info
            whoami_data = self._api.whoami()
            self._user_info = HFUserInfo.from_whoami(whoami_data)
            logger.info(f"HuggingFace client initialized for user: {self._user_info.username}")
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Invalid" in error_msg or "Unauthorized" in error_msg:
                self._init_error = f"Invalid HuggingFace token: {error_msg}"
            else:
                self._init_error = f"Failed to initialize HuggingFace client: {error_msg}"
            logger.error(self._init_error)

    @property
    def api(self) -> Optional[Any]:
        """Get the HfApi instance (None if not available)."""
        self._ensure_initialized()
        return self._api

    @property
    def is_enabled(self) -> bool:
        """Check if the client is properly configured and authenticated."""
        self._ensure_initialized()
        return self._api is not None and self._user_info is not None

    @property
    def is_pro(self) -> bool:
        """Check if user has Pro subscription."""
        self._ensure_initialized()
        if self._user_info:
            return self._user_info.is_pro
        return False

    @property
    def username(self) -> Optional[str]:
        """Get the authenticated username."""
        self._ensure_initialized()
        if self._user_info:
            return self._user_info.username
        return None

    @property
    def organizations(self) -> list:
        """Get list of organizations the user belongs to."""
        self._ensure_initialized()
        if self._user_info and self._user_info.orgs:
            return self._user_info.orgs
        return []

    @property
    def default_org(self) -> Optional[str]:
        """Get the default organization for operations."""
        return self._default_org

    @property
    def namespace(self) -> str:
        """Get the default namespace (org or username)."""
        self._ensure_initialized()
        if self._default_org:
            return self._default_org
        if self._user_info:
            return self._user_info.username
        return ""

    @property
    def token(self) -> Optional[str]:
        """Get the API token (if set)."""
        return self._token

    @property
    def user_info(self) -> Optional[HFUserInfo]:
        """Get full user information."""
        self._ensure_initialized()
        return self._user_info

    def require_enabled(self) -> None:
        """
        Require that HuggingFace integration is enabled.

        Raises:
            HFAuthError: If not enabled (no token or invalid token).
        """
        self._ensure_initialized()
        if not self.is_enabled:
            error_msg = self._init_error or "HuggingFace integration not configured"
            raise HFAuthError(error_msg)

    def require_pro(self) -> None:
        """
        Require HuggingFace Pro subscription.

        Raises:
            HFAuthError: If not enabled.
            HFProRequiredError: If user doesn't have Pro subscription.
        """
        self.require_enabled()
        if not self.is_pro:
            raise HFProRequiredError(
                f"This operation requires HuggingFace Pro subscription. "
                f"User '{self.username}' does not have Pro enabled. "
                f"Upgrade at https://huggingface.co/subscribe/pro"
            )

    def get_repo_id(self, name: str, namespace: Optional[str] = None) -> str:
        """
        Construct a full repository ID.

        Args:
            name: Repository name (without namespace)
            namespace: Optional namespace override (default: self.namespace)

        Returns:
            Full repo ID like "namespace/name"
        """
        ns = namespace or self.namespace
        if "/" in name:
            return name  # Already has namespace
        if ns:
            return f"{ns}/{name}"
        return name

    def __repr__(self) -> str:
        """String representation."""
        if self.is_enabled:
            pro_str = " [Pro]" if self.is_pro else ""
            return f"<HuggingFaceClient user={self.username}{pro_str}>"
        elif self._init_error:
            return f"<HuggingFaceClient disabled: {self._init_error}>"
        else:
            return "<HuggingFaceClient disabled>"


# =============================================================================
# Singleton Access
# =============================================================================

_hf_client: Optional[HuggingFaceClient] = None


def get_hf_client(
    token: Optional[str] = None,
    default_org: Optional[str] = None,
    force_new: bool = False,
) -> HuggingFaceClient:
    """
    Get or create the global HuggingFace client instance.

    Args:
        token: HF API token. If provided, creates new client.
        default_org: Default organization namespace.
        force_new: Force creation of new client instance.

    Returns:
        HuggingFaceClient instance (singleton unless force_new=True)
    """
    global _hf_client

    if force_new or _hf_client is None or token is not None:
        # Get settings if no token provided
        if token is None:
            try:
                from bashgym.config import get_settings
                settings = get_settings()
                token = settings.huggingface.token
                if default_org is None:
                    default_org = settings.huggingface.default_org
            except Exception:
                pass  # Settings not available, client will try cached token

        _hf_client = HuggingFaceClient(token=token, default_org=default_org)

    return _hf_client


def reset_hf_client() -> None:
    """Reset the global client instance (for testing)."""
    global _hf_client
    _hf_client = None
