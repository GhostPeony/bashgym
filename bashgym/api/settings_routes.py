"""
Settings API routes for Bash Gym.

Provides REST endpoints for managing environment variables / API keys:
- GET /api/settings/env       - List all API keys with masked values
- PUT /api/settings/env       - Update API keys in .env file
- POST /api/settings/env/test - Test an API key against its provider
"""

import os
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


# =============================================================================
# Constants
# =============================================================================

# Allowlist of env var names that can be read/written through this API
ALLOWED_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "NVIDIA_API_KEY",
    "HF_TOKEN",
}

# Display metadata per provider key
PROVIDER_META: Dict[str, Dict[str, str]] = {
    "ANTHROPIC_API_KEY": {"display_name": "Anthropic", "env_key": "ANTHROPIC_API_KEY"},
    "OPENAI_API_KEY": {"display_name": "OpenAI", "env_key": "OPENAI_API_KEY"},
    "GOOGLE_API_KEY": {"display_name": "Google (Gemini)", "env_key": "GOOGLE_API_KEY"},
    "NVIDIA_API_KEY": {"display_name": "NVIDIA", "env_key": "NVIDIA_API_KEY"},
    "HF_TOKEN": {"display_name": "HuggingFace", "env_key": "HF_TOKEN"},
}

# Test URLs per provider
PROVIDER_TEST_URLS: Dict[str, Dict[str, str]] = {
    "ANTHROPIC_API_KEY": {
        "url": "https://api.anthropic.com/v1/models",
        "auth_type": "x-api-key",
    },
    "OPENAI_API_KEY": {
        "url": "https://api.openai.com/v1/models",
        "auth_type": "bearer",
    },
    "GOOGLE_API_KEY": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "auth_type": "query_key",
    },
    "NVIDIA_API_KEY": {
        "url": "https://integrate.api.nvidia.com/v1/models",
        "auth_type": "bearer",
    },
    "HF_TOKEN": {
        "url": "https://huggingface.co/api/whoami-v2",
        "auth_type": "bearer",
    },
}


# =============================================================================
# Schemas
# =============================================================================

class EnvKeyStatus(BaseModel):
    """Status of a single environment variable / API key."""
    key: str = Field(description="Environment variable name")
    display_name: str = Field(description="Human-readable provider name")
    is_set: bool = Field(description="Whether a non-placeholder value is present")
    masked_value: str = Field(default="", description="Last 4 chars of value, or empty")
    source: str = Field(default="", description="Where the value comes from: 'env_file', 'environment', or ''")


class EnvKeysResponse(BaseModel):
    """Response for GET /api/settings/env."""
    keys: List[EnvKeyStatus] = Field(description="Status of all managed API keys")


class EnvUpdateRequest(BaseModel):
    """Request for PUT /api/settings/env."""
    values: Dict[str, str] = Field(description="Map of env var name -> new value")


class EnvTestRequest(BaseModel):
    """Request for POST /api/settings/env/test."""
    key: str = Field(description="Environment variable name to test (e.g. ANTHROPIC_API_KEY)")
    value: Optional[str] = Field(default=None, description="Value to test. If omitted, uses current env value.")


class EnvTestResponse(BaseModel):
    """Response for POST /api/settings/env/test."""
    key: str
    valid: bool
    message: str = ""
    status_code: Optional[int] = None


# =============================================================================
# Helpers
# =============================================================================

def _get_project_env_path() -> Path:
    """Return the path to the project .env file."""
    # Walk up from this file to find the project root (where .env lives)
    # bashgym/api/settings_routes.py -> project root is 2 levels up
    here = Path(__file__).resolve().parent.parent.parent
    return here / ".env"


def _is_placeholder(value: str) -> bool:
    """Return True if the value looks like a placeholder rather than a real key."""
    if not value:
        return True
    v = value.strip()
    if v.startswith("your-") or v.startswith("your_"):
        return True
    if v.endswith("-here") or v.endswith("_here"):
        return True
    return False


def _mask_value(value: str) -> str:
    """Mask an API key, showing only the last 4 characters."""
    if not value or _is_placeholder(value):
        return ""
    if len(value) <= 4:
        return "****"
    return "****" + value[-4:]


def _read_env_file() -> Dict[str, str]:
    """Read key=value pairs from the .env file (ignoring comments/blanks)."""
    env_path = _get_project_env_path()
    result: Dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        key = key.strip()
        val = val.strip()
        result[key] = val
    return result


def _write_env_values(updates: Dict[str, str]) -> None:
    """Update values in the .env file, preserving comments and ordering.

    If a key already exists in the file its line is replaced in-place.
    If a key is new it is appended at the end.
    """
    env_path = _get_project_env_path()

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    remaining = dict(updates)  # keys we still need to write
    new_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, _ = stripped.partition("=")
            key = key.strip()
            if key in remaining:
                new_lines.append(f"{key}={remaining.pop(key)}")
                continue
        new_lines.append(line)

    # Append any keys that weren't already in the file
    for key, val in remaining.items():
        new_lines.append(f"{key}={val}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/env", response_model=EnvKeysResponse)
async def get_env_keys():
    """Return masked API key status for all managed providers.

    Reads from both the .env file and the current process environment.
    Placeholder values (e.g. ``your-xxx-here``) are treated as unset.
    """
    env_file_values = _read_env_file()
    statuses: List[EnvKeyStatus] = []

    for env_key in ALLOWED_ENV_KEYS:
        meta = PROVIDER_META[env_key]

        # Check .env file first, then os.environ
        file_val = env_file_values.get(env_key, "")
        env_val = os.environ.get(env_key, "")

        # Determine effective value and source
        if file_val and not _is_placeholder(file_val):
            effective = file_val
            source = "env_file"
        elif env_val and not _is_placeholder(env_val):
            effective = env_val
            source = "environment"
        else:
            effective = ""
            source = ""

        statuses.append(EnvKeyStatus(
            key=env_key,
            display_name=meta["display_name"],
            is_set=bool(effective),
            masked_value=_mask_value(effective),
            source=source,
        ))

    return EnvKeysResponse(keys=statuses)


@router.put("/env")
async def update_env_keys(request: EnvUpdateRequest):
    """Update API keys in the .env file and current process environment.

    Only keys in the allowlist may be written.
    """
    # Validate all keys are in the allowlist
    for key in request.values:
        if key not in ALLOWED_ENV_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"Key '{key}' is not in the allowlist. Allowed: {ALLOWED_ENV_KEYS}",
            )

    # Write to .env file (preserves comments/ordering)
    try:
        _write_env_values(request.values)
    except (IOError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to write .env file: {e}")

    # Also update os.environ for the running process
    for key, val in request.values.items():
        if val:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)

    return {"success": True, "updated": list(request.values.keys())}


@router.post("/env/test", response_model=EnvTestResponse)
async def test_env_key(request: EnvTestRequest):
    """Test an API key by calling the provider's API.

    Supports Anthropic, OpenAI, Google/Gemini, NVIDIA, and HuggingFace.
    """
    import httpx

    if request.key not in ALLOWED_ENV_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"Key '{request.key}' is not in the allowlist. Allowed: {ALLOWED_ENV_KEYS}",
        )

    # Determine the value to test
    value = request.value
    if not value:
        value = os.environ.get(request.key, "")
        if not value or _is_placeholder(value):
            # Also check .env file
            env_file_values = _read_env_file()
            value = env_file_values.get(request.key, "")

    if not value or _is_placeholder(value):
        return EnvTestResponse(
            key=request.key,
            valid=False,
            message="No API key value provided or configured.",
        )

    test_info = PROVIDER_TEST_URLS.get(request.key)
    if not test_info:
        return EnvTestResponse(
            key=request.key,
            valid=False,
            message=f"No test configuration for key '{request.key}'.",
        )

    url = test_info["url"]
    auth_type = test_info["auth_type"]

    headers: Dict[str, str] = {}
    params: Dict[str, str] = {}

    if auth_type == "x-api-key":
        headers["x-api-key"] = value
        # Anthropic also requires anthropic-version header
        headers["anthropic-version"] = "2023-06-01"
    elif auth_type == "bearer":
        headers["Authorization"] = f"Bearer {value}"
    elif auth_type == "query_key":
        params["key"] = value

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params)

        if resp.status_code in (200, 201):
            return EnvTestResponse(
                key=request.key,
                valid=True,
                message="API key is valid.",
                status_code=resp.status_code,
            )
        elif resp.status_code == 401:
            return EnvTestResponse(
                key=request.key,
                valid=False,
                message="Invalid API key (401 Unauthorized).",
                status_code=resp.status_code,
            )
        elif resp.status_code == 403:
            return EnvTestResponse(
                key=request.key,
                valid=False,
                message="API key lacks required permissions (403 Forbidden).",
                status_code=resp.status_code,
            )
        else:
            return EnvTestResponse(
                key=request.key,
                valid=False,
                message=f"Unexpected response: HTTP {resp.status_code}.",
                status_code=resp.status_code,
            )

    except httpx.TimeoutException:
        return EnvTestResponse(
            key=request.key,
            valid=False,
            message="Request timed out after 10 seconds.",
        )
    except httpx.ConnectError as exc:
        return EnvTestResponse(
            key=request.key,
            valid=False,
            message=f"Connection failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error testing API key %s", request.key)
        return EnvTestResponse(
            key=request.key,
            valid=False,
            message=f"Unexpected error: {exc}",
        )
