"""
Secrets management for Bash Gym.

Stores API tokens and secrets locally in ~/.bashgym/secrets.json
Environment variables always take precedence over stored secrets.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_secrets_path() -> Path:
    """Get the path to the secrets file."""
    from bashgym.config import get_bashgym_dir
    return get_bashgym_dir() / "secrets.json"


def load_secrets() -> Dict[str, Any]:
    """Load secrets from the secrets file."""
    secrets_path = get_secrets_path()
    if not secrets_path.exists():
        return {}

    try:
        with open(secrets_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load secrets: {e}")
        return {}


def save_secrets(secrets: Dict[str, Any]) -> None:
    """Save secrets to the secrets file."""
    secrets_path = get_secrets_path()
    secrets_path.parent.mkdir(parents=True, exist_ok=True)

    with open(secrets_path, "w") as f:
        json.dump(secrets, f, indent=2)

    # Set restrictive permissions on Unix
    try:
        import os
        import stat
        os.chmod(secrets_path, stat.S_IRUSR | stat.S_IWUSR)  # 600
    except (ImportError, OSError):
        pass  # Windows or permission error


def get_secret(key: str) -> Optional[str]:
    """
    Get a secret value.

    Environment variables take precedence over stored secrets.

    Args:
        key: The secret key (e.g., "HF_TOKEN")

    Returns:
        The secret value or None if not set
    """
    import os

    # Environment variable takes precedence
    env_value = os.environ.get(key)
    if env_value:
        return env_value

    # Check stored secrets
    secrets = load_secrets()
    return secrets.get(key)


def set_secret(key: str, value: str) -> None:
    """
    Store a secret value.

    Args:
        key: The secret key (e.g., "HF_TOKEN")
        value: The secret value
    """
    secrets = load_secrets()
    secrets[key] = value
    save_secrets(secrets)
    logger.info(f"Saved secret: {key}")


def delete_secret(key: str) -> bool:
    """
    Delete a stored secret.

    Args:
        key: The secret key to delete

    Returns:
        True if the secret was deleted, False if it didn't exist
    """
    secrets = load_secrets()
    if key in secrets:
        del secrets[key]
        save_secrets(secrets)
        logger.info(f"Deleted secret: {key}")
        return True
    return False


def has_secret(key: str) -> bool:
    """
    Check if a secret is configured (either env var or stored).

    Args:
        key: The secret key to check

    Returns:
        True if the secret is configured
    """
    return get_secret(key) is not None


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """
    Mask a secret value for display.

    Args:
        value: The secret value to mask
        visible_chars: Number of characters to show at the end

    Returns:
        Masked string like "***abc123"
    """
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]
