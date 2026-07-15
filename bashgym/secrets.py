"""Durable secret storage for BashGym.

Real environment values take precedence for operator-managed deployments.
Submitted secrets use the OS credential manager when available, with the
legacy ``~/.bashgym/secrets.json`` file retained only as a restricted fallback.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "BashGym"
KEYRING_DISABLED_ENV = "BASHGYM_DISABLE_KEYRING"
SecretSource = Literal["env", "keyring", "stored", ""]

_file_lock = threading.RLock()


def get_secrets_path() -> Path:
    """Return the legacy restricted-file secret path."""
    from bashgym.config import get_bashgym_dir

    return get_bashgym_dir() / "secrets.json"


def is_placeholder_secret(value: str | None) -> bool:
    """Return whether a configured value is a template rather than a credential."""
    if not value or not value.strip():
        return True
    normalized = value.strip().lower()
    return (
        normalized.startswith(("your-", "your_", "<your-", "<your_"))
        or normalized.endswith(("-here", "_here", "-placeholder", "_placeholder"))
        or normalized in {"changeme", "change-me", "placeholder", "replace-me"}
    )


def _normalized_secret(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return None if is_placeholder_secret(normalized) else normalized


def _get_keyring_module():
    """Return a usable keyring module, or None on headless/unsupported systems."""
    if os.environ.get(KEYRING_DISABLED_ENV, "").lower() in {"1", "true", "yes", "on"}:
        return None
    try:
        import keyring

        backend = keyring.get_keyring()
        if float(getattr(backend, "priority", 0)) <= 0:
            return None
        return keyring
    except Exception as exc:  # pragma: no cover - backend discovery is platform-specific
        logger.debug("OS credential manager unavailable: %s", exc.__class__.__name__)
        return None


def _get_keyring_secret(key: str) -> str | None:
    keyring = _get_keyring_module()
    if keyring is None:
        return None
    try:
        return _normalized_secret(keyring.get_password(KEYRING_SERVICE, key))
    except Exception as exc:  # pragma: no cover - backend failures are platform-specific
        logger.warning("Could not read %s from the OS credential manager: %s", key, exc)
        return None


def _set_keyring_secret(key: str, value: str) -> bool:
    keyring = _get_keyring_module()
    if keyring is None:
        return False
    try:
        keyring.set_password(KEYRING_SERVICE, key, value)
        return keyring.get_password(KEYRING_SERVICE, key) == value
    except Exception as exc:  # pragma: no cover - backend failures are platform-specific
        logger.warning("Could not store %s in the OS credential manager: %s", key, exc)
        return False


def _delete_keyring_secret(key: str) -> bool:
    keyring = _get_keyring_module()
    if keyring is None:
        return False
    try:
        if keyring.get_password(KEYRING_SERVICE, key) is None:
            return False
        keyring.delete_password(KEYRING_SERVICE, key)
        return True
    except Exception as exc:  # pragma: no cover - backend failures are platform-specific
        logger.warning("Could not delete %s from the OS credential manager: %s", key, exc)
        return False


def load_secrets() -> dict[str, Any]:
    """Load the legacy fallback file without consulting environment or keyring."""
    secrets_path = get_secrets_path()
    if not secrets_path.exists():
        return {}

    try:
        with secrets_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load legacy secrets: %s", exc)
        return {}


def save_secrets(secrets: dict[str, Any]) -> None:
    """Atomically save the restricted-file fallback used without an OS keyring."""
    secrets_path = get_secrets_path()
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = secrets_path.with_suffix(f"{secrets_path.suffix}.tmp")

    with _file_lock:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(secrets, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        os.replace(temporary_path, secrets_path)


def _remove_legacy_secret(key: str) -> bool:
    with _file_lock:
        secrets = load_secrets()
        if key not in secrets:
            return False
        del secrets[key]
        save_secrets(secrets)
        return True


def _get_persisted_secret(key: str) -> tuple[str | None, SecretSource]:
    legacy_value = _normalized_secret(load_secrets().get(key))
    if legacy_value:
        # A fallback write is newer than any stale keyring value. Migrate it only
        # after verifying the OS credential write, then remove the plaintext copy.
        if _set_keyring_secret(key, legacy_value):
            _remove_legacy_secret(key)
            return legacy_value, "keyring"
        return legacy_value, "stored"

    keyring_value = _get_keyring_secret(key)
    if keyring_value:
        return keyring_value, "keyring"
    return None, ""


def get_secret_with_source(key: str) -> tuple[str | None, SecretSource]:
    """Resolve a secret and its safe source label without exposing the value."""
    persisted_value, persisted_source = _get_persisted_secret(key)
    environment_value = _normalized_secret(os.environ.get(key))
    if environment_value:
        return environment_value, "env"
    return persisted_value, persisted_source


def get_secret(key: str) -> str | None:
    """Resolve a secret using real environment values, keyring, then fallback file."""
    value, _ = get_secret_with_source(key)
    return value


def set_secret(key: str, value: str) -> None:
    """Store a submitted secret in the OS credential manager when available."""
    normalized = _normalized_secret(value)
    if not normalized:
        raise ValueError(f"Refusing to store an empty or placeholder secret for {key}")

    if _set_keyring_secret(key, normalized):
        _remove_legacy_secret(key)
        logger.info("Saved secret in OS credential manager: %s", key)
        return

    with _file_lock:
        secrets = load_secrets()
        secrets[key] = normalized
        save_secrets(secrets)
    logger.warning("Saved %s to restricted fallback file; OS credential manager unavailable", key)


def delete_secret(key: str) -> bool:
    """Delete a submitted secret from both durable storage backends."""
    deleted_from_keyring = _delete_keyring_secret(key)
    deleted_from_file = _remove_legacy_secret(key)
    deleted = deleted_from_keyring or deleted_from_file
    if deleted:
        logger.info("Deleted secret: %s", key)
    return deleted


def has_secret(key: str) -> bool:
    """Return whether a real environment or stored secret is configured."""
    return get_secret(key) is not None


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """Mask a secret value for display."""
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]
