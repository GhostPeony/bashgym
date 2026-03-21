"""
Device Registry — JSON-backed storage for SSH training devices.

Devices are persisted to ~/.bashgym/devices.json. The registry is safe for
concurrent async use via an asyncio.Lock.
"""

import asyncio
import json
import os
import platform
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug (lowercase, hyphens, max 50 chars)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:50]


def _normalize_key_path(path_str: str) -> str:
    """Expand %USERPROFILE% and ~ in a key path; store with forward slashes."""
    if not path_str:
        return path_str
    # Expand Windows %USERPROFILE%
    path_str = os.path.expandvars(path_str)
    # Expand ~
    path_str = os.path.expanduser(path_str)
    # Normalise to forward slashes for cross-platform storage
    return Path(path_str).as_posix()


def _get_devices_path() -> Path:
    """Return the canonical path to devices.json (~/.bashgym/devices.json)."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("USERPROFILE", Path.home()))
    else:
        base = Path.home()
    return base / ".bashgym" / "devices.json"


# ---------------------------------------------------------------------------
# Device dataclass
# ---------------------------------------------------------------------------


@dataclass
class Device:
    """Represents a remote SSH training device."""

    id: str
    name: str
    host: str
    port: int
    username: str
    key_path: str
    work_dir: str
    is_default: bool
    added_at: str  # ISO-8601 UTC
    last_seen: str | None  # ISO-8601 UTC or None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# DeviceRegistry
# ---------------------------------------------------------------------------


class DeviceRegistry:
    """JSON-backed registry for SSH training devices.

    All mutating methods acquire ``self._lock`` to prevent concurrent
    read-modify-write races in an async context.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path: Path = path if path is not None else _get_devices_path()
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal sync I/O (called while the lock is already held)
    # ------------------------------------------------------------------

    def _load_sync(self) -> list[Device]:
        """Load devices from disk synchronously. Returns empty list if missing."""
        if not self._path.exists():
            return []
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            devices: list[Device] = []
            for item in data.get("devices", []):
                devices.append(Device(**item))
            return devices
        except (json.JSONDecodeError, TypeError, KeyError):
            return []

    def _save_sync(self, devices: list[Device]) -> None:
        """Persist devices to disk synchronously."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"devices": [d.to_dict() for d in devices]}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public async I/O
    # ------------------------------------------------------------------

    async def load(self) -> list[Device]:
        """Load and return all devices from storage."""
        async with self._lock:
            return self._load_sync()

    async def save(self, devices: list[Device]) -> None:
        """Persist the given device list to storage."""
        async with self._lock:
            self._save_sync(devices)

    # ------------------------------------------------------------------
    # Read helpers (no mutation — acquire lock for consistent reads)
    # ------------------------------------------------------------------

    async def list_devices(self) -> list[Device]:
        """Return all registered devices."""
        async with self._lock:
            return self._load_sync()

    async def get_device(self, device_id: str) -> Device | None:
        """Return the device with the given ID, or None."""
        async with self._lock:
            for device in self._load_sync():
                if device.id == device_id:
                    return device
            return None

    async def get_default(self) -> Device | None:
        """Return the default device, or None if none is set."""
        async with self._lock:
            for device in self._load_sync():
                if device.is_default:
                    return device
            return None

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    async def add_device(
        self,
        name: str,
        host: str,
        username: str,
        key_path: str = "",
        work_dir: str = "~/bashgym-training",
        port: int = 22,
        is_default: bool = False,
    ) -> Device:
        """Add a new device and return it.

        Raises ``ValueError`` if a device with the same host + username already
        exists.  The first device added is automatically set as the default.
        """
        async with self._lock:
            devices = self._load_sync()

            # Duplicate check
            for existing in devices:
                if existing.host == host and existing.username == username:
                    raise ValueError(
                        f"A device for {username}@{host} already exists " f"(id={existing.id})."
                    )

            device_id = f"{_slugify(name)}-{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()

            # First device is always the default
            if not devices:
                is_default = True

            # If this device is the new default, unset all others
            if is_default:
                for d in devices:
                    d.is_default = False

            new_device = Device(
                id=device_id,
                name=name,
                host=host,
                port=port,
                username=username,
                key_path=_normalize_key_path(key_path),
                work_dir=work_dir,
                is_default=is_default,
                added_at=now,
                last_seen=None,
                capabilities={},
            )
            devices.append(new_device)
            self._save_sync(devices)
            return new_device

    async def update_device(self, device_id: str, updates: dict[str, Any]) -> Device:
        """Apply a partial update to a device and return the updated version.

        Raises ``KeyError`` if the device is not found.
        """
        async with self._lock:
            devices = self._load_sync()
            for device in devices:
                if device.id == device_id:
                    for key, value in updates.items():
                        if key == "key_path":
                            value = _normalize_key_path(value)
                        if hasattr(device, key):
                            setattr(device, key, value)
                    self._save_sync(devices)
                    return device
            raise KeyError(f"Device '{device_id}' not found.")

    async def remove_device(self, device_id: str) -> None:
        """Remove a device by ID.

        Raises ``KeyError`` if the device is not found.
        """
        async with self._lock:
            devices = self._load_sync()
            original_len = len(devices)
            devices = [d for d in devices if d.id != device_id]
            if len(devices) == original_len:
                raise KeyError(f"Device '{device_id}' not found.")
            self._save_sync(devices)

    async def set_default(self, device_id: str) -> Device:
        """Make the given device the default.

        Raises ``KeyError`` if the device is not found.
        """
        async with self._lock:
            devices = self._load_sync()
            target: Device | None = None
            for device in devices:
                if device.id == device_id:
                    target = device
                    break
            if target is None:
                raise KeyError(f"Device '{device_id}' not found.")
            for device in devices:
                device.is_default = device.id == device_id
            self._save_sync(devices)
            return target

    async def update_capabilities(self, device_id: str, capabilities: dict[str, Any]) -> Device:
        """Update a device's capabilities and refresh last_seen timestamp.

        Raises ``KeyError`` if the device is not found.
        """
        async with self._lock:
            devices = self._load_sync()
            for device in devices:
                if device.id == device_id:
                    device.capabilities = capabilities
                    device.last_seen = datetime.now(timezone.utc).isoformat()
                    self._save_sync(devices)
                    return device
            raise KeyError(f"Device '{device_id}' not found.")

    # ------------------------------------------------------------------
    # Auto-import from environment variables
    # ------------------------------------------------------------------

    async def auto_import_from_env(self) -> Device | None:
        """If devices.json does not yet exist and SSH_REMOTE_ENABLED=true,
        import a device from the SSH_REMOTE_* environment variables.

        Returns the imported Device, or None if conditions are not met.
        """
        if self._path.exists():
            return None

        enabled = os.environ.get("SSH_REMOTE_ENABLED", "false").lower()
        if enabled not in ("true", "1", "yes", "on"):
            return None

        host = os.environ.get("SSH_REMOTE_HOST", "").strip()
        username = os.environ.get("SSH_REMOTE_USER", "").strip()

        if not host or not username:
            return None

        port_str = os.environ.get("SSH_REMOTE_PORT", "22")
        try:
            port = int(port_str)
        except ValueError:
            port = 22

        key_path = os.environ.get("SSH_REMOTE_KEY_PATH", "~/.ssh/id_rsa")
        work_dir = os.environ.get("SSH_REMOTE_WORK_DIR", "~/bashgym-training")
        name = os.environ.get("SSH_REMOTE_NAME", f"{username}@{host}")

        return await self.add_device(
            name=name,
            host=host,
            username=username,
            port=port,
            key_path=key_path,
            work_dir=work_dir,
            is_default=True,
        )
