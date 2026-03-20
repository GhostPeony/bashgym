# SSH Device Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded `.env`-based SSH config with a plug-and-play device manager — JSON-backed registry, SSH config auto-discovery, enhanced GPU preflight, and a frontend UI for managing remote training devices.

**Architecture:** Backend data layer (`device_registry.py`, `device_discovery.py`) handles storage and parsing. API routes (`device_routes.py`) expose CRUD + preflight. Frontend (`DeviceManager.tsx` + `deviceStore.ts`) provides the UI. Training integration resolves `device_id` to `SSHConfig` in the route handler, keeping the trainer decoupled.

**Tech Stack:** Python (FastAPI, Pydantic, asyncio, asyncssh), React (Zustand, TypeScript), JSON file storage

**Spec:** `docs/superpowers/specs/2026-03-20-ssh-device-management-design.md`

---

### Task 1: Device registry data layer

**Files:**
- Create: `bashgym/device_registry.py`

- [ ] **Step 1: Create `bashgym/device_registry.py` with DeviceRegistry class**

This is the pure data layer. No HTTP concerns. Create the file with:

```python
"""
Device Registry — JSON-backed storage for SSH training devices.

Stores device configurations at ~/.bashgym/devices.json.
Thread-safe via asyncio.Lock for concurrent API handler access.
"""

import json
import asyncio
import re
import uuid
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class Device:
    """A remote SSH training device."""
    id: str
    name: str
    host: str
    port: int = 22
    username: str = ""
    key_path: str = "~/.ssh/id_rsa"
    work_dir: str = "~/bashgym-training"
    is_default: bool = False
    added_at: str = ""
    last_seen: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text.strip('-')[:50]


def _normalize_key_path(path_str: str) -> str:
    """Normalize SSH key path for cross-platform storage."""
    # Expand %USERPROFILE% on Windows
    if '%USERPROFILE%' in path_str:
        path_str = path_str.replace('%USERPROFILE%', os.environ.get('USERPROFILE', '~'))
    # Expand ~ and normalize
    expanded = str(Path(path_str).expanduser())
    # Store with forward slashes
    return expanded.replace("\\", "/")


def _get_devices_path() -> Path:
    """Get path to devices.json."""
    import platform
    if platform.system() == 'Windows':
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym" / "devices.json"


class DeviceRegistry:
    """Manages the device registry with async-safe file access."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or _get_devices_path()
        self._lock = asyncio.Lock()

    async def load(self) -> List[Device]:
        """Load all devices from disk."""
        async with self._lock:
            return self._load_sync()

    def _load_sync(self) -> List[Device]:
        """Load devices (not async-safe, use inside lock)."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [Device(**d) for d in data]
        except (json.JSONDecodeError, TypeError):
            return []

    async def save(self, devices: List[Device]) -> None:
        """Save all devices to disk."""
        async with self._lock:
            self._save_sync(devices)

    def _save_sync(self, devices: List[Device]) -> None:
        """Save devices (not async-safe, use inside lock)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([d.to_dict() for d in devices], indent=2),
            encoding="utf-8"
        )

    async def list_devices(self) -> List[Device]:
        """List all devices."""
        return await self.load()

    async def get_device(self, device_id: str) -> Optional[Device]:
        """Get a device by ID."""
        devices = await self.load()
        return next((d for d in devices if d.id == device_id), None)

    async def get_default(self) -> Optional[Device]:
        """Get the default device."""
        devices = await self.load()
        return next((d for d in devices if d.is_default), None)

    async def add_device(
        self,
        name: str,
        host: str,
        username: str,
        port: int = 22,
        key_path: str = "~/.ssh/id_rsa",
        work_dir: str = "~/bashgym-training",
        is_default: bool = False,
    ) -> Device:
        """Add a new device. Rejects duplicates (same host+username)."""
        async with self._lock:
            devices = self._load_sync()

            # Check for duplicates
            for d in devices:
                if d.host == host and d.username == username:
                    raise ValueError(f"Device already exists: {username}@{host} (id: {d.id})")

            device_id = f"{_slugify(name)}-{uuid.uuid4().hex[:8]}"
            device = Device(
                id=device_id,
                name=name,
                host=host,
                port=port,
                username=username,
                key_path=_normalize_key_path(key_path),
                work_dir=work_dir,
                is_default=is_default,
                added_at=datetime.now(timezone.utc).isoformat(),
            )

            # If this is the first device or marked default, ensure only one default
            if is_default or not devices:
                for d in devices:
                    d.is_default = False
                device.is_default = True

            devices.append(device)
            self._save_sync(devices)
            return device

    async def update_device(self, device_id: str, updates: Dict[str, Any]) -> Optional[Device]:
        """Update a device's fields."""
        async with self._lock:
            devices = self._load_sync()
            device = next((d for d in devices if d.id == device_id), None)
            if not device:
                return None

            for key, value in updates.items():
                if value is not None and hasattr(device, key) and key not in ('id', 'added_at'):
                    if key == 'key_path':
                        value = _normalize_key_path(value)
                    setattr(device, key, value)

            self._save_sync(devices)
            return device

    async def remove_device(self, device_id: str) -> bool:
        """Remove a device by ID."""
        async with self._lock:
            devices = self._load_sync()
            before = len(devices)
            devices = [d for d in devices if d.id != device_id]
            if len(devices) == before:
                return False
            self._save_sync(devices)
            return True

    async def set_default(self, device_id: str) -> Optional[Device]:
        """Set a device as the default."""
        async with self._lock:
            devices = self._load_sync()
            target = None
            for d in devices:
                if d.id == device_id:
                    d.is_default = True
                    target = d
                else:
                    d.is_default = False
            if target:
                self._save_sync(devices)
            return target

    async def update_capabilities(
        self, device_id: str, capabilities: Dict[str, Any]
    ) -> Optional[Device]:
        """Update a device's capabilities after preflight."""
        async with self._lock:
            devices = self._load_sync()
            device = next((d for d in devices if d.id == device_id), None)
            if not device:
                return None
            device.capabilities = capabilities
            device.last_seen = datetime.now(timezone.utc).isoformat()
            self._save_sync(devices)
            return device

    async def auto_import_from_env(self) -> Optional[Device]:
        """Import device from SSH_REMOTE_* env vars if devices.json doesn't exist."""
        if self._path.exists():
            return None

        from bashgym.config import get_settings
        settings = get_settings()
        if not settings.ssh.enabled or not settings.ssh.host:
            return None

        return await self.add_device(
            name=f"SSH ({settings.ssh.host})",
            host=settings.ssh.host,
            username=settings.ssh.username,
            port=settings.ssh.port,
            key_path=settings.ssh.key_path,
            work_dir=settings.ssh.remote_work_dir,
            is_default=True,
        )
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from bashgym.device_registry import DeviceRegistry, Device; print('Registry OK')"`
Expected: `Registry OK`

- [ ] **Step 3: Commit**

```bash
git add bashgym/device_registry.py
git commit -m "feat: add device registry data layer with JSON-backed storage

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: SSH config discovery

**Files:**
- Create: `bashgym/device_discovery.py`

- [ ] **Step 1: Create `bashgym/device_discovery.py`**

```python
"""
SSH Config Discovery — parses ~/.ssh/config to find candidate devices.

Handles Host blocks, HostName, User, Port, IdentityFile directives.
Supports one level of Include directives.
Filters out wildcards, localhost, and known code hosting services.
"""

import os
import glob
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


# Hosts to filter out (code hosting, not GPU devices)
FILTERED_HOSTS = {
    'github.com', 'gitlab.com', 'bitbucket.org', 'ssh.dev.azure.com',
    'vs-ssh.visualstudio.com', 'localhost', '127.0.0.1', '::1',
}


@dataclass
class SSHHostEntry:
    """A parsed SSH config host entry."""
    alias: str
    hostname: Optional[str] = None
    user: Optional[str] = None
    port: int = 22
    identity_file: Optional[str] = None


@dataclass
class SSHCandidate:
    """A candidate device from SSH config."""
    ssh_alias: str
    host: str
    username: Optional[str] = None
    port: int = 22
    key_path: Optional[str] = None
    already_added: bool = False
    existing_device_id: Optional[str] = None


def get_ssh_config_path() -> Path:
    """Get the platform-appropriate SSH config path."""
    if platform.system() == 'Windows':
        base = Path(os.environ.get("USERPROFILE", os.path.expanduser("~")))
    else:
        base = Path.home()
    return base / ".ssh" / "config"


def _parse_ssh_config_file(path: Path) -> List[SSHHostEntry]:
    """Parse a single SSH config file into host entries."""
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return []

    entries: List[SSHHostEntry] = []
    current: Optional[SSHHostEntry] = None

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Split on first whitespace or =
        parts = line.split(None, 1)
        if len(parts) < 2:
            parts = line.split('=', 1)
        if len(parts) < 2:
            continue

        key = parts[0].lower()
        value = parts[1].strip().strip('"')

        if key == 'host':
            # New host block — save previous
            if current:
                entries.append(current)
            # Skip wildcard entries
            if '*' in value or '?' in value:
                current = None
            else:
                current = SSHHostEntry(alias=value)
        elif current is not None:
            if key == 'hostname':
                current.hostname = value
            elif key == 'user':
                current.user = value
            elif key == 'port':
                try:
                    current.port = int(value)
                except ValueError:
                    pass
            elif key == 'identityfile':
                # Block-scoped: only set if not already set in this block
                if current.identity_file is None:
                    current.identity_file = value

    # Don't forget the last entry
    if current:
        entries.append(current)

    return entries


def _resolve_includes(config_path: Path) -> List[SSHHostEntry]:
    """Parse SSH config with one level of Include support."""
    entries: List[SSHHostEntry] = []

    if not config_path.exists():
        return entries

    try:
        content = config_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return entries

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith('include '):
            pattern = stripped.split(None, 1)[1].strip().strip('"')
            # Expand ~ in include path
            pattern = str(Path(pattern).expanduser())
            for included_path in sorted(glob.glob(pattern)):
                entries.extend(_parse_ssh_config_file(Path(included_path)))

    # Parse the main config file itself
    entries.extend(_parse_ssh_config_file(config_path))

    return entries


def _is_filtered(entry: SSHHostEntry) -> bool:
    """Check if an entry should be filtered out."""
    host = (entry.hostname or entry.alias).lower()
    return host in FILTERED_HOSTS


def discover_ssh_devices(
    existing_devices: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Discover SSH devices from ~/.ssh/config.

    Args:
        existing_devices: List of existing device dicts to cross-reference

    Returns:
        Dict with 'candidates' list and 'ssh_config_path'
    """
    config_path = get_ssh_config_path()
    existing = existing_devices or []

    # Build lookup for existing devices by host+username
    existing_lookup = {}
    for dev in existing:
        key = f"{dev.get('host', '')}:{dev.get('username', '')}"
        existing_lookup[key] = dev.get('id')

    entries = _resolve_includes(config_path)

    candidates = []
    for entry in entries:
        if _is_filtered(entry):
            continue

        host = entry.hostname or entry.alias
        username = entry.user
        lookup_key = f"{host}:{username or ''}"

        existing_id = existing_lookup.get(lookup_key)

        candidates.append(SSHCandidate(
            ssh_alias=entry.alias,
            host=host,
            username=username,
            port=entry.port,
            key_path=entry.identity_file,
            already_added=existing_id is not None,
            existing_device_id=existing_id,
        ))

    return {
        "candidates": [vars(c) for c in candidates],
        "ssh_config_path": str(config_path),
    }
```

- [ ] **Step 2: Verify it imports and discovers from your SSH config**

Run: `python -c "from bashgym.device_discovery import discover_ssh_devices; result = discover_ssh_devices(); print(f'Found {len(result[\"candidates\"])} candidates'); print(result['ssh_config_path'])"`

- [ ] **Step 3: Commit**

```bash
git add bashgym/device_discovery.py
git commit -m "feat: add SSH config discovery parser with Include support

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Device schemas

**Files:**
- Create: `bashgym/api/device_schemas.py`
- Modify: `bashgym/api/schemas.py:125-168` (add device_id to TrainingRequest)

- [ ] **Step 1: Create `bashgym/api/device_schemas.py`**

```python
"""Pydantic schemas for device management API."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class DeviceCreate(BaseModel):
    """Request to add a new device."""
    name: str = Field(..., min_length=1, max_length=100)
    host: str = Field(..., min_length=1, description="Hostname or IP address")
    port: int = Field(22, ge=1, le=65535)
    username: str = Field(..., min_length=1)
    key_path: str = Field("~/.ssh/id_rsa", description="Path to SSH private key")
    work_dir: str = Field("~/bashgym-training", description="Remote working directory")


class DeviceUpdate(BaseModel):
    """Request to update a device."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    host: Optional[str] = Field(None, min_length=1)
    port: Optional[int] = Field(None, ge=1, le=65535)
    username: Optional[str] = Field(None, min_length=1)
    key_path: Optional[str] = None
    work_dir: Optional[str] = None


class DeviceResponse(BaseModel):
    """Response for a device."""
    id: str
    name: str
    host: str
    port: int
    username: str
    key_path: str
    work_dir: str
    is_default: bool
    added_at: str
    last_seen: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class SSHCandidateResponse(BaseModel):
    """A candidate device from SSH config discovery."""
    ssh_alias: str
    host: str
    username: Optional[str] = None
    port: int = 22
    key_path: Optional[str] = None
    already_added: bool = False
    existing_device_id: Optional[str] = None


class DiscoverResponse(BaseModel):
    """Response from SSH config discovery."""
    candidates: List[SSHCandidateResponse]
    ssh_config_path: str
```

- [ ] **Step 2: Add `device_id` to `TrainingRequest` in `bashgym/api/schemas.py`**

After the `use_remote_ssh` field (line 157), add:

```python
    device_id: Optional[str] = Field(None, description="Target device ID for remote SSH training (uses default if omitted)")
```

- [ ] **Step 3: Verify schemas compile**

Run: `python -c "from bashgym.api.device_schemas import DeviceCreate, DiscoverResponse; from bashgym.api.schemas import TrainingRequest; r = TrainingRequest(device_id='test-123'); print(r.device_id); print('Schemas OK')"`
Expected: `test-123` then `Schemas OK`

- [ ] **Step 4: Commit**

```bash
git add bashgym/api/device_schemas.py bashgym/api/schemas.py
git commit -m "feat: add device management schemas and device_id to TrainingRequest

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Enhanced preflight with GPU detection

**Files:**
- Modify: `bashgym/gym/remote_trainer.py:45-51` (PreflightResult dataclass)
- Modify: `bashgym/gym/remote_trainer.py:71-110` (preflight_check method)

- [ ] **Step 1: Extend PreflightResult dataclass**

In `bashgym/gym/remote_trainer.py`, find the `PreflightResult` dataclass (line 45). Add new fields after `error`:

```python
@dataclass
class PreflightResult:
    """Result of pre-flight checks on the remote machine."""
    ok: bool
    python_version: Optional[str] = None
    disk_free_gb: Optional[float] = None
    error: Optional[str] = None
    # Enhanced device info
    hostname: Optional[str] = None
    os_info: Optional[str] = None
    cuda_version: Optional[str] = None
    gpus: Optional[List[Dict[str, Any]]] = None
```

Add `from typing import List, Dict, Any` to the imports if not already present.

- [ ] **Step 2: Add GPU/system commands to `preflight_check()`**

Find the `preflight_check` method. After the existing disk space check (around line 104) and before the return statement, add these additional SSH commands:

```python
            # Enhanced device info (non-fatal — missing nvidia-smi just means no GPU info)
            try:
                hostname_result = await conn.run('hostname', timeout=5)
                result.hostname = hostname_result.stdout.strip() if hostname_result.exit_status == 0 else None
            except Exception:
                pass

            try:
                os_result = await conn.run('uname -sr', timeout=5)
                result.os_info = os_result.stdout.strip() if os_result.exit_status == 0 else None
            except Exception:
                pass

            try:
                # GPU info via nvidia-smi
                gpu_csv = await conn.run(
                    'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits',
                    timeout=10
                )
                if gpu_csv.exit_status == 0 and gpu_csv.stdout.strip():
                    gpus = []
                    for line in gpu_csv.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            gpus.append({
                                "name": parts[0],
                                "vram_total_gb": round(float(parts[1]) / 1024, 1),
                                "vram_free_gb": round(float(parts[2]) / 1024, 1),
                            })
                    result.gpus = gpus if gpus else None

                # CUDA version from nvidia-smi header
                cuda_header = await conn.run('nvidia-smi | head -3', timeout=10)
                if cuda_header.exit_status == 0:
                    import re
                    cuda_match = re.search(r'CUDA Version:\s*([\d.]+)', cuda_header.stdout)
                    if cuda_match:
                        result.cuda_version = cuda_match.group(1)
            except Exception:
                pass  # No GPU / nvidia-smi not available — not a failure
```

- [ ] **Step 3: Add connection timeout**

Find the `_connect()` method. In the `asyncssh.connect()` call, add a `connect_timeout=10` parameter if not already present. If the method doesn't directly expose this, wrap the connect call with `asyncio.wait_for(asyncssh.connect(...), timeout=10)`.

- [ ] **Step 4: Verify it compiles**

Run: `python -c "from bashgym.gym.remote_trainer import PreflightResult; r = PreflightResult(ok=True, hostname='test', gpus=[{'name': 'A100'}]); print(r.hostname, r.gpus); print('Preflight OK')"`
Expected: `test [{'name': 'A100'}]` then `Preflight OK`

- [ ] **Step 5: Commit**

```bash
git add bashgym/gym/remote_trainer.py
git commit -m "feat: enhance preflight with GPU, CUDA, hostname, OS detection

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Device API routes

**Files:**
- Create: `bashgym/api/device_routes.py`

- [ ] **Step 1: Create `bashgym/api/device_routes.py`**

```python
"""Device management API routes."""

import logging
from fastapi import APIRouter, HTTPException

from bashgym.device_registry import DeviceRegistry
from bashgym.device_discovery import discover_ssh_devices
from bashgym.api.device_schemas import (
    DeviceCreate, DeviceUpdate, DeviceResponse,
    DiscoverResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/devices", tags=["Devices"])

# Singleton registry instance
_registry = DeviceRegistry()


def get_registry() -> DeviceRegistry:
    return _registry


@router.get("", response_model=list[DeviceResponse])
async def list_devices():
    """List all configured devices."""
    devices = await _registry.list_devices()
    return [d.to_dict() for d in devices]


@router.post("", response_model=DeviceResponse)
async def add_device(request: DeviceCreate):
    """Add a new device. Rejects duplicates (same host+username)."""
    try:
        device = await _registry.add_device(
            name=request.name,
            host=request.host,
            port=request.port,
            username=request.username,
            key_path=request.key_path,
            work_dir=request.work_dir,
        )
        return device.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.put("/{device_id}", response_model=DeviceResponse)
async def update_device(device_id: str, request: DeviceUpdate):
    """Update a device."""
    updates = request.dict(exclude_none=True)
    device = await _registry.update_device(device_id, updates)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
    return device.to_dict()


@router.delete("/{device_id}")
async def remove_device(device_id: str):
    """Remove a device."""
    removed = await _registry.remove_device(device_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
    return {"ok": True}


@router.post("/{device_id}/set-default", response_model=DeviceResponse)
async def set_default_device(device_id: str):
    """Set a device as the default training target."""
    device = await _registry.set_default(device_id)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")
    return device.to_dict()


@router.post("/{device_id}/preflight")
async def device_preflight(device_id: str):
    """Run preflight checks on a device (connection, Python, GPU, disk)."""
    device = await _registry.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail=f"Device not found: {device_id}")

    try:
        from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig

        ssh_config = SSHConfig(
            host=device.host,
            port=device.port,
            username=device.username,
            key_path=device.key_path,
            remote_work_dir=device.work_dir,
        )
        trainer = RemoteTrainer(ssh_config)
        result = await trainer.preflight_check()

        # Update device capabilities in registry
        if result.ok:
            capabilities = {
                "python_version": result.python_version,
                "disk_free_gb": result.disk_free_gb,
                "hostname": result.hostname,
                "os": result.os_info,
                "cuda_version": result.cuda_version,
                "gpus": result.gpus,
            }
            updated = await _registry.update_capabilities(device_id, capabilities)
            if updated:
                device = updated

        return {
            "ok": result.ok,
            "python_version": result.python_version,
            "disk_free_gb": result.disk_free_gb,
            "hostname": result.hostname,
            "os_info": result.os_info,
            "cuda_version": result.cuda_version,
            "gpus": result.gpus,
            "error": result.error,
            "device": device.to_dict(),
        }
    except Exception as e:
        logger.error(f"Preflight failed for {device_id}: {e}")
        return {
            "ok": False,
            "error": str(e),
            "device": device.to_dict(),
        }


@router.post("/discover", response_model=DiscoverResponse)
async def discover_devices():
    """Discover SSH devices from ~/.ssh/config."""
    devices = await _registry.list_devices()
    existing = [d.to_dict() for d in devices]
    result = discover_ssh_devices(existing_devices=existing)
    return result
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from bashgym.api.device_routes import router; print(f'{len(router.routes)} routes'); print('Device routes OK')"`

- [ ] **Step 3: Commit**

```bash
git add bashgym/api/device_routes.py
git commit -m "feat: add device management API routes (CRUD, preflight, discover)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Register device router and auto-import

**Files:**
- Modify: `bashgym/api/routes.py:61-72` (router imports)
- Modify: `bashgym/api/routes.py:628-647` (ssh preflight backward compat)

- [ ] **Step 1: Add device router import**

In `bashgym/api/routes.py`, after the existing router imports (around line 72), add:

```python
from bashgym.api.device_routes import router as device_router, get_registry as get_device_registry
```

- [ ] **Step 2: Register the device router in `create_app()`**

Find where other routers are registered with `app.include_router()` (around line 4091-4124). Add:

```python
    app.include_router(device_router)
```

- [ ] **Step 3: Add auto-import on startup**

Find the startup event handler (search for `@app.on_event("startup")` or `lifespan`). Add auto-import logic:

```python
    # Auto-import SSH device from .env on first run
    try:
        registry = get_device_registry()
        imported = await registry.auto_import_from_env()
        if imported:
            logger.info(f"Auto-imported SSH device from .env: {imported.name} ({imported.host})")
    except Exception as e:
        logger.warning(f"Failed to auto-import SSH device: {e}")
```

- [ ] **Step 4: Update the old `/api/ssh/preflight` endpoint**

Find the existing ssh_preflight endpoint (line 628). Update it to use the device registry as primary source, falling back to env vars:

```python
@app.get("/api/ssh/preflight", tags=["SSH"])
async def ssh_preflight():
    """Run pre-flight checks on the default remote training device.

    Backward-compatible endpoint. Prefers device registry, falls back to env vars.
    """
    # Try device registry first
    try:
        registry = get_device_registry()
        default_device = await registry.get_default()
        if default_device:
            from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig
            ssh_config = SSHConfig(
                host=default_device.host,
                port=default_device.port,
                username=default_device.username,
                key_path=default_device.key_path,
                remote_work_dir=default_device.work_dir,
            )
            trainer = RemoteTrainer(ssh_config)
            result = await trainer.preflight_check()
            return {
                "ok": result.ok,
                "python_version": result.python_version,
                "disk_free_gb": result.disk_free_gb,
                "error": result.error,
                "host": default_device.host,
                "username": default_device.username,
                "hostname": result.hostname,
                "os_info": result.os_info,
                "cuda_version": result.cuda_version,
                "gpus": result.gpus,
            }
    except Exception as e:
        logger.warning(f"Device registry preflight failed, falling back to env: {e}")

    # Fallback to env vars
    from bashgym.config import get_settings as _get_settings
    _s = _get_settings()
    if not _s.ssh.enabled:
        return {"ok": False, "error": "SSH remote training not enabled. Set SSH_REMOTE_ENABLED=true or add a device."}

    from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig
    config = SSHConfig.from_settings(_s.ssh)
    trainer = RemoteTrainer(config)
    result = await trainer.preflight_check()
    return {
        "ok": result.ok,
        "python_version": result.python_version,
        "disk_free_gb": result.disk_free_gb,
        "error": result.error,
        "host": _s.ssh.host,
        "username": _s.ssh.username,
    }
```

- [ ] **Step 5: Verify app creates**

Run: `python -c "from bashgym.api.routes import create_app; app = create_app(); print(f'App created with {len(app.routes)} routes')"`

- [ ] **Step 6: Commit**

```bash
git add bashgym/api/routes.py
git commit -m "feat: register device router, add auto-import, update ssh preflight

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Training integration (device_id → SSHConfig)

**Files:**
- Modify: `bashgym/api/routes.py:872-903` (training config construction)
- Modify: `bashgym/gym/trainer.py:1282-1310` (_train_with_remote_ssh)

- [ ] **Step 1: Resolve device_id in the training route handler**

In `bashgym/api/routes.py`, find the training start handler's config construction block (around line 872-903). After the config is set on `app.state.trainer` (line 904: `app.state.trainer.config = config`), add device resolution:

```python
                # Resolve device for remote SSH training
                if config.use_remote_ssh:
                    ssh_config = None
                    device_id_to_use = getattr(request, 'device_id', None)

                    try:
                        registry = get_device_registry()
                        if device_id_to_use:
                            device = await registry.get_device(device_id_to_use)
                        else:
                            device = await registry.get_default()

                        if device:
                            from bashgym.gym.remote_trainer import SSHConfig
                            ssh_config = SSHConfig(
                                host=device.host,
                                port=device.port,
                                username=device.username,
                                key_path=device.key_path,
                                remote_work_dir=device.work_dir,
                            )
                    except Exception as e:
                        logger.warning(f"Device resolution failed, falling back to env: {e}")

                    app.state.trainer.ssh_config = ssh_config  # May be None (falls back to env)
                else:
                    app.state.trainer.ssh_config = None
```

- [ ] **Step 2: Update `_train_with_remote_ssh()` in trainer.py**

In `bashgym/gym/trainer.py`, find `_train_with_remote_ssh` (line 1282). Update it to use `self.ssh_config` if set:

Replace the SSHConfig construction lines (1287-1291):

```python
        from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig

        # Use pre-resolved ssh_config from route handler, or fall back to env vars
        ssh_config = getattr(self, 'ssh_config', None)
        if ssh_config is None:
            from bashgym.config import get_settings
            settings = get_settings()
            ssh_config = SSHConfig.from_settings(settings.ssh)

        trainer = RemoteTrainer(ssh_config)
```

- [ ] **Step 3: Verify it compiles**

Run: `python -c "from bashgym.gym.trainer import Trainer; t = Trainer(); print('Trainer OK')"`
Run: `python -c "from bashgym.api.routes import create_app; app = create_app(); print('Routes OK')"`

- [ ] **Step 4: Commit**

```bash
git add bashgym/api/routes.py bashgym/gym/trainer.py
git commit -m "feat: resolve device_id to SSHConfig in route handler, trainer accepts pre-resolved config

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Frontend API client

**Files:**
- Modify: `frontend/src/services/api.ts`

- [ ] **Step 1: Add device types and API client**

In `frontend/src/services/api.ts`, add the device-related types and API object. Find the existing `sshApi` object and add the new `deviceApi` after it:

```typescript
// Device management types
export interface DeviceCapabilities {
  python_version?: string
  cuda_version?: string
  gpus?: Array<{ name: string; vram_total_gb: number; vram_free_gb: number }>
  disk_free_gb?: number
  hostname?: string
  os?: string
}

export interface Device {
  id: string
  name: string
  host: string
  port: number
  username: string
  key_path: string
  work_dir: string
  is_default: boolean
  added_at: string
  last_seen?: string
  capabilities?: DeviceCapabilities
}

export interface NewDevice {
  name: string
  host: string
  port?: number
  username: string
  key_path?: string
  work_dir?: string
}

export interface SSHCandidate {
  ssh_alias: string
  host: string
  username?: string
  port: number
  key_path?: string
  already_added: boolean
  existing_device_id?: string
}

export interface DiscoverResult {
  candidates: SSHCandidate[]
  ssh_config_path: string
}

export interface PreflightResult {
  ok: boolean
  python_version?: string
  disk_free_gb?: number
  hostname?: string
  os_info?: string
  cuda_version?: string
  gpus?: Array<{ name: string; vram_total_gb: number; vram_free_gb: number }>
  error?: string
  device?: Device
}

export const deviceApi = {
  list: () => request<Device[]>('/devices'),
  add: (device: NewDevice) => request<Device>('/devices', { method: 'POST', body: JSON.stringify(device), headers: { 'Content-Type': 'application/json' } }),
  update: (id: string, device: Partial<NewDevice>) => request<Device>(`/devices/${id}`, { method: 'PUT', body: JSON.stringify(device), headers: { 'Content-Type': 'application/json' } }),
  remove: (id: string) => request<{ ok: boolean }>(`/devices/${id}`, { method: 'DELETE' }),
  preflight: (id: string) => request<PreflightResult>(`/devices/${id}/preflight`, { method: 'POST' }),
  setDefault: (id: string) => request<Device>(`/devices/${id}/set-default`, { method: 'POST' }),
  discover: () => request<DiscoverResult>('/devices/discover', { method: 'POST' }),
}
```

Also add `device_id?: string` to the existing `TrainingRequest` interface (find it near line 27).

- [ ] **Step 2: Verify frontend compiles**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -20`
(Some existing errors may appear — just verify no NEW errors from the types you added)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add device management API client and types

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Device store

**Files:**
- Create: `frontend/src/stores/deviceStore.ts`

- [ ] **Step 1: Create `frontend/src/stores/deviceStore.ts`**

Follow the existing store pattern (see `trainingStore.ts`). Create:

```typescript
import { create } from 'zustand'
import { deviceApi, Device, NewDevice, SSHCandidate, PreflightResult } from '../services/api'

interface DeviceStore {
  devices: Device[]
  defaultDeviceId: string | null
  loading: boolean
  error: string | null

  fetchDevices: () => Promise<void>
  addDevice: (device: NewDevice) => Promise<Device | null>
  updateDevice: (id: string, updates: Partial<NewDevice>) => Promise<void>
  removeDevice: (id: string) => Promise<void>
  runPreflight: (id: string) => Promise<PreflightResult | null>
  setDefault: (id: string) => Promise<void>
  discoverFromSSHConfig: () => Promise<SSHCandidate[]>
}

export const useDeviceStore = create<DeviceStore>((set, get) => ({
  devices: [],
  defaultDeviceId: null,
  loading: false,
  error: null,

  fetchDevices: async () => {
    set({ loading: true, error: null })
    try {
      const devices = await deviceApi.list()
      const defaultDevice = devices.find(d => d.is_default)
      set({ devices, defaultDeviceId: defaultDevice?.id || null, loading: false })
    } catch (e: any) {
      set({ error: e.message || 'Failed to fetch devices', loading: false })
    }
  },

  addDevice: async (device: NewDevice) => {
    try {
      const added = await deviceApi.add(device)
      await get().fetchDevices()
      return added
    } catch (e: any) {
      set({ error: e.message || 'Failed to add device' })
      return null
    }
  },

  updateDevice: async (id: string, updates: Partial<NewDevice>) => {
    try {
      await deviceApi.update(id, updates)
      await get().fetchDevices()
    } catch (e: any) {
      set({ error: e.message || 'Failed to update device' })
    }
  },

  removeDevice: async (id: string) => {
    try {
      await deviceApi.remove(id)
      await get().fetchDevices()
    } catch (e: any) {
      set({ error: e.message || 'Failed to remove device' })
    }
  },

  runPreflight: async (id: string) => {
    try {
      const result = await deviceApi.preflight(id)
      // Refresh devices to get updated capabilities
      await get().fetchDevices()
      return result
    } catch (e: any) {
      set({ error: e.message || 'Preflight failed' })
      return null
    }
  },

  setDefault: async (id: string) => {
    try {
      await deviceApi.setDefault(id)
      await get().fetchDevices()
    } catch (e: any) {
      set({ error: e.message || 'Failed to set default' })
    }
  },

  discoverFromSSHConfig: async () => {
    try {
      const result = await deviceApi.discover()
      return result.candidates
    } catch (e: any) {
      set({ error: e.message || 'Discovery failed' })
      return []
    }
  },
}))
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/stores/deviceStore.ts
git commit -m "feat: add device management Zustand store

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: DeviceManager component

**Files:**
- Create: `frontend/src/components/training/DeviceManager.tsx`

- [ ] **Step 1: Create `frontend/src/components/training/DeviceManager.tsx`**

This is the main UI component. It shows device cards, add form, and discover button. Follow the botanical brutalist design system (see CLAUDE.md for design tokens).

The component should:

1. **Import and use** the `useDeviceStore` for all device operations
2. **Fetch devices on mount** via `useEffect(() => { fetchDevices() }, [])`
3. **Show device list** as cards with brutalist styling (2px borders, offset shadows, monospace labels)
4. **Each card shows:**
   - Device name + host in serif header
   - GPU info or "No GPU info — run Test Connection" in muted text
   - Disk free, CUDA version as mono labels
   - Status dot (green if `last_seen` within 1 hour, yellow if older, gray if never tested)
   - Actions row: Test Connection button, Set Default button (if not already), Edit, Remove
5. **"Add Device" button** toggles an inline form with fields: Name, Host, Username, Port, Key Path, Work Dir
6. **"Discover from SSH Config" button** calls `discoverFromSSHConfig()`, shows candidates as a list with "Add" buttons
7. **Empty state** shows "No remote devices configured" with both add and discover buttons prominent
8. **Loading/testing states** show spinner during preflight

Key UI details:
- Use `font-mono text-xs uppercase tracking-widest` for labels (brutalist)
- Use `border-2 border-text-primary` with `shadow-[3px_3px_0_0]` for cards
- Use accent colors for the default device border
- "Test Connection" button shows inline result (checkmark or error) after completion

This component will be imported and used by TrainingConfig in Task 11.

The exact implementation should follow existing component patterns in the codebase (check `SystemInfoPanel.tsx` or `TrainingConfig.tsx` for styling patterns). The component is self-contained — all state management goes through the store.

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit 2>&1 | grep -i "DeviceManager" | head -5`
Should have no errors specific to DeviceManager.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/training/DeviceManager.tsx
git commit -m "feat: add DeviceManager component with cards, add form, and SSH discovery

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: TrainingConfig integration

**Files:**
- Modify: `frontend/src/components/training/TrainingConfig.tsx:343-400` (backend selector)
- Modify: `frontend/src/stores/trainingStore.ts` (add deviceId)

- [ ] **Step 1: Add `deviceId` to trainingStore config**

In `frontend/src/stores/trainingStore.ts`, find the `TrainingConfig` interface (around line 21). Add:

```typescript
  deviceId?: string
```

- [ ] **Step 2: Replace the backend selector in TrainingConfig.tsx**

In `frontend/src/components/training/TrainingConfig.tsx`, find the backend selector section (lines 343-400). The three buttons (Local / DGX Spark / NeMo Cloud) need to change:

1. Import DeviceManager: `import { DeviceManager } from './DeviceManager'` (adjust import path based on file location)
2. Import useDeviceStore: `import { useDeviceStore } from '../../stores/deviceStore'`
3. Replace the "DGX Spark" button with a "Remote Device" button that, when selected, shows the DeviceManager component below the button grid
4. When a device is selected in DeviceManager or already has a default, set `config.deviceId` and `config.useRemoteSSH = true`

The backend selector becomes:
- **Local** — same as now
- **Remote Device** — replaces "DGX Spark". When selected, shows DeviceManager below. Sets `useRemoteSSH: true` and `deviceId: defaultDeviceId`
- **NeMo Cloud** — same as now

When `trainingBackend === 'remote_ssh'`, render `<DeviceManager />` below the button grid. The DeviceManager handles its own state; the training config just needs to read `defaultDeviceId` from `useDeviceStore` and set it on the config.

- [ ] **Step 3: Verify frontend compiles and renders**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -10`
Then start dev: `cd frontend && npm run dev` — verify the Training Config page loads without errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/training/TrainingConfig.tsx frontend/src/stores/trainingStore.ts
git commit -m "feat: integrate DeviceManager into TrainingConfig, replace DGX Spark button

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: End-to-end verification

**Files:** None (verification only)

- [ ] **Step 1: Verify backend imports**

```bash
python -c "
from bashgym.device_registry import DeviceRegistry
from bashgym.device_discovery import discover_ssh_devices
from bashgym.api.device_routes import router
from bashgym.api.device_schemas import DeviceCreate, DeviceResponse, DiscoverResponse
from bashgym.gym.remote_trainer import PreflightResult
print('All backend imports OK')
"
```

- [ ] **Step 2: Verify app creates with device routes**

```bash
python -c "
from bashgym.api.routes import create_app
app = create_app()
device_routes = [r for r in app.routes if hasattr(r, 'path') and '/devices' in r.path]
print(f'App: {len(app.routes)} total routes, {len(device_routes)} device routes')
assert len(device_routes) >= 7, f'Expected 7+ device routes, got {len(device_routes)}'
print('Device routes registered OK')
"
```

- [ ] **Step 3: Test SSH config discovery**

```bash
python -c "
from bashgym.device_discovery import discover_ssh_devices
result = discover_ssh_devices()
print(f'Discovered {len(result[\"candidates\"])} SSH candidates')
for c in result['candidates']:
    print(f'  {c[\"ssh_alias\"]} -> {c[\"host\"]}')
"
```

- [ ] **Step 4: Test device registry CRUD**

```bash
python -c "
import asyncio
from bashgym.device_registry import DeviceRegistry
from pathlib import Path
import tempfile, os

async def test():
    # Use temp file for testing
    tmp = Path(tempfile.mktemp(suffix='.json'))
    try:
        reg = DeviceRegistry(path=tmp)

        # Add device
        d = await reg.add_device(name='Test GPU', host='10.0.0.1', username='user')
        print(f'Added: {d.id} ({d.name})')
        assert d.is_default == True  # First device becomes default

        # List
        devices = await reg.list_devices()
        assert len(devices) == 1

        # Get default
        default = await reg.get_default()
        assert default.id == d.id

        # Update capabilities
        await reg.update_capabilities(d.id, {'gpus': [{'name': 'A100'}]})
        updated = await reg.get_device(d.id)
        assert updated.capabilities['gpus'][0]['name'] == 'A100'

        # Duplicate rejection
        try:
            await reg.add_device(name='Dup', host='10.0.0.1', username='user')
            assert False, 'Should have raised'
        except ValueError:
            print('Duplicate rejected OK')

        # Remove
        await reg.remove_device(d.id)
        assert len(await reg.list_devices()) == 0

        print('All CRUD tests passed')
    finally:
        if tmp.exists():
            os.unlink(tmp)

asyncio.run(test())
"
```

- [ ] **Step 5: Test auto-import from env**

```bash
python -c "
import asyncio
from bashgym.device_registry import DeviceRegistry
from pathlib import Path
import tempfile, os

async def test():
    tmp = Path(tempfile.mktemp(suffix='.json'))
    try:
        reg = DeviceRegistry(path=tmp)
        result = await reg.auto_import_from_env()
        if result:
            print(f'Auto-imported: {result.name} ({result.host})')
        else:
            print('No auto-import (SSH not configured or file exists)')
    finally:
        if tmp.exists():
            os.unlink(tmp)

asyncio.run(test())
"
```
