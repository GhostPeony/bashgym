"""
Device Management API Routes

REST endpoints for managing SSH training devices. Backed by DeviceRegistry
(JSON storage) and SSH config discovery via device_discovery.
"""

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from bashgym.api.device_schemas import (
    DeviceCreate,
    DeviceResponse,
    DeviceUpdate,
    DiscoverResponse,
)
from bashgym.device_discovery import discover_ssh_devices
from bashgym.device_registry import DeviceRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level registry singleton
# ---------------------------------------------------------------------------

_registry = DeviceRegistry()


def get_registry() -> DeviceRegistry:
    """Return the module-level DeviceRegistry singleton."""
    return _registry


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/devices", tags=["Devices"])


def _device_to_response(device) -> DeviceResponse:
    """Convert a Device dataclass to a DeviceResponse schema."""
    return DeviceResponse(**asdict(device))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[DeviceResponse])
async def list_devices():
    """Return all registered devices."""
    devices = await _registry.list_devices()
    return [_device_to_response(d) for d in devices]


@router.post("", response_model=DeviceResponse, status_code=201)
async def add_device(body: DeviceCreate):
    """Add a new device. Returns 409 if a device with the same host+username already exists."""
    try:
        device = await _registry.add_device(
            name=body.name,
            host=body.host,
            username=body.username,
            key_path=body.key_path,
            work_dir=body.work_dir,
            port=body.port,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _device_to_response(device)


@router.put("/{device_id}", response_model=DeviceResponse)
async def update_device(device_id: str, body: DeviceUpdate):
    """Partially update a device. Returns 404 if not found."""
    updates: dict[str, Any] = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        device = await _registry.update_device(device_id, updates)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _device_to_response(device)


@router.delete("/{device_id}")
async def remove_device(device_id: str):
    """Remove a device by ID. Returns 404 if not found."""
    try:
        await _registry.remove_device(device_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@router.post("/{device_id}/set-default", response_model=DeviceResponse)
async def set_default_device(device_id: str):
    """Set a device as the default. Returns 404 if not found."""
    try:
        device = await _registry.set_default(device_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _device_to_response(device)


@router.post("/{device_id}/preflight")
async def preflight_device(device_id: str):
    """Run a preflight check on a device and update its capabilities.

    Constructs an SSHConfig from the stored device record, creates a
    RemoteTrainer, and calls preflight_check(). On success the device's
    capabilities are updated in the registry.

    Returns the preflight result alongside the updated device record.
    """
    device = await _registry.get_device(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail=f"Device '{device_id}' not found.")

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
    except Exception as exc:
        logger.warning("Preflight failed for device %s: %s", device_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Connection or preflight error: {exc}",
        )

    # Persist capabilities when the check succeeded
    updated_device = device
    if result.ok:
        capabilities: dict[str, Any] = {}
        if result.python_version is not None:
            capabilities["python_version"] = result.python_version
        if result.disk_free_gb is not None:
            capabilities["disk_free_gb"] = result.disk_free_gb
        if result.hostname is not None:
            capabilities["hostname"] = result.hostname
        if result.os_info is not None:
            capabilities["os_info"] = result.os_info
        if result.cuda_version is not None:
            capabilities["cuda_version"] = result.cuda_version
        if result.gpus is not None:
            capabilities["gpus"] = result.gpus

        try:
            updated_device = await _registry.update_capabilities(device_id, capabilities)
        except KeyError:
            pass  # device was removed between check and update — non-fatal

    preflight_dict = {
        "ok": result.ok,
        "python_version": result.python_version,
        "disk_free_gb": result.disk_free_gb,
        "error": result.error,
        "hostname": result.hostname,
        "os_info": result.os_info,
        "cuda_version": result.cuda_version,
        "gpus": result.gpus,
    }

    return {
        "preflight": preflight_dict,
        "device": _device_to_response(updated_device).model_dump(),
    }


@router.post("/discover", response_model=DiscoverResponse)
async def discover_devices():
    """Parse ~/.ssh/config and return SSH device candidates.

    Cross-references candidates against the existing device list so each
    candidate carries an ``already_added`` flag and the matching
    ``existing_device_id`` when relevant.
    """
    existing = await _registry.list_devices()
    existing_dicts = [asdict(d) for d in existing]
    result = discover_ssh_devices(existing_devices=existing_dicts)
    return DiscoverResponse(**result)
