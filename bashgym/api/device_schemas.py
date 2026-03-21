"""Pydantic schemas for device management API."""

from typing import Any

from pydantic import BaseModel, Field


class DeviceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    host: str = Field(..., min_length=1, description="Hostname or IP address")
    port: int = Field(22, ge=1, le=65535)
    username: str = Field(..., min_length=1)
    key_path: str = Field("~/.ssh/id_rsa", description="Path to SSH private key")
    work_dir: str = Field("~/bashgym-training", description="Remote working directory")


class DeviceUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    host: str | None = Field(None, min_length=1)
    port: int | None = Field(None, ge=1, le=65535)
    username: str | None = Field(None, min_length=1)
    key_path: str | None = None
    work_dir: str | None = None


class DeviceResponse(BaseModel):
    id: str
    name: str
    host: str
    port: int
    username: str
    key_path: str
    work_dir: str
    is_default: bool
    added_at: str
    last_seen: str | None = None
    capabilities: dict[str, Any] | None = None


class SSHCandidateResponse(BaseModel):
    ssh_alias: str
    host: str
    username: str | None = None
    port: int = 22
    key_path: str | None = None
    already_added: bool = False
    existing_device_id: str | None = None


class DiscoverResponse(BaseModel):
    candidates: list[SSHCandidateResponse]
    ssh_config_path: str
