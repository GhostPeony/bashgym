"""Pydantic schemas for device management API."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class DeviceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    host: str = Field(..., min_length=1, description="Hostname or IP address")
    port: int = Field(22, ge=1, le=65535)
    username: str = Field(..., min_length=1)
    key_path: str = Field("~/.ssh/id_rsa", description="Path to SSH private key")
    work_dir: str = Field("~/bashgym-training", description="Remote working directory")


class DeviceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    host: Optional[str] = Field(None, min_length=1)
    port: Optional[int] = Field(None, ge=1, le=65535)
    username: Optional[str] = Field(None, min_length=1)
    key_path: Optional[str] = None
    work_dir: Optional[str] = None


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
    last_seen: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class SSHCandidateResponse(BaseModel):
    ssh_alias: str
    host: str
    username: Optional[str] = None
    port: int = 22
    key_path: Optional[str] = None
    already_added: bool = False
    existing_device_id: Optional[str] = None


class DiscoverResponse(BaseModel):
    candidates: List[SSHCandidateResponse]
    ssh_config_path: str
