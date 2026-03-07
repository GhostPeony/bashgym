"""
Remote training execution via SSH.

Uploads training scripts and datasets to a remote machine (e.g. DGX Spark),
executes training over SSH, streams logs back in real-time, and downloads
model artifacts on completion.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import asyncssh
    HAS_ASYNCSSH = True
except ImportError:
    HAS_ASYNCSSH = False
    asyncssh = None


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    username: str
    port: int = 22
    key_path: str = "~/.ssh/id_rsa"
    remote_work_dir: str = "~/bashgym-training"

    @classmethod
    def from_settings(cls, settings) -> "SSHConfig":
        return cls(
            host=settings.host,
            username=settings.username,
            port=settings.port,
            key_path=settings.key_path,
            remote_work_dir=settings.remote_work_dir,
        )


@dataclass
class PreflightResult:
    """Result of pre-flight checks on the remote machine."""
    ok: bool
    python_version: Optional[str] = None
    disk_free_gb: Optional[float] = None
    error: Optional[str] = None


class RemoteTrainer:
    """Execute training on a remote machine via SSH."""

    def __init__(self, config: SSHConfig):
        self.config = config

    async def _connect(self):
        """Open an SSH connection to the remote host."""
        key_path = Path(self.config.key_path).expanduser()
        return await asyncssh.connect(
            self.config.host,
            port=self.config.port,
            username=self.config.username,
            client_keys=[str(key_path)],
            known_hosts=None,
        )

    async def preflight_check(self) -> PreflightResult:
        """Verify the remote machine is ready for training."""
        try:
            conn = await self._connect()
        except Exception as e:
            return PreflightResult(ok=False, error=f"Connection failed: {e}")

        async with conn:
            # Check Python
            result = await conn.run("python3 --version", check=False)
            if result.exit_status != 0:
                return PreflightResult(ok=False, error="python3 not found on remote")
            python_version = result.stdout.strip()

            # Check Unsloth
            result = await conn.run('python3 -c "import unsloth"', check=False)
            if result.exit_status != 0:
                return PreflightResult(
                    ok=False,
                    python_version=python_version,
                    error="Unsloth not installed on remote. Run: pip install unsloth",
                )

            # Check disk space
            disk_free_gb = None
            result = await conn.run(
                f"df -BG --output=avail {self.config.remote_work_dir} 2>/dev/null | tail -1",
                check=False,
            )
            if result.exit_status == 0:
                try:
                    disk_free_gb = float(result.stdout.strip().rstrip("G"))
                except (ValueError, AttributeError):
                    pass

            return PreflightResult(
                ok=True,
                python_version=python_version,
                disk_free_gb=disk_free_gb,
            )
