"""
Remote training execution via SSH.

Uploads training scripts and datasets to a private compute target,
executes training over SSH, streams logs back in real-time, and downloads
model artifacts on completion.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import asyncssh

    HAS_ASYNCSSH = True
except ImportError:
    HAS_ASYNCSSH = False
    asyncssh = None


def _mib_to_gb(value: str) -> float | None:
    """Convert a MiB string to GB, or None for nvidia-smi's `[N/A]` (unified memory)."""
    try:
        return round(float(value) / 1024, 1)
    except (TypeError, ValueError):
        return None


def parse_nvidia_smi_gpus(stdout: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.free` CSV rows.

    Unified-memory devices can report VRAM as `[N/A]`; those
    become `None` so the caller can fall back to system RAM as the budget.
    """
    gpus: list[dict[str, Any]] = []
    for row in stdout.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) != 3:
            continue
        gpus.append(
            {
                "name": parts[0],
                "vram_total_gb": _mib_to_gb(parts[1]),
                "vram_free_gb": _mib_to_gb(parts[2]),
            }
        )
    return gpus


def parse_meminfo_gb(stdout: str) -> float | None:
    """Parse total system RAM (GiB) from a `/proc/meminfo` `MemTotal: NNN kB` line."""
    for line in stdout.splitlines():
        if line.lower().startswith("memtotal"):
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                return round(int(digits) / 1024 / 1024, 1)
    return None


def remote_compute_budget_gb(
    gpus: list[dict[str, Any]] | None, ram_gb: float | None
) -> dict[str, Any]:
    """Effective trainable memory budget, with unified-memory fallback to RAM.

    Discrete GPUs use the largest reported VRAM. When GPUs are present but report
    no discrete VRAM (unified memory), the budget is the system RAM instead.
    """
    discrete = [g["vram_total_gb"] for g in (gpus or []) if g.get("vram_total_gb") is not None]
    if discrete:
        return {"effective_vram_gb": max(discrete), "unified_memory": False}
    unified = bool(gpus) and ram_gb is not None
    return {"effective_vram_gb": ram_gb, "unified_memory": unified}


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
    python_version: str | None = None
    disk_free_gb: float | None = None
    error: str | None = None
    # Enhanced device info
    hostname: str | None = None
    os_info: str | None = None
    cuda_version: str | None = None
    gpus: list[dict[str, Any]] | None = None
    # Memory budget (effective_vram_gb falls back to RAM on unified-memory devices)
    ram_gb: float | None = None
    effective_vram_gb: float | None = None
    unified_memory: bool = False
    unsloth_available: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Full serialization for API responses."""
        return {
            "ok": self.ok,
            "python_version": self.python_version,
            "disk_free_gb": self.disk_free_gb,
            "error": self.error,
            "hostname": self.hostname,
            "os_info": self.os_info,
            "cuda_version": self.cuda_version,
            "gpus": self.gpus,
            "ram_gb": self.ram_gb,
            "effective_vram_gb": self.effective_vram_gb,
            "unified_memory": self.unified_memory,
            "unsloth_available": self.unsloth_available,
            "warnings": self.warnings,
        }

    def capabilities(self) -> dict[str, Any]:
        """Discovered capability fields to persist on the device registry.

        ``None`` discovery values are dropped so a transient probe miss never
        overwrites previously-known data. ``unified_memory`` is always included so
        the recommendation layer can route unified-memory budgets to RAM.
        """
        fields: dict[str, Any] = {
            "python_version": self.python_version,
            "disk_free_gb": self.disk_free_gb,
            "hostname": self.hostname,
            "os_info": self.os_info,
            "cuda_version": self.cuda_version,
            "gpus": self.gpus,
            "ram_gb": self.ram_gb,
            "effective_vram_gb": self.effective_vram_gb,
            "unsloth_available": self.unsloth_available,
        }
        caps = {key: value for key, value in fields.items() if value is not None}
        caps["unified_memory"] = self.unified_memory
        return caps


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
            connect_timeout=10,
        )

    async def preflight_check(self, *, require_unsloth: bool = True) -> PreflightResult:
        """Verify the remote machine is ready for training.

        ``require_unsloth`` defaults to True for the Unsloth backend. Set it False
        for plain-transformers backends on newer compute architectures where Unsloth
        cannot load) so a missing Unsloth is reported as a warning, not a failure.
        """
        try:
            conn = await self._connect()
        except Exception as e:
            return PreflightResult(ok=False, error=f"Connection failed: {e}")

        async with conn:
            warnings: list[str] = []

            # Check Python
            result = await conn.run(self._venv_cmd("python3 --version"), check=False)
            if result.exit_status != 0:
                return PreflightResult(ok=False, error="python3 not found on remote")
            python_version = result.stdout.strip()

            # Check Unsloth — required only for the Unsloth backend
            result = await conn.run(self._venv_cmd('python3 -c "import unsloth"'), check=False)
            unsloth_available = result.exit_status == 0
            if not unsloth_available:
                if require_unsloth:
                    return PreflightResult(
                        ok=False,
                        python_version=python_version,
                        unsloth_available=False,
                        error="Unsloth not installed on remote. Run: pip install unsloth",
                    )
                warnings.append(
                    "Unsloth not installed on remote; using the plain transformers backend."
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

            result = PreflightResult(
                ok=True,
                python_version=python_version,
                disk_free_gb=disk_free_gb,
                unsloth_available=unsloth_available,
                warnings=warnings,
            )

            # Hostname
            try:
                r = await conn.run("hostname", timeout=5)
                result.hostname = r.stdout.strip()
            except Exception:
                pass

            # OS info
            try:
                r = await conn.run("uname -sr", timeout=5)
                result.os_info = r.stdout.strip()
            except Exception:
                pass

            # GPU info
            try:
                r = await conn.run(
                    "nvidia-smi --query-gpu=name,memory.total,memory.free"
                    " --format=csv,noheader,nounits",
                    timeout=10,
                )
                gpus = parse_nvidia_smi_gpus(r.stdout)
                if gpus:
                    result.gpus = gpus
            except Exception:
                pass

            # CUDA version
            try:
                import re

                r = await conn.run("nvidia-smi | head -3", timeout=10)
                m = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
                if m:
                    result.cuda_version = m.group(1)
            except Exception:
                pass

            # System RAM — the effective budget for unified-memory devices
            try:
                r = await conn.run("cat /proc/meminfo | head -1", timeout=5)
                result.ram_gb = parse_meminfo_gb(r.stdout)
            except Exception:
                pass

            # Effective trainable budget (unified-memory fallback to RAM)
            budget = remote_compute_budget_gb(result.gpus, result.ram_gb)
            result.effective_vram_gb = budget["effective_vram_gb"]
            result.unified_memory = budget["unified_memory"]

            return result

    def _venv_cmd(self, cmd: str) -> str:
        """Wrap a command with venv activation."""
        venv = f"{self.config.remote_work_dir}/venv"
        return f"source {venv}/bin/activate 2>/dev/null; {cmd}"

    def _remote_run_dir(self, run_id: str) -> str:
        """Get the remote directory for a training run."""
        return f"{self.config.remote_work_dir}/{run_id}"

    async def _upload_files(
        self,
        conn,
        run_id: str,
        script_path: Path,
        dataset_path: Path,
    ) -> None:
        """Upload training script and dataset to remote via SFTP."""
        remote_dir = self._remote_run_dir(run_id)
        sftp = await conn.start_sftp_client()
        await sftp.makedirs(remote_dir, exist_ok=True)
        await sftp.put(str(script_path), f"{remote_dir}/{script_path.name}")
        await sftp.put(str(dataset_path), f"{remote_dir}/{dataset_path.name}")
        logger.info(f"Uploaded training files to {remote_dir}")

    async def _start_remote_training(
        self, conn, run_id: str, script_name: str = "train_sft.py"
    ) -> int:
        """Start training on the remote machine, return PID."""
        remote_dir = self._remote_run_dir(run_id)
        cmd = (
            f"cd {remote_dir} && "
            f"nohup bash -c '{self._venv_cmd(f'python3 {script_name}')}' > training.log 2>&1 & echo $!"
        )
        result = await conn.run(cmd, check=False)
        pid = int(result.stdout.strip())
        logger.info(f"Remote training started with PID {pid}")
        return pid

    async def _stream_logs(
        self,
        conn,
        run_id: str,
        remote_pid: int,
        log_callback: Callable[[str], None] | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        """Stream training logs from remote machine until process exits."""
        remote_dir = self._remote_run_dir(run_id)
        log_file = f"{remote_dir}/training.log"
        lines_read = 0

        while True:
            result = await conn.run(
                f"tail -n +{lines_read + 1} {log_file} 2>/dev/null",
                check=False,
            )
            if result.exit_status == 0 and result.stdout.strip():
                new_lines = result.stdout.strip().split("\n")
                for line in new_lines:
                    if log_callback:
                        log_callback(line)
                lines_read += len(new_lines)

            alive = await conn.run(f"kill -0 {remote_pid} 2>/dev/null", check=False)
            if alive.exit_status != 0:
                # Process exited — drain any remaining log lines
                result = await conn.run(
                    f"tail -n +{lines_read + 1} {log_file} 2>/dev/null",
                    check=False,
                )
                if result.exit_status == 0 and result.stdout.strip():
                    for line in result.stdout.strip().split("\n"):
                        if log_callback:
                            log_callback(line)
                break

            await asyncio.sleep(poll_interval)

    async def _download_artifacts(
        self,
        conn,
        run_id: str,
        local_output_dir: Path,
    ) -> None:
        """Download trained model artifacts from remote to local."""
        remote_dir = self._remote_run_dir(run_id)
        sftp = await conn.start_sftp_client()

        local_output_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["final", "merged"]:
            remote_path = f"{remote_dir}/{subdir}"
            try:
                entries = await sftp.listdir(remote_path)
                local_subdir = local_output_dir / subdir
                local_subdir.mkdir(parents=True, exist_ok=True)
                for entry in entries:
                    remote_file = f"{remote_path}/{entry}"
                    local_file = str(local_subdir / entry)
                    logger.info(f"Downloading {remote_file} -> {local_file}")
                    await sftp.get(remote_file, local_file)
            except Exception as e:
                logger.warning(f"Could not download {subdir}: {e}")

    async def pause_remote(self, remote_pid: int) -> bool:
        """Pause a remote training process."""
        try:
            conn = await self._connect()
            async with conn:
                result = await conn.run(f"kill -STOP {remote_pid}", check=False)
                return result.exit_status == 0
        except Exception as e:
            logger.error(f"Failed to pause remote PID {remote_pid}: {e}")
            return False

    async def resume_remote(self, remote_pid: int) -> bool:
        """Resume a paused remote training process."""
        try:
            conn = await self._connect()
            async with conn:
                result = await conn.run(f"kill -CONT {remote_pid}", check=False)
                return result.exit_status == 0
        except Exception as e:
            logger.error(f"Failed to resume remote PID {remote_pid}: {e}")
            return False

    async def cancel_remote(self, remote_pid: int) -> bool:
        """Cancel a remote training process."""
        try:
            conn = await self._connect()
            async with conn:
                result = await conn.run(f"kill -TERM {remote_pid}", check=False)
                return result.exit_status == 0
        except Exception as e:
            logger.error(f"Failed to cancel remote PID {remote_pid}: {e}")
            return False

    async def train_remote(
        self,
        run_id: str,
        script_path: Path,
        dataset_path: Path,
        local_output_dir: Path,
        log_callback: Callable[[str], None] | None = None,
        pid_callback: Callable[[int], None] | None = None,
        script_name: str = "train_sft.py",
        require_unsloth: bool = True,
    ) -> dict[str, Any]:
        """Full remote training orchestration.

        Runs the complete flow: preflight check -> upload files -> start
        training -> stream logs -> download artifacts.

        Args:
            run_id: Unique identifier for this training run.
            script_path: Local path to the training script.
            dataset_path: Local path to the training dataset.
            local_output_dir: Local directory to download artifacts into.
            log_callback: Optional callback invoked with each log line.
            pid_callback: Optional callback invoked with the remote PID once training starts.
            script_name: Name of the training script to execute on remote.
            require_unsloth: Gate the run on a remote Unsloth install. Set False
                for plain-transformers backends (e.g. Session Distillation) on
                compute targets where Unsloth cannot load.

        Returns:
            Dict with 'success' bool plus 'remote_pid'/'run_id' on success
            or 'error' string on failure.
        """
        # Pre-flight
        preflight = await self.preflight_check(require_unsloth=require_unsloth)
        if not preflight.ok:
            return {"success": False, "error": preflight.error}

        try:
            conn = await self._connect()
        except Exception as e:
            return {"success": False, "error": f"SSH connection failed: {e}"}

        async with conn:
            # Upload
            await self._upload_files(conn, run_id, script_path, dataset_path)

            # Execute
            remote_pid = await self._start_remote_training(conn, run_id, script_name)
            if pid_callback:
                pid_callback(remote_pid)

            # Stream logs until done
            await self._stream_logs(
                conn,
                run_id,
                remote_pid,
                log_callback=log_callback,
                poll_interval=2.0,
            )

            # Download artifacts
            await self._download_artifacts(conn, run_id, local_output_dir)

        return {
            "success": True,
            "remote_pid": remote_pid,
            "run_id": run_id,
        }
