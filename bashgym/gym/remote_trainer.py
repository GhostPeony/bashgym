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
    # Enhanced device info
    hostname: Optional[str] = None
    os_info: Optional[str] = None
    cuda_version: Optional[str] = None
    gpus: Optional[List[Dict[str, Any]]] = None


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

    async def preflight_check(self) -> PreflightResult:
        """Verify the remote machine is ready for training."""
        try:
            conn = await self._connect()
        except Exception as e:
            return PreflightResult(ok=False, error=f"Connection failed: {e}")

        async with conn:
            # Check Python
            result = await conn.run(self._venv_cmd("python3 --version"), check=False)
            if result.exit_status != 0:
                return PreflightResult(ok=False, error="python3 not found on remote")
            python_version = result.stdout.strip()

            # Check Unsloth
            result = await conn.run(self._venv_cmd('python3 -c "import unsloth"'), check=False)
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

            result = PreflightResult(
                ok=True,
                python_version=python_version,
                disk_free_gb=disk_free_gb,
            )

            # Hostname
            try:
                r = await conn.run('hostname', timeout=5)
                result.hostname = r.stdout.strip()
            except Exception:
                pass

            # OS info
            try:
                r = await conn.run('uname -sr', timeout=5)
                result.os_info = r.stdout.strip()
            except Exception:
                pass

            # GPU info
            try:
                r = await conn.run(
                    'nvidia-smi --query-gpu=name,memory.total,memory.free'
                    ' --format=csv,noheader,nounits',
                    timeout=10,
                )
                gpus = []
                for row in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in row.split(',')]
                    if len(parts) == 3:
                        gpus.append({
                            "name": parts[0],
                            "vram_total_gb": round(float(parts[1]) / 1024, 1),
                            "vram_free_gb": round(float(parts[2]) / 1024, 1),
                        })
                if gpus:
                    result.gpus = gpus
            except Exception:
                pass

            # CUDA version
            try:
                import re
                r = await conn.run('nvidia-smi | head -3', timeout=10)
                m = re.search(r'CUDA Version:\s*([\d.]+)', r.stdout)
                if m:
                    result.cuda_version = m.group(1)
            except Exception:
                pass

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

    async def _start_remote_training(self, conn, run_id: str) -> int:
        """Start training on the remote machine, return PID."""
        remote_dir = self._remote_run_dir(run_id)
        cmd = (
            f"cd {remote_dir} && "
            f"nohup bash -c '{self._venv_cmd('python3 train_sft.py')}' > training.log 2>&1 & echo $!"
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
        log_callback: Optional[Callable[[str], None]] = None,
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
        log_callback: Optional[Callable[[str], None]] = None,
        pid_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
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

        Returns:
            Dict with 'success' bool plus 'remote_pid'/'run_id' on success
            or 'error' string on failure.
        """
        # Pre-flight
        preflight = await self.preflight_check()
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
            remote_pid = await self._start_remote_training(conn, run_id)
            if pid_callback:
                pid_callback(remote_pid)

            # Stream logs until done
            await self._stream_logs(
                conn, run_id, remote_pid,
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
