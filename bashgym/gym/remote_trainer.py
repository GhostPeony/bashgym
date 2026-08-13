"""
Remote training execution via SSH.

Uploads training scripts and datasets to a private compute target and executes
training over SSH. Logs and model artifacts remain on the compute target;
callers receive opaque run references instead of local copies.
"""

import asyncio
import base64
import json
import logging
import shlex
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

    async def _resolve_work_dir(self, conn) -> str:
        """Expand a leading ``~`` in the remote work dir to the remote $HOME.

        SFTP operations treat ``~`` as a literal directory name, so every
        path handed to the SFTP client must be absolute.
        """
        work_dir = self.config.remote_work_dir
        if work_dir == "~" or work_dir.startswith("~/"):
            result = await conn.run('printf %s "$HOME"', check=False)
            home = result.stdout.strip()
            if home:
                work_dir = home + work_dir[1:]
        return work_dir

    def _remote_run_dir(self, run_id: str, work_dir: str | None = None) -> str:
        """Get the remote directory for a training run."""
        return f"{work_dir or self.config.remote_work_dir}/{run_id}"

    async def _upload_files(
        self,
        conn,
        run_id: str,
        script_path: Path,
        dataset_path: Path,
        work_dir: str | None = None,
    ) -> None:
        """Upload training script and dataset to remote via SFTP."""
        remote_dir = self._remote_run_dir(run_id, work_dir)
        sftp = await conn.start_sftp_client()
        await sftp.makedirs(remote_dir, exist_ok=True)
        await sftp.put(str(script_path), f"{remote_dir}/{script_path.name}")
        await sftp.put(str(dataset_path), f"{remote_dir}/{dataset_path.name}")
        logger.info(f"Uploaded training files to {remote_dir}")

    async def _start_remote_training(
        self, conn, run_id: str, script_name: str = "train_sft.py", work_dir: str | None = None
    ) -> int:
        """Start training on the remote machine, return PID.

        The wrapped command records the script's exit status to ``exit_code``
        in the run directory so the caller can distinguish a crashed run from
        a completed one.
        """
        remote_dir = self._remote_run_dir(run_id, work_dir)
        inner = (
            f"{self._venv_cmd(f'PYTHONUNBUFFERED=1 python3 {script_name}')}; echo $? > exit_code"
        )
        cmd = f"cd {remote_dir} && " f"nohup bash -c '{inner}' > training.log 2>&1 & echo $!"
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
        work_dir: str | None = None,
    ) -> None:
        """Stream bounded progress chunks while the complete log stays remote."""
        remote_dir = self._remote_run_dir(run_id, work_dir)
        dead_polls = 0
        offset = 0
        partial = ""
        log_file = f"{remote_dir}/training.log"
        chunk_script = (
            "import base64,json,pathlib,sys;"
            "p=pathlib.Path(sys.argv[1]);o=int(sys.argv[2]);n=int(sys.argv[3]);"
            "f=p.open('rb') if p.is_file() else None;"
            "f.seek(o) if f else None;d=f.read(n) if f else b'';"
            "print(json.dumps({'remote_log_chunk':True,'offset':f.tell() if f else o,"
            "'data':base64.b64encode(d).decode('ascii')},separators=(',',':')))"
        )

        async def emit_chunk() -> bool:
            nonlocal offset, partial
            command = " ".join(
                shlex.quote(value)
                for value in (
                    "python3",
                    "-c",
                    chunk_script,
                    log_file,
                    str(offset),
                    "65536",
                )
            )
            result = await conn.run(command, check=False)
            if result.exit_status != 0:
                return False
            try:
                payload = json.loads(result.stdout.strip())
                data = base64.b64decode(payload["data"], validate=True)
                next_offset = int(payload["offset"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return False
            if next_offset < offset or next_offset - offset != len(data):
                return False
            offset = next_offset
            if not data:
                return False
            text = partial + data.decode("utf-8", errors="replace")
            lines = text.split("\n")
            partial = lines.pop()
            if log_callback:
                for line in lines:
                    if line.endswith("\r"):
                        line = line[:-1]
                    if line:
                        try:
                            log_callback(line)
                        except Exception as exc:
                            logger.warning(f"Log callback error: {exc}")
            return True

        while True:
            await emit_chunk()
            alive = await conn.run(f"kill -0 {remote_pid} 2>/dev/null", check=False)
            if alive.exit_status != 0:
                # A failed kill -0 can be a transient channel error, not a dead
                # process. The run is authoritatively over when exit_code
                # exists; otherwise require several consecutive failures.
                done = await conn.run(f"test -f {remote_dir}/exit_code", check=False)
                if done.exit_status != 0:
                    dead_polls += 1
                    if dead_polls < 5:
                        await asyncio.sleep(poll_interval)
                        continue
                while await emit_chunk():
                    pass
                if partial and log_callback:
                    try:
                        log_callback(partial.rstrip("\r"))
                    except Exception as exc:
                        logger.warning(f"Log callback error: {exc}")
                break

            dead_polls = 0
            await asyncio.sleep(poll_interval)

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
        training -> stream progress -> return remote-resident references.

        Args:
            run_id: Unique identifier for this training run.
            script_path: Local path to the training script.
            dataset_path: Local path to the training dataset.
            local_output_dir: Controller-side run directory. It is retained for
                API compatibility and is never populated with remote artifacts.
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
            # SFTP does not expand "~" — resolve the work dir once and use it
            # for every path in this run.
            work_dir = await self._resolve_work_dir(conn)
            remote_dir = self._remote_run_dir(run_id, work_dir)

            # Upload
            await self._upload_files(conn, run_id, script_path, dataset_path, work_dir)

            # Execute
            remote_pid = await self._start_remote_training(conn, run_id, script_name, work_dir)
            if pid_callback:
                pid_callback(remote_pid)

            # Stream logs until done
            await self._stream_logs(
                conn,
                run_id,
                remote_pid,
                log_callback=log_callback,
                poll_interval=2.0,
                work_dir=work_dir,
            )

            # The training process has exited — verify it succeeded before
            # returning references, so a crashed run is never represented as a
            # completed remote model.
            result = await conn.run(f"cat {remote_dir}/exit_code 2>/dev/null", check=False)
            exit_code = result.stdout.strip()
            if exit_code != "0":
                return {
                    "success": False,
                    "remote_pid": remote_pid,
                    "run_id": run_id,
                    "error": f"Remote training exited with code {exit_code or 'unknown'}.",
                    "log_ref": f"ssh-run://{run_id}/training.log",
                }

            artifact_probe = await conn.run(
                f"artifact_manifest=1; for name in final merged; do "
                f"path={shlex.quote(remote_dir)}/$name; "
                'test -d "$path" && test ! -L "$path" && '
                'test -f "$path/config.json" && '
                'find "$path" -type f '
                "\\( -name '*.safetensors' -o -name 'pytorch_model*.bin' \\) "
                "-print -quit | grep -q . && printf '%s\\n' \"$name\"; done",
                check=False,
            )
            artifact_names = tuple(
                name for name in artifact_probe.stdout.splitlines() if name in {"final", "merged"}
            )
            if artifact_probe.exit_status != 0 or not artifact_names:
                return {
                    "success": False,
                    "remote_pid": remote_pid,
                    "run_id": run_id,
                    "error": "Remote training produced no model artifact.",
                    "log_ref": f"ssh-run://{run_id}/training.log",
                }

        return {
            "success": True,
            "remote_pid": remote_pid,
            "run_id": run_id,
            "remote_run_ref": f"ssh-run://{run_id}",
            "artifact_refs": [f"ssh-run://{run_id}/{name}" for name in artifact_names],
        }
