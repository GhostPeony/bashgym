"""
Training run state persistence and process control.

Saves/loads training run metadata to disk so the backend can reconnect
to orphaned training subprocesses after a restart.
"""

import os
import json
import ctypes
import platform
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("data/models")
STATE_FILENAME = "run_state.json"
MAX_LOG_LINES = 50


@dataclass
class TrainingRunState:
    """Persisted training run metadata."""

    run_id: str
    pid: int
    status: str  # pending/running/paused/completed/failed
    config: Dict[str, Any]
    started_at: str
    script_path: str
    dataset_path: str
    output_dir: str
    completed_at: Optional[str] = None
    last_metrics: Optional[Dict[str, Any]] = None
    last_log_lines: List[str] = field(default_factory=list)

    def state_path(self) -> Path:
        return Path(self.output_dir) / STATE_FILENAME

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_run_state(state: TrainingRunState) -> Path:
    """Write run state to {output_dir}/run_state.json."""
    path = state.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    logger.info(f"Saved run state: {path}")
    return path


def load_run_state(output_dir: str) -> Optional[TrainingRunState]:
    """Read run state from a model output directory."""
    path = Path(output_dir) / STATE_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TrainingRunState(**data)
    except Exception as e:
        logger.warning(f"Failed to load run state from {path}: {e}")
        return None


def list_run_states(models_dir: Optional[Path] = None) -> List[TrainingRunState]:
    """Scan data/models/run_*/run_state.json and return all persisted states."""
    models_dir = models_dir or DEFAULT_MODELS_DIR
    states = []
    if not models_dir.exists():
        return states
    for run_dir in sorted(models_dir.glob("run_*")):
        state = load_run_state(str(run_dir))
        if state:
            states.append(state)
    return states


def update_run_state(output_dir: str, **updates) -> Optional[TrainingRunState]:
    """Load state, apply partial updates, and save."""
    state = load_run_state(output_dir)
    if state is None:
        return None
    for key, value in updates.items():
        if hasattr(state, key):
            setattr(state, key, value)
    save_run_state(state)
    return state


# ---------------------------------------------------------------------------
# Process control (cross-platform)
# ---------------------------------------------------------------------------

def is_process_alive(pid: int) -> bool:
    """Check if a process is still running."""
    if platform.system() == "Windows":
        # On Windows, os.kill(pid, 0) doesn't work reliably.
        # Use OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION.
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        # Check exit code — STILL_ACTIVE (259) means running
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def suspend_process(pid: int) -> bool:
    """Suspend a process (SIGSTOP equivalent on Windows)."""
    if platform.system() == "Windows":
        kernel32 = ctypes.windll.kernel32
        PROCESS_SUSPEND_RESUME = 0x0800
        handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not handle:
            logger.error(f"Failed to open process {pid} for suspension")
            return False
        ntdll = ctypes.windll.ntdll
        result = ntdll.NtSuspendProcess(handle)
        kernel32.CloseHandle(handle)
        if result == 0:
            logger.info(f"Suspended process {pid}")
            return True
        logger.error(f"NtSuspendProcess failed for {pid}: NTSTATUS {result:#x}")
        return False
    else:
        import signal
        try:
            os.kill(pid, signal.SIGSTOP)
            logger.info(f"Sent SIGSTOP to process {pid}")
            return True
        except OSError as e:
            logger.error(f"Failed to suspend process {pid}: {e}")
            return False


def resume_process(pid: int) -> bool:
    """Resume a suspended process."""
    if platform.system() == "Windows":
        kernel32 = ctypes.windll.kernel32
        PROCESS_SUSPEND_RESUME = 0x0800
        handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if not handle:
            logger.error(f"Failed to open process {pid} for resume")
            return False
        ntdll = ctypes.windll.ntdll
        result = ntdll.NtResumeProcess(handle)
        kernel32.CloseHandle(handle)
        if result == 0:
            logger.info(f"Resumed process {pid}")
            return True
        logger.error(f"NtResumeProcess failed for {pid}: NTSTATUS {result:#x}")
        return False
    else:
        import signal
        try:
            os.kill(pid, signal.SIGCONT)
            logger.info(f"Sent SIGCONT to process {pid}")
            return True
        except OSError as e:
            logger.error(f"Failed to resume process {pid}: {e}")
            return False


def terminate_process(pid: int, timeout: float = 5.0) -> bool:
    """Terminate a process, escalating to kill after timeout."""
    if platform.system() == "Windows":
        kernel32 = ctypes.windll.kernel32
        PROCESS_TERMINATE = 0x0001
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if not handle:
            logger.error(f"Failed to open process {pid} for termination")
            return False
        result = kernel32.TerminateProcess(handle, 1)
        kernel32.CloseHandle(handle)
        if result:
            logger.info(f"Terminated process {pid}")
            return True
        logger.error(f"TerminateProcess failed for {pid}")
        return False
    else:
        import signal
        import time
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for graceful exit
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not is_process_alive(pid):
                    logger.info(f"Process {pid} terminated gracefully")
                    return True
                time.sleep(0.25)
            # Force kill
            os.kill(pid, signal.SIGKILL)
            logger.info(f"Force-killed process {pid}")
            return True
        except OSError as e:
            logger.error(f"Failed to terminate process {pid}: {e}")
            return False
