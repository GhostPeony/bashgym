"""
Worker Pool Dispatcher

Manages parallel Claude Code worker sessions via subprocess execution.
Workers are spawned using `claude -p` in headless mode, each operating
in an isolated git worktree.

Module: Orchestrator
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Dict, List, Optional

from bashgym.orchestrator.models import (
    TaskNode, WorkerConfig, WorkerResult,
)

logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages parallel Claude Code worker sessions.

    Spawns workers as subprocesses using the Claude CLI in headless mode.
    Each worker runs in its own event loop task and can be monitored,
    awaited, or cancelled.
    """

    def __init__(
        self,
        max_workers: int = 5,
        default_config: Optional[WorkerConfig] = None,
    ):
        self.max_workers = max_workers
        self.default_config = default_config or WorkerConfig()
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, WorkerResult] = {}
        self._output_buffers: Dict[str, List[str]] = {}
        self._completion_events: Dict[str, asyncio.Event] = {}
        self._any_complete = asyncio.Event()

    async def spawn_worker(
        self,
        task: TaskNode,
        config: Optional[WorkerConfig] = None,
    ) -> str:
        """Spawn a new Claude Code worker for a task.

        Creates a subprocess running:
            claude -p "<prompt>" --output-format json
                --max-turns N --max-budget-usd N
                --allowedTools "..."

        Args:
            task: TaskNode describing the work
            config: Optional worker config (uses default if None)

        Returns:
            Worker ID (same as task.id for simplicity)

        Raises:
            RuntimeError: If max workers reached
        """
        if len(self._processes) >= self.max_workers:
            raise RuntimeError(
                f"Max workers ({self.max_workers}) reached. "
                f"Wait for a worker to finish."
            )

        cfg = config or self.default_config
        worker_id = task.id

        # Build the prompt
        prompt = task.worker_prompt or task.description
        if not prompt:
            prompt = f"Complete this task: {task.title}\n\n{task.description}"

        # Build CLI command
        cmd = ["claude", "-p", prompt]
        cmd.extend(cfg.to_cli_args())

        # Set working directory to worktree if available
        cwd = str(task.worktree_path) if task.worktree_path else None

        logger.info(
            f"Spawning worker {worker_id} for task '{task.title}' "
            f"(max_turns={cfg.max_turns}, budget=${cfg.max_budget_usd})"
        )

        # Initialize tracking
        self._output_buffers[worker_id] = []
        self._completion_events[worker_id] = asyncio.Event()

        # Spawn as async task for non-blocking management
        async_task = asyncio.create_task(
            self._run_worker(worker_id, cmd, cwd, cfg.timeout_seconds)
        )
        self._tasks[worker_id] = async_task

        return worker_id

    async def _run_worker(
        self,
        worker_id: str,
        cmd: List[str],
        cwd: Optional[str],
        timeout: float,
    ) -> WorkerResult:
        """Execute a worker subprocess and collect results."""
        start_time = time.time()
        session_id = ""
        output_lines: List[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._processes[worker_id] = proc

            # Read output with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                duration = time.time() - start_time
                result = WorkerResult(
                    task_id=worker_id,
                    session_id=session_id,
                    success=False,
                    output="Worker timed out",
                    exit_code=-1,
                    duration_seconds=duration,
                    error=f"Timed out after {timeout}s",
                )
                self._results[worker_id] = result
                return result

            duration = time.time() - start_time
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            # Parse JSON output from Claude CLI
            output_text = stdout
            files_modified: List[str] = []
            tokens_used = 0
            cost_usd = 0.0

            try:
                # Claude JSON output may have multiple JSON objects (stream)
                # Try to parse the last complete JSON object
                for line in stdout.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            if "session_id" in parsed:
                                session_id = parsed["session_id"]
                            if "result" in parsed:
                                output_text = parsed["result"]
                            if "usage" in parsed:
                                usage = parsed["usage"]
                                tokens_used = usage.get("total_tokens", 0)
                            if "cost_usd" in parsed:
                                cost_usd = parsed["cost_usd"]
                    except json.JSONDecodeError:
                        output_lines.append(line)
            except Exception:
                pass

            self._output_buffers[worker_id] = output_lines

            success = proc.returncode == 0
            result = WorkerResult(
                task_id=worker_id,
                session_id=session_id,
                success=success,
                output=output_text[:10000],  # Cap output size
                exit_code=proc.returncode or 0,
                duration_seconds=duration,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                files_modified=files_modified,
                error=stderr[:2000] if not success else None,
            )

        except FileNotFoundError:
            duration = time.time() - start_time
            result = WorkerResult(
                task_id=worker_id,
                session_id="",
                success=False,
                output="",
                exit_code=-1,
                duration_seconds=duration,
                error="claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
            )

        except Exception as e:
            duration = time.time() - start_time
            result = WorkerResult(
                task_id=worker_id,
                session_id="",
                success=False,
                output="",
                exit_code=-1,
                duration_seconds=duration,
                error=str(e),
            )

        finally:
            # Cleanup
            self._processes.pop(worker_id, None)
            self._results[worker_id] = result
            self._completion_events[worker_id].set()
            self._any_complete.set()

        logger.info(
            f"Worker {worker_id} finished: "
            f"{'success' if result.success else 'failed'} "
            f"({result.duration_seconds:.1f}s, ${result.cost_usd:.2f})"
        )
        return result

    async def wait_for_worker(
        self,
        worker_id: str,
        timeout: Optional[float] = None,
    ) -> WorkerResult:
        """Wait for a specific worker to complete.

        Args:
            worker_id: Worker to wait for
            timeout: Optional timeout in seconds

        Returns:
            WorkerResult from the completed worker

        Raises:
            KeyError: If worker_id not found
            asyncio.TimeoutError: If timeout exceeded
        """
        if worker_id not in self._completion_events:
            if worker_id in self._results:
                return self._results[worker_id]
            raise KeyError(f"Worker '{worker_id}' not found")

        if timeout:
            await asyncio.wait_for(
                self._completion_events[worker_id].wait(),
                timeout=timeout,
            )
        else:
            await self._completion_events[worker_id].wait()

        return self._results[worker_id]

    async def wait_for_any(self, timeout: float = 600) -> WorkerResult:
        """Wait for any active worker to complete. Returns first result.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            WorkerResult from the first completed worker

        Raises:
            asyncio.TimeoutError: If no worker completes within timeout
            RuntimeError: If no workers are active
        """
        if not self._tasks:
            raise RuntimeError("No active workers to wait for")

        self._any_complete.clear()

        # Check if any are already done
        for worker_id, result in self._results.items():
            if worker_id in self._tasks:
                task = self._tasks.pop(worker_id)
                if task.done():
                    return result

        await asyncio.wait_for(self._any_complete.wait(), timeout=timeout)

        # Find the completed worker
        for worker_id, event in self._completion_events.items():
            if event.is_set() and worker_id in self._results:
                self._tasks.pop(worker_id, None)
                return self._results[worker_id]

        raise RuntimeError("No completed worker found after event signal")

    async def cancel_worker(self, worker_id: str) -> None:
        """Cancel a running worker.

        Args:
            worker_id: Worker to cancel
        """
        if worker_id in self._processes:
            proc = self._processes[worker_id]
            try:
                proc.terminate()
                await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()

        if worker_id in self._tasks:
            self._tasks[worker_id].cancel()
            self._tasks.pop(worker_id, None)

        self._processes.pop(worker_id, None)
        logger.info(f"Cancelled worker {worker_id}")

    async def cancel_all(self) -> None:
        """Cancel all active workers."""
        worker_ids = list(self._processes.keys())
        for worker_id in worker_ids:
            await self.cancel_worker(worker_id)

    def get_worker_output(self, worker_id: str) -> List[str]:
        """Get buffered output lines for a worker.

        Args:
            worker_id: Worker to get output for

        Returns:
            List of output lines
        """
        return self._output_buffers.get(worker_id, [])

    @property
    def available_slots(self) -> int:
        """Number of available worker slots."""
        return self.max_workers - len(self._processes)

    @property
    def active_count(self) -> int:
        """Number of currently running workers."""
        return len(self._processes)

    @property
    def total_cost(self) -> float:
        """Total cost across all completed workers."""
        return sum(r.cost_usd for r in self._results.values())
