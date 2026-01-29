"""
Agent Runner for The Arena

Manages the execution of Claude Code CLI within sandboxed environments.
Handles task submission, hook configuration, and output capture.

Instrumentation:
  - Guardrails check task prompts for injection
  - Profiler tracks task execution
  - PII filtering on captured outputs
"""

import os
import json
import subprocess
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from datetime import datetime, timezone
import threading
import queue

# Handle both package and standalone usage
try:
    from bashgym.arena.sandbox import SandboxManager, SandboxConfig, SandboxInstance
except ImportError:
    from .sandbox import SandboxManager, SandboxConfig, SandboxInstance

# Import instrumentation (optional)
try:
    from bashgym.core import get_instrumentation, Instrumentation
    HAS_INSTRUMENTATION = True
except ImportError:
    HAS_INSTRUMENTATION = False
    Instrumentation = None

if TYPE_CHECKING:
    from bashgym.core import Instrumentation


@dataclass
class AgentConfig:
    """Configuration for the agent runner."""

    # Claude Code settings
    claude_cli_path: str = "claude"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192

    # Execution settings
    timeout: int = 1800  # 30 minutes
    max_retries: int = 3

    # Hook settings
    hooks_enabled: bool = True
    hooks_path: str = ".claude/hooks"

    # Output settings
    capture_output: bool = True
    stream_output: bool = True

    # Safety settings
    require_approval: bool = False
    allowed_tools: List[str] = field(default_factory=lambda: [
        "Bash", "Read", "Write", "Edit", "Glob", "Grep"
    ])

    # Instrumentation
    enable_guardrails: bool = True
    enable_profiling: bool = True
    filter_pii_from_output: bool = True


@dataclass
class TaskResult:
    """Result of an agent task execution."""

    task_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    trace_path: Optional[Path] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Instrumentation fields
    injection_blocked: bool = False
    pii_redactions: int = 0


class AgentRunner:
    """
    Runs Claude Code CLI agents within sandboxed environments.

    Features:
    - Subprocess management for Claude CLI
    - Hook integration for trace capture
    - Output streaming and capture
    - Timeout and retry handling
    - Guardrails for prompt injection detection
    - Profiler for task execution tracking
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        instrumentation: Optional["Instrumentation"] = None
    ):
        """Initialize the agent runner."""
        self.config = config or AgentConfig()
        self.sandbox_manager = sandbox_manager or SandboxManager()
        self.active_tasks: Dict[str, subprocess.Popen] = {}

        # Instrumentation (guardrails + profiling)
        self._instrumentation = instrumentation
        if self._instrumentation is None and HAS_INSTRUMENTATION:
            if self.config.enable_guardrails or self.config.enable_profiling:
                self._instrumentation = get_instrumentation()

    @property
    def instrumentation(self) -> Optional["Instrumentation"]:
        """Get the instrumentation instance."""
        return self._instrumentation

    def prepare_workspace(
        self,
        sandbox: SandboxInstance,
        task_prompt: str,
        task_id: str
    ) -> None:
        """
        Prepare the workspace for agent execution.

        Sets up hooks, metadata, and initial state.
        """
        workspace = sandbox.workspace_path

        # Create hooks directory
        hooks_dir = workspace / ".claude" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)

        # Copy hook scripts from project
        project_hooks = Path(__file__).parent.parent.parent / ".claude" / "hooks"
        if project_hooks.exists():
            for hook_file in project_hooks.glob("*.py"):
                dest = hooks_dir / hook_file.name
                dest.write_text(hook_file.read_text())
                dest.chmod(0o755)

        # Create session metadata
        metadata = {
            "task_id": task_id,
            "initial_prompt": task_prompt,
            "workspace_id": sandbox.sandbox_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "model": self.config.model,
                "timeout": self.config.timeout
            }
        }

        metadata_path = workspace / ".session_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Create data directories
        (workspace / "data" / "gold_traces").mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "failed_traces").mkdir(parents=True, exist_ok=True)

    def build_claude_command(
        self,
        task_prompt: str,
        workspace_path: Path
    ) -> List[str]:
        """Build the Claude CLI command."""
        cmd = [
            self.config.claude_cli_path,
            "--print",  # Print output to stdout
            "--dangerously-skip-permissions",  # Skip permission prompts in sandbox
        ]

        # Add model if specified
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        # Add the task prompt
        cmd.extend(["--prompt", task_prompt])

        return cmd

    def run_task(
        self,
        task_prompt: str,
        task_id: Optional[str] = None,
        repository_url: Optional[str] = None,
        on_output: Optional[Callable[[str], None]] = None
    ) -> TaskResult:
        """
        Run an agent task in a sandboxed environment.

        Args:
            task_prompt: The task description for the agent
            task_id: Optional unique task identifier
            repository_url: Optional git repo to clone
            on_output: Optional callback for streaming output

        Returns:
            TaskResult with execution details
        """
        task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now(timezone.utc)

        # Create sandbox
        sandbox = self.sandbox_manager.create_sandbox(
            task_id=task_id,
            repository_url=repository_url
        )

        try:
            # Start sandbox
            self.sandbox_manager.start_sandbox(sandbox.sandbox_id)

            # Clone repository if specified
            if repository_url:
                self.sandbox_manager.clone_repository(sandbox.sandbox_id, repository_url)

            # Prepare workspace
            self.prepare_workspace(sandbox, task_prompt, task_id)

            # Build command
            cmd = self.build_claude_command(task_prompt, sandbox.workspace_path)

            # Set up environment
            env = os.environ.copy()
            env["CLAUDE_HOOKS_DIR"] = str(sandbox.workspace_path / ".claude" / "hooks")
            env["OUROBOROS_TASK_ID"] = task_id

            # Run the agent
            stdout_lines = []
            stderr_lines = []

            process = subprocess.Popen(
                cmd,
                cwd=str(sandbox.workspace_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            self.active_tasks[task_id] = process

            # Stream output
            def read_output(pipe, lines_list, callback):
                for line in iter(pipe.readline, ''):
                    lines_list.append(line)
                    if callback:
                        callback(line)
                pipe.close()

            stdout_thread = threading.Thread(
                target=read_output,
                args=(process.stdout, stdout_lines, on_output if self.config.stream_output else None)
            )
            stderr_thread = threading.Thread(
                target=read_output,
                args=(process.stderr, stderr_lines, None)
            )

            stdout_thread.start()
            stderr_thread.start()

            # Wait for completion with timeout
            try:
                exit_code = process.wait(timeout=self.config.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = -1

            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Check for trace file
            trace_path = sandbox.workspace_path / "current_session_trace.json"

            return TaskResult(
                task_id=task_id,
                success=exit_code == 0,
                exit_code=exit_code,
                stdout="".join(stdout_lines),
                stderr="".join(stderr_lines),
                duration_seconds=duration,
                trace_path=trace_path if trace_path.exists() else None,
                metadata={
                    "sandbox_id": sandbox.sandbox_id,
                    "workspace_path": str(sandbox.workspace_path)
                }
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            return TaskResult(
                task_id=task_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                error_message=str(e)
            )
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    async def run_task_async(
        self,
        task_prompt: str,
        task_id: Optional[str] = None,
        repository_url: Optional[str] = None,
        on_output: Optional[Callable[[str], None]] = None
    ) -> TaskResult:
        """
        Run an agent task with instrumentation (async).

        Checks task prompt for injection, profiles execution,
        and filters PII from output.

        Args:
            task_prompt: The task description for the agent
            task_id: Optional unique task identifier
            repository_url: Optional git repo to clone
            on_output: Optional callback for streaming output

        Returns:
            TaskResult with execution details and instrumentation stats
        """
        import asyncio

        task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        injection_blocked = False
        pii_redactions = 0

        # Start profiler trace
        trace_id = ""
        if self._instrumentation and self.config.enable_profiling:
            trace_id = self._instrumentation.start_trace(
                f"task:{task_id}",
                metadata={"prompt_preview": task_prompt[:200]}
            )

        try:
            # Check task prompt for injection
            if self._instrumentation and self.config.enable_guardrails:
                is_safe = await self._instrumentation.check_injection(
                    task_prompt,
                    location="runner.task_prompt"
                )
                if not is_safe:
                    injection_blocked = True
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr="Task prompt blocked: potential injection detected",
                        duration_seconds=0.0,
                        error_message="Injection detected in task prompt",
                        injection_blocked=True
                    )

                # Filter PII from task prompt
                original_prompt = task_prompt
                task_prompt = await self._instrumentation.filter_pii(
                    task_prompt,
                    location="runner.task_prompt"
                )
                if task_prompt != original_prompt:
                    pii_redactions += 1

            # Run the sync task in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.run_task(task_prompt, task_id, repository_url, on_output)
            )

            # Filter PII from output
            if self._instrumentation and self.config.filter_pii_from_output:
                if result.stdout:
                    original_stdout = result.stdout
                    result.stdout = await self._instrumentation.filter_pii(
                        result.stdout,
                        location="runner.stdout"
                    )
                    if result.stdout != original_stdout:
                        pii_redactions += 1

                if result.stderr:
                    original_stderr = result.stderr
                    result.stderr = await self._instrumentation.filter_pii(
                        result.stderr,
                        location="runner.stderr"
                    )
                    if result.stderr != original_stderr:
                        pii_redactions += 1

            # Update result with instrumentation stats
            result.injection_blocked = injection_blocked
            result.pii_redactions = pii_redactions

            return result

        finally:
            # End profiler trace
            if trace_id and self._instrumentation:
                self._instrumentation.end_trace(trace_id)

    def run_task_in_existing_sandbox(
        self,
        sandbox: SandboxInstance,
        task_prompt: str,
        task_id: Optional[str] = None
    ) -> TaskResult:
        """
        Run a task in an existing sandbox (for multi-step workflows).
        """
        task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now(timezone.utc)

        # Prepare workspace
        self.prepare_workspace(sandbox, task_prompt, task_id)

        # Execute via docker exec
        cmd_str = f"{self.config.claude_cli_path} --print --prompt '{task_prompt}'"

        result = self.sandbox_manager.execute_command(
            sandbox.sandbox_id,
            cmd_str,
            timeout=self.config.timeout
        )

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        return TaskResult(
            task_id=task_id,
            success=result["exit_code"] == 0,
            exit_code=result["exit_code"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            duration_seconds=duration,
            metadata={"sandbox_id": sandbox.sandbox_id}
        )

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.active_tasks:
            process = self.active_tasks[task_id]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.active_tasks[task_id]
            return True
        return False

    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        return list(self.active_tasks.keys())
