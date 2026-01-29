"""
Core Trace Capture Module

Shared logic for processing and storing traces from any AI coding tool.
This module provides the common interface that all adapters use.
"""

import os
import sys
import json
import uuid
import subprocess
import platform
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

# Cross-platform file locking
if platform.system() == 'Windows':
    import msvcrt

    def lock_file(f, exclusive=False):
        """Lock a file on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)
        except OSError:
            pass  # May already be locked

    def unlock_file(f):
        """Unlock a file on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def lock_file(f, exclusive=False):
        """Lock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def unlock_file(f):
        """Unlock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclass
class RepoInfo:
    """Information about a git repository."""
    path: str
    name: str
    git_remote: Optional[str] = None
    git_branch: Optional[str] = None
    is_git_repo: bool = False

    @classmethod
    def from_path(cls, path: Optional[Path] = None) -> 'RepoInfo':
        """Get repository info from a path (defaults to cwd)."""
        cwd = Path(path) if path else Path.cwd()

        info = cls(
            path=str(cwd),
            name=cwd.name,
        )

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, cwd=cwd, timeout=5
            )
            if result.returncode == 0:
                info.is_git_repo = True

                # Get remote URL
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True, text=True, cwd=cwd, timeout=5
                )
                if result.returncode == 0:
                    info.git_remote = result.stdout.strip()

                # Get current branch
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True, text=True, cwd=cwd, timeout=5
                )
                if result.returncode == 0:
                    info.git_branch = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return info


@dataclass
class TraceStep:
    """A single step in a trace (tool execution)."""
    step_id: str
    timestamp: str
    tool_name: str
    command: str
    output: str = ""
    exit_code: Optional[int] = None
    success: Optional[bool] = None
    cwd: str = ""
    repo: Optional[Dict[str, Any]] = None
    source_tool: str = "unknown"  # claude_code, opencode, aider, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        tool_name: str,
        command: str,
        output: str = "",
        exit_code: Optional[int] = None,
        source_tool: str = "unknown",
        repo_info: Optional[RepoInfo] = None,
        **metadata
    ) -> 'TraceStep':
        """Create a new trace step with auto-generated ID and timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat()
        step_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

        if repo_info is None:
            repo_info = RepoInfo.from_path()

        return cls(
            step_id=step_id,
            timestamp=timestamp,
            tool_name=tool_name,
            command=command,
            output=output[:10000],  # Truncate very long outputs
            exit_code=exit_code,
            success=exit_code == 0 if exit_code is not None else None,
            cwd=os.getcwd(),
            repo=asdict(repo_info),
            source_tool=source_tool,
            metadata=metadata
        )


@dataclass
class TraceSession:
    """A complete trace session with metadata."""
    session_id: str
    timestamp: str
    source_tool: str
    repos: List[Dict[str, Any]]
    primary_repo: Dict[str, Any]
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    trace: List[Dict[str, Any]]
    final_bash_script: str = ""

    @classmethod
    def from_steps(
        cls,
        steps: List[TraceStep],
        source_tool: str = "unknown",
        verification_passed: bool = False,
        **metadata
    ) -> 'TraceSession':
        """Create a session from a list of steps."""
        # Extract unique repos
        repos_dict = {}
        for step in steps:
            repo = step.repo or {}
            repo_path = repo.get("path", "")
            if repo_path and repo_path not in repos_dict:
                repos_dict[repo_path] = repo
        repos = list(repos_dict.values())
        primary_repo = repos[0] if repos else {"name": "unknown", "path": os.getcwd()}

        # Calculate summary
        total_steps = len(steps)
        successful_steps = sum(1 for s in steps if s.success is True)
        failed_steps = sum(1 for s in steps if s.success is False)

        # Extract bash commands
        bash_commands = [
            s.command for s in steps
            if s.tool_name.lower() == "bash" and s.success is True
        ]

        return cls(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_tool=source_tool,
            repos=repos,
            primary_repo=primary_repo,
            metadata={
                "verification_passed": verification_passed,
                **metadata
            },
            summary={
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
                "repos_count": len(repos)
            },
            trace=[asdict(s) for s in steps],
            final_bash_script="\n".join(bash_commands)
        )


class TraceCapture:
    """
    Main trace capture class.

    Handles reading/writing traces to the global ~/.bashgym/ directory.
    Used by all adapters (Claude Code, OpenCode, etc.)
    """

    RELEVANT_TOOLS = {"Bash", "Edit", "Write", "Read", "bash", "edit", "write", "read"}

    def __init__(self):
        self.bashgym_dir = self._get_bashgym_dir()
        self.traces_dir = self.bashgym_dir / "traces"
        self.gold_traces_dir = self.bashgym_dir / "gold_traces"
        self.failed_traces_dir = self.bashgym_dir / "failed_traces"

        # Ensure directories exist
        for d in [self.traces_dir, self.gold_traces_dir, self.failed_traces_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_bashgym_dir() -> Path:
        """Get the global Bash Gym directory (~/.bashgym/)."""
        if platform.system() == 'Windows':
            base = Path(os.environ.get("USERPROFILE", ""))
        else:
            base = Path.home()
        return base / ".bashgym"

    def get_session_id(self) -> str:
        """Get or create a session ID for the current session."""
        session_file = self.bashgym_dir / "current_session_id"

        if session_file.exists():
            return session_file.read_text().strip()

        session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_file.write_text(session_id)
        return session_id

    def get_trace_file(self) -> Path:
        """Get the path to the current session's trace file."""
        session_id = self.get_session_id()
        return self.traces_dir / f"session_{session_id}.json"

    def load_trace(self) -> List[Dict[str, Any]]:
        """Load the current trace from file."""
        trace_file = self.get_trace_file()
        if not trace_file.exists():
            return []

        try:
            with open(trace_file, 'r') as f:
                lock_file(f, exclusive=False)
                content = f.read()
                unlock_file(f)

            if not content.strip():
                return []
            return json.loads(content)
        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"Warning: Could not load trace file: {e}", file=sys.stderr)
            return []

    def save_trace(self, trace: List[Dict[str, Any]]) -> None:
        """Save the trace to file."""
        trace_file = self.get_trace_file()
        try:
            with open(trace_file, 'w') as f:
                lock_file(f, exclusive=True)
                json.dump(trace, f, indent=2, ensure_ascii=False)
                unlock_file(f)
        except (IOError, OSError) as e:
            print(f"Error: Could not write trace file: {e}", file=sys.stderr)

    def append_step(self, step: TraceStep) -> None:
        """Append a step to the current trace."""
        trace = self.load_trace()
        trace.append(asdict(step))
        self.save_trace(trace)

        repo_name = step.repo.get("name", "unknown") if step.repo else "unknown"
        print(f"[BashGym] Captured: {step.tool_name} - {step.command[:50]}... ({repo_name})")

    def is_relevant_tool(self, tool_name: str) -> bool:
        """Check if a tool should be captured."""
        return tool_name in self.RELEVANT_TOOLS

    def promote_to_gold(self, verification_passed: bool = True, **metadata) -> Optional[Path]:
        """Promote the current trace to gold_traces."""
        trace_data = self.load_trace()
        if not trace_data:
            return None

        # Convert to TraceStep objects
        steps = []
        for step_dict in trace_data:
            step = TraceStep(**{k: v for k, v in step_dict.items() if k in TraceStep.__dataclass_fields__})
            steps.append(step)

        # Determine source tool from steps
        source_tools = set(s.source_tool for s in steps if s.source_tool != "unknown")
        source_tool = source_tools.pop() if source_tools else "unknown"

        # Create session
        session = TraceSession.from_steps(
            steps,
            source_tool=source_tool,
            verification_passed=verification_passed,
            **metadata
        )

        # Generate filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        repo_name = session.primary_repo.get("name", "unknown")[:20]

        if verification_passed:
            filename = f"gold_{repo_name}_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            destination = self.gold_traces_dir / filename
            status = "✓"
        else:
            filename = f"failed_{repo_name}_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            destination = self.failed_traces_dir / filename
            status = "✗"

        try:
            with open(destination, 'w') as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
            print(f"[BashGym] {status} Trace saved to: {destination}")
            return destination
        except IOError as e:
            print(f"Error writing trace: {e}", file=sys.stderr)
            return None

    def cleanup_session(self) -> None:
        """Clean up the current session files."""
        session_file = self.bashgym_dir / "current_session_id"
        trace_file = self.get_trace_file()

        for f in [session_file, trace_file]:
            try:
                if f.exists():
                    f.unlink()
            except IOError:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about captured traces."""
        gold_count = len(list(self.gold_traces_dir.glob("*.json")))
        failed_count = len(list(self.failed_traces_dir.glob("*.json")))
        pending_count = len(list(self.traces_dir.glob("*.json")))

        return {
            "gold_traces": gold_count,
            "failed_traces": failed_count,
            "pending_traces": pending_count,
            "total": gold_count + failed_count + pending_count
        }
