"""
Orchestrator Data Models

Core data structures for the orchestration system:
- LLMProvider: Supported LLM providers for orchestration
- OrchestratorSpec: User-submitted development specification
- TaskNode: Single task in the decomposition DAG
- WorkerConfig: Configuration for a worker Claude Code session
- WorkerResult: Result from a completed worker session

Module: Orchestrator
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timezone


class LLMProvider(Enum):
    """Supported LLM providers for the orchestrator brain."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


_PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "model": "claude-opus-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/messages",
    },
    "openai": {
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions",
    },
    "gemini": {
        "model": "gemini-2.5-pro",
        "env_key": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "ollama": {
        "model": "qwen2.5-coder:32b",
        "env_key": "",
        "base_url": "http://localhost:11434/api/chat",
    },
}


@dataclass
class LLMConfig:
    """Configuration for the orchestrator's LLM provider.

    Controls which model decomposes specs into task DAGs.
    Workers always use Claude Code CLI regardless of this setting.

    If model is not explicitly set, it uses the provider's default:
    - anthropic: claude-opus-4-6
    - openai: gpt-4o
    - gemini: gemini-2.5-pro
    - ollama: qwen2.5-coder:32b
    """
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = ""  # Empty = use provider default
    api_key: str = ""  # Falls back to env vars per provider
    base_url: Optional[str] = None  # For Ollama or custom endpoints
    temperature: float = 0.3
    max_tokens: int = 4096

    # Class-level reference to provider defaults
    PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: _PROVIDER_DEFAULTS, repr=False
    )

    def __post_init__(self):
        # Auto-resolve model from provider if not explicitly set
        if not self.model:
            defaults = _PROVIDER_DEFAULTS.get(self.provider.value, {})
            self.model = defaults.get("model", "claude-opus-4-6")

    def get_api_key(self) -> str:
        """Resolve API key from config or environment."""
        import os
        if self.api_key:
            return self.api_key
        defaults = _PROVIDER_DEFAULTS.get(self.provider.value, {})
        env_key = defaults.get("env_key", "")
        return os.environ.get(env_key, "") if env_key else ""

    def get_base_url(self) -> str:
        """Resolve base URL from config or provider defaults."""
        if self.base_url:
            return self.base_url
        defaults = _PROVIDER_DEFAULTS.get(self.provider.value, {})
        return defaults.get("base_url", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class TaskStatus(Enum):
    """Status of a task in the orchestration DAG."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for task scheduling."""
    CRITICAL = 1    # Blocks many others
    HIGH = 2        # Core functionality
    NORMAL = 3      # Standard features
    LOW = 4         # Nice-to-have


@dataclass
class OrchestratorSpec:
    """User-submitted development specification.

    This is what the user provides to kick off orchestrated work.
    The orchestrator decomposes this into a TaskDAG.
    """
    title: str
    description: str
    constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    repository: Optional[str] = None
    base_branch: str = "main"
    max_budget_usd: float = 10.0
    max_workers: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "constraints": self.constraints,
            "acceptance_criteria": self.acceptance_criteria,
            "repository": self.repository,
            "base_branch": self.base_branch,
            "max_budget_usd": self.max_budget_usd,
            "max_workers": self.max_workers,
        }


@dataclass
class TaskNode:
    """Single task in the decomposition DAG.

    Represents one unit of work that a Claude Code worker will execute.
    Tasks form a directed acyclic graph via their dependencies.
    """
    id: str
    title: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    files_touched: List[str] = field(default_factory=list)  # Expected files
    estimated_turns: int = 20
    budget_usd: float = 2.0
    worker_prompt: str = ""
    worker_id: Optional[str] = None
    worktree_path: Optional[Path] = None
    result: Optional["WorkerResult"] = None
    retry_count: int = 0
    max_retries: int = 2
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "files_touched": self.files_touched,
            "estimated_turns": self.estimated_turns,
            "budget_usd": self.budget_usd,
            "worker_id": self.worker_id,
            "worktree_path": str(self.worktree_path) if self.worktree_path else None,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
        }


@dataclass
class WorkerConfig:
    """Configuration for a worker Claude Code session."""
    model: str = "sonnet"
    max_turns: int = 30
    max_budget_usd: float = 5.0
    allowed_tools: List[str] = field(default_factory=lambda: [
        "Read", "Edit", "Write", "Bash", "Glob", "Grep",
    ])
    disallowed_tools: List[str] = field(default_factory=list)
    system_prompt_append: str = ""
    worktree_path: Optional[Path] = None
    timeout_seconds: float = 600.0

    def to_cli_args(self) -> List[str]:
        """Convert config to claude CLI arguments."""
        args = [
            "--output-format", "json",
            "--max-turns", str(self.max_turns),
            "--max-budget-usd", str(self.max_budget_usd),
            "--verbose",
        ]
        if self.allowed_tools:
            args.extend(["--allowedTools", ",".join(self.allowed_tools)])
        if self.disallowed_tools:
            args.extend(["--disallowedTools", ",".join(self.disallowed_tools)])
        if self.system_prompt_append:
            args.extend(["--append-system-prompt", self.system_prompt_append])
        return args


@dataclass
class WorkerResult:
    """Result from a completed worker session."""
    task_id: str
    session_id: str
    success: bool
    output: str
    exit_code: int
    duration_seconds: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    trace_path: Optional[Path] = None
    files_modified: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "success": self.success,
            "output": self.output[:2000],  # Truncate for API responses
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "files_modified": self.files_modified,
            "error": self.error,
        }


@dataclass
class MergeResult:
    """Result from merging a worktree branch."""
    task_id: str
    branch: str
    success: bool
    conflicts: List[str] = field(default_factory=list)
    files_merged: List[str] = field(default_factory=list)
    error: Optional[str] = None
