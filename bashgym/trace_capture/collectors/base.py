"""
Base Collector and Record Types

Provides the abstract BaseCollector class and all record/result dataclasses
used by the collector subsystem.  Every concrete collector (subagent, edit,
plan, todo, prompt, debug, environment) inherits from BaseCollector and
produces typed records that can be serialized to JSON.

Patterns follow the existing codebase:
  - @dataclass for all data containers
  - pathlib for filesystem operations
  - platform-aware home directory resolution
  - dataclasses.asdict() for JSON serialization
  - field(default_factory=...) for mutable defaults
"""

import json
import os
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_claude_dir() -> Path:
    """Return the Claude Code configuration directory (~/.claude/)."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".claude"


def get_collected_dir() -> Path:
    """Return the default collected-data directory (~/.bashgym/collected/)."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym" / "collected"


# ---------------------------------------------------------------------------
# Record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CollectorRecord:
    """Base record produced by any collector.

    Every record carries a session identifier, a UTC timestamp, the
    source_type that created it, and an open metadata dict.
    """

    session_id: str
    timestamp: str
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of all fields."""
        return asdict(self)

    def save(self, directory: Path, filename: str) -> Path:
        """Write the record as a JSON file.

        Creates parent directories if they do not exist.
        Returns the path to the written file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return filepath


@dataclass
class SubagentRecord(CollectorRecord):
    """Record for a subagent (task agent) invocation."""

    agent_id: str = ""
    slug: str = ""
    parent_session_id: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    user_prompt: str = ""


@dataclass
class EditRecord(CollectorRecord):
    """Record for a file edit event."""

    file_path: str = ""
    content_hash: str = ""
    versions: List[Dict[str, Any]] = field(default_factory=list)
    diff: str = ""
    total_versions: int = 0


@dataclass
class PlanRecord(CollectorRecord):
    """Record for a plan artifact."""

    plan_name: str = ""
    content: str = ""
    word_count: int = 0


@dataclass
class TodoRecord(CollectorRecord):
    """Record for a todo/task list snapshot."""

    agent_id: str = ""
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    total_tasks: int = 0
    completed_tasks: int = 0
    pending_tasks: int = 0


@dataclass
class PromptRecord(CollectorRecord):
    """Record for a user prompt event."""

    project: str = ""
    prompt_text: str = ""
    pasted_content: str = ""


@dataclass
class DebugRecord(CollectorRecord):
    """Record for debug/observability data from a session."""

    system_prompts: List[str] = field(default_factory=list)
    full_thinking_blocks: List[str] = field(default_factory=list)
    api_call_count: int = 0
    total_latency_ms: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class EnvironmentRecord(CollectorRecord):
    """Record capturing the runtime environment of a session."""

    platform: str = ""
    shell: str = ""
    cwd: str = ""
    git_branch: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    shell_snapshot: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CollectorScanResult:
    """Summary returned by BaseCollector.scan()."""

    source_type: str
    total_found: int
    already_collected: int
    new_available: int
    estimated_size_bytes: int = 0


@dataclass
class CollectorBatchResult:
    """Summary returned by BaseCollector.collect_all()."""

    source_type: str
    collected: int
    skipped: int
    errors: List[str] = field(default_factory=list)
    records: List[CollectorRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BaseCollector abstract class
# ---------------------------------------------------------------------------

class BaseCollector(ABC):
    """Abstract base for all data collectors.

    Each concrete collector reads from a specific data source inside the
    Claude Code local directory tree and writes structured records to the
    collected output directory.

    Parameters
    ----------
    claude_dir : Path
        Root of the Claude Code configuration directory (typically ~/.claude/).
    collected_dir : Path
        Root of the output directory (typically ~/.bashgym/collected/).
    """

    def __init__(self, claude_dir: Path, collected_dir: Path) -> None:
        self.claude_dir = Path(claude_dir)
        self.collected_dir = Path(collected_dir)

    # -- Abstract interface --------------------------------------------------

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Unique string identifying this collector (e.g. 'subagent')."""
        ...

    @abstractmethod
    def scan(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorScanResult:
        """Scan for available data without collecting anything."""
        ...

    @abstractmethod
    def collect(self, session_id: str) -> List[CollectorRecord]:
        """Collect records for a single session."""
        ...

    @abstractmethod
    def collect_all(
        self,
        since: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        """Collect records for all available sessions."""
        ...

    # -- Concrete helpers ----------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Directory where this collector writes its records."""
        return self.collected_dir / self.source_type

    def _state_file(self) -> Path:
        """Path to the scan_state.json deduplication file."""
        return self.output_dir / "scan_state.json"

    def _load_collected_ids(self) -> Set[str]:
        """Load the set of already-collected session IDs from disk."""
        state_file = self._state_file()
        if not state_file.exists():
            return set()
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            return set(data.get("collected_ids", []))
        except (json.JSONDecodeError, IOError, OSError):
            return set()

    def _save_collected_id(self, session_id: str) -> None:
        """Persist a session ID to the deduplication state file."""
        state_file = self._state_file()
        state_file.parent.mkdir(parents=True, exist_ok=True)

        ids = self._load_collected_ids()
        ids.add(session_id)

        state_file.write_text(
            json.dumps({"collected_ids": sorted(ids)}, indent=2),
            encoding="utf-8",
        )
