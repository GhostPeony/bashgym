# Comprehensive .claude Data Collection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build modular collectors that capture all data from `~/.claude/` (subagents, file edits, plans, todos, prompts, debug logs, environments) and wire them into Peony's tools and the bashgym-setup CLI.

**Architecture:** Each collector implements a `BaseCollector` interface, produces typed dataclass records tagged by session_id, and stores them under `~/.bashgym/collected/`. A `ClaudeDataScanner` orchestrates all collectors. Peony's `import_traces` tool and `bashgym-setup` CLI call the scanner.

**Tech Stack:** Python 3.10+, dataclasses, pathlib, json, existing BashGym patterns (TraceStep/TraceSession from `bashgym/trace_capture/core.py`)

**Design Doc:** `docs/plans/2026-02-25-comprehensive-claude-data-collection-design.md`

---

## Task 1: BaseCollector Abstract Class + Record Types

**Files:**
- Create: `bashgym/trace_capture/collectors/__init__.py`
- Create: `bashgym/trace_capture/collectors/base.py`
- Create: `tests/test_collectors.py`

**Step 1: Write the test scaffolding**

```python
# tests/test_collectors.py
import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import asdict


class TestBaseCollector:
    """Tests for BaseCollector abstract interface and record types."""

    def test_collector_record_has_session_id_and_timestamp(self):
        from bashgym.trace_capture.collectors.base import CollectorRecord
        record = CollectorRecord(
            session_id="abc-123",
            timestamp="2026-02-25T10:00:00Z",
            source_type="test",
        )
        assert record.session_id == "abc-123"
        assert record.timestamp == "2026-02-25T10:00:00Z"
        assert record.source_type == "test"

    def test_collector_record_serializes_to_dict(self):
        from bashgym.trace_capture.collectors.base import CollectorRecord
        record = CollectorRecord(
            session_id="abc-123",
            timestamp="2026-02-25T10:00:00Z",
            source_type="test",
        )
        d = asdict(record)
        assert d["session_id"] == "abc-123"
        assert d["source_type"] == "test"

    def test_scan_result_contains_counts(self):
        from bashgym.trace_capture.collectors.base import CollectorScanResult
        result = CollectorScanResult(
            source_type="subagents",
            total_found=15,
            already_collected=3,
            new_available=12,
        )
        assert result.new_available == 12

    def test_batch_result_tracks_collected_and_errors(self):
        from bashgym.trace_capture.collectors.base import CollectorBatchResult
        result = CollectorBatchResult(
            source_type="subagents",
            collected=10,
            skipped=3,
            errors=["file not found: xyz.jsonl"],
        )
        assert result.collected == 10
        assert len(result.errors) == 1

    def test_base_collector_is_abstract(self):
        from bashgym.trace_capture.collectors.base import BaseCollector
        with pytest.raises(TypeError):
            BaseCollector()  # Can't instantiate abstract class
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_collectors.py::TestBaseCollector -v`
Expected: FAIL with ImportError

**Step 3: Implement BaseCollector and record types**

```python
# bashgym/trace_capture/collectors/base.py
"""
Base collector interface and shared record types.

All collectors implement BaseCollector and produce CollectorRecord subclasses
tagged with session_id and timestamp for cross-linking.
"""

import json
import platform
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Set


def get_claude_dir() -> Path:
    """Get Claude Code's data directory (~/.claude/)."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".claude"


def get_collected_dir() -> Path:
    """Get the BashGym collected data directory (~/.bashgym/collected/)."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("USERPROFILE", ""))
    else:
        base = Path.home()
    return base / ".bashgym" / "collected"


# --- Record Types ---

@dataclass
class CollectorRecord:
    """Base record produced by any collector. Every record carries session_id + timestamp."""
    session_id: str
    timestamp: str
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, directory: Path, filename: str) -> Path:
        """Save this record to a JSON file."""
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return filepath


@dataclass
class SubagentRecord(CollectorRecord):
    """A subagent conversation from projects/*/subagents/*.jsonl."""
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
    """A file edit from file-history/<session-id>/<hash>@v<n>."""
    file_path: str = ""
    content_hash: str = ""
    versions: List[Dict[str, Any]] = field(default_factory=list)  # [{version: 1, content: "..."}, ...]
    diff: str = ""  # Unified diff between first and last version
    total_versions: int = 0


@dataclass
class PlanRecord(CollectorRecord):
    """An implementation plan from plans/*.md."""
    plan_name: str = ""
    content: str = ""
    word_count: int = 0


@dataclass
class TodoRecord(CollectorRecord):
    """A todo/task list from todos/<session-id>-agent-<agent-id>.json."""
    agent_id: str = ""
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    total_tasks: int = 0
    completed_tasks: int = 0
    pending_tasks: int = 0


@dataclass
class PromptRecord(CollectorRecord):
    """A user prompt from history.jsonl or paste-cache/."""
    project: str = ""
    prompt_text: str = ""
    pasted_content: str = ""


@dataclass
class DebugRecord(CollectorRecord):
    """Processed debug log entry from debug/<session-id>.txt."""
    system_prompts: List[str] = field(default_factory=list)
    full_thinking_blocks: List[str] = field(default_factory=list)
    api_call_count: int = 0
    total_latency_ms: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class EnvironmentRecord(CollectorRecord):
    """Session environment from session-env/ and shell-snapshots/."""
    platform: str = ""
    shell: str = ""
    cwd: str = ""
    git_branch: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    shell_snapshot: str = ""


# --- Scan/Batch Result Types ---

@dataclass
class CollectorScanResult:
    """Result of a dry-run scan: what's available to collect."""
    source_type: str
    total_found: int = 0
    already_collected: int = 0
    new_available: int = 0
    estimated_size_bytes: int = 0


@dataclass
class CollectorBatchResult:
    """Result of a bulk collect operation."""
    source_type: str
    collected: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    records: List[CollectorRecord] = field(default_factory=list)


# --- Abstract Base ---

class BaseCollector(ABC):
    """Abstract base for all .claude data collectors."""

    def __init__(self):
        self.claude_dir = get_claude_dir()
        self.collected_dir = get_collected_dir()
        self._collected_ids: Optional[Set[str]] = None

    @property
    @abstractmethod
    def source_type(self) -> str:
        """The type of data this collector handles (e.g., 'subagents', 'edits')."""

    @property
    def output_dir(self) -> Path:
        """Directory where this collector stores its records."""
        return self.collected_dir / self.source_type

    def _load_collected_ids(self) -> Set[str]:
        """Load set of already-collected record IDs from scan_state."""
        state_file = self.collected_dir / "scan_state.json"
        if not state_file.exists():
            return set()
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
            return set(data.get(self.source_type, []))
        except (json.JSONDecodeError, IOError):
            return set()

    def _save_collected_id(self, record_id: str) -> None:
        """Mark a record as collected in scan_state."""
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()
        self._collected_ids.add(record_id)

        state_file = self.collected_dir / "scan_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            if state_file.exists():
                with open(state_file, "r") as f:
                    data = json.load(f)
            else:
                data = {}
            data[self.source_type] = list(self._collected_ids)
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not save scan state: {e}")

    @abstractmethod
    def scan(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorScanResult:
        """Find all uncollected records from this source (dry-run)."""

    @abstractmethod
    def collect(self, session_id: str) -> List[CollectorRecord]:
        """Collect records for a specific session."""

    @abstractmethod
    def collect_all(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        """Bulk collection with filtering."""
```

```python
# bashgym/trace_capture/collectors/__init__.py
"""
Modular collectors for .claude data sources.

Each collector handles one data type from ~/.claude/ and produces
typed records tagged with session_id for cross-linking.
"""

from .base import (
    BaseCollector,
    CollectorRecord,
    CollectorScanResult,
    CollectorBatchResult,
    SubagentRecord,
    EditRecord,
    PlanRecord,
    TodoRecord,
    PromptRecord,
    DebugRecord,
    EnvironmentRecord,
    get_claude_dir,
    get_collected_dir,
)

__all__ = [
    "BaseCollector",
    "CollectorRecord",
    "CollectorScanResult",
    "CollectorBatchResult",
    "SubagentRecord",
    "EditRecord",
    "PlanRecord",
    "TodoRecord",
    "PromptRecord",
    "DebugRecord",
    "EnvironmentRecord",
    "get_claude_dir",
    "get_collected_dir",
]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collectors.py::TestBaseCollector -v`
Expected: All 5 PASS

**Step 5: Commit**

```bash
git add bashgym/trace_capture/collectors/ tests/test_collectors.py
git commit -m "feat(collectors): add BaseCollector abstract class and record types"
```

---

## Task 2: SubagentCollector (P0 — highest value)

**Files:**
- Create: `bashgym/trace_capture/collectors/subagent.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `tests/test_collectors.py`

**Context:** Subagent JSONL files live at `projects/<slug>/<session-id>/subagents/agent-<id>.jsonl`. They use the same format as parent session JSONL — events with `type: "user"` and `type: "assistant"`. Each subagent file has fields like `agentId`, `sessionId`, `slug`, `gitBranch`. The existing `ClaudeSessionImporter.parse_session_file()` already knows how to parse this format.

**Step 1: Write the tests**

```python
# Add to tests/test_collectors.py

class TestSubagentCollector:
    """Tests for SubagentCollector."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create a mock .claude directory with subagent files."""
        projects = tmp_path / "projects"
        project = projects / "C--Users-Cade-projects-myapp"
        session_dir = project / "abc12345-1234-5678-abcd-1234567890ab"
        subagents = session_dir / "subagents"
        subagents.mkdir(parents=True)

        # Create a minimal subagent JSONL
        subagent_file = subagents / "agent-a11b22c33d44e55f6.jsonl"
        lines = [
            json.dumps({
                "parentUuid": None,
                "isSidechain": True,
                "userType": "external",
                "cwd": "C:\\Users\\Cade\\projects\\myapp",
                "sessionId": "abc12345-1234-5678-abcd-1234567890ab",
                "version": "2.1.51",
                "gitBranch": "main",
                "agentId": "a11b22c33d44e55f6",
                "slug": "iridescent-wiggling-adleman",
                "type": "user",
                "message": {"role": "user", "content": "Find all API endpoints in this project"},
            }),
            json.dumps({
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [
                        {"type": "text", "text": "I'll search for API endpoints."},
                        {"type": "tool_use", "id": "toolu_01", "name": "Grep", "input": {"pattern": "@app\\.(get|post|put|delete)", "path": "src/"}},
                    ],
                    "usage": {"input_tokens": 500, "output_tokens": 100},
                },
            }),
        ]
        subagent_file.write_text("\n".join(lines), encoding="utf-8")

        # Also create the parent session JSONL (empty is fine for this test)
        session_jsonl = project / "abc12345-1234-5678-abcd-1234567890ab.jsonl"
        session_jsonl.write_text("", encoding="utf-8")

        return tmp_path

    def test_scan_finds_subagent_files(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.subagent import SubagentCollector
        collector = SubagentCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.scan()
        assert result.source_type == "subagents"
        assert result.total_found == 1
        assert result.new_available == 1

    def test_collect_parses_subagent_jsonl(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.subagent import SubagentCollector
        collector = SubagentCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        records = collector.collect("abc12345-1234-5678-abcd-1234567890ab")
        assert len(records) == 1
        record = records[0]
        assert record.agent_id == "a11b22c33d44e55f6"
        assert record.slug == "iridescent-wiggling-adleman"
        assert record.parent_session_id == "abc12345-1234-5678-abcd-1234567890ab"
        assert record.total_tool_calls >= 1
        assert "Grep" in record.tools_used

    def test_collect_all_returns_batch_result(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.subagent import SubagentCollector
        collector = SubagentCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.collect_all()
        assert result.source_type == "subagents"
        assert result.collected == 1
        assert len(result.records) == 1

    def test_collect_skips_already_collected(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.subagent import SubagentCollector
        collector = SubagentCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        # Collect once
        collector.collect_all()
        # Collect again — should skip
        result = collector.collect_all()
        assert result.collected == 0
        assert result.skipped == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collectors.py::TestSubagentCollector -v`
Expected: FAIL with ImportError

**Step 3: Implement SubagentCollector**

```python
# bashgym/trace_capture/collectors/subagent.py
"""
SubagentCollector — parses subagent conversation JSONL files.

Source: ~/.claude/projects/<slug>/<session-id>/subagents/agent-<id>.jsonl
These are full conversation transcripts for Task/Explore subagents.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Set

from .base import (
    BaseCollector,
    CollectorRecord,
    CollectorScanResult,
    CollectorBatchResult,
    SubagentRecord,
)


class SubagentCollector(BaseCollector):
    """Collects subagent conversation traces from .claude/projects/*/subagents/."""

    @property
    def source_type(self) -> str:
        return "subagents"

    def _find_subagent_files(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> List[tuple]:
        """Find all subagent JSONL files. Returns list of (path, session_id, agent_id)."""
        projects_dir = self.claude_dir / "projects"
        if not projects_dir.exists():
            return []

        results = []
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            if project_filter and project_filter.lower() not in project_dir.name.lower():
                continue

            for session_dir in project_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                subagents_dir = session_dir / "subagents"
                if not subagents_dir.exists():
                    continue

                for jsonl_file in subagents_dir.glob("agent-*.jsonl"):
                    if since:
                        mtime = datetime.fromtimestamp(
                            jsonl_file.stat().st_mtime, tz=timezone.utc
                        )
                        if mtime < since:
                            continue

                    agent_id = jsonl_file.stem.replace("agent-", "")
                    session_id = session_dir.name
                    results.append((jsonl_file, session_id, agent_id))

        return results

    def _parse_subagent_file(
        self, filepath: Path, session_id: str, agent_id: str
    ) -> SubagentRecord:
        """Parse a subagent JSONL file into a SubagentRecord."""
        models_used: Set[str] = set()
        tools_used: Set[str] = set()
        steps = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_tool_calls = 0
        slug = ""
        user_prompt = ""
        timestamp = ""

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not slug and "slug" in event:
                    slug = event["slug"]
                if not timestamp and "timestamp" in event:
                    ts = event.get("timestamp")
                    if isinstance(ts, (int, float)):
                        timestamp = datetime.fromtimestamp(
                            ts / 1000, tz=timezone.utc
                        ).isoformat()
                    elif isinstance(ts, str):
                        timestamp = ts

                event_type = event.get("type")

                if event_type == "user":
                    message = event.get("message", {})
                    content = message.get("content", "")
                    if isinstance(content, str) and not user_prompt:
                        user_prompt = content[:2000]
                    elif isinstance(content, list) and not user_prompt:
                        texts = [
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                            if not isinstance(b, dict) or b.get("type") == "text"
                        ]
                        user_prompt = "\n".join(texts)[:2000]

                elif event_type == "assistant":
                    msg = event.get("message", {})
                    model = msg.get("model", "")
                    usage = msg.get("usage", {})

                    if model:
                        models_used.add(model)
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)

                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "tool_use":
                                tool_name = item.get("name", "")
                                tools_used.add(tool_name)
                                total_tool_calls += 1
                                steps.append({
                                    "tool_name": tool_name,
                                    "input": item.get("input", {}),
                                    "tool_use_id": item.get("id", ""),
                                    "model": model,
                                })

        return SubagentRecord(
            session_id=session_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            source_type=self.source_type,
            agent_id=agent_id,
            slug=slug,
            parent_session_id=session_id,
            steps=steps,
            models_used=sorted(models_used),
            tools_used=sorted(tools_used),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tool_calls=total_tool_calls,
            user_prompt=user_prompt,
        )

    def scan(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorScanResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        files = self._find_subagent_files(since=since, project_filter=project_filter)
        already = sum(
            1 for _, sid, aid in files if f"{sid}/{aid}" in self._collected_ids
        )

        return CollectorScanResult(
            source_type=self.source_type,
            total_found=len(files),
            already_collected=already,
            new_available=len(files) - already,
        )

    def collect(self, session_id: str) -> List[CollectorRecord]:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        files = self._find_subagent_files()
        records = []

        for filepath, sid, aid in files:
            if sid != session_id:
                continue
            record_id = f"{sid}/{aid}"
            if record_id in self._collected_ids:
                continue

            record = self._parse_subagent_file(filepath, sid, aid)
            record.save(self.output_dir, f"{aid}.json")
            self._save_collected_id(record_id)
            records.append(record)

        return records

    def collect_all(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        files = self._find_subagent_files(since=since, project_filter=project_filter)
        collected = 0
        skipped = 0
        errors = []
        records = []

        for filepath, sid, aid in files:
            record_id = f"{sid}/{aid}"
            if record_id in self._collected_ids:
                skipped += 1
                continue

            try:
                record = self._parse_subagent_file(filepath, sid, aid)
                record.save(self.output_dir, f"{aid}.json")
                self._save_collected_id(record_id)
                records.append(record)
                collected += 1
            except Exception as e:
                errors.append(f"{filepath}: {e}")

        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected,
            skipped=skipped,
            errors=errors,
            records=records,
        )
```

**Step 4: Update `__init__.py`**

Add to `bashgym/trace_capture/collectors/__init__.py`:
```python
from .subagent import SubagentCollector
# Add to __all__: "SubagentCollector"
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_collectors.py::TestSubagentCollector -v`
Expected: All 4 PASS

**Step 6: Commit**

```bash
git add bashgym/trace_capture/collectors/subagent.py bashgym/trace_capture/collectors/__init__.py tests/test_collectors.py
git commit -m "feat(collectors): add SubagentCollector for subagent conversation traces"
```

---

## Task 3: EditCollector (P0 — DPO pair source)

**Files:**
- Create: `bashgym/trace_capture/collectors/edit.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `tests/test_collectors.py`

**Context:** File history lives at `file-history/<session-id>/<content-hash>@v<version>`. Content hashes are hex strings. `@v1` is the file before an edit, `@v2` is after. The content is the raw file text. The same hash can have multiple versions if edited repeatedly. Map file hashes to actual file paths using the session JSONL's tool_use inputs (Edit/Write tools contain `file_path`).

**Step 1: Write the tests**

```python
class TestEditCollector:
    """Tests for EditCollector."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock file-history with versioned edits."""
        fh = tmp_path / "file-history" / "abc12345-1234-5678-abcd-1234567890ab"
        fh.mkdir(parents=True)

        # Simulate file edit: v1 (before) and v2 (after)
        (fh / "a1b2c3d4e5f6g7h8@v1").write_text("def hello():\n    pass\n", encoding="utf-8")
        (fh / "a1b2c3d4e5f6g7h8@v2").write_text("def hello():\n    print('hello')\n", encoding="utf-8")

        # Another file with 3 versions
        (fh / "f0f0f0f0f0f0f0f0@v1").write_text("# old\n", encoding="utf-8")
        (fh / "f0f0f0f0f0f0f0f0@v2").write_text("# middle\n", encoding="utf-8")
        (fh / "f0f0f0f0f0f0f0f0@v3").write_text("# final\n", encoding="utf-8")

        return tmp_path

    def test_scan_finds_edit_sessions(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.edit import EditCollector
        collector = EditCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.scan()
        assert result.source_type == "edits"
        assert result.total_found >= 1

    def test_collect_groups_versions(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.edit import EditCollector
        collector = EditCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        records = collector.collect("abc12345-1234-5678-abcd-1234567890ab")
        assert len(records) == 2  # Two distinct content hashes

        # Find the file with 3 versions
        multi = [r for r in records if r.total_versions == 3]
        assert len(multi) == 1
        assert multi[0].versions[0]["content"] == "# old\n"
        assert multi[0].versions[-1]["content"] == "# final\n"

    def test_collect_produces_diff(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.edit import EditCollector
        collector = EditCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        records = collector.collect("abc12345-1234-5678-abcd-1234567890ab")
        two_ver = [r for r in records if r.total_versions == 2][0]
        assert "pass" in two_ver.diff  # v1 content
        assert "hello" in two_ver.diff  # v2 content
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collectors.py::TestEditCollector -v`
Expected: FAIL with ImportError

**Step 3: Implement EditCollector**

```python
# bashgym/trace_capture/collectors/edit.py
"""
EditCollector — parses versioned file history snapshots.

Source: ~/.claude/file-history/<session-id>/<content-hash>@v<n>
Produces EditRecord with before/after content and unified diffs.
"""

import difflib
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

from .base import (
    BaseCollector,
    CollectorRecord,
    CollectorScanResult,
    CollectorBatchResult,
    EditRecord,
)

VERSION_PATTERN = re.compile(r"^(.+)@v(\d+)$")


class EditCollector(BaseCollector):
    """Collects file edit history from .claude/file-history/."""

    @property
    def source_type(self) -> str:
        return "edits"

    def _find_edit_sessions(
        self,
        since: Optional[datetime] = None,
    ) -> List[tuple]:
        """Find session dirs in file-history/. Returns list of (dir_path, session_id)."""
        fh_dir = self.claude_dir / "file-history"
        if not fh_dir.exists():
            return []

        results = []
        for session_dir in fh_dir.iterdir():
            if not session_dir.is_dir():
                continue
            if since:
                mtime = datetime.fromtimestamp(
                    session_dir.stat().st_mtime, tz=timezone.utc
                )
                if mtime < since:
                    continue
            results.append((session_dir, session_dir.name))
        return results

    def _parse_session_edits(self, session_dir: Path, session_id: str) -> List[EditRecord]:
        """Parse all versioned files in a session's file-history directory."""
        # Group files by content hash
        hash_versions: Dict[str, Dict[int, Path]] = defaultdict(dict)

        for entry in session_dir.iterdir():
            if not entry.is_file():
                continue
            match = VERSION_PATTERN.match(entry.name)
            if not match:
                continue
            content_hash = match.group(1)
            version = int(match.group(2))
            hash_versions[content_hash][version] = entry

        records = []
        for content_hash, versions_dict in hash_versions.items():
            sorted_versions = sorted(versions_dict.items())
            version_data = []
            for ver_num, ver_path in sorted_versions:
                try:
                    content = ver_path.read_text(encoding="utf-8", errors="replace")
                except IOError:
                    content = ""
                version_data.append({
                    "version": ver_num,
                    "content": content,
                })

            # Generate unified diff between first and last version
            diff = ""
            if len(version_data) >= 2:
                first = version_data[0]["content"].splitlines(keepends=True)
                last = version_data[-1]["content"].splitlines(keepends=True)
                diff_lines = difflib.unified_diff(
                    first, last,
                    fromfile=f"v{version_data[0]['version']}",
                    tofile=f"v{version_data[-1]['version']}",
                )
                diff = "".join(diff_lines)

            mtime = max(p.stat().st_mtime for p in versions_dict.values())

            records.append(EditRecord(
                session_id=session_id,
                timestamp=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                source_type=self.source_type,
                content_hash=content_hash,
                versions=version_data,
                diff=diff,
                total_versions=len(version_data),
            ))

        return records

    def scan(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorScanResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        sessions = self._find_edit_sessions(since=since)
        already = sum(1 for _, sid in sessions if sid in self._collected_ids)

        return CollectorScanResult(
            source_type=self.source_type,
            total_found=len(sessions),
            already_collected=already,
            new_available=len(sessions) - already,
        )

    def collect(self, session_id: str) -> List[CollectorRecord]:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        fh_dir = self.claude_dir / "file-history" / session_id
        if not fh_dir.exists():
            return []

        records = self._parse_session_edits(fh_dir, session_id)
        for record in records:
            record.save(
                self.output_dir,
                f"{session_id}_{record.content_hash[:16]}.json",
            )
        if records:
            self._save_collected_id(session_id)
        return records

    def collect_all(
        self,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        sessions = self._find_edit_sessions(since=since)
        collected = 0
        skipped = 0
        errors = []
        all_records = []

        for session_dir, session_id in sessions:
            if session_id in self._collected_ids:
                skipped += 1
                continue
            try:
                records = self._parse_session_edits(session_dir, session_id)
                for record in records:
                    record.save(
                        self.output_dir,
                        f"{session_id}_{record.content_hash[:16]}.json",
                    )
                all_records.extend(records)
                self._save_collected_id(session_id)
                collected += 1
            except Exception as e:
                errors.append(f"{session_dir}: {e}")

        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected,
            skipped=skipped,
            errors=errors,
            records=all_records,
        )
```

**Step 4: Update `__init__.py`**

Add to `bashgym/trace_capture/collectors/__init__.py`:
```python
from .edit import EditCollector
# Add to __all__: "EditCollector"
```

**Step 5: Run tests**

Run: `pytest tests/test_collectors.py::TestEditCollector -v`
Expected: All 3 PASS

**Step 6: Commit**

```bash
git add bashgym/trace_capture/collectors/edit.py bashgym/trace_capture/collectors/__init__.py tests/test_collectors.py
git commit -m "feat(collectors): add EditCollector for file-history diffs and DPO pairs"
```

---

## Task 4: PlanCollector (P1)

**Files:**
- Create: `bashgym/trace_capture/collectors/plan.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `tests/test_collectors.py`

**Context:** Plans are markdown files at `~/.claude/plans/<alliterative-name>.md`. They don't contain session IDs directly — linking requires matching by timestamp and project context. For Phase 1, store them as standalone records. Phase 2 links them to sessions via the cross-reference index.

**Step 1: Write the tests**

```python
class TestPlanCollector:

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        plans = tmp_path / "plans"
        plans.mkdir()
        (plans / "clever-jumping-fox.md").write_text(
            "# Auth System Plan\n\n## Problem\nNeed auth.\n\n## Approach\nUse JWT.\n",
            encoding="utf-8",
        )
        (plans / "wiggly-napping-pixel.md").write_text(
            "# Dashboard Redesign\n\nShort plan.\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_scan_finds_plan_files(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.plan import PlanCollector
        collector = PlanCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.scan()
        assert result.source_type == "plans"
        assert result.total_found == 2

    def test_collect_all_parses_markdown(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.plan import PlanCollector
        collector = PlanCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.collect_all()
        assert result.collected == 2
        auth_plan = [r for r in result.records if "Auth" in r.content][0]
        assert auth_plan.plan_name == "clever-jumping-fox"
        assert auth_plan.word_count > 5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_collectors.py::TestPlanCollector -v`

**Step 3: Implement PlanCollector**

```python
# bashgym/trace_capture/collectors/plan.py
"""
PlanCollector — imports implementation plans from ~/.claude/plans/*.md.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from .base import (
    BaseCollector,
    CollectorRecord,
    CollectorScanResult,
    CollectorBatchResult,
    PlanRecord,
)


class PlanCollector(BaseCollector):
    """Collects implementation plans from .claude/plans/."""

    @property
    def source_type(self) -> str:
        return "plans"

    def _find_plan_files(
        self, since: Optional[datetime] = None,
    ) -> List[Path]:
        plans_dir = self.claude_dir / "plans"
        if not plans_dir.exists():
            return []
        results = []
        for md_file in plans_dir.glob("*.md"):
            if since:
                mtime = datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc)
                if mtime < since:
                    continue
            results.append(md_file)
        return results

    def _parse_plan(self, filepath: Path) -> PlanRecord:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
        plan_name = filepath.stem

        return PlanRecord(
            session_id="",  # Plans don't carry session IDs directly
            timestamp=mtime.isoformat(),
            source_type=self.source_type,
            plan_name=plan_name,
            content=content,
            word_count=len(content.split()),
        )

    def scan(self, since=None, project_filter=None) -> CollectorScanResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()
        files = self._find_plan_files(since=since)
        already = sum(1 for f in files if f.stem in self._collected_ids)
        return CollectorScanResult(
            source_type=self.source_type,
            total_found=len(files),
            already_collected=already,
            new_available=len(files) - already,
        )

    def collect(self, session_id: str) -> List[CollectorRecord]:
        # Plans don't have session IDs — collect by plan name instead
        return []

    def collect_all(self, since=None, project_filter=None) -> CollectorBatchResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()
        files = self._find_plan_files(since=since)
        collected = 0
        skipped = 0
        errors = []
        records = []
        for filepath in files:
            if filepath.stem in self._collected_ids:
                skipped += 1
                continue
            try:
                record = self._parse_plan(filepath)
                record.save(self.output_dir, f"{filepath.stem}.json")
                self._save_collected_id(filepath.stem)
                records.append(record)
                collected += 1
            except Exception as e:
                errors.append(f"{filepath}: {e}")
        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected, skipped=skipped, errors=errors, records=records,
        )
```

**Step 4: Update `__init__.py`, run tests, commit**

Run: `pytest tests/test_collectors.py::TestPlanCollector -v`
Expected: All 2 PASS

```bash
git add bashgym/trace_capture/collectors/plan.py bashgym/trace_capture/collectors/__init__.py tests/test_collectors.py
git commit -m "feat(collectors): add PlanCollector for implementation plan documents"
```

---

## Task 5: PromptCollector (P1)

**Files:**
- Create: `bashgym/trace_capture/collectors/prompt.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `tests/test_collectors.py`

**Context:** `history.jsonl` contains every user prompt across all sessions: `{"display":"prompt text","pastedContents":{},"timestamp":1759817571796,"project":"C:\\Users\\..."}`. `paste-cache/<hash>.txt` contains pasted code/text. The `pastedContents` field in history links to paste-cache entries.

**Step 1: Write the tests**

```python
class TestPromptCollector:

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        history = tmp_path / "history.jsonl"
        lines = [
            json.dumps({"display": "Fix the auth bug", "pastedContents": {}, "timestamp": 1759817571796, "project": "C:\\Users\\Cade\\projects\\myapp"}),
            json.dumps({"display": "Add dark mode", "pastedContents": {"abc123": True}, "timestamp": 1759817600000, "project": "C:\\Users\\Cade\\projects\\myapp"}),
        ]
        history.write_text("\n".join(lines), encoding="utf-8")

        paste = tmp_path / "paste-cache"
        paste.mkdir()
        (paste / "abc123.txt").write_text("const theme = 'dark';", encoding="utf-8")

        return tmp_path

    def test_collect_all_reads_history(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.prompt import PromptCollector
        collector = PromptCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.collect_all()
        assert result.collected == 2
        auth = [r for r in result.records if "auth" in r.prompt_text.lower()][0]
        assert auth.project.endswith("myapp")

    def test_collect_links_paste_cache(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.prompt import PromptCollector
        collector = PromptCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.collect_all()
        dark = [r for r in result.records if "dark" in r.prompt_text.lower()][0]
        assert "theme" in dark.pasted_content
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement PromptCollector**

```python
# bashgym/trace_capture/collectors/prompt.py
"""
PromptCollector — imports user prompts from history.jsonl and paste-cache/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from .base import (
    BaseCollector,
    CollectorRecord,
    CollectorScanResult,
    CollectorBatchResult,
    PromptRecord,
)


class PromptCollector(BaseCollector):
    """Collects user prompts from .claude/history.jsonl and paste-cache/."""

    @property
    def source_type(self) -> str:
        return "prompts"

    def _read_paste_cache(self, paste_ids: dict) -> str:
        """Read pasted content from paste-cache/ files."""
        paste_dir = self.claude_dir / "paste-cache"
        if not paste_dir.exists():
            return ""
        parts = []
        for paste_id in paste_ids:
            paste_file = paste_dir / f"{paste_id}.txt"
            if paste_file.exists():
                try:
                    parts.append(paste_file.read_text(encoding="utf-8", errors="replace"))
                except IOError:
                    pass
        return "\n---\n".join(parts)

    def scan(self, since=None, project_filter=None) -> CollectorScanResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        history_file = self.claude_dir / "history.jsonl"
        if not history_file.exists():
            return CollectorScanResult(source_type=self.source_type)

        total = 0
        already = 0
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = event.get("timestamp", 0)
                if since and isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    if dt < since:
                        continue
                if project_filter:
                    project = event.get("project", "")
                    if project_filter.lower() not in project.lower():
                        continue
                record_id = str(ts)
                total += 1
                if record_id in self._collected_ids:
                    already += 1

        return CollectorScanResult(
            source_type=self.source_type,
            total_found=total,
            already_collected=already,
            new_available=total - already,
        )

    def collect(self, session_id: str) -> List[CollectorRecord]:
        # Prompts aren't session-scoped — use collect_all with project_filter
        return []

    def collect_all(self, since=None, project_filter=None) -> CollectorBatchResult:
        if self._collected_ids is None:
            self._collected_ids = self._load_collected_ids()

        history_file = self.claude_dir / "history.jsonl"
        if not history_file.exists():
            return CollectorBatchResult(source_type=self.source_type)

        collected = 0
        skipped = 0
        errors = []
        records = []

        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = event.get("timestamp", 0)
                record_id = str(ts)

                if record_id in self._collected_ids:
                    skipped += 1
                    continue

                if since and isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    if dt < since:
                        continue

                project = event.get("project", "")
                if project_filter and project_filter.lower() not in project.lower():
                    continue

                prompt_text = event.get("display", "")
                pasted = event.get("pastedContents", {})
                pasted_content = self._read_paste_cache(pasted) if pasted else ""

                timestamp = ""
                if isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()

                record = PromptRecord(
                    session_id="",  # Prompts aren't session-scoped
                    timestamp=timestamp,
                    source_type=self.source_type,
                    project=Path(project).name if project else "",
                    prompt_text=prompt_text[:5000],
                    pasted_content=pasted_content[:10000],
                )
                record.save(self.output_dir, f"prompt_{record_id}.json")
                self._save_collected_id(record_id)
                records.append(record)
                collected += 1

        return CollectorBatchResult(
            source_type=self.source_type,
            collected=collected, skipped=skipped, errors=errors, records=records,
        )
```

**Step 4: Update `__init__.py`, run tests, commit**

Run: `pytest tests/test_collectors.py::TestPromptCollector -v`

```bash
git add bashgym/trace_capture/collectors/prompt.py bashgym/trace_capture/collectors/__init__.py tests/test_collectors.py
git commit -m "feat(collectors): add PromptCollector for history.jsonl and paste-cache"
```

---

## Task 6: TodoCollector (P2)

**Files:**
- Create: `bashgym/trace_capture/collectors/todo.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `tests/test_collectors.py`

**Context:** Todo files are at `todos/<session-id>-agent-<agent-id>.json`. They contain JSON arrays (most are empty `[]`). Non-empty ones have task objects with statuses.

**Step 1: Write the tests**

```python
class TestTodoCollector:

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        todos = tmp_path / "todos"
        todos.mkdir()
        # Non-empty todo
        (todos / "abc12345-agent-abc12345.json").write_text(
            json.dumps([
                {"subject": "Fix auth", "status": "completed"},
                {"subject": "Add tests", "status": "pending"},
            ]),
            encoding="utf-8",
        )
        # Empty todo (should be skipped or marked empty)
        (todos / "def67890-agent-def67890.json").write_text("[]", encoding="utf-8")
        return tmp_path

    def test_collect_all_skips_empty_todos(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.todo import TodoCollector
        collector = TodoCollector()
        collector.claude_dir = mock_claude_dir
        collector.collected_dir = mock_claude_dir / "collected"

        result = collector.collect_all()
        # Only the non-empty todo should be collected
        assert result.collected == 1
        record = result.records[0]
        assert record.total_tasks == 2
        assert record.completed_tasks == 1
        assert record.pending_tasks == 1
```

**Step 2-4: Implement, test, commit**

Implementation follows the same pattern as PlanCollector. Parse the session ID and agent ID from the filename (`<session-id>-agent-<agent-id>.json`). Count task statuses. Skip files with empty arrays.

```bash
git commit -m "feat(collectors): add TodoCollector for task decomposition data"
```

---

## Task 7: EnvironmentCollector (P3)

**Files:**
- Create: `bashgym/trace_capture/collectors/environment.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `tests/test_collectors.py`

**Context:** `session-env/<session-id>/` contains environment data. `shell-snapshots/snapshot-bash-<timestamp>-<id>.sh` contains shell state (PATH, aliases). These are metadata enrichment — tag training examples with environment context.

**Step 1-4: Write tests, implement, verify, commit**

Implementation reads session-env directories (contents TBD — may be empty based on our exploration). Shell snapshots parse the bash script for PATH and alias definitions.

```bash
git commit -m "feat(collectors): add EnvironmentCollector for platform/shell context"
```

---

## Task 8: ClaudeDataScanner Orchestrator

**Files:**
- Create: `bashgym/trace_capture/collectors/scanner.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `tests/test_collectors.py`

**Step 1: Write the tests**

```python
class TestClaudeDataScanner:

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create a mock .claude with data for multiple collectors."""
        # Plans
        plans = tmp_path / "plans"
        plans.mkdir()
        (plans / "test-plan.md").write_text("# Plan\nContent.\n", encoding="utf-8")

        # History
        history = tmp_path / "history.jsonl"
        history.write_text(
            json.dumps({"display": "test prompt", "pastedContents": {}, "timestamp": 1700000000000, "project": "test"}),
            encoding="utf-8",
        )

        # File history
        fh = tmp_path / "file-history" / "session-001"
        fh.mkdir(parents=True)
        (fh / "abc123@v1").write_text("old", encoding="utf-8")
        (fh / "abc123@v2").write_text("new", encoding="utf-8")

        return tmp_path

    def test_scan_all_returns_all_sources(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        scanner.claude_dir = mock_claude_dir
        scanner.collected_dir = mock_claude_dir / "collected"

        results = scanner.scan_all()
        assert isinstance(results, dict)
        assert "plans" in results
        assert "prompts" in results
        assert "edits" in results

    def test_collect_all_runs_all_collectors(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        scanner.claude_dir = mock_claude_dir
        scanner.collected_dir = mock_claude_dir / "collected"

        results = scanner.collect_all()
        assert isinstance(results, dict)
        total = sum(r.collected for r in results.values())
        assert total >= 2  # At least plans + prompts

    def test_collect_specific_source(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        scanner.claude_dir = mock_claude_dir
        scanner.collected_dir = mock_claude_dir / "collected"

        results = scanner.collect_all(sources=["plans"])
        assert "plans" in results
        assert results["plans"].collected == 1
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement ClaudeDataScanner**

```python
# bashgym/trace_capture/collectors/scanner.py
"""
ClaudeDataScanner — orchestrates all collectors.

Provides a single interface to scan and collect all data from ~/.claude/.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from .base import CollectorScanResult, CollectorBatchResult, get_claude_dir, get_collected_dir
from .subagent import SubagentCollector
from .edit import EditCollector
from .plan import PlanCollector
from .prompt import PromptCollector
from .todo import TodoCollector
from .environment import EnvironmentCollector

ALL_SOURCES = ["subagents", "edits", "plans", "prompts", "todos", "environments"]


class ClaudeDataScanner:
    """Orchestrates all collectors for comprehensive .claude data capture."""

    def __init__(self):
        self.claude_dir = get_claude_dir()
        self.collected_dir = get_collected_dir()
        self._collectors = {
            "subagents": SubagentCollector,
            "edits": EditCollector,
            "plans": PlanCollector,
            "prompts": PromptCollector,
            "todos": TodoCollector,
            "environments": EnvironmentCollector,
        }

    def _get_collector(self, source_type: str):
        cls = self._collectors[source_type]
        collector = cls()
        collector.claude_dir = self.claude_dir
        collector.collected_dir = self.collected_dir
        return collector

    def scan_all(
        self,
        sources: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> Dict[str, CollectorScanResult]:
        sources = sources or ALL_SOURCES
        results = {}
        for source in sources:
            if source not in self._collectors:
                continue
            collector = self._get_collector(source)
            results[source] = collector.scan(since=since, project_filter=project_filter)
        return results

    def collect_all(
        self,
        sources: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> Dict[str, CollectorBatchResult]:
        sources = sources or ALL_SOURCES
        results = {}
        for source in sources:
            if source not in self._collectors:
                continue
            collector = self._get_collector(source)
            results[source] = collector.collect_all(since=since, project_filter=project_filter)
        return results

    def collect_source(
        self,
        source: str,
        since: Optional[datetime] = None,
        project_filter: Optional[str] = None,
    ) -> CollectorBatchResult:
        collector = self._get_collector(source)
        return collector.collect_all(since=since, project_filter=project_filter)

    def status(self) -> Dict[str, dict]:
        """Get current collection status for all sources."""
        status = {}
        for source in ALL_SOURCES:
            scan = self._get_collector(source).scan()
            status[source] = {
                "total": scan.total_found,
                "collected": scan.already_collected,
                "available": scan.new_available,
            }
        return status
```

**Step 4: Run tests, commit**

Run: `pytest tests/test_collectors.py::TestClaudeDataScanner -v`

```bash
git add bashgym/trace_capture/collectors/scanner.py tests/test_collectors.py
git commit -m "feat(collectors): add ClaudeDataScanner orchestrator for all data sources"
```

---

## Task 9: Peony Tool Expansion

**Files:**
- Modify: `bashgym/agent/tools.py` (add/expand tool definitions)
- Modify: `bashgym/api/agent_routes.py` (add tool execution handlers)
- Modify: `tests/test_collectors.py` (integration test)

**Context:** Peony tools are defined as dicts in `bashgym/agent/tools.py`. Tool execution happens in `bashgym/api/agent_routes.py` in the `_execute_tool()` function. The `import_traces` tool currently calls `ClaudeSessionImporter`. We need to expand it and add `scan_claude_data` and `get_collection_status`.

**Step 1: Write integration test**

```python
class TestPeonyToolIntegration:

    def test_import_traces_tool_definition_has_sources_param(self):
        from bashgym.agent.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.build_tools()
        import_tool = [t for t in tools if t["name"] == "import_traces"][0]
        props = import_tool["input_schema"]["properties"]
        assert "sources" in props

    def test_scan_claude_data_tool_exists(self):
        from bashgym.agent.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.build_tools()
        names = [t["name"] for t in tools]
        assert "scan_claude_data" in names

    def test_get_collection_status_tool_exists(self):
        from bashgym.agent.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.build_tools()
        names = [t["name"] for t in tools]
        assert "get_collection_status" in names
```

**Step 2: Update tool definitions in `bashgym/agent/tools.py`**

Expand the `import_traces` tool definition to accept `sources`, `days`, `project_filter`, and `dry_run` parameters. Add `scan_claude_data` and `get_collection_status` tool definitions.

**Step 3: Update tool execution in `bashgym/api/agent_routes.py`**

Wire the tools to call `ClaudeDataScanner`:
- `import_traces` → `scanner.collect_all(sources=..., since=...)`
- `scan_claude_data` → `scanner.scan_all()`
- `get_collection_status` → `scanner.status()`

**Step 4: Run tests, commit**

```bash
git commit -m "feat(agent): expand Peony tools with comprehensive data collection capabilities"
```

---

## Task 10: bashgym-setup CLI Expansion

**Files:**
- Modify: `bashgym/trace_capture/setup.py`
- Modify: `tests/test_collectors.py`

**Context:** The `main()` function in `setup.py` uses `argparse` with subcommands. Add `scan`, `status`, and expand `import-recent` to use `ClaudeDataScanner`.

**Step 1: Add CLI subcommands**

Add to argparse in `setup.py`:
- `bashgym-setup scan` → calls `ClaudeDataScanner.scan_all()`, prints table of available data
- `bashgym-setup status` → calls `ClaudeDataScanner.status()`, prints collection stats
- `bashgym-setup import-recent --source <type>` → adds `--source` argument to filter by collector type

**Step 2: Test the CLI manually**

```bash
python -m bashgym.trace_capture.setup scan
python -m bashgym.trace_capture.setup status
python -m bashgym.trace_capture.setup import-recent --days 7 --source subagents
```

**Step 3: Commit**

```bash
git commit -m "feat(cli): expand bashgym-setup with scan/status and source filtering"
```

---

## Task 11: DebugCollector (P2 — deferred, complex)

**Files:**
- Create: `bashgym/trace_capture/collectors/debug.py`
- Modify: `bashgym/trace_capture/collectors/__init__.py`
- Modify: `bashgym/trace_capture/collectors/scanner.py`
- Modify: `tests/test_collectors.py`

**Context:** Debug logs at `debug/<session-id>.txt` are 449MB total. They contain timestamped entries with API payloads. This is the most sensitive data source — needs aggressive PII filtering before storage. Implementation requires:

1. Parse the debug log format (timestamped lines with `[DEBUG]` prefixes)
2. Extract system prompts, full thinking blocks, API call metadata
3. Apply PII filtering (reuse patterns from `trace_processor.py`)
4. Store processed (not raw) records

**Note:** This task is intentionally less specified because the debug log format needs investigation. The implementing agent should read a sample debug file to understand the exact format before writing the parser.

**Step 1: Read a sample debug file to understand format**

```bash
head -100 ~/.claude/debug/<any-session-id>.txt
```

**Step 2: Write tests based on actual format**

**Step 3: Implement with PII filtering**

**Step 4: Commit**

```bash
git commit -m "feat(collectors): add DebugCollector for API traffic and full thinking blocks"
```

---

## Task 12: Cross-Reference Index

**Files:**
- Create: `bashgym/trace_capture/collectors/index.py`
- Modify: `bashgym/trace_capture/collectors/scanner.py`
- Modify: `tests/test_collectors.py`

**Context:** After all collectors have run, build `index.json` that maps `session_id -> {all related records}`. This is the bridge to Phase 2 (Session Graph).

**Step 1: Write the test**

```python
class TestCrossReferenceIndex:

    def test_build_index_links_records_by_session(self, mock_claude_dir):
        from bashgym.trace_capture.collectors.index import build_cross_reference_index
        # After running collectors, build index
        index = build_cross_reference_index(mock_claude_dir / "collected")
        assert isinstance(index, dict)
        # Each key is a session_id, each value has source-type keys
```

**Step 2: Implement**

Walk all `~/.bashgym/collected/*/` directories, read each JSON record's `session_id`, group by session_id, write `index.json`.

**Step 3: Wire into scanner**

Add `scanner.build_index()` method that calls after `collect_all()`.

**Step 4: Commit**

```bash
git commit -m "feat(collectors): add cross-reference index for session-to-records linking"
```

---

## Summary

| Task | Component | Priority | Effort |
|------|-----------|----------|--------|
| 1 | BaseCollector + Record Types | Foundation | Small |
| 2 | SubagentCollector | P0 | Medium |
| 3 | EditCollector | P0 | Medium |
| 4 | PlanCollector | P1 | Small |
| 5 | PromptCollector | P1 | Small |
| 6 | TodoCollector | P2 | Small |
| 7 | EnvironmentCollector | P3 | Small |
| 8 | ClaudeDataScanner | Foundation | Small |
| 9 | Peony Tool Expansion | Integration | Medium |
| 10 | bashgym-setup CLI | Integration | Small |
| 11 | DebugCollector | P2 | Large |
| 12 | Cross-Reference Index | Phase 2 prep | Small |
