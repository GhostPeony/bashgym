"""
Tests for the collector base classes, record types, and concrete collectors.

Tests cover:
  - CollectorRecord creation, serialization, and file persistence
  - Specialized record types (SubagentRecord, EditRecord, etc.)
  - CollectorScanResult and CollectorBatchResult data containers
  - BaseCollector abstract interface enforcement
  - get_claude_dir() and get_collected_dir() helper functions
  - Deduplication state loading/saving via scan_state.json
  - SubagentCollector: scanning, parsing, collecting subagent JSONL files
"""

import json
import os
import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from bashgym.trace_capture.collectors.base import (
    BaseCollector,
    CollectorRecord,
    SubagentRecord,
    EditRecord,
    PlanRecord,
    TodoRecord,
    PromptRecord,
    DebugRecord,
    EnvironmentRecord,
    CollectorScanResult,
    CollectorBatchResult,
    get_claude_dir,
    get_collected_dir,
)


# ---------------------------------------------------------------------------
# 1. CollectorRecord basics
# ---------------------------------------------------------------------------

class TestCollectorRecord:
    """Tests for the base CollectorRecord dataclass."""

    def test_collector_record_has_session_id_and_timestamp(self):
        """A CollectorRecord must carry session_id and timestamp."""
        record = CollectorRecord(
            session_id="sess-001",
            timestamp="2026-02-25T10:00:00Z",
            source_type="test",
        )
        assert record.session_id == "sess-001"
        assert record.timestamp == "2026-02-25T10:00:00Z"
        assert record.source_type == "test"

    def test_collector_record_has_default_metadata(self):
        """Metadata defaults to an empty dict."""
        record = CollectorRecord(
            session_id="sess-002",
            timestamp="2026-02-25T10:00:00Z",
            source_type="test",
        )
        assert record.metadata == {}

    def test_collector_record_serializes_to_dict(self):
        """to_json() returns a JSON-serializable dict with all fields."""
        record = CollectorRecord(
            session_id="sess-003",
            timestamp="2026-02-25T11:00:00Z",
            source_type="test",
            metadata={"key": "value"},
        )
        data = record.to_json()
        assert isinstance(data, dict)
        assert data["session_id"] == "sess-003"
        assert data["timestamp"] == "2026-02-25T11:00:00Z"
        assert data["source_type"] == "test"
        assert data["metadata"] == {"key": "value"}

    def test_collector_record_save_writes_json_file(self, tmp_path):
        """save() writes a JSON file to the given directory."""
        record = CollectorRecord(
            session_id="sess-004",
            timestamp="2026-02-25T12:00:00Z",
            source_type="test",
            metadata={"tool": "pytest"},
        )
        record.save(tmp_path, "test_record.json")

        written = tmp_path / "test_record.json"
        assert written.exists()

        data = json.loads(written.read_text(encoding="utf-8"))
        assert data["session_id"] == "sess-004"
        assert data["source_type"] == "test"
        assert data["metadata"]["tool"] == "pytest"

    def test_collector_record_save_creates_parent_dirs(self, tmp_path):
        """save() creates parent directories if they do not exist."""
        nested_dir = tmp_path / "a" / "b" / "c"
        record = CollectorRecord(
            session_id="sess-005",
            timestamp="2026-02-25T12:00:00Z",
            source_type="test",
        )
        record.save(nested_dir, "deep.json")
        assert (nested_dir / "deep.json").exists()


# ---------------------------------------------------------------------------
# 2. Specialized record types
# ---------------------------------------------------------------------------

class TestSpecializedRecords:
    """Tests for SubagentRecord, EditRecord, PlanRecord, etc."""

    def test_subagent_record_fields(self):
        """SubagentRecord carries agent details and token counts."""
        rec = SubagentRecord(
            session_id="s1",
            timestamp="t1",
            source_type="subagent",
            agent_id="agent-abc",
            slug="fix-typo",
            parent_session_id="parent-001",
            user_prompt="Fix the typo in README",
            total_input_tokens=1000,
            total_output_tokens=500,
            total_tool_calls=5,
        )
        data = rec.to_json()
        assert data["agent_id"] == "agent-abc"
        assert data["slug"] == "fix-typo"
        assert data["parent_session_id"] == "parent-001"
        assert data["total_input_tokens"] == 1000
        assert data["total_output_tokens"] == 500
        assert data["total_tool_calls"] == 5
        assert data["user_prompt"] == "Fix the typo in README"
        assert data["steps"] == []
        assert data["models_used"] == []
        assert data["tools_used"] == []

    def test_edit_record_fields(self):
        """EditRecord carries file path, versions, and diff info."""
        rec = EditRecord(
            session_id="s2",
            timestamp="t2",
            source_type="edit",
            file_path="/src/app.py",
            content_hash="abc123",
            total_versions=3,
            diff="@@ -1,2 +1,3 @@",
        )
        data = rec.to_json()
        assert data["file_path"] == "/src/app.py"
        assert data["content_hash"] == "abc123"
        assert data["versions"] == []
        assert data["total_versions"] == 3
        assert data["diff"] == "@@ -1,2 +1,3 @@"

    def test_plan_record_fields(self):
        """PlanRecord carries plan name, content, and word count."""
        rec = PlanRecord(
            session_id="s3",
            timestamp="t3",
            source_type="plan",
            plan_name="refactor-auth",
            content="Step 1: Extract interface...",
            word_count=42,
        )
        data = rec.to_json()
        assert data["plan_name"] == "refactor-auth"
        assert data["word_count"] == 42

    def test_todo_record_fields(self):
        """TodoRecord carries task lists and completion counts."""
        rec = TodoRecord(
            session_id="s4",
            timestamp="t4",
            source_type="todo",
            agent_id="agent-x",
            total_tasks=5,
            completed_tasks=3,
            pending_tasks=2,
        )
        data = rec.to_json()
        assert data["agent_id"] == "agent-x"
        assert data["tasks"] == []
        assert data["total_tasks"] == 5
        assert data["completed_tasks"] == 3
        assert data["pending_tasks"] == 2

    def test_prompt_record_fields(self):
        """PromptRecord carries project, prompt text, and pasted content."""
        rec = PromptRecord(
            session_id="s5",
            timestamp="t5",
            source_type="prompt",
            project="ghostwork",
            prompt_text="Implement the login page",
            pasted_content="<html>...</html>",
        )
        data = rec.to_json()
        assert data["project"] == "ghostwork"
        assert data["prompt_text"] == "Implement the login page"
        assert data["pasted_content"] == "<html>...</html>"

    def test_debug_record_fields(self):
        """DebugRecord carries system prompts, thinking blocks, and error info."""
        rec = DebugRecord(
            session_id="s6",
            timestamp="t6",
            source_type="debug",
            api_call_count=12,
            total_latency_ms=4500,
        )
        data = rec.to_json()
        assert data["system_prompts"] == []
        assert data["full_thinking_blocks"] == []
        assert data["api_call_count"] == 12
        assert data["total_latency_ms"] == 4500
        assert data["errors"] == []

    def test_environment_record_fields(self):
        """EnvironmentRecord carries platform/shell/env information."""
        rec = EnvironmentRecord(
            session_id="s7",
            timestamp="t7",
            source_type="environment",
            platform="win32",
            shell="bash",
            cwd="/home/user/project",
            git_branch="main",
        )
        data = rec.to_json()
        assert data["platform"] == "win32"
        assert data["shell"] == "bash"
        assert data["cwd"] == "/home/user/project"
        assert data["git_branch"] == "main"
        assert data["env_vars"] == {}
        assert data["shell_snapshot"] == {}


# ---------------------------------------------------------------------------
# 3. Result types
# ---------------------------------------------------------------------------

class TestResultTypes:
    """Tests for CollectorScanResult and CollectorBatchResult."""

    def test_scan_result_contains_counts(self):
        """CollectorScanResult reports totals for found/collected/new items."""
        result = CollectorScanResult(
            source_type="subagent",
            total_found=100,
            already_collected=60,
            new_available=40,
            estimated_size_bytes=2048,
        )
        assert result.source_type == "subagent"
        assert result.total_found == 100
        assert result.already_collected == 60
        assert result.new_available == 40
        assert result.estimated_size_bytes == 2048

    def test_batch_result_tracks_collected_and_errors(self):
        """CollectorBatchResult tracks successes, skips, and errors."""
        record = CollectorRecord(
            session_id="s1", timestamp="t1", source_type="test"
        )
        result = CollectorBatchResult(
            source_type="test",
            collected=5,
            skipped=2,
            errors=["timeout on session-3"],
            records=[record],
        )
        assert result.source_type == "test"
        assert result.collected == 5
        assert result.skipped == 2
        assert len(result.errors) == 1
        assert result.errors[0] == "timeout on session-3"
        assert len(result.records) == 1

    def test_batch_result_defaults(self):
        """CollectorBatchResult has sensible defaults for errors and records."""
        result = CollectorBatchResult(
            source_type="edit",
            collected=0,
            skipped=0,
        )
        assert result.errors == []
        assert result.records == []


# ---------------------------------------------------------------------------
# 4. BaseCollector abstract class
# ---------------------------------------------------------------------------

class TestBaseCollector:
    """Tests for the BaseCollector abstract interface."""

    def test_base_collector_is_abstract(self):
        """BaseCollector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCollector(
                claude_dir=Path("/tmp/claude"),
                collected_dir=Path("/tmp/collected"),
            )

    def test_concrete_collector_requires_all_methods(self):
        """A subclass must implement all abstract methods to be instantiated."""
        # Missing collect_all
        class PartialCollector(BaseCollector):
            @property
            def source_type(self) -> str:
                return "partial"

            def scan(self, since=None, project_filter=None):
                pass

            def collect(self, session_id):
                pass

        with pytest.raises(TypeError):
            PartialCollector(
                claude_dir=Path("/tmp/claude"),
                collected_dir=Path("/tmp/collected"),
            )

    def test_concrete_collector_can_be_instantiated(self, tmp_path):
        """A fully implemented subclass can be created."""
        class FullCollector(BaseCollector):
            @property
            def source_type(self) -> str:
                return "full"

            def scan(self, since=None, project_filter=None):
                return CollectorScanResult(
                    source_type="full",
                    total_found=0,
                    already_collected=0,
                    new_available=0,
                )

            def collect(self, session_id):
                return []

            def collect_all(self, since=None, project_filter=None):
                return CollectorBatchResult(
                    source_type="full",
                    collected=0,
                    skipped=0,
                )

        collector = FullCollector(
            claude_dir=tmp_path / ".claude",
            collected_dir=tmp_path / "collected",
        )
        assert collector.source_type == "full"
        assert collector.claude_dir == tmp_path / ".claude"
        assert collector.collected_dir == tmp_path / "collected"

    def test_output_dir_property(self, tmp_path):
        """output_dir returns collected_dir / source_type."""
        class StubCollector(BaseCollector):
            @property
            def source_type(self) -> str:
                return "mystuff"

            def scan(self, since=None, project_filter=None):
                return CollectorScanResult(
                    source_type="mystuff", total_found=0,
                    already_collected=0, new_available=0,
                )

            def collect(self, session_id):
                return []

            def collect_all(self, since=None, project_filter=None):
                return CollectorBatchResult(
                    source_type="mystuff", collected=0, skipped=0,
                )

        collector = StubCollector(
            claude_dir=tmp_path / ".claude",
            collected_dir=tmp_path / "collected",
        )
        assert collector.output_dir == tmp_path / "collected" / "mystuff"


# ---------------------------------------------------------------------------
# 5. Deduplication state
# ---------------------------------------------------------------------------

class TestDeduplicationState:
    """Tests for _load_collected_ids and _save_collected_id."""

    def _make_collector(self, tmp_path):
        """Create a minimal concrete collector for testing."""
        class StubCollector(BaseCollector):
            @property
            def source_type(self) -> str:
                return "stub"

            def scan(self, since=None, project_filter=None):
                return CollectorScanResult(
                    source_type="stub", total_found=0,
                    already_collected=0, new_available=0,
                )

            def collect(self, session_id):
                return []

            def collect_all(self, since=None, project_filter=None):
                return CollectorBatchResult(
                    source_type="stub", collected=0, skipped=0,
                )

        return StubCollector(
            claude_dir=tmp_path / ".claude",
            collected_dir=tmp_path / "collected",
        )

    def test_load_collected_ids_empty_when_no_state_file(self, tmp_path):
        """Returns empty set when scan_state.json does not exist."""
        collector = self._make_collector(tmp_path)
        ids = collector._load_collected_ids()
        assert ids == set()

    def test_save_and_load_collected_id(self, tmp_path):
        """Saved IDs can be loaded back."""
        collector = self._make_collector(tmp_path)
        collector._save_collected_id("session-aaa")
        collector._save_collected_id("session-bbb")

        ids = collector._load_collected_ids()
        assert "session-aaa" in ids
        assert "session-bbb" in ids

    def test_save_collected_id_is_idempotent(self, tmp_path):
        """Saving the same ID twice does not duplicate it."""
        collector = self._make_collector(tmp_path)
        collector._save_collected_id("session-dup")
        collector._save_collected_id("session-dup")

        ids = collector._load_collected_ids()
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# 6. Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for get_claude_dir() and get_collected_dir()."""

    def test_get_claude_dir_returns_path(self):
        """get_claude_dir() returns a Path ending in .claude."""
        result = get_claude_dir()
        assert isinstance(result, Path)
        assert result.name == ".claude"

    def test_get_collected_dir_returns_path(self):
        """get_collected_dir() returns a Path under ~/.bashgym/collected/."""
        result = get_collected_dir()
        assert isinstance(result, Path)
        assert result.name == "collected"
        assert ".bashgym" in str(result)

    def test_get_claude_dir_uses_userprofile_on_windows(self, monkeypatch):
        """On Windows, get_claude_dir uses USERPROFILE."""
        monkeypatch.setattr("bashgym.trace_capture.collectors.base.platform.system", lambda: "Windows")
        monkeypatch.setenv("USERPROFILE", "C:\\Users\\TestUser")
        result = get_claude_dir()
        assert result == Path("C:\\Users\\TestUser") / ".claude"

    def test_get_claude_dir_uses_home_on_unix(self, monkeypatch):
        """On non-Windows, get_claude_dir uses Path.home()."""
        monkeypatch.setattr("bashgym.trace_capture.collectors.base.platform.system", lambda: "Linux")
        home = Path.home()
        result = get_claude_dir()
        assert result == home / ".claude"


# ---------------------------------------------------------------------------
# 7. SubagentCollector
# ---------------------------------------------------------------------------

from bashgym.trace_capture.collectors.subagent import SubagentCollector


class TestSubagentCollector:
    """Tests for SubagentCollector: scanning, parsing, and collecting subagent JSONL files."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock .claude with subagent files."""
        projects = tmp_path / "projects"
        project = projects / "C--Users-Cade-projects-myapp"
        session_dir = project / "abc12345-1234-5678-abcd-1234567890ab"
        subagents = session_dir / "subagents"
        subagents.mkdir(parents=True)

        subagent_file = subagents / "agent-a11b22c33d44e55f6.jsonl"
        lines = [
            json.dumps({"parentUuid": None, "isSidechain": True, "userType": "external",
                "cwd": "C:\\Users\\Cade\\projects\\myapp",
                "sessionId": "abc12345-1234-5678-abcd-1234567890ab",
                "version": "2.1.51", "gitBranch": "main",
                "agentId": "a11b22c33d44e55f6",
                "slug": "iridescent-wiggling-adleman", "type": "user",
                "message": {"role": "user", "content": "Find all API endpoints"}}),
            json.dumps({"type": "assistant", "message": {
                "role": "assistant", "model": "claude-sonnet-4-5-20250929",
                "content": [
                    {"type": "text", "text": "I'll search for API endpoints."},
                    {"type": "tool_use", "id": "toolu_01", "name": "Grep",
                     "input": {"pattern": "@app\\.(get|post)", "path": "src/"}},
                ],
                "usage": {"input_tokens": 500, "output_tokens": 100}}}),
        ]
        subagent_file.write_text("\n".join(lines), encoding="utf-8")

        # Also a session JSONL (can be empty)
        (project / "abc12345-1234-5678-abcd-1234567890ab.jsonl").write_text("", encoding="utf-8")
        return tmp_path

    def test_scan_finds_subagent_files(self, mock_claude_dir, tmp_path):
        """scan() discovers subagent files and reports correct counts."""
        collector = SubagentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.scan()
        assert result.source_type == "subagents"
        assert result.total_found == 1
        assert result.new_available == 1

    def test_collect_parses_subagent_jsonl(self, mock_claude_dir, tmp_path):
        """collect() parses subagent JSONL and returns SubagentRecord with correct fields."""
        collector = SubagentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("abc12345-1234-5678-abcd-1234567890ab")
        assert len(records) >= 1
        rec = records[0]
        assert rec.agent_id == "a11b22c33d44e55f6"
        assert rec.slug == "iridescent-wiggling-adleman"
        assert rec.parent_session_id == "abc12345-1234-5678-abcd-1234567890ab"
        assert rec.total_tool_calls >= 1
        assert "Grep" in rec.tools_used
        assert "claude-sonnet-4-5-20250929" in rec.models_used
        assert rec.total_input_tokens == 500
        assert rec.total_output_tokens == 100
        assert rec.user_prompt == "Find all API endpoints"
        assert rec.source_type == "subagents"
        assert len(rec.steps) >= 1
        assert rec.steps[0]["tool_name"] == "Grep"

    def test_collect_all_returns_batch_result(self, mock_claude_dir, tmp_path):
        """collect_all() returns a CollectorBatchResult with correct totals."""
        collector = SubagentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        assert result.source_type == "subagents"
        assert result.collected == 1
        assert len(result.records) == 1

    def test_collect_skips_already_collected(self, mock_claude_dir, tmp_path):
        """collect_all() skips subagents that were already collected."""
        collector = SubagentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # Collect once
        first = collector.collect_all()
        assert first.collected == 1

        # Collect again — should skip
        second = collector.collect_all()
        assert second.collected == 0
        assert second.skipped == 1

    def test_scan_respects_project_filter(self, mock_claude_dir, tmp_path):
        """scan() filters subagent files by project directory name."""
        collector = SubagentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # Filter matches the project slug
        result = collector.scan(project_filter="myapp")
        assert result.total_found == 1

        # Filter does NOT match
        result = collector.scan(project_filter="nonexistent")
        assert result.total_found == 0

    def test_scan_respects_since_filter(self, mock_claude_dir, tmp_path):
        """scan() filters subagent files by modification time."""
        collector = SubagentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # Far-future date — nothing should match
        result = collector.scan(since="2099-01-01T00:00:00Z")
        assert result.total_found == 0

        # Far-past date — everything should match
        result = collector.scan(since="2000-01-01T00:00:00Z")
        assert result.total_found == 1

    def test_collect_handles_empty_subagent_file(self, tmp_path):
        """collect() handles an empty subagent JSONL gracefully."""
        projects = tmp_path / "projects"
        project = projects / "C--Users-Cade-projects-emptyapp"
        session_dir = project / "empty-session-id"
        subagents = session_dir / "subagents"
        subagents.mkdir(parents=True)
        (subagents / "agent-empty123.jsonl").write_text("", encoding="utf-8")

        collector = SubagentCollector(
            claude_dir=tmp_path,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("empty-session-id")
        assert records == []

    def test_collect_handles_malformed_json_lines(self, tmp_path):
        """collect() skips malformed JSON lines without crashing."""
        projects = tmp_path / "projects"
        project = projects / "C--Users-Cade-projects-badapp"
        session_dir = project / "bad-session-id"
        subagents = session_dir / "subagents"
        subagents.mkdir(parents=True)

        lines = [
            "not valid json at all",
            json.dumps({"type": "user", "sessionId": "bad-session-id",
                "agentId": "badagent1", "slug": "test-slug",
                "message": {"role": "user", "content": "Hello"}}),
            "{incomplete json",
        ]
        (subagents / "agent-badagent1.jsonl").write_text("\n".join(lines), encoding="utf-8")

        collector = SubagentCollector(
            claude_dir=tmp_path,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("bad-session-id")
        assert len(records) == 1
        assert records[0].agent_id == "badagent1"
        assert records[0].user_prompt == "Hello"
