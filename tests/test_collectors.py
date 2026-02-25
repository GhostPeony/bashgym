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
  - DebugCollector: scanning, parsing, collecting debug log files
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


# ---------------------------------------------------------------------------
# 8. EditCollector
# ---------------------------------------------------------------------------

from bashgym.trace_capture.collectors.edit import EditCollector


class TestEditCollector:
    """Tests for EditCollector: scanning, parsing, and collecting file-history snapshots."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock .claude with file-history directories."""
        fh = tmp_path / "file-history" / "abc12345-1234-5678-abcd-1234567890ab"
        fh.mkdir(parents=True)
        (fh / "a1b2c3d4e5f6g7h8@v1").write_text("def hello():\n    pass\n", encoding="utf-8")
        (fh / "a1b2c3d4e5f6g7h8@v2").write_text("def hello():\n    print('hello')\n", encoding="utf-8")
        (fh / "f0f0f0f0f0f0f0f0@v1").write_text("# old\n", encoding="utf-8")
        (fh / "f0f0f0f0f0f0f0f0@v2").write_text("# middle\n", encoding="utf-8")
        (fh / "f0f0f0f0f0f0f0f0@v3").write_text("# final\n", encoding="utf-8")
        return tmp_path

    def test_scan_finds_edit_sessions(self, mock_claude_dir, tmp_path):
        """scan() discovers edit sessions and reports correct counts."""
        collector = EditCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.scan()
        assert result.source_type == "edits"
        assert result.total_found >= 1
        assert result.new_available >= 1

    def test_collect_groups_versions(self, mock_claude_dir, tmp_path):
        """collect() groups versions by content hash and returns correct records."""
        collector = EditCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("abc12345-1234-5678-abcd-1234567890ab")
        assert len(records) == 2  # two content hashes

        # Find the record with 3 versions (f0f0f0f0f0f0f0f0)
        three_ver = [r for r in records if r.total_versions == 3]
        assert len(three_ver) == 1
        rec = three_ver[0]
        assert rec.content_hash == "f0f0f0f0f0f0f0f0"
        assert rec.versions[0]["content"] == "# old\n"
        assert rec.versions[-1]["content"] == "# final\n"

    def test_collect_produces_diff(self, mock_claude_dir, tmp_path):
        """collect() produces a unified diff between first and last version."""
        collector = EditCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("abc12345-1234-5678-abcd-1234567890ab")

        # Find the 2-version hash (a1b2c3d4e5f6g7h8)
        two_ver = [r for r in records if r.total_versions == 2]
        assert len(two_ver) == 1
        rec = two_ver[0]
        assert "pass" in rec.diff
        assert "hello" in rec.diff

    def test_collect_all_deduplicates(self, mock_claude_dir, tmp_path):
        """collect_all() skips sessions that were already collected."""
        collector = EditCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # First collection
        first = collector.collect_all()
        assert first.source_type == "edits"
        assert first.collected >= 1

        # Second collection — should skip
        second = collector.collect_all()
        assert second.collected == 0
        assert second.skipped >= 1


# ---------------------------------------------------------------------------
# 9. PlanCollector
# ---------------------------------------------------------------------------

from bashgym.trace_capture.collectors.plan import PlanCollector


class TestPlanCollector:
    """Tests for PlanCollector: scanning, parsing, and collecting plan markdown files."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock .claude with plans/ directory."""
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

    def test_scan_finds_plan_files(self, mock_claude_dir, tmp_path):
        """scan() discovers plan markdown files and reports correct counts."""
        collector = PlanCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.scan()
        assert result.source_type == "plans"
        assert result.total_found == 2

    def test_collect_all_parses_markdown(self, mock_claude_dir, tmp_path):
        """collect_all() parses plan markdown files and returns correct records."""
        collector = PlanCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        assert result.collected == 2
        assert len(result.records) == 2

        names = {r.plan_name for r in result.records}
        assert "clever-jumping-fox" in names
        assert "wiggly-napping-pixel" in names

        for rec in result.records:
            assert rec.word_count > 0
            assert rec.content != ""
            assert rec.source_type == "plans"

    def test_collect_by_session_id_returns_empty(self, mock_claude_dir, tmp_path):
        """collect() returns empty list since plans are not session-scoped."""
        collector = PlanCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("any-id")
        assert records == []

    def test_collect_all_deduplicates(self, mock_claude_dir, tmp_path):
        """collect_all() skips plans that were already collected."""
        collector = PlanCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # First collection
        first = collector.collect_all()
        assert first.collected == 2

        # Second collection — should skip all
        second = collector.collect_all()
        assert second.collected == 0
        assert second.skipped == 2


# ---------------------------------------------------------------------------
# 10. TodoCollector
# ---------------------------------------------------------------------------

try:
    from bashgym.trace_capture.collectors.todo import TodoCollector
    _has_todo_collector = True
except ImportError:
    _has_todo_collector = False


@pytest.mark.skipif(not _has_todo_collector, reason="TodoCollector not yet implemented")
class TestTodoCollector:
    """Tests for TodoCollector: scanning, parsing, and collecting todo JSON files."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock .claude with todos/ directory."""
        todos = tmp_path / "todos"
        todos.mkdir()
        # Non-empty todo file: 2 tasks (1 completed, 1 pending)
        (todos / "abc12345-1234-5678-abcd-1234567890ab-agent-abc12345-1234-5678-abcd-1234567890ab.json").write_text(
            json.dumps([
                {"subject": "Fix auth", "status": "completed"},
                {"subject": "Add tests", "status": "pending"},
            ]),
            encoding="utf-8",
        )
        # Empty todo file: should be skipped
        (todos / "def67890-1234-5678-abcd-1234567890ab-agent-def67890-1234-5678-abcd-1234567890ab.json").write_text(
            "[]",
            encoding="utf-8",
        )
        return tmp_path

    def test_source_type(self, mock_claude_dir, tmp_path):
        """source_type returns 'todos'."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        assert collector.source_type == "todos"

    def test_collect_all_skips_empty_todos(self, mock_claude_dir, tmp_path):
        """collect_all() skips files with empty arrays and collects non-empty ones."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        assert result.source_type == "todos"
        assert result.collected == 1
        assert len(result.records) == 1

        rec = result.records[0]
        assert rec.total_tasks == 2
        assert rec.completed_tasks == 1
        assert rec.pending_tasks == 1
        assert rec.agent_id == "abc12345-1234-5678-abcd-1234567890ab"
        assert rec.session_id == "abc12345-1234-5678-abcd-1234567890ab"

    def test_scan_finds_todo_files(self, mock_claude_dir, tmp_path):
        """scan() discovers non-empty todo files and reports correct counts."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.scan()
        assert result.source_type == "todos"
        # Only non-empty files are counted
        assert result.total_found >= 1
        assert result.new_available >= 1

    def test_collect_all_deduplicates(self, mock_claude_dir, tmp_path):
        """collect_all() skips todos that were already collected."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # First collection
        first = collector.collect_all()
        assert first.collected == 1

        # Second collection -- should skip
        second = collector.collect_all()
        assert second.collected == 0
        assert second.skipped >= 1

    def test_collect_returns_records_for_session(self, mock_claude_dir, tmp_path):
        """collect() returns todo records matching the given session_id."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("abc12345-1234-5678-abcd-1234567890ab")
        assert len(records) == 1
        assert records[0].total_tasks == 2

    def test_collect_returns_empty_for_unknown_session(self, mock_claude_dir, tmp_path):
        """collect() returns empty list for non-existent session_id."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("nonexistent-session-id")
        assert records == []

    def test_parse_filename_extracts_ids(self, mock_claude_dir, tmp_path):
        """_parse_filename() extracts session_id and agent_id from the filename."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        session_id, agent_id = collector._parse_filename(
            "abc12345-1234-5678-abcd-1234567890ab-agent-def67890-1234-5678-abcd-1234567890ab.json"
        )
        assert session_id == "abc12345-1234-5678-abcd-1234567890ab"
        assert agent_id == "def67890-1234-5678-abcd-1234567890ab"

    def test_parse_filename_returns_none_for_invalid(self, mock_claude_dir, tmp_path):
        """_parse_filename() returns (None, None) for filenames that don't match the pattern."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        session_id, agent_id = collector._parse_filename("not-a-valid-filename.json")
        assert session_id is None
        assert agent_id is None

    def test_collect_all_records_contain_tasks(self, mock_claude_dir, tmp_path):
        """Collected records contain the original task data."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        rec = result.records[0]
        assert len(rec.tasks) == 2
        subjects = {t["subject"] for t in rec.tasks}
        assert "Fix auth" in subjects
        assert "Add tests" in subjects

    def test_scan_respects_since_filter(self, mock_claude_dir, tmp_path):
        """scan() filters todo files by modification time."""
        collector = TodoCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # Far-future date -- nothing should match
        result = collector.scan(since="2099-01-01T00:00:00Z")
        assert result.total_found == 0

        # Far-past date -- everything non-empty should match
        result = collector.scan(since="2000-01-01T00:00:00Z")
        assert result.total_found >= 1

    def test_handles_malformed_json(self, tmp_path):
        """collect_all() handles malformed JSON files gracefully."""
        todos = tmp_path / "todos"
        todos.mkdir()
        (todos / "abc12345-1234-5678-abcd-1234567890ab-agent-abc12345-1234-5678-abcd-1234567890ab.json").write_text(
            "{not valid json",
            encoding="utf-8",
        )

        collector = TodoCollector(
            claude_dir=tmp_path,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        assert result.collected == 0


# ---------------------------------------------------------------------------
# 11. PromptCollector
# ---------------------------------------------------------------------------

try:
    from bashgym.trace_capture.collectors.prompt import PromptCollector
    _has_prompt_collector = True
except ImportError:
    _has_prompt_collector = False


@pytest.mark.skipif(not _has_prompt_collector, reason="PromptCollector not yet implemented")
class TestPromptCollector:
    """Tests for PromptCollector: scanning, parsing, and collecting user prompts from history.jsonl."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock .claude with history.jsonl and paste-cache/."""
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

    def test_scan_counts_history_lines(self, mock_claude_dir, tmp_path):
        """scan() counts all lines in history.jsonl and reports correct totals."""
        collector = PromptCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.scan()
        assert result.source_type == "prompts"
        assert result.total_found == 2

    def test_collect_all_reads_history(self, mock_claude_dir, tmp_path):
        """collect_all() reads history.jsonl and creates PromptRecords."""
        collector = PromptCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        assert result.source_type == "prompts"
        assert result.collected == 2
        assert len(result.records) == 2

        # Find the auth prompt and verify project extraction
        auth_records = [r for r in result.records if "auth" in r.prompt_text]
        assert len(auth_records) == 1
        assert auth_records[0].project.endswith("myapp")

    def test_collect_links_paste_cache(self, mock_claude_dir, tmp_path):
        """collect_all() reads paste-cache files and links pasted content."""
        collector = PromptCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()

        # Find the dark mode prompt
        dark_records = [r for r in result.records if "dark mode" in r.prompt_text]
        assert len(dark_records) == 1
        assert "theme" in dark_records[0].pasted_content

    def test_collect_all_deduplicates(self, mock_claude_dir, tmp_path):
        """collect_all() skips prompts that were already collected."""
        collector = PromptCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # First collection
        first = collector.collect_all()
        assert first.collected == 2

        # Second collection — should skip all
        second = collector.collect_all()
        assert second.collected == 0
        assert second.skipped == 2

    def test_collect_returns_empty_list(self, mock_claude_dir, tmp_path):
        """collect() returns empty list since prompts are not session-scoped."""
        collector = PromptCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("any-session-id")
        assert records == []

    def test_scan_respects_project_filter(self, mock_claude_dir, tmp_path):
        """scan() filters prompts by project name."""
        collector = PromptCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # Filter matches the project
        result = collector.scan(project_filter="myapp")
        assert result.total_found == 2

        # Filter does NOT match
        result = collector.scan(project_filter="nonexistent")
        assert result.total_found == 0

    def test_scan_respects_since_filter(self, mock_claude_dir, tmp_path):
        """scan() filters prompts by timestamp."""
        collector = PromptCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # Far-future date — nothing should match
        result = collector.scan(since="2099-01-01T00:00:00Z")
        assert result.total_found == 0

        # Far-past date — everything should match
        result = collector.scan(since="2000-01-01T00:00:00Z")
        assert result.total_found == 2

    def test_collect_all_truncates_long_content(self, tmp_path):
        """collect_all() truncates prompt_text to 5000 chars and pasted_content to 10000 chars."""
        history = tmp_path / "history.jsonl"
        long_prompt = "x" * 6000
        long_paste_id = "longpaste1"
        line = json.dumps({
            "display": long_prompt,
            "pastedContents": {long_paste_id: True},
            "timestamp": 1759817571796,
            "project": "C:\\Users\\Cade\\projects\\bigapp",
        })
        history.write_text(line, encoding="utf-8")

        paste = tmp_path / "paste-cache"
        paste.mkdir()
        (paste / f"{long_paste_id}.txt").write_text("y" * 12000, encoding="utf-8")

        collector = PromptCollector(
            claude_dir=tmp_path,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        assert result.collected == 1
        rec = result.records[0]
        assert len(rec.prompt_text) <= 5000
        assert len(rec.pasted_content) <= 10000


# ---------------------------------------------------------------------------
# 12. EnvironmentCollector
# ---------------------------------------------------------------------------

try:
    from bashgym.trace_capture.collectors.environment import EnvironmentCollector
    _has_environment_collector = True
except ImportError:
    _has_environment_collector = False


@pytest.mark.skipif(not _has_environment_collector, reason="EnvironmentCollector not yet implemented")
class TestEnvironmentCollector:
    """Tests for EnvironmentCollector: scanning, parsing, and collecting session environments and shell snapshots."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock .claude with session-env dirs and shell snapshots."""
        env = tmp_path / "session-env" / "abc12345-session-id"
        env.mkdir(parents=True)
        snaps = tmp_path / "shell-snapshots"
        snaps.mkdir()
        (snaps / "snapshot-bash-1760000000000-abc123.sh").write_text(
            "# Snapshot file\nexport PATH=/usr/bin:/usr/local/bin\nalias ll='ls -la'\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_scan_finds_environments(self, mock_claude_dir, tmp_path):
        """scan() discovers session-env dirs and shell snapshots."""
        collector = EnvironmentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.scan()
        assert result.source_type == "environments"
        assert result.total_found >= 1

    def test_collect_all_captures_shell_snapshot(self, mock_claude_dir, tmp_path):
        """collect_all() parses shell snapshot and returns records with PATH info."""
        collector = EnvironmentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        assert result.source_type == "environments"
        assert result.collected >= 1

        # Find a record that has shell_snapshot data
        snapshot_records = [r for r in result.records if r.shell_snapshot]
        assert len(snapshot_records) >= 1
        snap = snapshot_records[0]
        assert "PATH" in str(snap.shell_snapshot)

    def test_collect_all_deduplicates(self, mock_claude_dir, tmp_path):
        """collect_all() skips already-collected items on second call."""
        collector = EnvironmentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        # First collection
        first = collector.collect_all()
        assert first.collected >= 1

        # Second collection — should skip
        second = collector.collect_all()
        assert second.collected == 0
        assert second.skipped >= 1

    def test_collect_for_session_returns_env_record(self, mock_claude_dir, tmp_path):
        """collect() returns an EnvironmentRecord for a known session."""
        collector = EnvironmentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("abc12345-session-id")
        assert len(records) == 1
        rec = records[0]
        assert rec.session_id == "abc12345-session-id"
        assert rec.source_type == "environments"
        assert rec.platform != ""

    def test_collect_for_unknown_session_returns_empty(self, mock_claude_dir, tmp_path):
        """collect() returns empty list for a session that doesn't exist."""
        collector = EnvironmentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        records = collector.collect("nonexistent-session-id")
        assert records == []

    def test_parse_shell_snapshot_extracts_aliases(self, mock_claude_dir, tmp_path):
        """Shell snapshot parsing extracts alias definitions."""
        collector = EnvironmentCollector(
            claude_dir=mock_claude_dir,
            collected_dir=tmp_path / "collected",
        )
        result = collector.collect_all()
        snapshot_records = [r for r in result.records if r.shell_snapshot]
        assert len(snapshot_records) >= 1
        snap = snapshot_records[0]
        aliases = snap.shell_snapshot.get("aliases", {})
        assert "ll" in aliases
        assert aliases["ll"] == "ls -la"


# ---------------------------------------------------------------------------
# 10. ClaudeDataScanner orchestrator
# ---------------------------------------------------------------------------


class TestClaudeDataScanner:
    """Tests for the ClaudeDataScanner orchestrator."""

    @pytest.fixture
    def mock_claude_dir(self, tmp_path):
        """Create mock .claude with data for multiple collectors."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        # Plans
        plans_dir = claude_dir / "plans"
        plans_dir.mkdir()
        (plans_dir / "test-plan.md").write_text(
            "# Plan\nContent here.\n", encoding="utf-8"
        )

        # History (prompts)
        (claude_dir / "history.jsonl").write_text(
            json.dumps({
                "display": "test prompt",
                "pastedContents": {},
                "timestamp": 1700000000000,
                "project": "test",
            }),
            encoding="utf-8",
        )

        # File history (edits)
        fh = claude_dir / "file-history" / "session-001"
        fh.mkdir(parents=True)
        (fh / "abc123@v1").write_text("old", encoding="utf-8")
        (fh / "abc123@v2").write_text("new", encoding="utf-8")

        return claude_dir

    def _make_scanner(self, mock_claude_dir, tmp_path):
        """Create a ClaudeDataScanner pointed at mock directories."""
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner

        scanner = ClaudeDataScanner()
        scanner.claude_dir = mock_claude_dir
        scanner.collected_dir = tmp_path / "collected"
        return scanner

    def test_scan_all_returns_all_sources(self, mock_claude_dir, tmp_path):
        """scan_all() returns a dict keyed by every source type, each a CollectorScanResult."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        results = scanner.scan_all()
        from bashgym.trace_capture.collectors.scanner import ALL_SOURCES

        assert set(results.keys()) == set(ALL_SOURCES)
        for key, result in results.items():
            assert isinstance(result, CollectorScanResult)
            assert result.source_type == key

    def test_collect_all_runs_all_collectors(self, mock_claude_dir, tmp_path):
        """collect_all() collects from all sources; total collected >= 2 (plans + prompts)."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        results = scanner.collect_all()
        from bashgym.trace_capture.collectors.scanner import ALL_SOURCES

        assert set(results.keys()) == set(ALL_SOURCES)
        total_collected = sum(r.collected for r in results.values())
        # At minimum plans (1) + prompts (1) + edits (1) = 3
        assert total_collected >= 2

    def test_collect_specific_sources(self, mock_claude_dir, tmp_path):
        """collect_all(sources=['plans']) only collects plans."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        results = scanner.collect_all(sources=["plans"])
        assert list(results.keys()) == ["plans"]
        assert results["plans"].collected == 1

    def test_collect_source_single(self, mock_claude_dir, tmp_path):
        """collect_source('plans') returns a CollectorBatchResult with collected == 1."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        result = scanner.collect_source("plans")
        assert isinstance(result, CollectorBatchResult)
        assert result.collected == 1
        assert result.source_type == "plans"

    def test_status_returns_all_sources(self, mock_claude_dir, tmp_path):
        """status() returns dict with all source types, each with total/collected/available."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        status = scanner.status()
        from bashgym.trace_capture.collectors.scanner import ALL_SOURCES

        assert set(status.keys()) == set(ALL_SOURCES)
        for source, info in status.items():
            assert "total" in info
            assert "collected" in info
            assert "available" in info

    def test_invalid_source_ignored(self, mock_claude_dir, tmp_path):
        """collect_all(sources=['invalid']) returns an empty dict."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        results = scanner.collect_all(sources=["invalid"])
        assert results == {}

    def test_scan_all_with_specific_sources(self, mock_claude_dir, tmp_path):
        """scan_all(sources=['plans', 'prompts']) only scans those two."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        results = scanner.scan_all(sources=["plans", "prompts"])
        assert set(results.keys()) == {"plans", "prompts"}

    def test_collect_source_raises_on_unknown(self, mock_claude_dir, tmp_path):
        """collect_source() raises KeyError for unknown source types."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        with pytest.raises(KeyError):
            scanner.collect_source("nonexistent")

    def test_status_reflects_collection(self, mock_claude_dir, tmp_path):
        """After collecting plans, status shows them as collected."""
        scanner = self._make_scanner(mock_claude_dir, tmp_path)
        # Collect plans first
        scanner.collect_source("plans")
        status = scanner.status()
        assert status["plans"]["collected"] == 1
        assert status["plans"]["available"] == 0


# ---------------------------------------------------------------------------
# 11. Peony Tool Integration
# ---------------------------------------------------------------------------


class TestPeonyToolIntegration:
    """Tests for Peony tool definitions and execution wiring."""

    def test_import_traces_tool_definition_has_sources_param(self):
        """import_traces tool must define sources, days, project_filter, dry_run params."""
        from bashgym.agent.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.build_tools()
        import_tool = [t for t in tools if t["name"] == "import_traces"][0]
        props = import_tool["input_schema"]["properties"]
        assert "sources" in props
        assert "days" in props
        assert "project_filter" in props
        assert "dry_run" in props

    def test_scan_claude_data_tool_exists(self):
        """scan_claude_data tool must be present in the tool registry."""
        from bashgym.agent.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.build_tools()
        names = [t["name"] for t in tools]
        assert "scan_claude_data" in names

    def test_get_collection_status_tool_exists(self):
        """get_collection_status tool must be present in the tool registry."""
        from bashgym.agent.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.build_tools()
        names = [t["name"] for t in tools]
        assert "get_collection_status" in names

    def test_import_traces_sources_enum_matches_all_sources(self):
        """The sources enum should contain 'all' plus all source types from ALL_SOURCES."""
        from bashgym.agent.tools import ToolRegistry
        from bashgym.trace_capture.collectors.scanner import ALL_SOURCES
        registry = ToolRegistry()
        tools = registry.build_tools()
        import_tool = [t for t in tools if t["name"] == "import_traces"][0]
        sources_prop = import_tool["input_schema"]["properties"]["sources"]
        # The items enum should contain "all" plus all source types
        enum_values = sources_prop["items"]["enum"]
        for source in ALL_SOURCES:
            assert source in enum_values
        assert "all" in enum_values

    def test_execute_scan_claude_data_calls_scanner(self):
        """_execute_tool('scan_claude_data') calls scanner.scan_all() and returns JSON."""
        import asyncio
        from unittest.mock import patch, MagicMock
        from bashgym.trace_capture.collectors.base import CollectorScanResult

        mock_scanner = MagicMock()
        mock_scanner.scan_all.return_value = {
            "plans": CollectorScanResult(
                source_type="plans", total_found=5,
                already_collected=2, new_available=3,
            ),
        }

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            from bashgym.api.agent_routes import _execute_tool
            result = asyncio.run(_execute_tool("scan_claude_data", {}))

        data = json.loads(result)
        assert data["plans"]["total_found"] == 5
        assert data["plans"]["new_available"] == 3
        mock_scanner.scan_all.assert_called_once()

    def test_execute_get_collection_status_calls_scanner(self):
        """_execute_tool('get_collection_status') calls scanner.status() and returns JSON."""
        import asyncio
        from unittest.mock import patch, MagicMock

        mock_scanner = MagicMock()
        mock_scanner.status.return_value = {
            "plans": {"total": 10, "collected": 7, "available": 3},
        }

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            from bashgym.api.agent_routes import _execute_tool
            result = asyncio.run(_execute_tool("get_collection_status", {}))

        data = json.loads(result)
        assert data["plans"]["total"] == 10
        assert data["plans"]["collected"] == 7
        mock_scanner.status.assert_called_once()

    def test_execute_import_traces_dry_run_calls_scan_all(self):
        """_execute_tool('import_traces', {dry_run: True}) calls scanner.scan_all()."""
        import asyncio
        from unittest.mock import patch, MagicMock
        from bashgym.trace_capture.collectors.base import CollectorScanResult

        mock_scanner = MagicMock()
        mock_scanner.scan_all.return_value = {
            "plans": CollectorScanResult(
                source_type="plans", total_found=3,
                already_collected=1, new_available=2,
            ),
        }

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            from bashgym.api.agent_routes import _execute_tool
            result = asyncio.run(_execute_tool("import_traces", {
                    "sources": ["plans"],
                    "dry_run": True,
                }))

        data = json.loads(result)
        assert "plans" in data
        assert data["plans"]["new_available"] == 2
        mock_scanner.scan_all.assert_called_once()


# ---------------------------------------------------------------------------
# CLI subcommand tests (scan, status, --source)
# ---------------------------------------------------------------------------

class TestCLICommands:
    """Tests for bashgym-setup CLI subcommands (scan, status, --source)."""

    def test_scan_subcommand_parses(self, monkeypatch):
        """The 'scan' subcommand is recognized by argparse and calls scanner.scan_all()."""
        from bashgym.trace_capture.setup import main
        from unittest.mock import MagicMock

        monkeypatch.setattr("sys.argv", ["bashgym-setup", "--quiet", "scan"])

        mock_scanner = MagicMock()
        mock_scanner.scan_all.return_value = {}

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            result = main()

        assert result == 0
        mock_scanner.scan_all.assert_called_once()

    def test_status_subcommand_parses(self, monkeypatch):
        """The 'status' subcommand is recognized by argparse and calls scanner.status()."""
        from bashgym.trace_capture.setup import main
        from unittest.mock import MagicMock

        monkeypatch.setattr("sys.argv", ["bashgym-setup", "--quiet", "status"])

        mock_scanner = MagicMock()
        mock_scanner.status.return_value = {}

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            result = main()

        assert result == 0
        mock_scanner.status.assert_called_once()

    def test_import_recent_source_argument_exists(self, monkeypatch):
        """import-recent accepts --source argument and routes to scanner.collect_all()."""
        from bashgym.trace_capture.setup import main
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "sys.argv",
            ["bashgym-setup", "--quiet", "import-recent", "--source", "plans", "--days", "7"],
        )

        mock_scanner = MagicMock()
        mock_scanner.collect_all.return_value = {}

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            result = main()

        assert result == 0
        mock_scanner.collect_all.assert_called_once()

    def test_import_recent_default_source_is_sessions(self, monkeypatch):
        """import-recent without --source defaults to sessions (original behavior)."""
        from bashgym.trace_capture.setup import main
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "sys.argv",
            ["bashgym-setup", "--quiet", "import-recent", "--days", "7"],
        )

        # Patch import_recent to avoid real filesystem access
        mock_import_recent = MagicMock(return_value=[])

        with patch(
            "bashgym.trace_capture.setup.import_recent",
            mock_import_recent,
        ):
            result = main()

        assert result == 0
        mock_import_recent.assert_called_once()
        call_kwargs = mock_import_recent.call_args
        assert call_kwargs[1]["days"] == 7

    def test_import_recent_source_all_routes_to_scanner(self, monkeypatch):
        """import-recent --source all passes sources=None to scanner.collect_all()."""
        from bashgym.trace_capture.setup import main
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            "sys.argv",
            ["bashgym-setup", "--quiet", "import-recent", "--source", "all", "--days", "3"],
        )

        mock_scanner = MagicMock()
        mock_scanner.collect_all.return_value = {}

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            result = main()

        assert result == 0
        mock_scanner.collect_all.assert_called_once()
        call_kwargs = mock_scanner.collect_all.call_args
        # source='all' should pass sources=None to collect everything
        assert call_kwargs[1]["sources"] is None

    def test_scan_verbose_prints_table(self, monkeypatch, capsys):
        """The 'scan' subcommand prints a table when verbose (no --quiet)."""
        from bashgym.trace_capture.setup import main
        from unittest.mock import MagicMock

        monkeypatch.setattr("sys.argv", ["bashgym-setup", "scan"])

        mock_scanner = MagicMock()
        mock_scanner.scan_all.return_value = {
            "plans": CollectorScanResult(
                source_type="plans", total_found=10,
                already_collected=3, new_available=7,
            ),
        }

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Data Scan Results" in captured.out
        assert "plans" in captured.out

    def test_status_verbose_prints_table(self, monkeypatch, capsys):
        """The 'status' subcommand prints a table when verbose (no --quiet)."""
        from bashgym.trace_capture.setup import main
        from unittest.mock import MagicMock

        monkeypatch.setattr("sys.argv", ["bashgym-setup", "status"])

        mock_scanner = MagicMock()
        mock_scanner.status.return_value = {
            "plans": {"total": 10, "collected": 3, "available": 7},
        }

        with patch(
            "bashgym.trace_capture.collectors.scanner.ClaudeDataScanner",
            return_value=mock_scanner,
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Collection Status" in captured.out
        assert "plans" in captured.out


# ---------------------------------------------------------------------------
# 14. DebugCollector
# ---------------------------------------------------------------------------

# A minimal but realistic mock debug log file.  Mirrors the real format found
# at ~/.claude/debug/<session-id>.txt: each line starts with an ISO timestamp,
# a log level tag in brackets, and the log message.

MOCK_DEBUG_LOG = """\
2026-02-20T16:17:40.218Z [DEBUG] [init] configureGlobalMTLS starting
2026-02-20T16:17:40.218Z [DEBUG] [init] configureGlobalMTLS complete
2026-02-20T16:17:40.260Z [DEBUG] Applying permission update: Adding 1 allow rule(s)
2026-02-20T16:17:42.516Z [DEBUG] [SystemPrompt] path=simple proactive=false
2026-02-20T16:17:42.922Z [DEBUG] attribution header x-anthropic-billing-header: cc_version=2.1.49.b45; cc_entrypoint=cli; cch=e76c9;
2026-02-20T16:17:42.923Z [DEBUG] [API:request] Creating client, ANTHROPIC_CUSTOM_HEADERS present: false, has Authorization header: false
2026-02-20T16:17:42.923Z [DEBUG] [API:auth] OAuth token check starting
2026-02-20T16:17:42.924Z [DEBUG] [API:auth] OAuth token check complete
2026-02-20T16:17:43.091Z [DEBUG] autocompact: tokens=2123 threshold=167000 effectiveWindow=180000
2026-02-20T16:17:43.337Z [DEBUG] Tool search disabled for model 'claude-sonnet-4-5-20250929': model does not support tool_reference blocks.
2026-02-20T16:17:44.694Z [DEBUG] Stream started - received first chunk
2026-02-20T16:17:46.500Z [DEBUG] [useDeferredValue] Messages deferred by 2
2026-02-20T16:17:50.000Z [DEBUG] attribution header x-anthropic-billing-header: cc_version=2.1.49.107; cc_entrypoint=cli; cch=00000;
2026-02-20T16:17:50.001Z [DEBUG] [API:request] Creating client, ANTHROPIC_CUSTOM_HEADERS present: false, has Authorization header: false
2026-02-20T16:17:50.002Z [DEBUG] [API:auth] OAuth token check starting
2026-02-20T16:17:50.003Z [DEBUG] [API:auth] OAuth token check complete
2026-02-20T16:17:50.100Z [DEBUG] autocompact: tokens=45674 threshold=167000 effectiveWindow=180000
2026-02-20T16:17:50.200Z [DEBUG] Tool search disabled for model 'claude-sonnet-4-5-20250929': model does not support tool_reference blocks.
2026-02-20T16:17:51.500Z [DEBUG] Stream started - received first chunk
2026-02-20T16:17:55.000Z [ERROR] Error: LSP server plugin:rust-analyzer crashed with exit code 1
2026-02-20T16:17:55.500Z [ERROR] RangeError: stdout maxBuffer length exceeded
"""

MOCK_DEBUG_LOG_WITH_HAIKU = """\
2026-02-20T18:00:00.000Z [DEBUG] [init] configureGlobalMTLS starting
2026-02-20T18:00:01.000Z [DEBUG] Tool search disabled for model 'claude-haiku-4-5-20251001': model does not support tool_reference blocks.
2026-02-20T18:00:01.500Z [DEBUG] [SystemPrompt] path=simple proactive=false
2026-02-20T18:00:02.000Z [DEBUG] attribution header x-anthropic-billing-header: cc_version=2.1.50.6b4; cc_entrypoint=cli; cch=00000;
2026-02-20T18:00:02.001Z [DEBUG] [API:request] Creating client, ANTHROPIC_CUSTOM_HEADERS present: false, has Authorization header: false
2026-02-20T18:00:02.002Z [DEBUG] [API:auth] OAuth token check complete
2026-02-20T18:00:02.100Z [DEBUG] autocompact: tokens=4926 threshold=167000 effectiveWindow=180000
2026-02-20T18:00:03.000Z [DEBUG] Stream started - received first chunk
"""


class TestDebugCollector:
    """Tests for the DebugCollector that parses ~/.claude/debug/<session>.txt."""

    def _make_debug_dir(self, tmp_path, logs: dict):
        """Create a fake ~/.claude/debug/ directory with the given log files.

        Parameters
        ----------
        tmp_path : Path
            pytest temp directory.
        logs : dict
            Mapping of session_id -> log content string.

        Returns
        -------
        tuple of (claude_dir, collected_dir)
        """
        claude_dir = tmp_path / ".claude"
        debug_dir = claude_dir / "debug"
        debug_dir.mkdir(parents=True)
        for session_id, content in logs.items():
            (debug_dir / f"{session_id}.txt").write_text(content, encoding="utf-8")
        collected_dir = tmp_path / "collected"
        collected_dir.mkdir(parents=True)
        return claude_dir, collected_dir

    def test_source_type_is_debug(self, tmp_path):
        """DebugCollector.source_type returns 'debug'."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {})
        collector = DebugCollector(claude_dir, collected_dir)
        assert collector.source_type == "debug"

    def test_scan_finds_debug_files(self, tmp_path):
        """scan() correctly counts debug log files."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "aaa-bbb-ccc": MOCK_DEBUG_LOG,
            "ddd-eee-fff": MOCK_DEBUG_LOG_WITH_HAIKU,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        result = collector.scan()

        assert result.source_type == "debug"
        assert result.total_found == 2
        assert result.new_available == 2
        assert result.already_collected == 0
        assert result.estimated_size_bytes > 0

    def test_scan_respects_since_filter(self, tmp_path):
        """scan(since=...) only returns files modified after the given date."""
        from bashgym.trace_capture.collectors.debug import DebugCollector
        import time

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "old-session": MOCK_DEBUG_LOG,
        })
        # Set the file modification time to the distant past
        old_file = claude_dir / "debug" / "old-session.txt"
        old_time = 1700000000  # 2023-11-14
        os.utime(old_file, (old_time, old_time))

        collector = DebugCollector(claude_dir, collected_dir)
        result = collector.scan(since="2025-01-01T00:00:00Z")

        assert result.total_found == 0

    def test_scan_deduplicates_already_collected(self, tmp_path):
        """scan() subtracts already-collected sessions from new_available."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-001": MOCK_DEBUG_LOG,
            "sess-002": MOCK_DEBUG_LOG_WITH_HAIKU,
        })
        collector = DebugCollector(claude_dir, collected_dir)

        # Simulate sess-001 already collected
        collector._save_collected_id("sess-001")

        result = collector.scan()
        assert result.total_found == 2
        assert result.already_collected == 1
        assert result.new_available == 1

    def test_scan_returns_zero_when_no_debug_dir(self, tmp_path):
        """scan() returns zeros when the debug directory doesn't exist."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir = tmp_path / ".claude"  # no debug/ subdirectory
        claude_dir.mkdir(parents=True)
        collected_dir = tmp_path / "collected"
        collected_dir.mkdir(parents=True)

        collector = DebugCollector(claude_dir, collected_dir)
        result = collector.scan()
        assert result.total_found == 0
        assert result.new_available == 0

    def test_collect_single_session(self, tmp_path):
        """collect() parses a single debug log and returns a DebugRecord."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")

        assert len(records) == 1
        record = records[0]
        assert record.session_id == "sess-abc"
        assert record.source_type == "debug"

    def test_collect_extracts_api_call_count(self, tmp_path):
        """collect() counts API calls from '[API:request]' lines."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        # MOCK_DEBUG_LOG has 2 "[API:request]" lines
        assert record.api_call_count == 2

    def test_collect_extracts_models_used(self, tmp_path):
        """collect() extracts unique model names from debug logs."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        assert "claude-sonnet-4-5-20250929" in record.metadata.get("models_used", [])

    def test_collect_extracts_token_counts(self, tmp_path):
        """collect() sums token counts from autocompact lines."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        # Should extract max tokens seen (45674 from second autocompact line)
        assert record.metadata.get("max_tokens_seen", 0) == 45674
        # Should track all token snapshots
        assert len(record.metadata.get("token_snapshots", [])) == 2

    def test_collect_detects_system_prompt(self, tmp_path):
        """collect() detects [SystemPrompt] entries."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        # MOCK_DEBUG_LOG has one [SystemPrompt] line
        assert len(record.system_prompts) == 1
        assert "path=simple" in record.system_prompts[0]

    def test_collect_extracts_errors(self, tmp_path):
        """collect() captures [ERROR] log lines."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        # MOCK_DEBUG_LOG has 2 [ERROR] lines
        assert len(record.errors) == 2
        assert any("LSP server" in e for e in record.errors)
        assert any("maxBuffer" in e for e in record.errors)

    def test_collect_computes_latencies(self, tmp_path):
        """collect() computes latency between API request and stream start."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        # total_latency_ms should be sum of latencies for both API calls
        assert record.total_latency_ms > 0
        assert len(record.metadata.get("latencies_ms", [])) == 2

    def test_collect_missing_session_returns_empty(self, tmp_path):
        """collect() returns an empty list when the session file doesn't exist."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {})
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("nonexistent-session")

        assert records == []

    def test_collect_all_processes_all_files(self, tmp_path):
        """collect_all() processes all debug log files and returns a batch result."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-001": MOCK_DEBUG_LOG,
            "sess-002": MOCK_DEBUG_LOG_WITH_HAIKU,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        result = collector.collect_all()

        assert result.source_type == "debug"
        assert result.collected == 2
        assert result.skipped == 0
        assert len(result.records) == 2
        assert len(result.errors) == 0

    def test_collect_all_deduplicates(self, tmp_path):
        """collect_all() skips already-collected sessions."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-001": MOCK_DEBUG_LOG,
            "sess-002": MOCK_DEBUG_LOG_WITH_HAIKU,
        })
        collector = DebugCollector(claude_dir, collected_dir)

        # Pre-mark sess-001 as collected
        collector._save_collected_id("sess-001")

        result = collector.collect_all()
        assert result.collected == 1
        assert result.skipped == 1

    def test_collect_all_with_since_filter(self, tmp_path):
        """collect_all(since=...) respects time filter."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-old": MOCK_DEBUG_LOG,
        })
        old_file = claude_dir / "debug" / "sess-old.txt"
        old_time = 1700000000  # 2023-11-14
        os.utime(old_file, (old_time, old_time))

        collector = DebugCollector(claude_dir, collected_dir)
        result = collector.collect_all(since="2025-01-01T00:00:00Z")
        assert result.collected == 0

    def test_collect_extracts_multiple_models(self, tmp_path):
        """collect() identifies multiple distinct model names."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        mixed_log = """\
2026-02-20T18:00:00.000Z [DEBUG] [init] configureGlobalMTLS starting
2026-02-20T18:00:01.000Z [DEBUG] Tool search disabled for model 'claude-haiku-4-5-20251001': not supported.
2026-02-20T18:00:02.000Z [DEBUG] Tool search disabled for model 'claude-sonnet-4-5-20250929': not supported.
2026-02-20T18:00:03.000Z [DEBUG] Tool search disabled for model 'claude-haiku-4-5-20251001': not supported.
"""
        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-multi": mixed_log,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-multi")
        record = records[0]

        models = record.metadata.get("models_used", [])
        assert "claude-haiku-4-5-20251001" in models
        assert "claude-sonnet-4-5-20250929" in models
        # Should be deduplicated
        assert len(models) == 2

    def test_record_serializes_to_json(self, tmp_path):
        """DebugRecord produced by collect() is JSON-serializable."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        # Should not raise
        data = record.to_json()
        serialized = json.dumps(data)
        assert "sess-abc" in serialized

    def test_collect_stores_session_start_end_times(self, tmp_path):
        """collect() stores the first and last timestamps from the log."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        # The timestamp field should be set to the first log entry timestamp
        assert "2026-02-20" in record.timestamp
        # Metadata should contain the session duration info
        assert "first_timestamp" in record.metadata
        assert "last_timestamp" in record.metadata

    def test_collect_does_not_store_raw_content(self, tmp_path):
        """collect() must NOT store raw debug log content (PII safety)."""
        from bashgym.trace_capture.collectors.debug import DebugCollector

        claude_dir, collected_dir = self._make_debug_dir(tmp_path, {
            "sess-abc": MOCK_DEBUG_LOG,
        })
        collector = DebugCollector(claude_dir, collected_dir)
        records = collector.collect("sess-abc")
        record = records[0]

        data = json.dumps(record.to_json())
        # Should not contain raw log content like file paths from debug
        assert "configureGlobalMTLS" not in data
        assert "OAuth token check" not in data

    def test_debug_in_scanner_all_sources(self):
        """'debug' should be in ALL_SOURCES list."""
        from bashgym.trace_capture.collectors.scanner import ALL_SOURCES
        assert "debug" in ALL_SOURCES

    def test_scanner_includes_debug_collector(self):
        """ClaudeDataScanner should include the DebugCollector."""
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        assert "debug" in scanner._collectors

    def test_debug_collector_exported_from_init(self):
        """DebugCollector should be importable from the collectors package."""
        from bashgym.trace_capture.collectors import DebugCollector
        assert DebugCollector is not None
