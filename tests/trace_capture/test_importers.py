"""
Tests for new session history importers (Gemini, Copilot, OpenCode).

Tests cover:
  - GeminiSessionImporter: parsing session files, empty directories
  - CopilotSessionImporter: accept/reject metadata, session parsing
  - OpenCodeSessionImporter: file-based parsing, CLI fallback
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from bashgym.trace_capture.importers.gemini_history import GeminiSessionImporter
from bashgym.trace_capture.importers.copilot_history import CopilotSessionImporter
from bashgym.trace_capture.importers.opencode_history import OpenCodeSessionImporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> Path:
    """Write JSON data to a file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. GeminiSessionImporter
# ---------------------------------------------------------------------------

class TestGeminiSessionImporter:
    """Tests for Gemini CLI session import."""

    def _make_importer(self, tmp_path, monkeypatch):
        """Create a GeminiSessionImporter pointing at tmp_path."""
        bashgym_dir = tmp_path / ".bashgym"
        bashgym_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = bashgym_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            "bashgym.trace_capture.importers.gemini_history.GeminiSessionImporter._get_gemini_dir",
            staticmethod(lambda: tmp_path / ".gemini"),
        )
        # Patch TraceCapture to use tmp_path
        with patch("bashgym.trace_capture.importers.gemini_history.TraceCapture") as mock_tc:
            mock_tc_instance = mock_tc.return_value
            mock_tc_instance.bashgym_dir = bashgym_dir
            mock_tc_instance.traces_dir = traces_dir
            importer = GeminiSessionImporter()
        return importer

    def test_empty_directory_returns_empty(self, tmp_path, monkeypatch):
        """Empty directory yields no sessions."""
        importer = self._make_importer(tmp_path, monkeypatch)
        assert importer.find_session_files() == []

    def test_parse_valid_session_format_a(self, tmp_path, monkeypatch):
        """Parse a session with functionCall/functionResponse (Format A)."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_data = [
            {
                "role": "user",
                "parts": [{"text": "List files in the current directory"}],
                "timestamp": "2026-02-24T10:00:00Z",
            },
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "read_file", "args": {"path": "main.py"}}}
                ],
                "timestamp": "2026-02-24T10:00:01Z",
            },
            {
                "role": "tool",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "read_file",
                            "response": {"content": "print('hello')"},
                        }
                    }
                ],
                "timestamp": "2026-02-24T10:00:02Z",
            },
        ]

        session_file = tmp_path / "session-test01.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_file(session_file)

        assert len(steps) >= 1
        assert steps[0].source_tool == "gemini_cli"
        assert steps[0].tool_name == "read_file"
        assert meta["user_initial_prompt"] == "List files in the current directory"
        assert meta["conversation_turns"] == 1

    def test_parse_valid_session_format_c(self, tmp_path, monkeypatch):
        """Parse a session with flat tool metadata fields (Format C)."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_data = [
            {"role": "user", "parts": [{"text": "Run tests"}]},
            {
                "tool_name": "bash",
                "tool_input": {"command": "pytest"},
                "tool_output": "2 passed",
                "timestamp": "2026-02-24T10:00:01Z",
            },
        ]

        session_file = tmp_path / "session-test02.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_file(session_file)

        assert len(steps) == 1
        assert steps[0].tool_name == "bash"
        assert steps[0].output == "2 passed"
        assert steps[0].source_tool == "gemini_cli"

    def test_parse_empty_session(self, tmp_path, monkeypatch):
        """Empty JSON array returns no steps."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_file = tmp_path / "session-empty.json"
        _write_json(session_file, [])

        steps, meta = importer.parse_session_file(session_file)
        assert steps == []
        assert meta["user_initial_prompt"] is None

    def test_parse_session_dict_with_messages_key(self, tmp_path, monkeypatch):
        """Parse when session data is a dict with a 'messages' key."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_data = {
            "messages": [
                {"role": "user", "parts": [{"text": "Hello Gemini"}]},
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "test"}}}
                    ],
                },
            ]
        }

        session_file = tmp_path / "session-dict.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_file(session_file)
        assert len(steps) == 1
        assert steps[0].tool_name == "search"
        assert meta["user_initial_prompt"] == "Hello Gemini"

    def test_malformed_json_returns_empty(self, tmp_path, monkeypatch):
        """Malformed JSON gracefully returns empty results."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_file = tmp_path / "session-bad.json"
        session_file.write_text("not valid json{{{", encoding="utf-8")

        steps, meta = importer.parse_session_file(session_file)
        assert steps == []
        assert meta["user_initial_prompt"] is None


# ---------------------------------------------------------------------------
# 2. CopilotSessionImporter
# ---------------------------------------------------------------------------

class TestCopilotSessionImporter:
    """Tests for Copilot CLI session import."""

    def _make_importer(self, tmp_path, monkeypatch):
        """Create a CopilotSessionImporter pointing at tmp_path."""
        bashgym_dir = tmp_path / ".bashgym"
        bashgym_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = bashgym_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            "bashgym.trace_capture.importers.copilot_history.CopilotSessionImporter._get_copilot_dir",
            staticmethod(lambda: tmp_path / ".copilot"),
        )
        with patch("bashgym.trace_capture.importers.copilot_history.TraceCapture") as mock_tc:
            mock_tc_instance = mock_tc.return_value
            mock_tc_instance.bashgym_dir = bashgym_dir
            mock_tc_instance.traces_dir = traces_dir
            importer = CopilotSessionImporter()
        return importer

    def test_empty_directory_returns_empty(self, tmp_path, monkeypatch):
        """Empty directory yields no sessions."""
        importer = self._make_importer(tmp_path, monkeypatch)
        assert importer.find_session_files() == []

    def test_parse_session_with_suggestions(self, tmp_path, monkeypatch):
        """Parse session with accept/reject suggestions (Pattern A)."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_data = {
            "messages": [
                {"role": "user", "content": "Deploy the service", "timestamp": "2026-02-24T10:00:00Z"},
            ],
            "suggestions": [
                {"command": "docker compose up -d", "accepted": True, "output": "Starting..."},
                {"command": "rm -rf /", "accepted": False},
                {"command": "kubectl apply -f deploy.yaml", "accepted": True, "output": "deployed"},
            ],
        }

        session_file = tmp_path / "session-sug.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_file(session_file)

        # Should capture both messages and suggestions
        assert len(steps) >= 3  # 3 suggestions
        assert meta["accepted_commands"] == 2
        assert meta["rejected_commands"] == 1
        assert meta["user_initial_prompt"] == "Deploy the service"

        # Check user_accepted metadata is set
        suggestion_steps = [s for s in steps if s.tool_name == "command_suggestion"]
        assert any(s.metadata.get("user_accepted") is True for s in suggestion_steps)
        assert any(s.metadata.get("user_accepted") is False for s in suggestion_steps)

    def test_parse_session_with_commands(self, tmp_path, monkeypatch):
        """Parse session with flat command history (Pattern C)."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_data = {
            "commands": [
                {
                    "proposed": "git status",
                    "accepted": True,
                    "executed": "git status",
                    "output": "On branch main",
                    "exit_code": 0,
                },
                {
                    "proposed": "git push --force",
                    "accepted": False,
                    "correction": "git push",
                },
            ],
        }

        session_file = tmp_path / "session-cmds.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_file(session_file)

        assert len(steps) == 2
        assert meta["accepted_commands"] == 1
        assert meta["rejected_commands"] == 1

        # First command was accepted and has output
        assert steps[0].tool_name == "bash"
        assert steps[0].output == "On branch main"
        assert steps[0].metadata["user_accepted"] is True

        # Second command was rejected with correction
        assert steps[1].metadata["user_accepted"] is False
        assert steps[1].metadata["user_correction"] == "git push"

    def test_parse_session_with_tool_calls(self, tmp_path, monkeypatch):
        """Parse session with tool_calls in messages (Pattern A)."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_data = {
            "messages": [
                {"role": "user", "content": "Read the file", "timestamp": "2026-02-24T10:00:00Z"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "read_file", "arguments": {"path": "main.py"}, "output": "print('hello')"}
                    ],
                    "timestamp": "2026-02-24T10:00:01Z",
                },
            ],
        }

        session_file = tmp_path / "session-tc.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_file(session_file)
        assert len(steps) == 1
        assert steps[0].tool_name == "read_file"
        assert steps[0].source_tool == "copilot_cli"

    def test_parse_empty_session(self, tmp_path, monkeypatch):
        """Empty JSON object returns no steps."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_file = tmp_path / "session-empty.json"
        _write_json(session_file, {})

        steps, meta = importer.parse_session_file(session_file)
        assert steps == []

    def test_parse_malformed_json(self, tmp_path, monkeypatch):
        """Malformed JSON gracefully returns empty results."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_file = tmp_path / "session-bad.json"
        session_file.write_text("{broken json", encoding="utf-8")

        steps, meta = importer.parse_session_file(session_file)
        assert steps == []
        assert meta["user_initial_prompt"] is None


# ---------------------------------------------------------------------------
# 3. OpenCodeSessionImporter
# ---------------------------------------------------------------------------

class TestOpenCodeSessionImporter:
    """Tests for OpenCode session import."""

    def _make_importer(self, tmp_path, monkeypatch):
        """Create an OpenCodeSessionImporter pointing at tmp_path."""
        bashgym_dir = tmp_path / ".bashgym"
        bashgym_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = bashgym_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        storage_dir = tmp_path / "opencode_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            "bashgym.trace_capture.importers.opencode_history.OpenCodeSessionImporter._get_storage_dirs",
            staticmethod(lambda: [storage_dir]),
        )
        with patch("bashgym.trace_capture.importers.opencode_history.TraceCapture") as mock_tc:
            mock_tc_instance = mock_tc.return_value
            mock_tc_instance.bashgym_dir = bashgym_dir
            mock_tc_instance.traces_dir = traces_dir
            importer = OpenCodeSessionImporter()
        return importer

    def test_empty_directory_returns_empty(self, tmp_path, monkeypatch):
        """Empty storage directory yields no session files."""
        importer = self._make_importer(tmp_path, monkeypatch)
        assert importer.find_session_files() == []

    def test_cli_not_available(self, tmp_path, monkeypatch):
        """Falls back gracefully when CLI not available."""
        importer = self._make_importer(tmp_path, monkeypatch)
        # Patch shutil.which to return None for opencode
        monkeypatch.setattr("shutil.which", lambda _: None)
        importer._opencode_available = None  # Reset cache

        assert importer.opencode_available is False

    def test_parse_session_with_inline_messages(self, tmp_path, monkeypatch):
        """Parse session with inline messages containing tool_calls."""
        importer = self._make_importer(tmp_path, monkeypatch)

        # Create session file in the expected structure:
        # storage/session/<projectID>/<sessionID>.json
        storage_dir = importer.storage_dirs[0]
        session_dir = storage_dir / "session" / "proj1"
        session_dir.mkdir(parents=True)

        session_data = {
            "title": "Test Session",
            "project_id": "proj1",
            "messages": [
                {"role": "user", "content": "Fix the bug"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "bash", "input": {"command": "ls -la"}, "output": "total 42"}
                    ],
                },
            ],
        }

        session_file = session_dir / "sess-001.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_from_files(session_file)

        assert len(steps) == 1
        assert steps[0].tool_name == "bash"
        assert steps[0].source_tool == "opencode"
        assert meta["user_initial_prompt"] == "Fix the bug"
        assert meta["opencode_session_title"] == "Test Session"

    def test_parse_session_with_content_blocks(self, tmp_path, monkeypatch):
        """Parse session with tool_use/tool_result content blocks."""
        importer = self._make_importer(tmp_path, monkeypatch)

        storage_dir = importer.storage_dirs[0]
        session_dir = storage_dir / "session" / "proj2"
        session_dir.mkdir(parents=True)

        session_data = {
            "title": "Content Block Session",
            "project_id": "proj2",
            "messages": [
                {"role": "user", "content": "Read the config"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_001",
                            "name": "read_file",
                            "input": {"path": "config.json"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_001",
                            "content": '{"key": "value"}',
                            "is_error": False,
                        }
                    ],
                },
            ],
        }

        session_file = session_dir / "sess-002.json"
        _write_json(session_file, session_data)

        steps, meta = importer.parse_session_from_files(session_file)

        assert len(steps) >= 1
        # The tool_use step should have been matched with the tool_result
        read_steps = [s for s in steps if s.tool_name == "read_file"]
        assert len(read_steps) >= 1
        assert meta["user_initial_prompt"] == "Read the config"

    def test_parse_empty_session(self, tmp_path, monkeypatch):
        """Empty session file returns no steps."""
        importer = self._make_importer(tmp_path, monkeypatch)

        storage_dir = importer.storage_dirs[0]
        session_dir = storage_dir / "session" / "proj3"
        session_dir.mkdir(parents=True)

        session_file = session_dir / "sess-empty.json"
        _write_json(session_file, {"title": "Empty", "messages": []})

        steps, meta = importer.parse_session_from_files(session_file)
        assert steps == []

    def test_parse_malformed_json(self, tmp_path, monkeypatch):
        """Malformed JSON gracefully returns empty results."""
        importer = self._make_importer(tmp_path, monkeypatch)

        storage_dir = importer.storage_dirs[0]
        session_dir = storage_dir / "session" / "proj4"
        session_dir.mkdir(parents=True)

        session_file = session_dir / "sess-bad.json"
        session_file.write_text("not json!", encoding="utf-8")

        steps, meta = importer.parse_session_from_files(session_file)
        assert steps == []
        assert meta["user_initial_prompt"] is None

    def test_parse_cli_export(self, tmp_path, monkeypatch):
        """Parse session data from CLI export format."""
        importer = self._make_importer(tmp_path, monkeypatch)

        session_data = {
            "title": "CLI Export Session",
            "messages": [
                {"role": "user", "content": "Write a test"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "write_file", "input": {"path": "test.py"}, "output": "ok"}
                    ],
                },
            ],
        }

        steps, meta = importer.parse_session_from_cli(session_data, "cli-sess-001")

        assert len(steps) == 1
        assert steps[0].tool_name == "write_file"
        assert steps[0].source_tool == "opencode"
        assert meta["user_initial_prompt"] == "Write a test"
        assert meta["opencode_session_title"] == "CLI Export Session"

    def test_find_session_files(self, tmp_path, monkeypatch):
        """find_session_files discovers files in storage/session/*/*.json."""
        importer = self._make_importer(tmp_path, monkeypatch)

        storage_dir = importer.storage_dirs[0]
        session_dir = storage_dir / "session" / "proj_find"
        session_dir.mkdir(parents=True)

        # Create a couple session files
        _write_json(session_dir / "sess-a.json", {"title": "A"})
        _write_json(session_dir / "sess-b.json", {"title": "B"})

        files = importer.find_session_files()
        assert len(files) == 2
        # Each entry is (Path, datetime)
        for f, mtime in files:
            assert f.suffix == ".json"
            assert isinstance(mtime, datetime)
