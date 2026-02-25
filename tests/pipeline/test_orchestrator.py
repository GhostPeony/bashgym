"""Tests for Pipeline orchestrator."""

import json
import pytest
from pathlib import Path

from bashgym.pipeline.config import PipelineConfig
from bashgym.pipeline.orchestrator import Pipeline


class TestPipeline:

    def test_init(self, tmp_path):
        pipeline = Pipeline(config_path=tmp_path / "config.json", bashgym_dir=tmp_path / "bashgym")
        assert pipeline.config.watch_enabled is True

    def test_on_session_file_imports_and_classifies(self, tmp_path):
        """Full flow: file detected -> import -> classify -> route."""
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        session_file = session_dir / "test.jsonl"
        events = [
            {"type": "user", "message": {"content": "fix bug"}, "cwd": "/repo"},
            {
                "type": "assistant",
                "timestamp": "2026-02-24T12:00:00Z",
                "message": {
                    "model": "claude-sonnet-4-5",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                    ],
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "file.py", "is_error": False},
                    ]
                },
            },
        ]
        with open(session_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        cfg = PipelineConfig(classify_gold_min_steps=1, classify_gold_min_success_rate=0.5)
        pipeline = Pipeline(config_path=tmp_path / "config.json", bashgym_dir=tmp_path / "bashgym")
        pipeline.config = cfg
        pipeline._gate.config = cfg

        # Ensure the session ID is not already marked as imported from prior runs
        pipeline._importer.imported_sessions.discard("test")

        result = pipeline.handle_session_file(session_file)
        assert result is not None
        assert result["steps_imported"] >= 1
        assert result["classification"] in ("gold", "pending", "failed")

    def test_get_status_returns_counts(self, tmp_path):
        pipeline = Pipeline(config_path=tmp_path / "config.json", bashgym_dir=tmp_path / "bashgym")
        status = pipeline.get_status()
        assert "watcher_running" in status
        assert "gold_count" in status
        assert "pending_count" in status
        assert "failed_count" in status
