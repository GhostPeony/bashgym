"""Integration test: file appears → import → classify → threshold check."""

import json
import pytest
from pathlib import Path

from bashgym.pipeline.config import PipelineConfig
from bashgym.pipeline.orchestrator import Pipeline


class TestPipelineIntegration:

    def test_full_flow_with_direct_handling(self, tmp_path):
        """Write a JSONL file, verify the pipeline imports and classifies it."""
        events_received = []

        def on_event(event_type, payload):
            events_received.append((event_type, payload))

        # Set up directories
        sessions_dir = tmp_path / "claude" / "projects" / "test-project"
        sessions_dir.mkdir(parents=True)
        bashgym_dir = tmp_path / "bashgym"

        cfg = PipelineConfig(
            watch_debounce_seconds=1,
            classify_gold_min_steps=1,
            classify_gold_min_success_rate=0.5,
        )
        cfg_path = tmp_path / "config.json"
        cfg.save(cfg_path)

        pipeline = Pipeline(
            config_path=cfg_path,
            bashgym_dir=bashgym_dir,
            on_event=on_event,
        )

        # Override importer's claude_dir so it finds our temp files
        pipeline._importer.claude_dir = tmp_path / "claude"

        # Write a session file with a successful tool call
        session_file = sessions_dir / "test-session.jsonl"
        session_events = [
            {"type": "user", "message": {"content": "fix it"}, "cwd": "/repo"},
            {
                "type": "assistant", "timestamp": "2026-02-24T12:00:00Z",
                "message": {
                    "model": "claude-sonnet-4-5",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Bash",
                         "input": {"command": "echo fix"}},
                    ],
                },
            },
            {
                "type": "user", "message": {"content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": "fix", "is_error": False},
                ]},
            },
        ]
        with open(session_file, "w") as f:
            for e in session_events:
                f.write(json.dumps(e) + "\n")

        # Ensure the session isn't in the imported set
        pipeline._importer.imported_sessions.discard("test-session")

        # Process directly (skip watcher timing for deterministic test)
        result = pipeline.handle_session_file(session_file)

        assert result is not None
        assert result["steps_imported"] >= 1
        assert result["classification"] in ("gold", "pending", "failed")
        assert len(events_received) >= 2  # import + classified
        assert events_received[0][0] == "pipeline:import"
        assert events_received[1][0] == "pipeline:classified"

    def test_config_hot_reload(self, tmp_path):
        """Verify config changes propagate through the pipeline."""
        bashgym_dir = tmp_path / "bashgym"
        cfg = PipelineConfig(classify_gold_min_steps=10)
        cfg_path = tmp_path / "config.json"
        cfg.save(cfg_path)

        pipeline = Pipeline(config_path=cfg_path, bashgym_dir=bashgym_dir)
        assert pipeline.config.classify_gold_min_steps == 10

        new_config = pipeline.save_config({"classify_gold_min_steps": 5})
        assert new_config.classify_gold_min_steps == 5
        assert pipeline.config.classify_gold_min_steps == 5
        assert pipeline._gate.config.classify_gold_min_steps == 5

    def test_get_status_reflects_trace_dirs(self, tmp_path):
        """Verify get_status() counts trace files in correct directories."""
        bashgym_dir = tmp_path / "bashgym"
        cfg_path = tmp_path / "config.json"
        PipelineConfig().save(cfg_path)

        pipeline = Pipeline(config_path=cfg_path, bashgym_dir=bashgym_dir)

        # Override trace dirs to use temp directories
        gold_dir = tmp_path / "gold"
        pending_dir = tmp_path / "pending"
        failed_dir = tmp_path / "failed"
        gold_dir.mkdir()
        pending_dir.mkdir()
        failed_dir.mkdir()

        pipeline._trace_capture.gold_traces_dir = gold_dir
        pipeline._trace_capture.traces_dir = pending_dir
        pipeline._trace_capture.failed_traces_dir = failed_dir

        (gold_dir / "g1.json").write_text("{}")
        (gold_dir / "g2.json").write_text("{}")
        (pending_dir / "p1.json").write_text("{}")
        (failed_dir / "f1.json").write_text("{}")

        status = pipeline.get_status()
        assert status["gold_count"] == 2
        assert status["pending_count"] == 1
        assert status["failed_count"] == 1
        assert status["watcher_running"] is False
