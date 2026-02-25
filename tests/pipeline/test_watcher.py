"""Tests for ImportWatcher filesystem watcher."""

import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from bashgym.pipeline.config import PipelineConfig
from bashgym.pipeline.watcher import ImportWatcher


class TestImportWatcher:

    def test_init_creates_watcher(self, tmp_path):
        cfg = PipelineConfig(watch_debounce_seconds=1)
        watcher = ImportWatcher(
            config=cfg,
            watch_dir=tmp_path,
            on_import=MagicMock(),
        )
        assert watcher.config == cfg
        assert watcher._running is False

    def test_start_and_stop(self, tmp_path):
        watcher = ImportWatcher(
            config=PipelineConfig(watch_debounce_seconds=1),
            watch_dir=tmp_path,
            on_import=MagicMock(),
        )
        watcher.start()
        assert watcher._running is True
        watcher.stop()
        assert watcher._running is False

    def test_detects_new_jsonl_file(self, tmp_path):
        callback = MagicMock()
        watcher = ImportWatcher(
            config=PipelineConfig(watch_debounce_seconds=1),
            watch_dir=tmp_path,
            on_import=callback,
        )
        watcher.start()
        try:
            project_dir = tmp_path / "test-project"
            project_dir.mkdir()

            session = project_dir / "test-session.jsonl"
            session.write_text('{"type": "user", "message": {"content": "hi"}}\n')

            # Wait for debounce + processing
            time.sleep(3)

            assert callback.call_count >= 1
            called_path = callback.call_args[0][0]
            assert called_path.name == "test-session.jsonl"
        finally:
            watcher.stop()

    def test_disabled_watcher_does_not_start(self, tmp_path):
        watcher = ImportWatcher(
            config=PipelineConfig(watch_enabled=False),
            watch_dir=tmp_path,
            on_import=MagicMock(),
        )
        watcher.start()
        assert watcher._running is False

    def test_missing_watch_dir_waits(self, tmp_path):
        missing = tmp_path / "nonexistent"
        watcher = ImportWatcher(
            config=PipelineConfig(watch_debounce_seconds=1),
            watch_dir=missing,
            on_import=MagicMock(),
        )
        watcher.start()
        assert watcher._running is True
        watcher.stop()
