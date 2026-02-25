"""Tests for PipelineConfig."""

import json
import pytest
from bashgym.pipeline.config import PipelineConfig


class TestPipelineConfig:

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.watch_enabled is True
        assert cfg.watch_debounce_seconds == 30
        assert cfg.classify_enabled is True
        assert cfg.classify_gold_min_success_rate == 0.8
        assert cfg.classify_gold_min_steps == 10
        assert cfg.classify_fail_max_success_rate == 0.3
        assert cfg.generate_enabled is False
        assert cfg.generate_gold_threshold == 10
        assert cfg.train_enabled is False
        assert cfg.train_examples_threshold == 100

    def test_to_dict(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["watch_enabled"] is True
        assert d["classify_gold_min_success_rate"] == 0.8

    def test_from_dict(self):
        d = {"watch_enabled": False, "watch_debounce_seconds": 60}
        cfg = PipelineConfig.from_dict(d)
        assert cfg.watch_enabled is False
        assert cfg.watch_debounce_seconds == 60
        assert cfg.classify_enabled is True

    def test_save_and_load(self, tmp_path):
        cfg = PipelineConfig(watch_debounce_seconds=45, generate_enabled=True)
        path = tmp_path / "pipeline_config.json"
        cfg.save(path)
        loaded = PipelineConfig.load(path)
        assert loaded.watch_debounce_seconds == 45
        assert loaded.generate_enabled is True

    def test_load_missing_file_returns_defaults(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cfg = PipelineConfig.load(path)
        assert cfg.watch_enabled is True

    def test_load_corrupted_file_returns_defaults(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")
        cfg = PipelineConfig.load(path)
        assert cfg.watch_enabled is True
