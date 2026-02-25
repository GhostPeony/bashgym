"""Tests for QualityGate classification logic."""

import pytest
from pathlib import Path

from bashgym.pipeline.config import PipelineConfig
from bashgym.pipeline.quality_gate import QualityGate, Classification


class TestQualityGate:

    def test_high_quality_classified_gold(self):
        gate = QualityGate(PipelineConfig())
        result = gate.classify(success_rate=0.9, step_count=15)
        assert result == Classification.GOLD

    def test_low_quality_classified_failed(self):
        gate = QualityGate(PipelineConfig())
        result = gate.classify(success_rate=0.2, step_count=15)
        assert result == Classification.FAILED

    def test_borderline_classified_pending(self):
        gate = QualityGate(PipelineConfig())
        result = gate.classify(success_rate=0.5, step_count=15)
        assert result == Classification.PENDING

    def test_too_few_steps_is_pending(self):
        gate = QualityGate(PipelineConfig())
        result = gate.classify(success_rate=1.0, step_count=3)
        assert result == Classification.PENDING

    def test_custom_thresholds(self):
        cfg = PipelineConfig(
            classify_gold_min_success_rate=0.5,
            classify_gold_min_steps=5,
            classify_fail_max_success_rate=0.1,
        )
        gate = QualityGate(cfg)
        result = gate.classify(success_rate=0.6, step_count=8)
        assert result == Classification.GOLD

    def test_classify_disabled_returns_pending(self):
        cfg = PipelineConfig(classify_enabled=False)
        gate = QualityGate(cfg)
        result = gate.classify(success_rate=0.95, step_count=50)
        assert result == Classification.PENDING

    def test_route_trace_moves_to_gold_dir(self, tmp_path):
        gold_dir = tmp_path / "gold_traces"
        failed_dir = tmp_path / "failed_traces"
        gold_dir.mkdir()
        failed_dir.mkdir()

        trace_file = tmp_path / "traces" / "test.json"
        trace_file.parent.mkdir()
        trace_file.write_text('{"trace": []}')

        gate = QualityGate(PipelineConfig())
        dest = gate.route_trace(trace_file, Classification.GOLD, gold_dir, failed_dir)
        assert dest.parent == gold_dir
        assert dest.exists()
        assert not trace_file.exists()

    def test_route_trace_moves_to_failed_dir(self, tmp_path):
        gold_dir = tmp_path / "gold_traces"
        failed_dir = tmp_path / "failed_traces"
        gold_dir.mkdir()
        failed_dir.mkdir()

        trace_file = tmp_path / "traces" / "test.json"
        trace_file.parent.mkdir()
        trace_file.write_text('{"trace": []}')

        gate = QualityGate(PipelineConfig())
        dest = gate.route_trace(trace_file, Classification.FAILED, gold_dir, failed_dir)
        assert dest.parent == failed_dir

    def test_route_pending_stays_in_place(self, tmp_path):
        trace_file = tmp_path / "traces" / "test.json"
        trace_file.parent.mkdir()
        trace_file.write_text('{"trace": []}')

        gate = QualityGate(PipelineConfig())
        dest = gate.route_trace(trace_file, Classification.PENDING, tmp_path / "gold", tmp_path / "failed")
        assert dest == trace_file
