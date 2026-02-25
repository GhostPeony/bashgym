"""Tests for ThresholdMonitor watermark logic."""

import pytest
from pathlib import Path

from bashgym.pipeline.config import PipelineConfig
from bashgym.pipeline.threshold_monitor import ThresholdMonitor


class TestThresholdMonitor:

    def test_generate_threshold_not_reached(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        for i in range(3):
            (gold_dir / f"trace_{i}.json").write_text("{}")

        monitor = ThresholdMonitor(PipelineConfig(generate_enabled=True, generate_gold_threshold=10))
        assert monitor.should_generate(gold_dir) is False

    def test_generate_threshold_reached(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        for i in range(10):
            (gold_dir / f"trace_{i}.json").write_text("{}")

        monitor = ThresholdMonitor(PipelineConfig(generate_enabled=True, generate_gold_threshold=10))
        assert monitor.should_generate(gold_dir) is True

    def test_generate_disabled(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        for i in range(20):
            (gold_dir / f"trace_{i}.json").write_text("{}")

        monitor = ThresholdMonitor(PipelineConfig(generate_enabled=False))
        assert monitor.should_generate(gold_dir) is False

    def test_watermark_prevents_retrigger(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        for i in range(10):
            (gold_dir / f"trace_{i}.json").write_text("{}")

        monitor = ThresholdMonitor(PipelineConfig(generate_enabled=True, generate_gold_threshold=10))
        assert monitor.should_generate(gold_dir) is True

        monitor.mark_generate_triggered(gold_dir)
        assert monitor.should_generate(gold_dir) is False

        for i in range(10, 20):
            (gold_dir / f"trace_{i}.json").write_text("{}")
        assert monitor.should_generate(gold_dir) is True

    def test_train_threshold(self, tmp_path):
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        with open(examples_dir / "train.jsonl", "w") as f:
            for i in range(100):
                f.write('{"messages": []}\n')

        monitor = ThresholdMonitor(PipelineConfig(train_enabled=True, train_examples_threshold=100))
        assert monitor.should_train(examples_dir / "train.jsonl") is True

    def test_train_threshold_not_reached(self, tmp_path):
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        with open(examples_dir / "train.jsonl", "w") as f:
            for i in range(5):
                f.write('{"messages": []}\n')

        monitor = ThresholdMonitor(PipelineConfig(train_enabled=True, train_examples_threshold=100))
        assert monitor.should_train(examples_dir / "train.jsonl") is False

    def test_train_disabled(self, tmp_path):
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        with open(examples_dir / "train.jsonl", "w") as f:
            for i in range(200):
                f.write('{"messages": []}\n')

        monitor = ThresholdMonitor(PipelineConfig(train_enabled=False))
        assert monitor.should_train(examples_dir / "train.jsonl") is False

    def test_watermark_persists_across_instances(self, tmp_path):
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        for i in range(10):
            (gold_dir / f"trace_{i}.json").write_text("{}")

        watermark_file = tmp_path / "watermarks.json"

        m1 = ThresholdMonitor(
            PipelineConfig(generate_enabled=True, generate_gold_threshold=10),
            watermark_path=watermark_file,
        )
        assert m1.should_generate(gold_dir) is True
        m1.mark_generate_triggered(gold_dir)

        m2 = ThresholdMonitor(
            PipelineConfig(generate_enabled=True, generate_gold_threshold=10),
            watermark_path=watermark_file,
        )
        assert m2.should_generate(gold_dir) is False
