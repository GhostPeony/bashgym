"""Threshold monitor with watermark tracking for pipeline stages."""

import json
from pathlib import Path

from .config import PipelineConfig


class ThresholdMonitor:
    """Monitors counts and triggers downstream stages when thresholds are met."""

    def __init__(self, config: PipelineConfig, watermark_path: Path | None = None):
        self.config = config
        self._watermark_path = watermark_path
        self._watermarks = self._load_watermarks()

    def _load_watermarks(self) -> dict:
        if self._watermark_path and self._watermark_path.exists():
            try:
                return json.loads(self._watermark_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _save_watermarks(self) -> None:
        if self._watermark_path:
            self._watermark_path.parent.mkdir(parents=True, exist_ok=True)
            self._watermark_path.write_text(json.dumps(self._watermarks))

    def should_generate(self, gold_dir: Path) -> bool:
        if not self.config.generate_enabled:
            return False
        count = len(list(gold_dir.glob("*.json")))
        last_triggered = self._watermarks.get("generate_at", 0)
        return count >= last_triggered + self.config.generate_gold_threshold

    def mark_generate_triggered(self, gold_dir: Path) -> None:
        count = len(list(gold_dir.glob("*.json")))
        self._watermarks["generate_at"] = count
        self._save_watermarks()

    def should_train(self, examples_file: Path) -> bool:
        if not self.config.train_enabled:
            return False
        if not examples_file.exists():
            return False
        count = sum(1 for _ in open(examples_file, encoding="utf-8"))
        last_triggered = self._watermarks.get("train_at", 0)
        return count >= last_triggered + self.config.train_examples_threshold

    def mark_train_triggered(self, examples_file: Path) -> None:
        count = sum(1 for _ in open(examples_file, encoding="utf-8"))
        self._watermarks["train_at"] = count
        self._save_watermarks()
