"""Quality gate for hybrid auto-classification of imported traces."""

import shutil
from enum import Enum
from pathlib import Path

from .config import PipelineConfig


class Classification(str, Enum):
    GOLD = "gold"
    PENDING = "pending"
    FAILED = "failed"


class QualityGate:
    """Classifies imported traces as gold, pending, or failed."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def classify(self, success_rate: float, step_count: int) -> Classification:
        if not self.config.classify_enabled:
            return Classification.PENDING

        if success_rate <= self.config.classify_fail_max_success_rate:
            return Classification.FAILED

        if (
            success_rate >= self.config.classify_gold_min_success_rate
            and step_count >= self.config.classify_gold_min_steps
        ):
            return Classification.GOLD

        return Classification.PENDING

    def route_trace(
        self,
        trace_file: Path,
        classification: Classification,
        gold_dir: Path,
        failed_dir: Path,
    ) -> Path:
        if classification == Classification.GOLD:
            dest = gold_dir / trace_file.name
            shutil.move(str(trace_file), str(dest))
            return dest
        elif classification == Classification.FAILED:
            dest = failed_dir / trace_file.name
            shutil.move(str(trace_file), str(dest))
            return dest
        return trace_file  # pending stays in place
