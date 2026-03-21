"""Pipeline configuration with JSON persistence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Configuration for the auto-import pipeline.

    Persisted in ~/.bashgym/pipeline_config.json. Editable at runtime
    from the frontend Pipeline page.
    """

    # Stage 1: Watch
    watch_enabled: bool = True
    watch_debounce_seconds: int = 30

    # Stage 2: Classify
    classify_enabled: bool = True
    classify_gold_min_success_rate: float = 0.8
    classify_gold_min_steps: int = 10
    classify_fail_max_success_rate: float = 0.3

    # Stage 3: Generate examples
    generate_enabled: bool = False
    generate_gold_threshold: int = 10

    # Stage 4: Train
    train_enabled: bool = False
    train_examples_threshold: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError):
            return cls()
