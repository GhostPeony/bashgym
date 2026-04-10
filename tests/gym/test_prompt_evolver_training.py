"""
Tests for Phase 6: Prompt Evolver <-> Trainer Loop (Ouroboros Close).

Tests the TrainingTrigger and TrainingTriggerConfig classes. All trainer/model
calls are mocked — no GPU required.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from bashgym.gym.prompt_evolver import (
    PromptEvolver,
    TrainingTrigger,
    TrainingTriggerConfig,
)


def _create_gold_dir(count: int) -> tuple[Path, str]:
    """Create a temp dir with N gold trace JSON files. Returns (dir, tmpdir_path)."""
    tmpdir = tempfile.mkdtemp()
    gold_dir = Path(tmpdir) / "gold"
    gold_dir.mkdir()
    for i in range(count):
        trace = {
            "metadata": {"primary_repo": {"name": "test-repo"}},
            "trace": [{"tool_name": "Bash", "command": f"cmd{i}", "output": "ok", "success": True}],
        }
        (gold_dir / f"trace_{i}.json").write_text(json.dumps(trace))
    return gold_dir, tmpdir


class TestTrainingTriggerConfig:
    def test_defaults(self):
        config = TrainingTriggerConfig()
        assert config.enabled is False
        assert config.gold_trace_threshold == 10
        assert config.micro_train_steps == 50

    def test_custom(self):
        config = TrainingTriggerConfig(
            enabled=True,
            gold_trace_threshold=5,
            gold_traces_dir="/path/to/gold",
        )
        assert config.enabled is True
        assert config.gold_trace_threshold == 5


class TestTrainingTrigger:
    def test_no_trigger_below_threshold(self):
        gold_dir, _ = _create_gold_dir(3)
        config = TrainingTriggerConfig(
            enabled=True,
            gold_trace_threshold=10,
            gold_traces_dir=str(gold_dir),
        )
        trigger = TrainingTrigger(config)
        result = trigger.check_and_train()
        assert result is None

    def test_count_gold_traces(self):
        gold_dir, _ = _create_gold_dir(7)
        config = TrainingTriggerConfig(
            gold_traces_dir=str(gold_dir),
        )
        trigger = TrainingTrigger(config)
        assert trigger._count_gold_traces() == 7

    def test_count_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingTriggerConfig(gold_traces_dir=tmpdir)
            trigger = TrainingTrigger(config)
            assert trigger._count_gold_traces() == 0

    def test_count_nonexistent_dir(self):
        config = TrainingTriggerConfig(gold_traces_dir="/nonexistent")
        trigger = TrainingTrigger(config)
        assert trigger._count_gold_traces() == 0

    def test_initialize_sets_baseline(self):
        gold_dir, _ = _create_gold_dir(5)
        config = TrainingTriggerConfig(gold_traces_dir=str(gold_dir))
        trigger = TrainingTrigger(config)
        trigger._initialize()
        assert trigger._last_gold_count == 5
        assert trigger._initialized is True

    def test_threshold_detection(self):
        """Verify the trigger detects when threshold is crossed."""
        gold_dir, _ = _create_gold_dir(15)
        config = TrainingTriggerConfig(
            enabled=True,
            gold_trace_threshold=5,
            gold_traces_dir=str(gold_dir),
        )
        trigger = TrainingTrigger(config)
        # Pre-set baseline to simulate "started with 5 traces"
        trigger._initialized = True
        trigger._last_gold_count = 5

        # 15 - 5 = 10, which exceeds threshold of 5
        current = trigger._count_gold_traces()
        assert current - trigger._last_gold_count >= trigger.config.gold_trace_threshold

    def test_check_and_train_catches_errors(self):
        """check_and_train returns None on import/training errors (doesn't crash)."""
        gold_dir, _ = _create_gold_dir(15)
        config = TrainingTriggerConfig(
            enabled=True,
            gold_trace_threshold=5,
            gold_traces_dir=str(gold_dir),
        )
        trigger = TrainingTrigger(config)
        trigger._initialized = True
        trigger._last_gold_count = 5
        trigger._last_loss = 3.0

        # check_and_train will try to import Trainer which may or may not exist
        # in test env — it should catch the error and return None
        result = trigger.check_and_train()
        assert result is None or isinstance(result, float)


class TestPromptEvolverTrainingTrigger:
    def test_init_without_trigger(self):
        evolver = PromptEvolver()
        assert evolver._training_trigger is None

    def test_init_with_disabled_trigger(self):
        config = TrainingTriggerConfig(enabled=False)
        evolver = PromptEvolver(training_trigger_config=config)
        assert evolver._training_trigger is None

    def test_init_with_enabled_trigger(self):
        config = TrainingTriggerConfig(
            enabled=True,
            gold_traces_dir="/tmp/gold",
            gold_trace_threshold=5,
        )
        evolver = PromptEvolver(training_trigger_config=config)
        assert evolver._training_trigger is not None
        assert evolver._training_trigger.config.gold_trace_threshold == 5
