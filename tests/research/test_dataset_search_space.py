"""Unit tests for DatasetSearchSpace. No training, no network — all mocked."""
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bashgym.research.dataset_search_space import (
    DatasetCandidate,
    DatasetSearchSpace,
)


class _FakeConfig:
    """Minimal stand-in for TrainerConfig. Tests only care about attribute I/O.

    Using a plain class instead of MagicMock because MagicMock returns a new
    MagicMock for any attribute access, which breaks
    ``getattr(cfg, '_research_dataset_index', None)`` checks — it always
    returns something truthy.
    """
    def __init__(self):
        self.base_model = "test/model"
        self.max_steps = 10
        self.eval_strategy = "steps"
        self.eval_steps = 10
        self.save_steps = 999_999
        self.logging_steps = 5
        self.output_dir = "/tmp/fake"


def _base_config():
    return _FakeConfig()


def _candidates(n: int = 3) -> list[DatasetCandidate]:
    return [
        DatasetCandidate(
            repo_id=f"org/ds{i}",
            hf_score=float(9 - i),
            bashgym_format="sft",
        )
        for i in range(n)
    ]


class TestConstruction:
    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            DatasetSearchSpace(
                candidates=[],
                base_trainer_config=_base_config(),
                work_dir=Path("/tmp/x"),
            )

    def test_default_mode_is_simulate(self):
        ss = DatasetSearchSpace(
            candidates=_candidates(),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
        )
        assert ss._mode == "simulate"


class TestMutateAdvancesCursor:
    def test_mutate_returns_candidates_in_order(self):
        ss = DatasetSearchSpace(
            candidates=_candidates(3),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
        )
        c0 = ss.mutate(_base_config())
        c1 = ss.mutate(_base_config())
        c2 = ss.mutate(_base_config())
        assert c0._research_dataset_index == 0
        assert c1._research_dataset_index == 1
        assert c2._research_dataset_index == 2

    def test_mutate_past_end_clamps(self):
        ss = DatasetSearchSpace(
            candidates=_candidates(2),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
        )
        ss.mutate(_base_config())
        ss.mutate(_base_config())
        clamped = ss.mutate(_base_config())
        assert clamped._research_dataset_index == 1  # clamped to last


class TestSnapshot:
    def test_snapshot_returns_candidate_fields(self):
        ss = DatasetSearchSpace(
            candidates=_candidates(1),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
        )
        cfg = ss.mutate(_base_config())
        ss.candidates[0].eval_loss = 1.23
        snap = ss.get_config_snapshot(cfg)
        assert snap["repo_id"] == "org/ds0"
        assert snap["hf_score"] == 9.0
        assert snap["bashgym_format"] == "sft"
        assert snap["eval_loss"] == 1.23

    def test_snapshot_without_index_returns_nulls(self):
        ss = DatasetSearchSpace(
            candidates=_candidates(1),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
        )
        cfg = _base_config()
        snap = ss.get_config_snapshot(cfg)
        assert snap["repo_id"] is None
