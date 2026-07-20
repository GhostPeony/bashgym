"""Unit tests for DatasetSearchSpace. No training, no network — all mocked."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestEvaluateSimulate:
    def test_simulate_returns_inverse_hf_score(self):
        """Deterministic: simulated loss = 10.0 - hf_score."""
        ss = DatasetSearchSpace(
            candidates=_candidates(3),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
            mode="simulate",
        )
        cfg = ss.mutate(_base_config())  # picks candidate 0 (hf_score=9.0)
        metric = ss.evaluate(cfg, 1, 3)
        assert metric == pytest.approx(1.0, abs=0.01)

    def test_simulate_records_metric_on_candidate(self):
        ss = DatasetSearchSpace(
            candidates=_candidates(3),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
            mode="simulate",
        )
        cfg = ss.mutate(_base_config())
        ss.evaluate(cfg, 1, 3)
        assert ss.candidates[0].eval_loss is not None

    def test_simulate_is_fast(self):
        """Simulate mode should take <1s total for 3 candidates."""
        ss = DatasetSearchSpace(
            candidates=_candidates(3),
            base_trainer_config=_base_config(),
            work_dir=Path("/tmp/x"),
            mode="simulate",
        )
        start = time.time()
        for _ in range(3):
            cfg = ss.mutate(_base_config())
            ss.evaluate(cfg, 1, 3)
        assert time.time() - start < 1.0


class TestEvaluateReal:
    @patch("bashgym.research.dataset_search_space._get_trainer")
    @patch("bashgym.research.dataset_search_space._get_data_designer")
    def test_real_materializes_and_trains(self, mock_get_dd, mock_get_trainer, tmp_path):
        # Mock DataDesignerPipeline
        mock_pipeline = MagicMock()
        mock_df = MagicMock()
        mock_pipeline.from_dataset.return_value = mock_df
        mock_pipeline.export_nemo.return_value = {
            "train_path": str(tmp_path / "train.jsonl"),
            "val_path": str(tmp_path / "val.jsonl"),
            "train_count": 450,
            "val_count": 50,
        }
        mock_get_dd.return_value = MagicMock(return_value=mock_pipeline)

        # Mock Trainer
        mock_run = MagicMock()
        mock_run.metrics = {"eval_loss": 1.42, "final_loss": 1.55}
        mock_trainer = MagicMock()
        mock_trainer.train_sft.return_value = mock_run
        mock_get_trainer.return_value = MagicMock(return_value=mock_trainer)

        ss = DatasetSearchSpace(
            candidates=_candidates(1),
            base_trainer_config=_base_config(),
            work_dir=tmp_path,
            mode="real",
            num_records=500,
            train_steps=50,
        )
        cfg = ss.mutate(_base_config())
        metric = ss.evaluate(cfg, 1, 1)

        assert metric == pytest.approx(1.42)
        assert ss.candidates[0].eval_loss == 1.42
        assert ss.candidates[0].final_loss == 1.55
        assert ss.candidates[0].num_rows_generated == 500
        mock_pipeline.from_dataset.assert_called_once()

    @patch("bashgym.research.dataset_search_space._get_data_designer")
    def test_real_handles_materialize_failure(self, mock_get_dd, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline.from_dataset.side_effect = RuntimeError("network boom")
        mock_get_dd.return_value = MagicMock(return_value=mock_pipeline)

        ss = DatasetSearchSpace(
            candidates=_candidates(1),
            base_trainer_config=_base_config(),
            work_dir=tmp_path,
            mode="real",
        )
        cfg = ss.mutate(_base_config())
        metric = ss.evaluate(cfg, 1, 1)

        assert metric == 5.0
        assert "network boom" in (ss.candidates[0].error or "")

    @patch("bashgym.research.dataset_search_space._get_trainer")
    @patch("bashgym.research.dataset_search_space._get_data_designer")
    def test_real_handles_training_failure(self, mock_get_dd, mock_get_trainer, tmp_path):
        mock_pipeline = MagicMock()
        mock_pipeline.from_dataset.return_value = MagicMock()
        mock_pipeline.export_nemo.return_value = {
            "train_path": str(tmp_path / "train.jsonl"),
            "val_path": str(tmp_path / "val.jsonl"),
            "train_count": 10,
            "val_count": 5,
        }
        mock_get_dd.return_value = MagicMock(return_value=mock_pipeline)

        mock_trainer = MagicMock()
        mock_trainer.train_sft.side_effect = RuntimeError("OOM")
        mock_get_trainer.return_value = MagicMock(return_value=mock_trainer)

        ss = DatasetSearchSpace(
            candidates=_candidates(1),
            base_trainer_config=_base_config(),
            work_dir=tmp_path,
            mode="real",
        )
        cfg = ss.mutate(_base_config())
        metric = ss.evaluate(cfg, 1, 1)

        assert metric == 5.0
        assert "OOM" in (ss.candidates[0].error or "")
