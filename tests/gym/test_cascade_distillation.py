"""Tests for MOPD distillation — merging cascade domain checkpoints."""

import tempfile
from pathlib import Path

from bashgym.gym.cascade_scheduler import (
    CascadeConfig,
    CascadeResult,
    CascadeScheduler,
    MOPDConfig,
)


class TestMOPDConfig:
    def test_defaults(self):
        config = MOPDConfig()
        assert config.distillation_alpha == 0.5
        assert config.temperature == 2.0
        assert config.train_steps == 500

    def test_custom_checkpoints(self):
        config = MOPDConfig(
            domain_checkpoints={"file_operations": "/tmp/ckpt1"},
            domain_datasets={"file_operations": "/tmp/data1.jsonl"},
        )
        assert len(config.domain_checkpoints) == 1


class TestCreateMOPDConfig:
    def test_from_cascade_result(self):
        config = CascadeConfig(
            domains=["file_operations", "bash_commands"],
            output_dir=Path(tempfile.mkdtemp()),
        )
        scheduler = CascadeScheduler(config)

        # Create fake completed stages
        for stage in scheduler.stages:
            stage.status = "completed"
            stage.checkpoint_path = stage.output_path / "final"
            stage.checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Create filtered datasets
        filtered_dir = config.output_dir / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        for domain in ["file_operations", "bash_commands"]:
            (filtered_dir / f"{domain}.jsonl").write_text('{"test": true}\n')

        result = CascadeResult(
            stages=scheduler.stages,
            best_checkpoints={s.domain.name: s.checkpoint_path for s in scheduler.stages},
            total_duration_seconds=100.0,
            status="completed",
        )

        mopd_config = scheduler.create_mopd_config(result)
        assert len(mopd_config.domain_checkpoints) == 2
        assert len(mopd_config.domain_datasets) == 2
        assert "file_operations" in mopd_config.domain_checkpoints

    def test_skips_failed_stages(self):
        config = CascadeConfig(
            domains=["file_operations", "bash_commands"],
            output_dir=Path(tempfile.mkdtemp()),
        )
        scheduler = CascadeScheduler(config)

        # One completed, one failed
        scheduler.stages[0].status = "completed"
        scheduler.stages[0].checkpoint_path = Path(tempfile.mkdtemp())
        scheduler.stages[1].status = "failed"

        result = CascadeResult(
            stages=scheduler.stages,
            best_checkpoints={"file_operations": scheduler.stages[0].checkpoint_path},
            total_duration_seconds=50.0,
            status="completed",
        )

        mopd_config = scheduler.create_mopd_config(result)
        assert len(mopd_config.domain_checkpoints) == 1
        assert "bash_commands" not in mopd_config.domain_checkpoints
