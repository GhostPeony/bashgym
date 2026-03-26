"""Tests for AutoResearch -- SearchSpace ABC and HyperparamSearchSpace."""

from pathlib import Path
from unittest.mock import patch

import pytest

from bashgym.gym.autoresearch import (
    AutoResearchConfig,
    AutoResearcher,
    AutoResearchStatus,
    HyperparamSearchSpace,
    SearchSpace,
    _mutate_value,
    _simulate_loss,
)
from bashgym.gym.trainer import TrainerConfig


class TestSearchSpaceABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SearchSpace()

    def test_has_required_methods(self):
        assert hasattr(SearchSpace, "mutate")
        assert hasattr(SearchSpace, "evaluate")
        assert hasattr(SearchSpace, "get_config_snapshot")


class TestMutateValue:
    def test_bool_toggle(self):
        result = _mutate_value(True, {"type": "bool"}, 0.2)
        assert result is False

    def test_bool_toggle_false_to_true(self):
        result = _mutate_value(False, {"type": "bool"}, 0.2)
        assert result is True

    def test_float_within_bounds(self):
        spec = {"type": "float", "min": 0.0, "max": 1.0}
        for _ in range(20):
            result = _mutate_value(0.5, spec, 0.2)
            assert 0.0 <= result <= 1.0

    def test_float_log_scale(self):
        spec = {"type": "float", "min": 1e-6, "max": 1e-3, "log_scale": True}
        for _ in range(20):
            result = _mutate_value(2e-5, spec, 0.2)
            assert 1e-6 <= result <= 1e-3

    def test_int_choices(self):
        spec = {"type": "int", "min": 4, "max": 128, "choices": [4, 8, 16, 32, 64, 128]}
        for _ in range(20):
            result = _mutate_value(16, spec, 0.2)
            assert result in [4, 8, 16, 32, 64, 128]

    def test_int_choices_not_in_list(self):
        spec = {"type": "int", "min": 4, "max": 128, "choices": [4, 8, 16, 32, 64, 128]}
        result = _mutate_value(10, spec, 0.2)
        assert result in [4, 8, 16, 32, 64, 128]

    def test_int_continuous(self):
        spec = {"type": "int", "min": 8, "max": 256}
        for _ in range(20):
            result = _mutate_value(32, spec, 0.2)
            assert 8 <= result <= 256

    def test_unknown_type_returns_current(self):
        result = _mutate_value("hello", {"type": "string"}, 0.2)
        assert result == "hello"


class TestSimulateLoss:
    def test_returns_float(self):
        config = TrainerConfig()
        loss = _simulate_loss(config, 1, 50)
        assert isinstance(loss, float)

    def test_in_range(self):
        config = TrainerConfig()
        for i in range(20):
            loss = _simulate_loss(config, i, 50)
            assert 0.3 <= loss <= 5.0

    def test_extreme_lr_high_loss(self):
        config = TrainerConfig(learning_rate=1e-1)
        losses = [_simulate_loss(config, i, 50) for i in range(10)]
        avg = sum(losses) / len(losses)
        assert avg > 2.5  # Bad LR should produce higher loss

    def test_optimal_lr_lower_loss(self):
        config = TrainerConfig(learning_rate=2e-5, lora_r=32, lora_alpha=64)
        losses = [_simulate_loss(config, i, 50) for i in range(10)]
        avg = sum(losses) / len(losses)
        assert avg < 3.0


class TestHyperparamSearchSpace:
    def test_mutate_preserves_type(self):
        space = HyperparamSearchSpace(
            search_params=["learning_rate", "lora_r"],
            mutation_rate=1.0,  # Always mutate
        )
        config = TrainerConfig()
        mutated = space.mutate(config)
        assert isinstance(mutated, TrainerConfig)
        assert isinstance(mutated.learning_rate, float)
        assert isinstance(mutated.lora_r, int)

    def test_mutate_does_not_modify_original(self):
        space = HyperparamSearchSpace(
            search_params=["learning_rate"],
            mutation_rate=1.0,
        )
        config = TrainerConfig(learning_rate=2e-5)
        original_lr = config.learning_rate
        space.mutate(config)
        assert config.learning_rate == original_lr

    def test_mutate_skips_unknown_params(self):
        space = HyperparamSearchSpace(
            search_params=["nonexistent_param"],
            mutation_rate=1.0,
        )
        config = TrainerConfig()
        mutated = space.mutate(config)
        # Should still return a valid TrainerConfig without errors
        assert isinstance(mutated, TrainerConfig)

    def test_get_config_snapshot(self):
        space = HyperparamSearchSpace(
            search_params=["learning_rate", "lora_r"],
        )
        config = TrainerConfig(learning_rate=2e-5, lora_r=16)
        snapshot = space.get_config_snapshot(config)
        assert "learning_rate" in snapshot
        assert "lora_r" in snapshot
        assert snapshot["learning_rate"] == 2e-5

    def test_get_config_snapshot_excludes_non_search_params(self):
        space = HyperparamSearchSpace(
            search_params=["learning_rate"],
        )
        config = TrainerConfig(learning_rate=2e-5, lora_r=16)
        snapshot = space.get_config_snapshot(config)
        assert "learning_rate" in snapshot
        assert "lora_r" not in snapshot

    def test_evaluate_simulate_mode(self):
        space = HyperparamSearchSpace(
            search_params=["learning_rate"],
            mode="simulate",
        )
        config = TrainerConfig()
        with patch("bashgym.gym.autoresearch.time.sleep"):
            metric = space.evaluate(config, 1, 50)
        assert isinstance(metric, float)
        assert 0.3 <= metric <= 5.0


class TestAutoResearcher:
    def test_init_default_search_space(self):
        config = AutoResearchConfig()
        trainer_config = TrainerConfig()
        researcher = AutoResearcher(config, trainer_config)
        assert isinstance(researcher.search_space, HyperparamSearchSpace)

    def test_init_custom_search_space(self):
        config = AutoResearchConfig()
        trainer_config = TrainerConfig()
        custom_space = HyperparamSearchSpace(search_params=["lora_r"])
        researcher = AutoResearcher(config, trainer_config, search_space=custom_space)
        assert researcher.search_space is custom_space

    def test_mutate_config_delegates(self):
        config = AutoResearchConfig(search_params=["learning_rate"])
        trainer_config = TrainerConfig()
        researcher = AutoResearcher(config, trainer_config)
        mutated = researcher.mutate_config(trainer_config)
        assert isinstance(mutated, TrainerConfig)

    def test_status_idle_initially(self):
        config = AutoResearchConfig()
        researcher = AutoResearcher(config, TrainerConfig())
        assert researcher.status == AutoResearchStatus.IDLE

    def test_best_metric_starts_at_inf(self):
        config = AutoResearchConfig()
        researcher = AutoResearcher(config, TrainerConfig())
        assert researcher.best_metric == float("inf")

    def test_get_status_returns_dict(self):
        config = AutoResearchConfig()
        researcher = AutoResearcher(config, TrainerConfig())
        status = researcher.get_status()
        assert isinstance(status, dict)
        assert "status" in status
        assert "best_metric" in status
        assert status["status"] == AutoResearchStatus.IDLE
        assert status["best_metric"] is None  # inf is returned as None

    def test_stop_sets_running_false(self):
        config = AutoResearchConfig()
        researcher = AutoResearcher(config, TrainerConfig())
        researcher._running = True
        researcher.stop()
        assert not researcher._running

    def test_pause_sets_paused_status(self):
        config = AutoResearchConfig()
        researcher = AutoResearcher(config, TrainerConfig())
        researcher.pause()
        assert researcher.status == AutoResearchStatus.PAUSED

    def test_resume_from_paused(self):
        config = AutoResearchConfig()
        researcher = AutoResearcher(config, TrainerConfig())
        researcher.pause()
        assert researcher.status == AutoResearchStatus.PAUSED
        researcher.resume()
        assert researcher.status == AutoResearchStatus.RUNNING

    def test_resume_from_non_paused_no_change(self):
        config = AutoResearchConfig()
        researcher = AutoResearcher(config, TrainerConfig())
        researcher.status = AutoResearchStatus.IDLE
        researcher.resume()
        # Status stays idle since it wasn't paused
        assert researcher.status == AutoResearchStatus.IDLE


@pytest.mark.asyncio
class TestAutoResearcherLoop:
    async def test_run_loop_completes(self):
        config = AutoResearchConfig(max_experiments=3, mode="simulate")
        researcher = AutoResearcher(config, TrainerConfig())
        with patch("bashgym.gym.autoresearch.time.sleep"):
            best_config, experiments = await researcher.run_loop(Path("data/traces"))
        assert len(experiments) == 3
        assert researcher.status == AutoResearchStatus.COMPLETED

    async def test_run_loop_with_callback(self):
        callback_results = []

        async def callback(result, best_config, best_metric):
            callback_results.append(result)

        config = AutoResearchConfig(max_experiments=2, mode="simulate")
        researcher = AutoResearcher(config, TrainerConfig())
        with patch("bashgym.gym.autoresearch.time.sleep"):
            await researcher.run_loop(Path("data/traces"), callback=callback)
        assert len(callback_results) == 2

    async def test_run_loop_stop_early(self):
        config = AutoResearchConfig(max_experiments=100, mode="simulate")
        researcher = AutoResearcher(config, TrainerConfig())

        async def stop_after_two(result, best_config, best_metric):
            if result.experiment_id >= 2:
                researcher.stop()

        with patch("bashgym.gym.autoresearch.time.sleep"):
            await researcher.run_loop(Path("data/traces"), callback=stop_after_two)
        assert len(researcher.experiments) <= 3
        assert researcher.status == AutoResearchStatus.STOPPED

    async def test_run_loop_tracks_best_metric(self):
        config = AutoResearchConfig(max_experiments=5, mode="simulate")
        researcher = AutoResearcher(config, TrainerConfig())
        with patch("bashgym.gym.autoresearch.time.sleep"):
            await researcher.run_loop(Path("data/traces"))
        # After running, best_metric should no longer be inf
        assert researcher.best_metric < float("inf")
        assert researcher.best_metric >= 0.3

    async def test_run_loop_sets_timestamps(self):
        config = AutoResearchConfig(max_experiments=2, mode="simulate")
        researcher = AutoResearcher(config, TrainerConfig())
        with patch("bashgym.gym.autoresearch.time.sleep"):
            await researcher.run_loop(Path("data/traces"))
        assert researcher._started_at is not None
        assert researcher._completed_at is not None

    async def test_run_loop_experiment_results_have_correct_ids(self):
        config = AutoResearchConfig(max_experiments=3, mode="simulate")
        researcher = AutoResearcher(config, TrainerConfig())
        with patch("bashgym.gym.autoresearch.time.sleep"):
            await researcher.run_loop(Path("data/traces"))
        ids = [e.experiment_id for e in researcher.experiments]
        assert ids == [1, 2, 3]
