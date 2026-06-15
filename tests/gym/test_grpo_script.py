"""Tests for GRPO training script generation — verifies critical GRPOConfig parameters."""

from pathlib import Path

from bashgym.gym.trainer import GRPOTrainer, TrainerConfig, TrainingRun, TrainingStrategy


def _generate_script(config: TrainerConfig) -> str:
    """Helper to generate a GRPO script string from config."""
    trainer = GRPOTrainer(config)
    run = TrainingRun(
        run_id="test-grpo-001",
        strategy=TrainingStrategy.GRPO,
        base_model=config.base_model,
        dataset_path=Path("data/test.jsonl"),
        output_path=Path("/tmp/test-output"),
    )
    return trainer._generate_grpo_script(run)


class TestGRPOScriptGeneration:
    def test_use_vllm_false_by_default(self):
        script = _generate_script(TrainerConfig())
        assert "use_vllm=False" in script

    def test_use_vllm_true_when_configured(self):
        script = _generate_script(TrainerConfig(grpo_use_vllm=True))
        assert "use_vllm=True" in script

    def test_max_steps_passed_through(self):
        script = _generate_script(TrainerConfig(max_steps=200))
        assert "max_steps=200" in script

    def test_max_steps_default_minus_one(self):
        script = _generate_script(TrainerConfig())
        assert "max_steps=-1" in script

    def test_temperature_passed_through(self):
        script = _generate_script(TrainerConfig(grpo_temperature=0.7))
        assert "temperature=0.7" in script

    def test_temperature_custom_value(self):
        script = _generate_script(TrainerConfig(grpo_temperature=0.9))
        assert "temperature=0.9" in script

    def test_no_vllm_import(self):
        script = _generate_script(TrainerConfig())
        assert "import vllm" not in script
        assert "from vllm" not in script

    def test_reward_mode_in_script(self):
        script = _generate_script(TrainerConfig(grpo_reward_mode="execution"))
        assert 'REWARD_MODE = "execution"' in script

    def test_cascade_config_produces_correct_script(self):
        """Simulate what cascade scheduler passes to GRPO."""
        config = TrainerConfig(
            base_model="google/gemma-4-31B-it",
            strategy=TrainingStrategy.GRPO,
            grpo_num_generations=4,
            grpo_temperature=0.7,
            grpo_reward_mode="syntax",
            max_steps=200,
            load_in_4bit=False,
        )
        script = _generate_script(config)
        assert "use_vllm=False" in script
        assert "max_steps=200" in script
        assert "temperature=0.7" in script
        assert "load_in_4bit=False" in script
