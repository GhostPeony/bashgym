"""Tests for GRPO training script generation — verifies critical GRPOConfig parameters."""

from pathlib import Path

import pytest

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

    def test_target_modules_come_from_family_profile(self):
        """The generator consumes the ModelFamilyProfile rather than a hardcoded list."""
        from bashgym.families import resolve_family_profile

        config = TrainerConfig(base_model="google/gemma-4-31B-it")
        script = _generate_script(config)
        profile = resolve_family_profile(config.base_model)
        assert profile.lora_target_modules  # sanity
        for mod in profile.lora_target_modules:
            assert f"'{mod}'" in script, f"{mod} missing from generated GRPO script"


class TestGRPOBackendDispatch:
    def test_plain_backend_generates_plain_transformers_script(self):
        config = TrainerConfig(grpo_backend="plain", base_model="google/gemma-4-31B-it")
        script = _generate_script(config)
        assert "AutoModelForCausalLM" in script
        assert "from unsloth" not in script
        # Gemma 4 patch + multimodal excludes flow from the profile.
        assert "apply_patches(['gemma4_clippable_linear'])" in script
        assert "exclude_modules=['vision_tower', 'multi_modal_projector', 'audio_tower']" in script

    def test_plain_backend_no_gemma_patch_for_qwen(self):
        config = TrainerConfig(grpo_backend="plain", base_model="Qwen/Qwen3.6-35B-A3B")
        script = _generate_script(config)
        assert "apply_patches([])" in script
        assert "exclude_modules=[]" in script

    def test_unsloth_backend_generates_unsloth_script(self):
        config = TrainerConfig(grpo_backend="unsloth", base_model="google/gemma-4-31B-it")
        script = _generate_script(config)
        assert "FastLanguageModel" in script

    def test_both_backends_emit_valid_python(self):
        import ast

        for backend in ("plain", "unsloth"):
            config = TrainerConfig(grpo_backend=backend, base_model="google/gemma-4-31B-it")
            ast.parse(_generate_script(config))  # raises SyntaxError if escaping is wrong


class TestGRPOLossType:
    """GSPO / Dr. GRPO variant selection via GRPOConfig.loss_type."""

    def test_default_is_grpo(self):
        for backend in ("plain", "unsloth"):
            script = _generate_script(TrainerConfig(grpo_backend=backend))
            assert 'loss_type="grpo"' in script

    def test_gspo_threads_into_both_backends(self):
        for backend in ("plain", "unsloth"):
            script = _generate_script(TrainerConfig(grpo_backend=backend, grpo_loss_type="gspo"))
            assert 'loss_type="gspo"' in script

    def test_dr_grpo_variant(self):
        script = _generate_script(TrainerConfig(grpo_backend="plain", grpo_loss_type="dr_grpo"))
        assert 'loss_type="dr_grpo"' in script

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="grpo_loss_type"):
            _generate_script(TrainerConfig(grpo_loss_type="not_a_real_loss"))

    def test_gspo_script_still_valid_python(self):
        import ast

        for backend in ("plain", "unsloth"):
            script = _generate_script(TrainerConfig(grpo_backend=backend, grpo_loss_type="gspo"))
            ast.parse(script)
