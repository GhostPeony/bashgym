"""Tests for SFT/DPO backend dispatch (Unsloth vs plain transformers) + Liger wiring.

Mirrors test_grpo_script.py: generate the script string and assert on markers;
``ast.parse`` guards the f-string escaping in the plain generators.
"""

import ast
from pathlib import Path

from bashgym.gym.trainer import Trainer, TrainerConfig, TrainingRun, TrainingStrategy


def _sft(config: TrainerConfig) -> str:
    run = TrainingRun(
        run_id="t-sft",
        strategy=TrainingStrategy.SFT,
        base_model=config.base_model,
        dataset_path=Path("data/test.jsonl"),
        output_path=Path("/tmp/out"),
    )
    return Trainer(config)._generate_sft_script(run)


def _dpo(config: TrainerConfig) -> str:
    run = TrainingRun(
        run_id="t-dpo",
        strategy=TrainingStrategy.DPO,
        base_model=config.base_model,
        dataset_path=Path("data/test.jsonl"),
        output_path=Path("/tmp/out"),
    )
    return Trainer(config)._generate_dpo_script(run)


class TestSFTDispatch:
    def test_plain_backend(self):
        s = _sft(TrainerConfig(sft_backend="plain", base_model="google/gemma-4-31B-it"))
        assert "AutoModelForCausalLM" in s
        assert "from unsloth" not in s
        # family-correct patch + multimodal excludes flow from the profile
        assert "apply_patches(['gemma4_clippable_linear'])" in s
        assert "exclude_modules=['vision_tower', 'multi_modal_projector', 'audio_tower']" in s

    def test_unsloth_backend(self):
        s = _sft(TrainerConfig(sft_backend="unsloth", base_model="google/gemma-4-31B-it"))
        assert "FastLanguageModel" in s

    def test_both_emit_valid_python(self):
        for backend in ("plain", "unsloth"):
            ast.parse(_sft(TrainerConfig(sft_backend=backend, base_model="google/gemma-4-31B-it")))


class TestDPODispatch:
    def test_plain_backend(self):
        s = _dpo(TrainerConfig(dpo_backend="plain", base_model="google/gemma-4-31B-it"))
        assert "DPOTrainer" in s and "AutoModelForCausalLM" in s
        assert "ref_model=None" in s  # implicit-reference DPO
        assert "from unsloth" not in s

    def test_unsloth_backend(self):
        s = _dpo(TrainerConfig(dpo_backend="unsloth", base_model="google/gemma-4-31B-it"))
        assert "FastLanguageModel" in s

    def test_both_emit_valid_python(self):
        for backend in ("plain", "unsloth"):
            ast.parse(_dpo(TrainerConfig(dpo_backend=backend, base_model="google/gemma-4-31B-it")))


class TestLigerWiring:
    def test_liger_off_by_default(self):
        assert "use_liger_kernel=False" in _sft(TrainerConfig(sft_backend="plain"))
        assert "use_liger_kernel=False" in _dpo(TrainerConfig(dpo_backend="plain"))

    def test_liger_on_when_enabled(self):
        assert "use_liger_kernel=True" in _sft(TrainerConfig(sft_backend="plain", use_liger=True))
        assert "use_liger_kernel=True" in _dpo(TrainerConfig(dpo_backend="plain", use_liger=True))

    def test_liger_not_in_unsloth_path(self):
        # Liger is a plain-backend (transformers-native) concern; Unsloth has its own CE.
        assert "use_liger_kernel" not in _sft(TrainerConfig(sft_backend="unsloth"))

    def test_plain_sft_uses_family_targets(self):
        from bashgym.families import resolve_family_profile

        config = TrainerConfig(sft_backend="plain", base_model="google/gemma-4-31B-it")
        s = _sft(config)
        profile = resolve_family_profile(config.base_model)
        assert profile.lora_target_modules  # sanity
        for mod in profile.lora_target_modules:
            assert f"'{mod}'" in s, f"{mod} missing from plain SFT script"
