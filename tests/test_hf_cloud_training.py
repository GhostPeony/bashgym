"""
Tests for HuggingFace Cloud Training (Unsloth Jobs) integration.

Tests script adapter, cloud script generation, and hardware specs.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from bashgym.integrations.huggingface.script_adapter import (
    CloudScriptConfig,
    generate_cloud_script,
    _adapt_for_cloud,
    PEP_723_HEADER,
)
from bashgym.integrations.huggingface.jobs import HARDWARE_SPECS


# =============================================================================
# CloudScriptConfig Tests
# =============================================================================

class TestCloudScriptConfig:
    """Tests for CloudScriptConfig validation."""

    def test_valid_config(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/dataset",
            output_repo="user/model",
            base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        )
        errors = config.validate()
        assert errors == []

    def test_invalid_strategy(self):
        config = CloudScriptConfig(
            strategy="invalid",
            dataset_repo="user/dataset",
            output_repo="user/model",
        )
        errors = config.validate()
        assert any("strategy" in e.lower() for e in errors)

    def test_missing_dataset_repo(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="",
            output_repo="user/model",
        )
        errors = config.validate()
        assert any("dataset_repo" in e for e in errors)

    def test_missing_output_repo(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/dataset",
            output_repo="",
        )
        errors = config.validate()
        assert any("output_repo" in e for e in errors)

    def test_invalid_epochs(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/dataset",
            output_repo="user/model",
            num_epochs=0,
        )
        errors = config.validate()
        assert any("num_epochs" in e for e in errors)

    def test_invalid_learning_rate(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/dataset",
            output_repo="user/model",
            learning_rate=-1,
        )
        errors = config.validate()
        assert any("learning_rate" in e for e in errors)

    def test_all_strategies_valid(self):
        for strategy in ("sft", "dpo", "distillation"):
            config = CloudScriptConfig(
                strategy=strategy,
                dataset_repo="user/dataset",
                output_repo="user/model",
            )
            assert config.validate() == [], f"Strategy {strategy} should be valid"

    def test_default_values(self):
        config = CloudScriptConfig()
        assert config.strategy == "sft"
        assert config.batch_size == 1
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.max_seq_length == 2048
        assert config.hardware == "a10g-small"


# =============================================================================
# generate_cloud_script Tests
# =============================================================================

class TestGenerateCloudScript:
    """Tests for generate_cloud_script function."""

    @pytest.fixture
    def sft_config(self):
        return CloudScriptConfig(
            strategy="sft",
            dataset_repo="testuser/sft-dataset",
            output_repo="testuser/sft-model",
            base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            num_epochs=2,
            learning_rate=1e-4,
        )

    @pytest.fixture
    def dpo_config(self):
        return CloudScriptConfig(
            strategy="dpo",
            dataset_repo="testuser/dpo-dataset",
            output_repo="testuser/dpo-model",
            base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            dpo_beta=0.1,
        )

    @pytest.fixture
    def distillation_config(self):
        return CloudScriptConfig(
            strategy="distillation",
            dataset_repo="testuser/distill-dataset",
            output_repo="testuser/distill-model",
            base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        )

    def test_sft_script_has_pep723_header(self, sft_config):
        script = generate_cloud_script(sft_config)
        assert "# /// script" in script
        assert 'requires-python = ">=3.10"' in script
        assert '"unsloth[cu124]"' in script
        assert '"trl>=0.15"' in script
        assert '"datasets"' in script
        assert '"torch"' in script
        assert '"trackio"' in script
        assert '"huggingface_hub"' in script
        assert "# ///" in script

    def test_sft_script_has_hf_dataset_loading(self, sft_config):
        script = generate_cloud_script(sft_config)
        assert f'load_dataset("{sft_config.dataset_repo}"' in script
        # Should NOT contain local file loading
        assert 'data_files=' not in script

    def test_sft_script_has_hub_push(self, sft_config):
        script = generate_cloud_script(sft_config)
        assert f'push_to_hub("{sft_config.output_repo}")' in script

    def test_sft_script_has_trackio(self, sft_config):
        script = generate_cloud_script(sft_config)
        assert "import trackio" in script
        assert "trackio.init()" in script

    def test_sft_script_uses_unsloth(self, sft_config):
        script = generate_cloud_script(sft_config)
        assert "from unsloth import FastLanguageModel" in script
        assert "FastLanguageModel.from_pretrained" in script

    def test_sft_script_has_correct_model(self, sft_config):
        script = generate_cloud_script(sft_config)
        assert sft_config.base_model in script

    def test_sft_script_output_dir_is_local(self, sft_config):
        script = generate_cloud_script(sft_config)
        assert 'output_dir="./output"' in script

    def test_dpo_script_has_pep723_header(self, dpo_config):
        script = generate_cloud_script(dpo_config)
        assert "# /// script" in script

    def test_dpo_script_has_dpo_imports(self, dpo_config):
        script = generate_cloud_script(dpo_config)
        assert "DPOTrainer" in script or "DPOConfig" in script

    def test_dpo_script_has_hf_dataset(self, dpo_config):
        script = generate_cloud_script(dpo_config)
        assert f'load_dataset("{dpo_config.dataset_repo}"' in script

    def test_dpo_script_has_hub_push(self, dpo_config):
        script = generate_cloud_script(dpo_config)
        assert f'push_to_hub("{dpo_config.output_repo}")' in script

    def test_distillation_script_has_pep723_header(self, distillation_config):
        script = generate_cloud_script(distillation_config)
        assert "# /// script" in script

    def test_distillation_script_has_hub_push(self, distillation_config):
        script = generate_cloud_script(distillation_config)
        assert f'push_to_hub("{distillation_config.output_repo}")' in script

    def test_distillation_script_has_hf_dataset(self, distillation_config):
        script = generate_cloud_script(distillation_config)
        assert f'load_dataset("{distillation_config.dataset_repo}"' in script

    def test_invalid_config_raises(self):
        config = CloudScriptConfig(
            strategy="bad",
            dataset_repo="",
            output_repo="",
        )
        with pytest.raises(ValueError, match="Invalid cloud script config"):
            generate_cloud_script(config)

    def test_hyperparameters_propagate(self, sft_config):
        sft_config.lora_r = 32
        sft_config.lora_alpha = 64
        sft_config.max_seq_length = 4096
        script = generate_cloud_script(sft_config)
        assert "r=32" in script
        assert "lora_alpha=64" in script
        assert "max_seq_length = 4096" in script


# =============================================================================
# _adapt_for_cloud Tests
# =============================================================================

class TestAdaptForCloud:
    """Tests for the _adapt_for_cloud transformation function."""

    def test_prepends_pep723_header(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/data",
            output_repo="user/model",
        )
        result = _adapt_for_cloud("import torch\nprint('hello')", config)
        assert result.startswith("# /// script")

    def test_replaces_local_dataset_loading(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/data",
            output_repo="user/model",
        )
        script = 'dataset = load_dataset("json", data_files="/local/path/train.jsonl", split="train")'
        result = _adapt_for_cloud(script, config)
        assert 'load_dataset("user/data", split="train")' in result
        assert "/local/path" not in result

    def test_replaces_output_dir(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/data",
            output_repo="user/model",
        )
        script = 'output_dir="/home/user/.bashgym/models/run123"'
        result = _adapt_for_cloud(script, config)
        assert 'output_dir="./output"' in result

    def test_adds_hub_push(self):
        config = CloudScriptConfig(
            strategy="sft",
            dataset_repo="user/data",
            output_repo="user/model",
        )
        script = 'print("Training complete!")'
        result = _adapt_for_cloud(script, config)
        assert 'push_to_hub("user/model")' in result


# =============================================================================
# Hardware Specs Tests
# =============================================================================

class TestHardwareSpecs:
    """Tests for HARDWARE_SPECS pricing data."""

    def test_hardware_specs_has_gpu_tiers(self):
        gpu_tiers = [k for k, v in HARDWARE_SPECS.items() if v.get("gpu")]
        assert len(gpu_tiers) >= 4

    def test_all_gpu_tiers_have_cost(self):
        for tier_id, specs in HARDWARE_SPECS.items():
            if specs.get("gpu"):
                assert "cost_per_hour" in specs, f"{tier_id} missing cost_per_hour"
                assert specs["cost_per_hour"] > 0, f"{tier_id} has zero cost"

    def test_all_gpu_tiers_have_vram(self):
        for tier_id, specs in HARDWARE_SPECS.items():
            if specs.get("gpu"):
                assert "vram_gb" in specs, f"{tier_id} missing vram_gb"
                assert specs["vram_gb"] > 0, f"{tier_id} has zero vram"

    def test_all_gpu_tiers_require_pro(self):
        for tier_id, specs in HARDWARE_SPECS.items():
            if specs.get("gpu"):
                assert specs.get("pro_required", False), f"{tier_id} should require Pro"

    def test_known_tiers_exist(self):
        assert "t4-small" in HARDWARE_SPECS
        assert "a10g-small" in HARDWARE_SPECS
        assert "a10g-large" in HARDWARE_SPECS
        assert "a100-large" in HARDWARE_SPECS
        assert "h100" in HARDWARE_SPECS

    def test_cost_ordering(self):
        """Higher-tier GPUs should cost more per hour."""
        t4_cost = HARDWARE_SPECS["t4-small"]["cost_per_hour"]
        a10g_cost = HARDWARE_SPECS["a10g-small"]["cost_per_hour"]
        a100_cost = HARDWARE_SPECS["a100-large"]["cost_per_hour"]
        h100_cost = HARDWARE_SPECS["h100"]["cost_per_hour"]
        assert t4_cost < a10g_cost < a100_cost < h100_cost
