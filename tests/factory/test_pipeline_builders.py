"""Tests for shared pipeline builder infrastructure."""

from unittest.mock import MagicMock, patch

import pytest

from bashgym.factory.data_designer import PipelineConfig, ProviderSpec

# =========================================================================
# _env_key_for_provider
# =========================================================================


class TestEnvKeyForProvider:
    def test_nvidia_provider(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("nvidia") == "NVIDIA_API_KEY"

    def test_nvidia_nim_provider(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("nvidia-nim") == "NVIDIA_API_KEY"

    def test_anthropic_provider(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("anthropic") == "ANTHROPIC_API_KEY"

    def test_openai_provider(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("openai") == "OPENAI_API_KEY"

    def test_openrouter_provider(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("openrouter") == "OPENROUTER_API_KEY"

    def test_local_provider(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("local") == "LOCAL_API_KEY"

    def test_unknown_provider_defaults_nvidia(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("unknown_provider") == "NVIDIA_API_KEY"

    def test_empty_string_defaults_nvidia(self):
        from bashgym.factory.designer_pipelines import _env_key_for_provider

        assert _env_key_for_provider("") == "NVIDIA_API_KEY"


# =========================================================================
# build_base_config
# =========================================================================


class TestBuildBaseConfig:
    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", False)
    def test_raises_when_dd_unavailable(self):
        from bashgym.factory.designer_pipelines import build_base_config

        config = PipelineConfig()
        with pytest.raises(ImportError, match="data-designer"):
            build_base_config(config)

    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", True)
    @patch("bashgym.factory.designer_pipelines.dd", create=True)
    def test_single_provider(self, mock_dd):
        from bashgym.factory.designer_pipelines import build_base_config

        mock_builder_instance = MagicMock()
        mock_dd.DataDesignerConfigBuilder.return_value = mock_builder_instance
        mock_dd.ModelConfig.return_value = MagicMock()
        mock_dd.ModelProvider.return_value = MagicMock()
        mock_dd.InferenceParameters.return_value = MagicMock()

        config = PipelineConfig(
            provider="nvidia",
            provider_endpoint="https://nim.example.com",
        )
        build_base_config(config)

        # Should create 1 provider
        assert mock_dd.ModelProvider.call_count == 1
        provider_call = mock_dd.ModelProvider.call_args
        assert provider_call[1]["name"] == "nvidia"
        assert provider_call[1]["endpoint"] == "https://nim.example.com"

        # Should create 3 model configs (text, code, judge)
        assert mock_dd.ModelConfig.call_count == 3

        # Verify model aliases
        model_aliases = [c[1]["alias"] for c in mock_dd.ModelConfig.call_args_list]
        assert "text-model" in model_aliases
        assert "code-model" in model_aliases
        assert "judge-model" in model_aliases

    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", True)
    @patch("bashgym.factory.designer_pipelines.dd", create=True)
    def test_multi_provider(self, mock_dd):
        from bashgym.factory.designer_pipelines import build_base_config

        mock_builder_instance = MagicMock()
        mock_dd.DataDesignerConfigBuilder.return_value = mock_builder_instance
        mock_dd.ModelConfig.return_value = MagicMock()
        mock_dd.ModelProvider.return_value = MagicMock()
        mock_dd.InferenceParameters.return_value = MagicMock()

        providers = [
            ProviderSpec(name="nvidia", endpoint="https://nim.example.com"),
            ProviderSpec(name="anthropic", endpoint="https://api.anthropic.com"),
        ]
        config = PipelineConfig(providers=providers)
        build_base_config(config)

        # Should create 2 providers
        assert mock_dd.ModelProvider.call_count == 2

        provider_names = [c[1]["name"] for c in mock_dd.ModelProvider.call_args_list]
        assert "nvidia" in provider_names
        assert "anthropic" in provider_names

    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", True)
    @patch("bashgym.factory.designer_pipelines.dd", create=True)
    def test_multi_provider_with_explicit_api_key(self, mock_dd):
        from bashgym.factory.designer_pipelines import build_base_config

        mock_dd.DataDesignerConfigBuilder.return_value = MagicMock()
        mock_dd.ModelConfig.return_value = MagicMock()
        mock_dd.ModelProvider.return_value = MagicMock()
        mock_dd.InferenceParameters.return_value = MagicMock()

        providers = [
            ProviderSpec(
                name="nvidia",
                endpoint="https://nim.example.com",
                api_key="explicit-nvidia-key",
            ),
        ]
        config = PipelineConfig(providers=providers)
        build_base_config(config)

        provider_call = mock_dd.ModelProvider.call_args
        assert provider_call[1]["api_key"] == "explicit-nvidia-key"

    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", True)
    @patch("bashgym.factory.designer_pipelines.dd", create=True)
    def test_multi_provider_falls_back_to_env_key(self, mock_dd):
        from bashgym.factory.designer_pipelines import build_base_config

        mock_dd.DataDesignerConfigBuilder.return_value = MagicMock()
        mock_dd.ModelConfig.return_value = MagicMock()
        mock_dd.ModelProvider.return_value = MagicMock()
        mock_dd.InferenceParameters.return_value = MagicMock()

        providers = [
            ProviderSpec(
                name="anthropic",
                endpoint="https://api.anthropic.com",
                # api_key is None
            ),
        ]
        config = PipelineConfig(providers=providers)
        build_base_config(config)

        provider_call = mock_dd.ModelProvider.call_args
        assert provider_call[1]["api_key"] == "${ANTHROPIC_API_KEY}"

    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", True)
    @patch("bashgym.factory.designer_pipelines.dd", create=True)
    def test_temperature_settings_applied(self, mock_dd):
        from bashgym.factory.designer_pipelines import build_base_config

        mock_dd.DataDesignerConfigBuilder.return_value = MagicMock()
        mock_dd.ModelConfig.return_value = MagicMock()
        mock_dd.ModelProvider.return_value = MagicMock()
        mock_dd.InferenceParameters.return_value = MagicMock()

        config = PipelineConfig(
            temperature_text=0.9,
            temperature_code=0.3,
            temperature_judge=0.05,
        )
        build_base_config(config)

        # InferenceParameters should be called 3 times with the configured temps
        assert mock_dd.InferenceParameters.call_count == 3
        temp_calls = mock_dd.InferenceParameters.call_args_list

        temps_used = [c[1]["temperature"] for c in temp_calls]
        assert 0.9 in temps_used  # text
        assert 0.3 in temps_used  # code
        assert 0.05 in temps_used  # judge

    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", True)
    @patch("bashgym.factory.designer_pipelines.dd", create=True)
    def test_custom_model_names(self, mock_dd):
        from bashgym.factory.designer_pipelines import build_base_config

        mock_dd.DataDesignerConfigBuilder.return_value = MagicMock()
        mock_dd.ModelConfig.return_value = MagicMock()
        mock_dd.ModelProvider.return_value = MagicMock()
        mock_dd.InferenceParameters.return_value = MagicMock()

        config = PipelineConfig(
            text_model="custom/text",
            code_model="custom/code",
            judge_model="custom/judge",
        )
        build_base_config(config)

        model_names = [c[1]["model"] for c in mock_dd.ModelConfig.call_args_list]
        assert "custom/text" in model_names
        assert "custom/code" in model_names
        assert "custom/judge" in model_names

    @patch("bashgym.factory.designer_pipelines.DATA_DESIGNER_AVAILABLE", True)
    @patch("bashgym.factory.designer_pipelines.dd", create=True)
    def test_returns_config_builder(self, mock_dd):
        from bashgym.factory.designer_pipelines import build_base_config

        mock_builder_instance = MagicMock()
        mock_dd.DataDesignerConfigBuilder.return_value = mock_builder_instance
        mock_dd.ModelConfig.return_value = MagicMock()
        mock_dd.ModelProvider.return_value = MagicMock()
        mock_dd.InferenceParameters.return_value = MagicMock()

        config = PipelineConfig()
        result = build_base_config(config)

        assert result == mock_builder_instance


# =========================================================================
# Pipeline Registry
# =========================================================================


class TestPipelineRegistry:
    def test_pipelines_dict_exists(self):
        from bashgym.factory.designer_pipelines import PIPELINES

        assert isinstance(PIPELINES, dict)

    def test_pipelines_values_are_callable(self):
        """All registered pipelines should be callable functions."""
        from bashgym.factory.designer_pipelines import PIPELINES

        for name, builder_fn in PIPELINES.items():
            assert callable(builder_fn), f"Pipeline '{name}' is not callable"


# =========================================================================
# register_pipeline decorator
# =========================================================================


class TestRegisterPipeline:
    def test_register_pipeline_decorator(self):
        from bashgym.factory.designer_pipelines import (
            PIPELINES,
            register_pipeline,
        )

        @register_pipeline("test_registered_pipeline")
        def my_test_builder(config):
            return MagicMock()

        assert "test_registered_pipeline" in PIPELINES
        assert PIPELINES["test_registered_pipeline"] is my_test_builder

        # Cleanup
        del PIPELINES["test_registered_pipeline"]

    def test_register_pipeline_preserves_function(self):
        from bashgym.factory.designer_pipelines import (
            PIPELINES,
            register_pipeline,
        )

        @register_pipeline("test_preserved_fn")
        def original_function(config):
            return "original"

        assert original_function(None) == "original"

        # Cleanup
        del PIPELINES["test_preserved_fn"]
