"""Tests for Phase 5: provider presets, per-column routing, diversity samplers."""

import pytest

from bashgym.factory.data_designer import DATA_DESIGNER_AVAILABLE, PipelineConfig
from bashgym.factory.designer_pipelines import (
    _provider_name_for,
    nim_provider_spec,
    ollama_provider_spec,
)


class TestProviderPresets:
    def test_ollama_spec_normalizes_endpoint(self):
        spec = ollama_provider_spec(["text-model"], endpoint="http://192.168.50.173:11434")
        assert spec.endpoint == "http://192.168.50.173:11434/v1"
        assert spec.api_key == "ollama"  # non-empty, ignored by Ollama
        assert spec.models == ["text-model"]
        assert spec.name == "ollama"

    def test_ollama_spec_keeps_existing_v1(self):
        spec = ollama_provider_spec(["m"], endpoint="http://host:11434/v1")
        assert spec.endpoint == "http://host:11434/v1"

    def test_nim_spec(self):
        spec = nim_provider_spec(["judge-model"])
        assert spec.name == "nvidia"
        assert spec.endpoint == "https://integrate.api.nvidia.com/v1"
        assert spec.models == ["judge-model"]


class TestPerColumnRouting:
    def test_mixed_provider_routing(self):
        cfg = PipelineConfig(
            providers=[
                ollama_provider_spec(["text-model", "code-model"], endpoint="http://x:11434"),
                nim_provider_spec(["judge-model"]),
            ]
        )
        assert _provider_name_for("text-model", cfg) == "ollama"
        assert _provider_name_for("code-model", cfg) == "ollama"
        assert _provider_name_for("judge-model", cfg) == "nvidia"

    def test_unrouted_alias_falls_back_to_first(self):
        cfg = PipelineConfig(
            providers=[ollama_provider_spec(["text-model"]), nim_provider_spec(["judge-model"])]
        )
        # An alias declared by no provider falls back to the first.
        assert _provider_name_for("unknown-alias", cfg) == "ollama"

    @pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
    def test_build_model_providers_mixed(self):
        from bashgym.factory.designer_pipelines import build_model_providers

        cfg = PipelineConfig(
            providers=[
                ollama_provider_spec(["text-model"], endpoint="http://x:11434"),
                nim_provider_spec(["judge-model"]),
            ]
        )
        provs = build_model_providers(cfg)
        names = {p.name for p in provs}
        assert names == {"ollama", "nvidia"}
        ollama = next(p for p in provs if p.name == "ollama")
        assert ollama.api_key == "ollama"  # resolves via plaintext fallback


@pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
class TestDiversitySamplers:
    def test_subcategory_sampler(self):
        import data_designer.config as dd

        from bashgym.factory.designer_pipelines import subcategory_sampler

        col = subcategory_sampler(
            "framework", "language", {"python": ["fastapi"], "javascript": ["react"]}
        )
        assert col.name == "framework"
        assert col.sampler_type == dd.SamplerType.SUBCATEGORY

    def test_persona_sampler(self):
        import data_designer.config as dd

        from bashgym.factory.designer_pipelines import persona_sampler

        col = persona_sampler(with_synthetic_personas=True)
        assert col.name == "persona"
        assert col.sampler_type == dd.SamplerType.PERSON
