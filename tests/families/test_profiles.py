"""Tests for the model-agnostic family profile registry."""

import pytest

from bashgym.families.profiles import (
    GENERIC,
    REGISTRY,
    ModelFamilyProfile,
    resolve_family_profile,
)


class TestResolveFamilyProfile:
    @pytest.mark.parametrize(
        "model_id,family",
        [
            ("google/gemma-4-31B-it", "gemma4"),
            ("unsloth/gemma-4-E4B-it", "gemma4"),
            ("google/gemma-4-26B-A4B", "gemma4"),
            ("Qwen/Qwen3.6-35B-A3B", "qwen3"),
            ("Qwen/Qwen3-8B", "qwen3"),
            ("Qwen/Qwen2.5-Coder-7B-Instruct", "qwen2.5"),
            ("meta-llama/Llama-3.2-3B-Instruct", "llama3"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "generic"),
            ("some/unknown-model-xyz", "generic"),
        ],
    )
    def test_family_resolution(self, model_id, family):
        assert resolve_family_profile(model_id).family == family

    def test_case_insensitive(self):
        assert resolve_family_profile("GOOGLE/GEMMA-4-31B").family == "gemma4"

    def test_qwen25_not_misresolved_as_qwen3(self):
        # 'qwen2.5' must not match the qwen3 profile (no 'qwen3' substring).
        assert resolve_family_profile("Qwen/Qwen2.5-Coder-32B").family == "qwen2.5"

    def test_gemma4_has_clippable_patch_and_vision_excludes(self):
        p = resolve_family_profile("google/gemma-4-31B")
        assert "gemma4_clippable_linear" in p.patches
        assert "vision_tower" in p.lora_exclude_modules
        assert p.tool_call_format == "gemma4_delimited"

    def test_qwen3_no_gemma_patch(self):
        p = resolve_family_profile("Qwen/Qwen3.6-35B-A3B")
        assert "gemma4_clippable_linear" not in p.patches
        assert p.tool_call_format == "qwen_xml"
        assert p.lora_exclude_modules == ()

    def test_thinking_flags(self):
        assert resolve_family_profile("google/gemma-4-26B-it").thinking is True
        assert resolve_family_profile("Qwen/Qwen3-8B").thinking is True
        assert resolve_family_profile("meta-llama/Llama-3.2-3B").thinking is False


class TestRegistryInvariants:
    def test_all_profiles_have_lora_targets(self):
        for p in REGISTRY:
            assert p.lora_target_modules, f"{p.family} has empty lora_target_modules"
            assert isinstance(p.lora_target_modules, tuple)

    def test_generic_is_last_and_is_the_fallback(self):
        assert REGISTRY[-1] is GENERIC
        assert resolve_family_profile("literally/anything-1234") is GENERIC

    def test_profiles_are_frozen(self):
        p = resolve_family_profile("google/gemma-4-31B")
        with pytest.raises(Exception):
            p.family = "mutated"  # frozen dataclass

    def test_default_backend_is_valid_token(self):
        for p in REGISTRY:
            assert p.default_backend in ("auto", "unsloth", "plain", "trl_vllm")

    def test_match_method(self):
        prof = ModelFamilyProfile(family="x", match=("foo", "bar"))
        assert prof.matches("a/FOO-7b") is True
        assert prof.matches("a/baz") is False
