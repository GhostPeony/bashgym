"""Tests for backend selection (Unsloth vs plain-transformers as a switch)."""

from bashgym.families.backends import platform_probe, select_backend
from bashgym.families.profiles import GEMMA4, ModelFamilyProfile


class TestSelectBackend:
    def test_explicit_config_wins(self):
        # Explicit config_backend overrides everything, even the probe.
        assert select_backend(GEMMA4, "plain", probe={"unsloth_ok": True}) == "plain"
        assert select_backend(GEMMA4, "trl_vllm", probe={"unsloth_ok": True}) == "trl_vllm"

    def test_profile_default_used_when_config_auto(self):
        prof = ModelFamilyProfile(family="x", match=("x",), default_backend="unsloth")
        assert select_backend(prof, "auto", probe={"unsloth_ok": False}) == "unsloth"

    def test_auto_prefers_unsloth_when_available(self):
        assert select_backend(GEMMA4, "auto", probe={"unsloth_ok": True}) == "unsloth"

    def test_auto_falls_back_to_plain_without_unsloth(self):
        # The GB10/sm_121 reality: Unsloth not importable -> plain transformers+peft.
        assert select_backend(GEMMA4, "auto", probe={"unsloth_ok": False}) == "plain"


class TestPlatformProbe:
    def test_probe_shape(self):
        p = platform_probe()
        for key in ("machine", "is_aarch64", "is_sm121", "unsloth_ok"):
            assert key in p
        assert isinstance(p["is_aarch64"], bool)
        assert isinstance(p["unsloth_ok"], bool)
