"""Tests for the named correctness-patch registry."""

from bashgym.families.patches import PATCHES, apply_patches


class TestPatches:
    def test_gemma4_patch_registered(self):
        assert "gemma4_clippable_linear" in PATCHES
        assert callable(PATCHES["gemma4_clippable_linear"])

    def test_apply_empty_is_noop(self):
        assert apply_patches([]) == []
        assert apply_patches(None) == []

    def test_unknown_patch_skipped(self):
        assert apply_patches(["does_not_exist"]) == []

    def test_apply_gemma4_does_not_raise_and_returns_subset(self):
        # Whether it applies depends on the installed transformers having a gemma4
        # module; either way it must not raise and must return a subset of inputs.
        result = apply_patches(["gemma4_clippable_linear"])
        assert set(result).issubset({"gemma4_clippable_linear"})
