"""Named, model-specific correctness patches applied before model load.

Generated training scripts call ``apply_patches(profile.patches)`` at startup
inside the training subprocess, so a family's known transformers/PEFT quirks are
fixed in one declarative place instead of being copy-pasted into every generated
script. Each patch is idempotent and returns True only if it actually applied.
"""

from __future__ import annotations

from collections.abc import Callable


def _patch_gemma4_clippable_linear() -> bool:
    """Make ``Gemma4ClippableLinear`` a ``nn.Linear`` subclass so PEFT recognizes it.

    Gemma 4 ships ``Gemma4ClippableLinear`` inheriting from ``nn.Module``, which
    PEFT's LoRA injection cannot see, so adapters never attach to those layers.
    This rebinds it to an ``nn.Linear`` subclass that preserves the optional input/
    output clipping via a ``forward`` override.

    Source: https://huggingface.co/google/gemma-4-31B/discussions/3
    Returns False (no-op) when the running transformers has no gemma4 module.
    """
    try:
        import torch
        import torch.nn as nn
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return False

    class _PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features):
            nn.Linear.__init__(self, in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = nn.Linear.forward(self, x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = _PatchedClippableLinear
    return True


# name -> patch function. Functions must be idempotent and import their own deps lazily.
PATCHES: dict[str, Callable[[], bool]] = {
    "gemma4_clippable_linear": _patch_gemma4_clippable_linear,
}


def apply_patches(names) -> list[str]:
    """Apply the named patches in order; return the list that actually applied.

    Unknown names are skipped silently (a profile may name a patch whose target
    module isn't present in this environment). Safe to call with an empty list.
    """
    applied: list[str] = []
    for name in names or ():
        fn = PATCHES.get(name)
        if fn is not None and fn():
            applied.append(name)
    return applied
