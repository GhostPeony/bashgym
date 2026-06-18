"""Training backend selection: resolve a family profile + platform into a concrete backend.

This is where the long-standing "Unsloth vs plain-transformers" GRPO question becomes
a *switch* instead of an either/or: pick the backend from explicit config, else the
family default, else probe the platform (prefer Unsloth when importable, otherwise
plain transformers+peft — the path that works on GB10/sm_121 when Unsloth can't load).
"""

from __future__ import annotations

import platform as _platform

from .profiles import ModelFamilyProfile

VALID_BACKENDS = ("unsloth", "plain", "trl_vllm")


def platform_probe() -> dict:
    """Detect the traits that decide backend viability. All detection is best-effort."""
    machine = _platform.machine().lower()
    info: dict = {
        "machine": machine,
        "is_aarch64": machine in ("aarch64", "arm64"),
        "is_sm121": False,
        "unsloth_ok": False,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["is_sm121"] = torch.cuda.get_device_capability() == (12, 1)
    except Exception:
        pass
    try:
        import importlib.util

        # find_spec checks availability without the (heavy) side-effect of importing
        # unsloth's torch/triton stack — keeps probing fast and side-effect free.
        info["unsloth_ok"] = importlib.util.find_spec("unsloth") is not None
    except Exception:
        info["unsloth_ok"] = False
    return info


def select_backend(
    profile: ModelFamilyProfile,
    config_backend: str = "auto",
    probe: dict | None = None,
) -> str:
    """Resolve the concrete training backend.

    Priority: explicit ``config_backend`` (non-auto) > ``profile.default_backend``
    (non-auto) > auto resolution. Auto prefers Unsloth when importable, otherwise
    falls back to the plain transformers+peft path.
    """
    if config_backend and config_backend != "auto":
        return config_backend
    if profile.default_backend and profile.default_backend != "auto":
        return profile.default_backend
    p = probe if probe is not None else platform_probe()
    return "unsloth" if p.get("unsloth_ok") else "plain"
