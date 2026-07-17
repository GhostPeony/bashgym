"""BashGym public package surface.

The public convenience imports are lazy so lightweight commands such as
``bashgym --help`` do not import training, Data Designer, Docker, or API stacks.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"

_EXPORTS = {
    "Settings": ("bashgym.config", "Settings"),
    "get_settings": ("bashgym.config", "get_settings"),
    "SandboxManager": ("bashgym.arena", "SandboxManager"),
    "AgentRunner": ("bashgym.arena", "AgentRunner"),
    "DataFactory": ("bashgym.factory", "DataFactory"),
    "TraceProcessor": ("bashgym.factory", "TraceProcessor"),
    "Trainer": ("bashgym.gym", "Trainer"),
    "BashGymEnv": ("bashgym.gym", "BashGymEnv"),
    "ModelRouter": ("bashgym.gym", "ModelRouter"),
    "Verifier": ("bashgym.judge", "Verifier"),
}


def __getattr__(name: str) -> Any:
    """Resolve compatibility exports only when callers actually use them."""

    if name == "integrations":
        value = importlib.import_module("bashgym.integrations")
    elif name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, "integrations"})


__all__ = [*_EXPORTS, "integrations"]
