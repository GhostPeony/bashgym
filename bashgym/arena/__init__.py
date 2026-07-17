"""Sandboxed execution APIs with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "AgentConfig": "bashgym.arena.runner",
    "AgentRunner": "bashgym.arena.runner",
    "TaskResult": "bashgym.arena.runner",
    "SandboxConfig": "bashgym.arena.sandbox",
    "SandboxInstance": "bashgym.arena.sandbox",
    "SandboxManager": "bashgym.arena.sandbox",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
