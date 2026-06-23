"""DPPO backend capability probes.

The TMax plan calls for DPPO as a deliberate backend, not a flag. This module
only answers whether a usable DPPO-capable stack appears to be installed.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

VALID_DPPO_BACKENDS = ("auto", "verl", "skyrl", "tmax_open_instruct", "grpo_fallback")
DPPO_BACKEND_PRIORITY = ("verl", "skyrl", "tmax_open_instruct")


@dataclass(frozen=True)
class DPPOBackendCapability:
    name: str
    available: bool
    reason: str
    command: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "name": self.name,
            "available": self.available,
            "reason": self.reason,
            "command": self.command,
            "path": self.path,
        }


@dataclass(frozen=True)
class DPPOBackendSelection:
    requested: str
    selected: str
    available: bool
    fallback_to_grpo: bool
    reason: str
    capabilities: dict[str, DPPOBackendCapability]

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "available": self.available,
            "fallback_to_grpo": self.fallback_to_grpo,
            "reason": self.reason,
            "capabilities": {
                name: capability.to_dict() for name, capability in self.capabilities.items()
            },
        }


def probe_dppo_backends(
    *,
    env: Mapping[str, str] | None = None,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    which: Callable[[str], str | None] = shutil.which,
    path_exists: Callable[[str], bool] | None = None,
) -> dict[str, DPPOBackendCapability]:
    """Probe for known DPPO-capable external stacks without importing them."""

    env_map = env or os.environ
    exists = path_exists or (lambda path: Path(path).exists())

    verl_command = which("verl")
    verl_home = env_map.get("VERL_HOME", "")
    verl_available = bool(find_spec("verl") or verl_command or (verl_home and exists(verl_home)))
    skyrl_command = which("skyrl")
    skyrl_home = env_map.get("SKYRL_HOME", "")
    skyrl_available = bool(
        find_spec("skyrl") or skyrl_command or (skyrl_home and exists(skyrl_home))
    )
    tmax_home = env_map.get("TMAX_OPEN_INSTRUCT_DIR") or env_map.get("OPEN_INSTRUCT_ROOT", "")
    tmax_available = bool(tmax_home and exists(tmax_home))

    return {
        "verl": DPPOBackendCapability(
            name="verl",
            available=verl_available,
            reason="verl module/CLI/home detected" if verl_available else "verl not detected",
            command=verl_command,
            path=verl_home or None,
        ),
        "skyrl": DPPOBackendCapability(
            name="skyrl",
            available=skyrl_available,
            reason="SkyRL module/CLI/home detected" if skyrl_available else "SkyRL not detected",
            command=skyrl_command,
            path=skyrl_home or None,
        ),
        "tmax_open_instruct": DPPOBackendCapability(
            name="tmax_open_instruct",
            available=tmax_available,
            reason=(
                "TMax/open-instruct checkout detected"
                if tmax_available
                else "TMax/open-instruct checkout not detected"
            ),
            path=tmax_home or None,
        ),
    }


def select_dppo_backend(
    requested: str = "auto",
    *,
    capabilities: dict[str, DPPOBackendCapability] | None = None,
) -> DPPOBackendSelection:
    """Resolve the requested DPPO backend with GRPO as the explicit fallback."""

    normalized = (requested or "auto").strip().lower()
    if normalized not in VALID_DPPO_BACKENDS:
        raise ValueError(f"dppo_backend={requested!r} must be one of {list(VALID_DPPO_BACKENDS)}")
    caps = capabilities if capabilities is not None else probe_dppo_backends()

    if normalized == "grpo_fallback":
        return DPPOBackendSelection(
            requested=normalized,
            selected="grpo_fallback",
            available=False,
            fallback_to_grpo=True,
            reason="GRPO fallback explicitly requested",
            capabilities=caps,
        )

    if normalized == "auto":
        for candidate in DPPO_BACKEND_PRIORITY:
            capability = caps[candidate]
            if capability.available:
                return DPPOBackendSelection(
                    requested=normalized,
                    selected=candidate,
                    available=True,
                    fallback_to_grpo=False,
                    reason=capability.reason,
                    capabilities=caps,
                )
        return DPPOBackendSelection(
            requested=normalized,
            selected="grpo_fallback",
            available=False,
            fallback_to_grpo=True,
            reason="No DPPO-capable backend detected; use GRPO fallback",
            capabilities=caps,
        )

    capability = caps[normalized]
    return DPPOBackendSelection(
        requested=normalized,
        selected=normalized if capability.available else "grpo_fallback",
        available=capability.available,
        fallback_to_grpo=not capability.available,
        reason=(
            capability.reason
            if capability.available
            else f"Requested {normalized} but it is unavailable; use GRPO fallback"
        ),
        capabilities=caps,
    )
