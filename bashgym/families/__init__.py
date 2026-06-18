"""Model-agnostic training layer: declarative per-family recipes, correctness
patches, and backend selection consumed by the trainer, exporter, and evaluator.

Supporting a new open model = adding one ``ModelFamilyProfile`` to the registry,
not editing trainer/export/eval code.

Distinct from ``bashgym.models`` (metadata + registry for *trained* artifacts).
"""

from .backends import VALID_BACKENDS, platform_probe, select_backend
from .patches import PATCHES, apply_patches
from .profiles import (
    GENERIC,
    REGISTRY,
    ModelFamilyProfile,
    resolve_family_profile,
)
from .tools import sanitize_message_tool_calls, sanitize_tool_call, validate_tool_call

__all__ = [
    "ModelFamilyProfile",
    "REGISTRY",
    "GENERIC",
    "resolve_family_profile",
    "platform_probe",
    "select_backend",
    "VALID_BACKENDS",
    "PATCHES",
    "apply_patches",
    "sanitize_tool_call",
    "sanitize_message_tool_calls",
    "validate_tool_call",
]
