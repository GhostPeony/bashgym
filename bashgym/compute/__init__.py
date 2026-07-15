"""Compute target planning for local, SSH, and cloud GPU training."""

from bashgym.compute.activation import normalize_training_target_payload
from bashgym.compute.targets import (
    ComputeLauncher,
    ComputeTarget,
    get_compute_target,
    launch_plan,
    list_compute_targets,
    preflight_compute_target,
)

__all__ = [
    "ComputeLauncher",
    "ComputeTarget",
    "get_compute_target",
    "launch_plan",
    "list_compute_targets",
    "normalize_training_target_payload",
    "preflight_compute_target",
]
