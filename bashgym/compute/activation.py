"""Normalize an agent-selected compute target into executable training flags."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_training_target_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a request payload whose target label matches its execution backend.

    ``compute_target`` used to be provenance-only, which let an agent submit
    ``ssh:<device>`` while accidentally leaving ``use_remote_ssh`` false.  This
    function makes the target operational and fails closed for execution lanes
    that use a different API.
    """

    normalized = dict(payload)
    target = str(normalized.get("compute_target") or "").strip()
    use_remote = bool(normalized.get("use_remote_ssh", False))
    legacy_nemo_gym = bool(normalized.get("use_nemo_gym", False))
    use_nemo_customizer = bool(normalized.get("use_nemo_customizer", False)) or legacy_nemo_gym
    device_id = str(normalized.get("device_id") or "").strip()

    if use_remote and use_nemo_customizer:
        raise ValueError("use_remote_ssh and use_nemo_customizer cannot both be enabled")

    if use_nemo_customizer:
        normalized["use_nemo_customizer"] = True

    if not target:
        if use_remote:
            normalized["compute_target"] = f"ssh:{device_id}" if device_id else "ssh:remote"
        elif use_nemo_customizer:
            normalized["compute_target"] = "cloud:nemo-customizer"
        return normalized

    target_key = target.casefold()
    if target_key == "local":
        if use_remote or use_nemo_customizer:
            raise ValueError("compute_target 'local' conflicts with a remote/cloud backend flag")
        if device_id:
            raise ValueError("device_id is only valid with an ssh compute target")
        normalized["compute_target"] = "local"
        return normalized

    if target_key == "private":
        if use_nemo_customizer:
            raise ValueError("the private target alias cannot use NeMo Customizer")
        normalized["use_remote_ssh"] = True
        normalized["compute_target"] = "ssh:remote"
        return normalized

    if target_key.startswith("ssh:"):
        if use_nemo_customizer:
            raise ValueError("an ssh compute target cannot use NeMo Customizer")
        target_device = target.split(":", 1)[1].strip()
        if not target_device:
            raise ValueError("ssh compute targets must include a device id: ssh:<device_id>")
        if target_device.casefold() != "remote":
            if device_id and device_id != target_device:
                raise ValueError(
                    f"device_id '{device_id}' conflicts with compute_target '{target}'"
                )
            normalized["device_id"] = target_device
        normalized["use_remote_ssh"] = True
        normalized["compute_target"] = target
        return normalized

    if target_key in {
        "cloud",
        "cloud:nemo-customizer",
        "nemo-customizer",
        # Backward-compatible aliases. These never mean NeMo Gym or NeMo RL.
        "cloud:nemo",
        "nemo",
    }:
        if use_remote:
            raise ValueError("a cloud compute target cannot use the SSH backend")
        if target_key == "cloud" and not use_nemo_customizer:
            raise ValueError(
                "compute_target 'cloud' is ambiguous; use cloud:nemo-customizer, "
                "the Hugging Face Jobs activation path, or /api/training/managed/submit"
            )
        normalized["use_nemo_customizer"] = True
        normalized["compute_target"] = "cloud:nemo-customizer"
        return normalized

    if target_key in {"hf-jobs", "hf_jobs", "cloud:hf-jobs", "cloud:hf_jobs"}:
        raise ValueError(
            "Hugging Face Jobs is not launched by /api/training/start; use the "
            "Hugging Face Jobs activation path and ingest its job/artifact evidence into BashGym"
        )

    if target_key.startswith(("managed:", "skypilot", "dstack")):
        raise ValueError(
            f"compute_target '{target}' is not executable through /api/training/start; "
            "use its provider-specific activation surface"
        )

    raise ValueError(
        f"unsupported compute_target '{target}'; expected local, private, "
        "ssh:<device_id>, or cloud:nemo-customizer"
    )
