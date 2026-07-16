"""Fail-closed conversion of NeMo Gym runtime output into campaign evidence.

This module accepts only explicit rollout envelopes and exact refit receipts. It
does not treat process completion, checkpoint presence, or a successful exit code
as proof that the policy and generation workers were synchronized.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bashgym.campaigns.contracts import ActionAttempt
from bashgym.campaigns.nemo_gym_evidence import (
    NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
    NemoGymCampaignEvidence,
    NemoGymRefitEvidence,
    build_nemo_gym_campaign_evidence,
    write_nemo_gym_campaign_evidence,
)
from bashgym.environments.contracts import EnvironmentSpec

NEMO_GYM_TRAJECTORIES_FILENAME = "nemo_gym_trajectories.jsonl"
NEMO_GYM_REFIT_RECEIPT_FILENAME = "nemo_gym_refit_receipt.json"
NEMO_GYM_BUNDLE_MANIFEST_FILENAME = "nemo_gym_bundle_manifest.json"
NEMO_GYM_ENVIRONMENT_CONTRACT_FILENAME = "nemo_gym_environment_contract.json"

_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024
_MAX_TRAJECTORY_RECORD_BYTES = 4 * 1024 * 1024
_MAX_TRAJECTORY_RECORDS = 4096
_ROLLOUT_WRAPPERS = ("trajectory", "rollout", "result", "full_result")
_REFIT_FIELDS = ("refit", "refit_receipt")


def _regular_file(path: Path, *, label: str, byte_limit: int) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular file")
    resolved = candidate.resolve()
    if resolved.stat().st_size > byte_limit:
        raise ValueError(f"{label} exceeds the size limit")
    return resolved


def load_bounded_json_object(
    path: str | Path,
    *,
    label: str,
    byte_limit: int = _MAX_DOCUMENT_BYTES,
) -> dict[str, Any]:
    """Load one bounded UTF-8 JSON object from a regular, non-symlink file."""

    candidate = _regular_file(Path(path), label=label, byte_limit=byte_limit)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(payload)


def _bounded_mapping(record: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    try:
        encoded = json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"NeMo Gym trajectory record {index} is not JSON-compatible") from exc
    if len(encoded) > _MAX_TRAJECTORY_RECORD_BYTES:
        raise ValueError(f"NeMo Gym trajectory record {index} exceeds the size limit")
    return dict(record)


def load_nemo_gym_trajectory_records(
    source: str | Path | Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Load a bounded JSON array or JSONL stream without guessing its schema."""

    if isinstance(source, (str, Path)):
        candidate = _regular_file(
            Path(source),
            label="NeMo Gym trajectory output",
            byte_limit=_MAX_TRAJECTORY_BYTES,
        )
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("NeMo Gym trajectory output is not valid UTF-8") from exc
        if candidate.suffix.casefold() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("NeMo Gym trajectory JSON is invalid") from exc
            if not isinstance(payload, list):
                raise ValueError("NeMo Gym trajectory JSON must contain an array")
            raw_records: Sequence[Any] = payload
        else:
            raw_records = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > _MAX_TRAJECTORY_RECORD_BYTES:
                    raise ValueError(
                        f"NeMo Gym trajectory JSONL line {line_number} exceeds the size limit"
                    )
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"NeMo Gym trajectory JSONL line {line_number} is invalid"
                    ) from exc
                raw_records.append(parsed)
    else:
        if isinstance(source, (bytes, bytearray)):
            raise ValueError("NeMo Gym trajectories must be records or a file path")
        raw_records = source

    if not raw_records:
        raise ValueError("NeMo Gym trajectory output cannot be empty")
    if len(raw_records) > _MAX_TRAJECTORY_RECORDS:
        raise ValueError("NeMo Gym trajectory output contains too many records")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"NeMo Gym trajectory record {index} must be an object")
        records.append(_bounded_mapping(record, index=index))
    return tuple(records)


def _decode_wrapper(value: Any, *, field: str, index: int) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if field == "full_result" and isinstance(value, str):
        if len(value.encode("utf-8")) > _MAX_TRAJECTORY_RECORD_BYTES:
            raise ValueError(f"NeMo Gym trajectory record {index} full_result is too large")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"NeMo Gym trajectory record {index} full_result is not valid JSON"
            ) from exc
        if isinstance(decoded, Mapping):
            return dict(decoded)
    raise ValueError(f"NeMo Gym trajectory record {index} field {field!r} must contain an object")


def _explicit_rollout(record: Mapping[str, Any], *, index: int) -> tuple[dict[str, Any], bool]:
    """Extract one rollout from a small set of documented runtime wrappers."""

    present = [field for field in _ROLLOUT_WRAPPERS if field in record]
    direct = "session_id" in record or "example_index" in record
    if direct and present:
        raise ValueError(f"NeMo Gym trajectory record {index} has ambiguous rollout payloads")
    if len(present) > 1:
        raise ValueError(f"NeMo Gym trajectory record {index} has ambiguous rollout wrappers")
    if direct:
        return dict(record), False
    if not present:
        raise ValueError(f"NeMo Gym trajectory record {index} has no explicit rollout envelope")
    return _decode_wrapper(record[present[0]], field=present[0], index=index), True


def _exact_refit_binding(
    record: Mapping[str, Any],
    rollout: Mapping[str, Any],
    *,
    nested: bool,
    index: int,
) -> NemoGymRefitEvidence:
    candidates: list[Any] = [rollout[field] for field in _REFIT_FIELDS if field in rollout]
    if nested:
        candidates.extend(record[field] for field in _REFIT_FIELDS if field in record)
    if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
        raise ValueError(
            f"NeMo Gym trajectory record {index} requires one explicit exact refit receipt"
        )
    try:
        return NemoGymRefitEvidence.model_validate(dict(candidates[0]))
    except ValueError as exc:
        raise ValueError(
            f"NeMo Gym trajectory record {index} has an invalid exact refit receipt"
        ) from exc


def _environment_from_exact_contract(payload: Mapping[str, Any]) -> EnvironmentSpec:
    """Parse a contract while preserving explicit absence of optional fields."""

    raw = dict(payload)
    environment = EnvironmentSpec.from_dict(raw)
    raw_verifier = raw.get("verifier")
    if isinstance(raw_verifier, Mapping) and "path" not in raw_verifier:
        environment.verifier.path = None
    raw_build = raw.get("build")
    if isinstance(raw_build, Mapping) and "dockerfile" not in raw_build:
        environment.build.dockerfile = None
    if environment.to_dict() != raw:
        raise ValueError(
            "NeMo Gym environment contract must be explicit and canonically round-trippable"
        )
    return environment


def build_nemo_gym_evidence_from_outputs(
    attempt: ActionAttempt,
    *,
    bundle_manifest: Mapping[str, Any] | str | Path,
    environment_contract: Mapping[str, Any] | EnvironmentSpec | str | Path,
    trajectories: str | Path | Sequence[Mapping[str, Any]],
    refit_receipt_path: str | Path,
) -> NemoGymCampaignEvidence:
    """Convert actual Gym outputs only when every rollout proves one exact refit."""

    if isinstance(bundle_manifest, (str, Path)):
        bundle = load_bounded_json_object(bundle_manifest, label="NeMo Gym bundle manifest")
    else:
        bundle = dict(bundle_manifest)

    if isinstance(environment_contract, EnvironmentSpec):
        environment = environment_contract
    else:
        if isinstance(environment_contract, (str, Path)):
            raw_environment = load_bounded_json_object(
                environment_contract, label="NeMo Gym environment contract"
            )
        else:
            raw_environment = dict(environment_contract)
        environment = _environment_from_exact_contract(raw_environment)

    raw_refit = load_bounded_json_object(
        refit_receipt_path,
        label="NeMo Gym refit receipt",
        byte_limit=_MAX_DOCUMENT_BYTES,
    )
    try:
        exact_refit = NemoGymRefitEvidence.model_validate(raw_refit)
    except ValueError as exc:
        raise ValueError("NeMo Gym refit receipt is invalid or not synchronized") from exc

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(load_nemo_gym_trajectory_records(trajectories)):
        rollout, nested = _explicit_rollout(record, index=index)
        rollout_refit = _exact_refit_binding(record, rollout, nested=nested, index=index)
        if rollout_refit != exact_refit:
            raise ValueError(
                f"NeMo Gym trajectory record {index} is not bound to the exact refit receipt"
            )
        rollout["refit"] = exact_refit.model_dump(mode="json")
        rollout.pop("refit_receipt", None)
        normalized.append(rollout)

    return build_nemo_gym_campaign_evidence(
        attempt,
        bundle_manifest=bundle,
        environment=environment,
        rollout_payloads=normalized,
    )


def convert_nemo_gym_outputs(
    attempt: ActionAttempt,
    *,
    bundle_manifest: Mapping[str, Any] | str | Path,
    environment_contract: Mapping[str, Any] | EnvironmentSpec | str | Path,
    trajectories: str | Path | Sequence[Mapping[str, Any]],
    refit_receipt_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """Build the canonical, non-overwriting campaign evidence artifact."""

    evidence = build_nemo_gym_evidence_from_outputs(
        attempt,
        bundle_manifest=bundle_manifest,
        environment_contract=environment_contract,
        trajectories=trajectories,
        refit_receipt_path=refit_receipt_path,
    )
    destination = Path(output_directory).expanduser() / NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME
    return write_nemo_gym_campaign_evidence(destination, evidence)


__all__ = [
    "NEMO_GYM_BUNDLE_MANIFEST_FILENAME",
    "NEMO_GYM_ENVIRONMENT_CONTRACT_FILENAME",
    "NEMO_GYM_REFIT_RECEIPT_FILENAME",
    "NEMO_GYM_TRAJECTORIES_FILENAME",
    "build_nemo_gym_evidence_from_outputs",
    "convert_nemo_gym_outputs",
    "load_bounded_json_object",
    "load_nemo_gym_trajectory_records",
]
