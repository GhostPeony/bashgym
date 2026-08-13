"""Shared primitives for preference and reward artifact validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def first_text(record: dict[str, Any], *keys: str) -> str:
    """Return the first non-blank string stored under ``keys``."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return record metadata when it is an object, otherwise an empty object."""
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def validation_level(strict: bool) -> str:
    """Map strict-mode findings to failures and lightweight findings to warnings."""
    return "fail" if strict else "warn"


def load_json_records(
    path: str | Path,
    *,
    container_keys: Sequence[str],
    artifact_name: str,
) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL artifact without applying domain validation."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in container_keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        if isinstance(payload, list):
            return payload
        keys = "/".join(container_keys)
        raise ValueError(f"JSON {artifact_name} artifact must be a list or contain {keys}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        records.append(payload)
    return records
