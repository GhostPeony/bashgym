"""Strict validation for reward-model training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REWARD_EXAMPLE_VALIDATION_SCHEMA_VERSION = "bashgym.reward_example_validation.v1"
VALID_REWARD_TYPES = {
    "preference_reward",
    "outcome_reward",
    "process_reward",
    "orm",
    "prm",
}


def _text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _metadata_text(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _metadata_any(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _finding(
    *,
    code: str,
    level: str,
    message: str,
    index: int,
    example_id: str | None,
) -> dict[str, Any]:
    return {
        "code": code,
        "level": level,
        "message": message,
        "index": index,
        "example_id": example_id,
    }


def _level(strict: bool) -> str:
    return "fail" if strict else "warn"


def _example_id(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    return (
        _text(record, "reward_example_id", "example_id", "id")
        or _metadata_text(metadata, "reward_example_id", "example_id", "id")
        or ""
    )


def _reward_type(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    raw = (
        _text(record, "reward_type", "type")
        or _metadata_text(metadata, "reward_type", "type", "artifact_type")
        or ""
    )
    return raw.lower().strip()


def _has_prompt(record: dict[str, Any]) -> bool:
    return bool(_text(record, "prompt", "input", "question", "instruction"))


def _has_response_or_trajectory(record: dict[str, Any]) -> bool:
    if _text(record, "response", "completion", "answer", "chosen", "text"):
        return True
    for key in ("trajectory", "messages", "steps", "trace", "rollout"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            return True
    return False


def _has_reward_value(record: dict[str, Any], metadata: dict[str, Any]) -> bool:
    for key in ("reward", "score", "rating", "label", "target", "preference_score"):
        if record.get(key) not in (None, ""):
            return True
    return _metadata_any(metadata, "reward", "score", "rating", "label", "target") is not None


def _has_process_signal(record: dict[str, Any], metadata: dict[str, Any]) -> bool:
    for key in ("step_rewards", "process_rewards", "step_scores", "steps"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            return True
    return _metadata_any(metadata, "step_rewards", "process_rewards", "step_scores") is not None


def _has_quality(metadata: dict[str, Any]) -> bool:
    return any(
        _metadata_any(metadata, key) is not None
        for key in (
            "quality_score",
            "label_confidence",
            "judge_confidence",
            "calibration_score",
            "verifier_score",
        )
    )


def _has_decontamination(metadata: dict[str, Any]) -> bool:
    return any(
        _metadata_any(metadata, key) is not None
        for key in (
            "decontamination_manifest",
            "decontamination_status",
            "contamination_checked",
            "split_manifest",
        )
    )


def validate_reward_example_records(
    records: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate reward-model, ORM, and PRM training records.

    Lightweight mode checks structural usability. Strict mode turns missing
    provenance, reward-scale, quality, split, and contamination metadata into
    failures for serious reward-model evidence.
    """

    findings: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            findings.append(
                _finding(
                    code="invalid_record_type",
                    level="fail",
                    message="reward example record must be a JSON object",
                    index=index,
                    example_id=None,
                )
            )
            continue

        metadata = _metadata(record)
        example_id = _example_id(record, metadata)
        reward_type = _reward_type(record, metadata)
        has_process = _has_process_signal(record, metadata)

        normalized.append(
            {
                "example_id": example_id,
                "reward_type": reward_type,
                "has_prompt": _has_prompt(record),
                "has_process_signal": has_process,
            }
        )

        if not example_id:
            findings.append(
                _finding(
                    code="missing_example_id",
                    level="fail",
                    message="reward_example_id/example_id/id is required",
                    index=index,
                    example_id=None,
                )
            )
        if not reward_type:
            findings.append(
                _finding(
                    code="missing_reward_type",
                    level=_level(strict),
                    message="reward_type is required for reward-model artifacts",
                    index=index,
                    example_id=example_id or None,
                )
            )
        elif reward_type not in VALID_REWARD_TYPES:
            findings.append(
                _finding(
                    code="unknown_reward_type",
                    level="warn",
                    message=f"unknown reward_type {reward_type!r}",
                    index=index,
                    example_id=example_id or None,
                )
            )

        if not _has_prompt(record):
            findings.append(
                _finding(
                    code="missing_prompt",
                    level="fail",
                    message="prompt/input/question/instruction is required",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _has_response_or_trajectory(record):
            findings.append(
                _finding(
                    code="missing_response_or_trajectory",
                    level="fail",
                    message="response/completion/trajectory/messages/steps is required",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _has_reward_value(record, metadata) and not has_process:
            findings.append(
                _finding(
                    code="missing_reward_value",
                    level="fail",
                    message="reward/score/rating/label or process step rewards are required",
                    index=index,
                    example_id=example_id or None,
                )
            )

        if reward_type in {"process_reward", "prm"} and not has_process:
            findings.append(
                _finding(
                    code="missing_process_reward_steps",
                    level="fail",
                    message="process reward examples require step-level rewards or scored steps",
                    index=index,
                    example_id=example_id or None,
                )
            )

        if not _metadata_any(metadata, "reward_scale", "score_scale", "label_schema"):
            findings.append(
                _finding(
                    code="missing_reward_scale",
                    level=_level(strict),
                    message="reward scale or label schema is required for serious reward runs",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _metadata_any(metadata, "label_source", "judge_model", "verifier_id", "annotator"):
            findings.append(
                _finding(
                    code="missing_label_source",
                    level=_level(strict),
                    message="label source, judge model, verifier id, or annotator is required",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _metadata_any(metadata, "source_id", "source_ids", "source_manifest_path"):
            findings.append(
                _finding(
                    code="missing_source_provenance",
                    level=_level(strict),
                    message="source provenance is missing",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _has_quality(metadata):
            findings.append(
                _finding(
                    code="missing_quality_or_confidence",
                    level=_level(strict),
                    message="quality, confidence, calibration, or verifier score metadata is missing",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _metadata_any(metadata, "domain", "task_family"):
            findings.append(
                _finding(
                    code="missing_domain_or_task_family",
                    level=_level(strict),
                    message="domain/task-family metadata is missing",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _metadata_any(metadata, "split", "split_id", "split_policy"):
            findings.append(
                _finding(
                    code="missing_split_metadata",
                    level=_level(strict),
                    message="split metadata is missing",
                    index=index,
                    example_id=example_id or None,
                )
            )
        if not _has_decontamination(metadata):
            findings.append(
                _finding(
                    code="missing_decontamination_metadata",
                    level=_level(strict),
                    message="decontamination/split-manifest metadata is missing",
                    index=index,
                    example_id=example_id or None,
                )
            )

    fail_count = sum(1 for finding in findings if finding["level"] == "fail")
    warn_count = sum(1 for finding in findings if finding["level"] == "warn")
    return {
        "schema_version": REWARD_EXAMPLE_VALIDATION_SCHEMA_VERSION,
        "ok": fail_count == 0,
        "strict": strict,
        "total_records": len(records),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "findings": findings,
        "examples": normalized,
    }


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("examples", "records", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON reward artifact must be a list or contain examples/records/data")

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


def validate_reward_examples_file(
    path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    records = _load_records(path)
    result = validate_reward_example_records(records, strict=strict)
    result["path"] = str(path)
    return result
