"""Strict validation for DPO/preference pair artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PREFERENCE_PAIR_VALIDATION_SCHEMA_VERSION = "bashgym.preference_pair_validation.v1"


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
        if value not in (None, ""):
            return value
    return None


def _finding(
    *,
    code: str,
    level: str,
    message: str,
    index: int,
    pair_id: str | None,
) -> dict[str, Any]:
    return {
        "code": code,
        "level": level,
        "message": message,
        "index": index,
        "pair_id": pair_id,
    }


def _level(strict: bool) -> str:
    return "fail" if strict else "warn"


def _infer_generation_method(metadata: dict[str, Any]) -> str:
    explicit = _metadata_text(metadata, "pair_generation_method", "preference_source")
    if explicit:
        return explicit
    if metadata.get("decision_level"):
        return "decision_level_failure_recovery"
    if _metadata_any(metadata, "similarity", "embedding_similarity") is not None:
        return "embedding_similarity_trace_pair"
    if _metadata_any(metadata, "gold_trace", "failed_trace"):
        return "trace_pair"
    return ""


def _has_trace_provenance(metadata: dict[str, Any]) -> tuple[bool, bool]:
    decision_level = bool(metadata.get("decision_level") and metadata.get("source_trace"))
    has_chosen = bool(
        decision_level
        or _metadata_any(
            metadata,
            "chosen_trace_id",
            "gold_trace_id",
            "gold_trace",
            "source_trace",
        )
    )
    has_rejected = bool(
        decision_level
        or _metadata_any(
            metadata,
            "rejected_trace_id",
            "failed_trace_id",
            "failed_trace",
            "source_trace",
        )
    )
    return has_chosen, has_rejected


def _has_quality(metadata: dict[str, Any]) -> bool:
    return any(
        _metadata_any(metadata, key) is not None
        for key in (
            "quality_score",
            "chosen_quality_score",
            "rejected_quality_score",
            "chosen_verification_score",
            "rejected_verification_score",
            "score_delta",
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


def _pair_identity(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    return (
        _text(record, "pair_id", "id", "example_id")
        or _metadata_text(metadata, "pair_id", "id", "example_id")
        or ""
    )


def validate_preference_pair_records(
    records: list[dict[str, Any]],
    *,
    strict: bool = False,
    max_length_ratio: float = 3.0,
) -> dict[str, Any]:
    """Validate DPO/preference records.

    Lightweight mode checks structural safety. Strict mode turns missing
    provenance, quality, split, and contamination metadata into failures for
    serious runs.
    """

    findings: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            findings.append(
                _finding(
                    code="invalid_record_type",
                    level="fail",
                    message="preference pair record must be a JSON object",
                    index=index,
                    pair_id=None,
                )
            )
            continue

        metadata = _metadata(record)
        pair_id = _pair_identity(record, metadata)
        prompt = _text(record, "prompt", "input", "question")
        chosen = _text(record, "chosen", "chosen_response", "chosen_text")
        rejected = _text(record, "rejected", "rejected_response", "rejected_text")
        prompt_hash = _text(record, "prompt_hash") or _metadata_text(metadata, "prompt_hash")
        computed_prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16] if prompt else ""
        generation_method = _infer_generation_method(metadata)

        normalized.append(
            {
                "pair_id": pair_id,
                "prompt_hash": prompt_hash or computed_prompt_hash,
                "generation_method": generation_method,
                "chosen_chars": len(chosen),
                "rejected_chars": len(rejected),
            }
        )

        if not pair_id:
            findings.append(
                _finding(
                    code="missing_pair_id",
                    level="fail",
                    message="pair_id/id is required",
                    index=index,
                    pair_id=None,
                )
            )
        if not prompt:
            findings.append(
                _finding(
                    code="missing_prompt",
                    level="fail",
                    message="prompt is required",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not chosen:
            findings.append(
                _finding(
                    code="missing_chosen",
                    level="fail",
                    message="chosen/chosen_response is required",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not rejected:
            findings.append(
                _finding(
                    code="missing_rejected",
                    level="fail",
                    message="rejected/rejected_response is required",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if chosen and rejected and chosen.strip() == rejected.strip():
            findings.append(
                _finding(
                    code="identical_chosen_rejected",
                    level="fail",
                    message="chosen and rejected responses are identical",
                    index=index,
                    pair_id=pair_id or None,
                )
            )

        if chosen and rejected:
            shorter = max(1, min(len(chosen), len(rejected)))
            ratio = max(len(chosen), len(rejected)) / shorter
            if ratio > max_length_ratio:
                findings.append(
                    _finding(
                        code="extreme_length_ratio",
                        level="warn",
                        message=(
                            f"chosen/rejected length ratio {ratio:.2f} exceeds "
                            f"{max_length_ratio:.2f}"
                        ),
                        index=index,
                        pair_id=pair_id or None,
                    )
                )

        if not prompt_hash:
            findings.append(
                _finding(
                    code="missing_saved_prompt_hash",
                    level=_level(strict),
                    message="strict DPO evidence should save prompt_hash, not only compute it later",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not generation_method:
            findings.append(
                _finding(
                    code="missing_pair_generation_method",
                    level=_level(strict),
                    message="pair_generation_method is required for serious DPO runs",
                    index=index,
                    pair_id=pair_id or None,
                )
            )

        has_chosen_trace, has_rejected_trace = _has_trace_provenance(metadata)
        if not has_chosen_trace:
            findings.append(
                _finding(
                    code="missing_chosen_trace_provenance",
                    level=_level(strict),
                    message="chosen trace/source provenance is missing",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not has_rejected_trace:
            findings.append(
                _finding(
                    code="missing_rejected_trace_provenance",
                    level=_level(strict),
                    message="rejected trace/source provenance is missing",
                    index=index,
                    pair_id=pair_id or None,
                )
            )

        if not _metadata_any(metadata, "label_strength", "preference_strength", "label_source"):
            findings.append(
                _finding(
                    code="missing_label_strength",
                    level=_level(strict),
                    message="label strength/source metadata is required for serious DPO runs",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not _has_quality(metadata):
            findings.append(
                _finding(
                    code="missing_quality_scores",
                    level=_level(strict),
                    message="quality or verification score metadata is missing",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not _metadata_any(metadata, "domain", "task_family"):
            findings.append(
                _finding(
                    code="missing_domain_or_task_family",
                    level=_level(strict),
                    message="domain/task-family metadata is missing",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not _metadata_any(metadata, "split", "split_id", "split_policy"):
            findings.append(
                _finding(
                    code="missing_split_metadata",
                    level=_level(strict),
                    message="split metadata is missing",
                    index=index,
                    pair_id=pair_id or None,
                )
            )
        if not _has_decontamination(metadata):
            findings.append(
                _finding(
                    code="missing_decontamination_metadata",
                    level=_level(strict),
                    message="decontamination/split-manifest metadata is missing",
                    index=index,
                    pair_id=pair_id or None,
                )
            )

    fail_count = sum(1 for finding in findings if finding["level"] == "fail")
    warn_count = sum(1 for finding in findings if finding["level"] == "warn")
    return {
        "schema_version": PREFERENCE_PAIR_VALIDATION_SCHEMA_VERSION,
        "ok": fail_count == 0,
        "strict": strict,
        "total_records": len(records),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "findings": findings,
        "pairs": normalized,
    }


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("pairs", "records", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON preference artifact must be a list or contain pairs/records/data")

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


def validate_preference_pairs_file(
    path: str | Path,
    *,
    strict: bool = False,
    max_length_ratio: float = 3.0,
) -> dict[str, Any]:
    records = _load_records(path)
    result = validate_preference_pair_records(
        records,
        strict=strict,
        max_length_ratio=max_length_ratio,
    )
    result["path"] = str(path)
    return result
