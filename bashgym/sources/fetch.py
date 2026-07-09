"""Hugging Face fetch orchestration for BashGym source cards."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from bashgym.sources.catalog import SourceCard

SOURCE_FETCH_SCHEMA_VERSION = "bashgym.source_fetch.v1"
DEFAULT_SOURCE_FETCH_LIMIT = 1000
SOURCE_FETCH_APPROVAL_LIMIT = DEFAULT_SOURCE_FETCH_LIMIT

DatasetLoader = Callable[..., Iterable[Any]]


def source_fetch_approval_policy() -> dict[str, Any]:
    """Return the fail-closed policy for larger Hugging Face source pulls."""

    return {
        "default_limit": DEFAULT_SOURCE_FETCH_LIMIT,
        "approval_limit": SOURCE_FETCH_APPROVAL_LIMIT,
        "requires_reason_when": "limit is omitted/unbounded or greater than approval_limit",
        "reason": (
            "Large public-source fetches can take time, consume quota, or pull more data "
            "than intended. Provide an approval reason to record why the larger pull is safe."
        ),
    }


def _fetch_requires_approval(limit: int | None) -> bool:
    return limit is None or limit > SOURCE_FETCH_APPROVAL_LIMIT


def _fetch_request(
    card: SourceCard,
    *,
    split: str,
    subset: str | None,
    revision: str | None,
    limit: int | None,
) -> dict[str, Any]:
    return {
        "source_id": card.id,
        "huggingface_id": card.huggingface_id,
        "split": split,
        "subset": subset,
        "revision": revision,
        "limit": limit,
    }


def _load_huggingface_dataset(
    huggingface_id: str,
    *,
    split: str,
    subset: str | None = None,
    revision: str | None = None,
) -> Iterable[Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset fetch requires the training extras. "
            "Install with `pip install -e .[training]` or provide a local JSON/JSONL input."
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    if revision:
        kwargs["revision"] = revision
    if subset:
        return load_dataset(huggingface_id, subset, **kwargs)
    return load_dataset(huggingface_id, **kwargs)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _record_id(record: dict[str, Any], index: int) -> str:
    metadata = record.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    for key in ("source_record_id", "record_id", "id", "example_id", "pair_id"):
        value = record.get(key) or metadata_dict.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return f"record-{index + 1:06d}"


def _with_source_metadata(
    card: SourceCard,
    row: Any,
    *,
    index: int,
    split: str,
    subset: str | None,
    revision: str | None,
) -> dict[str, Any]:
    record = _jsonable(row)
    if not isinstance(record, dict):
        record = {"value": record}

    payload = dict(record)
    metadata = payload.get("metadata")
    metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
    source_record_id = _record_id(payload, index)
    source_fetch = metadata_dict.get("source_fetch")
    source_fetch_dict = dict(source_fetch) if isinstance(source_fetch, dict) else {}
    source_fetch_dict.update(
        {
            "schema_version": SOURCE_FETCH_SCHEMA_VERSION,
            "source_id": card.id,
            "source_name": card.name,
            "huggingface_id": card.huggingface_id,
            "source_record_id": source_record_id,
            "source_record_index": index,
            "split": split,
            "subset": subset,
            "revision": revision,
        }
    )
    metadata_dict["source_fetch"] = source_fetch_dict
    metadata_dict.setdefault("source_record_id", source_record_id)
    metadata_dict.setdefault("source_id", card.id)
    metadata_dict.setdefault("source_name", card.name)
    metadata_dict.setdefault("split", split)
    metadata_dict.setdefault("domain", card.domain)
    metadata_dict.setdefault("task_family", card.task_family)

    payload["metadata"] = metadata_dict
    payload.setdefault("source_record_id", source_record_id)
    payload.setdefault("split", split)
    return payload


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    report["report_path"] = str(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _read_cached_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cached_report_matches(
    report: dict[str, Any] | None,
    *,
    request: dict[str, Any],
    records_path: Path,
) -> bool:
    if not report or not report.get("ok"):
        return False
    if report.get("schema_version") != SOURCE_FETCH_SCHEMA_VERSION:
        return False
    if report.get("request") != request:
        return False
    return records_path.exists()


def fetch_source_records(
    card: SourceCard,
    *,
    output_dir: str | Path,
    split: str = "train",
    limit: int | None = DEFAULT_SOURCE_FETCH_LIMIT,
    subset: str | None = None,
    revision: str | None = None,
    approval_reason: str | None = None,
    force_refresh: bool = False,
    loader: DatasetLoader | None = None,
) -> dict[str, Any]:
    """Fetch a Hugging Face-backed source card into local JSONL records.

    The local JSONL is intentionally generic. Goal-specific conversion stays in
    ``prepare_source_artifacts`` so data provenance and policy checks remain in
    one place.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records_path = output_path / "source_records.jsonl"
    report_path = output_path / "source_fetch_report.json"
    request = _fetch_request(
        card,
        split=split,
        subset=subset,
        revision=revision,
        limit=limit,
    )
    approval_required = _fetch_requires_approval(limit)
    approval_reason_clean = approval_reason.strip() if approval_reason else None
    report: dict[str, Any] = {
        "schema_version": SOURCE_FETCH_SCHEMA_VERSION,
        "ok": True,
        "source_id": card.id,
        "source_name": card.name,
        "huggingface_id": card.huggingface_id,
        "split": split,
        "subset": subset,
        "revision": revision,
        "limit": limit,
        "output_dir": str(output_path),
        "records_path": str(records_path),
        "record_count": 0,
        "truncated": False,
        "request": request,
        "cache_enabled": True,
        "cache_hit": False,
        "force_refresh": force_refresh,
        "approval_policy": source_fetch_approval_policy(),
        "approval_required": approval_required,
        "approval_granted": not approval_required or bool(approval_reason_clean),
        "approval_reason": approval_reason_clean,
        "warnings": [],
        "errors": [],
    }
    if not card.huggingface_id:
        report["ok"] = False
        report["errors"].append("source_has_no_huggingface_id")
        _write_report(report_path, report)
        return report

    if approval_required and not approval_reason_clean:
        report["ok"] = False
        report["errors"].append("remote_fetch_approval_required")
        report["warnings"].append(
            f"Set an approval reason to fetch more than {SOURCE_FETCH_APPROVAL_LIMIT} records."
        )
        _write_report(report_path, report)
        return report

    if not force_refresh:
        cached_report = _read_cached_report(report_path)
        if _cached_report_matches(cached_report, request=request, records_path=records_path):
            cached = dict(cached_report or {})
            cached.update(
                {
                    "cache_enabled": True,
                    "cache_hit": True,
                    "force_refresh": False,
                    "approval_policy": source_fetch_approval_policy(),
                    "approval_required": approval_required,
                    "approval_granted": True,
                    "approval_reason": approval_reason_clean or cached.get("approval_reason"),
                    "request": request,
                    "output_dir": str(output_path),
                    "records_path": str(records_path),
                }
            )
            _write_report(report_path, cached)
            return cached

    if limit is None:
        report["warnings"].append("unbounded_source_fetch")

    load = loader or _load_huggingface_dataset
    try:
        dataset = load(card.huggingface_id, split=split, subset=subset, revision=revision)
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"source_fetch_failed: {exc}")
        _write_report(report_path, report)
        return report

    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            report["truncated"] = True
            break
        records.append(
            _with_source_metadata(
                card,
                row,
                index=index,
                split=split,
                subset=subset,
                revision=revision,
            )
        )

    _write_jsonl(records_path, records)
    report["record_count"] = len(records)
    if not records:
        report["ok"] = False
        report["errors"].append("no_source_records_fetched")
    _write_report(report_path, report)
    return report
