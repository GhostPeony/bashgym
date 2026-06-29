"""Structured agent status events and replay scrubbing helpers.

The status protocol gives agent integrations a narrow JSON marker to emit when
they need to report state. Ingesters should parse those markers instead of
guessing status from free-form terminal text.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import SurfaceType, TraceEvent, validate_event

AGENT_STATUS_SCHEMA_VERSION = "bashgym.agent_status.v1"
REPLAY_SCRUB_SCHEMA_VERSION = "bashgym.trace_replay_scrub.v1"
AGENT_STATUS_MARKER = "::bashgym-agent-status"

ALLOWED_AGENT_STATUSES = frozenset(
    {
        "planning",
        "researching",
        "running",
        "tool_call",
        "observing",
        "awaiting_input",
        "awaiting_approval",
        "blocked",
        "completed",
        "failed",
        "cancelled",
    }
)

_STATUS_ALIASES = {
    "approve": "awaiting_approval",
    "awaiting_human": "awaiting_input",
    "done": "completed",
    "error": "failed",
    "in_progress": "running",
    "needs_approval": "awaiting_approval",
    "needs_input": "awaiting_input",
    "observed": "observing",
    "research": "researching",
    "tool": "tool_call",
    "waiting": "awaiting_input",
}

_STATUS_KEYS = {
    "schema_version",
    "status",
    "state",
    "message",
    "summary",
    "source",
    "source_tool",
    "session_id",
    "step_id",
    "progress",
    "blocked_reason",
    "next_action",
    "metadata",
}

_OUTPUT_FIELD_NAMES = frozenset(
    {
        "content",
        "log",
        "logs",
        "output",
        "stderr",
        "stdout",
        "terminal_output",
        "transcript",
    }
)


@dataclass
class _ScrubStats:
    redactions: int = 0
    truncations: int = 0
    redacted_fields: set[str] = field(default_factory=set)
    truncated_fields: set[str] = field(default_factory=set)

    def to_dict(self, max_output_chars: int) -> dict[str, Any]:
        return {
            "redactions": self.redactions,
            "truncations": self.truncations,
            "redacted_fields": sorted(self.redacted_fields),
            "truncated_fields": sorted(self.truncated_fields),
            "max_output_chars": max_output_chars,
        }


_Replacement = str | Callable[[re.Match[str]], str]
_DEFAULT_REDACT_PATTERNS: tuple[tuple[str, re.Pattern[str], _Replacement], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        "named_secret",
        re.compile(
            r"(?i)([\"']?\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|bearer|"
            r"client[_-]?secret|hf[_-]?token|openai[_-]?api[_-]?key|password|"
            r"secret|token)\b[\"']?\s*[:=]\s*[\"']?)([^\"'\s,}]+)"
        ),
        lambda match: f"{match.group(1)}[REDACTED]",
    ),
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "[REDACTED_OPENAI_KEY]",
    ),
    (
        "github_token",
        re.compile(r"\bgh[opusr]_[A-Za-z0-9_]{20,}\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (
        "huggingface_token",
        re.compile(r"\bhf_[A-Za-z0-9_]{16,}\b"),
        "[REDACTED_HF_TOKEN]",
    ),
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED_AWS_ACCESS_KEY]",
    ),
)


def normalize_agent_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an agent-status payload.

    The status field is deliberately small and explicit. Free-form terminal
    output should stay in trace events; machine state goes through this schema.
    """

    if not isinstance(payload, dict):
        raise ValueError("agent status payload must be a dictionary")

    raw_status = str(payload.get("status") or payload.get("state") or "").strip()
    if not raw_status:
        raise ValueError("agent status payload requires a status")

    status = raw_status.lower().replace("-", "_").replace(" ", "_")
    status = _STATUS_ALIASES.get(status, status)
    if status not in ALLOWED_AGENT_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_AGENT_STATUSES))
        raise ValueError(f"unsupported agent status {raw_status!r}; expected one of: {allowed}")

    metadata = dict(payload.get("metadata") or {})
    for key, value in payload.items():
        if key not in _STATUS_KEYS:
            metadata[key] = value

    normalized: dict[str, Any] = {
        "schema_version": AGENT_STATUS_SCHEMA_VERSION,
        "status": status,
        "message": str(payload.get("message") or payload.get("summary") or ""),
        "source": str(payload.get("source") or payload.get("source_tool") or "unknown"),
    }

    for key in ("session_id", "step_id", "blocked_reason", "next_action"):
        value = payload.get(key)
        if value not in (None, ""):
            normalized[key] = str(value)

    progress = _normalize_progress(payload.get("progress"))
    if progress is not None:
        normalized["progress"] = progress

    if metadata:
        normalized["metadata"] = metadata

    return normalized


def format_agent_status_marker(payload: dict[str, Any]) -> str:
    """Return a one-line structured status marker for terminal integrations."""

    normalized = normalize_agent_status(payload)
    return f"{AGENT_STATUS_MARKER} {json.dumps(normalized, sort_keys=True)}"


def parse_agent_status_markers(text: str, *, strict: bool = False) -> list[dict[str, Any]]:
    """Parse structured status markers from terminal text.

    Only lines that begin with ``::bashgym-agent-status`` are inspected. This is
    intentionally not a broad terminal-log parser.
    """

    statuses: list[dict[str, Any]] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped.startswith(AGENT_STATUS_MARKER):
            continue
        raw_payload = stripped[len(AGENT_STATUS_MARKER) :].strip()
        if raw_payload.startswith(":"):
            raw_payload = raw_payload[1:].strip()
        try:
            decoded = json.loads(raw_payload)
            statuses.append(normalize_agent_status(decoded))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            if strict:
                raise ValueError(f"invalid agent status marker: {line}") from exc
    return statuses


def agent_status_to_event(
    payload: dict[str, Any],
    *,
    trace_id: str,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    source_tool: str | None = None,
    timestamp: str | datetime | None = None,
) -> TraceEvent:
    """Convert an agent-status payload into a validated contextual TraceEvent."""

    normalized = normalize_agent_status(payload)
    event_timestamp = timestamp or datetime.now(timezone.utc)
    event_span = span_id or uuid.uuid4().hex[:12]
    event_source = source_tool or normalized["source"]
    body = {
        "operation_type": "agent_status",
        "target": normalized.get("session_id", trace_id),
        "details": normalized,
    }
    event = TraceEvent(
        surface_type=SurfaceType.CONTEXTUAL,
        trace_id=trace_id,
        span_id=event_span,
        parent_span_id=parent_span_id,
        timestamp=event_timestamp,
        source_tool=event_source,
        body=body,
    )
    return validate_event(event)


def scrub_trace_replay(
    trace_or_events: Any,
    *,
    max_output_chars: int = 2000,
    redact_patterns: Iterable[tuple[str, re.Pattern[str], _Replacement]] | None = None,
) -> dict[str, Any]:
    """Redact secrets and summarize long outputs in a trace/replay object."""

    if max_output_chars < 1:
        raise ValueError("max_output_chars must be positive")

    stats = _ScrubStats()
    patterns = tuple(redact_patterns or _DEFAULT_REDACT_PATTERNS)
    plain = _to_plain(trace_or_events)
    scrubbed = _scrub_value(
        plain,
        path=(),
        stats=stats,
        max_output_chars=max_output_chars,
        patterns=patterns,
    )
    return {
        "schema_version": REPLAY_SCRUB_SCHEMA_VERSION,
        "ok": True,
        "scrubbed": scrubbed,
        "stats": stats.to_dict(max_output_chars),
    }


def scrub_trace_replay_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    max_output_chars: int = 2000,
) -> dict[str, Any]:
    """Scrub a JSON/JSONL trace replay file and optionally write the scrubbed payload."""

    source = Path(input_path)
    payload, input_format = _read_json_or_jsonl(source)
    result = scrub_trace_replay(payload, max_output_chars=max_output_chars)
    destination = Path(output_path) if output_path else None
    if destination:
        _write_scrubbed(destination, result["scrubbed"], input_format=input_format)

    stats = {
        **result["stats"],
        "input_format": input_format,
        "records": len(payload) if isinstance(payload, list) else 1,
    }
    return {
        "schema_version": REPLAY_SCRUB_SCHEMA_VERSION,
        "ok": True,
        "input_path": str(source.resolve()),
        "output_path": str(destination.resolve()) if destination else None,
        "stats": stats,
    }


def _normalize_progress(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        progress = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("progress must be numeric") from exc
    if progress > 1.0:
        progress = progress / 100.0
    if progress < 0.0 or progress > 1.0:
        raise ValueError("progress must be between 0 and 1, or 0 and 100")
    return progress


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _scrub_value(
    value: Any,
    *,
    path: tuple[str, ...],
    stats: _ScrubStats,
    max_output_chars: int,
    patterns: tuple[tuple[str, re.Pattern[str], _Replacement], ...],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_value(
                item,
                path=(*path, str(key)),
                stats=stats,
                max_output_chars=max_output_chars,
                patterns=patterns,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _scrub_value(
                item,
                path=path,
                stats=stats,
                max_output_chars=max_output_chars,
                patterns=patterns,
            )
            for item in value
        ]
    if isinstance(value, str):
        return _scrub_text(
            value,
            path=path,
            stats=stats,
            max_output_chars=max_output_chars,
            patterns=patterns,
        )
    return value


def _scrub_text(
    text: str,
    *,
    path: tuple[str, ...],
    stats: _ScrubStats,
    max_output_chars: int,
    patterns: tuple[tuple[str, re.Pattern[str], _Replacement], ...],
) -> str:
    field_name = path[-1].lower() if path else ""
    field_path = ".".join(path) if path else "<root>"
    scrubbed = text
    for _name, pattern, replacement in patterns:
        scrubbed, count = pattern.subn(replacement, scrubbed)
        if count:
            stats.redactions += count
            stats.redacted_fields.add(field_path)

    if field_name in _OUTPUT_FIELD_NAMES and len(scrubbed) > max_output_chars:
        omitted = len(scrubbed) - max_output_chars
        scrubbed = f"{scrubbed[:max_output_chars]}\n[truncated {omitted} chars by BashGym replay scrubber]"
        stats.truncations += 1
        stats.truncated_fields.add(field_path)

    return scrubbed


def _read_json_or_jsonl(path: Path) -> tuple[Any, str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return records, "jsonl"
    return json.loads(text), "json"


def _write_scrubbed(path: Path, scrubbed: Any, *, input_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl" or input_format == "jsonl":
        records = scrubbed if isinstance(scrubbed, list) else [scrubbed]
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
        return
    path.write_text(json.dumps(scrubbed, indent=2, sort_keys=True), encoding="utf-8")
