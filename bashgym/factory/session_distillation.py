"""Session Distillation artifact builder and validator.

Session Distillation turns a student's own trace into a targeted learning
record: original context, hinted context, and the same target action tokens.
The trainer can then re-score the target span under both contexts and apply a
masked KL/CE loss.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bashgym.factory.decision_extractor import Decision, DecisionExtractor

SESSION_DISTILLATION_HINT_TAG = "[Session Distillation Hint]"

VALID_TARGET_TYPES = {
    "tool_call",
    "command",
    "edit",
    "test_decision",
    "recovery_decision",
    "summary",
}

# Only target_span_only is implemented in the trainer. "target_tokens_only" was
# a placeholder that every consumer rejected; drop it until it is implemented.
VALID_MASK_POLICIES = {"target_span_only"}

ERROR_RE = re.compile(
    "|".join(
        [
            r"(?:error|ERROR)[\s:]",
            r"traceback",
            r"failed|FAILED",
            r"No such file",
            r"not found",
            r"command not found",
            r"Permission denied",
            r"syntax error|SyntaxError",
            r"ModuleNotFoundError|ImportError",
            r"FileNotFoundError|OSError",
            r"fatal:",
            r"ENOENT",
        ]
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SessionDistillationHint:
    """Reader output for one local mistake span."""

    decision_id: str
    problem_step_index: int
    target_span: dict[str, int]
    hint_text: str
    mistake_type: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionDistillationRecord:
    """JSONL record consumed by Session Distillation trainers."""

    record_id: str
    trace_id: str
    session_id: str
    decision_id: str
    step_index: int
    original_context: str
    hinted_context: str
    hint_text: str
    target_text: str
    target_type: str
    target_span: dict[str, int]
    loss_mask: dict[str, Any]
    reader_model: str
    reader_confidence: float
    verifier_outcome: str
    quality_score: float
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_session_distillation_records(records: list[dict[str, Any]]) -> list[str]:
    """Validate a list of Session Distillation JSONL payloads."""

    errors: list[str] = []
    for index, record in enumerate(records):
        for error in validate_session_distillation_record(record):
            errors.append(f"records[{index}].{error}")
    return errors


def validate_session_distillation_record(record: dict[str, Any]) -> list[str]:
    """Return specific validation errors for one Session Distillation record."""

    required_fields = [
        "record_id",
        "trace_id",
        "session_id",
        "decision_id",
        "step_index",
        "original_context",
        "hinted_context",
        "hint_text",
        "target_text",
        "target_type",
        "target_span",
        "loss_mask",
        "reader_model",
        "reader_confidence",
        "verifier_outcome",
        "quality_score",
        "source_metadata",
    ]
    errors: list[str] = []
    for field_name in required_fields:
        if field_name not in record:
            errors.append(f"{field_name} is required")

    if errors:
        return errors

    for field_name in (
        "record_id",
        "trace_id",
        "session_id",
        "decision_id",
        "original_context",
        "hinted_context",
        "hint_text",
        "target_text",
        "reader_model",
        "verifier_outcome",
    ):
        if not isinstance(record.get(field_name), str) or not record[field_name].strip():
            errors.append(f"{field_name} must be a non-empty string")

    if not isinstance(record.get("step_index"), int) or record["step_index"] < 0:
        errors.append("step_index must be a non-negative integer")

    target_type = record.get("target_type")
    if target_type not in VALID_TARGET_TYPES:
        errors.append(f"target_type must be one of {sorted(VALID_TARGET_TYPES)}")

    target_span = record.get("target_span")
    if not isinstance(target_span, dict):
        errors.append("target_span must be an object")
    else:
        start = target_span.get("start")
        end = target_span.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append("target_span.start and target_span.end must be integers")
        elif start < 0 or end <= start:
            errors.append("target_span must satisfy 0 <= start < end")
        elif isinstance(record.get("target_text"), str) and end > len(record["target_text"]):
            errors.append("target_span.end cannot exceed target_text length")
        elif isinstance(record.get("target_text"), str) and (
            start != 0 or end != len(record["target_text"])
        ):
            # The trainer masks the entire target_text; sub-span masking is not
            # implemented, so a partial span would be silently trained full.
            errors.append(
                "target_span must cover the entire target_text (sub-span masking unsupported)"
            )

    loss_mask = record.get("loss_mask")
    if not isinstance(loss_mask, dict):
        errors.append("loss_mask must be an object")
    else:
        policy = loss_mask.get("policy")
        if policy not in VALID_MASK_POLICIES:
            errors.append(f"loss_mask.policy must be one of {sorted(VALID_MASK_POLICIES)}")
        if "target_span" not in loss_mask:
            errors.append("loss_mask.target_span is required")

    for score_field in ("reader_confidence", "quality_score"):
        value = record.get(score_field)
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            errors.append(f"{score_field} must be between 0 and 1")

    if not isinstance(record.get("source_metadata"), dict):
        errors.append("source_metadata must be an object")

    if (
        isinstance(record.get("hinted_context"), str)
        and isinstance(record.get("original_context"), str)
        and record["hinted_context"] == record["original_context"]
    ):
        errors.append("hinted_context must differ from original_context")

    if (
        isinstance(record.get("hinted_context"), str)
        and SESSION_DISTILLATION_HINT_TAG not in record["hinted_context"]
    ):
        errors.append("hinted_context must include the Session Distillation hint tag")

    if (
        isinstance(record.get("original_context"), str)
        and SESSION_DISTILLATION_HINT_TAG in record["original_context"]
    ):
        errors.append("original_context must not include the Session Distillation hint tag")

    return errors


class HeuristicSessionDistillationReader:
    """Deterministic reader for early Session Distillation records."""

    reader_model = "heuristic-session-distillation-reader-v1"

    def analyze(
        self,
        steps: list[dict[str, Any]],
        decisions: list[Decision] | None = None,
    ) -> list[SessionDistillationHint]:
        decisions_by_step = {decision.step_index: decision for decision in decisions or []}
        hints: list[SessionDistillationHint] = []

        for index, step in enumerate(steps):
            decision = decisions_by_step.get(index)
            failed = _step_failed(step) or (decision is not None and decision.outcome == "FAILURE")
            if not failed:
                continue

            target_text = _target_text(step, decision)
            if not target_text:
                continue

            mistake_type = _mistake_type(step)
            confidence = 0.78 if decision and decision.outcome == "FAILURE" else 0.68
            hint_text = _hint_for_mistake(mistake_type, step, decision)
            reason = _failure_reason(step, decision)
            decision_id = _decision_id(index, decision)

            hints.append(
                SessionDistillationHint(
                    decision_id=decision_id,
                    problem_step_index=index,
                    target_span={"start": 0, "end": len(target_text)},
                    hint_text=hint_text,
                    mistake_type=mistake_type,
                    confidence=confidence,
                    reason=reason,
                )
            )

        return hints


def build_session_distillation_records(
    steps: list[dict[str, Any]],
    *,
    decisions: list[Decision] | None = None,
    task_prompt: str = "",
    trace_id: str = "",
    session_id: str = "",
    source_metadata: dict[str, Any] | None = None,
    min_confidence: float = 0.6,
) -> list[SessionDistillationRecord]:
    """Build Session Distillation records from normalized trace steps."""

    if not steps:
        return []

    if decisions is None:
        decisions = DecisionExtractor().extract(steps)

    reader = HeuristicSessionDistillationReader()
    hints = reader.analyze(steps, decisions)
    decisions_by_step = {decision.step_index: decision for decision in decisions}
    records: list[SessionDistillationRecord] = []

    for hint in hints:
        if hint.confidence < min_confidence:
            continue
        if hint.problem_step_index < 0 or hint.problem_step_index >= len(steps):
            continue

        step = steps[hint.problem_step_index]
        decision = decisions_by_step.get(hint.problem_step_index)
        target_text = _target_text(step, decision)
        if not target_text:
            continue

        target_span = {"start": 0, "end": len(target_text)}
        original_context = _build_original_context(
            steps,
            hint.problem_step_index,
            task_prompt=task_prompt,
            decision=decision,
        )
        hinted_context = inject_session_distillation_hint(original_context, hint.hint_text)
        verifier_outcome = "failed" if _step_failed(step) else "decision_failure"
        metadata = dict(source_metadata or {})
        metadata.update(
            {
                "mistake_type": hint.mistake_type,
                "reader_reason": hint.reason,
                "step_tool": step.get("tool", step.get("tool_name", "")),
            }
        )

        record_id = _record_id(
            trace_id=trace_id,
            session_id=session_id,
            decision_id=hint.decision_id,
            step_index=hint.problem_step_index,
            target_text=target_text,
            hint_text=hint.hint_text,
        )

        records.append(
            SessionDistillationRecord(
                record_id=record_id,
                trace_id=trace_id or "unknown-trace",
                session_id=session_id or "unknown-session",
                decision_id=hint.decision_id,
                step_index=hint.problem_step_index,
                original_context=original_context,
                hinted_context=hinted_context,
                hint_text=hint.hint_text,
                target_text=target_text,
                target_type=_target_type(step, decision),
                target_span=target_span,
                loss_mask={"policy": "target_span_only", "target_span": target_span},
                reader_model=reader.reader_model,
                reader_confidence=hint.confidence,
                verifier_outcome=verifier_outcome,
                quality_score=hint.confidence,
                source_metadata=metadata,
            )
        )

    return records


def build_session_distillation_records_from_traces(
    traces_dir: str | Path,
    *,
    min_confidence: float = 0.6,
    source_split: str = "",
    limit: int | None = None,
) -> list[SessionDistillationRecord]:
    """Build Session Distillation records from a directory of trace files.

    Reads each ``*.json`` session file (top-level ``trace`` step list, or a bare
    list of steps), runs the heuristic reader per session, and collects records
    for failed/recovery steps. Traces carry no ``trace_id`` on disk, so the file
    stem is used. Clean sessions yield nothing.
    """

    directory = Path(traces_dir)
    records: list[SessionDistillationRecord] = []

    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

        if isinstance(data, list):
            steps, metadata, session_id, repo = data, {}, "", None
        elif isinstance(data, dict):
            steps = data.get("trace", data.get("steps", []))
            metadata = data.get("metadata") or {}
            session_id = str(data.get("session_id", ""))
            repo = data.get("primary_repo")
        else:
            continue

        if not isinstance(steps, list) or not steps:
            continue

        source_metadata: dict[str, Any] = {"trace_file": path.name}
        if source_split:
            source_metadata["split"] = source_split
        if isinstance(repo, dict) and repo.get("name"):
            source_metadata["repo"] = repo["name"]

        records.extend(
            build_session_distillation_records(
                steps,
                task_prompt=str(metadata.get("user_initial_prompt", "")),
                trace_id=path.stem,
                session_id=session_id or path.stem,
                source_metadata=source_metadata,
                min_confidence=min_confidence,
            )
        )

        if limit is not None and len(records) >= limit:
            return records[:limit]

    return records


def inject_session_distillation_hint(original_context: str, hint_text: str) -> str:
    """Insert a structured hint immediately before the target action context."""

    return f"{original_context.rstrip()}\n\n{SESSION_DISTILLATION_HINT_TAG}\n{hint_text.strip()}\n"


def save_session_distillation_records(
    records: list[SessionDistillationRecord],
    output_path: str | Path,
) -> Path:
    """Write Session Distillation records as JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def _build_original_context(
    steps: list[dict[str, Any]],
    step_index: int,
    *,
    task_prompt: str,
    decision: Decision | None,
) -> str:
    parts: list[str] = []
    if task_prompt:
        parts.append(f"Task:\n{task_prompt.strip()}")
    if decision and decision.intent:
        parts.append(f"Current intent:\n{decision.intent.strip()}")

    start = max(0, step_index - 3)
    previous_steps = steps[start:step_index]
    if previous_steps:
        parts.append("Recent trace:")
        for idx, step in enumerate(previous_steps, start=start):
            command = _target_text(step, None)
            output = str(step.get("output", step.get("result", ""))).strip()
            if len(output) > 500:
                output = output[:500] + "\n... (truncated)"
            parts.append(f"- Step {idx}: {command}")
            if output:
                parts.append(f"  Output: {output}")

    parts.append("Next action:")
    return "\n".join(parts).strip()


def _record_id(
    *,
    trace_id: str,
    session_id: str,
    decision_id: str,
    step_index: int,
    target_text: str,
    hint_text: str,
) -> str:
    raw = "|".join([trace_id, session_id, decision_id, str(step_index), target_text, hint_text])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _decision_id(step_index: int, decision: Decision | None) -> str:
    if decision is None:
        return f"step-{step_index}"
    return f"decision-{decision.step_index}-{hashlib.sha1(decision.chosen.encode('utf-8')).hexdigest()[:8]}"


def _step_failed(step: dict[str, Any]) -> bool:
    if step.get("success") is False:
        return True
    exit_code = step.get("exit_code")
    if exit_code is not None and exit_code != 0:
        return True
    # A step that explicitly reports success is not a mistake, even when its
    # output merely mentions failure keywords (passing pytest xfail summaries,
    # docs describing "command not found", grepped log lines containing
    # "ERROR:"). Only fall back to output scanning when no success signal exists.
    if step.get("success") is True or exit_code == 0:
        return False
    output = str(step.get("output", step.get("result", "")))
    return bool(ERROR_RE.search(output))


def _target_text(step: dict[str, Any], decision: Decision | None) -> str:
    if decision and decision.chosen:
        return decision.chosen.strip()
    for field_name in ("command", "input", "content", "text"):
        value = step.get(field_name)
        if value:
            return str(value).strip()
    return ""


def _target_type(step: dict[str, Any], decision: Decision | None) -> str:
    if decision and decision.outcome == "FAILURE" and decision.tool_used:
        return "recovery_decision"
    tool = str(step.get("tool", step.get("tool_name", ""))).lower()
    command = str(step.get("command", step.get("input", ""))).lower()
    if tool == "bash":
        if any(token in command for token in ("pytest", "npm test", "ruff", "black --check")):
            return "test_decision"
        return "command"
    if tool in {"edit", "write", "multiedit"}:
        return "edit"
    return "tool_call"


def _mistake_type(step: dict[str, Any]) -> str:
    output = str(step.get("output", step.get("result", "")))
    if re.search(r"command not found|not recognized", output, re.IGNORECASE):
        return "missing_command"
    if re.search(r"No such file|FileNotFoundError|ENOENT", output, re.IGNORECASE):
        return "missing_path"
    if re.search(r"ModuleNotFoundError|ImportError", output, re.IGNORECASE):
        return "missing_dependency"
    if re.search(r"Permission denied", output, re.IGNORECASE):
        return "permission_error"
    if re.search(r"SyntaxError|syntax error", output, re.IGNORECASE):
        return "syntax_error"
    return "failed_action"


def _failure_reason(step: dict[str, Any], decision: Decision | None) -> str:
    if decision and decision.reasoning:
        return decision.reasoning[:300]
    output = str(step.get("output", step.get("result", ""))).strip()
    if output:
        return output[:300]
    return "The step failed according to trace success or exit status."


def _hint_for_mistake(
    mistake_type: str,
    step: dict[str, Any],
    decision: Decision | None,
) -> str:
    command = _target_text(step, decision)
    if mistake_type == "missing_command":
        return "Check whether the command or package is installed before repeating this action."
    if mistake_type == "missing_path":
        return "Verify the file path from the current working directory before using it."
    if mistake_type == "missing_dependency":
        return "Confirm the dependency is installed in the active environment before importing or running it."
    if mistake_type == "permission_error":
        return "Use a writable path or adjust permissions before running this command."
    if mistake_type == "syntax_error":
        return "Inspect the syntax near the reported location before rerunning the command."
    if command:
        return "Use the verifier output to revise this action before trying the same target again."
    return "Review the failed step and choose the smallest corrective action."
