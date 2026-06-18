"""Validate NeMo-format JSONL training examples for chat-template compatibility.

Catches the bug classes that silently break Gemma/Qwen chat templates at
training time: tool_calls arguments encoded as JSON strings instead of dicts,
missing/unknown roles, and structurally malformed examples.
"""

from __future__ import annotations

import json
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant", "tool"}
MAX_CONTENT_PREVIEW = 2000


def _validate_example(example: dict) -> list[str]:
    """Return human-readable warnings for one example (empty list = clean)."""
    warnings: list[str] = []
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["Example has no 'messages' list"]

    has_assistant = False
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            warnings.append(f"Message {i} is not an object")
            continue
        role = msg.get("role")
        if role not in VALID_ROLES:
            warnings.append(f"Message {i} has unknown role: {role!r}")
        if role == "assistant":
            has_assistant = True
        content = msg.get("content")
        if content is None and not msg.get("tool_calls"):
            warnings.append(f"Message {i} has neither content nor tool_calls")
        for j, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                warnings.append(f"Message {i} tool_call {j} is not an object")
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                warnings.append(
                    f"Message {i} tool_call {j}: arguments is a JSON string, not a dict "
                    "— breaks Gemma/Qwen chat templates"
                )

    if not has_assistant:
        warnings.append("Example has no assistant message — nothing to learn from")
    return warnings


def inspect_dataset(path: str | Path, offset: int = 0, limit: int = 20) -> dict:
    """Inspect a slice of a JSONL dataset.

    Returns total example count, the requested slice with per-example message
    previews and validation warnings, and a count of flagged examples in the slice.
    """
    path = Path(path)
    examples: list[dict] = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = total
            total += 1
            if idx < offset or len(examples) >= limit:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                examples.append({"index": idx, "messages": [], "warnings": ["Invalid JSON line"]})
                continue
            messages = example.get("messages") if isinstance(example, dict) else None
            preview = []
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        truncated = isinstance(content, str) and len(content) > MAX_CONTENT_PREVIEW
                        preview.append(
                            {
                                "role": msg.get("role"),
                                "content": (
                                    content[:MAX_CONTENT_PREVIEW]
                                    if isinstance(content, str)
                                    else content
                                ),
                                "tool_calls": msg.get("tool_calls"),
                                "truncated": truncated,
                            }
                        )
            examples.append(
                {
                    "index": idx,
                    "messages": preview,
                    "warnings": _validate_example(example if isinstance(example, dict) else {}),
                }
            )
    warned = sum(1 for e in examples if e["warnings"])
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "examples": examples,
        "with_warnings_in_slice": warned,
    }
