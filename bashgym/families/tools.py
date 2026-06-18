"""Tool-call normalization + per-family format validation.

Coerces tool-call arguments to dicts (required by the Gemma/Qwen chat templates) and
flags tool calls that won't render under a family's ``tool_call_format``. This is the
canonical home for the sanitization that is otherwise copy-pasted across converters,
importers, and training scripts.
"""

from __future__ import annotations

import json


def sanitize_tool_call(tc: dict) -> dict:
    """Return a copy of ``tc`` with ``function.arguments`` coerced from a JSON string to a dict."""
    if not isinstance(tc, dict):
        return tc
    tc = dict(tc)
    fn = tc.get("function")
    if isinstance(fn, dict):
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {"raw": args}
        tc["function"] = fn
    return tc


def sanitize_message_tool_calls(messages: list[dict]) -> list[dict]:
    """Sanitize ``tool_calls`` across a message list (returns new message dicts)."""
    out = []
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("tool_calls"), list):
            m = dict(m)
            m["tool_calls"] = [sanitize_tool_call(tc) for tc in m["tool_calls"]]
        out.append(m)
    return out


def validate_tool_call(tc: dict, tool_call_format: str) -> list[str]:
    """Return issues that would break rendering of ``tc`` under ``tool_call_format``.

    Empty list = renders fine. Catches the common breakers: missing name, string
    (un-parsed) arguments, and — for delimiter/xml formats where the template wraps
    string literals — non-JSON-serializable arguments.
    """
    issues: list[str] = []
    fn = tc.get("function") if isinstance(tc, dict) else None
    if not isinstance(fn, dict) or not fn.get("name"):
        issues.append("missing function.name")
        return issues
    args = fn.get("arguments")
    if isinstance(args, str):
        issues.append("arguments is a string (must be a dict for chat-template rendering)")
    elif args is not None and not isinstance(args, dict):
        issues.append("arguments must be a dict or None")
    if tool_call_format in ("gemma4_delimited", "qwen_xml") and isinstance(args, dict):
        try:
            json.dumps(args)
        except (TypeError, ValueError):
            issues.append("arguments not JSON-serializable")
    return issues
