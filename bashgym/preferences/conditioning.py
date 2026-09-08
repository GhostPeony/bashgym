"""Bindings for independently captured responses to the same immutable task."""

import hashlib
import json
import re
from typing import Any


def conditioning_binding(prompt: str, context: Any) -> str | None:
    """Hash full prompt plus task, repository snapshot and available tool schema.

    Context is supplied by capture/replay, never inferred from response similarity.
    Missing historical context cannot establish preference eligibility.
    """
    if not prompt or not isinstance(context, dict):
        return None
    if not isinstance(context.get("task_id"), str) or not context["task_id"].strip():
        return None
    for key in ("snapshot_digest", "tools_digest"):
        if not isinstance(context.get(key), str) or not re.fullmatch("[0-9a-f]{64}", context[key]):
            return None
    payload = {"prompt": prompt, "context": context}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
