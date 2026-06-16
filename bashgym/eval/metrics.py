"""Step-level tool-call metrics for held-out trace eval.

Compares a model's predicted tool call against the gold call with structured
signals — tool-name exact match and per-argument F1 (arguments parsed to dicts,
exact for non-strings, normalized for strings) — instead of the brittle substring
matching the old eval scripts used.
"""

from __future__ import annotations

import json


def _coerce_args(args) -> dict:
    """Tool-call arguments may be a dict or a JSON string; return a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {"_value": args}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def _tool_name(call: dict) -> str:
    """Extract the tool name from an OpenAI-style or flat tool call."""
    if not isinstance(call, dict):
        return ""
    fn = call.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(call.get("name") or "")


def _tool_args(call: dict) -> dict:
    if not isinstance(call, dict):
        return {}
    fn = call.get("function")
    if isinstance(fn, dict):
        return _coerce_args(fn.get("arguments"))
    return _coerce_args(call.get("arguments"))


def _norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def tool_name_match(predicted: dict, gold: dict) -> bool:
    """True if both name the same (non-empty) tool."""
    gold_name = _tool_name(gold)
    return gold_name != "" and _tool_name(predicted) == gold_name


def tool_arg_f1(predicted: dict, gold: dict) -> float:
    """Per-argument F1 between predicted and gold tool-call arguments.

    A key is a true positive when it is present in both with matching (normalized)
    values. Returns 1.0 when both have no arguments, 0.0 when exactly one does.
    """
    p, g = _tool_args(predicted), _tool_args(gold)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = sum(1 for k, v in g.items() if k in p and _norm(p[k]) == _norm(v))
    precision = tp / len(p)
    recall = tp / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_tool_call(predicted: dict, gold: dict) -> dict:
    """Combined step-level score: name match, arg F1, and full exact match."""
    name = tool_name_match(predicted, gold)
    arg_f1 = tool_arg_f1(predicted, gold)
    return {
        "name_match": name,
        "arg_f1": arg_f1,
        "exact_match": bool(name and arg_f1 == 1.0),
    }
