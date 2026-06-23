"""Concrete predictors for the held-out eval runner — call a served model and
return its predicted tool call.

``heldout.py`` is model-agnostic: it takes injected predictors. These adapt an
OpenAI-compatible chat endpoint (vLLM ``local-completions`` on the Spark, or
Ollama's ``/v1``) into the sync ``Predictor`` the runner expects. All network
lives behind an injected ``complete(messages) -> response`` callable, so the
prompt-building and tool-call parsing here stay unit-testable with a stub.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

Predictor = Callable[[dict], dict]
Completer = Callable[[list[dict]], Any]  # messages -> chat-completion response


def build_prompt_messages(example: dict) -> list[dict]:
    """Messages to send the model: everything up to (not including) the first gold
    assistant tool call — i.e. ask the model to produce that next call."""
    out: list[dict] = []
    for m in example.get("messages", []) or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            break  # stop before the gold answer we're predicting
        out.append({"role": m.get("role", "user"), "content": m.get("content", "") or ""})
    return out


def _extract_message(response: Any) -> dict:
    """Pull the assistant message out of an OpenAI- or Ollama-shaped response."""
    if isinstance(response, str):
        return {"content": response}
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            return msg
    if isinstance(response.get("message"), dict):  # Ollama-style {"message": {...}}
        return response["message"]
    return response  # assume it is already a message dict


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _first_json_object(text: str) -> Any | None:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_tool_call_from_text(text: str) -> dict:
    """Recover a tool call from free text — a fenced ```json block or an inline
    JSON object carrying ``name``/``function``."""
    if not text:
        return {}
    for cand in [*_FENCE_RE.findall(text), text]:
        obj = _first_json_object(cand)
        if isinstance(obj, dict) and ("name" in obj or "function" in obj):
            return obj
    return {}


def parse_tool_call(response: Any) -> dict:
    """Extract the predicted tool call from a chat-completion response.

    Prefers native ``message.tool_calls`` (OpenAI/vLLM/Ollama tool calling); falls
    back to parsing a JSON tool call out of the message text (fenced or inline).
    Returns ``{}`` when nothing is recoverable — scores as a miss, not a crash.
    """
    msg = _extract_message(response)
    tcs = msg.get("tool_calls") if isinstance(msg, dict) else None
    if isinstance(tcs, list) and tcs and isinstance(tcs[0], dict):
        return tcs[0]
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    return _parse_tool_call_from_text(content or "")


def endpoint_predictor(complete: Completer) -> Predictor:
    """Build a ``Predictor`` from a ``complete(messages) -> response`` callable.

    ``complete`` is the network seam — wrap vLLM/Ollama's chat endpoint with it.
    Errors from ``complete`` are swallowed to an empty call so one bad request
    scores as a miss instead of aborting the whole eval sweep.
    """

    def predict(example: dict) -> dict:
        messages = build_prompt_messages(example)
        try:
            resp = complete(messages)
        except (
            Exception
        ):  # noqa: BLE001 - a failed request must score as a miss, not crash the sweep
            return {}
        return parse_tool_call(resp)

    return predict


def openai_complete(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 60.0,
    tools: list | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> Completer:
    """A ``complete`` callable hitting an OpenAI-compatible chat endpoint.

    Point ``base_url`` at the vLLM ``local-completions`` server on the Spark
    (``http://192.168.50.173:8100/v1``) or Ollama's ``/v1``. ``httpx`` is imported
    lazily so this module stays import-light; the runner only touches it when
    actually serving a model.
    """
    import httpx

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def complete(messages: list[dict]) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    return complete
