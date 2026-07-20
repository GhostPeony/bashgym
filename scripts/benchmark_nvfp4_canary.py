#!/usr/bin/env python3
"""Run a small, reproducible OpenAI-compatible Gemma canary benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]

CASES = [
    {
        "id": "exact_text",
        "prompt": "Reply with exactly: NVFP4_READY",
        "expected_text": "NVFP4_READY",
    },
    {
        "id": "short_reasoning",
        "prompt": "What is 17 multiplied by 19? Reply with the number only.",
        "expected_text": "323",
    },
    {
        "id": "shell_tool",
        "prompt": (
            "List Python files in the current working directory. Use the available "
            "tool and do not invent results."
        ),
        "expected_tool": "run_command",
    },
    {
        "id": "file_tool",
        "prompt": "Read pyproject.toml using the available tool. Do not invent its contents.",
        "expected_tool": "read_file",
    },
]


def evaluate_response(case: dict[str, str], response: dict[str, Any]) -> bool:
    message = (response.get("choices") or [{}])[0].get("message") or {}
    if expected := case.get("expected_text"):
        return str(message.get("content") or "").strip() == expected
    if expected_tool := case.get("expected_tool"):
        calls = message.get("tool_calls") or []
        return any((call.get("function") or {}).get("name") == expected_tool for call in calls)
    return False


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [sample for sample in samples if "error" not in sample]
    latencies = [float(sample["elapsed_seconds"]) for sample in successful]
    throughputs = [
        float(sample["completion_tokens"]) / float(sample["elapsed_seconds"])
        for sample in successful
        if sample.get("completion_tokens") and sample.get("elapsed_seconds")
    ]
    return {
        "requests": len(samples),
        "successful_requests": len(successful),
        "passed_requests": sum(bool(sample.get("passed")) for sample in successful),
        "pass_rate": (
            sum(bool(sample.get("passed")) for sample in successful) / len(successful)
            if successful
            else 0.0
        ),
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "median_completion_tokens_per_second": (
            statistics.median(throughputs) if throughputs else None
        ),
    }


def call_model(endpoint: str, model: str, case: dict[str, str], timeout: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "temperature": 0,
        "max_tokens": 128,
    }
    if case.get("expected_tool"):
        payload.update({"tools": TOOLS, "tool_choice": "auto"})

    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = json.loads(response.read())
        elapsed = time.perf_counter() - started
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "model": model,
            "case_id": case["id"],
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(exc),
        }

    message = (body.get("choices") or [{}])[0].get("message") or {}
    return {
        "model": model,
        "case_id": case["id"],
        "elapsed_seconds": elapsed,
        "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
        "passed": evaluate_response(case, body),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8892")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        for model in args.model:
            for case in CASES:
                sample = call_model(args.endpoint, model, case, args.timeout)
                sample["repetition"] = repetition + 1
                samples.append(sample)
                print(
                    f"{model} {case['id']} pass={sample.get('passed', False)} "
                    f"elapsed={sample['elapsed_seconds']:.3f}s"
                )

    report = {
        "schema_version": "bashgym.nvfp4-canary-benchmark.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": args.endpoint,
        "models": args.model,
        "repetitions": args.repetitions,
        "cases": CASES,
        "summary": {
            model: summarize([sample for sample in samples if sample["model"] == model])
            for model in args.model
        },
        "samples": samples,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
