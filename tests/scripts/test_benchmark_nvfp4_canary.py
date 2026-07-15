import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_nvfp4_canary.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_nvfp4_canary", _PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_evaluate_response_handles_exact_text_and_tool_calls():
    assert benchmark.evaluate_response(
        {"expected_text": "323"}, {"choices": [{"message": {"content": "323"}}]}
    )
    assert benchmark.evaluate_response(
        {"expected_tool": "run_command"},
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"name": "run_command", "arguments": "{}"}}]
                    }
                }
            ]
        },
    )


def test_summarize_reports_medians_and_pass_rate():
    summary = benchmark.summarize(
        [
            {"elapsed_seconds": 1.0, "completion_tokens": 10, "passed": True},
            {"elapsed_seconds": 2.0, "completion_tokens": 10, "passed": False},
        ]
    )

    assert summary["pass_rate"] == pytest.approx(0.5)
    assert summary["median_latency_seconds"] == pytest.approx(1.5)
    assert summary["median_completion_tokens_per_second"] == pytest.approx(7.5)
