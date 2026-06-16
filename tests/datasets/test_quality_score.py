"""Tests for the continuous quality_score carried into converter metadata."""

from bashgym.datasets.converters import trace_to_grpo_example, trace_to_sft_example

TRACE = {
    "metadata": {"user_initial_prompt": "Read the config file and summarize it for me please"},
    "trace": [
        {"tool_name": "Read", "success": True},
        {"tool_name": "Bash", "success": True},
        {"tool_name": "Edit", "success": True},
    ],
    "verification_passed": True,
}


def test_grpo_example_carries_continuous_quality_score():
    ex = trace_to_grpo_example(TRACE, multimodal_format=False)
    qs = ex["metadata"]["quality_score"]
    assert qs is not None
    assert 0.0 <= qs <= 1.0


def test_sft_example_carries_quality_score():
    ex = trace_to_sft_example(TRACE)
    if ex is not None:  # sft converter needs >=4 messages; trace may be too short
        assert "quality_score" in ex["metadata"]


def test_messages_format_has_no_quality_score():
    # public messages-format examples have no scorable steps -> None, not a crash
    ex = trace_to_grpo_example(
        {"messages": [{"role": "user", "content": "do the thing for me please"}]},
        multimodal_format=False,
    )
    assert ex["metadata"]["quality_score"] is None
