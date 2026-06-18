"""Tests for trace->training-format converters, covering BOTH input shapes:
the bashgym-trace format (`trace` steps + `metadata`) and the OpenAI/NeMo
`messages` format that the cascade filter operates on.
"""

from bashgym.datasets.converters import (
    _extract_user_prompt,
    _summarize_tool_usage,
    trace_to_grpo_example,
)

READ_TOOL_MESSAGES = {
    "messages": [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}],
        }
    ]
}


class TestMessagesFormat:
    def test_tool_calls_summarized(self):
        assert "Read" in _summarize_tool_usage(READ_TOOL_MESSAGES)

    def test_user_prompt_extracted_from_str_content(self):
        ex = {"messages": [{"role": "user", "content": "Please read the config and summarize it"}]}
        assert "config" in _extract_user_prompt(ex)

    def test_user_prompt_extracted_from_list_content(self):
        ex = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Read the config and summarize"}],
                }
            ]
        }
        assert "config" in _extract_user_prompt(ex)

    def test_tool_only_example_converts_to_grpo(self):
        # The cascade-test regression: an assistant-only tool example must still
        # convert (via the tool-usage summary fallback) instead of yielding None.
        g = trace_to_grpo_example(READ_TOOL_MESSAGES, multimodal_format=False)
        assert g is not None
        assert g.get("prompt")


class TestTraceFormatStillWorks:
    def test_metadata_prompt_trace_format(self):
        ex = {
            "metadata": {"user_initial_prompt": "Read the config file and summarize it"},
            "trace": [{"tool_name": "Read"}],
        }
        g = trace_to_grpo_example(ex, multimodal_format=False)
        assert g is not None

    def test_empty_example_still_returns_none(self):
        assert trace_to_grpo_example({"messages": []}, multimodal_format=False) is None
