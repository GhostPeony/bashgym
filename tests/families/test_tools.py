"""Tests for tool-call sanitization + per-family format validation."""

from bashgym.families.tools import (
    sanitize_message_tool_calls,
    sanitize_tool_call,
    validate_tool_call,
)


def _tc(name="Read", args=None):
    return {"function": {"name": name, "arguments": args}}


class TestSanitize:
    def test_string_args_become_dict(self):
        out = sanitize_tool_call(_tc("Read", '{"path": "a.py"}'))
        assert out["function"]["arguments"] == {"path": "a.py"}

    def test_unparseable_args_fall_back_to_raw(self):
        out = sanitize_tool_call(_tc("X", "nope"))
        assert out["function"]["arguments"] == {"raw": "nope"}

    def test_dict_args_unchanged(self):
        out = sanitize_tool_call(_tc("X", {"a": 1}))
        assert out["function"]["arguments"] == {"a": 1}

    def test_message_level(self):
        msgs = [{"role": "assistant", "tool_calls": [_tc("Read", '{"p": 1}')]}]
        out = sanitize_message_tool_calls(msgs)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"p": 1}


class TestValidate:
    def test_valid_dict_args_no_issues(self):
        assert validate_tool_call(_tc("Read", {"path": "a"}), "gemma4_delimited") == []

    def test_missing_name(self):
        assert "missing function.name" in validate_tool_call({"function": {}}, "openai_json")

    def test_string_args_flagged(self):
        issues = validate_tool_call(_tc("Read", '{"path": "a"}'), "gemma4_delimited")
        assert any("string" in i for i in issues)

    def test_non_serializable_args_flagged_for_delimiter_formats(self):
        bad = _tc("X", {"obj": object()})
        assert any("serializable" in i for i in validate_tool_call(bad, "qwen_xml"))
