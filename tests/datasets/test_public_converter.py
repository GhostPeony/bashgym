"""Tests for normalizing public datasets into our messages format."""

from bashgym.datasets.converters import normalize_public_messages


class TestNormalizePublicMessages:
    def test_sharegpt_conversations_shape(self):
        ex = {
            "conversations": [
                {"from": "human", "value": "Fix the failing test in utils.py"},
                {"from": "gpt", "value": "Sure, let me look."},
            ]
        }
        out = normalize_public_messages(ex)
        assert out["messages"][0] == {"role": "user", "content": "Fix the failing test in utils.py"}
        assert out["messages"][1]["role"] == "assistant"

    def test_openhands_messages_shape_with_string_tool_args(self):
        ex = {
            "messages": [
                {"role": "user", "content": "read config"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "Read", "arguments": '{"path": "c.yaml"}'}}
                    ],
                },
            ]
        }
        out = normalize_public_messages(ex)
        args = out["messages"][1]["tool_calls"][0]["function"]["arguments"]
        assert args == {"path": "c.yaml"}  # JSON string coerced to dict

    def test_unparseable_tool_args_fall_back_to_raw(self):
        ex = {
            "messages": [
                {"role": "user", "content": "do it"},
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "X", "arguments": "oops"}}],
                },
            ]
        }
        out = normalize_public_messages(ex)
        assert out["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"raw": "oops"}

    def test_no_user_turn_returns_none(self):
        ex = {"messages": [{"role": "assistant", "content": "hello"}]}
        assert normalize_public_messages(ex) is None

    def test_empty_or_bad_input_returns_none(self):
        assert normalize_public_messages({}) is None
        assert normalize_public_messages({"messages": "not a list"}) is None

    def test_trajectory_key_supported(self):
        ex = {"trajectory": [{"role": "user", "content": "go"}]}
        assert normalize_public_messages(ex)["messages"][0]["content"] == "go"
