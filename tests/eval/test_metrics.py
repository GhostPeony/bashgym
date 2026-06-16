"""Tests for step-level tool-call metrics."""

from bashgym.eval.metrics import score_tool_call, tool_arg_f1, tool_name_match


def _call(name, args):
    return {"function": {"name": name, "arguments": args}}


class TestToolNameMatch:
    def test_same_name_matches(self):
        assert tool_name_match(_call("Read", {}), _call("Read", {})) is True

    def test_different_name(self):
        assert tool_name_match(_call("Read", {}), _call("Bash", {})) is False

    def test_empty_gold_never_matches(self):
        assert tool_name_match(_call("", {}), _call("", {})) is False

    def test_flat_and_openai_shapes_interop(self):
        assert tool_name_match({"name": "Read"}, _call("Read", {})) is True


class TestToolArgF1:
    def test_identical_args(self):
        assert tool_arg_f1(_call("Read", {"path": "a.py"}), _call("Read", {"path": "a.py"})) == 1.0

    def test_no_args_both(self):
        assert tool_arg_f1(_call("X", {}), _call("X", {})) == 1.0

    def test_one_side_missing_args(self):
        assert tool_arg_f1(_call("X", {}), _call("X", {"a": 1})) == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        # gold has 2 keys, predicted matches 1 and adds a wrong one
        f1 = tool_arg_f1(_call("X", {"a": 1, "wrong": 9}), _call("X", {"a": 1, "b": 2}))
        assert 0.0 < f1 < 1.0

    def test_json_string_arguments_are_parsed(self):
        # arguments given as a JSON string (the known Gemma/Qwen serialization quirk)
        assert tool_arg_f1(_call("X", '{"a": 1}'), _call("X", {"a": 1})) == 1.0

    def test_string_values_normalized(self):
        assert tool_arg_f1(_call("X", {"p": "A.py "}), _call("X", {"p": "a.py"})) == 1.0

    def test_value_mismatch(self):
        assert tool_arg_f1(_call("X", {"a": 1}), _call("X", {"a": 2})) == 0.0


class TestScoreToolCall:
    def test_exact_match_requires_name_and_args(self):
        s = score_tool_call(_call("Read", {"path": "a"}), _call("Read", {"path": "a"}))
        assert s["name_match"] and s["arg_f1"] == 1.0 and s["exact_match"]

    def test_name_match_but_wrong_args_not_exact(self):
        s = score_tool_call(_call("Read", {"path": "a"}), _call("Read", {"path": "b"}))
        assert s["name_match"] and not s["exact_match"]
