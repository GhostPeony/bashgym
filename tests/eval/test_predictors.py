"""Tests for served-model predictors (bashgym/eval/predictors.py)."""

from bashgym.eval.heldout import first_gold_tool_call
from bashgym.eval.metrics import score_tool_call
from bashgym.eval.predictors import build_prompt_messages, endpoint_predictor, parse_tool_call


def _example():
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "Bash", "arguments": {"command": "ls"}},
                    }
                ],
            },
        ]
    }


class TestBuildPrompt:
    def test_stops_before_gold_assistant_call(self):
        msgs = build_prompt_messages(_example())
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert all("tool_calls" not in m for m in msgs)


class TestParseToolCall:
    def test_openai_structured(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "Bash", "arguments": '{"command":"ls"}'},
                            }
                        ]
                    }
                }
            ]
        }
        assert parse_tool_call(resp)["function"]["name"] == "Bash"

    def test_ollama_structured(self):
        resp = {
            "message": {"tool_calls": [{"function": {"name": "Read", "arguments": {"file": "a"}}}]}
        }
        assert parse_tool_call(resp)["function"]["name"] == "Read"

    def test_text_json(self):
        resp = {
            "choices": [{"message": {"content": '{"name":"Bash","arguments":{"command":"ls"}}'}}]
        }
        assert parse_tool_call(resp)["name"] == "Bash"

    def test_fenced_json(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "content": 'Sure:\n```json\n{"name":"Grep","arguments":{"pattern":"x"}}\n```'
                    }
                }
            ]
        }
        assert parse_tool_call(resp)["name"] == "Grep"

    def test_no_call_returns_empty(self):
        assert parse_tool_call({"choices": [{"message": {"content": "I am not sure."}}]}) == {}
        assert parse_tool_call("plain text, no json") == {}


class TestEndpointPredictor:
    def test_predicts_via_complete_and_strips_gold(self):
        captured = {}

        def complete(messages):
            captured["msgs"] = messages
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": {"command": "ls"}},
                                }
                            ]
                        }
                    }
                ]
            }

        tc = endpoint_predictor(complete)(_example())
        assert tc["function"]["name"] == "Bash"
        assert [m["role"] for m in captured["msgs"]] == ["system", "user"]  # gold turn stripped

    def test_complete_error_scores_as_miss(self):
        def boom(messages):
            raise RuntimeError("network down")

        assert endpoint_predictor(boom)(_example()) == {}


class TestEndToEndScoring:
    def test_predicted_call_scores_against_gold(self):
        ex = _example()
        gold = first_gold_tool_call(ex)

        def complete(messages):
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": '{"command":"ls"}'},
                                }
                            ]
                        }
                    }
                ]
            }

        pred = endpoint_predictor(complete)(ex)
        s = score_tool_call(pred, gold)
        assert s["name_match"] and s["exact_match"]  # JSON-string args coerced and matched
