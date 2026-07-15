"""Tests for the model-agnostic SFT training CLI (pure-logic parts).

The script's heavy imports (torch/unsloth/trl) live inside main(), so loading it
as a module here is cheap and does not trigger any training.
"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
_SPEC = importlib.util.spec_from_file_location("train_model", _PATH)
train_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_model)


class TestSanitizeMessages:
    def test_tool_call_string_args_become_dict(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
            }
        ]
        out = train_model.sanitize_messages(msgs)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_unparseable_args_fall_back_to_raw(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}],
            }
        ]
        out = train_model.sanitize_messages(msgs)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"raw": "not json"}

    def test_dict_args_unchanged(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
            }
        ]
        out = train_model.sanitize_messages(msgs)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_none_content_becomes_empty(self):
        out = train_model.sanitize_messages([{"role": "user", "content": None}])
        assert out[0]["content"] == ""


class TestParseArgs:
    def test_required_and_defaults(self):
        ns = train_model.parse_args(["--base-model", "X", "--train", "t.jsonl", "--output", "o"])
        assert ns.base_model == "X"
        assert ns.max_seq_length == 8192
        assert ns.load_in_4bit is False

    def test_missing_required_exits(self):
        with pytest.raises(SystemExit):
            train_model.parse_args(["--train", "t.jsonl"])


def test_main_rejects_nvfp4_before_importing_training_runtime():
    with pytest.raises(SystemExit, match="not a fine-tuning base"):
        train_model.main(
            [
                "--base-model",
                "unsloth/gemma-4-12b-it-NVFP4",
                "--train",
                "t.jsonl",
                "--output",
                "o",
            ]
        )
