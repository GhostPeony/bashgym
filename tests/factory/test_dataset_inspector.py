"""Dataset inspector validates NeMo-format examples for chat-template compatibility."""

import json


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_valid_example_has_no_warnings(tmp_path):
    from bashgym.factory.dataset_inspector import inspect_dataset

    f = tmp_path / "train.jsonl"
    _write_jsonl(
        f,
        [
            {
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Fix the bug."},
                    {"role": "assistant", "content": "Done."},
                ]
            }
        ],
    )

    report = inspect_dataset(f, offset=0, limit=10)
    assert report["total"] == 1
    assert report["examples"][0]["warnings"] == []


def test_tool_calls_string_arguments_flagged(tmp_path):
    from bashgym.factory.dataset_inspector import inspect_dataset

    f = tmp_path / "train.jsonl"
    _write_jsonl(
        f,
        [
            {
                "messages": [
                    {"role": "user", "content": "Run ls."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "bash", "arguments": '{"command": "ls"}'}}
                        ],
                    },
                ]
            }
        ],
    )

    report = inspect_dataset(f, offset=0, limit=10)
    warnings = report["examples"][0]["warnings"]
    assert any("arguments" in w and "string" in w for w in warnings)


def test_role_and_structure_warnings(tmp_path):
    from bashgym.factory.dataset_inspector import inspect_dataset

    f = tmp_path / "train.jsonl"
    _write_jsonl(
        f,
        [
            {"messages": [{"role": "user", "content": "hi"}]},  # no assistant turn
            {"messages": [{"role": "alien", "content": "hi"}]},  # unknown role
            {"not_messages": True},  # malformed
        ],
    )

    report = inspect_dataset(f, offset=0, limit=10)
    assert report["total"] == 3
    assert any("assistant" in w for w in report["examples"][0]["warnings"])
    assert any("role" in w for w in report["examples"][1]["warnings"])
    assert any("messages" in w for w in report["examples"][2]["warnings"])


def test_offset_and_limit_slice(tmp_path):
    from bashgym.factory.dataset_inspector import inspect_dataset

    f = tmp_path / "train.jsonl"
    _write_jsonl(
        f,
        [
            {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
            for i in range(10)
        ],
    )

    report = inspect_dataset(f, offset=4, limit=3)
    assert report["total"] == 10
    assert [e["index"] for e in report["examples"]] == [4, 5, 6]


def test_invalid_json_line_reported(tmp_path):
    from bashgym.factory.dataset_inspector import inspect_dataset

    f = tmp_path / "train.jsonl"
    f.write_text(
        '{"messages": [{"role": "user", "content": "ok"}]}\nnot json at all\n', encoding="utf-8"
    )

    report = inspect_dataset(f, offset=0, limit=10)
    assert report["total"] == 2
    assert any("JSON" in w for w in report["examples"][1]["warnings"])
