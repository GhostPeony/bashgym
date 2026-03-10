"""Tests for ChatGPT conversation importer."""

import json
import tempfile
import zipfile
from pathlib import Path

import pytest


CHATGPT_CONVERSATION = {
    "title": "Help with Python",
    "create_time": 1700000000.0,
    "update_time": 1700001000.0,
    "mapping": {
        "root": {
            "id": "root",
            "message": None,
            "parent": None,
            "children": ["user1"]
        },
        "user1": {
            "id": "user1",
            "message": {
                "id": "msg_user1",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Write a fibonacci function in Python"]},
                "create_time": 1700000000.0,
                "metadata": {}
            },
            "parent": "root",
            "children": ["asst1"]
        },
        "asst1": {
            "id": "asst1",
            "message": {
                "id": "msg_asst1",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": ["Here's a fibonacci function:\n```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"]
                },
                "create_time": 1700000010.0,
                "metadata": {"model_slug": "gpt-4", "finish_details": {"type": "stop"}}
            },
            "parent": "user1",
            "children": ["user2"]
        },
        "user2": {
            "id": "user2",
            "message": {
                "id": "msg_user2",
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Now add memoization"]},
                "create_time": 1700000060.0,
                "metadata": {}
            },
            "parent": "asst1",
            "children": ["asst2"]
        },
        "asst2": {
            "id": "asst2",
            "message": {
                "id": "msg_asst2",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "text",
                    "parts": ["Here's the memoized version:\n```python\nfrom functools import lru_cache\n\n@lru_cache\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"]
                },
                "create_time": 1700000070.0,
                "metadata": {"model_slug": "gpt-4", "finish_details": {"type": "stop"}}
            },
            "parent": "user2",
            "children": []
        }
    },
    "current_node": "asst2"
}


def make_chatgpt_zip(conversations: list, tmp_dir: Path) -> Path:
    """Create a mock ChatGPT export zip."""
    zip_path = tmp_dir / "chatgpt_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
    return zip_path


class TestChatGPTImporter:
    def test_parse_conversation_extracts_steps(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()
        steps, metadata = importer.parse_conversation(CHATGPT_CONVERSATION)

        assert len(steps) >= 2
        assert metadata["source"] == "chatgpt"
        assert metadata["title"] == "Help with Python"

    def test_parse_conversation_preserves_order(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()
        steps, _ = importer.parse_conversation(CHATGPT_CONVERSATION)

        timestamps = [s.timestamp for s in steps]
        assert timestamps == sorted(timestamps)

    def test_parse_conversation_extracts_model(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()
        _, metadata = importer.parse_conversation(CHATGPT_CONVERSATION)

        assert "gpt-4" in metadata.get("models_used", [])

    def test_import_from_zip(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = make_chatgpt_zip([CHATGPT_CONVERSATION], tmp_path)
            results = importer.import_from_zip(zip_path, force=True)

            assert len(results) == 1
            assert results[0].steps_imported > 0
            assert results[0].error is None

    def test_import_from_zip_multiple_conversations(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()
        convos = [CHATGPT_CONVERSATION, {**CHATGPT_CONVERSATION, "title": "Second chat"}]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = make_chatgpt_zip(convos, tmp_path)
            results = importer.import_from_zip(zip_path)

            assert len(results) == 2

    def test_empty_conversation_skipped(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()
        empty_convo = {
            "title": "Empty",
            "create_time": 1700000000.0,
            "update_time": 1700000000.0,
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": []}
            },
            "current_node": "root"
        }
        steps, metadata = importer.parse_conversation(empty_convo)
        assert len(steps) == 0

    def test_tool_use_conversation(self):
        """ChatGPT conversations with Code Interpreter / tool use."""
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        convo_with_tools = {
            "title": "Code execution",
            "create_time": 1700000000.0,
            "update_time": 1700001000.0,
            "mapping": {
                "root": {"id": "root", "message": None, "parent": None, "children": ["u1"]},
                "u1": {
                    "id": "u1",
                    "message": {
                        "id": "m1", "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Run print('hello')"]},
                        "create_time": 1700000000.0, "metadata": {}
                    },
                    "parent": "root", "children": ["a1"]
                },
                "a1": {
                    "id": "a1",
                    "message": {
                        "id": "m2", "author": {"role": "assistant"},
                        "content": {"content_type": "code", "parts": ["print('hello')"]},
                        "create_time": 1700000010.0,
                        "metadata": {"model_slug": "gpt-4"}
                    },
                    "parent": "u1", "children": ["t1"]
                },
                "t1": {
                    "id": "t1",
                    "message": {
                        "id": "m3", "author": {"role": "tool"},
                        "content": {"content_type": "execution_output", "parts": ["hello\n"]},
                        "create_time": 1700000011.0, "metadata": {}
                    },
                    "parent": "a1", "children": ["a2"]
                },
                "a2": {
                    "id": "a2",
                    "message": {
                        "id": "m4", "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["The code ran successfully."]},
                        "create_time": 1700000012.0,
                        "metadata": {"model_slug": "gpt-4"}
                    },
                    "parent": "t1", "children": []
                }
            },
            "current_node": "a2"
        }

        importer = ChatGPTImporter()
        steps, metadata = importer.parse_conversation(convo_with_tools)

        tool_steps = [s for s in steps if s.tool_name != "conversation"]
        assert len(tool_steps) >= 1

    def test_skip_already_imported(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()

        # Force first import in case previous test runs left state
        result1 = importer.import_conversation(CHATGPT_CONVERSATION, force=True)
        assert result1.steps_imported > 0

        result2 = importer.import_conversation(CHATGPT_CONVERSATION)
        assert result2.skipped is True
        assert result2.steps_imported == 0

    def test_force_reimport(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()

        result1 = importer.import_conversation(CHATGPT_CONVERSATION, force=True)
        assert result1.steps_imported > 0

        result2 = importer.import_conversation(CHATGPT_CONVERSATION, force=True)
        assert result2.skipped is False
        assert result2.steps_imported > 0

    def test_no_conversations_json_in_zip(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("other_file.txt", "not conversations")

            results = importer.import_from_zip(zip_path)
            assert len(results) == 1
            assert results[0].error is not None

    def test_conversation_turns_counted(self):
        from bashgym.trace_capture.importers.chatgpt import ChatGPTImporter

        importer = ChatGPTImporter()
        _, metadata = importer.parse_conversation(CHATGPT_CONVERSATION)

        assert metadata["conversation_turns"] == 2  # Two user messages
