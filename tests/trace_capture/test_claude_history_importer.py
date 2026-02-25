"""
Tests for ClaudeSessionImporter enriched data extraction.

Tests cover:
  - Backward compatibility (existing fields still present)
  - Session metadata extraction (model, tokens, cost, git branch, etc.)
  - Per-step enrichment (thinking, text, model per step)
  - Tool expansion (all tools captured, not just 6)
  - Cost estimation (known/unknown models, prefix matching)
  - Edge cases (empty files, malformed lines, no assistant messages)
"""

import json
import tempfile
from pathlib import Path
from dataclasses import asdict

import pytest

from bashgym.trace_capture.importers.claude_history import ClaudeSessionImporter
from bashgym.trace_capture.core import estimate_cost_usd, TraceSession


# ---------------------------------------------------------------------------
# Helpers to build JSONL fixtures
# ---------------------------------------------------------------------------

def _assistant_event(
    tool_uses=None,
    model="claude-sonnet-4-5",
    input_tokens=100,
    output_tokens=50,
    cache_creation=0,
    cache_read=0,
    thinking_blocks=None,
    text_blocks=None,
    stop_reason=None,
    timestamp="2026-02-24T12:00:00Z",
    **extra_event,
):
    """Build an assistant event dict."""
    content = []
    if thinking_blocks:
        for t in thinking_blocks:
            content.append({"type": "thinking", "thinking": t})
    if text_blocks:
        for t in text_blocks:
            content.append({"type": "text", "text": t})
    if tool_uses:
        for tu in tool_uses:
            content.append({
                "type": "tool_use",
                "id": tu.get("id", "toolu_test123"),
                "name": tu.get("name", "Bash"),
                "input": tu.get("input", {"command": "echo hello"}),
            })
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "model": model,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
            "content": content,
            "stop_reason": stop_reason,
        },
        **extra_event,
    }


def _user_event(
    text=None,
    tool_results=None,
    tool_use_result=None,
    plan_content=None,
    timestamp="2026-02-24T12:00:01Z",
    **extra_event,
):
    """Build a user event dict."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if tool_results:
        for tr in tool_results:
            content.append({
                "type": "tool_result",
                "tool_use_id": tr.get("tool_use_id", "toolu_test123"),
                "content": tr.get("content", "command output here"),
                "is_error": tr.get("is_error", False),
            })
    event = {
        "type": "user",
        "timestamp": timestamp,
        "message": {"content": content},
        **extra_event,
    }
    if tool_use_result:
        event["toolUseResult"] = tool_use_result
    if plan_content:
        event["planContent"] = plan_content
    return event


def _write_session(events, tmp_path, name="test_session"):
    """Write events to a JSONL file and return the path."""
    session_file = tmp_path / f"{name}.jsonl"
    with open(session_file, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return session_file


# ---------------------------------------------------------------------------
# 1. Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Existing fields and shapes still present after enrichment."""

    def test_returns_steps_and_metadata_dict(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "Fix the bug"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Bash", "input": {"command": "ls"}}]),
            _user_event(tool_results=[{"tool_use_id": "t1", "content": "file.py"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, meta = importer.parse_session_file(f)

        assert isinstance(steps, list)
        assert len(steps) > 0
        assert isinstance(meta, dict)
        assert "user_initial_prompt" in meta

    def test_step_fields_unchanged(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "Hi"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Edit", "input": {"file": "a.py"}}]),
            _user_event(tool_results=[{"tool_use_id": "t1", "content": "done"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, _ = importer.parse_session_file(f)

        step = steps[0]
        assert step.tool_name == "Edit"
        assert step.source_tool == "claude_code"
        assert step.step_id  # non-empty
        assert step.timestamp  # non-empty

    def test_user_initial_prompt_accessible(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "Fix the login bug"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Read"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["user_initial_prompt"] == "Fix the login bug"

    def test_from_steps_creates_valid_session(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "Hello"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Bash", "input": {"command": "pwd"}}]),
            _user_event(tool_results=[{"tool_use_id": "t1", "content": "/repo"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, session_meta = importer.parse_session_file(f)

        user_prompt = session_meta.pop("user_initial_prompt", "Imported session")
        session = TraceSession.from_steps(
            steps, source_tool="claude_code",
            verification_passed=None, imported=True,
            user_initial_prompt=user_prompt,
            **session_meta
        )
        d = asdict(session)
        assert d["source_tool"] == "claude_code"
        assert "summary" in d
        assert d["summary"]["total_steps"] >= 1
        assert d["summary"]["tool_breakdown"]["Bash"] == 1

    def test_summary_has_tool_breakdown(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[
                {"id": "t1", "name": "Bash", "input": {"command": "ls"}},
                {"id": "t2", "name": "Read", "input": {"file_path": "a.py"}},
            ]),
            _assistant_event(tool_uses=[
                {"id": "t3", "name": "Bash", "input": {"command": "pwd"}},
            ]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, meta = importer.parse_session_file(f)

        user_prompt = meta.pop("user_initial_prompt", "")
        session = TraceSession.from_steps(steps, source_tool="claude_code", **meta)
        assert session.summary["tool_breakdown"] == {"Bash": 2, "Read": 1}


# ---------------------------------------------------------------------------
# 2. Session Metadata Extraction
# ---------------------------------------------------------------------------

class TestSessionMetadata:
    """Session-level accumulators are computed correctly."""

    def test_models_used(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(model="claude-opus-4-6", tool_uses=[{"id": "t1", "name": "Bash"}]),
            _assistant_event(model="claude-sonnet-4-5", tool_uses=[{"id": "t2", "name": "Read"}]),
            _assistant_event(model="claude-opus-4-6", tool_uses=[{"id": "t3", "name": "Edit"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["models_used"] == ["claude-opus-4-6", "claude-sonnet-4-5"]
        assert meta["model"] == "claude-opus-4-6"  # first alphabetically

    def test_token_sums(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(input_tokens=100, output_tokens=50, cache_creation=30, cache_read=200,
                             tool_uses=[{"id": "t1", "name": "Bash"}]),
            _assistant_event(input_tokens=200, output_tokens=100, cache_creation=70, cache_read=300,
                             tool_uses=[{"id": "t2", "name": "Bash"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["total_input_tokens"] == 300
        assert meta["total_output_tokens"] == 150
        assert meta["total_cache_creation_tokens"] == 100
        assert meta["total_cache_read_tokens"] == 500

    def test_all_user_prompts_multi_turn(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "Fix the login bug"}, "cwd": "/repo",
             "timestamp": "2026-02-24T12:00:00Z"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Bash"}]),
            {"type": "user", "message": {"content": "Now add tests"},
             "timestamp": "2026-02-24T12:05:00Z"},
            _assistant_event(tool_uses=[{"id": "t2", "name": "Write"}]),
            {"type": "user", "message": {"content": "Looks good, commit it"},
             "timestamp": "2026-02-24T12:10:00Z"},
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["conversation_turns"] == 3
        assert len(meta["all_user_prompts"]) == 3
        assert meta["all_user_prompts"][0]["text"] == "Fix the login bug"
        assert meta["all_user_prompts"][1]["text"] == "Now add tests"
        assert meta["all_user_prompts"][2]["text"] == "Looks good, commit it"

    def test_thinking_block_count(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(
                thinking_blocks=["Let me think about this...", "Actually, I should check first."],
                tool_uses=[{"id": "t1", "name": "Read"}],
            ),
            _assistant_event(
                thinking_blocks=["Now I understand."],
                tool_uses=[{"id": "t2", "name": "Edit"}],
            ),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["thinking_block_count"] == 3

    def test_git_branch_and_version(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo",
             "gitBranch": "feature/auth", "version": "2.1.39", "slug": "cool-session",
             "sessionId": "abc-123-def"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Bash"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["git_branch"] == "feature/auth"
        assert meta["claude_version"] == "2.1.39"
        assert meta["session_slug"] == "cool-session"
        assert meta["session_id_original"] == "abc-123-def"

    def test_subagent_metadata(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Task"}]),
            _user_event(
                tool_results=[{"tool_use_id": "t1", "content": "Agent result"}],
                tool_use_result={"totalTokens": 15000, "totalDurationMs": 8500, "totalToolUseCount": 5},
            ),
            _assistant_event(tool_uses=[{"id": "t2", "name": "Task"}]),
            _user_event(
                tool_results=[{"tool_use_id": "t2", "content": "Agent result 2"}],
                tool_use_result={"totalTokens": 20000, "totalDurationMs": 12000, "totalToolUseCount": 8},
            ),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["subagent_count"] == 2
        assert meta["subagent_total_tokens"] == 35000
        assert meta["subagent_total_duration_ms"] == 20500

    def test_plan_contents(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "plan this"},
             "cwd": "/repo", "planContent": "Step 1: Do X\nStep 2: Do Y"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Bash"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert len(meta["plan_contents"]) == 1
        assert "Step 1" in meta["plan_contents"][0]


# ---------------------------------------------------------------------------
# 3. Per-Step Enrichment
# ---------------------------------------------------------------------------

class TestPerStepEnrichment:
    """Individual TraceStep.metadata has model/tokens/thinking/text."""

    def test_step_has_model_and_tokens(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(
                model="claude-opus-4-6", input_tokens=500, output_tokens=200,
                tool_uses=[{"id": "t1", "name": "Bash", "input": {"command": "ls"}}],
            ),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, _ = importer.parse_session_file(f)

        assert steps[0].metadata["model"] == "claude-opus-4-6"
        assert steps[0].metadata["input_tokens"] == 500
        assert steps[0].metadata["output_tokens"] == 200

    def test_thinking_and_text_attached_to_step(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(
                thinking_blocks=["I need to read the file first."],
                text_blocks=["Let me check the code."],
                tool_uses=[{"id": "t1", "name": "Read", "input": {"file_path": "main.py"}}],
            ),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, _ = importer.parse_session_file(f)

        assert steps[0].metadata["thinking_content"] == "I need to read the file first."
        assert steps[0].metadata["assistant_text"] == "Let me check the code."

    def test_thinking_truncation(self, tmp_path):
        long_thinking = "X" * 5000
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(
                thinking_blocks=[long_thinking],
                tool_uses=[{"id": "t1", "name": "Bash"}],
            ),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()

        # Default truncation at 2000
        steps, _ = importer.parse_session_file(f)
        assert len(steps[0].metadata["thinking_content"]) == 2000

        # Custom truncation
        steps2, _ = importer.parse_session_file(f, thinking_max_chars=500)
        assert len(steps2[0].metadata["thinking_content"]) == 500

    def test_multiple_tool_uses_get_correct_thinking(self, tmp_path):
        """First thinking/text associate with first tool_use, second with second."""
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            {
                "type": "assistant",
                "timestamp": "2026-02-24T12:00:00Z",
                "message": {
                    "model": "claude-sonnet-4-5",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "content": [
                        {"type": "thinking", "thinking": "Think A"},
                        {"type": "text", "text": "Text A"},
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file": "a.py"}},
                        {"type": "thinking", "thinking": "Think B"},
                        {"type": "tool_use", "id": "t2", "name": "Edit", "input": {"file": "a.py"}},
                    ],
                    "stop_reason": None,
                },
            },
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, _ = importer.parse_session_file(f)

        assert len(steps) == 2
        assert steps[0].metadata["thinking_content"] == "Think A"
        assert steps[0].metadata["assistant_text"] == "Text A"
        assert steps[1].metadata["thinking_content"] == "Think B"
        assert steps[1].metadata["assistant_text"] is None  # no text before second tool_use


# ---------------------------------------------------------------------------
# 4. Tool Expansion
# ---------------------------------------------------------------------------

class TestToolExpansion:
    """All tools captured, not just the old 6-tool whitelist."""

    def test_task_tool_captured(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Task", "input": {"prompt": "research"}}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, meta = importer.parse_session_file(f)

        assert len(steps) == 1
        assert steps[0].tool_name == "Task"
        assert "Task" in meta["tools_used"]

    def test_mcp_and_webfetch_captured(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[
                {"id": "t1", "name": "WebFetch", "input": {"url": "https://example.com"}},
                {"id": "t2", "name": "mcp__pencil__batch_get", "input": {"filePath": "x.pen"}},
                {"id": "t3", "name": "WebSearch", "input": {"query": "test"}},
            ]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, meta = importer.parse_session_file(f)

        assert len(steps) == 3
        names = {s.tool_name for s in steps}
        assert names == {"WebFetch", "mcp__pencil__batch_get", "WebSearch"}
        assert meta["total_tool_calls"] == 3

    def test_excluded_tools_empty_by_default(self):
        assert ClaudeSessionImporter.EXCLUDED_TOOLS == set()


# ---------------------------------------------------------------------------
# 5. Cost Estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:
    """estimate_cost_usd returns correct values."""

    def test_known_model(self):
        # 1M input tokens of sonnet-4-5 = $3.00
        cost = estimate_cost_usd("claude-sonnet-4-5", 1_000_000, 0)
        assert cost == 3.0

    def test_known_model_output(self):
        # 1M output tokens of opus-4-6 = $75.00
        cost = estimate_cost_usd("claude-opus-4-6", 0, 1_000_000)
        assert cost == 75.0

    def test_cache_tokens(self):
        # haiku-4-5: 1M cache_creation = $1.0, 1M cache_read = $0.08
        cost = estimate_cost_usd("claude-haiku-4-5", 0, 0, 1_000_000, 1_000_000)
        assert cost == pytest.approx(1.08)

    def test_unknown_model_returns_zero(self):
        cost = estimate_cost_usd("gpt-4o", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_prefix_matching(self):
        # "claude-sonnet-4-5-20250929" should match "claude-sonnet-4-5"
        cost = estimate_cost_usd("claude-sonnet-4-5-20250929", 1_000_000, 0)
        assert cost == 3.0

    def test_estimated_cost_in_session_meta(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(
                model="claude-sonnet-4-5", input_tokens=1000, output_tokens=500,
                tool_uses=[{"id": "t1", "name": "Bash"}],
            ),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        # 1000 * 3/1M + 500 * 15/1M = 0.003 + 0.0075 = 0.0105
        assert meta["api_equivalent_cost_usd"] == pytest.approx(0.0105)


# ---------------------------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Graceful handling of unusual inputs."""

    def test_empty_jsonl(self, tmp_path):
        f = _write_session([], tmp_path)
        importer = ClaudeSessionImporter()
        steps, meta = importer.parse_session_file(f)

        assert steps == []
        assert meta["user_initial_prompt"] is None
        assert meta["total_input_tokens"] == 0

    def test_no_assistant_messages(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "Hello"}, "cwd": "/repo"},
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, meta = importer.parse_session_file(f)

        assert steps == []
        assert meta["user_initial_prompt"] == "Hello"
        assert meta["conversation_turns"] == 1
        assert meta["models_used"] == []

    def test_malformed_lines_skipped(self, tmp_path):
        session_file = tmp_path / "bad.jsonl"
        with open(session_file, "w") as f:
            f.write("not valid json\n")
            f.write('{"broken": true\n')  # missing closing brace
            f.write(json.dumps({
                "type": "user", "message": {"content": "valid"}, "cwd": "/repo"
            }) + "\n")
            f.write(json.dumps(_assistant_event(
                tool_uses=[{"id": "t1", "name": "Bash"}]
            )) + "\n")
        importer = ClaudeSessionImporter()
        steps, meta = importer.parse_session_file(session_file)

        # Should have parsed the valid lines
        assert len(steps) == 1
        assert meta["user_initial_prompt"] == "valid"

    def test_tool_result_matches_step(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[
                {"id": "t1", "name": "Bash", "input": {"command": "echo hello"}},
            ]),
            _user_event(tool_results=[
                {"tool_use_id": "t1", "content": "hello\n", "is_error": False},
            ]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, _ = importer.parse_session_file(f)

        assert steps[0].output == "hello\n"
        assert steps[0].success is True
        assert steps[0].exit_code == 0

    def test_error_tool_result(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[
                {"id": "t1", "name": "Bash", "input": {"command": "bad_cmd"}},
            ]),
            _user_event(tool_results=[
                {"tool_use_id": "t1", "content": "command not found", "is_error": True},
            ]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, _ = importer.parse_session_file(f)

        assert steps[0].output == "command not found"
        assert steps[0].success is False
        assert steps[0].exit_code == 1

    def test_user_content_as_plain_string(self, tmp_path):
        """User message content can be a plain string, not a list of blocks."""
        events = [
            {"type": "user", "message": {"content": "Fix the bug"}, "cwd": "/repo"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Bash"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        _, meta = importer.parse_session_file(f)

        assert meta["user_initial_prompt"] == "Fix the bug"

    def test_cwd_from_event(self, tmp_path):
        events = [
            {"type": "user", "message": {"content": "go"}, "cwd": "/home/user/project"},
            _assistant_event(tool_uses=[{"id": "t1", "name": "Bash"}]),
        ]
        f = _write_session(events, tmp_path)
        importer = ClaudeSessionImporter()
        steps, _ = importer.parse_session_file(f)

        assert steps[0].cwd == "/home/user/project"
        assert steps[0].repo["path"] == "/home/user/project"
        assert steps[0].repo["name"] == "project"
