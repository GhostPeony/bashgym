"""
Tests for High-Fidelity Trace Capture — PRD: bashgym_evolution.md

Verifies:
1. OperationalBody new fields (args, interrupt_reason, background_task_id, output_file_path, is_spilled)
2. Spill-to-disk for outputs > 1 MB
3. Backward compatibility with old traces missing new fields
"""

import tempfile
from pathlib import Path

from bashgym.trace_capture.schema import (
    OUTPUT_SPILL_THRESHOLD,
    OperationalBody,
    spill_output_to_disk,
    trace_step_to_events,
    validate_operational_event,
)


class TestOperationalBodyNewFields:
    """Test that the 5 new fields exist and have correct defaults."""

    def test_defaults_are_none_or_false(self):
        body = OperationalBody(tool_name="Bash", command="ls")
        assert body.args is None
        assert body.interrupt_reason is None
        assert body.background_task_id is None
        assert body.output_file_path is None
        assert body.is_spilled is False

    def test_args_parsing(self):
        body = OperationalBody(
            tool_name="Bash",
            command="git commit -m 'fix'",
            args=["git", "commit", "-m", "fix"],
        )
        assert body.args == ["git", "commit", "-m", "fix"]

    def test_interrupt_reason_values(self):
        for reason in ("timeout", "user_interrupt", "sigkill"):
            body = OperationalBody(
                tool_name="Bash",
                command="sleep 99",
                interrupt_reason=reason,
            )
            assert body.interrupt_reason == reason

    def test_background_task_id(self):
        body = OperationalBody(
            tool_name="Bash",
            command="make build &",
            background_task_id="abc123-uuid",
        )
        assert body.background_task_id == "abc123-uuid"

    def test_spilled_fields(self):
        body = OperationalBody(
            tool_name="Bash",
            command="cat bigfile",
            is_spilled=True,
            output_file_path="/tmp/spill.txt",
        )
        assert body.is_spilled is True
        assert body.output_file_path == "/tmp/spill.txt"


class TestSpillOutputToDisk:
    """Test the spill_output_to_disk function."""

    def test_writes_file_and_returns_preview(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            big_output = "x" * (OUTPUT_SPILL_THRESHOLD + 100)
            preview, filepath = spill_output_to_disk(
                big_output, "trace-1", "span-1", spill_dir=Path(tmpdir)
            )

            # Preview is truncated
            assert len(preview) < len(big_output)
            assert "[output spilled to disk" in preview

            # File was written with full content
            written = Path(filepath).read_text(encoding="utf-8")
            assert len(written) == len(big_output)
            assert written == big_output

    def test_file_naming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, filepath = spill_output_to_disk(
                "data", "trace-abc", "span-xyz", spill_dir=Path(tmpdir)
            )
            assert "trace-abc_span-xyz.txt" in filepath

    def test_creates_spill_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "spills"
            assert not nested.exists()
            spill_output_to_disk("data", "t", "s", spill_dir=nested)
            assert nested.exists()


class TestTraceStepToEventsHiFi:
    """Test that trace_step_to_events populates the new fields."""

    def test_args_from_step_dict(self):
        step = {
            "tool_name": "Bash",
            "command": "git status",
            "output": "clean",
            "args": ["git", "status"],
        }
        events = trace_step_to_events(step, trace_id="t1")
        op_body = events[0].body
        assert op_body["args"] == ["git", "status"]

    def test_interrupt_reason_from_step_dict(self):
        step = {
            "tool_name": "Bash",
            "command": "sleep 999",
            "output": "",
            "interrupt_reason": "timeout",
        }
        events = trace_step_to_events(step, trace_id="t1")
        assert events[0].body["interrupt_reason"] == "timeout"

    def test_interrupt_reason_from_metadata(self):
        step = {
            "tool_name": "Bash",
            "command": "sleep 999",
            "output": "",
            "metadata": {"interrupt_reason": "user_interrupt"},
        }
        events = trace_step_to_events(step, trace_id="t1")
        assert events[0].body["interrupt_reason"] == "user_interrupt"

    def test_background_task_id(self):
        step = {
            "tool_name": "Bash",
            "command": "make &",
            "output": "",
            "background_task_id": "bg-uuid-123",
        }
        events = trace_step_to_events(step, trace_id="t1")
        assert events[0].body["background_task_id"] == "bg-uuid-123"

    def test_spill_on_large_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch the default spill dir for this test
            import bashgym.trace_capture.schema as schema_mod

            original = schema_mod._DEFAULT_SPILL_DIR
            schema_mod._DEFAULT_SPILL_DIR = Path(tmpdir)
            try:
                big_output = "A" * (OUTPUT_SPILL_THRESHOLD + 500)
                step = {
                    "tool_name": "Bash",
                    "command": "cat hugefile",
                    "output": big_output,
                }
                events = trace_step_to_events(step, trace_id="t-spill")
                body = events[0].body
                assert body["is_spilled"] is True
                assert body["output_file_path"] is not None
                assert Path(body["output_file_path"]).exists()
                # Inline output is the truncated preview, not the full output
                assert len(body["output"]) < len(big_output)
            finally:
                schema_mod._DEFAULT_SPILL_DIR = original

    def test_no_spill_on_small_output(self):
        step = {
            "tool_name": "Bash",
            "command": "echo hi",
            "output": "hi",
        }
        events = trace_step_to_events(step, trace_id="t1")
        body = events[0].body
        assert body["is_spilled"] is False
        assert body["output_file_path"] is None
        assert body["output"] == "hi"

    def test_backward_compat_no_new_fields(self):
        """Old step dicts without new fields should still work."""
        step = {
            "tool_name": "Read",
            "command": "/path/to/file.py",
            "output": "contents",
            "exit_code": 0,
            "success": True,
            "cwd": "/home/user",
        }
        events = trace_step_to_events(step, trace_id="t1")
        body = events[0].body

        # Validate the operational body parses without error
        op = validate_operational_event(events[0])
        assert op.tool_name == "Read"
        assert op.args is None
        assert op.interrupt_reason is None
        assert op.background_task_id is None
        assert op.is_spilled is False
