import json

import pytest

from bashgym.trace_capture.schema import SurfaceType, validate_event
from bashgym.trace_capture.status_protocol import (
    AGENT_STATUS_MARKER,
    AGENT_STATUS_SCHEMA_VERSION,
    REPLAY_SCRUB_SCHEMA_VERSION,
    agent_status_to_event,
    format_agent_status_marker,
    normalize_agent_status,
    parse_agent_status_markers,
    scrub_trace_replay,
    scrub_trace_replay_file,
)


def test_agent_status_marker_round_trips_without_terminal_regex():
    marker = format_agent_status_marker(
        {
            "status": "in-progress",
            "message": "running verifier",
            "source_tool": "codex",
            "session_id": "sess-1",
            "progress": 75,
            "backend": "pytest",
        }
    )

    text = "\n".join(
        [
            "pytest collected 12 items",
            "status: failed to fetch",  # Should be ignored because it is not a marker.
            marker,
            "FAILED tests/test_example.py::test_example",
        ]
    )

    statuses = parse_agent_status_markers(text)

    assert marker.startswith(AGENT_STATUS_MARKER)
    assert statuses == [
        {
            "schema_version": AGENT_STATUS_SCHEMA_VERSION,
            "status": "running",
            "message": "running verifier",
            "source": "codex",
            "session_id": "sess-1",
            "progress": 0.75,
            "metadata": {"backend": "pytest"},
        }
    ]


def test_agent_status_parser_can_strictly_reject_bad_markers():
    bad_marker = f"{AGENT_STATUS_MARKER} {{not json}}"

    assert parse_agent_status_markers(bad_marker) == []
    with pytest.raises(ValueError, match="invalid agent status marker"):
        parse_agent_status_markers(bad_marker, strict=True)


def test_normalize_agent_status_rejects_unknown_status():
    with pytest.raises(ValueError, match="unsupported agent status"):
        normalize_agent_status({"status": "probably fine"})


def test_agent_status_to_event_is_valid_contextual_trace_event():
    event = agent_status_to_event(
        {
            "status": "blocked",
            "message": "waiting on GPU approval",
            "blocked_reason": "remote compute needs human approval",
            "source": "codex",
        },
        trace_id="trace-1",
        span_id="span-1",
        timestamp="2026-06-29T12:00:00+00:00",
    )

    assert validate_event(event) == event
    assert event.surface_type == SurfaceType.CONTEXTUAL
    assert event.body["operation_type"] == "agent_status"
    assert event.body["details"]["status"] == "blocked"
    assert event.body["details"]["blocked_reason"] == "remote compute needs human approval"


def test_scrub_trace_replay_redacts_and_truncates_legacy_trace_shape():
    payload = {
        "session_id": "trace-1",
        "trace": [
            {
                "tool_name": "Bash",
                "command": "OPENAI_API_KEY=sk-1234567890abcdefghijklmnop pytest",
                "output": "A" * 40 + " ghp_1234567890abcdefghijklmnop",
                "metadata": {"token": "hf_1234567890abcdefghijklmnop"},
            }
        ],
    }

    result = scrub_trace_replay(payload, max_output_chars=20)
    scrubbed = result["scrubbed"]
    encoded = json.dumps(scrubbed)

    assert result["schema_version"] == REPLAY_SCRUB_SCHEMA_VERSION
    assert result["ok"] is True
    assert "sk-1234567890abcdefghijklmnop" not in encoded
    assert "ghp_1234567890abcdefghijklmnop" not in encoded
    assert "hf_1234567890abcdefghijklmnop" not in encoded
    assert "[REDACTED]" in encoded
    assert "[truncated" in scrubbed["trace"][0]["output"]
    assert scrubbed["trace"][0]["tool_name"] == "Bash"
    assert result["stats"]["redactions"] >= 3
    assert result["stats"]["truncations"] == 1


def test_scrub_trace_replay_accepts_trace_event_objects():
    event = agent_status_to_event(
        {"status": "completed", "message": "OPENAI_API_KEY=sk-1234567890abcdefghijklmnop"},
        trace_id="trace-1",
        timestamp="2026-06-29T12:00:00+00:00",
    )

    result = scrub_trace_replay([event])
    scrubbed_message = result["scrubbed"][0]["body"]["details"]["message"]

    assert "sk-1234567890abcdefghijklmnop" not in scrubbed_message
    assert "[REDACTED]" in scrubbed_message


def test_scrub_trace_replay_file_preserves_jsonl_record_count(tmp_path):
    source = tmp_path / "replay.jsonl"
    source.write_text(
        json.dumps({"stdout": "HF_TOKEN=hf_1234567890abcdefghijklmnop"})
        + "\n"
        + json.dumps({"stderr": "ok"})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "replay.scrubbed.jsonl"

    report = scrub_trace_replay_file(source, output_path=output, max_output_chars=10)
    lines = output.read_text(encoding="utf-8").splitlines()

    assert report["ok"] is True
    assert report["stats"]["input_format"] == "jsonl"
    assert report["stats"]["records"] == 2
    assert len(lines) == 2
    assert "hf_1234567890abcdefghijklmnop" not in output.read_text(encoding="utf-8")
