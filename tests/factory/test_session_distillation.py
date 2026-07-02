import json

from bashgym.factory.session_distillation import (
    SESSION_DISTILLATION_HINT_TAG,
    build_session_distillation_records,
    build_session_distillation_records_from_traces,
    save_session_distillation_records,
    validate_session_distillation_record,
    validate_session_distillation_records,
)


def _write_trace(path, steps, *, session_id="s1", prompt="Do the task.", repo="ghostwork"):
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "metadata": {"user_initial_prompt": prompt},
                "primary_repo": {"name": repo},
                "trace": steps,
            }
        ),
        encoding="utf-8",
    )


def test_valid_session_distillation_record_has_no_errors():
    records = build_session_distillation_records(
        [
            {
                "tool": "Bash",
                "command": "pytest tests/missing_test.py",
                "output": "ERROR: file or directory not found: tests/missing_test.py",
                "success": False,
                "exit_code": 4,
            }
        ],
        task_prompt="Run the focused tests.",
        trace_id="trace-1",
        session_id="session-1",
        source_metadata={"source": "fixture"},
    )

    assert len(records) == 1
    payload = records[0].to_dict()
    assert validate_session_distillation_record(payload) == []
    assert payload["target_text"] == "pytest tests/missing_test.py"
    assert payload["target_span"] == {"start": 0, "end": len(payload["target_text"])}
    assert payload["loss_mask"]["policy"] == "target_span_only"
    assert SESSION_DISTILLATION_HINT_TAG not in payload["original_context"]
    assert SESSION_DISTILLATION_HINT_TAG in payload["hinted_context"]
    assert payload["source_metadata"]["source"] == "fixture"


def test_invalid_session_distillation_record_reports_specific_errors():
    errors = validate_session_distillation_record(
        {
            "record_id": "",
            "trace_id": "trace-1",
            "session_id": "session-1",
            "decision_id": "step-0",
            "step_index": 0,
            "original_context": "Task",
            "hinted_context": "Task",
            "hint_text": "Hint",
            "target_text": "cmd",
            "target_type": "unsupported",
            "target_span": {"start": 2, "end": 1},
            "loss_mask": {"policy": "full_context"},
            "reader_model": "reader",
            "reader_confidence": 1.2,
            "verifier_outcome": "failed",
            "quality_score": 0.5,
            "source_metadata": [],
        }
    )

    assert "record_id must be a non-empty string" in errors
    assert any("target_type must be one of" in error for error in errors)
    assert "target_span must satisfy 0 <= start < end" in errors
    assert any("loss_mask.policy must be one of" in error for error in errors)
    assert "loss_mask.target_span is required" in errors
    assert "reader_confidence must be between 0 and 1" in errors
    assert "source_metadata must be an object" in errors
    assert "hinted_context must differ from original_context" in errors


def test_partial_target_span_is_rejected_until_trainer_supports_it():
    # The trainer masks the whole target_text, so a record claiming a partial
    # span would be silently trained full. Reject it loudly until sub-span
    # masking is implemented.
    errors = validate_session_distillation_record(
        {
            "record_id": "r",
            "trace_id": "t",
            "session_id": "s",
            "decision_id": "d",
            "step_index": 0,
            "original_context": "Task:\nX\nNext action:",
            "hinted_context": f"Task:\nX\nNext action:\n\n{SESSION_DISTILLATION_HINT_TAG}\nhint\n",
            "hint_text": "hint",
            "target_text": "python app.py",
            "target_type": "command",
            "target_span": {"start": 0, "end": 6},  # partial: does not cover "python app.py"
            "loss_mask": {"policy": "target_span_only", "target_span": {"start": 0, "end": 6}},
            "reader_model": "reader",
            "reader_confidence": 0.7,
            "verifier_outcome": "failed",
            "quality_score": 0.7,
            "source_metadata": {},
        }
    )
    assert any("target_span must cover" in error for error in errors)


def test_clean_successful_trace_produces_no_records():
    records = build_session_distillation_records(
        [
            {
                "tool": "Bash",
                "command": "pytest tests/factory/test_session_distillation.py",
                "output": "1 passed",
                "success": True,
                "exit_code": 0,
            }
        ],
        task_prompt="Run tests.",
    )

    assert records == []


def test_successful_step_with_error_like_output_produces_no_records():
    # Regression: successful steps whose output merely contains failure keywords
    # (passing pytest xfail summaries, docs mentioning "command not found",
    # grepped log lines with "ERROR:") must not become mistake records.
    steps = [
        {
            "tool": "Bash",
            "command": "pytest -q",
            "output": "2 passed, 1 xfailed in 0.3s",
            "success": True,
            "exit_code": 0,
        },
        {
            "tool": "Read",
            "command": "cat README",
            "output": "See the command not found troubleshooting section",
            "success": True,
            "exit_code": 0,
        },
        {
            "tool": "Bash",
            "command": "pytest -q",
            "output": "0 failed, 5 passed",
            "success": True,
            "exit_code": 0,
        },
        {
            "tool": "Bash",
            "command": "grep ERROR log.txt",
            "output": "matched line: ERROR: (this is data)",
            "success": True,
            "exit_code": 0,
        },
    ]
    for step in steps:
        assert build_session_distillation_records([step], task_prompt="x") == [], step["output"]


def test_failed_step_without_explicit_success_flag_still_detected():
    # When a trace step carries no success/exit_code signal, error text in the
    # output is still the failure signal.
    records = build_session_distillation_records(
        [
            {
                "tool": "Bash",
                "command": "python x.py",
                "output": "Traceback (most recent call last): ModuleNotFoundError: No module named 'x'",
            }
        ],
        task_prompt="Run the script.",
    )
    assert len(records) == 1


def test_build_records_from_traces_uses_failed_steps_and_skips_clean(tmp_path):
    _write_trace(
        tmp_path / "failed.json",
        [
            {
                "tool_name": "Bash",
                "command": "python x.py",
                "output": "ModuleNotFoundError: No module named 'x'",
                "success": False,
                "exit_code": 1,
            }
        ],
        session_id="sess-failed",
    )
    _write_trace(
        tmp_path / "clean.json",
        [
            {
                "tool_name": "Bash",
                "command": "pytest -q",
                "output": "3 passed",
                "success": True,
                "exit_code": 0,
            }
        ],
        session_id="sess-clean",
    )

    records = build_session_distillation_records_from_traces(tmp_path)

    assert len(records) == 1
    payload = records[0].to_dict()
    assert validate_session_distillation_records([payload]) == []
    assert payload["trace_id"] == "failed"  # filename stem, since traces carry no trace_id
    assert payload["session_id"] == "sess-failed"
    assert payload["source_metadata"]["trace_file"] == "failed.json"
    assert payload["source_metadata"]["repo"] == "ghostwork"


def test_build_records_from_traces_respects_limit(tmp_path):
    for i in range(3):
        _write_trace(
            tmp_path / f"t{i}.json",
            [
                {
                    "tool_name": "Bash",
                    "command": f"cmd{i}",
                    "output": "error: boom",
                    "success": False,
                    "exit_code": 1,
                }
            ],
            session_id=f"s{i}",
        )
    records = build_session_distillation_records_from_traces(tmp_path, limit=2)
    assert len(records) == 2


def test_save_session_distillation_records_writes_jsonl(tmp_path):
    records = build_session_distillation_records(
        [
            {
                "tool": "Bash",
                "command": "python missing.py",
                "output": "python: can't open file 'missing.py': No such file or directory",
                "success": False,
                "exit_code": 2,
            }
        ],
        task_prompt="Run the script.",
        trace_id="trace-jsonl",
        session_id="session-jsonl",
    )
    output_path = save_session_distillation_records(
        records, tmp_path / "session_distillation_records.jsonl"
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "trace-jsonl"
    assert rows[0]["target_type"] == "command"
