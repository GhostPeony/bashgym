"""Tests for the Hermes Agent session importer (against a synthetic state.db)."""

import json
import sqlite3

import pytest

from bashgym.trace_capture.importers.hermes_history import (
    HermesSessionImporter,
    import_hermes_sessions,
)


@pytest.fixture
def hermes_db(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, "
        "tool_calls TEXT, tool_name TEXT, token_count INT)"
    )
    rows = [
        ("20260616_010101_abc12345", "user", "fix the bug in utils.py", None, None, 8),
        (
            "20260616_010101_abc12345",
            "assistant",
            "",
            json.dumps([{"function": {"name": "Read", "arguments": '{"path": "utils.py"}'}}]),
            "Read",
            12,
        ),
        ("20260616_020202_def67890", "user", "run the tests", None, None, 4),
    ]
    conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return db


class TestHermesImporter:
    def test_imports_sessions_grouped(self, hermes_db):
        result = import_hermes_sessions(hermes_db)
        assert result.n_sessions == 2
        assert result.n_messages == 3
        ids = {s["session_id"] for s in result.sessions}
        assert "20260616_010101_abc12345" in ids

    def test_source_tool_and_tool_calls_sanitized(self, hermes_db):
        result = import_hermes_sessions(hermes_db)
        sess = next(s for s in result.sessions if s["session_id"] == "20260616_010101_abc12345")
        assert sess["source_tool"] == "hermes"
        assistant = sess["messages"][1]
        # JSON-string tool args coerced to a dict
        assert assistant["tool_calls"][0]["function"]["arguments"] == {"path": "utils.py"}

    def test_missing_db_returns_empty(self, tmp_path):
        result = import_hermes_sessions(tmp_path / "nope.db")
        assert result.n_sessions == 0 and result.sessions == []

    def test_available(self, hermes_db, tmp_path):
        assert HermesSessionImporter(hermes_db).available() is True
        assert HermesSessionImporter(tmp_path / "nope.db").available() is False
