"""Import Hermes Agent sessions into BashGym traces.

Hermes Agent (Nous Research) stores sessions in SQLite at ``~/.hermes/state.db``:
a ``messages`` table carrying role, content, tool_calls (JSON), tool_name and
token_count, keyed by session id. This importer reads that store and emits BashGym
traces (``{session_id, source_tool: 'hermes', messages, metadata}``) — closing the
deploy→trace loop: a fine-tuned student runs inside Hermes, its sessions flow back
as training data.

The schema is introspected defensively (column names discovered via PRAGMA) so it
tolerates Hermes schema differences across versions; verify against a real
``~/.hermes/state.db`` once Hermes is deployed on the Spark.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STATE_DB = Path.home() / ".hermes" / "state.db"


def _sanitize_tool_calls(raw) -> list:
    """Coerce tool_calls (Hermes may store them as a JSON string) to dicts with dict args."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        tc = dict(tc)
        fn = tc.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    fn["arguments"] = {"raw": args}
            tc["function"] = fn
        out.append(tc)
    return out


@dataclass
class HermesImportResult:
    sessions: list[dict] = field(default_factory=list)
    n_sessions: int = 0
    n_messages: int = 0


class HermesSessionImporter:
    """Read Hermes' SQLite session store and emit BashGym traces."""

    def __init__(self, state_db: Path | str = DEFAULT_STATE_DB):
        self.state_db = Path(state_db)

    def available(self) -> bool:
        return self.state_db.exists()

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        try:
            return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        except sqlite3.Error:
            return set()

    def import_sessions(self) -> HermesImportResult:
        result = HermesImportResult()
        if not self.available():
            return result

        conn = sqlite3.connect(str(self.state_db))
        conn.row_factory = sqlite3.Row
        try:
            cols = self._columns(conn, "messages")
            if not cols:
                return result
            sid_col = next((c for c in ("session_id", "session", "sid") if c in cols), None)
            if sid_col is None:
                return result

            rows = conn.execute(f"SELECT * FROM messages ORDER BY {sid_col}, rowid").fetchall()
            by_session: dict[str, list[dict]] = {}
            for row in rows:
                d = dict(row)
                sid = str(d.get(sid_col))
                msg = {"role": d.get("role") or "user", "content": d.get("content") or ""}
                sanitized = _sanitize_tool_calls(d.get("tool_calls"))
                if sanitized:
                    msg["tool_calls"] = sanitized
                if d.get("tool_name"):
                    msg["tool_name"] = d["tool_name"]
                by_session.setdefault(sid, []).append(msg)
                result.n_messages += 1

            for sid, messages in by_session.items():
                result.sessions.append(
                    {
                        "session_id": sid,
                        "source_tool": "hermes",
                        "messages": messages,
                        "metadata": {
                            "session_id": sid,
                            "source": "hermes",
                            "message_count": len(messages),
                        },
                    }
                )
            result.n_sessions = len(result.sessions)
        finally:
            conn.close()
        return result


def import_hermes_sessions(state_db: Path | str = DEFAULT_STATE_DB) -> HermesImportResult:
    """Import all Hermes sessions from ``state_db`` (defaults to ~/.hermes/state.db)."""
    return HermesSessionImporter(state_db).import_sessions()
