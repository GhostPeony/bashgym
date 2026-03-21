"""SQLite database for user sessions and auth.

Stores GitHub OAuth users and httpOnly session tokens.
Tokens are stored as SHA-256 hashes — raw tokens never touch disk.
"""

import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Default: data/bashgym.db (inside Fly persistent volume mount)
_DB_PATH: Path = Path("data/bashgym.db")

SESSION_MAX_AGE_DAYS = 30


def set_db_path(path: Path) -> None:
    """Override the database path (call before init_db)."""
    global _DB_PATH
    _DB_PATH = path


def init_db() -> None:
    """Create tables if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                github_id   INTEGER UNIQUE NOT NULL,
                username    TEXT NOT NULL,
                display_name TEXT,
                avatar_url  TEXT,
                email       TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                last_login  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash  TEXT PRIMARY KEY,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at  TEXT NOT NULL,
                user_agent  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
        """)


@contextmanager
def get_conn():
    """Context manager for a SQLite connection with WAL mode and foreign keys."""
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_user(
    github_id: int,
    username: str,
    display_name: str | None = None,
    avatar_url: str | None = None,
    email: str | None = None,
) -> int:
    """Insert or update a GitHub user, return the user ID."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO users (github_id, username, display_name, avatar_url, email)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(github_id) DO UPDATE SET
                username     = excluded.username,
                display_name = excluded.display_name,
                avatar_url   = excluded.avatar_url,
                email        = excluded.email,
                last_login   = datetime('now')
            """,
            (github_id, username, display_name, avatar_url, email),
        )
        row = conn.execute("SELECT id FROM users WHERE github_id = ?", (github_id,)).fetchone()
        return row["id"]


def _hash_token(token: str) -> str:
    """SHA-256 hash a raw session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int, user_agent: str | None = None) -> str:
    """Create a new session, return the raw token (caller sets cookie)."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_MAX_AGE_DAYS)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (token_hash, user_id, expires_at, user_agent)
            VALUES (?, ?, ?, ?)
            """,
            (token_hash, user_id, expires_at.isoformat(), user_agent),
        )
    return raw_token


def get_session_user(token: str) -> dict | None:
    """Look up a session token and return the user dict, or None if invalid/expired."""
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.github_id, u.username, u.display_name, u.avatar_url, u.email
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ?
            """,
            (token_hash, now),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def delete_session(token: str) -> None:
    """Invalidate a session by its raw token."""
    token_hash = _hash_token(token)
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def cleanup_expired_sessions() -> int:
    """Delete expired sessions, return count removed."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cursor = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        return cursor.rowcount
