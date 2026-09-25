"""SQLite app DB for the paper-trading assistant: sessions, tickets, orders, evidence, cache.

A single writer lock serializes every statement (mirroring `kb/rdb.py`'s pattern), so the chat,
trading and research layers can share one small on-disk DB without worrying about SQLite's
own concurrent-write restrictions. Reads and writes both go through this lock; the schema uses
plain `CREATE TABLE IF NOT EXISTS` so later tasks can add columns with their own migrations
without this class knowing about them.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    lang TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT,
    content TEXT,
    meta TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS mandates (
    session_id TEXT PRIMARY KEY,
    version INTEGER,
    body TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    status TEXT,
    body TEXT,
    created_at TEXT,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    client_order_id TEXT UNIQUE,
    broker_order_id TEXT,
    body TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    kind TEXT,
    payload TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    session_id TEXT NOT NULL,
    id TEXT NOT NULL,
    connector TEXT,
    title TEXT,
    url TEXT,
    publisher TEXT,
    tier INTEGER,
    published_at TEXT,
    retrieved_at TEXT,
    snippet TEXT,
    content_sha256 TEXT,
    untrusted INTEGER,
    PRIMARY KEY (session_id, id)
);

CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    payload TEXT,
    stored_at TEXT
);
"""


class AppDB:
    """Thin, lock-serialized wrapper around one sqlite3 connection."""

    def __init__(self, path: Path | str):
        self._is_memory = str(path) == ":memory:"
        if not self._is_memory:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            if not self._is_memory:
                # WAL lets readers proceed while a write is in progress; irrelevant (and a no-op)
                # for ":memory:" databases, which don't support WAL.
                self._con.execute("PRAGMA journal_mode=WAL")
            self._con.executescript(_SCHEMA)
            self._con.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run one write/DDL statement. Serialized; commits before returning."""
        with self._lock:
            cur = self._con.execute(sql, params)
            self._con.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run one read statement and return rows as plain dicts."""
        with self._lock:
            cur = self._con.execute(sql, params)
            rows = cur.fetchall()
        return [dict(r) for r in rows]
