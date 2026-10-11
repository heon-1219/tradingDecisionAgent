"""SQLite-backed store for chat sessions: session metadata, message history, and the one `Mandate`
in force for a session.

A session id is `"S"` + 12 characters drawn from the RFC 4648 base32 alphabet via `secrets`
(mirroring `trading/proposals.py`'s ticket ids) - cryptographically random and unguessable, so it
is safe to use directly in a URL path segment. Every timestamp is ISO-8601 UTC
(`datetime.now(timezone.utc).isoformat()`), the same format the rest of the app DB uses (see
`test_appdb.py`'s fixtures), so `list()`'s `ORDER BY created_at` sorts correctly as plain text.

`append()` and `save_mandate()` are where session existence is validated on a write: an unknown
`session_id` raises `KeyError` rather than silently inserting an orphan row (neither the `messages`
table nor the `mandates` table has a foreign key on `session_id`). `append()` also raises
`ValueError` for an unrecognized `role`, before anything is written. `history()`, `messages()` and
`get_mandate()` are pure reads and simply return an empty list / `None` for an unknown session,
matching `get()`'s `None` for a single unknown row - there is nothing to validate on a read.

`save_mandate()` always computes the new version from what is currently stored (`previous + 1`,
or `1` if none), discarding whatever `version` the caller's `Mandate` happened to carry - the
caller never gets to pick a version number, only the store does. The read-then-write is protected
by a `threading.Lock` **per `AppDB` instance** (not per `SessionStore`, and not `AppDB`'s own
per-statement lock, which only serializes one statement at a time and would not stop two concurrent
`save_mandate()` calls from both reading version N and both writing N+1): `_mandate_lock_for(db)`
keeps one lock per `AppDB` in a module-level `weakref.WeakKeyDictionary` (guarded by its own small
module-level lock while looking it up/creating it), so every `SessionStore` built over the *same*
`AppDB` - however many instances - shares one lock and two concurrent callers, regardless of which
instance they call through, always land on two distinct, sequential versions. The dict is
weak-keyed so a `SessionStore`/`AppDB` pair that goes out of scope doesn't leak its lock forever.
Storage is a single-row-per-session UPSERT (`INSERT ... ON CONFLICT DO UPDATE`), matching the
`mandates` table's `session_id` primary key.
"""
from __future__ import annotations

import json
import secrets
import threading
import weakref
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from finagent.store.appdb import AppDB
from finagent.trading.models import Mandate

# RFC 4648 base32 alphabet (A-Z, 2-7) - no padding, no ambiguous characters (0/1/8 excluded).
_BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_SESSION_ID_RANDOM_CHARS = 12

_VALID_LANGS = frozenset({"en", "ko"})
_VALID_ROLES = frozenset({"user", "assistant", "event"})
_TITLE_MAX_LEN = 80
_DEFAULT_TITLE = "New chat"


def utcnow_iso() -> str:
    """The current tz-aware UTC time as ISO-8601 (e.g. `2026-01-01T00:00:00.123456+00:00`)."""
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    """`"S"` + 12 characters drawn from the RFC 4648 base32 alphabet via `secrets`."""
    return "S" + "".join(secrets.choice(_BASE32_ALPHABET) for _ in range(_SESSION_ID_RANDOM_CHARS))


def _normalize_title(title: str) -> str:
    """Strip surrounding whitespace and cap at 80 characters; an empty (or all-whitespace) title
    falls back to `"New chat"`."""
    stripped = (title or "").strip()
    if not stripped:
        return _DEFAULT_TITLE
    return stripped[:_TITLE_MAX_LEN]


def _json_default(obj: Any) -> str:
    """`json.dumps(..., default=...)` hook: a `Decimal` in `meta` is stored as its exact string
    form (never a float), per the project's "money is Decimal" constraint."""
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# One `threading.Lock` per `AppDB` instance, shared by every `SessionStore` built over it, so
# `save_mandate()`'s read-then-write version bump is atomic even across multiple `SessionStore`
# instances on the same underlying database (see the module docstring). Weak-keyed so a garbage
# collected `AppDB` takes its lock with it rather than pinning it in this module forever.
_mandate_locks: "weakref.WeakKeyDictionary[AppDB, threading.Lock]" = weakref.WeakKeyDictionary()
_mandate_locks_guard = threading.Lock()


def _mandate_lock_for(db: AppDB) -> threading.Lock:
    """The shared `save_mandate()` lock for `db`, creating it on first use."""
    with _mandate_locks_guard:
        lock = _mandate_locks.get(db)
        if lock is None:
            lock = threading.Lock()
            _mandate_locks[db] = lock
        return lock


class SessionStore:
    """Chat sessions, their message history, and per-session `Mandate` storage over one `AppDB`."""

    def __init__(self, db: AppDB):
        self._db = db
        self._mandate_lock = _mandate_lock_for(db)

    def create(self, title: str = "New chat", lang: str = "en") -> dict:
        """Create a new session. Raises `ValueError` if `lang` is not one of `{"en", "ko"}`."""
        if lang not in _VALID_LANGS:
            raise ValueError(f"unsupported lang {lang!r}; must be one of {sorted(_VALID_LANGS)}")
        session_id = _new_session_id()
        created_at = utcnow_iso()
        normalized_title = _normalize_title(title)
        self._db.execute(
            "INSERT INTO sessions (id, title, lang, created_at) VALUES (?,?,?,?)",
            (session_id, normalized_title, lang, created_at),
        )
        return {"id": session_id, "title": normalized_title, "created_at": created_at, "lang": lang}

    def list(self) -> list[dict]:
        """Every session, newest first (by `created_at`, then `rowid` as a tie-break for sessions
        created within the same instant). Each item is `{'id', 'title', 'created_at', 'lang'}` plus
        `'updated_at'` - the `created_at` of that session's most recent message, or the session's
        own `created_at` if it has none."""
        return self._db.query(
            "SELECT s.id AS id, s.title AS title, s.lang AS lang, s.created_at AS created_at, "
            "COALESCE("
            "  (SELECT MAX(m.created_at) FROM messages m WHERE m.session_id = s.id), s.created_at"
            ") AS updated_at "
            "FROM sessions s "
            "ORDER BY s.created_at DESC, s.rowid DESC"
        )

    def get(self, session_id: str) -> dict | None:
        """`{'id', 'title', 'created_at', 'lang'}` for `session_id`, or `None` if it doesn't exist."""
        rows = self._db.query(
            "SELECT id, title, lang, created_at FROM sessions WHERE id=?", (session_id,)
        )
        return rows[0] if rows else None

    def rename(self, session_id: str, title: str) -> None:
        """Rename a session (stripped and capped at 80 characters; blank falls back to `"New
        chat"`). Raises `KeyError` if `session_id` doesn't exist."""
        normalized = _normalize_title(title)
        cur = self._db.execute("UPDATE sessions SET title=? WHERE id=?", (normalized, session_id))
        if cur.rowcount == 0:
            raise KeyError(f"unknown session: {session_id!r}")

    def append(
        self,
        session_id: str,
        role: Literal["user", "assistant", "event"],
        content: str,
        meta: dict | None = None,
    ) -> int:
        """Append one message and return its id. Raises `ValueError` for a `role` outside
        `{"user", "assistant", "event"}`, and `KeyError` for an unknown `session_id`. `meta`, when
        given, is stored as JSON (a `Decimal` inside it is stored as its exact string form)."""
        if role not in _VALID_ROLES:
            raise ValueError(f"invalid role {role!r}; must be one of {sorted(_VALID_ROLES)}")
        if self.get(session_id) is None:
            raise KeyError(f"unknown session: {session_id!r}")
        meta_json = None if meta is None else json.dumps(meta, default=_json_default)
        cur = self._db.execute(
            "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
            (session_id, role, content, meta_json, utcnow_iso()),
        )
        return cur.lastrowid

    def history(self, session_id: str, limit: int = 12) -> list[dict]:
        """The last `limit` `user`/`assistant` messages for `session_id`, oldest first, each
        `{'role', 'content', 'created_at'}` (`event` rows are excluded). An unknown session simply
        has no rows, so this returns `[]`."""
        rows = self._db.query(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id=? AND role IN ('user', 'assistant') "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows.reverse()
        return rows

    def messages(self, session_id: str, limit: int = 50) -> list[dict]:
        """The last `limit` messages of *any* role (including `event`) for `session_id`, oldest
        first, each `{'role', 'content', 'meta', 'created_at'}` with `meta` parsed back from JSON
        (`None` if it was never set)."""
        rows = self._db.query(
            "SELECT role, content, meta, created_at FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows.reverse()
        for row in rows:
            row["meta"] = None if row["meta"] is None else json.loads(row["meta"])
        return rows

    def user_texts(self, session_id: str) -> list[str]:
        """The text of every `user` message of `session_id`, oldest first (`[]` for an unknown
        session). What the user wrote, never the assistant: `fetch_source` opens a link only when
        the user gave it (or a research tool returned it)."""
        rows = self._db.query(
            "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id",
            (session_id,),
        )
        return [row["content"] or "" for row in rows]

    def get_mandate(self, session_id: str) -> Mandate | None:
        """The `Mandate` currently stored for `session_id`, or `None` if none has been saved yet -
        including when `session_id` itself doesn't exist (a pure read, like `history`/`messages`;
        it does not raise)."""
        rows = self._db.query("SELECT body FROM mandates WHERE session_id=?", (session_id,))
        if not rows:
            return None
        return Mandate.model_validate_json(rows[0]["body"])

    def save_mandate(self, session_id: str, mandate: Mandate) -> Mandate:
        """Store `mandate` for `session_id`, always at version `previous + 1` (`1` if none existed
        yet) regardless of `mandate.version` - the caller's version is never trusted. Keeps exactly
        one row per session (UPSERT). Returns the stored `Mandate` (i.e. with the bumped version).
        Raises `KeyError` for an unknown `session_id` (like `append`/`rename`) rather than attaching
        an orphan mandate row to a session that was never created."""
        if self.get(session_id) is None:
            raise KeyError(f"unknown session: {session_id!r}")
        with self._mandate_lock:
            rows = self._db.query("SELECT version FROM mandates WHERE session_id=?", (session_id,))
            new_version = (rows[0]["version"] + 1) if rows else 1
            stored = mandate.model_copy(update={"version": new_version})
            body = stored.model_dump_json()
            self._db.execute(
                "INSERT INTO mandates (session_id, version, body, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "  version=excluded.version, body=excluded.body, updated_at=excluded.updated_at",
                (session_id, new_version, body, utcnow_iso()),
            )
            return stored
