"""Per-session evidence store and untrusted-text handling for research connectors.

Every connector result (EDGAR filings, web search snippets, grounding output) becomes a
`SourceRecord` here before it reaches the model, numbered `E1`, `E2`, ... per session so the
chat layer and the model can both refer to "E3" unambiguously. External text is never trusted
per the project's global constraints: `sanitize_untrusted` strips it down to plain text and
`wrap_untrusted` marks the boundary so the model can tell it apart from the system/user turns.
`EvidenceStore` also doubles as a generic TTL cache (`http_cache`) for connector responses.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..store.appdb import AppDB

# --- untrusted text -----------------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
# Drops an element (tags + contents) whose inline style hides it, e.g. <span style="display:none">.
_HIDDEN_BLOCK_RE = re.compile(
    r"<([a-zA-Z][\w-]*)\b[^>]*\bstyle\s*=\s*[\"'][^\"']*display\s*:\s*none[^\"']*[\"'][^>]*>.*?</\1\s*>",
    re.I | re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")
_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
_WS_RE = re.compile(r"\s+")


def sanitize_untrusted(text: str, max_chars: int = 4000) -> str:
    """Strip markup, hidden/zero-width text and comments out of externally-sourced text."""
    s = text or ""
    s = _COMMENT_RE.sub(" ", s)
    s = _SCRIPT_STYLE_RE.sub(" ", s)
    s = _HIDDEN_BLOCK_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s[:max_chars].rstrip()


# Matches the wrapper's own closing tag (however it's spaced/cased) so injected text can't fake
# the end of the untrusted block; nothing else in the body is touched.
_CLOSE_TAG_RE = re.compile(r"</\s*untrusted\b[^>]*>", re.I)


def wrap_untrusted(source_id: str, text: str) -> str:
    """Wrap sanitized external text as `<untrusted source="...">...</untrusted>`."""
    body = _CLOSE_TAG_RE.sub("&lt;/untrusted&gt;", text or "")
    return f'<untrusted source="{source_id}">{body}</untrusted>'


# --- records --------------------------------------------------------------------


@dataclass
class SourceRecord:
    id: str
    session_id: str
    connector: str
    title: str
    url: str | None
    publisher: str | None
    tier: int
    published_at: str | None
    retrieved_at: str
    snippet: str
    content_sha256: str
    untrusted: bool


@dataclass
class ResearchResult:
    text: str
    sources: list[SourceRecord]
    is_error: bool = False


def _row_to_source(row: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        id=row["id"],
        session_id=row["session_id"],
        connector=row["connector"],
        title=row["title"],
        url=row["url"],
        publisher=row["publisher"],
        tier=row["tier"],
        published_at=row["published_at"],
        retrieved_at=row["retrieved_at"],
        snippet=row["snippet"],
        content_sha256=row["content_sha256"],
        untrusted=bool(row["untrusted"]),
    )


class EvidenceStore:
    """Per-session evidence (backed by `AppDB.evidence`) plus a TTL cache (`AppDB.http_cache`)."""

    def __init__(self, db: AppDB, now: Callable[[], datetime] | None = None):
        self._db = db
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()  # serializes the count-then-insert numbering in add()

    def _timestamp(self) -> str:
        return self._now().isoformat(timespec="seconds")

    def add(
        self,
        session_id: str,
        *,
        connector: str,
        title: str,
        url: str | None,
        publisher: str | None,
        tier: int,
        published_at: str | None,
        snippet: str,
        untrusted: bool,
    ) -> SourceRecord:
        content_sha256 = hashlib.sha256((snippet or "").encode("utf-8")).hexdigest()
        dedupe_url = url or ""
        with self._lock:
            existing = self._db.query(
                "SELECT * FROM evidence WHERE session_id=? AND COALESCE(url,'')=? AND content_sha256=?",
                (session_id, dedupe_url, content_sha256),
            )
            if existing:
                return _row_to_source(existing[0])
            count = self._db.query(
                "SELECT COUNT(*) AS n FROM evidence WHERE session_id=?", (session_id,)
            )[0]["n"]
            source_id = f"E{count + 1}"
            retrieved_at = self._timestamp()
            self._db.execute(
                "INSERT INTO evidence (session_id, id, connector, title, url, publisher, tier, "
                "published_at, retrieved_at, snippet, content_sha256, untrusted) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    source_id,
                    connector,
                    title,
                    url,
                    publisher,
                    tier,
                    published_at,
                    retrieved_at,
                    snippet,
                    content_sha256,
                    int(untrusted),
                ),
            )
        return SourceRecord(
            id=source_id,
            session_id=session_id,
            connector=connector,
            title=title,
            url=url,
            publisher=publisher,
            tier=tier,
            published_at=published_at,
            retrieved_at=retrieved_at,
            snippet=snippet,
            content_sha256=content_sha256,
            untrusted=untrusted,
        )

    def get(self, session_id: str, source_id: str) -> SourceRecord | None:
        rows = self._db.query(
            "SELECT * FROM evidence WHERE session_id=? AND id=?", (session_id, source_id)
        )
        return _row_to_source(rows[0]) if rows else None

    def list(self, session_id: str) -> list[SourceRecord]:
        rows = self._db.query(
            "SELECT * FROM evidence WHERE session_id=? ORDER BY rowid", (session_id,)
        )
        return [_row_to_source(r) for r in rows]

    def cache_get(self, key: str, ttl_sec: float) -> Any | None:
        rows = self._db.query("SELECT payload, stored_at FROM http_cache WHERE key=?", (key,))
        if not rows:
            return None
        stored_at = datetime.fromisoformat(rows[0]["stored_at"])
        age = (self._now() - stored_at).total_seconds()
        if age > ttl_sec:
            return None
        return json.loads(rows[0]["payload"])

    def cache_put(self, key: str, payload: Any) -> None:
        self._db.execute(
            "INSERT INTO http_cache (key, payload, stored_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, stored_at=excluded.stored_at",
            (key, json.dumps(payload), self._timestamp()),
        )
