"""Guarded, read-only SQL access to the SQLite product store."""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


class SQLGuardError(ValueError):
    pass


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[dict]
    row_count: int
    truncated: bool


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace|reindex|load_extension)\b", re.I
)


class RDB:
    def __init__(self, path: Path, row_limit: int = 200, timeout_sec: float = 5.0):
        self.path, self.row_limit, self.timeout_sec = Path(path), row_limit, timeout_sec
        if not self.path.exists():
            raise FileNotFoundError(f"store not found: {self.path} (run scripts/build_store.py)")
        self._con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    # --- guard ------------------------------------------------------------
    def guard(self, sql: str) -> str:
        s = (sql or "").strip().rstrip(";").strip()
        s = re.sub(r"^```(?:sql)?\s*|\s*```$", "", s, flags=re.I).strip()
        if not s:
            raise SQLGuardError("Empty SQL.")
        if ";" in s:
            raise SQLGuardError("Only one SELECT statement can be run at a time.")
        if not re.match(r"^(select|with)\b", s, re.I):
            raise SQLGuardError("Only SELECT/WITH statements are allowed.")
        if _FORBIDDEN.search(s):
            raise SQLGuardError("Only read-only queries are allowed.")
        m = re.search(r"\blimit\s+(\d+)\s*$", s, re.I)
        if m is None:
            s = f"{s} LIMIT {self.row_limit + 1}"
        elif int(m.group(1)) > self.row_limit:
            s = re.sub(r"\blimit\s+\d+\s*$", f"LIMIT {self.row_limit + 1}", s, flags=re.I)
        return s

    # --- execution --------------------------------------------------------
    def query(self, sql: str) -> QueryResult:
        s = self.guard(sql)
        deadline = time.monotonic() + self.timeout_sec
        with self._lock:
            self._con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 5000)
            try:
                cur = self._con.execute(s)
                cols = [d[0] for d in cur.description]
                rows = [dict(r) for r in cur.fetchall()]
            except sqlite3.OperationalError as e:
                if "interrupted" in str(e).lower():
                    raise SQLGuardError(f"Query timed out ({self.timeout_sec:.0f}s). Narrow the conditions.") from e
                raise SQLGuardError(f"SQL error: {e}") from e
            except sqlite3.DatabaseError as e:
                raise SQLGuardError(f"SQL error: {e}") from e
            finally:
                self._con.set_progress_handler(None, 0)
        truncated = len(rows) > self.row_limit
        rows = rows[: self.row_limit]
        # show the caller's SQL (our +1 probe limit is an implementation detail)
        original = (sql or "").strip().rstrip(";").strip()
        original = re.sub(r"^```(?:sql)?\s*|\s*```$", "", original, flags=re.I).strip()
        shown = original
        if truncated:
            if re.search(r"\blimit\s+\d+\s*$", original, re.I):
                shown = re.sub(r"\blimit\s+\d+\s*$", f"LIMIT {self.row_limit}", original, flags=re.I)
            else:
                shown = f"{original} LIMIT {self.row_limit}"
        return QueryResult(shown, cols, rows, len(rows), truncated)

    def fetch_all(self, sql: str, params=()) -> list[tuple]:
        with self._lock:
            return [tuple(r) for r in self._con.execute(sql, params).fetchall()]

    def scalar(self, sql: str, params=()):
        rows = self.fetch_all(sql, params)
        return rows[0][0] if rows else None

    # --- catalog ----------------------------------------------------------
    def views(self) -> list[str]:
        return [r[0] for r in self.fetch_all("select name from sqlite_master where type='view' order by name")]

    def columns(self, view: str) -> list[str]:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", view):
            raise SQLGuardError("Invalid table name")
        return [r[1] for r in self.fetch_all(f"pragma table_info({view})")]

    def meta(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        try:
            rows = self.fetch_all("select table_name, source_id, korean_name, base_date, row_count from meta")
        except sqlite3.DatabaseError:
            return out
        for t, sid, k, b, n in rows:
            out[f"v_{t}"] = {"source_id": sid, "korean_name": k, "base_date": b, "row_count": n}
        return out
