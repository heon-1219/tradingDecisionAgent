import asyncio

from finagent.store.appdb import AppDB

_EXPECTED_TABLES = {
    "sessions",
    "messages",
    "mandates",
    "tickets",
    "orders",
    "ledger",
    "evidence",
    "http_cache",
}


def test_schema_created_twice_without_error(tmp_path):
    path = tmp_path / "store" / "app.db"
    AppDB(path)
    AppDB(path)  # re-opening the same file must not raise


def test_memory_db_creates_all_tables():
    db = AppDB(":memory:")
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert _EXPECTED_TABLES <= tables


def test_execute_and_query_roundtrip():
    db = AppDB(":memory:")
    db.execute(
        "INSERT INTO sessions (id, title, lang, created_at) VALUES (?,?,?,?)",
        ("s1", "hello", "en", "2026-01-01T00:00:00+00:00"),
    )
    rows = db.query("SELECT id, title FROM sessions WHERE id=?", ("s1",))
    assert rows == [{"id": "s1", "title": "hello"}]


def test_execute_returns_cursor_with_lastrowid():
    db = AppDB(":memory:")
    cur = db.execute(
        "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
        ("s1", "user", "hi", "{}", "2026-01-01T00:00:00+00:00"),
    )
    assert cur.lastrowid == 1


async def test_concurrent_execute_does_not_corrupt_data():
    db = AppDB(":memory:")

    async def insert(i: int) -> None:
        await asyncio.to_thread(
            db.execute,
            "INSERT INTO ledger (session_id, kind, payload, created_at) VALUES (?,?,?,?)",
            ("s1", "note", f"payload-{i}", "2026-01-01T00:00:00+00:00"),
        )

    await asyncio.gather(*(insert(i) for i in range(50)))

    rows = db.query("SELECT COUNT(*) AS n FROM ledger WHERE session_id=?", ("s1",))
    assert rows[0]["n"] == 50
    payloads = {r["payload"] for r in db.query("SELECT payload FROM ledger WHERE session_id=?", ("s1",))}
    assert payloads == {f"payload-{i}" for i in range(50)}
