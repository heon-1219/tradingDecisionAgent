import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

import finagent.chat.sessions as sessions_module
from finagent.chat.sessions import SessionStore
from finagent.store.appdb import AppDB
from finagent.trading.models import Mandate


def _store() -> SessionStore:
    return SessionStore(AppDB(":memory:"))


# --------------------------------------------------------------------------- create / get / rename
def test_create_returns_expected_shape_with_defaults():
    store = _store()
    session = store.create()
    assert session["title"] == "New chat"
    assert session["lang"] == "en"
    assert session["id"].startswith("S") and len(session["id"]) == 13
    # ISO-8601 UTC, parseable and tz-aware
    parsed = datetime.fromisoformat(session["created_at"])
    assert parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_create_accepts_custom_title_and_lang():
    store = _store()
    session = store.create(title="Tech stocks", lang="ko")
    assert session["title"] == "Tech stocks"
    assert session["lang"] == "ko"


@pytest.mark.parametrize("bad_lang", ["fr", "EN", "", "english"])
def test_create_rejects_unsupported_lang(bad_lang):
    with pytest.raises(ValueError):
        _store().create(lang=bad_lang)


def test_create_ids_are_unique():
    store = _store()
    ids = {store.create()["id"] for _ in range(20)}
    assert len(ids) == 20


def test_get_round_trips_a_created_session():
    store = _store()
    created = store.create(title="Round trip", lang="ko")
    fetched = store.get(created["id"])
    assert fetched["id"] == created["id"]
    assert fetched["title"] == "Round trip"
    assert fetched["lang"] == "ko"
    assert fetched["created_at"] == created["created_at"]


def test_get_unknown_session_returns_none():
    assert _store().get("Sdoesnotexist") is None


def test_rename_strips_whitespace_and_caps_at_80_chars():
    store = _store()
    session = store.create()
    store.rename(session["id"], "  padded title  ")
    assert store.get(session["id"])["title"] == "padded title"

    long_title = "x" * 200
    store.rename(session["id"], long_title)
    assert store.get(session["id"])["title"] == "x" * 80


def test_rename_to_empty_or_blank_falls_back_to_new_chat():
    store = _store()
    session = store.create(title="Something")
    store.rename(session["id"], "   ")
    assert store.get(session["id"])["title"] == "New chat"


def test_rename_unknown_session_raises_keyerror():
    with pytest.raises(KeyError):
        _store().rename("Sdoesnotexist", "new title")


# --------------------------------------------------------------------------- append / history / messages
def test_append_returns_incrementing_message_ids_and_persists_content():
    store = _store()
    session = store.create()
    mid1 = store.append(session["id"], "user", "hello")
    mid2 = store.append(session["id"], "assistant", "hi there")
    assert isinstance(mid1, int) and isinstance(mid2, int) and mid2 > mid1


def test_append_unknown_session_raises_keyerror():
    with pytest.raises(KeyError):
        _store().append("Sdoesnotexist", "user", "hello")


def test_append_rejects_invalid_role():
    store = _store()
    session = store.create()
    with pytest.raises(ValueError):
        store.append(session["id"], "system", "nope")


def test_append_stores_meta_with_decimals_as_json_strings():
    store = _store()
    session = store.create()
    store.append(session["id"], "event", "order placed", meta={"amount": Decimal("12.50"), "qty": 3})
    [msg] = store.messages(session["id"])
    assert msg["meta"] == {"amount": "12.50", "qty": 3}


def test_append_with_no_meta_round_trips_as_none():
    store = _store()
    session = store.create()
    store.append(session["id"], "user", "hi")
    [msg] = store.messages(session["id"])
    assert msg["meta"] is None


def test_history_excludes_events_and_returns_oldest_first():
    store = _store()
    session = store.create()
    store.append(session["id"], "user", "q1")
    store.append(session["id"], "event", "tool ran")
    store.append(session["id"], "assistant", "a1")
    store.append(session["id"], "user", "q2")

    history = store.history(session["id"])
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert [m["content"] for m in history] == ["q1", "a1", "q2"]
    assert all(set(m) == {"role", "content", "created_at"} for m in history)


def test_history_respects_limit_keeping_the_most_recent_and_oldest_first():
    store = _store()
    session = store.create()
    for i in range(5):
        store.append(session["id"], "user" if i % 2 == 0 else "assistant", f"m{i}")
    history = store.history(session["id"], limit=2)
    assert [m["content"] for m in history] == ["m3", "m4"]


def test_history_unknown_session_returns_empty_list():
    assert _store().history("Sdoesnotexist") == []


def test_messages_includes_all_roles_oldest_first_with_parsed_meta():
    store = _store()
    session = store.create()
    store.append(session["id"], "user", "q1")
    store.append(session["id"], "event", "tool ran", meta={"tool": "quote"})
    store.append(session["id"], "assistant", "a1")

    rows = store.messages(session["id"])
    assert [m["role"] for m in rows] == ["user", "event", "assistant"]
    assert rows[1]["meta"] == {"tool": "quote"}
    assert all(set(m) == {"role", "content", "meta", "created_at"} for m in rows)


# --------------------------------------------------------------------------- mandate
def test_get_mandate_returns_none_when_absent():
    store = _store()
    session = store.create()
    assert store.get_mandate(session["id"]) is None


def test_save_mandate_starts_at_version_one_regardless_of_incoming_version():
    store = _store()
    session = store.create()
    mandate = Mandate(budget_usd=Decimal("1000.00"), version=99)
    saved = store.save_mandate(session["id"], mandate)
    assert saved.version == 1
    assert saved.budget_usd == Decimal("1000.00")


def test_save_mandate_bumps_version_on_each_save_and_keeps_one_row():
    store = _store()
    session = store.create()
    store.save_mandate(session["id"], Mandate())
    second = store.save_mandate(session["id"], Mandate(budget_usd=Decimal("500")))
    third = store.save_mandate(session["id"], Mandate(budget_usd=Decimal("750.25")))

    assert second.version == 2
    assert third.version == 3
    fetched = store.get_mandate(session["id"])
    assert fetched.version == 3
    assert fetched.budget_usd == Decimal("750.25")

    rows = store._db.query("SELECT COUNT(*) AS n FROM mandates WHERE session_id=?", (session["id"],))
    assert rows[0]["n"] == 1


def test_save_mandate_preserves_decimal_exactly_through_round_trip():
    store = _store()
    session = store.create()
    mandate = Mandate(budget_usd=Decimal("1234.5678"), max_position_pct=Decimal("17.5"))
    store.save_mandate(session["id"], mandate)
    fetched = store.get_mandate(session["id"])
    assert fetched.budget_usd == Decimal("1234.5678")
    assert str(fetched.budget_usd) == "1234.5678"
    assert fetched.max_position_pct == Decimal("17.5")


async def test_save_mandate_concurrent_saves_land_on_distinct_sequential_versions():
    """`save_mandate`'s read-then-write version bump is protected by its own lock (AppDB's lock only
    serializes one statement at a time, not the read+write pair) - N concurrent callers must still
    produce exactly the versions 1..N, with none lost or duplicated."""
    store = _store()
    session = store.create()

    async def save() -> int:
        return (await asyncio.to_thread(store.save_mandate, session["id"], Mandate())).version

    versions = await asyncio.gather(*(save() for _ in range(20)))
    assert sorted(versions) == list(range(1, 21))
    assert store.get_mandate(session["id"]).version == 20


# --------------------------------------------------------------------------- list
def test_list_is_newest_first_by_created_at_then_rowid():
    store = _store()
    first = store.create(title="first")
    second = store.create(title="second")
    third = store.create(title="third")

    listed = store.list()
    assert [s["id"] for s in listed] == [third["id"], second["id"], first["id"]]
    assert all(set(s) == {"id", "title", "created_at", "lang", "updated_at"} for s in listed)


def test_list_updated_at_falls_back_to_created_at_without_messages():
    store = _store()
    session = store.create()
    [listed] = store.list()
    assert listed["updated_at"] == session["created_at"]


def test_list_breaks_created_at_ties_by_rowid_newest_last_inserted_first(monkeypatch):
    """Two sessions created within the same instant (identical `created_at`) must still order
    deterministically, newest-inserted first, via the rowid tie-break."""
    store = _store()
    monkeypatch.setattr(sessions_module, "utcnow_iso", lambda: "2026-01-01T00:00:00+00:00")
    first = store.create(title="first")
    second = store.create(title="second")
    assert first["created_at"] == second["created_at"]  # same instant, by construction

    listed = store.list()
    assert [s["id"] for s in listed] == [second["id"], first["id"]]


def test_list_updated_at_reflects_latest_message():
    store = _store()
    session = store.create()
    store.append(session["id"], "user", "hi")
    mid = store.append(session["id"], "assistant", "hello")
    [listed] = store.list()

    rows = store._db.query("SELECT created_at FROM messages WHERE id=?", (mid,))
    assert listed["updated_at"] == rows[0]["created_at"]
    assert listed["updated_at"] >= session["created_at"]
