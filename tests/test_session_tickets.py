"""Every ticket of a chat, and refreshing an expired one.

`TicketStore.for_session` lists a chat's tickets (id, status, created_at; oldest first) and
`GET /api/sessions/{id}` returns it as `tickets`, so a reloaded chat can show the tickets no turn
mentions (a requote made by Refresh quote or by a confirm). `TicketStore.latest` is the newest
ticket of a chat: `POST /api/tickets/{id}/refresh` re-quotes an expired ticket only when it is that
one (200 requote); an older expired ticket, or one refreshed already, is 409.

The API tests reuse `tests/test_api_chat.py`'s offline world (a scripted model, a fake broker and
fake market data); nothing here touches a network or the project's `.env`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finagent.store.appdb import AppDB
from finagent.trading.models import TicketStatus
from finagent.trading.proposals import TicketStore
from test_api_chat import ApiWorld, confirm, new_session, propose

NOW = datetime(2026, 9, 28, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clean_trading_env(monkeypatch):
    from finagent.app_state import ENV_NAMES

    for name in [*ENV_NAMES.values(), "API_ALLOW_REMOTE"]:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


# --- the store ----------------------------------------------------------------------------------


def store_with(rows: list[tuple[str, str, str, datetime]]) -> TicketStore:
    """A store over raw rows `(id, session_id, status, created_at)`, inserted in this order."""
    db = AppDB(":memory:")
    store = TicketStore(db, clock=lambda: NOW)
    for ticket_id, session_id, status, created in rows:
        db.execute(
            "INSERT INTO tickets (id, session_id, status, body, created_at, expires_at) VALUES (?,?,?,?,?,?)",
            (ticket_id, session_id, status, "{}", created.isoformat(), (created + timedelta(minutes=5)).isoformat()),
        )
    return store


def test_for_session_matches_the_session_exactly():
    store = store_with([
        ("T1", "s_1", "proposed", NOW),
        ("T2", "sa1", "proposed", NOW),   # what `s_1` would match as a LIKE pattern
        ("T3", "S_1", "expired", NOW),
        ("T4", "s_1 ", "expired", NOW),
        ("T5", "s%", "expired", NOW),
        ("T6", "s_1", "superseded", NOW + timedelta(seconds=1)),
    ])
    assert [t["id"] for t in store.for_session("s_1")] == ["T1", "T6"]
    assert [t["id"] for t in store.for_session("s%")] == ["T5"]
    assert store.for_session("nope") == []
    assert store.for_session("s_1")[0] == {"id": "T1", "status": "proposed", "created_at": NOW.isoformat()}


def test_for_session_is_oldest_first_and_ties_keep_the_insertion_order():
    store = store_with([
        ("TC", "s", "proposed", NOW + timedelta(seconds=2)),
        ("TA", "s", "expired", NOW),
        ("TB1", "s", "superseded", NOW + timedelta(seconds=1)),
        ("TB2", "s", "submitted", NOW + timedelta(seconds=1)),
        ("TA2", "s", "expired", NOW + timedelta(microseconds=500)),  # isoformat with and without micros
    ])
    assert [t["id"] for t in store.for_session("s")] == ["TA", "TA2", "TB1", "TB2", "TC"]
    assert store_with([]).latest("s") is None  # `latest` on real tickets: tests/test_execution.py


def test_for_session_keeps_the_newest_tickets_up_to_the_limit():
    store = store_with([(f"T{i:02}", "s", "expired", NOW + timedelta(seconds=i)) for i in range(60)])
    listed = store.for_session("s")
    assert len(listed) == 50
    assert listed[0]["id"] == "T10" and listed[-1]["id"] == "T59"
    assert [t["id"] for t in store.for_session("s", limit=3)] == ["T57", "T58", "T59"]
    assert store.for_session("s", limit=0) == []
    assert store.for_session("s", limit=-1) == []


# --- the API ------------------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path) -> ApiWorld:
    return ApiWorld(tmp_path)


def listed(client, sid: str) -> list[tuple[str, str]]:
    detail = client.get(f"/api/sessions/{sid}").json()
    assert all(set(t) == {"id", "status", "created_at"} for t in detail["tickets"])
    return [(t["id"], t["status"]) for t in detail["tickets"]]


def refresh(client, ticket_id: str, sid: str):
    return client.post(f"/api/tickets/{ticket_id}/refresh", json={"session_id": sid})


def test_session_detail_lists_every_ticket_including_requotes_and_only_its_own(world):
    with world.client() as client:
        sid = new_session(client)
        other = new_session(client)
        assert listed(client, sid) == []
        ticket = propose(client, world, sid)
        theirs = propose(client, world, other)
        world.clock.advance(30)
        r = refresh(client, ticket["id"], sid)
        assert r.status_code == 200, r.text
        requote = r.json()["new_ticket"]
        assert confirm(client, requote, sid).status_code == 200
        # the requote was made by a button, not a chat turn, and still comes back after a reload
        assert listed(client, sid) == [(ticket["id"], "superseded"), (requote["id"], "submitted")]
        assert listed(client, other) == [(theirs["id"], "proposed")]


def test_refresh_requotes_the_newest_ticket_after_it_expired_and_only_once(world):
    world.legs = [{"symbol": "AAPL", "side": "buy", "qty": "10", "type": "market"}]
    with world.client() as client:
        sid = new_session(client)
        ticket = propose(client, world, sid)
        world.clock.advance(301)
        world.data.set_price("AAPL", "201")
        r = refresh(client, ticket["id"], sid)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "requote"
        assert body["ticket"]["status"] == "expired"  # it stays expired
        fresh = body["new_ticket"]
        assert fresh["status"] == "proposed"
        assert client.get(f"/api/sessions/{sid}").json()["active_ticket"]["id"] == fresh["id"]
        # a second refresh of the expired ticket: a newer ticket exists
        again = refresh(client, ticket["id"], sid)
        assert again.status_code == 409, again.text
        assert again.json()["status"] == "rejected" and again.json()["new_ticket"] is None
        assert listed(client, sid) == [(ticket["id"], "expired"), (fresh["id"], "proposed")]
        # confirm and reject of an expired ticket are still 410
        assert confirm(client, ticket, sid).status_code == 410
        assert client.post(f"/api/tickets/{ticket['id']}/reject", json={"session_id": sid}).status_code == 410
        assert world.total_submits() == 0


def test_refresh_of_an_expired_ticket_that_is_not_the_newest_is_409(world):
    with world.client() as client:
        sid = new_session(client)
        old = propose(client, world, sid)
        world.clock.advance(301)
        client.get(f"/api/sessions/{sid}")  # runs expire_stale
        stored = world.runtime.tickets.get(old["id"])
        assert stored.status == TicketStatus.expired
        newer = world.runtime.tickets.create(  # a later proposal of the same chat
            sid, decisions=stored.decisions, thesis=stored.thesis, risks=stored.risks,
            invalidation=stored.invalidation, evidence_ids=stored.evidence_ids,
            mandate_version=stored.mandate_version, quote_as_of=world.clock(), ttl_sec=300,
        )
        r = refresh(client, old["id"], sid)
        assert r.status_code == 409, r.text
        assert r.json()["ticket"]["status"] == "expired" and r.json()["new_ticket"] is None
        assert listed(client, sid) == [(old["id"], "expired"), (newer.id, "proposed")]
        # another chat cannot refresh it at all
        assert refresh(client, old["id"], new_session(client)).status_code == 404
