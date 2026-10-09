"""Tests for `GET /api/sources/{session_id}` (`finagent.api_sources`): the evidence records behind
the web app's source chips.

The app is the real one (`create_app`) around the offline fake world's real `ChatRuntime`
(`scripts/_fakeworld.build_fake_world` with the scripted `DemoLLM`), so the route is exercised behind
the real guard and next to the real chat routes. Nothing here touches a network or the project's
`.env`.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import _fakeworld as fw  # noqa: E402

from finagent.api import create_app  # noqa: E402
from finagent.api_sources import SNIPPET_MAX  # noqa: E402

WRITE = {"X-FinAgent-Client": "1"}
FIELDS = {"id", "title", "url", "publisher", "tier", "published_at", "connector", "untrusted", "snippet"}


@pytest.fixture
def demo(tmp_path):
    world = fw.build_fake_world(llm=fw.DemoLLM(), tmp_dir=tmp_path)
    state = SimpleNamespace(settings=world.settings, chat=world.runtime)
    with TestClient(create_app(lambda: state), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        yield client, world
    fw.release_sqlite(world.runtime.db, getattr(world.registry, "rdb", None))


def events_of(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_an_unknown_session_is_404(demo):
    client, _world = demo
    r = client.get("/api/sources/nope")
    assert r.status_code == 404
    assert r.json() == {"error": "session_not_found", "message": "This chat session was not found."}


def test_a_new_session_has_no_sources(demo):
    client, _world = demo
    sid = client.post("/api/sessions", json={}, headers=WRITE).json()["id"]
    r = client.get(f"/api/sources/{sid}")
    assert r.status_code == 200 and r.json() == {"sources": []}
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["x-frame-options"] == "DENY"  # the guard's headers cover this route too


def test_records_come_back_in_order_with_exactly_the_documented_fields(demo):
    client, world = demo
    sid = world.sessions.create(title="a")["id"]
    other = world.sessions.create(title="b")["id"]
    long_snippet = "x" * (SNIPPET_MAX + 300)
    world.evidence.add(sid, connector="edgar", title="10-K 2025", url="https://www.sec.gov/a", publisher="SEC",
                       tier=1, published_at="2026-02-01", snippet=long_snippet, untrusted=False)
    world.evidence.add(sid, connector="news", title="<b>Headline</b> “quoted” 한글", url=None, publisher=None,
                       tier=2, published_at=None, snippet="short", untrusted=True)
    world.evidence.add(other, connector="web", title="Other chat", url="https://example.com/x", publisher="Ex",
                       tier=3, published_at=None, snippet="s", untrusted=True)

    body = client.get(f"/api/sources/{sid}").json()
    assert [s["id"] for s in body["sources"]] == ["E1", "E2"]
    first, second = body["sources"]
    assert set(first) == FIELDS and set(second) == FIELDS  # no session id, hash or retrieval time
    assert first == {"id": "E1", "title": "10-K 2025", "url": "https://www.sec.gov/a", "publisher": "SEC", "tier": 1,
                     "published_at": "2026-02-01", "connector": "edgar", "untrusted": False,
                     "snippet": "x" * SNIPPET_MAX}
    assert second["title"] == "<b>Headline</b> “quoted” 한글"  # as stored: the page renders it as text
    assert second["url"] is None and second["publisher"] is None and second["untrusted"] is True
    # another chat's sources are not listed, and ids are per chat
    assert [s["title"] for s in client.get(f"/api/sources/{other}").json()["sources"]] == ["Other chat"]


def test_every_source_cited_by_a_chat_turn_is_listed(demo):
    client, world = demo
    sid = client.post("/api/sessions", json={"title": "core"}, headers=WRITE).json()["id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "Build a low-cost core ETF portfolio with $5,000."},
                    headers=WRITE)
    assert r.status_code == 200
    events = events_of(r)
    answer = "".join(e["delta"] for e in events if e["type"] == "text-delta")
    cited = set(re.findall(r"\[(E\d+)\]", answer))
    ticket = next(e["ticket"] for e in events if e["type"] == "data-trade-proposal")
    listed = {s["id"]: s for s in client.get(f"/api/sources/{sid}").json()["sources"]}
    assert cited and cited <= set(listed)
    assert set(ticket["evidence_ids"]) <= set(listed)
    # what the stream said about a source matches what the endpoint says, and the endpoint adds the rest
    streamed = {s["id"]: s for e in events if e["type"] == "tool-output-available" for s in e["sources"]}
    assert streamed
    for sid_, source in streamed.items():
        assert {k: listed[sid_][k] for k in ("title", "url", "publisher", "tier")} == {
            k: source[k] for k in ("title", "url", "publisher", "tier")}
        assert listed[sid_]["snippet"] and len(listed[sid_]["snippet"]) <= SNIPPET_MAX
    assert world.broker.submits == 0


def test_the_route_sits_behind_the_local_guard(demo):
    client, world = demo
    sid = world.sessions.create()["id"]
    assert client.get(f"/api/sources/{sid}", headers={"Host": "evil.example"}).status_code == 400
    with TestClient(client.app, base_url="http://127.0.0.1", client=("203.0.113.9", 50000)) as remote:
        r = remote.get(f"/api/sources/{sid}")
        assert r.status_code == 403 and r.json()["error"] == "local_only"
    # read-only: there is no write method on the path
    assert client.post(f"/api/sources/{sid}", json={}, headers=WRITE).status_code == 405
    assert client.delete(f"/api/sources/{sid}", headers=WRITE).status_code == 405


def test_an_over_long_session_id_is_a_validation_error_not_a_lookup(demo):
    client, _world = demo
    r = client.get("/api/sources/" + "a" * 65)
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


def test_without_a_chat_runtime_the_route_answers_503():
    state = SimpleNamespace(chat=None, settings=SimpleNamespace(api_allow_remote=False))
    with TestClient(create_app(lambda: state), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        r = client.get("/api/sources/anything")
        assert r.status_code == 503 and r.json()["error"] == "chat_unavailable"
