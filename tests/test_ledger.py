"""Tests for `finagent.trading.ledger.Ledger` - the append-only, secret-redacting audit ledger."""
from __future__ import annotations

import json
from decimal import Decimal

from finagent.store.appdb import AppDB
from finagent.trading.ledger import Ledger

D = Decimal


def make_ledger() -> tuple[Ledger, AppDB]:
    db = AppDB(":memory:")
    return Ledger(db), db


# --- append / redaction --------------------------------------------------------------------


def test_append_returns_row_id():
    ledger, _db = make_ledger()
    first = ledger.append("sess-1", "note", {"a": 1})
    second = ledger.append("sess-1", "note", {"a": 2})
    assert isinstance(first, int)
    assert second == first + 1


def test_append_stores_payload_as_json_with_decimals_as_strings():
    ledger, db = make_ledger()
    ledger.append("sess-1", "order", {"qty": D("1.5"), "notional": D("100.00")})
    row = db.query("SELECT payload FROM ledger")[0]
    payload = json.loads(row["payload"])
    assert payload == {"qty": "1.5", "notional": "100.00"}


def test_append_redacts_top_level_secret_key():
    ledger, db = make_ledger()
    ledger.append("sess-1", "auth", {"alpaca_api_secret_key": "x"})
    row = db.query("SELECT payload FROM ledger")[0]
    payload = json.loads(row["payload"])
    assert payload == {"alpaca_api_secret_key": "[REDACTED]"}


def test_append_redacts_case_insensitively_and_multiple_patterns():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "auth",
        {
            "API_KEY_ID": "abc",
            "Secret": "shh",
            "Token": "tok",
            "PASSWORD": "pw",
            "Authorization": "Bearer abc",
            "symbol": "AAPL",
        },
    )
    payload = json.loads(db.query("SELECT payload FROM ledger")[0]["payload"])
    assert payload["API_KEY_ID"] == "[REDACTED]"
    assert payload["Secret"] == "[REDACTED]"
    assert payload["Token"] == "[REDACTED]"
    assert payload["PASSWORD"] == "[REDACTED]"
    assert payload["Authorization"] == "[REDACTED]"
    assert payload["symbol"] == "AAPL"


def test_append_redacts_nested_dict_keys():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "request",
        {"headers": {"APCA-API-KEY-ID": "abc", "Content-Type": "application/json"}},
    )
    payload = json.loads(db.query("SELECT payload FROM ledger")[0]["payload"])
    assert payload["headers"]["APCA-API-KEY-ID"] == "[REDACTED]"
    assert payload["headers"]["Content-Type"] == "application/json"


def test_append_redacts_keys_nested_inside_lists():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "batch",
        {"items": [{"secret_key": "s1"}, {"secret_key": "s2"}, {"symbol": "AAPL"}]},
    )
    payload = json.loads(db.query("SELECT payload FROM ledger")[0]["payload"])
    assert payload["items"][0]["secret_key"] == "[REDACTED]"
    assert payload["items"][1]["secret_key"] == "[REDACTED]"
    assert payload["items"][2]["symbol"] == "AAPL"


def test_append_redacts_header_like_string_values_even_under_innocuous_keys():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "log",
        {
            "message": "sent header APCA-API-SECRET-KEY: super-secret-value",
            "other": "sent header APCA-API-KEY-ID: abc123",
            "safe": "no secrets here",
        },
    )
    payload = json.loads(db.query("SELECT payload FROM ledger")[0]["payload"])
    assert payload["message"] == "[REDACTED]"
    assert payload["other"] == "[REDACTED]"
    assert payload["safe"] == "no secrets here"


def test_append_does_not_redact_unrelated_keys():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"symbol": "AAPL", "qty": D("1"), "side": "buy"})
    payload = json.loads(db.query("SELECT payload FROM ledger")[0]["payload"])
    assert payload == {"symbol": "AAPL", "qty": "1", "side": "buy"}


# --- export_jsonl ----------------------------------------------------------------------------


def test_export_jsonl_yields_lines_in_insertion_order():
    ledger, _db = make_ledger()
    ledger.append("sess-1", "note", {"n": 1})
    ledger.append("sess-1", "note", {"n": 2})
    ledger.append("sess-1", "note", {"n": 3})

    lines = list(ledger.export_jsonl("sess-1"))
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["payload"]["n"] for p in parsed] == [1, 2, 3]


def test_export_jsonl_lines_are_valid_json_with_expected_keys():
    ledger, _db = make_ledger()
    row_id = ledger.append("sess-1", "order", {"qty": D("2")})

    lines = list(ledger.export_jsonl("sess-1"))
    entry = json.loads(lines[0])
    assert entry["id"] == row_id
    assert entry["session_id"] == "sess-1"
    assert entry["kind"] == "order"
    assert "created_at" in entry and entry["created_at"]
    assert entry["payload"] == {"qty": "2"}


def test_export_jsonl_only_includes_the_given_session():
    ledger, _db = make_ledger()
    ledger.append("sess-1", "note", {"n": 1})
    ledger.append("sess-2", "note", {"n": 2})

    lines = list(ledger.export_jsonl("sess-1"))
    assert len(lines) == 1
    assert json.loads(lines[0])["session_id"] == "sess-1"


def test_export_jsonl_redacted_payload_round_trips_as_valid_json():
    ledger, _db = make_ledger()
    ledger.append("sess-1", "auth", {"api_key": "abc", "n": 1})

    entry = json.loads(next(ledger.export_jsonl("sess-1")))
    assert entry["payload"] == {"api_key": "[REDACTED]", "n": 1}


def test_export_jsonl_empty_session_yields_nothing():
    ledger, _db = make_ledger()
    assert list(ledger.export_jsonl("sess-none")) == []
