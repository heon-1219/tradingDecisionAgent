"""Tests for `finagent.trading.ledger.Ledger` - the append-only, secret-redacting audit ledger."""
from __future__ import annotations

import json
import time
from decimal import Decimal

from finagent.trading.models import OrderLeg
from finagent.store.appdb import AppDB
from finagent.trading.ledger import Ledger, _redact_free_text

D = Decimal


def make_ledger() -> tuple[Ledger, AppDB]:
    db = AppDB(":memory:")
    return Ledger(db), db


def _last_payload(db: AppDB) -> dict:
    row = db.query("SELECT payload FROM ledger ORDER BY id DESC LIMIT 1")[0]
    return json.loads(row["payload"])


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
    # Surgical redaction (fix round 2): the header *name* stays visible, only its value is hidden.
    assert payload["message"] == "sent header APCA-API-SECRET-KEY: [REDACTED]"
    assert payload["other"] == "sent header APCA-API-KEY-ID: [REDACTED]"
    assert payload["safe"] == "no secrets here"


def test_append_does_not_redact_unrelated_keys():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"symbol": "AAPL", "qty": D("1"), "side": "buy"})
    payload = json.loads(db.query("SELECT payload FROM ledger")[0]["payload"])
    assert payload == {"symbol": "AAPL", "qty": "1", "side": "buy"}


def test_append_redacts_credential_and_cookie_keys():
    ledger, db = make_ledger()
    ledger.append("sess-1", "auth", {"credentials": "LEAK", "cookie": "LEAK", "Session-Cookie": "LEAK"})
    payload = _last_payload(db)
    assert payload == {"credentials": "[REDACTED]", "cookie": "[REDACTED]", "Session-Cookie": "[REDACTED]"}


def test_append_redacts_underscore_and_alpaca_spelled_header_env_values():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "log",
        {
            "a": "env APCA_API_SECRET_KEY=shh",
            "b": "env APCA_API_KEY_ID=shh",
            "c": "env ALPACA_API_SECRET_KEY=shh",
            "d": "env ALPACA_API_KEY_ID=shh",
            "e": "no secret here",
        },
    )
    payload = _last_payload(db)
    # Surgical redaction (fix round 2): the env var name stays visible, only its value is hidden.
    assert payload["a"] == "env APCA_API_SECRET_KEY=[REDACTED]"
    assert payload["b"] == "env APCA_API_KEY_ID=[REDACTED]"
    assert payload["c"] == "env ALPACA_API_SECRET_KEY=[REDACTED]"
    assert payload["d"] == "env ALPACA_API_KEY_ID=[REDACTED]"
    assert payload["e"] == "no secret here"


def test_append_redacts_secret_dicts_nested_inside_a_tuple():
    ledger, db = make_ledger()
    ledger.append("sess-1", "batch", {"items": ({"secret_key": "LEAK"},)})
    payload = _last_payload(db)
    assert payload == {"items": [{"secret_key": "[REDACTED]"}]}


def test_append_redacts_a_header_represented_as_a_list_of_kv_pairs():
    ledger, db = make_ledger()
    ledger.append("sess-1", "request", {"headers": [("APCA-API-SECRET-KEY", "shh"), ("Accept", "json")]})
    payload = _last_payload(db)
    assert payload == {"headers": [["APCA-API-SECRET-KEY", "[REDACTED]"], ["Accept", "json"]]}


def test_append_redacts_a_tuple_containing_a_header_like_string():
    ledger, db = make_ledger()
    ledger.append("sess-1", "log", {"log": ("APCA-API-SECRET-KEY: shh",)})
    payload = _last_payload(db)
    # A single compound "name: value" string is treated as a name (kept), and free-text-scanned
    # for its own embedded value (redacted) - see test_append_redacts_header_like_string_values...
    assert payload == {"log": ["APCA-API-SECRET-KEY: [REDACTED]"]}


def test_append_a_2_item_pair_without_a_matching_name_is_not_treated_as_kv():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"pair": ["side", "buy"]})
    payload = _last_payload(db)
    assert payload == {"pair": ["side", "buy"]}


def test_append_never_raises_on_a_bytes_value():
    ledger, db = make_ledger()
    row_id = ledger.append("sess-1", "note", {"x": b"hello"})
    assert isinstance(row_id, int)
    payload = _last_payload(db)
    assert payload == {"x": "<bytes len=5>"}


def test_append_never_raises_on_a_non_str_dict_key():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {D("1"): "a"})
    payload = _last_payload(db)
    assert payload == {"1": "a"}


def test_append_never_raises_on_mixed_key_types():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {1: "a", "b": 2})
    payload = _last_payload(db)
    assert payload == {"1": "a", "b": 2}


def test_append_never_raises_on_a_set_value():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"x": {1, 2, 3}})
    payload = _last_payload(db)
    assert set(payload["x"]) == {1, 2, 3}


def test_append_never_raises_on_a_pydantic_model_value_and_still_redacts_it():
    ledger, db = make_ledger()
    leg = OrderLeg(symbol="AAPL", side="buy", qty=D("1"), type="market")
    ledger.append("sess-1", "note", {"leg": leg})
    payload = _last_payload(db)
    assert payload["leg"]["symbol"] == "AAPL"
    assert payload["leg"]["qty"] == "1"


def test_append_never_raises_on_a_tuple_containing_a_decimal():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"x": (D("1.5"),)})
    payload = _last_payload(db)
    assert payload == {"x": ["1.5"]}


def test_append_falls_back_to_a_placeholder_when_redaction_itself_fails(monkeypatch):
    import finagent.trading.ledger as ledger_module

    def _boom(_value):
        raise RuntimeError("simulated redaction bug")

    monkeypatch.setattr(ledger_module, "_redact", _boom)
    ledger, db = make_ledger()
    row_id = ledger.append("sess-1", "note", {"symbol": "AAPL"})
    assert isinstance(row_id, int)
    payload = _last_payload(db)
    assert payload == {"_ledger_error": "payload could not be serialized"}


# --- fix round 2: header-shaped values (HAR/Postman, flat lists, 3-tuples) --------------------


def test_append_redacts_a_har_style_name_value_header_entry():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "request",
        {"headers": [{"name": "APCA-API-SECRET-KEY", "value": "shh"}]},
    )
    payload = _last_payload(db)
    assert payload == {"headers": [{"name": "APCA-API-SECRET-KEY", "value": "[REDACTED]"}]}


def test_append_redacts_a_postman_style_key_value_header_entry():
    """The Postman convention uses "key" (not "name") for the header's name field; it must be
    kept visible exactly like HAR's "name" field, even though a bare "key" dict key would
    otherwise match the broader dict-key secret pattern."""
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "request",
        {"headers": [{"key": "APCA-API-SECRET-KEY", "value": "shh"}]},
    )
    payload = _last_payload(db)
    assert payload == {"headers": [{"key": "APCA-API-SECRET-KEY", "value": "[REDACTED]"}]}


def test_append_does_not_treat_an_ordinary_header_entry_as_secret():
    ledger, db = make_ledger()
    ledger.append("sess-1", "request", {"headers": [{"name": "Accept", "value": "application/json"}]})
    payload = _last_payload(db)
    assert payload == {"headers": [{"name": "Accept", "value": "application/json"}]}


def test_append_redacts_a_flat_four_item_header_list():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "request",
        {"h": ["APCA-API-KEY-ID", "id-value", "APCA-API-SECRET-KEY", "secret-value"]},
    )
    payload = _last_payload(db)
    assert payload == {"h": ["APCA-API-KEY-ID", "[REDACTED]", "APCA-API-SECRET-KEY", "[REDACTED]"]}


def test_append_redacts_only_the_element_immediately_after_a_header_name_in_a_3_tuple():
    ledger, db = make_ledger()
    ledger.append("sess-1", "request", {"h": [("APCA-API-SECRET-KEY", "shh", "unaffected")]})
    payload = _last_payload(db)
    assert payload == {"h": [["APCA-API-SECRET-KEY", "[REDACTED]", "unaffected"]]}


def test_append_redacts_a_header_name_pair_with_spaces():
    ledger, db = make_ledger()
    ledger.append("sess-1", "request", {"h": [("APCA API SECRET KEY", "shh")]})
    payload = _last_payload(db)
    assert payload == {"h": [["APCA API SECRET KEY", "[REDACTED]"]]}


def test_append_bytes_dict_key_never_leaks_raw_content_via_repr():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {b"LEAKrawbytes": 1})
    payload = _last_payload(db)
    assert "LEAKrawbytes" not in json.dumps(payload)
    assert list(payload.keys()) == ["<bytes len=12>"]


# --- fix round 2: free-text scanning (query params, Authorization/Bearer, name=value) ----------


def test_append_redacts_a_query_string_apikey_parameter():
    ledger, db = make_ledger()
    ledger.append("sess-1", "http", {"url": "https://h/v2/x?apikey=shh"})
    payload = _last_payload(db)
    assert payload == {"url": "https://h/v2/x?apikey=[REDACTED]"}


def test_append_redacts_a_query_string_key_parameter():
    ledger, db = make_ledger()
    ledger.append("sess-1", "http", {"error": "401 for url 'https://g/v1?key=shh'"})
    payload = _last_payload(db)
    assert payload == {"error": "401 for url 'https://g/v1?key=[REDACTED]'"}


def test_append_redacts_authorization_bearer_in_free_text():
    ledger, db = make_ledger()
    ledger.append("sess-1", "log", {"msg": "Authorization: Bearer shh"})
    payload = _last_payload(db)
    assert payload == {"msg": "Authorization: [REDACTED]"}


def test_append_redacts_a_bare_bearer_token_in_free_text():
    ledger, db = make_ledger()
    ledger.append("sess-1", "log", {"msg": "sending with Bearer shh now"})
    payload = _last_payload(db)
    assert payload == {"msg": "sending with Bearer [REDACTED] now"}


def test_append_redacts_a_generic_token_equals_value_in_free_text():
    ledger, db = make_ledger()
    ledger.append("sess-1", "log", {"msg": "token=shh"})
    payload = _last_payload(db)
    assert payload == {"msg": "token=[REDACTED]"}


# --- fix round 2: documented over-/under-redaction ---------------------------------------------


def test_append_does_not_treat_a_bare_ticker_key_as_a_pair_name():
    """A bare "KEY"/"KEYS" is a plausible stock ticker, not a credential-header name; it must not
    trigger the [name, value] pair heuristic (unlike a dict key literally named "key" - see the
    documented over-redaction test below, which is a different code path)."""
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"symbols": ["KEY", "AAPL"]})
    payload = _last_payload(db)
    assert payload == {"symbols": ["KEY", "AAPL"]}


def test_append_does_not_treat_an_unrelated_word_containing_key_as_a_pair_name():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"x": ["monkey", "banana"]})
    payload = _last_payload(db)
    assert payload == {"x": ["monkey", "banana"]}


def test_append_documented_over_redaction_of_a_dict_keyed_by_the_ticker_key():
    """Documented, accepted limitation (fix round 2, item (c)): dict-KEY redaction stays broad
    (any key containing "key", "secret", etc.), so a dict literally keyed by the ticker symbol
    "KEY" still has its value redacted wholesale. Task 2.4 keeps trading symbols out of dict
    keys, so this is not expected to occur in practice; this test documents the current, accepted
    behavior rather than asserting it as desirable."""
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"positions": {"KEY": {"qty": "10"}}})
    payload = _last_payload(db)
    assert payload == {"positions": {"KEY": "[REDACTED]"}}


# --- fix round 3: free-text ReDoS fix (linear-time scan) --------------------------------------
#
# The old `_FREE_TEXT_KV_RE` nested an unbounded name-charset quantifier around a late
# alternation - `[\w.\-]*(?:secret|token|...)[\w.\-]*` - which is catastrophically slow
# (quadratic-or-worse) on an adversarial string with no match at the end. Every case below is an
# instance of exactly that shape (a long run of name-charset characters, or a long run of
# separators, with nothing for the pattern to ultimately match) and must complete comfortably
# under the reviewer's 0.5s bar - measured here, not just asserted, since a regression here is a
# production hang, not a wrong answer.

_REDOS_TIME_BUDGET_SEC = 0.5


def _time_redact_free_text(text: str) -> float:
    start = time.perf_counter()
    _redact_free_text(text)
    return time.perf_counter() - start


def test_redact_free_text_handles_100k_dotted_run_quickly():
    assert _time_redact_free_text("a." * 50_000) < _REDOS_TIME_BUDGET_SEC  # 100,000 chars


def test_redact_free_text_handles_100k_hyphenated_run_quickly():
    assert _time_redact_free_text("a-" * 50_000) < _REDOS_TIME_BUDGET_SEC  # 100,000 chars


def test_redact_free_text_handles_dotted_traceback_shape_quickly():
    trace = "finagent.trading.proposals.TicketStore.transition." * 400  # ~20,000 chars
    assert _time_redact_free_text(trace) < _REDOS_TIME_BUDGET_SEC


def test_redact_free_text_handles_repeated_secret_equals_quickly():
    assert _time_redact_free_text("secret=" * 20_000) < _REDOS_TIME_BUDGET_SEC  # 140,000 chars


def test_redact_free_text_handles_long_runs_of_separators_quickly():
    assert _time_redact_free_text("=" * 100_000) < _REDOS_TIME_BUDGET_SEC
    assert _time_redact_free_text(":" * 100_000) < _REDOS_TIME_BUDGET_SEC


def test_redact_free_text_scales_roughly_linearly_not_quadratically():
    """Doubling the input should roughly double the time (allow generous slack for timing noise),
    not roughly quadruple it - the signature of the quadratic behavior this fix removes."""
    small = _time_redact_free_text("a." * 20_000)  # 40,000 chars
    large = _time_redact_free_text("a." * 80_000)  # 160,000 chars, 4x the input
    # A tiny `small` measurement (e.g. rounds to ~0) would make the ratio meaningless; guard it.
    assert small > 0
    assert large / small < 8  # linear predicts ~4x; quadratic would predict ~16x


def test_append_end_to_end_handles_a_huge_adversarial_payload_quickly():
    """The same guarantee, exercised through the public `append()` entry point rather than the
    internal helper directly."""
    ledger, _db = make_ledger()
    start = time.perf_counter()
    ledger.append("sess-1", "error", {"traceback": "a." * 50_000})
    assert time.perf_counter() - start < _REDOS_TIME_BUDGET_SEC


def test_redact_free_text_truncates_an_oversized_string_leaf():
    ledger, db = make_ledger()
    huge = "x" * 150_000
    ledger.append("sess-1", "note", {"blob": huge})
    payload = _last_payload(db)
    assert payload["blob"].startswith("x" * 100_000)
    assert "truncated 50000 chars" in payload["blob"]
    assert len(payload["blob"]) < len(huge)


# --- fix round 3: free-text over-redaction of ordinary prose -----------------------------------


def test_append_does_not_redact_a_titlecase_word_starting_ordinary_prose():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"msg": "Secret: Apple's new product launch delayed"})
    payload = _last_payload(db)
    assert payload == {"msg": "Secret: Apple's new product launch delayed"}


def test_append_does_not_redact_other_titlecase_labels_starting_prose():
    ledger, db = make_ledger()
    ledger.append(
        "sess-1", "note",
        {
            "a": "Token: some note about tokens",
            "b": "Password: a note about passwords in general",
            "c": "Signature: John Hancock signed first",
        },
    )
    payload = _last_payload(db)
    assert payload["a"] == "Token: some note about tokens"
    assert payload["b"] == "Password: a note about passwords in general"
    assert payload["c"] == "Signature: John Hancock signed first"


def test_append_redacts_lower_case_password_equals_in_free_text():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"msg": "password=hunter2"})
    payload = _last_payload(db)
    assert payload == {"msg": "password=[REDACTED]"}


def test_append_redacts_upper_snake_case_api_key_colon_in_free_text():
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"msg": "API_KEY: abc123"})
    payload = _last_payload(db)
    assert payload == {"msg": "API_KEY: [REDACTED]"}


def test_append_redacts_hyphenated_titlecase_x_api_key_in_free_text():
    """Even though "X-Api-Key" is Titlecase per-word, it contains a hyphen, so it's still
    identifier-shaped (unlike a single Titlecase word like "Secret")."""
    ledger, db = make_ledger()
    ledger.append("sess-1", "note", {"msg": "X-Api-Key: abc123"})
    payload = _last_payload(db)
    assert payload == {"msg": "X-Api-Key: [REDACTED]"}


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
