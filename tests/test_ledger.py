"""Tests for `finagent.trading.ledger.Ledger` - the append-only, secret-redacting audit ledger."""
from __future__ import annotations

import json
import time
from decimal import Decimal

import pytest

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


# --- fix round 4: separator-driven free-text scan, no name-length blind spot -------------------
#
# Round 3's candidate regex capped the name at 64 characters and anchored it at the start of the
# name run, so a run longer than 64 characters before the separator was never matched anywhere and
# its value was stored verbatim. The scan now walks backward from each separator over the full run.


def _append_and_read_back(text: str) -> tuple[str, str]:
    """Append `{"msg": text}` and return (the stored `msg`, the exported JSONL line)."""
    ledger, db = make_ledger()
    ledger.append("sess-1", "log", {"msg": text})
    return _last_payload(db)["msg"], next(ledger.export_jsonl("sess-1"))


def test_append_redacts_a_65_char_secret_name_end_to_end():
    name = "A" * (65 - len("_SECRET_KEY")) + "_SECRET_KEY"
    assert len(name) == 65
    stored, exported = _append_and_read_back(f"{name}=SUPERSECRETVALUE123")
    assert stored == f"{name}=[REDACTED]"
    assert "SUPERSECRETVALUE123" not in exported


def test_append_redacts_an_80_char_dotted_secret_name_end_to_end():
    name = "finagent.trading.brokers.alpaca.client.AlpacaClient._headers.APCA_API_SECRET_KEY"
    assert len(name) == 80
    stored, exported = _append_and_read_back(f"{name}=abcdEXPOSEDsecret123")
    assert stored == f"{name}=[REDACTED]"
    assert "EXPOSED" not in exported


@pytest.mark.parametrize("length", [63, 64, 65, 200, 5000])
def test_append_redacts_secret_names_of_any_length(length):
    name = "A" * (length - len("_SECRET_KEY")) + "_SECRET_KEY"
    assert len(name) == length
    stored, exported = _append_and_read_back(f"prefix {name}=LEAKvalue suffix")
    assert stored == f"prefix {name}=[REDACTED] suffix"
    assert "LEAKvalue" not in exported


def test_append_redacts_a_secret_keyword_at_the_start_of_a_very_long_name():
    name = "SECRET_KEY_" + "X" * 100
    stored, _exported = _append_and_read_back(f"{name}=v")
    assert stored == f"{name}=[REDACTED]"


def test_append_does_not_redact_a_long_non_secret_dotted_name():
    text = ".".join(["finagent.trading.proposals.TicketStore"] * 10) + ".max_retries=3"
    assert len(text) > 64
    stored, _exported = _append_and_read_back(text)
    assert stored == text


def test_append_redacts_a_mixed_case_path_whose_last_component_is_a_lower_case_secret():
    """The full run `MyApp.Config.password` is mixed-case with no `_`/`-`, so it isn't
    identifier-shaped - but its last dotted component is."""
    stored, _exported = _append_and_read_back("MyApp.Config.password=hunter2")
    assert stored == "MyApp.Config.password=[REDACTED]"


def test_append_keeps_the_titlecase_prose_rule_for_dotted_names():
    text = "see launch.Secret: Apple's new product"
    stored, _exported = _append_and_read_back(text)
    assert stored == text


def test_append_redacts_a_secret_pair_nested_in_another_pairs_value():
    stored, _exported = _append_and_read_back("config=APCA_API_SECRET_KEY=shhLEAK")
    assert stored == "config=APCA_API_SECRET_KEY=[REDACTED]"


def test_append_redacts_a_secret_pair_nested_under_a_secret_like_label():
    stored, _exported = _append_and_read_back("credentials:\n  api_key: LEAKvalue")
    assert "LEAKvalue" not in stored


# --- fix round 4: quoted names and values (dict reprs, JSON, quoted env values) ----------------


def test_append_redacts_quoted_values_in_a_python_dict_repr():
    stored, _exported = _append_and_read_back(
        "headers {'APCA-API-KEY-ID': 'PKLEAK', 'APCA-API-SECRET-KEY': 'skLEAK', 'Accept': 'json'}"
    )
    assert stored == (
        "headers {'APCA-API-KEY-ID': '[REDACTED]', 'APCA-API-SECRET-KEY': '[REDACTED]', "
        "'Accept': 'json'}"
    )


def test_append_redacts_quoted_values_in_json_text():
    stored, _exported = _append_and_read_back('{"api_key": "tvly-LEAK", "symbol": "AAPL"}')
    assert stored == '{"api_key": "[REDACTED]", "symbol": "AAPL"}'


def test_append_redacts_backslash_escaped_json_inside_a_string():
    text = '{\\"api_key\\": \\"escLEAK\\", \\"symbol\\": \\"AAPL\\"}'
    stored, _exported = _append_and_read_back(text)
    assert stored == '{\\"api_key\\": \\"[REDACTED]\\", \\"symbol\\": \\"AAPL\\"}'


def test_append_redacts_a_whole_quoted_env_value_including_spaces_and_escaped_quotes():
    stored, _exported = _append_and_read_back('APCA_API_SECRET_KEY="two LEAK \\" words" next')
    assert stored == 'APCA_API_SECRET_KEY="[REDACTED]" next'


def test_append_redacts_an_unclosed_quoted_value_through_the_end():
    stored, _exported = _append_and_read_back("password='LEAK and whatever follows")
    assert stored == "password='[REDACTED]"


def test_append_does_not_redact_quoted_titlecase_or_empty_values():
    for text in ("{'Secret': 'Apple'}", 'api_key=""', "{'symbol': 'AAPL'}"):
        stored, _exported = _append_and_read_back(text)
        assert stored == text


# --- fix round 4: free-text passes no longer run on each other's output ------------------------
#
# Found by fuzzing: when the passes ran in sequence, one pass's replacement could destroy context
# the next one needed. All passes now report spans against the same original text, and the union
# is replaced once.


def test_append_redacts_a_quoted_value_after_a_query_style_secret_param():
    # The query-param pass used to insert "[REDACTED]" for the (empty) unquoted value in front of
    # the quote, hiding the quoted value from the pair scan.
    stored, _exported = _append_and_read_back("GET https://h/x?api_key='LEAKvalue' failed")
    assert stored == "GET https://h/x?api_key='[REDACTED]' failed"


def test_append_redacts_a_secret_pair_inside_a_query_param_value():
    # The query-param pass used to swallow the nested name and its separator.
    stored, _exported = _append_and_read_back('x&secret=APCA_API_SECRET_KEY: "LEAKvalue"')
    assert stored == 'x&secret=[REDACTED] "[REDACTED]"'


def test_append_redacts_an_authorization_header_inside_a_secret_pair_value():
    stored, _exported = _append_and_read_back("api_key=Authorization: Basic LEAKvalue")
    assert stored == "api_key=[REDACTED] [REDACTED]"


# --- fix round 4: linear-time guarantees for the separator-driven scan -------------------------

_TRACEBACK_UNIT = "finagent.trading.proposals.TicketStore.transition."  # 50 chars

_ADVERSARIAL_100K = {
    "a. x50000": "a." * 50_000,
    "a- x50000": "a-" * 50_000,
    "x= x50000": "x=" * 50_000,
    "API_KEY: x12000": "API_KEY:" * 12_000,
    "a x99999 then =v": "a" * 99_999 + "=v",
    ":= x50000": ":=" * 50_000,
    "dotted traceback to 100k": _TRACEBACK_UNIT * 2_000,
    "whitespace run then :": " " * 99_999 + ":",
    "backslash run then quote-colon": "\\" * 99_998 + '":',
    "api_key=' x11111": "api_key='" * 11_111,
    'api_key=\\" x10000': 'api_key=\\"' * 10_000,
    "quoted value full of backslashes": 'api_key="' + "\\" * 99_990,
}


@pytest.mark.parametrize("label", list(_ADVERSARIAL_100K))
def test_redact_free_text_handles_100k_adversarial_strings_quickly(label):
    assert _time_redact_free_text(_ADVERSARIAL_100K[label]) < _REDOS_TIME_BUDGET_SEC


def test_redact_free_text_redacts_a_secret_at_the_end_of_a_100k_dotted_traceback_quickly():
    text = _TRACEBACK_UNIT * 1_999 + "_headers.APCA_API_SECRET_KEY=LEAKvalue"
    assert len(text) < 100_000
    start = time.perf_counter()
    result = _redact_free_text(text)
    assert time.perf_counter() - start < _REDOS_TIME_BUDGET_SEC
    assert result.endswith("_headers.APCA_API_SECRET_KEY=[REDACTED]")
    assert "LEAKvalue" not in result


def _best_time(text: str, repeats: int = 3) -> float:
    return min(_time_redact_free_text(text) for _ in range(repeats))


@pytest.mark.parametrize(
    "make",
    [
        lambda n: "x=" * (n // 2),
        lambda n: "a" * (n - 2) + "=v",
        lambda n: "API_KEY:" * (n // 8),
        lambda n: "api_key='" * (n // 9),
    ],
    ids=["many-separators", "one-long-name", "nested-secret-values", "unclosed-quotes"],
)
def test_redact_free_text_scales_linearly_across_input_doublings(make):
    """Three doublings (12.5k -> 100k chars, 8x the input): linear predicts ~8x the time,
    quadratic ~64x. A 24x bound leaves 3x slack for timing noise while still failing on a
    quadratic regression."""
    times = [_best_time(make(n)) for n in (12_500, 25_000, 50_000, 100_000)]
    assert times[0] > 0
    assert times[-1] / times[0] < 24


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


# --- Task 2.4: known-secret exact-value scrubbing ------------------------------------------------
#
# The pattern passes above key off *names* ("api_key=", "Authorization:"), so a secret embedded in
# a shape none of them recognizes used to be stored verbatim. `Ledger(db, known_secrets=...)` also
# replaces every verbatim occurrence of each configured secret value, in every string leaf and key,
# before any pattern pass runs.

KNOWN_SECRET = "PKTESTSECRET0123456789abcdefXYZ"
KNOWN_KEY_ID = "PKKEYID987654321"


def make_scrubbing_ledger(*secrets) -> tuple[Ledger, AppDB]:
    db = AppDB(":memory:")
    return Ledger(db, known_secrets=secrets or (KNOWN_SECRET, KNOWN_KEY_ID)), db


def _stored_text(db: AppDB) -> str:
    return "\n".join(row["payload"] for row in db.query("SELECT payload FROM ledger"))


@pytest.mark.parametrize(
    "shape",
    [
        '{"Authorization": "Basic %s"}',  # Basic auth inside JSON text (quoted Titlecase name)
        "apiKey=%s",  # camelCase name - not identifier-shaped, missed by the pair scan
        "clientSecret: %s",
        "https://user:%s@paper-api.alpaca.markets/v2/orders",  # URL userinfo
        "api_key => %s",  # an odd operator
        "'%s '",  # quoted, with a trailing space inside the quotes
        "api_key＝%s",  # fullwidth equals sign
        "api_key：%s",  # fullwidth colon
        "Mypassword=%s",  # mixed-case word fused onto a lower-case keyword
        "prefix%ssuffix",  # glued to other text with no separator at all
        "%s",  # the whole value
    ],
)
def test_known_secret_is_scrubbed_from_every_residual_free_text_shape(shape):
    ledger, db = make_scrubbing_ledger()
    ledger.append("sess-1", "note", {"message": shape % KNOWN_SECRET})
    stored = _stored_text(db)
    assert KNOWN_SECRET not in stored
    assert "[REDACTED]" in stored


def test_known_secret_is_scrubbed_from_dict_keys_nested_lists_tuples_sets_and_models():
    ledger, db = make_scrubbing_ledger()
    leg = OrderLeg(symbol="AAPL", side="buy", qty=D("1"), type="market")
    ledger.append(
        "sess-1",
        "note",
        {
            KNOWN_SECRET: "value",
            "nested": {"deeper": [("a", f"x{KNOWN_SECRET}y"), {f"k-{KNOWN_KEY_ID}": 1}]},
            "a_set": {f"s {KNOWN_SECRET}"},
            "leg": leg,
            "bytes_key": {b"raw": KNOWN_KEY_ID},
        },
    )
    stored = _stored_text(db)
    assert KNOWN_SECRET not in stored
    assert KNOWN_KEY_ID not in stored
    payload = _last_payload(db)
    assert payload["[REDACTED]"] == "value"  # the key itself was the secret
    assert payload["leg"]["symbol"] == "AAPL"  # an ordinary model still serializes normally


def test_known_secrets_are_stripped_and_short_or_empty_values_ignored():
    # Surrounding whitespace (a stray newline in a .env value) is stripped before matching; values
    # shorter than 8 characters, blank values and None (an unset setting) are ignored, so they can
    # never blank out ordinary short words.
    ledger, db = make_scrubbing_ledger(f"  {KNOWN_SECRET}\n", "short", "   ", None, "")
    ledger.append("sess-1", "note", {"message": f"{KNOWN_SECRET} short shortcut"})
    payload = _last_payload(db)
    assert payload["message"] == "[REDACTED] short shortcut"


def test_known_secret_containing_another_secret_is_scrubbed_whole():
    # Longest first: the longer secret is replaced as one unit, so no fragment of it survives.
    inner = "ABCDEFGH12"
    outer = f"zz{inner}zz-longer"
    ledger, db = make_scrubbing_ledger(inner, outer)
    ledger.append("sess-1", "note", {"message": f"<{outer}> and <{inner}>"})
    assert _last_payload(db)["message"] == "<[REDACTED]> and <[REDACTED]>"


def test_known_secret_scrubbing_leaves_payloads_without_secrets_unchanged():
    plain, plain_db = make_ledger()
    scrubbing, scrubbing_db = make_scrubbing_ledger()
    payload = {"symbol": "KEY", "qty": D("3"), "note": "api_key=abc", "legs": [{"i": 0}]}
    plain.append("sess-1", "note", payload)
    scrubbing.append("sess-1", "note", payload)
    assert _last_payload(scrubbing_db) == _last_payload(plain_db)


def test_known_secret_never_reaches_the_export():
    ledger, _db = make_scrubbing_ledger()
    ledger.append("sess-1", "broker_response", {"message": f"echo {KNOWN_SECRET}"})
    exported = "\n".join(ledger.export_jsonl("sess-1"))
    assert KNOWN_SECRET not in exported


def test_redact_text_applies_known_secrets_and_the_pattern_passes():
    ledger, _db = make_scrubbing_ledger()
    assert ledger.redact_text(f"apiKey={KNOWN_SECRET}") == "apiKey=[REDACTED]"
    assert ledger.redact_text("api_key=abc123") == "api_key=[REDACTED]"
    assert ledger.redact_text("insufficient buying power") == "insufficient buying power"


def test_known_secret_scrubbing_is_fast_on_a_100k_leaf():
    ledger, db = make_scrubbing_ledger(*(f"SECRETVALUE{i:04d}" for i in range(20)))
    text = ("x" * 99_000) + "SECRETVALUE0007"
    start = time.perf_counter()
    ledger.append("sess-1", "note", {"message": text})
    assert time.perf_counter() - start < 1.0
    assert "SECRETVALUE0007" not in _stored_text(db)


# A list or object value after a secret-like name is redacted whole (it used to stop at the
# first quote inside the brackets, leaking `"api_key": ["sk-..."]`).
@pytest.mark.parametrize("text, secrets", [
    ('{"api_key": ["sk-LISTSECRET-1"]}', ["sk-LISTSECRET-1"]),
    ("api_key=[sk-LISTSECRET-2, sk-LISTSECRET-3]", ["sk-LISTSECRET-2", "sk-LISTSECRET-3"]),
    ('"api_key": [["a"], "sk-LISTSECRET-4"]', ["sk-LISTSECRET-4"]),
    ('"secret": {"inner": "sk-LISTSECRET-5"}', ["sk-LISTSECRET-5"]),
    ('"api_key": ["sk-LISTSECRET-6"', ["sk-LISTSECRET-6"]),  # unclosed: fail safe to the end
    ("token: (sk-LISTSECRET-7, x)", ["sk-LISTSECRET-7"]),
])
def test_bracketed_values_after_secret_names_are_redacted_whole(text, secrets):
    out = _redact_free_text(text)
    for secret in secrets:
        assert secret not in out, out
    assert "[REDACTED]" in out


def test_bracketed_values_after_ordinary_names_are_kept():
    text = '{"symbols": ["AAPL", "VOO"], "qty": [1, 2]}'
    assert _redact_free_text(text) == text


def test_bracketed_secret_values_stay_linear():
    assert _time_redact_free_text('"api_key": [' * 20_000) < _REDOS_TIME_BUDGET_SEC
    assert _time_redact_free_text('"api_key": ["x", ' * 10_000) < _REDOS_TIME_BUDGET_SEC
