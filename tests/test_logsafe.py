"""Tests for `finagent.logsafe` - the process-wide secret scrub for `finagent.*` log records.

Every test logs through real loggers and reads the result the way an operator would: pytest's
`caplog` (a handler on the root logger) and a plain `StreamHandler` attached to a `finagent` logger.
The last test drives the executor's real `logger.exception` path with a fake broker. `conftest.py`
uninstalls the scrubber and clears its secrets after every test.
"""
from __future__ import annotations

import io
import logging
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from fakes_trading import FakeBroker, FakeData
from finagent import logsafe
from finagent.logsafe import install_log_scrubber, scrub_text, set_log_secrets, uninstall_log_scrubber
from finagent.store.appdb import AppDB
from finagent.trading.broker import BrokerError
from finagent.trading.execution import Executor
from finagent.trading.ledger import Ledger
from finagent.trading.models import Mandate, OrderLeg, TicketStatus
from finagent.trading.proposals import TicketStore
from finagent.trading.risk import evaluate

SECRET = "sk-live-0123456789abcdef"
OTHER = "tvly-other-zyxwvutsrqpon"
FAKE_KEY_ID = "PKFAKEKEYID0123456789"
FAKE_SECRET = "fakeSecretValue-9f8e7d6c5b4a3210zyxw"
FIXED_NOW = datetime(2026, 9, 28, 14, 0, 0, tzinfo=timezone.utc)

log = logging.getLogger("finagent.x")


@pytest.fixture
def scrubbed():
    install_log_scrubber()
    set_log_secrets([SECRET])


@pytest.fixture
def stream():
    """What a plain `StreamHandler` on a `finagent` logger writes."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    parent = logging.getLogger("finagent")
    parent.addHandler(handler)
    try:
        yield buffer
    finally:
        parent.removeHandler(handler)


def _log_chained_failure(logger: logging.Logger) -> None:
    try:
        try:
            raise ValueError(f"inner cause {SECRET}")
        except ValueError as inner:
            raise RuntimeError(f"outer failure {SECRET}") from inner
    except RuntimeError:
        logger.exception("order flow failed")


# --- exception text ---------------------------------------------------------------------------


def test_without_the_scrubber_the_secret_is_logged(caplog, stream):
    """The control: this is the leak the scrubber closes."""
    _log_chained_failure(log)
    assert SECRET in caplog.text and SECRET in stream.getvalue()


def test_exception_text_and_chained_cause_are_scrubbed(scrubbed, caplog, stream):
    _log_chained_failure(log)
    for text in (caplog.text, stream.getvalue()):
        assert SECRET not in text
        assert "order flow failed" in text
        assert "Traceback (most recent call last)" in text
        assert "RuntimeError: outer failure [redacted]" in text
        assert "ValueError: inner cause [redacted]" in text  # the chained cause
        assert "direct cause of the following exception" in text
    record = caplog.records[-1]
    # Nothing on the record can re-render the raw traceback: the text is cached, the exception gone.
    assert record.exc_info is None
    assert SECRET not in record.exc_text and "[redacted]" in record.exc_text


def test_an_implicitly_chained_exception_is_scrubbed(scrubbed, caplog):
    try:
        try:
            raise ValueError(f"first {SECRET}")
        except ValueError:
            raise KeyError("second")  # no `from`: chained through __context__
    except KeyError:
        log.exception("failed")
    assert SECRET not in caplog.text
    assert "ValueError: first [redacted]" in caplog.text
    assert "During handling of the above exception" in caplog.text


def test_exc_info_passed_explicitly_is_scrubbed(scrubbed, caplog):
    error = RuntimeError(f"explicit {SECRET}")
    log.warning("with exc_info", exc_info=error)
    log.error("no active exception", exc_info=True)  # (None, None, None): must not crash
    assert SECRET not in caplog.text
    assert "RuntimeError: explicit [redacted]" in caplog.text
    assert "no active exception" in caplog.text


# --- message and arguments --------------------------------------------------------------------


def test_message_args_are_scrubbed(scrubbed, caplog, stream):
    log.warning("key %s for %s", SECRET, "user")
    log.warning("config %s", {"value": SECRET})
    log.warning("named %(value)s", {"value": SECRET})
    log.warning(f"inline {SECRET}")
    for text in (caplog.text, stream.getvalue()):
        assert SECRET not in text
        assert "key [redacted] for user" in text
        assert "named [redacted]" in text and "inline [redacted]" in text
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert SECRET not in repr(record.args) and SECRET not in str(record.msg)


def test_a_message_that_cannot_be_formatted_does_not_leak_its_args(scrubbed, caplog, capsys):
    log.warning("%d items", SECRET)  # TypeError at format time
    log.warning("no placeholders", SECRET)
    captured = capsys.readouterr()
    assert SECRET not in caplog.text
    assert SECRET not in captured.err and SECRET not in captured.out
    assert "Logging error" not in captured.err
    assert "could not be formatted" in caplog.text


def test_a_percent_sign_in_the_scrubbed_message_is_kept(scrubbed, caplog):
    log.warning("progress %d%% of %s", 50, SECRET)
    assert "progress 50% of [redacted]" in caplog.text


def test_stack_info_is_scrubbed(scrubbed, caplog):
    log.warning("with stack", stack_info=True); _ = "sk-live-0123456789abcdef"  # noqa: E702 - the literal must sit on the logged source line
    record = caplog.records[-1]
    assert "Stack (most recent call last)" in record.stack_info
    assert "[redacted]" in record.stack_info
    assert SECRET not in record.stack_info and SECRET not in caplog.text


# --- scope ------------------------------------------------------------------------------------


def test_non_finagent_loggers_are_untouched(scrubbed, caplog):
    for name in ("other.lib", "finagentx", "thirdparty.finagent"):
        logging.getLogger(name).warning("value %s", SECRET)
    logging.getLogger().warning("root %s", SECRET)
    assert caplog.text.count(SECRET) == 4
    for record in caplog.records:
        assert record.args == (SECRET,)  # not even rendered


def test_the_whole_finagent_hierarchy_is_scrubbed(scrubbed, caplog):
    for name in ("finagent", "finagent.app", "finagent.trading.execution", "finagent.a.b.c"):
        logging.getLogger(name).warning("value %s", SECRET)
    assert SECRET not in caplog.text
    assert caplog.text.count("value [redacted]") == 4


def test_a_handler_added_later_is_covered_too(scrubbed):
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)  # e.g. a user's basicConfig, or uvicorn's handlers
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        _log_chained_failure(log)
    finally:
        root.removeHandler(handler)
    assert SECRET not in buffer.getvalue() and "outer failure [redacted]" in buffer.getvalue()


# --- install / secrets ------------------------------------------------------------------------


def test_install_is_idempotent(caplog):
    before = logging.getLogRecordFactory()
    install_log_scrubber()
    installed = logging.getLogRecordFactory()
    assert installed is not before
    install_log_scrubber()
    install_log_scrubber()
    assert logging.getLogRecordFactory() is installed  # not wrapped again
    set_log_secrets([SECRET])
    log.warning("once %s", SECRET)
    assert "once [redacted]" in caplog.text
    uninstall_log_scrubber()
    assert logging.getLogRecordFactory() is before
    uninstall_log_scrubber()  # a no-op when not installed
    assert logging.getLogRecordFactory() is before


def test_set_log_secrets_replaces_the_previous_set(scrubbed, caplog):
    log.warning("first %s %s", SECRET, OTHER)
    assert "first [redacted] " + OTHER in caplog.text
    set_log_secrets([OTHER])
    log.warning("second %s %s", SECRET, OTHER)
    assert f"second {SECRET} [redacted]" in caplog.text
    set_log_secrets([])
    log.warning("third %s %s", SECRET, OTHER)
    assert f"third {SECRET} {OTHER}" in caplog.text


def test_short_and_empty_values_are_ignored(scrubbed, caplog):
    set_log_secrets(["", "   ", "abc", "12345", None, 123456789, "abcdefg", "abcdefgh", f"  {SECRET}\n"])
    log.warning("abc 12345 123456789 abcdefg-x abcdefgh %s", SECRET)
    assert "abc 12345 123456789 abcdefg-x [redacted] [redacted]" in caplog.text


def test_the_ledger_and_the_log_scrubber_share_one_floor_of_eight_characters():
    from finagent.trading import ledger

    assert logsafe.MIN_SECRET_LEN == ledger.MIN_KNOWN_SECRET_LEN == 8
    book = Ledger(AppDB(":memory:"), known_secrets=["seven77", "eight888"])
    assert book.redact_text("a seven77 b eight888 c") == "a seven77 b [REDACTED] c"
    set_log_secrets(["seven77", "eight888"])
    assert scrub_text("a seven77 b eight888 c") == "a seven77 b [redacted] c"


def test_a_secret_containing_another_is_replaced_as_one_unit(scrubbed, caplog):
    set_log_secrets(["PKABCDEF", "PKABCDEF-extended-secret"])
    log.warning("a PKABCDEF-extended-secret b PKABCDEF c")
    assert "a [redacted] b [redacted] c" in caplog.text


# --- credential shapes ------------------------------------------------------------------------


@pytest.mark.parametrize("text, hidden, kept", [
    ("request headers APCA-API-KEY-ID: PKSHAPE0001 sent", "PKSHAPE0001", "APCA-API-KEY-ID"),
    ("request headers APCA-API-SECRET-KEY: shapeSecret0002 sent", "shapeSecret0002", "APCA-API-SECRET-KEY"),
    ("headers={'APCA-API-KEY-ID': 'PKSHAPE0003', 'APCA-API-SECRET-KEY': 'shapeSecret0004'}",
     "PKSHAPE0003", "APCA-API-KEY-ID"),
    ("headers={'APCA-API-KEY-ID': 'PKSHAPE0003', 'APCA-API-SECRET-KEY': 'shapeSecret0004'}",
     "shapeSecret0004", "APCA-API-SECRET-KEY"),
    ("Authorization: Bearer shapeToken0005.abc.def", "shapeToken0005", "Authorization"),
    ("sent Bearer shapeToken0006 upstream", "shapeToken0006", "Bearer"),
    ("GET https://example.com/v1/models?key=shapeKey0007&alt=json failed", "shapeKey0007", "alt=json"),
    ("GET https://example.com/search?q=apple&api_key=shapeKey0008", "shapeKey0008", "q=apple"),
])
def test_credential_shapes_are_masked(scrubbed, caplog, text, hidden, kept):
    log.warning(text)
    log.warning("as an argument: %s", text)
    try:
        raise RuntimeError(text)
    except RuntimeError:
        log.exception("as exception text")
    assert hidden not in caplog.text
    assert kept in caplog.text and "[redacted]" in caplog.text
    assert hidden not in scrub_text(text)


def test_ordinary_log_lines_are_left_alone(scrubbed, caplog):
    lines = [
        "chat runtime built: broker=True edgar=False web_search=True grounding=False",
        "chat turn 3f2a: tool get_quotes raised RuntimeError",
        "startup: SEC_USER_AGENT comes from the process environment, not .env",
        "alpaca key check: invalid (BrokerError)",
        "tool price_metrics: upstream HTTP 403",
    ]
    for line in lines:
        log.warning(line)
    for line in lines:
        assert line in caplog.text
    assert "[redacted]" not in caplog.text


# --- token counts are not credentials (the log pass only) ------------------------------------------

# The real templates of `finagent.agent.llm` that log provider error text; each provider message is
# the shape that provider sends. `test_the_llm_log_templates_are_the_real_ones` keeps them honest.
LLM_ERROR_TEMPLATES = (
    "anthropic error: %s",
    "openai-compatible error: %s",
    "effort/thinking unsupported for %s, retrying plain: %s",
    "%s rejected reasoning_effort=%s; continuing without it: %s",
)
ANTHROPIC_MAX_TOKENS = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
    "'max_tokens: 9000 > 8192, which is the maximum allowed number of output tokens for this model'}}"
)
OPENAI_USAGE = (
    "Error code: 400 - {'error': {'message': 'context too long', 'usage': {'prompt_tokens': 1200, "
    "'completion_tokens': 30, 'total_tokens': 1230}, 'max_completion_tokens': 8192}}"
)
GEMINI_COUNTS = "INVALID_ARGUMENT: thoughts_token_count=512 exceeds the budget; input_tokens: 90210"


def test_the_llm_log_templates_are_the_real_ones():
    import inspect

    from finagent.agent import llm as llm_module

    source = inspect.getsource(llm_module)
    for template in LLM_ERROR_TEMPLATES:
        assert f'"{template}"' in source, template


def test_token_counts_stay_readable_in_llm_error_logs(scrubbed, caplog):
    llm_log = logging.getLogger("finagent.llm")
    llm_log.error("anthropic error: %s", ANTHROPIC_MAX_TOKENS)
    llm_log.error("openai-compatible error: %s", OPENAI_USAGE)
    llm_log.warning("effort/thinking unsupported for %s, retrying plain: %s", "claude-x", ANTHROPIC_MAX_TOKENS)
    llm_log.warning("%s rejected reasoning_effort=%s; continuing without it: %s", "gemini-x", "high", GEMINI_COUNTS)
    text = caplog.text
    assert "[redacted]" not in text
    for visible in ("max_tokens: 9000 > 8192", "'prompt_tokens': 1200", "'completion_tokens': 30",
                    "'total_tokens': 1230", "'max_completion_tokens': 8192", "thoughts_token_count=512",
                    "input_tokens: 90210"):
        assert visible in text, visible


@pytest.mark.parametrize("text, hidden", [
    ("access_token=shapeToken0101", "shapeToken0101"),
    ("refresh_token: shapeToken0102", "shapeToken0102"),
    ("X-Auth-Token: shapeToken0103", "shapeToken0103"),
    ("api_token='shapeToken0104'", "shapeToken0104"),
    ("GET https://example.com/v1?token=shapeToken0105", "shapeToken0105"),
    ("token_secret: shapeToken0106", "shapeToken0106"),
])
def test_real_token_credentials_are_still_masked(scrubbed, caplog, text, hidden):
    logging.getLogger("finagent.llm").error("openai-compatible error: %s", text)
    assert hidden not in caplog.text and "[redacted]" in caplog.text


@pytest.mark.parametrize("text, hidden", [
    ("access_tokens=ya29.SECRETVALUE0001", "ya29.SECRETVALUE0001"),
    ("{'tokens': 'ya29.SECRETVALUE0002'}", "ya29.SECRETVALUE0002"),
    ("x-api-tokens: SECRETVALUE0003", "SECRETVALUE0003"),
    ("refresh_tokens=\"r-SECRETVALUE0004\"", "r-SECRETVALUE0004"),
    ("max_tokens: abc123SECRET0005", "abc123SECRET0005"),
    ("thoughts_token_count=SECRET0006", "SECRET0006"),
])
def test_a_tokens_name_with_a_value_that_is_not_a_number_is_masked(scrubbed, caplog, text, hidden):
    logging.getLogger("finagent.llm").error("openai-compatible error: %s", text)
    assert hidden not in caplog.text and "[redacted]" in caplog.text


@pytest.mark.parametrize("text", [
    "max_tokens: 9000", "max_tokens: 9,000", "'total_tokens': 1230}", "tokens=12.5",
    "'prompt_tokens': '1200'", "x-ratelimit-remaining-tokens: 5000", "thoughts_token_count=512)",
])
def test_a_tokens_name_with_a_number_stays_readable(scrubbed, caplog, text):
    logging.getLogger("finagent.llm").error("anthropic error: %s", text)
    assert text in caplog.text and "[redacted]" not in caplog.text


def test_a_known_secret_next_to_a_token_count_name_is_still_scrubbed(scrubbed, caplog):
    logging.getLogger("finagent.llm").error("anthropic error: %s", f"max_tokens: {SECRET} prompt_tokens={SECRET}")
    assert SECRET not in caplog.text
    assert "max_tokens: [redacted] prompt_tokens=[redacted]" in caplog.text


def test_the_ledger_pass_still_masks_token_count_names():
    """The exemption is for logs only: the ledger keeps its broader rule."""
    book = Ledger(AppDB(":memory:"))
    assert book.redact_text("max_tokens: 9000") == "max_tokens: [REDACTED]"
    book.append("sess-1", "note", {"text": "prompt_tokens=1200"})
    assert "1200" not in book._db.query("SELECT payload FROM ledger")[0]["payload"]


# --- AppState.load installs the scrubber before anything else can log -------------------------------


def test_app_state_load_installs_the_scrubber_with_the_settings_secrets_first(monkeypatch, caplog):
    """`AppState.load` is heavy (it reads the product store), so this stops it at its first real
    step and checks what it did before that: the factory is installed and knows every key."""
    import finagent.app_state as app_state
    from finagent.config import Settings

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        logging.getLogger("finagent.app").warning("opening the store with %s", "gm-key-0123456789")
        raise Stop()

    monkeypatch.setattr(app_state, "RDB", stop)
    settings = Settings(
        _env_file=None, llm_provider="stub", gemini_api_key="gm-key-0123456789",
        openai_api_key="sk-openai-0123456789", anthropic_api_key="sk-ant-0123456789",
        alpaca_api_key_id=FAKE_KEY_ID, alpaca_api_secret_key=FAKE_SECRET, tavily_api_key=OTHER,
    )
    before = logging.getLogRecordFactory()
    with pytest.raises(Stop):
        app_state.AppState.load(settings)
    assert logging.getLogRecordFactory() is not before
    assert "opening the store with [redacted]" in caplog.text
    log.warning("keys %s %s %s %s %s %s", "gm-key-0123456789", "sk-openai-0123456789", "sk-ant-0123456789",
                FAKE_KEY_ID, FAKE_SECRET, OTHER)
    assert "keys [redacted] [redacted] [redacted] [redacted] [redacted] [redacted]" in caplog.text


def test_scrubbing_is_linear_time_on_a_hostile_message(scrubbed):
    hostile = "a." * 200_000 + "=x " + ("api_key:" * 20_000)
    started = time.perf_counter()
    scrub_text(hostile)
    assert time.perf_counter() - started < 5.0


# --- the executor's real `logger.exception` path ----------------------------------------------


class _ExecWorld:
    def __init__(self):
        self.db = AppDB(":memory:")
        self.tickets = TicketStore(self.db, clock=lambda: FIXED_NOW)
        self.ledger = Ledger(self.db, known_secrets=[FAKE_KEY_ID, FAKE_SECRET])
        self.broker = FakeBroker()
        self.data = FakeData()
        self.settings = SimpleNamespace(trading_enabled=True, ticket_ttl_sec=300)

        async def no_sleep(_seconds):
            return None

        self.executor = Executor(self.broker, self.data, self.tickets, self.ledger, self.settings,
                                 mandate_for=lambda _session: Mandate(), sleep=no_sleep)

    async def propose(self):
        leg = OrderLeg(symbol="AAPL", side="buy", qty=Decimal("10"), type="limit", limit_price=Decimal("200.00"))
        ctx = await self.executor.build_context([leg.symbol])
        return self.tickets.create(
            "sess-1", decisions=evaluate([leg], ctx, Mandate()), thesis="t", risks="r", invalidation="i",
            evidence_ids=["E1"], mandate_version=1, quote_as_of=FIXED_NOW, ttl_sec=300,
        )

    def break_the_outcome_write(self) -> None:
        real = self.tickets.transition

        def failing(ticket_id, expect, to, **fields):
            if to in (TicketStatus.submitted, TicketStatus.failed):
                raise sqlite3.OperationalError("database is locked")
            return real(ticket_id, expect, to, **fields)

        self.tickets.transition = failing


async def _confirm_with_a_secret_in_the_broker_error(world: _ExecWorld):
    """The re-check fails with a broker error whose message carries the secret; writing that
    failure to the ticket fails too, so the second error escapes into the executor's
    `logger.exception` with the broker error chained to it."""
    ticket = await world.propose()
    world.broker.account_error = BrokerError(503, None, f"service unavailable {FAKE_SECRET}")
    world.break_the_outcome_write()
    result = await world.executor.confirm(ticket.id, "sess-1", ticket.hash)
    assert result.status == "error" and world.broker.submits == 0
    assert FAKE_SECRET not in result.message
    return result


class _ExplodingPayload(dict):
    """A payload whose traversal fails with the secret in the exception message."""

    def items(self):
        raise RuntimeError(f"cannot read payload {FAKE_SECRET}")


def test_the_ledgers_own_traceback_is_scrubbed(caplog):
    install_log_scrubber()
    set_log_secrets([FAKE_KEY_ID, FAKE_SECRET])
    db = AppDB(":memory:")
    ledger = Ledger(db, known_secrets=[FAKE_KEY_ID, FAKE_SECRET])
    ledger.append("sess-1", "note", _ExplodingPayload(a=1))  # never raises: a placeholder is stored
    assert "could not be serialized" in caplog.text and "Traceback (most recent call last)" in caplog.text
    assert "RuntimeError: cannot read payload [redacted]" in caplog.text
    assert FAKE_SECRET not in caplog.text
    assert FAKE_SECRET not in db.query("SELECT payload FROM ledger")[0]["payload"]


async def test_the_executor_traceback_leaks_the_secret_without_the_scrubber(caplog):
    """The control for the next test: `logger.exception` really logs the broker's message."""
    await _confirm_with_a_secret_in_the_broker_error(_ExecWorld())
    assert "confirm of ticket" in caplog.text and "Traceback (most recent call last)" in caplog.text
    assert FAKE_SECRET in caplog.text


async def test_the_executor_traceback_is_scrubbed(caplog):
    install_log_scrubber()
    set_log_secrets([FAKE_KEY_ID, FAKE_SECRET])
    await _confirm_with_a_secret_in_the_broker_error(_ExecWorld())
    # the real `logger.exception` calls ran, tracebacks and chained broker error included ...
    assert "confirm of ticket" in caplog.text and "could not persist the outcome" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "BrokerError: Alpaca broker error 503: service unavailable [redacted]" in caplog.text
    assert "OperationalError: database is locked" in caplog.text
    # ... and the secret is in none of them
    assert FAKE_SECRET not in caplog.text
    assert all(FAKE_SECRET not in (record.exc_text or "") for record in caplog.records)
