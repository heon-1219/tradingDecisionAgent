"""Process-wide secret scrub for log records of the `finagent` logger hierarchy.

Most of the code logs ids, counts and exception *types* only. A few places deliberately log a
traceback (`logger.exception` in `trading/execution.py` and `trading/ledger.py`: order problems
must be diagnosable), and a traceback prints exception messages, including chained causes, that
nothing else has cleaned. This module is the guarantee behind "secrets never enter the logs": once
`install_log_scrubber()` has run, every record of a `finagent` or `finagent.*` logger is cleaned
when it is created, before any handler sees it.

**How.** `install_log_scrubber()` wraps the process's log-record factory
(`logging.setLogRecordFactory`). A factory, unlike a filter attached to handlers, covers every
handler however logging is configured, and handlers added later: uvicorn's, pytest's `caplog`, a
user's `basicConfig`. For a `finagent` record the wrapper:

1. renders the message with its arguments once, scrubs it, and stores the result as `record.msg`
   with empty `args` (so the raw arguments are gone, and a `%` left in the text is not formatted
   again). A message that cannot be rendered is replaced by its scrubbed template plus a note; the
   arguments are dropped rather than left for logging's own error report to print;
2. renders the traceback of `exc_info` once (the standard format, chained causes included), scrubs
   it into `record.exc_text` - the cache every `logging.Formatter` uses - and clears `exc_info`, so
   no handler can render the raw exception again;
3. scrubs `stack_info`.

Records of every other logger pass through untouched (not even rendered).

**Limits.** Values passed through `extra=` are not covered (they are attributes a custom formatter
may print), and neither is the logger's *name*. A text longer than 100,000 characters is cut there
before it is scrubbed.

**What is scrubbed.** (a) Every exact occurrence of a known secret value (`set_log_secrets`),
longest first; (b) credential shapes, with the ledger's linear-time free-text passes
(`trading.ledger.redact_free_text`): `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY` and other
secret-named `name: value` / `name=value` pairs, `Authorization: ...`, `Bearer ...`, and
`key=` / `api_key=` / `token=` query parameters. Both become `[redacted]`; names stay visible. The
price of (b) is that a traceback's *source line* such as `token = load()` is shown as
`token = [redacted]`. One exemption, in this log pass only (the ledger keeps the broad rule): a
pair whose name ends in `tokens` or `token_count` (`max_tokens: 9000`, `'prompt_tokens': 1200`)
is a count, not a credential, and stays readable; a known secret value is still removed from it by
(a). Each scrubbed text is also capped at 100,000 characters (the ledger's leaf cap), so no pattern
ever runs on an unbounded string.

If scrubbing itself fails, the record's text is replaced by a fixed notice: fail safe, never leak.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

from .trading.ledger import MIN_KNOWN_SECRET_LEN, redact_free_text

REDACTED = "[redacted]"
# A shorter value is ignored: blanking a common substring out of every log line would destroy the
# logs while protecting nothing (real keys are far longer). One floor (8) for the ledger and the logs.
MIN_SECRET_LEN = MIN_KNOWN_SECRET_LEN

_ROOT = "finagent"
_PREFIX = _ROOT + "."
_MARK = "_finagent_log_scrubber"
_UNFORMATTED = " [log arguments could not be formatted]"
_UNSCRUBBABLE = "[log record withheld: it could not be scrubbed]"

_lock = threading.Lock()
_secrets: tuple[str, ...] = ()
_formatter = logging.Formatter()


def set_log_secrets(values: Iterable[Any] | None) -> None:
    """Replace the set of exact secret values scrubbed from `finagent` log records. Values are
    stripped; anything that is not a string, or is shorter than `MIN_SECRET_LEN`, is ignored."""
    global _secrets
    if values is None or isinstance(values, (str, bytes)):
        values = () if values is None else (values,)
    kept = {
        value.strip()
        for value in values
        if isinstance(value, str) and len(value.strip()) >= MIN_SECRET_LEN
    }
    # Longest first, so a secret that contains another is replaced as one unit.
    _secrets = tuple(sorted(kept, key=lambda s: (-len(s), s)))


def scrub_text(text: str) -> str:
    """`text` with every known secret value and every credential shape replaced by `[redacted]`."""
    for secret in _secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return redact_free_text(text, REDACTED, allow_token_counts=True)


def _scrub_record(record: logging.LogRecord) -> None:
    try:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string or argument must not leak the args
            message = str(record.msg) + _UNFORMATTED
        record.msg = scrub_text(message)
        record.args = ()
        exc_info = record.exc_info
        if exc_info:
            if record.exc_text:
                record.exc_text = scrub_text(record.exc_text)
            elif isinstance(exc_info, tuple) and exc_info[0] is not None:
                record.exc_text = scrub_text(_formatter.formatException(exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = scrub_text(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub_text(str(record.stack_info))
    except Exception:  # noqa: BLE001 - fail safe: an unscrubbed record is never emitted
        record.msg, record.args = _UNSCRUBBABLE, ()
        record.exc_info = record.exc_text = record.stack_info = None


def install_log_scrubber() -> None:
    """Wrap the log-record factory so every `finagent` / `finagent.*` record is scrubbed when it is
    created. Idempotent: while the scrubber is the current factory, calling this again changes
    nothing."""
    with _lock:
        inner = logging.getLogRecordFactory()
        if getattr(inner, _MARK, False):
            return

        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = inner(*args, **kwargs)
            name = record.name
            if isinstance(name, str) and (name == _ROOT or name.startswith(_PREFIX)):
                _scrub_record(record)
            return record

        setattr(factory, _MARK, True)
        factory.inner = inner  # type: ignore[attr-defined]
        logging.setLogRecordFactory(factory)


def uninstall_log_scrubber() -> None:
    """Put back the factory the scrubber wrapped (for tests). A no-op unless the scrubber is the
    current factory."""
    with _lock:
        current = logging.getLogRecordFactory()
        if getattr(current, _MARK, False):
            logging.setLogRecordFactory(current.inner)
