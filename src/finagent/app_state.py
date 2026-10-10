"""Builds every runtime component once (startup) and hands them to the API.

`AppState` is the single-question QA stack behind `/answer`. `ChatRuntime` is the chat and trading
stack behind `/api/*`; `AppState.load` builds it last, and a failure there only disables `/api/*`
(`chat` stays None and the router answers 503), never `/answer`.

**ChatRuntime.** One instance per process of each persistent store over one `AppDB`: `evidence`
(its id lock is per instance), `sessions`, `tickets` (its lifecycle lock is per instance), plus
exactly one `ChatAgent` (its per-session turn locks must be unique). The rest is *rebuildable*
from the current settings by `rebuild()`: the HTTP client, the ledger (its known-secret list), the
connectors (Gemini search grounding whenever a Gemini key exists, whatever the chat provider is),
the Alpaca broker and market data (only when both keys are set and the account is not known to be
a live one), the executor (only with a broker) and the toolbox (which writes its `tool_call` and
`proposal` audit entries to that ledger). The agent reaches the current toolbox through
`_ToolboxHandle`, so a rebuild takes effect at the next tool call; the toolbox's
one-proposal-per-turn slots (`turn_slots`) are owned by the runtime and shared by every toolbox it
builds, so a rebuild in the middle of a turn cannot reset them. Every component shares the *same*
mutable `settings` object, so the kill switch (`settings.trading_enabled = False`) is seen
everywhere at once.

**The model.** `AppState.load` creates the provider client once and wraps it in one `LLMHandle`.
The QA orchestrator (`/answer`), the runtime and the one `ChatAgent` hold that handle, so
`apply_settings` changes the model for all of them at once: it builds the new client from a copy of
the settings (`llm_factory`), and after the env file is written swaps it into the handle, retires
the old client (closed after the grace delay: a turn in flight still holds it through its
conversation) and has the agent drop its cached provider conversations (`reset_conversations`).
`check_llm` (one minimal model call) and `list_llm_models` go through injectable factories
(`llm_validator`, `llm_models`); production uses `llm_admin`.

**Paper only.** `AccountSnapshot.paper` is read wherever the account is: the status check
(`refresh_alpaca_status` at startup, and from `GET /api/settings` while the status is unsettled,
at most once a minute; concurrent callers share one check), `check_alpaca_keys` for candidate
keys, the portfolio route, the chat tool `get_account` and the executor's market context, which
every proposal, confirm re-check and refresh builds first. The executor and the toolbox refuse
(to trade, or to show the account) and call back `_drop_non_paper`. `False` means a live account:
the status is `not_paper` and the runtime rebuilds without a broker, executor or market data,
exactly as if the keys were missing, and remembers that key pair so that no later rebuild brings
them back. Only a write that applies the same pair again (`apply_settings`, after a check that
says otherwise) clears that mark: a check alone does not, since the write may still fail. Only the
broker that read the account is dropped: one already replaced is left alone. `None` (the response
did not say) keeps the broker: the hard-coded paper host is still the first line of defence.

**Retired model clients** are closed only after the longest turn the settings allow (either
mode), plus the nudge grace, one model call and a margin (`_llm_grace_sec`), so a deep turn that
started on the old client can finish on it.

A rebuild is build-then-swap: `_build()` constructs a full set of components and touches nothing
shared (a failure retires what it made and changes nothing); `_swap()` puts them in service
without awaiting, and closes the old HTTP clients only after `close_grace_sec` (120 s), so turns
and confirms already in flight finish on the old objects. `apply_settings()` is all or nothing:
build for the new values, write the env file, and only then update the settings in place and swap.
`lock` serialises settings rebuilds against confirm, reject, refresh and the kill switch; chat
turns do not take it. The one rebuild that may run without it is the not-paper drop from a chat
tool or a proposal (see `rebuild`).

**The kill switch** (`kill_switch`) turns trading off in memory, bumps `kill_generation` and writes
`TRADING_ENABLED=false` to the env file at once, before it waits for the lock; the broker steps
follow under the lock. A settings write that started before the press may not turn trading back
on (`KillSwitchPressed`). Re-enabling trading resumes the broker the kill switch suspended. Every
settings write is ledgered as `settings_changed` (field names only) and a non-paper drop as
`broker_dropped`, both under the system session.

`startup()` (from the app lifespan) warns, by name only, about writable settings that the process
environment-only settings (`config.env_fallbacks`), expires stale tickets, settles tickets stuck in
`submitting` (`Executor.recover_stuck`; without a broker it only logs how many there are) and
checks the Alpaca keys. Nothing here logs a secret or an exception message: failures are logged by
type.

**Log scrubbing.** The runtime installs `logsafe`'s process-wide scrub at construction and, on
every build, gives it the same accumulated known-secret list the ledger and the toolbox get
(current and replaced keys: Alpaca, Tavily, Gemini, OpenAI, Anthropic), so a traceback logged
anywhere under `finagent.*` cannot show a key. A candidate key (one being checked, or one given
only to list models) joins that list before the call is made, whether or not it is ever saved.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import llm_admin
from .agent.llm import LLMClient, LLMHandle, make_llm
from .agent.orchestrator import Orchestrator
from .agent.prompts import build_system_prompt, value_hints
from .agent.tools import ToolRegistry
from .chat.agent import ChatAgent, nudge_grace_sec
from .chat.sessions import SessionStore
from .chat.toolbox import ChatToolbox
from .config import ENV_NAMES, Settings, env_fallbacks, longest_turn_sec
from .envfile import update_env_file
from .kb.docs import DocStore
from .kb.entity import EntityResolver
from .kb.graph import KG
from .kb.lexicon import Lexicon
from .kb.rdb import RDB
from .kb.search import SearchIndex
from .kb.vector import VectorIndex, make_openai_embed_fn
from .logsafe import install_log_scrubber, set_log_secrets
from .ontology.loader import FallbackValidators, Ontology
from .paths import ROOT
from .research.edgar import Edgar
from .research.evidence import EvidenceStore
from .research.grounding import GroundedWeb
from .research.http import HttpClient
from .research.websearch import WebSearch
from .store.appdb import AppDB
from .trading.broker import AlpacaPaper, BrokerError
from .trading.execution import SYSTEM_SESSION, Executor
from .trading.ledger import Ledger
from .trading.marketdata import AlpacaData
from .trading.models import Mandate, TicketStatus
from .trading.proposals import TicketStore

log = logging.getLogger("finagent.app")

CLOSE_GRACE_SEC = 120.0  # for retired HTTP clients and brokers (a confirm in flight)
LLM_GRACE_MARGIN_SEC = 30.0  # on top of the longest turn, the nudge and one model call
# Alpaca statuses that `GET /api/settings` does not re-check; any other is re-checked, at most once
# every ALPACA_RECHECK_SEC.
SETTLED_ALPACA = frozenset({"connected", "not_paper"})
ALPACA_RECHECK_SEC = 60.0
# Settings fields whose secret values the ledger and the log scrubber remove verbatim.
SECRET_FIELDS = (
    "alpaca_api_key_id", "alpaca_api_secret_key", "gemini_api_key", "tavily_api_key",
    "anthropic_api_key", "openai_api_key",
)
_ALPACA_PAIR = frozenset({"alpaca_api_key_id", "alpaca_api_secret_key"})
# Settings that only the model client reads: changing them needs a new client (`_needs_new_llm`).
_LLM_CLIENT_FIELDS = frozenset({"llm_provider", "llm_model", "llm_effort"})
# Changing any of these needs new connectors, a new ledger and a new toolbox (they capture keys,
# or the known-secret list, at construction).
_REBUILD_FIELDS = frozenset(ENV_NAMES) - {"trading_enabled", "gemini_paid"} - _LLM_CLIENT_FIELDS
# Changing any of these changes what a chat conversation was built with (the client, or the mode in
# its system prompt): cached provider conversations are dropped.
_CONVERSATION_FIELDS = _LLM_CLIENT_FIELDS | {"gemini_paid", *llm_admin.KEY_FIELDS.values()}


class KillSwitchPressed(Exception):
    """`apply_settings` refused to turn trading on: the kill switch was pressed after the request
    started (see `ChatRuntime.kill_generation`). Nothing was written."""


@dataclass
class _Built:
    """One set of rebuildable components, constructed but not yet in service."""

    http: Any
    ledger: Ledger
    edgar: Any
    web: Any
    grounded: Any
    broker: Any
    data: Any
    executor: Any
    toolbox: Any


def alpaca_status_for(exc: BaseException) -> str:
    """`invalid` for rejected keys (401/403), else `unreachable` (network, 5xx, anything else)."""
    if isinstance(exc, BrokerError) and exc.status in (401, 403):
        return "invalid"
    return "unreachable"


def account_status(account: Any) -> str:
    """`not_paper` when the account says it is not a paper account (`paper is False`); `connected`
    when it is one or does not say (`True` / `None`)."""
    return "not_paper" if getattr(account, "paper", None) is False else "connected"


async def _aclose(component: Any) -> None:
    close = getattr(component, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception as exc:  # noqa: BLE001 - closing is best effort
        log.info("closing %s failed: %s", type(component).__name__, type(exc).__name__)


class _ToolboxHandle:
    """The one `ChatAgent`'s toolbox: forwards to the runtime's current `ChatToolbox`."""

    def __init__(self, runtime: "ChatRuntime"):
        self._runtime = runtime

    def specs(self):
        return self._runtime.toolbox.specs()

    async def execute(self, session_id: str, name: str, args: dict | None, *, turn_id: str):
        return await self._runtime.toolbox.execute(session_id, name, args, turn_id=turn_id)


class ChatRuntime:
    """The `/api/*` stack; see the module docstring. The factories and `clock`/`sleep` exist for
    tests; production uses the defaults (`AlpacaPaper` is never given a `client`)."""

    def __init__(
        self,
        settings,
        *,
        llm: LLMClient,
        registry=None,
        orchestrator=None,
        db: AppDB | None = None,
        env_path: Path | str | None = None,
        broker_factory: Callable[[str, str], Any] | None = None,
        data_factory: Callable[[str, str, Any], Any] | None = None,
        http_factory: Callable[[], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Any] | None = None,
        close_grace_sec: float = CLOSE_GRACE_SEC,
        llm_validator: Callable[[Any], Awaitable[str]] | None = None,
        llm_models: Callable[[str, str, Any], Awaitable[list[dict]]] | None = None,
        llm_factory: Callable[[Any], LLMClient] | None = None,
        close_sleep: Callable[[float], Awaitable[Any]] | None = None,
        monotonic: Callable[[], float] | None = None,
    ):
        install_log_scrubber()  # before anything here can log; the secret values follow in _build
        self.settings = settings
        # The one model handle (see the module docstring); a bare client is wrapped here.
        self.llm = LLMHandle.wrap(llm)
        self._llm_validator = llm_validator or llm_admin.validate_llm
        self._llm_models = llm_models or llm_admin.fetch_models
        self._llm_factory = llm_factory or make_llm
        self.registry = registry
        self.orchestrator = orchestrator
        self.env_path = Path(env_path) if env_path is not None else ROOT / ".env"
        self.db = db if db is not None else AppDB(settings.app_db_path)
        self.evidence = EvidenceStore(self.db)
        self.sessions = SessionStore(self.db)
        self.tickets = TicketStore(self.db, clock=clock) if clock is not None else TicketStore(self.db)
        self.lock = asyncio.Lock()
        # Bumped by every kill-switch press, before it waits for the lock: a settings write that
        # started earlier may not turn trading back on (`apply_settings`).
        self.kill_generation = 0
        # Whether the kill switch suspended trading at the broker (for the keys now in service):
        # re-enabling trading then resumes it. In memory only (the ledger keeps the history).
        self._broker_suspended_by_app = False
        # connected | missing | invalid | unreachable | not_paper; None = unchecked
        self.alpaca_status: str | None = None
        # The Alpaca key pair found to belong to a non-paper account: no broker is built for it.
        self._not_paper_keys: tuple[str, str] | None = None
        self._broker_factory = broker_factory or (lambda key_id, secret: AlpacaPaper(key_id, secret))
        self._data_factory = data_factory or (lambda key_id, secret, http: AlpacaData(key_id, secret, http))
        self._http_factory = http_factory or HttpClient
        self._sleep = sleep
        self._grace = close_grace_sec
        self._close_sleep = close_sleep or asyncio.sleep  # the grace-close wait (tests: virtual time)
        self._monotonic = monotonic or time.monotonic
        self._alpaca_checked_at: float | None = None
        # The status check for `GET /api/settings` that is in flight, shared by concurrent callers.
        self._status_check: asyncio.Task | None = None
        self._retired: list[Any] = []
        self._close_tasks: set[asyncio.Task] = set()
        self._secrets_seen: set[str] = set()
        # (session_id, turn_id) -> ticket id: the toolbox's one-proposal-per-turn slots. Owned here
        # and handed to every toolbox built, so a rebuild in the middle of a turn cannot reset them.
        self.turn_slots: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.http = self.ledger = self.edgar = self.web = self.grounded = None
        self.broker = self.data = self.executor = self.toolbox = None
        self.rebuild()
        self.agent = ChatAgent(self.llm, _ToolboxHandle(self), self.sessions, settings)

    # --- building --------------------------------------------------------------------------------

    def _mandate_for(self, session_id: str) -> Mandate:
        return self.sessions.get_mandate(session_id) or Mandate()

    def _known_secrets(self, *candidates: Any) -> list[str]:
        """Every secret value this process has used, current or replaced (turns still running on
        the old components may log an old key after a rebuild), plus `candidates` (the values a
        build in progress is about to use)."""
        values = [getattr(self.settings, name, None) for name in SECRET_FIELDS]
        for value in [*values, *candidates]:
            if isinstance(value, str) and value:
                self._secrets_seen.add(value)
        return sorted(self._secrets_seen)

    def rebuild(self) -> None:
        """(Re)create the rebuildable components from the current settings and swap them in. Old
        clients are closed after the grace delay. If building fails nothing is swapped.

        It is synchronous (no await), so it runs to the end without another task in between, and
        it does not need `lock`: `_drop_non_paper` calls it with the lock held (a confirm) and
        without it (the propose path, the `get_account` tool). That is safe because no lock-holder
        carries state across an await that a rebuild would invalidate: `apply_settings` builds,
        writes and swaps with no await between them, and a confirm or the kill switch keeps the
        executor it started with, whose clients stay open for the grace delay."""
        self._swap(self._build())

    def _build(self, overrides: dict[str, Any] | None = None) -> "_Built":
        """Construct the rebuildable components for the settings as they would be with `overrides`
        (key fields -> new value) applied, touching nothing shared (except the log scrubber's
        secret list, which only grows). On failure the clients made so far are retired and the
        exception propagates."""
        s = self.settings

        def value(name: str) -> Any:
            return overrides[name] if overrides and name in overrides else getattr(s, name, None)

        key_id, secret = value("alpaca_api_key_id"), value("alpaca_api_secret_key")
        tavily_key, sec_user_agent = value("tavily_api_key"), value("sec_user_agent")
        gemini_key = value("gemini_api_key")
        made: list[Any] = []
        try:
            http = self._http_factory()
            made.append(http)
            # One accumulated list for the three scrubs (ledger, logs, toolbox). The log scrubber
            # is process-wide, so it gets the list now, even if this build fails: it only ever grows.
            known_secrets = self._known_secrets(key_id, secret, tavily_key, gemini_key,
                                                value("openai_api_key"), value("anthropic_api_key"))
            set_log_secrets(known_secrets)
            ledger = Ledger(self.db, known_secrets=known_secrets)
            edgar = Edgar(http, self.evidence, sec_user_agent) if sec_user_agent else None
            web = WebSearch(http, self.evidence, tavily_key)
            # Gemini search grounding needs only a Gemini key: it is a research connector of its
            # own, whichever provider runs the chat.
            grounded = GroundedWeb(http, gemini_key, s.grounding_model) if gemini_key else None
            # A write that applies a key pair has checked it: the mark of a non-paper pair does not
            # hold that pair back (`apply_settings` clears the mark once the write is in).
            applies_pair = bool(overrides) and bool(_ALPACA_PAIR & overrides.keys())
            blocked = None if applies_pair else self._not_paper_keys
            broker = data = executor = on_not_paper = None
            if key_id and secret and (key_id, secret) != blocked:
                broker = self._broker_factory(key_id, secret)
                made.append(broker)
                data = self._data_factory(key_id, secret, http)
                extra = {"sleep": self._sleep} if self._sleep is not None else {}
                # Told by the executor, and by the toolbox's `get_account`, when this broker's account
                # is not a paper account: drops this broker, if it is still the one in service.
                on_not_paper = functools.partial(self._drop_non_paper, broker)
                executor = Executor(broker, data, self.tickets, ledger, s,
                                    mandate_for=self._mandate_for, on_not_paper=on_not_paper, **extra)
            toolbox = ChatToolbox(
                registry=self.registry, orchestrator=self.orchestrator, broker=broker, data=data,
                edgar=edgar, web=web, grounded=grounded, evidence=self.evidence,
                sessions=self.sessions, tickets=self.tickets, executor=executor, settings=s,
                turn_slots=self.turn_slots, ledger=ledger, known_secrets=known_secrets,
                on_not_paper=on_not_paper,
            )
        except BaseException:
            self._retire(made)
            raise
        return _Built(http, ledger, edgar, web, grounded, broker, data, executor, toolbox)

    def _swap(self, built: "_Built") -> None:
        """Put `built` in service in one synchronous step (no await, so no request sees a mix)."""
        old = [c for c in (self.http, self.broker) if c is not None]
        self.http, self.ledger, self.edgar = built.http, built.ledger, built.edgar
        self.web, self.grounded = built.web, built.grounded
        self.broker, self.data, self.executor = built.broker, built.data, built.executor
        self.toolbox = built.toolbox
        self._retire(old)
        log.info(
            "chat runtime built: broker=%s edgar=%s web_search=%s grounding=%s",
            built.broker is not None, built.edgar is not None,
            bool(getattr(self.settings, "tavily_api_key", None)), built.grounded is not None,
        )

    def _llm_grace_sec(self) -> float:
        """How long a retired model client must live: the longest turn the settings allow, the
        nudge grace, one more model call (`llm_timeout_sec`) and a margin."""
        try:
            timeout = float(getattr(self.settings, "llm_timeout_sec", 40.0))
        except (TypeError, ValueError):
            timeout = 40.0
        turn = longest_turn_sec(self.settings) + nudge_grace_sec(self.settings) + timeout
        return max(self._grace, turn + LLM_GRACE_MARGIN_SEC)

    def _retire(self, components: list[Any], grace: float | None = None) -> None:
        """Close `components` after `grace` seconds (default `close_grace_sec`)."""
        if not components:
            return
        self._retired.extend(components)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # not running yet: shutdown() closes them
        delay = self._grace if grace is None else grace

        async def close_later() -> None:
            await self._close_sleep(delay)
            for component in components:
                if component in self._retired:
                    self._retired.remove(component)
                    await _aclose(component)

        task = loop.create_task(close_later())
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)

    # --- lifecycle -------------------------------------------------------------------------------

    async def startup(self) -> None:
        """Warn about environment-only settings, expire stale tickets, settle stuck ones and check the
        Alpaca keys. Never raises."""
        for name in env_fallbacks(self.env_path):
            log.warning("startup: %s comes from the process environment, not .env; clearing it in "
                        "the app will not survive a restart", name)
        try:
            expired = self.tickets.expire_stale()
            if expired:
                log.info("startup: %d stale ticket(s) expired", expired)
        except Exception as exc:  # noqa: BLE001
            log.warning("startup: expiring stale tickets failed (%s)", type(exc).__name__)
        executor = self.executor
        try:
            if executor is not None:
                moved = await executor.recover_stuck()
                if moved:
                    log.warning("startup: %d ticket(s) stuck in 'submitting' were reconciled", moved)
            else:
                rows = self.db.query("SELECT COUNT(*) AS n FROM tickets WHERE status=?",
                                     (TicketStatus.submitting.value,))
                if rows and rows[0]["n"]:
                    log.warning("startup: %d ticket(s) stuck in 'submitting'; no broker is configured, "
                                "so they are left for a later startup", rows[0]["n"])
        except Exception as exc:  # noqa: BLE001
            log.warning("startup: stuck-ticket recovery failed (%s)", type(exc).__name__)
        await self.refresh_alpaca_status()

    async def shutdown(self) -> None:
        if self._status_check is not None:
            self._status_check.cancel()  # a no-op once it is done
        for task in list(self._close_tasks):
            task.cancel()
        for component in [*self._retired, self.http, self.broker, self.llm.current]:
            if component is not None:
                await _aclose(component)
        self._retired.clear()

    # --- Alpaca keys -----------------------------------------------------------------------------

    async def refresh_alpaca_status(self) -> str:
        """Check the broker in service: connected, missing, invalid, unreachable or not_paper. A
        non-paper account costs the runtime its broker (`drop_non_paper_broker`)."""
        broker = self.broker
        if broker is None:
            # No broker for the keys of a known non-paper account is `not_paper`, not `missing`.
            blocked = self._not_paper_keys is not None and self._not_paper_keys == self._alpaca_keys()
            status = "not_paper" if blocked else "missing"
        else:
            try:
                status = account_status(await broker.account())
            except Exception as exc:  # noqa: BLE001 - classified, logged by type only
                status = alpaca_status_for(exc)
                log.info("alpaca key check: %s (%s)", status, type(exc).__name__)
        self._alpaca_checked_at = self._monotonic()
        if status == "not_paper" and broker is not None:
            await self.drop_non_paper_broker(broker)
            return self.alpaca_status
        if broker is self.broker:  # not replaced meanwhile (a settings write sets its own status)
            self.alpaca_status = status
        return self.alpaca_status if self.alpaca_status is not None else status

    async def refresh_alpaca_status_if_due(self) -> str | None:
        """For `GET /api/settings`: check now if never checked; re-check an unsettled status
        (`unreachable`, `invalid`, `missing`) at most once every `ALPACA_RECHECK_SEC`.

        One check at a time: while one is in flight (Alpaca can take the broker's whole timeout to
        answer), a caller that finds the status unsettled awaits that same task instead of making
        its own `account()` call. The task is shielded: a caller that goes away (a closed request)
        neither cancels it nor stops the others from getting its result."""
        status = self.alpaca_status
        if status in SETTLED_ALPACA:
            return status
        check = self._status_check
        if check is None or check.done():
            checked = self._alpaca_checked_at
            if status is not None and checked is not None and self._monotonic() - checked < ALPACA_RECHECK_SEC:
                return status
            check = self._status_check = asyncio.ensure_future(self.refresh_alpaca_status())
            check.add_done_callback(self._status_check_done)
        return await asyncio.shield(check)

    def _status_check_done(self, check: asyncio.Task) -> None:
        if self._status_check is check:
            self._status_check = None
        if not check.cancelled():
            check.exception()  # marks it retrieved: the callers awaiting it get the error

    def _alpaca_keys(self) -> tuple[Any, Any]:
        return (getattr(self.settings, "alpaca_api_key_id", None),
                getattr(self.settings, "alpaca_api_secret_key", None))

    def _drop_non_paper(self, broker: Any) -> bool:
        """Synchronous core of `drop_non_paper_broker`, also the `on_not_paper` callback of the
        executor and of the toolbox's `get_account`. A confirm calls it while holding the lock; the
        propose path and the chat tool call it without. That is safe because `rebuild` is
        synchronous and no lock-holder carries state across an await that a rebuild would
        invalidate (see `rebuild`). Drops only `broker`, and only while it is still the one in
        service; returns whether it did."""
        if broker is None or broker is not self.broker:
            return False  # already replaced (new keys were saved meanwhile): nothing to drop
        executor = self.executor
        if executor is not None:
            # Built with this broker: a confirm still running on it submits no further leg.
            executor.mark_not_paper()
        self._not_paper_keys = self._alpaca_keys()
        self.alpaca_status = "not_paper"
        self._alpaca_checked_at = self._monotonic()
        log.warning("the Alpaca keys belong to a non-paper account: trading is not connected")
        self._audit("broker_dropped", {"reason": "not_paper"})
        self.rebuild()
        return True

    def _audit(self, kind: str, payload: dict[str, Any]) -> None:
        """A system ledger entry; a failed audit write never stops the action it records."""
        try:
            self.ledger.append(SYSTEM_SESSION, kind, payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s ledger entry failed (%s)", kind, type(exc).__name__)

    async def drop_non_paper_broker(self, broker: Any) -> bool:
        """`broker`, which has just read a non-paper account, is dropped if it is still in service:
        rebuild without a broker, executor or market data (as if the keys were missing) and
        remember the key pair, so no later rebuild brings them back. The keys stay in the settings
        and the env file; the status is `not_paper` until other keys are saved."""
        async with self.lock:
            return self._drop_non_paper(broker)

    async def check_alpaca_keys(self, key_id: str, secret: str) -> str:
        """Try candidate keys with a temporary broker (`account()` on the paper host) and close it.
        Returns connected, invalid, unreachable or not_paper (a live account's keys). It changes
        nothing: a mark on a non-paper pair is cleared only by `apply_settings`, once the write that
        applies the pair is in (the rest of the request may still fail). The candidate keys join
        the log scrub first, before anything is built or called with them."""
        self._note_secrets(key_id, secret)
        try:
            broker = self._broker_factory(key_id, secret)
        except ValueError:
            return "invalid"
        try:
            status = account_status(await broker.account())
        except Exception as exc:  # noqa: BLE001 - classified, logged by type only
            status = alpaca_status_for(exc)
            log.info("alpaca candidate key check: %s (%s)", status, type(exc).__name__)
            return status
        finally:
            await _aclose(broker)
        return status

    # --- the model provider ------------------------------------------------------------------------

    def _candidate(self, changes: dict[str, Any]) -> Any:
        """A copy of the settings as they would be with `changes` applied; the shared settings
        object is not touched."""
        copy = getattr(self.settings, "model_copy", None)
        if callable(copy):
            return copy(update=dict(changes))
        return SimpleNamespace(**{**vars(self.settings), **changes})

    def _note_secrets(self, *values: Any) -> None:
        """Add candidate secret values to the log scrubber (see the module docstring)."""
        set_log_secrets(self._known_secrets(*values))

    def _needs_new_llm(self, changes: dict[str, Any]) -> bool:
        """Whether `changes` alter what the model client was built from: the provider, the model,
        the effort, or the key of the provider that will be in use."""
        candidate = self._candidate(changes)
        key_field = llm_admin.KEY_FIELDS.get(getattr(candidate, "llm_provider", None))
        names = [*_LLM_CLIENT_FIELDS, *([key_field] if key_field else [])]
        return any(getattr(candidate, name, None) != getattr(self.settings, name, None) for name in names)

    async def check_llm(self, changes: dict[str, Any]) -> str:
        """One minimal model call for the configuration as it would be with `changes` (the
        injected validator; `llm_admin.validate_llm` in production), built from a copy of the
        settings. Returns ok, invalid_key, model_unavailable, rate_limited or unreachable; never
        raises, and never logs more than an exception type."""
        candidate = self._candidate(changes)
        self._note_secrets(*(getattr(candidate, name, None) for name in llm_admin.KEY_FIELDS.values()))
        try:
            status = await self._llm_validator(candidate)
        except Exception as exc:  # noqa: BLE001 - classified, logged by type only
            status = llm_admin.classify_llm_error(exc)
            log.info("llm check: %s (%s)", status, type(exc).__name__)
        return status if status in llm_admin.STATUSES else "unreachable"

    def llm_context(self, changes: dict[str, Any]) -> tuple[Any, Any, Any]:
        """(provider, model, that provider's key) as they would be with `changes` applied: what a
        live check validates. `PUT /api/settings` records it when it decides, and
        `apply_settings` compares it again under the lock."""
        candidate = self._candidate(changes)
        provider = getattr(candidate, "llm_provider", None)
        key_field = llm_admin.KEY_FIELDS.get(provider)
        return provider, getattr(candidate, "llm_model", None), getattr(candidate, key_field, None) if key_field else None

    async def _recheck_llm(self, changes: dict[str, Any]) -> str:
        """The re-check under the lock (see `apply_settings`): ok or rate_limited, else raises
        `llm_admin.LLMCheckFailed`."""
        provider, model, key = self.llm_context(changes)
        if provider not in llm_admin.KEY_FIELDS:
            return "ok"
        if not model:
            raise llm_admin.LLMCheckFailed("model_required")
        if not key:
            raise llm_admin.LLMCheckFailed("key_required")
        status = await self.check_llm(changes)
        if status not in ("ok", "rate_limited"):
            raise llm_admin.LLMCheckFailed(status)
        return status

    async def list_llm_models(self, provider: str, api_key: str) -> list[dict]:
        """The provider's unfiltered model list for `api_key` (the injected lister;
        `llm_admin.fetch_models` in production). Raises what the lister raises: the caller
        classifies it. The key is not saved; it only joins the scrub list."""
        self._note_secrets(api_key)
        return await self._llm_models(provider, api_key, self.settings)

    # --- settings writes -------------------------------------------------------------------------

    async def apply_settings(self, changes: dict[str, Any], *, alpaca_status: str | None = None,
                             llm_context: tuple[Any, Any, Any] | None = None,
                             kill_generation: int | None = None,
                             outcome: dict[str, Any] | None = None) -> str | None:
        """Apply `changes` (settings field -> value; None removes the key), all or nothing:
        1. build the new components for the changed keys, and the new model client if the
           provider, the model, the effort or that provider's key changes (a failure changes
           nothing);
        2. write the env file atomically (a failure discards what was built and changes nothing);
        3. only then update the shared settings in place, swap the components in and swap the
           model client into the handle. None of those can fail, so the env file, the settings,
           the components and the model never disagree. The old model client is retired (closed
           after `_llm_grace_sec`), and the agent drops its cached provider conversations.
        `llm_context` is what the caller's LLM decision was made for (`llm_context(changes)` at
        that time). If another write changed the provider, the model or a key in between, the
        configuration these changes now produce is checked again here, under the lock: a failure
        raises `llm_admin.LLMCheckFailed` and changes nothing. Returns that re-check's status
        (`ok` or `rate_limited`), or None when none was needed.

        **The kill switch wins.** `kill_generation` is `self.kill_generation` as the caller saw it
        when its request started (default: now). If the kill switch has been pressed since, a
        write that turns trading on is refused with `KillSwitchPressed` and changes nothing. The
        check runs after the last await before the write, so a press while this call waits for
        the lock or re-checks the model is always seen.

        **Re-enabling.** When the write turns trading on and the kill switch had suspended trading
        at the broker (for the keys still in service), trading is resumed there too
        (`set_suspend_trade(False)`, best effort). `outcome`, when given, receives
        `broker_resumed`: True, False (it failed: the next re-enable tries again) or None
        (nothing to resume). Every write is ledgered as `settings_changed` under the system
        session with the changed field names only, never a value (plus the new `trading_enabled`
        when that switch is among them, and `broker_resumed`)."""
        generation = self.kill_generation if kill_generation is None else kill_generation
        async with self.lock:
            recheck = None
            if llm_context is not None and self.llm_context(changes) != llm_context:
                recheck = await self._recheck_llm(changes)
            if changes.get("trading_enabled") is True and self.kill_generation != generation:
                raise KillSwitchPressed()
            made: list[Any] = []
            try:
                built = self._build(changes) if _REBUILD_FIELDS & set(changes) else None
                if built is not None:
                    made += [c for c in (built.http, built.broker) if c is not None]
                new_llm = self._llm_factory(self._candidate(changes)) if self._needs_new_llm(changes) else None
                if new_llm is not None:
                    made.append(new_llm)
                update_env_file(
                    self.env_path, {ENV_NAMES[name]: _env_value(value) for name, value in changes.items()}
                )
            except BaseException:
                self._retire(made)
                raise
            for name, value in changes.items():
                setattr(self.settings, name, value)
            if built is not None:
                self._swap(built)
            if new_llm is not None:
                # A turn in flight holds the old client until it ends: keep it that long.
                self._retire([self.llm.swap(new_llm)], grace=self._llm_grace_sec())
                log.info("model client swapped: %s", new_llm.name)
            if _CONVERSATION_FIELDS & set(changes):
                self.agent.reset_conversations()
            if _ALPACA_PAIR & set(changes):
                if self._not_paper_keys == self._alpaca_keys():
                    # This pair is applied now (the caller's check said it is no longer a live
                    # account, and the build above did not hold it back): it may have a broker.
                    self._not_paper_keys = None
                # The caller has just checked the new keys; without keys there is no broker.
                self.alpaca_status = "missing" if self.broker is None else alpaca_status
                self._alpaca_checked_at = self._monotonic()
                # Another account (or none): a suspension the app made is not this one's.
                self._broker_suspended_by_app = False
            audit: dict[str, Any] = {"fields": ", ".join(sorted(changes))}
            if "trading_enabled" in changes:
                audit["trading_enabled"] = bool(self.settings.trading_enabled)
            if changes.get("trading_enabled") is True:
                resumed = await self._resume_broker()
                if outcome is not None:
                    outcome["broker_resumed"] = resumed
                if resumed is not None:
                    audit["broker_resumed"] = resumed
            self._audit("settings_changed", audit)
            return recheck

    async def _resume_broker(self) -> bool | None:
        """Undo the kill switch's broker suspension, best effort: True when resumed, False when
        the call failed (the mark stays, so the next re-enable tries again), None when the app had
        not suspended anything."""
        if not self._broker_suspended_by_app:
            return None
        broker = self.broker
        if broker is None:
            return False
        try:
            await broker.set_suspend_trade(False)
        except Exception as exc:  # noqa: BLE001 - logged by type only; reported to the caller
            log.warning("re-enable: resuming trading at the broker failed (%s)", type(exc).__name__)
            return False
        self._broker_suspended_by_app = False
        return True

    def _persist_trading_off(self) -> bool:
        try:
            update_env_file(self.env_path, {ENV_NAMES["trading_enabled"]: "false"})
            return True
        except Exception as exc:  # noqa: BLE001 - the switch is already off in memory
            log.warning("kill switch: writing TRADING_ENABLED failed (%s)", type(exc).__name__)
            return False

    async def kill_switch(self, suspend_broker: bool) -> dict:
        """Trading off now, before waiting for the lock: the shared flag (a confirm in flight
        stops before its next leg), the kill generation (a settings write already under way can no
        longer turn trading on, see `apply_settings`) and `TRADING_ENABLED=false` in the env file
        (a restart keeps it off even while the broker steps below hang on their timeouts). Then,
        under the lock, the broker steps (`Executor.kill_switch`) and the ledger entry. Whether
        the broker was suspended is remembered, so that re-enabling can resume it."""
        self.settings.trading_enabled = False
        self.kill_generation += 1
        persisted = self._persist_trading_off()
        async with self.lock:
            self.settings.trading_enabled = False
            executor = self.executor
            if executor is not None:
                payload = await executor.kill_switch(suspend_broker, persisted=persisted)
                if payload.get("broker_suspended"):
                    self._broker_suspended_by_app = True
            else:
                self._audit("kill_switch", {
                    "trading_enabled": False, "suspend_broker": suspend_broker,
                    "broker_connected": False, "persisted": persisted,
                })
        return {
            "trading_enabled": False,
            "suspend_broker": suspend_broker,
            "broker_connected": executor is not None,
            "persisted": persisted,
        }


def _env_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass
class AppState:
    settings: Settings
    rdb: RDB
    index: SearchIndex
    resolver: EntityResolver
    kg: KG | None
    vectors: VectorIndex | None
    docs: DocStore
    ontology: Ontology | None
    lexicon: Lexicon
    llm: LLMClient
    tools: ToolRegistry
    orchestrator: Orchestrator
    system_prompt: str
    chat: ChatRuntime | None = None

    @classmethod
    def load(cls, settings: Settings, llm: LLMClient | None = None, build_graph: bool = True) -> "AppState":
        t0 = time.time()
        # Scrub finagent log records from the first line on (ChatRuntime, built last, keeps the
        # secret list current after that).
        install_log_scrubber()
        set_log_secrets([getattr(settings, name, None) for name in SECRET_FIELDS])
        rdb = RDB(settings.store_path, row_limit=settings.sql_row_limit, timeout_sec=settings.sql_timeout_sec)
        lexicon = Lexicon.load(settings.enrichment_dir)
        ontology: Ontology | None = None
        try:
            ontology = Ontology.load(settings.ontology_dir)
            log.info("ontology: %d triples", len(ontology.graph))
        except Exception as e:  # noqa: BLE001
            log.warning("ontology not loaded (%s); using fallback validators", e)
            ontology = FallbackValidators()  # type: ignore[assignment]
        index = SearchIndex.from_rdb(rdb)
        log.info("index: %d docs", len(index.docs))
        resolver = EntityResolver(index, lexicon)
        vectors: VectorIndex | None = None
        if settings.vector_backend != "off":
            embed_fn = None
            if settings.vector_backend == "openai":
                try:
                    embed_fn = make_openai_embed_fn(settings, settings.vector_embed_model)
                except Exception as e:  # noqa: BLE001
                    log.warning("embedding backend unavailable (%s); falling back to tfidf-lsa", e)
            try:
                vectors = VectorIndex.build(
                    index.docs, dims=settings.vector_dims, max_df=settings.vector_max_df,
                    cache_path=settings.vector_cache_path if embed_fn is None else None,
                    embed_fn=embed_fn, log_fn=log.info,
                )
            except Exception as e:  # noqa: BLE001 - retrieval must still work without vectors
                log.warning("vector index unavailable (%s); lexical + graph only", e)
        kg: KG | None = None
        if build_graph:
            kg = KG.build(rdb, lexicon, ontology, log=log.info)
        docs = DocStore.load(settings.docs_dir)
        # One handle for the QA orchestrator and the chat runtime: a provider or model switch
        # (PUT /api/settings) swaps the client behind it for both.
        llm = LLMHandle.wrap(llm or make_llm(settings))
        tools = ToolRegistry(rdb, index, resolver, kg, docs, ontology, lexicon, settings, vectors=vectors)
        system_prompt = build_system_prompt(ontology, rdb.meta(), docs.titles(), settings, hints=value_hints(rdb))
        orch = Orchestrator(settings, llm, tools, resolver, ontology, lexicon, system_prompt)
        chat: ChatRuntime | None = None
        try:
            chat = ChatRuntime(settings, llm=llm, registry=tools, orchestrator=orch)
        except Exception as exc:  # noqa: BLE001 - /answer must keep working without /api/*
            log.error("chat runtime unavailable (%s); /api/* will answer 503", type(exc).__name__)
        log.info("app state ready in %.1fs (llm=%s/%s)", time.time() - t0, llm.name, settings.llm_model)
        return cls(settings, rdb, index, resolver, kg, vectors, docs, ontology, lexicon, llm, tools, orch,
                   system_prompt, chat)
