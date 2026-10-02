"""Builds every runtime component once (startup) and hands them to the API.

`AppState` is the single-question QA stack behind `/answer`. `ChatRuntime` is the chat and trading
stack behind `/api/*`; `AppState.load` builds it last, and a failure there only disables `/api/*`
(`chat` stays None and the router answers 503), never `/answer`.

**ChatRuntime.** One instance per process of each persistent store over one `AppDB`: `evidence`
(its id lock is per instance), `sessions`, `tickets` (its lifecycle lock is per instance), plus
exactly one `ChatAgent` (its per-session turn locks must be unique). The rest is *rebuildable*
from the current settings by `rebuild()`: the HTTP client, the ledger (its known-secret list), the
connectors, the Alpaca broker and market data (only when both keys are set), the executor (only
with a broker) and the toolbox. The agent reaches the current toolbox through `_ToolboxHandle`, so
a rebuild takes effect at the next tool call; the toolbox's one-proposal-per-turn slots
(`turn_slots`) are owned by the runtime and shared by every toolbox it builds, so a rebuild in the
middle of a turn cannot reset them. Every component shares the *same* mutable `settings` object,
so the kill switch (`settings.trading_enabled = False`) is seen everywhere at once.

A rebuild is build-then-swap: `_build()` constructs a full set of components and touches nothing
shared (a failure retires what it made and changes nothing); `_swap()` puts them in service
without awaiting, and closes the old HTTP clients only after `close_grace_sec` (120 s), so turns
and confirms already in flight finish on the old objects. `apply_settings()` is all or nothing:
build for the new values, write the env file, and only then update the settings in place and swap.
`lock` serialises rebuilds against confirm, reject, refresh and the kill switch; chat turns do not
take it.

`startup()` (from the app lifespan) warns, by name only, about writable settings that the process
environment-only settings (`config.env_fallbacks`), expires stale tickets, settles tickets stuck in
`submitting` (`Executor.recover_stuck`; without a broker it only logs how many there are) and
checks the Alpaca keys. Nothing here logs a secret or an exception message: failures are logged by
type.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent.llm import LLMClient, make_llm
from .agent.orchestrator import Orchestrator
from .agent.prompts import build_system_prompt, value_hints
from .agent.tools import ToolRegistry
from .chat.agent import ChatAgent
from .chat.sessions import SessionStore
from .chat.toolbox import ChatToolbox
from .config import ENV_NAMES, Settings, env_fallbacks
from .envfile import update_env_file
from .kb.docs import DocStore
from .kb.entity import EntityResolver
from .kb.graph import KG
from .kb.lexicon import Lexicon
from .kb.rdb import RDB
from .kb.search import SearchIndex
from .kb.vector import VectorIndex, make_openai_embed_fn
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

CLOSE_GRACE_SEC = 120.0
# Settings fields whose secret values the ledger scrubs verbatim.
SECRET_FIELDS = (
    "alpaca_api_key_id", "alpaca_api_secret_key", "gemini_api_key", "tavily_api_key",
    "anthropic_api_key", "openai_api_key",
)
# Changing any of these needs new connectors (they capture keys at construction).
_REBUILD_FIELDS = frozenset(ENV_NAMES) - {"trading_enabled"}


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
    ):
        self.settings = settings
        self.llm = llm
        self.registry = registry
        self.orchestrator = orchestrator
        self.env_path = Path(env_path) if env_path is not None else ROOT / ".env"
        self.db = db if db is not None else AppDB(settings.app_db_path)
        self.evidence = EvidenceStore(self.db)
        self.sessions = SessionStore(self.db)
        self.tickets = TicketStore(self.db, clock=clock) if clock is not None else TicketStore(self.db)
        self.lock = asyncio.Lock()
        self.alpaca_status: str | None = None  # connected|missing|invalid|unreachable, None = unchecked
        self._broker_factory = broker_factory or (lambda key_id, secret: AlpacaPaper(key_id, secret))
        self._data_factory = data_factory or (lambda key_id, secret, http: AlpacaData(key_id, secret, http))
        self._http_factory = http_factory or HttpClient
        self._sleep = sleep
        self._grace = close_grace_sec
        self._retired: list[Any] = []
        self._close_tasks: set[asyncio.Task] = set()
        self._secrets_seen: set[str] = set()
        # (session_id, turn_id) -> ticket id: the toolbox's one-proposal-per-turn slots. Owned here
        # and handed to every toolbox built, so a rebuild in the middle of a turn cannot reset them.
        self.turn_slots: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.http = self.ledger = self.edgar = self.web = self.grounded = None
        self.broker = self.data = self.executor = self.toolbox = None
        self.rebuild()
        self.agent = ChatAgent(llm, _ToolboxHandle(self), self.sessions, settings)

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
        """(Re)create the rebuildable components from the current settings and swap them in. Call
        it under `lock` once the app is running. Old clients are closed after the grace delay. If
        building fails nothing is swapped."""
        self._swap(self._build())

    def _build(self, overrides: dict[str, Any] | None = None) -> "_Built":
        """Construct the rebuildable components for the settings as they would be with `overrides`
        (key fields -> new value) applied, touching nothing shared. On failure the clients made so
        far are retired and the exception propagates."""
        s = self.settings

        def value(name: str) -> Any:
            return overrides[name] if overrides and name in overrides else getattr(s, name, None)

        key_id, secret = value("alpaca_api_key_id"), value("alpaca_api_secret_key")
        tavily_key, sec_user_agent = value("tavily_api_key"), value("sec_user_agent")
        made: list[Any] = []
        try:
            http = self._http_factory()
            made.append(http)
            ledger = Ledger(self.db, known_secrets=self._known_secrets(key_id, secret, tavily_key))
            edgar = Edgar(http, self.evidence, sec_user_agent) if sec_user_agent else None
            web = WebSearch(http, self.evidence, tavily_key)
            grounded = None
            if getattr(s, "llm_provider", None) == "gemini" and s.gemini_api_key:
                grounded = GroundedWeb(http, s.gemini_api_key, s.grounding_model)
            broker = data = executor = None
            if key_id and secret:
                broker = self._broker_factory(key_id, secret)
                made.append(broker)
                data = self._data_factory(key_id, secret, http)
                extra = {"sleep": self._sleep} if self._sleep is not None else {}
                executor = Executor(broker, data, self.tickets, ledger, s,
                                    mandate_for=self._mandate_for, **extra)
            toolbox = ChatToolbox(
                registry=self.registry, orchestrator=self.orchestrator, broker=broker, data=data,
                edgar=edgar, web=web, grounded=grounded, evidence=self.evidence,
                sessions=self.sessions, tickets=self.tickets, executor=executor, settings=s,
                turn_slots=self.turn_slots,
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

    def _retire(self, components: list[Any]) -> None:
        if not components:
            return
        self._retired.extend(components)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # not running yet: shutdown() closes them

        async def close_later() -> None:
            await asyncio.sleep(self._grace)
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
        for task in list(self._close_tasks):
            task.cancel()
        for component in [*self._retired, self.http, self.broker]:
            if component is not None:
                await _aclose(component)
        self._retired.clear()

    # --- Alpaca keys -----------------------------------------------------------------------------

    async def refresh_alpaca_status(self) -> str:
        broker = self.broker
        if broker is None:
            status = "missing"
        else:
            try:
                await broker.account()
                status = "connected"
            except Exception as exc:  # noqa: BLE001 - classified, logged by type only
                status = alpaca_status_for(exc)
                log.info("alpaca key check: %s (%s)", status, type(exc).__name__)
        self.alpaca_status = status
        return status

    async def check_alpaca_keys(self, key_id: str, secret: str) -> str:
        """Try candidate keys with a temporary broker (`account()` on the paper host) and close it.
        Returns connected, invalid or unreachable."""
        try:
            broker = self._broker_factory(key_id, secret)
        except ValueError:
            return "invalid"
        try:
            await broker.account()
            return "connected"
        except Exception as exc:  # noqa: BLE001 - classified, logged by type only
            status = alpaca_status_for(exc)
            log.info("alpaca candidate key check: %s (%s)", status, type(exc).__name__)
            return status
        finally:
            await _aclose(broker)

    # --- settings writes -------------------------------------------------------------------------

    async def apply_settings(self, changes: dict[str, Any], *, alpaca_status: str | None = None) -> None:
        """Apply `changes` (settings field -> value; None removes the key), all or nothing:
        1. build the new components for the changed keys (a failure changes nothing);
        2. write the env file atomically (a failure discards what was built and changes nothing);
        3. only then update the shared settings in place and swap the components in. Neither of
           those can fail, so the env file, the settings and the components never disagree."""
        async with self.lock:
            built = self._build(changes) if _REBUILD_FIELDS & set(changes) else None
            try:
                update_env_file(
                    self.env_path, {ENV_NAMES[name]: _env_value(value) for name, value in changes.items()}
                )
            except BaseException:
                if built is not None:
                    self._retire([c for c in (built.http, built.broker) if c is not None])
                raise
            for name, value in changes.items():
                setattr(self.settings, name, value)
            if built is not None:
                self._swap(built)
            if {"alpaca_api_key_id", "alpaca_api_secret_key"} & set(changes):
                # The caller has just checked the new keys; without keys there is no broker.
                self.alpaca_status = "missing" if self.broker is None else alpaca_status

    async def kill_switch(self, suspend_broker: bool) -> dict:
        """Trading off now (before waiting for the lock, so an in-flight confirm stops before its
        next leg), then the broker steps and the env file under the lock."""
        self.settings.trading_enabled = False
        async with self.lock:
            executor = self.executor
            if executor is not None:
                await executor.kill_switch(suspend_broker)
            else:
                self.settings.trading_enabled = False
                try:
                    self.ledger.append(SYSTEM_SESSION, "kill_switch", {
                        "trading_enabled": False, "suspend_broker": suspend_broker,
                        "broker_connected": False,
                    })
                except Exception as exc:  # noqa: BLE001
                    log.warning("kill switch ledger entry failed (%s)", type(exc).__name__)
            try:
                update_env_file(self.env_path, {ENV_NAMES["trading_enabled"]: "false"})
                persisted = True
            except Exception as exc:  # noqa: BLE001 - the switch is already off in memory
                log.warning("kill switch: writing TRADING_ENABLED failed (%s)", type(exc).__name__)
                persisted = False
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
        llm = llm or make_llm(settings)
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
