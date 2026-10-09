from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import DATA_DIR, ONTOLOGY_DIR, ROOT


class Settings(BaseSettings):
    # An empty value ("GEMINI_API_KEY=") means "not set" in both `.env` and the environment.
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings):
        """`.env` wins over the process environment (pydantic-settings' default is the reverse).

        The app writes `.env` itself (PUT /api/settings, the kill switch), so what it saved must be
        what the next start reads: a stale `GEMINI_API_KEY` or a `TRADING_ENABLED=true` exported in
        some shell must not undo it. The environment still supplies whatever `.env` leaves out,
        which is how a container without a `.env` file is configured."""
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # LLM
    # English edition: Google Gemini through its OpenAI-compatible endpoint. The competition build had to
    # use NCP HyperCLOVA X (organizer rule); Anthropic and OpenAI remain available as alternates.
    llm_provider: Literal["gemini", "anthropic", "openai", "stub"] = "gemini"
    llm_model: str = "gemini-3.8-flash"
    # Sent to Gemini as reasoning_effort (xhigh/max map to high) and to Anthropic as output effort.
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Output-token ceiling per call. Gemini's thinking models count thinking tokens against it, so it must
    # leave room for thinking plus the answer; the visible answer itself is short (54 answers recorded in the
    # competition: median 108 characters, longest 610 - G-026, a list of nine monthly-dividend ETFs).
    # History: the competition build capped this at 1536 for HyperCLOVA X, which has no thinking tokens, after
    # a 4096 ceiling let one 92.9 s call write a methodology essay instead of the answer (the data had been
    # retrieved two rounds earlier). Here the prompt's answer rules and the per-question time budget guard
    # against that instead; lower this if you see long, slow answers.
    llm_max_tokens: int = 8192
    # GEMINI_API_KEY (preferred) or GOOGLE_API_KEY; create one at https://aistudio.google.com/apikey
    gemini_api_key: str | None = Field(default=None, validation_alias=AliasChoices("gemini_api_key", "google_api_key"))
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # The user's statement that the Gemini key is on a paid plan. A free and a paid Gemini key cannot
    # be told apart reliably from the outside, so this is an explicit switch: false (the default)
    # keeps the chat agent in shallow mode while the provider is gemini (see `chat_limits`).
    gemini_paid: bool = False
    # Free-tier keys have low per-minute limits, so 429s are common. Without retries the whole question fails.
    rate_limit_retries: int = 4
    request_min_interval_sec: float = 0.0  # optional minimum spacing between calls
    llm_timeout_sec: float = 40.0
    anthropic_api_key: str | None = None
    anthropic_fallbacks: bool = True
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Vector index
    vector_backend: Literal["lsa", "openai", "off"] = "lsa"
    vector_dims: int = 384
    vector_max_df: float = 0.005
    vector_embed_model: str = "text-embedding-3-small"

    # Agent budget
    # Competition rule, kept as the default: an answer after 300 s counts as no answer; within that, late
    # answers are still scored but lose points.
    time_budget_sec: float = 120.0
    max_tool_rounds: int = 8
    sql_row_limit: int = 200
    sql_timeout_sec: float = 5.0

    # Paths
    data_dir: Path = DATA_DIR
    ontology_dir: Path = ONTOLOGY_DIR

    @property
    def store_path(self) -> Path:
        return self.data_dir / "store" / "products.db"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def enrichment_dir(self) -> Path:
        return self.data_dir / "enrichment"

    @property
    def constituents_dir(self) -> Path:
        return self.data_dir / "enrichment" / "constituents"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "external" / "docs"

    @property
    def vector_cache_path(self) -> Path:
        return self.data_dir / "store" / "vector_cache.npz"

    # Paper-trading assistant (research + trading layers)
    # Alpaca paper-trading credentials. The trading host itself is a hard-coded constant in
    # trading/broker.py (never derived from settings, per the paper-only constraint) - these keys
    # only authenticate against it.
    alpaca_api_key_id: str | None = None
    alpaca_api_secret_key: str | None = None
    # Tavily web search API key (research/websearch.py); Jina Reader (r.jina.ai) needs no key.
    tavily_api_key: str | None = None
    # SEC EDGAR requires a descriptive User-Agent with contact info on every request, or it 403s.
    sec_user_agent: str | None = None
    # Gemini's native google_search grounding model (research/grounding.py). Kept separate from
    # llm_model since grounding goes through a different SDK code path and its results are never
    # cached (the whole point is a fresh search).
    grounding_model: str = "gemini-3.8-flash"
    # Kill switch: when False, the chat agent can still research but ticket confirmation is refused.
    trading_enabled: bool = True

    # App DB: sessions, messages, mandates, tickets, orders, ledger, evidence, http_cache.
    app_db_path: Path = DATA_DIR / "store" / "app.db"

    # Chat turn budget - separate from the QA agent's time_budget_sec/max_tool_rounds above, since
    # the chat agent's tool loop (research + trading) has different latency characteristics.
    # Two modes, resolved in exactly one place, `chat_limits()` below; nothing else reads these four.
    # Deep (a paid key: OpenAI, Anthropic, or Gemini with GEMINI_PAID=true): room for real research.
    chat_time_budget_sec: float = 300.0
    chat_max_rounds: int = 12
    # Shallow (Gemini without GEMINI_PAID): a free-tier key allows about 5 model calls a minute and
    # 20 a day, so a turn gets few rounds. 150 s still leaves room for a 429 backoff or two: at
    # 90 s both live evaluation scenarios ran out of time (2026-10-02) before they could propose.
    shallow_time_budget_sec: float = 150.0
    shallow_max_rounds: int = 4
    # How long a proposed ticket stays confirmable before it expires and must be regenerated.
    ticket_ttl_sec: int = 300
    # /api/* (chat, tickets, settings, kill switch) answers loopback clients only. True lets any
    # client address through (the Host allowlist and the anti-CSRF header still apply). UNSAFE
    # unless you put your own authentication in front: /api/* can place paper orders and write keys.
    api_allow_remote: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --- research depth: shallow and deep ----------------------------------------------------------------

ChatMode = Literal["shallow", "deep"]
# Used only when a settings object lacks a field or holds something unusable (tests pass bare
# namespaces); `Settings` carries the same numbers as its defaults.
_DEEP_LIMITS = (12, 300.0)
_SHALLOW_LIMITS = (4, 150.0)


@dataclass(frozen=True)
class ChatLimits:
    """What one chat turn may use: the mode, the tool rounds and the seconds."""

    mode: ChatMode
    max_rounds: int
    time_budget_sec: float


def chat_mode(settings: object) -> ChatMode:
    """`shallow` when the provider is gemini and the key is not declared paid (`gemini_paid`);
    `deep` otherwise: OpenAI and Anthropic keys are always paid, and the `stub` provider (tests)
    counts as deep."""
    if getattr(settings, "llm_provider", None) == "gemini" and getattr(settings, "gemini_paid", False) is not True:
        return "shallow"
    return "deep"


def chat_limits(settings: object) -> ChatLimits:
    """The ONE place that resolves a chat turn's limits. `ChatAgent` (rounds, time budget), the
    chat system prompt (its Budget section) and `GET /api/settings` all read through it."""
    return _limits_for(settings, chat_mode(settings))


def longest_turn_sec(settings: object) -> float:
    """The longest time budget a chat turn can have under these settings, whatever the mode (a
    turn keeps the limits it started with, so a client it holds must outlive either budget)."""
    return max(_limits_for(settings, "shallow").time_budget_sec, _limits_for(settings, "deep").time_budget_sec)


def _limits_for(settings: object, mode: ChatMode) -> ChatLimits:
    prefix, (rounds, seconds) = ("shallow", _SHALLOW_LIMITS) if mode == "shallow" else ("chat", _DEEP_LIMITS)
    try:
        rounds = max(1, int(getattr(settings, f"{prefix}_max_rounds", rounds)))
    except (TypeError, ValueError):
        pass
    try:
        seconds = float(getattr(settings, f"{prefix}_time_budget_sec", seconds))
    except (TypeError, ValueError):
        pass
    return ChatLimits(mode, rounds, seconds)


# Settings field -> .env name, for the fields the app itself writes (PUT /api/settings and the
# kill switch).
ENV_NAMES = {
    "alpaca_api_key_id": "ALPACA_API_KEY_ID",
    "alpaca_api_secret_key": "ALPACA_API_SECRET_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
    "sec_user_agent": "SEC_USER_AGENT",
    "trading_enabled": "TRADING_ENABLED",
    "llm_provider": "LLM_PROVIDER",
    "llm_model": "LLM_MODEL",
    "llm_effort": "LLM_EFFORT",
    "gemini_api_key": "GEMINI_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "gemini_paid": "GEMINI_PAID",
}


def env_fallbacks(env_path: Path | str | None = None) -> list[str]:
    """Names (never values) of the app-writable settings that only the process environment
    provides: set there, and absent or empty in the `.env` file (which wins when it has a value).
    Saving such a setting from the app works, since that writes it to `.env`; clearing it there
    does not survive a restart, because the environment value comes back."""
    path = Path(env_path) if env_path is not None else ROOT / ".env"
    in_file = {name.upper() for name, value in dotenv_values(path).items() if value} if path.is_file() else set()
    in_env = {name.upper() for name, value in os.environ.items() if value}
    return sorted(name for name in ENV_NAMES.values() if name in in_env and name not in in_file)
