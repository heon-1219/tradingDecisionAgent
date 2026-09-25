from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import DATA_DIR, ONTOLOGY_DIR, ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

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
    chat_time_budget_sec: float = 90.0
    chat_max_rounds: int = 8
    # How long a proposed ticket stays confirmable before it expires and must be regenerated.
    ticket_ttl_sec: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
