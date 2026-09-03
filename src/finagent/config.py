from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import DATA_DIR, ONTOLOGY_DIR, ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: Literal["anthropic", "openai", "stub"] = "anthropic"
    llm_model: str = "claude-opus-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 4096
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
    time_budget_sec: float = 50.0
    max_tool_rounds: int = 6
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
