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
    # 주최측 공지: 답변 생성 파이프라인은 NCP HyperCLOVA X 필수.
    llm_provider: Literal["clova", "anthropic", "openai", "stub"] = "clova"
    llm_model: str = "HCX-005"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 4096  # CLOVA function calling 은 1024 이상이어야 동작
    clova_api_key: str | None = None
    clova_base_url: str = "https://clovastudio.stream.ntruss.com/v1/openai"
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
    # 주최측 공지: 300초는 물리적 미응답 처리. 그 안이면 늦어도 채점되나 점수 차등.
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
