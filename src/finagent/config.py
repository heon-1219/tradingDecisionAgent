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
    # 지연은 출력 생성에 지배된다. 느린 두 문항을 라운드별로 계측하니 도구 실행은 0.53초/0.11초였고,
    # 105초·119초의 대부분이 마지막 LLM 호출 한 번이었다(각각 53초·92.9초). 92.9초짜리 응답이 만든 것은
    # 답이 아니라 "…상위 10개를 찾기 위해서는 다음과 같은 접근 방법을 사용할 수 있습니다" 라는 방법론
    # 서술이었다. 데이터는 이미 두 라운드 전에 확보돼 있었다. 4096 이라는 천장이 그 서술을 가능하게 했다.
    #
    # 상한의 근거(측정):
    #   * 기록된 답변 54개의 길이: 중앙값 108자, 95백분위 497자, 최대 610자(G-026, 월배당 ETF 9개 나열).
    #   * CLOVA 토크나이저 실측 한국어 비율은 1.85~1.90자/토큰 → 최대 610자 = 321토큰.
    #   * 현실적 최악: 교차질의 10행 표 + 근거 + evidence_refs 를 final_answer 인자로 직렬화 = 1,508자 = 813토큰.
    #   * 1536 은 그 최악 대비 723토큰(약 1,340자) 여유이면서, 수천 토큰짜리 서술은 불가능하게 만든다.
    #   * 1024 로 더 내리면 최악 대비 여유가 211토큰뿐이라 표가 한 행만 길어져도 잘린다.
    # CLOVA function calling 은 1024 이상이어야 동작하므로 그 아래로는 내릴 수 없다(llm.MIN_TOOL_TOKENS).
    llm_max_tokens: int = 1536
    clova_api_key: str | None = None
    clova_base_url: str = "https://clovastudio.stream.ntruss.com/v1/openai"
    # CLOVA 테스트앱은 요청 빈도 제한이 낮아 429 가 자주 난다. 재시도하지 않으면 그 문항은 통째로 실패한다.
    rate_limit_retries: int = 4
    request_min_interval_sec: float = 0.0  # 필요 시 호출 간 최소 간격
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
