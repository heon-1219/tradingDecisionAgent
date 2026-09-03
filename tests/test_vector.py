from finagent.kb.search import Doc, SearchIndex
from finagent.kb.vector import VectorIndex, hybrid_search, rrf_fuse

DOCS = [
    Doc("1", "v_etf_kr", "KODEX 2차전지산업", "KODEX 2차전지산업 삼성자산운용 배터리 양극재 리튬 주식 국내", {}),
    Doc("2", "v_etf_kr", "TIGER 2차전지테마", "TIGER 2차전지테마 미래에셋자산운용 배터리 리튬 주식 국내", {}),
    Doc("3", "v_etf_gl", "VanEck China Semiconductor ETF", "VanEck China Semiconductor ETF SMHC China Equity semiconductor chips", {}),
    Doc("4", "v_etf_gl", "Global X Robotics & Artificial Intelligence ETF", "Global X Robotics Artificial Intelligence ETF BOTZ robotics automation Equity", {}),
    Doc("5", "v_fund_pub", "미래에셋국민참여형국민성장혼합자산투자신탁", "국민성장 혼합자산 KOSPI200 종합채권 미래에셋자산운용", {}),
    Doc("6", "v_bond_kr", "한국전력공사채권1012", "한국전력공사채권 특수채 AAA 일반특수법인채", {}),
]


def test_vector_index_builds_and_searches(tmp_path):
    vi = VectorIndex.build(DOCS, dims=8, cache_path=tmp_path / "v.npz", log_fn=lambda *a: None)
    assert len(vi) == 6 and vi.stats.backend == "tfidf-lsa" and 0 < vi.stats.dims <= 8  # clamped by vocabulary size
    hits = vi.search("배터리 양극재", k=3)
    assert hits and {h.id for h in hits[:2]} <= {"1", "2"}
    en = vi.search("semiconductor chips china", k=2)
    assert en and en[0].id == "3"
    assert vi.search("robotics automation", table="v_etf_gl")[0].id == "4"
    assert vi.search("", k=3) == []


def test_vector_cache_roundtrip(tmp_path):
    cache = tmp_path / "v.npz"
    VectorIndex.build(DOCS, dims=8, cache_path=cache, log_fn=lambda *a: None)
    assert cache.exists()
    again = VectorIndex.build(DOCS, dims=8, cache_path=cache, log_fn=lambda *a: None)
    assert again.stats.cached and again.search("배터리", k=1)


def test_vector_with_injected_embed_fn():
    def embed(texts):  # deterministic toy embedding: [len, count of 'ETF']
        return [[float(len(t)), float(t.count("ETF")) + 0.1] for t in texts]

    vi = VectorIndex.build(DOCS, embed_fn=embed, log_fn=lambda *a: None)
    assert vi.stats.backend == "embedding-api" and vi.stats.dims == 2 and vi.search("ETF", k=2)


def test_empty_index():
    vi = VectorIndex.build([], log_fn=lambda *a: None)
    assert len(vi) == 0 and vi.search("x") == []


def test_rrf_fuse_prefers_agreement():
    lex = SearchIndex(DOCS).search("2차전지", k=5)
    vec = VectorIndex.build(DOCS, dims=8, log_fn=lambda *a: None).search("배터리 리튬", k=5)
    fused = rrf_fuse([lex, vec], k=3)
    assert fused and {h.id for h in fused[:2]} <= {"1", "2"}
    # a doc ranked by both sources outranks one found by a single source
    only_lex = [h for h in lex if h.id not in {v.id for v in vec}]
    if only_lex:
        assert fused[0].id != only_lex[-1].id
    assert rrf_fuse([[], []], k=3) == []


def test_hybrid_pins_exact_lexical_match():
    """A named product must never be displaced by a semantically-similar one."""
    lex = SearchIndex(DOCS).search("KODEX 2차전지산업", k=5)
    assert lex[0].id == "1" and lex[0].score >= 100
    vec = VectorIndex.build(DOCS, dims=8, log_fn=lambda *a: None).search("리튬 배터리", k=5)
    fused = hybrid_search(lex, vec, k=3)
    assert fused[0].id == "1"


def test_hybrid_adds_vector_recall():
    lex = [h for h in SearchIndex(DOCS).search("국민성장", k=5)]
    vec = VectorIndex.build(DOCS, dims=8, log_fn=lambda *a: None).search("혼합자산 KOSPI200", k=5)
    fused = hybrid_search(lex, vec, k=4)
    ids = {h.id for h in fused}
    assert "5" in ids  # the fund is reachable from either leg
    assert len(fused) <= 4
    assert hybrid_search([], [], k=3) == []
