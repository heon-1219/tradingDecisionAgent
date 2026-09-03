"""Dense vector (semantic) index — the "Vector" leg of the RDB + Vector + Graph KnowledgeBase.

Default backend is deterministic and offline: TF-IDF over the shared tokenizer (words + Korean
character bigrams) reduced with truncated SVD (LSA) to dense, L2-normalised vectors, searched by
cosine similarity.  This needs no embedding API key, so evaluation never depends on a second
provider, and it is reproducible run to run.

An embedding-API backend is available by injecting `embed_fn` (see `make_openai_embed_fn`); set
`VECTOR_BACKEND=openai` to use it.

`rrf_fuse` combines this index with the BM25 index (reciprocal rank fusion) for hybrid retrieval.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .search import Doc, Hit, tokenize

log = logging.getLogger("finagent.vector")

DEFAULT_DIMS = 384
# Prune tokens appearing in more than this fraction of documents.  Korean product names are full of
# boilerplate (증권투자신탁, 사모, 파생형, 상장지수...) which otherwise dominates the LSA space and makes
# every short query collapse onto the same cluster; 0.005 was chosen by measuring retrieval probes.
DEFAULT_MAX_DF = 0.005


@dataclass
class VectorStats:
    backend: str
    docs: int
    dims: int
    build_seconds: float
    cached: bool


def _identity(tokens):  # TfidfVectorizer wants a callable analyzer
    return tokens


class VectorIndex:
    """Cosine-similarity search over dense document vectors."""

    def __init__(self, ids, tables, titles, payloads, matrix: np.ndarray, encode_query, stats: VectorStats):
        self.ids, self.tables, self.titles, self.payloads = ids, tables, titles, payloads
        self.matrix = matrix
        self._encode_query = encode_query
        self.stats = stats
        self._tables_arr = np.array(tables) if tables else np.array([])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        docs: list[Doc],
        dims: int = DEFAULT_DIMS,
        cache_path: Path | None = None,
        embed_fn=None,
        max_df: float = DEFAULT_MAX_DF,
        log_fn=log.info,
    ) -> "VectorIndex":
        t0 = time.time()
        if not docs:
            return cls([], [], [], [], np.zeros((0, 1), dtype=np.float32), lambda q: np.zeros(1, dtype=np.float32),
                       VectorStats("empty", 0, 0, 0.0, False))
        ids = [d.id for d in docs]
        tables = [d.table for d in docs]
        titles = [d.title for d in docs]
        payloads = [d.payload for d in docs]
        texts = [f"{d.title} {d.text}" for d in docs]

        if embed_fn is not None:
            matrix = np.asarray(embed_fn(texts), dtype=np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-9)

            def encode_query(q: str) -> np.ndarray:
                v = np.asarray(embed_fn([q])[0], dtype=np.float32)
                return v / max(float(np.linalg.norm(v)), 1e-9)

            stats = VectorStats("embedding-api", len(docs), int(matrix.shape[1]), time.time() - t0, False)
            log_fn("vector index: %d docs, %d dims (embedding api) in %.1fs", len(docs), matrix.shape[1], time.time() - t0)
            return cls(ids, tables, titles, payloads, matrix, encode_query, stats)

        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        signature = hashlib.sha1(
            (f"{len(docs)}|{dims}|{max_df}|" + "|".join(ids[:: max(1, len(ids) // 500)])).encode("utf-8")
        ).hexdigest()[:16]

        # min_df=2 drops hapax noise on a real corpus, but would empty a small one
        min_df = 2 if len(docs) >= 100 else 1
        eff_max_df = max_df if len(docs) >= 1000 else 1.0
        tfidf = TfidfVectorizer(
            analyzer=_identity, min_df=min_df, max_df=eff_max_df, sublinear_tf=True, dtype=np.float32
        )
        sparse = tfidf.fit_transform(tokenize(t) for t in texts)
        dims = int(min(dims, max(2, min(sparse.shape) - 1)))
        cached = False
        matrix = None
        components = None
        if cache_path is not None:
            cp = Path(cache_path)
            if cp.exists():
                try:
                    z = np.load(cp, allow_pickle=False)
                    if str(z["signature"]) == signature and int(z["dims"]) == dims:
                        matrix, components = z["matrix"], z["components"]
                        cached = True
                except Exception as e:  # noqa: BLE001 - a stale/corrupt cache must never break startup
                    log.warning("vector cache unusable (%s); rebuilding", e)
        if matrix is None:
            svd = TruncatedSVD(n_components=dims, random_state=0)
            matrix = svd.fit_transform(sparse).astype(np.float32)
            components = svd.components_.astype(np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-9)
            if cache_path is not None:
                try:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_path, matrix=matrix, components=components, dims=dims, signature=signature)
                except Exception as e:  # noqa: BLE001
                    log.warning("could not write vector cache: %s", e)

        comp = components

        def encode_query(q: str) -> np.ndarray:
            sp = tfidf.transform([tokenize(q)])
            v = (sp @ comp.T).astype(np.float32).ravel()
            return v / max(float(np.linalg.norm(v)), 1e-9)

        stats = VectorStats("tfidf-lsa", len(docs), dims, time.time() - t0, cached)
        log_fn("vector index: %d docs, %d dims (tfidf-lsa%s) in %.1fs", len(docs), dims, ", cached" if cached else "", time.time() - t0)
        return cls(ids, tables, titles, payloads, matrix, encode_query, stats)

    # ----------------------------------------------------------------- search
    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int = 10, table: str | None = None) -> list[Hit]:
        if not self.ids or not query or not query.strip():
            return []
        q = self._encode_query(query)
        if not np.any(q):
            return []
        scores = self.matrix @ q
        if table is not None:
            scores = np.where(self._tables_arr == table, scores, -1.0)
        k = max(1, min(k, len(self.ids)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(self.ids[i], self.tables[i], self.titles[i], float(scores[i]), self.payloads[i])
            for i in top
            if scores[i] > 0.01
        ]


def rrf_fuse(
    ranked_lists: list[list[Hit]], k: int = 10, rrf_k: int = 60, weights: list[float] | None = None
) -> list[Hit]:
    """Reciprocal rank fusion: robust to the different score scales of BM25 and cosine."""
    scores: dict[tuple[str, str], float] = {}
    best: dict[tuple[str, str], Hit] = {}
    for i, hits in enumerate(ranked_lists):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, h in enumerate(hits, start=1):
            key = (h.table, h.id)
            scores[key] = scores.get(key, 0.0) + w / (rrf_k + rank)
            if key not in best or h.score > best[key].score:
                best[key] = h
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [Hit(best[key].id, best[key].table, best[key].title, round(score, 5), best[key].payload) for key, score in ordered]


EXACT_SCORE = 100.0  # SearchIndex adds +100 for an exact/near-exact title match


def hybrid_search(lexical: list[Hit], vector: list[Hit], k: int = 10, vec_ratio: float = 0.3) -> list[Hit]:
    """Lexical head + reserved vector-recall tail.

    Shaped by measurement rather than taste.  Plain RRF was tried first and rejected: because RRF
    rewards agreement between legs, a document ranked mid-list by *both* legs can outrank the best
    lexical hit, and on this corpus that put an unrelated fund at rank 1 for "배터리 양극재 관련 상품".
    Since a wrong lead candidate can mislead the answer generator, the head of the list is now
    lexical order (so hybrid never ranks worse than lexical where lexical is strong), while the last
    `vec_ratio` of the slots is reserved for documents only the semantic leg found — which is what
    recovers cases like "space exploration satellite" -> Procure Space ETF or a paraphrased query.
    When lexical returns fewer than `k` hits, the vector leg fills the remainder.
    """
    if k <= 0:
        return []
    lex_keys = {(h.table, h.id) for h in lexical}
    vec_only = [h for h in vector if (h.table, h.id) not in lex_keys]
    n_vec = min(len(vec_only), max(1, int(round(k * vec_ratio))) if vec_only else 0)
    head = lexical[: max(0, k - n_vec)]
    out = list(head)
    for h in vec_only:
        if len(out) >= k:
            break
        out.append(h)
    for h in lexical[len(head):]:  # lexical fills anything the vector leg could not
        if len(out) >= k:
            break
        if (h.table, h.id) not in {(x.table, x.id) for x in out}:
            out.append(h)
    return out[:k]


def make_openai_embed_fn(settings, model: str = "text-embedding-3-small", batch: int = 256):
    """Embedding backend for `VECTOR_BACKEND=openai` (any OpenAI-compatible embeddings endpoint)."""
    import openai

    client = openai.OpenAI(api_key=settings.openai_api_key or None, base_url=settings.openai_base_url or None)

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            chunk = [t[:8000] for t in texts[i : i + batch]]
            resp = client.embeddings.create(model=model, input=chunk)
            out.extend(d.embedding for d in resp.data)
        return out

    return embed
