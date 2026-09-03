"""Precompute everything that can be cached on disk, so server startup is shorter.

Run after scripts/build_store.py (the Dockerfile does both at image build time):
  python scripts/build_store.py && python scripts/warmup.py

Currently warms the vector index cache (data/store/vector_cache.npz, ~80s otherwise paid on
every boot).  The knowledge graph is rebuilt at startup because re-parsing a million triples is
not faster than constructing them.
"""
from __future__ import annotations

import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from finagent.config import get_settings  # noqa: E402
from finagent.kb.rdb import RDB  # noqa: E402
from finagent.kb.search import SearchIndex  # noqa: E402
from finagent.kb.vector import VectorIndex  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = get_settings()
    if s.vector_backend != "lsa":
        print(f"vector_backend={s.vector_backend}: nothing to warm up (cache is for the offline lsa backend)")
        return 0
    t0 = time.time()
    rdb = RDB(s.store_path)
    index = SearchIndex.from_rdb(rdb)
    vi = VectorIndex.build(
        index.docs, dims=s.vector_dims, max_df=s.vector_max_df, cache_path=s.vector_cache_path
    )
    print(f"warmed vector cache: {len(vi)} docs, {vi.stats.dims} dims -> {s.vector_cache_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
