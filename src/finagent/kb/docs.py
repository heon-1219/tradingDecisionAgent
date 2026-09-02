"""External markdown document store: front-matter parsing, heading-based chunking, BM25 search."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .search import Doc, SearchIndex

_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_MAX_CHUNK = 1200


@dataclass
class Chunk:
    doc_id: str
    title: str
    source: str
    as_of: str
    heading: str
    text: str
    products: list[str] = field(default_factory=list)
    score: float = 0.0


def parse_front_matter(raw: str) -> tuple[dict, str]:
    m = _FM.match(raw)
    if not m:
        return {}, raw
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            meta[k.strip()] = [x.strip().strip("\"'") for x in v[1:-1].split(",") if x.strip()]
        else:
            meta[k.strip()] = v.strip("\"'")
    return meta, raw[m.end():]


def chunk_markdown(body: str) -> list[tuple[str, str]]:
    """Split on markdown headings; further split long sections by paragraph."""
    parts: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            while len(text) > _MAX_CHUNK:
                cut = text.rfind("\n\n", 0, _MAX_CHUNK)
                cut = cut if cut > 200 else _MAX_CHUNK
                parts.append((heading, text[:cut].strip()))
                text = text[cut:].strip()
            parts.append((heading, text))
        buf.clear()

    for line in body.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            flush()
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
        else:
            buf.append(line)
    flush()
    return parts


class DocStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._index = SearchIndex(
            [Doc(str(i), "docs", f"{c.title} / {c.heading}", f"{c.title} {c.heading} {c.text}", {}) for i, c in enumerate(chunks)]
        )

    @classmethod
    def load(cls, d: Path) -> "DocStore":
        chunks: list[Chunk] = []
        d = Path(d)
        if not d.exists():
            return cls([])
        for f in sorted(d.glob("*.md")):
            if f.name.lower() == "readme.md":
                continue
            meta, body = parse_front_matter(f.read_text(encoding="utf-8"))
            title = str(meta.get("title") or f.stem)
            prods = meta.get("products") or []
            for heading, text in chunk_markdown(body):
                chunks.append(
                    Chunk(f.stem, title, str(meta.get("source") or ""), str(meta.get("as_of") or ""), heading, text, list(prods))
                )
        return cls(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def titles(self) -> list[str]:
        return sorted({c.title for c in self.chunks})

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        out = []
        for h in self._index.search(query, k=k):
            c = self.chunks[int(h.id)]
            out.append(Chunk(c.doc_id, c.title, c.source, c.as_of, c.heading, c.text, c.products, h.score))
        return out
