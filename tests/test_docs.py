from pathlib import Path

from finagent.kb.docs import DocStore, chunk_markdown, parse_front_matter

ROOT = Path(__file__).resolve().parents[1]


def test_docstore_search(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntitle: 국민성장펀드 개요\nsource: https://example.org\nas_of: 2026-08-20\nproducts: [\"미래에셋국민참여형\"]\n---\n"
        "# 구조\n국민성장펀드는 모펀드-자펀드 구조이다.\n\n# 전략\nAI·반도체 등 첨단산업에 투자한다.\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("ignored", encoding="utf-8")
    ds = DocStore.load(tmp_path)
    assert len(ds) == 2
    hits = ds.search("국민성장펀드 구조")
    assert hits and hits[0].title == "국민성장펀드 개요" and "모펀드" in hits[0].text and hits[0].as_of == "2026-08-20"
    assert hits[0].products == ["미래에셋국민참여형"]


def test_front_matter_and_chunking():
    meta, body = parse_front_matter("---\ntitle: t\nproducts: []\n---\n# A\nx\n## B\n" + "y" * 3000 + "\n")
    assert meta["title"] == "t" and meta["products"] == []
    chunks = chunk_markdown(body)
    assert chunks[0] == ("A", "x") and all(len(t) <= 1200 for _, t in chunks)


def test_real_docs_load():
    ds = DocStore.load(ROOT / "data" / "external" / "docs")
    assert len(ds) > 5
    assert ds.search("국민성장펀드 투자전략")[0].title.startswith("국민성장펀드")
