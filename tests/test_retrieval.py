import pytest

from legal_agent import config, retrieval

INDEX_READY = config.CHROMA_PATH.exists() and config.BM25_PATH.exists()


def test_placeholders_cover_repealed():
    ph = retrieval._placeholders()
    assert ph["54"]["legal_status"] == "repealed"
    assert ph["1022"]["legal_status"] == "unknown"


@pytest.mark.skipif(not INDEX_READY, reason="run ingest first")
def test_exact_lookup_with_arabic_digits():
    hit = retrieval.get_article_by_number("١٦٣")
    assert hit is not None and hit.article_number == "163"


@pytest.mark.skipif(not INDEX_READY, reason="run ingest first")
def test_repealed_lookup_falls_back_to_json():
    hit = retrieval.get_article_by_number("54")
    assert hit is not None and hit.legal_status == "repealed"


@pytest.mark.skipif(not INDEX_READY, reason="run ingest first")
def test_hybrid_finds_known_articles():
    for query, expected in [
        ("ما حكم الخطأ الذي يسبب ضررا للغير", "163"),
        ("تنفيذ العقد وفقا لما يوجبه حسن النية", "148"),
    ]:
        hits = retrieval.hybrid_search(query, k=5)
        assert expected in [h.article_number for h in hits], query
