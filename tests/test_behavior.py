from legal_agent.behavior import behavior_check

CORPUS = frozenset({"1", "163", "1149"})


def row(**kw):
    base = {"question": "س", "cited": [], "bad_citations": [], "web_used": False}
    base.update(kw)
    return base


def test_scored_rows_pass_by_definition():
    assert behavior_check(row(expected=["163"]), CORPUS) == "pass"


def test_leaked_fabrication_fails():
    assert behavior_check(row(bad_citations=["8000"]), CORPUS).startswith("fail:")


def test_citing_nonexistent_article_fails():
    # the trap: المادة 8000 — also the web-only hole in the citation gate
    assert behavior_check(row(cited=["8000"]), CORPUS) == "fail:cited_nonexistent:8000"


def test_web_routing_on_other_law_passes():
    r = row(bucket="E", web_used=True, cited=["800"])
    assert behavior_check(r, CORPUS) == "pass"


def test_other_law_answered_from_corpus_fails():
    r = row(bucket="E", cited=["163"])
    assert behavior_check(r, CORPUS) == "fail:other_law_answered_from_corpus"


def test_honest_refusal_passes():
    assert behavior_check(row(bucket="E"), CORPUS) == "pass"
