"""Deterministic behavior scoring for eval rows with no expected article.

These rows are traps and routing tests (nonexistent articles, other-law
questions, current events). The correct response is known in advance and is
mechanically checkable — no LLM judge needed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def corpus_numbers() -> frozenset[str]:
    data = json.loads((ROOT / "data" / "civil_law_records.json").read_text(encoding="utf-8"))
    return frozenset(r["article_number"] for r in data["records"] if r.get("article_number"))


def behavior_check(row: dict, corpus: frozenset[str]) -> str:
    """Return 'pass' or 'fail:<reason>' for a behavior-only row.

    Rows with expected articles return 'pass' (they are scored elsewhere).
    The checks, in order:
    - leaked fabrications: bad_citations survived the retry gate
    - nonexistent article cited: catches web-only answers, which the
      citation gate cannot verify (no evidence to compare against)
    - other-law question answered from the civil corpus instead of routing
      to the web or refusing
    """
    if row.get("expected"):
        return "pass"
    if row.get("bad_citations"):
        return "fail:bad_citations"
    if not row.get("web_used"):
        # same exemption as the citation gate: a web answer may legitimately
        # cite another law's article numbers, which we cannot verify code-side
        nonexistent = [n for n in row.get("cited", []) if n not in corpus]
        if nonexistent:
            return "fail:cited_nonexistent:" + ",".join(nonexistent)
    if row.get("bucket") == "E" and not row.get("web_used") and row.get("cited"):
        return "fail:other_law_answered_from_corpus"
    return "pass"
