"""Terminal runner: ask a legal question, see a grounded answer.

Interactive REPL:    python -m legal_agent.cli
One-shot question:   python -m legal_agent.cli "ما نص المادة 163؟"

P1 flow (no agent yet): hybrid retrieval -> evidence-grounded Arabic answer
-> code-verified citations -> evidence table + disclaimer.
"""

from __future__ import annotations

import json
import sys

from . import citations, nodes, retrieval

DISCLAIMER = "\n⚠ هذه معلومات عامة وليست استشارة قانونية."

_B = "\033[1m"; _UNVERIFIED = "\033[93m"; _R = "\033[0m"


def run(question: str) -> None:
    hits = retrieval.hybrid_search(question, k=5)
    evidence = retrieval.get_article_by_number(question.split()[-1]) if _looks_exact(question) else None
    if evidence:
        hits = [evidence] + [h for h in hits if h.id != evidence.id][:4]
    print(f"[retrieval] {len(hits)} articles: {', '.join('م' + h.article_number for h in hits)}")

    answer_text = nodes.answer(question, hits)

    evidence_numbers = {h.article_number for h in hits}
    bad = citations.check(answer_text, evidence_numbers)
    print(f"\n{_B}{answer_text}{_R}")
    if bad:
        print(f"{_UNVERIFIED}\n⚠ استشهد بنصوص خارج الأدلة المعروضة: {', '.join(sorted(bad))}{_R}")
        print(DISCLAIMER)
        return
    print(f"\n{_B}المصادر:{_R}")
    for h in hits:
        if h.article_number in answer_text or True:
            status = {"in_force": "ساري", "repealed": "ملغى"}.get(h.legal_status, h.legal_status)
            print(f"  المادة {h.article_number} [{status}] — صفحة {h.page_start} — {h.parent_titles}")
    print(DISCLAIMER)


def _looks_exact(question: str) -> bool:
    flows = {"ما", "نص", "القول", "ما هي", "المادة"}
    return "المادة" in question and len(question.split()) <= 6


def main() -> None:
    args = sys.argv[1:]
    if args:
        run(" ".join(args))
        return
    print("LegalAI — اسأل عن القانون المدني المصري. اكتب 'exit' للخروج.")
    while True:
        try:
            question = input("\nس > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit", "خروج"}:
            break
        run(question)
    print("ودعً")


if __name__ == "__main__":
    main()
