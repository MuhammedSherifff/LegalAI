"""Terminal runner: ask a legal question, see an agentic, cited answer.

Interactive REPL:    python -m legal_agent
One-shot question:   python -m legal_agent "ما نص المادة 163؟"

The agent (graph.py) decides which tools to call, answers from their results,
and a code-side gate verifies every cited article number.
"""

from __future__ import annotations

import sys

from . import citations, graph

DISCLAIMER = "\n⚠ هذه معلومات عامة وليست استشارة قانونية."

_B = "\033[1m"; _UNVERIFIED = "\033[93m"; _R = "\033[0m"

_STATUS_AR = {"in_force": "ساري", "repealed": "ملغى", "unknown": "غير مؤكدة"}


def run(question: str) -> None:
    try:
        state = graph.ask(question)
    except Exception as e:  # noqa: BLE001 — provider errors must not kill the REPL
        print(f"{_UNVERIFIED}\n⚠ تعذر تنفيذ الطلب: {type(e).__name__}: {str(e)[:200]}{_R}")
        return
    evidence = state.get("evidence", [])
    seen: list[str] = []
    for e in evidence:
        if e["article"] not in seen:
            seen.append(e["article"])
    print(f"[agent] {len(seen)} article(s) retrieved: {', '.join('م' + n for n in seen)}")

    answer_text = state.get("answer", "")
    print(f"\n{_B}{answer_text}{_R}")

    bad = state.get("bad_citations") or set()
    if bad:
        print(f"{_UNVERIFIED}\n⚠ استشهد بنصوص خارج الأدلة المعروضة: {', '.join(sorted(bad))}{_R}")
        print(DISCLAIMER)
        return

    cited = citations.extract_citations(answer_text)
    sources = {e["article"]: e for e in evidence if e["article"] in cited}
    if sources:
        print(f"\n{_B}المصادر:{_R}")
        for article, e in sources.items():
            status = _STATUS_AR.get(e["status"], e["status"])
            print(f"  المادة {article} [{status}] — صفحة {e['page']} — {e['section']}")
    elif evidence:
        print("\n(لم يستشهد النموذج بأي مادة من الأدلة المعروضة)")
    else:
        print("\n(لا نصوص من الفهرس المحلي — الإجابة مستندة إلى بحث الويب)")
    print(DISCLAIMER)


def main() -> None:
    args = sys.argv[1:]
    if args:
        run(" ".join(args))
        return
    print("LegalAI — اسأل عن القانون المصري. اكتب 'exit' للخروج.")
    while True:
        try:
            question = input("\nس > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit", "خروج"}:
            break
        run(question)
    print("وداعًا")


if __name__ == "__main__":
    main()
