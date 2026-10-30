"""LLM-as-judge over a run JSONL from eval_answers.py.

Two modes, chosen per row:
- civil rows (have `expected`): the answer is judged against the verbatim law
  text of the expected article(s) from the eval file — real ground truth.
- web rows (`web_used`): there is no ground truth, so the answer is judged
  for internal consistency against the exact web snippets the agent used.
  Verdicts here mean "claims unsupported by its own sources", NOT truth.

The judge runs on a different provider than the agent by default (Groq vs
Gemini) — same-model judging is measurably more lenient.

Usage:
    python scripts/judge.py --run eval/runs/golden_prompt_rule3.jsonl --out eval/runs/judged.jsonl
    python scripts/judge.py --run eval/runs/ext516.jsonl --out eval/runs/judged_ext.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legal_agent import config, llm  # noqa: E402

# Same-model judging is lenient — set LEGALAI_JUDGE_LLM to another provider when its key is live
JUDGE_SPEC = __import__("os").getenv("LEGALAI_JUDGE_LLM", "gemini:gemini-3.5-flash-lite")

CIVIL_PROMPT = """أنت محكم دقيق متخصص في القانون المدني المصري. سيُعطى لك سؤال، ونص قانون صحيح (المرجع)، وإجابة سابقة من مساعد قانوني.
مهمتك: هل الإجابة متوافقة مع نص القانون المرجع؟ لا تحكم على الأسلوب ولا على الاكتمال، فقط على صحة المضمون القانوني.
- grounded: كل ما ذكرته الإجابة من أحكام متوافق مع النص.
- partial: جوهر الإجابة صحيح لكن فيه خطأ أو تبسيط مضلل في تفصيلة.
- unsupported: الإجابة تخالف النص أو تنسب إليه حكما غير وارد فيه.
أعد JSON فقط بهذا الشكل: {{"verdict": "...", "reason": "جملة واحدة بالعربية"}}
السؤال: {question}
نص القانون المرجع:
{ground_truth}
الإجابة المحكومة:
{answer}"""

WEB_PROMPT = """أنت محكم دقيق. سيُعطى لك سؤال، ومقتطفات من مصادر ويب استخدمها مساعد قانوني، وإجابته.
مهمتك: هل كل المعلومات القانونية في الإجابة مدعومة بهذه المقتطفات؟ (هذا فحص اتساق مع المصادر، وليس تحققًا من صحتها المطلقة)
- grounded: كل المعلومات مدعومة بالمقتطفات.
- partial: معظمها مدعوم مع تفصيلة غير مدعومة.
- unsupported: في الإجابة معلومات قانونية جوهرية لا سند لها في المقتطفات.
أعد JSON فقط بهذا الشكل: {{"verdict": "...", "reason": "جملة واحدة بالعربية"}}
السؤال: {question}
المقتطفات:
{sources}
الإجابة المحكومة:
{answer}"""


def load_ground_truth(path: str) -> dict[str, str]:
    """Question -> verbatim law text, from the eval file when it carries one."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    out = {}
    for it in items:
        if it.get("article_text") and it["question"] not in out:
            out[it["question"]] = it["article_text"]
    return out


def load_corpus_texts() -> dict[str, str]:
    """Article number -> verbatim text, for eval files without article_text."""
    data = json.loads((ROOT / "data" / "civil_law_records.json").read_text(encoding="utf-8"))
    out = {}
    for r in data["records"]:
        if r.get("text") and r.get("article_number"):
            out.setdefault(r["article_number"], r["text"])
    return out


def parse_verdict(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    return data if data.get("verdict") in {"grounded", "partial", "unsupported"} else None


def judge_row(row: dict, truth: dict[str, str], corpus: dict[str, str], judge_llm) -> dict | None:
    if row.get("error") or not row.get("answer"):
        return None
    if row.get("expected"):
        gt = truth.get(row["question"]) or "\n\n".join(
            f"[المادة {n}] {corpus.get(n, '')}" for n in row["expected"])
        if not gt.strip():
            return {"verdict": "no_truth", "reason": "لا يوجد نص مرجعي"}
        prompt = CIVIL_PROMPT.format(
            question=row["question"], ground_truth=gt, answer=row["answer"])
    elif row.get("web_used") and row.get("web_results"):
        sources = "\n\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:600]}" for r in row["web_results"])
        prompt = WEB_PROMPT.format(question=row["question"], sources=sources, answer=row["answer"])
    else:
        return None
    for attempt in range(3):
        try:
            out = parse_verdict(judge_llm.invoke(prompt).text or "")
            if out:
                return out
        except Exception as e:  # noqa: BLE001 — transient 429/503 on free tiers
            if attempt == 2:
                return {"verdict": "judge_error", "reason": f"{type(e).__name__}: {e}"}
            time.sleep(5 * (attempt + 1))
    return {"verdict": "judge_error", "reason": "unparseable output"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run JSONL from eval_answers.py")
    ap.add_argument("--eval-file", default="eval/golden_set.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    truth = load_ground_truth(ROOT / args.eval_file)
    corpus = load_corpus_texts()
    rows = [json.loads(l) for l in Path(args.run).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"resuming: {len(done)} already judged")

    judge_llm = llm.get_llm(JUDGE_SPEC)
    counts: dict[tuple, list] = {}
    with out.open("a", encoding="utf-8") as f:
        for n, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            verdict = judge_row(row, truth, corpus, judge_llm)
            if verdict is None:
                continue
            kind = "civil" if row.get("expected") else ("web" if row.get("web_used") else "other")
            f.write(json.dumps({"id": row["id"], "kind": kind, **verdict}, ensure_ascii=False) + "\n")
            f.flush()
            counts.setdefault((kind, verdict["verdict"]), []).append(row["id"])
            print(f"[{n}/{len(rows)}] #{row['id']} {kind}:{verdict['verdict']} {verdict['reason'][:60]}",
                  flush=True)

    print(f"\n=== judge summary ({JUDGE_SPEC}, agent: {config.LLM}) ===")
    for (kind, v), ids in sorted(counts.items()):
        total = sum(len(v_) for (k, _), v_ in counts.items() if k == kind)
        print(f"  {kind:6s} {v:12s} {len(ids):3d}/{total} ({len(ids)/total:.0%})")


if __name__ == "__main__":
    main()
