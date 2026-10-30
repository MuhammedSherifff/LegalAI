"""Answer-level eval: run the real agent over an eval file and log one JSON row per question.

Deterministic metrics only (no judge yet): citation hit, tool usage, web share,
rounds, latency, errors. Writes JSONL and can resume an interrupted run.

Usage:
    python scripts/eval_answers.py --eval-file eval/external_egyptian_legal_v2.json --out eval/runs/ext516.jsonl
    python scripts/eval_answers.py --limit 20
    python scripts/eval_answers.py --bucket Scenario --scope civil
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legal_agent import citations, config, agent
from legal_agent.behavior import behavior_check, corpus_numbers


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["items"]


def tool_names(state: dict) -> list[str]:
    names: list[str] = []
    for m in state.get("messages", []):
        if isinstance(m, AIMessage) and m.tool_calls:
            names.extend(c["name"] for c in m.tool_calls)
    return names


def run_one(idx: int, item: dict) -> dict:
    t0 = time.perf_counter()
    state: dict = {}
    error = None
    for attempt in range(3):
        try:
            state = agent.ask(item["question"])
            error = None
            break
        except Exception as e:  # noqa: BLE001 - one bad item must not stop the run
            error = f"{type(e).__name__}: {e}"
            if attempt < 2:
                time.sleep(10 * (attempt + 1))  # transient 429/503 on free tiers
    latency = time.perf_counter() - t0

    answer = state.get("answer") or ""
    cited = sorted(citations.extract_citations(answer))
    expected = item.get("expected") or []
    tools = tool_names(state)
    return {
        "id": idx,
        "question": item["question"],
        "bucket": item.get("bucket"),
        "scope": item.get("scope", "civil" if expected else "other"),
        "expected": expected,
        "article": item.get("article"),
        "answer": answer,
        "cited": cited,
        "citation_hit": bool(set(expected) & set(cited)) if expected else None,
        "bad_citations": sorted(state.get("bad_citations") or []),
        "tools": tools,
        "web_results": state.get("web_results", []),
        "web_used": "web_search" in tools,
        "tool_rounds": state.get("tool_rounds"),
        "attempts": state.get("attempts"),
        "latency_s": round(latency, 2),
        "llm": config.LLM,
        "error": error,
    }


def summarize(rows: list[dict]) -> None:
    corpus = corpus_numbers()
    ok = [r for r in rows if not r["error"]]
    failed = [r for r in rows if r["error"]]
    print(f"\n=== {len(rows)} questions, {len(ok)} ok, {len(failed)} failed ===")
    print(f"llm: {config.LLM} | embedder: {config.EMBEDDER}")

    if ok:
        print(f"avg latency: {sum(r['latency_s'] for r in ok) / len(ok):.1f}s | "
              f"max: {max(r['latency_s'] for r in ok):.1f}s")
        rounds = [r["tool_rounds"] for r in ok if r["tool_rounds"] is not None]
        if rounds:
            print(f"tool rounds: avg {sum(rounds) / len(rounds):.1f} | max {max(rounds)}")
        print(f"web used: {sum(r['web_used'] for r in ok)}/{len(ok)}")
        print(f"bad citations left: {sum(1 for r in ok if r['bad_citations'])}/{len(ok)}")

    by_scope = defaultdict(list)
    for r in ok:
        by_scope[r["scope"]].append(r)
    for scope, rs in sorted(by_scope.items()):
        hits = [r for r in rs if r["citation_hit"] is not None]
        line = f"scope={scope:10s} n={len(rs):3d} web={sum(r['web_used'] for r in rs)/len(rs):.0%}"
        if hits:
            line += f" citation_hit={sum(r['citation_hit'] for r in hits)/len(hits):.0%} (n={len(hits)})"
        print(line)

    behavior = [behavior_check(r, corpus) for r in ok]
    passed = [v for v in behavior if v == "pass"]
    print(f"\nhonest behavior (rows without expected): {len(passed)}/{len(behavior)}")
    for v, n in Counter(behavior).most_common():
        if v != "pass":
            print(f"  {v} x{n}")
    print("buckets:", dict(by_bucket))
    if failed:
        print("\nfailed items:")
        for r in failed[:10]:
            print(f"  #{r['id']} {r['error'][:100]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-file", default="eval/golden_set.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bucket", action="append", help="only these buckets (repeatable)")
    ap.add_argument("--scope", choices=["civil", "other_law", "other"])
    args = ap.parse_args()

    items = load_items(args.eval_file)
    if args.bucket:
        items = [(i, it) for i, it in enumerate(items) if it.get("bucket") in args.bucket]
    else:
        items = list(enumerate(items))
    if args.scope:
        items = [(i, it) for i, it in items
                 if it.get("scope", "civil" if it.get("expected") else "other") == args.scope]
    if args.limit:
        items = items[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    rows: list[dict] = []
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        done = {r["id"] for r in rows}
        print(f"resuming: {len(done)} rows already in {out}")

    t_start = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for n, (idx, item) in enumerate(items, 1):
            if idx in done:
                continue
            row = run_one(idx, item)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            status = "ERR" if row["error"] else ("hit" if row["citation_hit"] else
                                                 ("web" if row["web_used"] else "corpus"))
            print(f"[{n}/{len(items)}] #{idx} {row['latency_s']:5.1f}s {status:5s} "
                  f"{row['question'][:70]}", flush=True)

    summarize(rows)
    print(f"\ntotal wall: {(time.perf_counter() - t_start) / 60:.1f} min | log: {out}")


if __name__ == "__main__":
    main()
