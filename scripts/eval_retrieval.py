"""Retrieval bake-off: recall@k, MRR and speed per embedder and search mode.

Builds a throwaway Chroma index per embedder (same records, same BM25) and
scores eval/golden_set.json. Never touches the canonical index.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --embedders qwen3-0.6b,e5-small
    python scripts/eval_retrieval.py --embedders gemini-2 --k 5
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legal_agent import config, retrieval  # noqa: E402
from legal_agent.embeddings import available_embedders, get_embedder  # noqa: E402
from legal_agent.ingest import build_chroma, load_records  # noqa: E402


@contextlib.contextmanager
def config_override(**values):
    old = {k: getattr(config, k) for k in values}
    for k, v in values.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def load_golden() -> list[dict]:
    with open(ROOT / "eval" / "golden_set.json", encoding="utf-8") as f:
        return json.load(f)["items"]


def evaluate(name: str, records: list[dict], golden: list[dict], k: int, modes: list[str]) -> list[dict]:
    """One throwaway index per embedder, scored against every search mode."""
    with tempfile.TemporaryDirectory() as tmp:
        with config_override(
            CHROMA_PATH=Path(tmp), COLLECTION_NAME=f"eval_{name}", EMBEDDER=name
        ):
            embedder = get_embedder(name)
            t0 = time.perf_counter()
            build_chroma(records, embedder)
            index_secs = time.perf_counter() - t0

            scores = {m: [0, 0.0, []] for m in modes}  # hits, rr, latencies
            scored = 0
            for item in golden:
                if not item.get("expected"):  # behavior/web/trap items aren't retrieval-scored
                    continue
                scored += 1
                for mode in modes:
                    search = getattr(retrieval, mode + "_search")
                    t0 = time.perf_counter()
                    nums = [h.article_number for h in search(item["question"], k=k)]
                    scores[mode][2].append(time.perf_counter() - t0)
                    rank = next((i for i, n in enumerate(nums, 1) if n in item["expected"]), None)
                    if rank:
                        scores[mode][0] += 1
                        scores[mode][1] += 1.0 / rank

    get_embedder.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    n = scored
    return [
        {
            "embedder": name,
            "mode": mode,
            "dim": embedder.dim,
            "index_secs": index_secs,
            f"recall@{k}": hits / n,
            "mrr": rr / n,
            "query_ms": 1000 * sum(lats) / n,
        }
        for mode, (hits, rr, lats) in scores.items()
    ]



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedders", default="qwen3-0.6b,e5-small,bgem3")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", default="hybrid", help="comma list: vector,sparse,hybrid")
    args = ap.parse_args()

    records = load_records()
    golden = load_golden()
    names = [n.strip() for n in args.embedders.split(",") if n.strip()]
    unknown = set(names) - set(available_embedders())
    if unknown:
        sys.exit(f"unknown embedders {sorted(unknown)}, available: {available_embedders()}")

    results = []
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for name in names:
        print(f"\n=== {name} / {'+'.join(modes)} ===", flush=True)
        try:
            results.extend(evaluate(name, records, golden, args.k, modes))
        except Exception as e:  # noqa: BLE001 — report and continue the bake-off
            print(f"FAILED: {type(e).__name__}: {str(e)[:200]}")
    if not results:
        sys.exit("no embedder completed")

    header = f"{'embedder':<14} {'mode':<8} {'dim':>5} {'index_s':>8} {'query_ms':>9} {'recall@k':>9} {'mrr':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in sorted(results, key=lambda r: (-r[f"recall@{args.k}"], -r["mrr"])):
        print(
            f"{r['embedder']:<14} {r['mode']:<8} {r['dim']:>5} {r['index_secs']:>8.1f} "
            f"{r['query_ms']:>9.1f} {r[f'recall@{args.k}']:>9.2f} {r['mrr']:>6.2f}"
        )


if __name__ == "__main__":
    main()
