"""Retrieval bake-off: recall@k, MRR and speed per embedder.

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


def evaluate(name: str, records: list[dict], golden: list[dict], k: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        with config_override(
            CHROMA_PATH=Path(tmp), COLLECTION_NAME=f"eval_{name}", EMBEDDER=name
        ):
            embedder = get_embedder(name)
            t0 = time.perf_counter()
            build_chroma(records, embedder)
            index_secs = time.perf_counter() - t0

            latencies, hits_at_k, rr = [], 0, 0.0
            for item in golden:
                t0 = time.perf_counter()
                nums = [h.article_number for h in retrieval.hybrid_search(item["question"], k=k)]
                latencies.append(time.perf_counter() - t0)
                rank = next((i for i, n in enumerate(nums, 1) if n in item["expected"]), None)
                if rank:
                    hits_at_k += 1
                    rr += 1.0 / rank

    get_embedder.cache_clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    n = len(golden)
    return {
        "embedder": name,
        "dim": embedder.dim,
        "index_secs": index_secs,
        f"recall@{k}": hits_at_k / n,
        "mrr": rr / n,
        "query_ms": 1000 * sum(latencies) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedders", default="qwen3-0.6b,e5-small,bgem3")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    records = load_records()
    golden = load_golden()
    names = [n.strip() for n in args.embedders.split(",") if n.strip()]
    unknown = set(names) - set(available_embedders())
    if unknown:
        sys.exit(f"unknown embedders {sorted(unknown)}, available: {available_embedders()}")

    results = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        try:
            results.append(evaluate(name, records, golden, args.k))
        except Exception as e:  # noqa: BLE001 — report and continue the bake-off
            print(f"FAILED: {type(e).__name__}: {str(e)[:200]}")
    if not results:
        sys.exit("no embedder completed")

    header = f"{'embedder':<14} {'dim':>5} {'index_s':>8} {'query_ms':>9} {'recall@k':>9} {'mrr':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in sorted(results, key=lambda r: (-r[f"recall@{args.k}"], -r["mrr"])):
        print(
            f"{r['embedder']:<14} {r['dim']:>5} {r['index_secs']:>8.1f} "
            f"{r['query_ms']:>9.1f} {r[f'recall@{args.k}']:>9.2f} {r['mrr']:>6.2f}"
        )


if __name__ == "__main__":
    main()
