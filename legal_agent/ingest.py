"""Build the indices from data/civil_law_records.json.

Chroma (vector) + BM25 (lexical) over the same loaded records, so the two
indices always agree. Usage:
    python -m legal_agent.ingest [--embedder NAME] [--check QUERY]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .embeddings import get_embedder

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Arabic word + digit tokens. Arabic-Indic digits become ASCII so
    '١٦٣' and '163' match either way. Shared by ingest and query side."""
    return [
        t if t.isdigit() else t
        for t in _TOKEN_RE.findall(text.translate(AR_DIGITS))
    ]


def load_records() -> list[dict]:
    with open(config.DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    records = [r for r in data["records"] if r.get("text")]
    dropped = len(data["records"]) - len(records)
    print(f"loaded {len(records)} records, dropped {dropped} empty-text records")
    return records


def enrich(record: dict) -> str:
    """The string that gets embedded: number + hierarchy + text."""
    number = record.get("article_number") or ""
    parents = " / ".join(record.get("parent_titles") or [])
    return f"{number}\n{parents}\n{record['text']}"


def _clean_meta(record: dict) -> dict:
    """Chroma metadata: str/int/float/bool only — None becomes a query-time flag."""
    meta = {
        "law_id": record["law_id"],
        "article_number": record.get("article_number"),
        "legal_status": record.get("legal_status"),
        "parent_titles": " / ".join(record.get("parent_titles") or []),
        "page_start": record.get("source_page_start"),
        "page_end": record.get("source_page_end"),
        "title": record.get("title"),
    }
    return {k: (v if v is not None else "") for k, v in meta.items()}


def build_chroma(records: list[dict], embedder) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedder": embedder.name},
    )
    embeddings = embedder.embed_passages([enrich(r) for r in records])
    metas = [_clean_meta(r) for r in records]
    batch = 256
    for i in range(0, len(records), batch):
        collection.add(
            ids=[r["id"] for r in records[i : i + batch]],
            documents=[r["text"] for r in records[i : i + batch]],
            embeddings=np.asarray(embeddings[i : i + batch], dtype=np.float32).tolist(),
            metadatas=metas[i : i + batch],
        )
    print(f"chroma: indexed {len(records)} into {config.COLLECTION_NAME} ({embedder.name})")
    return collection


def build_bm25(records: list[dict]) -> None:
    corpus = [tokenize(enrich(r)) for r in records]
    bm25 = BM25Okapi(corpus)
    config.BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"embedder": None, "records": records, "bm25": bm25}, f)
    print(f"bm25: indexed {len(records)} -> {config.BM25_PATH.name}")


def check(query: str) -> None:
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    collection = client.get_collection(config.COLLECTION_NAME)
    embedder = get_embedder()
    res = collection.query(
        query_embeddings=[embedder.embed_query(query).tolist()],
        n_results=5,
    )
    print(f"\nquery: {query}")
    for i, (doc_id, meta, doc) in enumerate(
        zip(res["ids"][0], res["metadatas"][0], res["documents"][0]), 1
    ):
        status = meta["legal_status"]
        print(f"  {i}. {meta['article_number']} [{status}] p.{meta['page_start']}")
        print(f"     {doc[:110]}...")
    if config.BM25_PATH.exists():
        with open(config.BM25_PATH, "rb") as f:
            payload = pickle.load(f)
        scores = payload["bm25"].get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:5]
        print("  bm25 top-5:")
        for rank, idx in enumerate(top, 1):
            if scores[idx] <= 0:
                break
            r = payload["records"][idx]
            print(
                f"  {rank}. {r['article_number']} [{r['legal_status']}] "
                f"score={scores[idx]:.1f} :: {r['text'][:90]}..."
            )
    else:
        print("  (bm25 index not built)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Chroma + BM25 indices")
    ap.add_argument("--embedder", default=None, help="e5-small | bgem3")
    ap.add_argument("--check", default=None, metavar="QUERY", help="run sample query against built indices")
    args = ap.parse_args()

    if args.check:
        check(args.check)
        return

    if args.embedder:
        config.EMBEDDER = args.embedder

    from collections import Counter

    records = load_records()
    print(Counter(r["legal_status"] for r in records))
    build_chroma(records, get_embedder())
    build_bm25(records)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        sys.exit(f"data file not found: {config.DATA_PATH}")
