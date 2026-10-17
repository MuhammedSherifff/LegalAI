"""Retrieval engine: two search methods + hybrid fusion. Not agent-facing.

Vector (Chroma + embedder query instruction) and sparse (BM25) fail in
opposite ways; hybrid fuses them by reciprocal rank so one query benefits
from both. Agents consume tools.py, which calls hybrid_search.
"""

from __future__ import annotations

import functools
import json
import pickle
from dataclasses import dataclass

import chromadb

from . import config
from .embeddings import get_embedder_for_collection
from .ingest import tokenize


@dataclass
class Hit:
    id: str
    article_number: str
    legal_status: str
    title: str
    parent_titles: str
    page_start: int
    page_end: int
    text: str
    score: float


def _collection() -> chromadb.Collection:
    return chromadb.PersistentClient(path=str(config.CHROMA_PATH)).get_collection(
        config.COLLECTION_NAME
    )


def vector_search(query: str, k: int = 5) -> list[Hit]:
    coll = _collection()
    emb = get_embedder_for_collection(coll).embed_query(query).tolist()
    res = coll.query(query_embeddings=[emb], n_results=k)
    return [
        Hit(
            id=rid,
            article_number=m["article_number"],
            legal_status=m["legal_status"],
            title=m.get("title") or "",
            parent_titles=m.get("parent_titles") or "",
            page_start=m["page_start"],
            page_end=m["page_end"],
            text=doc,
            score=float(d),
        )
        for rid, m, doc, d in zip(
            res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]
        )
    ]


@functools.lru_cache(maxsize=1)
def _bm25_payload() -> dict:
    # shortcut: cached for process lifetime — restart after re-running ingest
    with open(config.BM25_PATH, "rb") as f:
        return pickle.load(f)


def sparse_search(query: str, k: int = 5) -> list[Hit]:
    payload = _bm25_payload()
    scores = payload["bm25"].get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        Hit(
            id=payload["records"][i]["id"],
            article_number=payload["records"][i]["article_number"],
            legal_status=payload["records"][i]["legal_status"],
            title=payload["records"][i].get("title") or "",
            parent_titles=" / ".join(payload["records"][i].get("parent_titles") or []),
            page_start=payload["records"][i]["source_page_start"],
            page_end=payload["records"][i]["source_page_end"],
            text=payload["records"][i]["text"],
            score=float(scores[i]),
        )
        for i in ranked
        if scores[i] > 0
    ]


def hybrid_search(query: str, k: int = 5, k_rrf: int = 60) -> list[Hit]:
    """Reciprocal-rank fusion over the two result lists (deduplicated by id)."""
    vec = vector_search(query, k)
    lex = sparse_search(query, k)
    by_id: dict[str, tuple[Hit, float]] = {}
    for hits in (vec, lex):
        for rank, hit in enumerate(hits, 1):
            _, rrf = by_id.get(hit.id, (None, 0.0))
            by_id[hit.id] = (hit, rrf + 1.0 / (k_rrf + rank))
    fused = sorted(by_id.values(), key=lambda t: t[1], reverse=True)
    return [
        Hit(
            id=hit.id,
            article_number=hit.article_number,
            legal_status=hit.legal_status,
            title=hit.title,
            parent_titles=hit.parent_titles,
            page_start=hit.page_start,
            page_end=hit.page_end,
            text=hit.text,
            score=rrf,
        )
        for hit, rrf in fused[:k]
    ]


@functools.lru_cache(maxsize=1)
def _placeholders() -> dict[str, dict]:
    """Article number -> record for repealed/unknown articles. These carry no
    text and are absent from the index, but exact lookups should still say so."""
    with open(config.DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {
        r["article_number"]: r
        for r in data["records"]
        if not r["text"] and r["law_id"] == "civ_1948"
    }


def get_article_by_number(number: str) -> Hit | None:
    """Exact metadata lookup '163' or '٦٣٠'-style input, digits normalized."""
    num = str(number).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    res = _collection().get(
        where={"article_number": {"$eq": num}}, limit=1,
        include=["documents", "metadatas"],
    )
    if not res["ids"]:
        r = _placeholders().get(num)
        if r is None:
            return None
        return Hit(
            id=r["id"],
            article_number=num,
            legal_status=r["legal_status"],
            title="",
            parent_titles="",
            page_start=r.get("source_page_start") or 0,
            page_end=r.get("source_page_end") or 0,
            text=r.get("source_note") or "لا يوجد نص منشور لهذه المادة",
            score=1.0,
        )
    m = res["metadatas"][0]
    return Hit(
        id=res["ids"][0],
        article_number=m["article_number"],
        legal_status=m["legal_status"],
        title=m.get("title") or "",
        parent_titles=m.get("parent_titles") or "",
        page_start=m["page_start"],
        page_end=m["page_end"],
        text=res["documents"][0],
        score=1.0,
    )
