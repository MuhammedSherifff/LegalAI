"""Embedder abstraction: one interface, swappable implementations.

Ingest embeds passages; retrieval embeds the query. Each embedder hides its
own quirks (e.g. e5 task prefixes, Gemini taskType) so callers never see them.
"""

from __future__ import annotations

import functools
import json
import time
import urllib.request

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


class BaseEmbedder:
    """Interface: embed_passages for indexing, embed_query for search."""

    name: str = "base"
    dim: int = 0

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _st(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_id, device=config.DEVICE)
            except Exception:
                # already-downloaded model in the HF cache must never re-download
                self._model = SentenceTransformer(
                    self.model_id, device=config.DEVICE, local_files_only=True
                )
            if config.DEVICE.startswith("cuda"):
                self._model.half()
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._st().encode(
            texts,
            batch_size=config.BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class E5Embedder(BaseEmbedder):
    """multilingual-e5-small. Trained with task prefixes: omitting them
    silently degrades retrieval quality, so they live here, not in callers."""

    name = "e5-small"
    model_id = "intfloat/multilingual-e5-small"
    dim = 384

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.encode([f"query: {text}"])[0]


class Bgem3Embedder(BaseEmbedder):
    """BAAI/bge-m3. No prefixes needed; heavier (~2GB) and slower on CPU."""

    name = "bgem3"
    model_id = "BAAI/bge-m3"
    dim = 1024

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class Qwen3Embedder(BaseEmbedder):
    """Qwen/Qwen3-Embedding-0.6B. Only the query gets the official
    instruction prefix; passages are embedded raw."""

    name = "qwen3-0.6b"
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    dim = 1024

    _INSTRUCTION = (
        "Instruct: Given an Arabic legal question, "
        "retrieve the civil code articles that answer it\nQuery: "
    )

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.encode([self._INSTRUCTION + text])[0]


class GeminiEmbedder(BaseEmbedder):
    """google/gemini-embedding-2 via REST (validated shapes, zero deps).

    Query/passage are separated server-side by taskType, not by a prefix.
    embedding-2 aggregates a multi-input request into one vector, so you
    cannot batch: passage embedding = one request per article. Throttle
    knob keeps free-tier RPM in limits. Quality is the highest here
    (~MTEB multilingual 69.9), tradeoff is network + quota."""

    name = "gemini-2"
    model_id = "gemini-embedding-2"
    dim = 1024
    _URL = (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/gemini-embedding-2:embedContent"
    )

    def __init__(self) -> None:
        from os import getenv

        from dotenv import load_dotenv

        load_dotenv(config.PROJECT_ROOT / ".env")
        self._key = getenv("GEMINI_API_KEY")
        if not self._key:
            raise ValueError("GEMINI_API_KEY missing — add it to .env")
        self._sleep = config.GEMINI_MIN_INTERVAL
        self._last = 0.0

    def _call(self, text: str, task_type: str) -> np.ndarray:
        wait = self._sleep - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        body = {
            "model": f"models/{self.model_id}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.dim,
        }
        req = urllib.request.Request(
            self._URL,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
        self._last = time.monotonic()
        return np.asarray(out["embedding"]["values"], dtype=np.float32)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        return vec / np.linalg.norm(vec)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._normalize(self._call(t, "RETRIEVAL_DOCUMENT")) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._normalize(self._call(text, "RETRIEVAL_QUERY"))


_EMBEDDERS = {c.name: c for c in (E5Embedder, Bgem3Embedder, Qwen3Embedder, GeminiEmbedder)}


@functools.lru_cache(maxsize=None)
def get_embedder(name: str | None = None) -> BaseEmbedder:
    """Return the (cached) embedder instance by name; default from config."""
    name = (name or config.EMBEDDER).lower()
    try:
        return _EMBEDDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown embedder {name!r}, available: {sorted(_EMBEDDERS)}"
        ) from None
