"""Central configuration: paths, index names, active model picks."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "civil_law_records.json"
CHROMA_PATH = PROJECT_ROOT / "chroma"
BM25_PATH = PROJECT_ROOT / "indices" / "bm25.pkl"

COLLECTION_NAME = "civil_code_1948"

# Active embedder, overridable via env or --embedder flag on ingest.
EMBEDDER = os.getenv("LEGALAI_EMBEDDER", "qwen3-0.6b")
# Embedding compute device + encode batch. fp16 GPU fits <4GB cards.
DEVICE = os.getenv("LEGALAI_DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("LEGALAI_BATCH", "16"))
# Gemini embedding throttle: min seconds between API calls (free-tier RPM guard).
GEMINI_MIN_INTERVAL = float(os.getenv("LEGALAI_GEMINI_INTERVAL", "0.2"))

# Chat model: "provider:model". Default = only model passing the full tool probe.
LLM = os.getenv("LEGALAI_LLM", "groq:qwen/qwen3.8-27b")
