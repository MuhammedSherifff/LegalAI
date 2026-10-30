"""Central configuration: paths, index names, active model picks."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "civil_law_records.json"
CHROMA_PATH = PROJECT_ROOT / "chroma"
BM25_PATH = PROJECT_ROOT / "indices" / "bm25.pkl"

COLLECTION_NAME = "civil_code_1948"


EMBEDDER = os.getenv("LEGALAI_EMBEDDER", "qwen3-0.6b")



# note for later eb2a add interval between gemini calls, to avoid rate limit errors
DEVICE = os.getenv("LEGALAI_DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("LEGALAI_BATCH", "16"))
GEMINI_MIN_INTERVAL = float(os.getenv("LEGALAI_GEMINI_INTERVAL", "0.2"))
LLM = os.getenv("LEGALAI_LLM", "gemini:gemini-3.5-flash-lite")
