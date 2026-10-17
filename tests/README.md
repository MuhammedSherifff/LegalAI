# tests/

pytest suite (`.venv/bin/python -m pytest`).

Coverage:

- `test_citations.py` — every citation phrasing the model may emit, fabricated-citation detection
- `test_graph.py` — agent loop with a scripted fake model: tool call → grounded answer, fabrication retry, web-only exemption
- `test_retrieval.py` — exact lookup (Arabic digits), repealed fallback, hybrid hits for known articles (skips if indices not built)
- `test_embeddings_guard.py` — index/embedder mismatch refusal
- `test_ingest.py` — Arabic-Indic digit normalization in tokenization
