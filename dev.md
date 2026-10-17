# dev.md — status, decisions, known limits

## Current state

Working agentic RAG (P1–P4 done):

- Corpus: Egyptian Civil Code 1948, 1151 records (1149 code + 2 enactment),
  extracted from `data/CivilLaw.pdf` by `scripts/extract_civil_law.py`
  (verified reproducible: rerunning yields byte-identical JSON).
- Index: 1094 in-force records (57 repealed/unknown placeholders carry no text
  and stay out of the index); Chroma + BM25, fused with reciprocal-rank.
- Agent: LangGraph loop (`graph.py`) over 3 tools (`tools.py`): hybrid search,
  exact article fetch (with repealed-article fallback from the JSON), web search.
- Citation gate (`citations.py`): cited articles must appear in tool results;
  violations bounce back once (max 2 retries), then surface as a warning.
  Web-only answers are exempt (other laws can't be verified code-side).
- CLI (`python -m legal_agent`), 16 pytest tests, packaging via pyproject.toml.

## Decisions log

- **Embedder**: qwen3-0.6b is canonical (index built with it). Swapping requires
  re-ingest; `get_embedder_for_collection` refuses mismatched queries.
- **Repealed articles**: not indexed (no text to embed). Exact lookups answer
  "ملغاة" with the repeal note from the JSON (`retrieval._placeholders`).
- **Flat package layout** (no `src/`): single service, not a published library.
- **BM25 payload cached** for process lifetime — restart after re-ingest.
- **PDF rename**: `CivialLaw.pdf` -> `CivilLaw.pdf` (typo); `source_pdf` in the
  JSON updated to match.
- **Chat model**: `gemini:gemini-3.5-flash-lite` is the config default
  (probed 2026-10-18: exact article 4s, semantic multi-article 19s/12 refs,
  web/labor 6s; no looping with the round cap in place).
  `LEGALAI_LLM=groq:qwen/qwen3.8-27b` is the verified fallback — one malformed
  tool-call 400 (`tool_use_failed`, parameter `url` instead of `query`) was
  observed on it, which is why the retry guard exists.
- **Provider guards (graph.py)**: malformed tool calls retried twice with a
  corrective note; tool rounds capped at 4, then `finalize` forces one
  no-tools answer call; the CLI catches graph errors so the REPL survives
  provider failures.

## Known limits / deferred

- **Embedder bake-off incomplete**: qwen3-0.6b (recall@5 0.88, MRR 0.78) >
  e5-small (0.82 / 0.62). bge-m3 (~2.2GB download) and gemini-2 (~1100 API
  calls) deferred by user choice. Resume:
  `python scripts/eval_retrieval.py --embedders bgem3` / `--embedders gemini-2`.
- Golden set is 17 questions, one field of law; expand before trusting the
  comparison.
- Retrieval misses some vocabulary-bridging questions (e.g. "الشرط الجزائي" ->
  art. 223). Candidate fixes: query rewriting node, reranker
  (`bge-reranker-base` is already in the HF cache), or a better embedder.
- `paragraphs` / `cross_refs` fields exist in the JSON but nothing reads them.
- No CI. No streaming in CLI. Web search is Tavily-only (Exa swap noted in
  tools.py if ever needed).

## Next candidates (unordered)

1. Finish embedder bake-off, pin the winner, re-ingest.
2. Query-rewrite node for vocabulary mismatch; reranker experiment.
3. Expand golden set (other law fields for web path, hard retrieval cases).
4. Eval harness for answers (not just retrieval): grounding, citation accuracy.
