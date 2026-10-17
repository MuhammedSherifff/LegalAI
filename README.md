# LegalAI

Agentic RAG over the Egyptian Civil Code. Local-first (Chroma + BM25), answers
in Arabic with a code-side citation gate, and web search for law fields outside
the indexed corpus.

## Setup

```bash
python -m venv .venv            # or: uv venv
uv pip install -e ".[dev,extract]"
cp .env.example .env            # fill the provider keys you use
```

## Build the indices

```bash
python -m legal_agent.ingest                        # Chroma + BM25 from data/civil_law_records.json
python -m legal_agent.ingest --check "التقادم المسقط" # sanity query against the built indices
```

The index rebuild must use the same embedder that queries use — a mismatch is
refused at query time (see `embeddings.get_embedder_for_collection`).

## Use

```bash
python -m legal_agent                     # interactive REPL
python -m legal_agent "ما نص المادة 163؟"  # one-shot
```

The agent decides which tools to call (civil-code search, exact article fetch,
web search), answers from their results, and every cited article number is
verified against them. Fabricated citations are bounced back once, then shown
as a warning.

## Tests

```bash
.venv/bin/python -m pytest
```

## Retrieval bake-off

```bash
python scripts/eval_retrieval.py                       # qwen3-0.6b, e5-small, bgem3
python scripts/eval_retrieval.py --embedders gemini-2  # API embedder, ~1100 calls
```

Scores `eval/golden_set.json` (recall@5, MRR, speed) on throwaway indices;
never touches the canonical one. Partial results so far: qwen3-0.6b recall@5
0.88 / MRR 0.78, e5-small 0.82 / 0.62 (bge-m3 and Gemini pending).

## Layout

```
legal_agent/        package: config, embeddings, llm, ingest, retrieval, tools,
                    nodes (prompts), graph (agent loop), citations, cli
data/               corpus: CivilLaw.pdf + civil_law_records.json (generated)
scripts/            extract_civil_law.py (PDF -> JSON), eval_retrieval.py
eval/               golden_set.json
tests/              pytest suite
indices/, chroma/   built artifacts (gitignored — rebuild with ingest)
```

## Regenerating the corpus

```bash
python scripts/extract_civil_law.py    # data/CivilLaw.pdf -> data/civil_law_records.json
python -m legal_agent.ingest           # rebuild indices after
```

Status, decisions and known limits: [dev.md](dev.md).
