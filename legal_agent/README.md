# legal_agent/

The only package. Agentic RAG over the Egyptian Civil Code.

## Module map (planned)

| Module | Role |
|---|---|
| `config.py` | env keys, model names, constants |
| `embeddings.py` | embedder abstraction: e5-small / bge-m3, `get_embedder()` |
| `llm.py` | provider adapter: Groq / Gemini / OpenRouter via one init |
| `ingest.py` | JSON → Chroma + BM25 build script |
| `retrieval.py` | hybrid search, filters, `get_article_by_number` |
| `web_search.py` | Tavily tool |
| `nodes.py` | planner, evidence_gate, rewrite, answerer (graph nodes) |
| `graph.py` | LangGraph `StateGraph` wiring |
| `citations.py` | post-answer citation verification (code, not LLM) |
| `eval.py` | golden 20-question eval: retrieval hits, no fabricated citations |
| `cli.py` | `python -m legal_agent` REPL, pretty-printed answers |

Modules appear here as phases progress — see `../dev.md` for status and design decisions.
