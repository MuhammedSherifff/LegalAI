# legal_agent/

The only package. Agentic RAG over the Egyptian Civil Code.

## Module map

| Module | Role |
|---|---|
| `config.py` | paths, index names, active model/embedder picks (env-overridable) |
| `embeddings.py` | embedder abstraction (e5-small / bge-m3 / qwen3-0.6b / gemini-2), index-consistency guard |
| `llm.py` | provider adapter: Groq / Gemini / OpenRouter via one `get_llm()` |
| `ingest.py` | JSON → Chroma + BM25 build script (`python -m legal_agent.ingest`) |
| `retrieval.py` | hybrid search (RRF), exact article lookup, repealed fallback |
| `tools.py` | the 3 agent-facing tools: retrieve / exact fetch / web search |
| `nodes.py` | prompt layer — the one file where answer behavior gets tuned |
| `graph.py` | LangGraph agent loop + citation gate (`graph.ask(question)`) |
| `citations.py` | regex citation extraction + evidence-set enforcement |
| `cli.py` / `__main__.py` | terminal REPL (`python -m legal_agent`) |

Status, decisions and known limits live in [`../dev.md`](../dev.md).
