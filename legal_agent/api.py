"""FastAPI wrapper: one endpoint over the agent, plus the static UI.

Run:
    uvicorn legal_agent.api:app --host 0.0.0.0 --port 8000

Startup fails fast if the index or the embedder is missing — better than
every request discovering it at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import agent, citations, retrieval

STATIC_DIR = Path(__file__).parent / "static"
MAX_HISTORY_TURNS = 20  # protect the context window; the UI trims, this enforces


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[Turn] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _warmup() -> None:
    """Touch both index paths once so a broken setup dies at boot, not per request."""
    retrieval.hybrid_search("اختبار الإقلاع", k=1)


def _web_used(state: dict) -> bool:
    from langchain_core.messages import AIMessage

    return any(
        isinstance(m, AIMessage) and any(c["name"] == "web_search" for c in m.tool_calls or [])
        for m in state.get("messages", [])
    )


app = FastAPI(title="LegalAI", description="أسئلة قانونية على القانون المدني المصري")


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    _warmup()
    try:
        state = agent.ask(
            req.question, history=[t.model_dump() for t in req.history[-MAX_HISTORY_TURNS:]]
        )
    except Exception as e:  # noqa: BLE001 — surface provider errors as 502, not stack traces
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {str(e)[:300]}") from e

    answer = state.get("answer", "")
    cited = citations.extract_citations(answer)
    seen, sources = set(), []
    for e in state.get("evidence", []):
        if e["article"] in cited and e["article"] not in seen:
            seen.add(e["article"])
            sources.append(e)
    return {
        "answer": answer,
        "sources": sources,
        "bad_citations": sorted(state.get("bad_citations") or []),
        "articles_retrieved": len({e["article"] for e in state.get("evidence", [])}),
        "web_used": _web_used(state),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
