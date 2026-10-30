"""Agent-facing tools: exactly what the LLM sees and may call.

One retrieval tool (hybrid), one exact-fetch tool, one web tool. Tools return
strings, never raise — failures come back as {"error": ...} so the agent can
decide what to do instead of crashing the graph.
"""

from __future__ import annotations

import json
import urllib.request

from langchain_core.tools import tool

from . import retrieval
from .retrieval import Hit

_TEXT_CHARS = 400       # enough to answer + cite, keeps context small
_SNIPPET_CHARS = 300
_LAW_NAME = {"civ_1948": "القانون المدني المصري", "civ_1948_enact": "قانون إصدار القانون المدني"}
_LAW_NAME_DEFAULT = "القانون المدني المصري"


def _hit_to_dict(hit: Hit) -> dict:
    return {
        "article": hit.article_number,
        "status": hit.legal_status,
        "law": _LAW_NAME.get(hit.id.split("_art_")[0], _LAW_NAME_DEFAULT),
        "section": hit.parent_titles,
        "page": hit.page_start,
        "text": hit.text[:_TEXT_CHARS],
    }


def _format(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _tool_fn(fn):
    """Wrap a function so tool errors return honest JSON instead of raising."""
    import functools

    @functools.wraps(fn)  # keeps name, docstring, schema for @tool
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e: 
            return _format({"error": f"{type(e).__name__}: {e}"})
    return wrapper


@tool
@_tool_fn
def retrieve_civil_law(query: str, k: int = 5) -> str:
    """Search the local Egyptian civil code index for the articles most relevant
    to the query. Use this for any question about Egyptian civil law
    (contracts, obligations, torts, property, inheritance exceptions...).
    Returns the top-k articles with number, status, and text."""
    hits = retrieval.hybrid_search(query, k=k)
    return _format({"results": [_hit_to_dict(h) for h in hits]})


@tool
@_tool_fn
def get_article_by_number(number: str) -> str:
    """Fetch the exact, complete text of a civil code article by its number.
    Use when the question names a specific article (e.g. 'ما نص المادة ١٦٣').
    Accepts '163' or Arabic-Indic '١٦٣'."""
    hit = retrieval.get_article_by_number(number)
    if hit is None:
        return _format({"error": f"لا توجد مادة برقم {number} في القانون المدني المصري"})
    return _format({"results": [_hit_to_dict(hit)]})


@tool
@_tool_fn
def web_search(query: str) -> str:
    """Search public web sources for legal information OUTSIDE the Egyptian
    civil code (other law fields, recent amendments, official announcements).
    Results carry title and URL; prefer official sources when composing the answer."""
    results = [{"title": r["title"], "url": r["url"], "content": r["content"][:_SNIPPET_CHARS]}
               for r in _tavily_search(query)]
    return _format({"results": results})


def _tavily_search(query: str) -> list[dict]:
    from dotenv import load_dotenv

    from . import config

    load_dotenv(config.PROJECT_ROOT / ".env")
    import os
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY missing — add it to .env")
    body = {
        "api_key": key,
        "query": query,
        "max_results": 5,
        "search_depth": "basic",
        "include_answer": False,
    }
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    return out.get("results", [])


ALL_TOOLS = [retrieve_civil_law, get_article_by_number, web_search]

# Exa try for later i guess
