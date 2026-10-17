"""LangGraph agent: tool loop + code-side citation gate.

agent -> (tool calls) -> tools -> agent ... -> verify -> retry or END.
On corpus-grounded answers the model may only cite articles that appeared in
tool results; violations are bounced back once, then surfaced to the caller
in `bad_citations`. Web-only answers pass unverified (see verify()).
"""

from __future__ import annotations

import json
import operator
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from . import citations, nodes
from .llm import get_llm
from .tools import ALL_TOOLS

MAX_RETRIES = 2
_EVIDENCE_TOOLS = {"retrieve_civil_law", "get_article_by_number"}


class State(TypedDict, total=False):
    question: str
    messages: Annotated[list, operator.add]
    evidence: Annotated[list[dict], operator.add]
    attempts: int
    answer: str
    bad_citations: set[str]


def agent(state: State) -> dict:
    llm = get_llm().bind_tools(ALL_TOOLS)
    return {"messages": [llm.invoke(state["messages"])]}


def _evidence_from(tool_name: str, content: str) -> list[dict]:
    if tool_name not in _EVIDENCE_TOOLS:
        return []
    try:
        data = json.loads(content)
    except ValueError:
        return []
    return data.get("results", [])


def call_tools(state: State) -> dict:
    by_name = {t.name: t for t in ALL_TOOLS}
    messages, evidence = [], []
    for call in state["messages"][-1].tool_calls:
        tool = by_name.get(call["name"])
        if tool is None:
            content = json.dumps({"error": f"unknown tool {call['name']!r}"}, ensure_ascii=False)
        else:
            try:
                content = tool.invoke(call["args"])
            except Exception as e:  # noqa: BLE001 — the message IS the result
                content = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        evidence.extend(_evidence_from(call["name"], content))
    return {"messages": messages, "evidence": evidence}


def verify(state: State) -> dict:
    """The gate only applies to corpus-grounded answers: citations can be
    checked against tool results solely for the indexed civil code. Web-only
    answers cite other laws from sources we cannot verify code-side."""
    answer_text = state["messages"][-1].text
    numbers = {e["article"] for e in state.get("evidence", [])}
    if not numbers:
        return {"answer": answer_text, "bad_citations": set()}
    bad = citations.check(answer_text, numbers)
    if bad and state.get("attempts", 0) < MAX_RETRIES:
        return {
            "messages": [HumanMessage(nodes.correction(bad))],
            "attempts": state.get("attempts", 0) + 1,
        }
    return {"answer": answer_text, "bad_citations": bad}


def _route_after_agent(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"


def _route_after_verify(state: State) -> str:
    return "agent" if state.get("bad_citations") is None else "end"


@lru_cache(maxsize=1)
def _compiled():
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", call_tools)
    g.add_node("verify", verify)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", _route_after_verify, {"agent": "agent", "end": END})
    return g.compile()


def ask(question: str) -> State:
    """Run the agent to completion and return the final state."""
    return _compiled().invoke(
        {
            "question": question,
            "messages": [SystemMessage(nodes.SYSTEM), HumanMessage(question)],
            "evidence": [],
            "attempts": 0,
        },
        config={"recursion_limit": 30},
    )
