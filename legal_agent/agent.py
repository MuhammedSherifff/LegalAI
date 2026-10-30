"""LangGraph agent: tool loop + code-side citation gate.

agent -> (tool calls) -> tools -> agent ... -> verify -> retry or END.
On corpus-grounded answers the model may only cite articles that appeared in
tool results; violations are bounced back once, then surfaced to the caller
in `bad_citations`. Web-only answers pass unverified (see verify()).

Two provider guards: malformed tool calls are retried with a corrective note
(Groq rejects them with tool_use_failed), and a tool-round cap forces a final
answer without tools (Gemini 3.5 Flash-Lite otherwise loops on tool calls).
"""

from __future__ import annotations

import json
import operator
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from . import citations, prompts
from .llm import get_llm
from .tools import ALL_TOOLS

MAX_RETRIES = 2
MAX_TOOL_CALL_RETRIES = 2
MAX_TOOL_ROUNDS = 4
_EVIDENCE_TOOLS = {"retrieve_civil_law", "get_article_by_number"}


class State(TypedDict, total=False):
    question: str
    messages: Annotated[list, operator.add]
    evidence: Annotated[list[dict], operator.add]
    web_results: Annotated[list[dict], operator.add]
    attempts: int
    tool_rounds: int
    answer: str
    bad_citations: set[str]


def agent(state: State) -> dict:
    llm = get_llm().bind_tools(ALL_TOOLS)
    messages = list(state["messages"])
    for attempt in range(MAX_TOOL_CALL_RETRIES + 1):
        try:
            return {"messages": [llm.invoke(messages)]}
        except Exception as e:  
            if "tool" not in str(e).lower() or attempt == MAX_TOOL_CALL_RETRIES:
                raise
            messages.append(HumanMessage(prompts.tool_call_correction()))


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
    messages, evidence, web = [], [], []
    for call in state["messages"][-1].tool_calls:
        tool = by_name.get(call["name"])
        if tool is None:
            content = json.dumps({"error": f"unknown tool {call['name']!r}"}, ensure_ascii=False)
        else:
            try:
                content = tool.invoke(call["args"])
            except Exception as e:  
                content = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        evidence.extend(_evidence_from(call["name"], content))
        if call["name"] == "web_search":
            try:
                web.extend(json.loads(content).get("results", []))
            except ValueError:
                pass
    out = {"messages": messages, "evidence": evidence,
           "tool_rounds": state.get("tool_rounds", 0) + 1}
    if web:
        out["web_results"] = web
    return out


def finalize(state: State) -> dict:
    """Cap reached: one answer-producing call with tools removed."""
    messages = list(state["messages"]) + [HumanMessage(prompts.FORCE_ANSWER)]
    return {"messages": [get_llm().invoke(messages)]}


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
            "messages": [HumanMessage(prompts.correction(bad))],
            "attempts": state.get("attempts", 0) + 1,
        }
    return {"answer": answer_text, "bad_citations": bad}


def _route_after_agent(state: State) -> str:
    last = state["messages"][-1]
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return "verify"
    if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
        return "finalize"
    return "tools"


def _route_after_verify(state: State) -> str:
    return "agent" if state.get("bad_citations") is None else "end"


@lru_cache(maxsize=1)
def _compiled():
    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", call_tools)
    g.add_node("finalize", finalize)
    g.add_node("verify", verify)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "verify": "verify", "finalize": "finalize"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "verify")
    g.add_conditional_edges("verify", _route_after_verify, {"agent": "agent", "end": END})
    return g.compile()


def ask(question: str, history: list[dict] | None = None) -> State:
    """Run the agent to completion and return the final state.

    `history` carries prior turns of the same conversation, oldest first:
    [{"role": "user" | "assistant", "content": str}, ...]. It is replayed
    before the new question so follow-ups resolve context and pronouns.
    """
    prefix = [
        HumanMessage(t["content"]) if t["role"] == "user" else AIMessage(t["content"])
        for t in (history or [])
    ]
    return _compiled().invoke(
        {
            "question": question,
            "messages": [SystemMessage(prompts.SYSTEM), *prefix, HumanMessage(question)],
            "evidence": [],
            "web_results": [],
            "attempts": 0,
            "tool_rounds": 0,
        },
        config={"recursion_limit": 30},
    )
