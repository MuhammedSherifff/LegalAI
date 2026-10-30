import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from legal_agent import agent


class ScriptedModel(BaseChatModel):
    """Returns canned AIMessages in order; optional queued raises first."""

    responses: list
    errors: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.errors:
            raise self.errors.pop(0)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


@tool
def retrieve_civil_law(query: str) -> str:
    """Stub retrieval."""
    return json.dumps(
        {
            "results": [
                {
                    "article": "163",
                    "status": "in_force",
                    "law": "القانون المدني المصري",
                    "section": "العمل غير المشروع",
                    "page": 19,
                    "text": "كل خطأ سبب ضررا للغير يلزم من ارتكبه بالتعويض",
                }
            ]
        },
        ensure_ascii=False,
    )


def scripted(monkeypatch, responses, tools=(), errors=()):
    model = ScriptedModel(responses=list(responses), errors=list(errors))
    monkeypatch.setattr(agent, "get_llm", lambda: model)
    monkeypatch.setattr(agent, "ALL_TOOLS", list(tools))
    return model


def test_tool_loop_and_grounded_answer(monkeypatch):
    scripted(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_civil_law", "args": {"query": "ضرر"}, "id": "c1"}
                ],
            ),
            AIMessage(content="وفقا للمادة 163 يلزم المتسبب بالتعويض."),
        ],
        tools=[retrieve_civil_law],
    )
    state = agent.ask("ما حكم من سبب ضررا للغير؟")
    assert "163" in state["answer"]
    assert {e["article"] for e in state["evidence"]} == {"163"}
    assert state["bad_citations"] == set()
    assert state["attempts"] == 0


def test_fabricated_citation_triggers_retry(monkeypatch):
    scripted(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "retrieve_civil_law", "args": {"query": "ضرر"}, "id": "c1"}
                ],
            ),
            AIMessage(content="وفقا للمادة 999 يجوز ذلك."),
            AIMessage(content="وفقا للمادة 163 يلزم المتسبب بالتعويض."),
        ],
        tools=[retrieve_civil_law],
    )
    state = agent.ask("سؤال")
    assert state["attempts"] == 1
    assert state["bad_citations"] == set()
    assert "999" not in state["answer"]


def test_persistent_fabrication_surfaces(monkeypatch):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "retrieve_civil_law", "args": {"query": "ضرر"}, "id": "c1"}],
    )
    fabricated = AIMessage(content="نص المادة 999 واضح.")
    scripted(monkeypatch, [tool_call, fabricated] + [fabricated] * 2, tools=[retrieve_civil_law])
    state = agent.ask("سؤال")
    assert state["bad_citations"] == {"999"}
    assert "999" in state["answer"]


def test_web_only_answer_skips_gate(monkeypatch):
    scripted(
        monkeypatch,
        [AIMessage(content="وفقا للمادة 318 من قانون العقوبات تُعاقب السرقة بالحبس.")],
    )
    state = agent.ask("ما عقوبة السرقة؟")
    assert state["bad_citations"] == set()
    assert state["attempts"] == 0


def _tool_call(i: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "retrieve_civil_law", "args": {"query": f"س{i}"}, "id": f"c{i}"}
        ],
    )


def test_tool_round_cap_forces_final_answer(monkeypatch):
    scripted(
        monkeypatch,
        [_tool_call(i) for i in range(agent.MAX_TOOL_ROUNDS + 1)]
        + [AIMessage(content="إجابة نهائية دون استشهاد.")],
        tools=[retrieve_civil_law],
    )
    state = agent.ask("سؤال يغري بالبحث المتكرر")
    assert state["tool_rounds"] == agent.MAX_TOOL_ROUNDS
    assert state["answer"] == "إجابة نهائية دون استشهاد."


def test_malformed_tool_call_retried(monkeypatch):
    scripted(
        monkeypatch,
        [AIMessage(content="الإجابة بعد إعادة المحاولة.")],
        errors=[ValueError("tool call validation failed: missing properties: 'query'")],
    )
    state = agent.ask("سؤال")
    assert state["answer"] == "الإجابة بعد إعادة المحاولة."


def test_non_tool_error_propagates(monkeypatch):
    scripted(monkeypatch, [], errors=[RuntimeError("invalid api key")])
    with pytest.raises(RuntimeError, match="invalid api key"):
        agent.ask("سؤال")
