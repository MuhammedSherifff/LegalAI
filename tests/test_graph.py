import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from legal_agent import graph


class ScriptedModel(BaseChatModel):
    """Returns canned AIMessages in order; ignores bind_tools."""

    responses: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
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


def scripted(monkeypatch, responses, tools=()):
    model = ScriptedModel(responses=list(responses))
    monkeypatch.setattr(graph, "get_llm", lambda: model)
    monkeypatch.setattr(graph, "ALL_TOOLS", list(tools))
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
    state = graph.ask("ما حكم من سبب ضررا للغير؟")
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
    state = graph.ask("سؤال")
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
    state = graph.ask("سؤال")
    assert state["bad_citations"] == {"999"}
    assert "999" in state["answer"]


def test_web_only_answer_skips_gate(monkeypatch):
    scripted(
        monkeypatch,
        [AIMessage(content="وفقا للمادة 318 من قانون العقوبات تُعاقب السرقة بالحبس.")],
    )
    state = graph.ask("ما عقوبة السرقة؟")
    assert state["bad_citations"] == set()
    assert state["attempts"] == 0
