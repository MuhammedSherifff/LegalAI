"""API contract tests: agent.ask mocked, no LLM, no index."""

import pytest
from fastapi.testclient import TestClient

import legal_agent.api as api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_warmup", lambda: None)
    return TestClient(api.app)


def fake_ask(question, history=None):
    fake_ask.seen = (question, history)
    return {
        "messages": [],
        "answer": "نص المادة (المادة 163) واضح.",
        "evidence": [{"article": "163", "status": "in_force", "page": 40, "section": "الباب الأول"}],
        "bad_citations": set(),
        "tool_rounds": 1,
    }


def test_ask_shape(client, monkeypatch):
    monkeypatch.setattr(api.agent, "ask", fake_ask)
    r = client.post("/ask", json={"question": "ما نص المادة 163؟"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("نص")
    assert body["sources"][0]["article"] == "163"
    assert body["bad_citations"] == []
    assert body["articles_retrieved"] == 1
    assert body["web_used"] is False


def test_history_passed_and_trimmed(client, monkeypatch):
    captured = {}

    def spy(question, history=None):
        captured["history"] = history
        return {"messages": [], "answer": "ok", "evidence": [], "bad_citations": set()}

    monkeypatch.setattr(api.agent, "ask", spy)
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"t{i}"} for i in range(25)]
    r = client.post("/ask", json={"question": "س؟", "history": turns})
    assert r.status_code == 200
    assert len(captured["history"]) == api.MAX_HISTORY_TURNS


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
