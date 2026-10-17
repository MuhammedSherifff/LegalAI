"""Graph-node building blocks. P1: the answerer only.

Blending rules live here because prompt iteration is the main tuning loop —
nodes.py is the one file where behavior gets adjusted. llm.py stays clean
(nowhere does it see a message), retrieval stays numeric.
"""

from __future__ import annotations

from .llm import get_llm
from .retrieval import Hit

_STATUS_AR = {"in_force": "ساري", "repealed": "ملغى", "unknown": "الحالة غير مؤكدة"}

_SYSTEM = (
    "أنت مساعد قانوني متخصص في القانون المصري. ترد بالعربية الفصحى المبسطة.\n"
    "القواعد الصارمة:\n"
    "1. اعتمد حصرا على الأدلة المرقمة المعطاة إليك؛ لا تذكر أي مادة أو نص غير موجود في الأدلة.\n"
    "2. استشهد في موقع الاستشهاد بصيغة (المادة رقم) مثل: (المادة 163).\n"
    "3. إذا لم تكفِ الأدلة للإجابة كاملة، قل ذلك بصراحة وحدد ما هو موجود وما ينقص — ولا تختلق إجابة.\n"
    "4. إذا كان خارج نطاق القانون المدني المصري، قل ذلك ووضح أي جهة تصلح للسؤال.\n"
    "5. هذه معلومات عامة وليست استشارة قانونية — لا تتظاهر بغير ذلك."
)


def _evidence_block(evidence: list[Hit]) -> str:
    chunks = []
    for i, hit in enumerate(evidence, 1):
        status = _STATUS_AR.get(hit.legal_status, hit.legal_status)
        chunks.append(
            f"[{i}] المادة {hit.article_number} — الحالة: {status} — صفحة {hit.page_start}\n"
            f"{hit.text}"
        )
    return "\n\n".join(chunks)


def build_messages(question: str, evidence: list[Hit]) -> list:
    """Message list for the answerer: system rules + numbered evidence + question."""
    evidence_text = _evidence_block(evidence) or "لا توجد أدلة."
    return [
        ("system", _SYSTEM),
        (
            "human",
            "الأدلة المستندية:\n\n"
            f"{evidence_text}\n\n"
            "السؤال:\n"
            f"{question}",
        ),
    ]


def answer(question: str, evidence: list[Hit]) -> str:
    """One LLM call: Arabic evidence-grounded answer with (المادة N) citations."""
    return get_llm().invoke(build_messages(question, evidence)).text
