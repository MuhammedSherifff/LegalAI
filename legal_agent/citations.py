"""Code-side citation verification — the hard gate.

The answer may only cite article numbers that exist in the evidence set it
was built from. Extraction is regex (deterministic), never an LLM call.
"""

from __future__ import annotations

import re

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# (المادة 163) | (المادة ١٦٣) | مادة 163 — allow both bracketed and bare forms
_CITED = re.compile(r"المادة\s+([0-9٠-٩]+)")


def extract_citations(text: str) -> set[str]:
    """ASCII-normalized article numbers cited in an Arabic answer."""
    return {m.translate(_AR_DIGITS) for m in _CITED.findall(text)}


def check(answer_text: str, evidence_numbers: set[str]) -> set[str]:
    """Cited numbers missing from the evidence set. Empty set = verified."""
    return extract_citations(answer_text) - evidence_numbers
