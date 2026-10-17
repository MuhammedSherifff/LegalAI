"""Code-side citation verification — the hard gate.

The answer may only cite article numbers that exist in the evidence set it
was built from. Extraction is regex (deterministic), never an LLM call.

Known limits: article ranges ("المواد من 5 إلى 10") are not expanded, and a
joined list only keeps what follows the noun ("المادة 163 و164" works,
"163 و164" alone does not). Both fail toward *fewer* flagged numbers.
"""

from __future__ import annotations

import re

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# مادة/المادة/المواد (+رقم), or the abbreviation م standing as its own token.
# Prefixed forms (للمادة، بالمادة، والمادة) match via their مادة tail.
_CITED = re.compile(
    r"(?:المواد|المادة|مادة|(?:^|[\s(،؛.:])م)"
    r"\s*(?:رقم\s*)?[.:]?\s*\(?\s*(?P<num>[0-9٠-٩]+)"
    r"(?P<tail>(?:\s*(?:و|,|،)\s*[0-9٠-٩]+)*)"
)


def extract_citations(text: str) -> set[str]:
    """ASCII-normalized article numbers cited in Arabic text."""
    out: set[str] = set()
    for m in _CITED.finditer(text):
        out.add(m.group("num").translate(_AR_DIGITS))
        out.update(t.translate(_AR_DIGITS) for t in re.findall(r"[0-9٠-٩]+", m.group("tail")))
    return out


def check(answer_text: str, evidence_numbers: set[str]) -> set[str]:
    """Cited numbers missing from the evidence set. Empty set = verified."""
    return extract_citations(answer_text) - evidence_numbers
