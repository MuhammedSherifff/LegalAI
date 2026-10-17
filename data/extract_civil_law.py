import json
import re
import sys
from collections import Counter

import pymupdf

PDF_PATH = "/home/muhamed/Desktop/Spamming/LegalAI/CivialLaw.pdf"
OUT_PATH = "/home/muhamed/Desktop/Spamming/LegalAI/civil_law_records.json"

LAW_CODE = "civ_1948"
LAW_ENACT = "civ_1948_enact"

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
AR_TO_WEST = str.maketrans({ar: str(i) for i, ar in enumerate(AR_DIGITS)})
WEST_TO_AR = str.maketrans({str(i): ar for i, ar in enumerate(AR_DIGITS)})

ALEFS = "اأإآ"
AR_LETTERS = "أبجدهوزحطيكلمنسعفصقرشتثخذضظغ"
LETTER_NUM = {ch: i for i, ch in enumerate(AR_LETTERS)}

NORM_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ة"})
DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

ORDINAL_TOKENS = {
    "اولا": 1, "ااول": 1, "ثانيا": 2, "ثالثا": 3, "رابعا": 4,
    "خامسا": 5, "سادسا": 6, "سابعا": 7, "ثامنا": 8, "تاسعا": 9,
}
HEAD_ORDINALS = {"الاول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5, "السادس": 6}
HEAD_LEVEL_BASE = {"pre": 1, "p": 2, "b": 3, "c": 4, "s": 5, "t": 6, "u": 7}


def norm(s):
    s = s.translate(NORM_MAP)
    return DIACRITICS.sub("", s)


def is_arabic_font(name):
    return any(name.startswith(s) for s in (
        "Arial", "Sakkal", "Traditional", "Arabic", "Amiri", "Simplified"))


def fix_digit_runs(s):
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] in AR_DIGITS:
            j = i
            while j < n and s[j] in AR_DIGITS:
                j += 1
            out.append(s[i:j][::-1])
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def assemble_line(chars):
    stream = [(c, x, x) for c, x in chars]
    fused = []
    i = 0
    n = len(stream)
    while i < n:
        c, x, _ = stream[i]
        if (i > 0 and c in ALEFS and abs(x - stream[i - 1][1]) < 0.6
                and i + 1 < n and stream[i + 1][0] == "ل" and stream[i + 1][1] < x):
            key = min(x, stream[i + 1][1])
            fused.append((stream[i + 1][0], stream[i + 1][1], key))
            fused.append((c, x, key))
            i += 2
        else:
            fused.append((c, x, x))
            i += 1
    fused.sort(key=lambda t: -t[2])
    return fix_digit_runs("".join(t[0] for t in fused))


def clean_spaces(s):
    s = s.replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def tidy(s):
    s = clean_spaces(s)
    s = re.sub(r"\s+([،؛.:!؟])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s+\(", " (", s)
    s = re.sub(r"  +", " ", s)
    return s.strip()


class Line:
    __slots__ = ("page", "y", "text", "bold")

    def __init__(self, page, y, text, bold):
        self.page = page
        self.y = y
        self.text = text
        self.bold = bold

    def __repr__(self):
        return f"p{self.page+1} {self.text!r}"


def build_visual_lines(pdf):
    out = []
    for pno, page in enumerate(pdf):
        pending = []
        for block in page.get_text("rawdict")["blocks"]:
            if "lines" not in block:
                continue
            for rline in block["lines"]:
                for span in rline["spans"]:
                    if not is_arabic_font(span["font"]):
                        continue
                    bold = "Bold" in span["font"]
                    for ch in span["chars"]:
                        pending.append((ch["origin"][1], ch["c"], ch["origin"][0], bold))
        pending.sort(key=lambda t: (t[0], -t[2]))
        groups = []
        for ch in pending:
            if groups and ch[0] - groups[-1][0] <= 3.0:
                groups[-1][1].append(ch)
                if ch[3]:
                    groups[-1][2][0] = True
            else:
                groups.append((ch[0], [ch], [ch[3]]))
        for yy, chars, bold_flag in groups:
            txt = assemble_line([(c[1], c[2]) for c in chars])
            out.append(Line(pno, round(yy, 1), clean_spaces(txt), bold_flag[0]))
    return out


ARTICLE_RX = re.compile(r"^[\(\s]*مادة[\(\s]*([٠-٩0-9]+)[\)\s]*$")


def article_number_from(s):
    m = ARTICLE_RX.match(clean_spaces(s))
    return int(m.group(1).translate(AR_TO_WEST)) if m else None


REPEAL_START = re.compile(r"^\*?\s*(ألغيت\s+)?المواد\s*من\s*([٠-٩0-9]+)\s*إلى\s*([٠-٩0-9]+)")
REPEAL_ANY = re.compile(r"المواد\s*من\s*([٠-٩0-9]+)\s*إلى\s*([٠-٩0-9]+)\s*ملغاة")


def repeal_range(s):
    for rx in (REPEAL_START, REPEAL_ANY):
        m = rx.search(s)
        if m:
            a = int(m.group(2 if rx is REPEAL_START else 1).translate(AR_TO_WEST))
            b = int(m.group(3 if rx is REPEAL_START else 2).translate(AR_TO_WEST))
            if 1 <= a <= b <= 1200:
                return a, b
    return None


def match_heading(text, bold):
    t = clean_spaces(text)
    if not t or len(t) > 42:
        return None
    n = norm(t)
    if re.match(r"^باب\s+تمهيدي\b", n):
        return ("pre", 0, "inline")
    if not bold:
        return None
    m = re.match(r"^(القسم|الكتاب|الباب|الفصل)\s+(\S+)\s*[:\-–]*\s*(.*)$", n)
    if m:
        level = {"القسم": "p", "الكتاب": "b", "الباب": "c", "الفصل": "s"}[m.group(1)]
        num = HEAD_ORDINALS.get(m.group(2))
        if num is None or num == 0:
            return None
        if len(m.group(3).strip()) > 1:
            return (level, num, "inline")
        return (level, num, None)
    tok, sep, rest = n.partition(":")
    key = tok.strip().replace("-", "").strip()
    if sep and key in ORDINAL_TOKENS:
        num = ORDINAL_TOKENS[key]
        if len(rest.strip()) > 1:
            return ("t", num, "inline")
        return ("t", num, None)
    m = re.match(r"^([٠-٩0-9]+)\s*[-–]?\s*(\S.*)$", t)
    if m:
        return ("u", int(m.group(1).translate(AR_TO_WEST)), "inline")
    return None


def title_from(line, level, hnum):
    raw = clean_spaces(line)
    if level == "t":
        m = re.match(r"^[^:]+[:\-–]+\s*(\S.*)$", raw)
        if m:
            return tidy(m.group(1))
    if level == "u":
        m = re.match(r"^[٠-٩0-9]+\s*[-–:،]?\s*(\S.*)$", raw)
        if m:
            return tidy(m.group(1))
    if level == "pre":
        m = re.match(r"^باب\s+تمهيدي\b[:\-–]*\s*(\S.*)$", raw)
        if m:
            return tidy(m.group(1))
        return "أحكام عامة"
    return None


class HeadingStack:
    def __init__(self, law_id):
        self.law_id = law_id
        self.counters = Counter()
        self.path = []
        self.titles = []

    def push(self, level, title):
        base = HEAD_LEVEL_BASE[level]
        while len(self.path) >= base:
            self.path.pop()
            self.titles.pop()
        self.counters[level] += 1
        slug = f"{self.law_id}_h_{level}{self.counters[level]}"
        self.path.append(slug)
        self.titles.append(title)
        return list(self.path), list(self.titles)

    def current(self):
        return list(self.path), list(self.titles)


def make_article(law_id, num, body_lines, path, titles, p_start, p_end):
    raw = [l.text for l in body_lines if l.text]
    text = tidy(" ".join(raw))
    paragraphs = split_paragraphs(raw)
    return {
        "id": f"{law_id}_art_{num}",
        "law_id": law_id,
        "record_type": "article",
        "article_number": str(num),
        "title": None,
        "text": text,
        "parent_path": list(path),
        "parent_titles": list(titles),
        "paragraphs": paragraphs,
        "cross_refs": cross_refs(text),
        "legal_status": "in_force",
        "source_page_start": p_start + 1,
        "source_page_end": p_end + 1,
        "source_note": None,
    }


def cross_refs(text):
    found = []
    for m in re.finditer(r"(?:المادة|المواد|الفصل)\s*[(]?\s*([٠-٩0-9]+)", text):
        n = int(m.group(1).translate(AR_TO_WEST))
        if 1 <= n <= 1149 and n not in found:
            found.append(n)
    return found or None


PARA_RX = re.compile(r"^\(?([٠-٩0-9]+)\)?\s*[:،.؟]?\s*(.*)$")
ITEM_RX = re.compile(r"^([أ-ي])\s*(?:[-–]\s*|\)\s*)(.*)$")


def split_paragraphs(raw_lines):
    paragraphs = []
    cur = None
    items = []
    item_lines = []
    cur_lines = []
    lead = []
    prev_d = 0
    in_item = False

    def flush():
        nonlocal cur, items, cur_lines, item_lines, in_item, prev_d
        if cur is not None:
            paragraphs.append({
                "id": str(cur).translate(WEST_TO_AR),
                "text": tidy(" ".join(cur_lines)),
                "items": items or None,
            })
        cur, items, cur_lines, item_lines, in_item = None, [], [], [], False

    for raw in raw_lines:
        line = clean_spaces(raw)
        if not line:
            continue
        m = PARA_RX.match(line)
        d = int(m.group(1).translate(AR_TO_WEST)) if m else None
        if d is not None and (d == prev_d + 1 or (d == 1 and prev_d > 0 and cur is not None)):
            flush()
            prev_d = d
            cur = d
            first = m.group(2).strip()
            cur_lines = [first] if first else []
            continue
        im = ITEM_RX.match(line)
        if im and cur is not None:
            lid = im.group(1)
            expected = LETTER_NUM.get(AR_LETTERS[len(items)]) if items else None
            if expected is None or LETTER_NUM[lid] == expected + 1:
                if items:
                    items.append({"id": lid, "text": tidy(im.group(2))})
                else:
                    items = [{"id": lid, "text": tidy(im.group(2))}]
                in_item = True
                continue
        if in_item and items:
            items[-1]["text"] = tidy(items[-1]["text"] + " " + line)
        elif cur is not None:
            cur_lines.append(line)
        else:
            lead.append(line)
    flush()
    for p in paragraphs:
        if p["items"] is None:
            del p["items"]
    if lead:
        pre = {"id": "٠", "text": tidy(" ".join(lead))}
        paragraphs.insert(0, pre)
    return paragraphs


def extract_enactment(lines):
    stack = HeadingStack(LAW_ENACT)
    stack.push("t", "قانون الإصدار")
    records = []
    num = None
    buf = []
    p0 = None
    for idx, ln in enumerate(lines):
        n = article_number_from(ln.text)
        if n is not None:
            if num is not None:
                records.append(make_article(LAW_ENACT, num, buf, *stack.current(), p0, lines[idx - 1].page))
            num, buf, p0 = n, [], ln.page
        else:
            buf.append(ln)
    if num is not None:
        records.append(make_article(LAW_ENACT, num, buf, *stack.current(), p0, lines[-1].page))
    return records


def extract_code(lines):
    stack = HeadingStack(LAW_CODE)
    records = []
    notes = {}
    num = None
    buf = []
    p0 = None
    note_open = None
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        t = ln.text
        if note_open is not None:
            if t and (article_number_from(t) or match_heading(t, ln.bold) or repeal_range(t)):
                a, b, raw, pg = note_open
                notes[(a, b)] = {"from": a, "to": b,
                                 "raw": tidy(raw), "page": pg}
                note_open = None
            else:
                note_open[2] += " " + t
                i += 1
                continue
        r = repeal_range(t) if t and ('ملغاة' in t or 'الغيت' in norm(t) or 'ألغيت' in t) else None
        if r:
            note_open = [r[0], r[1], t, ln.page + 1]
            i += 1
            continue
        nnum = article_number_from(t)
        if nnum is not None:
            if num is not None:
                records.append(make_article(LAW_CODE, num, buf, *stack.current(), p0, lines[i - 1].page))
            num, buf, p0 = nnum, [], ln.page
            i += 1
            continue
        h = match_heading(t, ln.bold)
        if h is not None:
            if num is not None:
                records.append(make_article(LAW_CODE, num, buf, *stack.current(), p0, lines[i - 1].page))
                num, buf, p0 = None, [], None
            level, hnum, inline = h
            title = title_from(t, level, hnum) if inline == "inline" else None
            if not title and i + 1 < n:
                nxt = lines[i + 1]
                if (nxt.bold and 0 < len(nxt.text) <= 60
                        and article_number_from(nxt.text) is None
                        and match_heading(nxt.text, nxt.bold) is None
                        and not nxt.text.isdigit()):
                    title = nxt.text
                    i += 1
            if level == "b" and stack.counters["p"] == 1 and hnum == 3:
                stack.push("p", "الحقوق العينية")
            stack.push(level, title or f"{hnum}")
            i += 1
            continue
        buf.append(ln)
        i += 1
    if note_open is not None:
        a, b, raw, pg = note_open
        notes[(a, b)] = {"from": a, "to": b, "raw": tidy(raw), "page": pg}
    if num is not None:
        records.append(make_article(LAW_CODE, num, buf, *stack.current(), p0, lines[-1].page))
    return records, list(notes.values())


KNOWN_MISSING_STATUS = {1022: "unknown"}


def make_placeholder(num, status, raw_note, page):
    return {
        "id": f"{LAW_CODE}_art_{num}",
        "law_id": LAW_CODE,
        "record_type": "article",
        "article_number": str(num),
        "title": f"المادة {str(num).translate(WEST_TO_AR)}",
        "text": "",
        "parent_path": [],
        "parent_titles": [],
        "paragraphs": [],
        "cross_refs": None,
        "legal_status": status,
        "source_page_start": page,
        "source_page_end": page,
        "source_note": raw_note,
    }


def build_placeholders(records, notes):
    existing = {int(r["article_number"]) for r in records if r["law_id"] == LAW_CODE}
    out = []
    holder_path = {}
    for r in records:
        holder_path[int(r["article_number"])] = r
    for nt in notes:
        for num in range(nt["from"], nt["to"] + 1):
            if num in existing:
                holder_path[num]["legal_status"] = "repealed"
                holder_path[num]["source_note"] = nt["raw"]
                continue
            out.append(make_placeholder(num, "repealed", nt["raw"], nt["page"]))
    for num, status in KNOWN_MISSING_STATUS.items():
        if num not in existing:
            out.append(make_placeholder(num, status, None, None))
    return out


def validate(records):
    errs = []
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        errs.append(f"duplicate ids {dupes[:5]}")
    code = [r for r in records if r["law_id"] == LAW_CODE]
    nums = {int(r["article_number"]): r for r in code}
    expected = set(range(1, 1150))
    got = set(nums)
    if got - expected:
        errs.append(f"unexpected numbers {sorted(got - expected)[:10]}")
    if expected - got:
        errs.append(f"missing numbers {sorted(expected - got)[:10]}")
    orphan = [i for i, r in nums.items() if r["legal_status"] == "in_force" and not r["parent_path"]]
    if orphan:
        errs.append(f"in_force without parent {orphan[:10]}")
    empty = [i for i, r in nums.items() if r["legal_status"] == "in_force" and not r["text"]]
    if empty:
        errs.append(f"in_force without text {empty[:10]}")
    if errs:
        for e in errs:
            print("VALIDATION FAIL:", e)
        sys.exit(1)
    print(f"[OK] validation passed: {len(code)} code articles (expected {len(expected)}), "
          f"{len([r for r in records if r['law_id'] == LAW_ENACT])} enactment articles")


def main():
    pdf = pymupdf.open(PDF_PATH)
    lines = build_visual_lines(pdf)
    print(f"[1] Arabic lines: {len(lines)}")
    split_at = next((idx for idx, ln in enumerate(lines) if "نصوص القانون" in ln.text), None)
    if split_at is None:
        sys.exit("doc boundary not found")
    enact = [l for l in lines[:split_at] if l.text]
    code = [l for l in lines[split_at + 1:] if l.text]
    print(f"[2] enactment lines: {len(enact)}, code lines: {len(code)}")

    enact_records = extract_enactment(enact)
    code_records, notes = extract_code(code)
    print(f"[3] code articles: {len(code_records)}, notes: {notes}")

    code_records += build_placeholders(code_records, notes)
    records = enact_records + code_records
    print(f"[4] total records: {len(records)}")

    validate(records)

    laws = [
        {"law_id": LAW_CODE, "title_ar": "القانون المدني المصري", "doc_type": "civil_code",
         "promulgated": "1948-07-16", "effective": "1949-10-15", "source_pdf": "CivialLaw.pdf"},
        {"law_id": LAW_ENACT, "title_ar": "قانون الإصدار", "doc_type": "enactment_law",
         "promulgated": "1948-07-16", "effective": "1949-10-15", "source_pdf": "CivialLaw.pdf"},
    ]
    data = {"laws": laws, "records": records}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"[5] wrote {OUT_PATH}")

    sample = next(r for r in code_records if r["article_number"] == "121")
    print("sample 121:", json.dumps(sample, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
