"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

# FTEC5660 HW4: student-editable functions and their helpers.
# Paste into hw4.py between the "Your code" and "Provided code below" markers.
from collections import Counter, defaultdict
import math
import re
from typing import Any
from bs4 import BeautifulSoup, Tag

# Feature switches for the reproducible one-change-at-a-time ablation experiment.
# The submitted/default configuration enables all switches.
PIPELINE_FLAGS = {
    "table_mode": True,
    "structure_chunk": True,
    "metadata_filter": True,
    "tokeniser": True,
    "metric_boost": True,
    "balanced_comparison": True,
    "abstention": True,
}
_KNOWN_COMPANIES: set[str] = set()

_STOP = set("a an the and or of at by in on for from to was were is are what which who how much many did does do its it their with as than that this end year years fiscal usd million millions percent per share amount value company held hold have had more higher less lower did report reports reported please during into based do".split())
_LEGAL = {"inc", "corp", "corporation", "limited", "ltd", "plc", "llc", "co"}


def _normal(s):
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ")).strip()


def _company_key(s):
    return " ".join(t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if t not in _LEGAL)


def _clean_company(s):
    s = _normal(re.sub(r"\([^)]*\)", "", str(s)))
    s = re.sub(r"\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|plc)\.?$", "", s, flags=re.I)
    return s.strip(" ,.–-'")


def _doc_metadata(file_name, soup):
    h1 = soup.find("h1")
    title = _normal(h1.get_text(" ", strip=True)) if h1 else ""
    head = _normal(soup.get_text(" ", strip=True)[:1700])
    company = _clean_company(title)
    # In a research report, the broker is in the top line but the <h1> is the company.
    company = re.sub(r"\s*\([^)]*(?:NYSE|NASDAQ|LSE|HKEX|ticker|:)[^)]*\)$", "", company, flags=re.I)
    company = _clean_company(company)
    if "earnings conference call" in head.lower() or "earnings call" in head.lower():
        doc_type = "earnings_call"
    elif "results review" in head.lower() or ("equity research" in head.lower() and "target price" in head.lower()):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    # The fiscal year of calls / research is the year reviewed, not the conference date.
    fy = re.search(r"(?:Annual\s+Report|Full\s+Year|FY)\s*(20\d\d)", head, re.I)
    if fy is None:
        fy = re.search(r"FY\s*(20\d\d)\s+results review", head, re.I)
    if fy is None:
        fy = re.search(r"fiscal year ended\s+\d{1,2}\s+\w+\s+(20\d\d)", head, re.I)
    year = int(fy.group(1)) if fy else 2025
    _KNOWN_COMPANIES.add(company)
    return {"source": file_name, "company": company, "fiscal_year": year, "doc_type": doc_type}


def table_to_text(table_html: str) -> str:
    """Preserve each HTML table cell's row label, column header and printed value."""
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return ""
    cap = table.find("caption")
    caption = _normal(cap.get_text(" ", strip=True)) if cap else ""
    rows = table.find_all("tr")
    if not rows:
        return caption
    def values(tr):
        return [_normal(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"], recursive=False)]
    # Most source tables contain one <thead> header row; fall back to the first row.
    head = table.find("thead")
    hr = (head.find_all("tr")[-1] if head and head.find("tr") else rows[0])
    headers = values(hr)
    if not headers:
        return caption
    out = [caption] if caption else []
    for tr in rows:
        if tr is hr or tr.find_parent("thead") is not None:
            continue
        cells = values(tr)
        if not cells:
            continue
        row_label = cells[0]
        if not row_label:
            continue
        # A full-width section header is kept as context, not interpreted as data.
        if len(cells) == 1:
            out.append(row_label)
            continue
        bits = []
        for j, cell in enumerate(cells[1:], 1):
            if cell:
                col = headers[j] if j < len(headers) else f"Column {j}"
                bits.append(f"{col}: {cell}")
        if not bits:
            out.append(row_label)
            continue
        line = f"{row_label} | " + " | ".join(bits)
        if len(line) <= 400:
            out.append(line)
        else:
            # Requirement limits every scored line to 400 characters.
            out.extend(f"{row_label} | {b}" for b in bits if len(row_label) + len(b) + 3 <= 400)
    return "\n".join(out)


def _plain_split(text, max_chars=1450, overlap=100):
    """A no-extra-dependency stand-in for recursive character splitting."""
    result = []
    text = str(text)
    if not text:
        return result
    i = 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        if j < len(text):
            # Prefer breaking at paragraph, sentence or space boundaries.
            for sep in ["\n\n", "\n", ". ", " "]:
                pos = text.rfind(sep, i + max_chars // 2, j)
                if pos != -1:
                    j = pos + len(sep)
                    break
        if j <= i:
            j = min(len(text), i + max_chars)
        result.append(text[i:j].strip())
        if j == len(text):
            break
        i = max(i+1, j - overlap)
    return [s for s in result if s]


def _legacy_document(soup, meta):
    body = soup.body or soup
    if PIPELINE_FLAGS["table_mode"]:
        for table in list(body.find_all("table")):
            p = soup.new_tag("p")
            p.string = table_to_text(str(table))
            table.replace_with(p)
    plain = body.get_text("\n", strip=True)
    # Baseline must use RecursiveCharacterTextSplitter (as instructed in Task 5).
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        pieces = RecursiveCharacterTextSplitter(chunk_size=1450, chunk_overlap=100).split_text(plain)
    except ImportError:
        pieces = _plain_split(plain)
    return [Document(page_content=x[:1500], metadata=dict(meta)) for x in pieces]


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Parse one document into at most 1500-character source-grounded passages."""
    soup = BeautifulSoup(html, "lxml")
    for x in soup.find_all(["style", "script", "nav", "footer"]):
        x.decompose()
    meta = _doc_metadata(file_name, soup)
    if not PIPELINE_FLAGS["structure_chunk"]:
        return _legacy_document(soup, meta)
    body = soup.body or soup
    chunks: list[Document] = []
    section = ""
    buffer: list[str] = []
    buffer_length = 0
    def emit(text, title=""):
        for part in _plain_split(text, max_chars=1450, overlap=60):
            if part:
                chunks.append(Document(page_content=part[:1500], metadata={**meta, "section": title}))
    def flush():
        nonlocal buffer, buffer_length
        if buffer:
            emit("\n".join(buffer), section)
        buffer, buffer_length = [], 0
    def append_line(txt):
        nonlocal buffer_length
        txt = _normal(txt)
        if not txt:
            return
        if len(txt) > 1420:
            flush()
            emit(txt, section)
            return
        if buffer and buffer_length + len(txt) + 1 > 1420:
            flush()
        buffer.append(txt)
        buffer_length += len(txt) + 1
    # Iterate document order without picking up nested cells / tags twice.
    for tag in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if tag.find_parent("table") is not None:
            continue
        if tag.name in {"h1", "h2", "h3", "h4"}:
            flush()
            section = _normal(tag.get_text(" ", strip=True))
            append_line(section)
        elif tag.name == "table":
            flush()
            table_text = table_to_text(str(tag))
            lines = table_text.splitlines()
            if not lines:
                continue
            caption = lines[0] if tag.find("caption") else ""
            prefix = (section + "\n" if section else "") + (caption + "\n" if caption else "")
            part = prefix
            data_lines = lines[1:] if caption else lines
            for line in data_lines:
                if len(part) + len(line) + 1 > 1450 and part.strip() != prefix.strip():
                    emit(part.strip(), section)
                    part = prefix
                part += line + "\n"
            if part.strip():
                emit(part.strip(), section)
        elif tag.name == "li":
            if tag.find_parent("li") is None and tag.find_parent("table") is None:
                append_line(tag.get_text(" ", strip=True))
        elif tag.name == "p":
            if tag.find_parent("li") is None:
                append_line(tag.get_text(" ", strip=True))
    flush()
    return chunks


def _capital_name(s):
    s = _normal(s).strip(" ,:?.")
    # Restrict to 2-6 capitalised words; avoids matching metric phrases.
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){1,5}", s))


def _find_companies(question):
    q = _normal(question)
    names = []
    def add(name):
        name = _clean_company(name)
        if _capital_name(name) and _company_key(name) not in {_company_key(n) for n in names}:
            names.append(name)
    # Comparisons are separated by 'or' after a colon, or 'of X and Y'.
    m = re.search(r":\s*([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){1,5})\s+or\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){1,5})\s*\??\s*$", q)
    if m:
        add(m.group(1)); add(m.group(2))
    m = re.search(r"\bof\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){1,5})\s+and\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){1,5})\s+(?:held|had|reported|earned|generated)\b", q)
    if m:
        add(m.group(1)); add(m.group(2))
    # Possessives: includes unknown companies, vital for abstention.
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){1,5})['’](?:s\b|(?=\s))", q):
        add(m.group(1))
    # Non-possessive company after 'did', 'for', 'give', 'audited' or 'is'.
    verbs = r"(?:hold|report|generate|earn|announce|spend|record|have|provide|disclose|invest|generate|have|publish|forecast)"
    for m in re.finditer(r"\bdid\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){1,5})\s+" + verbs + r"\b", q):
        add(m.group(1))
    for m in re.finditer(r"\b(?:for|give|audited|is)\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){1,5})(?=\s+(?:in|headquartered|its|shares|at|for|on)|\s*\?|,|$)", q):
        add(m.group(1))
    # Explicit known entities fill gaps in questions of less predictable wording.
    # Do not choose a known name that overlaps with an extracted *different* company.
    for company in sorted(_KNOWN_COMPANIES, key=len, reverse=True):
        ck = _company_key(company)
        if re.search(r"(?<![A-Za-z])" + re.escape(company) + r"(?![A-Za-z])", q, re.I):
            if not any(ck in _company_key(n) or _company_key(n) in ck for n in names):
                add(company)
    # Match known case against the canonical spelling if found in a question.
    canonical = {_company_key(c): c for c in _KNOWN_COMPANIES}
    return [canonical.get(_company_key(n), n) for n in names]


def parse_query(question: str) -> dict:
    q = _normal(question)
    # Take the question's named year, not the document's own fiscal year.
    y = re.search(r"\bFY\s*[- ]?(20\d\d)\b", q, re.I)
    if not y:
        y = re.search(r"\b(?:31\s+December|December\s+31|end\s+of)\s*,?\s*(20\d\d)\b", q, re.I)
    if not y:
        y = re.search(r"\b(20\d\d)\b", q)
    year = int(y.group(1)) if y else None
    low = q.lower()
    if re.search(r"earnings?\s+(?:conference\s+)?call|fourth.quarter\s+call|q4.*call", low):
        dtype = "earnings_call"
    elif re.search(r"\b(?:broker|investment rating|target price|results review|forecast|equity research|analyst estimate|estimate changes)\b", low):
        dtype = "research_report"
    else:
        dtype = "annual_report"
    return {"companies": _find_companies(q), "fiscal_year": year, "doc_type": dtype}


def _tokens(text):
    # Important: 'accounts receivable' and FY2026E should remain searchable.
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", str(text).lower())


def _bm25_build(chunks, tokeniser=True):
    toks = [(_tokens(d.page_content) if tokeniser else d.page_content.lower().split()) for d in chunks]
    freq = [Counter(t) for t in toks]
    df = Counter()
    for f in freq:
        df.update(f.keys())
    n = len(chunks)
    mean_len = sum(map(len, toks))/max(n, 1)
    idf = {w: math.log(1 + (n - count + .5)/(count + .5)) for w,count in df.items()}
    return {"freq": freq, "lengths": [len(t) for t in toks], "idf": idf, "mean": mean_len}


def build_index(chunks: list[Document]) -> Any:
    """Compact in-memory BM25 plus metadata; build from supplied corpus each run."""
    grouped = defaultdict(list)
    for i, ch in enumerate(chunks):
        grouped[_company_key(ch.metadata.get("company", ""))].append(i)
    return {"chunks": chunks, "bm25": _bm25_build(chunks, True), "bm25_raw": _bm25_build(chunks, False), "groups": grouped}


def _metric_info(q):
    """Pure question-driven metric/section synonyms, not any public answer or company."""
    low = q.lower()
    if "net accounts receivable" in low or "receivables" in low:
        return ["accounts receivable, net", "net accounts receivable", "accounts receivable"], ["receivable"]
    if "long-term borrowing" in low or "long term borrowing" in low:
        return ["long-term borrowings", "long-term borrowing", "borrowings"], ["borrowing"]
    if "cash and cash equivalents" in low:
        return ["cash and cash equivalents"], ["cash"]
    if "net cash from operating" in low:
        return ["net cash from operating activities", "operating activities"], ["cash flow"]
    if "chief financial officer" in low or re.search(r"\bcfo\b", low):
        return ["chief financial officer"], ["executive officers"]
    if "chief executive officer" in low or re.search(r"\bceo\b", low):
        return ["chief executive officer"], ["executive officers"]
    if "auditor" in low or "audit firm" in low:
        return ["independent auditor", "audited", "auditor"], ["auditor"]
    if "target price" in low:
        return ["target price", "12-month"], ["valuation"]
    if "rating" in low:
        return ["rating", "investment rating"], ["rating"]
    if "employees" in low or "headcount" in low or "people employed" in low:
        return ["total employees", "employees", "employed"], ["employee"]
    if "share repurchase" in low or "buyback" in low:
        return ["share repurchase", "repurchase programme"], ["repurchase"]
    if "revenue growth" in low and "guide" in low:
        return ["revenue growth", "guidance", "guide"], ["outlook"]
    known_metrics = ["gross profit", "operating income", "net income", "total assets", "total equity", "inventories", "revenue", "long-term debt"]
    for t in known_metrics:
        if t in low:
            return [t], ["financial statements"]
    if "headquartered" in low or "headquarters" in low:
        return ["headquarters", "headquartered"], ["headquarters"]
    # Generic non-financial metrics such as "spend on X" must have source evidence.
    m = re.search(r"\b(?:spend|spent)\s+on\s+(.+?)\s+in\s+FY\s*20\d\d", q, re.I)
    if m:
        return [_normal(m.group(1)).lower()], []
    # Generic unseen metrics: prefer terms around 'X's <metric> in/at ...'
    m = re.search(r"['’]s\s+(.+?)(?:\s+at\b|\s+in\b|\s+for\b|\?|,|$)", q)
    if m:
        txt = re.sub(r"\b(?:what|were|was|how|much|value|the|its|of)\b", " ", m.group(1), flags=re.I)
        txt = _normal(txt).lower()
        if txt:
            return [txt], []
    return [], []


def _question_details(question, companies):
    # Extract a segment or region qualifier to avoid generic revenue lines.
    q = _normal(question)
    qualifier = ""
    m = re.search(r"['’]s\s+(.+?)\s+segment\b", q, re.I)
    if m:
        qualifier = m.group(1)
    elif re.search(r"\brevenue\b", q, re.I):
        m = re.search(r"\b(?:in|from)\s+([A-Z][A-Za-z,\s]+?)\s+in\s+FY\s*20\d\d", q)
        if m:
            qualifier = m.group(1)
        else:
            m = re.search(r"\bfrom\s+(?:its|their|the)\s+(.+?)\s+in\s+FY\s*20\d\d", q, re.I)
            if m:
                qualifier = m.group(1)
    return qualifier.lower().strip()


def _bm_score(index, ix, query_terms, use_raw=False):
    stat = index["bm25_raw"] if use_raw else index["bm25"]
    tf = stat["freq"][ix]
    n = stat["lengths"][ix]
    avg = max(stat["mean"], 1)
    score = 0.
    for w,c in Counter(query_terms).items():
        f = tf.get(w, 0)
        if f:
            idf = stat["idf"].get(w, 0.0)
            score += idf * f*2.5/(f + 1.5*(.25+.75*n/avg)) * min(c, 2)
    return score


def _score_passage(index, i, q, query, names, metric, qualifier):
    ch = index["chunks"][i]
    t = ch.page_content.lower()
    section = ch.metadata.get("section", "").lower()
    # Pure BM25 is the baseline; enhancements are individually controlled.
    if PIPELINE_FLAGS["tokeniser"]:
        words = [w for w in _tokens(q) if w not in _STOP and not w.isdigit() and len(w)>2]
        for n in names:
            for w in _tokens(n):
                if w in words:
                    words.remove(w)
    else:
        words = q.lower().split()
    sc = _bm_score(index, i, words, use_raw=not PIPELINE_FLAGS["tokeniser"])
    if not PIPELINE_FLAGS["metric_boost"]:
        return sc
    phrases, sections = metric
    if phrases:
        bonuses = [12.0/(j+1) for j,p in enumerate(phrases) if p in t]
        if bonuses:
            sc += max(bonuses)
        else:
            sc -= 12.0
    if any(s in section or s in t[:140] for s in sections):
        sc += 4.0
    if qualifier:
        if qualifier in t:
            sc += 12.0
        else:
            sc -= 10.0
    year = query.get("fiscal_year")
    if year is not None and str(year) in t:
        sc += 2.0
    if query.get("doc_type") == "research_report" and year and year>=2026:
        if f"fy{year}" in t.replace(" ", "") or str(year) in t:
            sc += 4.0
    if "auditor" in q.lower() and "independent auditor" in t:
        sc += 6.0
    if "chief financial officer" in q.lower() and "executive officers" in t:
        sc += 8.0
    return sc


def _eligible_year(ch, q_year, doc_type):
    fy = ch.metadata.get("fiscal_year")
    if not isinstance(fy, int):
        return True
    if q_year is None:
        return True
    if doc_type == "annual_report":
        # Annual statements have two adjacent calendar-year columns.
        return fy == q_year or fy == q_year + 1
    if doc_type == "earnings_call":
        # The earnings call reviews FY2025 and may guide FY2026.
        return q_year in {fy, fy+1}
    if doc_type == "research_report":
        # A FY2025 report contains FY2024A ... FY2027E estimates.
        return fy-1 <= q_year <= fy+2
    return True


def _has_support(chunks, question, metric, qualifier):
    """Conservative, data-derived safeguard for unsupported metrics (Task 3)."""
    if not chunks:
        return False
    phrases, _ = metric
    contents = [c.page_content.lower() for c in chunks]
    if not phrases:
        return True
    if not any(any(p in c for p in phrases) for c in contents):
        return False
    if qualifier and not any(qualifier in c for c in contents):
        return False
    # Unknown metric phrases must actually appear, not merely share 'revenue'.
    return True


def retrieve(index: Any, question: str) -> list[Document]:
    chunks = index["chunks"]
    if not chunks:
        return []
    info = parse_query(question)
    names = info["companies"]
    qyear = info["fiscal_year"]
    dtype = info["doc_type"]
    comparing = len(names)>=2
    metric = _metric_info(question)
    qualifier = _question_details(question, names)
    if not PIPELINE_FLAGS["metadata_filter"]:
        candidates = list(range(len(chunks)))
    else:
        # Exact company match prevents false hits on similar names and unknown entities.
        by_company = index["groups"]
        if names and any(_company_key(n) not in by_company for n in names):
            return []
        if qyear is not None and qyear < 2023 and dtype == "annual_report":
            return []
        candidates = [i for i,ch in enumerate(chunks)
                      if (not names or _company_key(ch.metadata.get("company", "")) in {_company_key(n) for n in names})
                      and ch.metadata.get("doc_type") == dtype
                      and _eligible_year(ch,qyear,dtype)]
    if not candidates:
        return []
    scored = [(float(_score_passage(index, i, question, info, names, metric, qualifier)), i) for i in candidates]
    # Prefer the original year's report over a FY2025 prior-year column, especially for CFOs/auditors.
    if PIPELINE_FLAGS["metadata_filter"] and dtype=="annual_report" and qyear is not None:
        scored = [(s+1.5 if chunks[i].metadata.get("fiscal_year")==qyear else s, i) for s,i in scored]
    scored.sort(key=lambda pair:(pair[0], -pair[1]), reverse=True)
    chosen = []
    if PIPELINE_FLAGS["balanced_comparison"] and comparing and PIPELINE_FLAGS["metadata_filter"]:
        # Allocate evidence per named company, preventing all five results from one firm.
        for name in names[:2]:
            ck = _company_key(name)
            pool = [(s,i) for s,i in scored if _company_key(chunks[i].metadata.get("company", ""))==ck]
            chosen.extend(i for s,i in pool[:2])
    for _, i in scored:
        if i not in chosen:
            chosen.append(i)
        if len(chosen)>=5:
            break
    final = [chunks[i] for i in chosen[:5]]
    if PIPELINE_FLAGS["abstention"]:
        if not _has_support(final, question, metric, qualifier):
            return []
        # Exclude weak matches for arbitrary undisclosed measures: require a metric label.
        if scored and scored[0][0] < 0.5:
            return []
    return final
# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
