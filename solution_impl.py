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