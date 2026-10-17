"""Turn raw sources into ``Document`` objects, grouped by agent domain."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pandas as pd
from bs4 import BeautifulSoup

from .. import config
from .chunking import Document
from .fetch_sources import html_to_text


def _meta(path: Path) -> dict:
    m = path.with_suffix(".meta.json")
    return json.loads(m.read_text()) if m.exists() else {}


def _pretty(stem: str) -> str:
    return re.sub(r"[_\-]+", " ", stem).strip()


# --- PDFs --------------------------------------------------------------------
def is_readable(text: str, min_ratio: float = 0.6) -> bool:
    """Reject pages whose fonts decode to symbol soup (some PDFs embed custom encodings)."""
    words = text.split()
    wordlike = sum(1 for w in words if sum(ch.isalnum() for ch in w) >= max(1, len(w) // 2))
    return bool(words) and wordlike / len(words) >= min_ratio


def load_pdf(path: Path, domain: str, doc_type: str, title: str | None = None,
             source: str | None = None, extra: dict | None = None) -> Iterator[Document]:
    """One Document per page so citations can point at page numbers."""
    with pymupdf.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if len(text.split()) < 20 or not is_readable(text):
                continue
            yield Document(
                doc_id=path.stem, title=title or _pretty(path.stem), source=source or str(path),
                domain=domain, doc_type=doc_type, text=text, page=i, extra=extra or {},
            )


# --- Web pages -----------------------------------------------------------------
def load_web_page(path: Path, domain: str, doc_type: str = "web") -> Document | None:
    meta = _meta(path)
    page_title, text = html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    if len(text.split()) < 20:
        return None
    return Document(doc_id=path.stem, title=meta.get("title") or page_title, source=meta.get("url", str(path)),
                    domain=domain, doc_type=doc_type, text=text)


# --- SEC 10-K ----------------------------------------------------------------
ITEM_ORDER = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
              "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"]
ITEM_NAMES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety", "5": "Market for Equity",
    "6": "Reserved", "7": "MD&A", "7A": "Market Risk", "8": "Financial Statements",
    "9": "Accountant Changes", "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Foreign Jurisdictions", "10": "Directors and Governance", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Relationships", "14": "Accountant Fees", "15": "Exhibits",
    "16": "10-K Summary",
}
_ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\b\.?", re.I)


def tenk_lines(html: str) -> list[str]:
    """Readable lines from 10-K HTML; table rows are flattened to 'a | b | c'."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    for t in soup.find_all(style=re.compile(r"display:\s*none")):
        t.decompose()
    for tr in soup.find_all("tr"):
        cells = [" ".join(td.get_text(" ").split()) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", "%", ")"}]
        tr.replace_with(soup.new_string("\n" + " | ".join(cells) + "\n"))
    lines = [" ".join(ln.split()) for ln in soup.get_text("\n").splitlines()]
    return [ln for ln in lines if ln]


def find_item_starts(lines: list[str]) -> list[tuple[int, str]]:
    """Line indices where each 10-K Item begins, skipping the TOC and backward cross-references.

    The TOC lists every item within a few lines; real section headers appear in
    order afterwards. We start after the first 'Item 16' (end of TOC) when present
    and only accept headers that move forward in ITEM_ORDER.
    """
    heads = []
    for i, ln in enumerate(lines):
        m = _ITEM_RE.match(ln)
        if m and len(ln) < 140 and m.group(1).upper() in ITEM_ORDER:
            heads.append((i, m.group(1).upper()))
    toc_end = next((i for i, k in heads if k == "16"), -1)
    if toc_end >= 0 and any(i > toc_end and k == "1" for i, k in heads):
        heads = [(i, k) for i, k in heads if i > toc_end]
    starts, last = [], -1
    for i, k in heads:
        pos = ITEM_ORDER.index(k)
        if pos > last:
            starts.append((i, k))
            last = pos
    return starts


def split_10k_items(lines: list[str]) -> dict[str, str]:
    starts = find_item_starts(lines)
    sections = {}
    for n, (i, k) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        sections[k] = "\n".join(lines[i:end])
    return sections


def section_name(key: str) -> str:
    return f"Item {key} {ITEM_NAMES.get(key, '')}".strip()


def load_10k(path: Path) -> Iterator[Document]:
    """EDGAR HTML 10-K -> one Document per Item."""
    meta = _meta(path)
    company = meta.get("company", path.stem.split("-")[0])
    sections = split_10k_items(tenk_lines(path.read_text(encoding="utf-8", errors="ignore")))
    for key, text in sections.items():
        yield Document(
            doc_id=path.stem, title=meta.get("title", path.stem), source=meta.get("url", str(path)),
            domain="sec", doc_type="10-K", text=text, section=section_name(key),
            extra={"company": company, "item": key, "filing_date": meta.get("filing_date", ""),
                   "fiscal_year": int(meta.get("report_date", "0")[:4] or 0)},
        )


# --- SEC filings supplied as PDFs (UnitedStatesSecuritiesandExchangeCommission(SEC)/<Company>/*.pdf) ---
_FY_END_RE = re.compile(r"for the (fiscal year|quarterly period) ended\s+([A-Za-z]+\s+\d{1,2}\s*,\s*(\d{4}))", re.I)
_FORM_RE = re.compile(r"\bFORM\s+(10-K|10-Q|20-F)\b", re.I)
_PAGE_NOISE_RE = re.compile(r"^(table of contents|\d{1,3}|page \d+( of \d+)?)$", re.I)


def sec_pdf_meta(path: Path) -> dict:
    """Company comes from the folder name; form type and period end from the cover page."""
    with pymupdf.open(path) as pdf:
        cover = " ".join(" ".join(pdf[i].get_text().split()) for i in range(min(2, len(pdf))))
        pages = len(pdf)
    fy = _FY_END_RE.search(cover)
    form = _FORM_RE.search(cover)
    company = path.parent.name
    form_type = form.group(1).upper() if form else "10-K"
    fiscal_year = int(fy.group(3)) if fy else 0
    return {"company": company.lower(), "company_name": company, "form": form_type,
            "period_end": fy.group(2) if fy else "", "fiscal_year": fiscal_year, "pages": pages,
            "title": f"{company} Form {form_type} FY{fiscal_year}" if fiscal_year else f"{company} Form {form_type}"}


def load_10k_pdf(path: Path) -> Iterator[Document]:
    """10-K PDF -> one Document per (Item, page) so citations carry both section and page."""
    meta = sec_pdf_meta(path)
    page_lines: list[tuple[int, str]] = []
    with pymupdf.open(path) as pdf:
        for pno, page in enumerate(pdf, start=1):
            for ln in page.get_text("text").splitlines():
                ln = " ".join(ln.split())
                if ln and not _PAGE_NOISE_RE.match(ln):
                    page_lines.append((pno, ln))
    lines = [ln for _, ln in page_lines]
    starts = find_item_starts(lines)
    # Lines before Item 1 are the cover page and table of contents.
    bounds = [(0, "Cover")] + starts
    for n, (start, key) in enumerate(bounds):
        end = bounds[n + 1][0] if n + 1 < len(bounds) else len(lines)
        by_page: dict[int, list[str]] = {}
        for pno, ln in page_lines[start:end]:
            by_page.setdefault(pno, []).append(ln)
        for pno, plines in by_page.items():
            text = "\n".join(plines)
            if len(text.split()) < 20 or not is_readable(text):
                continue
            yield Document(
                doc_id=f"{meta['company']}-{meta['form'].lower()}-fy{meta['fiscal_year']}", title=meta["title"],
                source=str(path.relative_to(config.ROOT)), domain="sec", doc_type=meta["form"], text=text,
                page=pno, section="Cover page" if key == "Cover" else section_name(key),
                extra={"company": meta["company"], "item": key, "fiscal_year": meta["fiscal_year"],
                       "period_end": meta["period_end"]},
            )


# --- ClinicalTrials.gov exports -----------------------------------------------
TRIAL_FILES = {
    "covid": "ctg-studies_covid.csv",
    "diabetes": "ctg-studies_diabetes.csv",
    "heart_attack": "ctg-studies_Hearattack.csv",
    "knee_injuries": "ctg-studies_KneeInjuries.csv",
}
TRIAL_COLS = ["NCT Number", "Study Title", "Study URL", "Study Status", "Brief Summary", "Study Results",
              "Conditions", "Interventions", "Primary Outcome Measures", "Sponsor", "Phases",
              "Enrollment", "Study Type", "Start Date", "Completion Date"]
# Paper folder name -> the same condition keys used for trials.
PAPER_GROUPS = {"covid": "covid", "diabetes": "diabetes", "heart_attack": "heart_attack", "kneeinjuries": "knee_injuries"}
_STATUS_RANK = {"COMPLETED": 0, "ACTIVE_NOT_RECRUITING": 1, "RECRUITING": 2, "ENROLLING_BY_INVITATION": 3}


def load_trials_frame(condition: str, cap: int | None = config.CLINICAL_TRIALS_PER_CONDITION) -> pd.DataFrame:
    """Load one condition's export, prioritising completed trials with results, newest first."""
    df = pd.read_csv(config.CLINICAL_DIR / TRIAL_FILES[condition], usecols=lambda c: c in TRIAL_COLS,
                     dtype=str, keep_default_na=False)
    df = df[df["Brief Summary"].str.len() > 40].copy()
    df["_status"] = df["Study Status"].map(_STATUS_RANK).fillna(9)
    df["_results"] = (df["Study Results"].str.upper() != "YES").astype(int)
    df["_start"] = pd.to_datetime(df["Start Date"], errors="coerce", format="mixed")
    df = df.sort_values(["_status", "_results", "_start"], ascending=[True, True, False])
    if cap:
        df = df.head(cap)
    df["condition_group"] = condition
    return df.drop(columns=["_status", "_results", "_start"]).reset_index(drop=True)


def trial_to_document(row: pd.Series) -> Document:
    def clip(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + "…"
    text = (
        f"{row['Study Title']}\n"
        f"Conditions: {row['Conditions']}\nInterventions: {clip(row['Interventions'], 400)}\n"
        f"Status: {row['Study Status']}; Phase: {row['Phases'] or 'N/A'}; Type: {row['Study Type']}; "
        f"Enrollment: {row['Enrollment']}; Sponsor: {row['Sponsor']}\n"
        f"Summary: {clip(row['Brief Summary'], 1200)}\n"
        f"Primary outcomes: {clip(row['Primary Outcome Measures'], 400)}"
    )
    return Document(
        doc_id=row["NCT Number"], title=f"{row['NCT Number']}: {clip(row['Study Title'], 120)}",
        source=row["Study URL"], domain="clinical", doc_type="clinical_trial", text=text,
        extra={"condition_group": row["condition_group"], "status": row["Study Status"],
               "phase": row["Phases"], "has_results": row["Study Results"], "enrollment": row["Enrollment"],
               "sponsor": row["Sponsor"]},
    )


# --- Domain corpora ----------------------------------------------------------
def technology_docs() -> Iterator[Document]:
    raw = config.RAW_DIR / "technology"
    for pdf in sorted(raw.glob("*.pdf")):
        meta = _meta(pdf)
        yield from load_pdf(pdf, "technology", "datasheet", title=f"{meta.get('title', _pretty(pdf.stem))} Data Sheet",
                            source=meta.get("url"), extra={"vendor": "cisco"})
    for html in sorted(raw.glob("*.html")):
        doc = load_web_page(html, "technology")
        if doc:
            doc.extra["vendor"] = "google" if "google" in doc.source else "cisco"
            yield doc


def clinical_docs(include_trials: bool = True) -> Iterator[Document]:
    for pdf in sorted(config.CLINICAL_DIR.glob("*/*.pdf")):
        group = PAPER_GROUPS.get(pdf.parent.name.lower(), pdf.parent.name.lower())
        yield from load_pdf(pdf, "clinical", "research_paper", extra={"condition_group": group})
    for html in sorted((config.RAW_DIR / "clinical").glob("*.html")):
        doc = load_web_page(html, "clinical", "product_page")
        if doc:
            doc.extra["vendor"] = "agilent"
            yield doc
    if include_trials:
        for cond in TRIAL_FILES:
            df = load_trials_frame(cond)
            for _, row in df.iterrows():
                yield trial_to_document(row)


def sec_filings() -> list[dict]:
    """All SEC filings available: local PDFs first, then EDGAR downloads not already covered.

    A filing is identified by (company, form, fiscal_year); the local PDF wins because it
    supports page-level citations.
    """
    filings, seen = [], set()
    for pdf in sorted(config.SEC_LOCAL_DIR.glob("*/*.pdf")) if config.SEC_LOCAL_DIR.exists() else []:
        m = sec_pdf_meta(pdf)
        key = (m["company"], m["form"], m["fiscal_year"])
        if key not in seen:
            seen.add(key)
            filings.append({**m, "path": pdf, "kind": "pdf"})
    for html in sorted((config.RAW_DIR / "sec").glob("*.html")):
        m = _meta(html)
        fy = int(m.get("report_date", "0")[:4] or 0)
        key = (m.get("company", html.stem.split("-")[0]), m.get("form", "10-K"), fy)
        if key not in seen:
            seen.add(key)
            filings.append({"company": key[0], "company_name": key[0].title(), "form": key[1],
                            "fiscal_year": fy, "period_end": m.get("report_date", ""), "title": m.get("title", ""),
                            "path": html, "kind": "html"})
    return sorted(filings, key=lambda f: (f["company"], f["fiscal_year"]))


def sec_docs() -> Iterator[Document]:
    for f in sec_filings():
        yield from (load_10k_pdf(f["path"]) if f["kind"] == "pdf" else load_10k(f["path"]))


def csr_docs() -> Iterator[Document]:
    for pdf in sorted(config.FOOD_DIR.glob("*.pdf")):
        year = re.search(r"(20\d\d)", pdf.stem)
        yield from load_pdf(
            pdf, "csr", "report",
            title=f"FAO The State of Food Security and Nutrition in the World {year.group(1) if year else ''}".strip(),
            extra={"year": int(year.group(1)) if year else 0},
        )


DOMAIN_LOADERS = {
    "technology": technology_docs,
    "clinical": clinical_docs,
    "sec": sec_docs,
    "csr": csr_docs,
}
