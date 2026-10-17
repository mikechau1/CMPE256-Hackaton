"""SEC agent: Form 10-K filings (local PDFs in the SEC folder, plus EDGAR downloads)."""
from __future__ import annotations

from functools import lru_cache

from ..ingest.loaders import sec_filings
from .base import DomainAgent, tool

ITEMS = {"business": "1", "risk_factors": "1A", "cybersecurity": "1C", "legal": "3", "mdna": "7",
         "market_risk": "7A", "financial_statements": "8", "controls": "9A"}


@lru_cache(maxsize=1)
def available_filings() -> list[dict]:
    return [{k: f[k] for k in ("company", "company_name", "form", "fiscal_year", "period_end")} for f in sec_filings()]


def companies() -> list[str]:
    return sorted({f["company"] for f in available_filings()}) or ["cisco", "agilent"]


def fiscal_years() -> list[int]:
    return sorted({f["fiscal_year"] for f in available_filings() if f["fiscal_year"]})


def filings_summary() -> str:
    lines = [f"- {f['company_name']} Form {f['form']} FY{f['fiscal_year']} (period ended {f['period_end']})"
             for f in available_filings()]
    return "\n".join(lines) or "- (no filings indexed)"


class SECAgent(DomainAgent):
    domain = "sec"
    prefix = "SEC"
    role = f"""\
Domain: securities filings and regulatory/financial risk. Knowledge base (split by Item: Business,
Risk Factors, MD&A, Market Risk, Financial Statements, ...; cited by page):
{filings_summary()}

Fiscal years differ by company (e.g. Cisco's ends in late July, Agilent's on October 31,
Alphabet's on December 31). When the question is about the "latest" or "current" position, use
each company's most recent fiscal year and say which one; for trends, compare fiscal years.
Quote figures exactly as filed with their period. Separate financial, competitive, operational,
regulatory and cyber risks. Risk factors are management disclosures, not predictions. Do not give
investment advice; frame findings as business/vendor-risk insight. If a company or year is not
in the knowledge base, say so."""

    tools = [
        tool("search_filings", "Hybrid search over the SEC filings, optionally by company, fiscal year and section.",
             {"query": {"type": "string"},
              "company": {"type": "string", "enum": companies() + ["any"]},
              "fiscal_year": {"type": "integer", "enum": fiscal_years(),
                              "description": "Restrict to one fiscal year's filing."},
              "section": {"type": "string", "enum": list(ITEMS) + ["any"]}},
             ["query"]),
        tool("compare_companies", "Retrieve the same topic from several companies' filings side by side.",
             {"topic": {"type": "string"},
              "companies": {"type": "array", "items": {"type": "string", "enum": companies()},
                            "description": "Defaults to all companies."},
              "fiscal_year": {"type": "integer", "enum": fiscal_years(),
                              "description": "Optional; default is each company's latest filing."},
              "section": {"type": "string", "enum": list(ITEMS) + ["any"]}},
             ["topic"]),
        tool("list_filings", "List the filings available in the knowledge base.", {}, []),
    ]

    @staticmethod
    def _filters(company: str | None, section: str | None, fiscal_year: int | None = None) -> dict:
        f: dict = {}
        if company in companies():
            f["company"] = company
        if section in ITEMS:
            f["item"] = ITEMS[section]
        if fiscal_year in fiscal_years():
            f["fiscal_year"] = fiscal_year
        return f

    @staticmethod
    def _latest_year(company: str) -> int | None:
        years = [f["fiscal_year"] for f in available_filings() if f["company"] == company]
        return max(years) if years else None

    def handle(self, name: str, args: dict) -> str:
        if name == "search_filings":
            return self.search("sec", args["query"], filters=self._filters(
                args.get("company"), args.get("section"), args.get("fiscal_year")))
        if name == "compare_companies":
            parts = []
            for company in args.get("companies") or companies():
                fy = args.get("fiscal_year") or self._latest_year(company)
                filters = self._filters(company, args.get("section"), fy)
                if "fiscal_year" not in filters:  # requested year not filed by this company
                    filters = self._filters(company, args.get("section"), self._latest_year(company))
                parts.append(f"=== {company.upper()} FY{filters.get('fiscal_year', '?')} ===\n"
                             + self.search("sec", args["topic"], k=4, filters=filters))
            return "\n\n".join(parts)
        if name == "list_filings":
            return filings_summary()
        return super().handle(name, args)
