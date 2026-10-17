"""Intent detection / routing.

* ``llm_route``         - Claude (fast model) multi-label classifier with confidence (production router)
* ``EmbeddingRouter``   - nearest-centroid baseline over labelled example queries (for evaluation)
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from . import config
from .llm import LLM, get_llm
from .schemas import RouteOut

ROUTER_TEMPLATE = """\
You are the intent router of a management-consulting platform. First decide whether the request is
in scope, then choose every specialist agent it needs (multi-label, most relevant first).

KNOWLEDGE BASES (the ONLY information the platform has):
- technology: Cisco Catalyst 8300/8200/8000V, ISR 4000 and ASR 1000 routers (datasheets, product
  pages: failures, replacement parts, specs, licensing); Google Cloud pages on MCP and RAG.
- clinical: clinical research papers and ClinicalTrials.gov studies on COVID-19, diabetes, heart
  attack and knee injuries; Agilent lab/diagnostic products (Seahorse XF, NanoZoomer digital
  pathology scanners, Artisan Link Pro special stains). Advice is for organizations, not individuals.
- sec: SEC 10-K filings of {sec_companies} - financial performance, revenue, competitive,
  operational, regulatory and cyber risk, vendor-risk analysis. Google Cloud products go to
  technology; Alphabet/Google financials go to sec.
- csr: FAO State of Food Security and Nutrition reports (2023, 2024) - food security, hunger,
  nutrition, ESG / CSR and sustainability programs and how technology investments support them.
- procurement: ONLY the security / fire-alarm product order history (cameras, NVRs, detectors,
  control panels, keypads; Bosch, Hikvision, DSC, Honeywell, GE Interlogix, ...) - order bundles and
  "frequently bought together". Do NOT add procurement just because a request mentions ordering,
  buying or replacement parts: Cisco/network parts belong to technology, lab components to clinical.

SCOPE: in_scope=true if any part of the request is about these business topics, even when it is
broad or vague. in_scope=false when nothing in it is, e.g. general knowledge, trivia, sports, weather,
recipes, entertainment, writing poems or code, personal medical advice for an individual, consumer
gadgets, or companies/products outside the knowledge bases (e.g. Tesla or Apple financials, IBM
products - no IBM documents exist). Put every uncovered part of a request in out_of_scope_parts
(for a fully out-of-scope request, the whole request). When in_scope=false, still return
domains=["technology"], confidence 0 and needs_clarification=false.

Pick only domains that add real value. For in-scope requests always route to at least one domain,
even when broad or vague - the specialists answer best-effort and list what they still need. Set
needs_clarification=true when important details are missing (and phrase one clarification_question
for the customer); this does NOT stop the agents from working."""


@lru_cache(maxsize=1)
def router_system() -> str:
    """Router prompt with the SEC companies actually present in the knowledge base."""
    try:
        from .ingest.loaders import sec_filings
        names = sorted({f["company_name"] for f in sec_filings()})
    except Exception:  # knowledge base not built yet
        names = []
    if len(names) > 1:
        companies = ", ".join(names[:-1]) + " and " + names[-1]
    else:
        companies = names[0] if names else "Cisco and Agilent"
    return ROUTER_TEMPLATE.format(sec_companies=companies)


def llm_route(request: str, llm: LLM | None = None) -> RouteOut:
    llm = llm or get_llm()
    out = llm.parse(RouteOut, f"Customer request:\n{request}", system=router_system(),
                    model=config.FAST_MODEL, effort="low", max_tokens=2000)
    seen, domains = set(), []
    for d in out.domains:
        if d not in seen:
            seen.add(d)
            domains.append(d)
    # The procurement agent only knows the security/fire-alarm order history. The model tends to add it
    # whenever a request says "order" or "replacement part"; drop it from multi-domain routes that never
    # mention that product family.
    if "procurement" in domains and len(domains) > 1 and not mentions_security_products(request):
        domains.remove("procurement")
    out.domains = domains or ["technology"]
    out.confidence = max(0.0, min(1.0, out.confidence))
    return out


SECURITY_PRODUCT_TERMS = re.compile(
    r"\b(security|fire|alarm|intrusion|camera|cctv|nvr|dvr|detector|smoke|carbon monoxide|keypad|control panel|"
    r"sensor|glass ?break|bosch|hikvision|dahua|hanwha|pelco|honeywell|interlogix|dsc|frequently bought|"
    r"bought together|bundle|cross-sell)\b", re.I)


def mentions_security_products(request: str) -> bool:
    return bool(SECURITY_PRODUCT_TERMS.search(request))


class EmbeddingRouter:
    """Baseline: cosine similarity to per-domain centroids of example queries; multi-label by margin."""

    def __init__(self, examples: list[tuple[str, list[str]]], margin: float = 0.03):
        from .retrieval.hybrid import embed_queries
        self._embed = embed_queries
        vecs = embed_queries([q for q, _ in examples])
        self.centroids = {}
        for d in config.DOMAINS:
            idx = [i for i, (_, labels) in enumerate(examples) if d in labels]
            if idx:
                c = vecs[idx].mean(axis=0)
                self.centroids[d] = c / np.linalg.norm(c)
        self.margin = margin

    def route(self, request: str) -> RouteOut:
        q = self._embed([request])[0]
        sims = sorted(((float(c @ q), d) for d, c in self.centroids.items()), reverse=True)
        best = sims[0][0]
        domains = [d for s, d in sims if s >= best - self.margin]
        return RouteOut(in_scope=True, out_of_scope_parts=[], domains=domains, confidence=round(best, 3),
                        rationale="nearest centroid",
                        needs_clarification=False, clarification_question="")
