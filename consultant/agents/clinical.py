"""Clinical agent: clinical research PDFs, ClinicalTrials.gov studies, Agilent diagnostic products."""
from __future__ import annotations

from ..recommenders.instruments import recommend_instruments
from ..recommenders.trials import recommend_trials
from .base import DomainAgent, as_json, tool

CONDITIONS = ["covid", "diabetes", "heart_attack", "knee_injuries"]


class ClinicalAgent(DomainAgent):
    domain = "clinical"
    prefix = "CLN"
    role = """\
Domain: healthcare and medical diagnostics. Knowledge base: 20 peer-reviewed papers (COVID-19,
diabetes, heart attack, knee injuries - mostly ML/IoT for diagnosis and monitoring),
ClinicalTrials.gov study records for the same four conditions, and Agilent product pages
(Seahorse XF cell metabolic analysis, NanoZoomer digital pathology scanners, Artisan Link Pro
special stains). Agilent's FY2025 10-K business section describes its wider product lines.

For instrument/product questions use recommend_instruments and verify claims in the docs.
For evidence about interventions use recommend_trials and search_clinical_docs. You are not
giving medical advice to patients; you advise organizations. State regulatory status (IVD vs
research-use) only when the evidence says so."""

    tools = [
        tool("search_clinical_docs", "Hybrid search over clinical research papers, trial records and Agilent pages.",
             {"query": {"type": "string"},
              "source_type": {"type": "string", "enum": ["research_paper", "clinical_trial", "product_page", "any"]},
              "condition": {"type": "string", "enum": CONDITIONS + ["any"]}},
             ["query"]),
        tool("recommend_trials", "Rank the most relevant, highest-evidence ClinicalTrials.gov studies for a need.",
             {"need": {"type": "string"}, "condition": {"type": "string", "enum": CONDITIONS + ["any"]}},
             ["need"]),
        tool("recommend_instruments", "Match laboratory requirements to Agilent products in the catalog.",
             {"requirements": {"type": "string"}}, ["requirements"]),
        tool("search_agilent_10k", "Search Agilent's FY2025 10-K (business, products, segments).",
             {"query": {"type": "string"}}, ["query"]),
    ]

    def handle(self, name: str, args: dict) -> str:
        if name == "search_clinical_docs":
            filters = {}
            if args.get("source_type") not in (None, "any"):
                filters["doc_type"] = args["source_type"]
            if args.get("condition") not in (None, "any"):
                filters["condition_group"] = args["condition"]
            return self.search("clinical", args["query"], filters=filters)
        if name == "recommend_trials":
            cond = args.get("condition")
            rows = recommend_trials(args["need"], None if cond in (None, "any") else cond)
            for r in rows:
                ev = self.register_record(title=r["title"], doc_id=r["nct_id"], source=r["url"], score=r["score"],
                                          text=r["snippet"])
                r["ref"] = ev.ref
                r.pop("snippet")
                r.pop("chunk_id")
            return as_json(rows)
        if name == "recommend_instruments":
            res = recommend_instruments(args["requirements"])
            for p in res["products"]:
                ev = self.register_record(title=f"Agilent product: {p['name']}", doc_id=p["name"][:40],
                                          source=p["url"], score=p["score"], text=p["summary"])
                p["ref"] = ev.ref
            return as_json(res)
        if name == "search_agilent_10k":
            return self.search("sec", args["query"], filters={"company": "agilent"})
        return super().handle(name, args)
