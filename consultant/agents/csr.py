"""Corporate Responsibility agent: FAO SOFI 2023/2024 food security reports (+ tech levers)."""
from __future__ import annotations

from .base import DomainAgent, tool


class CSRAgent(DomainAgent):
    domain = "csr"
    prefix = "CSR"
    role = """\
Domain: corporate social responsibility, ESG and food security. Knowledge base: FAO "The State
of Food Security and Nutrition in the World" (SOFI) 2023 and 2024 reports. You may also search
the technology knowledge base for technology levers (networking, cloud, RAG/AI, MCP).

Tie recommendations to measurable outcomes that SOFI uses (prevalence of undernourishment,
moderate/severe food insecurity (FIES), cost and affordability of a healthy diet, stunting,
wasting) and to corporate ESG reporting. Distinguish what the reports state from your inferred
link between a technology investment and an outcome (the latter is a hypothesis)."""

    tools = [
        tool("search_food_security", "Hybrid search over the SOFI 2023/2024 reports.",
             {"query": {"type": "string"}, "year": {"type": "integer", "enum": [2023, 2024],
                                                    "description": "Restrict to one report edition."}},
             ["query"]),
        tool("search_tech_docs", "Search the technology knowledge base (Cisco/Google) for enabling technologies.",
             {"query": {"type": "string"}}, ["query"]),
    ]

    def handle(self, name: str, args: dict) -> str:
        if name == "search_food_security":
            filters = {"year": args["year"]} if args.get("year") in (2023, 2024) else None
            return self.search("csr", args["query"], filters=filters)
        if name == "search_tech_docs":
            return self.search("technology", args["query"], k=4)
        return super().handle(name, args)
