"""Technology agent: Cisco routers (datasheets, product pages) and Google Cloud MCP/RAG docs."""
from __future__ import annotations

from ..recommenders.components import load_catalog, recommend_components
from .base import DomainAgent, as_json, tool


class TechnologyAgent(DomainAgent):
    domain = "technology"
    prefix = "TEC"
    role = """\
Domain: enterprise technology. Knowledge base: Cisco Catalyst 8300/8200/8000V, ISR 4000 and
ASR 1000 datasheets and product pages; Google Cloud pages on MCP and RAG. No IBM documents
were supplied - if IBM is asked about, state that the knowledge base has no IBM sources.

For hardware failures: identify the platform, treat each failure cause as a hypothesis,
name the diagnostics (CLI commands) that confirm or rule it out, and only then recommend
compatible replacement parts (use recommend_components and cite its CMP refs). Never claim
a component has failed without diagnostic evidence. Include preventive maintenance."""

    tools = [
        tool("search_tech_docs", "Hybrid search over Cisco/Google technical documentation.",
             {"query": {"type": "string"},
              "vendor": {"type": "string", "enum": ["cisco", "google", "any"],
                         "description": "Restrict to one vendor's documents."}},
             ["query"]),
        tool("recommend_components",
             "Rank compatible Cisco replacement parts for a symptom description. Returns fault-mode "
             "hypotheses with confirming diagnostics and preventive actions, plus candidate parts.",
             {"symptoms": {"type": "string", "description": "Symptoms and router model in plain words."},
              "series": {"type": "string", "enum": ["Catalyst 8300", "Catalyst 8200", "Catalyst 8000V",
                                                    "ISR 4000", "ASR 1000"],
                         "description": "Router series if known."},
              "component_types": {"type": "array", "items": {"type": "string"},
                                  "description": "Optional filter, e.g. power_supply, fan, interface_module, "
                                                 "transceiver, memory, storage, license."}},
             ["symptoms"]),
        tool("lookup_part", "Look up Cisco catalog rows by product ID substring (e.g. 'PWR-CC1', 'NIM-ES2').",
             {"pid": {"type": "string"}}, ["pid"]),
    ]

    def handle(self, name: str, args: dict) -> str:
        if name == "search_tech_docs":
            vendor = args.get("vendor", "any")
            filters = {"vendor": vendor} if vendor in ("cisco", "google") else None
            return self.search("technology", args["query"], filters=filters)
        if name == "recommend_components":
            res = recommend_components(args["symptoms"], series=args.get("series"),
                                       component_types=args.get("component_types") or None)
            for p in res["parts"]:
                ev = self.register_record(
                    title=f"Cisco {p['series']} catalog: {p['pid']}", doc_id=p["pid"], page=p["page"],
                    source=p["url"] or p["source"], score=p["score"],
                    text=f"{p['pid']} ({p['component_type']}): {p['description']}")
                p["ref"] = ev.ref
            return as_json(res)
        if name == "lookup_part":
            q = args["pid"].upper()
            rows = [p for p in load_catalog() if q in p["pid"]][:15]
            return as_json(rows) if rows else f"No catalog part matches '{q}'."
        return super().handle(name, args)
