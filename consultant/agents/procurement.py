"""Procurement agent: market-basket recommendations over historical security/fire-system orders."""
from __future__ import annotations

from ..recommenders.model_store import production_model
from .base import DomainAgent, as_json, tool


class ProcurementAgent(DomainAgent):
    domain = "procurement"
    prefix = "PRC"
    role = """\
Domain: procurement and cross-sell for security / fire-safety products (Honeywell, Bosch,
Hanwha, Pelco, GE Interlogix, DSC, Dahua, Hikvision). Knowledge base: 1,000 historical orders.
Recommendations come from the checkout recommender: Apriori association rules (mlxtend) whose
antecedent is exactly the current basket ("customers who bought this exact combination also
bought X"), keeping only positively associated items (lift > 1). Report confidence, lift and how
many orders back each rule; few orders or lift near 1 means weak evidence - say so. If the tool
returns status no_pattern or unknown_items, the combination is not in the order history: say that
no data-backed add-on exists rather than suggesting items. You may look at sub-combinations
(e.g. each item alone) and say clearly that those are per-item suggestions. Translate results into
procurement guidance (bundles, stocking, compatibility checks)."""

    tools = [
        tool("find_products", "Find catalog items by name, brand, type or SKU.",
             {"query": {"type": "string"}}, ["query"]),
        tool("recommend_bundle", "Checkout recommender: items to add given the exact combination already in "
                                 "the order, or a no_pattern/unknown_items status when history has no support.",
             {"items": {"type": "array", "items": {"type": "string"}, "description": "Exact item names."},
              "category": {"type": "string", "description": "Optional: camera, recorder, control_panel, keypad, "
                                                             "smoke_detector, co_detector, motion_detector, "
                                                             "glassbreak_detector, sensor"}},
             ["items"]),
        tool("top_association_rules", "Strongest Apriori association rules (by lift) across all orders.",
             {"n": {"type": "integer"}}, []),
    ]

    def _register(self, rows: list[dict], kind: str) -> None:
        for r in rows:
            name = r.get("item") or f"{{{', '.join(r['antecedents'])}}} -> {r['consequent']}"
            ev = self.register_record(
                title=f"Order history {kind}: {name}", doc_id=name[:60], source="Retail market-basket CSV",
                score=min(1.0, r.get("lift", r.get("score", 0)) / 2),
                text=", ".join(f"{k}={v}" for k, v in r.items() if k != "ref"))
            r["ref"] = ev.ref

    def handle(self, name: str, args: dict) -> str:
        rec, meta = production_model()
        if name == "find_products":
            return as_json(rec.find_items(args["query"], k=8))
        if name == "recommend_bundle":
            items = [i for q in args["items"] for i in (rec.find_items(q, 1) if q not in rec.item_count else [q])]
            res = rec.recommend_for_cart(items, k=6, category_filter=args.get("category"))
            self._register(res["recommendations"], "recommendation")
            return as_json(res | {"model_version": meta["version"]})
        if name == "top_association_rules":
            rows = rec.top_rules(int(args.get("n", 10)))
            self._register(rows, "rule")
            return as_json(rows)
        return super().handle(name, args)
