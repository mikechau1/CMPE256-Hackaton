"""Checkout market-basket recommender using the Apriori algorithm (mlxtend).

Checkout flow:
    empty cart -> no recommendations
    each time an item is added -> recommendations for the *current cart combination*
Only rules whose antecedent equals the whole cart are used, i.e. "customers who bought exactly
this combination also bought X". If that combination does not occur often enough in the order
history (fewer than MIN_SUPPORT of orders), or no item is positively associated with it
(lift <= MIN_LIFT), the recommender abstains instead of showing irrelevant items.

Model:
    orders -> one-hot matrix (TransactionEncoder)
           -> frequent itemsets (apriori, min_support, max_len = largest order)
           -> association rules (association_rules) indexed by antecedent

Rule metrics:  support(A->X) = P(A and X),  confidence = P(X | A),  lift = confidence / P(X)
"""
from __future__ import annotations

import inspect
import math
import random
import re
from functools import lru_cache

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from .. import config

# Tuned on the 1,000 orders: a combination must appear in >= 2 orders (0.002) to be trusted.
# This serves every 1-item cart and ~88% of 2-item carts seen in history, while larger, rarer
# combinations correctly fall back to "no pattern".
MIN_SUPPORT = 0.002
MIN_CONFIDENCE = 0.0
MIN_LIFT = 1.1       # relevant = at least 10% more likely than average; 1.1 keeps coverage identical
                     # to 1.0 on validation and drops ~90 near-independent rules (lift 1.00-1.05)
MAX_LEN = 5          # largest order has 5 items

_SKU_RE = re.compile(r"\(SKU:\s*([^)]+)\)")

CATEGORY_KEYWORDS = [
    ("camera", ["camera", "ipc", "dome", "bullet"]),
    ("recorder", ["nvr", "dvr", "recorder"]),
    ("control_panel", ["control panel", "alarm kit", "panel"]),
    ("keypad", ["keypad", "wireless key", "keyfob"]),
    ("smoke_detector", ["smoke"]),
    ("co_detector", ["carbon monoxide", " co "]),
    ("motion_detector", ["motion", "pir", "d7050", "detector base"]),
    ("glassbreak_detector", ["glassbreak", "glass break"]),
    ("sensor", ["contact", "sensor", "door", "window"]),
]


def read_orders_csv(path) -> list[frozenset[str]]:
    df = pd.read_csv(path)
    items = df.drop(columns=["transaction_id"], errors="ignore")
    return [frozenset(v.strip() for v in row if isinstance(v, str) and v.strip())
            for row in items.itertuples(index=False)]


def load_transactions(path=config.RETAIL_CSV) -> list[frozenset[str]]:
    return read_orders_csv(path)


def sku(item: str) -> str:
    m = _SKU_RE.search(item)
    return m.group(1).strip() if m else item


def category(item: str) -> str:
    low = f" {item.lower()} "
    for cat, kws in CATEGORY_KEYWORDS:
        if any(k in low for k in kws):
            return cat
    return "other"


def _association_rules(itemsets: pd.DataFrame, n_transactions: int, min_confidence: float) -> pd.DataFrame:
    kwargs = {"metric": "confidence", "min_threshold": min_confidence}
    if "num_itemsets" in inspect.signature(association_rules).parameters:  # mlxtend >= 0.23.2
        kwargs["num_itemsets"] = n_transactions
    return association_rules(itemsets, **kwargs)


class BasketRecommender:
    def __init__(self, transactions: list[frozenset[str]], min_support: float = MIN_SUPPORT,
                 min_confidence: float = MIN_CONFIDENCE, min_lift: float = MIN_LIFT, max_len: int = MAX_LEN):
        self.n = len(transactions)
        self.params = {"min_support": min_support, "min_confidence": min_confidence, "min_lift": min_lift,
                       "max_len": max_len}
        self.min_lift = min_lift
        self.min_count = max(1, math.ceil(min_support * self.n - 1e-9))
        encoder = TransactionEncoder()
        rows = [sorted(t) for t in transactions]
        self.onehot = pd.DataFrame(encoder.fit(rows).transform(rows), columns=encoder.columns_)
        self.items: list[str] = list(encoder.columns_)
        self.item_count = self.onehot.sum().astype(int).to_dict()
        self.itemsets = apriori(self.onehot, min_support=min_support, use_colnames=True, max_len=max_len)
        cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
        if len(self.itemsets) and (self.itemsets["itemsets"].map(len) > 1).any():
            rules = _association_rules(self.itemsets, self.n, min_confidence)[cols]
        else:
            rules = pd.DataFrame(columns=cols)
        rules = rules[rules["consequents"].map(len) == 1].copy()  # recommend one item per rule
        rules["consequent"] = rules["consequents"].map(lambda s: next(iter(s)))
        rules["n_antecedents"] = rules["antecedents"].map(len)
        rules["count"] = (rules["support"] * self.n).round().astype(int)
        self.rules = rules.sort_values(["confidence", "lift"], ascending=False).reset_index(drop=True)
        # Exact-cart lookup: antecedent itemset -> its rules (already sorted by confidence, lift).
        self.by_antecedent: dict[frozenset, pd.DataFrame] = {
            ante: grp for ante, grp in self.rules.groupby("antecedents", sort=False)}
        self._by_sku = {sku(i).lower(): i for i in self.items}
        self._by_name = {i.lower(): i for i in self.items}

    # -- catalog helpers ------------------------------------------------------------
    def resolve(self, ref: str) -> str | None:
        """Map an exact item name or SKU to the catalog item name."""
        ref = ref.strip()
        return self._by_name.get(ref.lower()) or self._by_sku.get(sku(ref).lower())

    def find_items(self, query: str, k: int = 5) -> list[str]:
        """Fuzzy lookup by SKU or name tokens."""
        q = set(re.findall(r"[a-z0-9\-]+", query.lower()))
        scored = []
        for item in self.items:
            toks = set(re.findall(r"[a-z0-9\-]+", item.lower()))
            overlap = len(q & toks)
            if sku(item).lower() in query.lower():
                overlap += 5
            if overlap:
                scored.append((overlap, item))
        return [i for _, i in sorted(scored, reverse=True)[:k]]

    def combination_count(self, cart: list[str]) -> int:
        """Number of historical orders containing every item in the cart."""
        if not cart:
            return self.n
        return int(self.onehot[list(cart)].all(axis=1).sum())

    # -- checkout recommendations ---------------------------------------------------------
    def recommend_for_cart(self, cart: list[str], k: int = 5, category_filter: str | None = None) -> dict:
        """Recommendations for the current cart, or an explicit abstention status.

        status: ok | empty_cart | unknown_items | no_pattern
        """
        resolved, unknown = [], []
        for ref in cart:
            item = self.resolve(ref)
            (resolved if item else unknown).append(item or ref)
        resolved = list(dict.fromkeys(resolved))
        base = {"cart": [sku(i) for i in resolved], "recommendations": [], "combination_orders": 0}
        if unknown:
            return base | {"status": "unknown_items", "unknown": unknown,
                           "message": f"Not in the product history: {', '.join(unknown)}. No recommendations shown."}
        if not resolved:
            return base | {"status": "empty_cart", "message": "Cart is empty - add an item to get recommendations."}
        count = self.combination_count(resolved)
        base["combination_orders"] = count
        rules = self.by_antecedent.get(frozenset(resolved))
        if rules is None:
            if count == 0:
                why = "these items have never been bought together"
            elif count < self.min_count:
                why = f"this combination appears in only {count} past order(s) (need at least {self.min_count})"
            else:  # the cart is common enough, but no single item was added to it often enough
                why = (f"this combination appears in {count} past orders, but no other item was bought with it "
                       f"in at least {self.min_count} of them")
            return base | {"status": "no_pattern",
                           "message": f"No reliable pattern: {why}. No recommendations shown."}
        rules = rules[(rules["lift"] > self.min_lift) & ~rules["consequent"].isin(resolved)]
        if category_filter:
            rules = rules[rules["consequent"].map(category) == category_filter]
        if rules.empty:
            return base | {"status": "no_pattern",
                           "message": "No item is bought with this combination more often than by chance "
                                      f"(lift > {self.min_lift}). No recommendations shown."}
        recs = []
        for r in rules.head(k).itertuples(index=False):
            recs.append({
                "item": r.consequent, "sku": sku(r.consequent), "category": category(r.consequent),
                "score": round(float(r.confidence), 4), "confidence": round(float(r.confidence), 3),
                "lift": round(float(r.lift), 3), "support": round(float(r.support), 4), "orders": int(r.count),
                "antecedents": [sku(a) for a in sorted(r.antecedents)],
                "reason": f"{r.count} of {count} orders with this cart also had {sku(r.consequent)} "
                          f"(conf {r.confidence:.2f}, lift {r.lift:.2f})",
            })
        return base | {"status": "ok", "message": f"Based on {count} past orders containing this cart.",
                       "recommendations": recs}

    def recommend(self, cart: list[str], k: int = 5, category_filter: str | None = None) -> list[dict]:
        return self.recommend_for_cart(cart, k, category_filter)["recommendations"]

    def explain(self, cart: list[str], item: str, max_examples: int = 3) -> dict:
        """Plain-language reasoning behind recommending `item` for `cart`, computed from the order data."""
        cart = [c for c in dict.fromkeys(self.resolve(c) for c in cart) if c]
        item = self.resolve(item) or item
        with_cart = self.onehot[cart].all(axis=1) if cart else pd.Series(True, index=self.onehot.index)
        both = with_cart & self.onehot[item]
        n_cart, n_both, n_item = int(with_cart.sum()), int(both.sum()), int(self.item_count[item])
        conf = n_both / n_cart if n_cart else 0.0
        base = n_item / self.n
        lift = conf / base if base else 0.0
        order_ids = getattr(self, "order_ids", None) or [f"#{i + 1}" for i in range(self.n)]
        rows = both[both].index[:max_examples]
        examples = [{"order_id": order_ids[i],
                     "items": [sku(c) for c in self.items if self.onehot.at[i, c]]} for i in rows]
        strength = ("solid" if n_both >= 10 else "moderate" if n_both >= 5 else "thin")
        return {
            "cart": [sku(c) for c in cart], "item": sku(item), "item_name": item,
            "orders_with_cart": n_cart, "orders_with_cart_and_item": n_both, "orders_with_item": n_item,
            "total_orders": self.n, "confidence": round(conf, 4), "base_rate": round(base, 4),
            "lift": round(lift, 3), "support": round(n_both / self.n, 4),
            "rule": f"{{{', '.join(sku(c) for c in cart)}}} -> {sku(item)}",
            "evidence_strength": strength, "example_orders": examples,
            "pairing_reason": pairing_reason(cart, item),
            "brand_note": brand_note(cart, item),
        }


    def top_rules(self, n: int = 20, min_lift: float = 1.0) -> list[dict]:
        rules = self.rules[self.rules["lift"] >= min_lift].sort_values(["lift", "support"], ascending=False).head(n)
        return [{"antecedents": [sku(a) for a in sorted(r.antecedents)], "consequent": sku(r.consequent),
                 "support": round(float(r.support), 4), "confidence": round(float(r.confidence), 3),
                 "lift": round(float(r.lift), 3), "count": int(r.count)}
                for r in rules.itertuples(index=False)]

    def summary(self) -> dict:
        return {"n_transactions": self.n, "n_items": len(self.items), "n_itemsets": len(self.itemsets),
                "n_rules": len(self.rules), "min_combination_orders": self.min_count, **self.params}


# -- product knowledge used to explain *why* a data-driven pairing makes sense ------------------
_PAIRINGS = [  # (cart category, recommended category, reason)
    ("camera", "recorder", "Network cameras need a recorder (NVR/DVR) to store and play back footage."),
    ("recorder", "camera", "A recorder is bought to record cameras, so orders add cameras to feed it."),
    ("camera", "camera", "Multi-camera sites mix models for different coverage (indoor/outdoor, wide/zoom)."),
    ("control_panel", "keypad", "Keypads are how users arm and disarm the alarm control panel."),
    ("keypad", "control_panel", "A keypad needs an alarm control panel to operate."),
    ("smoke_detector", "co_detector", "Fire-safety installs commonly pair smoke and carbon-monoxide detection."),
    ("co_detector", "smoke_detector", "Fire-safety installs commonly pair carbon-monoxide and smoke detection."),
]
_DETECTORS = {"smoke_detector", "co_detector", "motion_detector", "glassbreak_detector", "sensor"}
_INTRUSION = {"motion_detector", "glassbreak_detector", "sensor"}


def pairing_reason(cart: list[str], item: str) -> str:
    rec_cat = category(item)
    cart_cats = [category(c) for c in cart]
    for cart_cat, cat, reason in _PAIRINGS:
        if cart_cat in cart_cats and rec_cat == cat:
            return reason
    if "control_panel" in cart_cats and rec_cat in _DETECTORS:
        return "Detectors and sensors connect to the alarm control panel as protection zones."
    if rec_cat == "control_panel" and any(c in _DETECTORS for c in cart_cats):
        return "Detectors and sensors need an alarm control panel to report to."
    if rec_cat in _INTRUSION and any(c in _INTRUSION for c in cart_cats):
        return "Intrusion systems layer perimeter (door/window, glass-break) and interior (motion) sensors."
    return ""  # no standard technical dependency - the pairing is purely what customers did


def brand_note(cart: list[str], item: str) -> str:
    brand = item.split()[0]
    others = sorted({c.split()[0] for c in cart} - {brand})
    if others:
        return (f"{brand} differs from the brand(s) in your cart ({', '.join(others)}) - "
                "confirm compatibility before ordering.")
    return ""


@lru_cache(maxsize=8)
def get_recommender(min_support: float = MIN_SUPPORT, min_confidence: float = MIN_CONFIDENCE) -> BasketRecommender:
    """Ad-hoc model fitted on the Retail CSV (for experiments); production code uses model_store."""
    return BasketRecommender(load_transactions(), min_support=min_support, min_confidence=min_confidence)


# -- validation ----------------------------------------------------------------
def evaluate_checkout_simulation(transactions: list[frozenset[str]] | None = None, k: int = 5,
                                 test_frac: float = 0.2, seed: int = 7, **params) -> dict:
    """Replay held-out orders as checkouts: add items one at a time, recommend after each add.

    For each step (cart = first t items, t < order size) we record whether recommendations were
    shown (coverage) and, when shown, whether they contain any item the customer went on to add
    (hit@k), precision@k, and whether the very next item was recommended (next_item_hit@k).
    A popularity baseline (always shows the k most popular items) is reported for reference.
    """
    tx = list(transactions) if transactions is not None else load_transactions()
    rng = random.Random(seed)
    idx = list(range(len(tx)))
    rng.shuffle(idx)
    cut = int(len(tx) * (1 - test_frac))
    train, test = [tx[i] for i in idx[:cut]], [tx[i] for i in idx[cut:] if len(tx[i]) >= 2]
    rec = BasketRecommender(train, **params)
    popular = [i for i, _ in sorted(rec.item_count.items(), key=lambda x: -x[1])]

    def blank():
        return {"steps": 0, "shown": 0, "hits": 0, "precision_sum": 0.0, "next_hits": 0}
    by_size: dict[int, dict] = {}
    overall, pop = blank(), blank()
    for order in test:
        seq = sorted(order)
        rng.shuffle(seq)  # order in which the shopper adds items
        for t in range(1, len(seq)):
            cart, remaining, nxt = seq[:t], set(seq[t:]), seq[t]
            recs = [r["item"] for r in rec.recommend(cart, k=k)]
            pop_recs = [i for i in popular if i not in cart][:k]
            for stats, shown_recs in ((overall, recs), (by_size.setdefault(t, blank()), recs), (pop, pop_recs)):
                stats["steps"] += 1
                if shown_recs:
                    good = len(remaining & set(shown_recs))
                    stats["shown"] += 1
                    stats["hits"] += good > 0
                    stats["precision_sum"] += good / len(shown_recs)
                    stats["next_hits"] += nxt in shown_recs

    def finalize(s: dict) -> dict:
        shown = max(1, s["shown"])
        return {"steps": s["steps"], "coverage": round(s["shown"] / max(1, s["steps"]), 4),
                f"hit@{k}": round(s["hits"] / shown, 4), f"precision@{k}": round(s["precision_sum"] / shown, 4),
                f"next_item_hit@{k}": round(s["next_hits"] / shown, 4)}

    return {"apriori": finalize(overall) | {"model": rec.summary()},
            "apriori_by_cart_size": {str(t): finalize(s) for t, s in sorted(by_size.items())},
            "popularity_baseline": finalize(pop),
            "random_baseline_next_item_hit": round(k / max(1, len(rec.items) - 1), 4),
            "test_orders": len(test)}
