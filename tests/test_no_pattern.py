"""Requirement: when the cart combination does not exist in the order history, the recommender must
abstain gracefully instead of showing irrelevant items.

Checked against the real Retail order history with an independent brute-force oracle (bitmask
counting over the raw CSV, no mlxtend), for every 1-, 2- and 3-item cart plus sampled 4-5 item carts.
"""
import functools
import itertools
import random

import pytest
from fastapi.testclient import TestClient

from consultant.recommenders.basket import BasketRecommender, load_transactions, sku
from consultant.services.basket_api import Monitor, create_app

TX = load_transactions()
N = len(TX)
REC = BasketRecommender(TX)
ITEMS = REC.items
MASK = {i: sum(1 << n for n, t in enumerate(TX) if i in t) for i in ITEMS}
FULL = (1 << N) - 1
BY_SKU = {sku(i): i for i in ITEMS}


def count(combo) -> int:
    """Oracle: number of orders containing every item in combo."""
    return bin(functools.reduce(lambda a, b: a & b, (MASK[i] for i in combo), FULL)).count("1")


def relevant_items(cart) -> dict[str, float]:
    """Oracle: items bought with the exact cart in >= min_count orders and more often than average."""
    c = count(cart)
    if c < REC.min_count:
        return {}
    out = {}
    for x in ITEMS:
        if x in cart:
            continue
        both = count((*cart, x))
        lift = (both / c) / (REC.item_count[x] / N)
        if both >= REC.min_count and lift > REC.min_lift:
            out[x] = lift
    return out


def check_cart(cart) -> str:
    res = REC.recommend_for_cart(list(cart), k=40)
    oracle = relevant_items(cart)
    if not oracle:  # combination missing / too rare / nothing positively associated
        assert res["status"] == "no_pattern", (cart, res["status"])
        assert res["recommendations"] == []
        assert "No recommendations shown" in res["message"]
        return "abstain"
    assert res["status"] == "ok", (cart, res["message"])
    shown = {r["item"] for r in res["recommendations"]}
    assert shown == set(oracle), cart  # nothing irrelevant shown, nothing relevant missed
    for r in res["recommendations"]:
        both, c = count((*cart, r["item"])), count(cart)
        assert r["item"] not in cart
        assert r["orders"] == both and r["confidence"] == pytest.approx(both / c, abs=1e-3)
        assert r["lift"] > 1.0 and r["lift"] == pytest.approx(oracle[r["item"]], abs=1e-3)
    return "ok"


@pytest.mark.parametrize("size", [1, 2, 3])
def test_every_cart_of_size_matches_oracle(size):
    outcomes = {"ok": 0, "abstain": 0}
    for cart in itertools.combinations(ITEMS, size):
        outcomes[check_cart(cart)] += 1
    assert outcomes["abstain"] > 0 or size == 1  # the abstention path is genuinely exercised
    print(f"size {size}: {outcomes}")


def test_large_carts_from_orders_and_random_samples():
    rng = random.Random(0)
    carts = [tuple(sorted(t))[:k] for t in TX for k in (4, 5) if len(t) >= k]
    carts += [tuple(rng.sample(ITEMS, k)) for k in (4, 5) for _ in range(300)]
    outcomes = [check_cart(c) for c in carts]
    assert "abstain" in outcomes


def test_named_cases_explain_why():
    never = REC.recommend_for_cart([BY_SKU["XRN-2010"], BY_SKU["IMP1110-1ES"]])
    assert never["status"] == "no_pattern" and never["combination_orders"] == 0
    assert "never been bought together" in never["message"]

    once = REC.recommend_for_cart([BY_SKU["0806-001"], BY_SKU["6160"]])
    assert once["combination_orders"] == 1 and "only 1 past order" in once["message"]

    # Common pair, but no third item was ever added to it in >= 2 orders.
    pair = REC.recommend_for_cart([BY_SKU["0806-001"], BY_SKU["D7050"]])
    assert pair["status"] == "no_pattern" and pair["combination_orders"] >= REC.min_count
    assert "no other item was bought with it" in pair["message"]
    for res in (never, once, pair):
        assert res["recommendations"] == []


def test_unknown_items_never_get_recommendations():
    for cart in (["Unknown Gadget"], [BY_SKU["B5512"], "Unknown Gadget"]):
        res = REC.recommend_for_cart(cart)
        assert res["status"] == "unknown_items" and res["recommendations"] == []


def test_checkout_sequence_recommendations_appear_disappear_and_return():
    cart = [BY_SKU["B5512"]]
    assert REC.recommend_for_cart(cart)["status"] == "ok"
    cart.append(BY_SKU["6160"])
    assert REC.recommend_for_cart(cart)["status"] == "ok"
    cart.append(BY_SKU["WS4916"])  # this combination never occurred
    res = REC.recommend_for_cart(cart)
    assert res["status"] == "no_pattern" and res["recommendations"] == []
    cart.remove(BY_SKU["WS4916"])  # back to a known combination
    assert REC.recommend_for_cart(cart)["status"] == "ok"


def test_api_returns_empty_recommendations_for_unseen_combination(tmp_path):
    app = create_app(model=REC, metadata={"version": "real-data"}, monitor=Monitor(log_path=tmp_path / "l.jsonl"))
    with TestClient(app) as c:
        r = c.post("/recommend", json={"cart": ["XRN-2010", "IMP1110-1ES"]})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "no_pattern" and body["recommendations"] == []
        assert c.get("/metrics").json()["serving"]["abstention_rate"] == 1.0
