import pytest

from consultant.recommenders.basket import BasketRecommender, category, evaluate_checkout_simulation, sku

CAM, NVR, PANEL, SMOKE, KEY = ("cam (SKU: C1)", "nvr (SKU: N1)", "panel (SKU: P1)", "smoke (SKU: S1)",
                               "keypad (SKU: K1)")
# {cam, panel} -> keypad in 3/3 orders; cam alone -> keypad in 3/5; smoke never with keypad.
TX = [frozenset(t) for t in [{CAM, PANEL, KEY}] * 3 + [{CAM, NVR}] * 2 + [{SMOKE, NVR}] * 2 + [{SMOKE, PANEL}]]


def model(**kw) -> BasketRecommender:
    return BasketRecommender(TX, **({"min_support": 0.2} | kw))


def test_rule_metrics_match_definitions():
    rec = model()
    r = rec.by_antecedent[frozenset({CAM})]
    r = r[r["consequent"] == KEY].iloc[0]
    assert r["support"] == pytest.approx(3 / 8)
    assert r["confidence"] == pytest.approx(3 / 5)               # P(keypad | cam)
    assert r["lift"] == pytest.approx((3 / 5) / (3 / 8))          # conf / P(keypad)


def test_recommendations_update_with_cart_combination():
    rec = model()
    first = rec.recommend_for_cart([CAM])
    assert first["status"] == "ok" and first["combination_orders"] == 5
    second = rec.recommend_for_cart([CAM, PANEL])
    assert second["status"] == "ok" and second["combination_orders"] == 3
    top = second["recommendations"][0]
    assert top["item"] == KEY and top["confidence"] == pytest.approx(1.0)
    assert top["antecedents"] == ["C1", "P1"]  # rule antecedent is the whole cart
    assert all(r["item"] not in (CAM, PANEL) for r in second["recommendations"])


def test_unseen_combination_abstains():
    rec = model()
    out = rec.recommend_for_cart([CAM, SMOKE])  # never bought together
    assert out["status"] == "no_pattern" and out["recommendations"] == []
    assert "never been bought together" in out["message"]


def test_rare_combination_below_support_abstains():
    rec = model()  # min_support 0.2 of 8 orders -> combination must appear in >= 2 orders
    out = rec.recommend_for_cart([SMOKE, PANEL])  # appears in exactly 1 order
    assert out["status"] == "no_pattern" and out["combination_orders"] == 1


def test_only_positively_associated_items_shown():
    rec = model()
    recs = rec.recommend_for_cart([NVR], k=10)["recommendations"]
    assert recs and all(r["lift"] > 1.0 for r in recs)


def test_empty_unknown_and_sku_resolution():
    rec = model()
    assert rec.recommend_for_cart([])["status"] == "empty_cart"
    unk = rec.recommend_for_cart([CAM, "Laptop X"])
    assert unk["status"] == "unknown_items" and unk["unknown"] == ["Laptop X"] and unk["recommendations"] == []
    assert rec.recommend_for_cart(["C1", "p1"])["recommendations"][0]["sku"] == "K1"  # SKUs, case-insensitive


def test_checkout_simulation_reports_coverage_by_cart_size():
    res = evaluate_checkout_simulation(TX * 5, k=3, test_frac=0.25, min_support=0.05)
    assert set(res) >= {"apriori", "apriori_by_cart_size", "popularity_baseline"}
    assert 0 <= res["apriori"]["coverage"] <= 1 and res["apriori_by_cart_size"]["1"]["steps"] > 0


def test_helpers():
    assert sku("Bosch B5512 Control Panel (SKU: B5512)") == "B5512"
    assert category("Hanwha XRN-2010 NVR (SKU: XRN-2010)") == "recorder"
    assert category("Honeywell 5800CO Carbon Monoxide Detector (SKU: 5800CO)") == "co_detector"


def test_explain_gives_counts_rule_examples_and_reasons():
    rec = model()
    e = rec.explain([CAM, PANEL], KEY)
    assert (e["orders_with_cart"], e["orders_with_cart_and_item"], e["orders_with_item"]) == (3, 3, 3)
    assert e["confidence"] == 1.0 and e["base_rate"] == pytest.approx(3 / 8)
    assert e["lift"] == pytest.approx(1.0 / (3 / 8), abs=1e-3)
    assert e["rule"] == "{C1, P1} -> K1" and e["evidence_strength"] == "thin"
    assert len(e["example_orders"]) == 3 and all({"C1", "P1", "K1"} <= set(o["items"]) for o in e["example_orders"])
    assert "arm and disarm" in e["pairing_reason"]  # panel -> keypad product knowledge
    # explanation must agree with what recommend_for_cart reported
    top = rec.recommend_for_cart([CAM, PANEL])["recommendations"][0]
    assert top["orders"] == e["orders_with_cart_and_item"] and top["lift"] == pytest.approx(e["lift"], abs=1e-3)


def test_brand_note_only_when_brands_differ():
    from consultant.recommenders.basket import brand_note
    assert brand_note(["Bosch B5512 Control Panel (SKU: B5512)"], "Honeywell 6160 Keypad (SKU: 6160)").startswith("Honeywell")
    assert brand_note(["Bosch B5512 Control Panel (SKU: B5512)"], "Bosch D7050 Detector (SKU: D7050)") == ""
