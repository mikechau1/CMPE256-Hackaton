"""Checkout microservice: API contract, monitoring, order history and validation-gated retraining."""
import json

from fastapi.testclient import TestClient

from consultant.recommenders import model_store
from consultant.services.basket_api import Monitor, create_app
from tests.test_basket import CAM, KEY, PANEL, SMOKE, TX, model


def client(tmp_path, trainer=None):
    orders_csv = tmp_path / "placed_orders.csv"
    app = create_app(model=model(), metadata={"version": "vtest", "validation": {"apriori": {"hit@5": 0.3}}},
                     trainer=trainer, order_sink=lambda items: model_store.append_order(items, orders_csv),
                     order_source=lambda: model_store.list_orders(orders_csv),
                     order_deleter=lambda oid: model_store.delete_order(oid, orders_csv),
                     order_clearer=lambda: model_store.clear_orders(orders_csv),
                     monitor=Monitor(log_path=tmp_path / "log.jsonl"))
    return TestClient(app), orders_csv


def test_recommend_flow_and_monitoring(tmp_path):
    c, _ = client(tmp_path)
    with c:
        assert c.get("/health").json()["model_version"] == "vtest"
        assert c.post("/recommend", json={"cart": [], "session_id": "s"}).json()["status"] == "empty_cart"
        r1 = c.post("/recommend", json={"cart": ["C1"], "session_id": "s", "k": 3}).json()
        assert r1["status"] == "ok" and r1["model_version"] == "vtest"
        r2 = c.post("/recommend", json={"cart": ["C1", "K1"], "session_id": "s"}).json()  # accepted a rec
        assert r2["cart"] == ["C1", "K1"]
        r3 = c.post("/recommend", json={"cart": [CAM, SMOKE]}).json()
        assert r3["status"] == "no_pattern" and r3["recommendations"] == []
        m = c.get("/metrics").json()["serving"]
    assert m["requests"] == 4 and m["status_counts"]["no_pattern"] == 1
    assert m["uptake_rate"] == 1.0  # K1 was recommended after C1, then added
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[1])["status"] == "ok"


def test_validation_errors(tmp_path):
    c, _ = client(tmp_path)
    with c:
        assert c.post("/recommend", json={"cart": ["C1"], "k": 0}).status_code == 422
        assert c.post("/orders", json={"items": []}).status_code == 422


def test_orders_are_recorded_as_history_not_training(tmp_path):
    c, _ = client(tmp_path)
    with c:
        before = c.post("/recommend", json={"cart": ["C1"]}).json()
        out = c.post("/orders", json={"items": ["C1", "P1", "K1"]}).json()
        assert out["stored"] and out["used_for_training"] is False
        assert out["order"]["items"] == [CAM, PANEL, KEY] and out["order"]["order_id"].startswith("ORD-")
        c.post("/orders", json={"items": ["S1"]})
        history = c.get("/orders").json()
        assert [o["items"] for o in history] == [[SMOKE], [CAM, PANEL, KEY]]  # newest first
        after = c.post("/recommend", json={"cart": ["C1"]}).json()
    assert after["recommendations"] == before["recommendations"]  # placing orders changes nothing


def test_retrain_hot_swaps_only_promoted_models(tmp_path):
    calls = []

    def trainer(force):
        calls.append(force)
        promoted = len(calls) == 1
        return {"version": f"v{len(calls)}", "promoted": promoted, "_model": model()}

    c, _ = client(tmp_path, trainer)
    with c:
        assert c.post("/retrain", json={}).json()["promoted"] is True
        assert c.get("/health").json()["model_version"] == "v1"
        assert c.post("/retrain", json={}).json()["promoted"] is False
        assert c.get("/health").json()["model_version"] == "v1"  # rejected model is not served


def test_model_store_train_gate_and_order_append(tmp_path):
    first = model_store.train(model_dir=tmp_path, transactions=TX * 10, min_support=0.05)
    assert first["promoted"] and model_store.latest_metadata(tmp_path)["version"] == first["version"]
    rec, meta = model_store.load_latest(tmp_path)
    assert meta["version"] == first["version"] and rec.recommend_for_cart(["C1"])["status"] == "ok"
    csv_path = tmp_path / "placed_orders.csv"
    first_order = model_store.append_order([CAM, KEY], csv_path)
    model_store.append_order([PANEL], csv_path)
    history = model_store.list_orders(csv_path)
    assert [o["items"] for o in history] == [[PANEL], [CAM, KEY]] and history[1] == first_order


def test_training_ignores_placed_orders(tmp_path, monkeypatch):
    orders_csv = tmp_path / "placed_orders.csv"
    monkeypatch.setattr(model_store, "PLACED_ORDERS_CSV", orders_csv)
    for _ in range(50):  # a burst of identical orders that would create strong new rules if trained on
        model_store.append_order([CAM, SMOKE, KEY], orders_csv)
    meta = model_store.train(model_dir=tmp_path / "models")
    assert meta["model"]["n_transactions"] == 1000  # only the original Retail orders


def test_delete_orders_via_api(tmp_path):
    c, _ = client(tmp_path)
    with c:
        a = c.post("/orders", json={"items": ["C1", "K1"]}).json()["order"]["order_id"]
        b = c.post("/orders", json={"items": ["S1"]}).json()["order"]["order_id"]
        c.post("/orders", json={"items": ["N1"]})
        assert c.delete(f"/orders/{a}").json() == {"deleted": a}
        assert [o["order_id"] for o in c.get("/orders").json()][-1] == b  # a is gone, others kept
        assert c.delete(f"/orders/{a}").status_code == 404                 # already deleted
        assert c.delete("/orders/ORD-does-not-exist").status_code == 404
        assert c.delete("/orders").json() == {"deleted_count": 2}
        assert c.get("/orders").json() == []
        assert c.delete("/orders").json() == {"deleted_count": 0}


def test_delete_order_store_keeps_other_orders(tmp_path):
    csv_path = tmp_path / "placed_orders.csv"
    ids = [model_store.append_order([CAM], csv_path)["order_id"] for _ in range(3)]  # same millisecond
    assert len(set(ids)) == 3  # ids stay unique
    assert model_store.delete_order(ids[1], csv_path) is True
    assert [o["order_id"] for o in model_store.list_orders(csv_path)] == [ids[2], ids[0]]
    assert model_store.delete_order(ids[1], csv_path) is False
    assert model_store.delete_order("x", tmp_path / "missing.csv") is False
    assert model_store.clear_orders(csv_path) == 2 and model_store.list_orders(csv_path) == []


def test_explain_endpoint(tmp_path):
    c, _ = client(tmp_path)
    with c:
        e = c.post("/explain", json={"cart": ["C1", "P1"], "item": "K1"}).json()
        assert e["orders_with_cart_and_item"] == 3 and e["rule"] == "{C1, P1} -> K1"
        assert c.post("/explain", json={"cart": ["C1"], "item": "nope"}).status_code == 404
