"""Checkout recommendation microservice (FastAPI).

Run:  uvicorn consultant.services.basket_api:app --port 8000
Docs: http://localhost:8000/docs

Endpoints
    GET  /health                 liveness + serving model version
    GET  /items?q=               product catalog (for the storefront search box)
    POST /recommend              {cart: [...names or SKUs], k, session_id} -> ranked items or abstention
    POST /explain                {cart, item} -> why that item is recommended (counts, lift, example orders)
    GET  /orders                 order history (placed orders, newest first)
    POST /orders                 {items: [...]} record a placed order (history only - never used for training)
    DELETE /orders/{order_id}    delete one past order;  DELETE /orders  delete the whole history
    POST /retrain                {force} retrain on the original orders, validate, hot-swap if promoted
    GET  /rules?n=               strongest association rules
    GET  /metrics                monitoring: traffic, statuses, abstention rate, latency, uptake

Set BASKET_RETRAIN_HOURS=24 to retrain automatically in the background.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..recommenders import model_store
from ..recommenders.basket import BasketRecommender, category, sku

LOG_PATH = config.DATA_DIR / "logs" / "basket_requests.jsonl"


class RecommendRequest(BaseModel):
    cart: list[str] = Field(default_factory=list, description="Item names or SKUs currently in the cart.")
    k: int = Field(5, ge=1, le=20)
    session_id: str | None = Field(None, description="Lets monitoring measure recommendation uptake.")


class ExplainRequest(BaseModel):
    cart: list[str] = Field(min_length=1)
    item: str


class OrderRequest(BaseModel):
    items: list[str] = Field(min_length=1)


class RetrainRequest(BaseModel):
    force: bool = False


class Monitor:
    """In-process serving metrics; every request is also appended to a JSONL log."""

    def __init__(self, log_path: Path | None = LOG_PATH):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.status = Counter()
        self.latency_ms: deque[float] = deque(maxlen=5000)
        self.last_recs: dict[str, set[str]] = {}   # session -> last recommended SKUs
        self.last_cart: dict[str, set[str]] = {}
        self.adds = 0          # items added after a recommendation was shown
        self.accepted = 0      # ...that were one of the recommended items
        self.log_path = log_path

    def record(self, req: RecommendRequest, resp: dict, ms: float) -> None:
        with self.lock:
            self.requests += 1
            self.status[resp["status"]] += 1
            self.latency_ms.append(ms)
            if req.session_id:
                cart = set(resp["cart"])
                new = cart - self.last_cart.get(req.session_id, set())
                shown = self.last_recs.get(req.session_id)
                if shown and new:
                    self.adds += len(new)
                    self.accepted += len(new & shown)
                self.last_cart[req.session_id] = cart
                self.last_recs[req.session_id] = {r["sku"] for r in resp["recommendations"]}
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "session": req.session_id, "cart": resp["cart"],
                                    "status": resp["status"], "n_recs": len(resp["recommendations"]),
                                    "model": resp.get("model_version"), "ms": round(ms, 2)}) + "\n")

    def snapshot(self) -> dict:
        with self.lock:
            lat = sorted(self.latency_ms)

            def pct(p):
                return round(lat[min(len(lat) - 1, int(p * len(lat)))], 2) if lat else None
            n = max(1, self.requests)
            return {"uptime_s": round(time.time() - self.started), "requests": self.requests,
                    "status_counts": dict(self.status),
                    "recommendation_rate": round(self.status["ok"] / n, 3),
                    "abstention_rate": round((self.status["no_pattern"] + self.status["unknown_items"]) / n, 3),
                    "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95),
                    "uptake_rate": round(self.accepted / self.adds, 3) if self.adds else None,
                    "items_added_after_recs": self.adds}


def create_app(model: BasketRecommender | None = None, metadata: dict | None = None,
               trainer=None, order_sink=None, order_source=None, order_deleter=None, order_clearer=None,
               monitor: Monitor | None = None) -> FastAPI:
    """App factory; tests inject a toy model/trainer, production loads the promoted model."""
    state = {"model": model, "meta": metadata or {}}
    trainer = trainer or (lambda force: model_store.train(force=force))
    order_sink = order_sink or model_store.append_order
    order_source = order_source or model_store.list_orders
    order_deleter = order_deleter or model_store.delete_order
    order_clearer = order_clearer or model_store.clear_orders
    monitor = monitor or Monitor()
    swap_lock = threading.Lock()

    def retrain(force: bool) -> dict:
        meta = trainer(force)
        if meta.get("promoted"):
            new_model = meta.pop("_model", None) or model_store.load_model(meta["version"])
            with swap_lock:
                state["model"], state["meta"] = new_model, meta
        meta.pop("_model", None)
        return meta

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["model"] is None:
            state["model"], state["meta"] = model_store.load_latest()
        task = None
        hours = float(os.getenv("BASKET_RETRAIN_HOURS", "0") or 0)
        if hours > 0:
            async def periodic():
                while True:
                    await asyncio.sleep(hours * 3600)
                    await asyncio.to_thread(retrain, False)
            task = asyncio.create_task(periodic())
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Checkout Recommender", version="1.0",
                  description="Apriori market-basket recommendations for the checkout cart.", lifespan=lifespan)

    def current() -> tuple[BasketRecommender, dict]:
        with swap_lock:
            return state["model"], state["meta"]

    @app.get("/health")
    def health():
        m, meta = current()
        return {"status": "ok" if m is not None else "loading", "model_version": meta.get("version"),
                "trained_at": meta.get("trained_at")}

    @app.get("/items")
    def items(q: str | None = None, limit: int = 50):
        m, _ = current()
        names = m.find_items(q, k=limit) if q else m.items[:limit]
        return [{"item": i, "sku": sku(i), "category": category(i), "orders": m.item_count[i]} for i in names]

    @app.post("/recommend")
    def recommend(req: RecommendRequest):
        t0 = time.perf_counter()
        m, meta = current()
        resp = m.recommend_for_cart(req.cart, k=req.k) | {"model_version": meta.get("version")}
        monitor.record(req, resp, (time.perf_counter() - t0) * 1000)
        return resp

    @app.post("/explain")
    def explain(req: ExplainRequest):
        """Reasoning behind one recommendation: counts, lift, Apriori rule, example past orders."""
        m, _ = current()
        if m.resolve(req.item) is None or any(m.resolve(c) is None for c in req.cart):
            raise HTTPException(status_code=404, detail="unknown item in cart or recommendation")
        return m.explain(req.cart, req.item)

    @app.post("/orders")
    def orders(req: OrderRequest):
        m, _ = current()
        resolved = [m.resolve(i) or i for i in req.items]
        return {"stored": True, "order": order_sink(resolved), "used_for_training": False}

    @app.get("/orders")
    def order_history(limit: int = 50):
        return order_source()[:limit]

    @app.delete("/orders/{order_id}")
    def delete_order(order_id: str):
        if not order_deleter(order_id):
            raise HTTPException(status_code=404, detail=f"order {order_id} not found")
        return {"deleted": order_id}

    @app.delete("/orders")
    def delete_all_orders():
        return {"deleted_count": order_clearer()}

    @app.post("/retrain")
    def retrain_endpoint(req: RetrainRequest):
        try:
            return retrain(req.force)
        except Exception as e:  # surface training failures without killing the service
            raise HTTPException(status_code=500, detail=f"retraining failed: {e}") from e

    @app.get("/rules")
    def rules(n: int = 20):
        m, _ = current()
        return m.top_rules(n)

    @app.get("/metrics")
    def metrics():
        _, meta = current()
        return {"serving": monitor.snapshot(),
                "model": {"version": meta.get("version"), "trained_at": meta.get("trained_at"),
                          **meta.get("model", {})},
                "validation": meta.get("validation", {}).get("apriori")}

    return app


app = create_app()
