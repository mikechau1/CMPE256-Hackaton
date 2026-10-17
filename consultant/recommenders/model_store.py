"""Versioned storage, validation-gated promotion and retraining for the checkout recommender.

Layout (data/models/):
    basket_<version>.pkl   fitted BasketRecommender
    basket_<version>.json  metadata: params, data size, validation metrics
    latest.json            pointer to the promoted version

Training data = the original Retail CSV (1,000 orders) only. Orders placed through the checkout
UI or the API are kept as order history (data/retail/placed_orders.csv) and never used for
training, so they cannot change what the model recommends.

Run: python -m consultant.recommenders.model_store [--force] [--min-support 0.002]
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from .basket import MIN_SUPPORT, BasketRecommender, evaluate_checkout_simulation, load_transactions

MODEL_DIR = config.DATA_DIR / "models"
PLACED_ORDERS_CSV = config.DATA_DIR / "retail" / "placed_orders.csv"  # order history only, not training data
MAX_ITEMS_PER_ORDER = 20
# Promotion gate: a retrained model may not lose more than this on validation hit@5 / coverage.
MAX_HIT_DROP = 0.05
MAX_COVERAGE_DROP = 0.10

_lock = threading.Lock()


def append_order(items: list[str], orders_csv: Path = PLACED_ORDERS_CSV) -> dict:
    """Record a placed order in the order history (not used for training). Returns the saved order."""
    order = {"order_id": f"ORD-{int(time.time() * 1000)}-{secrets.token_hex(2)}",  # unique even within 1 ms
             "placed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
             "items": list(items)[:MAX_ITEMS_PER_ORDER]}
    with _lock:
        orders_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = orders_csv.exists()
        with open(orders_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["order_id", "placed_at"] + [f"item_{i}" for i in range(1, MAX_ITEMS_PER_ORDER + 1)])
            w.writerow([order["order_id"], order["placed_at"], *order["items"]])
    return order


def list_orders(orders_csv: Path = PLACED_ORDERS_CSV) -> list[dict]:
    """Placed orders, newest first."""
    if not orders_csv.exists():
        return []
    with open(orders_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    orders = [{"order_id": r[0], "placed_at": r[1], "items": [x for x in r[2:] if x.strip()]} for r in rows if r]
    return orders[::-1]


def delete_order(order_id: str, orders_csv: Path = PLACED_ORDERS_CSV) -> bool:
    """Remove one order from the history. Returns False if no such order exists."""
    with _lock:
        if not orders_csv.exists():
            return False
        with open(orders_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        kept = [rows[0]] + [r for r in rows[1:] if r and r[0] != order_id]
        if len(kept) == len(rows):
            return False
        with open(orders_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(kept)
        return True


def clear_orders(orders_csv: Path = PLACED_ORDERS_CSV) -> int:
    """Delete the whole order history. Returns how many orders were removed."""
    with _lock:
        n = len(list_orders(orders_csv)) if orders_csv.exists() else 0
        orders_csv.unlink(missing_ok=True)
        return n


def latest_metadata(model_dir: Path = MODEL_DIR) -> dict | None:
    p = model_dir / "latest.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_model(version: str, model_dir: Path = MODEL_DIR) -> BasketRecommender:
    with open(model_dir / f"basket_{version}.pkl", "rb") as f:
        return pickle.load(f)


def train(min_support: float = MIN_SUPPORT, force: bool = False, model_dir: Path = MODEL_DIR,
          transactions: list[frozenset[str]] | None = None) -> dict:
    """Fit, validate, save and (if it passes the gate) promote a new model version.

    Trains on the original 1,000 Retail orders; placed orders are deliberately excluded.
    """
    tx = transactions if transactions is not None else load_transactions()
    t0 = time.time()
    validation = evaluate_checkout_simulation(tx, min_support=min_support)
    model = BasketRecommender(tx, min_support=min_support)  # final model uses all data
    version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    meta = {"version": version, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_seconds": round(time.time() - t0, 2), "model": model.summary(),
            "validation": {k: v for k, v in validation.items() if k != "apriori"}
            | {"apriori": {k: v for k, v in validation["apriori"].items() if k != "model"}}}

    prev = latest_metadata(model_dir)
    promoted, reason = True, "first model" if prev is None else "passed validation gate"
    if prev is not None and not force:
        old, new = prev["validation"]["apriori"], meta["validation"]["apriori"]
        if new["hit@5"] < old["hit@5"] - MAX_HIT_DROP:
            promoted, reason = False, f"hit@5 dropped {old['hit@5']} -> {new['hit@5']}"
        elif new["coverage"] < old["coverage"] - MAX_COVERAGE_DROP:
            promoted, reason = False, f"coverage dropped {old['coverage']} -> {new['coverage']}"
    if force and prev is not None:
        reason = "forced"
    meta |= {"promoted": promoted, "promotion_reason": reason,
             "previous_version": prev["version"] if prev else None}

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / f"basket_{version}.pkl", "wb") as f:
        pickle.dump(model, f)
    (model_dir / f"basket_{version}.json").write_text(json.dumps(meta, indent=2))
    if promoted:
        (model_dir / "latest.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_latest(model_dir: Path = MODEL_DIR) -> tuple[BasketRecommender, dict]:
    """Promoted model, training one first if none exists yet."""
    meta = latest_metadata(model_dir)
    if meta is None or not (model_dir / f"basket_{meta['version']}.pkl").exists():
        meta = train(model_dir=model_dir)
    return load_model(meta["version"], model_dir), meta


_cached: tuple[BasketRecommender, dict] | None = None


def production_model() -> tuple[BasketRecommender, dict]:
    """Process-wide cached latest model (used by the Procurement agent and the UI)."""
    global _cached
    meta = latest_metadata()
    if _cached is None or meta is None or _cached[1]["version"] != meta["version"]:
        _cached = load_latest()
    return _cached


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain the checkout recommender")
    ap.add_argument("--min-support", type=float, default=MIN_SUPPORT)
    ap.add_argument("--force", action="store_true", help="promote even if validation gets worse")
    args = ap.parse_args()
    meta = train(min_support=args.min_support, force=args.force)
    v = meta["validation"]["apriori"]
    print(f"{meta['version']}: {meta['model']['n_transactions']} orders, {meta['model']['n_rules']} rules; "
          f"validation coverage {v['coverage']}, hit@5 {v['hit@5']}, next-item hit@5 {v['next_item_hit@5']}; "
          f"promoted={meta['promoted']} ({meta['promotion_reason']})")


if __name__ == "__main__":
    main()
