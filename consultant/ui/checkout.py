"""Streamlit checkout page: a storefront-style cart with live Apriori recommendations."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from datetime import datetime

from ..recommenders.basket import category, sku
from ..recommenders.model_store import append_order, clear_orders, delete_order, list_orders, production_model

CATEGORY_STYLE = {  # category -> (icon, label)
    "camera": ("📷", "Cameras"),
    "recorder": ("💾", "Recorders"),
    "control_panel": ("🎛️", "Control panels"),
    "keypad": ("🔢", "Keypads"),
    "smoke_detector": ("🔥", "Smoke detectors"),
    "co_detector": ("⚠️", "CO detectors"),
    "motion_detector": ("🏃", "Motion detectors"),
    "glassbreak_detector": ("🪟", "Glass-break"),
    "sensor": ("🚪", "Sensors"),
    "other": ("📦", "Other"),
}
CART_KEY, PICK_KEY = "checkout_cart", "checkout_pick"


def icon(item: str) -> str:
    return CATEGORY_STYLE.get(category(item), CATEGORY_STYLE["other"])[0]


def display_name(item: str) -> str:
    return re.sub(r"\s*\(SKU:[^)]*\)", "", item).strip()


def strength(lift: float) -> tuple[str, str]:
    """Human label + badge color for how much more likely the item is than average."""
    if lift >= 3:
        return "Strong match", "green"
    if lift >= 1.5:
        return "Good match", "blue"
    return "Possible match", "gray"


# -- cart callbacks (run before the rerun, so the page redraws once) ----------------------
def _cart() -> list[str]:
    return st.session_state.setdefault(CART_KEY, [])


def add_item(item: str) -> None:
    if item and item not in _cart():
        _cart().append(item)
        st.toast(f"Added **{display_name(item)}**", icon="🛒")


def add_picked() -> None:
    add_item(st.session_state.get(PICK_KEY))
    st.session_state[PICK_KEY] = None


def remove_item(item: str) -> None:
    if item in _cart():
        _cart().remove(item)


def clear_cart() -> None:
    st.session_state[CART_KEY] = []


def complete_checkout() -> None:
    cart = list(_cart())
    if cart:
        order = append_order(cart)
        st.session_state[CART_KEY] = []
        st.toast(f"Order **{order['order_id']}** placed ({len(cart)} items). See it under *Your orders*.", icon="✅")


def reorder(items: list[str], known: set[str]) -> None:
    """Replace the cart with a past order's items (skipping any no longer in the catalog)."""
    st.session_state[CART_KEY] = [i for i in dict.fromkeys(items) if i in known]
    st.toast("Cart refilled from your past order", icon="🔁")


def remove_order(order_id: str) -> None:
    if delete_order(order_id):
        st.toast(f"Deleted order **{order_id}**", icon="🗑️")
    else:
        st.toast(f"Order {order_id} was already deleted", icon="ℹ️")


def remove_all_orders() -> None:
    n = clear_orders()
    st.toast(f"Deleted {n} order{'s' if n != 1 else ''}", icon="🗑️")


# -- page ---------------------------------------------------------------------------
def render() -> None:
    rec, meta = production_model()
    cart = _cart()

    st.subheader("🛒 Checkout")
    st.caption("Add products to your cart. Each time the cart changes, we suggest what other customers bought "
               "with this exact combination.")

    left, right = st.columns([2, 3], gap="large")
    with left:
        _product_picker(rec, cart)
        _cart_panel(rec, cart)
    with right:
        _recommendation_panel(rec, cart)

    _orders_panel(rec)
    _how_it_works(rec, meta)


def _product_picker(rec, cart: list[str]) -> None:
    with st.container(border=True):
        st.markdown("**Add products**")
        cats = sorted({category(i) for i in rec.items}, key=lambda c: CATEGORY_STYLE.get(c, ("", c))[1])
        chosen = st.pills("Category", cats, selection_mode="single", key="checkout_category",
                          format_func=lambda c: " ".join(CATEGORY_STYLE.get(c, CATEGORY_STYLE["other"])),
                          label_visibility="collapsed")
        options = [i for i in rec.items if i not in cart and (chosen is None or category(i) == chosen)]
        st.selectbox("Search products", options, index=None, key=PICK_KEY, on_change=add_picked,
                     format_func=lambda i: f"{icon(i)}  {display_name(i)}  ·  {sku(i)}",
                     placeholder="Search by name, brand or SKU…", label_visibility="collapsed")


def _cart_panel(rec, cart: list[str]) -> None:
    with st.container(border=True):
        st.markdown(f"**Your cart** · {len(cart)} item{'s' if len(cart) != 1 else ''}")
        if not cart:
            st.caption("Your cart is empty. Search above, or start with a best-seller:")
            popular = sorted(rec.items, key=lambda i: -rec.item_count[i])[:6]
            cols = st.columns(2)
            for n, item in enumerate(popular):
                cols[n % 2].button(f"{icon(item)} {sku(item)}", key=f"pop_{sku(item)}", width="stretch",
                                   help=display_name(item), on_click=add_item, args=(item,))
            return
        for item in cart:
            a, b = st.columns([6, 1], vertical_alignment="center")
            a.markdown(f"{icon(item)} **{display_name(item)}**  \n:gray[SKU {sku(item)}]")
            b.button("✕", key=f"rm_{sku(item)}", help="Remove from cart", on_click=remove_item, args=(item,))
        st.divider()
        c1, c2 = st.columns(2)
        c1.button("Clear cart", width="stretch", on_click=clear_cart)
        c2.button("Complete checkout", type="primary", width="stretch", on_click=complete_checkout,
                  help="Places the order and adds it to Your orders. Orders are not used to train the recommender.")


def _recommendation_panel(rec, cart: list[str]) -> None:
    res = rec.recommend_for_cart(cart, k=6)
    st.markdown("#### ✨ Frequently bought together")
    status = res["status"]
    if status == "empty_cart":
        with st.container(border=True):
            st.markdown("👈 **Add your first item** to see what customers usually buy with it.")
            st.caption("Suggestions update every time your cart changes.")
        return
    if status != "ok":
        with st.container(border=True):
            st.markdown("🤔 **No suggestions for this combination yet**")
            st.caption(res["message"].replace("No recommendations shown.", "").strip())
            if status == "no_pattern" and len(cart) > 1:
                st.caption("We only suggest items with proven purchase history for your exact cart. "
                           "Removing an item may bring suggestions back.")
        return
    st.caption(f"Customers whose order contained your cart: **{res['combination_orders']}** past orders")
    recs = res["recommendations"]
    for row in range(0, len(recs), 2):
        cols = st.columns(2)
        for col, r in zip(cols, recs[row:row + 2]):
            with col.container(border=True):
                label, color = strength(r["lift"])
                st.markdown(f"### {icon(r['item'])}")
                st.markdown(f"**{display_name(r['item'])}**  \n:gray[SKU {r['sku']}]")
                st.markdown(f":{color}-badge[{label}] :violet-badge[{r['lift']:.1f}× more likely]")
                st.progress(min(1.0, r["confidence"]),
                            text=f"Bought by {r['confidence']:.0%} of matching orders ({r['orders']} orders)")
                with st.expander("💡 Why this?"):
                    _why_this(rec.explain(cart, r["item"]))
                st.button("＋ Add to cart", key=f"rec_{r['sku']}", width="stretch", type="secondary",
                          on_click=add_item, args=(r["item"],))


STRENGTH_STYLE = {
    "solid": ("green", "Solid evidence", "Backed by 10+ past orders."),
    "moderate": ("orange", "Moderate evidence", "Backed by 5-9 past orders."),
    "thin": ("gray", "Thin evidence", "Only a handful of past orders - treat it as a hint, not a rule."),
}


def _why_this(e: dict) -> None:
    """Explain one recommendation in plain language from the order data."""
    cart_txt = " + ".join(e["cart"])
    st.markdown(f"**What the data shows**  \n"
                f"Of the **{e['orders_with_cart']}** past orders that contained **{cart_txt}**, "
                f"**{e['orders_with_cart_and_item']}** also included **{e['item']}**.")
    st.progress(min(1.0, e["confidence"]), text=f"With your cart: {e['confidence']:.0%} of orders")
    st.progress(min(1.0, e["base_rate"]), text=f"All shoppers: {e['base_rate']:.0%} of orders "
                                                f"({e['orders_with_item']} of {e['total_orders']})")
    st.markdown(f"So shoppers with your cart are **{e['lift']:.1f}×** as likely to add it as an average shopper.")

    color, label, note = STRENGTH_STYLE[e["evidence_strength"]]
    st.markdown(f":{color}-badge[{label}] :gray-badge[rule {e['rule']}] "
                f":gray-badge[support {e['support']:.1%} · confidence {e['confidence']:.0%} · lift {e['lift']:.2f}]")
    st.caption(note)

    if e["pairing_reason"]:
        st.markdown(f"**Why it fits** · {e['pairing_reason']}")
        st.caption("Product knowledge, shown to explain the pattern - the recommendation itself comes from the "
                   "order data above.")
    else:
        st.caption("No standard technical dependency between these product types - this suggestion reflects "
                   "what customers bought together.")
    if e["brand_note"]:
        st.warning(e["brand_note"], icon="⚠️")

    if e["example_orders"]:
        st.markdown("**Example past orders**")
        st.markdown("\n".join(
            f"- Order {o['order_id']}: " + ", ".join(f"**{i}**" if i in (*e["cart"], e["item"]) else i
                                                       for i in o["items"])
            for o in e["example_orders"]))


def _format_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %Y · %I:%M %p")
    except ValueError:
        return iso


def _orders_panel(rec) -> None:
    orders = list_orders()
    head, tools = st.columns([5, 1], vertical_alignment="center")
    head.markdown(f"#### 📦 Your orders · {len(orders)}")
    if not orders:
        st.caption("Orders you place with **Complete checkout** will appear here.")
        return
    with tools.popover("🗑️ Delete all", width="stretch"):
        st.markdown(f"Delete **all {len(orders)} orders**? This can't be undone.")
        st.button("Yes, delete all", type="primary", key="orders_delete_all", width="stretch",
                  on_click=remove_all_orders)
    known = set(rec.items)
    show_all = len(orders) <= 5 or st.toggle(f"Show all {len(orders)} orders", key="orders_show_all")
    for n, order in enumerate(orders if show_all else orders[:5]):
        items, oid = order["items"], order["order_id"]
        with st.expander(f"**{oid}** · {_format_time(order['placed_at'])} · "
                         f"{len(items)} item{'s' if len(items) != 1 else ''}", expanded=n == 0):
            a, b = st.columns([5, 1], vertical_alignment="center")
            a.markdown("\n".join(f"- {icon(i)} **{display_name(i)}** :gray[· SKU {sku(i)}]" for i in items))
            with b:
                st.button("🔁 Reorder", key=f"reorder_{oid}", width="stretch",
                          help="Replace your cart with these items", on_click=reorder, args=(items, known))
                with st.popover("🗑️ Delete", width="stretch"):
                    st.markdown(f"Delete order **{oid}**? This can't be undone.")
                    st.button("Yes, delete", type="primary", key=f"delete_{oid}", width="stretch",
                              on_click=remove_order, args=(oid,))


def _how_it_works(rec, meta: dict) -> None:
    with st.expander("ℹ️ How recommendations work · model details"):
        m, v = meta["model"], meta["validation"]
        st.markdown(
            f"- Suggestions come from **Apriori association rules** mined from {m['n_transactions']:,} past orders "
            f"({m['n_rules']:,} rules).\n"
            f"- Only rules for your **exact cart combination** are used, and only items bought **more often than "
            f"average** with it (lift > {m['min_lift']}).\n"
            f"- A combination needs at least **{m['min_combination_orders']} past orders**; otherwise nothing is "
            "shown rather than guessing.\n"
            f"- Model `{meta['version']}` trained {meta.get('trained_at', '')} on the original order history only; "
            "orders you place appear under *Your orders* but never change the recommendations.\n"
            "- Serve it as an API with `uvicorn consultant.services.basket_api:app`.")
        c1, c2, c3 = st.columns(3)
        a = v["apriori"]
        c1.metric("Coverage (carts with suggestions)", f"{a['coverage']:.0%}")
        c2.metric("Hit@5 when shown", f"{a['hit@5']:.0%}")
        c3.metric("Next-item hit@5", f"{a['next_item_hit@5']:.0%}",
                  delta=f"{a['next_item_hit@5'] - v['popularity_baseline']['next_item_hit@5']:+.0%} vs popularity",
                  delta_color="normal")
        st.markdown("**Validation by cart size** (held-out orders replayed as checkouts)")
        st.dataframe(pd.DataFrame(v["apriori_by_cart_size"]).T.rename_axis("cart size"), width="stretch")
        st.markdown("**Strongest rules**")
        st.dataframe(pd.DataFrame(rec.top_rules(10)), hide_index=True, width="stretch")
