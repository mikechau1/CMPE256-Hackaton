"""Streamlit UI for the multi-agent consulting platform.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import pickle
import queue
import threading
import time

import pandas as pd
import streamlit as st

from consultant import config
from consultant.ui.report import render_ledger, render_report

st.set_page_config(page_title="AI Consulting Platform", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

SAMPLES = {
    "Cisco router failure": "Our Cisco Catalyst 8300 enterprise router is experiencing intermittent connectivity "
                            "failures. Which replacement parts should we order, and what actions should we take "
                            "to prevent future outages?",
    "Lab instrument": "Our pathology lab wants to digitize routine clinical histopathology slides for primary "
                      "diagnosis at high throughput. Which Agilent instrument fits, and what should we check?",
    "Vendor risk (SEC)": "What are the financial, competitive, and operational risks associated with Cisco and "
                         "Agilent based on their 2025 10-K filings?",
    "Food security (CSR)": "Which technology investments could improve food security and sustainability "
                           "outcomes, and how should we measure impact?",
    "Procurement bundle": "We are ordering a Bosch B5512 control panel and a Hikvision 4MP IP camera. What "
                          "else do customers usually order with these?",
    "Cross-domain": "We run diagnostic labs on Agilent instruments connected over Cisco ISR 4000 routers that "
                    "keep dropping links. Should we standardise on these vendors given their 2025 financial "
                    "risks, and how could the upgrade support our food-security ESG commitments?",
}


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


@st.cache_data(show_spinner=False)
def index_stats() -> dict:
    stats = {}
    for d in ("technology", "clinical", "sec", "csr"):
        p = config.INDEX_DIR / f"{d}_chunks.pkl"
        if p.exists():
            with open(p, "rb") as f:
                metas = pickle.load(f)["metas"]
            stats[d] = {"chunks": len(metas), "documents": len({m["doc_id"] for m in metas})}
    return stats


# --- top bar -------------------------------------------------------------------
st.session_state.setdefault("turns", [])
stats = index_stats()
max_loops = config.MAX_GOAL_LOOPS  # set by the top-bar slider; read by run_engagement()


def render_top_bar(on_consult: bool) -> int:
    """Pinned header. Chat-only controls (Clear chat) appear only on the Consult page."""
    _dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    # Our bar is pinned to the top of the page. Streamlit's header stays (it holds the sidebar
    # expand button and the menu) but is transparent so it floats over the bar; Deploy is hidden.
    st.markdown(f"""<style>
    header[data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stAppDeployButton"] {{ display: none; }}
    [data-testid="stMainBlockContainer"] {{ padding-top: 0; }}
    .st-key-topbar {{
        position: sticky; top: 0; z-index: 999;
        background: {"#0e1117" if _dark else "#ffffff"};
        border-bottom: 1px solid {"rgba(250,250,250,0.15)" if _dark else "rgba(49,51,63,0.15)"};
        padding: 0.6rem 0 0.4rem; margin-bottom: 0.5rem;
    }}
    .st-key-topbar h3 {{ margin: 0; padding: 0; }}
    </style>""", unsafe_allow_html=True)
    with st.container(key="topbar"):
        t1, t2, t3, t4, t5 = st.columns([3, 2, 2, 2, 1.3], vertical_alignment="center")
        with t1:
            st.markdown("### 🧭 AI Consulting Platform")
            st.caption("Manager agent → specialist agents → evidence-grounded recommendations")
        t2.markdown(f"**Agents** `{config.AGENT_MODEL}`  \n**Router/judge** `{config.FAST_MODEL}`")
        with t3:
            if stats:
                with st.popover(f"📚 Knowledge bases · {sum(s['chunks'] for s in stats.values()):,} chunks",
                                width="stretch"):
                    st.dataframe(pd.DataFrame(stats).T, width="stretch")
            else:
                st.warning("No indexes - run `python -m consultant.ingest.build_index`.")
        loops = t4.slider("Max evaluate-goal iterations", 1, 3, config.MAX_GOAL_LOOPS)
        if on_consult and t5.button("Clear chat", width="stretch", help="Remove the conversation history"):
            st.session_state.turns = []
            st.rerun()
    if not has_api_key():
        st.error("ANTHROPIC_API_KEY not set - add it to .env. Retrieval and recommender tabs still work.")
    return loops



def history_text(turns) -> str:
    parts = []
    for t in turns[-3:]:
        parts.append(f"Customer: {t['request']}")
        if t.get("result"):
            rep = t["result"].report
            parts.append(f"Consultant summary: {rep.executive_summary}")
            if rep.information_needed:
                parts.append("Consultant asked for: " + "; ".join(rep.information_needed))
    return "\n".join(parts)


def run_engagement(request: str) -> None:
    from consultant.agents.manager import ManagerAgent
    from consultant.llm import LLM

    events: queue.Queue = queue.Queue()
    holder: dict = {}
    history = history_text(st.session_state.turns)

    def worker():
        try:
            mgr = ManagerAgent(llm=LLM(), on_event=lambda k, p: events.put((k, p)), max_loops=max_loops)
            holder["result"] = mgr.run(request, history=history)
        except Exception as e:  # surface errors in the UI
            holder["error"] = e
        finally:
            events.put(("done", {}))

    status = st.status("Manager agent working…", expanded=True)
    ledger_box = status.empty()
    tool_box = status.empty()
    tool_lines: list[str] = []
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        try:
            kind, payload = events.get(timeout=0.3)
        except queue.Empty:
            continue
        if kind == "done":
            break
        if "ledger" in payload:
            render_ledger(payload["ledger"], ledger_box)
        if kind == "tool_call":
            tool_lines.append(f"🔧 **{config.DOMAIN_LABELS[payload['agent']]}** → `{payload['tool']}` "
                              f"{json.dumps(payload['args'])[:140]}")
            tool_box.markdown("\n\n".join(tool_lines[-8:]))
        if kind == "evaluation":
            g = payload["goal"]
            status.write(f"🎯 Goal check: resolved={g.resolved}, score={g.score:.2f}"
                         + (f", gaps: {'; '.join(g.gaps)}" if g.gaps else ""))
    t.join()
    if "error" in holder:
        status.update(label="Engagement failed", state="error")
        st.exception(holder["error"])
        st.session_state.turns.append({"request": request, "result": None, "error": str(holder["error"])})
    else:
        status.update(label="Engagement complete", state="complete", expanded=False)
        st.session_state.turns.append({"request": request, "result": holder["result"]})


# --- Consult tab ----------------------------------------------------------------
def page_consult() -> None:
    st.subheader("Ask the Management Consulting Manager Agent")
    cols = st.columns(len(SAMPLES))
    clicked = None
    for col, (name, q) in zip(cols, SAMPLES.items()):
        if col.button(name, width="stretch"):
            clicked = q
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["request"])
        with st.chat_message("assistant"):
            if turn.get("result"):
                render_report(turn["result"])
            else:
                st.error(turn.get("error", "No result"))
    prompt = st.chat_input("Describe your business problem…") or clicked
    if prompt:
        if not has_api_key():
            st.error("Set ANTHROPIC_API_KEY in .env to run the agents.")
        else:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                run_engagement(prompt)
            st.rerun()

# --- Knowledge base explorer -------------------------------------------------------
def page_knowledge_base() -> None:
    st.subheader("Hybrid retrieval explorer")
    c1, c2, c3 = st.columns([3, 1, 1])
    q = c1.text_input("Query", "Catalyst 8300 power supply options")
    domain = c2.selectbox("Domain", [d for d in ("technology", "clinical", "sec", "csr") if d in index_stats()] or ["technology"])
    mode = c3.selectbox("Mode", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    if q and index_stats():
        from consultant.retrieval.hybrid import get_retriever
        t0 = time.time()
        hits = get_retriever(domain).search(q, k=8, mode=mode)
        st.caption(f"{len(hits)} results in {time.time() - t0:.2f}s")
        for h in hits:
            with st.expander(f"{h.score:.3f} · {h.citation()}"):
                st.caption(h.source)
                st.write(h.text)

# --- Checkout recommender ------------------------------------------------------------
def page_checkout() -> None:
    from consultant.ui.checkout import render
    render()


# --- Evaluation --------------------------------------------------------------------
def page_evaluation() -> None:
    st.subheader("Evaluation results")
    report_md = config.EVAL_DIR / "results" / "report.md"
    if report_md.exists():
        st.markdown(report_md.read_text(encoding="utf-8"))
    else:
        st.info("Run `python -m eval.run_eval` to generate routing, retrieval, recommendation and response-quality metrics.")


# --- Navigation: collapsible left sidebar ---------------------------------------------
page = st.navigation([
    st.Page(page_consult, title="Consult", icon="💬", url_path="consult", default=True),
    st.Page(page_checkout, title="Checkout recommender", icon="🛒", url_path="checkout"),
    st.Page(page_knowledge_base, title="Knowledge base", icon="🔎", url_path="knowledge-base"),
    st.Page(page_evaluation, title="Evaluation", icon="📊", url_path="evaluation"),
], position="sidebar")
max_loops = render_top_bar(on_consult=page.title == "Consult")
page.run()
