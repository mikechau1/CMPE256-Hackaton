"""Human-friendly rendering of a ConsultingReport: only the summary is open; every section starts collapsed."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from .. import config
from ..confidence import label as confidence_label

KB_TOPICS = ("Cisco routers & Google Cloud (MCP/RAG) · clinical research, trials & Agilent instruments · "
             "SEC 10-K filings of Agilent, Alphabet & Cisco · food security & CSR (FAO SOFI) · "
             "security / fire-alarm product orders")
STATUS_ICON = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌", "needs_input": "🙋"}
CONF_COLOR = {"high": "green", "medium": "orange", "low": "gray"}
PRIORITY = {"P1": ("red", "P1 · Do now"), "P2": ("orange", "P2 · Next"), "P3": ("blue", "P3 · Later")}
AGENT_ICON = {"technology": "🖧", "clinical": "🧪", "sec": "📈", "csr": "🌾", "procurement": "🛒"}


def render_ledger(ledger, container) -> None:
    """Task & progress ledger (also streamed live while the manager works)."""
    with container.container():
        if ledger.route:
            st.markdown(f"**Routing:** {', '.join(config.DOMAIN_LABELS[d] for d in ledger.route.domains)} "
                        f"· confidence {ledger.route.confidence:.2f}  \n_{ledger.route.rationale}_")
        if ledger.approach:
            st.markdown(f"**Resolution approach:** {ledger.approach}")
        if ledger.subtasks:
            rows = [{"#": s.id, "status": f"{STATUS_ICON[s.status]} {s.status}", "agent": config.DOMAIN_LABELS[s.agent],
                     "iter": s.iteration, "task": s.instruction, "result": s.summary[:160]} for s in ledger.subtasks]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for line in ledger.log:
            st.caption(line)


# -- small formatting helpers ---------------------------------------------------------
def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _sources(refs: list[str], evidence: dict) -> str:
    """'📎 Catalyst 8300 Data Sheet, p.19 · Cisco Form 10-K FY2026, Item 7, p.33' (with refs for traceability)."""
    parts = [f"{_short(evidence[r].citation(), 60)} `{r}`" for r in refs if r in evidence]
    return "📎 " + " · ".join(parts) if parts else "📎 _no source cited_"


def _conf_badge(level: str) -> str:
    return f":{CONF_COLOR.get(level, 'gray')}-badge[{level} confidence]"


# -- sections -----------------------------------------------------------------------
def _header(res) -> None:
    rep = res.report
    agents = sorted({r.agent for r in res.agent_results})
    cited = {c for it in [*rep.findings, *rep.recommendations] for c in it.citations}
    badges = [
        # One source of truth: the computed score (retrieval strength, citation coverage, goal check).
        f":{CONF_COLOR[confidence_label(res.confidence)]}-badge[{confidence_label(res.confidence).title()} "
        f"confidence · {res.confidence:.0%}]",
        ":violet-badge[" + (", ".join(f"{AGENT_ICON.get(a, '')} {config.DOMAIN_LABELS[a]}" for a in agents)
                            or "no agents") + "]",
        f":gray-badge[📚 {len(cited)} cited of {len(res.evidence)} sources]",
        ":green-badge[✓ Resolved]" if res.ledger.resolved else ":orange-badge[◐ Partly resolved - details needed]",
    ]
    st.markdown(" ".join(badges))
    with st.container(border=True):
        st.markdown("##### 📋 Summary")
        st.markdown(rep.executive_summary)
    if getattr(res, "scope_note", ""):
        st.info(res.scope_note, icon="📭")


def _recommendations(rep, evidence: dict) -> None:
    if not rep.recommendations:
        return
    order = {"P1": 0, "P2": 1, "P3": 2}
    recs = sorted(rep.recommendations, key=lambda r: order.get(r.priority, 9))
    with st.expander(f"✅ **Recommendations** · {len(recs)}"):
        for n, r in enumerate(recs, 1):
            color, label = PRIORITY.get(r.priority, ("gray", r.priority))
            with st.container(border=True):
                st.markdown(f":{color}-badge[{label}] {_conf_badge(r.confidence)}")
                st.markdown(f"**{n}. {r.action}**")
                st.markdown(f"**Why:** {r.rationale}")
                a, b = st.columns(2)
                a.markdown(f"👤 **Owner / timeline:** {r.owner_timeline}")
                b.markdown(f"📏 **How to measure:** {r.kpi}")
                st.caption(_sources(r.citations, evidence))


def _findings(rep, evidence: dict) -> None:
    if not rep.findings:
        return
    confirmed = [f for f in rep.findings if f.status == "confirmed"]
    hypotheses = [f for f in rep.findings if f.status != "confirmed"]
    with st.expander(f"🔍 **Findings** · {len(confirmed)} confirmed, {len(hypotheses)} "
                     f"{'hypothesis' if len(hypotheses) == 1 else 'hypotheses'}"):
        for title, items, note in (("🟢 Confirmed by the documents", confirmed, None),
                                   ("🟡 Hypotheses - not yet confirmed", hypotheses,
                                    "Likely explanations that still need checking (e.g. diagnostics or data "
                                    "you can provide).")):
            if not items:
                continue
            st.markdown(f"**{title}**")
            if note:
                st.caption(note)
            for f in items:
                st.markdown(f"- {f.statement} {_conf_badge(f.confidence)}")
                st.caption("  " + _sources(f.citations, evidence))


def _simple_list(title: str, items: list[str], numbered: bool = False, note: str = "") -> None:
    if not items:
        return
    with st.expander(f"{title} · {len(items)}"):
        if note:
            st.caption(note)
        st.markdown("\n".join(f"{i}. {it}" if numbered else f"- {it}" for i, it in enumerate(items, 1)))


def _sources_section(res) -> None:
    rep = res.report
    cited = list(dict.fromkeys(c for it in [*rep.recommendations, *rep.findings] for c in it.citations))
    others = [r for r in res.evidence if r not in cited]
    with st.expander(f"📚 **Sources** · {len(cited)} cited, {len(others)} also consulted"):
        for group, refs in (("Cited in this report", cited), ("Also consulted", others[:25])):
            if not refs:
                continue
            st.markdown(f"**{group}**")
            rows = [{"ref": r, "source": res.evidence[r].citation(), "type": res.evidence[r].doc_type,
                     "relevance": round(res.evidence[r].score, 2), "excerpt": _short(" ".join(res.evidence[r].text.split()), 220),
                     "link": res.evidence[r].source} for r in refs if r in res.evidence]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"relevance": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                                                    format="%.2f"),
                                        "excerpt": st.column_config.TextColumn(width="large")})


def _agents_section(res) -> None:
    with st.expander("🤖 **How the agents worked** · routing, plan and tool calls"):
        for r in res.agent_results:
            st.markdown(f"**{AGENT_ICON.get(r.agent, '')} {config.DOMAIN_LABELS[r.agent]} agent** "
                        f"{_conf_badge(r.confidence_label)} :gray-badge[{len(r.tool_calls)} tool calls]")
            st.markdown(r.output.summary)
            if r.tool_calls:
                st.caption("Searched: " + " · ".join(
                    f"`{c['tool']}` {_short(json.dumps(c['args'], ensure_ascii=False), 90)}" for c in r.tool_calls))
            st.divider()
        st.markdown("**Task & progress ledger**")
        render_ledger(res.ledger, st.empty())
        u = res.usage
        st.caption(f"LLM calls {u.get('calls', 0)} · input tokens {u.get('input_tokens', 0):,} "
                   f"(cache read {u.get('cache_read_input_tokens', 0):,}, cache write "
                   f"{u.get('cache_creation_input_tokens', 0):,}) · output tokens {u.get('output_tokens', 0):,}")


# -- entry point ----------------------------------------------------------------------
def render_report(res) -> None:
    rep = res.report
    if getattr(res, "status", "answered") == "out_of_scope":
        st.info(f"**{rep.executive_summary}**", icon="📭")
        st.caption(f"This platform answers questions about: {KB_TOPICS}.")
        if res.agent_results:  # grounding backstop case: show what was searched
            with st.expander("Why? The specialists searched but found no supporting evidence"):
                render_ledger(res.ledger, st.empty())
        return

    _header(res)
    _recommendations(rep, res.evidence)
    _simple_list("🙋 **Information we need from you**", rep.information_needed,
                 note="Reply in the chat with these details and the manager will re-run the analysis with them.")
    _simple_list("➡️ **Next steps**", rep.next_steps, numbered=True)
    _findings(rep, res.evidence)
    _simple_list("⚠️ **Risks & dependencies**", rep.risks)
    _sources_section(res)
    _agents_section(res)
