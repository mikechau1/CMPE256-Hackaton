"""Similar-trial recommender over ClinicalTrials.gov exports.

Candidates come from the hybrid retriever (restricted to trial documents); they
are re-ranked with contextual evidence-quality boosts: completed status, posted
results, later phase and larger enrollment.
"""
from __future__ import annotations

import math

from ..retrieval.hybrid import get_retriever

PHASE_BOOST = {"PHASE4": 0.06, "PHASE3": 0.05, "PHASE2|PHASE3": 0.04, "PHASE2": 0.03}


def recommend_trials(need: str, condition_group: str | None = None, k: int = 5) -> list[dict]:
    retriever = get_retriever("clinical")
    filters = {"doc_type": "clinical_trial"}
    if condition_group:
        filters["condition_group"] = condition_group
    candidates = retriever.search(need, k=max(15, 3 * k), filters=filters, mode="hybrid_rerank")
    out = []
    for ev in candidates:
        m = retriever.metas[retriever.pos[ev.chunk_id]]
        boost, why = 0.0, []
        if m.get("status") == "COMPLETED":
            boost += 0.05
            why.append("completed")
        if str(m.get("has_results", "")).upper() == "YES":
            boost += 0.05
            why.append("results posted")
        pb = PHASE_BOOST.get(str(m.get("phase", "")), 0.0)
        if pb:
            boost += pb
            why.append(str(m.get("phase")).lower())
        try:
            enroll = int(float(m.get("enrollment") or 0))
        except ValueError:
            enroll = 0
        if enroll:
            boost += min(0.05, 0.01 * math.log10(enroll + 1))
            why.append(f"n={enroll}")
        out.append({"nct_id": m["doc_id"], "title": ev.title, "url": ev.source, "relevance": ev.score,
                    "score": round(ev.score + boost, 4), "status": m.get("status"), "phase": m.get("phase"),
                    "has_results": m.get("has_results"), "enrollment": enroll, "sponsor": m.get("sponsor"),
                    "condition_group": m.get("condition_group"), "why": ", ".join(why), "chunk_id": ev.chunk_id,
                    "snippet": ev.text[:500]})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:k]
