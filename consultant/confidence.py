"""Confidence / uncertainty scoring.

Agent confidence blends four observable signals rather than trusting the model's
self-report alone:
  * retrieval strength  - mean of the top cross-encoder relevance scores the agent saw
  * citation coverage   - share of findings/recommendations backed by a valid evidence ref
  * confirmed ratio     - confirmed findings vs hypotheses
  * self-assessment     - the agent's own 0-1 estimate
"""
from __future__ import annotations

from .schemas import AgentResultOut, ConfidenceLabel, Evidence

WEIGHTS = {"retrieval": 0.35, "coverage": 0.3, "confirmed": 0.1, "self": 0.25}


def label(score: float) -> ConfidenceLabel:
    return "high" if score >= 0.7 else "medium" if score >= 0.45 else "low"


def citation_coverage(out: AgentResultOut, valid_refs: set[str]) -> float:
    items = [*out.findings, *out.recommendations]
    if not items:
        return 0.0
    return sum(1 for it in items if any(c in valid_refs for c in it.citations)) / len(items)


def agent_confidence(out: AgentResultOut, evidence: list[Evidence]) -> tuple[float, dict]:
    valid = {e.ref for e in evidence}
    # Search results carry reranker probabilities; recommender rows carry their own scores.
    scores = sorted((e.score for e in evidence if e.doc_type not in {"recommendation"}), reverse=True)[:5]
    retrieval = sum(scores) / len(scores) if scores else 0.0
    coverage = citation_coverage(out, valid)
    confirmed = (sum(f.status == "confirmed" for f in out.findings) / len(out.findings)) if out.findings else 0.0
    self_conf = max(0.0, min(1.0, out.self_confidence))
    parts = {"retrieval": round(retrieval, 3), "coverage": round(coverage, 3),
             "confirmed": round(confirmed, 3), "self": round(self_conf, 3)}
    score = sum(WEIGHTS[k] * v for k, v in parts.items())
    if out.missing_information:  # unresolved unknowns cap certainty
        score = min(score, 0.69)
    return round(score, 3), parts


def overall_confidence(agent_scores: list[float], route_conf: float, goal_score: float) -> float:
    if not agent_scores:
        return 0.0
    return round(0.5 * (sum(agent_scores) / len(agent_scores)) + 0.2 * route_conf + 0.3 * goal_score, 3)
