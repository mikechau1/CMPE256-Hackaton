"""Live health check of every domain agent through the full Manager pipeline.

Guards against "empty" engagements (0 agents / 0 evidence, e.g. the manager bouncing a vague
request back as a clarification). Each case must:
  * route to the expected domain(s) and actually run those agents, with no failed subtasks
  * retrieve evidence and produce cited findings plus recommendations
Cases mix specific questions with the deliberately vague example requests from the brief.

Run: python -m eval.agent_check [--workers 4] [--only clinical,sec]
Writes eval/results/agent_check.json. Costs a few cents with the default Haiku agent model.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from consultant import config
from consultant.agents.manager import ManagerAgent
from consultant.llm import LLM
from consultant.schemas import OUT_OF_SCOPE_MESSAGE

OUT = ["out_of_scope"]  # expected outcome marker for irrelevant requests

CASES = [
    # (id, expected domains, request)
    ("tech-brief", ["technology"], "Which replacement components should we order to resolve a Cisco router failure?"),
    ("tech-vague", ["technology"], "Our network keeps going down. What should we do?"),
    ("tech-specific", ["technology"], "Our Catalyst 8300 shows CRC errors and the link flaps on the SFP port. "
                                      "Which parts should we replace and what should we check first?"),
    ("tech-gcp", ["technology"], "How could we use MCP servers and RAG on Google Cloud for our internal documents?"),
    ("clin-brief", ["clinical"], "Which diagnostic instrument or replacement component best meets our laboratory "
                                 "requirements?"),
    ("clin-vague", ["clinical"], "What instrument should our lab buy?"),
    ("clin-specific", ["clinical"], "We need to digitize histopathology slides for primary diagnosis at high "
                                    "throughput. Which Agilent instrument fits?"),
    ("clin-trials", ["clinical"], "Which completed clinical trials evaluated mobile apps for type 2 diabetes?"),
    ("sec-brief", ["sec"], "What are the financial, competitive, and operational risks associated with these "
                           "companies?"),
    ("sec-specific", ["sec"], "How did Cisco's fiscal 2026 revenue compare with fiscal 2025?"),
    ("sec-alphabet", ["sec"], "What antitrust and regulatory risks does Alphabet disclose?"),
    ("csr-brief", ["csr"], "Which technology investments could improve food security and sustainability outcomes?"),
    ("csr-vague", ["csr"], "How can our company help fight hunger?"),
    ("proc-specific", ["procurement"], "We are ordering a Bosch B5512 control panel. What else do customers add?"),
    ("proc-vague", ["procurement"], "What security products are usually bought together?"),
    ("cross", ["technology", "sec", "csr"], "Should we standardise our branch network on Cisco given its financial "
                                            "risks, and how would that support our food-security ESG goals?"),
    # Irrelevant requests: must be declined with the "no information" message and no agents.
    ("oos-49ers", OUT, "What is the 49ers record right now?"),
    ("oos-recipe", OUT, "What is a good recipe for lasagna?"),
    ("oos-sports", OUT, "Who won the 2022 FIFA World Cup?"),
    ("oos-tesla", OUT, "What was Tesla's revenue in 2025?"),
    ("oos-medical", OUT, "I have a headache, what pill should I take?"),
    ("oos-code", OUT, "Write a Python function to reverse a linked list."),
    # Mixed: answer the covered part, flag the uncovered part.
    ("mixed", ["technology"], "Our Catalyst 8300 keeps rebooting - what should we check? Also, what's a good "
                              "lasagna recipe?"),
]


def run_case(case: tuple) -> dict:
    cid, expected, request = case
    llm = LLM()
    t0 = time.time()
    try:
        res = ManagerAgent(llm=llm).run(request)
    except Exception as e:  # report, don't crash the sweep
        return {"id": cid, "request": request, "error": f"{type(e).__name__}: {e}", "passed": False,
                "problems": ["engagement raised"]}
    rep = res.report
    ran = {r.agent for r in res.agent_results}
    if expected == OUT:
        problems = []
        if res.status != "out_of_scope":
            problems.append(f"answered instead of declining (status={res.status})")
        if rep.executive_summary != OUT_OF_SCOPE_MESSAGE:
            problems.append("wrong decline message")
        if res.agent_results or rep.recommendations:
            problems.append("agents ran / recommendations shown for an irrelevant request")
        return {"id": cid, "request": request, "expected": expected, "status": res.status,
                "routed": res.ledger.route.domains, "agents_ran": sorted(ran), "evidence": len(res.evidence),
                "findings": 0, "recommendations": len(rep.recommendations), "cited_share": None,
                "clarification_flagged": False, "seconds": round(time.time() - t0, 1), "llm_calls": llm.usage["calls"],
                "cost_usd": None, "summary": rep.executive_summary, "problems": problems, "passed": not problems}
    failed = [s for s in res.ledger.subtasks if s.status == "failed"]
    items = [*rep.findings, *rep.recommendations]
    problems = []
    if not res.agent_results:
        problems.append("no agents ran")
    missing = [d for d in expected if d not in ran]
    if missing:
        problems.append(f"expected agent(s) did not run: {missing}")
    if failed:
        problems.append(f"failed subtasks: {[(s.agent, s.summary[:80]) for s in failed]}")
    if not res.evidence:
        problems.append("no evidence retrieved")
    if not rep.recommendations:
        problems.append("no recommendations")
    if not rep.findings:
        problems.append("no findings")
    if not any(it.citations for it in items):
        problems.append("nothing cited")
    if res.status != "answered":
        problems.append(f"in-scope request was not answered (status={res.status})")
    if cid == "mixed" and "not relevant to the knowledge base" not in res.scope_note:
        problems.append("uncovered part of a mixed request was not flagged")
    u = llm.usage
    return {
        "id": cid, "request": request, "expected": expected, "routed": res.ledger.route.domains,
        "clarification_flagged": res.ledger.route.needs_clarification, "agents_ran": sorted(ran),
        "evidence": len(res.evidence), "findings": len(rep.findings), "recommendations": len(rep.recommendations),
        "cited_share": round(sum(bool(it.citations) for it in items) / len(items), 2) if items else 0.0,
        "information_needed": rep.information_needed[:3], "confidence": res.confidence,
        "seconds": round(time.time() - t0, 1), "llm_calls": u["calls"],
        "cost_usd": round(u["input_tokens"] / 1e6 * 0.10 + u.get("cache_creation_input_tokens", 0) / 1e6 * 0.125
                          + u["cache_read_input_tokens"] / 1e6 * 0.01 + u["output_tokens"] / 1e6 * 0.50, 4)
        if config.AGENT_MODEL == "claude-haiku-5-5" else None,
        "status": res.status, "scope_note": res.scope_note,
        "summary": rep.executive_summary[:300], "problems": problems, "passed": not problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", help="comma list of id prefixes, e.g. clin,sec")
    args = ap.parse_args()
    cases = [c for c in CASES if not args.only or c[0].split("-")[0] in args.only.split(",")]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_case, cases))
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark} {r['id']:14} routed={r.get('routed')} ran={r.get('agents_ran')} "
              f"evidence={r.get('evidence')} findings={r.get('findings')} recs={r.get('recommendations')} "
              f"cited={r.get('cited_share')} clarify_flag={r.get('clarification_flagged')} "
              f"{r.get('seconds')}s ${r.get('cost_usd')}" + ("" if r["passed"] else f"  <- {r['problems']}"))
    passed = sum(r["passed"] for r in rows)
    cost = sum(r.get("cost_usd") or 0 for r in rows)
    out = Path(__file__).parent / "results" / "agent_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"agent_model": config.AGENT_MODEL, "passed": passed, "total": len(rows),
                               "cases": rows}, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(rows)} passed - total about ${cost:.3f} - details in {out}")


if __name__ == "__main__":
    main()
