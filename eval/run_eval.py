"""Evaluation harness: routing, retrieval, recommendation and end-to-end response quality.

Run:  python -m eval.run_eval                  # everything (LLM parts need ANTHROPIC_API_KEY)
      python -m eval.run_eval --only retrieval,recommendation
      python -m eval.run_eval --skip-llm       # offline metrics only
Writes eval/results/<section>.json and eval/results/report.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from consultant import config

DATA = Path(__file__).parent / "datasets"
OUT = Path(__file__).parent / "results"


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def has_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def ndcg(rels: list[int], k: int, n_relevant: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg / ideal if ideal else 0.0


# --- Retrieval ------------------------------------------------------------------
def eval_retrieval() -> dict:
    from consultant.retrieval.hybrid import get_retriever
    items = load("retrieval.jsonl")
    modes = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    per_mode = {m: {"hit@1": [], "hit@5": [], "recall@10": [], "mrr": [], "ndcg@10": [], "latency_s": []} for m in modes}
    per_domain: dict[str, dict[str, list]] = {}
    skipped = []
    for it in items:
        try:
            r = get_retriever(it["domain"])
        except Exception as e:
            skipped.append({"id": it["id"], "reason": f"index missing: {e}"})
            continue
        phrase = norm(it["answer_phrase"])
        gold = {cid for cid, t in zip(r.ids, r.texts) if phrase in norm(t)}
        if not gold:
            skipped.append({"id": it["id"], "reason": "answer phrase not found in index"})
            continue
        for m in modes:
            t0 = time.time()
            hits = [e.chunk_id for e in r.search(it["query"], k=10, mode=m)]
            per_mode[m]["latency_s"].append(time.time() - t0)
            rels = [1 if h in gold else 0 for h in hits]
            first = next((i for i, x in enumerate(rels) if x), None)
            per_mode[m]["hit@1"].append(1 if rels[:1] == [1] else 0)
            per_mode[m]["hit@5"].append(1 if any(rels[:5]) else 0)
            per_mode[m]["recall@10"].append(sum(rels) / min(len(gold), 10))
            per_mode[m]["mrr"].append(1 / (first + 1) if first is not None else 0.0)
            per_mode[m]["ndcg@10"].append(ndcg(rels, 10, len(gold)))
            if m == "hybrid_rerank":
                per_domain.setdefault(it["domain"], {"hit@5": [], "mrr": []})
                per_domain[it["domain"]]["hit@5"].append(per_mode[m]["hit@5"][-1])
                per_domain[it["domain"]]["mrr"].append(per_mode[m]["mrr"][-1])

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    result = {"n_queries": len(items) - len(skipped), "skipped": skipped,
              "by_mode": {m: {k: avg(v) for k, v in d.items()} for m, d in per_mode.items()},
              "by_domain_hybrid_rerank": {d: {k: avg(v) for k, v in x.items()} | {"n": len(x["mrr"])}
                                          for d, x in per_domain.items()}}
    save("retrieval", result)
    return result


# --- Routing --------------------------------------------------------------------
def multilabel_metrics(gold: list[list[str]], pred: list[list[str]]) -> dict:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    exact = jacc = primary = 0
    for g, p in zip(gold, pred):
        gs, ps = set(g), set(p)
        exact += gs == ps
        jacc += len(gs & ps) / len(gs | ps)
        primary += bool(p) and p[0] in gs
        for d in config.DOMAINS:
            tp[d] += d in gs and d in ps
            fp[d] += d not in gs and d in ps
            fn[d] += d in gs and d not in ps
    per = {}
    for d in config.DOMAINS:
        prec = tp[d] / (tp[d] + fp[d]) if tp[d] + fp[d] else 0.0
        rec = tp[d] / (tp[d] + fn[d]) if tp[d] + fn[d] else 0.0
        per[d] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p, micro_r = TP / (TP + FP) if TP + FP else 0, TP / (TP + FN) if TP + FN else 0
    n = len(gold)
    return {"n": n, "primary_accuracy": round(primary / n, 3), "exact_match": round(exact / n, 3),
            "jaccard": round(jacc / n, 3),
            "micro_f1": round(2 * micro_p * micro_r / (micro_p + micro_r), 3) if micro_p + micro_r else 0.0,
            "macro_f1": round(sum(v["f1"] for v in per.values()) / len(per), 3), "per_domain": per}


def confusion(gold: list[list[str]], pred: list[list[str]]) -> dict:
    m = {g: {p: 0 for p in config.DOMAINS} for g in config.DOMAINS}
    for g, p in zip(gold, pred):
        if p:
            m[g[0]][p[0]] += 1
    return m


def eval_routing(use_llm: bool) -> dict:
    from consultant.routing import EmbeddingRouter, llm_route
    rows = load("routing.jsonl")
    train = [(r["query"], r["labels"]) for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    gold = [r["labels"] for r in test]
    result = {}
    base = EmbeddingRouter(train)
    preds = [base.route(r["query"]).domains for r in test]
    result["embedding_baseline"] = multilabel_metrics(gold, preds) | {"confusion_primary": confusion(gold, preds)}
    if use_llm:
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()
        with ThreadPoolExecutor(8) as pool:
            outs = list(pool.map(lambda r: llm_route(r["query"]), test))
        preds = [o.domains for o in outs]
        result["llm_router"] = multilabel_metrics(gold, preds) | {
            "confusion_primary": confusion(gold, preds),
            "mean_confidence": round(sum(o.confidence for o in outs) / len(outs), 3),
            "avg_latency_s": round((time.time() - t0) / len(test) * 8, 2),
            "errors": [{"query": r["query"], "gold": g, "pred": p} for r, g, p in zip(test, gold, preds)
                       if set(g) != set(p)]}
    save("routing", result)
    return result


# --- Recommendation ---------------------------------------------------------------
def eval_recommendation() -> dict:
    from consultant.recommenders.basket import evaluate_checkout_simulation
    from consultant.recommenders.components import recommend_components
    from consultant.recommenders.instruments import recommend_instruments
    cases = load("recommendation.jsonl")
    comp, inst = [], []
    for c in cases:
        if c["kind"] == "component":
            r = recommend_components(c["query"], k=5)
            types = [p["component_type"] for p in r["parts"]]
            rels = [1 if t in c["gold_types"] else 0 for t in types]
            comp.append({"id": c["id"], "precision@5": sum(rels) / 5, "hit@1": rels[0] if rels else 0,
                         "ndcg@5": ndcg(rels, 5, 5),
                         "series_correct": r["series_detected"] == [c["series"]],
                         "compatible_rate": sum(p["series"] == c["series"] for p in r["parts"]) / max(1, len(r["parts"])),
                         "top_hypothesis": r["hypotheses"][0]["fault_mode"], "top_types": types})
        else:
            r = recommend_instruments(c["query"], k=3)
            names = [p["name"].lower() for p in r["products"]]
            rank = next((i for i, n in enumerate(names) if c["gold_contains"] in n), None)
            inst.append({"id": c["id"], "hit@1": int(rank == 0), "mrr": 1 / (rank + 1) if rank is not None else 0.0,
                         "top": names[:1]})

    def mean(rows, k):
        return round(sum(float(r[k]) for r in rows) / len(rows), 3) if rows else None
    result = {
        "components": {"n": len(comp), **{k: mean(comp, k) for k in ("precision@5", "hit@1", "ndcg@5", "series_correct",
                                                                       "compatible_rate")}, "cases": comp},
        "instruments": {"n": len(inst), "hit@1": mean(inst, "hit@1"), "mrr": mean(inst, "mrr"), "cases": inst},
        "checkout_basket": evaluate_checkout_simulation(k=5),
    }
    save("recommendation", result)
    return result


# --- End-to-end response quality (LLM-as-judge) ---------------------------------------
class JudgeOut(BaseModel):
    groundedness: int = Field(description="1-5: claims are supported by the cited evidence excerpts.")
    relevance: int = Field(description="1-5: answers the customer's actual question.")
    actionability: int = Field(description="1-5: concrete next steps with owners and KPIs.")
    uncertainty_handling: int = Field(description="1-5: separates confirmed facts from hypotheses; asks for missing info.")
    unsupported_claims: list[str] = Field(description="Claims not supported by the provided evidence.")
    rationale: str


class SupportOut(BaseModel):
    supported: bool
    reason: str


JUDGE_SYSTEM = """You are a strict evaluator of AI consulting reports. Score only on the evidence excerpts
provided; anything not traceable to them is unsupported. Use the full 1-5 range."""


def eval_e2e(limit: int | None) -> dict:
    from consultant.agents.manager import ManagerAgent
    from consultant.llm import LLM
    cases = load("e2e.jsonl")[:limit] if limit else load("e2e.jsonl")
    judge = LLM()
    rows = []
    for c in cases:
        t0 = time.time()
        llm = LLM()
        try:
            res = ManagerAgent(llm=llm).run(c["query"])
        except Exception as e:
            rows.append({"id": c["id"], "error": str(e)})
            continue
        latency = time.time() - t0
        rep = res.report
        text = json.dumps(rep.model_dump())
        items = [*rep.findings, *rep.recommendations]
        cited = sum(1 for it in items if it.citations)
        # Citation precision: is each cited claim supported by the evidence it cites (up to 3 refs)?
        # Checked for the first 8 cited findings/recommendations.
        cited_items = [it for it in items if any(r in res.evidence for r in it.citations)][:8]
        support = []
        for it in cited_items:
            claim = getattr(it, "statement", None) or f"{it.action} - {it.rationale}"
            refs = [r for r in it.citations if r in res.evidence][:3]
            ev_text = "\n\n".join(f"[{r}] {res.evidence[r].citation()}:\n{res.evidence[r].text[:2000]}" for r in refs)
            s = judge.parse(SupportOut, f"Claim: {claim}\n\nCited evidence:\n{ev_text}\n\n"
                            "Is the claim's factual content supported (fully or substantially) by the cited evidence? "
                            "Recommendations count as supported when the facts motivating them are in the evidence.",
                            system=JUDGE_SYSTEM, model=config.FAST_MODEL, effort="low", max_tokens=1000)
            support.append(s.supported)
        # Give the judge every cited excerpt (not just the first N registered).
        cited_refs = list(dict.fromkeys(r for it in items for r in it.citations if r in res.evidence))
        excerpts = {ref: f"{res.evidence[ref].citation()}: {res.evidence[ref].text[:900]}" for ref in cited_refs[:60]}
        j = judge.parse(JudgeOut, f"Customer request:\n{c['query']}\n\nReport:\n{text}\n\nEvidence excerpts:\n"
                        f"{json.dumps(excerpts, indent=1)}", system=JUDGE_SYSTEM, model=config.FAST_MODEL,
                        effort="medium", max_tokens=3000)
        routed = res.ledger.route.domains
        rows.append({
            "id": c["id"], "routed": routed,
            "routing_recall": len(set(c["expected_domains"]) & set(routed)) / len(c["expected_domains"]),
            "must_mention_rate": sum(m.lower() in text.lower() for m in c["must_mention"]) / len(c["must_mention"]),
            "citation_coverage": round(cited / len(items), 3) if items else 0.0,
            "citation_precision": round(sum(support) / len(support), 3) if support else None,
            "groundedness": j.groundedness, "relevance": j.relevance, "actionability": j.actionability,
            "uncertainty_handling": j.uncertainty_handling, "unsupported_claims": j.unsupported_claims,
            "confidence": res.confidence, "iterations": res.ledger.iterations, "resolved": res.ledger.resolved,
            "latency_s": round(latency, 1), "llm_calls": llm.usage["calls"],
            "input_tokens": llm.usage["input_tokens"], "output_tokens": llm.usage["output_tokens"],
            "judge_rationale": j.rationale})
        print(f"  {c['id']}: routed={routed} grounded={j.groundedness} relevance={j.relevance} "
              f"cit_prec={rows[-1]['citation_precision']} {latency:.0f}s")
    ok = [r for r in rows if "error" not in r]
    keys = ["routing_recall", "must_mention_rate", "citation_coverage", "citation_precision", "groundedness", "relevance",
            "actionability", "uncertainty_handling", "confidence", "latency_s", "llm_calls"]
    summary = {k: round(sum(r[k] for r in ok if r[k] is not None) / max(1, sum(r[k] is not None for r in ok)), 3)
               for k in keys} if ok else {}
    # Calibration check: does our confidence track judged groundedness?
    if len(ok) >= 3:
        import numpy as np
        summary["confidence_vs_groundedness_corr"] = round(float(np.corrcoef(
            [r["confidence"] for r in ok], [r["groundedness"] for r in ok])[0, 1]), 3)
    result = {"n": len(cases), "errors": len(rows) - len(ok), "summary": summary, "cases": rows}
    save("e2e", result)
    return result


# --- Report ---------------------------------------------------------------------
def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def write_report() -> None:
    def get(name):
        p = OUT / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    lines = ["# Evaluation report", "", f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    rt = get("routing")
    if rt:
        test = [r for r in load("routing.jsonl") if r["split"] == "test"]
        multi = sum(len(r["labels"]) > 1 for r in test)
        lines += [f"## Routing accuracy ({len(test)} labelled test queries, {multi} multi-domain)", ""]
        rows = [{"router": k, **{m: v[m] for m in ("primary_accuracy", "exact_match", "jaccard", "micro_f1", "macro_f1")}}
                for k, v in rt.items()]
        lines += [md_table(rows, ["router", "primary_accuracy", "exact_match", "jaccard", "micro_f1", "macro_f1"]), ""]
        best = rt.get("llm_router") or rt["embedding_baseline"]
        lines += ["Per-domain F1 (" + ("LLM router" if "llm_router" in rt else "baseline") + "): "
                  + ", ".join(f"{d} {v['f1']}" for d, v in best["per_domain"].items()), ""]
    rv = get("retrieval")
    if rv:
        lines += [f"## Retrieval ({rv['n_queries']} queries with phrase-verified gold chunks)", ""]
        rows = [{"mode": m, **v} for m, v in rv["by_mode"].items()]
        for r in rows:
            r["latency_s"] = round(r["latency_s"], 3) if r.get("latency_s") is not None else None
        lines += [md_table(rows, ["mode", "hit@1", "hit@5", "recall@10", "mrr", "ndcg@10", "latency_s"]), ""]
        lines += ["Hybrid + rerank by domain:", "",
                  md_table([{"domain": d, **v} for d, v in rv["by_domain_hybrid_rerank"].items()],
                           ["domain", "n", "hit@5", "mrr"]), ""]
        if rv["skipped"]:
            lines += [f"Skipped: {rv['skipped']}", ""]
    rc = get("recommendation")
    if rc:
        c, i = rc["components"], rc["instruments"]
        lines += ["## Recommendation relevance", "",
                  f"**Cisco components** ({c['n']} symptom cases, gold = correct component types): "
                  f"precision@5 {c['precision@5']}, hit@1 {c['hit@1']}, nDCG@5 {c['ndcg@5']}, "
                  f"series detected {c['series_correct']}, compatible-part rate {c['compatible_rate']}", "",
                  f"**Agilent instruments** ({i['n']} requirement cases): hit@1 {i['hit@1']}, MRR {i['mrr']}", ""]
        b = rc.get("checkout_basket")
        if b:
            mdl = b["apriori"]["model"]
            cols = ["coverage", "hit@5", "precision@5", "next_item_hit@5", "steps"]
            lines += [f"**Checkout recommender (Apriori, mlxtend)**: {mdl['n_rules']} rules, a cart combination must "
                      f"appear in >= {mdl['min_combination_orders']} orders, lift > {mdl['min_lift']}. Held-out orders "
                      f"({b['test_orders']}) are replayed as checkouts, adding one item at a time; hit@5 and "
                      "precision@5 count recommendations that the shopper went on to add, measured only when "
                      "recommendations were shown (coverage).", "",
                      md_table([{"model": "apriori", **{k: b["apriori"][k] for k in cols}},
                                {"model": "popularity (baseline)", **b["popularity_baseline"]}], ["model"] + cols), "",
                      "By cart size (Apriori):", "",
                      md_table([{"cart_size": t, **v} for t, v in b["apriori_by_cart_size"].items()], ["cart_size"] + cols),
                      f"\nRandom baseline next-item hit@5: {b['random_baseline_next_item_hit']}", ""]
    e2 = get("e2e")
    if e2 and e2.get("summary"):
        s = e2["summary"]
        lines += [f"## End-to-end response quality ({e2['n']} consulting requests, judge: {config.FAST_MODEL})", "",
                  md_table([s], list(s.keys())), "",
                  md_table(e2["cases"], ["id", "routed", "groundedness", "relevance", "actionability",
                                         "uncertainty_handling", "citation_coverage", "citation_precision",
                                         "confidence", "iterations", "latency_s"]), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {OUT / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma list: routing,retrieval,recommendation,e2e")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--e2e-limit", type=int)
    args = ap.parse_args()
    sections = set(args.only.split(",")) if args.only else {"routing", "retrieval", "recommendation", "e2e"}
    use_llm = has_key() and not args.skip_llm
    if not use_llm:
        print("No API key (or --skip-llm): LLM router and end-to-end evaluation are skipped.")
    if "retrieval" in sections:
        print("Retrieval...", json.dumps(eval_retrieval()["by_mode"], indent=1))
    if "routing" in sections:
        r = eval_routing(use_llm)
        print("Routing...", {k: (v["primary_accuracy"], v["exact_match"], v["macro_f1"]) for k, v in r.items()})
    if "recommendation" in sections:
        r = eval_recommendation()
        print("Recommendation...", {k: v for k, v in r["components"].items() if k != "cases"})
    if "e2e" in sections and use_llm:
        print("End-to-end...")
        eval_e2e(args.e2e_limit)
    write_report()


if __name__ == "__main__":
    main()
