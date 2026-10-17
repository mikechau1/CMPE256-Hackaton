# Evaluation report

_Generated 2026-10-17 14:30_

## Routing accuracy (53 labelled test queries, 11 multi-domain)

| router | primary_accuracy | exact_match | jaccard | micro_f1 | macro_f1 |
|---|---|---|---|---|---|
| embedding_baseline | 0.943 | 0.774 | 0.887 | 0.892 | 0.899 |
| llm_router | 1.0 | 0.906 | 0.95 | 0.957 | 0.968 |

Per-domain F1 (LLM router): technology 0.878, clinical 0.963, sec 1.0, csr 1.0, procurement 1.0

## Retrieval (46 queries with phrase-verified gold chunks)

| mode | hit@1 | hit@5 | recall@10 | mrr | ndcg@10 | latency_s |
|---|---|---|---|---|---|---|
| bm25 | 0.3696 | 0.6957 | 0.65 | 0.5157 | 0.5049 | 0.009 |
| dense | 0.4783 | 0.7391 | 0.729 | 0.6006 | 0.5751 | 0.348 |
| hybrid | 0.5 | 0.913 | 0.7812 | 0.6658 | 0.6377 | 0.033 |
| hybrid_rerank | 0.5217 | 0.8696 | 0.8047 | 0.6766 | 0.637 | 1.55 |

Hybrid + rerank by domain:

| domain | n | hit@5 | mrr |
|---|---|---|---|
| technology | 12 | 1.0 | 0.7569 |
| sec | 16 | 0.8125 | 0.6901 |
| csr | 9 | 0.7778 | 0.6111 |
| clinical | 9 | 0.8889 | 0.6111 |

## Recommendation relevance

**Cisco components** (10 symptom cases, gold = correct component types): precision@5 0.9, hit@1 1.0, nDCG@5 0.931, series detected 1.0, compatible-part rate 1.0

**Agilent instruments** (4 requirement cases): hit@1 0.75, MRR 0.875

**Checkout recommender (Apriori, mlxtend)**: 2759 rules, a cart combination must appear in >= 2 orders, lift > 1.0. Held-out orders (200) are replayed as checkouts, adding one item at a time; hit@5 and precision@5 count recommendations that the shopper went on to add, measured only when recommendations were shown (coverage).

| model | coverage | hit@5 | precision@5 | next_item_hit@5 | steps |
|---|---|---|---|---|---|
| apriori | 0.5698 | 0.2282 | 0.0607 | 0.104 | 523 |
| popularity (baseline) | 1.0 | 0.2906 | 0.0639 | 0.1625 | 523 |

By cart size (Apriori):

| cart_size | coverage | hit@5 | precision@5 | next_item_hit@5 | steps |
|---|---|---|---|---|---|
| 1 | 1.0 | 0.285 | 0.062 | 0.12 | 200 |
| 2 | 0.6282 | 0.1122 | 0.0582 | 0.0714 | 156 |
| 3 | 0.0 | 0.0 | 0.0 | 0.0 | 108 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 | 59 |

Random baseline next-item hit@5: 0.1282

## End-to-end response quality (11 consulting requests, judge: claude-haiku-5-5)

| routing_recall | must_mention_rate | citation_coverage | citation_precision | groundedness | relevance | actionability | uncertainty_handling | confidence | latency_s | llm_calls | confidence_vs_groundedness_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 0.955 | 0.751 | 0.769 | 3.0 | 4.455 | 4.0 | 4.818 | 0.623 | 73.7 | 9.909 | 0.14 |

| id | routed | groundedness | relevance | actionability | uncertainty_handling | citation_coverage | citation_precision | confidence | iterations | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | ['technology'] | 2 | 4 | 4 | 5 | 0.5 | 0.667 | 0.419 | 1 | 42.8 |
| E2 | ['technology'] | 2 | 4 | 4 | 4 | 0.545 | 0.667 | 0.629 | 1 | 52.9 |
| E3 | ['clinical'] | 3 | 4 | 4 | 5 | 0.8 | 0.875 | 0.645 | 1 | 71.9 |
| E4 | ['sec'] | 3 | 5 | 4 | 4 | 0.938 | 0.5 | 0.771 | 2 | 144.2 |
| E5 | ['csr', 'technology'] | 4 | 5 | 4 | 5 | 0.81 | 0.875 | 0.542 | 1 | 66.5 |
| E6 | ['procurement'] | 3 | 4 | 4 | 5 | 0.444 | 1.0 | 0.414 | 1 | 47.3 |
| E7 | ['clinical'] | 3 | 4 | 4 | 5 | 0.889 | 0.75 | 0.715 | 1 | 56.0 |
| E8 | ['technology', 'sec', 'csr', 'clinical'] | 4 | 5 | 4 | 5 | 0.763 | 0.75 | 0.616 | 1 | 110.6 |
| E9 | ['technology'] | 3 | 5 | 4 | 5 | 0.727 | 0.75 | 0.62 | 1 | 46.8 |
| E10 | ['sec'] | 3 | 5 | 4 | 5 | 0.909 | 1.0 | 0.786 | 2 | 91.8 |
| E11 | ['sec', 'technology'] | 3 | 4 | 4 | 5 | 0.933 | 0.625 | 0.695 | 1 | 79.9 |
