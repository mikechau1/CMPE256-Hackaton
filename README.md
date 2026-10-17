# AI Management Consulting Platform: Multi-Agent RAG + Recommenders

CMPE256 Hackathon prototype. A **Management Consulting Manager Agent** classifies customer
requests and routes them to specialist agents. It tracks the work in a **task & progress
ledger** and loops on a **"resolved?" goal check**. When only the customer can supply a missing
fact, it asks the **human participant**. Finally it consolidates the evidence into
recommendations with citations, confidence, KPIs and next steps.

```
Input ─► Manager Agent ─ route (intent detection, multi-label + confidence)
            │            ─ plan  (ledger: resolution approach + subtasks)
            │            ─ dispatch in parallel ─► Technology | Clinical | SEC | CSR | Procurement
            │            ─ evaluate goal ── No ─► follow-up subtasks (≤2 loops) or ask the human
            │                          └─ Yes ─► synthesize consulting report
            ▼
  Report: executive summary · findings (confirmed vs hypothesis, cited) · prioritized actions
          with owner/timeline + KPI · risks · information needed · next steps · confidence
```

| Agent | Knowledge sources | Tools |
|---|---|---|
| **Technology** | Cisco Catalyst 8300 / 8200 / 8000V, ISR 4000, ASR 1000 datasheets + product pages; Google Cloud MCP & RAG pages | `search_tech_docs`, `recommend_components`, `lookup_part` |
| **Clinical** | 20 research PDFs (COVID, diabetes, heart attack, knee), ~8.7k ClinicalTrials.gov studies, Agilent product pages | `search_clinical_docs`, `recommend_trials`, `recommend_instruments`, `search_agilent_10k` |
| **SEC** | Form 10-K PDFs in `UnitedStatesSecuritiesandExchangeCommission(SEC)/<Company>/`: Cisco FY2025 + FY2026, Agilent FY2025, Alphabet FY2024 + FY2025. Split by Item and cited by page. EDGAR HTML is a fallback for anything missing. | `search_filings` (company / fiscal year / section filters), `compare_companies`, `list_filings` |
| **Corporate Responsibility** | FAO SOFI 2023 + 2024 (food security & nutrition) | `search_food_security`, `search_tech_docs` |
| **Procurement** | 1,000 security/fire-alarm orders (Retail CSV), via the Apriori checkout recommender | `find_products`, `recommend_bundle`, `top_association_rules` |

## How it meets the challenge

| Requirement | Implementation |
|---|---|
| Multi-agent orchestration | `consultant/agents/manager.py`: route, then plan, then parallel dispatch, then goal-evaluation loop, then synthesis. Ledger events stream live to the UI. |
| Domain knowledge grounding | `consultant/ingest/*`: PDFs are indexed per page. 10-Ks are split by Item (TOC and cross-references skipped, tables flattened to `a \| b \| c`). Web pages have boilerplate stripped. Trials become one document each. |
| RAG with citations | `consultant/retrieval/hybrid.py`: BM25 plus dense `bge-small` embeddings, merged with Reciprocal Rank Fusion, then reranked by a cross-encoder. Every chunk gets a ref such as `SEC-12`, and refs the model invents are removed. |
| Recommendation system | `consultant/recommenders/`:<br>• Cisco parts: a series-compatibility filter, then fault-mode priors, then semantic similarity, with diversity re-ranking.<br>• Agilent instruments: similarity plus requirement-constraint matching.<br>• Clinical trials: hybrid retrieval plus evidence-quality boosts.<br>• Checkout baskets: **Apriori** (mlxtend). For the current cart it uses only rules whose antecedent is the *exact* cart combination with lift > 1.1. A cart that never appeared (or appeared in fewer than 2 orders) gets no recommendations. Served as a FastAPI microservice; see [Checkout recommender](#checkout-recommender-market-basket-microservice). |
| Cross-domain reasoning | The planner gives each agent its own slice of the request. Agents can call tools from other domains (CSR uses tech docs; Clinical uses the Agilent 10-K). The synthesizer merges and deduplicates the findings. |
| Explainability & uncertainty | Each finding is labeled confirmed or hypothesis. Hardware causes stay hypotheses and come with the CLI diagnostics that would confirm them. Confidence combines retrieval strength, citation coverage, the confirmed ratio, the agent's self-assessment and the goal score (`consultant/confidence.py`). |
| Evaluation | `eval/run_eval.py` measures routing, retrieval, recommendations and end-to-end quality, with an LLM judge for the last (results below). |
| Business value | Each recommendation has an action, a reason, a priority (P1–P3), an owner and timeline, a KPI, cited evidence, and a confidence level. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # set ANTHROPIC_API_KEY and SEC_USER_AGENT ("Name email")

python -m consultant.ingest.fetch_sources   # Cisco/Agilent/GCP pages+PDFs (+ EDGAR 10-Ks as fallback) -> data/raw
python -m consultant.ingest.cisco_catalog   # Cisco parts catalog -> data/processed/cisco_catalog.json
python -m consultant.ingest.build_index     # chunks + Chroma per domain -> data/index (clinical ~15 min on CPU)

streamlit run app.py
```

* **Models.** `claude-haiku-5-5` runs all agents by default because it is cheap: a full
  4-agent cross-domain engagement costs about $0.03. Set `AGENT_MODEL=claude-sonnet-5-5` in
  `.env` for higher-quality reports. Sonnet also enables the server-side refusal fallback.
  Embeddings and reranking run locally on CPU.
* **Source access.** cisco.com and agilent.com block scripted clients (HTTP 403). The fetcher
  falls back to Wayback Machine snapshots and records `fetched_from` in each `.meta.json`.
* **Tests.** `python -m pytest tests` runs 20 unit tests: chunking, RRF, the 10-K splitter,
  basket math, fault-mode matching, and manager orchestration with a fake LLM (routing dedupe,
  ledger states, loop termination, ask-human, hallucinated-citation removal).

### Adding SEC filings
Drop 10-K or 10-Q PDFs into `UnitedStatesSecuritiesandExchangeCommission(SEC)/<CompanyName>/`, then run
`python -m consultant.ingest.build_index --domain sec`. The company comes from the folder name.
The form type and fiscal year are read from the cover page ("For the fiscal year ended ...").
Item sections are detected after the table of contents. The SEC agent's tool options and its
list of available filings update automatically. When a local PDF and an EDGAR download cover the
same company and year, the PDF wins.

## Checkout recommender (market-basket microservice)

This implements the retailer brief: recommendations at checkout that update as the cart changes.

1. **Empty cart:** nothing is shown.
2. **First item added:** Apriori rules {item} → X are ranked by confidence, with lift > 1.1 required
   (at least 10% more likely than average).
   Example: B5512 → Hikvision NVR, 13 of 92 orders, lift 1.46.
3. **Each further add** (a recommended item or anything else): recommendations are recomputed for
   the **exact current combination**. Example: {B5512, 6160} → Hikvision 4MP camera, lift 2.97.
4. **Combination not in history:** if the cart appears in fewer than 2 past orders, or no item has
   lift > 1.1, or an item is unknown, the response has status `no_pattern` / `unknown_items`, an
   explanation, and **no recommendations**. Irrelevant fallbacks are never shown.

```bash
python -m consultant.recommenders.model_store          # train + validate + version -> data/models/
uvicorn consultant.services.basket_api:app --port 8000 # API docs at http://localhost:8000/docs
curl -X POST localhost:8000/recommend -H "Content-Type: application/json"      -d '{"cart": ["B5512", "6160"], "k": 5, "session_id": "abc"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /recommend` | `{cart, k, session_id}` returns `status`, `message`, ranked items with confidence, lift, order count and reason |
| `POST /explain` | `{cart, item}` returns why that item is recommended: order counts, cart vs all-shopper rates, lift, the Apriori rule, evidence strength, example past orders, product-fit and brand notes |
| `GET /items?q=` | Catalog search for the storefront |
| `POST` / `GET` / `DELETE /orders`, `DELETE /orders/{id}` | Record, list or delete placed orders (`data/retail/placed_orders.csv`). History only, **never used for training** |
| `POST /retrain` | Retrain on the original 1,000 orders (e.g. after changing thresholds), validate, and hot-swap only if promoted |
| `GET /metrics` | Monitoring: requests, status mix, abstention rate, p50/p95 latency, uptake (share of recommended items the shopper then added), serving model and its validation |
| `GET /rules`, `GET /health` | Strongest rules; liveness and model version |

* **Model validation.** Each training run replays held-out orders as checkouts (the metrics above).
  A new version is promoted only if hit@5 drops by no more than 0.05 and coverage by no more
  than 0.10 versus the serving model (`--force` overrides). Every version is kept as
  `data/models/basket_<version>.pkl` / `.json`.
* **Monitoring.** Each request is logged to `data/logs/basket_requests.jsonl`, and live
  aggregates are served at `/metrics`.
* **Training data.** The model always trains on the original 1,000 orders only. Orders placed
  in the UI (*Your orders*, with **Reorder** and **Delete**) or via `POST /orders` are kept as order history and
  never change recommendations. `BASKET_RETRAIN_HOURS` / `python -m consultant.recommenders.model_store`
  rebuild the model from that same data, which is useful after changing `MIN_SUPPORT` / `MIN_LIFT`.
* **UI and agents.** The Streamlit "Checkout recommender" tab simulates the cart with **Add**
  buttons. The Procurement agent uses the same promoted model.

## Demo script (Streamlit "Consult" tab buttons)
1. **Cisco router failure.** Routes to Technology. All causes are labeled hypotheses, `show`
   commands are requested, compatible C8300 PSUs and NIMs are ranked, and preventive actions
   are listed. The goal check asks the human for `show version`/logs before any parts order.
2. **Lab instrument.** Clinical recommends NanoZoomer (IVD whole-slide imaging) and explains
   which requirements it meets and which it doesn't.
3. **Vendor risk (SEC).** Compares Cisco vs Agilent financial, competitive and operational
   risks with exact 10-K figures.
4. **Food security (CSR).** Recommends technology levers tied to SOFI metrics (FIES,
   undernourishment, healthy-diet cost).
5. **Cross-domain.** Technology, SEC, CSR and Clinical run in parallel, and the consolidated
   report keeps each claim's source.
6. Reply to the "Information needed" prompt to see the human-in-the-loop re-run with context.

Other tabs: **Knowledge base** (compare BM25 / dense / hybrid / rerank live), **Evaluation**
(renders `eval/results/report.md`).

## Evaluation

```bash
python -m eval.run_eval                 # all sections; LLM parts need the API key
python -m eval.run_eval --skip-llm      # offline metrics only
python -m eval.agent_check             # live health check: 16 requests (incl. vague ones) across all agents
```

`eval.agent_check` fails a case if the expected agent doesn't run, a subtask fails, no evidence is retrieved, or
the report has no cited findings and recommendations. Vague requests are answered best-effort: the router's
clarification question goes into *Information needed* instead of blocking the agents. Only near-empty
requests (fewer than 4 words) are asked about first.

**Scope gate.** The router knows exactly what the knowledge bases contain. An unrelated request (sports,
recipes, weather, coding, other companies' financials, personal medical advice) gets
*"The agent cannot answer this question because it is not relevant to the knowledge base."* with **no agents
run**. Mixed requests answer the covered part and flag the rest. As a backstop, if the agents find no
evidence-backed finding, the reply is *"There is no information from the knowledge base to answer the request."*

* **Routing:** 50 labeled test queries, 10 of them multi-domain. The Claude router is compared
  against a nearest-centroid embedding baseline trained on 24 separate examples. Metrics:
  primary accuracy, exact match, Jaccard, and micro/macro F1.
* **Retrieval:** 40 queries across 4 domains. A chunk is relevant if it contains a verified
  answer phrase, so gold labels are checkable. Metrics: Hit@1/5, Recall@10, MRR and nDCG@10
  for BM25, dense, hybrid, and hybrid + rerank.
* **Recommendation:**
  * Cisco: 10 symptom cases, scored by precision@5, hit@1 and nDCG@5 of component type, plus
    series compatibility.
  * Agilent: 4 cases, scored by hit@1 and MRR.
  * Checkout baskets: Apriori is fit on 80% of orders. Each held-out order is replayed as a checkout,
    adding one item at a time. Reported: coverage (share of steps with recommendations shown),
    hit@5 / precision@5 (recommended items the shopper went on to add), next-item hit@5, and a
    breakdown by cart size, with popularity and random as reference baselines.
* **End-to-end:** 10 consulting requests. A Haiku judge scores groundedness, relevance,
  actionability and uncertainty handling (1–5) and checks citation precision. We also report
  citation coverage, routing recall, latency and cost, plus the correlation between our
  confidence score and judged groundedness.

Results: see [`eval/results/report.md`](eval/results/report.md).

## Limitations (stated honestly)
* **IBM:** no IBM sources were supplied. The Technology agent says so instead of answering from memory.
* **Agilent:** the "Agilent PDFs" mentioned in the brief are not in the repo. The Clinical agent
  uses the 6 Agilent product pages plus Agilent's 10-K instead. That catalog is small.
* **Knee review PDF:** one paper (`A_Comprehensive_Review_of_Knee_Abnormalities...`) decodes to
  symbol soup because of embedded font encoding. It is filtered out and would need OCR.
* **Clinical trials:** capped at 2,500 per condition, preferring completed trials with results
  (`CLINICAL_TRIALS_PER_CONDITION`).
* **Retail baskets:** co-purchase signal is weak. On held-out checkouts Apriori reaches
  next-item hit@5 0.10 when it shows recommendations (random 0.13, popularity 0.16).
  Popularity has no notion of the cart and would always show the same best-sellers, which
  the brief rules out as irrelevant. The recommender abstains on most 3+ item carts because
  those combinations are rare in 1,000 orders.
* **Cisco parts catalog:** extracted from datasheets with regex and rules. Descriptions can be
  noisy, and compatibility means "listed in that series' datasheet".
