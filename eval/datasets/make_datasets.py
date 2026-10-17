"""Hand-labelled evaluation sets (routing, recommendation, end-to-end). Re-run to regenerate the .jsonl files.

retrieval.jsonl is maintained directly: each query has an answer_phrase whose presence in a
chunk defines relevance (verified to exist in the knowledge base).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

ROUTING_TRAIN = [
    ("Our Cisco router keeps dropping WAN links, what parts should we replace?", ["technology"]),
    ("Compare the Catalyst 8200 and 8300 for a branch office", ["technology"]),
    ("How does Google Cloud support the Model Context Protocol?", ["technology"]),
    ("Design a RAG architecture on Google Cloud for our documents", ["technology"]),
    ("What IPsec throughput does the ASR 1000 deliver?", ["technology"]),
    ("Which Agilent instrument should our lab buy for cell metabolism assays?", ["clinical"]),
    ("What does research say about machine learning for heart attack prediction?", ["clinical"]),
    ("Find completed clinical trials on continuous glucose monitoring", ["clinical"]),
    ("Recommend a digital pathology scanner for our hospital", ["clinical"]),
    ("How accurate are deep learning models at detecting COVID-19 from lung images?", ["clinical"]),
    ("Summarize Cisco's risk factors from its latest 10-K", ["sec"]),
    ("How did Agilent's revenue and margins change in fiscal 2025?", ["sec"]),
    ("What is Cisco's debt and cash position?", ["sec"]),
    ("Compare competitive risks disclosed by Cisco and Agilent", ["sec"]),
    ("What antitrust risks does Alphabet disclose in its 10-K?", ["sec"]),
    ("How many people are chronically hungry in the world?", ["csr"]),
    ("What ESG initiatives improve food security?", ["csr"]),
    ("How affordable is a healthy diet in low-income countries?", ["csr"]),
    ("What is the state of child stunting and wasting?", ["csr"]),
    ("What should we add to an order with a Bosch control panel?", ["procurement"]),
    ("Which security cameras are often bought together with NVRs?", ["procurement"]),
    ("Recommend a bundle of smoke and CO detectors for a new site", ["procurement"]),
    ("Is Cisco a financially safe vendor for our network refresh, and which routers fit?", ["technology", "sec"]),
    ("Could cloud data platforms help NGOs reduce hunger, and what tech should they deploy?", ["csr", "technology"]),
    ("Should our hospital rely on Agilent given its financial risks, and which instruments fit?", ["clinical", "sec"]),
]

ROUTING_TEST = [
    ("Our Cisco enterprise router is experiencing intermittent connectivity failures. Which replacement parts should we order?", ["technology"]),
    ("The ISR 4331 reloads every few days and the logs show power events", ["technology"]),
    ("Which NIM modules give us 8 PoE ports on a Catalyst 8200?", ["technology"]),
    ("Is the Catalyst 8000V suitable for running in AWS or Google Cloud?", ["technology"]),
    ("Explain how MCP servers let AI agents call enterprise tools", ["technology"]),
    ("What IBM products would help modernize our mainframe integration?", ["technology"]),
    ("Our ASR 1006-X fan tray alarm keeps firing - what do we do?", ["technology"]),
    ("How do we size DRAM on edge routers running SD-WAN and security?", ["technology"]),
    ("Which diagnostic instrument or replacement component best meets our laboratory requirements?", ["clinical"]),
    ("We need a whole-slide imaging system for primary diagnosis", ["clinical"]),
    ("What evidence exists for wearable IoT monitoring of heart rate to detect heart attacks?", ["clinical"]),
    ("Recommend clinical trials on knee exoskeleton rehabilitation", ["clinical"]),
    ("How do AI models improve glycemic control for type 1 diabetes?", ["clinical"]),
    ("Which special stains automation does Agilent offer for histology labs?", ["clinical"]),
    ("Are there phase 3 COVID-19 vaccine trials with posted results?", ["clinical"]),
    ("What machine learning methods detect knee abnormalities in imaging?", ["clinical"]),
    ("What are the financial, competitive, and operational risks associated with Cisco and Agilent?", ["sec"]),
    ("How much did Cisco spend on share buybacks and dividends in fiscal 2025?", ["sec"]),
    ("What does Agilent disclose about tariffs and China exposure?", ["sec"]),
    ("Summarize Cisco's cybersecurity risk management disclosures", ["sec"]),
    ("What is Agilent's segment breakdown and which segment grew fastest?", ["sec"]),
    ("Did Cisco's Splunk acquisition change its debt profile?", ["sec"]),
    ("What legal proceedings does Cisco report?", ["sec"]),
    ("Which technology investments could improve food security and sustainability outcomes?", ["csr", "technology"]),
    ("How does urbanization affect food security and diets?", ["csr"]),
    ("What financing is needed to end hunger by 2030?", ["csr"]),
    ("Which regions have the highest prevalence of undernourishment?", ["csr"]),
    ("How can our company's CSR program measure impact on nutrition?", ["csr"]),
    ("Is food insecurity worse for women than men?", ["csr"]),
    ("What does the FAO say about the cost of a healthy diet?", ["csr"]),
    ("Customers buying the Hikvision 4MP camera also buy what?", ["procurement"]),
    ("We are quoting a DSC PowerSeries Neo alarm kit - suggest add-on items", ["procurement"]),
    ("What are the strongest product associations in our order history?", ["procurement"]),
    ("Build a starter bundle for a small office intrusion alarm", ["procurement"]),
    ("Which detectors are frequently ordered with Bosch detector bases?", ["procurement"]),
    ("Our Cisco ISR routers keep failing; should we switch vendors given Cisco's financial outlook?", ["technology", "sec"]),
    ("We operate Agilent lab instruments over Cisco networks; assess vendor risks and network fixes", ["technology", "clinical", "sec"]),
    ("How can IoT and cloud networking reduce food loss in supply chains, and which Cisco platforms fit?", ["csr", "technology"]),
    ("Does Agilent's 10-K mention sustainability, and how does it relate to global food security?", ["sec", "csr"]),
    ("Which Agilent products support food safety testing, and how does that tie to SOFI hunger trends?", ["clinical", "csr"]),
    ("Our hospital network outages hurt diagnostics; recommend Cisco parts and lab continuity steps", ["technology", "clinical"]),
    ("Evaluate Cisco as a supplier: financial health, product fit for branches, and ESG", ["sec", "technology", "csr"]),
    ("We need cameras and NVRs for our lab; what bundle and how reliable is Cisco for the network?", ["procurement", "technology"]),
    ("Can RAG on Google Cloud help analyze clinical trial data for diabetes?", ["technology", "clinical"]),
    ("Rank Cisco and Agilent by exposure to supply chain disruption", ["sec"]),
    ("What tech does FAO recommend for monitoring food crises?", ["csr"]),
    ("Our C8300 shows CRC errors on the SFP port", ["technology"]),
    ("Which completed diabetes trials evaluated mobile app coaching?", ["clinical"]),
    ("Remaining Agilent share repurchase authorization?", ["sec"]),
    ("Glassbreak detector add-ons for an existing GE Interlogix install", ["procurement"]),
    ("How much did Google Cloud revenue grow according to Alphabet's latest annual report?", ["sec"]),
    ("How did Cisco's fiscal 2026 revenue compare with fiscal 2025?", ["sec"]),
    ("Should we build our RAG platform on Google Cloud given Alphabet's financial and regulatory risks?", ["technology", "sec"]),
]

RECOMMENDATION = [
    {"id": "C1", "kind": "component", "query": "Catalyst 8300 router has intermittent connectivity failures and interface CRC errors",
     "gold_types": ["interface_module", "transceiver"], "series": "Catalyst 8300"},
    {"id": "C2", "kind": "component", "query": "Our C8300 reloads unexpectedly and logs show power supply alarms",
     "gold_types": ["power_supply"], "series": "Catalyst 8300"},
    {"id": "C3", "kind": "component", "query": "Catalyst 8300 overheating, temperature alarms and fan warnings in the logs",
     "gold_types": ["fan"], "series": "Catalyst 8300"},
    {"id": "C4", "kind": "component", "query": "Catalyst 8200 memory exhaustion with MALLOC failures after enabling SD-WAN security",
     "gold_types": ["memory"], "series": "Catalyst 8200"},
    {"id": "C5", "kind": "component", "query": "C8200 LTE backup link keeps dropping; cellular modem disconnects",
     "gold_types": ["cellular_module"], "series": "Catalyst 8200"},
    {"id": "C6", "kind": "component", "query": "ISR 4331 throughput capped at 100 Mbps even though the WAN is 500 Mbps",
     "gold_types": ["license"], "series": "ISR 4000"},
    {"id": "C7", "kind": "component", "query": "ASR 1006-X packet forwarding capacity saturated during peak traffic",
     "gold_types": ["forwarding_processor", "license"], "series": "ASR 1000"},
    {"id": "C8", "kind": "component", "query": "Catalyst 8300 voice calls fail, DSP resources exhausted on FXS ports",
     "gold_types": ["dsp_voice_module"], "series": "Catalyst 8300"},
    {"id": "C9", "kind": "component", "query": "Catalyst 8200 bootflash errors and router stuck in rommon on boot",
     "gold_types": ["storage"], "series": "Catalyst 8200"},
    {"id": "C10", "kind": "component", "query": "ASR 1000 40GE link flaps and optic errors on the EPA",
     "gold_types": ["interface_module", "transceiver"], "series": "ASR 1000"},
    {"id": "I1", "kind": "instrument", "query": "We need an IVD whole slide scanner for primary histopathology diagnosis at high throughput",
     "gold_contains": "nanozoomer"},
    {"id": "I2", "kind": "instrument", "query": "Our drug discovery lab measures mitochondrial respiration and glycolysis in live cells",
     "gold_contains": "seahorse"},
    {"id": "I3", "kind": "instrument", "query": "Automate special stains for our histology lab with a consistent and safe workflow",
     "gold_contains": "artisan"},
    {"id": "I4", "kind": "instrument", "query": "Digitize pathology slides and manage digital images for remote review",
     "gold_contains": "pathology"},
]

E2E = [
    {"id": "E1", "query": "Our Cisco enterprise router is experiencing intermittent connectivity failures. Which replacement parts "
                          "should we order, and what actions should we take to prevent future outages?",
     "expected_domains": ["technology"], "must_mention": ["diagnos", "hypothes"]},
    {"id": "E2", "query": "Our Cisco Catalyst 8300 reloads unexpectedly a few times a week. What should we replace and what "
                          "should we check first?", "expected_domains": ["technology"], "must_mention": ["power"]},
    {"id": "E3", "query": "Which diagnostic instrument best meets our pathology lab's requirement to digitize clinical slides for "
                          "primary diagnosis?", "expected_domains": ["clinical"], "must_mention": ["NanoZoomer"]},
    {"id": "E4", "query": "What are the financial, competitive, and operational risks associated with Cisco and Agilent in 2025?",
     "expected_domains": ["sec"], "must_mention": ["Cisco", "Agilent"]},
    {"id": "E5", "query": "Which technology investments could improve food security and sustainability outcomes, and how do we "
                          "measure impact?", "expected_domains": ["csr", "technology"], "must_mention": ["food insecurity"]},
    {"id": "E6", "query": "We are ordering a Bosch B5512 control panel. What else should we include in the order?",
     "expected_domains": ["procurement"], "must_mention": ["lift"]},
    {"id": "E7", "query": "What does the research say about machine learning for predicting heart attacks, and which completed "
                          "trials are relevant?", "expected_domains": ["clinical"], "must_mention": ["trial"]},
    {"id": "E8", "query": "We run Agilent lab instruments over Cisco ISR 4000 routers that keep dropping links. Should we "
                          "standardise on these vendors given their 2025 financial risks, and how could the upgrade support "
                          "our food-security ESG commitments?",
     "expected_domains": ["technology", "sec", "csr"], "must_mention": ["Cisco", "Agilent"]},
    {"id": "E9", "query": "Which IBM servers should we buy to replace our failing Cisco routers?",
     "expected_domains": ["technology"], "must_mention": ["IBM"]},
    {"id": "E10", "query": "How did Agilent's revenue change in fiscal 2025 and what is management doing about tariffs?",
     "expected_domains": ["sec"], "must_mention": ["6,948"]},
    {"id": "E11", "query": "How did Cisco's fiscal 2026 results compare with fiscal 2025, and how does Alphabet's Google Cloud "
                           "growth in 2025 compare as a competitor in AI infrastructure?",
     "expected_domains": ["sec"], "must_mention": ["63.3", "Alphabet"]},
]


def write(name: str, rows: list[dict]) -> None:
    with open(HERE / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    write("routing.jsonl",
          [{"id": f"RT{i:02d}", "split": "train", "query": q, "labels": l} for i, (q, l) in enumerate(ROUTING_TRAIN, 1)]
          + [{"id": f"RQ{i:02d}", "split": "test", "query": q, "labels": l} for i, (q, l) in enumerate(ROUTING_TEST, 1)])
    write("recommendation.jsonl", RECOMMENDATION)
    write("e2e.jsonl", E2E)
    print(f"routing: {len(ROUTING_TRAIN)} train / {len(ROUTING_TEST)} test; "
          f"recommendation: {len(RECOMMENDATION)}; e2e: {len(E2E)}")
