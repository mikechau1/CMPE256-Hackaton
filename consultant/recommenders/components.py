"""Cisco replacement-component recommender.

Pipeline (content-based + contextual):
1. Detect the router series from the request (compatibility = parts listed in that
   series' datasheet).
2. Map symptoms to candidate fault modes (keyword evidence) -> each fault mode
   weights component types.
3. Rank compatible parts by  0.6 * fault/type prior + 0.4 * semantic similarity
   between the request and the part description.
Every fault mode is returned as a *hypothesis* with the diagnostics that would
confirm or rule it out; nothing is presented as a confirmed failure.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np

from .. import config

FAULT_MODES = [
    {
        "name": "Interface module, optic or cabling fault",
        "keywords": ["intermittent", "flap", "crc", "input error", "link down", "interface down", "packet loss",
                     "connectivity", "drops", "port", "sfp", "optic", "link"],
        "component_types": {"interface_module": 1.0, "transceiver": 0.9},
        "diagnostics": [
            "show interfaces | include line protocol|CRC|input errors|resets",
            "show logging | include LINK|LINEPROTO|OIR|TRANSCEIVER",
            "show hw-module subslot all oir   (module operational state)",
            "show interfaces transceiver detail   (optic Rx/Tx power, if SFP-based)",
        ],
        "preventive": ["Monitor interface error counters via SNMP/telemetry with alert thresholds",
                       "Keep spare NIMs/optics for critical WAN links; label and inspect cabling"],
    },
    {
        "name": "Power supply degradation or input power instability",
        "keywords": ["reboot", "reload", "power", "brownout", "shut down", "shuts down", "restart", "outage",
                     "crash", "psu", "ups"],
        "component_types": {"power_supply": 1.0, "fan": 0.3},
        "diagnostics": [
            "show environment all   (PSU status, voltages)",
            "show version | include uptime|reload reason|Last reload",
            "show logging | include PWR|POWER|PEM|ENVIRONMENT",
        ],
        "preventive": ["Use dual/redundant power supplies on separate circuits where the platform supports it",
                       "Put routers on a monitored UPS and log power events"],
    },
    {
        "name": "Thermal condition or fan failure",
        "keywords": ["overheat", "temperature", "fan", "thermal", "hot", "airflow", "noise"],
        "component_types": {"fan": 1.0, "power_supply": 0.3},
        "diagnostics": ["show environment temperature", "show environment | include Fan",
                        "show logging | include TEMP|FAN|THERMAL"],
        "preventive": ["Verify rack airflow direction matches fan/PSU airflow (front-to-back vs reverse)",
                       "Alert on inlet temperature; schedule dust/filter maintenance"],
    },
    {
        "name": "Memory exhaustion or DRAM fault",
        "keywords": ["memory", "malloc", "traceback", "leak", "high cpu", "slow", "hang", "freeze", "crash"],
        "component_types": {"memory": 1.0},
        "diagnostics": ["show platform resources", "show processes memory sorted",
                        "show logging | include MALLOC|SYS-2|%PLATFORM", "check for crashinfo files: dir crashinfo:"],
        "preventive": ["Right-size DRAM for enabled services (SD-WAN, security, apps)",
                       "Track memory trend; upgrade IOS XE to a recommended release to clear known leaks"],
    },
    {
        "name": "Storage / bootflash fault",
        "keywords": ["bootflash", "storage", "disk", "flash", "boot", "rommon", "file system"],
        "component_types": {"storage": 1.0},
        "diagnostics": ["dir bootflash:", "show platform hardware ... (storage health)", "fsck on bootflash if errors"],
        "preventive": ["Keep a golden image and config backup off-box", "Replace aging M.2/USB storage proactively"],
    },
    {
        "name": "Cellular WAN module issue",
        "keywords": ["lte", "cellular", "5g", "sim", "wireless wan", "backup link", "modem"],
        "component_types": {"cellular_module": 1.0},
        "diagnostics": ["show cellular 0/x/0 all", "show cellular 0/x/0 radio", "verify SIM/APN with carrier"],
        "preventive": ["Monitor RSSI/RSRP; use external antennas in weak-signal sites"],
    },
    {
        "name": "Voice / DSP resource issue",
        "keywords": ["voice", "call", "dsp", "fxs", "fxo", "pri", "telephony", "fax"],
        "component_types": {"dsp_voice_module": 1.0, "interface_module": 0.3},
        "diagnostics": ["show voice dsp group all", "show controllers t1/e1", "show voice port summary"],
        "preventive": ["Size PVDM DSP capacity for peak call volume"],
    },
    {
        "name": "Throughput or license capacity limit",
        "keywords": ["throughput", "bandwidth", "performance", "throttl", "capacity", "saturat", "slow", "latency"],
        "component_types": {"license": 1.0, "forwarding_processor": 0.6},
        "diagnostics": ["show platform hardware throughput level", "show license summary",
                        "show platform hardware qfp active datapath utilization"],
        "preventive": ["Capacity-plan WAN growth; upgrade performance/boost license before saturation"],
    },
]

SERIES_PATTERNS = {
    "Catalyst 8300": r"\b(c?8300|catalyst\s*8300)",
    "Catalyst 8200": r"\b(c?8200l?|catalyst\s*8200)",
    "Catalyst 8000V": r"\b(c?8000v|catalyst\s*8000v)",
    "ISR 4000": r"\b(isr\s?4\d{3}|isr\s*4000|4000 series|isr4k)",
    "ASR 1000": r"\b(asr\s?100\d|asr\s*1000|asr1k)",
}
EXCLUDE_BY_DEFAULT = {"chassis", "mounting", "other"}


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    path = config.PROCESSED_DIR / "cisco_catalog.json"
    if not path.exists():
        from ..ingest.cisco_catalog import main as build
        build()
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _catalog_vectors() -> np.ndarray:
    from ..retrieval.hybrid import embed_passages
    cat = load_catalog()
    return embed_passages([f"{p['pid']} {p['component_type'].replace('_', ' ')}: {p['description']}" for p in cat])


def detect_series(text: str) -> list[str]:
    low = text.lower()
    return [s for s, pat in SERIES_PATTERNS.items() if re.search(pat, low)]


def detect_pids(text: str) -> list[str]:
    pids = {p["pid"] for p in load_catalog()}
    return [p for p in pids if p.lower() in text.lower()]


def score_fault_modes(text: str) -> list[dict]:
    low = text.lower()
    scored = []
    for fm in FAULT_MODES:
        hits = [k for k in fm["keywords"] if re.search(r"\b" + re.escape(k), low)]
        if hits:
            scored.append({**fm, "matched_symptoms": hits, "weight": len(hits)})
    if not scored:  # no symptom evidence: fall back to the two most common hardware causes
        scored = [{**FAULT_MODES[0], "matched_symptoms": [], "weight": 1},
                  {**FAULT_MODES[1], "matched_symptoms": [], "weight": 1}]
    total = sum(s["weight"] for s in scored)
    for s in scored:
        s["weight"] = round(s["weight"] / total, 3)
    return sorted(scored, key=lambda s: s["weight"], reverse=True)


def diversify(ranked: list[dict], k: int, penalty: float = 0.08) -> list[dict]:
    """Greedy re-rank that penalises repeated component types so each hypothesis gets candidates."""
    picked, pool, seen = [], list(ranked), {}
    while pool and len(picked) < k:
        best = max(pool, key=lambda r: r["score"] - penalty * seen.get(r["component_type"], 0))
        pool.remove(best)
        seen[best["component_type"]] = seen.get(best["component_type"], 0) + 1
        picked.append(best)
    return picked


def recommend_components(request: str, series: str | None = None, k: int = 8,
                         component_types: list[str] | None = None) -> dict:
    from ..retrieval.hybrid import embed_queries
    cat = load_catalog()
    series_list = [series] if series else detect_series(request)
    modes = score_fault_modes(request)
    type_prior: dict[str, float] = {}
    for m in modes:
        for ctype, w in m["component_types"].items():
            type_prior[ctype] = max(type_prior.get(ctype, 0.0), m["weight"] * w)
    top = max(type_prior.values()) if type_prior else 1.0
    type_prior = {t: v / top for t, v in type_prior.items()}

    qv = embed_queries([request])[0]
    sims = _catalog_vectors() @ qv
    ranked = []
    for i, part in enumerate(cat):
        if series_list and part["series"] not in series_list:
            continue
        if component_types and part["component_type"] not in component_types:
            continue
        if not component_types and part["component_type"] in EXCLUDE_BY_DEFAULT:
            continue
        prior = type_prior.get(part["component_type"], 0.0)
        score = 0.6 * prior + 0.4 * float(sims[i])
        ranked.append({**part, "score": round(score, 4), "type_prior": round(prior, 3),
                       "similarity": round(float(sims[i]), 3),
                       "reason": f"{part['component_type'].replace('_', ' ')} listed for {part['series']} "
                                 f"(datasheet p.{part['page']}); fault-mode prior {prior:.2f}, "
                                 f"text similarity {float(sims[i]):.2f}"})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return {
        "series_detected": series_list,
        "series_known": bool(series_list),
        "pids_mentioned": detect_pids(request),
        "hypotheses": [{"fault_mode": m["name"], "weight": m["weight"], "status": "hypothesis",
                        "matched_symptoms": m["matched_symptoms"], "confirm_with": m["diagnostics"],
                        "preventive": m["preventive"]} for m in modes],
        "parts": diversify(ranked, k),
        "note": ("Parts are compatible candidates, not confirmed failures. Order only after diagnostics "
                 "isolate the faulty component." + ("" if series_list else
                 " Router series/model not identified - ask the customer for 'show version' output.")),
    }
