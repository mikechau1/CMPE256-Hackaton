"""Extract a structured Cisco component catalog from the downloaded datasheets.

Product IDs (PIDs) are pulled with a regex from each datasheet page together with
the text that follows them in the ordering/spec tables. Each part is tagged with
the router series whose datasheet lists it (our compatibility signal) and a
component type inferred from PID prefix and description keywords.

Run: python -m consultant.ingest.cisco_catalog  -> data/processed/cisco_catalog.json
"""
from __future__ import annotations

import json
import re

import pymupdf

from .. import config

PID_RE = re.compile(
    r"\b(?:C\d{4}[A-Z]?|ISR4\d{3}|ASR1\d{3}|ASR1000|C-NIM|C-SM|NIM|SM-X|PWR|SFP|GLC|MEM|MEMUSB|SSD|M2USB|"
    r"FL|P-LTE[A-Z0-9]*|ESP\d*|EPA|SPA|PVDM)[A-Z0-9]*(?:-[A-Z0-9.\/]+)+\b"
)

# datasheet file stem -> series
SERIES_BY_DOC = {
    "datasheet-c78-744088": "Catalyst 8300",
    "nb-06-cat8200-series-edge-plat-ds-cte-en": "Catalyst 8200",
    "catalyst-8000v-edge-software-ds": "Catalyst 8000V",
    "data-sheet-c78-732542": "ISR 4000",
    "datasheet-c78-731632": "ASR 1000",
}

# (component_type, PID regex, description keywords) - first match wins
TYPE_RULES = [
    # ASR1000-* are line cards/processors; ASR1001..ASR1009 are chassis.
    ("chassis", r"^(C8300-[12]N[12]S|C8200L?-1N|ISR4\d{3}|ASR100[1-9]-)", ["platform", "router", "chassis"]),
    ("power_supply", r"^PWR|PWR-", ["power supply", "ac power", "dc power"]),
    ("fan", r"FAN", ["fan tray", "fan module"]),
    ("license", r"^FL-|^DNA|^L-", ["license", "boost", "performance"]),
    ("cellular_module", r"LTE|5G", ["lte", "cellular", "5g"]),
    ("dsp_voice_module", r"PVDM|FXS|FXO|BRI|E/M", ["voice", "dsp", "fxs", "fxo"]),
    ("memory", r"^MEM-", ["dram", "memory"]),
    ("storage", r"SSD|M2USB|MEMUSB", ["ssd", "storage", "usb flash"]),
    ("transceiver", r"^SFP|^GLC|QSFP|CPAK", ["transceiver", "optics"]),
    ("forwarding_processor", r"ESP", ["embedded services processor"]),
    ("route_processor", r"-RP\d|^RP", ["route processor"]),
    ("mounting", r"-RM-|-WM-|RFID|BLANK", ["rack mount", "wall mount", "blank"]),
    ("interface_module", r"NIM|SM-X|C-SM|EPA|SPA|SIP|MIP|-\d+TGE|-\d+T$", ["interface module", "ethernet", "serial", "t1/e1", "dsl"]),
]


def classify(pid: str, desc: str) -> str:
    d = desc.lower()
    for ctype, pid_pat, kws in TYPE_RULES:
        if re.search(pid_pat, pid):
            return ctype
    for ctype, _, kws in TYPE_RULES:
        if any(k in d for k in kws):
            return ctype
    return "other"


def _context(lines: list[str], i: int, pid: str) -> str:
    """Description = remainder of the PID's line + following non-PID lines (table cells)."""
    rest = lines[i].split(pid, 1)[1].strip(" |:-")
    parts = [rest] if rest else []
    for ln in lines[i + 1:i + 4]:
        if PID_RE.search(ln) or len(" ".join(parts)) > 220:
            break
        parts.append(ln.strip())
    return " ".join(p for p in parts if p)[:300]


def build_catalog() -> list[dict]:
    parts: dict[tuple[str, str], dict] = {}
    for pdf_path in sorted((config.RAW_DIR / "technology").glob("*.pdf")):
        series = SERIES_BY_DOC.get(pdf_path.stem, pdf_path.stem)
        meta_path = pdf_path.with_suffix(".meta.json")
        url = json.loads(meta_path.read_text()).get("url", "") if meta_path.exists() else ""
        with pymupdf.open(pdf_path) as pdf:
            for pno, page in enumerate(pdf, start=1):
                lines = [ln.strip() for ln in page.get_text("text").splitlines() if ln.strip()]
                for i, ln in enumerate(lines):
                    for pid in PID_RE.findall(ln):
                        if len(pid) < 5 or pid in ("SM-X", "C-NIM", "NIM-SSD"):
                            continue
                        key = (series, pid)
                        desc = _context(lines, i, pid)
                        entry = parts.get(key)
                        # Keep the most descriptive mention (ordering tables beat prose).
                        if entry is None or len(desc) > len(entry["description"]):
                            parts[key] = {
                                "pid": pid, "series": series, "description": desc,
                                "component_type": classify(pid, desc),
                                "source": f"{pdf_path.stem}.pdf", "url": url, "page": pno,
                            }
    return sorted(parts.values(), key=lambda p: (p["series"], p["component_type"], p["pid"]))


def main() -> None:
    catalog = build_catalog()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "cisco_catalog.json"
    out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    from collections import Counter
    print(f"{len(catalog)} parts -> {out}")
    print(Counter((p["series"], p["component_type"]) for p in catalog))


if __name__ == "__main__":
    main()
