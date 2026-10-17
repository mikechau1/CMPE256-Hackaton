"""Agilent instrument/product recommender over the fetched Agilent product pages.

Score = semantic similarity (requirement vs product description)
      + 0.1 per matched hard requirement (e.g. IVD/clinical use, throughput, automation).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from .. import config

REQUIREMENT_TERMS = {
    "ivd": ["ivd", "in vitro diagnostic", "primary diagnostic", "clinical diagnostic"],
    "ruo": ["research use", "ruo", "drug discovery", "research"],
    "high_throughput": ["high-throughput", "high throughput", "throughput", "fast", "high-speed", "automated"],
    "pathology": ["pathology", "histopathology", "slide", "tissue", "stain"],
    "cell_metabolism": ["metabolic", "metabolism", "mitochondrial", "glycolysis", "oxygen consumption", "seahorse"],
    "imaging": ["imaging", "scanner", "whole slide", "image"],
    "staining": ["special stains", "stain", "staining"],
}


@lru_cache(maxsize=1)
def load_products() -> list[dict]:
    from ..ingest.loaders import load_web_page
    products = []
    for html in sorted((config.RAW_DIR / "clinical").glob("*.html")):
        doc = load_web_page(html, "clinical", "product_page")
        if not doc:
            continue
        low = doc.text.lower()
        tags = sorted(t for t, kws in REQUIREMENT_TERMS.items() if any(k in low for k in kws))
        products.append({"name": doc.title, "url": doc.source, "doc_id": doc.doc_id,
                         "description": doc.text[:1500], "tags": tags})
    return products


@lru_cache(maxsize=1)
def _vectors() -> np.ndarray:
    from ..retrieval.hybrid import embed_passages
    return embed_passages([f"{p['name']}\n{p['description']}" for p in load_products()])


def requirement_tags(text: str) -> list[str]:
    low = text.lower()
    return sorted(t for t, kws in REQUIREMENT_TERMS.items() if any(re.search(r"\b" + re.escape(k), low) for k in kws))


def recommend_instruments(requirements: str, k: int = 3) -> dict:
    from ..retrieval.hybrid import embed_queries
    products = load_products()
    if not products:
        return {"products": [], "note": "No Agilent product pages indexed - run fetch_sources."}
    need = requirement_tags(requirements)
    sims = _vectors() @ embed_queries([requirements])[0]
    ranked = []
    for p, s in zip(products, sims):
        matched = [t for t in need if t in p["tags"]]
        missing = [t for t in need if t not in p["tags"]]
        score = float(s) + 0.1 * len(matched)
        ranked.append({"name": p["name"], "url": p["url"], "score": round(score, 4), "similarity": round(float(s), 3),
                       "matched_requirements": matched, "unmet_requirements": missing,
                       "summary": p["description"][:400]})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return {"requirement_tags": need, "products": ranked[:k],
            "note": "Catalog limited to the Agilent pages supplied in Agilent.txt; verify specs with Agilent."}
