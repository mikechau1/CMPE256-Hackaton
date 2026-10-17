"""Build per-domain knowledge bases: chunk pickles (for BM25) + Chroma dense collections.

Run: python -m consultant.ingest.build_index [--domain technology] [--no-trials]
"""
from __future__ import annotations

import argparse
import collections
import pickle
import time

from .. import config
from ..retrieval.hybrid import chroma_client, chunks_path, embed_passages
from .chunking import chunk_document
from .loaders import DOMAIN_LOADERS, clinical_docs


def build_domain(domain: str, include_trials: bool = True) -> int:
    t0 = time.time()
    docs = clinical_docs(include_trials) if domain == "clinical" else DOMAIN_LOADERS[domain]()
    chunks, seen = [], set()
    for doc in docs:
        for ch in chunk_document(doc):
            if ch.chunk_id not in seen:
                seen.add(ch.chunk_id)
                chunks.append(ch)
    if not chunks:
        print(f"[{domain}] no documents found - did you run fetch_sources?")
        return 0
    by_type = collections.Counter(c.metadata["doc_type"] for c in chunks)
    print(f"[{domain}] {len(chunks)} chunks {dict(by_type)}; embedding...")

    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]
    # Prefix the title so short chunks keep their document context in vector space.
    vectors = embed_passages([f"{m['title']}\n{t}" for m, t in zip(metas, texts)], show_progress=True)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(chunks_path(domain), "wb") as f:
        pickle.dump({"ids": ids, "texts": texts, "metas": metas}, f)

    client = chroma_client()
    name = f"kb_{domain}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    step = 2000
    for i in range(0, len(ids), step):
        col.add(ids=ids[i:i + step], embeddings=vectors[i:i + step].tolist(), metadatas=metas[i:i + step])
    print(f"[{domain}] done in {time.time() - t0:.0f}s")
    return len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAIN_LOADERS), action="append",
                    help="domain(s) to build; default all")
    ap.add_argument("--no-trials", action="store_true", help="skip ClinicalTrials.gov CSVs (faster)")
    args = ap.parse_args()
    for d in args.domain or list(DOMAIN_LOADERS):
        build_domain(d, include_trials=not args.no_trials)


if __name__ == "__main__":
    main()
