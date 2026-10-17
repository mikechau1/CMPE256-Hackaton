"""Hybrid retrieval: BM25 + dense (bge) fused with Reciprocal Rank Fusion, then cross-encoder rerank."""
from __future__ import annotations

import math
import pickle
import re
import threading
from functools import lru_cache
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from .. import config
from ..schemas import Evidence

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.\/]*[a-z0-9]|[a-z0-9]")
_STOP = set("the a an and or of to in for on with by is are was were be as at from that this it its into".split())
_model_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, device="cpu")


def embed_passages(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    with _model_lock:
        return embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress, convert_to_numpy=True)


def embed_queries(queries: list[str]) -> np.ndarray:
    with _model_lock:
        return embedder().encode([BGE_QUERY_PREFIX + q for q in queries], normalize_embeddings=True,
                                 convert_to_numpy=True)


def chroma_client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(config.INDEX_DIR / "chroma"),
                                     settings=Settings(anonymized_telemetry=False))


def chunks_path(domain: str):
    return config.INDEX_DIR / f"{domain}_chunks.pkl"


def _matches(meta: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        if isinstance(v, (list, tuple, set)):
            if meta.get(k) not in v:
                return False
        elif meta.get(k) != v:
            return False
    return True


def _chroma_where(filters: dict | None) -> dict | None:
    if not filters:
        return None
    clauses = [{k: {"$in": list(v)}} if isinstance(v, (list, tuple, set)) else {k: v} for k, v in filters.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class HybridRetriever:
    def __init__(self, domain: str):
        self.domain = domain
        with open(chunks_path(domain), "rb") as f:
            data = pickle.load(f)
        self.ids: list[str] = data["ids"]
        self.texts: list[str] = data["texts"]
        self.metas: list[dict] = data["metas"]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.collection = chroma_client().get_collection(f"kb_{domain}")

    def __len__(self) -> int:
        return len(self.ids)

    # -- individual retrievers ------------------------------------------------
    def bm25_search(self, query: str, n: int, filters: dict | None = None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if _matches(self.metas[i], filters):
                out.append(self.ids[i])
                if len(out) >= n:
                    break
        return out

    def dense_search(self, query: str, n: int, filters: dict | None = None) -> list[str]:
        q = embed_queries([query])[0].tolist()
        res = self.collection.query(query_embeddings=[q], n_results=n, where=_chroma_where(filters),
                                    include=[])
        return res["ids"][0]

    # -- public -----------------------------------------------------------------
    def search(self, query: str, k: int = config.DEFAULT_TOP_K, filters: dict | None = None,
               mode: Mode = "hybrid_rerank", n_candidates: int = config.CANDIDATES_PER_RETRIEVER) -> list[Evidence]:
        if mode == "bm25":
            ranked = [(cid, 1.0 / r) for r, cid in enumerate(self.bm25_search(query, k, filters), 1)]
        elif mode == "dense":
            ranked = [(cid, 1.0 / r) for r, cid in enumerate(self.dense_search(query, k, filters), 1)]
        else:
            fused = rrf_fuse([self.bm25_search(query, n_candidates, filters),
                              self.dense_search(query, n_candidates, filters)])
            if mode == "hybrid":
                ranked = fused[:k]
            else:
                pool = [cid for cid, _ in fused[: max(3 * k, 20)]]
                if not pool:
                    return []
                with _model_lock:
                    logits = reranker().predict([(query, self.texts[self.pos[c]]) for c in pool])
                probs = [1 / (1 + math.exp(-float(x))) for x in logits]
                ranked = sorted(zip(pool, probs), key=lambda x: x[1], reverse=True)[:k]
        return [self._evidence(cid, score) for cid, score in ranked]

    def _evidence(self, cid: str, score: float) -> Evidence:
        i = self.pos[cid]
        m = self.metas[i]
        return Evidence(
            chunk_id=cid, doc_id=m["doc_id"], title=m["title"], source=m["source"],
            page=m["page"] if m.get("page", -1) >= 0 else None, section=m.get("section") or None,
            domain=m["domain"], doc_type=m["doc_type"], text=self.texts[i], score=round(float(score), 4),
        )


_retrievers: dict[str, HybridRetriever] = {}
_retriever_lock = threading.Lock()


def get_retriever(domain: str) -> HybridRetriever:
    with _retriever_lock:
        if domain not in _retrievers:
            _retrievers[domain] = HybridRetriever(domain)
        return _retrievers[domain]
