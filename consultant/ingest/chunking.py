"""Paragraph-aware chunker that preserves document metadata on every chunk."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .. import config

# bge-small truncates at 512 tokens; ~0.75 words/token for English prose.
WORDS_PER_CHUNK = int(config.CHUNK_TOKENS * 0.75)
OVERLAP_WORDS = int(config.CHUNK_OVERLAP * 0.75)


@dataclass
class Document:
    """One logical unit of source text (a PDF page, a 10-K item, a web page, a trial)."""

    doc_id: str
    title: str
    source: str
    domain: str
    doc_type: str
    text: str
    page: int | None = None
    section: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


def _split_long(paragraph: str, limit: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= limit:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, cur = [], []
    for s in sentences:
        sw = s.split()
        if len(sw) > limit:  # pathological run-on (tables): hard wrap
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.extend(" ".join(sw[i:i + limit]) for i in range(0, len(sw), limit))
            continue
        if len(cur) + len(sw) > limit and cur:
            out.append(" ".join(cur))
            cur = []
        cur.extend(sw)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_text(text: str, size: int = WORDS_PER_CHUNK, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Pack paragraphs into ~size-word chunks with a word overlap between neighbours."""
    paras: list[str] = []
    for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9•\-])", text):
        p = " ".join(p.split())
        if p:
            paras.extend(_split_long(p, size))
    chunks, cur, cur_len = [], [], 0
    for p in paras:
        n = len(p.split())
        if cur and cur_len + n > size:
            chunks.append("\n".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur, cur_len = ([tail] if tail else []), len(tail.split())
        cur.append(p)
        cur_len += n
    if cur:
        chunks.append("\n".join(cur))
    # Drop near-empty fragments (page numbers, stray headers).
    return [c for c in chunks if len(c.split()) >= 8]


def chunk_document(doc: Document) -> list[Chunk]:
    out = []
    for i, text in enumerate(chunk_text(doc.text)):
        h = hashlib.md5(f"{doc.doc_id}|{doc.page}|{doc.section}|{i}".encode()).hexdigest()[:12]
        meta = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source": doc.source,
            "domain": doc.domain,
            "doc_type": doc.doc_type,
            "page": doc.page if doc.page is not None else -1,
            "section": doc.section or "",
            **{k: v for k, v in doc.extra.items() if isinstance(v, (str, int, float, bool))},
        }
        out.append(Chunk(chunk_id=f"{doc.domain[:3]}-{h}", text=text, metadata=meta))
    return out
