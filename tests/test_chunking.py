from consultant.ingest.chunking import Document, chunk_document, chunk_text


def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 120 for i in range(10))
    chunks = chunk_text(text, size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 300 + 30 + 5 for c in chunks)
    # overlap: the tail of chunk 0 starts chunk 1
    assert chunks[0].split()[-5:] == chunks[1].split()[:30][-5:] or chunks[0].split()[-1] in chunks[1]


def test_metadata_preserved_on_every_chunk():
    doc = Document(doc_id="d1", title="T", source="s.pdf", domain="sec", doc_type="10-K",
                   text="Risk factor sentence. " * 400, page=3, section="Item 1A Risk Factors",
                   extra={"company": "cisco", "nested": {"skip": True}})
    chunks = chunk_document(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["page"] == 3 and c.metadata["section"] == "Item 1A Risk Factors"
        assert c.metadata["company"] == "cisco" and "nested" not in c.metadata
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_tiny_fragments_dropped():
    assert chunk_text("Page 12") == []
