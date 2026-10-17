from consultant.retrieval.hybrid import _chroma_where, _matches, rrf_fuse, tokenize


def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    order = [d for d, _ in fused]
    assert set(order[:2]) == {"a", "b"}
    assert order.index("c") > 1 and order.index("d") > 1


def test_rrf_single_list_preserves_order():
    assert [d for d, _ in rrf_fuse([["x", "y", "z"]])] == ["x", "y", "z"]


def test_tokenize_keeps_part_numbers():
    toks = tokenize("Order PWR-CC1-650WAC for the C8300-1N1S-6T")
    assert "pwr-cc1-650wac" in toks and "c8300-1n1s-6t" in toks and "the" not in toks


def test_filters():
    assert _matches({"company": "cisco", "item": "1A"}, {"company": "cisco"})
    assert not _matches({"company": "agilent"}, {"company": "cisco"})
    assert _matches({"year": 2024}, {"year": [2023, 2024]})
    assert _chroma_where({"a": 1}) == {"a": 1}
    assert _chroma_where({"a": 1, "b": [2, 3]}) == {"$and": [{"a": 1}, {"b": {"$in": [2, 3]}}]}
