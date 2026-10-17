from consultant.ingest.loaders import split_10k_items, tenk_lines

TOC = ["Item 1.", "Business", "Item 1A.", "Risk Factors", "Item 7.", "MD&A", "Item 16.", "Summary"]
BODY = ["Item 1.", "We sell routers.", "Item 1A.", "Supply chain risk.",
        "Item 1A. Risk Factors", "(a backward cross-reference that must be ignored)",
        "Item 7.", "Revenue grew 5%.", "Item 16.", "None."]


def test_skips_toc_and_backward_refs():
    sections = split_10k_items(TOC + BODY)
    assert list(sections) == ["1", "1A", "7", "16"]
    assert "Supply chain risk." in sections["1A"]
    assert "backward cross-reference" in sections["1A"]
    assert sections["7"].startswith("Item 7.") and "Revenue grew 5%." in sections["7"]


def test_table_rows_flattened():
    html = "<table><tr><td>Revenue</td><td>$</td><td>56,654</td></tr></table><p>Item 1. Business</p>"
    lines = tenk_lines(html)
    assert "Revenue | 56,654" in lines


def test_cover_page_period_parsing():
    from consultant.ingest.loaders import _FORM_RE, _FY_END_RE
    cover = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 10-K ANNUAL REPORT ... "
             "For the fiscal year ended July 25, 2026 or TRANSITION REPORT")
    m = _FY_END_RE.search(cover)
    assert m.group(2) == "July 25, 2026" and m.group(3) == "2026"
    assert _FORM_RE.search(cover).group(1) == "10-K"


def test_pdf_page_lines_keep_section_and_page():
    # find_item_starts operates on the flattened PDF line list; pages are tracked alongside.
    from consultant.ingest.loaders import find_item_starts
    lines = ["Item 1. Business", "Item 1A. Risk Factors", "Item 16. Summary",  # TOC
             "Item 1. Business", "We sell.", "Item 1A. Risk Factors", "Tariffs.", "Item 7. MD&A", "Revenue."]
    assert find_item_starts(lines) == [(3, "1"), (5, "1A"), (7, "7")]
