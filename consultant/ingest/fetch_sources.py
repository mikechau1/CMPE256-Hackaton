"""Download the remote sources the repo only lists as URLs.

* Cisco datasheet PDFs and product pages      -> data/raw/technology/
* Google Cloud pages                          -> data/raw/technology/
* Agilent product pages                       -> data/raw/clinical/
* Cisco and Agilent FY2025 10-K from EDGAR    -> data/raw/sec/  (only used for filings that are not
  already in the UnitedStatesSecuritiesandExchangeCommission(SEC) folder)

Run: python -m consultant.ingest.fetch_sources [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .. import config

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
# Raw (id_) snapshot from the most recent capture; cisco.com/agilent.com block scripted clients.
ARCHIVE_PREFIX = "https://web.archive.org/web/{year}id_/"
ARCHIVE_YEARS = (2026, 2025, 2024)
URL_RE = re.compile(r"https?://\S+")

# company -> (CIK, fiscal year whose 10-K we want)
SEC_COMPANIES = {"cisco": (858877, 2025), "agilent": (1090872, 2025)}


def slug(text: str, n: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n]


def parse_url_list(path: Path) -> list[tuple[str, str]]:
    """Return (title, url) pairs; the title is the nearest non-URL line above a URL."""
    pairs, last_title = [], path.stem
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = URL_RE.search(line)
        if m:
            url = m.group(0).split("?_gl=")[0]
            pairs.append((last_title, url))
        else:
            last_title = re.sub(r"^\d+\.\s*", "", line)
    return pairs


def html_to_text(html: str) -> tuple[str, str]:
    """Extract readable text; drops navigation/boilerplate (short lines, menus)."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    desc = ""
    for m in soup.find_all("meta"):
        if m.get("name") == "description" or m.get("property") == "og:description":
            desc = (m.get("content") or "").strip()
            break
    for tag in soup(["script", "style", "noscript", "svg", "nav"]):  # not <header>: some sites leave it unclosed
        tag.decompose()
    lines, seen = [], set()
    for ln in (soup.find("main") or soup.body or soup).get_text("\n").splitlines():
        ln = " ".join(ln.split())
        # Menu items and buttons are short; spec lines usually have >= 5 words.
        if len(ln.split()) >= 5 and ln not in seen:
            seen.add(ln)
            lines.append(ln)
    text = "\n".join(([desc] if desc and desc not in seen else []) + lines)
    return title, text


def fetch(url: str, ua: str = BROWSER_UA) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Language": "en-US,en"}, timeout=60)
    r.raise_for_status()
    return r


def fetch_with_archive(url: str) -> tuple[requests.Response, str]:
    """Fetch directly; on a bot-protection 403 fall back to the Wayback Machine raw snapshot."""
    try:
        return fetch(url), url
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code not in (401, 403, 429):
            raise
    last_err: Exception | None = None
    for year in ARCHIVE_YEARS:  # an older capture may succeed where the newest is a blocked page
        for attempt in range(4):
            try:
                r = fetch(ARCHIVE_PREFIX.format(year=year) + url)
                return r, r.url
            except requests.HTTPError as e:
                last_err = e
                if e.response is not None and e.response.status_code == 429:
                    time.sleep(15 * (attempt + 1))  # archive.org rate limit
                    continue
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                time.sleep(15 * (attempt + 1))
    raise last_err  # type: ignore[misc]


def save_page(title: str, url: str, out_dir: Path, force: bool) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    is_pdf = url.lower().endswith(".pdf")
    dest = out_dir / (slug(Path(url).stem if is_pdf else title + "-" + Path(url.rstrip("/")).name)
                      + (".pdf" if is_pdf else ".html"))
    if dest.exists() and not force:
        print(f"  cached  {dest.name}")
        return dest
    try:
        r, fetched_from = fetch_with_archive(url)
    except requests.RequestException as e:
        print(f"  FAILED  {url}: {e}")
        return None
    dest.write_bytes(r.content)
    meta = {"title": title, "url": url, "fetched_from": fetched_from, "doc_type": "pdf" if is_pdf else "web"}
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    via = " via Wayback Machine" if fetched_from != url else ""
    print(f"  saved   {dest.name} ({len(r.content) // 1024} KB){via}")
    time.sleep(4)
    return dest


def fetch_10k(company: str, cik: int, fiscal_year: int, out_dir: Path, force: bool) -> Path | None:
    if not config.SEC_USER_AGENT:
        print("  SKIPPED SEC: set SEC_USER_AGENT in .env (EDGAR requires a contact User-Agent).")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{company}-10k-fy{fiscal_year}.html"
    if dest.exists() and not force:
        print(f"  cached  {dest.name}")
        return dest
    subs = fetch(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", config.SEC_USER_AGENT).json()
    recent = subs["filings"]["recent"]
    chosen = None
    for i, form in enumerate(recent["form"]):
        if form == "10-K" and recent["reportDate"][i].startswith(str(fiscal_year)):
            chosen = i
            break
    if chosen is None:
        print(f"  FAILED  no FY{fiscal_year} 10-K found for {company}")
        return None
    acc = recent["accessionNumber"][chosen].replace("-", "")
    doc = recent["primaryDocument"][chosen]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
    r = fetch(url, config.SEC_USER_AGENT)
    dest.write_bytes(r.content)
    meta = {"company": company, "cik": cik, "form": "10-K", "url": url,
            "filing_date": recent["filingDate"][chosen], "report_date": recent["reportDate"][chosen],
            "title": f"{subs.get('name', company.title())} Form 10-K FY{fiscal_year}"}
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved   {dest.name} ({len(r.content) // 1024} KB) filed {meta['filing_date']}")
    return dest


def main(force: bool = False) -> None:
    raw = config.RAW_DIR
    print("Technology: Cisco")
    for title, url in parse_url_list(config.CISCO_SOURCES):
        save_page(title, url, raw / "technology", force)
    print("Technology: Google Cloud")
    for title, url in parse_url_list(config.GCP_SOURCES):
        save_page(title, url, raw / "technology", force)
    print("Clinical: Agilent")
    for title, url in parse_url_list(config.AGILENT_SOURCES):
        save_page(title, url, raw / "clinical", force)
    print("SEC: EDGAR 10-K filings")
    for company, (cik, fy) in SEC_COMPANIES.items():
        try:
            fetch_10k(company, cik, fy, raw / "sec", force)
        except requests.RequestException as e:
            print(f"  FAILED  {company}: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download cached files")
    main(ap.parse_args().force)
