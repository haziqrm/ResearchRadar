"""Pull recent arXiv q-fin papers (including cross-lists) and cache them.

Usage:
    python -m src.live.arxiv_feed [--months 24] [--refresh]

Queries the arXiv API one calendar month at a time, paging within a month,
at most one request every 3 seconds (arXiv's rule). Raw responses are cached in
data/cache/arxiv/raw/, so a rerun only fetches what's missing. Months that ended
more than CLOSED_AFTER_DAYS ago are treated as final; newer months are refetched
when you pass --refresh.

Writes data/cache/arxiv/papers.csv with PAPER_COLUMNS, one row per arXiv ID.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import feedparser
import pandas as pd
import requests

from src.live import PAPERS_CSV, RAW_DIR

API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "ResearchRadar/0.1 (hackathon research project)"
REQUEST_GAP_SECONDS = 3.0
PAGE_SIZE = 1000
CLOSED_AFTER_DAYS = 7
MAX_RETRIES = 4

Q_FIN_CATEGORIES = ["CP", "EC", "GN", "MF", "PM", "PR", "RM", "ST", "TR"]
PAPER_COLUMNS = ["arxiv_id", "title", "abstract", "published", "updated",
                 "primary_category", "categories", "authors"]

_last_request = 0.0


def month_starts(end: date, months: int) -> list[date]:
    """First day of each of the `months` calendar months up to and including end's."""
    y, m = end.year, end.month
    out = []
    for _ in range(months):
        out.append(date(y, m, 1))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return out[::-1]


def next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def build_query(start: date, end: date) -> str:
    """Search query for q-fin papers submitted in [start, end)."""
    cats = " OR ".join(f"cat:q-fin.{c}" for c in Q_FIN_CATEGORIES)
    last = end - timedelta(days=1)
    return f"({cats}) AND submittedDate:[{start:%Y%m%d}0000 TO {last:%Y%m%d}2359]"


def _raw_path(month: date, offset: int) -> Path:
    return RAW_DIR / f"{month:%Y-%m}_{offset:05d}.xml"


def _get(params: dict) -> str:
    """One polite API request, retried with backoff on errors or empty pages."""
    global _last_request
    for attempt in range(MAX_RETRIES):
        wait = _last_request + REQUEST_GAP_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            resp = requests.get(API_URL, params=params, timeout=60,
                                headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  request failed ({exc}); retrying")
            time.sleep(REQUEST_GAP_SECONDS * 2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_page(month: date, offset: int, use_cache: bool) -> str:
    """Raw Atom XML for one page of one month, from cache when allowed."""
    path = _raw_path(month, offset)
    if use_cache and path.exists():
        return path.read_text(encoding="utf-8")

    params = {"search_query": build_query(month, next_month(month)),
              "start": offset, "max_results": PAGE_SIZE,
              "sortBy": "submittedDate", "sortOrder": "ascending"}
    for attempt in range(MAX_RETRIES):
        xml = _get(params)
        feed = feedparser.parse(xml)
        total = int(feed.feed.get("opensearch_totalresults", 0))
        # arXiv sometimes returns an empty page for a query that has results.
        if feed.entries or total <= offset:
            break
        print(f"  empty page for {month:%Y-%m} at {offset} (total {total}); retrying")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return xml


def parse_feed(xml: str) -> tuple[list[dict], int]:
    """Rows in PAPER_COLUMNS order, plus the query's total result count."""
    feed = feedparser.parse(xml)
    rows = []
    for e in feed.entries:
        # id looks like http://arxiv.org/abs/2410.01234v2; drop the version.
        arxiv_id = e.id.rsplit("/abs/", 1)[-1].rsplit("v", 1)[0]
        rows.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(e.title.split()),
            "abstract": " ".join(e.summary.split()),
            "published": e.published[:10],
            "updated": e.updated[:10],
            "primary_category": e.get("arxiv_primary_category", {}).get("term", ""),
            "categories": " ".join(t["term"] for t in e.get("tags", [])),
            "authors": "; ".join(a.get("name", "") for a in e.get("authors", [])),
        })
    return rows, int(feed.feed.get("opensearch_totalresults", len(rows)))


def fetch_month(month: date, today: date, refresh: bool) -> list[dict]:
    closed = next_month(month) + timedelta(days=CLOSED_AFTER_DAYS) <= today
    use_cache = closed or not refresh
    rows, offset = [], 0
    while True:
        page, total = parse_feed(fetch_page(month, offset, use_cache))
        rows += page
        offset += PAGE_SIZE
        if not page or offset >= total:
            break
    return rows


def fetch_recent(months: int = 24, refresh: bool = False,
                 today: date | None = None) -> pd.DataFrame:
    """Fetch (or load from cache) the last `months` months of q-fin papers."""
    today = today or date.today()
    rows = []
    for month in month_starts(today, months):
        got = fetch_month(month, today, refresh)
        print(f"{month:%Y-%m}: {len(got)} papers")
        rows += got
    df = pd.DataFrame(rows, columns=PAPER_COLUMNS)
    return (df.sort_values(["arxiv_id", "updated"])
              .drop_duplicates("arxiv_id", keep="last")
              .sort_values("published")
              .reset_index(drop=True))


def load_papers() -> pd.DataFrame:
    if not PAPERS_CSV.exists():
        raise FileNotFoundError(f"{PAPERS_CSV} missing; run python -m src.live.arxiv_feed")
    return pd.read_csv(PAPERS_CSV, dtype=str, keep_default_na=False)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Pull recent arXiv q-fin papers.")
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--refresh", action="store_true",
                        help="refetch months that are still open")
    args = parser.parse_args(argv)
    df = fetch_recent(args.months, args.refresh)
    PAPERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PAPERS_CSV, index=False)
    print(f"wrote {len(df)} papers to {PAPERS_CSV}")
    return PAPERS_CSV


if __name__ == "__main__":
    main()
