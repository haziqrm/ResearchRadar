"""Research attention per paper: citations by year and early citations.

OpenAlex's counts_by_year only covers recent years, so it cannot give early
citations for papers from the 1980s-90s. Instead we count the works that cite
a paper, grouped by their publication year:

    GET /works?filter=cites:<id>,publication_year:<y1>-<y2>&group_by=publication_year

Every response is cached as JSON in data/cache/openalex. Network access only
happens inside fetch_citing_counts_by_year().

Paper -> OpenAlex ID mapping lives in data/paper_ids.csv (signal, openalex_id),
filled by hand after checking each match.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.signal_decay import OPENALEX_CACHE_DIR, PAPER_IDS_CSV, ROOT_DIR

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
PER_PAGE = 100

_ID = re.compile(r"^W\d+$")


def get_openalex_credentials() -> tuple[str, str]:
    """Return (OPENALEX_API_KEY, OPENALEX_EMAIL) from the environment, loading .env."""
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv("OPENALEX_API_KEY", "").strip(), os.getenv("OPENALEX_EMAIL", "").strip()


def load_paper_ids(path: Path = PAPER_IDS_CSV) -> dict[str, str]:
    """Map signal -> OpenAlex work ID (e.g. W2034567890) from data/paper_ids.csv."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str).dropna()
    bad = [i for i in df["openalex_id"] if not _ID.match(i.strip())]
    if bad:
        raise ValueError(f"paper_ids.csv has malformed OpenAlex IDs: {bad}")
    return {s.strip(): i.strip() for s, i in zip(df["signal"], df["openalex_id"])}


def fetch_citing_counts_by_year(
    openalex_id: str,
    start_year: int,
    end_year: int,
    cache_dir: Path = OPENALEX_CACHE_DIR,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Count works citing ``openalex_id`` per publication year in [start_year, end_year].

    Returns columns year, cited_by_count (years with no citations are absent).
    Uses the cached response if present; otherwise needs OPENALEX_API_KEY.
    """
    if not _ID.match(openalex_id):
        raise ValueError(f"Not an OpenAlex work ID: {openalex_id!r}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"cites_{openalex_id}_{start_year}_{end_year}.json"

    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        api_key, email = get_openalex_credentials()
        if not api_key:
            raise RuntimeError("OPENALEX_API_KEY is not set; add it to .env (see .env.example)")
        params = {
            "filter": f"cites:{openalex_id},publication_year:{start_year}-{end_year}",
            "group_by": "publication_year",
            "per_page": PER_PAGE,
            "api_key": api_key,
        }
        if email:
            params["mailto"] = email
        resp = (session or requests).get(OPENALEX_WORKS_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

    groups = payload.get("group_by", [])
    return pd.DataFrame(
        {"year": [int(g["key"]) for g in groups], "cited_by_count": [int(g["count"]) for g in groups]},
        columns=["year", "cited_by_count"],
    )


def compute_early_citations(counts: pd.DataFrame | None, publication_year: int) -> int | None:
    """Sum citations in years 1 to 3 after publication (pub_year+1 to pub_year+3).

    counts has columns year, cited_by_count from fetch_citing_counts_by_year;
    years with no row count as zero. Returns None if counts is None (no data).
    """
    if counts is None:
        return None
    first, last = publication_year + 1, publication_year + 3
    window = counts[(counts["year"] >= first) & (counts["year"] <= last)]
    return int(window["cited_by_count"].sum())


def early_citations(openalex_id: str, publication_year: int, **kwargs) -> int | None:
    """Fetch (or read cached) citations and return the years 1 to 3 total."""
    counts = fetch_citing_counts_by_year(
        openalex_id, publication_year + 1, publication_year + 3, **kwargs
    )
    return compute_early_citations(counts, publication_year)
