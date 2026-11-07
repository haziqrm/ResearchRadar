"""Research attention per paper: early citations, raw and adjusted for era.

OpenAlex's counts_by_year only covers recent years, so it cannot give early
citations for papers from the 1980s-90s. Instead we count the works that cite
a paper, grouped by their publication year:

    GET /works?filter=cites:<id>,publication_year:<y1>-<y2>&group_by=publication_year

Raw counts are not comparable across decades (far more papers are published
now, and OpenAlex records old references less completely). So each paper is
also ranked against peers: articles in the same journal and publication year,
counted over the same citation window. See HYPOTHESES.md (H3).

Every response is cached as JSON in data/cache/openalex. Network access only
happens inside _get_json(). Paper -> OpenAlex ID mapping lives in
data/paper_ids.csv (signal, openalex_id), checked by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import stats

from src.signal_decay import OPENALEX_CACHE_DIR, PAPER_IDS_CSV, ROOT_DIR

OPENALEX_API = "https://api.openalex.org"
OPENALEX_WORKS_URL = f"{OPENALEX_API}/works"
PER_PAGE = 100
PEER_SAMPLE_SIZE = 50
PEER_SEED = 0
_REQUEST_GAP_SECONDS = 0.2  # stay well under OpenAlex's per-second limit
_MAX_TRIES = 6

_ID = re.compile(r"^W\d+$")


def get_openalex_credentials() -> tuple[str, str]:
    """Return (OPENALEX_API_KEY, OPENALEX_EMAIL) from the environment, loading .env."""
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv("OPENALEX_API_KEY", "").strip(), os.getenv("OPENALEX_EMAIL", "").strip()


def _get_json(
    path: str,
    params: dict,
    cache_name: str,
    cache_dir: Path = OPENALEX_CACHE_DIR,
    session: requests.Session | None = None,
) -> dict:
    """GET an OpenAlex endpoint, caching the JSON under cache_name.

    The API key is added to the request but never written to the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    api_key, email = get_openalex_credentials()
    if not api_key:
        raise RuntimeError("OPENALEX_API_KEY is not set; add it to .env (see .env.example)")
    full = {**params, "api_key": api_key}
    if email:
        full["mailto"] = email
    for attempt in range(_MAX_TRIES):
        resp = (session or requests).get(f"{OPENALEX_API}/{path}", params=full, timeout=30)
        status = getattr(resp, "status_code", 200)
        if status != 429 and status < 500:
            break
        headers = getattr(resp, "headers", {}) or {}
        if status == 429 and str(headers.get("X-RateLimit-Remaining", "1")) == "0":
            raise RuntimeError(
                f"OpenAlex daily budget used up; it resets in {headers.get('X-RateLimit-Reset', '?')} s. "
                "Cached responses are kept, so rerun later to continue."
            )
        wait = float(headers.get("Retry-After") or 2 ** attempt)
        time.sleep(min(wait, 60))
    if status >= 400:
        # Never include the request URL: it carries the API key.
        raise RuntimeError(f"OpenAlex request to /{path} failed with HTTP {status}")
    payload = resp.json()
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    if session is None:
        time.sleep(_REQUEST_GAP_SECONDS)
    return payload


def load_paper_ids(path: Path = PAPER_IDS_CSV) -> dict[str, str]:
    """Map signal -> OpenAlex work ID (e.g. W2034567890) from data/paper_ids.csv."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str).dropna()
    bad = [i for i in df["openalex_id"] if not _ID.match(i.strip())]
    if bad:
        raise ValueError(f"paper_ids.csv has malformed OpenAlex IDs: {bad}")
    return {s.strip(): i.strip() for s, i in zip(df["signal"], df["openalex_id"])}


def _check_id(openalex_id: str) -> None:
    if not _ID.match(openalex_id):
        raise ValueError(f"Not an OpenAlex work ID: {openalex_id!r}")


# ---------- raw early citations ----------

def fetch_citing_counts_by_year(
    openalex_id: str,
    start_year: int,
    end_year: int,
    cache_dir: Path = OPENALEX_CACHE_DIR,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Count works citing ``openalex_id`` per publication year in [start_year, end_year].

    Returns columns year, cited_by_count (years with no citations are absent).
    """
    _check_id(openalex_id)
    payload = _get_json(
        "works",
        {
            "filter": f"cites:{openalex_id},publication_year:{start_year}-{end_year}",
            "group_by": "publication_year",
            "per_page": PER_PAGE,
        },
        f"cites_{openalex_id}_{start_year}_{end_year}",
        cache_dir,
        session,
    )
    groups = payload.get("group_by", [])
    return pd.DataFrame(
        {"year": [int(g["key"]) for g in groups], "cited_by_count": [int(g["count"]) for g in groups]},
        columns=["year", "cited_by_count"],
    )


def compute_early_citations(counts: pd.DataFrame | None, publication_year: int) -> int | None:
    """Sum citations in years 1 to 3 after publication (pub_year+1 to pub_year+3).

    counts has columns year, cited_by_count; years with no row count as zero.
    Returns None if counts is None (no data).
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


# ---------- era adjustment: rank against journal-year peers ----------

def fetch_work(openalex_id: str, **kwargs) -> dict:
    """Return id, publication_year and journal (source) id for one work."""
    _check_id(openalex_id)
    w = _get_json(
        f"works/{openalex_id}",
        {"select": "id,publication_year,primary_location"},
        f"work_{openalex_id}",
        **kwargs,
    )
    source = ((w.get("primary_location") or {}).get("source") or {}).get("id")
    return {
        "id": w["id"].rsplit("/", 1)[-1],
        "publication_year": w["publication_year"],
        "source_id": source.rsplit("/", 1)[-1] if source else None,
    }


def fetch_peer_ids(source_id: str, year: int, **kwargs) -> list[str]:
    """All articles in one journal-year with at least one citation ever (drops front matter)."""
    ids, cursor = [], "*"
    while cursor:
        page = _get_json(
            "works",
            {
                "filter": f"primary_location.source.id:{source_id},publication_year:{year},"
                          "type:article,cited_by_count:>0",
                "select": "id",
                "per_page": PER_PAGE,
                "cursor": cursor,
            },
            f"peers_{source_id}_{year}_{hashlib.md5(cursor.encode()).hexdigest()[:8]}",
            **kwargs,
        )
        ids += [r["id"].rsplit("/", 1)[-1] for r in page.get("results", [])]
        cursor = page.get("meta", {}).get("next_cursor")
    return ids


def citing_count(openalex_id: str, start_year: int, end_year: int, **kwargs) -> int:
    """Number of works citing openalex_id published in [start_year, end_year]."""
    _check_id(openalex_id)
    payload = _get_json(
        "works",
        {"filter": f"cites:{openalex_id},publication_year:{start_year}-{end_year}", "per_page": 1,
         "select": "id"},
        f"count_{openalex_id}_{start_year}_{end_year}",
        **kwargs,
    )
    return int(payload["meta"]["count"])


def sample_peers(peer_ids: list[str], exclude: str, n: int = PEER_SAMPLE_SIZE, seed: int = PEER_SEED) -> list[str]:
    """Deterministic random sample of up to n peers, excluding the paper itself."""
    pool = sorted(p for p in peer_ids if p != exclude)
    if len(pool) <= n:
        return pool
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def peer_percentile(value: float, peer_values: list[float]) -> float | None:
    """Percentile (0-100) of value among peers; ties count half. None if no peers."""
    if not peer_values:
        return None
    peers = np.asarray(peer_values, dtype=float)
    return float(100 * ((peers < value).sum() + 0.5 * (peers == value).sum()) / len(peers))


def era_adjusted_attention(
    openalex_id: str, publication_year: int, peer_n: int = PEER_SAMPLE_SIZE, **kwargs
) -> dict:
    """Early citations of a paper and its percentile among journal-year peers.

    Peers: up to ``peer_n`` articles from the same journal and the same
    OpenAlex publication year, with the same citation window
    (publication_year+1 to +3, using the given publication year).
    """
    window = (publication_year + 1, publication_year + 3)
    own = citing_count(openalex_id, *window, **kwargs)
    work = fetch_work(openalex_id, **kwargs)
    if not work["source_id"]:
        return {"early_citations": own, "peer_n": 0, "peer_median": None, "percentile": None}
    peers = sample_peers(fetch_peer_ids(work["source_id"], work["publication_year"], **kwargs),
                         openalex_id, n=peer_n)
    peer_counts = [citing_count(p, *window, **kwargs) for p in peers]
    return {
        "early_citations": own,
        "peer_n": len(peer_counts),
        "peer_median": float(np.median(peer_counts)) if peer_counts else None,
        "percentile": peer_percentile(own, peer_counts),
    }


# ---------- H3: attention versus decay ----------

def attention_decay_test(attention: pd.Series, decay: pd.Series) -> dict:
    """Spearman correlation of attention and decay, exact one-sided permutation p.

    H3 expects a positive correlation (more attention, more decay). The p value
    is the share of all orderings of ``decay`` whose correlation is at least the
    observed one. Rows with missing values are dropped. Needs at least 3 rows.
    """
    df = pd.DataFrame({"a": attention, "d": decay}).dropna()
    n = len(df)
    if n < 3:
        return {"n": n, "spearman": np.nan, "p_one_sided": np.nan}
    a = stats.rankdata(df["a"])
    d = stats.rankdata(df["d"])
    observed = stats.spearmanr(a, d).statistic
    if n <= 8:
        perms = (np.array(p) for p in permutations(d))
        rhos = np.array([stats.spearmanr(a, p).statistic for p in perms])
    else:  # too many orderings to list; sample them
        rng = np.random.default_rng(0)
        rhos = np.array([stats.spearmanr(a, rng.permutation(d)).statistic for _ in range(20_000)])
    p = float((rhos >= observed - 1e-12).mean())
    return {"n": n, "spearman": float(observed), "p_one_sided": p}
