"""Keep the monthly map up to date while the website runs (Person D).

Called by html_main.py in a background thread, so the site is up straight away.
It runs once at start and again every LIVE_REFRESH_HOURS (default 2; 0 = start only).
Each run:

1. arXiv: refetch the months from the newest cached paper up to today (free, no key)
   and add any new papers to data/cache/arxiv/papers.csv.
2. Gemini: tag the papers that have no map area yet in ONE request, at most
   map_tags.BATCH_SIZE (50) of them, newest first. No retries, so a run never
   costs more than one call, and a run with nothing new to tag makes none. A larger
   backlog is worked through over later runs. (Free tier: 20 requests a day; every
   2 hours is at most 12.)
3. Rewrite data/arxiv_map_tags.csv, which /api/network/monthly reads on every request,
   so a page reload shows the new papers.

Any failure (no network, no GEMINI_API_KEY, google-genai not installed, quota, bad
answer) is printed as one line and the site runs on the data it already has.
Set LIVE_REFRESH=0 to skip it (tests, offline demos).

Usage outside the site:  python -m src.live.startup_refresh
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.common.contracts import ROOT
from src.live import PAPERS_CSV
from src.live import arxiv_feed as feed
from src.live import map_tags as mt
from src.live.tagging import DEFAULT_MODEL

DEFAULT_EVERY_HOURS = "2"   # re-check while the site runs (LIVE_REFRESH_HOURS)


def _log(msg: str) -> None:
    print(f"live refresh: {msg}", flush=True)


def fetch_new_papers(papers: pd.DataFrame, today: date | None = None,
                     fetch_month=feed.fetch_month) -> pd.DataFrame:
    """papers plus anything arXiv has from the newest paper's month to today."""
    today = today or date.today()
    newest = date.fromisoformat(papers["published"].max()) if len(papers) else today
    rows, month = [], date(newest.year, newest.month, 1)
    while month <= today:
        rows += fetch_month(month, today, True)   # refresh=True: open months are refetched
        month = feed.next_month(month)
    fresh = pd.DataFrame(rows, columns=feed.PAPER_COLUMNS).astype(str)
    return (pd.concat([papers, fresh], ignore_index=True)
              .sort_values(["arxiv_id", "updated"])
              .drop_duplicates("arxiv_id", keep="last")
              .sort_values("published")
              .reset_index(drop=True))


def with_tagged_papers(papers: pd.DataFrame, path: Path = mt.MAP_TAGS_CSV) -> pd.DataFrame:
    """papers plus every paper already in the tags CSV. The arXiv cache (PAPERS_CSV) is
    git-ignored, so on any machine but the one that built it, papers starts empty; the
    committed CSV is then the only record of the 3,687 tagged papers. Without them the
    rewrite would keep only this month's papers."""
    if not path.exists():
        return papers
    tagged = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = tagged[~tagged["arxiv_id"].isin(papers["arxiv_id"])]
    if missing.empty:
        return papers
    extra = pd.DataFrame({c: missing[c] if c in missing else "" for c in feed.PAPER_COLUMNS})
    extra["updated"] = extra["published"]
    return (pd.concat([papers, extra.astype(str)], ignore_index=True)
              .sort_values("published").reset_index(drop=True))


def untagged(papers: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """The newest BATCH_SIZE papers with no cached Gemini answer."""
    todo = papers[~papers["arxiv_id"].isin(cache)]
    return todo.sort_values("published", ascending=False).head(mt.BATCH_SIZE)


def known_tags(model: str, path: Path = mt.MAP_TAGS_CSV,
               cache_path: Path = mt.MAP_CACHE) -> dict:
    """Areas already in the CSV, overlaid with the model's cached answers, so a run
    with a different GEMINI_MODEL never blanks papers tagged before."""
    tags = {}
    if path.exists():
        for r in pd.read_csv(path, dtype=str, keep_default_na=False).itertuples():
            if r.area:   # "" = never tagged; "none" is an answer (stored as "")
                tags[r.arxiv_id] = {"arxiv_id": r.arxiv_id, "confidence": r.confidence,
                                    "area": "" if r.area == "none" else r.area}
    tags.update(mt.load_cache(cache_path, model))
    return tags


def gemini_client():
    """(client, model), or None when there is no key or no SDK."""
    load_dotenv(ROOT / ".env")
    load_dotenv(find_dotenv(usecwd=True))   # also the folder the site was started from
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from google import genai
    except ImportError:
        return None
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    return genai.Client(api_key=key), model


def tag_once(batch: pd.DataFrame, client, model: str, cache: dict,
             cache_path: Path = mt.MAP_CACHE, call=mt.call_gemini) -> int:
    """One Gemini request for batch; answers go into cache and the cache file."""
    answers = mt.clean_answers(call(client, model, mt.build_prompt(batch)),
                               batch["arxiv_id"].tolist())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for a in answers:
            a["model"] = model
            f.write(json.dumps(a) + "\n")
            cache[a["arxiv_id"]] = a
    return len(answers)


def write_table(papers: pd.DataFrame, cache: dict, path: Path = mt.MAP_TAGS_CSV) -> bool:
    """Write via a temp file, so a request reading the CSV never sees half of it.
    Refuses (returns False) to write fewer papers than the CSV already has: a refresh
    only ever adds papers, so a smaller table means something upstream went wrong."""
    table = mt.build_table(papers, cache)
    if path.exists():
        existing = len(pd.read_csv(path, dtype=str, usecols=["arxiv_id"]))
        if len(table) < existing:
            _log(f"not writing {path.name}: {len(table)} papers would replace {existing}")
            return False
    tmp = path.with_suffix(".tmp")
    table.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return True


def refresh(client_factory=gemini_client, fetch_month=feed.fetch_month,
            call=mt.call_gemini) -> None:
    if os.getenv("LIVE_REFRESH", "1").strip() == "0":
        return
    try:
        papers = feed.load_papers()
    except FileNotFoundError:
        papers = pd.DataFrame(columns=feed.PAPER_COLUMNS)
    papers = with_tagged_papers(papers, mt.MAP_TAGS_CSV)

    before = len(papers)
    try:
        papers = fetch_new_papers(papers, fetch_month=fetch_month)
        if len(papers) > before:
            PAPERS_CSV.parent.mkdir(parents=True, exist_ok=True)
            papers.to_csv(PAPERS_CSV, index=False)
        _log(f"arXiv: {len(papers) - before} new papers")
    except Exception as exc:
        _log(f"arXiv unavailable ({type(exc).__name__}); using cached papers")
    if papers.empty:
        return

    try:
        got = client_factory()
    except Exception as exc:
        got = None
        _log(f"Gemini client failed ({type(exc).__name__})")
    if got is None:
        _log("no Gemini key or SDK; map not re-tagged")
        return
    client, model = got
    cache = known_tags(model, mt.MAP_TAGS_CSV, mt.MAP_CACHE)
    batch = untagged(papers, cache)
    if batch.empty and len(papers) == before:
        _log("nothing new to tag")
        return
    if not batch.empty:
        try:
            n = tag_once(batch, client, model, cache, mt.MAP_CACHE, call=call)
            left = (~papers["arxiv_id"].isin(cache)).sum()
            _log(f"Gemini tagged {n} papers in 1 call ({left} still untagged)")
        except Exception as exc:
            _log(f"Gemini unavailable ({type(exc).__name__}: {str(exc)[:100]}); "
                 "map not re-tagged")
            if len(papers) == before:
                return
    try:
        write_table(papers, cache, mt.MAP_TAGS_CSV)
    except OSError as exc:
        _log(f"could not write {mt.MAP_TAGS_CSV.name} ({exc})")


def every_hours() -> float:
    """LIVE_REFRESH_HOURS (default 2); 0 or junk = only check at start."""
    try:
        return max(float(os.getenv("LIVE_REFRESH_HOURS", DEFAULT_EVERY_HOURS)), 0.0)
    except ValueError:
        return 0.0


def start_in_background(stop: threading.Event | None = None) -> threading.Thread:
    """Run refresh() now and then every LIVE_REFRESH_HOURS, without holding up the
    server; never raises. The thread is a daemon, so it ends with the site."""
    stop = stop or threading.Event()

    def run():
        while True:
            try:
                refresh()
            except Exception as exc:   # the site must keep running whatever happens here
                _log(f"skipped ({type(exc).__name__}: {str(exc)[:100]})")
            hours = every_hours()
            if not hours or stop.wait(hours * 3600):
                return
    thread = threading.Thread(target=run, name="live-refresh", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    refresh()
