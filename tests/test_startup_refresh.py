import json
from datetime import date

import pandas as pd
import pytest

from src.live import arxiv_feed as feed
from src.live import map_tags as mt
from src.live import startup_refresh as sr

AREA = "Behavioral finance"


def paper(i, published):
    return {"arxiv_id": f"p{i}", "title": f"t{i}", "abstract": "a", "published": published,
            "updated": published, "primary_category": "q-fin.ST", "categories": "q-fin.ST",
            "authors": "x"}


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Two papers already tagged; the paths the refresh writes point into tmp_path."""
    monkeypatch.delenv("LIVE_REFRESH", raising=False)
    old = pd.DataFrame([paper(0, "2026-10-30"), paper(1, "2026-11-01")])
    papers_csv, tags_csv, cache = (tmp_path / n for n in
                                   ("papers.csv", "tags.csv", "cache.jsonl"))
    old.to_csv(papers_csv, index=False)
    mt.build_table(old, {"p0": {"area": AREA, "confidence": 0.9},
                         "p1": {"area": "", "confidence": 0.5}}).to_csv(tags_csv, index=False)
    monkeypatch.setattr(sr, "PAPERS_CSV", papers_csv)
    monkeypatch.setattr(feed, "load_papers",
                        lambda: pd.read_csv(papers_csv, dtype=str, keep_default_na=False))
    monkeypatch.setattr(mt, "MAP_TAGS_CSV", tags_csv)
    monkeypatch.setattr(mt, "MAP_CACHE", cache)
    return tags_csv


def arxiv_with(new):
    return lambda month, today, refresh: [p for p in new if p["published"][:7] == f"{month:%Y-%m}"]


def answer_all(calls):
    def call(client, model, prompt):
        calls.append(prompt)
        ids = [p["id"] for p in json.loads(prompt.split("Papers:\n", 1)[1])]
        return json.dumps([{"id": i, "area": AREA, "confidence": 1} for i in ids])
    return call


def tags(path):
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(t.arxiv_id, t.area))


def test_new_papers_are_tagged_in_one_call_and_old_tags_kept(files):
    calls = []
    new = [paper(i, "2026-11-07") for i in range(2, 2 + mt.BATCH_SIZE + 5)]
    sr.refresh(client_factory=lambda: (None, "m"), fetch_month=arxiv_with(new),
               call=answer_all(calls))
    got = tags(files)
    assert len(calls) == 1                               # never more than one request
    assert got["p0"] == AREA and got["p1"] == "none"     # earlier answers untouched
    assert sum(a == AREA for a in got.values()) == 1 + mt.BATCH_SIZE
    assert sum(a == "" for a in got.values()) == 5       # left for the next start


def test_no_key_means_no_call_and_no_change(files):
    before = files.read_text()
    sr.refresh(client_factory=lambda: None, fetch_month=arxiv_with([]),
               call=lambda *a: pytest.fail("Gemini called without a key"))
    assert files.read_text() == before


def test_gemini_and_arxiv_errors_leave_the_map_as_it_was(files):
    def offline(*a):
        raise ConnectionError("offline")
    before = files.read_text()
    sr.refresh(client_factory=lambda: (None, "m"), fetch_month=offline, call=offline)
    assert files.read_text() == before


def test_switched_off(files, monkeypatch):
    monkeypatch.setenv("LIVE_REFRESH", "0")
    sr.refresh(client_factory=lambda: pytest.fail("ran"), fetch_month=lambda *a: pytest.fail("ran"))


def test_fetch_starts_at_the_newest_papers_month():
    asked = []
    papers = pd.DataFrame([paper(0, "2026-09-20")]).astype(str)
    sr.fetch_new_papers(papers, today=date(2026, 11, 8),
                        fetch_month=lambda m, t, r: asked.append(m) or [])
    assert asked == [date(2026, 9, 1), date(2026, 10, 1), date(2026, 11, 1)]


def test_rechecks_on_a_timer_until_stopped(monkeypatch):
    import threading
    runs, stop = [], threading.Event()
    def fake_refresh():
        runs.append(1)
        if len(runs) == 3:
            stop.set()
    monkeypatch.setattr(sr, "refresh", fake_refresh)
    monkeypatch.setenv("LIVE_REFRESH_HOURS", str(0.01 / 3600))   # 10 ms
    sr.start_in_background(stop).join(timeout=5)
    assert len(runs) == 3


def test_hours_zero_checks_only_at_start(monkeypatch):
    runs = []
    monkeypatch.setattr(sr, "refresh", lambda: runs.append(1))
    monkeypatch.setenv("LIVE_REFRESH_HOURS", "0")
    thread = sr.start_in_background()
    thread.join(timeout=5)
    assert not thread.is_alive() and runs == [1]


def test_no_arxiv_cache_keeps_every_tagged_paper(files, monkeypatch):
    """The arXiv cache is git-ignored, so most machines start without it. The refresh
    must then start from the tags CSV, not from this month's papers alone."""
    def no_cache():
        raise FileNotFoundError("no cache")
    monkeypatch.setattr(feed, "load_papers", no_cache)
    calls = []
    sr.refresh(client_factory=lambda: (None, "m"),
               fetch_month=arxiv_with([paper(9, "2026-11-07")]), call=answer_all(calls))
    got = tags(files)
    assert got["p0"] == AREA and got["p1"] == "none"     # old papers still there
    assert got["p9"] == AREA and len(got) == 3           # and the new one added
    assert len(calls) == 1


def test_never_writes_a_smaller_table(files):
    before = files.read_text()
    one = pd.DataFrame([paper(0, "2026-10-30")]).astype(str)
    assert sr.write_table(one, {"p0": {"area": AREA, "confidence": 1}}, files) is False
    assert files.read_text() == before
