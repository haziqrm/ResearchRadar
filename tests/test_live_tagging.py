import json

import pandas as pd
import pytest

from src.live import tagging as tg
from src.live.tagging import AREAS  # D's original 8 areas, kept for the signal match

SIGNALS = ["Size", "Value", "Momentum", "Short-term reversal", "Long-term reversal",
           "Profitability", "Investment"]


# --- keyword tagger --------------------------------------------------------
@pytest.mark.parametrize("title, abstract, area, signal", [
    ("Can ChatGPT forecast stock returns?", "We prompt large language models.",
     "LLMs in finance", ""),
    ("Bitcoin liquidity", "Decentralized finance and stablecoins.", "Crypto and DeFi", ""),
    ("Deep learning for the limit order book", "We model the order book with LSTMs.",
     "Market microstructure", ""),
    ("Rough volatility and option pricing", "Implied volatility surfaces.",
     "Derivatives and volatility", ""),
    ("Momentum crashes", "Time-series momentum and the cross-section of stock returns.",
     "Factor and anomaly research", "Momentum"),
    ("A note on Brownian motion", "Pure mathematics.", "", ""),
])
def test_keyword_tag(title, abstract, area, signal):
    tag = tg.keyword_tag(title, abstract, SIGNALS)
    assert (tag["area"], tag["signal"], tag["tagger"]) == (area, signal, "keyword")


def test_option_does_not_match_optimal():
    assert tg.keyword_tag("Optimal stopping", "optimality", SIGNALS)["area"] == ""


# --- Gemini tagger (fake client) -------------------------------------------
def papers_frame(n):
    return pd.DataFrame({"arxiv_id": [f"2501.{i:05d}" for i in range(n)],
                         "title": ["t"] * n, "abstract": ["a"] * n,
                         "published": ["2025-01-06"] * n})


def test_clean_answers_forces_fixed_lists():
    raw = json.dumps([
        {"id": "1", "area": "LLMs in finance", "signal": "Momentum", "confidence": 0.9},
        {"id": "2", "area": "Astrology", "signal": "Vibes", "confidence": 3},
        {"id": "99", "area": "LLMs in finance", "signal": "none", "confidence": 1},
    ])
    out = tg.clean_answers(raw, ["1", "2"], SIGNALS)
    assert out == [
        {"arxiv_id": "1", "area": "LLMs in finance", "signal": "Momentum",
         "confidence": 0.9, "tagger": "gemini"},
        {"arxiv_id": "2", "area": "", "signal": "", "confidence": 1.0, "tagger": "gemini"},
    ]


def test_gemini_batches_and_caches(tmp_path):
    cache = tmp_path / "tags.jsonl"
    calls = []

    def fake_call(client, model, prompt):
        ids = [p["id"] for p in json.loads(prompt.split("Papers:\n", 1)[1])]
        calls.append(ids)
        return json.dumps([{"id": i, "area": AREAS[0], "signal": "none",
                            "confidence": 0.5} for i in ids])

    papers = papers_frame(60)
    tg.gemini_tag(papers, SIGNALS, None, "m", cache, gap=0, call=fake_call)
    assert [len(c) for c in calls] == [50, 10]                 # default batch size
    assert len(tg.load_gemini_cache(cache)) == 60

    calls.clear()                                   # rerun: everything cached
    tg.gemini_tag(papers, SIGNALS, None, "m", cache, gap=0, call=fake_call)
    assert calls == []


def test_switching_model_retags_everything(tmp_path):
    cache = tmp_path / "tags.jsonl"
    seen = []

    def fake_call(client, model, prompt):
        ids = [p["id"] for p in json.loads(prompt.split("Papers:\n", 1)[1])]
        seen.append(model)
        return json.dumps([{"id": i, "area": AREAS[0], "signal": "none",
                            "confidence": 1} for i in ids])

    papers = papers_frame(10)
    tg.gemini_tag(papers, SIGNALS, None, "old", cache, gap=0, call=fake_call)
    tg.gemini_tag(papers, SIGNALS, None, "new", cache, gap=0, call=fake_call)
    assert seen == ["old", "new"]
    assert {r["model"] for r in tg.load_gemini_cache(cache, "new").values()} == {"new"}
    assert len(tg.load_gemini_cache(cache)) == 10


def test_gemini_quota_stops_but_keeps_progress(tmp_path):
    cache = tmp_path / "tags.jsonl"
    n = {"calls": 0}

    def flaky(client, model, prompt):
        n["calls"] += 1
        if n["calls"] > 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        ids = [p["id"] for p in json.loads(prompt.split("Papers:\n", 1)[1])]
        return json.dumps([{"id": i, "area": AREAS[1], "signal": "none",
                            "confidence": 1} for i in ids])

    with pytest.raises(tg.QuotaExhausted):
        tg.gemini_tag(papers_frame(50), SIGNALS, None, "m", cache, gap=0, call=flaky,
                      batch_size=25)
    assert len(tg.load_gemini_cache(cache)) == 25


def test_gemini_wrong_model_stops_immediately(tmp_path):
    n = {"calls": 0}

    def missing(client, model, prompt):
        n["calls"] += 1
        raise RuntimeError("404 NOT_FOUND. model is no longer available")

    with pytest.raises(RuntimeError, match="404"):
        tg.gemini_tag(papers_frame(50), SIGNALS, None, "m", tmp_path / "t.jsonl",
                      gap=0, call=missing)
    assert n["calls"] == 1


class FakeAPIError(Exception):
    def __init__(self, code, msg):
        super().__init__(f"{code} {msg}")
        self.code = code


@pytest.mark.parametrize("exc, kind", [
    (FakeAPIError(429, "RESOURCE_EXHAUSTED. Please retry in 10h23m40.440284047s."), "quota"),
    (FakeAPIError(404, "NOT_FOUND. model is no longer available"), "fatal"),
    (FakeAPIError(503, "UNAVAILABLE. high demand"), "retry"),
    (FakeAPIError(403, "PERMISSION_DENIED. The caller does not have permission"), "retry"),
    (RuntimeError("connection reset after 4040 ms"), "retry"),
])
def test_error_kind_uses_status_not_digits(exc, kind):
    assert tg.error_kind(exc) == kind


def test_daily_quota_stops_at_once(tmp_path):
    n = {"calls": 0}

    def daily(client, model, prompt):
        n["calls"] += 1
        raise FakeAPIError(429, "RESOURCE_EXHAUSTED. Generate content requests per day "
                                "limit exceeded. retry in 10h23m40.440284047s")

    with pytest.raises(tg.QuotaExhausted):
        tg.gemini_tag(papers_frame(30), SIGNALS, None, "m", tmp_path / "t.jsonl",
                      gap=0, call=daily)
    assert n["calls"] == 1


def test_tag_papers_falls_back_to_keywords():
    papers = pd.DataFrame({"arxiv_id": ["a", "b"], "published": ["2025-01-06"] * 2,
                           "title": ["Bitcoin", "Bitcoin"], "abstract": ["", ""]})
    cache = {"a": {"area": AREAS[2], "signal": "", "confidence": 0.7, "tagger": "gemini"}}
    out = tg.tag_papers(papers, SIGNALS, cache)
    assert list(out.columns) == tg.TAG_COLUMNS
    assert out["tagger"].tolist() == ["gemini", "keyword"]
    assert out["area"].tolist() == [AREAS[2], "Crypto and DeFi"]


def test_committed_gemini_tags_survive_a_keyless_run(tmp_path):
    path = tmp_path / "arxiv_tagged.csv"
    pd.DataFrame([
        {"arxiv_id": "a", "published": "2025-01-06", "title": "x", "area": AREAS[2],
         "signal": "Momentum", "confidence": "0.8", "tagger": "gemini"},
        {"arxiv_id": "b", "published": "2025-01-06", "title": "y", "area": "",
         "signal": "", "confidence": "", "tagger": "gemini"},
        {"arxiv_id": "c", "published": "2025-01-06", "title": "z", "area": AREAS[0],
         "signal": "", "confidence": "0.3", "tagger": "keyword"},
    ]).to_csv(path, index=False)
    tags = tg.committed_gemini_tags(path)
    assert set(tags) == {"a", "b"}  # keyword rows are recomputed, not reused
    assert tags["a"] == {"area": AREAS[2], "signal": "Momentum", "confidence": 0.8,
                         "tagger": "gemini"}
    assert tags["b"]["area"] == "" and tags["b"]["confidence"] == 0.0
    assert tg.committed_gemini_tags(tmp_path / "missing.csv") == {}
