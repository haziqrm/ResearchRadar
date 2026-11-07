import json

import numpy as np
import pandas as pd
import pytest

from src.common import contracts
from src.common.contracts import AREAS
from src.live import crowding as cr
from src.live import tagging as tg

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
    assert [len(c) for c in calls] == [25, 25, 10]
    assert len(tg.load_gemini_cache(cache)) == 60

    calls.clear()                                   # rerun: everything cached
    tg.gemini_tag(papers, SIGNALS, None, "m", cache, gap=0, call=fake_call)
    assert calls == []


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
        tg.gemini_tag(papers_frame(50), SIGNALS, None, "m", cache, gap=0, call=flaky)
    assert len(tg.load_gemini_cache(cache)) == 25


def test_tag_papers_falls_back_to_keywords():
    papers = pd.DataFrame({"arxiv_id": ["a", "b"], "published": ["2025-01-06"] * 2,
                           "title": ["Bitcoin", "Bitcoin"], "abstract": ["", ""]})
    cache = {"a": {"area": AREAS[2], "signal": "", "confidence": 0.7, "tagger": "gemini"}}
    out = tg.tag_papers(papers, SIGNALS, cache)
    assert list(out.columns) == tg.TAG_COLUMNS
    assert out["tagger"].tolist() == ["gemini", "keyword"]
    assert out["area"].tolist() == [AREAS[2], "Crypto and DeFi"]


# --- crowding --------------------------------------------------------------
def tagged_frame(weeks, per_week, surge_area=None, surge=0, start="2025-01-06"):
    """Same mix of areas every week; optionally extra papers in the last week."""
    rows = []
    mondays = pd.date_range(start, periods=weeks, freq="7D")
    for i, monday in enumerate(mondays):
        for area in AREAS:
            k = per_week + (surge if area == surge_area and i == weeks - 1 else 0)
            rows += [{"published": monday.strftime("%Y-%m-%d"), "area": area}] * k
    rows.append({"published": mondays[0].strftime("%Y-%m-%d"), "area": ""})  # untagged
    return pd.DataFrame(rows)


def test_crowding_matches_contract_and_flat_is_zero():
    out = cr.crowding_scores(cr.weekly_counts(tagged_frame(40, 5)))
    contracts.validate("crowding", out)
    assert out["date"].nunique() == 40 - cr.WINDOW_WEEKS - cr.MIN_BASELINE_WEEKS + 1
    assert np.allclose(out["crowding_score"], 0)
    assert (out["paper_count"] == 5).all()


def test_crowding_flags_a_surge():
    out = cr.crowding_scores(cr.weekly_counts(tagged_frame(40, 5, "Crypto and DeFi", 15)))
    last = out[out["date"] == out["date"].max()].set_index("area")["crowding_score"]
    assert last["Crypto and DeFi"] > 2
    assert (last.drop("Crypto and DeFi") < 0).all()


def test_overall_growth_alone_is_not_crowding():
    small = tagged_frame(20, 2)
    big = tagged_frame(20, 6, start="2025-05-26")      # 20 weeks after small starts
    out = cr.crowding_scores(cr.weekly_counts(pd.concat([small, big])))
    assert np.allclose(out["crowding_score"], 0)


def test_weeks_with_no_papers_are_filled():
    df = pd.DataFrame({"published": ["2025-01-06", "2025-01-27"],
                       "area": [AREAS[0], AREAS[0]]})
    counts = cr.weekly_counts(df)
    assert len(counts) == 4 and counts[AREAS[0]].tolist() == [1, 0, 0, 1]
