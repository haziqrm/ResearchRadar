"""Person D's panels on the website: live_data.py and its routes in html_main.py."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import html_main
import live_data


def test_crowding_latest_period_sorted_with_examples():
    c = live_data.get_crowding()
    assert "error" not in c
    scores = [t["score"] for t in c["topics"]]
    assert scores == sorted(scores, reverse=True)
    assert len(c["topics"]) == 8
    if c["unit"] == "year":                          # OpenAlex: most cited paper this year
        ex = [t["example"] for t in c["topics"] if t["example"]]
        assert ex and all(e["year"] == int(c["period_start"][:4]) for e in ex)


def test_backtest_rows_and_series():
    bt = live_data.get_backtest()
    assert [r["key"] for r in bt["rows"]] == live_data.BACKTEST_SHOWN
    assert {s["key"] for s in bt["series"]} == set(live_data.BACKTEST_CHART)
    assert bt["start"] < bt["end"]


def test_briefing_card_matches_crowding():
    b, c = live_data.get_briefing(), live_data.get_crowding()
    assert b["text"] and b["card"]
    assert b["card"]["unit"] == c["unit"]
    assert b["card"]["busiest"][0]["area"] == c["topics"][0]["topic"]


def test_missing_file_gives_error(tmp_path, monkeypatch):
    monkeypatch.setattr(live_data, "DATA_DIR", tmp_path)
    assert "error" in live_data.get_crowding()
    assert "error" in live_data.get_backtest()
    assert "error" in live_data.get_briefing()


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), html_main.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers["Content-Type"], r.read()


@pytest.mark.parametrize("path", ["/api/crowding", "/api/backtest", "/api/briefing", "/api/heading"])
def test_routes_return_json(server, path):
    status, ctype, body = get(server + path)
    assert status == 200 and ctype == "application/json"
    assert "error" not in json.loads(body)


def test_page_loads_live_panels_and_audio(server):
    _, _, page = get(server + "/")
    for needle in [b'id="heading-drawer"', b'id="heading-btn"', b'id="brief-modal"',
                   b"/static/js/live.js", b"/static/css/live.css"]:
        assert needle in page
    status, ctype, audio = get(server + "/media/briefing.mp3")
    assert status == 200 and ctype == "audio/mpeg" and len(audio) > 1000


def test_crowding_history_colours_the_map():
    h = live_data.get_crowding_history()
    assert h["unit"] == "year"
    assert set(h["scores"]) == {t["topic"] for t in live_data.get_crowding()["topics"]}
    assert h["scores"]["Deep learning and LLM forecasting"]["2026"] > 2


def test_page_title_and_no_year_watermark(server):
    _, _, page = get(server + "/")
    assert b"<title>Research Lifecycle Radar</title>" in page
    assert b"net-year" not in page and b"net-legend" in page


def test_heading_covers_the_map_topics_with_forecast_and_reading():
    h = live_data.get_heading()
    assert len(h["topics"]) == 8
    diffs = [t["change"]["diff"] for t in h["topics"]]
    assert diffs == sorted(diffs, reverse=True)              # biggest rise first
    assert h["forecast_passed"] is False                     # labelled exploratory on the page
    assert all(t["read_first"] and t["read_first"]["title"] for t in h["topics"])


def test_heading_change_compares_shares_of_each_years_papers():
    """Up/down on last year uses shares, so the partial latest year does not look like a fall."""
    h = live_data.get_heading()
    totals = live_data._papers_per_year()
    now, prev = h["period_start"][:4], h["prev_start"][:4]
    for t in h["topics"]:
        c = t["change"]
        assert c["unit"] == "share"
        assert c["now"] == round(100 * c["papers"] / totals[now], 1)
        assert c["prev"] == round(100 * c["papers_prev"] / totals[prev], 1)
        assert c["dir"] == ("up" if c["diff"] > live_data.FLAT_SHARE else
                            "down" if c["diff"] < -live_data.FLAT_SHARE else "flat")


def test_heading_forecast_ranks_match_the_drill_down():
    """One denominator for topic ranks: among the 8 map topics, in the table and the
    drill-down alike. "Predicted to grow" is the forecast's own top 5 of all 25."""
    import radar_data
    fc = radar_data.get_forecasts()
    by_topic = {t["topic"]: t for t in fc["topics"]}
    h = live_data.get_heading()["topics"]
    ranks = sorted(t["forecast"]["rank"] for t in h)
    assert ranks == list(range(1, len(h) + 1)) and fc["n_map"] == len(h) == 8
    for t in h:
        f, g = t["forecast"], by_topic[t["topic"]]
        assert f["rank"] == g["map_rank"] and f["of"] == fc["n_map"]
        assert f["rank_all"] == g["rank"] and f["of_all"] == fc["n_topics"] == len(fc["topics"])
        assert f["likely"] == (f["rank_all"] <= f["top"] == fc["top_topics"])
    # same order as the overall forecast
    assert sorted(h, key=lambda t: t["forecast"]["rank"]) == sorted(h, key=lambda t: t["forecast"]["rank_all"])


def test_read_first_is_cited_and_recent():
    h = live_data.get_heading()
    lo, hi = h["read_first_years"]
    assert hi - lo + 1 == live_data.READ_FIRST_YEARS
    for t in h["topics"]:
        r = t["read_first"]
        assert lo <= r["year"] <= hi
        assert r["per_year"] == round(r["citations"] / (hi - r["year"] + 1), 1)
    # it beats this year's most cited paper on citations per year for every topic
    this_year = {t["topic"]: t["example"] for t in live_data.get_crowding()["topics"]}
    assert all(t["read_first"]["per_year"] >= (this_year[t["topic"]] or {}).get("citations", 0)
               for t in h["topics"])


def test_readable_names_for_subtopics_and_signals():
    import radar_data
    fc = radar_data.get_forecasts()
    assert all(s["name"] and s["name"] != s["label"] for s in fc["subtopics"])   # all 75 named
    topic = radar_data.get_topic("Information and stock reactions")
    assert all(s["name"] != s["label"] for s in topic["subtopics"] if s["id"] >= 0)  # -1: "other"
    sig = {s["code"]: s["name"] for s in radar_data.get_signals()["signals"]}
    assert sig["MomOffSeason11YrPlus"] == "Off season reversal years 11 to 15"   # OSAP's description
    assert sig["Mom"] == "Momentum"


def test_briefing_card_has_a_date_for_the_new_dot():
    assert live_data.get_briefing()["card"]["generated"]


def test_cut_parts_are_gone(server):
    _, _, page = get(server + "/")
    assert b'id="research"' not in page
    # the affected path (Person C) is back, on by default, in place of crowding colours
    assert b'id="path-btn"' in page and b"lg-ramp" not in page
