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
    for needle in [b'id="heading-panel"', b'id="help-drawer"', b'id="brief-modal"',
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
    scores = [t["score"] for t in h["topics"]]
    assert scores == sorted(scores, reverse=True)
    ranks = sorted(t["forecast"]["rank"] for t in h["topics"] if t["forecast"])
    assert ranks == list(range(1, len(ranks) + 1))          # re-ranked among the map's topics
    assert all(t["forecast"]["of"] == len(ranks) for t in h["topics"] if t["forecast"])
    assert h["forecast_passed"] is False                     # labelled exploratory on the page
    assert all(t["read_first"] and t["read_first"]["title"] for t in h["topics"])


def test_briefing_card_has_a_date_for_the_new_dot():
    assert live_data.get_briefing()["card"]["generated"]


def test_cut_parts_are_gone(server):
    _, _, page = get(server + "/")
    assert b'id="path-btn"' not in page and b'id="research"' not in page
