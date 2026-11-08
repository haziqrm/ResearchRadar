"""The Research Radar site: data functions and HTTP endpoints."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest

import html_main
import real_data as rd


# ---------------------------------------------------------------- data functions

@pytest.mark.parametrize("fn", ["get_topic_network", "get_research_snapshot", "get_decay",
                                "get_backtest", "get_briefing"])
def test_every_panel_works_on_committed_files(fn):
    out = getattr(rd, fn)()
    json.dumps(out)                                  # must be JSON-safe (no NaN objects)
    assert "sample" in out


def write_network(tmp_path, nodes, edges):
    data = tmp_path / "data"
    (data / "sample").mkdir(parents=True)
    pd.DataFrame(nodes, columns=["paper_id", "title", "year", "area", "citations",
                                 "lifecycle_stage"]).to_csv(data / "network_nodes.csv", index=False)
    pd.DataFrame(edges, columns=["source_id", "target_id"]).to_csv(
        data / "network_edges.csv", index=False)
    return data


def test_network_rolls_papers_up_to_areas(tmp_path, monkeypatch):
    nodes = ([(f"a{i}", "t", 2010 + i % 10, "A", 0, "peaking") for i in range(40)]
             + [(f"b{i}", "t", 2015 + i % 5, "B", 0, "emerging") for i in range(10)]
             + [(f"c{i}", "t", 2019, "C", 0, "saturated") for i in range(10)])
    edges = ([("a1", "b1")] * 6 + [("b2", "a3")] * 2      # A-B in both directions: one link
             + [("a1", "a2")] * 50                         # same area: dropped
             + [("b1", "c1")] * 3 + [("x", "a1")])         # unknown paper: dropped
    data = write_network(tmp_path, nodes, edges)
    monkeypatch.setattr(rd, "DATA_DIR", data)
    monkeypatch.setattr(rd, "SAMPLE_DIR", data / "sample")

    net = rd.get_topic_network()
    assert net["sample"] is False and net["directed"] is False
    assert net["periods"] == {"last": "2010-2014", "this": "2015-2019"}
    by = {n["id"]: n for n in net["nodes"]}
    assert by["A"]["total"] == 40 and by["A"]["papers"] == 20 and by["A"]["papers_last_week"] == 20
    assert by["B"]["papers"] == 10 and by["B"]["papers_last_week"] == 0
    assert by["B"]["stage"] == "emerging"
    pairs = {(l["source"], l["target"]): l for l in net["links"]}
    assert set(pairs) == {("A", "B"), ("B", "C")}
    assert pairs[("A", "B")]["shared"] == 8
    assert max(l["strength"] for l in net["links"]) == 1.0


def test_network_drops_a_year_still_in_progress():
    years = pd.Series([2024] * 100 + [2025] * 100 + [2026] * 20)
    assert rd._periods(years) == ((2016, 2020), (2021, 2025))


def test_network_falls_back_to_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "DATA_DIR", tmp_path)          # no real files here
    net = rd.get_topic_network()
    assert net["sample"] is True and net["nodes"]


def test_crowding_snapshot_sorted_with_weekly_change(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sample").mkdir(parents=True)
    pd.DataFrame([["2026-10-26", "Crypto and DeFi", 5, 0.1],
                  ["2026-10-26", "LLMs in finance", 2, 0.0],
                  ["2026-11-02", "Crypto and DeFi", 3, -0.5],
                  ["2026-11-02", "LLMs in finance", 6, 2.4]],
                 columns=["date", "area", "paper_count", "crowding_score"]).to_csv(
        data / "crowding.csv", index=False)
    monkeypatch.setattr(rd, "DATA_DIR", data)
    snap = rd.get_research_snapshot()
    assert snap["week_start"] == "2026-11-02"
    assert [t["topic"] for t in snap["topics"]] == ["LLMs in finance", "Crypto and DeFi"]
    llm = snap["topics"][0]
    assert (llm["papers"], llm["papers_change"], llm["papers_4w"]) == (6, 4, 8)


def test_decay_headline_reads_osap_summary():
    d = rd.get_decay()
    if d["headline"] is None:
        pytest.skip("osap_summary.csv not present")
    h = d["headline"]
    assert 0 < h["after_publication"] < h["after_sample"] <= 100


def test_backtest_chart_leaves_out_the_market():
    bt = rd.get_backtest()
    assert "market" not in {s["key"] for s in bt["series"]}


# ---------------------------------------------------------------- HTTP

@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), html_main.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_page_and_every_endpoint_respond(server):
    status, ctype, body = get(server + "/")
    assert status == 200 and b"Research Radar" in body and b"Stock Tracker" not in body
    for path in html_main.ROUTES:
        status, ctype, body = get(server + path)
        assert status == 200, (path, body[:200])
        json.loads(body)
    for css in html_main.CSS_FILES:
        assert get(server + f"/static/css/{css}")[0] == 200


def test_static_files_cannot_escape_the_folder(server):
    assert get(server + "/static/../html_main.py")[0] == 404
    assert get(server + "/static/%2e%2e/html_main.py")[0] == 404


def test_audio_served_when_present(server):
    status, ctype, _ = get(server + "/media/briefing.mp3")
    if html_main.AUDIO_FILE.exists():
        assert status == 200 and ctype == "audio/mpeg"
    else:
        assert status == 404
