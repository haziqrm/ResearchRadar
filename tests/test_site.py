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
                                "get_backtest", "get_briefing", "get_summary"])
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


def test_monthly_map_gives_each_area_a_share_per_month(tmp_path, monkeypatch):
    from src.live.map_tags import MAP_AREAS
    areas = list(MAP_AREAS)
    nodes = [(f"{i}-{j}", "t", 2020, a, 0, "peaking") for i, a in enumerate(areas) for j in range(3)]
    nodes += [("x0", "t", 2020, "Option pricing", 0, "peaking")]       # not chosen: dropped
    edges = [("0-0", "1-0")] * 3 + [("0-1", "x0")] * 9
    data = write_network(tmp_path, nodes, edges)
    a = areas[0]
    pd.DataFrame([("1", "2026-09-10", "s1", a, 0.9), ("2", "2026-09-20", "off", "", ""),
                  ("3", "2026-10-02", "o1", a, 0.9), ("4", "2026-10-09", "o2", a, 0.9),
                  ("5", "2026-10-30", "o3", a, 0.9), ("6", "2026-10-05", "off", "", ""),
                  ("7", "2026-11-02", "n1", a, 0.9)],                  # Nov in progress: dropped
                 columns=["arxiv_id", "published", "title", "area", "confidence"]
                 ).to_csv(data / rd.MAP_TAGS_NAME, index=False)
    monkeypatch.setattr(rd, "DATA_DIR", data)
    monkeypatch.setattr(rd, "SAMPLE_DIR", data / "sample")

    net = rd.get_topic_network()
    assert net["mode"] == "monthly" and [n["id"] for n in net["nodes"]] == areas
    assert [m["key"] for m in net["months"]] == ["2026-09", "2026-10"]
    sep, oct_ = net["nodes"][0]["series"]
    assert (sep["papers"], sep["share"], sep["share3"]) == (1, 50.0, 50.0)
    assert (oct_["papers"], oct_["share"], oct_["share3"], oct_["newest"]) == (3, 75.0, 66.7, "o3")
    assert net["nodes"][1]["series"][1] == {"papers": 0, "share": 0.0, "share3": 0.0, "newest": None}
    assert net["arxiv_total"] == 6 and net["papers_tagged"] == 4
    assert [(l["source"], l["target"]) for l in net["links"]] == [tuple(sorted(areas[:2]))]


def write_crowding(data, rows):
    pd.DataFrame(rows, columns=["date", "area", "paper_count", "crowding_score"]).to_csv(
        data / "crowding.csv", index=False)


def test_network_nodes_carry_live_area_and_score(tmp_path, monkeypatch):
    nodes = [(f"o{i}", "t", 2020, "Option pricing", 0, "peaking") for i in range(5)] + \
            [(f"e{i}", "t", 2020, "Oil and energy prices", 0, "peaking") for i in range(5)]
    data = write_network(tmp_path, nodes, [("o1", "e1")])
    (data / "area_map.csv").write_text("b_area,d_area\nOption pricing,Derivatives and volatility\n"
                                       "Oil and energy prices,Not tracked\n", encoding="utf-8")
    write_crowding(data, [["2026-10-26", "Derivatives and volatility", 4, 0.5],
                          ["2026-11-02", "Derivatives and volatility", 5, 1.25]])
    monkeypatch.setattr(rd, "DATA_DIR", data)
    monkeypatch.setattr(rd, "SAMPLE_DIR", data / "sample")
    net = rd.get_topic_network()
    by = {n["id"]: n for n in net["nodes"]}
    assert net["area_map"] is True and net["crowding_week"] == "2026-11-02"
    assert (by["Option pricing"]["d_area"], by["Option pricing"]["crowding_score"]) == \
        ("Derivatives and volatility", 1.25)                       # latest week, not older
    assert (by["Oil and energy prices"]["d_area"], by["Oil and energy prices"]["crowding_score"]) == \
        ("Not tracked", None)


def test_unmapped_map_area_fails_loudly(tmp_path, monkeypatch):
    data = write_network(tmp_path, [("x", "t", 2020, "Brand new B area", 0, "emerging")], [])
    (data / "area_map.csv").write_text("b_area,d_area\nOption pricing,Derivatives and volatility\n",
                                       encoding="utf-8")
    monkeypatch.setattr(rd, "DATA_DIR", data)
    with pytest.raises(ValueError, match="Brand new B area"):
        rd.get_topic_network()


def test_area_map_rejects_unknown_live_area(tmp_path, monkeypatch):
    (tmp_path / "area_map.csv").write_text("b_area,d_area\nIPOs,Factor research\n", encoding="utf-8")
    monkeypatch.setattr(rd, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown live areas"):
        rd.load_area_map(["IPOs"])


def test_no_area_map_means_no_colouring(tmp_path, monkeypatch):
    data = write_network(tmp_path, [("x", "t", 2020, "Option pricing", 0, "peaking")], [])
    monkeypatch.setattr(rd, "DATA_DIR", data)
    net = rd.get_topic_network()
    assert net["area_map"] is False
    assert net["nodes"][0]["d_area"] is None and net["nodes"][0]["crowding_score"] is None


def test_committed_area_map_covers_every_b_area():
    path = rd.DATA_DIR / "area_map.csv"
    nodes = rd.DATA_DIR / "network_nodes.csv"
    if not path.exists():
        pytest.skip("area_map.csv not present")
    m = pd.read_csv(path)
    assert set(m["d_area"]) <= set(rd.D_AREAS) | {rd.NOT_TRACKED}
    assert not m["b_area"].duplicated().any()
    if nodes.exists():                       # B's real network, when it's on this branch
        assert rd.load_area_map(pd.read_csv(nodes)["area"].dropna().unique()) is not None


def test_synthetic_network_has_the_same_shape():
    import synthetic_data
    real, fake = rd.get_topic_network(), synthetic_data.get_topic_network()
    assert {"area_map", "crowding_week"} <= set(fake)
    assert set(real["nodes"][0]) >= set(fake["nodes"][0]) >= {"id", "papers", "papers_last_week",
                                                             "d_area", "crowding_score"}


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


def test_partial_week_compares_the_same_days(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "sample").mkdir(parents=True)
    write_crowding(data, [["2026-10-26", "Crypto and DeFi", 9, 0.0],
                          ["2026-11-02", "Crypto and DeFi", 2, 0.5]])
    # this week: Mon-Wed so far. Last week: 1 paper Mon-Wed, 8 more on Thu-Sun.
    rows = ([("1", "2026-11-02"), ("2", "2026-11-04")] + [("3", "2026-10-27")]
            + [(str(i), "2026-10-30") for i in range(10, 18)])
    pd.DataFrame([(i, d, "t", "Crypto and DeFi", "", 0.9, "gemini") for i, d in rows],
                 columns=["arxiv_id", "published", "title", "area", "signal", "confidence",
                          "tagger"]).to_csv(data / "arxiv_tagged.csv", index=False)
    monkeypatch.setattr(rd, "DATA_DIR", data)
    snap = rd.get_research_snapshot()
    row = snap["topics"][0]
    assert (row["papers"], row["papers_last_week"], row["papers_change"]) == (2, 1, 1)
    assert snap["papers_compare"] == "Mon–Wed vs same days last week"


def test_summary_answers_the_brief_on_committed_files():
    s = rd.get_summary()
    assert set(s) == {"hidden", "inputs", "predict", "decide", "sample"}
    if s["hidden"]:
        assert s["hidden"]["after_publication"] < s["hidden"]["after_sample"]
    if s["decide"] and s["decide"]["naive"]:
        lo, hi = s["decide"]["naive"]["vs_equal_ci"]
        assert hi < 0                     # the "reliably worse" claim on the page must hold


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
