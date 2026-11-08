"""Live layer data for the website (Person D), standard library only.

Reads D's committed outputs in data/, so the page runs offline from a fresh clone:

  get_heading()    "Where research is heading": change on last year, forecast and a paper per
                   map topic (the paper: most cited per year over the last five years)
  get_crowding_history()  crowding score per area and year (colours the map)
  get_crowding()   crowding score per research-map area for the latest period
                   (crowding.csv + crowding_meta.json; a year for OpenAlex, a week for arXiv)
  get_backtest()   walk-forward allocation vs baselines, with 95% intervals
  get_briefing()   the decision card, the briefing text, and whether there is audio

The audio itself is served by html_main.py at /media/briefing.mp3 (AUDIO_FILE).
If a file is missing, each function returns {"error": ...} so the page can say so.
"""
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
AUDIO_FILE = DATA_DIR / "briefing.mp3"

BACKTEST_SHOWN = ["decay_tilt", "equal_weight", "naive_insample", "market"]
BACKTEST_LABELS = {"decay_tilt": "Decay tilt (ours)", "equal_weight": "Equal weight",
                   "naive_insample": "Trust the paper", "market": "Market"}
BACKTEST_CHART = ["decay_tilt", "equal_weight", "naive_insample"]  # market would flatten them
CHART_EVERY = 3        # monthly points are plenty for a small chart; keep every 3rd


def _rows(name):
    path = DATA_DIR / name
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _missing(name, how):
    return {"error": f"data/{name} not found; run: {how}"}


def _num(x, places=2):
    try:
        return round(float(x), places)
    except (TypeError, ValueError):
        return None


def _meta():
    path = DATA_DIR / "crowding_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"source": "arxiv", "unit": "week", "window": 4, "baseline": 52}


def _example_papers(meta, year):
    """One paper per area: most cited this year (OpenAlex) or newest (arXiv)."""
    out = {}
    if meta["source"] == "openalex":
        for r in _rows("network_nodes.csv") or []:
            if r["year"] != str(year):
                continue
            best = out.get(r["area"])
            if best is None or int(r["citations"] or 0) > best["citations"]:
                out[r["area"]] = {"title": " ".join(r["title"].split()), "year": int(r["year"]),
                                  "citations": int(r["citations"] or 0), "id": r["paper_id"]}
    else:
        for r in _rows("arxiv_map_tags.csv") or []:
            best = out.get(r["area"])
            if best is None or r["published"] > best["published"]:
                out[r["area"]] = {"title": " ".join(r["title"].split()),
                                  "published": r["published"], "id": r["arxiv_id"]}
    return out


def get_crowding():
    rows = _rows("crowding.csv")
    if rows is None:
        return _missing("crowding.csv", "python -m src.live.crowding")
    meta = _meta()
    periods = sorted({r["date"] for r in rows})
    latest, prev = periods[-1], (periods[-2] if len(periods) > 1 else None)
    before = {r["area"]: int(r["paper_count"]) for r in rows if r["date"] == prev}
    examples = _example_papers(meta, latest[:4])
    topics = []
    for r in rows:
        if r["date"] != latest:
            continue
        now, last = int(r["paper_count"]), before.get(r["area"], 0)
        topics.append({"topic": r["area"], "score": _num(r["crowding_score"]),
                       "papers": now, "papers_prev": last,
                       "example": examples.get(r["area"])})
    topics.sort(key=lambda t: t["score"] if t["score"] is not None else -99, reverse=True)
    return {"source": meta["source"], "unit": meta["unit"], "window": meta["window"],
            "baseline": meta.get("baseline"), "period_start": latest, "prev_start": prev,
            "topics": topics}


def _app_json(name):
    path = DATA_DIR / "app" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


READ_FIRST_YEARS = 5   # "Read first" looks at papers from the last five years


def _read_first(latest_year, years=READ_FIRST_YEARS):
    """One paper per area: the most cited per year since publication among papers from
    the last `years` years (OpenAlex sample). Citations per year stops a paper from this
    year (few citations yet) or from long ago (decades to collect them) from winning on
    age alone."""
    out = {}
    for r in _rows("network_nodes.csv") or []:
        year = int(r["year"])
        if not latest_year - years < year <= latest_year:
            continue
        cites = int(r["citations"] or 0)
        per_year = cites / (latest_year - year + 1)
        best = out.get(r["area"])
        if best is None or per_year > best["per_year"]:
            out[r["area"]] = {"title": " ".join(r["title"].split()), "year": year,
                              "citations": cites, "per_year": round(per_year, 1), "id": r["paper_id"]}
    return out


FLAT_SHARE = 0.5   # a change under half a percentage point of the year's papers counts as level


def _papers_per_year():
    """All sampled papers per year (OpenAlex), every area: the denominator for shares."""
    out = {}
    for r in _rows("network_nodes.csv") or []:
        out[r["year"]] = out.get(r["year"], 0) + 1
    return out


def _change(now, prev, total_now, total_prev):
    """Up or down on last year. With yearly totals (OpenAlex) this compares the share of
    each year's papers, so a partial year does not look like a fall; without them (arXiv
    weeks, equal length) it compares counts."""
    if total_now and total_prev:
        a, b = round(100 * now / total_now, 1), round(100 * prev / total_prev, 1)
        unit = "share"
    else:
        a, b, unit = now, prev, "papers"
    d = round(a - b, 1)
    flat = FLAT_SHARE if unit == "share" else 0
    return {"unit": unit, "now": a, "prev": b, "diff": d,
            "dir": "up" if d > flat else "down" if d < -flat else "flat",
            "papers": now, "papers_prev": prev}


def get_heading():
    """"Where research is heading": the map's topics with their change on last year
    (_change), Person A's growth forecast ranked among the map's topics
    (radar_data.map_rank, the same rank the drill-down shows) and a paper to read first
    (_read_first). Sorted by the change, biggest rise first."""
    from radar_data import map_rank
    crowd = get_crowding()
    if "error" in crowd:
        return crowd
    fc = _app_json("forecasts.json") or {}
    growth = {t["topic"]: t for t in fc.get("topics", [])}
    top = sum(1 for t in growth.values() if t["predicted_boom"])
    on_map = map_rank(list(growth.values()))
    passed = next((t["passed"] for t in fc.get("tests", []) if t.get("key") == "H11"), None)
    latest = int(crowd["period_start"][:4])
    reading = _read_first(latest) if crowd["source"] == "openalex" else {}
    totals = _papers_per_year() if crowd["unit"] == "year" else {}
    prev_label = (crowd.get("prev_start") or "")[:4] if crowd["unit"] == "year" else crowd.get("prev_start")
    rows = []
    for t in crowd["topics"]:
        g = growth.get(t["topic"])
        rows.append({
            "topic": t["topic"], "score": t["score"], "papers": t["papers"],
            "change": _change(t["papers"], t["papers_prev"], totals.get(str(latest)),
                              totals.get(prev_label)),
            "forecast": None if g is None or t["topic"] not in on_map else {
                "rank": on_map[t["topic"]], "of": len(on_map),
                "rank_all": g["rank"], "of_all": len(growth), "top": top,
                "likely": bool(g["predicted_boom"])},
            "read_first": reading.get(t["topic"]) or t["example"],
        })
    rows.sort(key=lambda r: r["change"]["diff"], reverse=True)
    return {"period_start": crowd["period_start"], "prev_start": crowd.get("prev_start"),
            "unit": crowd["unit"], "forecast_passed": passed, "read_first_years": [latest - READ_FIRST_YEARS + 1, latest],
            "topics": rows}


def get_crowding_history():
    """Crowding score per area and year, for colouring the map: {"scores": {area: {year: s}}}.
    Only yearly (OpenAlex) scores line up with the map's year slider; weekly arXiv
    scores return no scores, so the map falls back to grey."""
    rows = _rows("crowding.csv")
    if rows is None:
        return _missing("crowding.csv", "python -m src.live.crowding")
    meta = _meta()
    scores = {}
    if meta["unit"] == "year":
        for r in rows:
            scores.setdefault(r["area"], {})[r["date"][:4]] = _num(r["crowding_score"])
    return {"source": meta["source"], "unit": meta["unit"], "scores": scores}


def get_backtest():
    results, summary = _rows("backtest_results.csv"), _rows("backtest_summary.csv")
    if results is None or summary is None:
        return _missing("backtest_results.csv", "python -m src.live.backtest")
    by_key = {r["strategy"]: r for r in summary}
    rows = []
    for k in BACKTEST_SHOWN:
        r = by_key.get(k)
        if r is None:
            continue
        vs = k not in ("equal_weight", "market")
        rows.append({"key": k, "label": BACKTEST_LABELS[k], "sharpe": _num(r["sharpe"]),
                     "ci": [_num(r["sharpe_ci_low"]), _num(r["sharpe_ci_high"])],
                     "vs_equal": _num(r["sharpe_minus_equal_weight"]) if vs else None,
                     "vs_equal_ci": [_num(r["diff_ci_low"]), _num(r["diff_ci_high"])]
                     if vs else None})
    series = []
    for k in BACKTEST_CHART:
        pts = [[r["date"], _num(r["cumulative_return"], 4)]
               for r in results if r["strategy"] == k]
        if pts:
            series.append({"key": k, "label": BACKTEST_LABELS[k], "points": pts[::CHART_EVERY]})
    dates = [r["date"] for r in results]
    return {"rows": rows, "series": series, "start": min(dates), "end": max(dates)}


def get_briefing():
    card_path, text_path = DATA_DIR / "briefing_card.json", DATA_DIR / "briefing.txt"
    if not text_path.exists():
        return _missing("briefing.txt", "python -m src.live.briefing")
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else None
    return {"card": card, "text": text_path.read_text(encoding="utf-8").strip(),
            "audio": AUDIO_FILE.exists()}
