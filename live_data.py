"""Live layer data for the website (Person D), standard library only.

Reads D's committed outputs in data/, so the page runs offline from a fresh clone:

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
