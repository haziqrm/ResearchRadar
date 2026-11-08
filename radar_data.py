"""Topic radar and signal-decay data for the website (Person A), standard library only.

Reads the JSON that `python -m src.topic_radar.app_export` writes to data/app/:

  get_topic(name)   one topic's papers (2-D positions, year, title, sub-topic) and sub-topics
  get_topics()      the list of topic names with their size and forecast rank
  get_forecasts()   live topic and sub-topic rankings, and every test result (H9-H13)
  get_signals()     decay curve, haircut distribution, signal forecasts, H10 portfolios

If the export has not been run, each returns {"error": ...} so the page can say so.
"""
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "data" / "app"
_cache = {}


def _load(name):
    path = APP_DIR / name
    if not path.exists():
        return None
    stamp = path.stat().st_mtime
    if name not in _cache or _cache[name][0] != stamp:  # reload after a new export
        _cache[name] = (stamp, json.loads(path.read_text(encoding="utf-8")))
    return _cache[name][1]


def _missing(name):
    return {"error": f"data/app/{name} not found; run: python -m src.topic_radar.app_export"}


def get_topics():
    data = _load("topic_papers.json")
    if data is None:
        return _missing("topic_papers.json")
    rows = [{"name": t["name"], "n": t["n"], "rank": t["rank"], "stage": t["stage"],
             "predicted_boom": t["predicted_boom"]} for t in data["topics"].values()]
    return {"topics": sorted(rows, key=lambda r: (r["rank"] is None, r["rank"] or 0))}


def get_topic(name):
    data = _load("topic_papers.json")
    if data is None:
        return _missing("topic_papers.json")
    topic = data["topics"].get(name)
    if topic is None:
        return {"error": f"unknown topic: {name}"}
    return {**topic, "prior": data["prior"], "recent": data["recent"], "note": data["note"]}


def get_forecasts():
    return _load("forecasts.json") or _missing("forecasts.json")


def get_signals():
    return _load("signals.json") or _missing("signals.json")
