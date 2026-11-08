"""Topic radar and signal-decay data for the website (Person A), standard library only.

Reads the JSON that `python -m src.topic_radar.app_export` writes to data/app/:

  get_topic(name)   one topic's papers (2-D positions, year, title, sub-topic) and sub-topics
  get_topics()      the list of topic names with their size and forecast rank
  get_forecasts()   live topic and sub-topic rankings, and every test result (H9-H13)
  get_signals()     decay curve, haircut distribution, signal forecasts, H10 portfolios

If the export has not been run, each returns {"error": ...} so the page can say so.

Readable names are added on the way out, so the export doesn't need rerunning:
  data/subtopic_names.csv   a name per sub-topic, written by reading each sub-topic's
                            papers (the export's `label` is a keyword list; it is kept)
  data/signal_names.csv     OSAP's own description of each signal code (from its
                            signal documentation), e.g. MomOffSeason11YrPlus ->
                            "Off season reversal years 11 to 15"
"""
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
APP_DIR = DATA_DIR / "app"
_cache = {}


def _names(file, key):
    path = DATA_DIR / file
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r[key]: r["name"] for r in csv.DictReader(f)}


def _sub_name(names, sid, label):
    return names.get(str(sid)) or label


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


def _citations():
    """Citation count per paper (B's network_nodes.csv), for the suggested-reading popup."""
    path = DATA_DIR / "network_nodes.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["paper_id"]: int(r["citations"] or 0) for r in csv.DictReader(f)}


def get_topic(name):
    data = _load("topic_papers.json")
    if data is None:
        return _missing("topic_papers.json")
    topic = data["topics"].get(name)
    if topic is None:
        return {"error": f"unknown topic: {name}"}
    names = _names("subtopic_names.csv", "subtopic_id")
    subs = [{**s, "name": _sub_name(names, s["id"], s["label"])} for s in topic["subtopics"]]
    cites = _citations()
    papers = [{**p, "c": cites.get(p["id"])} for p in topic["papers"]]  # c: citations
    return {**topic, "subtopics": subs, "papers": papers, "prior": data["prior"], "recent": data["recent"],
            "note": data["note"]}


def get_forecasts():
    fc = _load("forecasts.json")
    if fc is None:
        return _missing("forecasts.json")
    names = _names("subtopic_names.csv", "subtopic_id")
    subs = [{**s, "name": _sub_name(names, s["subtopic_id"], s["label"])} for s in fc["subtopics"]]
    # The site ranks topics among the ones on the map (map_rank of n_map). `rank` stays
    # the rank among all forecast topics, which "predicted to grow" (top_topics) uses.
    on_map = map_rank(fc["topics"])
    topics = [{**t, "map_rank": on_map.get(t["topic"])} for t in fc["topics"]]
    return {**fc, "topics": topics, "subtopics": subs, "n_topics": len(fc["topics"]),
            "n_map": len(on_map), "top_topics": sum(1 for t in fc["topics"] if t["predicted_boom"])}


def map_rank(topics):
    """{topic: rank among the map's topics}, ordered by the forecast's own rank."""
    from research_data import AREAS
    ranked = sorted((t for t in topics if t["topic"] in AREAS), key=lambda t: t["rank"])
    return {t["topic"]: i + 1 for i, t in enumerate(ranked)}


def get_signals():
    sig = _load("signals.json")
    if sig is None:
        return _missing("signals.json")
    names = _names("signal_names.csv", "signal")
    rows = [{**s, "code": s["signal"], "name": names.get(s["signal"], s["name"])} for s in sig["signals"]]
    return {**sig, "signals": rows}
