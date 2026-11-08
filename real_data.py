"""Real data for the Research Radar site: same shape of functions as synthetic_data.py.

Every function reads the team's committed output files and returns plain JSON-able
data. Nothing calls a live API, so the demo runs offline from a fresh clone.

  get_topic_network()      B: research map, papers rolled up to one node per area
  get_research_snapshot()  D: this week's crowding score per area (crowding.csv)
  get_decay()              A: do published signals stop working?
  get_backtest()           D: walk-forward allocation vs baselines, with intervals
  get_briefing()           D: the decision card and spoken briefing

If a real file is missing, the sample in data/sample/ is used and the result says
so ("sample": true), so the page can label it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
AUDIO_FILE = DATA_DIR / "briefing.mp3"

NETWORK_PERIOD_YEARS = 5    # network toggle compares the last two 5-year periods
LINKS_PER_NODE = 2          # keep each area's strongest links (plus any very strong)
STRONG_LINK = 0.6


def _read(name: str) -> tuple[pd.DataFrame, bool]:
    """data/<name>.csv, else the sample. Returns (frame, is_sample)."""
    real = DATA_DIR / f"{name}.csv"
    if real.exists():
        return pd.read_csv(real), False
    return pd.read_csv(SAMPLE_DIR / f"{name}.csv"), True


def _num(x, places=None):
    """JSON-safe number: NaN becomes None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), places) if places is not None else float(x)


# ---------------------------------------------------------------- B: network

def _periods(years: pd.Series) -> tuple[tuple[int, int], tuple[int, int]]:
    """The last two complete NETWORK_PERIOD_YEARS-year periods in the data.
    The newest year is dropped when it has under half the papers of the year
    before (a year still in progress)."""
    counts = years.value_counts().sort_index()
    end = int(counts.index.max())
    if end - 1 in counts.index and counts[end] < 0.5 * counts[end - 1]:
        end -= 1
    n = NETWORK_PERIOD_YEARS
    return (end - 2 * n + 1, end - n), (end - n + 1, end)


def get_topic_network():
    """One node per area, sized by papers in each period; links = papers in two areas
    that share references, scaled by area size so big areas don't link to everything."""
    nodes_df, sample = _read("network_nodes")
    edges_df, _ = _read("network_edges")
    nodes_df = nodes_df.dropna(subset=["area"])
    (l0, l1), (t0, t1) = _periods(nodes_df["year"])

    area_of = dict(zip(nodes_df["paper_id"], nodes_df["area"]))
    size = nodes_df["area"].value_counts()
    in_period = lambda a, b: nodes_df[nodes_df["year"].between(a, b)]["area"].value_counts()
    now, before = in_period(t0, t1), in_period(l0, l1)
    stage = (nodes_df.groupby("area")["lifecycle_stage"]
             .agg(lambda s: s.value_counts().idxmax()) if "lifecycle_stage" in nodes_df
             else pd.Series(dtype=str))

    nodes = [{"id": a, "papers": int(now.get(a, 0)), "papers_last_week": int(before.get(a, 0)),
              "total": int(size[a]), "stage": str(stage.get(a, ""))}
             for a in size.index]

    e = edges_df.assign(s=edges_df["source_id"].map(area_of), t=edges_df["target_id"].map(area_of))
    e = e.dropna(subset=["s", "t"])
    e = e[e["s"] != e["t"]]
    pair = e.apply(lambda r: tuple(sorted((r["s"], r["t"]))), axis=1) if len(e) else pd.Series([], dtype=object)
    counts = pair.value_counts()
    links = []
    if len(counts):
        raw = {p: c / math.sqrt(size[p[0]] * size[p[1]]) for p, c in counts.items()}
        top = max(raw.values())
        strength = {p: v / top for p, v in raw.items()}
        keep = set()
        for a in size.index:
            mine = sorted((p for p in strength if a in p), key=strength.get, reverse=True)
            keep.update(mine[:LINKS_PER_NODE])
        keep.update(p for p, v in strength.items() if v >= STRONG_LINK)
        links = [{"source": p[0], "target": p[1], "strength": round(strength[p], 3),
                  "shared": int(counts[p])} for p in sorted(keep, key=strength.get, reverse=True)]

    return {"nodes": nodes, "links": links, "directed": False, "sample": sample,
            "papers_total": int(len(nodes_df)), "links_total": int(len(edges_df)),
            "periods": {"last": f"{l0}-{l1}", "this": f"{t0}-{t1}"}}


# ---------------------------------------------------------------- D: crowding

def get_research_snapshot():
    """This week's crowding score per area, most crowded first."""
    crowding, sample = _read("crowding")
    weeks = sorted(crowding["date"].unique())
    week, prev = weeks[-1], (weeks[-2] if len(weeks) > 1 else None)
    last4 = crowding[crowding["date"].isin(weeks[-4:])].groupby("area")["paper_count"].sum()
    before = (crowding[crowding["date"] == prev].set_index("area")["paper_count"]
              if prev else pd.Series(dtype=int))

    newest = {}
    tagged_path = DATA_DIR / "arxiv_tagged.csv"
    if tagged_path.exists():
        tagged = pd.read_csv(tagged_path, dtype=str, keep_default_na=False)
        for area, g in tagged[tagged["area"] != ""].groupby("area"):
            r = g.sort_values("published").iloc[-1]
            newest[area] = {"title": r["title"], "arxiv_id": r["arxiv_id"],
                            "published": r["published"]}

    rows = []
    for r in crowding[crowding["date"] == week].itertuples():
        last = int(before.get(r.area, 0))
        rows.append({"topic": r.area, "score": _num(r.crowding_score, 2),
                     "papers": int(r.paper_count), "papers_last_week": last,
                     "papers_change": int(r.paper_count) - last,
                     "papers_4w": int(last4.get(r.area, 0)),
                     "newest": newest.get(r.area)})
    rows.sort(key=lambda r: r["score"] if r["score"] is not None else -99, reverse=True)
    return {"week_start": week, "topics": rows, "sample": sample}


# ---------------------------------------------------------------- A: decay

def get_decay():
    """Headline from 209 OSAP signals plus the 7 Ken French signals."""
    decay, sample = _read("decay_results")
    signals = [{"signal": r.signal, "published": str(r.publication_date)[:4],
                "pre": _num(r.pre_mean * 100, 2), "post": _num(r.post_mean * 100, 2),
                "kept": _num(r.post_mean / r.pre_mean * 100, 0) if r.pre_mean else None,
                "diff_low": _num(r.ci_low * 100, 2), "diff_high": _num(r.ci_high * 100, 2),
                "clear": bool(r.ci_high < 0 or r.ci_low > 0)}
               for r in decay.itertuples()]

    headline = None
    path = DATA_DIR / "osap_summary.csv"
    if path.exists():
        s = pd.read_csv(path)
        get = lambda test, stat: s[(s["test"] == test) & (s["statistic"].str.startswith(stat))].iloc[0]
        try:
            oos, post = get("H4 McLean-Pontiff", "b_oos"), get("H4 McLean-Pontiff", "b_post")
            extra = get("H4 McLean-Pontiff", "b_post - b_oos")
            placebo = s[s["test"] == "H5 pooled placebo"].iloc[0]
            headline = {
                "n_signals": int(post["n"]),
                "after_sample": _num((1 + oos["value"]) * 100, 0),
                "after_publication": _num((1 + post["value"]) * 100, 0),
                "after_publication_ci": [_num((1 + post["ci_low"]) * 100, 0),
                                         _num((1 + post["ci_high"]) * 100, 0)],
                "publication_p": _num(extra["p_value"], 3),
                "placebo_n": int(placebo["n"]),
                "placebo_p": _num(placebo["p_value"], 4),
            }
        except (IndexError, KeyError):
            headline = None
    return {"signals": signals, "headline": headline, "sample": sample}


# ---------------------------------------------------------------- D: backtest

BACKTEST_SHOWN = ["decay_tilt", "equal_weight", "naive_insample", "market"]
BACKTEST_LABELS = {"decay_tilt": "Decay tilt (ours)", "equal_weight": "Equal weight",
                   "naive_insample": "Trust the paper", "market": "Market"}


def get_backtest():
    """Sharpe ratios with 95% intervals (after costs) and cumulative returns."""
    results, sample = _read("backtest_results")
    summary_path = DATA_DIR / "backtest_summary.csv"
    rows = []
    if summary_path.exists():
        s = pd.read_csv(summary_path).set_index("strategy")
        for k in BACKTEST_SHOWN:
            if k in s.index:
                r = s.loc[k]
                rows.append({"key": k, "label": BACKTEST_LABELS[k],
                             "sharpe": _num(r["sharpe"], 2),
                             "ci": [_num(r["sharpe_ci_low"], 2), _num(r["sharpe_ci_high"], 2)],
                             "vs_equal": None if k in ("equal_weight", "market")
                             else _num(r["sharpe_minus_equal_weight"], 2),
                             "vs_equal_ci": None if k in ("equal_weight", "market")
                             else [_num(r["diff_ci_low"], 2), _num(r["diff_ci_high"], 2)]})
    series = []
    # The market (before costs) grows ~30x since 1981 and would flatten the
    # strategies to a line at zero, so the chart compares the factor strategies only.
    for k in ["decay_tilt", "equal_weight", "naive_insample"]:
        g = results[results["strategy"] == k]
        if len(g):
            # monthly points are plenty for a small chart; keep every 3rd month
            g = g.iloc[::3]
            series.append({"key": k, "label": BACKTEST_LABELS[k],
                           "points": [[d, _num(v, 4)] for d, v in
                                      zip(g["date"], g["cumulative_return"])]})
    return {"rows": rows, "series": series, "sample": sample,
            "start": str(results["date"].min()), "end": str(results["date"].max())}


# ---------------------------------------------------------------- D: briefing

def get_briefing():
    card_path, text_path = DATA_DIR / "briefing_card.json", DATA_DIR / "briefing.txt"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else None
    text = (text_path.read_text(encoding="utf-8").strip() if text_path.exists()
            else (SAMPLE_DIR / "briefing.txt").read_text(encoding="utf-8").strip())
    return {"card": card, "text": text, "audio": AUDIO_FILE.exists(),
            "sample": card is None}
