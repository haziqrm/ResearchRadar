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
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
AUDIO_FILE = DATA_DIR / "briefing.mp3"

# B's map has 25 descriptive areas; D's live feed has the 8 shared AREAS (25 would
# leave ~1.4 arXiv papers per area per week, too sparse for a crowding score).
# data/area_map.csv (b_area, d_area) rolls each map area up into one live area.
AREA_MAP_NAME = "area_map.csv"
NOT_TRACKED = "Not tracked"
D_AREAS = [
    "Factor and anomaly research", "Machine learning for returns", "Market microstructure",
    "Portfolio and risk", "Derivatives and volatility", "Alternative data and text",
    "Crypto and DeFi", "LLMs in finance",
]

# When data/arxiv_map_tags.csv exists (src.live.map_tags), the map shows only the 8
# chosen B areas, month by month, sized by each one's share of arXiv q-fin papers.
MAP_TAGS_NAME = "arxiv_map_tags.csv"

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


def load_area_map(map_areas) -> dict[str, str] | None:
    """{map area: live area or NOT_TRACKED}, or None when area_map.csv is missing.

    A map area that already is one of the 8 live areas (as in the sample network)
    maps to itself. Raises ValueError if any map area is left unmapped or points
    at an unknown live area, so a renamed B area can't silently lose its colour."""
    path = DATA_DIR / AREA_MAP_NAME
    if not path.exists():
        return None
    m = pd.read_csv(path)
    if list(m.columns) != ["b_area", "d_area"]:
        raise ValueError(f"{AREA_MAP_NAME} columns {list(m.columns)} != ['b_area', 'd_area']")
    bad = sorted(set(m["d_area"]) - set(D_AREAS) - {NOT_TRACKED})
    if bad:
        raise ValueError(f"{AREA_MAP_NAME} maps to unknown live areas: {bad}")
    mapping = dict(zip(m["b_area"], m["d_area"]))
    mapping.update({a: a for a in D_AREAS if a not in mapping})
    missing = sorted(set(map_areas) - set(mapping))
    if missing:
        raise ValueError(f"{AREA_MAP_NAME} has no row for map areas: {missing}")
    return mapping


def latest_crowding() -> tuple[str | None, dict[str, float]]:
    """(week, {live area: crowding score}) for the newest week in crowding.csv."""
    crowding, _ = _read("crowding")
    week = crowding["date"].max()
    latest = crowding[crowding["date"] == week]
    return week, {r.area: _num(r.crowding_score, 2) for r in latest.itertuples()}


def _area_links(edges_df, area_of: dict, size: pd.Series, only=None) -> list[dict]:
    """Paper-pair links rolled up to area pairs, scaled by area size so big areas
    don't link to everything, then rescaled so the strongest shown link is 1.
    With `only`, links outside those areas are dropped before rescaling."""
    e = edges_df.assign(s=edges_df["source_id"].map(area_of), t=edges_df["target_id"].map(area_of))
    e = e.dropna(subset=["s", "t"])
    e = e[e["s"] != e["t"]]
    if only is not None:
        e = e[e["s"].isin(only) & e["t"].isin(only)]
    if not len(e):
        return []
    counts = e.apply(lambda r: tuple(sorted((r["s"], r["t"]))), axis=1).value_counts()
    raw = {p: c / math.sqrt(size[p[0]] * size[p[1]]) for p, c in counts.items()}
    top = max(raw.values())
    strength = {p: v / top for p, v in raw.items()}
    keep = set()
    for a in (only if only is not None else size.index):
        mine = sorted((p for p in strength if a in p), key=strength.get, reverse=True)
        keep.update(mine[:LINKS_PER_NODE])
    keep.update(p for p, v in strength.items() if v >= STRONG_LINK)
    return [{"source": p[0], "target": p[1], "strength": round(strength[p], 3),
             "shared": int(counts[p])} for p in sorted(keep, key=strength.get, reverse=True)]


def _monthly_network(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict:
    """The 8 chosen map areas, month by month: each area's share of that month's arXiv
    q-fin papers. Shares, not counts, because q-fin itself grew (about 130 papers a
    month in early 2025, 240 by late 2026). `share3` is the 3-month rolling share,
    which the map sizes by, so areas with a few papers a month don't flicker.
    arXiv gives no reference lists, so the links come from B's OpenAlex papers: how
    much the two areas' literatures cite the same work."""
    tags = pd.read_csv(DATA_DIR / MAP_TAGS_NAME, dtype={"arxiv_id": str},
                       keep_default_na=False, parse_dates=["published"])
    from src.live.map_tags import MAP_AREAS
    areas = list(MAP_AREAS)
    missing = sorted(set(areas) - set(nodes_df["area"]))
    if missing:
        raise ValueError(f"map areas not in network_nodes.csv: {missing}")
    tags["month"] = tags["published"].dt.to_period("M")
    total = tags.groupby("month").size().sort_index()
    if len(total) > 1 and total.iloc[-1] < 0.5 * total.iloc[-2]:
        total = total.iloc[:-1]                         # a month still in progress
    months = total.index
    counts = (tags[tags["area"].isin(areas)].groupby(["month", "area"]).size()
              .unstack(fill_value=0).reindex(index=months, columns=areas, fill_value=0))
    share = counts.div(total, axis=0) * 100
    share3 = counts.rolling(3, min_periods=1).sum().div(total.rolling(3, min_periods=1).sum(), axis=0) * 100

    stage = nodes_df.groupby("area")["lifecycle_stage"].agg(lambda s: s.value_counts().idxmax())         if "lifecycle_stage" in nodes_df else pd.Series(dtype=str)
    size = nodes_df["area"].value_counts()
    nodes = []
    for a in areas:
        mine = tags[(tags["area"] == a) & tags["month"].isin(months)].sort_values("published")
        newest = mine.groupby("month")["title"].last()
        series = [{"papers": int(counts.at[m, a]), "share": _num(share.at[m, a], 1),
                   "share3": _num(share3.at[m, a], 1), "newest": newest.get(m)}
                  for m in months]
        nodes.append({"id": a, "series": series, "total": int(size[a]),
                      "papers": series[-1]["papers"], "papers_last_week": series[0]["papers"],
                      # first and last month, so the summary can say how attention moved
                      "share": series[-1]["share3"], "share_before": series[0]["share3"],
                      "stage": str(stage.get(a, "")),
                      "d_area": None, "crowding_score": None})
    area_of = dict(zip(nodes_df["paper_id"], nodes_df["area"]))
    return {"nodes": nodes, "links": _area_links(edges_df, area_of, size, only=areas),
            "directed": False, "sample": False, "mode": "monthly",
            "months": [{"key": str(m), "label": m.strftime("%b %Y"), "papers": int(total[m])}
                       for m in months],
            "periods": {"last": months[0].strftime("%b %Y"), "this": months[-1].strftime("%b %Y")},
            "papers_total": int(len(nodes_df)), "links_total": int(len(edges_df)),
            "arxiv_total": int(total.sum()),
            "papers_tagged": int(counts.values.sum()),
            "area_map": False, "crowding_week": None}


def get_topic_network():
    """One node per area, sized by papers in each period; links = papers in two areas
    that share references, scaled by area size so big areas don't link to everything.
    Each node also carries its live area (d_area) and that area's latest crowding score."""
    nodes_df, sample = _read("network_nodes")
    edges_df, _ = _read("network_edges")
    nodes_df = nodes_df.dropna(subset=["area"])
    if (DATA_DIR / MAP_TAGS_NAME).exists() and not sample:
        return _monthly_network(nodes_df, edges_df)
    (l0, l1), (t0, t1) = _periods(nodes_df["year"])

    area_of = dict(zip(nodes_df["paper_id"], nodes_df["area"]))
    size = nodes_df["area"].value_counts()
    in_period = lambda a, b: nodes_df[nodes_df["year"].between(a, b)]["area"].value_counts()
    now, before = in_period(t0, t1), in_period(l0, l1)
    stage = (nodes_df.groupby("area")["lifecycle_stage"]
             .agg(lambda s: s.value_counts().idxmax()) if "lifecycle_stage" in nodes_df
             else pd.Series(dtype=str))

    area_map = load_area_map(size.index)
    week, scores = latest_crowding() if area_map else (None, {})
    # B samples the same number of papers per year, so sizes are shares of research,
    # not volume. Report the shares so the page can say how each area moved.
    share = lambda counts, a: _num(counts.get(a, 0) / counts.sum() * 100, 1) if counts.sum() else None
    nodes = []
    for a in size.index:
        d_area = area_map.get(a) if area_map else None
        nodes.append({"id": a, "papers": int(now.get(a, 0)),
                      "papers_last_week": int(before.get(a, 0)),
                      "share": share(now, a), "share_before": share(before, a),
                      "total": int(size[a]), "stage": str(stage.get(a, "")),
                      "d_area": d_area,
                      "crowding_score": scores.get(d_area) if d_area != NOT_TRACKED else None})

    links = _area_links(edges_df, area_of, size)

    return {"nodes": nodes, "links": links, "directed": False, "sample": sample,
            "papers_total": int(len(nodes_df)), "links_total": int(len(edges_df)),
            "periods": {"last": f"{l0}-{l1}", "this": f"{t0}-{t1}"},
            "area_map": area_map is not None, "crowding_week": week}


# ---------------------------------------------------------------- D: crowding

def get_research_snapshot():
    """This week's crowding score per area, most crowded first.

    The newest week is usually still in progress, so 'papers' compares like with
    like: this week's days so far against the same weekdays last week (from
    arxiv_tagged.csv). Without that file it falls back to whole weeks."""
    crowding, sample = _read("crowding")
    weeks = sorted(crowding["date"].unique())
    week, prev = weeks[-1], (weeks[-2] if len(weeks) > 1 else None)
    last4 = crowding[crowding["date"].isin(weeks[-4:])].groupby("area")["paper_count"].sum()
    this_n = crowding[crowding["date"] == week].set_index("area")["paper_count"]
    before = (crowding[crowding["date"] == prev].set_index("area")["paper_count"]
              if prev else pd.Series(dtype=int))
    days_label = "this week vs last week"

    newest = {}
    tagged_path = DATA_DIR / "arxiv_tagged.csv"
    if tagged_path.exists():
        tagged = pd.read_csv(tagged_path, dtype=str, keep_default_na=False)
        tagged = tagged[tagged["area"] != ""]
        for area, g in tagged.groupby("area"):
            r = g.sort_values("published").iloc[-1]
            newest[area] = {"title": r["title"], "arxiv_id": r["arxiv_id"],
                            "published": r["published"]}
        # plain dates and ISO strings: pandas 2.2 date arithmetic warns on numpy 2.5
        pub = tagged["published"].str[:10]
        start, end = date.fromisoformat(str(week)[:10]), date.fromisoformat(pub.max())
        if start <= end < start + timedelta(days=6):            # week still in progress
            in_range = lambda a, b: tagged[(pub >= a.isoformat()) & (pub <= b.isoformat())][
                "area"].value_counts()
            this_n = in_range(start, end)
            before = in_range(start - timedelta(days=7), end - timedelta(days=7))
            days_label = (f"{start:%a}–{end:%a} vs same days last week"
                          if end > start else f"{start:%a} vs last {start:%a}")

    rows = []
    for r in crowding[crowding["date"] == week].itertuples():
        now, last = int(this_n.get(r.area, 0)), int(before.get(r.area, 0))
        rows.append({"topic": r.area, "score": _num(r.crowding_score, 2),
                     "papers": now, "papers_last_week": last, "papers_change": now - last,
                     "papers_4w": int(last4.get(r.area, 0)),
                     "newest": newest.get(r.area)})
    rows.sort(key=lambda r: r["score"] if r["score"] is not None else -99, reverse=True)
    return {"week_start": week, "topics": rows, "sample": sample, "papers_compare": days_label}


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


# ---------------------------------------------------------------- summary strip

def _summary_row(name: str, test: str, stat_prefix: str = ""):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path)
    hit = s[(s["test"] == test) & s["statistic"].astype(str).str.startswith(stat_prefix)]
    return hit.iloc[0] if len(hit) else None


def get_summary():
    """The brief's questions answered in numbers: what's hidden, how we found it,
    what predicts (and what doesn't), and the decision. Any missing piece is None."""
    out = {"hidden": None, "inputs": [], "predict": {}, "decide": None}

    decay = get_decay()["headline"]
    snap = get_research_snapshot()
    top = snap["topics"][0] if snap["topics"] else None
    if decay:
        out["hidden"] = {**decay, "crowded_area": top and top["topic"],
                         "crowded_score": top and top["score"], "week": snap["week_start"]}

    # inputs: market data + alternative data (the HRT "diverse inputs" pillar)
    factors_path = DATA_DIR / "factors.csv"
    if factors_path.exists():
        f = pd.read_csv(factors_path)
        out["inputs"].append({"what": "Factor returns", "n": int(f["signal"].nunique()),
                              "unit": "series", "detail": f"Ken French, monthly since {str(f['date'].min())[:4]}"})
    if decay:
        out["inputs"].append({"what": "Published signals", "n": decay["n_signals"], "unit": "signals",
                              "detail": "Open Source Asset Pricing, with paper dates"})
    tagged = DATA_DIR / "arxiv_tagged.csv"
    if tagged.exists():
        n = len(pd.read_csv(tagged, usecols=["arxiv_id"]))
        audit = DATA_DIR / "tag_audit.csv"
        acc = None
        if audit.exists():
            a = pd.read_csv(audit, dtype=str, keep_default_na=False)
            marked = a[a["correct_area"].str.lower().isin(["y", "n"])]
            if len(marked):
                acc = f"{(marked['correct_area'].str.lower() == 'y').sum()}/{len(marked)}"
        out["inputs"].append({"what": "arXiv q-fin papers", "n": n, "unit": "papers",
                              "detail": "2 years, tagged by Gemini" + (f", {acc} correct by hand" if acc else "")})
    net = get_topic_network()
    if not net["sample"]:
        out["inputs"].append({"what": "OpenAlex papers", "n": net["papers_total"], "unit": "papers",
                              "detail": f"2000-2026, {net['links_total']:,} shared-reference links"})

    # predictions that work (A's month-ahead tests) and ones that don't
    mom = _summary_row("osap_h9_summary", "H9a factor momentum")
    vol = _summary_row("osap_h9_summary", "H9b volatility")
    vm = _summary_row("osap_h9_summary", "H9b descriptive", "vol_managed")
    ew = _summary_row("osap_h9_summary", "H9b descriptive", "equal_weight")
    if mom is not None:
        out["predict"]["momentum"] = {"spread_pct": _num(mom["value"] * 100, 2),
                                      "t": _num(mom["t_stat"], 1), "months": int(mom["n_months"])}
    if vol is not None:
        out["predict"]["volatility"] = {"t": _num(vol["t_stat"], 1),
                                        "sharpe_managed": _num(vm["value"], 2) if vm is not None else None,
                                        "sharpe_plain": _num(ew["value"], 2) if ew is not None else None}
    model_path = DATA_DIR / "osap_model_summary.csv"
    if model_path.exists():
        m = pd.read_csv(model_path)
        models = m[~m["model"].str.startswith("baseline")]
        out["predict"]["decay_model"] = {
            "n_models": int(len(models)),
            "beat_baseline": int((~models["verdict"].str.contains("does not beat")).sum())}
    att = _summary_row("osap_summary", "H6 attention vs decay")
    if att is not None:
        out["predict"]["attention"] = {"rho": _num(att["value"], 2), "p": _num(att["p_value"], 2),
                                       "n": int(att["n"])}
    if not net["sample"]:
        movers = [n for n in net["nodes"] if n["share"] is not None and n["share_before"]]
        movers.sort(key=lambda n: n["share"] / n["share_before"], reverse=True)
        out["predict"]["attention_moved"] = [
            {"area": n["id"], "before": n["share_before"], "after": n["share"]} for n in movers[:2]]
        out["predict"]["periods"] = net["periods"]

    # the decision
    brief, bt = get_briefing(), get_backtest()
    rows = {r["key"]: r for r in bt["rows"]}
    if brief["card"]:
        out["decide"] = {"for_month": brief["card"]["allocation"]["for_month"],
                         "held": brief["card"]["allocation"]["held"],
                         "tilt": rows.get("decay_tilt"), "naive": rows.get("naive_insample")}
    out["sample"] = bool(snap["sample"] or net["sample"] or bt["sample"])
    return out


# ---------------------------------------------------------------- D: briefing

def get_briefing():
    card_path, text_path = DATA_DIR / "briefing_card.json", DATA_DIR / "briefing.txt"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else None
    text = (text_path.read_text(encoding="utf-8").strip() if text_path.exists()
            else (SAMPLE_DIR / "briefing.txt").read_text(encoding="utf-8").strip())
    return {"card": card, "text": text, "audio": AUDIO_FILE.exists(),
            "sample": card is None}
