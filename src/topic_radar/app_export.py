"""Write ready-to-draw JSON for the website (data/app/), so the site itself needs only the
standard library.

  data/app/topic_papers.json  per topic: every paper (2-D position from t-SNE on its
                              SPECTER2 vector, year, title, sub-topic, seed flag) and the
                              topic's sub-topics with counts and forecast rank
  data/app/forecasts.json     live topic and sub-topic rankings, plus every test result
                              (H9-H13) with noise-world values for the histograms
  data/app/signals.json       decay curve, haircut distribution, signal forecasts, H10
                              portfolios

Run after the radar (python -m src.topic_radar.run and .subtopic_forecast):
    python -m src.topic_radar.app_export
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.common import contracts
from src.common.contracts import DATA_DIR
from src.signal_decay.haircut import CAVEAT, share_distribution
from src.topic_radar.panel import LAST_TARGET_YEAR, group_view, load_corpus, refit_model
from src.topic_radar.subtopic_forecast import K_SUB, TOP_SUB

APP_DIR = DATA_DIR / "app"
MIN_IN_TOPIC = 8  # sub-topics with fewer papers inside a topic are shown as "other"
PRIOR = (2020, 2022)
RECENT = (2023, 2025)
SEED = 0


def _clean(value):
    """JSON-safe: NaN -> None, numpy -> python."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), 5)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write(name: str, obj) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    (APP_DIR / name).write_text(json.dumps(_clean(obj), separators=(",", ":")), encoding="utf-8")


def layout(unit: np.ndarray, seed: int = SEED) -> np.ndarray:
    """2-D t-SNE positions scaled into [0, 1] (aspect kept)."""
    n = len(unit)
    if n < 5:
        return np.column_stack([np.linspace(0.2, 0.8, n), np.full(n, 0.5)])
    xy = TSNE(n_components=2, perplexity=min(30.0, (n - 1) / 3), init="pca", learning_rate="auto",
              random_state=seed).fit_transform(np.asarray(unit, dtype=np.float64))
    xy = xy - xy.min(axis=0)
    return xy / max(xy.max(), 1e-9)


def island_positions(centroids: np.ndarray, radii: np.ndarray, gap: float | None = None,
                     iters: int = 2000) -> np.ndarray:
    """Centres for round islands: similar sub-topics close together, none overlapping.

    Classical MDS on the cosine distance between sub-topic centroids gives the rough map;
    then any two islands closer than their radii plus a gap are pushed apart.
    """
    m = len(radii)
    if m == 1:
        return np.zeros((1, 2))
    c = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    d2 = np.clip(1 - c @ c.T, 0, None) ** 2
    j = np.eye(m) - 1.0 / m
    vals, vecs = np.linalg.eigh(-0.5 * j @ d2 @ j)
    pos = vecs[:, -2:] * np.sqrt(np.clip(vals[-2:], 0, None))
    spread = np.median(np.linalg.norm(pos[:, None] - pos[None], axis=2)[np.triu_indices(m, 1)])
    target = pos * (3 * radii.mean() / max(spread, 1e-9))
    gap = 0.3 * radii.mean() if gap is None else gap
    rng = np.random.default_rng(SEED)
    pos = target + rng.normal(scale=1e-3 * radii.mean(), size=target.shape)  # split exact ties

    def collide(pos):
        moved = False
        for a in range(m):
            for b in range(a + 1, m):
                d = pos[b] - pos[a]
                dist = max(float(np.hypot(*d)), 1e-9)
                need = radii[a] + radii[b] + gap
                if dist < need:
                    push = (need - dist) / 2 * d / dist
                    pos[a] -= push
                    pos[b] += push
                    moved = True
        return moved

    for _ in range(300):  # pull towards the similarity map, then resolve overlaps
        pos += 0.1 * (target - pos)
        collide(pos)
    for _ in range(iters):  # finish with overlaps only, so none remain
        if not collide(pos):
            break
    return pos


def island_layout(unit: np.ndarray, subs: np.ndarray, seed: int = SEED) -> np.ndarray:
    """2-D positions with one round island per sub-topic, scaled into [0, 1] (aspect kept).

    Inside an island, papers are placed by t-SNE on their own vectors (so near = similar);
    island size grows with the square root of its paper count.
    """
    ids = sorted(set(subs.tolist()))
    radii = np.array([np.sqrt((subs == s).sum()) for s in ids], dtype=float)
    centroids = np.array([unit[subs == s].astype(np.float64).mean(axis=0) for s in ids])
    centres = island_positions(centroids, radii)
    xy = np.zeros((len(subs), 2))
    for s, r, centre in zip(ids, radii, centres):
        idx = np.flatnonzero(subs == s)
        if len(idx) < 5:
            ang = np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
            inner = np.column_stack([np.cos(ang), np.sin(ang)]) * (0.5 if len(idx) > 1 else 0.0)
        else:
            inner = layout(unit[idx], seed) - 0.5
            inner = inner - inner.mean(axis=0)
            inner = inner / max(np.linalg.norm(inner, axis=1).max(), 1e-9)
        xy[idx] = centre + inner * r
    xy = xy - xy.min(axis=0)
    return xy / max(xy.max(), 1e-9)


def topic_links(edges: np.ndarray, idx: np.ndarray) -> list:
    """Shared-reference links with both papers in the topic, as [i, j] positions in its paper list."""
    pos = np.full(int(max(edges.max(), idx.max())) + 1, -1)
    pos[idx] = np.arange(len(idx))
    a, b = pos[edges[:, 0]], pos[edges[:, 1]]
    keep = (a >= 0) & (b >= 0)
    return np.column_stack([a[keep], b[keep]]).tolist()


def topic_citations(cites: np.ndarray, idx: np.ndarray, group_of: np.ndarray, names: dict) -> tuple:
    """Direct citations for one topic.

    cites: (m, 2) corpus indices [citing, cited]. Returns the citations with both papers in
    the topic as [citing, cited] positions in its paper list, and the ones that leave the
    topic as [position, other topic, "out" (this paper cites it) | "in" (cited by it), count].
    """
    pos = np.full(len(group_of), -1)
    pos[idx] = np.arange(len(idx))
    a, b = pos[cites[:, 0]], pos[cites[:, 1]]
    inside = np.column_stack([a, b])[(a >= 0) & (b >= 0)].tolist()
    ext = {}
    for (ci, cd), pa, pb in zip(cites, a, b):
        if pa >= 0 and pb < 0:
            key = (int(pa), names[int(group_of[cd])], "out")
        elif pb >= 0 and pa < 0:
            key = (int(pb), names[int(group_of[ci])], "in")
        else:
            continue
        ext[key] = ext.get(key, 0) + 1
    return inside, [[p, t, d, n] for (p, t, d), n in sorted(ext.items())]


def load_citations(corpus) -> np.ndarray:
    """In-sample direct citations as corpus indices; empty if not fetched yet."""
    path = DATA_DIR / "paper_citations.csv"
    if not path.exists():
        return np.zeros((0, 2), dtype=int)
    c = pd.read_csv(path)
    index = {p: i for i, p in enumerate(corpus.paper_id)}
    pairs = [(index[x], index[y]) for x, y in zip(c["citing_id"], c["cited_id"]) if x in index and y in index]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def topic_papers(corpus, sub_labels: np.ndarray, units: pd.DataFrame, live: pd.DataFrame,
                 seeds: pd.DataFrame, stages: pd.Series) -> dict:
    cites = load_citations(corpus)
    unit_info = units.set_index("subtopic_id")
    seed_rank = dict(zip(seeds["paper_id"], seeds["rank"]))
    topics = {}
    for g, name in sorted(corpus.group_names.items()):
        idx = np.flatnonzero(corpus.frozen_group == g)
        if not len(idx):
            continue
        subs = sub_labels[idx].copy()
        counts = pd.Series(subs).value_counts()
        small = counts[counts < MIN_IN_TOPIC].index
        subs[np.isin(subs, small)] = -1
        xy = island_layout(corpus.unit[idx], subs)
        links = topic_links(corpus.edges, idx)
        inside, external = topic_citations(cites, idx, corpus.frozen_group, corpus.group_names)
        years = corpus.year[idx]
        sub_rows = []
        for s in sorted(set(subs.tolist()), key=lambda s: (s == -1, -int((subs == s).sum()))):
            mine = subs == s
            info = unit_info.loc[s] if s in unit_info.index else None
            sub_rows.append({
                "id": int(s),
                "label": "other (small sub-topics)" if s == -1 else info["label"],
                "n": int(mine.sum()),
                "prior": int((mine & (years >= PRIOR[0]) & (years <= PRIOR[1])).sum()),
                "recent": int((mine & (years >= RECENT[0]) & (years <= RECENT[1])).sum()),
                "rank": None if info is None else int(info["rank"]),
                "predicted_boom": None if info is None else bool(info["predicted_boom"]),
                "growth_score": None if info is None else float(info["growth_score"]),
            })
        row = live.get(name, {})
        topics[name] = {
            "name": name,
            "n": int(len(idx)),
            "stage": stages.get(name, ""),
            "rank": row.get("rank"),
            "predicted_boom": row.get("predicted_boom"),
            "growth_score": row.get("growth_score"),
            "subtopics": sub_rows,
            "links": links,
            "cites": inside,
            "external": external,
            "papers": [
                {"id": corpus.paper_id[i], "t": corpus.title[i], "y": int(corpus.year[i]),
                 "x": round(float(xy[j, 0]), 4), "v": round(float(xy[j, 1]), 4), "s": int(subs[j]),
                 "seed": seed_rank.get(corpus.paper_id[i])}
                for j, i in enumerate(idx)
            ],
        }
    return {"topics": topics, "prior": list(PRIOR), "recent": list(RECENT),
            "note": "One island per sub-topic, placed so similar sub-topics sit close; inside an island, "
                    "t-SNE on SPECTER2 vectors. Sub-topics: the 75 forecast units (H13); fewer than 8 "
                    "papers in a topic are grouped as 'other'. Links: shared references. Cites: direct citations inside the sample (OpenAlex reference lists; "
                    "about a third of papers have none, so influence is undercounted)."}


def _pass_rows(path, mode="refit"):
    ev = pd.read_csv(path)
    rows = ev[(ev["metric"] == "pass_test") & (ev.get("groups_mode", mode) == mode)]
    keep = [c for c in ["model", "value", "noise_world_mean", "noise_world_p95", "p_noise_worlds",
                        "holm_p", "p_one_sided"] if c in rows.columns]
    return rows[keep].to_dict("records")


def forecasts(live_topics: pd.DataFrame, units: pd.DataFrame) -> dict:
    tests = []
    for key, eval_csv, null_csv, label, question, pass_col in [
        ("H11", "topic_forecast_eval.csv", "topic_forecast_null.csv", "Topic growth forecast (25 topics)",
         "Can we rank which topics grow over the next 3 years?", "h11_pass"),
        ("H13", "subtopic_forecast_eval.csv", "subtopic_forecast_null.csv", "Sub-topic growth forecast (75)",
         "Can we rank which sub-topics grow over the next 3 years?", "h13_pass"),
    ]:
        ev = pd.read_csv(DATA_DIR / eval_csv)
        null_path = DATA_DIR / null_csv
        null = pd.read_csv(null_path)["spearman_ridge_minus_random"].tolist() if null_path.exists() else []
        real = ev[(ev["metric"] == "spearman") & (ev["model"] == "ridge") & (ev["groups_mode"] == "refit")]
        tests.append({"key": key, "label": label, "question": question,
                      "passed": bool(ev[pass_col].dropna().iloc[0]),
                      "real_spearman": float(real["value"].iloc[0]), "noise_spearman": null,
                      "tests": _pass_rows(DATA_DIR / eval_csv)})
    seed = pd.read_csv(DATA_DIR / "seed_eval.csv")
    tests.append({"key": "H12", "label": "Seed papers (reading list)",
                  "question": "Can we pick recent papers that later work clusters around?",
                  "passed": bool(seed["h12_pass"].iloc[0]),
                  "precision": seed[seed["metric"] == "precision_at_20"][["model", "value"]].to_dict("records"),
                  "tests": seed[seed["metric"] == "pass_test"][["model", "value", "holm_p"]].to_dict("records")})
    h10 = pd.read_csv(DATA_DIR / "osap_h10_summary.csv")
    h10 = h10[(h10["period"] == "2000-2025") & (h10["vs"] == "baseline_a") & h10["model"].isin(["ridge", "hgb"])]
    tests.append({"key": "H10", "label": "Next-month volatility model",
                  "question": "Does a trained model forecast each signal's volatility better than the 12-month rule?",
                  "passed": bool((h10["model"].eq("ridge") & (h10["p_holm"] < 0.05)).any()),
                  "tests": h10[["model", "t_stat", "p_holm", "verdict"]].to_dict("records")})
    h9 = pd.read_csv(DATA_DIR / "osap_h9_summary.csv") if (DATA_DIR / "osap_h9_summary.csv").exists() else None
    if h9 is not None:
        tests.append({"key": "H9", "label": "Momentum and volatility rules",
                      "question": "Do past winners keep winning, and does last year's volatility predict next month's?",
                      "passed": True, "table": h9.head(10).to_dict("records")})
    subs = units.sort_values("rank")
    return {
        "topics": live_topics.sort_values("rank").to_dict("records"),
        "subtopics": subs[["subtopic_id", "label", "parent_group", "papers_2020_22", "papers_2023_25",
                           "growth_score", "rank", "predicted_boom"]].to_dict("records"),
        "top_subtopics": int(TOP_SUB), "n_subtopics": int(K_SUB),
        "tests": tests,
        "status": "exploratory: the topic and sub-topic forecasts did not pass their pre-registered tests",
    }


def signals() -> dict:
    curve = pd.read_csv(DATA_DIR / "decay_curve_robust.csv")
    table = pd.read_csv(DATA_DIR / "signal_forecasts.csv")
    decay = contracts.load("decay_results")[["source", "signal", "share_kept", "paper"]]
    table = table.merge(decay, on=["source", "signal"], how="left")
    portfolios = pd.read_csv(DATA_DIR / "osap_h10_portfolio.csv")
    return {
        "curve": curve.to_dict("records"),
        "haircut": {**share_distribution(), "caveat": CAVEAT},
        "signals": table[["source", "signal", "name", "paper", "as_of", "mom_12m", "momentum_on",
                          "vol_forecast", "vol_baseline", "share_kept"]].to_dict("records"),
        "portfolios": portfolios[["portfolio", "cost_bps", "sharpe", "max_drawdown"]].to_dict("records"),
    }


def main() -> None:
    corpus = load_corpus()
    model = refit_model(corpus, LAST_TARGET_YEAR, k=K_SUB)  # same fit as the H13 live run (seed 0)
    sub_labels = group_view(model, corpus.vectors)["group"].to_numpy()
    units = pd.read_csv(DATA_DIR / "subtopic_forecasts.csv")
    area = pd.read_csv(DATA_DIR / "area_forecasts.csv")
    live = area[area["actual_boom"].isna()].sort_values("growth_score", ascending=False).copy()
    live["rank"] = np.arange(1, len(live) + 1)
    live_map = {r["area"]: {"rank": int(r["rank"]), "predicted_boom": bool(r["predicted_boom"]),
                            "growth_score": float(r["growth_score"])} for _, r in live.iterrows()}
    nodes = pd.read_csv(DATA_DIR / "network_nodes.csv")
    stages = nodes.groupby("area")["lifecycle_stage"].agg(lambda s: s.mode().iloc[0])
    seeds = pd.read_csv(DATA_DIR / "seed_papers.csv")
    _write("topic_papers.json", topic_papers(corpus, sub_labels, units, live_map, seeds, stages))
    _write("forecasts.json", forecasts(live.rename(columns={"area": "topic"}), units))
    _write("signals.json", signals())
    for f in sorted(APP_DIR.glob("*.json")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
