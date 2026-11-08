"""H11: rank B's research groups by how much their share of new papers will grow.

Research growth at origin t: G = share(t+1..t+3) / share(t-2..t) - 1, counts smoothed by
+0.5. Boom = top 5 groups by G. Features use papers with year <= t only, and the primary
test refits the groups at every origin on papers <= t. B's frozen groups (fitted on all
years) are reported as a secondary, optimistic row. Pass rules are in HYPOTHESES.md.

Run: python -m src.topic_radar.forecast
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import contracts
from src.topic_radar import FORECAST_EVAL_CSV
from src.topic_radar.panel import (
    LAST_TARGET_YEAR, N_GROUPS, Corpus, frozen_model, group_view, load_corpus, refit_model,
)
from src.topic_radar.stats import (
    GAP, MIN_TRAIN_ORIGINS, holm, logit_predict, ridge_predict, spearman, summarise, top_k,
    zscore_within,
)

ORIGINS = list(range(2005, LAST_TARGET_YEAR - GAP + 1))  # 2005..2022
LIVE_ORIGIN = LAST_TARGET_YEAR
TOP = 5
SMOOTH = 0.5
FEATURES = ["past_growth", "acceleration", "share", "volatility", "bridging", "fuzziness", "spread"]
BASE_RATE = TOP / N_GROUPS
SPLIT_SEED = 0


def feature_half(year: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """True for a random half of each year's papers (features); False for the other (target)."""
    rng = np.random.default_rng(seed)
    half = np.zeros(len(year), dtype=bool)
    for y in np.unique(year):
        idx = np.flatnonzero(year == y)
        half[rng.permutation(idx)[: len(idx) // 2]] = True
    return half


def _counts(year: np.ndarray, group: np.ndarray, k: int) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame({"year": year, "group": group})
    counts = pd.crosstab(frame["year"], frame["group"]).reindex(columns=range(k), fill_value=0)
    return counts, frame.groupby("year").size()


def _share(counts: pd.DataFrame, totals: pd.Series, years) -> pd.Series:
    years = [y for y in years if y in counts.index]
    return (counts.loc[years].sum() + SMOOTH) / max(totals.loc[years].sum(), 1)


def group_features(year: np.ndarray, group: np.ndarray, top_prob: np.ndarray, dist: np.ndarray,
                   edges: np.ndarray, t: int, k: int = N_GROUPS,
                   half: np.ndarray | None = None) -> pd.DataFrame:
    """One row per group at origin t; `growth` (the target) is NaN when t + 3 is not observed.

    Split-half design (HYPOTHESES.md, amendment): features use only papers with half=True;
    the target's current share uses papers with half=False, and its future share uses all
    papers from t+1..t+3. Sampling noise in the features is then independent of the target.
    half=None uses every paper for both (no split).
    """
    a = np.ones(len(year), dtype=bool) if half is None else np.asarray(half, bool)
    b = a if half is None else ~a
    counts, totals = _counts(year[a], group[a], k)
    now = _share(counts, totals, range(t - 2, t + 1))
    out = pd.DataFrame({"group": range(k)})
    out["share"] = now.to_numpy()
    out["past_growth"] = (now / _share(counts, totals, range(t - 5, t - 2)) - 1).to_numpy()
    out["acceleration"] = (_share(counts, totals, [t]) / _share(counts, totals, [t - 1]) - 1).to_numpy()
    yearly = pd.concat([_share(counts, totals, [y]) for y in range(t - 4, t + 1)], axis=1)
    out["volatility"] = (yearly.std(axis=1) / yearly.mean(axis=1)).to_numpy()

    past = (year <= t) & a
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    e = edges[past[edges[:, 0]] & past[edges[:, 1]]]
    ga, gb = group[e[:, 0]], group[e[:, 1]]
    touch = np.bincount(ga, minlength=k) + np.bincount(gb, minlength=k)
    cross = np.bincount(ga[ga != gb], minlength=k) + np.bincount(gb[ga != gb], minlength=k)
    out["bridging"] = np.where(touch > 0, cross / np.maximum(touch, 1), np.nan)

    cur = (year == t) & a
    stats_now = pd.DataFrame({"group": group[cur], "p": top_prob[cur], "d": dist[cur]}).groupby("group")
    out["fuzziness"] = out["group"].map(stats_now["p"].mean())
    out["spread"] = out["group"].map(stats_now["d"].mean())

    if t + GAP <= LAST_TARGET_YEAR:
        b_counts, b_totals = _counts(year[b], group[b], k)
        base = _share(b_counts, b_totals, range(t - 2, t + 1))
        all_counts, all_totals = _counts(year, group, k)
        future = _share(all_counts, all_totals, range(t + 1, t + GAP + 1))
        out["growth"] = (future / base - 1).to_numpy()
    else:
        out["growth"] = np.nan
    out["origin"] = t
    return out


def label_targets(panel: pd.DataFrame, top: int = TOP) -> pd.DataFrame:
    """target = log(1+G); fit_target = target demeaned within origin; boom = top `top` by G."""
    panel = panel.copy()
    panel["target"] = np.log1p(panel["growth"])
    panel["fit_target"] = panel["target"] - panel.groupby("origin")["target"].transform("mean")
    panel["boom"] = np.where(panel["target"].isna(), np.nan, 0.0)
    for t, g in panel[panel["target"].notna()].groupby("origin"):
        panel.loc[g.index[top_k(g["growth"].to_numpy(), top)], "boom"] = 1.0
    return panel


def build_panel(corpus: Corpus, mode: str, origins=ORIGINS, models: dict | None = None,
                k: int = N_GROUPS, top: int = TOP, frozen=None) -> tuple[pd.DataFrame, dict]:
    """Panel of unit features at each origin; also the per-paper labels used.

    mode "refit": models[t] (or a fresh refit with k units) per origin; mode "frozen": one
    model for every origin (`frozen`, default B's frozen groups).
    """
    if mode == "frozen":
        fixed = group_view(frozen if frozen is not None else frozen_model(), corpus.vectors)
    half = feature_half(corpus.year)
    rows, labels = [], {}
    for t in origins:
        if mode == "frozen":
            view = fixed
        else:
            model = models[t] if models and t in models else refit_model(corpus, t, k=k)
            view = group_view(model, corpus.vectors)
        labels[t] = view["group"].to_numpy()
        rows.append(group_features(corpus.year, labels[t], view["top_prob"].to_numpy(),
                                   view["dist"].to_numpy(), corpus.edges, t, k=k, half=half))
    return finish_panel(pd.concat(rows, ignore_index=True), top=top), labels


def finish_panel(panel: pd.DataFrame, top: int = TOP) -> pd.DataFrame:
    """Keep raw past growth and share for the baselines, z-score features, add targets."""
    panel = panel.assign(raw_past_growth=panel["past_growth"], raw_share=panel["share"])
    return label_targets(zscore_within(panel, FEATURES), top=top)


def walk_forward(panel: pd.DataFrame, with_logit: bool = True, base_rate: float = BASE_RATE) -> pd.DataFrame:
    """Out-of-sample ridge and logistic scores for every origin with enough training origins."""
    out = []
    for t in sorted(panel["origin"].unique()):
        train = panel[(panel["origin"] <= t - GAP) & panel["target"].notna()]
        if train["origin"].nunique() < MIN_TRAIN_ORIGINS:
            continue
        test = panel[panel["origin"] == t].copy()
        test["growth_score"], test["alpha"] = ridge_predict(train, test, FEATURES)
        if with_logit:
            test["boom_probability"], test["C"] = logit_predict(train, test, FEATURES)
        else:
            test["boom_probability"] = base_rate
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


SCORERS = {
    "ridge": lambda g: g["growth_score"],
    "momentum": lambda g: g["raw_past_growth"],
    "mean_reversion": lambda g: -g["raw_past_growth"],
    "size": lambda g: g["raw_share"],
}


def per_origin_metrics(pred: pd.DataFrame, top: int = TOP, base_rate: float = BASE_RATE) -> pd.DataFrame:
    """Per origin: Spearman and hits in the top `top` (column p{top}_<scorer>), Brier."""
    rows = []
    for t, g in pred[pred["target"].notna()].groupby("origin"):
        actual = set(top_k(g["growth"].to_numpy(), top))
        row = {"origin": t}
        for name, fn in SCORERS.items():
            s = fn(g).to_numpy()
            row[f"spearman_{name}"] = spearman(s, g["growth"])
            row[f"p{top}_{name}"] = len(actual & set(top_k(s, top)))
        y = g["boom"].astype(float).to_numpy()
        row["brier_logit"] = float(np.mean((g["boom_probability"].to_numpy() - y) ** 2))
        row["brier_base_rate"] = float(np.mean((base_rate - y) ** 2))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("origin")


def pass_stats(top: int = TOP, k: int = N_GROUPS) -> dict:
    random_hits = top * top / k  # expected hits in the top `top` when picking at random
    return {
        "spearman_ridge_minus_random": lambda m: m["spearman_ridge"],
        f"p{top}_ridge_minus_random": lambda m: m[f"p{top}_ridge"] - random_hits,
        "spearman_ridge_minus_momentum": lambda m: m["spearman_ridge"] - m["spearman_momentum"],
    }


PASS_STATS = pass_stats()
NULL_WORLDS = 200
NULL_CSV = FORECAST_EVAL_CSV.with_name("topic_forecast_null.csv")  # one row per noise world


def null_worlds(year: np.ndarray, half: np.ndarray, top_prob: np.ndarray, dist: np.ndarray,
                edges: np.ndarray, shares: np.ndarray, n_worlds: int = NULL_WORLDS,
                seed: int = 1, origins=ORIGINS, top: int = TOP) -> pd.DataFrame:
    """Mean pass statistics on worlds where units never truly grow (only sampling noise).

    Each paper's unit is drawn at random with constant shares; everything else is kept.
    """
    rng = np.random.default_rng(seed)
    k = len(shares)
    stats_fns = pass_stats(top, k)
    rows = []
    for _ in range(n_worlds):
        labels = rng.choice(k, size=len(year), p=shares / shares.sum())
        panel = finish_panel(pd.concat(
            [group_features(year, labels, top_prob, dist, edges, t, k=k, half=half)
             for t in origins], ignore_index=True), top=top)
        m = per_origin_metrics(walk_forward(panel, with_logit=False, base_rate=top / k), top=top,
                               base_rate=top / k)
        rows.append({name: float(fn(m).mean()) for name, fn in stats_fns.items()})
    return pd.DataFrame(rows)


def evaluate(metrics: pd.DataFrame, mode: str, null: pd.DataFrame | None = None,
             top: int = TOP, k: int = N_GROUPS, pass_column: str = "h11_pass") -> pd.DataFrame:
    """Summary table. With `null` (noise worlds), the pass tests use the noise-world p-values
    (HYPOTHESES.md amendment); otherwise the block-bootstrap p-values."""
    rows = []

    def add(metric, model, x, **extra):
        rows.append({"metric": metric, "model": model, "groups_mode": mode, "n_origins": len(x),
                     **summarise(np.asarray(x, float)), **extra})

    for name in SCORERS:
        add("spearman", name, metrics[f"spearman_{name}"])
        add(f"precision_at_{top}", name, metrics[f"p{top}_{name}"])
    add("spearman", "random", np.zeros(len(metrics)))
    add(f"precision_at_{top}", "random", np.full(len(metrics), top * top / k))
    add("brier", "logistic", metrics["brier_logit"])
    add("brier", "base_rate", metrics["brier_base_rate"])
    start = len(rows)
    for name, fn in pass_stats(top, k).items():
        add("pass_test", name, fn(metrics))
        if null is not None:
            real = rows[-1]["value"]
            rows[-1]["p_noise_worlds"] = float((1 + (null[name] >= real).sum()) / (len(null) + 1))
            rows[-1]["noise_world_mean"] = float(null[name].mean())
            rows[-1]["noise_world_p95"] = float(null[name].quantile(0.95))
    key = "p_noise_worlds" if null is not None else "p_one_sided"
    for i, p in zip(range(start, len(rows)), holm([r[key] for r in rows[start:]])):
        rows[i]["holm_p"] = p
    add("context", "spearman_ridge_minus_mean_reversion",
        metrics["spearman_ridge"] - metrics["spearman_mean_reversion"])
    out = pd.DataFrame(rows)
    tests = out[out["metric"] == "pass_test"]
    passed = bool((tests["value"] > 0).all() and (tests["holm_p"] < 0.05).all())
    out[pass_column] = passed if mode == "refit" else np.nan
    return out


def live_forecast(train_panel: pd.DataFrame, corpus: Corpus, frozen=None, k: int = N_GROUPS,
                  top: int = TOP) -> pd.DataFrame:
    """Forecast from origin 2025: models trained on origins <= 2022, frozen units at 2025."""
    live, labels = build_panel(corpus, "frozen", origins=[LIVE_ORIGIN], k=k, top=top, frozen=frozen)
    train = train_panel[(train_panel["origin"] <= LIVE_ORIGIN - GAP) & train_panel["target"].notna()]
    live["growth_score"], _ = ridge_predict(train, live, FEATURES)
    live["boom_probability"], _ = logit_predict(train, live, FEATURES)
    live.attrs["labels"] = labels[LIVE_ORIGIN]
    return live


def to_contract(backtest: pd.DataFrame, live: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    rows = []
    for df, is_live in [(backtest, False), (live, True)]:
        for t, g in df.groupby("origin"):
            chosen = set(g.index[top_k(g["growth_score"].to_numpy(), TOP)])
            for i, r in g.iterrows():
                rows.append({"area": names[int(r["group"])], "year": int(t),
                             "growth_score": round(float(r["growth_score"]), 4),
                             "boom_probability": round(float(r["boom_probability"]), 4),
                             "predicted_boom": int(i in chosen),
                             "actual_boom": np.nan if is_live else int(bool(r["boom"]))})
    out = pd.DataFrame(rows)
    out["actual_boom"] = out["actual_boom"].astype("Int64")
    return out


def run(corpus: Corpus | None = None) -> dict:
    corpus = corpus or load_corpus()
    models = {t: refit_model(corpus, t) for t in ORIGINS}
    view = group_view(frozen_model(), corpus.vectors)
    shares = np.bincount(view["group"], minlength=N_GROUPS).astype(float)
    null = null_worlds(corpus.year, feature_half(corpus.year), view["top_prob"].to_numpy(),
                       view["dist"].to_numpy(), corpus.edges, shares)
    results, evals = {"null": null}, []
    for mode in ["refit", "frozen"]:
        panel, labels = build_panel(corpus, mode, models=models if mode == "refit" else None)
        pred = walk_forward(panel)
        metrics = per_origin_metrics(pred)
        evals.append(evaluate(metrics, mode, null=null))
        results[mode] = {"panel": panel, "labels": labels, "pred": pred, "metrics": metrics}
    live = live_forecast(results["refit"]["panel"], corpus)
    contract = to_contract(results["frozen"]["pred"], live, corpus.group_names)
    contracts.save("area_forecasts", contract)
    evaluation = pd.concat(evals, ignore_index=True)
    evaluation.to_csv(FORECAST_EVAL_CSV, index=False)
    null.to_csv(NULL_CSV, index=False)
    results.update({"live": live, "eval": evaluation, "models": models, "contract": contract})
    return results


def main() -> None:
    res = run()
    ev = res["eval"]
    cols = ["metric", "model", "groups_mode", "value", "ci_low", "ci_high", "holm_p"]
    print(ev[cols].round(3).to_string(index=False))
    print("H11 pass:", bool(ev["h11_pass"].dropna().iloc[0]))
    live = res["contract"][res["contract"]["actual_boom"].isna()].sort_values("growth_score", ascending=False)
    print(live.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
