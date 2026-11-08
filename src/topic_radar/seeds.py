"""H12: pick recent papers that the next three years of work will cluster around.

Seed score of a candidate at origin t = number of papers from t+1..t+3 with cosine
similarity >= tau to it (tau = 99th percentile of random-pair similarity). Candidates are
papers from t-2..t; true seeds are the top 10% by seed score. Features use papers <= t
only, including the level-1 growth score of the paper's group (groups refit on <= t).
Pass rule is in HYPOTHESES.md.

Run: python -m src.topic_radar.seeds   (runs level 1 and level 2 first)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.topic_radar import SEED_EVAL_CSV, SEED_PAPERS_CSV
from src.topic_radar.panel import LAST_TARGET_YEAR, Corpus
from src.topic_radar.stats import (
    GAP, MIN_TRAIN_ORIGINS, holm, ridge_predict, spearman, summarise, top_k, zscore_within,
)

TAU_QUANTILE = 99
TAU_PAIRS = 200_000
CANDIDATE_YEARS = 3  # t-2..t
HORIZON = 3  # t+1..t+3
SEED_SHARE = 0.10
PICK = 20
LIST_PER_GROUP = 10
SMOOTH = 0.5
FEATURES = ["age", "group_growth", "local_growth", "density", "novelty", "bridging"]
SEED = 0


def similarity_threshold(unit: np.ndarray, pairs: int = TAU_PAIRS, seed: int = SEED) -> float:
    """99th percentile of cosine similarity between random distinct paper pairs."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(unit), pairs)
    b = rng.integers(0, len(unit), pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    chunk = 10_000  # bounded memory
    sims = np.concatenate([np.einsum("ij,ij->i", unit[a[i:i + chunk]], unit[b[i:i + chunk]])
                           for i in range(0, len(a), chunk)])
    return float(np.percentile(sims, TAU_QUANTILE))


def candidate_features(corpus: Corpus, t: int, labels: np.ndarray, group_score: dict[int, float],
                       tau: float) -> pd.DataFrame:
    """Features (papers <= t only) and seed score (papers t+1..t+3) for candidates t-2..t."""
    year = corpus.year
    cand = np.flatnonzero((year >= t - CANDIDATE_YEARS + 1) & (year <= t))
    past = np.flatnonzero(year <= t)
    sims = corpus.unit[cand] @ corpus.unit[past].T
    near = sims >= tau
    near[np.arange(len(cand)), np.searchsorted(past, cand)] = False  # not itself
    py = year[past]
    recent = (py >= t - 2) & (py <= t)
    before = (py >= t - 5) & (py <= t - 3)
    out = pd.DataFrame({"idx": cand, "origin": t, "year": year[cand]})
    out["age"] = t - year[cand]
    out["density"] = near.sum(axis=1)
    out["local_growth"] = np.log((near[:, recent].sum(axis=1) + SMOOTH) / (near[:, before].sum(axis=1) + SMOOTH))
    older = py[None, :] < year[cand][:, None]
    best_old = np.where(older, sims, -np.inf).max(axis=1)
    out["novelty"] = np.where(np.isfinite(best_old), 1 - best_old, np.nan)
    out["group"] = labels[cand]
    out["group_growth"] = out["group"].map(group_score).astype(float)

    e = corpus.edges
    if len(e):
        ok = (year[e[:, 0]] <= t) & (year[e[:, 1]] <= t)
        e = e[ok]
        cross = labels[e[:, 0]] != labels[e[:, 1]]
        n = len(year)
        touch = np.bincount(e[:, 0], minlength=n) + np.bincount(e[:, 1], minlength=n)
        crossed = np.bincount(e[cross, 0], minlength=n) + np.bincount(e[cross, 1], minlength=n)
        out["bridging"] = np.where(touch[cand] > 0, crossed[cand] / np.maximum(touch[cand], 1), np.nan)
    else:
        out["bridging"] = np.nan

    fut = np.flatnonzero((year >= t + 1) & (year <= t + HORIZON))
    if t + HORIZON <= LAST_TARGET_YEAR and len(fut):
        out["seed_score"] = (corpus.unit[cand] @ corpus.unit[fut].T >= tau).sum(axis=1)
    else:
        out["seed_score"] = np.nan
    return out


def label_seeds(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["target"] = np.log1p(panel["seed_score"])
    panel["fit_target"] = panel["target"] - panel.groupby("origin")["target"].transform("mean")
    panel["true_seed"] = False
    for _, g in panel[panel["target"].notna()].groupby("origin"):
        k = int(np.ceil(SEED_SHARE * len(g)))
        panel.loc[g.index[top_k(g["seed_score"].to_numpy(), k)], "true_seed"] = True
    return panel


def build_panel(corpus: Corpus, labels: dict[int, np.ndarray], scores: pd.DataFrame, tau: float) -> pd.DataFrame:
    """scores: origin, group, growth_score (level-1 out-of-sample, refit groups)."""
    rows = []
    for t, g in scores.groupby("origin"):
        rows.append(candidate_features(corpus, int(t), labels[int(t)],
                                       dict(zip(g["group"], g["growth_score"])), tau))
    panel = pd.concat(rows, ignore_index=True)
    panel["raw_density"], panel["raw_age"] = panel["density"], panel["age"]
    return label_seeds(zscore_within(panel, FEATURES))


def walk_forward(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t in sorted(panel["origin"].unique()):
        train = panel[(panel["origin"] <= t - GAP) & panel["target"].notna()]
        if train["origin"].nunique() < MIN_TRAIN_ORIGINS:
            continue
        test = panel[panel["origin"] == t].copy()
        test["score"], test["alpha"] = ridge_predict(train, test, FEATURES)
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def per_origin_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in pred[pred["target"].notna()].groupby("origin"):
        seeds = g["true_seed"].to_numpy()
        row = {"origin": t, "n_candidates": len(g), "random": float(seeds.mean())}
        for name, s in [("ridge", g["score"]), ("density", g["raw_density"]), ("newest", -g["raw_age"])]:
            row[name] = float(seeds[top_k(s.to_numpy(), PICK)].mean())
        row["spearman_ridge"] = spearman(g["score"], g["seed_score"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("origin")


INFERENCE_ORIGINS = [2013, 2016, 2019, 2022]  # candidates and futures do not overlap
RANDOM_DRAWS = 5000
PERMUTATIONS = 2000


def _precision(score: np.ndarray, seeds_mask: np.ndarray, rng: np.random.Generator) -> float:
    tie = rng.random(len(score))
    return float(seeds_mask[np.lexsort((tie, -score))[:PICK]].mean())


def permutation_tests(pred: pd.DataFrame, origins=INFERENCE_ORIGINS, seed: int = SEED) -> dict:
    """p-values on non-overlapping origins: ridge vs random picks, and ridge vs density.

    vs density: within each origin, split the ridge score into its linear fit on density
    plus a leftover, and shuffle only the leftover. If the leftover says nothing about
    seeds, shuffling it does not hurt, so the real precision is not unusual.
    """
    rng = np.random.default_rng(seed)
    groups = [g for t, g in pred[pred["target"].notna()].groupby("origin") if t in set(origins)]
    if not groups:
        return {"n_origins": 0, "ridge": np.nan, "p_vs_random": np.nan, "p_vs_density": np.nan}
    real = np.mean([g["true_seed"].to_numpy()[top_k(g["score"].to_numpy(), PICK)].mean() for g in groups])
    rand = np.array([np.mean([rng.choice(g["true_seed"].to_numpy(), PICK, replace=False).mean()
                              for g in groups]) for _ in range(RANDOM_DRAWS)])
    parts = []  # score = (part explained by density) + leftover, per origin
    for g in groups:
        d, s = g["raw_density"].to_numpy(float), g["score"].to_numpy(float)
        x = np.column_stack([np.ones_like(d), d])
        fitted = x @ np.linalg.lstsq(x, s, rcond=None)[0]
        parts.append((fitted, s - fitted, g["true_seed"].to_numpy()))
    perm = np.empty(PERMUTATIONS)
    for b in range(PERMUTATIONS):
        perm[b] = np.mean([_precision(f + rng.permutation(r), seeds_mask, rng)
                           for f, r, seeds_mask in parts])
    return {"n_origins": len(groups), "ridge": float(real),
            "random_mean": float(rand.mean()), "density_shuffle_mean": float(perm.mean()),
            "p_vs_random": float((1 + (rand >= real).sum()) / (RANDOM_DRAWS + 1)),
            "p_vs_density": float((1 + (perm >= real).sum()) / (PERMUTATIONS + 1))}


def evaluate(metrics: pd.DataFrame, tests: dict | None = None) -> pd.DataFrame:
    """Descriptive rows over all origins; pass tests from `permutation_tests` (HYPOTHESES.md)."""
    rows = []

    def add(metric, model, x, **extra):
        rows.append({"metric": metric, "model": model, "n_origins": len(x),
                     **summarise(np.asarray(x, float)), **extra})

    for name in ["ridge", "density", "newest", "random"]:
        add("precision_at_20", name, metrics[name])
    add("spearman", "ridge", metrics["spearman_ridge"])
    add("context", "ridge_minus_density", metrics["ridge"] - metrics["density"])
    add("context", "ridge_minus_random", metrics["ridge"] - metrics["random"])
    out = pd.DataFrame(rows)
    if tests is None:
        out["h12_pass"] = np.nan
        return out
    adj = holm([tests["p_vs_random"], tests["p_vs_density"]])
    extra = pd.DataFrame([
        {"metric": "pass_test", "model": "ridge_vs_random", "n_origins": tests["n_origins"],
         "value": tests["ridge"] - tests["random_mean"], "p_one_sided": tests["p_vs_random"], "holm_p": adj[0]},
        {"metric": "pass_test", "model": "ridge_vs_density", "n_origins": tests["n_origins"],
         "value": tests["ridge"] - tests["density_shuffle_mean"], "p_one_sided": tests["p_vs_density"],
         "holm_p": adj[1]}])
    out = pd.concat([out, extra], ignore_index=True)
    out["h12_pass"] = bool((extra["value"] > 0).all() and (extra["holm_p"] < 0.05).all())
    return out


def reading_list(corpus: Corpus, panel: pd.DataFrame, live: pd.DataFrame, subtopics: pd.DataFrame) -> pd.DataFrame:
    """Top papers from 2023-2025 per frozen group, scored by a model trained on origins <= 2022."""
    train = panel[(panel["origin"] <= LAST_TARGET_YEAR - GAP) & panel["target"].notna()]
    live = live.copy()
    live["score"], _ = ridge_predict(train, live, FEATURES)
    live["paper_id"] = corpus.paper_id[live["idx"]]
    live["title"] = corpus.title[live["idx"]]
    live["group"] = live["group"].map(corpus.group_names)
    live = live.merge(subtopics[["paper_id", "subtopic"]], on="paper_id", how="left")
    live = live.sort_values(["group", "score"], ascending=[True, False])
    live["rank"] = live.groupby("group").cumcount() + 1
    cols = ["paper_id", "title", "year", "group", "subtopic", "score", "rank"]
    return live[live["rank"] <= LIST_PER_GROUP][cols].round({"score": 4}).reset_index(drop=True)


def run(corpus: Corpus, level1: dict, subtopic_papers: pd.DataFrame) -> dict:
    tau = similarity_threshold(corpus.unit)
    refit = level1["refit"]
    scores = refit["pred"][["origin", "group", "growth_score"]]
    panel = build_panel(corpus, refit["labels"], scores, tau)
    pred = walk_forward(panel)
    metrics = per_origin_metrics(pred)
    evaluation = evaluate(metrics, permutation_tests(pred)).assign(tau=tau)
    evaluation.to_csv(SEED_EVAL_CSV, index=False)

    live_groups = level1["live"]
    live_scores = dict(zip(live_groups["group"], live_groups["growth_score"]))
    live = candidate_features(corpus, LAST_TARGET_YEAR, live_groups.attrs["labels"], live_scores, tau)
    live["raw_density"], live["raw_age"] = live["density"], live["age"]
    live = zscore_within(live, FEATURES)
    papers = reading_list(corpus, panel, live, subtopic_papers)
    papers.to_csv(SEED_PAPERS_CSV, index=False)
    return {"tau": tau, "panel": panel, "pred": pred, "metrics": metrics, "eval": evaluation,
            "papers": papers}
