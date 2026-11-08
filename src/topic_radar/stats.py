"""Shared evaluation tools: within-origin z-scores, walk-forward fits, block bootstrap, Holm."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

from src.signal_decay.vol_model import holm  # noqa: F401  (re-exported)

GAP = 3  # a target at origin s ends at s + 3, so a model scored at t trains on s <= t - 3
MIN_TRAIN_ORIGINS = 3
TUNE_MIN_ORIGINS = 8
VAL_ORIGINS = 3
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
RIDGE_DEFAULT = 10.0
LOGIT_CS = [0.01, 0.1, 1.0, 10.0]
LOGIT_DEFAULT = 1.0
BLOCK = 3
DRAWS = 5000


def zscore_within(panel: pd.DataFrame, cols: list[str], by: str = "origin") -> pd.DataFrame:
    """z-score each column within each origin; missing values become 0 (the origin mean)."""
    out = panel.copy()
    for c in cols:
        g = out.groupby(by)[c]
        sd = g.transform("std").replace(0, np.nan)
        out[c] = ((out[c] - g.transform("mean")) / sd).fillna(0.0)
    return out


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def train_origins(panel: pd.DataFrame, t: int) -> pd.DataFrame:
    return panel[(panel["origin"] <= t - GAP) & panel["target"].notna()]


def _per_origin(df: pd.DataFrame, pred: np.ndarray, score: Callable) -> float:
    df = df.assign(_p=pred)
    return float(np.mean([score(g["_p"], g) for _, g in df.groupby("origin")]))


def choose(train: pd.DataFrame, features: list[str], make: Callable, grid: list, default,
           score: Callable, predict: Callable):
    """Pick a setting on the last VAL_ORIGINS training origins (inner train keeps the gap)."""
    origins = sorted(train["origin"].unique())
    if len(origins) < TUNE_MIN_ORIGINS:
        return default
    val = origins[-VAL_ORIGINS:]
    inner = train[train["origin"] <= val[0] - GAP]
    if inner["origin"].nunique() < MIN_TRAIN_ORIGINS:
        return default
    held = train[train["origin"].isin(val)]
    if inner["fit_target"].nunique() < 2:
        return default
    best, best_score = default, -np.inf
    for value in grid:
        model = make(value).fit(inner[features], inner["fit_target"])
        s = _per_origin(held, predict(model, held[features]), score)
        if s > best_score + 1e-12:
            best, best_score = value, s
    return best


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, float]:
    alpha = choose(train, features, lambda a: Ridge(alpha=a), RIDGE_ALPHAS, RIDGE_DEFAULT,
                   lambda p, g: spearman(p, g["target"]), lambda m, x: m.predict(x))
    model = Ridge(alpha=alpha).fit(train[features], train["fit_target"])
    return model.predict(test[features]), alpha


def logit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                  label: str = "boom") -> tuple[np.ndarray, float]:
    t2 = train.assign(fit_target=train[label].astype(int))
    if t2["fit_target"].nunique() < 2:
        return np.full(len(test), t2["fit_target"].mean()), np.nan

    def make(c):
        return LogisticRegression(C=c, max_iter=2000)

    def loss(p, g):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        y = g[label].astype(int).to_numpy()
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    c = choose(t2, features, make, LOGIT_CS, LOGIT_DEFAULT, loss, lambda m, x: m.predict_proba(x)[:, 1])
    model = make(c).fit(t2[features], t2["fit_target"])
    return model.predict_proba(test[features])[:, 1], c


def block_bootstrap(x: np.ndarray, block: int = BLOCK, draws: int = DRAWS, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap of the mean of a per-origin series (in origin order)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.full(draws, np.nan)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]
    return x[idx].mean(axis=1)


def summarise(x: np.ndarray, seed: int = 0) -> dict:
    """Mean, 95% CI and one-sided p-value (mean <= 0) from the block bootstrap."""
    boot = block_bootstrap(x, seed=seed)
    return {"value": float(np.mean(x)), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "p_one_sided": float((np.sum(boot <= 0) + 1) / (len(boot) + 1))}


def top_k(score, k: int, seed: int = 0) -> np.ndarray:
    """Indices of the k largest scores; ties broken by a fixed random order."""
    score = np.asarray(score, float)
    tie = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tie, -score))[:k]
