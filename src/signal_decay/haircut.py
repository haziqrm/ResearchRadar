"""Expected post-publication value of a claimed backtest return.

CAVEAT: this is a historical average applied as an assumption, not a prediction for
any one signal. It covers replicable published signals only (the OSAP predictors);
signals that were never published, or did not replicate, are not represented.

How it works: share_kept = post-publication mean / in-sample mean for each OSAP signal
(data/decay_results.csv, source "osap"). haircut(x) multiplies x by the 25th percentile,
median and 75th percentile of share_kept, each with a 95% bootstrap interval that
resamples papers (signals from the same paper move together).

share_kept outside [0, 1]: values are not clipped. Below 0 means the edge reversed after
publication; above 1 means it grew. Percentiles are barely moved by such extremes, so they
are kept as measured, and the output reports how often each case occurred.

Usage:
    from src.signal_decay.haircut import haircut
    haircut(0.01)   # a claimed 1% a month
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.common import contracts

CAVEAT = ("Historical average applied as an assumption, not a prediction for any one signal; "
          "replicable published signals only (OSAP).")
QUANTILES = {"p25": 25, "median": 50, "p75": 75}
BOOTSTRAP_DRAWS = 2000
SEED = 0


def share_distribution(results: pd.DataFrame | None = None, draws: int = BOOTSTRAP_DRAWS,
                       seed: int = SEED) -> dict:
    """Percentiles of OSAP share_kept with paper-bootstrap intervals."""
    df = contracts.load("decay_results") if results is None else results
    df = df[(df["source"] == "osap") & df["share_kept"].notna()]
    s = df["share_kept"].to_numpy(float)
    papers = df["paper"].to_numpy()
    out = {"n_signals": len(s), "n_papers": int(pd.Series(papers).nunique()),
           "share_below_zero": float((s < 0).mean()), "share_above_one": float((s > 1).mean())}
    uniq = np.unique(papers)
    idx = {p: np.flatnonzero(papers == p) for p in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(QUANTILES)))
    for i in range(draws):
        rows = np.concatenate([idx[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        boot[i] = np.percentile(s[rows], list(QUANTILES.values()))
    for j, (name, q) in enumerate(QUANTILES.items()):
        out[name] = float(np.percentile(s, q))
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = (float(v) for v in np.percentile(boot[:, j], [2.5, 97.5]))
    return out


@lru_cache(maxsize=1)
def _default_distribution() -> tuple:
    return tuple(share_distribution().items())


def haircut(claimed_value: float, distribution: dict | None = None) -> dict:
    """Expected range of what survives publication from a claimed (positive) backtest return.

    Historical average applied as an assumption, not a prediction for any one signal;
    covers replicable published signals only. claimed_value must be >= 0 (use the
    long-short return in the direction the paper trades). Returns the claimed value times
    the 25th percentile, median and 75th percentile of share_kept, with 95% intervals,
    plus how often signals kept less than 0 or more than 1 of their edge.
    """
    if not np.isfinite(claimed_value) or claimed_value < 0:
        raise ValueError("claimed_value must be a finite number >= 0 (the long-short return as claimed)")
    d = distribution if distribution is not None else dict(_default_distribution())
    out = {"claimed_value": float(claimed_value), "caveat": CAVEAT}
    for name in QUANTILES:
        for suffix in ("", "_ci_low", "_ci_high"):
            out[f"{name}{suffix}"] = claimed_value * d[f"{name}{suffix}"]
    out.update({k: d[k] for k in ("share_below_zero", "share_above_one", "n_signals", "n_papers")})
    return out
