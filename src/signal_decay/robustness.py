"""Robustness checks for the decay results (see HYPOTHESES.md).

- placebo_test / pooled_placebo_test (H2): is the drop at the real publication
  date bigger than the drop at fake dates?
- bootstrap_decay (half-life rule): block-bootstrap intervals for the half-life
  and for the retained share (post_mean / pre_mean).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.signal_decay.decay import MIN_MONTHS_EACH_SIDE, expected_path, fit_decay_curve, fit_half_life

BOOTSTRAP_DRAWS = 1000
BLOCK_MONTHS = 12
SEED = 0
PLACEBO_DRAWS = 10_000
HALF_LIFE_MIN_FIT_RATE = 0.80
HALF_LIFE_MAX_UPPER = 600.0
HALF_LIFE_MAX_NULL_P = 0.10


# ---------- H2: placebo publication dates ----------

def split_differences(returns: pd.Series, min_months: int = MIN_MONTHS_EACH_SIDE) -> pd.Series:
    """post_mean - pre_mean for every admissible split month.

    Index = first post month. A split is admissible if both sides have at
    least min_months observations.
    """
    r = returns.dropna().sort_index()
    x = r.to_numpy()
    n = len(x)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    k = np.arange(min_months, n - min_months + 1)  # number of pre months
    if len(k) == 0:
        return pd.Series(dtype=float)
    pre = csum[k] / k
    post = (csum[n] - csum[k]) / (n - k)
    return pd.Series(post - pre, index=r.index[k])


def placebo_test(returns: pd.Series, publication_date, min_months: int = MIN_MONTHS_EACH_SIDE) -> dict:
    """Share of admissible fake splits with a drop at least as large as the real one."""
    diffs = split_differences(returns, min_months)
    r = returns.dropna().sort_index()
    post = r.index >= pd.Timestamp(publication_date)
    real = r[post].mean() - r[~post].mean()
    if diffs.empty:
        return {"real_diff": real, "placebo_n": 0, "placebo_p": np.nan}
    return {
        "real_diff": float(real),
        "placebo_n": int(len(diffs)),
        "placebo_p": float((diffs <= real).mean()),
    }


def pooled_placebo_test(
    series_by_signal: dict[str, pd.Series],
    publication_dates: dict[str, pd.Timestamp],
    draws: int = PLACEBO_DRAWS,
    seed: int = SEED,
    min_months: int = MIN_MONTHS_EACH_SIDE,
) -> dict:
    """Mean real drop across signals against the mean of one random fake split per signal."""
    rng = np.random.default_rng(seed)
    reals, fakes = [], []
    for sig, series in series_by_signal.items():
        diffs = split_differences(series, min_months).to_numpy()
        if len(diffs) == 0:
            continue
        reals.append(placebo_test(series, publication_dates[sig], min_months)["real_diff"])
        fakes.append(rng.choice(diffs, size=draws, replace=True))
    if not reals:
        return {"n_signals": 0, "mean_real_diff": np.nan, "p_one_sided": np.nan}
    mean_real = float(np.mean(reals))
    mean_fake = np.mean(np.vstack(fakes), axis=0)
    return {
        "n_signals": len(reals),
        "mean_real_diff": mean_real,
        "mean_fake_diff": float(mean_fake.mean()),
        "p_one_sided": float((mean_fake <= mean_real).mean()),
    }


# ---------- half-life and retained-share intervals ----------

def _block_resample(x: np.ndarray, rng: np.random.Generator, block: int = BLOCK_MONTHS) -> np.ndarray:
    """Moving-block bootstrap: glue random contiguous blocks until length matches."""
    n = len(x)
    if n <= block:
        return rng.choice(x, size=n, replace=True)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([x[s : s + block] for s in starts])[:n]


def bootstrap_decay(
    returns: pd.Series,
    publication_date,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> dict:
    """Bootstrap intervals for the retained share and the half-life.

    Retained share (post_mean / pre_mean): pre and post returns are block-
    resampled separately (12-month blocks); order does not matter for a mean.

    Half-life: residual block bootstrap. Fit the decay curve, take residuals
    around the fitted monthly path, block-resample the residuals, add them back
    to the path and refit. Resampling the raw returns would scramble their time
    order and erase the decay being measured.

    Returns retained_share, retained_ci_low, retained_ci_high, half_life_ci_low,
    half_life_ci_high, half_life_fit_rate (share of draws where decay was found),
    half_life_reportable (the HYPOTHESES.md rule). If the point estimate finds
    no decay, the half-life bootstrap is skipped and it is not reportable.
    """
    r = returns.dropna().sort_index()
    pub = pd.Timestamp(publication_date)
    pre, post = r[r.index < pub].to_numpy(), r[r.index >= pub].to_numpy()
    out = dict(retained_share=np.nan, retained_ci_low=np.nan, retained_ci_high=np.nan,
               half_life_ci_low=np.nan, half_life_ci_high=np.nan, half_life_fit_rate=np.nan,
               half_life_null_p=np.nan, half_life_reportable=False)
    if len(pre) < 2 or len(post) < 2:
        return out

    rng = np.random.default_rng(seed)
    pre_mean = pre.mean()
    out["retained_share"] = post.mean() / pre_mean if pre_mean > 0 else np.nan

    ratios = []
    for _ in range(draws):
        bpre, bpost = _block_resample(pre, rng), _block_resample(post, rng)
        if bpre.mean() > 0:
            ratios.append(bpost.mean() / bpre.mean())
    if ratios and pre_mean > 0:
        out["retained_ci_low"], out["retained_ci_high"] = np.percentile(ratios, [2.5, 97.5])

    post_index = r.index[r.index >= pub]
    point = fit_half_life(r, pub)
    fit = fit_decay_curve(post) if point is not None else None
    if fit is None:
        return out
    path = expected_path(*fit, n=len(post))
    resid = post - path
    half_lives = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # curve_fit warns on hard draws; failures count as no decay
        for _ in range(draws):
            bpost = path + _block_resample(resid, rng)
            hl = fit_half_life(pd.Series(bpost, index=post_index), pub)
            half_lives.append(np.nan if hl is None else hl)

        # Null of no decay: resample residuals around a flat line and refit.
        # p = share of null draws with a decay rate at least as steep as observed.
        flat = post.mean()
        resid0 = post - flat
        k_obs = fit[1]
        null_hits = 0
        for _ in range(draws):
            f0 = fit_decay_curve(flat + _block_resample(resid0, rng))
            null_hits += f0 is not None and f0[1] >= k_obs
    out["half_life_null_p"] = float(null_hits / draws)

    hl = np.asarray(half_lives)
    found = hl[~np.isnan(hl)]
    out["half_life_fit_rate"] = float(len(found) / draws)
    if len(found):
        out["half_life_ci_low"], out["half_life_ci_high"] = np.percentile(found, [2.5, 97.5])
    out["half_life_reportable"] = bool(
        out["half_life_null_p"] < HALF_LIFE_MAX_NULL_P
        and out["half_life_fit_rate"] >= HALF_LIFE_MIN_FIT_RATE
        and np.isfinite(out["half_life_ci_high"])
        and out["half_life_ci_high"] <= HALF_LIFE_MAX_UPPER
    )
    return out
