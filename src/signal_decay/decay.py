"""Before/after-publication return tests and half-life fitting.

Split rule: a month is "post" if its month-end date is on or after
publication_date, so the publication month itself is post. Everything earlier
is "pre". The sample-end-to-publication window (McLean and Pontiff) is not
used yet. TODO: add it if papers get a verified sample_end date.

ci_low / ci_high in decay_results.csv are the 95% confidence interval of
(post_mean - pre_mean), from an OLS of returns on a constant and a post dummy
with Newey-West (HAC) standard errors, which allows for autocorrelated and
heteroskedastic monthly returns. A Welch interval is also returned for
comparison and written to decay_details.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit

MIN_MONTHS_EACH_SIDE = 120  # team rule: at least 10 years before and after
HAC_LAGS = 12


def _side_stats(x: pd.Series) -> dict:
    """n, mean, standard error and 95% t-interval for one sample."""
    x = x.dropna()
    n = len(x)
    if n < 2:
        return {"n": n, "mean": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return {"n": n, "mean": mean, "se": se, "ci_low": mean - t * se, "ci_high": mean + t * se}


def _welch(pre: dict, post: dict) -> tuple[float, float, float]:
    """Welch standard error and 95% CI for post minus pre."""
    diff = post["mean"] - pre["mean"]
    se = np.sqrt(pre["se"] ** 2 + post["se"] ** 2)
    if not (np.isfinite(se) and se > 0):
        return np.nan, np.nan, np.nan
    dof = se**4 / (pre["se"] ** 4 / (pre["n"] - 1) + post["se"] ** 4 / (post["n"] - 1))
    t = stats.t.ppf(0.975, df=dof)
    return se, diff - t * se, diff + t * se


def _status(pre_n: int, post_n: int, min_months: int) -> str:
    """'ok', or why the signal falls below the minimum window."""
    reasons = []
    if pre_n < min_months:
        reasons.append(f"insufficient_pre:{pre_n}")
    if post_n < min_months:
        reasons.append(f"insufficient_post:{post_n}")
    return ";".join(reasons) or "ok"


def before_after_test(
    returns: pd.Series,
    publication_date,
    hac_lags: int = HAC_LAGS,
    min_months: int = MIN_MONTHS_EACH_SIDE,
) -> dict:
    """Compare mean monthly returns before vs after a publication date.

    returns is a Series indexed by date. Returns a dict with:
      pre_* / post_*        n, mean, se, ci_low, ci_high for each side (plain t-interval)
      diff                  post_mean - pre_mean
      diff_se, diff_ci_low, diff_ci_high, t_stat, p_value   Newey-West (HAC)
      welch_se, welch_ci_low, welch_ci_high                 Welch, for comparison
      status                'ok' or e.g. 'insufficient_pre:84' (stats still computed)
    """
    r = returns.dropna().sort_index()
    pub = pd.Timestamp(publication_date)
    post_mask = r.index >= pub
    pre = _side_stats(r[~post_mask])
    post = _side_stats(r[post_mask])

    out = {f"pre_{k}": v for k, v in pre.items()}
    out.update({f"post_{k}": v for k, v in post.items()})
    out["diff"] = post["mean"] - pre["mean"]
    out["welch_se"], out["welch_ci_low"], out["welch_ci_high"] = _welch(pre, post)

    hac = dict(diff_se=np.nan, diff_ci_low=np.nan, diff_ci_high=np.nan, t_stat=np.nan, p_value=np.nan)
    if pre["n"] >= 2 and post["n"] >= 2:
        X = sm.add_constant(post_mask.astype(float))
        fit = sm.OLS(r.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        lo, hi = fit.conf_int(alpha=0.05)[1]
        hac = dict(
            diff_se=fit.bse[1], diff_ci_low=lo, diff_ci_high=hi,
            t_stat=fit.tvalues[1], p_value=fit.pvalues[1],
        )
    out.update(hac)
    out["status"] = _status(pre["n"], post["n"], min_months)
    return out


def _exp_decay(t, a, k):
    return a * np.exp(-k * t)


def fit_decay_curve(post: np.ndarray, window: int = 36) -> tuple[float, float] | None:
    """Fit m(t) = a * exp(-k t) to the rolling ``window``-month mean of post returns.

    t counts rolling windows from the first full one. Returns (a, k), or None if
    the series is too short, the starting mean is not positive, or the fit fails.
    """
    if len(post) < window + 2:
        return None
    m = pd.Series(post).rolling(window).mean().dropna().to_numpy()
    if m[0] <= 0:
        return None
    t = np.arange(len(m), dtype=float)
    try:
        (a, k), _ = curve_fit(_exp_decay, t, m, p0=(m[0], 0.01), bounds=([0, -1], [np.inf, 1]))
    except (RuntimeError, ValueError):
        return None
    return float(a), float(k)


def expected_path(a: float, k: float, n: int, window: int = 36) -> np.ndarray:
    """Monthly expected return mu(s) = c * exp(-k s), s = 0..n-1, implied by a rolling-mean fit.

    The rolling mean of c * exp(-k s) starting at month j equals
    c * exp(-k j) * (1 - exp(-k w)) / (w * (1 - exp(-k))), so c follows from a.
    """
    if abs(k) < 1e-12:
        c = a
    else:
        c = a * window * (1 - np.exp(-k)) / (1 - np.exp(-k * window))
    return c * np.exp(-k * np.arange(n))


def fit_half_life(
    returns: pd.Series,
    publication_date,
    window: int = 36,
    min_post_months: int = 60,
    max_half_life: float = 600.0,
) -> float | None:
    """Months for the post-publication mean return to halve, or None.

    Fits m(t) = a * exp(-k t) to the rolling ``window``-month mean of
    post-publication returns, t in months. Half-life = ln 2 / k.

    Returns None (left empty in the output) when there are fewer than
    ``min_post_months`` post months, the starting mean is not positive, the
    fit fails, k <= 0 (no decay), or the half-life exceeds ``max_half_life``
    (too slow to tell apart from no decay).
    """
    r = returns.dropna().sort_index()
    post = r[r.index >= pd.Timestamp(publication_date)]
    if len(post) < max(min_post_months, window + 2):
        return None
    fit = fit_decay_curve(post.to_numpy(), window)
    if fit is None:
        return None
    _, k = fit
    if k <= 0:
        return None
    half_life = float(np.log(2) / k)
    return half_life if half_life <= max_half_life else None
