"""H10: trained models for a published signal's next-month volatility.

Rules are fixed in HYPOTHESES.md (H10). Compares a HAR-style ridge (primary) and
gradient boosting (secondary) against the H9b rules, with an expanding walk-forward
by calendar year and a 12-month gap. Every feature for month t uses data up to t-1.
H9 code and results are reused by import and not changed.

Usage:
    python -m src.signal_decay.vol_model [--charts] [--refresh-vix]

Writes A-owned files only:
    data/osap_h10_summary.csv    tests (primary 2000-2025, secondary 1996-1999)
    data/osap_h10_folds.csv      per test year: settings chosen, rows, mean losses
    data/osap_h10_portfolio.csv  volatility-managed portfolios before and after costs
Caches the FRED VIX download in data/cache/fred/ (git-ignored).
"""

from __future__ import annotations

import argparse
import io
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import contracts
from src.signal_decay import CACHE_DIR, CHARTS_DIR, DATA_DIR
from src.signal_decay.forecasts import ABS_SCALE, hac_mean_test, sharpe
from src.signal_decay.osap import load_ls_returns, load_signal_doc, predictor_table

EPS = 1e-4
EULER_GAMMA = 0.5772156649015329
DEFAULT_C = float(np.exp((EULER_GAMMA + np.log(2)) / 2))  # E[log|z|] = -(gamma + ln 2) / 2 for z ~ N(0, 1)
SAMPLE_START = pd.Timestamp("1990-01-01")
TEST_YEARS = list(range(1996, 2026))
PRIMARY = (2000, 2025)
SECONDARY = (1996, 1999)
MIN_TUNE_YEARS = 8
VAL_YEARS = 3
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
RIDGE_DEFAULT_ALPHA = 1.0
HGB_GRID = [dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=leaf, max_iter=300)
            for lr, leaves, leaf in product([0.05, 0.1], [15, 31], [50, 200])]
HGB_DEFAULT = dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200, max_iter=300)
WEIGHT_CAP = 5.0
COSTS_BPS = [0, 10, 25]
BOOTSTRAP_DRAWS = 2000
SEED = 0

VOL_FEATURES = ["rms_1", "rms_3", "rms_6", "rms_12", "rms_all", "vol_of_vol", "cross_vol",
                "mkt_rms_1", "mkt_rms_12", "vix"]
FEATURES = VOL_FEATURES + ["age"]

VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
FRED_CACHE_DIR = CACHE_DIR / "fred"
SUMMARY_CSV = DATA_DIR / "osap_h10_summary.csv"
FOLDS_CSV = DATA_DIR / "osap_h10_folds.csv"
PORTFOLIO_CSV = DATA_DIR / "osap_h10_portfolio.csv"


# ---------- inputs ----------

def parse_vix(text: str) -> pd.Series:
    """Month-end VIX close (last trading day of each month), indexed by month-end date."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"])
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna()
    s = df.groupby(df["date"] + pd.offsets.MonthEnd(0))["vix"].last()
    s.index.name = "date"
    return s


def load_vix(cache_dir: Path = FRED_CACHE_DIR, refresh: bool = False) -> pd.Series:
    path = cache_dir / "VIXCLS.csv"
    if path.exists() and not refresh:
        return parse_vix(path.read_text())
    try:
        resp = requests.get(VIX_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download VIX from FRED ({type(exc).__name__}); H10 needs it.") from None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    return parse_vix(resp.text)


def build_features(returns: pd.DataFrame, pub_year: pd.Series, market: pd.Series, vix: pd.Series,
                   start: pd.Timestamp = SAMPLE_START) -> pd.DataFrame:
    """One row per post-publication signal-month with every feature at t-1 or earlier.

    returns: tidy (date, signal, return) for all predictors; market: Mkt-RF by month-end;
    vix: month-end VIX level. Rows with any missing feature are dropped, so every model
    and baseline is scored on the same rows.
    """
    returns = returns.assign(date=pd.to_datetime(returns["date"]))
    wide = returns.pivot(index="date", columns="signal", values="return").sort_index()
    a, sq = wide.abs(), wide**2
    mats = {"abs_r": a}
    for k in (1, 3, 6, 12):
        mats[f"rms_{k}"] = np.sqrt(sq.rolling(k, min_periods=k).mean().shift(1))
    mats["rms_all"] = np.sqrt(sq.expanding(min_periods=36).mean().shift(1))
    mats["vol_of_vol"] = a.rolling(12, min_periods=12).std().shift(1)
    long = pd.concat({k: v.stack(dropna=False) for k, v in mats.items()}, axis=1)
    long.index.names = ["date", "signal"]
    long = long.reset_index()

    cross = a.mean(axis=1).shift(1)
    m = market.copy()
    m.index = pd.to_datetime(m.index) + pd.offsets.MonthEnd(0)
    m = m.sort_index()
    mkt_1 = np.sqrt((m**2).shift(1))
    mkt_12 = np.sqrt((m**2).rolling(12, min_periods=12).mean().shift(1))
    v = vix.copy()
    v.index = pd.to_datetime(v.index) + pd.offsets.MonthEnd(0)
    vix_prev = (v.sort_index() / 100 / np.sqrt(12)).shift(1, freq="ME")

    long["cross_vol"] = long["date"].map(cross)
    long["mkt_rms_1"] = long["date"].map(mkt_1)
    long["mkt_rms_12"] = long["date"].map(mkt_12)
    long["vix"] = long["date"].map(vix_prev)
    y = long["signal"].map(pub_year)
    long["age"] = (long["date"].dt.year - (y + 1)) * 12 + long["date"].dt.month - 1
    long = long[(long["date"].dt.year > y) & (long["date"] >= start)]
    long = long.dropna(subset=["abs_r"] + FEATURES)
    long["vol_a"] = long["rms_12"]  # H9b 12-month rule
    long["vol_b"] = long["rms_all"]  # H9b long-run rule
    return long.sort_values(["date", "signal"]).reset_index(drop=True)


# ---------- models ----------

def design(df: pd.DataFrame) -> np.ndarray:
    X = np.log(df[VOL_FEATURES].to_numpy(float) + EPS)
    return np.column_stack([X, df["age"].to_numpy(float)])


def h9_loss(abs_r: np.ndarray, vol: np.ndarray) -> np.ndarray:
    return np.abs(abs_r - ABS_SCALE * vol)


def best_scale(e: np.ndarray, abs_r: np.ndarray) -> float:
    """c minimising sum |abs_r - ABS_SCALE * c * e|: a weighted median of abs_r / (ABS_SCALE * e)."""
    w = ABS_SCALE * e
    ratio = abs_r / w
    order = np.argsort(ratio)
    cum = np.cumsum(w[order])
    return float(ratio[order][np.searchsorted(cum, cum[-1] / 2)])


def fit_ridge(df: pd.DataFrame, alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(design(df), np.log(df["abs_r"] + EPS))


def ridge_vol(model, df: pd.DataFrame, c: float) -> np.ndarray:
    return np.exp(model.predict(design(df))) * c


def fit_hgb(df: pd.DataFrame, params: dict):
    return HistGradientBoostingRegressor(loss="absolute_error", random_state=SEED, **params).fit(
        design(df), df["abs_r"] / ABS_SCALE)


def choose_settings(train: pd.DataFrame, test_year: int) -> dict:
    """Ridge alpha and c, and boosting settings, from time-based validation inside the training years."""
    n_years = test_year - 1 - SAMPLE_START.year  # training years: 1990 .. test_year - 2
    if n_years < MIN_TUNE_YEARS:
        return {"tuned": False, "alpha": RIDGE_DEFAULT_ALPHA, "c": DEFAULT_C, "hgb": HGB_DEFAULT}
    yrs = train["date"].dt.year
    val_first = test_year - 1 - VAL_YEARS
    val = train[yrs >= val_first]
    inner = train[yrs < val_first - 1]  # one-year gap before validation
    a_val = val["abs_r"].to_numpy()

    best = None
    for alpha in RIDGE_ALPHAS:
        m = fit_ridge(inner, alpha)
        e = np.exp(m.predict(design(val)))
        c = best_scale(e, a_val)
        loss = h9_loss(a_val, c * e).mean()
        if best is None or loss < best[0]:
            best = (loss, alpha, c)
    hgb_best = min(HGB_GRID, key=lambda p: h9_loss(a_val, fit_hgb(inner, p).predict(design(val))).mean())
    return {"tuned": True, "alpha": float(best[1]), "c": float(best[2]), "hgb": hgb_best}


def walk_forward(df: pd.DataFrame, test_years: list[int] = TEST_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding walk-forward by calendar year with a 12-month gap. Returns (predictions, folds)."""
    preds, folds = [], []
    years = df["date"].dt.year
    for y in test_years:
        train, test = df[years <= y - 2], df[years == y]
        if train.empty or test.empty:
            continue
        s = choose_settings(train, y)
        ridge = fit_ridge(train, s["alpha"])
        hgb = fit_hgb(train, s["hgb"])
        out = test[["date", "signal", "abs_r", "vol_a", "vol_b"]].copy()
        out["vol_ridge"] = ridge_vol(ridge, test, s["c"])
        out["vol_hgb"] = hgb.predict(design(test))
        # Scale-corrected baselines (added before any real-data run; see HYPOTHESES.md):
        # one constant per rule, fitted on this fold's training rows only.
        ta = train["abs_r"].to_numpy()
        c_a, c_b = best_scale(train["vol_a"].to_numpy(), ta), best_scale(train["vol_b"].to_numpy(), ta)
        out["vol_a_scaled"], out["vol_b_scaled"] = out["vol_a"] * c_a, out["vol_b"] * c_b
        out["fold"] = y
        preds.append(out)
        a = out["abs_r"].to_numpy()
        folds.append({"test_year": y, "train_first": int(train["date"].dt.year.min()),
                      "train_last": int(train["date"].dt.year.max()), "n_train": len(train), "n_test": len(test),
                      "tuned": s["tuned"], "ridge_alpha": s["alpha"], "ridge_c": s["c"],
                      "scale_a": c_a, "scale_b": c_b,
                      **{f"hgb_{k}": v for k, v in s["hgb"].items()},
                      **{f"loss_{m}": h9_loss(a, out[f"vol_{m}"].to_numpy()).mean()
                         for m in ["a", "b", "a_scaled", "b_scaled", "ridge", "hgb"]}})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(folds)


# ---------- evaluation ----------

def signal_bootstrap_ci(diff: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> tuple[float, float]:
    """95% CI of the mean monthly cross-signal loss difference, resampling signals."""
    D = diff.pivot(index="date", columns="signal", values="d")
    M = D.notna().to_numpy(float)
    Dv = D.fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    n = D.shape[1]
    out = []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        den = M @ w
        ok = den > 0
        out.append(((Dv @ w)[ok] / den[ok]).mean())
    return tuple(np.percentile(out, [2.5, 97.5]))


def compare(pred: pd.DataFrame, model: str, base: str, period: tuple[int, int], ci: bool = True) -> dict:
    p = pred[pred["date"].dt.year.between(*period)]
    a = p["abs_r"].to_numpy()
    d = h9_loss(a, p[f"vol_{model}"].to_numpy()) - h9_loss(a, p[f"vol_{base}"].to_numpy())
    diff = pd.DataFrame({"date": p["date"], "signal": p["signal"], "d": d})
    monthly = diff.groupby("date")["d"].mean()
    res = hac_mean_test(-monthly)  # mean(-d) > 0  <=>  model loss lower
    out = {"model": model, "vs": f"baseline_{base}", "period": f"{period[0]}-{period[1]}",
           "mean_loss_diff": -res["mean"], "t_stat": res["t_stat"], "p_one_sided": res["p_one_sided"],
           "n_months": res["n_months"], "n_rows": len(p), "n_signals": p["signal"].nunique(),
           "mae_model": h9_loss(a, p[f"vol_{model}"].to_numpy()).mean(),
           "mae_base": h9_loss(a, p[f"vol_{base}"].to_numpy()).mean(),
           "spearman_model": stats.spearmanr(p[f"vol_{model}"], a).statistic,
           "spearman_base": stats.spearmanr(p[f"vol_{base}"], a).statistic}
    out["ci_low"], out["ci_high"] = signal_bootstrap_ci(diff) if ci else (np.nan, np.nan)
    return out


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


# ---------- decision check ----------

def portfolio_weights(pred: pd.DataFrame, vol_col: str | None, cap: float = WEIGHT_CAP) -> pd.DataFrame:
    """Monthly weights (date x signal). vol_col None = plain equal weight.

    Raw weight 1/vol^2, capped at cap x the average raw weight, scaled so average gross
    exposure over the period is 1 (the scale does not change any Sharpe ratio).
    """
    n = pred.groupby("date")["signal"].transform("size")
    if vol_col is None:
        w = 1.0 / n
    else:
        raw = 1.0 / pred[vol_col] ** 2
        raw = np.minimum(raw, cap * raw.mean())
        w = raw / n
        w = w / w.groupby(pred["date"]).sum().mean()
    return pd.DataFrame({"date": pred["date"], "signal": pred["signal"], "w": w}).pivot(
        index="date", columns="signal", values="w").fillna(0.0)


def portfolio_returns(pred: pd.DataFrame, W: pd.DataFrame, cost_bps: float) -> tuple[pd.Series, pd.Series]:
    """Net monthly return and one-way turnover. pred needs a signed `ret` column (month-t return)."""
    r = pred.pivot(index="date", columns="signal", values="ret").reindex_like(W).fillna(0.0)
    gross = (W * r).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = np.nan
    net = gross - cost_bps / 1e4 * turnover.fillna(0.0)
    return net, turnover


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def break_even_bps(pred: pd.DataFrame, W_model: pd.DataFrame, W_base: pd.DataFrame, hi: float = 500.0) -> float:
    """Cost (bps per unit turnover) at which the model's net Sharpe equals the base's."""
    f = lambda c: sharpe(portfolio_returns(pred, W_model, c)[0]) - sharpe(portfolio_returns(pred, W_base, c)[0])
    if f(0.0) <= 0:
        return 0.0
    if f(hi) > 0:
        return np.inf
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def decision_check(pred: pd.DataFrame, period: tuple[int, int] = PRIMARY) -> pd.DataFrame:
    p = pred[pred["date"].dt.year.between(*period)]
    versions = {"equal_weight": None, "h9b_rule": "vol_a", "ridge": "vol_ridge", "hgb": "vol_hgb"}
    W = {k: portfolio_weights(p, v) for k, v in versions.items()}
    rows = []
    for k in versions:
        for c in COSTS_BPS:
            net, turnover = portfolio_returns(p, W[k], c)
            rows.append({"portfolio": k, "period": f"{period[0]}-{period[1]}", "cost_bps": c,
                         "sharpe": sharpe(net), "mean_monthly_return": net.mean(),
                         "avg_turnover": turnover.mean(), "max_drawdown": max_drawdown(net),
                         "break_even_bps_vs_h9b": break_even_bps(p, W[k], W["h9b_rule"])
                         if k in ("ridge", "hgb") and c == 0 else np.nan})
    return pd.DataFrame(rows)


def plot_h10(pred: pd.DataFrame, port_inputs: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = pred[pred["date"].dt.year.between(*PRIMARY)]
    a = p["abs_r"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in ["ridge", "hgb", "a_scaled", "b"]:
        d = h9_loss(a, p[f"vol_{m}"].to_numpy()) - h9_loss(a, p["vol_a"].to_numpy())
        axes[0].plot(pd.Series(d, index=p["date"]).groupby(level=0).mean().cumsum(), label=f"{m} - H9b rule")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("H10: cumulative loss difference vs H9b 12-month rule\n(below zero = better than the rule)")
    for k, v in {"h9b_rule": "vol_a", "ridge": "vol_ridge", "hgb": "vol_hgb", "equal_weight": None}.items():
        net, _ = portfolio_returns(port_inputs, portfolio_weights(port_inputs, v), 10)
        axes[1].plot((1 + net).cumprod(), label=f"{k} (Sharpe {sharpe(net):.2f})")
    axes[1].set_yscale("log")
    axes[1].set_title("Volatility-managed portfolios, 10 bps illustrative cost")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H10: trained volatility models vs the H9b rule.")
    parser.add_argument("--charts", action="store_true")
    parser.add_argument("--refresh-vix", action="store_true")
    args = parser.parse_args(argv)

    pt = predictor_table(load_signal_doc())
    returns = load_ls_returns()
    returns = returns[returns["signal"].isin(pt["signal"])]
    kf = contracts.load("factors")
    market = kf[kf["signal"] == "Mkt-RF"].set_index("date")["return"]
    df = build_features(returns, pt.set_index("signal")["year"], market, load_vix(refresh=args.refresh_vix))
    pred, folds = walk_forward(df)
    pred = pred.merge(returns.rename(columns={"return": "ret"}), on=["date", "signal"], how="left")

    rows = []
    for period in (PRIMARY, SECONDARY):
        for model in ("ridge", "hgb"):
            for base in ("a", "b", "a_scaled", "b_scaled"):
                rows.append(compare(pred, model, base, period))
        for base in ("a", "b"):
            rows.append(compare(pred, f"{base}_scaled", base, period))
    summary = pd.DataFrame(rows)
    # Only the two pre-registered tests (ridge, boosting vs unscaled baseline A, 2000-2025) enter Holm.
    prim = ((summary["period"] == f"{PRIMARY[0]}-{PRIMARY[1]}") & (summary["vs"] == "baseline_a")
            & summary["model"].isin(["ridge", "hgb"]))
    summary["role"] = np.where(prim & (summary["model"] == "ridge"), "primary",
                               np.where(prim, "secondary", "descriptive"))
    summary["p_holm"] = np.nan
    summary.loc[prim, "p_holm"] = holm(summary.loc[prim, "p_one_sided"].tolist())
    summary["verdict"] = np.where(prim, np.where(summary["p_holm"] < 0.05, "beats H9b rule",
                                                 "does not beat H9b rule"), "")
    summary.to_csv(SUMMARY_CSV, index=False)
    folds.to_csv(FOLDS_CSV, index=False)
    decision_check(pred).to_csv(PORTFOLIO_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}, {FOLDS_CSV}, {PORTFOLIO_CSV}")
    if args.charts:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_h10(pred, pred[pred["date"].dt.year.between(*PRIMARY)], CHARTS_DIR / "osap_h10_vol_model.png")
        print(f"Saved chart to {CHARTS_DIR / 'osap_h10_vol_model.png'}")


if __name__ == "__main__":
    main()
