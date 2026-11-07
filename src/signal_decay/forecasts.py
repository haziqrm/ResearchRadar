"""H9: month-ahead forecasts for published signals (factor momentum and volatility).

Rules are fixed in HYPOTHESES.md (H9). Every forecast for month t uses returns up
to month t-1 only, and only post-publication months are scored.

Usage:
    python -m src.signal_decay.forecasts [--charts]

Writes A-owned files only:
    data/osap_h9_summary.csv      H9a and H9b tests plus descriptive portfolio stats
    data/signal_forecasts.csv     latest forecast per signal (OSAP and Ken French), for D and C

For D's backtest, `forecast_panel(returns)` returns the full month-by-month history
(date, signal, return, mom_12m, vol_forecast, vol_baseline) without writing a file.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.common import contracts
from src.signal_decay import CHARTS_DIR, DATA_DIR
from src.signal_decay.osap import load_ls_returns, load_signal_doc, predictor_table

WINDOW = 12
BASELINE_MIN_MONTHS = 36
MIN_SIGNALS = 10
HAC_LAGS = 12
ABS_SCALE = np.sqrt(2 / np.pi)  # E|r| = sqrt(2/pi) * sigma for a normal with mean zero

SUMMARY_CSV = DATA_DIR / "osap_h9_summary.csv"
FORECASTS_CSV = DATA_DIR / "signal_forecasts.csv"


# ---------- forecasts (t uses data up to t-1) ----------

def forecast_panel(returns: pd.DataFrame) -> pd.DataFrame:
    """Month-by-month forecasts per signal from tidy (date, signal, return).

    mom_12m       mean return over t-12..t-1 (all 12 months required)
    vol_forecast  sqrt(mean squared return over t-12..t-1)
    vol_baseline  sqrt(mean squared return over all months before t), at least 36
    """
    returns = returns.assign(date=pd.to_datetime(returns["date"]))
    wide = returns.pivot(index="date", columns="signal", values="return").sort_index()
    sq = wide**2
    out = {
        "return": wide,
        "mom_12m": wide.rolling(WINDOW, min_periods=WINDOW).mean().shift(1),
        "vol_forecast": np.sqrt(sq.rolling(WINDOW, min_periods=WINDOW).mean().shift(1)),
        "vol_baseline": np.sqrt(sq.expanding(min_periods=BASELINE_MIN_MONTHS).mean().shift(1)),
    }
    panel = pd.concat({k: v.stack(dropna=False) for k, v in out.items()}, axis=1)
    panel.index.names = ["date", "signal"]
    return panel.reset_index().dropna(subset=["return"])


def post_publication(panel: pd.DataFrame, pub_year: pd.Series) -> pd.DataFrame:
    """Keep months after each signal's publication year (pub_year indexed by signal)."""
    y = panel["signal"].map(pub_year)
    return panel[panel["date"].dt.year > y].copy()


# ---------- tests ----------

def hac_mean_test(x: pd.Series, lags: int = HAC_LAGS) -> dict:
    """Mean of a monthly series with a Newey-West t and a one-sided p for mean > 0."""
    x = x.dropna()
    fit = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = float(fit.tvalues[0])
    return {"n_months": len(x), "mean": float(fit.params[0]), "t_stat": t,
            "p_one_sided": float(stats.norm.sf(t))}


def momentum_test(panel: pd.DataFrame, min_signals: int = MIN_SIGNALS) -> tuple[dict, pd.DataFrame]:
    """H9a: monthly spread between past winners (mom_12m > 0) and the rest."""
    p = panel.dropna(subset=["mom_12m"])
    p = p.assign(winner=p["mom_12m"] > 0)
    g = p.groupby("date")
    monthly = pd.DataFrame({
        "n": g.size(),
        "n_winners": g["winner"].sum(),
        "all": g["return"].mean(),
        "winners": p[p["winner"]].groupby("date")["return"].mean(),
        "losers": p[~p["winner"]].groupby("date")["return"].mean(),
    })
    monthly = monthly[(monthly["n"] >= min_signals) & monthly["winners"].notna() & monthly["losers"].notna()]
    monthly["spread"] = monthly["winners"] - monthly["losers"]
    res = hac_mean_test(monthly["spread"])
    res["share_positive"] = float((monthly["spread"] > 0).mean()) if len(monthly) else np.nan
    return res, monthly


def volatility_test(panel: pd.DataFrame) -> tuple[dict, pd.Series]:
    """H9b: monthly mean of (model loss - baseline loss); negative means the model is better."""
    p = panel.dropna(subset=["vol_forecast", "vol_baseline"])
    a = p["return"].abs()
    diff = (a - ABS_SCALE * p["vol_forecast"]).abs() - (a - ABS_SCALE * p["vol_baseline"]).abs()
    monthly = diff.groupby(p["date"]).mean()
    res = hac_mean_test(-monthly)  # test mean(-diff) > 0, i.e. model loss lower
    res["mean_loss_diff"] = -res.pop("mean")
    res["corr_model"] = float(np.corrcoef(a, p["vol_forecast"])[0, 1])
    res["corr_baseline"] = float(np.corrcoef(a, p["vol_baseline"])[0, 1])
    return res, monthly


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(np.sqrt(12) * x.mean() / x.std()) if len(x) > 1 and x.std() > 0 else np.nan


def vol_managed(panel: pd.DataFrame, min_signals: int = MIN_SIGNALS) -> pd.DataFrame:
    """Equal-weight vs 1/variance-weighted portfolios of published signals (descriptive)."""
    p = panel.dropna(subset=["vol_forecast"])
    p = p[p["vol_forecast"] > 0]
    g = p.groupby("date")
    plain = g["return"].mean()
    w = 1 / p["vol_forecast"] ** 2
    managed = (p["return"] * w).groupby(p["date"]).sum() / g.size()
    keep = g.size() >= min_signals
    plain, managed = plain[keep], managed[keep]
    managed = managed * plain.std() / managed.std()  # same full-sample volatility; Sharpe unchanged
    return pd.DataFrame({"equal_weight": plain, "vol_managed": managed})


# ---------- live snapshot ----------

def latest_forecasts(returns: pd.DataFrame, source: str, names: dict[str, str] | None = None) -> pd.DataFrame:
    """Forecast for the month after the last return, per signal."""
    returns = returns.assign(date=pd.to_datetime(returns["date"]))
    wide = returns.pivot(index="date", columns="signal", values="return").sort_index()
    last = wide.index.max()
    rows = []
    for sig in wide.columns:
        r = wide[sig].dropna()
        if len(r) < WINDOW or r.index.max() != last:
            continue
        recent = r.iloc[-WINDOW:]
        rows.append({
            "source": source, "signal": sig, "name": (names or {}).get(sig, sig),
            "as_of": last.date().isoformat(),
            "mom_12m": recent.mean(), "momentum_on": bool(recent.mean() > 0),
            "vol_forecast": float(np.sqrt((recent**2).mean())),
            "vol_baseline": float(np.sqrt((r**2).mean())) if len(r) >= BASELINE_MIN_MONTHS else np.nan,
        })
    return pd.DataFrame(rows)


def plot_h9(monthly_mom: pd.DataFrame, port: pd.DataFrame, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for col in ["all", "winners", "losers"]:
        axes[0].plot((1 + monthly_mom[col]).cumprod(), label=col)
    axes[0].set_yscale("log")
    axes[0].set_title("H9a: published signals split by past-12-month return")
    for col in port.columns:
        axes[1].plot((1 + port[col]).cumprod(), label=f"{col} (Sharpe {sharpe(port[col]):.2f})")
    axes[1].set_yscale("log")
    axes[1].set_title("H9b: equal-weight vs volatility-managed (before costs)")
    for ax in axes:
        ax.set_ylabel("Growth of $1 (log scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="H9: factor momentum and volatility forecasts.")
    parser.add_argument("--charts", action="store_true")
    args = parser.parse_args(argv)

    pt = predictor_table(load_signal_doc())
    returns = load_ls_returns()
    returns = returns[returns["signal"].isin(pt["signal"])]
    panel = post_publication(forecast_panel(returns), pt.set_index("signal")["year"])

    mom, monthly_mom = momentum_test(panel)
    vol, _ = volatility_test(panel)
    port = vol_managed(panel)
    rows = [
        {"test": "H9a factor momentum", "statistic": "winners - losers, mean monthly return",
         "value": mom["mean"], "t_stat": mom["t_stat"], "p_one_sided": mom["p_one_sided"],
         "n_months": mom["n_months"], "note": f"spread positive in {mom['share_positive']:.0%} of months"},
        {"test": "H9b volatility", "statistic": "model - baseline absolute error, mean",
         "value": vol["mean_loss_diff"], "t_stat": vol["t_stat"], "p_one_sided": vol["p_one_sided"],
         "n_months": vol["n_months"],
         "note": f"corr with |r|: 12-month {vol['corr_model']:.2f}, long-run {vol['corr_baseline']:.2f}"},
    ]
    for col in ["all", "winners", "losers"]:
        rows.append({"test": "H9a descriptive", "statistic": f"{col}: annualised Sharpe",
                     "value": sharpe(monthly_mom[col]), "n_months": len(monthly_mom),
                     "note": f"mean monthly return {monthly_mom[col].mean():.4f}"})
    for col in port.columns:
        rows.append({"test": "H9b descriptive", "statistic": f"{col}: annualised Sharpe",
                     "value": sharpe(port[col]), "n_months": len(port), "note": "before trading costs"})
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    kf = contracts.load("factors")
    names = {}
    try:
        papers = contracts.load("papers")
        names = dict(zip(papers["ken_french_column"], papers["signal"]))
        kf = kf[kf["signal"].isin(names)]
    except FileNotFoundError:
        pass
    snapshot = pd.concat([latest_forecasts(returns, "osap"), latest_forecasts(kf, "ken_french", names)],
                         ignore_index=True)
    snapshot.to_csv(FORECASTS_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV} and {FORECASTS_CSV} ({len(snapshot)} signals)")
    if args.charts:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_h9(monthly_mom, port, CHARTS_DIR / "osap_h9_forecasts.png")
        print(f"Saved chart to {CHARTS_DIR / 'osap_h9_forecasts.png'}")


if __name__ == "__main__":
    main()
