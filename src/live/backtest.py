"""Walk-forward allocation backtest across the published Ken French signals.

Usage:
    python -m src.live.backtest [--boot 2000]

The question: an investor who reads the journals and trades only signals that have
already been published, should they believe the paper's numbers (face value), spread
evenly, or expect the edge to decay and tilt accordingly?

Each month-end m we decide weights for month m+1 using only returns dated <= m:
    universe   signals whose paper was published by m (papers.csv publication_date)
    pre_mean   mean return before publication (fully known once the paper is out)
    post_mean  mean of the post-publication months seen so far (n_post of them)
    edge       (n_post * post_mean + PRIOR_MONTHS * KEEP * pre_mean) / (n_post + PRIOR_MONTHS)
               i.e. start at KEEP of the paper's edge, then let the signal's own
               post-publication record take over as it builds up
    net edge   edge - COST[signal]  (rough monthly trading cost per unit held)

Strategies (all are long-short excess returns, weights sum to 1 or the book is cash):
    decay_tilt       weight ~ net edge / variance; ABSTAIN from a signal whose net edge
                     is under MIN_EDGE, and hold cash if every signal is abstained
    naive_insample   weight ~ pre_mean / variance: believes the paper, ignores decay
    equal_weight     1/N over published signals
    market           Mkt-RF, the market's excess return (no cost charged)
Each factor strategy is reported after costs, and again before costs with a
"_gross" suffix.

decay_results.csv is deliberately not used: its post_mean averages the whole future,
so trading on it would be look-ahead. Everything here is rebuilt month by month from
factors.csv and papers.csv.

Intervals: Sharpe ratios get a 95% circular block bootstrap interval (12-month
blocks), and each strategy's Sharpe minus equal_weight's (same cost treatment)
uses the same resampled months, so the difference interval is paired.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import contracts
from src.common.contracts import DATA_DIR

KEEP = 0.5            # prior share of the paper's edge that survives publication
PRIOR_MONTHS = 60     # how many post-publication months the prior is worth
MIN_EDGE = 0.001      # abstain below 0.1% a month expected edge after costs
MIN_HISTORY_MONTHS = 120  # a signal needs 10 years before publication to be traded
BLOCK_MONTHS = 12
SEED = 20261107

# Rough monthly cost of holding 1 unit of each long-short factor (its own
# rebalancing). Ballpark of Novy-Marx and Velikov (2016): low-turnover sorts cost a
# few basis points a month, momentum more, short-term reversal most of its edge.
COST: dict[str, float] = {
    "Size": 0.0005,
    "Value": 0.0005,
    "Profitability": 0.0005,
    "Investment": 0.0005,
    "Long-term reversal": 0.0010,
    "Momentum": 0.0020,
    "Short-term reversal": 0.0060,
}
MARKET_COLUMN = "Mkt-RF"

SUMMARY_CSV = DATA_DIR / "backtest_summary.csv"
WEIGHTS_CSV = DATA_DIR / "backtest_weights.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(signal returns wide by month-end, market returns, publication dates by signal)."""
    factors = contracts.load("factors")
    papers = contracts.load("papers")
    factors["date"] = pd.to_datetime(factors["date"])
    wide = factors.pivot(index="date", columns="signal", values="return").sort_index()
    column = papers.set_index("signal")["ken_french_column"]
    missing = sorted(set(column) - set(wide.columns))
    if missing:
        raise ValueError(f"factors.csv has no columns {missing} named in papers.csv")
    returns = wide[column.to_list()].set_axis(column.index.to_list(), axis=1)
    pub = pd.to_datetime(papers.set_index("signal")["publication_date"])
    return returns, wide[MARKET_COLUMN], pub


def expected_edges(returns: pd.DataFrame, pub: pd.Series, m: pd.Timestamp,
                   keep: float = KEEP) -> pd.DataFrame:
    """Per published signal at month-end m: pre_mean, edge, net_edge, var. No data after m."""
    rows = {}
    for s in returns.columns:
        if pub[s] > m:
            continue
        r = returns[s].loc[:m].dropna()
        pre = r[r.index < pub[s]]
        if len(pre) < MIN_HISTORY_MONTHS:
            continue
        post = r[r.index >= pub[s]]
        n = len(post)
        edge = (post.sum() + PRIOR_MONTHS * keep * pre.mean()) / (n + PRIOR_MONTHS)
        rows[s] = {"pre_mean": pre.mean(), "edge": edge,
                   "net_edge": edge - COST.get(s, 0.0), "var": r.var()}
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["pre_mean", "edge", "net_edge", "var"])


def _normalise(raw: pd.Series) -> pd.Series:
    raw = raw.clip(lower=0)
    return raw / raw.sum() if raw.sum() > 0 else raw * 0.0


def weights_at(est: pd.DataFrame) -> dict[str, pd.Series]:
    """Weights per factor strategy from one month's estimates (index = signals)."""
    if est.empty:
        return {k: pd.Series(dtype=float) for k in ("decay_tilt", "naive_insample", "equal_weight")}
    tilt = est["net_edge"].where(est["net_edge"] >= MIN_EDGE, 0.0) / est["var"]
    return {
        "decay_tilt": _normalise(tilt),
        "naive_insample": _normalise(est["pre_mean"] / est["var"]),
        "equal_weight": pd.Series(1.0 / len(est), index=est.index),
    }


def run_backtest(returns: pd.DataFrame, market: pd.Series, pub: pd.Series,
                 keep: float = KEEP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly returns wide by strategy, decay_tilt weights long)."""
    months = returns.index
    cost = pd.Series(COST).reindex(returns.columns).fillna(0.0)
    out: dict[pd.Timestamp, dict[str, float]] = {}
    weight_rows = []
    for m, nxt in zip(months[:-1], months[1:]):
        est = expected_edges(returns, pub, m, keep)
        if est.empty:
            continue  # nothing published yet
        r_next = returns.loc[nxt]
        row = {}
        for name, w in weights_at(est).items():
            gross = float((w * r_next[w.index]).sum())
            row[name + "_gross"] = gross
            row[name] = gross - float((w * cost[w.index]).sum())
        row["market"] = float(market.loc[nxt])
        out[nxt] = row
        w = weights_at(est)["decay_tilt"]
        for s in est.index:
            weight_rows.append({"date": nxt.strftime("%Y-%m-%d"), "signal": s,
                                "weight": round(float(w.get(s, 0.0)), 4),
                                "expected_edge_net": round(float(est.at[s, "net_edge"]), 5)})
    monthly = pd.DataFrame.from_dict(out, orient="index").sort_index()
    return monthly, pd.DataFrame(weight_rows)


def to_contract(monthly: pd.DataFrame) -> pd.DataFrame:
    """Long backtest_results table; cumulative_return compounds monthly returns."""
    cumulative = (1 + monthly).cumprod() - 1
    long = pd.DataFrame({
        "date": np.tile(monthly.index.strftime("%Y-%m-%d"), monthly.shape[1]),
        "strategy": np.repeat(monthly.columns, len(monthly)),
        "monthly_return": monthly.to_numpy().T.ravel().round(6),
        "cumulative_return": cumulative.to_numpy().T.ravel().round(6),
    })
    return long


def sharpe(r: np.ndarray, axis: int = 0) -> np.ndarray:
    sd = r.std(axis=axis, ddof=1)
    return np.where(sd > 0, r.mean(axis=axis) / np.where(sd > 0, sd, 1) * np.sqrt(12), 0.0)


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def block_bootstrap_index(n: int, reps: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """reps x n month indices from a circular block bootstrap."""
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return idx.reshape(reps, -1)[:, :n]


def summarise(monthly: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = block_bootstrap_index(len(monthly), reps, BLOCK_MONTHS, rng)
    boot = {c: sharpe(monthly[c].to_numpy()[idx], axis=1) for c in monthly.columns}
    rows = []
    for c in monthly.columns:
        r = monthly[c]
        base = "equal_weight_gross" if c.endswith("_gross") else "equal_weight"
        diff = boot[c] - boot[base]
        years = len(r) / 12
        rows.append({
            "strategy": c,
            "sharpe": float(sharpe(r.to_numpy())),
            "sharpe_ci_low": float(np.percentile(boot[c], 2.5)),
            "sharpe_ci_high": float(np.percentile(boot[c], 97.5)),
            "ann_return": float((1 + r).prod() ** (1 / years) - 1),
            "ann_vol": float(r.std(ddof=1) * np.sqrt(12)),
            "max_drawdown": max_drawdown(r),
            "n_months": len(r),
            "sharpe_minus_equal_weight": float(sharpe(r.to_numpy()) - sharpe(monthly[base].to_numpy())),
            "diff_ci_low": float(np.percentile(diff, 2.5)),
            "diff_ci_high": float(np.percentile(diff, 97.5)),
        })
    return pd.DataFrame(rows).round(4)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Walk-forward allocation backtest.")
    parser.add_argument("--boot", type=int, default=2000, help="bootstrap repetitions")
    args = parser.parse_args(argv)

    returns, market, pub = load_inputs()
    monthly, weights = run_backtest(returns, market, pub)
    path = contracts.save("backtest_results", to_contract(monthly))
    summary = summarise(monthly, args.boot)
    summary.to_csv(SUMMARY_CSV, index=False)
    weights.to_csv(WEIGHTS_CSV, index=False)

    print(f"wrote {path} ({monthly.index[0]:%Y-%m} to {monthly.index[-1]:%Y-%m}, "
          f"{len(monthly)} months), {SUMMARY_CSV.name}, {WEIGHTS_CSV.name}")
    print(f"{'strategy':22s} {'sharpe':>7s} {'95% CI':>15s} {'ann ret':>8s} "
          f"{'max dd':>7s} {'vs EW':>7s} {'95% CI':>15s}")
    for r in summary.itertuples():
        print(f"{r.strategy:22s} {r.sharpe:7.2f} [{r.sharpe_ci_low:5.2f}, {r.sharpe_ci_high:5.2f}] "
              f"{r.ann_return:8.2%} {r.max_drawdown:7.1%} {r.sharpe_minus_equal_weight:+7.2f} "
              f"[{r.diff_ci_low:+5.2f}, {r.diff_ci_high:+5.2f}]")
    latest = weights[weights["date"] == weights["date"].max()]
    print(f"decay_tilt weights for {latest['date'].iloc[0]}:")
    for r in latest.sort_values("weight", ascending=False).itertuples():
        print(f"  {r.weight:6.1%}  net edge {r.expected_edge_net:+.2%}/month  {r.signal}")
    return path


if __name__ == "__main__":
    main()
