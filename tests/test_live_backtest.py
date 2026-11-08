import numpy as np
import pandas as pd
import pytest

from src.common import contracts
from src.live import backtest as bt


def make_data(n_months=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1950-01-31", periods=n_months, freq="ME")
    returns = pd.DataFrame({
        "Value": 0.004 + 0.02 * rng.standard_normal(n_months),
        "Momentum": 0.006 + 0.03 * rng.standard_normal(n_months),
    }, index=idx)
    market = pd.Series(0.005 + 0.04 * rng.standard_normal(n_months), index=idx)
    pub = pd.Series({"Value": idx[150], "Momentum": idx[200]})
    return returns, market, pub


def test_unpublished_and_short_history_signals_are_excluded():
    returns, _, pub = make_data()
    assert bt.expected_edges(returns, pub, returns.index[149]).empty
    est = bt.expected_edges(returns, pub, returns.index[160])
    assert list(est.index) == ["Value"]
    pub_early = pub.copy()
    pub_early["Value"] = returns.index[bt.MIN_HISTORY_MONTHS - 1]
    assert bt.expected_edges(returns, pub_early, returns.index[160]).index.tolist() == []


def test_no_look_ahead():
    returns, market, pub = make_data()
    m = returns.index[220]
    before = bt.expected_edges(returns, pub, m)
    future = returns.copy()
    future.loc[future.index > m] = 9.99
    pd.testing.assert_frame_equal(before, bt.expected_edges(future, pub, m))
    # the month after m uses weights from m, so only month m+1 onward changes
    a, _ = bt.run_backtest(returns, market, pub)
    b, _ = bt.run_backtest(future, market, pub)
    pd.testing.assert_frame_equal(a.loc[:m], b.loc[:m])


def test_edge_starts_at_keep_of_paper_edge():
    returns, _, pub = make_data()
    m = pub["Value"]  # publication month: no post months observed beyond it yet
    est = bt.expected_edges(returns, pub, m - pd.offsets.MonthEnd(0))
    pre = returns["Value"][returns.index < pub["Value"]].mean()
    post = returns["Value"][returns.index >= pub["Value"]].loc[:m]
    expected = (post.sum() + bt.PRIOR_MONTHS * bt.KEEP * pre) / (len(post) + bt.PRIOR_MONTHS)
    assert est.at["Value", "edge"] == pytest.approx(expected)
    assert est.at["Value", "net_edge"] == pytest.approx(expected - bt.COST["Value"])


def test_abstention_goes_to_cash():
    est = pd.DataFrame({"pre_mean": [0.01, 0.01], "edge": [0.0, 0.0],
                        "net_edge": [-0.001, 0.0005], "var": [0.001, 0.001]},
                       index=["Value", "Momentum"])
    w = bt.weights_at(est)
    assert w["decay_tilt"].sum() == 0
    assert w["equal_weight"].tolist() == [0.5, 0.5]
    assert w["naive_insample"].sum() == pytest.approx(1)


def test_costs_lower_net_return_and_contract_compounds():
    returns, market, pub = make_data()
    monthly, weights = bt.run_backtest(returns, market, pub)
    assert (monthly["equal_weight"] < monthly["equal_weight_gross"]).all()
    assert set(weights["signal"]) <= {"Value", "Momentum"}
    long = bt.to_contract(monthly)
    contracts.validate("backtest_results", long)
    ew = long[long["strategy"] == "equal_weight"]
    assert ew["cumulative_return"].iloc[-1] == pytest.approx(
        (1 + monthly["equal_weight"]).prod() - 1, abs=1e-5)


def test_summary_paired_difference_is_zero_for_baseline():
    returns, market, pub = make_data()
    monthly, _ = bt.run_backtest(returns, market, pub)
    s = bt.summarise(monthly, reps=200).set_index("strategy")
    assert s.loc["equal_weight", ["diff_ci_low", "diff_ci_high"]].tolist() == [0, 0]
    assert s.loc["market", "sharpe_ci_low"] <= s.loc["market", "sharpe"] <= s.loc["market", "sharpe_ci_high"]


def test_block_bootstrap_shape_and_blocks():
    idx = bt.block_bootstrap_index(25, 3, 12, np.random.default_rng(0))
    assert idx.shape == (3, 25)
    assert ((np.diff(idx[:, :12], axis=1) % 25) == 1).all()
