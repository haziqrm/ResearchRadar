"""Tests for the signal-decay module. None of these touch the network."""

import importlib
import json
import socket
import sys

import numpy as np
import pandas as pd
import pytest

from src.common import contracts

MODULES = [
    "src.signal_decay",
    "src.signal_decay.ken_french",
    "src.signal_decay.osap",
    "src.signal_decay.papers",
    "src.signal_decay.decay",
    "src.signal_decay.attention",
    "src.signal_decay.plots",
    "src.signal_decay.run_decay",
]

PAPER_COLS = contracts.CONTRACTS["papers"][1]


def _papers(rows):
    return pd.DataFrame(rows, columns=PAPER_COLS)


def _monthly(values, start="1960-01-31"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="ME"))


@pytest.fixture
def no_network(monkeypatch):
    """Make any socket connection attempt fail loudly."""

    def _blocked(*args, **kwargs):
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


# ---------- imports ----------

@pytest.mark.parametrize("name", MODULES)
def test_module_imports_without_network(name, no_network, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    sys.modules.pop(name, None)  # force a fresh import under the network block
    assert importlib.import_module(name) is not None


# ---------- (a) Ken French parsing, reversal and 3-factor series ----------

KF_3F_TEXT = (
    "This file was created by CMPT_ME_BEME_RETS using the 202609 CRSP database.\n"
    "\n"
    ",Mkt-RF,SMB,HML,RF\n"
    "192607,    2.89,   -2.55,   -2.39,    0.22\n"
    "192608,    2.64,   -1.14,    3.81,    0.25\n"
    "\n"
    " Annual Factors: January-December \n"
    ",Mkt-RF,SMB,HML,RF\n"
    "1927,   29.47,   -2.46,   -3.75,    3.12\n"
    "\n"
    "Copyright 2026 Kenneth R. French\n"
)

KF_ST_TEXT = (
    "This file was created using the 202609 CRSP database.\n"
    "Missing data are indicated by -99.99.\n"
    "\n"
    ",ST_Rev\n"
    "192602,   -0.11\n"
    "192603,  -99.99\n"
    "192604,    1.20\n"
    "\n"
    "Annual Factors: January-December\n"
)


def test_parse_skips_header_annual_and_footer():
    from src.signal_decay.ken_french import parse_monthly_section

    wide = parse_monthly_section(KF_3F_TEXT)
    assert len(wide) == 2
    assert wide.index[0] == pd.Timestamp("1926-07-31")
    assert wide.loc["1926-08-31", "HML"] == pytest.approx(0.0381)


def test_parse_reversal_and_missing_sentinel():
    from src.signal_decay.ken_french import parse_monthly_section, to_tidy

    tidy = to_tidy(parse_monthly_section(KF_ST_TEXT), {"ST_Rev": "ST_Rev"})
    assert list(tidy.columns) == ["date", "signal", "return"]
    assert len(tidy) == 2  # the -99.99 month is dropped
    assert tidy["date"].min() == pd.Timestamp("1926-02-28")


def test_three_factor_columns_renamed_and_rf_dropped():
    from src.signal_decay.ken_french import DATASETS, parse_monthly_section, to_tidy

    tidy = to_tidy(parse_monthly_section(KF_3F_TEXT), DATASETS["F-F_Research_Data_Factors"])
    assert set(tidy["signal"]) == {"Mkt-RF", "SMB_3f", "HML_3f"}


def test_signal_names_unique_and_benchmark_not_testable():
    from src.signal_decay.ken_french import ALL_SIGNALS, DATASETS, TESTABLE_SIGNALS

    names = [n for m in DATASETS.values() for n in m.values()]
    assert len(names) == len(set(names))
    assert {"ST_Rev", "LT_Rev", "SMB_3f", "HML_3f", "SMB", "HML", "Mom"} <= TESTABLE_SIGNALS
    assert "Mkt-RF" in ALL_SIGNALS and "Mkt-RF" not in TESTABLE_SIGNALS


def test_download_uses_cache(tmp_path, no_network):
    import zipfile

    from src.signal_decay.ken_french import download_ken_french

    with zipfile.ZipFile(tmp_path / "F-F_ST_Reversal_Factor_CSV.zip", "w") as zf:
        zf.writestr("F-F_ST_Reversal_Factor.csv", KF_ST_TEXT)
    df = download_ken_french({"F-F_ST_Reversal_Factor": {"ST_Rev": "ST_Rev"}}, raw_dir=tmp_path)
    assert set(df["signal"]) == {"ST_Rev"}


# ---------- (b) papers.csv validation ----------

GOOD_ROW = ["momentum", "Paper A", "1993-03-01", "journal_issue", "Mom", "yes"]


def test_sample_papers_csv_validates():
    from src.signal_decay.papers import validate_papers

    df = pd.read_csv(contracts.path_for("papers", sample=True), dtype=str)
    out = validate_papers(df)
    assert list(out.columns) == PAPER_COLS


def test_real_papers_csv_validates_if_present():
    from src.signal_decay.papers import load_papers

    if not contracts.path_for("papers").exists():
        pytest.skip("data/papers.csv not created yet")
    load_papers()


def test_load_papers_never_falls_back_to_sample(tmp_path):
    from src.signal_decay.papers import load_papers

    with pytest.raises(FileNotFoundError):
        load_papers(tmp_path / "papers.csv")


@pytest.mark.parametrize(
    "row, message",
    [
        (["x", "p", "1992-13-40", "journal_issue", "HML", "yes"], "publication_date"),
        (["x", "p", "1992-06-01", "journal_issue", "Mkt-RF", "yes"], "benchmark"),
        (["x", "p", "1992-06-01", "journal_issue", "NotAFactor", "yes"], "unknown ken_french_column"),
        (["x", "p", "1992-06-01", "journal_issue", "HML", "maybe"], "source_verified"),
        (["x", "p", "1992-06-01", "", "HML", "yes"], "date_rule missing"),
    ],
)
def test_validator_rejects_bad_rows(row, message):
    from src.signal_decay.papers import validate_papers

    with pytest.raises(ValueError, match=message):
        validate_papers(_papers([GOOD_ROW, row]))


def test_validator_rejects_mixed_date_rules_and_duplicates():
    from src.signal_decay.papers import validate_papers

    with pytest.raises(ValueError, match="mixed date rules"):
        validate_papers(_papers([GOOD_ROW, ["value", "B", "1992-06-01", "ssrn_first", "HML", "yes"]]))
    with pytest.raises(ValueError, match="duplicate signals"):
        validate_papers(_papers([GOOD_ROW, GOOD_ROW]))


def test_unverified_rows_excluded_unless_allowed():
    from src.signal_decay.papers import select_for_results, validate_papers

    papers = validate_papers(_papers([GOOD_ROW, ["value", "B", "1992-06-01", "journal_issue", "HML", "no"]]))
    inc, exc = select_for_results(papers)
    assert list(inc["signal"]) == ["momentum"] and list(exc["signal"]) == ["value"]

    inc, exc = select_for_results(papers, allow_unverified=True)
    assert len(inc) == 2 and exc.empty
    assert inc.set_index("signal").loc["value", "unverified"]


# ---------- decay test ----------

def test_before_after_detects_known_drop():
    from src.signal_decay.decay import before_after_test

    rng = np.random.default_rng(0)
    values = np.r_[np.full(240, 0.010), np.full(240, 0.002)] + rng.normal(0, 0.002, 480)
    returns = _monthly(values, "1980-01-31")
    pub = returns.index[240]

    res = before_after_test(returns, pub)

    assert res["pre_n"] == 240 and res["post_n"] == 240 and res["status"] == "ok"
    assert res["pre_mean"] == pytest.approx(0.010, abs=0.001)
    assert res["post_mean"] == pytest.approx(0.002, abs=0.001)
    assert res["diff"] == pytest.approx(res["post_mean"] - res["pre_mean"])
    for prefix in ("diff_", "welch_"):  # true drop inside both CIs, and CIs exclude zero
        assert res[prefix + "ci_low"] < -0.008 < res[prefix + "ci_high"] < 0


def test_publication_month_counts_as_post():
    from src.signal_decay.decay import before_after_test

    returns = _monthly(np.zeros(10), "1993-01-31")
    res = before_after_test(returns, "1993-03-01", min_months=1)
    assert res["pre_n"] == 2 and res["post_n"] == 8


def test_newey_west_wider_than_welch_for_autocorrelated_returns():
    from src.signal_decay.decay import before_after_test

    rng = np.random.default_rng(1)
    e = rng.normal(0, 0.01, 600)
    x = np.zeros(600)
    for t in range(1, 600):
        x[t] = 0.6 * x[t - 1] + e[t]  # strong positive autocorrelation
    res = before_after_test(_monthly(x), _monthly(x).index[300])
    assert res["diff_se"] > 1.3 * res["welch_se"]


def test_short_window_is_flagged_not_dropped():
    from src.signal_decay.decay import before_after_test

    returns = _monthly(np.random.default_rng(2).normal(0.01, 0.02, 300))
    res = before_after_test(returns, returns.index[84])
    assert res["status"] == "insufficient_pre:84"
    assert np.isfinite(res["pre_mean"])  # stats still computed for the details file


# ---------- (c) charts ----------

def _factor_frame():
    s = _monthly(np.full(240, 0.01), "1980-01-31")
    return pd.DataFrame({"date": s.index, "signal": "Mom", "return": s.values})


def test_plot_marks_publication_and_saves(tmp_path):
    from src.signal_decay.papers import validate_papers
    from src.signal_decay.plots import plot_before_after, save_all_charts

    papers = validate_papers(_papers([GOOD_ROW]))
    fig = plot_before_after("momentum", factors=_factor_frame(), papers=papers)
    vlines = [ln for ln in fig.axes[0].get_lines() if ln.get_label() == "publication"]
    assert len(vlines) == 1

    paths = save_all_charts(papers, factors=_factor_frame(), out_dir=tmp_path)
    assert paths[0].exists() and paths[0].stat().st_size > 0


def test_plot_unknown_signal_raises():
    from src.signal_decay.papers import validate_papers
    from src.signal_decay.plots import plot_before_after

    with pytest.raises(ValueError, match="No returns"):
        plot_before_after("nope", factors=_factor_frame(), papers=validate_papers(_papers([GOOD_ROW])))


# ---------- (d) half-life ----------

def test_half_life_recovers_known_value():
    from src.signal_decay.decay import fit_half_life

    t = np.arange(300)
    post = 0.01 * np.exp(-np.log(2) / 60 * t)
    returns = _monthly(np.r_[np.full(120, 0.01), post])
    hl = fit_half_life(returns, returns.index[120])
    assert hl == pytest.approx(60, rel=0.10)


@pytest.mark.parametrize(
    "post",
    [np.full(300, 0.01), 0.002 + np.arange(300) * 1e-5, np.full(30, 0.01)],
    ids=["flat", "rising", "too_short"],
)
def test_half_life_none_without_clear_decay(post):
    from src.signal_decay.decay import fit_half_life

    returns = _monthly(np.r_[np.full(120, 0.01), post])
    assert fit_half_life(returns, returns.index[120]) is None


# ---------- (e) OpenAlex citations ----------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return _FakeResponse(self.payload)


PAYLOAD = {"group_by": [{"key": "1994", "count": 5}, {"key": "1996", "count": 7}]}


def test_citations_fetch_then_cache(tmp_path, monkeypatch):
    from src.signal_decay import attention

    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    session = _FakeSession(PAYLOAD)

    n = attention.early_citations("W123", 1993, cache_dir=tmp_path, session=session)
    assert n == 12
    assert session.calls[0]["filter"] == "cites:W123,publication_year:1994-1996"
    assert session.calls[0]["per_page"] == 100
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "test-key" not in json.dumps(cached)

    monkeypatch.delenv("OPENALEX_API_KEY")
    assert attention.early_citations("W123", 1993, cache_dir=tmp_path, session=session) == 12
    assert len(session.calls) == 1  # second call served from cache


def test_citations_missing_key_is_clear(tmp_path, monkeypatch):
    from src.signal_decay import attention

    monkeypatch.setattr(attention, "get_openalex_credentials", lambda: ("", ""))
    with pytest.raises(RuntimeError, match="OPENALEX_API_KEY"):
        attention.fetch_citing_counts_by_year("W123", 1994, 1996, cache_dir=tmp_path)


def test_early_citations_window():
    from src.signal_decay.attention import compute_early_citations

    counts = pd.DataFrame({"year": [1993, 1994, 1995, 1996, 1997], "cited_by_count": [1, 2, 3, 4, 5]})
    assert compute_early_citations(counts, 1993) == 2 + 3 + 4
    assert compute_early_citations(counts.iloc[0:0], 1993) == 0
    assert compute_early_citations(None, 1993) is None


def test_paper_ids_reject_malformed(tmp_path):
    from src.signal_decay.attention import load_paper_ids

    path = tmp_path / "paper_ids.csv"
    path.write_text("signal,openalex_id\nmomentum,https://openalex.org/W1\n")
    with pytest.raises(ValueError, match="malformed"):
        load_paper_ids(path)


# ---------- pipeline ----------

def test_build_results_matches_contract_and_flags_rows():
    from src.signal_decay.papers import select_for_results, validate_papers
    from src.signal_decay.run_decay import build_results

    s = _monthly(np.r_[np.full(240, 0.01), np.full(240, 0.002)], "1970-01-31")
    factors = pd.DataFrame({"date": s.index, "signal": "Mom", "return": s.values})
    papers = validate_papers(_papers([
        ["momentum", "A", s.index[240].strftime("%Y-%m-%d"), "journal_issue", "Mom", "yes"],
        ["late", "B", s.index[420].strftime("%Y-%m-%d"), "journal_issue", "Mom", "yes"],
        ["value", "C", "1992-06-01", "journal_issue", "HML", "no"],
    ]))
    inc, exc = select_for_results(papers)
    results, details = build_results(factors, inc, exc, citations=False)

    contracts.validate("decay_results", results)
    assert list(results["signal"]) == ["momentum", "late"]
    late = results.set_index("signal").loc["late"]
    assert pd.isna(late["pre_mean"]) and pd.isna(late["ci_low"])  # below 120 post months
    status = details.set_index("signal")["status"]
    assert status["momentum"] == "ok"
    assert status["late"] == "insufficient_post:60"
    assert status["value"] == "excluded_unverified"
