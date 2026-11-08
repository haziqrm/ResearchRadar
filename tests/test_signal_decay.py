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
    "src.signal_decay.robustness",
    "src.signal_decay.run_decay",
    "src.signal_decay.run_osap",
    "src.signal_decay.decay_model",
    "src.signal_decay.decay_model_h8",
    "src.signal_decay.forecasts",
    "src.signal_decay.vol_model",
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
    results, details = build_results(factors, inc, exc, citations=False, bootstrap_draws=50)

    contracts.validate("decay_results", results)
    assert list(results["signal"]) == ["momentum", "late"]
    late = results.set_index("signal").loc["late"]
    assert pd.isna(late["pre_mean"]) and pd.isna(late["ci_low"])  # below 120 post months
    status = details.set_index("signal")["status"]
    assert status["momentum"] == "ok"
    assert status["late"] == "insufficient_post:60"
    assert status["value"] == "excluded_unverified"


# ---------- era-adjusted attention (H3 input) ----------

class _RoutingSession:
    """Fake OpenAlex: answers work lookups, peer listings and citing counts offline."""

    def __init__(self, counts):
        self.counts, self.calls = counts, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        f = params.get("filter", "")
        if url.endswith("/works/W1"):
            payload = {"id": "https://openalex.org/W1", "publication_year": 1993,
                       "primary_location": {"source": {"id": "https://openalex.org/S9"}}}
        elif f.startswith("primary_location.source.id"):
            payload = {"results": [{"id": f"https://openalex.org/{w}"} for w in self.counts],
                       "meta": {"next_cursor": None}}
        else:
            wid = f.split(",")[0].split(":")[1]
            payload = {"meta": {"count": self.counts[wid]}}
        return _FakeResponse(payload)


def test_era_adjusted_attention_ranks_against_peers(tmp_path, monkeypatch):
    from src.signal_decay import attention

    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")
    counts = {"W1": 30, "W2": 5, "W3": 10, "W4": 30, "W5": 50}
    session = _RoutingSession(counts)
    att = attention.era_adjusted_attention("W1", 1993, cache_dir=tmp_path, session=session)

    assert att["early_citations"] == 30
    assert att["peer_n"] == 4  # the paper itself is not its own peer
    assert att["peer_median"] == 20
    assert att["percentile"] == pytest.approx(100 * (2 + 0.5) / 4)

    calls = session.calls
    attention.era_adjusted_attention("W1", 1993, cache_dir=tmp_path, session=session)
    assert session.calls == calls  # everything served from cache


def test_peer_sample_is_deterministic_and_capped():
    from src.signal_decay.attention import sample_peers

    pool = [f"W{i}" for i in range(200)]
    a, b = sample_peers(pool, "W0", n=50), sample_peers(list(reversed(pool)), "W0", n=50)
    assert a == b and len(a) == 50 and "W0" not in a


# ---------- H3 test ----------

def test_attention_decay_test_exact_permutation():
    from src.signal_decay.attention import attention_decay_test

    a = pd.Series([10, 20, 30, 40, 50, 60, 70])
    res = attention_decay_test(a, a * 0.01)  # perfectly increasing
    assert res["spearman"] == pytest.approx(1.0)
    assert res["p_one_sided"] == pytest.approx(1 / 5040)

    res = attention_decay_test(a, -a)  # perfectly decreasing: no support for H3
    assert res["p_one_sided"] == pytest.approx(1.0)


def test_attention_decay_test_needs_three_points():
    from src.signal_decay.attention import attention_decay_test

    res = attention_decay_test(pd.Series([1.0, np.nan]), pd.Series([0.5, 0.2]))
    assert res["n"] == 1 and np.isnan(res["p_one_sided"])


# ---------- H2 placebo ----------

def test_split_differences_match_direct_means():
    from src.signal_decay.robustness import split_differences

    s = _monthly(np.random.default_rng(3).normal(0, 0.01, 300))
    diffs = split_differences(s, min_months=120)
    assert len(diffs) == 300 - 2 * 120 + 1
    k = 150
    direct = s.iloc[k:].mean() - s.iloc[:k].mean()
    assert diffs.loc[s.index[k]] == pytest.approx(direct)


def test_placebo_flags_real_break_and_not_noise():
    from src.signal_decay.robustness import placebo_test

    rng = np.random.default_rng(4)
    broken = _monthly(np.r_[np.full(300, 0.01), np.full(300, 0.0)] + rng.normal(0, 0.003, 600))
    assert placebo_test(broken, broken.index[300])["placebo_p"] < 0.05

    noise = _monthly(rng.normal(0.005, 0.02, 600))
    assert placebo_test(noise, noise.index[300])["placebo_p"] > 0.05


def test_pooled_placebo_detects_common_break():
    from src.signal_decay.robustness import pooled_placebo_test

    rng = np.random.default_rng(5)
    series, dates = {}, {}
    for i, brk in enumerate([200, 300, 400]):
        v = np.r_[np.full(brk, 0.01), np.full(600 - brk, 0.0)] + rng.normal(0, 0.003, 600)
        series[f"s{i}"] = _monthly(v)
        dates[f"s{i}"] = series[f"s{i}"].index[brk]
    res = pooled_placebo_test(series, dates, draws=2000)
    assert res["n_signals"] == 3 and res["p_one_sided"] < 0.05


# ---------- half-life and retained-share intervals ----------

def test_bootstrap_interval_covers_known_half_life():
    from src.signal_decay.robustness import bootstrap_decay

    rng = np.random.default_rng(6)
    t = np.arange(400)
    post = 0.01 * np.exp(-np.log(2) / 60 * t) + rng.normal(0, 0.001, 400)
    s = _monthly(np.r_[np.full(240, 0.01) + rng.normal(0, 0.001, 240), post])
    res = bootstrap_decay(s, s.index[240], draws=200)
    assert res["half_life_fit_rate"] > 0.8
    assert res["half_life_ci_low"] < 60 < res["half_life_ci_high"]
    assert res["half_life_reportable"]
    assert res["retained_ci_low"] < res["retained_share"] < res["retained_ci_high"]


def test_noisy_series_half_life_not_reportable():
    from src.signal_decay.robustness import bootstrap_decay

    s = _monthly(np.random.default_rng(7).normal(0.005, 0.04, 600))
    res = bootstrap_decay(s, s.index[300], draws=200)
    assert not res["half_life_reportable"]


def test_build_summary_rows():
    from src.signal_decay.papers import select_for_results, validate_papers
    from src.signal_decay.run_decay import SUMMARY_COLUMNS, build_results, build_summary

    rng = np.random.default_rng(8)
    s = _monthly(np.r_[np.full(300, 0.01), np.full(300, 0.002)] + rng.normal(0, 0.003, 600), "1940-01-31")
    factors = pd.DataFrame({"date": s.index, "signal": "Mom", "return": s.values})
    papers = validate_papers(_papers([["momentum", "A", s.index[300].strftime("%Y-%m-%d"),
                                       "journal_issue", "Mom", "yes"]]))
    inc, exc = select_for_results(papers)
    _, details = build_results(factors, inc, exc, citations=False, bootstrap_draws=20)
    summary = build_summary(factors, inc, details)
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert summary.set_index("test").loc["H2 pooled placebo", "p_value"] < 0.05
    assert summary.set_index("test").loc["H3 attention vs decay", "n"] == 0  # no citations


# ---------- OSAP extension (H4-H6) ----------

@pytest.mark.parametrize(
    "authors, expected",
    [
        ("Banz", ["Banz"]),
        ("Jegadeesh and Titman", ["Jegadeesh", "Titman"]),
        ("Chan, Jegadeesh and Lakonishok", ["Chan", "Jegadeesh", "Lakonishok"]),
        ("Jegadeesh et al.", ["Jegadeesh"]),
        ("Avramov et al", ["Avramov"]),
    ],
)
def test_parse_surnames(authors, expected):
    from src.signal_decay.osap import parse_surnames

    assert parse_surnames(authors) == expected


@pytest.mark.parametrize(
    "surname, name, ok",
    [
        ("Titman", "Sheridan D. Titman", True),
        ("Novy-Marx", "Robert Novy‐Marx", True),
        ("DeBondt", "Werner F. M. De Bondt", True),
        ("Li", "William Lin", False),
        ("Fama", "Eugene F. Fama Jr.", True),
    ],
)
def test_surname_matches(surname, name, ok):
    from src.signal_decay.osap import surname_matches

    assert surname_matches(surname, name) is ok


def _osap_papers():
    return pd.DataFrame({"signal": ["A", "B"], "authors": ["X", "Y"], "year": [1990, 1995],
                         "journal": ["JF", "JF"], "sample_start": [1960, 1960],
                         "sample_end": [1985, 1985], "gs_cites": [100.0, 50.0]})


def _osap_returns(is_level=0.01, oos_level=0.008, post_level=0.004, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1955-01-31", "2020-12-31", freq="ME")
    rows = []
    for sig, year in [("A", 1990), ("B", 1995)]:
        y = dates.year
        mu = np.where(y <= 1985, is_level, np.where(y <= year, oos_level, post_level))
        rows.append(pd.DataFrame({"date": dates, "signal": sig, "return": mu + rng.normal(0, 0.002, len(dates))}))
    return pd.concat(rows, ignore_index=True)


def test_period_panel_labels_and_drops_pre_sample():
    from src.signal_decay.osap import period_panel

    panel = period_panel(_osap_returns(), _osap_papers())
    a = panel[panel["signal"] == "A"].set_index("date")["period"]
    assert a.index.min() == pd.Timestamp("1960-01-31")  # months before sample start dropped
    assert a["1985-12-31"] == "IS" and a["1986-01-31"] == "OOS"
    assert a["1990-12-31"] == "OOS" and a["1991-01-31"] == "POST"  # end-of-publication-year rule


def test_mp_regression_recovers_known_drops():
    from src.signal_decay.osap import mp_regression, period_panel

    panel = period_panel(_osap_returns(0.01, 0.008, 0.004), _osap_papers())
    h4 = mp_regression(panel)
    assert h4["n_signals"] == 2
    assert h4["b_oos"] == pytest.approx(-0.2, abs=0.03)
    assert h4["b_post"] == pytest.approx(-0.6, abs=0.03)
    assert h4["post_minus_oos"] < 0 and h4["post_minus_oos_p_one_sided"] < 0.05


def test_mp_regression_excludes_nonpositive_in_sample():
    from src.signal_decay.osap import mp_regression, period_panel

    r = _osap_returns()
    r.loc[(r["signal"] == "B") & (r["date"].dt.year <= 1985), "return"] -= 0.05
    h4 = mp_regression(period_panel(r, _osap_papers()))
    assert h4["n_signals"] == 1 and h4["n_excluded_nonpositive_is"] == 1


def test_signal_period_stats_relative_decay():
    from src.signal_decay.osap import period_panel, signal_period_stats

    st = signal_period_stats(period_panel(_osap_returns(), _osap_papers())).set_index("signal")
    assert st.loc["A", "relative_decay"] == pytest.approx(0.6, abs=0.05)
    assert st.loc["A", "n_is"] == 26 * 12


def test_within_year_percentile_uses_other_papers_only():
    from src.signal_decay.osap import within_year_percentile

    v = pd.Series([10.0, 20.0, 30.0, 5.0])
    y = pd.Series([1990, 1990, 1990, 2000])
    pct = within_year_percentile(v, y)
    assert pct.tolist()[:3] == [0.0, 50.0, 100.0]
    assert np.isnan(pct.iloc[3])  # no other paper that year


def test_match_paper_unique_ambiguous_and_no_journal(tmp_path, monkeypatch):
    from src.signal_decay import osap

    monkeypatch.setenv("OPENALEX_API_KEY", "test-key")

    def author(n):
        return {"author": {"display_name": n}, "raw_author_name": n}

    class Session:
        def __init__(self, works):
            self.works = works

        def get(self, url, params=None, timeout=None):
            if url.endswith("/sources"):
                return _FakeResponse({"results": [{"id": "https://openalex.org/S1",
                                                   "display_name": "The Journal of Finance"}]})
            return _FakeResponse({"results": self.works})

    jt = {"id": "https://openalex.org/W9", "display_name": "Returns to buying winners",
          "publication_year": 1993, "authorships": [author("N. Jegadeesh"), author("Sheridan Titman")]}
    other = {"id": "https://openalex.org/W8", "display_name": "Other", "publication_year": 1993,
             "authorships": [author("N. Jegadeesh")]}

    m = osap.match_paper("Jegadeesh and Titman", 1993, "JF", cache_dir=tmp_path / "a", session=Session([jt, other]))
    assert m["match_status"] == "matched" and m["openalex_id"] == "W9"

    m = osap.match_paper("Jegadeesh", 1993, "JF", cache_dir=tmp_path / "b", session=Session([jt, other]))
    assert m["match_status"] == "ambiguous" and m["n_candidates"] == 2

    m = osap.match_paper("Someone", 2001, "WP", cache_dir=tmp_path / "c", session=Session([]))
    assert m["match_status"] == "no_journal"


def test_openalex_errors_never_leak_the_key(tmp_path, monkeypatch):
    from src.signal_decay import attention

    monkeypatch.setenv("OPENALEX_API_KEY", "secret-key-123")

    class Bad:
        status_code, headers = 500, {}

    class Session:
        def get(self, url, params=None, timeout=None):
            return Bad()

    with pytest.raises(RuntimeError) as exc:
        attention._get_json("works", {"filter": "x"}, "bad", cache_dir=tmp_path, session=Session())
    assert "secret-key-123" not in str(exc.value) and "HTTP 500" in str(exc.value)


def test_pooled_placebo_skips_signals_short_before_publication():
    from src.signal_decay.robustness import pooled_placebo_test

    rng = np.random.default_rng(9)
    long = _monthly(rng.normal(0.005, 0.01, 600))
    short_pre = _monthly(rng.normal(0.005, 0.01, 600))
    res = pooled_placebo_test({"a": long, "b": short_pre},
                              {"a": long.index[300], "b": short_pre.index[50]}, draws=500)
    assert res["n_signals"] == 1 and np.isfinite(res["mean_real_diff"])


def test_openalex_retries_server_errors(tmp_path, monkeypatch):
    from src.signal_decay import attention

    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    monkeypatch.setattr(attention.time, "sleep", lambda s: None)

    class R:
        def __init__(self, code):
            self.status_code, self.headers = code, {}

        def json(self):
            return {"ok": True}

    class Session:
        def __init__(self):
            self.codes = [504, 502, 200]

        def get(self, url, params=None, timeout=None):
            return R(self.codes.pop(0))

    assert attention._get_json("works", {}, "retry", cache_dir=tmp_path, session=Session()) == {"ok": True}


# ---------- H7 decay predictor ----------

def _model_frame(n_papers=60, seed=0, signal_in_feature=True):
    """Synthetic signals: retained share depends on is_tstat when signal_in_feature."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_papers):
        year = 1980 + p % 36
        for k in range(2):
            t = rng.uniform(2, 8)
            target = (1.2 - 0.12 * t if signal_in_feature else 0.5) + rng.normal(0, 0.05)
            rows.append({
                "signal": f"s{p}_{k}", "authors": f"A{p}", "year": year, "journal": "JF",
                "paper_key": f"A{p}|{year}|JF", "target": target,
                "op_tstat": np.nan if k else t, "is_mean": 0.01, "is_tstat": t,
                "sample_years": 20, "pub_lag": 3, "pub_year": float(year),
                "citation_pct": rng.uniform(0, 100), "data_type": "Accounting",
                "journal_group": "top3", "discrete": 0.0, "clear": 1.0, "equal_weighted": 1.0,
            })
    return pd.DataFrame(rows)


def test_feature_matrix_uses_training_medians_and_flags():
    from src.signal_decay.decay_model import feature_matrix

    df = _model_frame(4)
    X = feature_matrix(df, {"op_tstat": 99.0})
    missing = df["op_tstat"].isna()
    assert (X.loc[missing, "op_tstat"] == 99.0).all()
    assert (X.loc[missing, "op_tstat_missing"] == 1.0).all()
    assert X.isna().sum().sum() == 0


def test_model_beats_baseline_when_signal_exists():
    from src.signal_decay.decay_model import fit_and_evaluate

    res = fit_and_evaluate(_model_frame(signal_in_feature=True), train_last_year=2000)
    s = res["summary"].set_index("model")
    assert s.loc["ridge", "verdict"] == "beats baseline"
    test_years = res["predictions"]["year"]
    assert test_years.min() > 2000  # no test paper from the training period


def test_model_does_not_beat_baseline_on_noise():
    from src.signal_decay.decay_model import fit_and_evaluate

    res = fit_and_evaluate(_model_frame(signal_in_feature=False), train_last_year=2000)
    s = res["summary"].set_index("model")
    assert s.loc["ridge", "verdict"] == "does not beat baseline"
    assert s.loc["random_forest", "verdict"] == "does not beat baseline"


def test_paper_bootstrap_resamples_whole_papers():
    from src.signal_decay.decay_model import paper_bootstrap_ci

    err_model = np.array([0.1, 0.1, 0.5, 0.5])
    err_base = np.array([0.3, 0.3, 0.3, 0.3])
    lo, hi = paper_bootstrap_ci(err_model, err_base, np.array(["a", "a", "b", "b"]), draws=500)
    assert lo == pytest.approx(-0.2) and hi == pytest.approx(0.2)


# ---------- H8 walk-forward predictor ----------

def _h8_doc():
    rows = [("A", 1990, "momentum"), ("B", 1995, "momentum"), ("C", 1996, "value"), ("D", 1999, "momentum")]
    return pd.DataFrame([{
        "Acronym": sig, "Cat.Signal": "Predictor", "Authors": f"Au{sig}", "Year": y, "Journal": "JF",
        "SampleStartYear": 1960, "SampleEndYear": y - 2, "GScholarCites202509": 10, "Cat.Economic": cat,
    } for sig, y, cat in rows])


def test_h8_features_use_pre_publication_data():
    from src.signal_decay.decay_model_h8 import compute_h8_features

    dates = pd.date_range("1960-01-31", "2010-12-31", freq="ME")
    rng = np.random.default_rng(0)
    base = rng.normal(0.01, 0.02, len(dates))
    series = {"A": base, "B": base * 2, "C": rng.normal(0.01, 0.02, len(dates)), "D": rng.normal(0.01, 0.02, len(dates))}
    tidy = lambda d: pd.concat([pd.DataFrame({"date": dates, "signal": k, "return": v}) for k, v in d.items()])
    liquid = tidy({k: v * 0.5 for k, v in series.items()})
    f = compute_h8_features(_h8_doc(), tidy(series), liquid).set_index("signal")

    assert f.loc["A", "liquid_share"] == pytest.approx(0.5)
    assert np.isnan(f.loc["A", "prior_max_corr"])           # nothing published before A
    assert f.loc["B", "prior_max_corr"] == pytest.approx(1.0)  # B is a scaled copy of A
    assert f.loc["D", "category_heat"] == 1                 # B (1995) is within 5 years; A (1990) is not
    assert f.loc["C", "category_heat"] == 0                 # different category


def _h8_frame(signal_in_feature=True, seed=0):
    df = _model_frame(n_papers=80, seed=seed, signal_in_feature=False)
    rng = np.random.default_rng(seed + 1)
    df["year"] = 1980 + (np.arange(len(df)) // 2) % 37
    df["paper_key"] = df["authors"] + "|" + df["year"].astype(str) + "|JF"
    df["liquid_share"] = rng.uniform(0, 1, len(df))
    df["prior_max_corr"] = rng.uniform(0, 1, len(df))
    df["category_heat"] = rng.integers(0, 5, len(df)).astype(float)
    if signal_in_feature:
        df["target"] = 1.0 - df["liquid_share"] + rng.normal(0, 0.05, len(df))
    return df


def test_h8_walk_forward_finds_signal_without_lookahead():
    from src.signal_decay.decay_model_h8 import FOLDS, walk_forward

    res = walk_forward(_h8_frame(True))
    s = res["summary"].set_index(["fold", "model"])
    assert s.loc[("pooled", "ridge_h8"), "verdict"] == "beats baseline"
    assert s.loc[("pooled", "ridge_h7_only"), "verdict"] == "does not beat baseline"
    p = res["predictions"]
    for start, end in FOLDS:
        years = p.loc[p["fold"] == f"{start}-{end}", "year"]
        assert years.between(start, end).all()
        assert s.loc[(f"{start}-{end}", "baseline"), "n_train"] == (_h8_frame(True)["year"] < start).sum()


def test_h8_walk_forward_rejects_noise():
    from src.signal_decay.decay_model_h8 import walk_forward

    s = walk_forward(_h8_frame(False))["summary"].set_index(["fold", "model"])
    assert s.loc[("pooled", "ridge_h8"), "verdict"] == "does not beat baseline"


# ---------- checks added after the evaluation diagnosis ----------

def test_no_feature_uses_post_publication_citations():
    from src.signal_decay.decay_model import NUMERIC, feature_matrix
    from src.signal_decay.decay_model_h8 import feature_matrix_h8

    assert "citation_pct" not in NUMERIC  # counts citations in Y+1..Y+3, not known at publication
    df = _h8_frame(True)
    assert not any("citation" in c for c in feature_matrix(df, {}).columns)
    assert not any("citation" in c for c in feature_matrix_h8(df, {}).columns)


def test_features_ignore_post_publication_returns():
    """Changing only POST-period returns changes the target but not a single feature."""
    from src.signal_decay.decay_model import build_dataset, feature_matrix, training_medians, NUMERIC

    dates = pd.date_range("1950-01-31", "2025-12-31", freq="ME")
    rng = np.random.default_rng(3)
    docs, rets = [], []
    for i in range(12):
        y = 1980 + 2 * i
        docs.append({"Acronym": f"S{i}", "Cat.Signal": "Predictor", "Authors": f"A{i}", "Year": y, "Journal": "JF",
                     "SampleStartYear": y - 25, "SampleEndYear": y - 3, "GScholarCites202509": 1,
                     "Cat.Data": "Price", "Cat.Form": "continuous", "Predictability in OP": "1_clear",
                     "Stock Weight": "EW", "T-Stat": 3.0})
        rets.append(pd.DataFrame({"date": dates, "signal": f"S{i}", "return": rng.normal(0.01, 0.02, len(dates))}))
    doc, ret = pd.DataFrame(docs), pd.concat(rets, ignore_index=True)
    a = build_dataset(doc, ret, None)
    pub = ret["signal"].map(doc.set_index("Acronym")["Year"])
    shocked = ret.assign(**{"return": np.where(ret["date"].dt.year > pub, ret["return"] + 0.05, ret["return"])})
    b = build_dataset(doc, shocked, None)
    med = training_medians(a, NUMERIC)
    pd.testing.assert_frame_equal(feature_matrix(a, med), feature_matrix(b, med))
    assert not np.allclose(a["target"], b["target"])


def test_baseline_uses_training_rows_only():
    from src.signal_decay.decay_model import fit_and_evaluate

    df = _model_frame(signal_in_feature=False)
    moved = df.copy()
    moved.loc[moved["year"] > 2000, "target"] += 5.0  # wild test targets must not move the baseline
    a = fit_and_evaluate(df, train_last_year=2000)["predictions"]["pred_baseline"]
    b = fit_and_evaluate(moved, train_last_year=2000)["predictions"]["pred_baseline"]
    assert np.allclose(a, b)
    assert np.allclose(a, np.median(df.loc[df["year"] <= 2000, "target"]))


def test_heavily_penalised_median_regression_is_the_baseline():
    """The median regression nests the constant baseline: with no signal it falls back to it."""
    from src.signal_decay.decay_model import fit_and_evaluate

    res = fit_and_evaluate(_model_frame(signal_in_feature=False), train_last_year=2000)
    p = res["predictions"]
    assert np.allclose(p["pred_median_regression"], p["pred_baseline"], atol=0.02)
    assert p["pred_median_regression"].std() < 0.01  # flat: the L1 penalty zeroed the coefficients


def test_summary_reports_mae_mse_and_both_constants():
    from src.signal_decay.decay_model import fit_and_evaluate

    s = fit_and_evaluate(_model_frame(), train_last_year=2000)["summary"].set_index("model")
    assert {"baseline", "baseline_mean", "ridge", "median_regression", "random_forest"} <= set(s.index)
    for col in ["test_mae", "test_mse", "mse_diff_vs_baseline", "alpha", "alpha_at_grid_edge"]:
        assert col in s.columns
    assert s.loc["baseline", "mse_diff_vs_baseline"] == 0.0


def test_model_survives_infinite_and_all_missing_inputs():
    from src.signal_decay.decay_model import fit_and_evaluate

    df = _model_frame()
    df.loc[df.index[:3], "is_tstat"] = np.inf
    df["op_tstat"] = np.nan
    s = fit_and_evaluate(df, train_last_year=2000)["summary"]
    assert np.isfinite(s["test_mae"]).all()


def test_zero_variance_signal_gets_missing_tstat_not_infinity():
    from src.signal_decay.decay_model import build_dataset

    dates = pd.date_range("1960-01-31", "2010-12-31", freq="ME")
    doc = pd.DataFrame([{"Acronym": "Z", "Cat.Signal": "Predictor", "Authors": "A", "Year": 1990, "Journal": "JF",
                         "SampleStartYear": 1960, "SampleEndYear": 1987, "GScholarCites202509": 1,
                         "Cat.Data": "Price", "Cat.Form": "continuous", "Predictability in OP": "1_clear",
                         "Stock Weight": "EW", "T-Stat": 3.0}])
    ret = pd.DataFrame({"date": dates, "signal": "Z", "return": 0.01})
    df = build_dataset(doc, ret, None)
    assert df["is_tstat"].isna().all()


# ---------- H9 month-ahead forecasts ----------

def _h9_returns(n_signals=30, months=400, persistent_mean=False, vol_clusters=False, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1980-01-31", periods=months, freq="ME")
    frames = []
    for i in range(n_signals):
        if persistent_mean:  # slow-moving true mean: past winners keep winning
            mu = np.repeat(rng.normal(0, 0.01, months // 60 + 1), 60)[:months]
        else:
            mu = np.zeros(months)
        sigma = np.repeat(rng.choice([0.01, 0.06], months // 24 + 1), 24)[:months] if vol_clusters             else np.full(months, 0.03)
        frames.append(pd.DataFrame({"date": dates, "signal": f"S{i}", "return": mu + sigma * rng.standard_normal(months)}))
    return pd.concat(frames, ignore_index=True)


def test_h9_forecasts_use_only_past_months():
    from src.signal_decay.forecasts import forecast_panel

    r = _h9_returns(3, 100)
    a = forecast_panel(r)
    last = r["date"].max()
    shocked = r.assign(**{"return": np.where(r["date"] == last, 9.0, r["return"])})
    b = forecast_panel(shocked)
    cols = ["mom_12m", "vol_forecast", "vol_baseline"]
    pd.testing.assert_frame_equal(a[cols], b[cols])  # month t's forecast ignores month t
    s0 = a[a["signal"] == "S0"].set_index("date")
    raw = r[r["signal"] == "S0"].set_index("date")["return"]
    assert s0["mom_12m"].iloc[12] == pytest.approx(raw.iloc[:12].mean())
    assert s0["mom_12m"].iloc[:12].isna().all() and s0["vol_baseline"].iloc[:36].isna().all()


def test_h9_post_publication_filter():
    from src.signal_decay.forecasts import forecast_panel, post_publication

    panel = post_publication(forecast_panel(_h9_returns(2, 120)), pd.Series({"S0": 1985, "S1": 1987}))
    assert panel.loc[panel["signal"] == "S0", "date"].dt.year.min() == 1986
    assert panel.loc[panel["signal"] == "S1", "date"].dt.year.min() == 1988


def test_h9_momentum_detected_only_when_present():
    from src.signal_decay.forecasts import forecast_panel, momentum_test

    yes, _ = momentum_test(forecast_panel(_h9_returns(persistent_mean=True)))
    no, _ = momentum_test(forecast_panel(_h9_returns(persistent_mean=False)))
    assert yes["p_one_sided"] < 0.05 and yes["mean"] > 0
    assert no["p_one_sided"] > 0.05


def test_h9_volatility_detected_only_when_clustered():
    from src.signal_decay.forecasts import forecast_panel, volatility_test

    yes, _ = volatility_test(forecast_panel(_h9_returns(vol_clusters=True)))
    no, _ = volatility_test(forecast_panel(_h9_returns(vol_clusters=False)))
    assert yes["p_one_sided"] < 0.05 and yes["mean_loss_diff"] < 0
    assert no["p_one_sided"] > 0.05


def test_h9_vol_managed_matches_plain_volatility():
    from src.signal_decay.forecasts import forecast_panel, vol_managed

    port = vol_managed(forecast_panel(_h9_returns(vol_clusters=True)))
    assert port["vol_managed"].std() == pytest.approx(port["equal_weight"].std())


def test_h9_latest_forecasts_snapshot():
    from src.signal_decay.forecasts import latest_forecasts

    r = _h9_returns(2, 60)
    snap = latest_forecasts(r, "osap", {"S0": "Zero"}).set_index("signal")
    raw = r[r["signal"] == "S0"]["return"].iloc[-12:]
    assert snap.loc["S0", "name"] == "Zero" and snap.loc["S1", "name"] == "S1"
    assert snap.loc["S0", "mom_12m"] == pytest.approx(raw.mean())
    assert snap.loc["S0", "momentum_on"] == (raw.mean() > 0)


# ---------- H10 trained volatility model ----------

def _h10_inputs(clustered=True, n=25, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1984-01-31", "2010-12-31", freq="ME")
    T = len(dates)
    frames = []
    for i in range(n):
        if clustered:  # persistent log-volatility (AR(1)): recent months beat a 12-month average
            lv = np.zeros(T)
            for t in range(1, T):
                lv[t] = 0.9 * lv[t - 1] + 0.35 * rng.standard_normal()
            sigma = 0.03 * np.exp(lv)
        else:
            sigma = np.full(T, 0.03)
        frames.append(pd.DataFrame({"date": dates, "signal": f"S{i}", "return": sigma * rng.standard_normal(T)}))
    returns = pd.concat(frames, ignore_index=True)
    pub = pd.Series({f"S{i}": 1984 for i in range(n)})
    market = pd.Series(0.04 * rng.standard_normal(T), index=dates)
    vix = pd.Series(rng.uniform(10, 40, T), index=dates)
    return returns, pub, market, vix


def test_h10_features_use_only_past_data():
    from src.signal_decay.vol_model import FEATURES, build_features

    r, pub, mkt, vix = _h10_inputs()
    a = build_features(r, pub, mkt, vix)
    last = r["date"].max()
    r2 = r.assign(**{"return": np.where(r["date"] == last, 5.0, r["return"])})
    mkt2, vix2 = mkt.copy(), vix.copy()
    mkt2.loc[last], vix2.loc[last] = 5.0, 500.0
    b = build_features(r2, pub, mkt2, vix2)
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES])  # month-t features ignore month t
    assert not np.allclose(a["abs_r"], b["abs_r"])


def test_h10_vix_is_previous_month_end_on_monthly_scale():
    from src.signal_decay.vol_model import build_features, parse_vix

    daily = "\n".join(["observation_date,VIXCLS", "1999-12-30,20", "1999-12-31,24", "2000-01-31,.",
                       "2000-01-28,30"])
    v = parse_vix(daily)
    assert v.loc["1999-12-31"] == 24 and v.loc["2000-01-31"] == 30  # last valid close of each month
    r, pub, mkt, vix = _h10_inputs()
    f = build_features(r, pub, mkt, vix)
    row = f[f["date"] == pd.Timestamp("2000-06-30")].iloc[0]
    assert row["vix"] == pytest.approx(vix.loc["2000-05-31"] / 100 / np.sqrt(12))


def test_h10_walk_forward_keeps_twelve_month_gap_and_ignores_test_data():
    from src.signal_decay.vol_model import build_features, walk_forward

    r, pub, mkt, vix = _h10_inputs()
    df = build_features(r, pub, mkt, vix)
    pred, folds = walk_forward(df, test_years=[2003, 2004])
    assert (folds["train_last"] <= folds["test_year"] - 2).all()
    assert folds["tuned"].all()  # 1990..2001 = 12 training years >= 8
    shocked = df.copy()
    shocked.loc[shocked["date"].dt.year == 2004, "abs_r"] *= 10
    _, folds2 = walk_forward(shocked, test_years=[2004])
    cols = ["ridge_alpha", "ridge_c", "hgb_learning_rate", "hgb_max_leaf_nodes", "hgb_min_samples_leaf"]
    pd.testing.assert_frame_equal(folds.loc[folds["test_year"] == 2004, cols].reset_index(drop=True),
                                  folds2[cols].reset_index(drop=True))


def test_h10_short_history_uses_fixed_defaults():
    from src.signal_decay.vol_model import DEFAULT_C, HGB_DEFAULT, choose_settings

    s = choose_settings(pd.DataFrame({"date": pd.to_datetime(["1990-01-31"])}), 1996)
    assert not s["tuned"] and s["c"] == pytest.approx(1.887, abs=1e-3) and s["c"] == DEFAULT_C
    assert s["hgb"] == HGB_DEFAULT and s["alpha"] == 1.0


def test_h10_ridge_finds_persistent_volatility_and_no_spurious_skill():
    from src.signal_decay.vol_model import build_features, compare, walk_forward

    years = list(range(2006, 2011))
    r, pub, mkt, vix = _h10_inputs(clustered=True)
    pred, _ = walk_forward(build_features(r, pub, mkt, vix), test_years=years)
    assert compare(pred, "ridge", "a", (2006, 2010), ci=False)["p_one_sided"] < 0.05
    # Constant volatility: the scale-corrected long-run average is optimal, so ridge must not beat it.
    r, pub, mkt, vix = _h10_inputs(clustered=False)
    pred, _ = walk_forward(build_features(r, pub, mkt, vix), test_years=years)
    assert compare(pred, "ridge", "b_scaled", (2006, 2010), ci=False)["p_one_sided"] > 0.05


def test_h10_best_scale_and_holm():
    from src.signal_decay.vol_model import ABS_SCALE, best_scale, holm

    e = np.ones(5)
    a = ABS_SCALE * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert best_scale(e, a) == pytest.approx(3.0)  # median ratio minimises absolute error
    assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])


def test_h10_portfolio_cap_scaling_and_costs():
    from src.signal_decay.vol_model import WEIGHT_CAP, max_drawdown, portfolio_returns, portfolio_weights

    dates = pd.to_datetime(["2000-01-31", "2000-01-31", "2000-02-29", "2000-02-29"])
    pred = pd.DataFrame({"date": dates, "signal": ["x", "y", "x", "y"],
                         "vol": [0.001, 0.05, 0.05, 0.05], "ret": [0.01, -0.02, 0.03, 0.01]})
    W = portfolio_weights(pred, "vol")
    assert W.sum(axis=1).mean() == pytest.approx(1.0)  # average gross exposure 1
    raw = 1 / pred["vol"] ** 2
    capped = np.minimum(raw, WEIGHT_CAP * raw.mean())
    assert W.loc["2000-01-31", "x"] / W.loc["2000-01-31", "y"] == pytest.approx(capped[0] / capped[1])
    gross, _ = portfolio_returns(pred, W, 0)
    net, turnover = portfolio_returns(pred, W, 25)
    assert np.isnan(turnover.iloc[0])
    assert net.iloc[1] == pytest.approx(gross.iloc[1] - 0.0025 * turnover.iloc[1])
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
