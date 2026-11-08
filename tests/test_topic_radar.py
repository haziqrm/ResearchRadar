"""Synthetic tests for the topic radar (no real data needed except the contract check)."""

import numpy as np
import pandas as pd
import pytest

from src.common import contracts
from src.topic_radar import forecast, seeds, subtopics
from src.topic_radar.panel import make_corpus, refit_model
from src.topic_radar.stats import block_bootstrap, summarise, top_k

YEARS = range(2000, 2026)


def _papers(trend_groups=(0, 1, 2, 3, 4), k=25, per_year=380, seed=0, trend=0.08):
    """Synthetic year/group arrays; trend groups grow their share every year (or none)."""
    rng = np.random.default_rng(seed)
    year, group = [], []
    for y in YEARS:
        w = np.ones(k)
        for g in trend_groups:
            w[g] *= np.exp(trend * (y - 2000))
        group.append(rng.choice(k, size=per_year, p=w / w.sum()))
        year.append(np.full(per_year, y))
    return np.concatenate(year), np.concatenate(group)


def _panel(year, group, origins=forecast.ORIGINS, split=True):
    n = len(year)
    edges = np.zeros((0, 2), dtype=int)
    half = forecast.feature_half(year) if split else None
    rows = [forecast.group_features(year, group, np.ones(n), np.ones(n), edges, t, half=half)
            for t in origins]
    return forecast.finish_panel(pd.concat(rows, ignore_index=True))


def _metrics(panel):
    return forecast.per_origin_metrics(forecast.walk_forward(panel, with_logit=False))


def test_without_split_noise_looks_predictable_with_split_it_does_not():
    noise = _papers(trend_groups=())
    assert _metrics(_panel(*noise, split=False))["spearman_ridge"].mean() > 0.4  # lucky groups shrink back
    assert abs(_metrics(_panel(*noise))["spearman_ridge"].mean()) < 0.15


def test_planted_trend_is_found_with_split():
    assert _metrics(_panel(*_papers(trend=0.2)))["spearman_ridge"].mean() > 0.25


def test_noise_worlds_pass_a_strong_trend_and_fail_noise():
    year, _ = _papers()
    n = len(year)
    null = forecast.null_worlds(year, forecast.feature_half(year), np.ones(n), np.ones(n),
                                np.zeros((0, 2), int), np.ones(25), n_worlds=19, seed=5)
    ev = forecast.evaluate(_metrics(_panel(*_papers(trend=0.3))), "refit", null=null)
    row = ev[ev["model"] == "spearman_ridge_minus_random"].iloc[0]
    assert row["p_noise_worlds"] == pytest.approx(1 / 20)  # beats every noise world
    ev_noise = forecast.evaluate(_metrics(_panel(*_papers(trend_groups=(), seed=7))), "refit", null=null)
    assert not bool(ev_noise["h11_pass"].iloc[0])


def test_features_ignore_papers_after_origin_and_the_target_half():
    year, group = _papers(seed=1)
    n, t = len(year), 2012
    half = forecast.feature_half(year)
    edges = np.array([[0, n - 1], [1, 2]])
    a = forecast.group_features(year, group, np.ones(n), np.ones(n), edges, t, half=half)
    group2 = group.copy()
    changed = (year > t) | ~half  # later papers and the target half
    group2[changed] = np.random.default_rng(9).integers(0, 25, changed.sum())
    b = forecast.group_features(year, group2, np.ones(n), np.ones(n), edges, t, half=half)
    pd.testing.assert_frame_equal(a.drop(columns="growth"), b.drop(columns="growth"))
    assert not np.allclose(a["growth"], b["growth"])
    assert pd.Series(half).groupby(year).mean().between(0.49, 0.51).all()


def test_walk_forward_trains_only_on_origins_ending_by_t():
    panel = _panel(*_papers(seed=2))
    t = 2015
    base = forecast.walk_forward(panel).query("origin == @t")["growth_score"].to_numpy()
    changed = panel.copy()
    late = changed["origin"] > t - 3
    changed.loc[late, ["target", "fit_target"]] = 99.0
    changed.loc[late, "boom"] = 1.0
    again = forecast.walk_forward(changed).query("origin == @t")["growth_score"].to_numpy()
    assert np.allclose(base, again)


def test_no_target_after_2025():
    year, group = _papers()
    n = len(year)
    row = forecast.group_features(year, group, np.ones(n), np.ones(n), np.zeros((0, 2), int), 2023)
    assert row["growth"].isna().all()
    assert max(forecast.ORIGINS) == 2022


def _corpus(n_per_year=12, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(2000, 2026), n_per_year)
    ids = [f"W{i}" for i in range(len(years))]
    vec = rng.normal(size=(len(years), dim))
    nodes = pd.DataFrame({"paper_id": ids, "title": "a b c", "year": years})
    frozen = pd.Series(rng.integers(0, 3, len(ids)), index=ids)
    edges = pd.DataFrame({"source_id": ids[:5], "target_id": ids[-5:]})
    return make_corpus(nodes, ids, vec, frozen, edges, {0: "x", 1: "y", 2: "z"})


def test_refit_groups_use_only_papers_up_to_origin():
    c = _corpus()
    t = 2010
    m1 = refit_model(c, t, k=3)
    c.vectors[c.year > t] += 50.0  # wildly change later papers
    m2 = refit_model(c, t, k=3)
    past = c.vectors[c.year <= t]
    assert np.allclose(m1.predict_proba(past), m2.predict_proba(past))


def test_seed_score_counts_later_papers_within_horizon_above_tau():
    c = _corpus()
    t = 2010
    target = np.flatnonzero(c.year == t)[0]
    v = c.unit.copy()
    copies = {2011: 2, 2013: 1, 2014: 3, 2009: 4}  # 2014 is outside t+1..t+3; 2009 is before t
    for y, k in copies.items():
        for i in np.flatnonzero(c.year == y)[:k]:
            v[i] = v[target]
    c.unit = v
    labels = np.zeros(len(c), dtype=int)
    out = seeds.candidate_features(c, t, labels, {0: 0.0}, tau=0.999)
    row = out[out["idx"] == target].iloc[0]
    assert row["seed_score"] == 3  # 2 from 2011 + 1 from 2013
    assert row["density"] >= 4  # the 2009 copies count as past neighbours
    assert row["novelty"] == pytest.approx(0.0, abs=1e-5)  # vectors are float32


def _seed_pred(informative, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in seeds.INFERENCE_ORIGINS:
        density = rng.normal(size=400)
        extra = rng.normal(size=400)
        target = density + extra
        true = np.zeros(400, bool)
        true[np.argsort(-target)[:40]] = True
        score = density + (extra if informative else 0.0)
        rows.append(pd.DataFrame({"origin": t, "target": target, "true_seed": true,
                                  "raw_density": density, "score": score}))
    return pd.concat(rows, ignore_index=True)


def test_seed_tests_reward_only_information_beyond_density():
    good = seeds.permutation_tests(_seed_pred(True))
    assert good["p_vs_random"] < 0.01 and good["p_vs_density"] < 0.05
    density_only = seeds.permutation_tests(_seed_pred(False))
    assert density_only["p_vs_random"] < 0.01  # density alone beats random
    assert density_only["p_vs_density"] > 0.05  # but adds nothing beyond density


def test_subtopic_level_uses_top_15_of_75_and_random_hits_of_3():
    from src.topic_radar import subtopic_forecast as sf

    assert sf.K_SUB == 75 and sf.TOP_SUB == 15 and sf.BASE_RATE_SUB == pytest.approx(0.2)
    m = pd.DataFrame({"spearman_ridge": [0.1], "spearman_momentum": [0.0], "p15_ridge": [3.0]})
    stats = forecast.pass_stats(15, 75)
    assert stats["p15_ridge_minus_random"](m).iloc[0] == pytest.approx(0.0)  # 15 * 15 / 75 = 3
    # Mechanics check with a large sample: at the real 380 papers/year a moderate +8%/year
    # trend is not detectable at 75 units (about 2.9 hits vs 3 random), so use 3,000/year.
    year, group = _papers(k=75, trend_groups=tuple(range(15)), trend=0.08, per_year=3000)
    n = len(year)
    rows = [forecast.group_features(year, group, np.ones(n), np.ones(n), np.zeros((0, 2), int), t, k=75,
                                    half=forecast.feature_half(year)) for t in forecast.ORIGINS]
    panel = forecast.finish_panel(pd.concat(rows, ignore_index=True), top=15)
    assert panel.groupby("origin")["boom"].sum().dropna().eq(15).all()
    m = forecast.per_origin_metrics(forecast.walk_forward(panel, with_logit=False, base_rate=0.2), top=15,
                                    base_rate=0.2)
    assert m["p15_ridge"].mean() > 3.0 and m["spearman_ridge"].mean() > 0.2


def test_describe_units_counts_and_parent_group():
    from src.topic_radar import subtopic_forecast as sf

    c = _corpus()
    labels = (c.year % 2).astype(int)
    out = sf.describe_units(c, labels, k=2)
    assert out["n_papers"].sum() == len(c)
    assert out.loc[0, "papers_2020_22"] == int(((c.year % 2 == 0) & (c.year >= 2020) & (c.year <= 2022)).sum())
    assert set(out["parent_group"]) <= {"x", "y", "z"}


def test_rising_needs_holm_significance_and_ten_recent_papers():
    table = pd.DataFrame({"p_value": [1e-6, 1e-6, 0.2], "recent_count": [30, 5, 40]})
    out = subtopics.flag_rising(table)
    assert out["rising"].tolist() == [True, False, False]
    assert subtopics.rising_test(10, 100, 30, 100) < 0.01
    assert subtopics.rising_test(30, 100, 10, 100) > 0.99


def test_bootstrap_and_top_k():
    assert np.allclose(block_bootstrap(np.full(10, 2.0)), 2.0)
    s = summarise(np.full(12, 0.5))
    assert s["value"] == 0.5 and s["ci_low"] == 0.5 and s["p_one_sided"] < 0.001
    assert set(top_k([5, 1, 4, 3, 2], 2)) == {0, 2}
    ties = top_k([1, 1, 1, 1], 2)
    assert len(set(ties)) == 2


def test_contract_output_validates_and_live_rows_have_no_actual():
    names = contracts.area_names()
    k = len(names)
    rows = pd.DataFrame({"group": list(range(k)) * 2, "origin": [2020] * k + [2025] * k,
                         "growth_score": np.linspace(-1, 1, 2 * k), "boom_probability": 0.2,
                         "boom": [True] * 5 + [False] * (2 * k - 5)})
    out = forecast.to_contract(rows[rows.origin == 2020], rows[rows.origin == 2025],
                               dict(enumerate(names)))
    contracts.validate("area_forecasts", out)
    assert out.loc[out["year"] == 2025, "actual_boom"].isna().all()
    assert out.groupby("year")["predicted_boom"].sum().eq(5).all()


def test_islands_never_overlap_and_similar_sit_closer():
    from src.topic_radar.app_export import island_positions
    rng = np.random.default_rng(0)
    base = rng.normal(size=(2, 16))
    # four sub-topics: 0 and 1 near-identical, 2 and 3 near-identical, the pairs unlike
    cents = np.vstack([base[0], base[0] + 0.01, base[1], base[1] + 0.01])
    radii = np.array([3.0, 3.5, 2.5, 3.0])
    pos = island_positions(cents, radii)
    for a in range(4):
        for b in range(a + 1, 4):
            assert np.hypot(*(pos[a] - pos[b])) >= radii[a] + radii[b] - 1e-6
    d = lambda a, b: np.hypot(*(pos[a] - pos[b]))
    within = (d(0, 1) + d(2, 3)) / 2
    across = (d(0, 2) + d(0, 3) + d(1, 2) + d(1, 3)) / 4
    assert within < across


def test_island_layout_keeps_sub_topics_apart():
    from src.topic_radar.app_export import island_layout
    rng = np.random.default_rng(1)
    unit = np.vstack([rng.normal(size=(30, 8)) + 3 * k for k in range(3)])
    subs = np.repeat([0, 1, -1], 30)
    xy = island_layout(unit, subs)
    assert xy.min() >= 0 and xy.max() <= 1 + 1e-9
    for s in (0, 1, -1):
        mine, other = xy[subs == s], xy[subs != s]
        centre = mine.mean(axis=0)
        reach = np.linalg.norm(mine - centre, axis=1).max()
        assert np.linalg.norm(other - centre, axis=1).min() > reach  # no foreign paper inside


def test_topic_links_only_inside_the_topic():
    from src.topic_radar.app_export import topic_links
    edges = np.array([[0, 2], [2, 5], [1, 3], [5, 0]])
    idx = np.array([0, 2, 5])
    assert sorted(map(tuple, topic_links(edges, idx))) == [(0, 1), (1, 2), (2, 0)]


def test_topic_citations_split_inside_and_leaving():
    from src.topic_radar.app_export import topic_citations
    group_of = np.array([0, 0, 1, 0, 1])
    names = {0: "A", 1: "B"}
    cites = np.array([[0, 1], [3, 0], [0, 2], [4, 3], [2, 4]])  # citing, cited
    inside, external = topic_citations(cites, np.array([0, 1, 3]), group_of, names)
    assert sorted(map(tuple, inside)) == [(0, 1), (2, 0)]
    assert external == [[0, "B", "out", 1], [2, "B", "in", 1]]


def test_in_sample_citations_keep_only_sampled_pairs():
    from src.topic_radar.citations import in_sample_citations
    refs = {"W1": ["W2", "W9", "W1"], "W2": ["W3"], "W3": []}
    out = in_sample_citations(refs, {"W1": 2020, "W2": 2015, "W3": 2010})
    assert sorted(zip(out["citing_id"], out["cited_id"])) == [("W1", "W2"), ("W2", "W3")]
    assert list(out["cited_year"]) == [2015, 2010]
