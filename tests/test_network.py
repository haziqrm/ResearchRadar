import numpy as np
import pandas as pd
import pytest

from src.common.contracts import validate
from src.research_map.network import (
    build_network,
    build_nodes,
    coupling_edges,
    group_shares,
    lifecycle_stages,
)


def test_coupling_links_papers_with_enough_shared_references():
    edges = coupling_edges(
        ["P3", "P1", "P2", "P4"],
        [["a", "b", "c"], ["a", "b", "x"], ["a", "z"], []],
        min_shared=2,
    )
    assert edges[["source_id", "target_id"]].values.tolist() == [["P1", "P3"]]
    assert edges.loc[0, "shared"] == 2
    assert edges.loc[0, "strength"] == pytest.approx(2 / 3)


def test_coupling_keeps_top_k_per_paper_and_is_undirected():
    refs = [["a", "b", "c", "d"], ["a", "b", "c", "d"], ["a", "b", "c"], ["a", "b"]]
    edges = coupling_edges(["A", "B", "C", "D"], refs, k=1, min_shared=2)
    pairs = set(map(tuple, edges[["source_id", "target_id"]].values))
    assert ("A", "B") in pairs  # identical lists are each other's best partner
    assert all(s < t for s, t in pairs) and len(pairs) == len(edges)
    for paper in "ABCD":
        assert paper in {p for pair in pairs for p in pair}


def test_coupling_ignores_duplicate_refs_and_self_links_and_handles_empty():
    edges = coupling_edges(["A", "B"], [["a", "a", "b"], ["a", "b"]], min_shared=2)
    assert edges["shared"].tolist() == [2]
    assert coupling_edges(["A"], [[]]).empty
    with pytest.raises(ValueError):
        coupling_edges(["A"], [[], []])
    with pytest.raises(ValueError):
        coupling_edges(["A"], [[]], k=0)


def test_coupling_does_not_depend_on_reference_order():
    rng = np.random.default_rng(1)
    refs = [list(rng.choice(12, 5, replace=False).astype(str)) for _ in range(40)]
    ids = [f"P{i:02d}" for i in range(40)]
    forward = coupling_edges(ids, refs, k=2, min_shared=2)
    backward = coupling_edges(ids, [r[::-1] for r in refs], k=2, min_shared=2)
    pd.testing.assert_frame_equal(forward, backward)


def test_group_shares_rows_sum_to_one():
    shares = group_shares([2020, 2020, 2021, 2021], [0, 1, 1, 1], n_groups=3)
    assert shares.loc[2020].tolist() == [0.5, 0.5, 0.0]
    assert shares.loc[2021].tolist() == [0.0, 1.0, 0.0]


def make_counts(prior, recent):
    """Papers per year for 2018-2023: three years of ``prior``, three of ``recent``."""
    rows = [prior] * 3 + [recent] * 3
    return pd.DataFrame(rows, index=range(2018, 2024), columns=range(len(prior)))


def test_lifecycle_stages_follow_the_rule():
    # 1000 papers a year. 0: small, growing fast. 1: large, growing. 2: shrinking.
    # 3: large, 38% -> 36%, which is within chance at this sample size.
    counts = make_counts([20, 400, 200, 380], [60, 480, 100, 360])
    stages = lifecycle_stages(counts, window=3).set_index("group")
    assert stages.loc[0, "stage"] == "emerging"
    assert stages.loc[1, "stage"] == "peaking"  # significant growth from a large base
    assert stages.loc[2, "stage"] == "saturated"
    assert stages.loc[3, "stage"] == "peaking"  # change not larger than chance
    assert stages.loc[0, "growth"] == pytest.approx(2.0)
    assert (stages.loc[0, "prior_papers"], stages.loc[0, "recent_papers"]) == (60, 180)
    assert stages.loc[3, "q_value"] > 0.05
    assert (stages["q_value"] >= stages["p_value"] - 1e-12).all()


def test_small_noisy_group_is_not_emerging():
    # 3 -> 9 papers is +200%, but chance alone can produce that
    counts = make_counts([1, 499, 500], [3, 497, 500])
    stage = lifecycle_stages(counts, window=3).set_index("group").loc[0]
    assert stage["growth"] == pytest.approx(2.0)
    assert stage["p_value"] > 0.05 and stage["stage"] == "peaking"


def test_lifecycle_handles_new_and_vanished_groups_and_validates():
    counts = make_counts([0, 30, 470, 500], [30, 0, 470, 500])
    stages = lifecycle_stages(counts, window=3).set_index("group")
    assert stages.loc[0, "stage"] == "emerging" and np.isinf(stages.loc[0, "growth"])
    assert stages.loc[1, "stage"] == "saturated" and stages.loc[1, "growth"] == -1.0
    assert set(stages.loc[[2, 3], "stage"]) == {"peaking"}
    with pytest.raises(ValueError, match="no papers"):
        lifecycle_stages(counts, window=4)
    with pytest.raises(ValueError):
        lifecycle_stages(counts, window=0)
    with pytest.raises(ValueError):
        lifecycle_stages(counts, alpha=0)


def test_group_with_no_papers_in_either_window_is_stable():
    stage = lifecycle_stages(make_counts([0, 500, 500], [0, 500, 500]),
                             window=3).set_index("group").loc[0]
    assert stage["growth"] == 0 and stage["stage"] == "peaking"


def test_build_nodes_matches_contract(tmp_path, monkeypatch):
    from src.common import contracts

    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    contracts.save("groups", pd.DataFrame({"group_id": [0, 1], "name": ["Alpha", "Beta"],
                                           "keywords": ["", ""]}))
    papers = pd.DataFrame({"paper_id": ["W1", "W2"], "title": ["t1", "t2"],
                           "year": [2020, 2021], "citations": [5, None]})
    nodes = build_nodes(papers, [1, 0], {0: "Alpha", 1: "Beta"},
                        {0: "emerging", 1: "saturated"})
    validate("network_nodes", nodes)
    assert nodes["area"].tolist() == ["Beta", "Alpha"]
    assert nodes["citations"].tolist() == [5, 0]
    assert nodes["lifecycle_stage"].tolist() == ["saturated", "emerging"]


def test_build_network_end_to_end():
    rng = np.random.default_rng(0)
    rows, enc = [], []
    for i in range(60):
        year = 2015 + i % 9
        group = i % 3
        refs = [f"g{group}r{j}" for j in rng.choice(8, 4, replace=False)]
        rows.append({"paper_id": f"W{i}", "title": f"t{i}", "year": year,
                     "citations": i, "references": refs})
        enc.append({"paper_id": f"W{i}", "top_topic": group})
    papers, encodings = pd.DataFrame(rows), pd.DataFrame(enc)
    nodes, edges, stages = build_network(papers, encodings, {0: "A", 1: "B", 2: "C"})
    assert len(nodes) == 60 and set(nodes["area"]) == {"A", "B", "C"}
    assert set(stages["stage"]) <= {"emerging", "peaking", "saturated"}
    # edges only join papers from the same reference pool, so the same group
    area = dict(zip(nodes["paper_id"], nodes["area"]))
    assert len(edges) > 0
    assert all(area[s] == area[t] for s, t in zip(edges["source_id"], edges["target_id"]))
    assert list(stages.columns) == ["group", "name", "prior_papers", "recent_papers",
                                    "prior_share", "recent_share", "growth", "p_value",
                                    "q_value", "stage"]


def test_build_network_requires_encodings():
    papers = pd.DataFrame({"paper_id": ["W1"], "title": ["t"], "year": [2020],
                           "citations": [1], "references": [[]]})
    with pytest.raises(ValueError, match="encodings"):
        build_network(papers, pd.DataFrame({"paper_id": ["X"], "top_topic": [0]}), {0: "A"})
