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


def test_group_shares_rows_sum_to_one():
    shares = group_shares([2020, 2020, 2021, 2021], [0, 1, 1, 1], n_groups=3)
    assert shares.loc[2020].tolist() == [0.5, 0.5, 0.0]
    assert shares.loc[2021].tolist() == [0.0, 1.0, 0.0]


def make_shares():
    # columns: 0 small and growing fast, 1 big and growing modestly, 2 shrinking, 3 big flat
    prior = [0.02, 0.40, 0.20, 0.38]
    recent = [0.06, 0.48, 0.10, 0.36]
    years = list(range(2018, 2024))
    rows = [prior] * 3 + [recent] * 3
    return pd.DataFrame(rows, index=years, columns=range(4))


def test_lifecycle_stages_follow_the_rule():
    stages = lifecycle_stages(make_shares(), window=3).set_index("group")
    assert stages.loc[0, "stage"] == "emerging"
    assert stages.loc[1, "stage"] == "peaking"
    assert stages.loc[2, "stage"] == "saturated"
    assert stages.loc[3, "stage"] == "saturated"
    assert stages.loc[0, "growth"] == pytest.approx(2.0)


def test_lifecycle_handles_new_and_vanished_groups_and_validates():
    shares = make_shares()
    shares[4] = [0, 0, 0, 0.01, 0.01, 0.01]
    shares[5] = [0.01, 0.01, 0.01, 0, 0, 0]
    stages = lifecycle_stages(shares, window=3).set_index("group")
    assert stages.loc[4, "stage"] == "emerging" and stages.loc[5, "stage"] == "saturated"
    with pytest.raises(ValueError, match="no papers"):
        lifecycle_stages(shares, window=4)
    with pytest.raises(ValueError):
        lifecycle_stages(shares, window=0)


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
    assert list(stages.columns) == ["group", "name", "prior_share", "recent_share",
                                    "growth", "stage"]


def test_build_network_requires_encodings():
    papers = pd.DataFrame({"paper_id": ["W1"], "title": ["t"], "year": [2020],
                           "citations": [1], "references": [[]]})
    with pytest.raises(ValueError, match="encodings"):
        build_network(papers, pd.DataFrame({"paper_id": ["X"], "top_topic": [0]}), {0: "A"})
