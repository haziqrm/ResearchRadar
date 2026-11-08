"""Build the research map: ``network_nodes.csv`` and ``network_edges.csv``.

    python -m src.research_map.network

Nodes are the sampled papers (see ``build_encodings --per-year``).  Each node's
``area`` is its strongest research group and its ``lifecycle_stage`` is that
group's current stage.

Edges link papers that cite the same earlier work (bibliographic coupling, as
Connected Papers does).  Direct citations are too sparse in a random sample:
only about a quarter of sampled papers cite another sampled paper.  Each paper
keeps its ``k`` strongest partners, ranked by shared references divided by the
geometric mean of both reference-list lengths, so papers with very long
reference lists do not link to everything.  Edges are undirected and listed
once, with ``source_id < target_id``.

Lifecycle stages compare each group's share of the corpus in the last
``window`` years with the ``window`` years before.  Shares, not counts, so a
year in which OpenAlex indexed more papers overall does not look like growth:

* emerging  - share grew by at least ``EMERGING_GROWTH`` from a below-median base
* saturated - share grew by at most ``SATURATED_GROWTH`` (flat or shrinking)
* peaking   - everything else (still growing, at high volume or modestly)
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.common import contracts

EMERGING_GROWTH = 0.25
SATURATED_GROWTH = 0.05
DEFAULT_WINDOW = 3
DEFAULT_K = 5
DEFAULT_MIN_SHARED = 2


def coupling_edges(
    paper_ids: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
    min_shared: int = DEFAULT_MIN_SHARED,
) -> pd.DataFrame:
    """Undirected edges between papers that share at least ``min_shared`` references.

    Returns ``source_id, target_id, shared, strength`` sorted by strength.
    """
    if len(paper_ids) != len(references):
        raise ValueError("paper_ids and references must have the same length")
    if k < 1 or min_shared < 1:
        raise ValueError("k and min_shared must be at least 1")
    columns = ["source_id", "target_id", "shared", "strength"]

    vocab: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for row, refs in enumerate(references):
        for ref in set(refs or ()):
            rows.append(row)
            cols.append(vocab.setdefault(ref, len(vocab)))
    if not rows:
        return pd.DataFrame(columns=columns)

    n = len(paper_ids)
    incidence = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, len(vocab))
    )
    shared = (incidence @ incidence.T).tocsr()
    shared.setdiag(0)
    shared.eliminate_zeros()
    lengths = np.asarray(incidence.sum(axis=1)).ravel()

    pairs: dict[tuple[int, int], tuple[int, float]] = {}
    for i in range(n):
        start, end = shared.indptr[i], shared.indptr[i + 1]
        partners = shared.indices[start:end]
        counts = shared.data[start:end]
        keep = counts >= min_shared
        partners, counts = partners[keep], counts[keep]
        if len(partners) == 0:
            continue
        strength = counts / np.sqrt(lengths[i] * lengths[partners])
        for j in np.argsort(-strength, kind="stable")[:k]:
            a, b = sorted((i, int(partners[j])))
            pairs[(a, b)] = (int(counts[j]), float(strength[j]))

    ids = np.asarray(paper_ids)
    frame = pd.DataFrame(
        [(ids[a], ids[b], c, s) for (a, b), (c, s) in pairs.items()], columns=columns
    )
    swap = frame["source_id"] > frame["target_id"]
    frame.loc[swap, ["source_id", "target_id"]] = frame.loc[swap, ["target_id", "source_id"]].to_numpy()
    return frame.sort_values(["strength", "source_id"], ascending=[False, True],
                             ignore_index=True)


def group_shares(years: Sequence[int], groups: Sequence[int], n_groups: int) -> pd.DataFrame:
    """Share of each year's papers whose strongest group is ``g`` (years x groups)."""
    frame = pd.DataFrame({"year": years, "group": groups})
    counts = pd.crosstab(frame["year"], frame["group"]).reindex(
        columns=range(n_groups), fill_value=0
    )
    return counts.div(counts.sum(axis=1), axis=0)


def lifecycle_stages(
    shares: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    last_year: int | None = None,
) -> pd.DataFrame:
    """Current stage per group from the change in share between two windows.

    Returns ``group, prior_share, recent_share, growth, stage``.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    last = int(last_year if last_year is not None else shares.index.max())
    recent_years = range(last - window + 1, last + 1)
    prior_years = range(last - 2 * window + 1, last - window + 1)
    missing = [y for y in [*prior_years, *recent_years] if y not in shares.index]
    if missing:
        raise ValueError(f"no papers for years {missing}")

    recent = shares.loc[list(recent_years)].mean()
    prior = shares.loc[list(prior_years)].mean()
    growth = (recent / prior.replace(0, np.nan)) - 1
    growth = growth.fillna(np.inf).where(recent > 0, -1.0)  # appeared / vanished
    small_base = recent < recent.median()

    stage = np.where(
        (growth >= EMERGING_GROWTH) & small_base, "emerging",
        np.where(growth <= SATURATED_GROWTH, "saturated", "peaking"),
    )
    return pd.DataFrame({
        "group": shares.columns.astype(int),
        "prior_share": prior.to_numpy(),
        "recent_share": recent.to_numpy(),
        "growth": growth.to_numpy(),
        "stage": stage,
    })


def build_nodes(
    papers: pd.DataFrame,
    top_group: Sequence[int],
    names: Mapping[int, str],
    stages: Mapping[int, str],
) -> pd.DataFrame:
    """The ``network_nodes`` contract table."""
    if len(papers) != len(top_group):
        raise ValueError("papers and top_group must have the same length")
    groups = [int(g) for g in top_group]
    nodes = pd.DataFrame({
        "paper_id": papers["paper_id"].to_numpy(),
        "title": papers["title"].to_numpy(),
        "year": pd.to_numeric(papers["year"]).astype("Int64").to_numpy(),
        "area": [names[g] for g in groups],
        "citations": pd.to_numeric(papers["citations"]).fillna(0).astype(int).to_numpy(),
        "lifecycle_stage": [stages[g] for g in groups],
    })
    return nodes[contracts.CONTRACTS["network_nodes"][1]]


def build_network(
    papers: pd.DataFrame,
    encodings: pd.DataFrame,
    names: Mapping[int, str],
    *,
    k: int = DEFAULT_K,
    min_shared: int = DEFAULT_MIN_SHARED,
    window: int = DEFAULT_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(nodes, edges, stages)`` for papers that have encodings."""
    merged = papers.merge(encodings[["paper_id", "top_topic"]], on="paper_id", how="inner")
    if merged.empty:
        raise ValueError("no papers have encodings; run build_encodings first")
    top = merged["top_topic"].astype(int).to_numpy()
    shares = group_shares(merged["year"].astype(int).to_numpy(), top, len(names))
    stages = lifecycle_stages(shares, window=window)
    stages.insert(1, "name", stages["group"].map(names))
    stage_of = dict(zip(stages["group"], stages["stage"]))

    nodes = build_nodes(merged, top, names, stage_of)
    edges = coupling_edges(merged["paper_id"].tolist(), merged["references"].tolist(),
                           k=k, min_shared=min_shared)
    return nodes, edges, stages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write network_nodes.csv and network_edges.csv.")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="strongest shared-reference partners kept per paper")
    parser.add_argument("--min-shared", type=int, default=DEFAULT_MIN_SHARED,
                        help="minimum shared references for an edge")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="years per window for lifecycle stages")
    args = parser.parse_args(argv)

    from src.research_map.encoding.fetch import load_cache, load_manifest, works_to_frame
    from src.research_map.encoding.groups import load_group_names

    cache = load_cache()
    ids = [i for i in load_manifest()["paper_ids"] if i in cache]
    papers = works_to_frame([cache[i] for i in ids])
    encodings = contracts.load("paper_encodings")
    nodes, edges, stages = build_network(
        papers, encodings, load_group_names(),
        k=args.k, min_shared=args.min_shared, window=args.window,
    )
    nodes_path = contracts.save("network_nodes", nodes)
    edges_path = contracts.save("network_edges", edges[["source_id", "target_id"]])

    linked = len(set(edges["source_id"]) | set(edges["target_id"]))
    print(f"Wrote {nodes_path} ({len(nodes)} papers) and {edges_path} ({len(edges)} edges; "
          f"{linked / len(nodes):.0%} of papers have at least one)")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(stages.sort_values("growth", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
