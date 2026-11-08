"""Level 2 (descriptive): sub-topics inside each of B's frozen groups, and which are rising.

Each group's papers are split with k-means on normalised vectors (k = clip(n/45, 2, 8))
and named with B's c-TF-IDF labels on titles. A sub-topic is `rising` when its share of
its group went up from 2020-22 to 2023-25 (one-sided two-proportion z-test, Holm across
all sub-topics, adjusted p < 0.05) and it has at least 10 recent papers.

This says what is driving a group now. It is not a forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from src.research_map.encoding.labels import topic_labels
from src.topic_radar import SUBTOPICS_CSV
from src.topic_radar.panel import Corpus, load_corpus
from src.topic_radar.stats import holm

PRIOR = range(2020, 2023)
RECENT = range(2023, 2026)
PAPERS_PER_SUBTOPIC = 45
K_MIN, K_MAX = 2, 8
MIN_RECENT = 10
ALPHA = 0.05
SEED = 0


def n_subtopics(n_papers: int) -> int:
    return int(np.clip(round(n_papers / PAPERS_PER_SUBTOPIC), K_MIN, K_MAX))


def rising_test(x_prior: int, n_prior: int, x_recent: int, n_recent: int) -> float:
    """One-sided p-value that the recent share is higher than the prior share."""
    if n_prior == 0 or n_recent == 0:
        return 1.0
    pooled = (x_prior + x_recent) / (n_prior + n_recent)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_prior + 1 / n_recent))
    if se == 0:
        return 1.0
    z = (x_recent / n_recent - x_prior / n_prior) / se
    return float(stats.norm.sf(z))


def flag_rising(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["p_holm"] = holm(table["p_value"].tolist()) if len(table) else []
    table["rising"] = (table["p_holm"] < ALPHA) & (table["recent_count"] >= MIN_RECENT)
    return table


def build_subtopics(corpus: Corpus) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sub-topic table, per-paper assignment with paper_id, group, subtopic_id, label)."""
    rows, papers = [], []
    for g in sorted(corpus.group_names):
        idx = np.flatnonzero(corpus.frozen_group == g)
        if len(idx) < 2 * K_MIN:
            continue
        k = n_subtopics(len(idx))
        sub = KMeans(k, n_init=10, random_state=SEED).fit_predict(corpus.unit[idx])
        labels = topic_labels(list(corpus.title[idx]), sub, n_terms=4, n_topics=k)
        years = corpus.year[idx]
        in_prior, in_recent = np.isin(years, list(PRIOR)), np.isin(years, list(RECENT))
        n1, n2 = int(in_prior.sum()), int(in_recent.sum())
        for s in range(k):
            mine = sub == s
            x1, x2 = int((mine & in_prior).sum()), int((mine & in_recent).sum())
            growth = ((x2 + 0.5) / max(n2, 1)) / ((x1 + 0.5) / max(n1, 1)) - 1
            rows.append({"group": corpus.group_names[g], "subtopic_id": s, "label": labels[s],
                         "n_papers": int(mine.sum()), "prior_count": x1, "recent_count": x2,
                         "group_prior": n1, "group_recent": n2, "growth": growth,
                         "p_value": rising_test(x1, n1, x2, n2)})
        papers.append(pd.DataFrame({"paper_id": corpus.paper_id[idx], "group": corpus.group_names[g],
                                    "subtopic_id": sub, "subtopic": [labels[s] for s in sub]}))
    table = flag_rising(pd.DataFrame(rows))
    cols = ["group", "subtopic_id", "label", "n_papers", "prior_count", "recent_count",
            "growth", "p_holm", "rising"]
    return table[cols], pd.concat(papers, ignore_index=True)


def run(corpus: Corpus | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    table, papers = build_subtopics(corpus or load_corpus())
    table.to_csv(SUBTOPICS_CSV, index=False)
    return table, papers


def main() -> None:
    table, _ = run()
    print(table.sort_values("growth", ascending=False).head(20).round(3).to_string(index=False))
    print(f"rising: {int(table['rising'].sum())} of {len(table)} sub-topics")


if __name__ == "__main__":
    main()
