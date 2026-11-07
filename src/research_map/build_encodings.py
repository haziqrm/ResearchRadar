"""Build paper encodings end to end.

    python -m src.research_map.build_encodings --query "momentum anomaly" --max 3000

Steps: fetch (OpenAlex, cached) -> embed (incremental) -> discover topics ->
label topics -> write ``data/paper_encodings.csv`` and ``data/paper_vectors.parquet``.

``--fake`` swaps SPECTER2 for a fast hash embedder and writes to
``data/cache/fake_run/`` so real outputs are never touched.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import contracts
from src.research_map.encoding.embed import Embedder, FakeEmbedder, Specter2Embedder
from src.research_map.encoding.fetch import fetch_papers
from src.research_map.encoding.labels import topic_labels
from src.research_map.encoding.persist import (
    build_encodings_frame,
    save_encodings,
)
from src.research_map.encoding.store import embed_missing
from src.research_map.encoding.topics import (
    DEFAULT_MAX_TOPICS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD,
    fit_topics,
)

EMBED_CACHE_NAME = "vectors_cache.parquet"


@dataclass
class PipelineResult:
    frame: pd.DataFrame  # the paper_encodings contract table
    vectors: np.ndarray
    summary: pd.DataFrame  # one row per topic
    n_new_embeddings: int
    csv_path: Path
    parquet_path: Path


def run_pipeline(
    papers: pd.DataFrame,
    embedder: Embedder,
    *,
    data_dir: Path | str | None = None,
    k_range: tuple[int, int] = (2, 12),
    threshold: float = DEFAULT_THRESHOLD,
    max_topics: int = DEFAULT_MAX_TOPICS,
    temperature: float = DEFAULT_TEMPERATURE,
    progress: Callable[[str], None] = lambda message: None,
) -> PipelineResult:
    """Embed ``papers``, discover topics, and write the encodings files."""
    if papers.empty:
        raise ValueError("no papers to encode")
    base = Path(data_dir) if data_dir is not None else contracts.DATA_DIR
    papers = papers.drop_duplicates("paper_id").reset_index(drop=True)

    progress(f"Embedding {len(papers)} papers with {embedder.name} (cached ones are reused)")
    ids, vectors, n_new = embed_missing(papers, embedder, base / "cache" / EMBED_CACHE_NAME)
    progress(f"  {n_new} new embeddings, {len(ids) - n_new} from cache")

    progress("Discovering topics")
    result = fit_topics(vectors, k_range=k_range, threshold=threshold,
                        max_topics=max_topics, temperature=temperature)
    labels = topic_labels(
        papers["text"].tolist(), result.top_topic.tolist(),
        n_topics=result.model.n_topics,
    )
    progress(f"  {result.model.n_topics} topics (chosen by BIC)")

    frame = build_encodings_frame(
        ids, result.top_topic.tolist(), result.assigned, labels,
        embedder.name, papers["text_quality"].tolist(),
    )
    csv_path, parquet_path = save_encodings(frame, vectors, data_dir=data_dir)
    progress(f"Wrote {csv_path} and {parquet_path}")

    return PipelineResult(
        frame=frame,
        vectors=vectors,
        summary=_summarize(papers, result.top_topic, result.assigned, labels),
        n_new_embeddings=n_new,
        csv_path=csv_path,
        parquet_path=parquet_path,
    )


def _summarize(
    papers: pd.DataFrame,
    top_topic: np.ndarray,
    assigned: Sequence[Sequence[tuple[int, float]]],
    labels: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for topic, label in sorted(labels.items()):
        members = np.flatnonzero(top_topic == topic)
        rows.append({
            "topic": topic,
            "label": label,
            "papers": len(members),
            "also_in_other_topics": sum(
                1 for i in members if len(assigned[i]) > 1
            ),
            "examples": " || ".join(papers["title"].iloc[members[:2]].tolist()),
        })
    return pd.DataFrame(rows, columns=["topic", "label", "papers",
                                       "also_in_other_topics", "examples"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper encodings from OpenAlex.")
    parser.add_argument("--query", required=True, action="append",
                        help="OpenAlex search query; repeat to combine several queries")
    parser.add_argument("--max", type=int, default=1000, dest="max_results",
                        help="maximum number of papers to fetch per query (default 1000)")
    parser.add_argument("--filters", default=None, help='OpenAlex filter, e.g. "from_publication_date:2015-01-01"')
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-topics", type=int, default=DEFAULT_MAX_TOPICS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="softens topic memberships; 1 = raw mixture posteriors")
    parser.add_argument("--batch-size", type=int, default=16, help="SPECTER2 batch size")
    parser.add_argument("--fake", action="store_true",
                        help="use a fast fake embedder; writes to data/cache/fake_run/")
    parser.add_argument("--out-dir", default=None, help="write outputs here instead of data/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    if args.fake and out_dir is None:
        out_dir = contracts.DATA_DIR / "cache" / "fake_run"
    embedder: Embedder = (
        FakeEmbedder() if args.fake else Specter2Embedder(batch_size=args.batch_size)
    )

    frames = []
    for query in args.query:
        print(f"Fetching up to {args.max_results} papers for {query!r}")
        frames.append(fetch_papers(query, args.max_results, filters=args.filters))
    papers = pd.concat(frames, ignore_index=True).drop_duplicates("paper_id")
    if papers.empty:
        print("No papers found; nothing written.")
        return 1

    result = run_pipeline(
        papers, embedder, data_dir=out_dir,
        k_range=(args.k_min, args.k_max),
        threshold=args.threshold, max_topics=args.max_topics,
        temperature=args.temperature, progress=print,
    )
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(result.summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
