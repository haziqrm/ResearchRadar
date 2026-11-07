"""Fetch papers from OpenAlex and cache the raw works on disk.

Raw works are stored as JSON lines in ``data/cache/openalex/works.jsonl``
(git-ignored).  A work is written again only if it is new or its cached copy
lacks fields we now request; when an id appears twice the later line wins.
The API key is handled by :class:`OpenAlexClient` and is never logged here.

Two ways to build a corpus:

* :func:`fetch_papers` - search results for a query (ranked by relevance).
* :func:`fetch_yearly_sample` - a seeded random sample of the same size from
  every publication year, plus the yearly population counts.  Use this one for
  anything that measures growth: relevance ranking favours old, highly cited
  papers, so counts from search results do not track real volume.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.common.contracts import DATA_DIR
from src.research_map.encoding.text import record_from_work

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache" / "openalex"
CACHE_FILE = "works.jsonl"
MANIFEST_FILE = "corpus_manifest.json"

SELECT_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "abstract_inverted_index",
    "keywords",
    "topics",
    "primary_topic",
    "publication_year",
    "cited_by_count",
    "counts_by_year",
    "referenced_works",
    "authorships",
)

PAPER_COLUMNS: list[str] = [
    "paper_id", "title", "abstract", "keywords", "topics",
    "text", "text_quality", "year", "citations", "references", "authors",
]

# OpenAlex topics that make up the quant-finance corpus (primary topic only).
QUANT_TOPICS: dict[str, str] = {
    "T10047": "Financial Markets and Investment Strategies",
    "T10067": "Stochastic processes and financial applications",
    "T10282": "Financial Risk and Volatility Modeling",
    "T11326": "Stock Market Forecasting Methods",
    "T11413": "Risk and Portfolio Optimization",
    "T11059": "Market Dynamics and Volatility",
    "T11270": "Complex Systems and Time Series Analysis",
}


def corpus_filter(
    topic_ids: Iterable[str] = QUANT_TOPICS,
    extra: str | None = None,
) -> str:
    """OpenAlex filter for the corpus: English papers with abstracts in the topics."""
    ids = [t.strip() for t in topic_ids if t.strip()]
    if not ids:
        raise ValueError("at least one topic id is required")
    parts = [
        f"primary_topic.id:{'|'.join(ids)}",
        "type:article|preprint|review",
        "language:en",
        "has_abstract:true",
    ]
    if extra:
        parts.append(extra.strip().strip(","))
    return ",".join(parts)


class _WorksSource(Protocol):
    def iter_works(self, query: str, **kwargs: Any) -> Any: ...


def _short_id(value: Any) -> str | None:
    raw = value.get("id") if isinstance(value, Mapping) else value
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().rstrip("/").rsplit("/", 1)[-1]


def _is_complete(work: Mapping[str, Any]) -> bool:
    return all(name in work for name in SELECT_FIELDS)


def load_cache(cache_dir: Path | str = CACHE_DIR) -> dict[str, dict[str, Any]]:
    """Return cached raw works keyed by short paper id (later lines win)."""
    path = Path(cache_dir) / CACHE_FILE
    works: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return works
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-write can leave a truncated final line; skip it.
                logger.warning("Skipping corrupt cache line %d in %s", line_number, path)
                continue
            paper_id = _short_id(work) if isinstance(work, dict) else None
            if paper_id:
                works[paper_id] = work
    return works


class _CacheWriter:
    """Appends works to the cache as they arrive, so interrupted runs keep progress."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / CACHE_FILE
        self.cached = load_cache(self.dir)
        self.new_count = 0
        self._handle = None

    def __enter__(self) -> "_CacheWriter":
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as existing:
                existing.seek(-1, 2)
                needs_newline = existing.read(1) != b"\n"
            if needs_newline:  # a crash left a partial last line; keep it separate
                with self.path.open("a", encoding="utf-8") as existing:
                    existing.write("\n")
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is not None:
            self._handle.close()

    def add(self, work: Mapping[str, Any]) -> Mapping[str, Any]:
        """Cache ``work`` if it is new or the cached copy is missing fields."""
        paper_id = _short_id(work)
        cached = self.cached.get(paper_id) if paper_id else None
        if cached is not None and (_is_complete(cached) or not _is_complete(work)):
            return cached
        assert self._handle is not None
        self._handle.write(json.dumps(work, ensure_ascii=False) + "\n")
        self._handle.flush()
        if paper_id:
            self.cached[paper_id] = dict(work)
        self.new_count += 1
        return work


def _short_ids(items: Any, key: str | None = None) -> list[str]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        value = item.get(key) if key and isinstance(item, Mapping) else item
        short = _short_id(value)
        if short:
            out.append(short)
    return out


def works_to_frame(works: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten raw OpenAlex works into the paper DataFrame."""
    rows = []
    for work in works:
        try:
            rec = record_from_work(work)
        except ValueError:
            continue
        rows.append({
            "paper_id": rec.paper_id,
            "title": rec.title,
            "abstract": rec.abstract,
            "keywords": list(rec.keywords),
            "topics": list(rec.topics),
            "text": rec.text,
            "text_quality": rec.text_quality,
            "year": work.get("publication_year"),
            "citations": work.get("cited_by_count"),
            "references": _short_ids(work.get("referenced_works")),
            "authors": _short_ids(work.get("authorships"), key="author"),
        })
    return pd.DataFrame(rows, columns=PAPER_COLUMNS)


def fetch_papers(
    query: str,
    max_results: int = 1000,
    *,
    client: _WorksSource | None = None,
    filters: str | None = None,
    cache_dir: Path | str = CACHE_DIR,
) -> pd.DataFrame:
    """Search OpenAlex and return the matching papers as a DataFrame.

    Only works matching this query are returned, in the order OpenAlex gave them.
    """
    if max_results < 0:
        raise ValueError("max_results cannot be negative")
    if client is None:
        from src.research_map.openalex import OpenAlexClient

        client = OpenAlexClient.from_env()

    matched: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    with _CacheWriter(cache_dir) as cache:
        for work in client.iter_works(
            query,
            filters=filters,
            max_results=max_results,
            select=list(SELECT_FIELDS),
        ):
            paper_id = _short_id(work)
            if paper_id is None or paper_id in seen:
                continue
            seen.add(paper_id)
            matched.append(cache.add(work))

    logger.info("Fetched %d papers (%d written to cache)", len(matched), cache.new_count)
    return works_to_frame(matched)


@dataclass
class YearlySample:
    papers: pd.DataFrame
    population: dict[int, int]  # works matching the corpus filter, per year
    sampled: dict[int, int]  # works in the sample, per year


def fetch_yearly_sample(
    filters: str,
    years: Iterable[int],
    per_year: int,
    *,
    seed: int = 0,
    client: Any | None = None,
    cache_dir: Path | str = CACHE_DIR,
    progress: Callable[[str], None] = lambda message: None,
) -> YearlySample:
    """Randomly sample ``per_year`` works from each year matching ``filters``.

    Every year gets the same sample size, so each year's group shares are
    estimated equally well; ``population`` lets counts be scaled back up.  The
    corpus definition is saved to ``corpus_manifest.json`` in the cache folder.
    """
    years = sorted({int(y) for y in years})
    if not years:
        raise ValueError("years cannot be empty")
    if not 1 <= per_year <= 10_000:
        raise ValueError("per_year must be between 1 and 10,000")
    if client is None:
        from src.research_map.openalex import OpenAlexClient

        client = OpenAlexClient.from_env()

    counts = client.count_works_by(
        f"{filters},publication_year:{years[0]}-{years[-1]}", "publication_year"
    )
    population = {y: int(counts.get(str(y), 0)) for y in years}

    matched: list[Mapping[str, Any]] = []
    sampled: dict[int, int] = {}
    seen: set[str] = set()
    with _CacheWriter(cache_dir) as cache:
        for year in years:
            size = min(per_year, population[year])
            taken = 0
            if size:
                for work in client.sample_works(
                    f"{filters},publication_year:{year}", size,
                    seed=seed, select=list(SELECT_FIELDS),
                ):
                    paper_id = _short_id(work)
                    if paper_id is None or paper_id in seen:
                        continue
                    seen.add(paper_id)
                    matched.append(cache.add(work))
                    taken += 1
            sampled[year] = taken
            progress(f"  {year}: sampled {taken} of {population[year]}")

    manifest = {
        "filters": filters,
        "seed": seed,
        "per_year": per_year,
        "years": {str(y): {"population": population[y], "sampled": sampled[y]} for y in years},
        "paper_ids": [_short_id(w) for w in matched],
    }
    (Path(cache_dir) / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    logger.info("Sampled %d papers (%d written to cache)", len(matched), cache.new_count)
    return YearlySample(works_to_frame(matched), population, sampled)


def load_manifest(cache_dir: Path | str = CACHE_DIR) -> dict[str, Any]:
    """The saved corpus definition from the last :func:`fetch_yearly_sample` run."""
    path = Path(cache_dir) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run a yearly sample first")
    return json.loads(path.read_text(encoding="utf-8"))
