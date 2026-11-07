"""Fetch papers from OpenAlex and cache the raw works on disk.

Raw works are stored as JSON lines in ``data/cache/openalex/works.jsonl``
(git-ignored).  Re-running a query never stores a work twice, and the cache is
reused by later queries.  The API key is handled by :class:`OpenAlexClient`
and is never logged here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.common.contracts import DATA_DIR
from src.research_map.encoding.text import record_from_work

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache" / "openalex"
CACHE_FILE = "works.jsonl"

SELECT_FIELDS: tuple[str, ...] = (
    "id",
    "title",
    "abstract_inverted_index",
    "keywords",
    "topics",
    "publication_year",
    "cited_by_count",
)

PAPER_COLUMNS: list[str] = [
    "paper_id", "title", "abstract", "keywords", "topics",
    "text", "text_quality", "year", "citations",
]


class _WorksSource(Protocol):
    def iter_works(self, query: str, **kwargs: Any) -> Any: ...


def _short_id(work: Mapping[str, Any]) -> str | None:
    raw = work.get("id")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().rstrip("/").rsplit("/", 1)[-1]


def load_cache(cache_dir: Path | str = CACHE_DIR) -> dict[str, dict[str, Any]]:
    """Return cached raw works keyed by short paper id."""
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


def works_to_frame(works: list[Mapping[str, Any]]) -> pd.DataFrame:
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

    New works are appended to the cache as they arrive, so an interrupted run
    keeps its progress.  Works already cached are not written again.  Only
    works matching this query are returned, in the order OpenAlex gave them.
    """
    if max_results < 0:
        raise ValueError("max_results cannot be negative")
    if client is None:
        from src.research_map.openalex import OpenAlexClient

        client = OpenAlexClient.from_env()

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cached = load_cache(cache_path)

    matched: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    new_count = 0
    file_path = cache_path / CACHE_FILE
    if file_path.exists() and file_path.stat().st_size:
        with file_path.open("rb") as existing:
            existing.seek(-1, 2)
            needs_newline = existing.read(1) != b"\n"
        if needs_newline:  # a crash left a partial last line; keep it on its own line
            with file_path.open("a", encoding="utf-8") as existing:
                existing.write("\n")
    with file_path.open("a", encoding="utf-8") as handle:
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
            if paper_id in cached:
                matched.append(cached[paper_id])
                continue
            handle.write(json.dumps(work, ensure_ascii=False) + "\n")
            handle.flush()
            cached[paper_id] = dict(work)
            matched.append(work)
            new_count += 1

    logger.info("Fetched %d papers (%d new, %d from cache)",
                len(matched), new_count, len(matched) - new_count)
    return works_to_frame(matched)
