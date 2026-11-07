"""Turn raw OpenAlex works into clean records and embedding input text.

OpenAlex ships abstracts as an inverted index (``{word: [positions]}``), and
``keywords`` / ``topics`` as lists of objects with a ``display_name``.  This
module rebuilds the plain text and assembles the string fed to the embedder:

    title [SEP] abstract. Keywords: a; b. Topics: c; d.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

SEP = "[SEP]"

QUALITY_FULL = "full"  # title and abstract present
QUALITY_PARTIAL = "partial"  # no abstract, but keywords/topics add context
QUALITY_TITLE_ONLY = "title_only"  # nothing beyond the title


@dataclass(frozen=True)
class PaperRecord:
    """Clean, flat view of one OpenAlex work."""

    paper_id: str
    title: str
    abstract: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    topics: tuple[str, ...] = field(default_factory=tuple)
    text: str = ""
    text_quality: str = QUALITY_TITLE_ONLY


def reconstruct_abstract(inverted_index: Mapping[str, Iterable[int]] | None) -> str:
    """Rebuild abstract text from OpenAlex's ``abstract_inverted_index``."""
    if not inverted_index:
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or positions is None:
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned.append((position, word))
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


def _names(items: Iterable[Any] | None) -> tuple[str, ...]:
    """Extract unique display names, keeping first-seen order (case-insensitive)."""
    if not items:
        return ()
    seen: set[str] = set()
    names: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            raw = item.get("display_name") or item.get("keyword")
        else:
            raw = item
        if not isinstance(raw, str):
            continue
        name = " ".join(raw.split())
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            names.append(name)
    return tuple(names)


def _clean(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def build_text(
    title: str,
    abstract: str = "",
    keywords: Iterable[str] = (),
    topics: Iterable[str] = (),
) -> str:
    """Assemble the embedder input: ``title [SEP] abstract`` plus a short suffix."""
    title, abstract = _clean(title), _clean(abstract)
    body = f"{title} {SEP} {abstract}" if abstract else title
    extras: list[str] = []
    keywords, topics = tuple(keywords), tuple(topics)
    if keywords:
        extras.append(f"Keywords: {'; '.join(keywords)}.")
    if topics:
        extras.append(f"Topics: {'; '.join(topics)}.")
    if not extras:
        return body
    separator = " " if body.endswith((".", "!", "?")) else ". "
    return f"{body}{separator}{' '.join(extras)}"


def text_quality(abstract: str, keywords: Iterable[str], topics: Iterable[str]) -> str:
    if abstract:
        return QUALITY_FULL
    if tuple(keywords) or tuple(topics):
        return QUALITY_PARTIAL
    return QUALITY_TITLE_ONLY


def record_from_work(work: Mapping[str, Any]) -> PaperRecord:
    """Build a :class:`PaperRecord` from a raw OpenAlex work.

    The paper id is the short OpenAlex id (``W123``), not the full URL.
    Missing or ``None`` fields are tolerated.
    """
    raw_id = work.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("OpenAlex work has no id")
    paper_id = raw_id.strip().rstrip("/").rsplit("/", 1)[-1]

    title = _clean(work.get("title") or work.get("display_name"))
    abstract = _clean(reconstruct_abstract(work.get("abstract_inverted_index")))
    keywords = _names(work.get("keywords"))
    topics = _names(work.get("topics"))

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        keywords=keywords,
        topics=topics,
        text=build_text(title, abstract, keywords, topics),
        text_quality=text_quality(abstract, keywords, topics),
    )
