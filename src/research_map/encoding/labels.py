"""Readable topic names via class-based TF-IDF (c-TF-IDF).

All papers whose strongest topic is ``t`` are joined into one document; terms
that are frequent in that document but rare in the other topics' documents get
the highest score.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# Words that appear in nearly every abstract (or come from the text template).
BOILERPLATE: frozenset[str] = frozenset({
    "sep", "keywords", "topics", "paper", "papers", "study", "studies", "results",
    "result", "using", "use", "used", "based", "show", "shows", "shown", "find",
    "finds", "found", "propose", "proposes", "proposed", "present", "presents",
    "approach", "method", "methods", "new", "novel", "analysis", "article",
    "research", "evidence", "also", "may", "can", "however", "within", "among",
    "effect", "effects", "model", "models", "data", "empirical", "we", "our",
})
STOP_WORDS: list[str] = sorted(set(ENGLISH_STOP_WORDS) | BOILERPLATE)


def _fallback(topic: int) -> str:
    return f"topic {topic}"


def topic_labels(
    texts: Sequence[str],
    top_topic: Sequence[int],
    *,
    n_terms: int = 4,
    n_topics: int | None = None,
) -> dict[int, str]:
    """Return ``{topic_id: "term, term, ..."}`` for topics ``0..n_topics-1``.

    ``top_topic[i]`` is the strongest topic of ``texts[i]``.  Topics without any
    paper, or without usable words, get a ``"topic N"`` placeholder.
    """
    if len(texts) != len(top_topic):
        raise ValueError("texts and top_topic must have the same length")
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1")
    topics_present = sorted({int(t) for t in top_topic})
    if n_topics is None:
        n_topics = (max(topics_present) + 1) if topics_present else 0
    labels = {t: _fallback(t) for t in range(n_topics)}
    if not topics_present:
        return labels

    documents = [
        " ".join(text for text, t in zip(texts, top_topic) if int(t) == topic)
        for topic in topics_present
    ]
    vectorizer = CountVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        lowercase=True,
    )
    try:
        counts = vectorizer.fit_transform(documents).toarray().astype(float)
    except ValueError:  # empty vocabulary
        return labels
    terms = vectorizer.get_feature_names_out()

    # c-TF-IDF: term share within the topic times log(1 + avg words / term total)
    words_per_topic = counts.sum(axis=1, keepdims=True)
    tf = counts / np.where(words_per_topic == 0, 1.0, words_per_topic)
    avg_words = words_per_topic.mean()
    idf = np.log1p(avg_words / np.maximum(counts.sum(axis=0), 1.0))
    scores = tf * idf

    for row, topic in enumerate(topics_present):
        chosen: list[str] = []
        for index in np.argsort(-scores[row], kind="stable"):
            if scores[row, index] <= 0 or len(chosen) >= n_terms:
                break
            term = str(terms[index])
            parts = set(term.split())
            # skip a word already covered by a chosen phrase, and phrases whose
            # words are all already covered
            if any(parts <= set(c.split()) for c in chosen):
                continue
            chosen = [c for c in chosen if not set(c.split()) <= parts] + [term]
        if chosen:
            labels[topic] = ", ".join(chosen)
    return labels
