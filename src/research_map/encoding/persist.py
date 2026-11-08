"""Write and read the paper encodings.

Two files are kept in sync:

* ``data/paper_encodings.csv`` - the readable contract (see ``CONTRACTS``).
* ``data/paper_vectors.parquet`` - the heavy 768-dim vectors keyed by ``paper_id``.

In the CSV, ``topic_ids`` and ``topic_scores`` are ``;``-separated and
``topic_labels`` is `` | ``-separated; the three are parallel lists.  They are
empty for a paper that is not clearly in any topic.  (Named ``persist`` so it
does not shadow the stdlib ``io`` module.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import contracts
from src.research_map.encoding.store import load_vectors, save_vectors

CONTRACT = "paper_encodings"
LIST_SEP = ";"
LABEL_SEP = " | "
VECTORS_FILE = "paper_vectors.parquet"


def vectors_path(data_dir: Path | str | None = None) -> Path:
    """Where the vectors live (follows ``contracts.DATA_DIR`` by default)."""
    return Path(data_dir or contracts.DATA_DIR) / VECTORS_FILE


@dataclass
class Encodings:
    """Encodings table plus vectors aligned to its row order (None for sample data)."""

    frame: pd.DataFrame
    vectors: np.ndarray | None


def parse_ids(value: object) -> list[int]:
    return [int(x) for x in str(value).split(LIST_SEP) if x != ""] if pd.notna(value) else []


def parse_scores(value: object) -> list[float]:
    return [float(x) for x in str(value).split(LIST_SEP) if x != ""] if pd.notna(value) else []


def parse_labels(value: object) -> list[str]:
    return [x for x in str(value).split(LABEL_SEP) if x != ""] if pd.notna(value) else []


def build_encodings_frame(
    paper_ids: Sequence[str],
    top_topic: Sequence[int],
    assigned: Sequence[Sequence[tuple[int, float]]],
    labels: Mapping[int, str],
    model: str,
    text_quality: Sequence[str],
) -> pd.DataFrame:
    """Assemble the contract DataFrame from the pipeline outputs."""
    n = len(paper_ids)
    if not (len(top_topic) == len(assigned) == len(text_quality) == n):
        raise ValueError("all per-paper inputs must have the same length")
    rows = []
    for pid, top, topics, quality in zip(paper_ids, top_topic, assigned, text_quality):
        rows.append({
            "paper_id": pid,
            "top_topic": int(top),
            "topic_ids": LIST_SEP.join(str(t) for t, _ in topics),
            "topic_scores": LIST_SEP.join(f"{s:.3f}" for _, s in topics),
            "topic_labels": LABEL_SEP.join(labels.get(t, f"topic {t}") for t, _ in topics),
            "model": model,
            "text_quality": quality,
        })
    return pd.DataFrame(rows, columns=contracts.CONTRACTS[CONTRACT][1])


def save_encodings(
    frame: pd.DataFrame,
    vectors: np.ndarray,
    data_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Validate, then write ``paper_encodings.csv`` and ``paper_vectors.parquet``.

    ``vectors`` rows must align with ``frame`` rows.  Files go to ``data/`` unless
    ``data_dir`` is given (used for scratch runs).  Nothing is written if any
    check fails.  Returns ``(csv_path, parquet_path)``.
    """
    contracts.validate(CONTRACT, frame)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(frame):
        raise ValueError(
            f"vectors has {len(vectors)} rows but encodings has {len(frame)}"
        )
    if frame["paper_id"].duplicated().any():
        raise ValueError("paper_encodings has duplicate paper_id values")
    models = frame["model"].unique()
    if len(models) > 1:
        raise ValueError(f"paper_encodings mixes embedding models: {list(models)}")

    model = str(models[0]) if len(models) else ""
    parquet = save_vectors(frame["paper_id"].tolist(), vectors, model, vectors_path(data_dir))
    if data_dir is None:
        csv = contracts.save(CONTRACT, frame)
    else:
        csv = Path(data_dir) / f"{CONTRACT}.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv, index=False)
    return csv, parquet


def load_encodings() -> Encodings:
    """Load encodings; real files if present, else the sample (without vectors).

    For real files the vectors are re-ordered to match the CSV, and an error is
    raised if the two files disagree on ids or on the embedding model.
    """
    real_csv = contracts.path_for(CONTRACT)
    frame = contracts.load(CONTRACT)
    if not real_csv.exists():
        return Encodings(frame, None)

    ids, vectors, model = load_vectors(vectors_path())
    if set(ids) != set(frame["paper_id"]) or len(ids) != len(frame):
        raise ValueError(
            "paper_encodings.csv and paper_vectors.parquet disagree on paper ids "
            f"({len(frame)} vs {len(ids)})"
        )
    csv_models = set(frame["model"].unique())
    if model is not None and csv_models and csv_models != {model}:
        raise ValueError(
            f"embedding model mismatch: csv {sorted(csv_models)} vs parquet {model!r}"
        )
    position = {pid: i for i, pid in enumerate(ids)}
    return Encodings(frame, vectors[[position[p] for p in frame["paper_id"]]])
