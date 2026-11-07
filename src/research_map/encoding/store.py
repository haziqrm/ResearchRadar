"""Parquet store for paper vectors, with incremental embedding.

``data/paper_vectors.parquet`` has two columns: ``paper_id`` and ``vector``
(a list of float32).  Vectors are reused when the model matches, so growing a
corpus only embeds the new papers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.contracts import DATA_DIR
from src.research_map.encoding.embed import Embedder

VECTORS_PATH = DATA_DIR / "paper_vectors.parquet"
MODEL_KEY = b"embedding_model"


def save_vectors(
    ids: list[str],
    vectors: np.ndarray,
    model: str,
    path: Path | str = VECTORS_PATH,
) -> Path:
    """Write ids and an ``(n, dim)`` float32 matrix to parquet."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(ids) != len(vectors):
        raise ValueError("ids and vectors must have the same length (vectors 2-D)")
    if len(set(ids)) != len(ids):
        raise ValueError("paper ids must be unique")

    dim = vectors.shape[1]
    table = pa.table({
        "paper_id": pa.array(ids, type=pa.string()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1), type=pa.float32()), dim
        ),
    })
    table = table.replace_schema_metadata({MODEL_KEY: model.encode("utf-8")})
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    return out


def load_vectors(
    path: Path | str = VECTORS_PATH,
) -> tuple[list[str], np.ndarray, str | None]:
    """Return ``(ids, vectors, model_name)``; an empty store if the file is missing."""
    file = Path(path)
    if not file.exists():
        return [], np.zeros((0, 0), dtype=np.float32), None
    table = pq.read_table(file)
    metadata = table.schema.metadata or {}
    model = metadata.get(MODEL_KEY)
    ids = table.column("paper_id").to_pylist()
    column = table.column("vector").combine_chunks()
    if len(ids) == 0:
        return [], np.zeros((0, 0), dtype=np.float32), model.decode() if model else None
    dim = column.type.list_size
    vectors = column.values.to_numpy(zero_copy_only=False).reshape(len(ids), dim)
    return ids, vectors.astype(np.float32), model.decode() if model else None


def embed_missing(
    papers: pd.DataFrame,
    embedder: Embedder,
    path: Path | str = VECTORS_PATH,
    *,
    batch_size: int = 256,
) -> tuple[list[str], np.ndarray, int]:
    """Embed only papers not yet stored, persist, and return vectors for ``papers``.

    ``papers`` needs ``paper_id`` and ``text``.  If the stored vectors came from a
    different model they are discarded and everything is re-embedded.  Returns
    ``(ids, vectors, n_new)`` aligned with the row order of ``papers``.
    """
    if papers["paper_id"].duplicated().any():
        raise ValueError("papers contains duplicate paper_id values")

    stored_ids, stored_vectors, stored_model = load_vectors(path)
    if stored_ids and stored_model != embedder.name:
        stored_ids, stored_vectors = [], np.zeros((0, 0), dtype=np.float32)
    lookup = {pid: i for i, pid in enumerate(stored_ids)}

    wanted = papers["paper_id"].tolist()
    missing = papers[~papers["paper_id"].isin(lookup)]
    new_vectors = np.zeros((0, embedder.dim), dtype=np.float32)
    if len(missing):
        parts = [
            embedder.embed(missing["text"].iloc[i:i + batch_size].tolist())
            for i in range(0, len(missing), batch_size)
        ]
        new_vectors = np.concatenate(parts, axis=0)

    all_ids = stored_ids + missing["paper_id"].tolist()
    all_vectors = (
        np.concatenate([stored_vectors.reshape(len(stored_ids), -1), new_vectors], axis=0)
        if stored_ids else new_vectors
    )
    if len(missing) or not Path(path).exists():
        save_vectors(all_ids, all_vectors, embedder.name, path)

    index = {pid: i for i, pid in enumerate(all_ids)}
    result = all_vectors[[index[pid] for pid in wanted]] if wanted else all_vectors[:0]
    return wanted, result, len(missing)
